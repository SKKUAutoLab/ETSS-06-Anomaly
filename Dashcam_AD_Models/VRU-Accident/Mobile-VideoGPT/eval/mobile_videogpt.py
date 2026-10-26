from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

import sys
sys.path.append('Mobile-VideoGPT')
from mobilevideogpt.constants import *
from mobilevideogpt.conversation import conv_templates
from mobilevideogpt.model.builder import load_pretrained_model
from mobilevideogpt.mm_utils import tokenizer_image_token, get_model_name_from_path
from eval.video_encoding import _get_rawvideo_dec, read_frame_mod#, read_pil_frames
import os
from PIL import Image
import time

@register_model("mobile_videogpt")
class MobileVideoGPT(lmms):
    """
    MobileVideoGPT Model
    "https://github.com/Amshaker/Mobile-VideoGPT"
    """

    def __init__(
        self,
        pretrained: str = "pretrained_path",
        model_base: str = "model_base_path",
        conv_mode: str = "qwen2_instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # Load model
        model_name = get_model_name_from_path(pretrained)
        
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=pretrained, model_base=None, model_name=model_name, lora_weights=False)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        vision_tower.load_model(model.config.mm_vision_tower)
        video_processor = vision_tower.image_processor

        image_vision_tower = model.get_image_vision_tower()
        image_vision_tower.load_model()
        image_processor = image_vision_tower.image_processor
        
        
        self._model = model.to("cuda")
        self.image_processor = image_processor
        self.video_processor = video_processor
        self._tokenizer = tokenizer
        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.conv_mode = conv_mode

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._word_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for MobileVideoGPTPlusQwen2")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path):
        # TODO: Implement this (Should handle reading video, gif and frame. Should handle bound argument).
        return _, _
    
    def uniform_sample(self, m, n):
        assert n <= m
        stride = (m - 1) / (n - 1) if n > 1 else 0  # Calculate the stride
        return [int(round(i * stride)) for i in range(n)]
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        total_tokens=0
        total_time=0
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            
            if isinstance(visuals[0], Image.Image):
                # process the list of image
                video_frames, context_frames, slice_len = read_pil_frames(visuals, self.image_processor,
                                                                                self.video_processor,
                                                                                max_frames=NUM_FRAMES,
                                                                                image_resolution=224,
                                                                                num_video_frames=NUM_FRAMES,
                                                                                num_context_images=NUM_CONTEXT_IMAGES)
            else:
                assert len(visuals) == 1, f"visuals = {visuals}, \nexpected 1. Found {len(visuals)}"
                if os.path.isdir(visuals[0]):
                    video_frames, context_frames, slice_len = read_frame_mod(visuals[0], self.image_processor,
                                                                                self.video_processor,
                                                                                max_frames=NUM_FRAMES,
                                                                                image_resolution=224,
                                                                                num_video_frames=NUM_FRAMES,
                                                                                num_context_images=NUM_CONTEXT_IMAGES)
                else:
                    video_frames, context_frames, slice_len = _get_rawvideo_dec(visuals[0], self.image_processor,
                                                                                self.video_processor,
                                                                                max_frames=NUM_FRAMES,
                                                                                image_resolution=224,
                                                                                num_video_frames=NUM_FRAMES,
                                                                                num_context_images=NUM_CONTEXT_IMAGES)

            qs = contexts[0]
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs
            
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            gen_kwargs = all_gen_kwargs[0]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 16
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = 1
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            with torch.inference_mode():
                output_ids = self.model.generate(
                        input_ids,
                        images=torch.stack(video_frames, dim=0).half().cuda(),
                        context_images=torch.stack(context_frames, dim=0).half().cuda(),
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache)
            stop_str = conv.sep
            outputs = self.tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            res.append(outputs)
            pbar.update(1)

        res = re_ords.get_original(res)
        
        pbar.close()
        return res

    def generate_until_multi_round(
        self,
        prompts: List[str],
        max_new_tokens: int,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        outputs = [
            self.generate(prompt=p, max_new_tokens=max_new_tokens, stop=stop, **kwargs)
            for p in prompts
        ]
        return outputs
                        
