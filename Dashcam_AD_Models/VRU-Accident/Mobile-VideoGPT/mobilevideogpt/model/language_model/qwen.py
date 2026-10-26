from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.generation.utils import GenerateOutput

from einops import rearrange
from mobilevideogpt.constants import *
from mobilevideogpt.model.arch import MetaModel, MobileVideoGPTMetaForCausalLM, apply_adaptive_avg_pooling

import math
import torch.nn.functional as F

class MobileVideoGPTQwenConfig(Qwen2Config):
    model_type = "MobileVideoGPT_qwen"


class MobileVideoGPTQwenModel(MetaModel, Qwen2Model):
    config_class = MobileVideoGPTQwenConfig

    def __init__(self, config: Qwen2Config):
        super(MobileVideoGPTQwenModel, self).__init__(config)


class MobileVideoGPTQwenForCausalLM(Qwen2ForCausalLM, MobileVideoGPTMetaForCausalLM):
    config_class = MobileVideoGPTQwenConfig

    def __init__(self, config,**kwargs):
        print("config",config)
        self.num_select_k_frames_in_chunk = kwargs.pop('num_select_k_frames_in_chunk', 4)
        self.topK = kwargs.pop('topk', True)
        Qwen2ForCausalLM.__init__(self, config)
        
        config.model_type = "MobileVideoGPT_qwen"
        config.rope_scaling = None
        
        self.model = MobileVideoGPTQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        context_images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.Tensor] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
                    input_ids, attention_mask, past_key_values, labels, images, context_images)

            
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        context_images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            inputs, attention_mask, _, inputs_embeds, _ = self.prepare_inputs_labels_for_multimodal(
                inputs, attention_mask, None, None, images, context_images)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        context_images = kwargs.pop("context_images", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if context_images is not None:
            inputs["context_images"] = context_images
        return inputs

AutoConfig.register("MobileVideoGPT_qwen", MobileVideoGPTQwenConfig)
AutoModelForCausalLM.register(MobileVideoGPTQwenConfig, MobileVideoGPTQwenForCausalLM)
