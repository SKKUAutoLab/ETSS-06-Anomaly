from mobilevideogpt.mm_utils import tokenizer_image_token
from mobilevideogpt.conversation import conv_templates
from eval.video_encoding import _get_rawvideo_dec
from mobilevideogpt.constants import *

def preprocess_input(model, tokenizer, video_path, prompt):
    vision_tower = model.get_vision_tower()
    vision_tower.load_model(model.config.mm_vision_tower)
    video_processor = vision_tower.image_processor
    image_vision_tower = model.get_image_vision_tower()
    image_vision_tower.load_model()
    image_processor = image_vision_tower.image_processor
    conv_mode = "qwen2_instruct"

    video_frames, context_frames, slice_len = _get_rawvideo_dec(
        video_path,
        image_processor,
        video_processor,
        max_frames=NUM_FRAMES,
        image_resolution=224,
        num_video_frames=NUM_FRAMES,
        num_context_images=NUM_CONTEXT_IMAGES,
    )

    # Prepare the prompt
    qs = prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize the prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
        0).cuda()

    return input_ids, video_frames, context_frames, conv.sep
