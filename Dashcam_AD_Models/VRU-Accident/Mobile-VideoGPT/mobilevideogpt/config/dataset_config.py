import os

DATASET_DIR = os.environ.get("DATASET_DIR", "playground/data")

CC3M_595K = {
    "annotation_path": f"{DATASET_DIR}/pretraining/CC3M-595K/chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/CC3M-595K",
}
COCO_CAP = {
    "annotation_path": f"{DATASET_DIR}/pretraining/COCO/coco_cap_chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/COCO/train2014",
}
COCO_REG = {
    "annotation_path": f"{DATASET_DIR}/pretraining/COCO/coco_reg_chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/COCO/train2014",
}
COCO_REC = {
    "annotation_path": f"{DATASET_DIR}/pretraining/COCO/coco_rec_chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/COCO/train2014",
}
CONV_VideoChatGPT = {
    "annotation_path": f"{DATASET_DIR}/annotations/conversation_videochatgpt.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}
CLASSIFICATION_K710 = {
    "annotation_path": f"{DATASET_DIR}/annotations/classification_k710.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/k710",
}
CLASSIFICATION_SSV2 = {
    "annotation_path": f"{DATASET_DIR}/annotations/classification_ssv2.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/ssv2",
}
REASONING_NExTQA = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_next_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/NExTQA",
}
REASONING_CLEVRER_QA = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_clevrer_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
}
REASONING_CLEVRER_MC = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_clevrer_mc.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
}
VQA_WEBVID_QA = {
    "annotation_path": f"{DATASET_DIR}/annotations/vqa_webvid_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/webvid",
}
LONGVU_VIDEO_QA = {
    "annotation_path": f"{DATASET_DIR}/annotations/longvu.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/longvu",
}
ST_ACADEMIC_MC_0_30_S = {
    "annotation_path": f"{DATASET_DIR}/annotations/0_30_s_academic_mc_v0_1_qa_processed_orig.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/LLaVA-Video-178K/0_30_s_academic_v0_1",
}
ST_ACADEMIC_OE_0_30_S = {
    "annotation_path":  f"{DATASET_DIR}/annotations/0_30_s_academic_oe_v0_1_qa_processed_orig.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/LLaVA-Video-178K/0_30_s_academic_v0_1",
}
ST_ACADEMIC_MC_1_2_M ={
    "annotation_path":  f"{DATASET_DIR}/annotations/1_2_m_academic_mc_v0_1_qa_processed_orig.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/LLaVA-Video-178K/1_2_m_academic_v0_1",
}
ST_ACADEMIC_OE_1_2_M = {
    "annotation_path":  f"{DATASET_DIR}/annotations/1_2_m_academic_oe_v0_1_qa_processed_orig.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/LLaVA-Video-178K/1_2_m_academic_v0_1",
}
ST_ACADEMIC_MC_2_3_M ={
    "annotation_path":  f"{DATASET_DIR}/annotations/2_3_m_academic_mc_v0_1_qa_processed_orig.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/LLaVA-Video-178K/2_3_m_academic_v0_1",
}
ST_ACADEMIC_OE_2_3_M ={
    "annotation_path":  f"{DATASET_DIR}/annotations/2_3_m_academic_oe_v0_1_qa_processed_orig.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/LLaVA-Video-178K/2_3_m_academic_v0_1",
}
ST_ACADEMIC_MC_30_60_S ={
    "annotation_path":  f"{DATASET_DIR}/annotations/30_60_s_academic_mc_v0_1_qa_processed_orig.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/LLaVA-Video-178K/30_60_s_academic_v0_1",
}
ST_ACADEMIC_OE_30_60_S ={
    "annotation_path":  f"{DATASET_DIR}/annotations/30_60_s_academic_oe_v0_1_qa_processed_orig.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/LLaVA-Video-178K/30_60_s_academic_v0_1",
}
ACTIVITYNET_QA_0_30_S ={
    "annotation_path": f"{DATASET_DIR}/annotations/0_30_s_activitynetqa_oe_qa_processed_videos.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/LLaVA-Video-178K/0_30_s_activitynetqa"
}
ACTIVITYNET_QA_30_60_S ={
    "annotation_path":  f"{DATASET_DIR}/annotations/30_60_s_activitynetqa_oe_qa_processed_videos.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/LLaVA-Video-178K/30_60_s_activitynetqa"
}
ACTIVITYNET_QA_1_2_M ={
    "annotation_path":  f"{DATASET_DIR}/annotations/1_2_m_activitynetqa_oe_qa_processed_videos.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/LLaVA-Video-178K/1_2_m_activitynetqa"
}
ACTIVITYNET_QA_2_3_M ={
    "annotation_path":  f"{DATASET_DIR}/annotations/2_3_m_activitynetqa_oe_qa_processed_videos.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/LLaVA-Video-178K/2_3_m_activitynetqa"
}
ST_PERCEPTIONTEST_0_30_S ={
    "annotation_path":  f"{DATASET_DIR}/annotations/0_30_s_perceptiontest_mc_qa_processed_single_videos.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/LLaVA-Video-178K/0_30_s_perceptiontest"
}
ST_PERCEPTIONTEST_30_60_S ={
    "annotation_path":  f"{DATASET_DIR}/annotations/30_60_s_perceptiontest_mc_qa_processed_single_videos.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/LLaVA-Video-178K/30_60_s_perceptiontest"
}
