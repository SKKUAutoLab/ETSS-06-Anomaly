export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
set -x
model_id="videomae_large_unfreeze_4blocks_bce1_prog10_pref01_risk_exp_above3_head_v4_margin_ranking_seed42"
torchrun --standalone --nproc_per_node=2 test_encoder_anticipation_ddp.py --batch-size 200 --num-workers 1 --checkpoint ./outputs_DAD/${model_id}/best_mauc_0_1.pt --root ./datasets/DAD --metadata-json ./annotations/dad_anno.json --subset DAD --split-name test --run-sliding-window --eval-score-key risk_score --eval-anno-json ./annotations/dad_anno.json --save-dir ./results_DAD/${model_id}
