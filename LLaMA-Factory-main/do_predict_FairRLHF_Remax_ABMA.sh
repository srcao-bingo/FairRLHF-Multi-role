
# conda activate llama_factory

USERNAME="student2020"
PPO_LR=1e-6
MODEL_TYPE='llama2-7b'

DATASET='subtler_age_bias_different_age_group_and_attribute_test'


MODEL_PATH="/home/${USERNAME}/log/step3_remax_diff_PPO_llama2_7b_hf-2024-04-10-2023-coef=0.9/actor"

PREDICT_RES_PATH="predict_res/ReMax-${MODEL_TYPE}-FairRLHF-lr=${PPO_LR}"

MAX_SAMPLES=296


CUDA_VISIBLE_DEVICES=2 python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path ${MODEL_PATH} \
    --dataset ${DATASET} \
    --template default \
    --finetuning_type lora \
    --output_dir ${PREDICT_RES_PATH} \
    --per_device_eval_batch_size 1 \
    --max_samples ${MAX_SAMPLES} \
    --predict_with_generate \
    --fp16 