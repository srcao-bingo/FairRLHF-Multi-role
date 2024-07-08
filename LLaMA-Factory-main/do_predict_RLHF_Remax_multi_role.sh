
# conda activate llama_factory

USERNAME="student2021"
PPO_LR=1e-6
MODEL_TYPE='llama2-7b'

DATASET='multi_role_test'


MODEL_PATH="/home/${USERNAME}/log/step3_remax-llama2_7b_hf-multi_role-lr=1e-6-2024-05-28-08-15-49-2023/actor"

PREDICT_RES_PATH="predict_res/ReMax-${MODEL_TYPE}-RLHF-multi-role-lr=${PPO_LR}-3000-1.0"

# MAX_SAMPLES=296
MAX_SAMPLES=296


CUDA_VISIBLE_DEVICES=1 python src/train_bash.py \
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