

# conda activate llama_factory

USERNAME="student2020"
PPO_LR=1e-6
MODEL_TYPE='llama2-7b'

DATASET='age_bias_rephase_question_test'


MODEL_PATH="/home/${USERNAME}/log/step3_remax-RLHF_llama_2_7b_hf-2024-04-06-2023/actor"

PREDICT_RES_PATH="predict_res/ReMax-${MODEL_TYPE}-RLHF-lr=${PPO_LR}"

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