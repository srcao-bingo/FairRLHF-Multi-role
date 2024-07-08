#!/bin/bash
# conda activate llm
cd /home/student2021/srcao/ReMax-master/step3_rlhf_finetuning/

set -e
set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# DeepSpeed Team
# DATA_PATH="/home/znli/datasets/Dahoas/rm-static"
DATA_PATH="age_bias"
BASE_PATH="/home/student2021/srcao/export-model"
ACTOR_MODEL_PATH="${BASE_PATH}/llama2_7b_sft"
REWARD_MODEL_PATH="${BASE_PATH}/llama2_7b_rm"
ACTOR_ZERO_STAGE=2
REWARD_ZERO_STAGE=3
REFERENCE_ZERO_STAGE=3
OUTPUT=$1
SEED=2023

if [ "$OUTPUT" == "" ]; then
    TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
    OUTPUT="./log/step3_remax-meta_llama_Llama_2_7b_hf-$TIME_STEP-$SEED"
fi

mkdir -p $OUTPUT


ACTOR_LR=1e-6

# --data_split 2,4,4 \

nohup deepspeed --include localhost:0,1,2,3 main.py \
   --algo "remax" \
   --data_path $DATA_PATH \
   --data_output_path "/tmp/data_files/llama2" \
   --data_split 0,0,1 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --reward_model_name_or_path $REWARD_MODEL_PATH \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 512 \
   --max_prompt_seq_len 1024 \
   --actor_learning_rate ${ACTOR_LR} \
   --actor_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --disable_actor_dropout \
   --disable_reward_dropout \
   --num_warmup_steps 0 \
   --kl_ctl 0.05 \
   --gamma 0.99 \
   --deepspeed \
   --offload \
   --offload_reward_model \
   --offload_reference_model \
   --actor_bf16 \
   --reward_bf16 \
   --seed $SEED \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --reward_zero_stage $REWARD_ZERO_STAGE \
   --reference_zero_stage $REFERENCE_ZERO_STAGE \
   --enable_hybrid_engine \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --print_answers \
   --save_answers \
   --save_at_final \
   &> $OUTPUT/training.log
