
conda activate llama_factory

USERNAME='student2020'

MODEL_TYPE='llama2-7b'

DATASET="subtler_age_bias_different_age_group_and_attribute_test"

MODEL_PATH="/home/${USERNAME}/base-model/${MODEL_TYPE}"

SFT_CHECKPOINT="sft_checkpoint/${MODEL_TYPE}-subtler_age_bias_different_age_group_and_attribute_train"

PREDICT_RES_PATH="predict_res/sft-${MODEL_TYPE}-${DATASET}"




CUDA_VISIBLE_DEVICES=0 nohup python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path ${MODEL_PATH} \
    --dataset ${DATASET} \
    --template default \
    --adapter_name_or_path ${SFT_CHECKPOINT} \
    --finetuning_type lora \
    --output_dir ${PREDICT_RES_PATH} \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --fp16 &