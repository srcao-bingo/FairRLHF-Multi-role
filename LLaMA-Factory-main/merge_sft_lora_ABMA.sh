

USERNAME="student2020"

MODEL_TYPE='llama2-7b'
MODEL_PATH="/home/${USERNAME}/base-model/${MODEL_TYPE}"
SFT_CHECKPOINT="sft_checkpoint/${MODEL_TYPE}-subtler_age_bias_different_age_group_and_attribute_train"

EXPORT_MODEL="/home/${USERNAME}/export-model/${MODEL_TYPE}-sft-subtler-age-bias"


CUDA_VISIBLE_DEVICES=1 python src/export_model.py \
    --model_name_or_path ${MODEL_PATH} \
    --adapter_name_or_path ${SFT_CHECKPOINT} \
    --template default \
    --finetuning_type lora \
    --export_dir ${EXPORT_MODEL}\
    --export_size 2 \
    --export_legacy_format False