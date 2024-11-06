# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

DATA_PATH='./your/data/path'
CKPT_PATH='./your/checkpoint/path'
LOG_PATH='./your/log/path'
GPU=0

mkdir $CKPT_PATH
mkdir $LOG_PATH

CUDA_VISIBLE_DEVICES=$GPU python finetune.py \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path $DATA_PATH \
    --output_dir $CKPT_PATH \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 \
    --learning_rate 3e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name lora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r 32 --lora_alpha 64 --lora_init 'sketchy' \
    --nora_sigma 0.02 \
    --use_gradient_checkpointing|tee -a $LOG_PATH/finetune.log

CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset boolq \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $CKPT_PATH|tee -a $LOG_PATH/boolq.txt

CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset piqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $CKPT_PATH|tee -a $LOG_PATH/piqa.txt

CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset social_i_qa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $CKPT_PATH|tee -a $LOG_PATH/social_i_qa.txt

CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset hellaswag \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $CKPT_PATH|tee -a $LOG_PATH/hellaswag.txt

CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset winogrande \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $CKPT_PATH|tee -a $LOG_PATH/winogrande.txt

CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset ARC-Challenge \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $CKPT_PATH|tee -a $LOG_PATH/ARC-Challenge.txt

CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset ARC-Easy \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
   --lora_weights $CKPT_PATH|tee -a $LOG_PATH/ARC-Easy.txt

CUDA_VISIBLE_DEVICES=$GPU python commonsense_evaluate.py \
    --model LLaMA2-7B \
    --adapter LoRA \
    --dataset openbookqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $CKPT_PATH|tee -a $LOG_PATH/openbookqa.txt
