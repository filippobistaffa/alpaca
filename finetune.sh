#!/bin/bash

GPUS="2"
CPUS=$(( $GPUS * 32 ))

tmpfile=$(mktemp)
sbatch 1> $tmpfile <<EOF
#!/bin/bash
#SBATCH --job-name=finetune-alpaca
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=246GB
#SBATCH --gres=gpu:a100:$GPUS
#SBATCH --output=alpaca.out
#SBATCH --error=alpaca.err
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=filippo.bistaffa@gmail.com

module load python/3.9.9

#export TORCH_CPP_LOG_LEVEL=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,P2P

torchrun --nproc_per_node=$GPUS train.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_config fsdp_config.json \
    --tf32 True
EOF
