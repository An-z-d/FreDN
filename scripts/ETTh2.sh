#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1

# ========== Basic Configuration ==========
model_name="${1:-FreqDL}"
des=${2:-des}
DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/FreDN
seed=2025
module_first=1
auxi_mode='rfft'
auxi_type='complex'
auxi_loss="MAE"
lambda=0.0
rl=$lambda
ax=$(python3 -c "print(1 - $lambda)")

# ========== Input/Output Length Combinations ==========
s_list=(96 192 336 512 720)
pl_list=(96 192 336 720)

# ========== pred_len => Parameter Sets ==========
# dropout hidden_layers hidden_size learning_rate lradj
declare -A param_map=(
    [96]="0.1 1 64 0.002 type1"
    [192]="0.3 3 16 0.01 type1"
    [336]="0.3 2 16 0.01 type1"
    [720]="0.2 2 128 0.01 type1"
)

# ========== Execute Tasks ==========
for sl in "${s_list[@]}"; do
    for pl in "${pl_list[@]}"; do

        JOB_DIR=$OUTPUT_DIR/${model_name}_ETTh2
        mkdir -p $JOB_DIR

        # Get parameter set for current pred_len
        IFS=' ' read -r dropout hidden_layers hidden_size learning_rate lradj <<< "${param_map[$pl]}"

        echo "Running task: seq_len=${sl}, pred_len=${pl} | dropout=$dropout, hidden_layers=$hidden_layers, hidden_size=$hidden_size, lr=$learning_rate, lradj=$lradj"

        python3 -u run.py \
            --des ${des} \
            --embed_size 8 \
            --batch_size 32 \
            --hidden_layers ${hidden_layers} \
            --hidden_size ${hidden_size} \
            --learning_rate ${learning_rate} \
            --dropout ${dropout} \
            --lradj ${lradj} \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh2.csv \
            --model_id "ETTh2_${sl}_${pl}" \
            --enc_in 7 \
            --model ${model_name} \
            --data ETTh2 \
            --features M \
            --seq_len ${sl} \
            --pred_len ${pl} \
            --itr 1 \
            --patience 5 \
            --auxi_lambda ${ax} \
            --rec_lambda ${rl} \
            --auxi_loss ${auxi_loss} \
            --module_first ${module_first} \
            --fix_seed ${seed} \
            --checkpoints $JOB_DIR/checkpoints/ \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type}
    done
done

echo "All tasks completed"
