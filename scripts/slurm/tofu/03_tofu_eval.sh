#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 3:00:00
#SBATCH -A project_name
#SBATCH -J eval_tofu

ml Mamba
conda activate open-unlearning
ml cuda

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

export PROJECT_ROOT="/project/xxx-project_name"

export DATASET_ROOT="datasets"
export MODEL_ROOT="/path/to/model_folder"
export HF_DATASETS_CACHE="/path/hf_cache/datasets"
export TRANSFORMERS_CACHE="/path/hf_cache/models"
# ===============================
# Config
# ===============================
model="Llama-3.1-8B-Instruct"
base_tasks=(
    "tofu_Llama-3.1-8B-Instruct_forget10_GradDiff_2"
    "tofu_Llama-3.1-8B-Instruct_forget10_GradDiff_3"
    )

RUN_EVAL=true
RUN_LM_EVAL=false

forget_split=forget10
holdout_split=holdout10
retain_split=retain90

# ===============================
# Pipeline
# ===============================
for base_task in ${base_tasks[@]}; do

    # Only used when explicitly needed (e.g., eval loading)

    echo "Model: $model"
    echo "Task: $base_task"

    # ---------- EVAL ----------
    checkpoints_to_eval=(50 100 150 200 250) 
    if $RUN_EVAL; then
        for ckpt_num in "${checkpoints_to_eval[@]}"; do
        ckpt="checkpoint-${ckpt_num}" 
        ckpt_model_path=$ckpt

        python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
            mode=eval \
            task_name=${base_task} \
            model=${model} \
            paths.output_dir=saves/eval/${base_task}/${ckpt} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${base_task}/${ckpt_model_path} \
            model.tokenizer_args.pretrained_model_name_or_path=saves/unlearn/${base_task}/${ckpt_model_path}
        done
    fi

    # ----------LM EVAL ----------
    if $RUN_LM_EVAL; then
        for ckpt_num in "${checkpoints_to_eval[@]}"; do
        ckpt="checkpoint-${ckpt_num}"
        ckpt_model_path=$ckpt

        python src/eval.py \
            experiment=eval/my/lm_eval.yaml \
            mode=eval \
            task_name=${base_task} \
            model=${model} \
            paths.output_dir=saves/eval/${base_task}/${ckpt} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${base_task}/${ckpt_model_path}
        done
    fi

done