#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 64
#SBATCH --gpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00
#SBATCH -A project_name
#SBATCH -J my_full_en

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
models=("Llama-3.1-8B-Instruct")
base_task_name="base"

RUN_TRAIN=true
RUN_EVAL=false
RUN_LM_EVAL=false 
gpu_count=4  # only used when training
# Must match the number of GPUs allocated by Slurm

# ===============================
# Pipeline
# ===============================
for model in "${models[@]}"; do

  base_task="${model}_${base_task_name}"

  # Only used when explicitly needed (e.g., eval loading)

  echo "Model: $model"
  echo "Task: $base_task"

  # ---------- TRAIN ----------
  if $RUN_TRAIN; then
    accelerate launch \
      --num_processes $gpu_count \
      --config_file configs/accelerate/default_config.yaml \
      --main_process_port $MASTER_PORT \
      src/train.py experiment=finetune/my/default.yaml \
      mode=finetune \
      task_name=${base_task} \
      model=${model} \
      data/datasets@data.train=my_full_en \
      trainer.args.per_device_train_batch_size=1 \
      trainer.args.gradient_accumulation_steps=8 \
      trainer.args.learning_rate=1e-5 \
      trainer.args.lr_scheduler_type=linear \
      trainer.args.ddp_find_unused_parameters=true \
      trainer.args.gradient_checkpointing=true \
      trainer.args.num_train_epochs=5 \
      +trainer.args.warmup_ratio=0.05
  fi

  # ---------- EVAL ----------
  checkpoints_to_eval=(190) 
  if $RUN_EVAL; then
    for ckpt_num in "${checkpoints_to_eval[@]}"; do
      ckpt="checkpoint-${ckpt_num}"
      ckpt_model_path=$ckpt

      python src/eval.py \
        experiment=eval/my/default.yaml \
        mode=eval \
        task_name=${base_task} \
        model=${model} \
        paths.output_dir=saves/eval/${base_task} \
        model.model_args.pretrained_model_name_or_path=saves/finetune/${base_task}/${ckpt_model_path}
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
        model.model_args.pretrained_model_name_or_path=saves/finetune/${base_task}/${ckpt_model_path} 
    done
  fi

done