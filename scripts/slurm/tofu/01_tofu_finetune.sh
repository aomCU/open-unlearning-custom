#!/bin/bash
#SBATCH -p gpu              # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 64                      # Specify number of nodes and processors per task
#SBATCH --gpus-per-task=4               # Specify the number of GPUs
#SBATCH --ntasks-per-node=1             # Specify tasks per node
#SBATCH -t 2:00:00                      # Specify maximum time limit (hour: minute: second)
#SBATCH -A project_name                # Specify project name
#SBATCH -J tofu_finetune                # Specify job name
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
gpu_count=4
# Must match the number of GPUs allocated by Slurm

# ########################################################################################################################
# ########################################### FULL Finetuned TOFU models #################################################
# ########################################################################################################################


for model in "${models[@]}"; do
    accelerate launch --num_processes=$gpu_count --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
    src/train.py experiment=finetune/tofu/default.yaml \
    task_name=tofu_${model}_full \
    model=${model} \
    data/datasets@data.train=TOFU_QA_full \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true 
    
done