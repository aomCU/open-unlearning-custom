#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 64
#SBATCH --gpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:30:00
#SBATCH -A project_name
#SBATCH -J en_6

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
models=( "Llama-3.1-8B-Instruct" )
finetune_name="fullparem_1/checkpoint-385"
trainers_experiments=("GradDiff unlearn/my/default.yaml")
gpu_count=4
# Must match the number of GPUs allocated by Slurm

num_of_run=n6

splits=(
    "my_en_forget my_en_retain"
    #"my_th_forget my_th_retain" 
)

per_device_train_batch_size=1 
gradient_accumulation_steps=8

########################################################################################################################
########################################### Unlearn  ###################################################################
########################################################################################################################


for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    retain_split=$(echo $split | cut -d' ' -f2)

    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            trainer=$(echo $trainer_experiment | cut -d' ' -f1)
            experiment=$(echo $trainer_experiment | cut -d' ' -f2)
            model_path=saves/finetune/${model}_${finetune_name}"
            task_name=my_${model}_${forget_split}_${trainer}_${num_of_run}
            echo ${task_name}: Unlearning ${model_path} using ${trainer}
            # Unlearn
            accelerate launch \
            --num_processes $gpu_count \
            --config_file configs/accelerate/default_config.yaml \
            --main_process_port $MASTER_PORT \
            src/train.py --config-name=unlearn.yaml \
            experiment=${experiment} \
            trainer=${trainer} \
            task_name=${task_name} \
            model=${model} \
            data/datasets@data.forget=${forget_split} \
            data/datasets@data.retain=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            retain_logs_path=null \
            trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
            trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            trainer.args.learning_rate=1e-5 \
            trainer.args.num_train_epochs=5 \
            trainer.args.logging_steps=1 \
            trainer.method_args.gamma=0.4 \
            trainer.method_args.alpha=0.6 \
            trainer.method_args.retain_loss_type=NLL 
        done
    done
done
