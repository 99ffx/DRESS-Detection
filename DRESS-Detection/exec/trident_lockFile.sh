#!/bin/bash
#SBATCH --job-name=trident_feature_extraction
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --container-mounts=/data/pathology/users/faii/:/data/pathology/users/faii/,/data/temporary/faii/:/data/temporary/faii/
#SBATCH --container-image="dodrio1.umcn.nl#siradakittipaisarnkul/dress_trident:1.0"
#SBATCH --output=/home/%u/logs/trident-%j.out
#SBATCH --error=/home/%u/logs/trident-%j.err


# Move to project directory
cd /data/temporary/faii

source /opt/conda/bin/activate trident

export HF_TOKEN=""

/opt/conda/envs/trident/bin/huggingface-cli login --token "$HF_TOKEN" 

/opt/conda/envs/trident/bin/python DRESS-Detection/run_batch_of_slides.py --task feat --wsi_dir Dataset/missing --job_dir Result/trident_processed_MDE --patch_encoder virchow2 --mag 20 --patch_size 256 

echo "Finish"