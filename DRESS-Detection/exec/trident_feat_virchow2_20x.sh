#!/bin/bash
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --container-mounts=/data/temporary/faii/:/data/temporary/faii/,data/pathology/users/faii:/data/pathology/users/faii/
#SBATCH --container-image="dodrio1.umcn.nl#siradakittipaisarnkul/dress_trident:1.0"
#SBATCH --output=/home/%u/trident_logs/trident-%j.out
#SBATCH --error=/home/%u/trident_logs/trident-%j.err
#SBATCH --job-name=trident_feature

port=3256
node=$(oaks-lab)
USERNAME=$(siradakittipaisarnkul)
SSH_ID_RSA_FOLDER="/home/${USERNAME}/.ssh/id_rsa"

cd /data/temporary/faii

source /opt/conda/bin/activate trident

export HF_TOKEN=""

/opt/conda/envs/trident/bin/huggingface-cli login --token "$HF_TOKEN"

# echo "Starting SSH on port 3256..."
# echo "Connect with:"
# echo "vscode://vscode-remote/ssh-remote+$USER@$(hostname):3256/home/$USER?ssh=/home/$USER/.ssh/id_rsa"

python DRESS-Detection/run_batch_of_slides.py --task coords --wsi_dir Dataset/DRESS --job_dir Result/trident_processed_DRESS --mag 10 --patch_size 256 --overlap 0

python DRESS-Detection/run_batch_of_slides.py --task feat --wsi_dir Dataset/DRESS --job_dir Result/trident_processed_DRESS --patch_encoder uni_v2 --mag 10 --patch_size 256

echo "Finished processing slides."