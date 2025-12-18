#!/bin/bash
#SBATCH -J CS_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch,batch_augi
#SBATCH -w augi3
#SBATCH -t 10-00:00:00
#SBATCH -o ../../logs/Out_folder/%N_%x_%j.out
#SBATCH -e ../../logs/Error/%x_%j.err

source ~/.slurm-telegram.sh # Slurm-Telegram function 로드
tg_job_start   # JOB 시작 알림

set -e
trap 'code=$?; tg_job_failed "$code"; tg_send_errlog "$ERR_FILE" 100; exit $code' ERR SIGINT SIGTERM

start_time=$(date +%s)
echo "Before changing directory:"
pwd
cd ../..
echo "After changing directory:"
pwd
which python
hostname

# ---------- 실제 작업 -----------------
# ------------------------------------

source /data/inseong/anaconda3/etc/profile.d/conda.sh
conda activate change-captioning-baseline

# python /data/inseong/skrr/DIRL_Capstone/train.py \
#     --cfg /data/inseong/skrr/DIRL_Capstone/configs/dynamic/transformer_levir_default.yaml


for snapshot in $(seq 6000 1000 13000); do
    # Test after training
    python /data/inseong/skrr/DIRL_Capstone/test.py \
    --cfg /data/inseong/skrr/DIRL_Capstone/configs/dynamic/transformer_levir_default.yaml \
    --snapshot $snapshot \
    --save_masks

    results_dir="/data/inseong/skrr/DIRL_Capstone/experiments/DIRL+CCR_levir_skip_only_1/test_output_${snapshot}/captions"

    python evaluate_dc.py \
    --results_dir $results_dir \
    --anno /data/inseong/skrr/DIRL_Capstone/data_loader/LEVIR-MCI-dataset/images_flattened/annotations/change_captions_reformat.json

done
# ---------실제 작업 종료-----------------
# -------------------------------------

exit_code=$?
runtime=$(( $(date +%s) - start_time ))

if [ $exit_code -eq 0 ]; then
  tg_job_done "$runtime"
else
  tg_job_failed "$exit_code"
  tg_send_errlog "$ERR_FILE" 100
fi

exit $exit_code
