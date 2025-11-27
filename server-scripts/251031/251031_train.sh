#!/bin/bash
#SBATCH -J CS251031
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


source /data/heedong/anaconda3/etc/profile.d/conda.sh
conda activate change-captioning-baseline

#

for snapshot in $(seq 6000 1000 13000); do
    # Test after training
    python /data/heedong/DIRL_Capstone/test.py \
    --cfg /data/heedong/DIRL_Capstone/configs/dynamic/transformer_levir.yaml \
    --snapshot $snapshot

    results_dir="/data/heedong/DIRL_Capstone/experiments/DIRL+CCR_levir/test_output_${snapshot}/captions"

    python evaluate_dc.py \
    --results_dir $results_dir \
    --anno /data/heedong/DIRL_Capstone/data_loader/LEVIR-MCI-dataset/images_flattened/annotations/change_captions_reformat.json

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