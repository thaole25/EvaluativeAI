#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0,1,2,3,4,5,6,42,10,100,149,222,289,377,445,498,578,789,999,3407
#SBATCH --output="o-pretrained_%a.out"

source ~/venvs/venv-3.10.4-cuda12.2/bin/activate

cd ../..

models=(resnext50 resnet50 resnet152)

for model in "${models[@]}"; do
    time python main.py -m $model -nc 8 --seed ${SLURM_ARRAY_TASK_ID} --run-mode concept --reducer NMF --algo ice
    time python ice_clfs.py -m $model -nc 8 --seed ${SLURM_ARRAY_TASK_ID} --reducer NMF --algo ice --clf gnb
    time python main.py -m $model -nc 8 --seed ${SLURM_ARRAY_TASK_ID} --run-mode woe --reducer NMF --algo ice
    sleep 1
    time python main.py --model $model --lr 0.01 --n-samples 50 --seed ${SLURM_ARRAY_TASK_ID} --run-mode concept --algo pcbm
    time python main.py --model $model --lr 0.01 --n-samples 50 --seed ${SLURM_ARRAY_TASK_ID} --run-mode woe --algo pcbm
    sleep 1
done;