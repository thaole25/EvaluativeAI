#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0,1,2,3,4,5,6,42,10,100,149,222,289,377,445,498,578,789,999,3407
#SBATCH --output="o-scratch_%a.out"

source ~/venvs/venv-3.10.4-cuda12.2/bin/activate

cd ../..

models=(resnext50 resnet50 resnet152)

for model in "${models[@]}"; do
    time python train_cnn.py -m $model -r True --seed ${SLURM_ARRAY_TASK_ID}
    sleep 1
    time python learn_concepts_dataset.py --concept-dataset derm7pt --dataset ham10000 --m $model --C 0.01 --n-samples 50 --out-dir="pcbm_output" --seed ${SLURM_ARRAY_TASK_ID}
    sleep 1
done;

nconcepts=(5 10 15 20 25 30 35 40)
reducers=(NMF PCA)

for model in "${models[@]}"; do
    for reducer in "${reducers[@]}"; do
        for nc in "${nconcepts[@]}"; do
            time python main.py -m $model -nc $nc --seed ${SLURM_ARRAY_TASK_ID} --run-mode concept --reducer $reducer --algo ice -rc True --save-model True
            time python main.py -m $model -nc $nc --seed ${SLURM_ARRAY_TASK_ID} --run-mode woe --reducer $reducer --algo ice --save-model True
            sleep 1
        done;
    done;
done;

smallconcepts=(4 6 7 8 9 11 12)
for model in "${models[@]}"; do
    for nc in "${smallconcepts[@]}"; do
        time python main.py -m $model -nc $nc --seed ${SLURM_ARRAY_TASK_ID} --run-mode concept --reducer NMF --algo ice -rc True --save-model True
        time python main.py -m $model -nc $nc --seed ${SLURM_ARRAY_TASK_ID} --run-mode woe --reducer NMF --algo ice --save-model True
        sleep 1
    done;
done;

for model in "${models[@]}"; do
    time python ice_clfs.py -m $model -nc 8 --seed ${SLURM_ARRAY_TASK_ID} --reducer NMF --algo ice --clf gnb
    sleep 1
    time python main.py -m $model --lr 0.01 --n-samples 50 --seed ${SLURM_ARRAY_TASK_ID} --run-mode concept --algo pcbm --pcbm-classifier ridge -rc True --save-model True
    time python main.py -m $model --lr 0.01 --n-samples 50 --seed ${SLURM_ARRAY_TASK_ID} --run-mode woe --algo pcbm --pcbm-classifier ridge --save-model True
done;

# For the app
time python main.py -m resnext50 -nc 8 --seed 3 --run-mode woe --reducer NMF --algo ice -rc True -sa True --threshold 0.6 --debug True
time python main.py -m resnext50 --lr 0.01 --n-samples 50 --seed 3 --run-mode woe --algo pcbm --pcbm-classifier ridge -rc True -sa True --threshold 0.6 --debug True
