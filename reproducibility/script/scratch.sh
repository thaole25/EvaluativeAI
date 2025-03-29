#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
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
done

nconcepts=(5 10 15 20 25 30 35 40)
reducers=(NMF PCA)
ICE_CLFS=(ridge gnb)
WOE_CLFS=(original) # mlp lda gnb logistic

for model in "${models[@]}"; do
    for reducer in "${reducers[@]}"; do
        for nc in "${nconcepts[@]}"; do
            # train-clf is False, clf is None, learn by estimating the weights
            time python main.py -m $model -nc $nc --seed ${SLURM_ARRAY_TASK_ID} --run-mode concept --reducer $reducer --algo ice -rc True --save-model True --ice-clf NA

            for iceclf in "${ICE_CLFS[@]}"; do
                time python main.py -m $model -nc $nc --seed ${SLURM_ARRAY_TASK_ID} --run-mode concept --reducer $reducer --algo ice -rc True --save-model True --train-clf True --ice-clf $iceclf
            done

            for woeclf in "${WOE_CLFS[@]}"; do
                time python main.py -m $model -nc $nc --seed ${SLURM_ARRAY_TASK_ID} --run-mode woe --reducer $reducer --algo ice --woe-clf $woeclf --ice-clf NA
            done
            sleep 1
        done
    done
done

smallconcepts=(4 6 7 8 9 11 12)
for model in "${models[@]}"; do
    for nc in "${smallconcepts[@]}"; do
        time python main.py -m $model -nc $nc --seed ${SLURM_ARRAY_TASK_ID} --run-mode concept --reducer NMF --algo ice -rc True --save-model True --ice-clf NA

        for iceclf in "${ICE_CLFS[@]}"; do
            time python main.py -m $model -nc $nc --seed ${SLURM_ARRAY_TASK_ID} --run-mode concept --reducer NMF --algo ice -rc True --save-model True --train-clf True --ice-clf $iceclf
        done

        for woeclf in "${WOE_CLFS[@]}"; do
            time python main.py -m $model -nc $nc --seed ${SLURM_ARRAY_TASK_ID} --run-mode woe --reducer NMF --algo ice --woe-clf $woeclf --ice-clf NA
        done
        sleep 1
    done
done

for model in "${models[@]}"; do
    time python main.py -m $model --lr 0.01 --n-samples 50 --seed ${SLURM_ARRAY_TASK_ID} --run-mode concept --algo pcbm --pcbm-classifier ridge -rc True --save-model True

    for woeclf in "${WOE_CLFS[@]}"; do
        time python main.py -m $model --lr 0.01 --n-samples 50 --seed ${SLURM_ARRAY_TASK_ID} --run-mode woe --algo pcbm --pcbm-classifier ridge --woe-clf $woeclf
    done
done
