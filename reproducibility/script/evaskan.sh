#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output="o-app.out"

source ~/venvs/venv-3.12.3-cuda12.5.1/bin/activate

cd ../..

DATASET=skin
MODEL=resnext50
NC=7
SEED=445
REDUCER=NMF
ICE_CLF=gnb
WOE_CLF=original
IS_TRAIN_CLF=True

time python main.py --dataset $DATASET -m $MODEL -nc $NC --seed $SEED --run-mode woe --reducer $REDUCER --algo ice --woe-clf $WOE_CLF --ice-clf NA -rc True --train-clf True --ice-clf $ICE_CLF --debug True -sa True
