#!/bin/bash

cd ..

# custom config
DATA= /path/to/datasets  # data path
TRAINER=CoOp

DATASET=$1
CFG=$2  # config file
CTP=end  # class token position (end or middle)
NCTX=$3  # number of context tokens
SHOTS=$4  # number of shots (1, 2, 4, 8, 16)
CSC=$5  # class-specific context (False or True)
KGCN=$6
CARD=$7
PRUNE=$8
NCLASS=$9

for SEED in 1 2 3
do
    DIR=${DATASET}/${TRAINER}/${KGCN}_kgcn/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=${CARD} python3 train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.KGCN ${KGCN} \
        TRAINER.COOP.PRUNE ${PRUNE} \
        TRAINER.COOP.DATASET ${DATASET} \
        TRAINER.COOP.NCLASS ${NCLASS} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done