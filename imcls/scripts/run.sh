##################### 下面是AAAI 的实验，为了可视化的。
#!/bin/bash

cd ..

DATA=~/data/mixstyle
DASSL=~/syn_project/mixstyle-release-master/Dassl.pytorch


D1=art_painting
D2=cartoon
D3=photo
D4=sketch

DATASET=pacs
TRAINER=Vanilla2
NET=resnet50_advs_lc01234
MIX=random
ADV_WEIGHT=10.0
MIX_WEIGHT=1.0

for SEED in $(seq 1 1)
do
    for SETUP in $(seq 1 1)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi

        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --adv_weight ${ADV_WEIGHT} \
        --mix_weight ${MIX_WEIGHT} \
        --trainer ${TRAINER} \
        --source-domains ${T} \
        --target-domains ${S1} ${S2} ${S3}  \
        --dataset-config-file ${DASSL}/configs/datasets/dg/${DATASET}.yaml \
        --config-file configs/trainers/mixstyle/${DATASET}_${MIX}.yaml \
        --output-dir aaaivisual/${DATASET}/${TRAINER}_singles/${NET}_advw${ADV_WEIGHT}_mixw${MIX_WEIGHT}/${MIX}/${T}/seed${SEED} \
        MODEL.BACKBONE.NAME ${NET}

#        python train.py --vis \
#        --model-dir aaaivisual/${DATASET}/${TRAINER}_singles/${NET}_advw${ADV_WEIGHT}_mixw${MIX_WEIGHT}/${MIX}/${T}/seed1/ \
#        --load-epoch 50 \
#        --adv_weight ${ADV_WEIGHT} \
#        --mix_weight ${MIX_WEIGHT} \
#        --root ${DATA} \
#        --seed ${SEED} \
#        --trainer ${TRAINER} \
#        --source-domains ${T} \
#        --target-domains ${S1} ${S2} ${S3}  \
#        --dataset-config-file ${DASSL}/configs/datasets/dg/${DATASET}.yaml \
#        --config-file configs/trainers/mixstyle/${DATASET}_${MIX}.yaml \
#        --output-dir aaaivisual/${DATASET}/${TRAINER}_singles/${NET}_advw${ADV_WEIGHT}_mixw${MIX_WEIGHT}/${MIX}/${T}/seed${SEED} \
#        MODEL.BACKBONE.NAME ${NET}
    done
done


