#!/bin/bash



DATA=/data0/yabin/data/mixstyle

ADV_WEIGHT=10.0
MIX_WEIGHT=1.0

for SEED in $(seq 1 2)
do
    python main.py \
    --config-file cfgs/cfg_osnet_domprior.yaml \
    -s market1501 \
    -t dukemtmcreid \
    --root ${DATA} \
    --adv_weight ${ADV_WEIGHT} \
    --mix_weight ${MIX_WEIGHT} \
    model.name osnet_x1_0_advstyle0123_a0d1 \
    data.save_dir direction_strength_nodsu_onlyadvstyle/osnet_x1_0_advstyle0123_a0d1${ADV_WEIGHT}_mixw${MIX_WEIGHT}/market2duke

    python main.py \
    --config-file cfgs/cfg_osnet_domprior.yaml \
    -s dukemtmcreid \
    -t market1501 \
    --root ${DATA} \
    --adv_weight ${ADV_WEIGHT} \
    --mix_weight ${MIX_WEIGHT} \
    model.name osnet_x1_0_advstyle0123_a0d1 \
    data.save_dir direction_strength_nodsu_onlyadvstyle/osnet_x1_0_advstyle0123_a0d1${ADV_WEIGHT}_mixw${MIX_WEIGHT}/duke2market
done

#ADV_WEIGHT=10.0
#MIX_WEIGHT=1.0
#
#for SEED in $(seq 1 2)
#do
#    python main.py \
#    --config-file cfgs/cfg_r50_domprior.yaml \
#    -s market1501 \
#    -t dukemtmcreid \
#    --root ${DATA} \
#    --adv_weight ${ADV_WEIGHT} \
#    --mix_weight ${MIX_WEIGHT} \
#    model.name resnet50_fc512_advstylec0123_a0d1 \
#    data.save_dir direction_strength_nodsu_onlyadvstyle/resnet50_fc512_advstylec0123_a0d1_advw${ADV_WEIGHT}_mixw${MIX_WEIGHT}/market2duke
#
#    python main.py \
#    --config-file cfgs/cfg_r50_domprior.yaml \
#    -s dukemtmcreid \
#    -t market1501 \
#    --root ${DATA} \
#    --adv_weight ${ADV_WEIGHT} \
#    --mix_weight ${MIX_WEIGHT} \
#    model.name resnet50_fc512_advstylec0123_a0d1 \
#    data.save_dir direction_strength_nodsu_onlyadvstyle/resnet50_fc512_advstylec0123_a0d1_advw${ADV_WEIGHT}_mixw${MIX_WEIGHT}/duke2market
#done
