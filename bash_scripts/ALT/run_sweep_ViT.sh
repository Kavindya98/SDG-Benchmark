#!/bin/bash

algorithms=ALT_ViT
datasets=PACS 
data_dir=/media/SSD2/Dataset


for command in delete_incomplete launch
do
    for backbone in DeitSmall ViTBase T2T14 DeiTBase
    do 
        output_dir=./Results/${datasets}/${algorithms}/${backbone}
        python -u -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
        --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
        --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
        --hparams """{\"num_blocks\":0,\"backbone\":\"${backbone}\",\"mixing\":false,\"pre_epoch\":4,\"lr_adv\":0.00005,\"batch_size\":64,\"lr\":5e-05,\"clw\":0.75,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"fixed_featurizer\":false,\"empty_head\":true}"""
    done
done


