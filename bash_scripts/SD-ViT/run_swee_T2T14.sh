#!/bin/bash

algorithms=SDViT
datasets=PACS # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet
backbone=T2T14 # DeitSmall, T2T14
data_dir=/media/SSD2/Dataset
output_dir=./Results/${datasets}/unsplit/${algorithms}/${backbone}


for lambda1 in 0.5 # 0.5 0.2 0.1
    do
    for lambda2 in 3.0 # 3.0 5.0 
        do
        for command in delete_incomplete launch
        do
            python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
            --output_dir=${output_dir}/sweep_RB_loss_${lambda1}_KL_Div_${lambda2}  --command_launcher multi_gpu --algorithms ${algorithms}  \
            --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
            --hparams """{\"backbone\":\"${backbone}\",\"batch_size\":64,\"lr\":5e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"fixed_featurizer\":false,\"empty_head\":true,\"RB_loss_weight\":$lambda1,\"KL_Div_Temperature\":$lambda2}"""
        done
    done
done

