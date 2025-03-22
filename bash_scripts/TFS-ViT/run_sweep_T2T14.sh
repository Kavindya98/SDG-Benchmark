#!/bin/bash

algorithms=TFSViT
alpha=0.1
datasets=PACS 
backbone=T2T14 
data_dir=/media/SSD2/Dataset
output_dir=./Results/${datasets}/${algorithms}/${backbone}


for n_layers in 1 #1 2 3 4  # number of random layers to apply TFS (n in the paper) 1 2 4
do
    for d_rate in 0.8 # 0.1 0.3 0.5 0.8 # the rate of token selection and replacement (d in the paper) 0.3 0.5 0.8
        do
        for command in delete_incomplete launch
        do
            python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
            --output_dir=${output_dir}/sweep_drate_${d_rate}_nlay_${n_layers}  --command_launcher multi_gpu --algorithms ${algorithms}  \
            --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
            --hparams """{\"backbone\":\"${backbone}\",\"batch_size\":64,\"lr\":5e-05,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"fixed_featurizer\":false,\"empty_head\":true,\"num_layers\":$n_layers,\"d_rate\":$d_rate,\"alpha\":$alpha}"""
        done
    done
done

