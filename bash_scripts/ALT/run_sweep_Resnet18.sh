#!/bin/bash

algorithms=ALT_CNN
datasets=PACS  
data_dir=/media/SSD2/Dataset # Change this to the directory where you store the PACS dataset
output_dir=./Results/${datasets}/${algorithms}/Resnet18

for command in delete_incomplete launch
do
    python -m domainbed.scripts.sweep ${command} --data_dir=${data_dir} \
    --output_dir=${output_dir}  --command_launcher multi_gpu --algorithms ${algorithms}  \
    --single_domain_gen  --datasets ${datasets}  --n_hparams 1 --n_trials 3  \
    --hparams """{\"num_blocks\":0,\"mixing\":false,\"scheduler\":true,\"pre_epoch\":4,\"lr_adv\":0.00005,\"batch_size\":64,\"lr\":0.004,\"clw\":0.75,\"resnet_dropout\":0.0,\"weight_decay\":0.0,\"resnet18\":true,\"empty_fc\":true,\"fixed_featurizer\":false}"""
done

