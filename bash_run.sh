#!/bin/bash
export CUDA_VISIBLE_DEVICES="4"

# ml-10m wd: 0.0001 lr:0.0001
# ml-10m lr 0.01 wd 1e-6
# ml-10m lr 0.001 wd 1e-5


logs='log_ml10m/test/sample_0.001'
data=ml-10m
for b in 2048
do
    for lr in 0.001
    do
        for w in  0.00001
        do
            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 1 --batch_size $b --data_name ${data} --epoch 50;

            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 1 --batch_size $b --data_name ${data} --sample_from_batch --epoch 50;
            
            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name ${data} --epoch 50;

            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name ${data} --sample_from_batch --epoch 50;

            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 4 --batch_size $b --data_name ${data} --epoch 50;

            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 4 --batch_size $b --data_name ${data} --sample_from_batch --epoch 50;

            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 5 --batch_size $b --data_name ${data} --epoch 50;

            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 5 --batch_size $b --data_name ${data} --sample_from_batch --epoch 50;

            python run2.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 6 --batch_size $b --data_name ${data} --epoch 50 --sample_size ${b};
        done
    done
done