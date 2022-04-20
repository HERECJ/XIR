#!/bin/bash
export CUDA_VISIBLE_DEVICES="7"


logs='log_ml100k/test2/sample'
for b in 256
do
    for lr in  0.001
    do
        for w in  0.001 
        do
            python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 1 --batch_size $b --data_name ml-100k --epoch 100;

            python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 1 --batch_size $b --data_name ml-100k --sample_from_batch --epoch 100;
            
            python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name ml-100k --epoch 100;

            python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name ml-100k --sample_from_batch --epoch 100;

            python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 4 --batch_size $b --data_name ml-100k --epoch 100;

            python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 4 --batch_size $b --data_name ml-100k --sample_from_batch --epoch 100;

            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 3 --batch_size $b --data_name ml-100k --epoch 100;

            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 3 --batch_size $b --data_name ml-100k --sample_from_batch --epoch 100;
        done
    done
done