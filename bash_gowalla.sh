#!/bin/bash
export CUDA_VISIBLE_DEVICES="3"

# bz 4096 lr 0.01 wd 1e-5

# bz 2048 lr 0.01 wd 1e-6

# bz 2048 lr 0.001 wd 1e-5

# logs='log_gowalla/tune/4096/softmax'
logs='log_gowalla/test/sample_0.001'
data=gowalla
for b in 2048
do
#     # for lr in  0.01 0.001 0.0001 0.00001
    for lr in  0.001
    do
#         # for w in  0.001 0.0001
        for w in 0.00001
        do  
            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 1 --batch_size $b --data_name ${data} --epoch 100;

            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 1 --batch_size $b --data_name ${data} --sample_from_batch --epoch 100;
            
            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name ${data} --epoch 100;

            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name ${data} --sample_from_batch --epoch 100;

            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 4 --batch_size $b --data_name ${data} --epoch 100;

            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 4 --batch_size $b --data_name ${data} --sample_from_batch --epoch 100;

            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 5 --batch_size $b --data_name ${data} --epoch 100;

            # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 5 --batch_size $b --data_name ${data} --sample_from_batch --epoch 100;

            python run2.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 6 --batch_size $b --data_name ${data} --epoch 100 --sample_size ${b};




#             # python run_evl.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name gowalla  --epoch 100 --loss softmax;
            
#             # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name gowalla  --epoch 100;
        done
    done
done



# logs='log_gowalla/tune/mix_sample'
# data=gowalla
# for b in 2048
# do
#     # for lr in  0.01 0.001 0.0001 0.00001
#     for lr in  0.001
#     do
#         for w in 0.0001 0.00001 0.000001
#         do  
#             python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name ${data} --epoch 50;

#             python run2.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 6 --batch_size $b --data_name ${data} --epoch 50 --sample_size ${b};

#         done
#     done
# done



# logs='log_gowalla/tune/fewer'
# data=gowalla
# for b in 2048
# do
#     # for lr in  0.01 0.001 0.0001 0.00001
#     for lr in  0.001
#     do
#         # for w in  0.001 0.0001
#         for w in 0.0001 0.00001 0.000001
#         do  
#             python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name ${data} --epoch 100;

#             python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name ${data} --sample_from_batch --epoch 100;

#             # python run_evl.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name gowalla  --epoch 100 --loss softmax;
            
#             # python run1.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name gowalla  --epoch 100;
#         done
#     done
# done