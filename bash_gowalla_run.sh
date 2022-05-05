#!/bin/bash
export CUDA_VISIBLE_DEVICES="5"

data=gowalla
lr=0.01
w=0.00001
root_dic="log_${data}/${lr}_final"

# ============Compare with baselines =========================
export CUDA_VISIBLE_DEVICES="5"; (
logs="${root_dic}/baselines";
epoch=5;
b=2048;
# for b in 2048
for s in 10 20 30 40 50  # default s=10 
do
    # Sampled Softmax with no debias
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 1 --batch_size $b --data_name ${data} --epoch ${epoch} --seed ${s};

    # Sampled Softmax with pop debias
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name ${data} --epoch ${epoch} --seed ${s};

    # Resample 
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 3 --batch_size $b --data_name ${data} --epoch ${epoch} --seed ${s};

    # Mix Negatives
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 4 --batch_size $b --data_name ${data} --epoch ${epoch} --seed ${s} --sample_size ${b};

    # With Last Batch PopDebias
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 5 --batch_size $b --data_name ${data} --epoch ${epoch} --seed ${s};

    # Resample 2B->B with last batch
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 6 --batch_size $b --data_name ${data} --epoch ${epoch} --seed ${s};

    # Sample 7 : sampling bias correction

    # Resample with Cache
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 8 --batch_size $b --data_name ${data} --epoch ${epoch} --seed ${s} --lambda 0.2;
done ) &



# ============ Compare with different sample_num for resample =============

export CUDA_VISIBLE_DEVICES="4"; (
logs="${root_dic}/diff_resample_num/res"
epoch=5
b=2048
# for b in 2048
for s in 1024 512 256 128 64 32 16 8 4  # default s=10 
do   
    # Resample 
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 3 --batch_size $b --data_name ${data} --epoch ${epoch} --sample_from_batch --sample_size ${s};
done 

# ============ Compare Time with different sample_num for resample =============
logs="${root_dic}/diff_resample_num/time"
epoch=10
b=2048
# for b in 2048
for s in 1024 512 256 128 64 32 16 8 4  # default s=10 
do   
    # Resample 
    python run_time.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 3 --batch_size $b --data_name ${data} --epoch ${epoch} --sample_from_batch --sample_size ${s};
done ) & 


# ============ Compare with batch_size =========================
export CUDA_VISIBLE_DEVICES="5"; (
logs="${root_dic}/diff_bz/"
epoch=5
# b=2048
# for b in 2048
for b in 8192 4096 1024 512 256 128
do
    # Sampled Softmax with pop debias
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name ${data} --epoch ${epoch};

    # Resample 
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 3 --batch_size $b --data_name ${data} --epoch ${epoch};

    # Mix Negatives
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 4 --batch_size $b --data_name ${data} --epoch ${epoch} --sample_size ${b};


    # Resample 2B->B with last batch
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 6 --batch_size $b --data_name ${data} --epoch ${epoch};

    # Resample with Cache
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 8 --batch_size $b --data_name ${data} --epoch ${epoch} --lambda 0.2;
done ) & 


# ============ Compare with lambda =========================
export CUDA_VISIBLE_DEVICES="1"; (
logs="${root_dic}/diff_lamdba/"
epoch=5
b=2048
# for b in 2048
for la in 0.0 0.2 0.5 0.8 1.0 
do
    
    # Resample with Cache
    python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 8 --batch_size $b --data_name ${data} --epoch ${epoch} --lambda ${la};
done ) &

wait

# split them into multiple files