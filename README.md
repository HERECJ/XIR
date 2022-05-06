# Implementation for Batch Resample


## Datasets
### Data Link
[dataurl](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI)
https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI

### Preprocess
1. Edit `datasets/raw_data/XXX/XXX.yaml`
2. Rewrite&Run   `datasets/data_preprocess/inter2mtx.py`
3. The data file is saved in `datasets/clean_data`

## Run Examples
+ See files `XXX.sh`

+ The tuning process: fix the learning rate (1e-2 or 1e-3) and tune the weight decay, with the running command `python run.py --device cuda --learning_rate ${lr} --weight_decay ${w} --log_path ${logs} --debias 2 --batch_size $b --data_name ${data} --epoch 50`, it means the sampled softmax with the pop debias

## Requirements
+ See `requirement.txt`
+ or manually update packages 

## Others
PPT