# Implementation for BIR and XIR
Cache-Augmented Inbatch Importance Resampling for Training Recommender Retriever, NeurIPS 2022


## Datasets
### Data Link
[dataurl](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI)
https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI

### Preprocess
1. Edit `datasets/raw_data/XXX/XXX.yaml`
2. Rewrite&Run   `datasets/data_preprocess/inter2mtx.py`
3. The data file is saved in `datasets/clean_data`

## Algs
+ `--debias 1` : SSL
+ `--debias 2` : SSL-Pop
+ `--debias 4` : MNS
+ `--debias 7` : G-Tower
+ `--debias 3` : BIR, inbatch importance resampling
+ `--debias 8` : XIR, cache-augmented inbatch importance sampling

## Run Examples
+ See files `bash_run.sh`


## Requirements
+ See `requirement.txt`
+ or manually update packages 
