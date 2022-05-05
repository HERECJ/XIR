#!/bin/bash
#print the directory and file
  
# for file in ./log_gowalla/tune/small_bz/0.001/*/log.txt
for file in ./log_ta-feng/tune/sample/0.001/*/log.txt
do
    echo $file
    tail -1 $file 
done