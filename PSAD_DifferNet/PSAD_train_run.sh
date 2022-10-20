#!/bin/bash



## all categories
cate_list=("adaptor_plug" "almond" "can" "capsule" "chestnut" "eraser" "medicine_wrapper" "pill" "toothpaste" "zipper")

for i in ${cate_list[@]}
do

    echo ${i} | tee -a PSADresult/PSAD_train_result_${i}.txt    
    python -u main.py --label ${i} | tee -a PSADresult/PSAD_train_result_${i}.txt 
    echo "\n" | tee -a PSADresult/PSAD_train_result_${i}.txt
done


