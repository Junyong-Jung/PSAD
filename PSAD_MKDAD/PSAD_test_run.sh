#!/bin/bash



##all category
cate_list=("adaptor_plug" "almond" "can" "capsule" "chestnut" "eraser" "medicine_wrapper" "pill" "toothpaste" "zipper")

for i in ${cate_list[@]}
do

    echo ${i} | tee -a PSADresult/PSAD_test_result_${i}.txt    
    python -u test.py --category ${i} | tee -a PSADresult/PSAD_test_result_${i}.txt 
done