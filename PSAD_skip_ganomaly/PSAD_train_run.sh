#!/bin/bash



cate_list=("adaptor_plug" "almond" "can" "capsule" "chestnut" "eraser" "medicine_wrapper" "pill" "toothpaste" "zipper")

for i in "${cate_list[@]}";
do
    echo "Running PSAD_${i}"
    python -u train.py \
    --dataset "${i}" | tee -a "PSADresult/PSAD_train_result${j}_${i}.txt"
done
