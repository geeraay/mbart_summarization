#! /bin/bash

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

DATE=`date +%Y%m%d`
data_bin="arxiv"
# model="bert-large-cased"
# model="roberta-large"
GPU=0
max_steps=3500
batch=8
#update_freq=4
#for single GPU
update_freq=16
lr=5e-5
save_steps=500
eval_steps=500
# lr=5e-5

while getopts ":g:p:d:m:" arg; do
    case $arg in
        g) GPU="$OPTARG"
        ;;
        p) LOADDIR="$OPTARG"
        ;;
        d) data_bin="$OPTARG"
        ;;
	m) model="$OPTARG"
	;;
        \?) echo "Invalid option -$OPTARG" >&2
        ;;
    esac
done

if [ "$data_bin" = "cnndm" ];
then
  max_steps=7000
  batch=4
  update_freq=32
  #lr=5e-5
elif [ "$data_bin" = "arxiv" ];
then
  max_steps=50000
  update_freq=4
  lr=5e-5
elif [ "$data_bin" = "big_patent" ];
then
  max_steps=300000
  update_freq=8
  model="bert-large-uncased"
  eval_steps=5000
  save_steps=5000
  lr=5e-5
fi

GPUSTR=$(printf "$GPU" | tr , _)

if [[ -v LOADDIR ]];
then
  SAVE=${LOADDIR}
  extra="--tokenizer_name ${model}"
  stdout="eval.log"
  model=${LOADDIR}

  export WANDB_MODE="dryrun"
else
  SAVE=checkpoint/seqlabel/${data_bin}/${DATE}/${data_bin}.${model}.bsz${batch}.uf${update_freq}.gpu${GPUSTR}
  rm -r ${SAVE}; mkdir -p ${SAVE}
  extra="--overwrite_output_dir --do_train --do_eval"
  stdout="stdout.log"
fi

mkdir -p ${SAVE}/wandb
mkdir -p datasets/${data_bin}/hf_cache

# wandb variables
export WANDB_NAME="${data_bin}.${DATE}.${model}.bsz${batch}.uf${update_freq}.gpu${GPUSTR}.${HOSTNAME}"
export WANDB_DIR="${SAVE}/wandb"

# CUDA_VISIBLE_DEVICES=${GPU} python ctrlsum/token-classification/main.py \
CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --nproc_per_node 1 ctrlsum/token-classification/main.py \
  --data_dir datasets/${data_bin}/ \
  --model_name_or_path ${model} \
  --output_dir ${SAVE} \
  --num_train_epochs 3 \
  --max_steps ${max_steps} \
  --max_seq_length 512 \
  --per_device_train_batch_size ${batch} \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps ${update_freq} \
  --save_steps ${save_steps} \
  --eval_steps ${eval_steps} \
  --threshold 0.1 \
  --learning_rate ${lr} \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --logging_steps 100 \
  --save_total_limit 10 \
  --logging_dir ${SAVE} \
  --seed 1 \
  --do_predict \
  --eval_split test \
  --evaluate_during_training \
  ${extra} \
  | tee -a ${SAVE}/${stdout}
  # --fp16 \
  # --overwrite_cache \

export WANDB_MODE="run"
