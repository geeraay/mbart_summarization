#! /bin/bash

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

DATE=`date +%Y%m%d`
data_bin="cnndm"
dropout=0.1
label_smoothing=0.1
train_steps=30000
warmup_updates=500
lr=3e-05
src='oraclewordsource'
tgt='target'
update_freq=64
#for single gpu
max_tokens=1024
save_interval_updates=1000
keep_interval_updates=1
log_interval=100

criterion='label_smoothed_cross_entropy'

GPU=0
checkpoint="checkpoint_best.pt"

while getopts ":g:p:d:l:b:s:t:i:" arg; do
    case $arg in
        g) GPU="$OPTARG"
        ;;
        p) LOADDIR="$OPTARG"
        ;;
        d) data_bin="$OPTARG"
        ;;
        l) checkpoint="$OPTARG"
        ;;
        b) bartpath="$OPTARG"
        ;;
        s) src="$OPTARG"
        ;;
        t) tgt="$OPTARG"
        ;;
	i) lang_id="$OPTARG"
	;;	
        \?) echo "Invalid option -$OPTARG" >&2
        ;;
    esac
done

if [ "$data_bin" = "cnndm" ];
then
    train_steps=20000
fi


if [[ -v LOADDIR ]];
then
    add_load_string=""
    cstring="_continue"
else
    add_load_string="--reset-optimizer --reset-dataloader --reset-meters"
    cstring=""
fi


GPUSTR=$(printf "$GPU" | tr , _)

SAVE=checkpoint/${data_bin}/${DATE}/${data_bin}.${criterion}.${src}-${tgt}.lsm${label_smoothing}.drop${dropout}.uf${update_freq}.gpu${GPUSTR}${cstring}
TENSORBOARD=${SAVE}/tensorboard

rm -r ${SAVE}; mkdir -p ${SAVE} ${TENSORBOARD}

if [[ -v LOADDIR ]];
then
    echo "load from ${LOADDIR}/${checkpoint}"
    cp ${LOADDIR}/${checkpoint} ${SAVE}/checkpoint_load.pt
    restore_file=${SAVE}/checkpoint_load.pt
else
    echo "load BART large"
    restore_file=${bartpath}
fi

echo "start training"
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,af_ZA,az_AZ,bn_IN,fa_IR,he_IL,hr_HR,id_ID,ka_GE,km_KH,mk_MK,ml_IN,mn_MN,mr_IN,pl_PL,ps_AF,pt_XX,sv_SE,sw_KE,ta_IN,te_IN,th_TH,tl_XX,uk_UA,ur_PK,xh_ZA,gl_ES,sl_SI

CUDA_VISIBLE_DEVICES=${GPU} fairseq-train data-bin/${data_bin}/mbart/${src} \
    --restore-file ${restore_file} \
    --encoder-normalize-before --decoder-normalize-before \
    --max-tokens ${max_tokens} \
    --task translation_from_pretrained_bart \
    --source-lang ${src} --target-lang ${tgt} \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --required-batch-size-multiple 1 \
    --arch mbart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr ${lr} --total-num-update ${train_steps} --warmup-updates ${warmup_updates} \
    --max-update ${train_steps} \
    --update-freq ${update_freq} \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --log-format simple --log-interval ${log_interval} \
    --best-checkpoint-metric ppl \
    --save-dir ${SAVE} \
    --save-interval-updates ${save_interval_updates} --tensorboard-logdir ${TENSORBOARD}\
    --validate-interval 500 --keep-interval-updates ${keep_interval_updates} --save-interval 500 --no-epoch-checkpoints\
    --patience 5 \
    --langs ${langs} --lang-id ${lang_id} \
    --ddp-backend no_c10d \
    ${add_load_string} \
    | tee -a ${SAVE}/stdout.log

