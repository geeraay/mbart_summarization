#! /bin/bash

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

gpu=0
src=test.oraclewordnssource
tgt=test.target
exp=.
data=cnndm
outfix=default
lenpen=1
minlen=1
extra=""

export CLASSPATH=stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar

while getopts ":g:s:p:d:t:o:l:m:e:i:" arg; do
    case $arg in
        g) gpu="$OPTARG"
        ;;
        s) src="$OPTARG"
        ;;
        p) exp="$OPTARG"
        ;;
        d) data="$OPTARG"
        ;;
        t) tgt="$OPTARG"
        ;;
        o) outfix="$OPTARG"
        ;;
        l) lenpen="$OPTARG"
        ;;
        m) minlen="$OPTARG"
        ;;
        e) extra="$OPTARG"
        ;;
        i) lang_id="$OPTARG"
        ;;
        \?) echo "Invalid option -$OPTARG" >&2
        ;;
    esac
done

IFS=','
read -a gpuarr <<< "$gpu"

nsplit=${#gpuarr[@]}
# echo "nsplit ${nsplit}"

split -n l/${nsplit} -d datasets/${data}/${src} datasets/${data}/${src}.

echo "start splitting into ${nsplit} pieces and generating summaries on test dataset this may take a while"
for ((i=0;i<nsplit;i++))
do
    # echo "i $i"
    gpu_s=${gpuarr[$i]}
    # echo "gpu ${gpu_s}"
    printf -v j "%02d" $i
    # echo "j: $j"
    CUDA_VISIBLE_DEVICES=${gpu_s} python scripts/generate_mbart.py --exp ${exp} --src ${src}.$j --dataset ${data} --outfix ${outfix} --lenpen ${lenpen} --min-len ${minlen} --lang-id ${lang_id} ${extra} &
done

# wait for the decoding to finish
wait

> ${exp}/${src}.${outfix}.hypo

echo "start joining the splits"

for ((i=0;i<nsplit;i++))
do
    printf -v j "%02d" $i
    cat ${exp}/${src}.$j.${outfix}.hypo >> ${exp}/${src}.${outfix}.hypo
    rm ${exp}/${src}.$j.${outfix}.hypo
    rm datasets/${data}/${src}.$j
done

echo "start tokenizing with stanford PTBTokenizer"

cat ${exp}/${src}.${outfix}.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${exp}/${src}.${outfix}.hypo.tokenized
cat datasets/${data}/${tgt} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${exp}/${tgt}.tokenized
files2rouge ${exp}/${tgt}.tokenized ${exp}/${src}.${outfix}.hypo.tokenized  > ${exp}/${src}.${outfix}.rouge


echo "start computing BERTScore"
# compute BERTScore
cp datasets/${data}/${tgt} ${exp}/
CUDA_VISIBLE_DEVICES=${gpu} bert-score -r ${exp}/${tgt} -c ${exp}/${src}.${outfix}.hypo --lang en --rescale_with_baseline > ${exp}/${src}.${outfix}.bertscore

