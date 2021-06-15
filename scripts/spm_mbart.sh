while getopts ":s:t:d:" arg; do
    case $arg in
        s) SRC="$OPTARG"
        ;;
        t) TGT="$OPTARG"
        ;;
	d) dataset="$OPTARG"
	;;
        \?) echo "Invalid option -$OPTARG" >&2
        ;;
    esac
done

SPM=/workspace/sentencepiece/build/src/spm_encode
MODEL=/workspace/ctrlsum/mbart50.pretrained/sentence.bpe.model
DATA=/workspace/ctrlsum/datasets/${dataset}
TRAIN=train
VALID=val
TEST=test

${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${SRC} > ${DATA}/${VALID}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${TGT} > ${DATA}/${VALID}.spm.${TGT} &
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${SRC} > ${DATA}/${TEST}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${TGT} > ${DATA}/${TEST}.spm.${TGT} 
