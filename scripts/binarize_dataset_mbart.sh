while getopts ":s:t:d:l:" arg; do
    case $arg in
        s) SRC="$OPTARG"
        ;;
        t) TGT="$OPTARG"
        ;;
	d) dataset="$OPTARG"
        ;;
	l) dictionary="$OPTARG"
	;;
	\?) echo "Invalid option -$OPTARG" >&2
        ;;
    esac
done

DICT=${dictionary}
DATA=datasets/${dataset}
TRAIN=train
VALID=val
TEST=test
DEST=data-bin/${dataset}/mbart/${SRC}


fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/${TRAIN}.spm \
  --validpref ${DATA}/${VALID}.spm \
  --testpref ${DATA}/${TEST}.spm \
  --destdir ${DEST} \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 70
