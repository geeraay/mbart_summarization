# CTRLsum
This is PyTorch implementation of multilingual BART for monolingual summarization. This repo inspired from [salesforce/ctrl-sum](https://github.com/salesforce/ctrl-sum) which is implementation from this paper:

```
CTRLsum: Towards Generic Controllable Text Summarization
Junxian He, Wojciech Kryściński, Bryan McCann, Nazneen Rajani, Caiming Xiong
arXiv 2020
```
 


## Dependencies
The code requires Python 3, [PyTorch](https://pytorch.org/) (>=1.5.0), and [fairseq](https://github.com/pytorch/fairseq) (the code is tested on this [commit](https://github.com/pytorch/fairseq/commit/fad3cf0769843e767155f4d0af18a61b9a804f59))

Install dependencies:
```bash
# manually install fairseq
git clone https://github.com/pytorch/fairseq

# this repo is tested on a commit of fairseq from May 2020:
# fad3cf0769843e767155f4d0af18a61b9a804f59
cd fairseq
git reset --hard fad3cf07

# the BART interface in fairseq does not support prefix-constrained decoding
# as of creating this README, thus we need to make several modifications to 
# fairseq before installing it
cp ../ctrlsum_mbart/fairseq_task.py fairseq/tasks/fairseq_task.py
cp ../ctrlsum_mbart/sequence_generator.py fairseq/
cp ../ctrlsum_mbart/hub_interface.py fairseq/models/bart/
cp ../ctrlsum_mbart/translation.py fairseq/tasks
cp ../ctrlsum_mbart/translation_from_pretrained_bart.py fairseq/tasks

# install fairseq
pip install --editable ./

cd ..

# install other requirements
pip install -r requirements.txt
```
## Model mBART Checkpoint
We use [mBART 50 pretrained model](https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.pretrained.tar.gz), download it, then extract to `mbart50.pretrained/` then trim it with this script:
```
trim_mbart by: python scripts/trim_mbart.py --pre-train-dir ./mbart50.pretrained --ft-dict ./ft/dict.en_XX.txt --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,af_ZA,az_AZ,bn_IN,fa_IR,he_IL,hr_HR,id_ID,ka_GE,km_KH,mk_MK,ml_IN,mn_MN,mr_IN,pl_PL,ps_AF,pt_XX,sv_SE,sw_KE,ta_IN,te_IN,th_TH,tl_XX,uk_UA,ur_PK,xh_ZA,gl_ES,sl_SI --output ./ft/model.pt
```
## Data Preparation
Prepare your data files into `datasets/[dataset name]`, which should consist of six data files as `[train/val/test].[source/target]`. These data files are raw text with each row representing one example. We take `cnndm` dataset as an example to preprocess the dataset (see [here](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md) for instructions to obtain the cnndm dataset).

Then create spm for source document and target document by: 

```bash
 bash scripts/spm_mbart.sh -s source -t target
```

Then preprocess data with source and target language by:
```bash
# notice that dataset corresponds with folder in datasets/[dataset name], and please see mbart50.pretrained for available language dictionaries. As for english dataset we use dict.en_XX.txt

bash scripts/binarize_dataset_mbart.sh -s source -t target -d [dataset name] -l mbart50.pretrained/[language dictionary]
```
## Train mBART for Monolingual Summarization only
For training step, use this script:

```bash
# Please use integer number separated with comma for example: 0,1,2,3 if you would want to use multiple GPUs. As for language id please refer to available languages from Model mBART Checkpoint section, for English dataset we use en_XX
bash scripts/train_mbart.sh -d [dataset name] -s source -t target -b ft/model.pt -g [GPU id] -i [language id]
```
## Train CTRLsum

### Data Processing
```bash
# this command runs the preprocessing pipeline including tokenization, truncation, and 
# keywords extraction. It will generate all required data files to train CTRLsum into 
# `datasets/cnndm`. Example obtained files can be found in `datasets/example_dataset`
# Some optional arguments can be found in preprocess.py
python scripts/preprocess.py cnndm --mode pipeline
```
For the generated files in the `datasets/cnndm`, the suffix `oracleword` represents the keywords (after keyword dropout) file, `oraclewordsource` represents the concatenated keywords and source. `oraclewordns` represents the original keywords without keyword dropout. The `.jsonl` files are potentially used to train the tagger later.

Then create spm for source document and target document by: 

```bash
 bash scripts/spm_mbart.sh -s source -t target
```

Then preprocess data with source and target language by:
```bash
# notice that dataset corresponds with folder in datasets/[dataset name], and please see mbart50.pretrained for available language dictionaries. As for english dataset we use dict.en_XX.txt

bash scripts/binarize_dataset_mbart.sh -s source -t target -d [dataset name] -l mbart50.pretrained/[language dictionary]
```
### Train the summarization model on multiple GPUs:

```
bash scripts/train_mbart.sh -g [GPUs] -d [dataset name] -b [bart checkpoint path (.pt file)]
```
`GPUs` are GPU ids separated by `,`. All our experiments are on 8 GPUs accumulating 8 gradient steps, resulting in an effective batch size of 1024x8x8 tokens in total. You probably need to increase the `update_freq` variable in `train_bart.sh` if you use less GPUs to match the effective batch size. The saved models are in dir `checkpoint`. The training arguments can be found in `train_bart.sh`.



### Train the keyword tagger (optional):
Note that the keyword tagger is required only in uncontrolled summarization setting and certain control settings which require automatic keywords (like length control in the paper)
```bash
# this requires to give 4 gpus for training by default,
# you need to change the --nproc_per_node value if you 
# train with different number of gpus
bash scripts/train_seqlabel.sh -g [GPUs] -d [dataset name]
```

The effective batch size we used for different datasets can be found in the training script as `number of gpus x batch x update_freq` as for cnndm dataset we use 128 total batch size. It could be achieve with `1 gpu x 2 batch x 64 update_freq `



## Evaluate CTRLsum
Here we include evaluation for uncontrolled summarization settings. 

### Obtain automatic keywords from a trained tagger:

```bash
# run prediction from the tagger which outputs confidence values for every token
# `checkpoint directory` is the directory that contains the `pytorch_model.bin` checkpoint.
# the results are saved in the checkpoint directory as test_predictions.txt
# the model name can be obtained from huggingface for corresponding language.
bash scripts/train_seqlabel.sh -g [GPUs] -d [dataset name] -p [checkpoint directory] -m [model name]


# obtain keywords by selecting confident words, `threshold, maximum-word, and summary-size` 
# are three hyperparameters in this step, please check Appendix A in the paper for specific
# values we used for different datasets, the performance is relatively robust
# this command will yield a file `.predwordsource` in `datasets/[dataset name]` which can be
# used as input to the summarization model to obtain uncontrolled summaries
python scripts/preprocess.py [dataset name] \
		--split test \
		--mode process_tagger_prediction \
		--tag-pred [the tagger prediction file path, named as test_predictions.txt] \
		--threshold [confidence threshold] \
		--maximum-word [maximum number of keywords] \
		--summary-size [number of sentences from which to identify keywords]

# for example 
python scripts/preprocess.py cnndm --split test --mode process_tagger_prediction --threshold 0.25 --maximum-word 30 --summary-size 10 --lang en --tag-pred checkpoint/seqlabel/cnndm/20210528/cnndm.bert-large-cased.bsz4.uf32.gpu7/test_predictions.txt

#or
python scripts/preprocess.py liputan6 --split test --mode process_tagger_prediction --threshold 0.25 --maximum-word 30 --summary-size 10 --lang id --tag-pred checkpoint/seqlabel/liputan6/20210522/liputan6.indobenchmark/indobert-large-p2.bsz8.uf4.gpu3/test_predictions.txt
```

### Metrics:

We report ROUGE scores and [BERTScore](https://github.com/Tiiiger/bert_score) in the paper. The ROUGE scores in the paper are computed using [files2rouge](https://github.com/pltrdy/files2rouge) which is a wrapper of a wrapper of the original ROUGE perl scripts. Please refer to `scripts/test_bart.sh` for our evaluation script:

```bash
# you will need the Stanford corenlp java toolkit to run this, we use it for tokenization
# this script computes ROUGE and (optionally) BERTScore.
bash scripts/test_bart.sh -g [GPUs] -s [source file name, NOT full path] -d [dataset] -p [ctrlsum checkpoint directory]
```


