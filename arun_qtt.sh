#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
export BERT_BASE_DIR=../chinese_model
export GLUE_DIR=../task/qtt/data/

# do_train=false
do_train=true
# do_eval=false
do_eval=true
do_predict=false
# do_predict=true

output_dir=/mnt/wlh/predict/bert_qtt/0406
# output_dir=../output

nohup python run_classifier.py \
  --task_name=videoCls \
  --do_train=$do_train \
  --do_eval=$do_eval \
  --data_dir=$GLUE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=32 \
  --train_batch_size=256 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --output_dir=$output_dir \
  --do_predict=$do_predict \
> log_0408_qtt.txt &
 

# --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
