#!/usr/bin/env bash

export BERT_BASE_DIR=../data/bert_model/chinese_model
export GLUE_DIR=../data/doc_title_corpus/for_bert
CONFIG_DIR=../data/doc_title_corpus/

do_train=false
#do_train=true
do_eval=false
#do_eval=true
do_predict=false
do_predict=true

output_dir=./tmp/doctitle_output/

python run_classifier_doc.py \
  --task_name=doctitlecls \
  --do_train=$do_train \
  --do_eval=$do_eval \
  --data_dir=$GLUE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=32 \
  --train_batch_size=64 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --output_dir=$output_dir \
  --save_checkpoints_steps=5000 \
  --do_predict=$do_predict
 


