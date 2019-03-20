export BERT_BASE_DIR=gs://gz16/bert_pretrain_model/chinese_L-12_H-768_A-12
# export BERT_LARGE_DIR=gs://gz16/bert_pretrain_model/uncased_L-24_H-1024_A-16
export SQUAD_DIR=gs://gz16/data/Illegal_Evidence/for_QA
export BASE_OUTPUT_DIR=gs://gz16/output/illegal_evidence_QA_output
# export LARGE_OUTPUT_DIR=gs://gz16/output/squad2_large
# export CUDA_VISIBLE_DEVICES=0

python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/qa_train.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/qa_eval.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=462 \
  --doc_stride=128 \
  --output_dir=$BASE_OUTPUT_DIR \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --max_query_length=47 \
  --version_2_iwith_negative=True
