export BERT_BASE_DIR=gs://gz16/bert_pretrain_model/uncased_L-12_H-768_A-12
export BERT_LARGE_DIR=gs://gz16/bert_pretrain_model/uncased_L-24_H-1024_A-16
export SQUAD_DIR=gs://gz16/data/squad2.0
export BASE_OUTPUT_DIR=gs://gz16/output/squad2_base
export LARGE_OUTPUT_DIR=gs://gz16/output/squad2_large
export THRESH=-6.4825356006622314
# export CUDA_VISIBLE_DEVICES=0

python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=$LARGE_OUTPUT_DIR \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True \
  --null_score_diff_threshold=$THRESH
