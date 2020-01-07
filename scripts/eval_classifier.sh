BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12
EXPTS_DIR=gs://zero_shot_entity_link/tmp
TFRecords=gs://zero_shot_entity_link/data
USE_TPU=true

domain='val/elder_scrolls'

EXP_NAME=BERT_fntn

python3 run_classifier.py \
  --do_train=false \
  --do_eval=true \
  --data_dir=$TFRecords \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --max_seq_length=256 \
  --num_cands=64 \
  --output_dir=$EXPTS_DIR/$EXP_NAME \
  --eval_domain=$domain \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME \
	--output_eval_file gs://zero_shot_entity_link/tmp/eval.txt
