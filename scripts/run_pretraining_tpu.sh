BERT_BASE_DIR=gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12
EXPTS_DIR=gs://zero_shot_entity_link/tmp/pretrain
TFRecords=gs://zero_shot_entity_link/tmp/my_pretrain_data/bert_ms256
USE_TPU=true

split=val
domain='coronation_street'

# WB -> Src+Tgt
EXP_NAME=BERT_srctgtlm
INIT=$BERT_BASE_DIR/bert_model.ckpt
INPUT_FILE=$TFRecords/train/*,$TFRecords/$split/${domain}.tfrecord

python3 run_pretraining.py \
  --input_file=$INPUT_FILE \
  --output_dir=$EXPTS_DIR/$EXP_NAME \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT \
  --train_batch_size=256 \
  --max_seq_length=256 \
  --num_train_steps=10000 \
  --num_warmup_steps=500 \
  --save_checkpoints_steps=2000 \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME \
  --learning_rate=2e-5

# WB -> Src+Tgt -> Tgt
EXP_NAME=BERT_srctgtlm_tgtlm_${domain}
INIT=$EXPTS_DIR/BERT_srctgtlm/model.ckpt-10000
INPUT_FILE=$TFRecords/$split/${domain}.tfrecord

python3 run_pretraining.py \
  --input_file=$INPUT_FILE \
  --output_dir=$EXPTS_DIR/$EXP_NAME \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT \
  --train_batch_size=256 \
  --max_seq_length=256 \
  --num_train_steps=10000 \
  --num_warmup_steps=500 \
  --save_checkpoints_steps=2000 \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME \
  --learning_rate=2e-5
