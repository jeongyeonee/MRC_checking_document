#!/bin/bash

MODEL_TYPE="kobert"
MODEL_NAME="monologg/kobert"
DATA_DIR="../../data/"
BATCH_SIZE=16
PRED_FILE="data_valid_230406.json"
EPOCHS=30


TRAIN_FILE="data_train_230406.json"
OUTPUT_DIR="../model/230424/opt7_all_EPOCH30/"
STEP=100
#NOANS_PROP=0
python train_korquality.py --model_type ${MODEL_TYPE} \
                     --model_name_or_path ${MODEL_NAME} \
                     --output_dir ${OUTPUT_DIR} \
                     --data_dir ${DATA_DIR} \
                     --train_file ${TRAIN_FILE} \
                     --predict_file ${PRED_FILE} \
                     --evaluate_during_training \
                     --per_gpu_train_batch_size ${BATCH_SIZE} \
                     --per_gpu_eval_batch_size ${BATCH_SIZE} \
                     --max_seq_length 512 \
                     --logging_steps ${STEP} \
                     --save_steps ${STEP} \
                     --do_train \
                     --num_train_epochs ${EPOCHS} \
                     --version_2_with_negative 
                     #--noans_prop ${NOANS_PROP}
                     
TRAIN_FILE="data_train_230406_1by1.json"
OUTPUT_DIR="../model/230424/opt8_1by1_EPOCH30/"
STEP=50
#NOANS_PROP=0
python train_korquality.py --model_type ${MODEL_TYPE} \
                     --model_name_or_path ${MODEL_NAME} \
                     --output_dir ${OUTPUT_DIR} \
                     --data_dir ${DATA_DIR} \
                     --train_file ${TRAIN_FILE} \
                     --predict_file ${PRED_FILE} \
                     --evaluate_during_training \
                     --per_gpu_train_batch_size ${BATCH_SIZE} \
                     --per_gpu_eval_batch_size ${BATCH_SIZE} \
                     --max_seq_length 512 \
                     --logging_steps ${STEP} \
                     --save_steps ${STEP} \
                     --do_train \
                     --num_train_epochs ${EPOCHS} \
                     --version_2_with_negative 
                     #--noans_prop ${NOANS_PROP}
                     
TRAIN_FILE="data_train_230406_extract.json"
OUTPUT_DIR="../model/230424/opt9_extract_EPOCH30/"
STEP=20
#NOANS_PROP=0
python train_korquality.py --model_type ${MODEL_TYPE} \
                     --model_name_or_path ${MODEL_NAME} \
                     --output_dir ${OUTPUT_DIR} \
                     --data_dir ${DATA_DIR} \
                     --train_file ${TRAIN_FILE} \
                     --predict_file ${PRED_FILE} \
                     --evaluate_during_training \
                     --per_gpu_train_batch_size ${BATCH_SIZE} \
                     --per_gpu_eval_batch_size ${BATCH_SIZE} \
                     --max_seq_length 512 \
                     --logging_steps ${STEP} \
                     --save_steps ${STEP} \
                     --do_train \
                     --num_train_epochs ${EPOCHS} \
                     --version_2_with_negative 
                     #--noans_prop ${NOANS_PROP}
                     


TRAIN_FILE="data_train_230406.json"
OUTPUT_DIR="../model/230424/opt10_chunk_all_EPOCH30/"
STEP=100
python train_korquality.py --model_type ${MODEL_TYPE} \
                     --model_name_or_path ${MODEL_NAME} \
                     --output_dir ${OUTPUT_DIR} \
                     --data_dir ${DATA_DIR} \
                     --train_file ${TRAIN_FILE} \
                     --predict_file ${PRED_FILE} \
                     --evaluate_during_training \
                     --per_gpu_train_batch_size ${BATCH_SIZE} \
                     --per_gpu_eval_batch_size ${BATCH_SIZE} \
                     --max_seq_length 512 \
                     --logging_steps ${STEP} \
                     --save_steps ${STEP} \
                     --do_train \
                     --num_train_epochs ${EPOCHS} \
                     --version_2_with_negative \
                     --noans_prop ${NOANS_PROP}
                     
OUTPUT_DIR="../model/230424/opt11_chunk_1by1_EPOCH30/"
STEP=50
NOANS_PROP=1
python train_korquality.py --model_type ${MODEL_TYPE} \
                     --model_name_or_path ${MODEL_NAME} \
                     --output_dir ${OUTPUT_DIR} \
                     --data_dir ${DATA_DIR} \
                     --train_file ${TRAIN_FILE} \
                     --predict_file ${PRED_FILE} \
                     --evaluate_during_training \
                     --per_gpu_train_batch_size ${BATCH_SIZE} \
                     --per_gpu_eval_batch_size ${BATCH_SIZE} \
                     --max_seq_length 512 \
                     --logging_steps ${STEP} \
                     --save_steps ${STEP} \
                     --do_train \
                     --num_train_epochs ${EPOCHS} \
                     --version_2_with_negative \
                     --noans_prop ${NOANS_PROP}
                     
OUTPUT_DIR="../model/230424/opt12_chunk_extract_EPOCH30/"
STEP=20
NOANS_PROP=0
python train_korquality.py --model_type ${MODEL_TYPE} \
                     --model_name_or_path ${MODEL_NAME} \
                     --output_dir ${OUTPUT_DIR} \
                     --data_dir ${DATA_DIR} \
                     --train_file ${TRAIN_FILE} \
                     --predict_file ${PRED_FILE} \
                     --evaluate_during_training \
                     --per_gpu_train_batch_size ${BATCH_SIZE} \
                     --per_gpu_eval_batch_size ${BATCH_SIZE} \
                     --max_seq_length 512 \
                     --logging_steps ${STEP} \
                     --save_steps ${STEP} \
                     --do_train \
                     --num_train_epochs ${EPOCHS} \
                     --version_2_with_negative \
                     --noans_prop ${NOANS_PROP}