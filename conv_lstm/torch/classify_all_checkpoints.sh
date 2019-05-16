#!/bin/bash

# =========
# This script needs to be invoked via bash and not zsh. 
# e.g. /bin/bash classify_all_checkpoints.sh
# =========

GPU_ID=0
# DATA_DIR='../../openface_data/face_gestures/dataseto_text/correct_gests_38'
DATA_DIR='../../data_cardiff/openface_h5'
# TRAIN_SEQ_H5='../../openface_data/gest_seq_38/ang_velocity_1.h5'
TRAIN_SEQ_H5='../../data_cardiff/gest_list_1.h5'
#MODEL_DIR='./results_multi_scale_conv_lstm/ang_velocity/exp1_norm_landmarks_non_max_supp_lr_0001'
MODEL_DIR='/file3/mohit/final_experiments/cardiff_num_scales_3'
ZFACE_H5_DIR='../../data_cardiff/zface_h5'
OPENFACE_MEAN_H5='../../data_cardiff/openface_mean_std_cache.h5'

CHECKPOINT_FILES=`find "$MODEL_DIR" -name "*.t7"`
for f in $CHECKPOINT_FILES
do
  echo $f
  # TODO(Mohit): This command might need to change if `f` has `_` in its name
  cp_num=`echo $f | cut -d '_' -f6 | cut -d'.' -f1`
  echo $cp_num
  if [[ $cp_num -gt "0" ]] && [[ $cp_num -lt "51" ]]
  then
    echo 'Evaluating ', $f
    cp_save_dir="$MODEL_DIR/pred_cp_$cp_num"
    comm="CUDA_VISIBLE_DEVICES=$GPU_ID th classify.lua \
      -data_dir '$DATA_DIR' \
      -zface_h5_dir '$ZFACE_H5_DIR' \
      -train_seq_h5 '$TRAIN_SEQ_H5' \
      -openface_mean_h5 '$OPENFACE_MEAN_H5' \
      -init_from '$f' \
      -use_cpm_features 0 \
      -save '$cp_save_dir'"
    echo $comm
    eval $comm
  fi
done

