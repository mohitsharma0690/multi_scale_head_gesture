FIPCO_DATA_DIR='../../../Fipco_data'

VIDEO_FILES=`find "$FIPCO_DATA_DIR" -name "*.mp4"`
for f in $VIDEO_FILES
do
  echo $f
  cp_num=`echo $f | cut -d '/' -f5`
  echo $cp_num
  echo 'Evaluating ' $f $cp_num
  json_save_dir="$FIPCO_DATA_DIR/$cp_num/json"
  jpeg_save_dir="$FIPCO_DATA_DIR/$cp_num/jpeg"
  mkdir -p "$FIPCO_DATA_DIR/$cp_num"
  mkdir_json_comm="mkdir -p $json_save_dir"
  mkdir_jpeg_comm="mkdir -p $jpeg_save_dir"
  eval $mkdir_json_comm
  eval $mkdir_jpeg_comm

  comm="../../../caffe_rtpose/build/examples/rtpose/rtpose.bin \
        --video '$f' \
        --no_display \
        --num_gpu 2 \
        --no_frame_drops \
        --write_json '$json_save_dir' \
        --write_frames '$jpeg_save_dir'"
  echo $comm
  eval $comm
done

