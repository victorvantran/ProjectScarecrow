TFLITE_OUTPUT="C:/tensorflow1/models/research/object_detection/cat.tflite"
FROZEN_GRAPH_INPUT="C:/tensorflow1/models/research/object_detection/inference_graph/tflite_graph.pb"
INPUT_TENSORS="normalized_input_image_tensor"
OUTPUT_TENSORS="TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3"


tflite_convert \
  --output_file=$TFLITE_OUTPUT \
  --graph_def_file=$FROZEN_GRAPH_INPUT \
  --enable_v1_converter \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays=$INPUT_TENSORS \
  --output_arrays=$OUTPUT_TENSORS \
  --mean_values=128 \
  --std_dev_values=128 \
  --input_shapes=1,300,300,3 \
  --change_concat_input_ranges=false \
  --allow_nudging_weights_to_use_fast_gemm_kernel=true \
  --allow_custom_ops \
  --default_ranges_min=0 \
  --default_ranges_max=6



tflite_convert \
  --output_file="C:/tensorflow1/models/research/object_detection/cat.tflite" \
  --graph_def_file="C:/tensorflow1/models/research/object_detection/inference_graph/tflite_graph.pb" \
  --enable_v1_converter \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays="normalized_input_image_tensor" \
  --output_arrays="TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3" \
  --mean_values=128 \
  --std_dev_values=128 \
  --input_shapes=1,300,300,3 \
  --change_concat_input_ranges=false \
  --allow_nudging_weights_to_use_fast_gemm_kernel=true \
  --allow_custom_ops \
  --default_ranges_min=-6 \
  --default_ranges_max=6