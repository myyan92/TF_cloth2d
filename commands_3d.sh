python3 train_eval_3d.py train --gin_config gin_configs/b-spline_data_pred_node_STN_2_3d.gin \
 > train_log.txt
mv train_log.txt b-spline_data_pred_node_STN_2_3d/train_log.txt

python3 train_eval_3d.py train --gin_config gin_configs/b-spline_data_pred_node_STN_2_3d.gin \
 --gin_bindings 'Model_STNv2_3d.fc_sizes = [1024, 256]' \
 --gin_bindings 'SAVE_DIR="b-spline_data_pred_node_STN_2_3d_smaller"' \
 > train_log.txt
mv train_log.txt b-spline_data_pred_node_STN_2_3d_smaller/train_log.txt

python3 train_eval_3d.py train --gin_config gin_configs/b-spline_data_pred_node_STN_2_3d.gin \
 --gin_bindings 'Model_STNv2_3d.fc_sizes = [1024, 1024, 256, 256]' \
 --gin_bindings 'SAVE_DIR="b-spline_data_pred_node_STN_2_3d_larger"' \
 > train_log.txt
mv train_log.txt b-spline_data_pred_node_STN_2_3d_larger/train_log.txt

python3 train_eval_3d.py train --gin_config gin_configs/b-spline_data_pred_node_STN_2_3d.gin \
 --gin_bindings 'Model_STNv2_3d.finetune_vgg=True'  \
 --gin_bindings 'Trainner.snapshot="b-spline_data_pred_node_STN_2_3d/model-15"' \
 --gin_bindings 'SAVE_DIR="b-spline_data_pred_node_STN_2_3d_finetune"' \
 > train_log.txt
mv train_log.txt b-spline_data_pred_node_STN_2_3d_finetune/train_log.txt

