python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node.gin \
--gin_bindings 'Trainner.train_dataset="./datasets/cloth2d_train_sim_seq_small.tfrecords"' \
--gin_bindings 'Trainner.eval_dataset="./datasets/cloth2d_test_sim_seq.tfrecords"' \
--gin_bindings 'SAVE_DIR="./simseq_data_pred_node_small/"' \
--gin_bindings 'Trainner.num_epoch=500' \
 > train_log.txt
mv train_log.txt simseq_data_pred_node_small/train_log_small.txt

python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node_STN.gin \
--gin_bindings 'Trainner.train_dataset="./datasets/cloth2d_train_sim_seq_small.tfrecords"' \
--gin_bindings 'Trainner.eval_dataset="./datasets/cloth2d_test_sim_seq.tfrecords"' \
--gin_bindings 'SAVE_DIR="./simseq_data_pred_node_STN_small/"' \
--gin_bindings 'Trainner.num_epoch=500' \
 > train_log.txt
mv train_log.txt simseq_data_pred_node_STN_small/train_log_small.txt

python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node_STN_2.gin \
--gin_bindings 'Trainner.train_dataset="./datasets/cloth2d_train_sim_seq_small.tfrecords"' \
--gin_bindings 'Trainner.eval_dataset="./datasets/cloth2d_test_sim_seq.tfrecords"' \
--gin_bindings 'SAVE_DIR="./simseq_data_pred_node_STN_2_small/"' \
--gin_bindings 'Trainner.num_epoch=500' \
 > train_log.txt
mv train_log.txt simseq_data_pred_node_STN_2_small/train_log_small.txt

python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node.gin \
--gin_bindings 'Trainner.train_dataset="./datasets/cloth2d_train_sim_seq_medium.tfrecords"' \
--gin_bindings 'Trainner.eval_dataset="./datasets/cloth2d_test_sim_seq.tfrecords"' \
--gin_bindings 'SAVE_DIR="./simseq_data_pred_node_medium/"' \
--gin_bindings 'Trainner.num_epoch=50' \
 > train_log.txt
mv train_log.txt simseq_data_pred_node_medium/train_log_medium.txt

python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node_STN.gin \
--gin_bindings 'Trainner.train_dataset="./datasets/cloth2d_train_sim_seq_medium.tfrecords"' \
--gin_bindings 'Trainner.eval_dataset="./datasets/cloth2d_test_sim_seq.tfrecords"' \
--gin_bindings 'SAVE_DIR="./simseq_data_pred_node_STN_medium/"' \
--gin_bindings 'Trainner.num_epoch=50' \
 > train_log.txt
mv train_log.txt simseq_data_pred_node_STN_medium/train_log_medium.txt

python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node_STN_2.gin \
--gin_bindings 'Trainner.train_dataset="./datasets/cloth2d_train_sim_seq_medium.tfrecords"' \
--gin_bindings 'Trainner.eval_dataset="./datasets/cloth2d_test_sim_seq.tfrecords"' \
--gin_bindings 'SAVE_DIR="./simseq_data_pred_node_STN_2_medium/"' \
--gin_bindings 'Trainner.num_epoch=50' \
 > train_log.txt
mv train_log.txt simseq_data_pred_node_STN_2_medium/train_log_medium.txt

python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node.gin \
--gin_bindings 'Trainner.train_dataset="./datasets/cloth2d_train_sim_seq.tfrecords"' \
--gin_bindings 'Trainner.eval_dataset="./datasets/cloth2d_test_sim_seq.tfrecords"' \
--gin_bindings 'SAVE_DIR="./simseq_data_pred_node_large/"' \
--gin_bindings 'Trainner.num_epoch=20' \
 > train_log.txt
mv train_log.txt simseq_data_pred_node_large/train_log_large.txt

python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node_STN.gin \
--gin_bindings 'Trainner.train_dataset="./datasets/cloth2d_train_sim_seq.tfrecords"' \
--gin_bindings 'Trainner.eval_dataset="./datasets/cloth2d_test_sim_seq.tfrecords"' \
--gin_bindings 'SAVE_DIR="./simseq_data_pred_node_STN_large/"' \
--gin_bindings 'Trainner.num_epoch=20' \
 > train_log.txt
mv train_log.txt simseq_data_pred_node_STN_large/train_log_large.txt

python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node_STN_2.gin \
--gin_bindings 'Trainner.train_dataset="./datasets/cloth2d_train_sim_seq.tfrecords"' \
--gin_bindings 'Trainner.eval_dataset="./datasets/cloth2d_test_sim_seq.tfrecords"' \
--gin_bindings 'SAVE_DIR="./simseq_data_pred_node_STN_2_large/"' \
--gin_bindings 'Trainner.num_epoch=20' \
 > train_log.txt
mv train_log.txt simseq_data_pred_node_STN_2_large/train_log_large.txt

