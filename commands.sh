#python3 runner.py train --gin_config gin_configs/b-spline_data_pred_knot_nodeloss.gin \
# > train_log.txt
#mv train_log.txt b-spline_data_pred_knot_nodeloss/

#python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node.gin \
# > train_log.txt
#mv train_log.txt b-spline_data_pred_node/

python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node_STN.gin \
--gin_bindings 'Model_STN.learning_rate=0.0001' \
 > train_log.txt
mv train_log.txt b-spline_data_pred_node_STN/train_log_0.0001.txt

python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node_STN.gin \
--gin_bindings 'Model_STN.learning_rate=0.0001' \
--gin_bindings 'Model_STN.momentum=0.5' \
 > train_log.txt
mv train_log.txt b-spline_data_pred_node_STN/train_log_0.0001_0.5.txt

#python3 runner.py train --gin_config gin_configs/b-spline_data_pred_knot_nodeloss.gin \
#--gin_bindings 'Trainner.train_dataset="./datasets/cloth2d_train_action2.tfrecords"' \
#--gin_bindings 'Trainner.eval_dataset="./datasets/cloth2d_test_action2.tfrecords"' \
#--gin_bindings 'SAVE_DIR="./rollout_data_pred_knot_nodeloss/"' \
# > train_log.txt
#mv train_log.txt rollout_data_pred_knot_nodeloss/

python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node.gin \
--gin_bindings 'Trainner.train_dataset="./datasets/cloth2d_train_action2.tfrecords"' \
--gin_bindings 'Trainner.eval_dataset="./datasets/cloth2d_test_action2.tfrecords"' \
--gin_bindings 'SAVE_DIR="./rollout_data_pred_node/"' \
--gin_bindings 'Model.learning_rate=0.0001' \
 > train_log.txt
mv train_log.txt rollout_data_pred_node/train_log_0.0001.txt

python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node.gin \
--gin_bindings 'Trainner.train_dataset="./datasets/cloth2d_train_action2.tfrecords"' \
--gin_bindings 'Trainner.eval_dataset="./datasets/cloth2d_test_action2.tfrecords"' \
--gin_bindings 'SAVE_DIR="./rollout_data_pred_node/"' \
--gin_bindings 'Model.learning_rate=0.0001' \
--gin_bindings 'Model.momentum=0.5' \
 > train_log.txt
mv train_log.txt rollout_data_pred_node/train_log_0.0001_0.5.txt

python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node_STN.gin \
--gin_bindings 'Trainner.train_dataset="./datasets/cloth2d_train_action2.tfrecords"' \
--gin_bindings 'Trainner.eval_dataset="./datasets/cloth2d_test_action2.tfrecords"' \
--gin_bindings 'SAVE_DIR="./rollout_data_pred_node_STN/"' \
--gin_bindings 'Model_STN.learning_rate=0.0001' \
 > train_log.txt
mv train_log.txt rollout_data_pred_node_STN/train_log_0.0001.txt

python3 runner.py train --gin_config gin_configs/b-spline_data_pred_node_STN.gin \
--gin_bindings 'Trainner.train_dataset="./datasets/cloth2d_train_action2.tfrecords"' \
--gin_bindings 'Trainner.eval_dataset="./datasets/cloth2d_test_action2.tfrecords"' \
--gin_bindings 'SAVE_DIR="./rollout_data_pred_node_STN/"' \
--gin_bindings 'Model_STN.learning_rate=0.0001' \
--gin_bindings 'Model_STN.momentum=0.5' \
 > train_log.txt
mv train_log.txt rollout_data_pred_node_STN/train_log_0.0001_0.5.txt

