export CUDA_VISIBLE_DEVICES=0
python3 runner.py train --gin_config b-spline_data_pred_node_STN.gin \
 > train_log_1.txt &

export CUDA_VISIBLE_DEVICES=1
python3 runner.py train --gin_config b-spline_data_pred_node_STN_2.gin \
 > train_log_2.txt &

wait

mv train_log_1.txt ./train_simseq_pretrain_bspline_STN/train_log.txt
mv train_log_2.txt ./train_simseq_pretrain_bspline_STNv2/train_log.txt

shutdown -h +5
