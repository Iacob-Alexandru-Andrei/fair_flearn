export CUDA_VISIBLE_DEVICES=GPU:0
python3  -u main.py --dataset=$1 --optimizer=$2  \
            --learning_rate=0.01 \
            --learning_rate_lambda=0.01 \
            --num_rounds=20 \
            --eval_every=1 \
            --clients_per_round=10 \
            --batch_size=64 \
            --q=$4 \
            --model='svm' \
            --sampling=$5  \
            --num_epochs=1 \
            --data_partition_seed=$3 \
            --log_interval=10 \
            --static_step_size=0 \
            --track_individual_accuracy=0 \
            --output="./log_$1/$2_samp$5_run$3_q$4"