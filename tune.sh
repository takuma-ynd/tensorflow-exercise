#!/bin/bash
echo "log file path?(default:tune.log): "
DEFAULT_LOG_FILE="tune.log"
TENSORBOARD_LOG_FILE="./tensorboard"
read LOG_FILE
if [[ $LOG_FILE == "" ]]
then
    LOG_FILE=$DEFAULT_LOG_FILE
fi

# array=(sth0 sth1 sth2) <-- bashにおける配列の定義方法
batch_size_list=(1 5 10 50)
embeddings_size_list=(50 100 200 400)
learning_rate_list=(0.0001 0.001 0.01)

# exp( -4 * log(10)) == 10^(-4)
# これをbc -l にパイプで渡して計算させる
l2_coef_list=($(echo "e(-4 * l(10))" | bc -l) $(echo "e(-5 * l(10))" | bc -l) $(echo "e(-6 * l(10))" | bc -l))
dropout_rate_list=(0.0 0.25 0.5)

# ${array[@]} --> (sth0 sth1 sth2)
# $array --> sth0

for batch_size in ${batch_size_list[@]}; do
    for embeddings_size in ${embeddings_size_list[@]}; do
    	for learning_rate in ${learning_rate_list[@]}; do
    	    for l2_coef in ${l2_coef_list[@]}; do
    	        for dropout_rate in ${dropout_rate_list[@]}; do
		    echo "----------TRAINING----------"
    		    echo "batch_size:$batch_size"
    		    echo "embeddings_size:$embeddings_size"
    		    echo "learning_rate:$learning_rate"
    		    echo "l2_coef:$l2_coef"
    		    echo "dropout_rate:$dropout_rate"
		    echo ""
		    python3 logreg_minibatch.py train.txt devel.txt --batch-size $batch_size --dim $embeddings_size --learning-rate $learning_rate --l2-coef $l2_coef --dropout-rate $dropout_rate --logdir $TENSORBOARD_LOG_FILE --eval-logfile $LOG_FILE --eval-log
    		done
    	    done
    	done
    done
done
