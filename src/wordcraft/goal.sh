#! /bin/bash
n_seeds=$1
method=$2
experiment_name=$3

source ./env/bin/activate


for i in $( seq 1 $n_seeds)
do
    if [[ $method == "residual" ]]
    then
        python -m run \
                --xpid "${experiment_name}_residual_${i}"  \
                --env 'wordcraft-multistep-goal-v0' \
                --split by_goal  \
                --train_ratio 0.8  \
                --depths 1    \
                --num_distractors 1 \
                --feature_type glove  \
                --discounting 0.99  \
                --reward_clipping none  \
                --learning_rate 0.0001   \
                --entropy_cost 0.1  \
                --unroll_length 2   \
                --total_steps 3000000   \
                --arch abstraction   \
                --num_actors 2  \
                --num_threads 4   \
                --batch_size 64   \
                --test_interval 50000   \
                --num_test_episodes 500  \
                --n_abstractions 9 \
                --collapse \
                --start_level 9 \
                --end_level 15 \
                --residual \
                --seed ${i}
    fi
    if [[ $method == "sum" ]]
    then
        python -m run \
                --xpid "${experiment_name}_sum_${i}"  \
                --env 'wordcraft-multistep-goal-v0' \
                --split by_goal  \
                --train_ratio 0.8  \
                --depths 1    \
                --num_distractors 8 \
                --feature_type glove  \
                --discounting 0.99  \
                --reward_clipping none  \
                --learning_rate 0.0001   \
                --entropy_cost 0.1  \
                --unroll_length 2   \
                --total_steps 3000000   \
                --arch abstraction   \
                --num_actors 2  \
                --num_threads 4   \
                --batch_size 64   \
                --test_interval 50000   \
                --num_test_episodes 500  \
                --n_abstractions 9 \
                --collapse \
                --start_level 9 \
                --end_level 15 \
                --seed {$i}
    fi
    if [[ $method == "baseline" ]]
    then
        python -m run \
                --xpid "${experiment_name}_baseline_${i}"  \
                --env 'wordcraft-multistep-goal-v0' \
                --split by_goal  \
                --train_ratio 0.8  \
                --depths 1    \
                --num_distractors 1 \
                --feature_type glove  \
                --discounting 0.99  \
                --reward_clipping none  \
                --learning_rate 0.0001   \
                --entropy_cost 0.1  \
                --unroll_length 2   \
                --total_steps 3000000   \
                --arch abstraction   \
                --num_actors 2  \
                --num_threads 4   \
                --batch_size 64   \
                --test_interval 50000   \
                --num_test_episodes 500  \
                --seed {$i}
    fi
done

