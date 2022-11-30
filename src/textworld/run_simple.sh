#! /bin/bash

source ./env/bin/activate 

n_seeds=$1
start_run=$2
residual=$3
prior_kg=$4
n_abstractions=$5
hyperbolic=$6
dataset=$7

if [ "$dataset" = "train" ]
then
    if ($hyperbolic)
    then
    python -u train_agent.py --agent_type simple \
                                --game_dir ./games/class_games_distractors \
                                --game_name *.ulx \
                                --difficulty_level easy \
                                --residual $residual \
                                --n_abstractions $n_abstractions  \
                                --bidirectional \
                                --cuda 0 \
                                --prior_kg $prior_kg \
                                --word_emb_type numberbatch \
                                --nruns $n_seeds \
                                --start_run $start_run \
                                --beta 0.0001 \
                                --lr 0.00003 \
                                --token_extractor 'max' \
                                --hyperbolic                           
    else
    python -u train_agent.py --agent_type simple \
                                --game_dir ./games/class_games_distractors \
                                --game_name *.ulx \
                                --difficulty_level easy \
                                --residual $residual \
                                --n_abstractions $n_abstractions  \
                                --bidirectional \
                                --cuda 0 \
                                --prior_kg $prior_kg \
                                --word_emb_type numberbatch \
                                --nruns $n_seeds \
                                --start_run $start_run \
                                --beta 0.0001 \
                                --lr 0.00003 \
                                --token_extractor 'max' 
    fi
elif [ "$dataset" = "valid" ]
then
    if ($hyperbolic)
    then
    python -u test_agent.py --agent_type simple \
                                --game_dir ./games/class_games_distractors \
                                --game_name *.ulx \
                                --difficulty_level easy \
                                --residual $residual \
                                --n_abstractions $n_abstractions  \
                                --bidirectional \
                                --cuda 0 \
                                --prior_kg $prior_kg \
                                --word_emb_type numberbatch \
                                --nruns $n_seeds \
                                --start_run $start_run \
                                --beta 0.0001 \
                                --lr 0.00003 \
                                --token_extractor 'max' \
                                --hyperbolic \
                                --split "valid"                         
    else
    python -u test_agent.py --agent_type simple \
                                --game_dir ./games/class_games_distractors \
                                --game_name *.ulx \
                                --difficulty_level easy \
                                --residual $residual \
                                --n_abstractions $n_abstractions  \
                                --bidirectional \
                                --cuda 0 \
                                --prior_kg $prior_kg \
                                --word_emb_type numberbatch \
                                --nruns $n_seeds \
                                --start_run $start_run \
                                --beta 0.0001 \
                                --lr 0.00003 \
                                --token_extractor 'max' \
                                --split "valid"
    fi
else
    if ($hyperbolic)
    then
    python -u test_agent.py --agent_type simple \
                                --game_dir ./games/class_games_distractors \
                                --game_name *.ulx \
                                --difficulty_level easy \
                                --residual $residual \
                                --n_abstractions $n_abstractions  \
                                --bidirectional \
                                --cuda 0 \
                                --prior_kg $prior_kg \
                                --word_emb_type numberbatch \
                                --nruns $n_seeds \
                                --start_run $start_run \
                                --beta 0.0001 \
                                --lr 0.00003 \
                                --token_extractor 'max' \
                                --hyperbolic \
                                --split "test"                           
    else
    python -u test_agent.py --agent_type simple \
                                --game_dir ./games/class_games_distractors \
                                --game_name *.ulx \
                                --difficulty_level easy \
                                --residual $residual \
                                --n_abstractions $n_abstractions  \
                                --bidirectional \
                                --cuda 0 \
                                --prior_kg $prior_kg \
                                --word_emb_type numberbatch \
                                --nruns $n_seeds \
                                --start_run $start_run \
                                --beta 0.0001 \
                                --lr 0.00003 \
                                --token_extractor 'max' \
                                --split "test" 
    fi
fi