#! /bin/bash

source ./env/bin/activate

n_seeds=$1 
start_run=$2
dataset=$3

if [ "$dataset" = "train" ]
then
python -u train_agent.py --agent_type knowledgeaware \
                        --game_dir ./games/class_games_distractors\
                        --game_name *.ulx \
                        --difficulty_level easy \
                        --residual 'sum' \
                        --prior_kg 'manual' \
                        --graph_type world \
                        --graph_mode evolve \
                        --graph_emb_type numberbatch \
                        --world_evolve_type "CDC" \
                        --word_emb_type numberbatch \
                        --n_abstractions 1 \
                        --cuda 0 \
                        --bidirectional \
                        --nruns $n_seeds \
                        --start_run $start_run \
                        --lr 0.00003 \
                        --beta 0.0001 \
                        --token_extractor 'max' 
elif [ "$dataset" = "valid" ]
then
python -u test_agent.py --agent_type knowledgeaware \
                        --game_dir ./games/class_games_distractors\
                        --game_name *.ulx \
                        --difficulty_level easy \
                        --residual 'sum' \
                        --prior_kg 'manual' \
                        --graph_type world \
                        --graph_mode evolve \
                        --graph_emb_type numberbatch \
                        --world_evolve_type "CDC" \
                        --word_emb_type numberbatch \
                        --n_abstractions 1 \
                        --cuda 0 \
                        --bidirectional \
                        --nruns $n_seeds \
                        --start_run $start_run \
                        --lr 0.00003 \
                        --beta 0.0001 \
                        --token_extractor 'max' \
                        --split "valid" 
else
python -u test_agent.py --agent_type knowledgeaware \
                        --game_dir ./games/class_games_distractors\
                        --game_name *.ulx \
                        --difficulty_level easy \
                        --residual 'sum' \
                        --prior_kg 'manual' \
                        --graph_type world \
                        --graph_mode evolve \
                        --graph_emb_type numberbatch \
                        --world_evolve_type "CDC" \
                        --word_emb_type numberbatch \
                        --n_abstractions 1 \
                        --cuda 0 \
                        --bidirectional \
                        --nruns $n_seeds \
                        --start_run $start_run \
                        --lr 0.00003 \
                        --beta 0.0001 \
                        --token_extractor 'max' \
                        --split "test"
fi