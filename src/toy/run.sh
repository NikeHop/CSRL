algorithm=$1
n_seeds=$2


source ./env/bin/activate

python main.py --config_path ./configs/ \
                --n_seeds $n_seeds \
                --algorithm $algorithm                                                                                                  









