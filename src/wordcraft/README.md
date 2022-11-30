# Experiments in the Wordcraft environment

The code is adapted from the [GitHub](https://github.com/minqi/wordcraft) (accompanying the paper [WordCraft: An Environment for Benchmarking Commonsense Agents](https://larel-ws.github.io/assets/pdfs/wordcraft_an_environment_for_benchmarking_commonsense_agents.pdf)).
To run the code you must have set up the virtual environment as explained in the main [README.md](https://github.com/NikeHop/Temp/blob/master/README.md).

## Dependencies 

Prerequisites: python3.8, venv

From inside the wordcraft directory run the following code.

```
python3 -m venv ./env
source ./env/bin/activate
pip install -r requirements.txt
```

## Experiments

Three types of experiments were performed in the paper. For all scripts the arguments are: 

* Argument 1: #seeds
* Argument 2: type of method (residual/sum/baseline)
* Argument 3: experiment name

### Generalise to unseen goal - 1 distractor 

bash goal.sh 5 residual unseen_goal_1

### Generalise to unseen recipes - 8 distractors

bash recipe.sh 5 residual unseen_recipe_8

### Generalise to unseen goals - 1 distractor - Low data 

bash low_data.sh 5 residual low_data_goal_1

The logs of the experiments can be found in the ./log directory. 