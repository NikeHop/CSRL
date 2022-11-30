# Experiments in the TextWorld Commonsense Environment

The code was adapted to support state abstraction from the [Tetxworld Commonsense GitHub](https://github.com/IBM/commonsense-rl/), related to the paper [Text-based RL Agents with Commonsense Knowledge: New Challenges, Environments and Baselines.](https://arxiv.org/abs/2010.03790).

## Dependencies

Prerequisites: python3.8, venv

Run the following lines in the textworld directory. 

```
python3 -m venv ./env
source ./bin/env/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

## Games

The games for the experiments can be found in the ./games/class_games_distractors/easy directory. There are 90 training games and 10 validation and 10 test games. The directory also contains the `entities.txt` file with the game entities and for each split a `conceptnet_subgraph.txt` file which contains the subpart of ConceptNet that represents the LocatedAt-relations of the game entities.

## Embeddings

Two types of embeddings were used for the experiments: [Numberbatch embeddings](https://github.com/commonsense/conceptnet-numberbatch) and [hyperbolic embeddings](https://hazyresearch.stanford.edu/hyperE/) trained on WordNet. They can be downloaded as follows:

```
mkdir embeddings
cd embeddings 
wget https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
gzip -d numberbatch-en-19.08.txt.gz
wget https://cdn-133.anonfiles.com/0dddp5F9y1/e62c166a-1667213299/hypernym_noun.100d.txt
```

## Knowledge Graphs

The experiments looked at three different knowledge graphs: Wordnet, DBpedia & ConceptNet.

* Knowledge from WordNet is obtained via nltk
* The DBpedia knowledge is obtained by querying a SPARQL endpoint
* For ConceptNet, the ./kgs directory contains a csv-file with all relevant IsA-relations. Since each node in ConceptNet might have multiple 'IsA'-relations the size of the subclass tree grows exponentially with the number of layers. To construct the csv-file a three layer hierarchy with a maximum of five 'IsA'-relations per element was considered. 

The constructed knowledge graphs will be saved to the ./kgs directory.

## Running experiments

**To run an agent with abstraction:**

bash run_simple.sh $1 $2 $3 $4 $5 $6 $7

* $1: number of seeds 
* $2: which run to start at (starting from 0)
* $3: residual or sum algorithm (residual/sum)
* $4: prior knowledge type (conceptnet/wordnet/dbpedia/manual/own)
* $5: number of abstractions 
* $6: use additional hyperbolic embeddings (true/false)
* $7: whether to run on train/valid/test dataset

The number of abstractions used per type of knowledge graph are:

* ConceptNet: 3
* DBpedia: 6
* WordNet: 7
* Manual: 1 (No abstraction, words are simply copied)
* Own: 2 (Self-defined abstraction)

E.g. the command to run the agent with WordNet prior knowledge and the residual algorithm for 5 seeds starting with the first run is:

bash run_simple.sh 5 0 residual wordnet 7 false train

**To run the agent that takes a subgraph of ConceptNet as input:**

bash run_knowledgeaware.sh $1 $2 $3

* $1: number of seeds
* $2: which run to start at (starting from 0)
* $3: whether to run on train/valid/test dataset

