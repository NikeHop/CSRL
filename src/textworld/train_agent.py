import argparse
import csv
import os
import random
from time import time

import numpy as np
import spacy
import tqdm
import torch
import pickle

import agent
from utils import extractor
from utils.format import format_settings
from utils.generic import getUniqueFileHandler
from utils.kg import construct_kg, load_manual_graphs, RelationExtractor
from utils.textworld_utils import get_goal_graph
from utils.nlp import Tokenizer
from games import dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(agent, opt, random_action=False):
    print("Starting play")
    # Prepare logging
    if not os.path.exists(opt.logs_dir):
        os.makedirs(opt.logs_dir)
    logfile = open(os.path.join(opt.logs_dir,opt.log_filename), "a+")
    logger = csv.DictWriter(
        logfile,
        fieldnames=[
            "episode",
            "average_train_reward",
            "average_train_steps",
            "average_entropy",
        ],
    )

    filter_examine_cmd = False
    infos_to_request = agent.infos_to_request
    infos_to_request.max_score = True  # Needed to normalize the scores.
    game_path = (
        opt.game_dir
        + "/"
        + (
            str(opt.difficulty_level) + "/" + opt.mode
            if opt.difficulty_level != ""
            else opt.game_dir + "/" + opt.mode
        )
    )

    manual_world_graphs = {}

    if opt.graph_emb_type and "world" in opt.graph_type:
        print("Loading Knowledge Graph ... ", end="")
        agent.kg_graph, _, _ = construct_kg(game_path + "/conceptnet_subgraph.txt")
        print(" DONE")

    # Creat the environment
    if opt.game_name:
        game_path = game_path + "/" + opt.game_name

    env, game_file_names = dataset.get_game_env(
        game_path,
        infos_to_request,
        opt.max_step_per_episode,
        opt.seed,
        opt.batch_size,
        opt.mode,
        opt.verbose,
    )

    # Collect some statistics: nb_steps, final reward.
    total_games_count = len(game_file_names)
    game_identifiers, avg_moves, avg_scores, avg_norm_scores, max_poss_scores = (
        [],
        [],
        [],
        [],
        [],
    )

    for no_episode in tqdm.tqdm(range(opt.nepisodes)):
        if not random_action:
            random.seed(no_episode)
            np.random.seed(no_episode)
            torch.manual_seed(no_episode)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(no_episode)
            env.seed(no_episode)

        # Reset values
        agent.start_episode(opt.batch_size)
        avg_eps_moves, avg_eps_scores, avg_eps_norm_scores = (
            [],
            [],
            [],
        )  # evaluation metrics
        num_games = opt.games_per_episode
        game_max_scores = []
        game_names = []

        # Per game loop
        while num_games > 0:
            obs, infos = env.reset()  # Start new episode.

            # Filter out examine and look command to reduce action space
            if filter_examine_cmd:
                for commands_ in infos[
                    "admissible_commands"
                ]:  # [open refri, take apple from refrigeration]
                    for cmd_ in [
                        cmd
                        for cmd in commands_
                        if cmd.split()[0] in ["examine", "look"]
                    ]:
                        commands_.remove(cmd_)

            # Record values per game
            batch_size = len(obs)
            num_games -= len(obs)
            game_goal_graphs = [None] * batch_size 
            max_scores = []
            game_ids = []
            game_manual_world_graph = [
                None
            ] * batch_size  
            
            # Working with the game-infos
            for b, game in enumerate(infos["game"]):
                max_scores.append(
                    game.max_score
                )  # Record the max score for the game - important for normalization
                if "uuid" in game.metadata:
                    game_id = game.metadata["uuid"].split("-")[-1]
                    game_ids.append(game_id)
                    game_names.append(game_id)
                    game_max_scores.append(game.max_score)

            if not game_ids:
                game_ids = range(num_games, num_games + batch_size)
                game_names.extend(game_ids)

            # Set parameters for this step
            commands = ["restart"] * len(obs)
            scored_commands = [[] for b in range(batch_size)]
            last_scores = [0.0] * len(obs)
            scores = [0.0] * len(obs)
            dones = [False] * len(obs)
            nb_moves = [0] * len(obs)
            agent.reset_parameters(opt.batch_size)
            episode_entropies = []

            for step_no in range(opt.max_step_per_episode):
                nb_moves = [
                    step + int(not done) for step, done in zip(nb_moves, dones)
                ]  # nb-moves ?
                if agent.graph_emb_type and (
                    "local" in agent.graph_type or "world" in agent.graph_type
                ):
                    # prune_nodes = opt.prune_nodes if no_episode >= opt.prune_episode and no_episode % 25 ==0 and step_no % 10 == 0 else False
                    prune_nodes = opt.prune_nodes
                    agent.update_current_graph(
                        obs,
                        commands,
                        scored_commands,
                        infos,
                        opt.graph_mode,
                        prune_nodes,
                    )

                commands, entropy = agent.act(
                    obs, scores, dones, infos, scored_commands, random_action
                )
                episode_entropies.append(entropy)
                obs, scores, dones, infos = env.step(commands)

                for b in range(batch_size):
                    if scores[b] - last_scores[b] > 0:
                        last_scores[b] = scores[b]
                        scored_commands[b].append(commands[b])

                if all(dones):
                    break
                if step_no == opt.max_step_per_episode - 1:
                    dones = [True for _ in dones]

            agent.act(
                obs, scores, dones, infos, scored_commands, random_action
            )  # Let the agent know the game is done.
            

            if opt.verbose:
                # One episode, one game, print the number of steps needed and the achieved score
                print(f"Number of moves needed {nb_moves}")
                print(f"Scores {scores}")
                print(".", end="")
            avg_eps_moves.extend(nb_moves)
            avg_eps_scores.extend(scores)
            avg_eps_norm_scores.extend(
                [score / max_score for score, max_score in zip(scores, max_scores)]
            )

        if opt.verbose:
            print("*", end="")
        agent.end_episode()
        game_identifiers.append(game_names)
        avg_moves.append(avg_eps_moves)  # episode x # games
        avg_scores.append(avg_eps_scores)
        avg_norm_scores.append(avg_eps_norm_scores)
        max_poss_scores.append(game_max_scores)

        # What do we want to log for each episode
        episode_dict = {
            "episode": no_episode,
            "average_train_reward": np.mean(avg_eps_norm_scores),
            "average_train_steps": np.mean(avg_eps_moves),
            "average_entropy": torch.mean(torch.tensor(episode_entropies)).item(),
        }

        logger.writerow(episode_dict)
        logfile.flush()

    env.close()
    game_identifiers = np.array(game_identifiers)
    avg_moves = np.array(avg_moves)
    avg_scores = np.array(avg_scores)
    avg_norm_scores = np.array(avg_norm_scores)
    max_poss_scores = np.array(max_poss_scores)
    if opt.verbose:
        idx = np.apply_along_axis(np.argsort, axis=1, arr=game_identifiers)
        game_avg_moves = np.mean(
            np.array(list(map(lambda x, y: y[x], idx, avg_moves))), axis=0
        )
        game_norm_scores = np.mean(
            np.array(list(map(lambda x, y: y[x], idx, avg_norm_scores))), axis=0
        )
        game_avg_scores = np.mean(
            np.array(list(map(lambda x, y: y[x], idx, avg_scores))), axis=0
        )

        msg = "\nGame Stats:\n-----------\n" + "\n".join(
            "  Game_#{} = Score: {:5.2f} Norm_Score: {:5.2f} Moves: {:5.2f}/{}".format(
                game_no, avg_score, norm_score, avg_move, opt.max_step_per_episode
            )
            for game_no, (norm_score, avg_score, avg_move) in enumerate(
                zip(game_norm_scores, game_avg_scores, game_avg_moves)
            )
        )

        print(msg)

        total_avg_moves = np.mean(game_avg_moves)
        total_avg_scores = np.mean(game_avg_scores)
        total_norm_scores = np.mean(game_norm_scores)
        msg = (
            opt.mode
            + " stats: avg. score: {:4.2f}; norm. avg. score: {:4.2f}; avg. steps: {:5.2f}; \n"
        )
        print(msg.format(total_avg_scores, total_norm_scores, total_avg_moves))

        ## Dump log files ......
        str_result = {
            opt.mode + "game_ids": game_identifiers,
            opt.mode + "max_scores": max_poss_scores,
            opt.mode + "scores_runs": avg_scores,
            opt.mode + "norm_score_runs": avg_norm_scores,
            opt.mode + "moves_runs": avg_moves,
        }
        if not os.path.exists(opt.results_dir):
            os.makedirs(opt.results_dir)
        results_ofile = getUniqueFileHandler(
            opt.results_filename + "_" + opt.mode + "_results"
        )
        pickle.dump(str_result, results_ofile)
    return avg_scores, avg_norm_scores, avg_moves


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser(add_help=False)

    # game files and other directories
    parser.add_argument(
        "--game_dir",
        default="./games/twc",
        help="Location of the game e.g ./games/testbed",
    )
    parser.add_argument(
        "--games_per_episode",
        type=int,
        default=5,
        help="How many games to play per episode",
    )
    parser.add_argument(
        "--game_name",
        help="Name of the game file e.g., kitchen_cleanup_10quest_1.ulx, *.ulx, *.z8",
    )
    parser.add_argument(
        "--results_dir", default="./results", help="Path to the results files"
    )
    parser.add_argument("--logs_dir", default="./logs", help="Path to the logs files")
    parser.add_argument("--cuda", default=0, type=int, help="Which gpu to use")

    # optional arguments (if game_name is given) for game files
    parser.add_argument(
        "--batch_size", type=int, default="1", help="Number of the games per batch"
    )
    parser.add_argument(
        "--difficulty_level",
        default="easy",
        choices=["easy", "medium", "hard"],
        help="difficulty level of the games",
    )

    # Experiments
    parser.add_argument("--initial_seed", type=int, default=42)
    parser.add_argument("--nruns", type=int, default=10)
    parser.add_argument("--start_run", type=int, default=0)
    parser.add_argument("--no_train_episodes", type=int, default=100)
    parser.add_argument("--no_eval_episodes", type=int, default=5)
    parser.add_argument("--train_max_step_per_episode", type=int, default=50)
    parser.add_argument("--eval_max_step_per_episode", type=int, default=50)
    parser.add_argument("--verbose", action="store_true", default=True)

    parser.add_argument("--emb_size", type=int, default=300, help="embedding dimension")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=300,
        help="num of hidden units for embeddings",
    )
    parser.add_argument(
        "--hist_scmds_size",
        type=int,
        default=3,
        help="Number of recent scored command history to use. Useful when the game has intermediate reward.",
    )
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument(
        "--token_extractor",
        default="max_bag_of_words",
        help="token extractor: (any or max)",
    )
    parser.add_argument(
        "--corenlp_url",
        default="http://localhost:9000/",
        help="URL for Stanford CoreNLP OpenIE Server for the relation extraction for the local graph",
    )

    parser.add_argument(
        "--noun_only_tokens",
        action="store_true",
        default=False,
        help=" Allow only noun for the token extractor",
    )
    parser.add_argument(
        "--use_stopword",
        action="store_true",
        default=False,
        help=" Use stopwords for the token extractor",
    )
    parser.add_argument(
        "--agent_type",
        default="knowledgeaware",
        choices=["random", "simple", "knowledgeaware"],
        help="Agent type for the text world: (random, simple, knowledgeable)",
    )
    parser.add_argument(
        "--graph_type",
        default="",
        choices=["", "local", "world"],
        help="What type of graphs to be generated",
    )
    parser.add_argument(
        "--graph_mode",
        default="evolve",
        choices=["full", "evolve"],
        help="Give Full ground truth graph or evolving knowledge graph: (full, evolve)",
    )
    parser.add_argument(
        "--local_evolve_type",
        default="direct",
        choices=["direct", "ground"],
        help="Type of the generated/evolving strategy for local graph",
    )
    parser.add_argument(
        "--world_evolve_type",
        default="cdc",
        choices=["DC", "CDC", "NG", "NG+prune", "manual"],
        help="Type of the generated/evolving strategy for world graph",
    )
    parser.add_argument(
        "--prune_nodes",
        action="store_true",
        default=False,
        help=" Allow pruning of low-probability nodes in the world-graph",
    )
    parser.add_argument(
        "--prune_start_episode",
        type=int,
        default=1,
        help="Starting the pruning from this episode",
    )

    # Embeddings
    parser.add_argument(
        "--emb_loc", default="embeddings/", help="Path to the embedding location"
    )
    parser.add_argument(
        "--word_emb_type",
        default="random",
        help="Embedding type for the observation and the actions: ..."
        "(random, glove, numberbatch, fasttext). Use utils.generic.load_embedings ..."
        " to take car of the custom embedding locations",
    )
    parser.add_argument(
        "--graph_emb_type",
        help="Knowledge Graph Embedding type for actions: (numberbatch, complex)",
    )
    parser.add_argument(
        "--egreedy_epsilon",
        type=float,
        default=0.0,
        help="Epsilon for the e-greedy exploration",
    )

    # Options related to the state abstraction
    parser.add_argument(
        "--prior_kg", type=str, default="wordnet", help="Type of prior knowledge to use"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to visualize the extracted class tree",
    )
    parser.add_argument(
        "--load_class_tree",
        action="store_true",
        default=False,
        help="Whether to load the extracted class tree or query it",
    )
    parser.add_argument(
        "--kg_data",
        type=str,
        default="./kgs",
        help="Directory that contains all data related to KGs",
    )
    parser.add_argument(
        "--concept_max_ent",
        type=int,
        default=5,
        help="In case of ConceptNet, how many superclasses per level for each class",
    )
    parser.add_argument(
        "--concept_n_levels",
        type=int,
        default=2,
        help="Number of abstraction levels, how many abstraction levels",
    )
    parser.add_argument(
        "--vocab_directory",
        type=str,
        default="./games/twc",
        help="Directory in which to find the vocabulary for the random embeddings",
    )
    parser.add_argument(
        "--n_abstractions", type=int, default=10, help="How many abstraction levels"
    )
    parser.add_argument(
        "--collapse",
        action="store_true",
        default=False,
        help="Whether to collapse the class graph",
    )
    parser.add_argument(
        "--start_level",
        type=int,
        default=10,
        help="At which level of the tree we should start collapsing",
    )
    parser.add_argument(
        "--end_level",
        type=int,
        default=10,
        help="At which level of the tree should we end collapsing",
    )
    parser.add_argument(
        "--residual",
        type=str,
        default="",
        help="Whether to use sum or residual approach",
    )
    parser.add_argument(
        "--beta", type=float, default=0.0001, help="Weight of regularization"
    )
    parser.add_argument("--lr", type=float, default=0.00003, help="Weight")
    parser.add_argument(
        "--bidirectional",
        default=False,
        action="store_true",
        help="Are GRUs bidirectional",
    )
    parser.add_argument(
        "--hyperbolic",
        action="store_true",
        default=False,
        help="Whether to add hyperbolic embeddings to numberbatch embeddings",
    )

    opt = parser.parse_args()

    # Determine device
    if opt.cuda == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(
            f"cuda:{opt.cuda}" if torch.cuda.is_available() else "cpu"
        )

    random.seed(opt.initial_seed)
    np.random.seed(opt.initial_seed)
    torch.manual_seed(opt.initial_seed)  # For reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.initial_seed)
        torch.backends.cudnn.deterministic = True
    # yappi.start()

    scores_runs = []
    norm_score_runs = []
    moves_runs = []
    test_scores_runs = []
    test_norm_score_runs = []
    test_moves_runs = []

    random_action = False
    if opt.agent_type == "random":
        random_action = True
        opt.graph_emb_type = None
    if opt.agent_type == "simple":
        opt.graph_type = ""
        opt.graph_emb_type = None

    # Reset prune start episodes if pruning is not selected
    if not opt.prune_nodes:
        opt.prune_start_episode = opt.no_train_episodes

    tk_extractor = extractor.get_extractor(opt.token_extractor)

    experiment_settings = {
        "agent_type": opt.agent_type,
        "hyperbolic": opt.hyperbolic,
        "priorKG_type": opt.prior_kg,
        "lr": opt.lr,
        "residual": opt.residual,
        "collapse": opt.collapse,
        "n_abstractions": opt.n_abstractions,
        "beta": opt.beta,
    }
    experiment_id = format_settings(experiment_settings)
    graph = None
    seeds = [random.randint(1, 100) for _ in range(opt.nruns)]
    
    for n in range(opt.start_run,opt.nruns):
        opt.run_no = n
        opt.seed = seeds[n]
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)  # For reproducibility
        if torch.cuda.is_available():
            torch.cuda.manual_seed(opt.seed)

        opt.results_filename = os.path.join(
            opt.results_dir, experiment_id + f"_run_{n+1}_of_{opt.nruns}"
        )
        opt.log_filename = experiment_id + f"_run_{n+1}_of_{opt.nruns}.csv"
            
        tokenizer = Tokenizer(
            noun_only_tokens=opt.noun_only_tokens,
            use_stopword=opt.use_stopword,
            ngram=opt.ngram,
            extractor=tk_extractor,
        )
        rel_extractor = RelationExtractor(tokenizer, openie_url=opt.corenlp_url)
        myagent = agent.KnowledgeAwareAgent(
            graph, opt, tokenizer, rel_extractor, device
        )
        myagent.type = opt.agent_type

        print("Training ...")
        myagent.train(opt.batch_size)  # Tell the agent it should update its parameters.
        opt.mode = "train"
        opt.nepisodes = opt.no_train_episodes  # for training
        opt.max_step_per_episode = opt.train_max_step_per_episode
        starttime = time()
        print("\n RUN ", n, "\n")
        scores, norm_scores, moves = play(myagent, opt, random_action=random_action)
        print("Trained in {:.2f} secs".format(time() - starttime))

        # Save train model
        if not os.path.exists(opt.results_dir):
            os.makedirs(opt.results_dir)
        torch.save(
            myagent.model.state_dict(),
            getUniqueFileHandler(opt.results_filename, ext=".pt"),
        )
