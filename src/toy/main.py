import argparse
from datetime import datetime
import os
import random

import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from algorithms import train_reinforce, train_dqn
from env import Env
from networks import Policy, Q_network


if __name__ == "__main__":
    # Parse config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs",
        help="Directory of the configuration file",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="reinforce",
        help="Whether to use dqn or reinforce",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=-1,
        help="Number of random seeds over which to run the experiment",
    )

    # Arguments to overwrite config values
    parser.add_argument(
        "--training_parameters.n_samples",
        type=int,
        default=-1,
        help="Number of batches to sample",
    )
    parser.add_argument(
        "--training_parameters.base",
        action="store_true",
        default=-1,
        help="Whether the agent only has the base state available",
    )
    parser.add_argument(
        "--training_parameters.true_abstraction",
        type=int,
        default=-1,
        help="Which abstraction layer to use for the oracle",
    )
    parser.add_argument(
        "--training_parameters.learning_alg",
        type=str,
        default=-1,
        help="Sum ord Residual learning algorithm",
    )

    parser.add_argument(
        "--environment.noise_prob",
        type=float,
        default=-1,
        help="Noise probability, likelihood of resampling an action at the leave",
    )
    parser.add_argument(
        "--environment.branching_per_layer",
        type=list,
        default=-1,
        help="The branching factor for each factor of the tree 7881 means \
             first layer has a brachning factor of 7 second layer has \
             a branching factor of 8 etc.",
    )

    args = parser.parse_args()

    # Load the configuration file
    if args.algorithm.lower()=="dqn":
        args.config_path = os.path.join(args.config_path, "dqn.yaml")
    elif args.algorithm.lower()=="reinforce":
        args.config_path = os.path.join(args.config_path, "reinforce.yaml")
    else:
        raise NotImplementedError('This algorithm does not exist')

    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Overwrite config with command line arguments
    for arg in vars(args):
        argument_value = getattr(args, arg)
        # Preprocess lists
        if type(argument_value) == list:
            argument_value = list(map(int, argument_value))
        if argument_value != -1:
            category = arg.split(".")
            if len(category) == 1:
                category = category[0]
                config[category] = argument_value
            elif len(category) == 2:
                category, category_argument = category
                config[category][category_argument] = argument_value
            else:
                raise Exception(
                    "The depth of the config file  should not be larger \
                                 than 2"
                )

    # Prepare Logging
    time_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    log_dir = os.path.join("./log", config["experiment_name"], time_string)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for _ in range(args.n_seeds):
        # Set seeds
        seed = random.randint(0, 10000)
        torch.manual_seed(seed)
        random.seed(seed)

        # Log configuration
        logger = SummaryWriter(log_dir=os.path.join(log_dir, str(seed)))
        config_string = ""
        for elem, value in config.items():
            config_string += f"{elem} : {value} "
        logger.add_text("Configuration", config_string, 0)

        # Build env
        env = Env(
            config["environment"]["branching_per_layer"],
            config["environment"]["n_actions"],
            config["environment"]["fixed_layer"],
            config["environment"]["noise_prob"],
        )
        if config["visualize"]:
            env.visualize_action_space(log_dir)

        # Train
        if args.algorithm.lower() == "dqn":
            model = Q_network(config, env)
            train_dqn(config, env, model, logger)
        elif args.algorithm.lower() == "reinforce":
            model = Policy(config, env)
            train_reinforce(config, env, model, logger)
        else:
            raise NotImplementedError("This algorithm does not exist")
