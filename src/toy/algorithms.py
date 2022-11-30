#####################################################################
# Contains:
# * Reinforce + Value function baseline algorithm
# * DQN - training algorithm
####################################################################

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import tqdm

from env import Env
from networks import Policy,Q_network

def train_reinforce(config:dict,
                    env:Env,
                    policy:Policy,
                    logger:SummaryWriter)->None:

    """
    Tensor-Dimensions:
    S: number of abstraction levels
    B: batch-size
    A: number of actions
    """
    n_evaluations= 0
    optimizer = Adam(policy.parameters(),lr=config['training_parameters']['lr'])
    n_policy_abstractions = env.n_abstractions
    if config['training_parameters']['true_abstraction']:
        n_policy_abstractions = 1
    if config['training_parameters']['base']:
        n_policy_abstractions = 1

    for i in tqdm.tqdm(range(config['training_parameters']['n_samples'])):

        # Env-Loop: One-step: No next-state & dones
        states = env.sample(config['training_parameters']['batch_size'])
        actions, logits, baseline = policy(states) # Bx1, SxBxA, Bx1
        rewards = torch.tensor(env.step(actions),dtype=torch.float).reshape(-1,1) # Bx1
        
        # Calculate loss 
        if config['training_parameters']['learning_alg']=='residual':
            # Get policy + action logits for each abstraction level
            cum_sum_logits = torch.cumsum(logits.detach(),dim=0)
            logits[1:] += cum_sum_logits[:-1]
            prob = F.softmax(logits,dim=-1)
            log_prob = F.log_softmax(logits,dim=-1)
            log_action_prob = log_prob.gather(2,actions.repeat(n_policy_abstractions,1,1))
            
            entropy_loss = (log_prob*prob).sum()
            baseline_loss = ((rewards-baseline)**2).sum()
            policy_loss = (-log_action_prob *(rewards-baseline.detach())).sum()

            loss = policy_loss \
                   +config['training_parameters']['baseline']*baseline_loss \
                   +config['training_parameters']['entropy']*entropy_loss
        
        elif config['training_parameters']['learning_alg']=='sum':
            sum_of_logits = torch.sum(logits,dim=0)
            prob = F.softmax(sum_of_logits,dim=-1)
            log_prob = F.log_softmax(sum_of_logits,dim=-1)
            log_action_prob = log_prob.gather(1,actions)

            entropy_loss = (log_prob*prob).sum()
            baseline_loss = ((rewards-baseline)**2).sum()
            policy_loss = (-log_action_prob*(rewards-baseline.detach())).sum()

            loss = policy_loss \
                   + config['training_parameters']['baseline']*baseline_loss \
                   + config['training_parameters']['entropy']*entropy_loss

        else:
            raise NotImplementedError('This learning algorithm has not been implemented')

        # Update 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        logger.add_scalar('Policy Loss',torch.mean(policy_loss),i)
        logger.add_scalar('Baseline Loss',torch.mean(baseline_loss),i)
        logger.add_scalar('Loss',loss,i)
        logger.add_scalar('Reward',torch.mean(rewards).detach(),i)

        # Evaluation
        if i%config['training_parameters']['evaluation_frequency']==0:
            with torch.no_grad():

                # Evaluation on the training set
                reward = 0 
                for i in range(config['training_parameters']['evaluation_samples']):
                    states = env.sample(config['training_parameters']['batch_size'])
                    actions, _ ,_ = policy(states,greedy=True)
                    rewards = env.step(actions)
                    average_reward = torch.tensor(rewards,dtype=torch.float).mean()
                    reward += average_reward.item()
                reward /= config['training_parameters']['evaluation_samples']
                logger.add_scalar('Training Evaluation',reward,n_evaluations)

                # Evaluation on the test set
                reward = 0
                for i in range(config['training_parameters']['evaluation_samples']):
                    states = env.sample(config['training_parameters']['batch_size'],test=True)
                    actions, _, _ = policy(states,greedy=True)
                    rewards = env.step(actions)
                    average_reward = torch.tensor(rewards,dtype=torch.float).mean()
                    reward += average_reward.item()
                reward /= config['training_parameters']['evaluation_samples']
                logger.add_scalar('Test Evaluation',reward,n_evaluations)

            n_evaluations += 1


def train_dqn(config:dict,
              env:Env,
              q_network:Q_network,
              logger:SummaryWriter)->None:

    n_evaluations= 0
    optimizer = Adam(q_network.parameters(),lr=config['training_parameters']['lr'])
    
    for i in tqdm.tqdm(range(config['training_parameters']['n_samples'])):

        # Env-Loop: One-step: No next-state & dones
        states = env.sample(config['training_parameters']['batch_size'])
        actions, action_values = q_network(states) # Bx1, SxBxA
        rewards = torch.tensor(env.step(actions),dtype=torch.float).reshape(-1,1)
        
        # Determine_loss
        if config['training_parameters']['learning_alg']=='residual':
            loss = 0
            residual = rewards 
            for a_values in action_values:
                a_value = a_values.gather(1,actions)
                diff = (a_value-residual)
                loss += (diff**2).sum()
                residual = -diff.detach()

        elif config['training_parameters']['learning_alg']=='sum':
            action_values = torch.sum(action_values,dim=0)
            action_values = action_values.gather(1,actions)
            loss = ((rewards-action_values)**2).sum()

        else:
            raise NotImplementedError('This learning algorithm has not been implemented')

        # Update 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        logger.add_scalar('Loss',loss,i)
        logger.add_scalar('Reward',torch.mean(rewards).detach(),i)
        logger.add_scalar('Epsilon',q_network.epsilon,i)

        # Evaluation
        if i%config['training_parameters']['evaluation_frequency']==0:
            with torch.no_grad():

                # Evaluation on the training set
                reward = 0 
                for i in range(config['training_parameters']['evaluation_samples']):
                    states = env.sample(config['training_parameters']['batch_size'])
                    actions, action_values = q_network(states,greedy=True)
                    rewards = env.step(actions)
                    average_reward = torch.tensor(rewards,dtype=torch.float).mean()
                    reward += average_reward.item()
                reward /= config['training_parameters']['evaluation_samples']
                logger.add_scalar('Training Evaluation',reward,n_evaluations)

                # Evaluation on the test set
                reward = 0
                for i in range(config['training_parameters']['evaluation_samples']):
                    states = env.sample(config['training_parameters']['batch_size'],test=True)
                    actions, action_values = q_network(states,greedy=True)
                    rewards = env.step(actions)
                    average_reward = torch.tensor(rewards,dtype=torch.float).mean()
                    reward += average_reward.item()
                reward /= config['training_parameters']['evaluation_samples']
                logger.add_scalar('Test Evaluation',reward,n_evaluations)

            n_evaluations += 1