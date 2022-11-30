import random 
from typing import List, Tuple

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from env import Env 

class Policy(nn.Module):
    
    def __init__(self,config:dict, env:Env)->None:
        super().__init__()

        # Environment information 
        self.n_actions = env.n_actions
        self.n_abstractions = env.n_abstractions
        self.n_states = env.n_states+1
        self.base = config['training_parameters']['base']
        self.true_abstraction = config['training_parameters']['true_abstraction']
        
        
        # Embedding Layer 
        self.embeddings = nn.Embedding(embedding_dim=config['network']['embedding_dim'],
                                       num_embeddings=self.n_states)
        
        # Parallel-MLP
        if config['training_parameters']['true_abstraction'] or config['training_parameters']['base']:
            self.policy = ParallelFeedforwardNetwork(config['network']['n_layers'],
                                                    1,
                                                    config['network']['embedding_dim'],
                                                    config['network']['hidden_dim'],
                                                    self.n_actions)
        else:
            self.policy = ParallelFeedforwardNetwork(config['network']['n_layers'],
                                                    self.n_abstractions,
                                                    config['network']['embedding_dim'],
                                                    config['network']['hidden_dim'],
                                                    self.n_actions)

        self.value = nn.Sequential(nn.Linear(config['network']['embedding_dim'],128),
                                   nn.ReLU(),
                                   nn.Linear(128,1))

        
    def forward(self, states:List[list], greedy:bool=False)->Tuple[torch.tensor]:
        '''
        Tensor-Dimensions:
        S: number of abstractions
        B: batch-size
        D: embedding-dimension
        '''

        # Encode State
        encoded_state = self.embeddings(torch.tensor(states,dtype=torch.long).transpose(1,0)) # SxBxD
        if self.true_abstraction:
            encoded_state = encoded_state[self.true_abstraction:self.true_abstraction+1] 
        if self.base:
            encoded_state = encoded_state[-1:]

        # Policy
        logits = self.policy(encoded_state) # SxBxD
        if greedy:
            actions = torch.argmax(F.softmax(torch.sum(logits,dim=0),dim=-1).squeeze(1),dim=-1) # Bx1
        else:
            actions = torch.multinomial(F.softmax(torch.sum(logits,dim=0),dim=-1).squeeze(1),num_samples=1) # Bx1
    
        # Value 
        baseline = self.value(encoded_state[-1].squeeze(1)) # Bx1

        return (actions, logits, baseline)


class Q_network(nn.Module):

    def __init__(self,config:dict, env:Env)->None:
        super().__init__()

        # Environment information 
        self.n_actions = env.n_actions
        self.n_abstractions = env.n_abstractions
        self.n_states = env.n_states+1
        self.base = config['training_parameters']['base']
        self.true_abstraction = config['training_parameters']['true_abstraction']
        self.epsilon = config['training_parameters']['epsilon']
        self.epsilon_decay = config['training_parameters']['epsilon_decay']
        
        # Learning algorithm
        self.learning_alg = config['training_parameters']['learning_alg']

        # Embedding Layer 
        self.embeddings = nn.Embedding(embedding_dim=config['network']['embedding_dim'],
                                       num_embeddings=self.n_states)
        
        # Parallel-MLP
        if config['training_parameters']['true_abstraction'] or config['training_parameters']['base']:
            self.q_network = ParallelFeedforwardNetwork(config['network']['n_layers'],
                                                    1,
                                                    config['network']['embedding_dim'],
                                                    config['network']['hidden_dim'],
                                                    self.n_actions)
        else:
            self.q_network = ParallelFeedforwardNetwork(config['network']['n_layers'],
                                                    self.n_abstractions,
                                                    config['network']['embedding_dim'],
                                                    config['network']['hidden_dim'],
                                                    self.n_actions)

        
    def forward(self,states:List[list],greedy:bool=False)->None:
        
        '''
        Tensor Dimension
        S: number of abstractions 
        B: batch size 
        D: embedding dimension
        '''
        
        # Encode state 
        encoded_state = self.embeddings(torch.tensor(states,dtype=torch.long).transpose(1,0)) # SxBxD
        if self.true_abstraction:
            encoded_state = encoded_state[self.true_abstraction:self.true_abstraction+1]
        if self.base:
            encoded_state = encoded_state[-1:]

        # Q-values
        action_values = self.q_network(encoded_state)
        if greedy:
            sum_action_values = torch.sum(action_values,dim=0).squeeze(1)
            actions = torch.argmax(sum_action_values,dim=-1).reshape(-1,1)
        else:
            if random.random()>self.epsilon:
                sum_action_values = torch.sum(action_values)
                actions = torch.argmax(sum_action_values,dim=-1).reshape(-1,1)
            else:
                bs = action_values.shape[1]
                n_actions = action_values.shape[-1]
                actions = [random.sample(list(range(n_actions)),1)[0] for _ in range(bs)]
                actions = torch.tensor(actions,dtype=torch.long).reshape(-1,1)

        self.epsilon *= self.epsilon_decay

        return actions, action_values
    
# NN layers
class ParallelLinearLayer(nn.Module):
    
    def __init__(self,
                 n_abstractions:int,
                 input_dimension:int,
                 output_dimension:int)->None:

        super().__init__()

        self.n_abstractions = n_abstractions
        self.input_dim = input_dimension
        self.output_dim = output_dimension

        # Create weights and biases 
        self.weights = nn.Parameter(torch.zeros(self.n_abstractions,self.input_dim,self.output_dim))
        self.biases = nn.Parameter(torch.zeros(self.n_abstractions,self.output_dim))

        # Initialize weights 
        nn.init.xavier_uniform_(self.weights)

    def forward(self,state:torch.tensor)->torch.tensor:

        '''
        Tensor Dimensions
        S: Number of abstractions 
        B: Batch size 
        D1/D2: embedding dimensions
        '''
        output = torch.bmm(state,self.weights)  + self.biases.unsqueeze(1) # S,B,D1 -> S,B,D2
        return output

class ParallelFeedforwardNetwork(nn.Module):

    def __init__(self,
                 n_layers:int,
                 n_abstractions:int,
                 input_dimension:int,
                 hidden_dimensions:int,
                 output_dimension:int)->None:
        
        super().__init__()

        dimensions = [input_dimension] + hidden_dimensions

        layers = []
        for i in range(n_layers):
            layers.append(ParallelLinearLayer(n_abstractions,dimensions[i],dimensions[i+1]))
            layers.append(nn.ReLU())

        layers.append(ParallelLinearLayer(n_abstractions,dimensions[n_layers],output_dimension))

        self.layers = nn.Sequential(*layers)

    def forward(self,state:torch.tensor)->torch.tensor:
        """
        Tensor Dimensions:
        S: Number of abstractions
        B: Batch size 
        D_I/D_O: input and output embedding size
        """
        return self.layers(state) # # S,B,D_I -> S,B,D_O





