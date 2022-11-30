import math 

import torch 
import torch.nn as nn
import torch.nn.functional as F

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
        nn.init.xavier_uniform(self.weights)

    def forward(self,state:torch.tensor)->torch.tensor:
        '''
        state (tensor): N,B,S,D -> N,B,S,D
        '''
        output = torch.matmul(state,self.weights.unsqueeze(1))  + self.biases.unsqueeze(1).unsqueeze(1)
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

    def forward(self,state):
        return self.layers(state)

        
class ParallelMHALayer(nn.Module):

    def __init__(self,
                 n_layers,
                 n_abstractions,
                 embedding_dim):

        super().__init__()

        
        self.n_abstractions = n_abstractions
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Create weights
        self.keys = nn.Parameter(torch.zeros(self.n_layers,self.n_abstractions,self.embedding_dim,self.embedding_dim))
        self.queries = nn.Parameter(torch.zeros(self.n_layers,self.n_abstractions,self.embedding_dim,self.embedding_dim))
        self.values = nn.Parameter(torch.zeros(self.n_layers,self.n_abstractions,self.embedding_dim,self.embedding_dim))

        self.linear_layers = nn.ModuleList()
        for i in range(n_layers):
            self.linear_layers.append(ParallelFeedforwardNetwork(1,n_abstractions,self.embedding_dim,[256],self.embedding_dim))
        
        # Initialize weights
        nn.init.xavier_uniform(self.keys)
        nn.init.xavier_uniform(self.queries)
        nn.init.xavier_uniform(self.values)

    def forward(self,state):
        '''
        state (tensor): NxBxSxD, where:
        N: abstraction-size
        B: batch-size
        S: state-size 
        D: dimension-size 
        '''

        N,B,S,D = state.shape
        
        for i in range(self.n_layers):
            # Embed vectors:
            K = torch.matmul(state,self.keys[i].unsqueeze(1))
            Q = torch.matmul(state,self.queries[i].unsqueeze(1))
            V = torch.matmul(state,self.values[i].unsqueeze(1))

            # Compute attention
            A = F.softmax((1/math.sqrt(self.embedding_dim)) * torch.matmul(Q,K.permute(0,1,3,2)),dim=-1) # N,B,S,S # probability each row is a query and contains the prob over the keys
            
            # Compute output
            new_state = torch.matmul(A,V)
            state = state + new_state 
            
            # Apply Linear Layer
            new_state = self.linear_layers[i](state)
            state = state + new_state
            
        return state