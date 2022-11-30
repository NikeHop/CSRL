from enum import IntEnum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DeviceAwareModule
from utils.layers import ParallelFeedforwardNetwork, ParallelLinearLayer, ParallelMHALayer


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		m.weight.data.normal_(0, 1)
		m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
		if m.bias is not None:
			m.bias.data.fill_(0)



class SelfAttentionAbstractionACNet(DeviceAwareModule):

	def __init__(self,
		env,
		observation_space,
		key_size=128,
		value_size=128,
		hidden_size=128,
		rb=None,
		n_abstractions=1,
		residual=False):

		super().__init__()

		self.rb = rb

		self.table_size = observation_space['table_features'].shape[1]
		goal_feature_size = observation_space['goal_features'].shape[-1]
		self.selection_size, selection_feature_size = observation_space['selection_features'].shape[1:]
		assert goal_feature_size == selection_feature_size, 'Goal and selection feature sizes must match.'

		self.key_size = key_size
		self.value_size = value_size
		self.hidden_size = hidden_size
		self.input_feature_size = goal_feature_size

		self.residual = residual 
		self.n_abstractions = n_abstractions
		self.env = env 
		self.num_actions = env.max_table_size
		
		self.q = ParallelLinearLayer(self.n_abstractions+1,self.input_feature_size*(1+self.selection_size),key_size)
		self.k = ParallelLinearLayer(self.n_abstractions+1,self.input_feature_size,key_size)
		self.v = nn.Linear(self.input_feature_size, value_size)

		self.baseline = nn.Linear(self.value_size,1)


	def forward(self, input, core_state=(), greedy=False, **kwargs):
		# Dimensions: T: Timesteps, B: Batch-Size, N: # of abstractions S:State-Size D:Embedding-Dimension 
		# input['goal_features]: TxBxNxD
		# input['selection_features]: TxBxNxSxD
		# input['table_features]: TxBxNxSxD
		
		goal_features = input['goal_features']
		T, B, *_ = goal_features.shape
		
		goal_features = goal_features.flatten(0, 1)
		selection_features = input['selection_features'].flatten(0, 1)
		table_features = input['table_features'].flatten(0, 1)
		table_features = table_features.permute(1,0,2,3) # N x T*B x S x D
		
		# Concatenate goal ry = feature and selected feature to obtain query vector
		input_features = [goal_features,] + [selection_features[:,:,i,:] for i in range(self.selection_size)]
		x = torch.cat(input_features, -1)
		x = x.permute(1,0,2).unsqueeze(2) # N x T*B x 1 x S*D 
		
		# Projecting the query vector onto the correct dimension + computing attention
		query = self.q(x) # N x T*B x 1 x D 
		keys = self.k(table_features) # N x T*B x S x D
		self_attn = (torch.matmul(query,keys.permute(0,1,3,2))/np.sqrt(self.key_size)) # NxT*Bx1xS
		self_attn_squeezed = self_attn.squeeze(-2)
		policy_logits = torch.sum(self_attn_squeezed,dim=0) # T*BxK
		policy_logits_abstract = self_attn_squeezed # NxT*Bx1
		
		# Compute action
		if greedy:
			action = torch.argmax(policy_logits, dim=1)
		else:
			action = torch.multinomial(F.softmax(policy_logits, dim=-1), num_samples=1)

		# Compute baseline
		values = torch.bmm(self_attn[0], self.v(table_features[0])).squeeze(1)
		baseline = self.baseline(values)

		# Reshaping
		baseline = baseline.view(T, B)
		policy_logits = policy_logits.view(T, B, self.num_actions)
		policy_logits_abstract = policy_logits_abstract.view(self.n_abstractions+1,T,B,self.num_actions).permute(1,2,0,3) # TxBxNx1
		action = action.view(T,B)
		
		return dict(policy_logits=policy_logits, baseline=baseline, action=action, policy_logits_abstract=policy_logits_abstract), core_state

	def initial_state(self, batch_size):
		return tuple()