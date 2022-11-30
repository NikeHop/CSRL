import os
from typing import Tuple

import numpy as np
import gym
from gym.utils import seeding
import torch

from utils import seed as utils_seed
from utils.kgs.class_trees import ClassTree
from utils.kgs.wordnet import get_class_tree_wordnet

from utils.word2feature import FeatureMap
from wordcraft.recipe_book import Recipe, RecipeBook, Task


NO_RECIPE_PENALTY = -0.1
IRRELEVANT_RECIPE_PENALTY = -0.1
GOAL_REWARD = 1.0
SUBGOAL_REWARD = 1.0


class WordCraftEnv(gym.Env):
	"""
	Simple text-only RL environment for crafting multi-step recipes.

	At a high level, the state consists of a goal, the inventory, and the current selection.
	"""
	def __init__(
		self,
		seed=None,
		data_path='datasets/alchemy2.json',
		recipe_book_path=None,
		max_depth=1,
		split='by_recipe',
		train_ratio=1.0,
		feature_type='glove',
		random_feature_size=300,
		num_distractors=0,
		uniform_distractors=False,
		max_mix_steps=1,
		subgoal_rewards=True,
		n_abstractions=1,
        collapse=False,
        start_level=15,
        end_level=15,
        priorKG_type='Wordnet',
	):
		super().__init__()

		# Controls from which recipes is sampled
		self.eval_mode = False

		# Set random, numpy, torch seed
		if seed is None:
			seed = int.from_bytes(os.urandom(4), byteorder="little")
		self.set_seed(seed)
		utils_seed(seed)

		# Load recipe book
		if recipe_book_path is not None:
			self.recipe_book = RecipeBook.load(recipe_book_path)
			self.recipe_book.set_seed(seed)
			max_depth = self.recipe_book.max_depth
		else:
			self.recipe_book = RecipeBook(data_path=data_path,
										  max_depth=max_depth,
										  split=split,
										  train_ratio=train_ratio,
										  seed=seed)

		# Extract prior knowledge from KG
		self.prior_kg = priorKG_type
		if self.prior_kg=='Wordnet':
			entities = self.recipe_book.entities
			class_tree = get_class_tree_wordnet(entities,
												collapse,
												start_level,
												end_level)
			# Build up datastructure								
			self.class_tree = ClassTree(class_tree,self.recipe_book.entity2index)

			# Initialize feature vectors
			words = [word for word,_ in sorted(self.class_tree.word2id.items(),key=lambda x: x[1])]
			self.feature_map = FeatureMap(words=words,
										  feature_type=feature_type,
										  random_feature_size=random_feature_size,
										  seed=seed)
		else:
			raise NotImplementedError('This prior knowledge type does not exist')
		
		self.max_selection_size = self.recipe_book.max_recipe_size
		self.max_mix_steps = max(max_mix_steps or max_depth, max_depth)
		self.max_steps = self.max_selection_size*self.max_mix_steps

		self.sample_depth = max_depth
	
		self.subgoal_rewards = subgoal_rewards
		self.max_depth = max_depth
		self.num_distractors = num_distractors
		self.uniform_distractors = uniform_distractors

		self.max_table_size = 2**max_depth + num_distractors + self.max_mix_steps

		############################# State abstraction arguments  ##################################

		self.n_abstractions = n_abstractions
		self.n_state_layers = self.n_abstractions+1

		#############################################################################################

		self.task = None
		self.distractors = []
		self.goal_features = np.zeros((self.n_state_layers,self.feature_map.feature_dim))

		self._reset_table()
		self._reset_selection()
		self._reset_history()

		self.episode_step = 0
		self.episode_mix_steps = 0
		self.episode_reward = 0
		self.done = False

		obs = self.reset()
		num_entities = len(self.class_tree.word2id)
		dspaces = {
			'goal_index': gym.spaces.MultiDiscrete([num_entities]*(self.n_state_layers)),
			'goal_features': gym.spaces.Box(shape=self.goal_features.shape, low=-1., high=1.),
			'table_index': gym.spaces.MultiDiscrete((self.n_state_layers)*(self.max_table_size)*[num_entities]),
			'table_features': gym.spaces.Box(shape=self.table_features.shape, low=-1., high=1.),
			'selection_index': gym.spaces.MultiDiscrete((self.n_state_layers)*self.max_selection_size*[num_entities]),
			'selection_features': gym.spaces.Box(shape=self.selection_features.shape, low=-1., high=1.),
		}

		self.observation_space = gym.spaces.Dict(dspaces)
		self.action_space = gym.spaces.Discrete(self.max_table_size) 

	def reset(self):
		self.episode_step = 0
		self.episode_mix_steps = 0
		self.episode_reward = 0
		self.done = False
		self.task = self.recipe_book.sample_task(depth=self.sample_depth)
		self.distractors = self.recipe_book.sample_distractors(self.task, self.num_distractors, uniform=self.uniform_distractors)
		self._reset_selection()
		self._reset_table()
		self._reset_history()

		return self._get_observation()

	def get_correct_actions(self):
		optimal_actions = [i for i,elem in enumerate(self.table) if elem not in self.distractors]
		return optimal_actions
		
	def eval(self, split='test'):
		self.eval_mode = True
		self.recipe_book.test_mode = (split == 'test')

	def train(self):
		self.eval_mode = False
		self.recipe_book.test_mode = False

	def set_seed(self, seed):
		self.np_random, self.seed = seeding.np_random(seed)

	def sample_depth(self, depth):
		self.sample_depth = depth

	def __max_table_size_for_depth(self, depth):
		return 2**depth - 1

	def _reset_table(self):
		self.table = []
		table_index = []
		self.table_features = np.zeros((self.n_state_layers,self.max_table_size, self.feature_map.feature_dim))

		# In case a task has been sampled
		if self.task:
			# Build state with abstraction
			self.table = list(self.task.base_entities + self.distractors)
			self.np_random.shuffle(self.table)
			_, self.table = self._form_abstraction([self.class_tree.word2id[word] for word in self.table])

			# Get feature embeddings for state
			state_size = len(self.table)//(self.n_state_layers)
			features = []
			for i,e in enumerate(self.table):
				features.append(self.feature_map.feature(e))
				if (i+1)%state_size==0:
					features += [np.zeros(self.feature_map.feature_dim)]*(self.max_table_size-state_size)
			self.table_features[:,:,:] = np.array(features).reshape(self.n_state_layers,self.max_table_size,-1)

			# Get indices for state
			for i,e in enumerate(self.table):
				table_index.append(self.class_tree.word2id[e])
				if (i+1)%state_size==0:
					table_index += [-1]*(self.max_table_size-state_size)
			self.table_index = np.array(table_index)
			
	def _reset_selection(self):
		self.selection = []
		self.selection_index = -np.ones(self.max_selection_size*(self.n_state_layers),dtype=int)
		self.selection_features = np.zeros((self.n_state_layers,self.max_selection_size,self.feature_map.feature_dim))

	def _reset_history(self):
		self.subgoal_history = set()
	
	def set_task(self,goal,base_entities):
		'''
		goal: (str) The goal entitiy 
		base_entities: (list) List of ingredients for the recipe
		'''
		recipe = Recipe(base_entities)
		task = Task(goal,base_entities,(),recipe)
		self.task = task
	
	def _get_observation(self):
		"""
		Note, includes indices for each inventory and selection item,
		since torchbeast stores actions in a shared_memory tensor shared among actor processes
		"""
		self.goal_index, self.goal_words = self._form_abstraction([self.class_tree.word2id[self.task.goal]])
		self.goal_features = np.array([self.feature_map.feature(goal_word) for goal_word in self.goal_words])
		
		
		obs = {
			'goal_index': self.goal_index,
			'goal_features': self.goal_features,
			'table_index': self.table_index,
			'table_features': self.table_features,
			'selection_index': self.selection_index,
			'selection_features': self.selection_features,
		}

		
		return obs
	
	def _form_abstraction(self,indeces:list)->Tuple[list,list]:
		"""
		indeces: list of indeces size N 
		that describes a state on the base abstraction level

		return: 
		list of indeces of size N*abstractions of all the abstracted indeces,
		list of all the abstracted words 
		"""

		current_state_indeces = indeces
		abstract_state_indeces = []
		abstract_state_words = []

		for i in range(self.n_abstractions):
			new_abstract_indeces = []
			for index in current_state_indeces:
				if index in self.class_tree.abstraction_layers[self.class_tree.depth-i]:
					new_abstract_indeces.append(self.class_tree.class2superclass[index][0])
				else:
					new_abstract_indeces.append(index)

			abstract_state_indeces += current_state_indeces
			abstract_state_words += [self.class_tree.id2word[idx] for idx in current_state_indeces]
			current_state_indeces = new_abstract_indeces
		
		abstract_state_indeces += current_state_indeces
		abstract_state_words += [self.class_tree.id2word[idx] for idx in current_state_indeces]
		
		return abstract_state_indeces, abstract_state_words

	def step(self, action):
		reward = 0
		if self.done: # no-op if env is done
			return self._get_observation(), reward, self.done, {}

		# Handle invalid actions
		invalid_action = not (0 <= action < self.max_table_size)
		if invalid_action:
			self.episode_step += 1
			if self.episode_step >= self.max_steps:
				self.done = True

		i = self.table_index[action]
		if i!=-1:
			e = self.class_tree.id2word[i]
		else:
			e = 'placeholder'
		
		# Update selection
		selection_size = len(self.selection)
		if selection_size < self.max_selection_size:
			self.selection.append(e)
			# Create abstraction and add to selection
			if i!=-1:
				selected_entities, selected_words = self._form_abstraction([i])
				for k,ent in enumerate(selected_entities):
					# selection_index=[ent1_level1,ent2_level1,ent1_level2,ent2_level2]
					self.selection_index[k*(self.max_selection_size)+selection_size] = ent
				self.selection_features[:,selection_size,:] = np.array([self.feature_map.feature(word) for word in selected_words])
			selection_size = len(self.selection)
		

		# Evaluate selection
		if selection_size == self.max_selection_size:
			self.episode_mix_steps += 1
			recipe = Recipe(self.selection)
			result = self.recipe_book.evaluate_recipe(recipe)
			if result is None:
				reward = NO_RECIPE_PENALTY if not self.eval_mode else 0
			elif result == self.task.goal:
				reward = GOAL_REWARD
				self.done = True
			elif result in self.task.intermediate_entities:
				reward = 0
				if result not in self.subgoal_history:
					self.subgoal_history.add(result)
					reward = SUBGOAL_REWARD if self.subgoal_rewards and not self.eval_mode else 0
			else:
				reward = IRRELEVANT_RECIPE_PENALTY if not self.eval_mode else 0
			self.episode_reward += reward

			# Add results to selectable entities
			if result:
				result_i = self.class_tree.word2id[result]
				idds, words = self._form_abstraction([result_i])
				
				# Update table
				state_size = len(self.table)//(self.n_state_layers)
				new_table = []
				k = 0 
				for i,elem in enumerate(self.table):
					new_table.append(elem)
					if (i+1)%state_size==0:
						new_table.append(words[k])
						k +=1 
				self.table = new_table 
				
				# Update table_index
				k = 0
				i = 0 
				while k<len(idds):
					if (i+1)%state_size==0:
						self.table_index[i+1] = idds[k]
						k += 1
					i += 1

				# Update table features
				for i in range(self.n_state_layers):
					self.table_features[i,state_size,:] = self.feature_map.feature(words[i])

			# Clear selection
			self._reset_selection()
		
		
		self.episode_step += 1
		if self.episode_mix_steps >= self.max_mix_steps or self.episode_step >= self.max_steps:
			self.done = True

		obs = self._get_observation()
		return obs, reward, self.done, {}

	def _display_ascii(self, mode='human'):
		"""
		Render the env state as ascii:
		Combine the ingredients to make *torch*
		-------------------------------------------------------
		1:fire, 2:wind, 3:sand, 4:star, 5:wood, 6:stick, 7:coal
		-------------------------------------------------------
		(on hand): stick
		Subgoal rewards: 0
		"""
		goal_str = f'Combine the ingredients to make *{self.task.goal}*'
		if mode == 'human':
			table_str = f"{', '.join([f'{i+1}:{e}' for i, e in enumerate(self.table)])}"
		else:
			table_str = f"{', '.join(self.table)}"
		selection_str = f"(on hand): {', '.join(self.selection)}"
		hr = ''.join(['-']*50)

		# output = f'\n{goal_str}\n\n{hr}\n{table_str}\n{hr}\n\n{selection_str}\n\nSubgoal rewards: {self.episode_reward}\n'
		output = f'\n{goal_str}\n\n{hr}\n{table_str}\n{hr}\n\n{selection_str}\n\n'

		print(output)

	def render(self, mode='human'):
		self._display_ascii(mode)

gym.envs.registration.register(
		id='wordcraft-multistep-goal-v0',
		entry_point=f"{__name__}:WordCraftEnv",
	)

if __name__=="__main__":
	pass