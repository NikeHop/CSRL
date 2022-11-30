import random
from typing import Tuple, Union, List

from graphviz import Graph


class Env(object):
    def __init__(
        self,
        branching_per_layer: List[int],
        n_actions: int,
        fixed_layer: int,
        noise_prob: float,
    ) -> None:

        self.observation_space = StateTree(
            branching_per_layer, n_actions, fixed_layer, noise_prob
        )
        self.state2action = self.observation_space.state2action
        self.current_state = None
        self.n_actions = n_actions
        self.n_states = self.observation_space.n_states
        self.n_abstractions = self.observation_space.n_layers

    def sample(self, batch_size: int, test=False) -> List[List[int]]:

        batch, leaves = self.observation_space.sample(batch_size, test)
        self.current_state = leaves
        return batch

    def step(self, actions: List[int]) -> List[int]:

        rewards = []
        for leaf, action in zip(self.current_state, actions):
            if self.state2action[leaf] == action:
                rewards.append(1)
            else:
                rewards.append(0)
        return rewards

    def visualize_action_space(self, log_dir: str) -> None:
        self.observation_space.visualize_action_space(log_dir)


class StateTree(object):
    def __init__(
        self,
        branching_per_layer: int,
        n_actions: int,
        fixed_layer: int,
        noise_prob: float,
    ) -> None:

        # Tree data structure: dict[int]=list
        # Maps nodes to its children
        self.tree = {}
        self.branching_per_layer = branching_per_layer

        # Each training leave is associated with a test leave
        self.train2test = {}
        self.train_leaves = []
        self.test_leaves = []

        # Map the state to the abstraction level
        self.state2level = {}

        # Build up tree data structure
        n_states = 1  # Counter
        previous_states = [1]  # Queue to add newly created states
        self.n_layers = len(branching_per_layer)

        for k, factor in enumerate(branching_per_layer):
            new_previous_states = []
            for elem in previous_states:
                # Map state to current level
                self.state2level[elem] = k + 1
                # Number of children determined by branching factor
                self.tree[elem] = [
                    i for i in range(n_states + 1, n_states + 1 + factor)
                ]
                # Update state counter and queue
                n_states += factor
                new_previous_states += self.tree[elem]

                # Last but one layer, memorize which state are leaves
                if k == self.n_layers - 1:
                    self.train_leaves += self.tree[elem]
                    for leave in self.tree[elem]:
                        self.train2test[leave] = n_states + 1
                        self.test_leaves.append(n_states + 1)
                        n_states += 1

            previous_states = new_previous_states

        self.n_states = n_states
        self.test2train = {value: key for key, value in self.train2test.items()}

        # Map states to actions for visualization & Reward calculation
        self.state2action = {}
        self.noise_prob = noise_prob
        self.n_actions = n_actions
        self.fixed_layer = fixed_layer

        self._map_state2action(1, "-", 0)

    def _map_state2action(self, node: int, action: Union[int, str], depth: int) -> None:

        # If leaf return
        if node not in self.tree:
            if action != "-":
                # Action is determined by parent state
                if random.random() < self.noise_prob:
                    # Sample a new action
                    sampled_action = random.sample(list(range(self.n_actions)), 1)[0]
                    self.state2action[node] = sampled_action
                else:
                    self.state2action[node] = action
                # Test leaf gets action of parent
                self.state2action[self.train2test[node]] = action
            else:
                # Action is not determined by parent state
                sampled_action = random.sample(list(range(self.n_actions)), 1)[0]
                self.state2action[node] = sampled_action
                self.state2action[self.train2test[node]] = sampled_action

            return

        # Set action
        self.state2action[node] = action

        # Recursive call on children
        children = self.tree[node]
        for child in children:
            if depth == self.fixed_layer:
                # Abstraction layer on which to dertermine the action
                action = random.sample(list(range(self.n_actions)), 1)[0]
                self._map_state2action(child, action, depth + 1)
            else:
                # Pass on action given by parent
                self._map_state2action(child, action, depth + 1)

    def sample(self, batch_size: int, test: bool = False) -> Tuple[list]:

        # List of path starting at the root
        # sampled from the tree
        batch = []
        leaves = []

        for _ in range(batch_size):
            # Saves the path sampled from the tree
            new_state = []
            # Each path starts at the root
            current_state = 1
            while current_state in self.tree:
                current_state = random.sample(self.tree[current_state], 1)[0]
                new_state.append(current_state)
            leaves.append(new_state[-1])

            # At test time exchange training leaf with test leaf
            if test:
                new_state[-1] = self.train2test[new_state[-1]]
                leaves[-1] = self.train2test[leaves[-1]]

            batch.append(new_state)

        return batch, leaves

    def visualize_action_space(self, log_dir: str) -> None:
        # Visualize graph with graphviz
        g = Graph()
        nodes = {}
        for node, children in self.tree.items():
            if node not in nodes:
                # Label the node with action
                if node in self.state2action:
                    if node not in self.tree:
                        # Label for the leaf (Train/Test)
                        g.node(
                            str(node),
                            label=str(self.state2action[node])
                            + str(self.state2action[self.train2test[child]]),
                        )
                    else:
                        # Action at inner node
                        g.node(str(node), label=str(self.state2action[node]))
                else:
                    # Action decided on lower abstraction level
                    g.node(str(node), label="--")

            for child in children:
                if child not in nodes:
                    # Label the node with action
                    if child in self.state2action:
                        if child not in self.tree:
                            # Label for the leaf (Train/Test)
                            g.node(
                                str(child),
                                label=str(self.state2action[child])
                                + str(self.state2action[self.train2test[child]]),
                            )
                        else:
                            # Action at inner node
                            g.node(str(child), label=str(self.state2action[child]))
                    else:
                        # Action decided on lower abstraction level
                        g.node(str(child), label="--")
                # Add edge between parents and children
                g.edge(str(node), str(child))
        g.render("env", directory=log_dir)
