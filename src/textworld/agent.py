import math
from utils import nlp
import torch
import numpy as np
from torch import optim
from textworld import EnvInfos
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from typing import Mapping, Any
from collections import defaultdict
import scorer
import re
from utils.generic import load_embeddings, max_len, to_tensor
from utils.textworld_utils import (
    serialize_facts,
    process_full_facts,
    process_step_facts,
)
from utils.kg import (
    construct_graph,
    add_triplets_to_graph,
    shortest_path_subgraph,
    khop_neighbor_graph
)
from utils.extractor import any_substring_extraction
from utils.prior_kg.class_tree import ClassTree, MultiClassTree
from utils.prior_kg.utils import load_entities
from utils.prior_kg.wordnet import get_class_tree_wordnet
from utils.prior_kg.dbpedia import get_class_tree_dbpedia
from utils.prior_kg.conceptnet import get_class_tree_conceptnet
from utils.prior_kg.own_kg import MANUAL_CLASSES


# Agent must have train(), test(), act() functions and infos_to_request as properties


class KnowledgeAwareAgent:
    """Knowledgeable Neural Agent for playing TextWorld games."""

    UPDATE_FREQUENCY = 20
    LOG_FREQUENCY = 20  # in episodes
    GAMMA = 0.9
    type = "KnowledgeAware"

    def __init__(
        self, graph, opt, tokenizer=None, rel_extractor=None, device=None
    ) -> None:
        print("Initializing Knowledge-Aware Neural Agent")
        self.opt = opt
        self.seed = opt.seed
        self.hidden_size = opt.hidden_size
        self.device = device
        self.local_evolve_type = opt.local_evolve_type
        self.world_evolve_type = opt.world_evolve_type
        self._initialized = False
        self._epsiode_has_started = False
        self.sentinel_node = (
            True  # Sentinel node is added to local/world to allow attention module
        )
        self.epsilon = opt.egreedy_epsilon
        self.tokenizer = tokenizer
        self.rel_extractor = rel_extractor
        self.pruned_concepts = []

        self.emb_loc = opt.emb_loc
        self.word_emb_type = opt.word_emb_type
        self.graph_emb_type = opt.graph_emb_type
        self.word_emb, self.graph_emb = None, None
        self.word2id, self.node2id, self.hyperbolic_word2id = {}, {}, {}
        self._episode_has_started = False

        # Additional properties needed for abstraction
        self.n_abstractions = opt.n_abstractions
        self.number_of_dones = 0
        self.residual = opt.residual
        self.beta = opt.beta
        self.entities = load_entities(opt)

        if opt.prior_kg == "wordnet":
            class_tree = get_class_tree_wordnet(
                self.entities,
                opt.collapse,
                opt.start_level,
                opt.end_level,
                opt.load_class_tree,
                opt.kg_data,
                visualize=opt.visualize,
            )
            self.class_tree = ClassTree(class_tree, self.entities, "wordnet")
        elif opt.prior_kg == "dbpedia":
            class_tree = get_class_tree_dbpedia(
                self.entities, opt.load_class_tree, opt.kg_data, visualize=opt.visualize
            )
            self.class_tree = ClassTree(class_tree, self.entities, "dbpedia")
        elif opt.prior_kg == "manual":
            # Class tree object so the code runs as is, but when abstraction is performed the same
            # word is just copied
            class_tree = get_class_tree_wordnet(
                self.entities,
                opt.collapse,
                opt.start_level,
                opt.end_level,
                opt.load_class_tree,
                opt.kg_data,
                visualize=opt.visualize,
            )
            self.class_tree = ClassTree(class_tree, self.entities, "wordnet")
        elif opt.prior_kg == "own":
            self.class_tree = ClassTree(MANUAL_CLASSES, self.entities , "own")
        elif opt.prior_kg == "conceptnet":
            class_tree = get_class_tree_conceptnet(
                self.entities,
                opt.concept_max_ent,
                opt.concept_n_levels,
                opt.load_class_tree,
                opt.kg_data,
            )
            self.class_tree = MultiClassTree(class_tree, 2)
        else:
            raise NotImplementedError("This priorKG does not exist")

        if opt.prior_kg=="manual":
            self.class_tree.manual = True
        else:
            self.class_tree.manual = False

        self.max_depth = self.class_tree.depth
        self.n_abstractions = opt.n_abstractions

        # Rewrite the word2id, id2word of the class tree to make sure all entities are one string
        new_word2id = {}
        for word, idd in self.class_tree.word2id.items():
            word = "_".join(word.split(" "))
            if word not in new_word2id:
                new_word2id[word] = idd
        self.class_tree.word2id = new_word2id
        self.class_tree.id2word = {
            idd: word for word, idd in self.class_tree.word2id.items()
        }

        # Determine correct embeddings
        if self.word_emb_type is not None:
            self.word_emb, self.hyperbolic_emb = load_embeddings(
                opt, self.emb_loc, self.word_emb_type
            )

            # Add UNK embedding to hyperbolic embddings
            if self.hyperbolic_emb != None:
                self.hyperbolic_emb["<UNK>"] = np.zeros(100)
                self.hyperbolic_word_vocab = self.hyperbolic_emb.vocab

                for i, w in enumerate(self.hyperbolic_word_vocab):
                    self.hyperbolic_word2id[w] = i

            # Building up vocabulary
            self.word_vocab = self.word_emb.vocab
            for i, w in enumerate(self.word_vocab):
                self.word2id[w] = i

            # Check whether all the classes are part of the vocab, if not add
            current_idd = len(self.word2id)
            embeddings = nn.Embedding(
                num_embeddings=len(self.class_tree.word2id), embedding_dim=opt.emb_size
            )
            counter = 0
            for word in self.class_tree.word2id:
                if word not in self.word2id:
                    # In case of ConceptNet, form the class embeddings by averaging the embeddings of all classes
                    # that are part of the class set
                    if "class_group" in word:
                        cls = self.class_tree.class_name2group[word]
                        compounds_embeddings = []
                        for cl in cls:
                            if cl in self.word2id:
                                compounds_embeddings.append(self.word_emb[cl])

                    # Create embedding in case of entity not present in vocab
                    else:
                        compounds = word.split("_")
                        compounds_embeddings = []
                        for elem in compounds:
                            if elem in self.word2id:
                                compounds_embeddings.append(self.word_emb[elem])

                    # Add it to vocabulary and embeddings
                    self.word2id[word] = current_idd
                    if len(compounds_embeddings) == 0:
                        counter += 1
                        self.word_emb[word] = np.array(
                            embeddings(torch.tensor(counter)).detach()
                        )
                    else:
                        self.word_emb[word] = np.mean(
                            np.array(compounds_embeddings), axis=0
                        )
                    current_idd += 1

            # Do the same for the hyperbolic embeddings
            self.numberbatchID2hyperbolicID = {}
            if opt.hyperbolic:
                for word in self.word2id.keys():
                    if word in self.hyperbolic_word2id:
                        self.numberbatchID2hyperbolicID[
                            self.word2id[word]
                        ] = self.hyperbolic_word2id[word]
                    else:
                        self.numberbatchID2hyperbolicID[
                            self.word2id[word]
                        ] = self.hyperbolic_word2id["<UNK>"]
        self.id2word = {value: key for key, value in self.word2id.items()}

        # Graphs
        self.graph_type = opt.graph_type
        self.reset_graph()

        if self.graph_emb_type is not None and (
            "local" in self.graph_type or "world" in self.graph_type
        ):
            self.graph_emb, _ = load_embeddings(opt, self.emb_loc, self.graph_emb_type)
            self.kg_graph = graph
            self.node_vocab = self.graph_emb.vocab
            for i, w in enumerate(self.node_vocab):
                self.node2id[w] = i

        torch.manual_seed(opt.seed)
        self.model = scorer.CommandScorerWithKG(
            opt.bidirectional,
            self.class_tree,
            self.word_emb,
            self.hyperbolic_emb,
            self.numberbatchID2hyperbolicID,
            self.graph_emb,
            self.graph_type,
            self.n_abstractions,
            hidden_size=self.hidden_size,
            device=device,
            opt=opt,
        )

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        # 0.00003
        self.optimizer = optim.Adam(self.model.parameters(), opt.lr)
        self.hist_scmds_size = opt.hist_scmds_size
        self.stats = {"episode": defaultdict(list), "game": defaultdict(list)}
        self.mode = "test"

    def start_episode(self, batch_size):
        # Called at the beginning of each episode
        self._episode_has_started = True
        if self.mode == "train":
            self.no_train_episodes += 1
            self.transitions = [[] for _ in range(batch_size)]
            self.stats["game"] = defaultdict(list)
        self.reset_parameters(batch_size)

    def end_episode(self):
        # Called at the end of each episode
        self._episode_has_started = False
        self.reset_graph()

        if self.mode == "train":
            for k, v in self.stats["game"].items():
                self.stats["episode"][k].append(np.mean(v, axis=0))
            if self.no_train_episodes % self.LOG_FREQUENCY == 0:
                msg = "{}. ".format(self.no_train_episodes)
                msg += "  " + "  ".join(
                    "{}: {:5.2f}".format(k, np.mean(v, axis=0))
                    for k, v in self.stats["episode"].items()
                )
                print(msg)
                self.stats["episode"] = defaultdict(list)  # reset stat

    def train(self, batch_size=1):
        self.mode = "train"
        self.model.train()
        self.no_train_step = 0
        self.no_train_episodes = 0

    def test(self, batch_size=1):
        self.mode = "test"
        self.model.eval()
        self.model.reset_hidden(batch_size)

    def reset_parameters(self, batch_size):
        # Called at the beginning of each batch
        self.agent_loc = ["" for _ in range(batch_size)]
        self.last_done = [False] * batch_size
        if self.mode == "train":
            self.last_score = tuple([0.0] * batch_size)
            self.batch_stats = [
                {"max": defaultdict(list), "mean": defaultdict(list)}
                for _ in range(batch_size)
            ]
        self.model.reset_hidden(batch_size)
        self.reset_graph()

    def reset_graph(self):
        self.world_graph = {}
        self.local_graph = {}
        self.rel_extractor.agent_loc = ""
        self.current_facts = defaultdict(
            set
        )  # unserialized facts, use serialize_facts() from utils

    @property
    def infos_to_request(self) -> EnvInfos:
        return EnvInfos(
            description=True,
            inventory=True,
            admissible_commands=True,
            won=True,
            lost=True,
            location=True,
            last_action=True,
            game=True,
            facts=True,
            entities=True,
            policy_commands=True,
            intermediate_reward=True,
        )  # Last line needed for ground truth local graph

    def get_local_graph(
        self, obs, hint, infos, cmd, prev_facts, graph_mode, prune_nodes
    ):

        if graph_mode == "full":
            # We get the full true state graph
            current_facts = process_full_facts(infos["game"], infos["facts"])
            current_triplets = serialize_facts(current_facts)  # planning graph triplets
            local_graph, entities = construct_graph(current_triplets)
        else:
            # For Graph type Evolve!
            if self.local_evolve_type == "direct":  # Similar to KG-DQN

                # Ok incredibly long state ???
                state = "{}\n{}\n{}".format(
                    obs, infos["description"], infos["inventory"]
                )
                # What are the hint! -> The commands that scored
                hint_str = " ".join(hint)
                prev_action = cmd
                if cmd == "restart":
                    prev_action = None

                # How do we extract the triplets!
                local_graph, current_facts = self.rel_extractor.fetch_triplets(
                    state + hint_str, infos["local_graph"], prev_action=prev_action
                )
                entities = self.rel_extractor.kg_vocab

            else:  # Ground-Truth, from textworld
                # Here partial observability is maintained ok
                current_facts = process_step_facts(
                    prev_facts, infos["game"], infos["facts"], infos["last_action"], cmd
                )
                current_triplets = serialize_facts(
                    current_facts
                )  # planning graph triplets
                local_graph, entities = construct_graph(current_triplets)
        return local_graph, current_facts, entities

    def get_world_graph(self, obs, hint, infos, graph_mode, prune_nodes):
        # hints could be list of goals or recipe to follow.
        # Options to choose for evolve graph: DC, CDC,  neighbours (with/ without pruning), manual
        add_edges = []
        if graph_mode == "full":
            if "goal_graph" in infos and infos["goal_graph"]:
                add_edges = [
                    [e.replace("_", " ") for e in edge] + ["AtLocation"]
                    for edge in infos["goal_graph"].edges
                ]

            entities = []
            prev_entities = infos["entities"]
            for entity in prev_entities:
                et_arr = re.split(r"[- ]+", entity)
                entity_nodes = any_substring_extraction(
                    entity, self.kg_graph, ngram=len(et_arr)
                )
                jentity = "_".join(et_arr)

                if not entity_nodes:  # No valid entry in the kg_graph
                    entity_nodes = set(jentity)
                for en in entity_nodes:
                    if en != jentity:
                        add_edges.append([en, jentity, "RelatedTo"])
                entities.extend(entity_nodes)
            graph = shortest_path_subgraph(self.kg_graph, nx.DiGraph(), entities)
            world_graph, entities = add_triplets_to_graph(graph, add_edges)
        else:
            prev_entities = (
                list(infos["world_graph"].nodes) if infos["world_graph"] else []
            )
            state = "{}\n{}".format(obs, infos["description"])
            hint_str = " ".join(hint)
            state_entities = self.tokenizer.extract_world_graph_entities(
                state, self.kg_graph
            )

            hint_entities = self.tokenizer.extract_world_graph_entities(
                hint_str, self.kg_graph
            )

            inventory_entities = self.tokenizer.extract_world_graph_entities(
                infos["inventory"], self.kg_graph
            )

            new_entities = list(
                (state_entities | hint_entities | inventory_entities)
                - set(prev_entities + self.tokenizer.ignore_list + self.pruned_concepts)
            )
            # list of entities
            world_graph = infos["world_graph"]
            node_weights = {}

            # Here the update of the graph is happening, do not touch
            if not nx.is_empty(world_graph):
                node_weights = nx.get_node_attributes(world_graph, "weight")

                # if 'sentinel_weight' in world_graph.graph:
                # sentinel_weight = world_graph.graph['sentinel_weight']

            if self.world_evolve_type == "DC":
                entities = prev_entities + new_entities
                world_graph = self.kg_graph.subgraph(entities).copy()
            elif "NG" in self.world_evolve_type:  # Expensive option
                if new_entities:
                    # Setting max_khop_degree to higher value results in adding high-degree nodes ==> noise
                    # cutoff =1 select paths of length 2 between every pair of nodes.
                    new_graph = khop_neighbor_graph(
                        self.kg_graph, new_entities, cutoff=1, max_khop_degree=100
                    )
                    world_graph = nx.compose(world_graph, new_graph)
            elif self.world_evolve_type == "manual":
                assert (
                    "manual_world_graph" in infos
                    and infos["manual_world_graph"]
                    and "graph" in infos["manual_world_graph"]
                ), "No valid manual world graph found. Use other options"
                select_entities = list(
                    set(infos["manual_world_graph"]["entities"]).intersection(
                        set(new_entities)
                    )
                )
                new_graph = khop_neighbor_graph(
                    infos["manual_world_graph"]["graph"], select_entities, cutoff=1
                )
                world_graph = nx.compose(world_graph, new_graph)
            else:  # default options = CDC
                if new_entities or inventory_entities:
                    command_entities = []
                    for cmd in infos["admissible_commands"]:
                        if "put" in cmd or "insert" in cmd:
                            entities = self.tokenizer.extract_world_graph_entities(
                                cmd, self.kg_graph
                            )
                            command_entities.extend(entities)
                    world_graph = shortest_path_subgraph(
                        self.kg_graph,
                        world_graph,
                        new_entities,
                        inventory_entities,
                        command_entities,
                    )

            # Prune Nodes, pruning based on what ?
            if (
                prune_nodes
                and not nx.is_empty(world_graph)
                and len(world_graph.nodes) > 10
            ):
                prune_count = int(len(world_graph.nodes) / 30)
                for _ in range(prune_count):
                    if any(node_weights):
                        rnode = min(node_weights, key=node_weights.get)
                        self.pruned_concepts.append(rnode)
                        # print('pruning ' + rnode)
                        world_graph.graph["sentinel_weight"] += node_weights[rnode]
                        if rnode in world_graph:
                            world_graph.remove_node(rnode)

            world_graph.remove_edges_from(nx.selfloop_edges(world_graph))
            entities = list(world_graph.nodes)

        return world_graph, entities

    def update_current_graph(
        self, obs, cmd, hints, infos, graph_mode, prune_nodes=False
    ):
        # hints could be list of goals or recipe to follow.

        batch_size = len(obs)
        info_per_batch = [{k: v[i] for k, v in infos.items()} for i in range(len(obs))]
        for b in range(batch_size):

            if "local" in self.graph_type:
                self.rel_extractor.agent_loc = self.agent_loc[b]
                info_per_batch[b]["local_graph"] = (
                    self.local_graph[b] if b in self.local_graph else nx.DiGraph()
                )
                local_graph, current_facts, _ = self.get_local_graph(
                    obs[b],
                    hints[b],
                    info_per_batch[b],
                    cmd[b],
                    self.current_facts[b],
                    graph_mode,
                    prune_nodes,
                )
                self.agent_loc[b] = self.rel_extractor.agent_loc
                self.local_graph[b] = local_graph
                self.current_facts[b] = current_facts

            if "world" in self.graph_type:
                info_per_batch[b]["world_graph"] = (
                    self.world_graph[b] if b in self.world_graph else nx.DiGraph()
                )
                # info_per_batch[b]["goal_graph"] = infos["goal_graph"][b] if 'goal_graph' in infos else None
                world_graph, _ = self.get_world_graph(
                    obs[b], hints[b], info_per_batch[b], graph_mode, prune_nodes
                )

                self.world_graph[b] = world_graph

    def _process(self, texts, vocabulary, sentinel=False):
        new_texts = []
        masks = []
        for word in texts:
            text, mask = self.tokenizer.extract_entity_ids(
                word, vocabulary, self.class_tree, self.n_abstractions
            )
            new_texts.append(text)
            masks.append(mask)
        texts = new_texts
        # Nested List: Dimensions: Batch, Sentence, Abstraction
        max_len = max(len(l) for l in texts)
        num_items = (
            len(texts) + 1 if sentinel else len(texts)
        )  # Add sentinel entry here for the attention mechanism
        if "<PAD>" in vocabulary:
            padded = (
                np.ones((num_items, max_len, self.n_abstractions)) * vocabulary["<PAD>"]
            )
        else:
            print(
                "Warning: No <PAD> found in the embedding vocabulary. Using the id:0 for now."
            )
            padded = np.zeros((num_items, max_len, self.n_abstractions))

        padded_mask = np.zeros((num_items, max_len, self.n_abstractions))
        for i, text in enumerate(texts):
            padded[i, : len(text)] = text
            padded_mask[i, : len(text)] = masks[i]

        padded_tensor = to_tensor(padded, self.device)
        padded_mask = to_tensor(padded_mask, self.device)

        return padded_tensor, padded_mask

    def _discount_rewards(self, batch_id, last_values):
        returns, advantages = [], []
        R = last_values.data
        for t in reversed(range(len(self.transitions[batch_id]))):
            rewards, _, _, values, _ = self.transitions[batch_id][t]
            R = torch.tensor(rewards) + self.GAMMA * R
            adv = R - values
            returns.append(R)
            advantages.append(adv)

        return returns[::-1], advantages[::-1]

    def _detached_cum_sum(self, logits):
        cum_sum = 0
        detached_logits = []
        for elem in logits:
            new_logits = elem + cum_sum
            detached_logits.append(new_logits)
            cum_sum = new_logits.detach()
        detached_logits = torch.stack(detached_logits, dim=0)
        return detached_logits

    def act(
        self,
        obs: str,
        score: int,
        done: bool,
        infos: Mapping[str, Any],
        scored_commands: list,
        random_action=False,
    ):
        batch_size = len(obs)
        if not self._episode_has_started:
            self.start_episode(batch_size)

        just_finished = [done[b] != self.last_done[b] for b in range(batch_size)]
        sel_rand_action_idx = [
            np.random.choice(len(infos["admissible_commands"][b]))
            for b in range(batch_size)
        ]
        if random_action:
            return [
                infos["admissible_commands"][b][sel_rand_action_idx[b]]
                for b in range(batch_size)
            ]

        torch.autograd.set_detect_anomaly(True)
        input_t = []

        # Build agent's observation: feedback + look + inventory.
        state = [
            "{}\n{}\n{}\n{}".format(
                obs[b],
                infos["description"][b],
                infos["inventory"][b],
                " \n".join(scored_commands[b]),
            )
            for b in range(batch_size)
        ]

        # Tokenize and pad the input and the commands to chose from.
        state_tensor, mask_state = self._process(state, self.word2id)

        # Preprocess Commands to guarantee reproducibility
        for b in range(batch_size):
            new_commands = []
            for elem in infos["admissible_commands"][b]:
                new_command = []
                new_command = " ".join(
                    [word for word in elem.split(" ") if word != "the"]
                )
                new_commands.append(new_command)
            new_commands = list(sorted(list(set(new_commands))))
            infos["admissible_commands"][b] = new_commands

        command_list = []
        for b in range(batch_size):
            cmd_b, mask_command = self._process(
                infos["admissible_commands"][b], self.word2id
            )
            command_list.append(cmd_b)

        max_num_candidate = max_len(infos["admissible_commands"])
        max_num_word = max([cmd.size(1) for cmd in command_list])
        commands_tensor = to_tensor(
            np.zeros(
                (batch_size, max_num_candidate, max_num_word, self.n_abstractions)
            ),
            self.device,
        )
        for b in range(batch_size):
            commands_tensor[
                b, : command_list[b].size(0), : command_list[b].size(1)
            ] = command_list[b]

        localkg_tensor = torch.FloatTensor()
        localkg_adj_tensor = torch.FloatTensor()
        worldkg_tensor = torch.FloatTensor()
        worldkg_adj_tensor = torch.FloatTensor()
        localkg_hint_tensor = torch.FloatTensor()
        worldkg_hint_tensor = torch.FloatTensor()

        if self.graph_emb_type is not None and (
            "local" in self.graph_type
            or "world" in self.graph_type
            or "class" in self.graph_type
        ):
            # prepare Local graph and world graph ....
            # Extra empty node (sentinel node) for no value_lossattention option
            #  (Xiong et al ICLR 2017 and https://arxiv.org/pdf/1612.01887.pdf)
            if "world" in self.graph_type:
                # Get the entities of the current world graph
                world_entities = []
                for b in range(batch_size):
                    world_entities.extend(self.world_graph[b].nodes())
                world_entities = set(world_entities)
                wentities2id = dict(zip(world_entities, range(len(world_entities))))
                max_num_nodes = (
                    len(wentities2id) + 1 if self.sentinel_node else len(wentities2id)
                )
                wentities2id_new = {}
                for key, value in wentities2id.items():
                    new_key = " ".join(key.split("_"))
                    wentities2id_new[new_key] = value
                worldkg_tensor, _ = self._process(
                    wentities2id_new, self.node2id, sentinel=self.sentinel_node
                )
                world_adj_matrix = np.zeros(
                    (batch_size, self.n_abstractions, max_num_nodes, max_num_nodes),
                    dtype="float32",
                )
                for b in range(batch_size):
                    # get adjacentry matrix for each batch based on the all_entities
                    triplets = [
                        list(edges)
                        for edges in self.world_graph[b].edges.data("relation")
                    ]
                    for [e1, e2, r] in triplets:
                        e1 = wentities2id[e1]
                        e2 = wentities2id[e2]
                        world_adj_matrix[b, :, e1, e2] = 1.0
                        world_adj_matrix[b, :, e2, e1] = 1.0  # reverse relation
                    for e1 in list(self.world_graph[b].nodes):
                        e1 = wentities2id[e1]
                        world_adj_matrix[b, :, e1, e1] = 1.0
                    if self.sentinel_node:  # Fully connected sentinel
                        world_adj_matrix[b, -1, :] = np.ones(
                            (max_num_nodes), dtype="float32"
                        )
                        world_adj_matrix[b, :, -1] = np.ones(
                            (max_num_nodes), dtype="float32"
                        )
                worldkg_adj_tensor = to_tensor(
                    world_adj_matrix, self.device, type="float"
                )

            if "local" in self.graph_type:
                local_entities = []
                for b in range(batch_size):
                    local_entities.extend(self.local_graph[b].nodes())
                local_entities = set(local_entities)
                lentities2id = dict(zip(local_entities, range(len(local_entities))))
                max_num_nodes = (
                    len(lentities2id) + 1 if self.sentinel_node else len(lentities2id)
                )
                localkg_tensor = self._process(
                    lentities2id, self.word2id, sentinel=self.sentinel_node
                )
                local_adj_matrix = np.zeros(
                    (batch_size, max_num_nodes, max_num_nodes), dtype="float32"
                )
                for b in range(batch_size):
                    # get adjacentry matrix for each batch based on the all_entities
                    triplets = [
                        list(edges)
                        for edges in self.local_graph[b].edges.data("relation")
                    ]
                    for [e1, e2, r] in triplets:
                        e1 = lentities2id[e1]
                        e2 = lentities2id[e2]
                        local_adj_matrix[b][e1][e2] = 1.0
                        local_adj_matrix[b][e2][e1] = 1.0
                    for e1 in list(self.local_graph[b].nodes):
                        e1 = lentities2id[e1]
                        local_adj_matrix[b][e1][e1] = 1.0
                    if self.sentinel_node:
                        local_adj_matrix[b][-1, :] = np.ones(
                            (max_num_nodes), dtype="float32"
                        )
                        local_adj_matrix[b][:, -1] = np.ones(
                            (max_num_nodes), dtype="float32"
                        )
                localkg_adj_tensor = to_tensor(
                    local_adj_matrix, self.device, type="float"
                )

            if len(scored_commands[0]) > 0:
                # Get the scored commands as one string
                hint_str = [
                    " \n".join(scored_commands[b][-self.hist_scmds_size :])
                    for b in range(batch_size)
                ]
            else:
                hint_str = [
                    obs[b] + " \n" + infos["inventory"][b] for b in range(batch_size)
                ]

            worldkg_hint_tensor, _ = self._process(
                hint_str, self.node2id
            )  

        input_t.append(state_tensor)
        input_t.append(commands_tensor)
        input_t.append([mask_state, mask_command])
        input_t.append(localkg_tensor)
        input_t.append(localkg_hint_tensor)
        input_t.append(localkg_adj_tensor)
        input_t.append(worldkg_tensor)
        input_t.append(worldkg_hint_tensor)
        input_t.append(worldkg_adj_tensor)

        # Apply model and map index to action
        outputs, indexes, values = self.model(*input_t, test=self.mode)
        outputs, indexes, values = (
            outputs,
            indexes.view(batch_size),
            values.view(batch_size),
        )
        sel_action_idx = [indexes[b] for b in range(batch_size)]
        action = [
            infos["admissible_commands"][b][sel_action_idx[b]]
            for b in range(batch_size)
        ]

        # Update Model
        if any(done):
            self.number_of_dones += 1

        if any(done):
            for b in range(batch_size):
                if done[b]:
                    self.model.reset_hidden_per_batch(b)
                    action[b] = "look"

        if self.mode == "test":
            self.last_done = done
            return action

        self.no_train_step += 1
        last_score = list(self.last_score)
        for b, score_b in enumerate(score):
            # Update local/world graph attention weights
            if "world" in self.graph_type:
                with torch.no_grad():
                    att_wts = self.model.world_attention[b].flatten().cpu().numpy()
                edge_attr = dict(zip(wentities2id.keys(), att_wts))
                nx.set_node_attributes(self.world_graph[b], edge_attr, "weight")
                self.world_graph[b].graph["sentinel_weight"] = att_wts[-1]
            if "local" in self.graph_type:
                with torch.no_grad():
                    att_wts = (
                        self.model.local_attention[b]
                        .flatten()
                        .cpu()
                        .numpy()(score_b - last_score[b])
                    )
                edge_attr = dict(zip(lentities2id.keys(), att_wts))
                nx.set_node_attributes(self.local_graph[b], edge_attr, "weight")
                self.local_graph[b].graph["sentinel_weight"] = att_wts[-1]

            if self.transitions[b]:
                reward = score_b - last_score[b]
                reward = reward + 100 if infos["won"][b] else reward
                reward = reward - 100 if infos["lost"][b] else reward
                self.transitions[b][-1][0] = reward  # Update reward information.
                last_score[b] = score_b

            if self.no_train_step % self.UPDATE_FREQUENCY == 0 or just_finished[b]:
                returns, advantages = self._discount_rewards(b, values[b])
                batch_loss = 0
                for transition, ret, advantage in zip(
                    self.transitions[b], returns, advantages
                ):
                    reward, indexes_, outputs_, values_, done_ = transition
                    if done_:
                        continue
                    advantage = advantage.detach()  # Block gradients flow here.

                    if self.residual == "residual":
                        outputs = self._detached_cum_sum(torch.flip(outputs_, (0,)))
                        probs = F.softmax(
                            outputs, dim=-1
                        )  # abstraction x batch x n_cmds
                        log_probs = F.log_softmax(outputs, dim=-1)
                        log_action_probs = log_probs[
                            list(range(log_probs.shape[0])), indexes_
                        ]
                        policy_loss = torch.mean(-log_action_probs * advantage)
                        value_loss = 0.5 * (values_ - ret) ** 2.0
                        entropy = (-probs * log_probs).sum()
                        confidence = log_action_probs[-1]

                    else:
                        # Sum approach
                        outputs = torch.sum(outputs_, dim=0)
                        probs = F.softmax(outputs, dim=-1)
                        log_probs = F.log_softmax(outputs, dim=-1)
                        log_action_probs = log_probs[indexes_]
                        # log_action_probs = log_probs.gather(1, indexes_.view(batch_size, 1))
                        policy_loss = -log_action_probs * advantage
                        value_loss = 0.5 * (values_ - ret) ** 2.0
                        entropy = (-probs * log_probs).sum()
                        confidence = log_action_probs

                    batch_loss += policy_loss + 0.5 * value_loss - self.beta * entropy

                    self.batch_stats[b]["mean"]["reward"].append(reward)
                    self.batch_stats[b]["mean"]["policy"].append(policy_loss.item())
                    self.batch_stats[b]["mean"]["value"].append(value_loss.item())
                    self.batch_stats[b]["mean"]["entropy"].append(entropy.item())
                    self.batch_stats[b]["mean"]["confidence"].append(
                        torch.exp(confidence).item()
                    )

                if batch_loss != 0:
                    batch_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 40)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.batch_stats[b]["mean"]["loss"].append(batch_loss.item())

                self.transitions[b] = []
            else:
                # Keep information about transitions for Truncated Backpropagation Through Time.
                # Reward will be set on the next call
                self.transitions[b].append(
                    [None, indexes[b], outputs[:, b], values[b], done[b]]
                )
            self.batch_stats[b]["max"]["score"].append(
                score_b / infos["game"][b].max_score
            )

        self.last_score = tuple(last_score)
        self.last_done = done

        if all(done):  # Used at the end of the batch to update epsiode stats
            for b in range(batch_size):
                for k, v in self.batch_stats[b]["mean"].items():
                    self.stats["game"][k].append(np.mean(v, axis=0))
                for k, v in self.batch_stats[b]["max"].items():
                    self.stats["game"][k].append(np.max(v, axis=0))

        if self.epsilon > 0.0:
            rand_num = torch.rand(
                (1,), device=self.device
            )  # np.random.uniform(low=0.0, high=1.0, size=(1,))
            less_than_epsilon = (rand_num < self.epsilon).long()  # batch
            greater_than_epsilon = 1 - less_than_epsilon
            choosen_idx = (
                less_than_epsilon * sel_rand_action_idx
                + greater_than_epsilon * sel_action_idx
            )
            action = infos["admissible_commands"][choosen_idx]

        with torch.no_grad():
            if self.residual=="residual":
                outputs = torch.cumsum(outputs,dim=0)
                entropy = -torch.sum(F.softmax(outputs,dim=-1)*F.log_softmax(outputs,dim=-1)).item()
            else:
                outputs = torch.sum(outputs,dim=0)
                entropy = -torch.sum(F.softmax(outputs,dim=-1)*F.log_softmax(outputs,dim=-1)).item()

        return action, entropy
