from collections import defaultdict
from copy import deepcopy
import pickle
import os

from graphviz import Graph
from nltk.corpus import wordnet
import tqdm


def prune_class_tree(class_tree: dict, instances: list) -> None:
    """
    Removes chains of subclasses from the class tree as they do not aid with abstraction.
    """
    children = [(child, "entity.n.01") for child in class_tree["entity.n.01"]]
    while len(children) > 0:
        subclass, parent = children.pop()
        if len(class_tree[subclass]) == 1 and subclass not in instances:
            children.append((class_tree[subclass][0], parent))
            class_tree[parent].append(class_tree[subclass][0])
            del class_tree[subclass]
        else:
            children += [(child, subclass) for child in class_tree[subclass]]


def print_class_tree(class_tree: dict, directory: str) -> None:
    """
    Saves a visualization of the class tree in the given directory
    """
    g = Graph("wordnet_class_tree")
    nodes = {}
    for node, children in class_tree.items():
        if node not in nodes:
            g.node(node, label=node)
        for child in children:
            if child not in nodes:
                g.node(child, label=child)
            g.edge(node, child)

    g.render(directory=directory)


def invert_tree(tree):
    """
    Reverses the mapping from (class->subclasses) to (class->superclass).
    """
    inverted_tree = defaultdict(list)
    for node, children in tree.items():
        for child in children:
            inverted_tree[child].append(node)
    return inverted_tree


def merge_class_trees(class_trees: list) -> dict:
    """
    Takes a list of class trees and merges them into one class tree.
    """
    base_tree = class_trees[0]
    for cl_tree in class_trees:
        for node, children in cl_tree.items():
            if node not in base_tree:
                base_tree[node] = children
            else:
                for child in children:
                    if child not in base_tree[node]:
                        base_tree[node].append(child)

    # Remove the semantics and only keep the concept names
    base_tree_stringified = defaultdict(list)
    for node, children in base_tree.items():
        if isinstance(node, tuple):
            base_tree_stringified[node[0]] = children
        else:
            base_tree_stringified[node] = children
    return base_tree_stringified


def get_class_hierachies(instances: list) -> list:
    """
    For a list of entities, query the class tree from Wordnet for each entity.
    Due to simplifying assumptions the class tree here is a chain of superclass relations
    """
    class_trees = []
    for instance in tqdm.tqdm(instances):
        cl_hierachy = defaultdict(list)
        instance_components = instance.split(" ")
        if len(instance_components) > 1:
            # For compound nouns choose the last noun to query class
            synsets = wordnet.synsets(instance_components[-1])
            synsets = [synset for synset in synsets if synset.pos() == "n"]
            if len(synsets) > 0:
                synset = synsets[0]
            else:
                # No synset found, map instance to root node
                cl_hierachy[instance] = []
                cl_hierachy[instance_components[-1]].append(instance)
                cl_hierachy["entity.n.01"] = [instance_components[-1]]
                class_trees.append(cl_hierachy)
                continue
        else:
            # Not a compound noun, proceed normally
            synsets = wordnet.synsets(instance)
            synsets = [synset for synset in synsets if synset.pos() == "n"]
            if len(synsets) > 0:
                synset = synsets[0]
            else:
                # No synset found, map instance to root node
                cl_hierachy["entity.n.01"].append(instance)
                class_trees.append(cl_hierachy)
                continue

        # Query hypernyms recursively:
        hypernyms = [
            hypernym
            for hypernym in synset.hypernyms() + synset.instance_hypernyms()
            if hypernym.pos() == "n"
        ]
        if len(hypernyms) == 0:
            # No hypernym exist
            if len(instance_components) > 1:
                cl_hierachy[instance] = []
                cl_hierachy[instance_components[-1]].append(instance)
                cl_hierachy["entity.n.01"] = [instance_components[-1]]
            else:
                cl_hierachy["entity.n.01"].append(instance)
            class_trees.append(cl_hierachy)
            continue
        else:
            # If hypernym exist the instance must be registered in the class hierachy
            if len(instance_components) > 1:
                cl_hierachy[instance_components[-1]].append(instance)
                cl_hierachy[instance] = []
            else:
                cl_hierachy[instance] = []
            i = 0
            while len(hypernyms) > 0:
                # Only look at first hypernym
                hypernym = hypernyms[0]

                # Determine names to put into class hierachy
                synset_name = synset.name()
                if i == 0 and synset_name != instance:
                    # Register instance name not synset name in class hierarchy
                    if len(instance_components) > 1:
                        synset_name = instance_components[-1]
                    else:
                        synset_name = instance

                # Add to current hypernym to class tree
                cl_hierachy[hypernym.name()].append(synset_name)

                # Prepare next recursion
                synset = hypernym
                hypernyms = [
                    h
                    for h in hypernym.hypernyms() + hypernym.instance_hypernyms()
                    if h.pos() == "n"
                ]
                i += 1
        class_trees.append(cl_hierachy)
    return class_trees


def collapse_class_tree(
    class_tree: dict, instances: list, start: int, end: int
) -> None:
    """
    Collapse layers of the class tree to reduce the number of abstraction layers
    """
    # Determine the classes on each level
    classes_per_level = {0: class_tree["entity.n.01"]}
    current_classes = class_tree["entity.n.01"]
    current_level = 1
    while len(current_classes) != 0:
        next_level_classes = []
        for elem in current_classes:
            next_level_classes += class_tree[elem]
        classes_per_level[current_level] = next_level_classes
        current_level += 1
        current_classes = next_level_classes

    # Take the classes from the start level,
    # for each class check whether their subclass is an instance of a class
    # or in the set of end_classes, if not remove.

    start_classes = classes_per_level[start]
    end_classes = classes_per_level[end]
    for cl in start_classes:
        subclasses = class_tree[cl]
        new_subclasses = []
        while len(subclasses) > 0:
            c = subclasses.pop()
            if c in end_classes:
                new_subclasses.append(c)
            elif c in instances:
                new_subclasses.append(c)
            else:
                subclasses += class_tree[c]
                class_tree.pop(c)
        class_tree[cl] = new_subclasses


def get_class_tree_wordnet(
    instances: list,
    collapse: bool,
    start: int,
    end: int,
    load: bool,
    kg_directory: str,
    visualize=False,
):
    """
    Takes a list of entities and returns the class tree from WordNet as a dictionary
    """

    if load:
        with open(os.path.join(kg_directory, "wordnet.pkl"), "rb+") as f:
            class_tree = pickle.load(f)
    else:
        class_trees = get_class_hierachies(instances)
        class_tree = merge_class_trees(class_trees)
        prune_class_tree(class_tree, instances)
        if collapse:
            collapse_class_tree(class_tree, instances, start, end)
        class_tree = invert_tree(class_tree)
        if not os.path.exists(kg_directory):
            os.mkdir(kg_directory)
        with open(os.path.join(kg_directory, "wordnet.pkl"), "wb+") as f:
            pickle.dump(class_tree, file=f)

    if visualize:
        print_class_tree(class_tree, "./visualisations")

    return class_tree
