from argparse import Namespace
import os
from typing import List, Union, Tuple

from utils.prior_kg.class_tree import ClassTree, MultiClassTree


def load_entities(opt: Namespace) -> List:
    entities = []
    with open(
        os.path.join(opt.game_dir, opt.difficulty_level, "entities.txt"), "r"
    ) as f:
        for line in f:
            entities.append(line.strip("\n"))
    entities = list(entities)
    return entities


def text_abstraction(
    text: List, class_tree: Union[ClassTree, MultiClassTree], n_abstractions: int
) -> Tuple[List, List]:
    """
    Given a list of words form their abstractions by mapping objects to their superclasses
    """
    if len(text) == 0:
        empty1 = []
        empty2 = []
        return empty1, empty2

    # Split any compound words
    new_text = []
    for word in text:
        new_text += word.split("_")
    text = new_text

    max_len_entity = 2  # At most two word can form an entity to abstract away from
    queues = {i + 1: [] for i in range(max_len_entity)}

    abstracted_text = []
    mask = []
    compound_word = 0
    for w in range(len(text) - 1):
        if compound_word:
            compound_word -= 1
            continue

        new_word = []
        new_mask = []

        # Check the next n-words whether they form an entity
        found_entity = False
        for key in queues.keys():
            queues[key] = text[w : w + 1 + key - 1]

        # If the manual flag holds, words are copied to the next abstraction level
        # without any abstraction
        if not class_tree.manual:
            for i in sorted(queues.keys(), reverse=True):
                entity_string = "_".join(queues[i])
                if entity_string in class_tree.word2id:
                    found_entity = True
                    for k in range(n_abstractions):
                        new_word.append(entity_string)
                        new_mask.append(0)
                        entity_id = class_tree.word2id[entity_string]
                        if (
                            entity_id
                            in class_tree.abstractions_up_to_down[class_tree.depth - k]
                        ):
                            entity_id = class_tree.class2superclass[entity_id][0]
                            entity_string = class_tree.id2word[entity_id]
                    if i != 1:
                        compound_word = i - 1
                    break

        # Add elem if it cannot be abstracted
        if not found_entity:
            for _ in range(n_abstractions):
                new_word.append(queues[1][0])
                new_mask.append(1)

        abstracted_text.append(new_word)
        mask.append(new_mask)

    # Do the same procedure as in for-loop for last word
    # Outside of for loop because one word look ahead not possible
    new_word = []
    new_mask = []
    if not compound_word:
        entity_string = text[-1]
        if entity_string in class_tree.word2id and not class_tree.manual:
            for k in range(n_abstractions):
                new_word.append(entity_string)
                new_mask.append(0)
                entity_id = class_tree.word2id[entity_string]
                if (
                    entity_id
                    in class_tree.abstractions_up_to_down[class_tree.depth - k]
                ):
                    entity_id = class_tree.class2superclass[entity_id][0]
                    entity_string = class_tree.id2word[entity_id]
        else:
            for _ in range(n_abstractions):
                new_word.append(text[-1])
                new_mask.append(1)
        abstracted_text.append(new_word)
        mask.append(new_mask)
    return abstracted_text, mask