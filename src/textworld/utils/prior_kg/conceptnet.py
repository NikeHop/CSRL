from collections import defaultdict
import pickle
import os
import random
from typing import List

import pandas as pd  
import tqdm

def get_superclasses_local(cl:str,dataframe:pd.DataFrame)->list:
    """
    Get all superclasses for a given class
    """
    superclasses = []
    rows = dataframe.loc[dataframe['start']==f'/c/en/{cl}']
    for id,row in rows.iterrows():
        end = row['end'].split('/')[-1]
        superclasses.append(end)
    return superclasses

def get_single_class_tree(cl:str,
                          max_per_level:int,
                          n_levels:int,
                          dataframe:pd.DataFrame)->dict:
    '''
    Given a dataframe of triplets and a class extracts the corresponding class tree
    Since elements in ConceptNet can have multiple superclasses, a class tree takes the format
    level->superclasses, where the number of superclasses is constrained by max_per_level
    '''

    class_tree = defaultdict(list)
    level = 0 
    class_tree = {}
    if " " in cl:
        cl="_".join(cl.split(" "))
    superclasses = [cl]

    # As long as superclasses exist and the depth of the superclass tree is below n-levels
    # Find the superclasses for each class 
    while len(superclasses)>0 and level<=n_levels:
        level += 1
        next_level_superclasses = {}
        for c in superclasses:
            # In case of a compound noun, ConceptNet concatenates them with '_'
            if " " in c:
                c = "_".join(c.split(" "))
                # Remove articles
                c = c.strip("a_")
            next_level_superclasses[c] = get_superclasses_local(c,dataframe)
            # In case no superclass is found but it is a compound noun, try the last component
            if len(next_level_superclasses[c])==0:
                compounds = c.split('_')
                if len(compounds)>1:
                    next_level_superclasses[c] = get_superclasses_local(compounds[-1],dataframe)
                
        # Choose max_per_level randomly
        if len(superclasses)>max_per_level:
            # Choose the classes from which to sample the superclasses 
            chosen_base_classes = random.sample(superclasses,max_per_level)
            next_level_superclasses_alias = {}
            for cl in chosen_base_classes:
                next_level_superclasses_alias[cl] = random.sample(next_level_superclasses[cl],1)

        else:
            # Choose at least one of each, the rest sample randomly if max_per_level is too high
            next_level_superclasses_alias = {}
            rest = max_per_level-len(superclasses)
            sample = (max_per_level<sum([len(value) for value in next_level_superclasses.values()]))
            for key, value in next_level_superclasses.items():
                if len(value)>0:
                    if sample:
                        add = random.randint(0,rest)
                        add_on = value[1:1+add]
                        if not isinstance(add_on,list):
                            add_on = list(add_on)
                        next_level_superclasses_alias[key] = [value[0]] + add_on
                        rest -= min(len(value)-1,add)
                    else:
                        next_level_superclasses_alias[key] = value
            
        # Filter out doubles
        class_tree.update(next_level_superclasses_alias)
        # Update the superclasses for the next level
        superclasses = []
        for value in next_level_superclasses_alias.values():
            superclasses += value
    
    # Write class trees in form of level and superclasses
    new_class_tree = {}
    if len(class_tree)!=0:
        start_superclasses = class_tree[cl]
        new_class_tree[0] = start_superclasses
        for level in range(1,n_levels):
            current_classes = []
            previous_classes = new_class_tree[level-1]
            for elem in previous_classes:
                if elem in class_tree:
                    current_classes += class_tree[elem]
            new_class_tree[level] = current_classes
    
    return new_class_tree

def extract_allowable_classes(class_trees:dict,entities:list)->dict:
    allowed_classes = defaultdict(lambda :0)
    for ent, tree in class_trees.items():
        for key, cls_ in tree.items():
            for cl in cls_:
                allowed_classes[cl]+=1
            
    allowed_classes = {key:True for key,value in allowed_classes.items() if value>1}
    allowed_classes.update({ent: True for ent in entities})

    return allowed_classes

def prune_class_tree(class_trees:dict,allowed_classes:dict)->dict:
    pruned_class_trees = {}
    for ent, cl_tree in class_trees.items():
        new_tree = {}
        for level, cls_ in cl_tree.items():
            new_tree[level] = [cl for cl in cls_ if cl in allowed_classes]
            if len(new_tree[level])==0:
                new_tree[level]=cls_
        pruned_class_trees[ent] = cl_tree

    return pruned_class_trees 

def get_class_tree_conceptnet(entities:List[str],max_per_level:int,n_levels:int,load:bool,kg_directory:str)->dict:
    class_trees = {}

    # Load subnet 
    class_file = os.path.join(kg_directory,'class_conceptnet.csv')
    dataframe = pd.read_csv(class_file,header='infer')

    # For each class get a mapping between level and superclasses
    print('Extract graphs out of Concept Net for each entity')
    if not load:
        n_not_found_entities = 0 
        for entity in tqdm.tqdm(entities):
            cl_tree = get_single_class_tree(entity,max_per_level,n_levels,dataframe)
            if cl_tree=={}:
                n_not_found_entities+=1
            class_trees[entity] = cl_tree
        
        with open(os.path.join(kg_directory,'concept_net.pkl'),'wb+') as f:
            pickle.dump(class_trees,f)
    else:
        with open(os.path.join(kg_directory,'concept_net.pkl'),'rb') as f:
            class_trees = pickle.load(f)
    
    # Fill ins in case of missing superclasses
    new_class_tree = {}
    for elem,item in class_trees.items():
        # If no superclasses could be found, at the class itself as superclass
        if len(item)==0:
            new_class_tree[elem] = {i:[elem] for i in range(n_levels)}
        # Fill up the superclasses for empty levels by copying prior levels
        elif len(item[n_levels-1])==0:
            for l, superclasses in item.items():
                if len(superclasses)==0:
                    max_level = l
            last_classes = item[l-1]
            item.update({i:last_classes for i in range(max_level,n_levels)})
            new_class_tree[elem]=item
        else:
            new_class_tree[elem] = item 
    
    class_trees = new_class_tree

    # Only superclasses that appear more than once can help with abstraction
    # Remove other superclasses
    allowable_classes = extract_allowable_classes(class_trees,entities)
    class_trees = prune_class_tree(class_trees,allowable_classes)
    
    return class_trees
           










