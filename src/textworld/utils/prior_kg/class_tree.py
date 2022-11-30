from collections import defaultdict

import numpy as np 
import torch 

class ClassTree(object):
    
    def __init__(self, class_tree, entities,type_):
        self.raw_dict = class_tree
        self.class2superclass = defaultdict(list)
        self.class2subclass = defaultdict(list)
        
        # The root node should not be a key
        if type_=='dbpedia':
            if 'thing' in self.raw_dict:
                del self.raw_dict['thing']

        self.entities2id = {name: idd+1 for idd, name in enumerate(entities)}
        current_id = len(self.entities2id) + 1

        for class_, superclasses in self.raw_dict.items():
            superclass = superclasses[0].split('.')[0]
            class_ = class_.split('.')[0]

            if class_  not in self.entities2id:
                self.entities2id[class_] = current_id
                current_id += 1
            class_id = self.entities2id[class_]
        
            if superclass not in self.entities2id:
                self.entities2id[superclass] = current_id
                current_id += 1
            superclass_id = self.entities2id[superclass]
            self.class2superclass[class_id] = [superclass_id]
            
        # Invert class2superclass relation
        for class_,superclasses in self.class2superclass.items():
            for superclass in superclasses: 
                self.class2subclass[superclass].append(class_)
        
        self.id2entities = {value:key for key,value in self.entities2id.items()}
        
        # Determine the roots and the leaves
        roots, leaves = [], []
        for id in self.entities2id.values():
            if id not in self.class2subclass:
                leaves.append(id)
            if id not in self.class2superclass:
                roots.append(id)

        
        # Determine the different levels of abstractions
        self.abstractions_down_to_up = defaultdict(list)
        self.abstractions_up_to_down = defaultdict(list)

        # Determine the tree layers starting from the bottom (down-to-up)
        self.depth = 1
        current_classes = leaves
        self.abstractions_down_to_up[self.depth] = current_classes

        while len(current_classes)>0:
            new_classes = []
            for cl in current_classes:
                new_classes += [super_cl for super_cl in self.class2superclass[cl]]
            if len(new_classes)!=0:
                self.abstractions_down_to_up[self.depth] = new_classes
                self.depth += 1
            current_classes = new_classes

        # Determine the tree layers starting from the top (up to down)
        self.depth = 1
        current_classes = roots
        self.abstractions_up_to_down[self.depth] = current_classes

        while len(current_classes)>0:
            new_classes = []
            for cl in current_classes:
                new_classes += [super_cl for super_cl in self.class2subclass[cl]]
            if len(new_classes)!=0:
                self.abstractions_up_to_down[self.depth] = new_classes
                self.depth += 1
            current_classes = new_classes
        self.depth -= 1

       
        # Add 0 as padding variable to the class2subclass and class2superclass 
        self.class2subclass[0].append(0)
        self.class2superclass[0].append(0)

        # Map IDs to words not semantics 
        self.word2id = self.entities2id
        self.id2word = self.id2entities

        
        

class MultiClassTree():

    def __init__(self,class_trees,n_levels):
        '''
        class_trees is a dictionary with: (key:entity, value:class_tree)
        '''
        self.depth = n_levels

        # Build up abstraction
        abstraction = {i:{} for i in range(n_levels,-1,-1)}
        self.word2id = {}
        wordid = 0
        self.class_name2group = defaultdict(list)
        class_group = 0

        for elem,tree in class_trees.items():
            if elem not in self.word2id:
                self.word2id[elem]=wordid
                wordid += 1

            for level, superclasses in tree.items():
                # Give this group of superclasses a name 
                class_name = f'class_group_{class_group}'
                class_group += 1
                self.class_name2group[class_name] = superclasses
                self.word2id[class_name] = wordid
                wordid += 1
                
                abstraction[self.depth-level][self.word2id[elem]] = self.word2id[class_name]

        self.id2word = {value:key for key, value in self.word2id.items()}
        self.abstractions_up_to_down = abstraction

        self.class2superclass = {}
        for level, value in self.abstractions_up_to_down.items():
            for elem, superclass in value.items():
                self.class2superclass[elem] = [superclass]
