from collections import defaultdict

class ClassTree(object):
    
    def __init__(self, 
                 class_tree:dict,
                 entities2id:dict)->None:
        
        # Maps each entity (object/class) to its superclass
        self.raw_dict = class_tree 
        self.class2superclass = defaultdict(list)
        self.class2subclass = defaultdict(list)
        
        self.entities2id = {entity: idd+1 for entity,idd in entities2id.items()}
        current_id = len(self.entities2id) + 1
        for class_, superclasses in self.raw_dict.items():
            superclass = superclasses[0]

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
            self.class2subclass[superclasses[0]].append(class_)
        
        # Create vocabulary
        self.id2entities = {value:key for key,value in self.entities2id.items()}
        self.id2word = {}
        for key, value in self.id2entities.items():
            w = value.split('.')[0]
            self.id2word[key] = w
        self.word2id = {value: key for key, value in self.id2word.items()}

        ##### Determine the classes for each layer of the tree ########

        # Determine the roots and the leaves
        roots, leaves = [], []
        for id in self.entities2id.values():
            if id not in self.class2subclass:
                leaves.append(id)
            if id not in self.class2superclass:
                roots.append(id)

        
        # Determine the different levels of abstractions
        self.abstraction_layers = defaultdict(dict)
        self.depth = 0 
        current_classes = roots

        self.abstraction_layers[self.depth] = {cl:True for cl in current_classes}
        self.depth += 1
        while len(current_classes)>0:
            new_classes = []
            # For each class get the subclass
            for cl in current_classes:
                new_classes += [super_cl for super_cl in self.class2subclass[cl]]
            
            # If there still subclasses, create a new abstraction layer
            if len(new_classes)!=0:
                self.abstraction_layers[self.depth] = {new_cl:True for new_cl in new_classes}
                self.depth += 1
            current_classes = new_classes
        self.depth -= 1

        # Add 0 as padding variable to the class2subclass and class2superclass 
        self.class2subclass[0].append(0)
        self.class2superclass[0].append(0)



            
        
