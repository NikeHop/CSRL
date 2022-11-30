from collections import defaultdict
import os
from typing import List, Dict, Tuple

from graphviz import Graph
import matplotlib.pyplot as plt 
import pickle 

import gym 
from SPARQLWrapper import SPARQLWrapper, JSON
import tqdm 


def print_class_tree(class_tree:dict,directory:str)->None:
    """
    Saves a visualization of the class tree in the given directory
    """
    g = Graph('wordnet_class_tree')
    nodes = {}
    for node, children in class_tree.items():
        if node not in nodes:
            g.node(node,label=node)
        for child in children:
            if child not in nodes:
                g.node(child,label=child)
            g.edge(node,child)
    
    g.render(directory=directory)
    
def invert_tree(tree):
    inverted_tree = defaultdict(list)
    for node, child in tree.items():
        inverted_tree[child].append(node)

    return inverted_tree

def get_instance_ids(instances:List[str],load:bool,sparql_wrapper:SPARQLWrapper,kg_directory:str)->Tuple[List,Dict]:
    """
    Given a list of entities tries to find the corresponding label in DBpedia.
    """

    if not load:
        # Generate all possible queries: "Word", "Word"@en
        entities_with_response = {}
        entities_without_response = []
        
        for ent in tqdm.tqdm(instances):
            # Create queries 
            no_response = True
            ent = ent.lower()
            Ent = ent[0].upper() + ent[1:]
            ent_literals = ['"' + Ent + '"@en','"' + Ent + '"']
            
            for literal in ent_literals:
                query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?entity
                WHERE {{ ?entity rdfs:label  {literal} }}
                """

                # Try query 
                sparql_wrapper.setQuery(query)
                sparql_wrapper.setReturnFormat(JSON)
                results = sparql_wrapper.query().convert()

                if len(results['results']['bindings'])>0:
                    for elem in results['results']['bindings']:
                        splitted_ent_id = elem['entity']['value'].split('/')
                        # Things to check:

                        # Is there a category in the name 
                        if 'Category' in splitted_ent_id[-1]:
                            continue

                        # Is it either part of the ontology or resource
                        if splitted_ent_id[-2] not in ['ontology','resource']:
                            continue 
                        
                        ent_id = elem['entity']['value']
                        entities_with_response[ent] = (literal,ent_id)
                        no_response = False
                        break
                        
            if no_response:
                entities_without_response.append(ent)
        
        with open(os.path.join(kg_directory,'./dbpedia_ids.pkl'),'wb+') as f:
            pickle.dump({'entities_without_response':entities_without_response, 'entities_with_response':entities_with_response},f)

    else:
        with open(os.path.join(kg_directory,'./dbpedia_ids.pkl'),'rb+') as f :
            entities = pickle.load(f)
            entities_with_response = entities['entities_with_response']
            entities_without_response = entities['entities_without_response']

    return entities_with_response, entities_without_response

def get_subclass_structure(entities_with_response:dict,
                           entities_without_response:list,
                           load:bool,
                           sparql_wrapper:SPARQLWrapper,
                           kg_directory:str)->dict:
    """
    Query DBpedia for superclasses.
    """
    print(f'Number of entities without response {len(entities_without_response)}')
    if not load:
        class2superclass = {}
        # Map all entities that could not be resolved to the root node
        class2superclass.update({elem:'Thing' for elem in entities_without_response})

        object_queue = [value[1] for value in entities_with_response.values()]
        dbpedia_name2instance = {value[1].split('/')[-1]:key for key,value in entities_with_response.items()}
        idd_type2relation = {'ontology':'rdfs:subClassOf', 'resource':'rdf:type'}
        
        while len(object_queue)>0:
            
            idd = object_queue.pop(0)
            current_object_name = idd.split('/')[-1]

            # Check whether it is a resource or ontology 
            idd_type = idd.split('/')[-2]
            
            query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
                SELECT ?type
                WHERE {{ <{idd}> {idd_type2relation[idd_type]} ?type  }}
                """
        
            sparql_wrapper.setQuery(query)
            sparql_wrapper.setReturnFormat(JSON)
            results = sparql_wrapper.query().convert()

            if len(results['results']['bindings'])==0:
                # Transform idd into name
                name = idd.split('/')[-1]
                class2superclass[name]="Thing"

            else:
                object_ids = []
                object_names = []
                only_thing = True

                for elem in results['results']['bindings']:
                    object_idd = elem['type']['value']
                    object_name = object_idd.split('/')[-1]
                    relation_type = object_idd.split('/')[-2]

                    if object_name!='owl#Thing' and relation_type not in ['ontology', 'resource']:
                        continue

                    object_names.append(object_name)
                    object_ids.append(object_idd)

                    if object_name!='owl#Thing':
                        only_thing = False
                
                if only_thing:
                    current_object_name = idd.split('/')[-1]
                    class2superclass[current_object_name] = 'Thing'
                else:
                    for super_id, super_name in zip(object_ids,object_names):
                        if super_name!='owl#Thing':
                            class2superclass[current_object_name]=super_name
                            object_queue.append(super_id)
                            break

        # Lowercase everything 
        new_class2superclass = {}
        for elem, value in class2superclass.items():
            if elem in dbpedia_name2instance:
                elem = dbpedia_name2instance[elem]
            new_class2superclass[elem.lower()]=[value.lower()]
        class2superclass = new_class2superclass

        # We need to pickle
        with open(os.path.join(kg_directory,'./dbpedia_class2subclass.pkl'),'wb+') as f:
                pickle.dump(class2superclass,f)
    else:
         with open(os.path.join(kg_directory,'./dbpedia_class2subclass.pkl',),'rb+') as f:
            class2superclass = pickle.load(f)
    
    return class2superclass


def get_class_tree_dbpedia(instances:List[str],load:bool,kg_directory:str,visualize=False)->dict:
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    # Obtain the ID's of the instances in the DBPedia-KG
    entities_with_response, entities_without_response = get_instance_ids(instances,load,sparql,kg_directory)
    class2superclass = get_subclass_structure(entities_with_response, entities_without_response, load, sparql, kg_directory)
    class_tree = class2superclass
    
    if visualize:
        print_class_tree(class_tree,'./')

    return class_tree