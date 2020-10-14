# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:11:53 2020

@authors: Andrea Allen & John Meluso
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import pickle

def clique_tree(height, children):
    '''
    A clique tree graph wherein every node (but the bottom nodes) have the same
    number of interconnected children.

    Parameters
    ----------
    height : INT
        The number of generations of children down from the top node.
    children : TYPE
        The numbef or children of each node.

    Returns
    -------
    T_clique : NetworkX Graph
        A clique tree graph.

    '''

    # Build balanced tree
    T = nx.generators.balanced_tree(children, height)
    T_clique = nx.Graph()
    T_sibs = nx.Graph()
    #print(list(nx.edges(T)))
    T_clique.add_edges_from(list(nx.edges(T)))
    T_sibs.add_nodes_from(list(nx.nodes(T)))


    # Add sibling edges
    for node_ in T.nodes():
        node_edges = np.array(list(T.adj[node_].keys()))
        kids = node_edges[node_edges > node_]
        pars = node_edges[node_edges < node_]
        new_edges = list(itertools.combinations(kids,2))
        T_clique.add_edges_from(new_edges)
        T_sibs.add_edges_from(new_edges)
        T_clique.add_node(node_, children=kids,parents=pars)

    for node_ in T.nodes():

        # Get sibling edges
        non_sib_edges = np.array(list(T.adj[node_].keys()))
        all_edges = np.array(list(T_clique.adj[node_].keys()))
        sibs = np.sort(list(list(set(non_sib_edges)-set(all_edges)) +
                     list(set(all_edges)-set(non_sib_edges))))
        T_clique.add_node(node_, siblings=sibs)

        # Get grandparent edges
        grandpars = np.unique([T_clique.nodes[pars_]['parents'] for pars_
                     in T_clique.nodes[node_]['parents']])
        T_clique.add_node(node_, grandparents=grandpars)

        # Get grandchildren edges
        grandkids = np.unique([T_clique.nodes[kids_]['children'] for kids_
                     in T_clique.nodes[node_]['children']])
        T_clique.add_node(node_, grandchildren=grandkids)

    # Get directed graph adjacency matrix of parentage & siblings
    parents = nx.to_numpy_array(T)
    for ii in T.nodes():
        for jj in T.nodes():
            if ii < jj:
                parents[ii,jj] = 0
    siblings = nx.to_numpy_array(T_sibs)

    # Plot the graph
    print(list(nx.edges(T_clique)))
    nx.draw_spring(T_clique, with_labels=True)
    plt.show()
    pickle.dump(T_clique, open("cliquetree_org.pickle","wb"))
    pickle.dump(parents, open("cliquetree_parents.pickle","wb"))
    pickle.dump(siblings, open("cliquetree_siblings.pickle","wb"))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    clique_tree(4, 5)





