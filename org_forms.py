# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:11:53 2020

@authors: Andrea Allen & John Meluso
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools

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

    T = nx.generators.balanced_tree(children, height)
    T_clique = nx.Graph()
    print(list(nx.edges(T)))
    T_clique.add_edges_from(list(nx.edges(T)))
    for node_ in T.nodes():
        node_edges = np.array(list(T.adj[node_].keys()))
        kids = node_edges[node_edges > node_]
        T_clique.add_node(node_, children=kids)
        T_clique.add_edges_from(list(itertools.combinations(kids,2)))
    nx.draw_spring(T_clique, with_labels=True)
    plt.show()
    return T_clique

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    clique_tree(2, 5)