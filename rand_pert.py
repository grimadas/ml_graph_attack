#!/usr/bin/python
import networkx as nx
from random import choice
from random import random
import community
import itertools


def random_walk(G, v, t):
    """
    Random walk in graph G 
    :param G: Networkx graph
    :param v: Starting vertex
    :param t: Number of hops
    :return: terminated vertex z
    """
    if t == 0:
        return v
    else:
        neighbours = G[v]
        z = choice(list(neighbours.keys()))
        return random_walk(G, z, t - 1)


def rand_pertub(G, M, t):
    """
    Random perturbation anonymization of graph G
    :param G: NetworkX original graph 
    :param M: Number of trials
    :param t: Number of significant hops
    :return: _G - sanitized graph
    """
    _G = nx.Graph()
    for u in G.nodes():
        count = 1
        for v in G[u]:
            loop = 1
            z = u
            while (z == u or _G.has_edge(u, z)) and loop <= M:
                # Perform t-1 random walk 
                z = random_walk(G, v, t - 1)
                loop += 1
            if loop <= M:
                if count == 1:
                    _G.add_edge(u, z)
                else:
                    deg_u = G.degree(u)

                    if random() <= (0.5 * deg_u - 1) / (deg_u - 1):
                        _G.add_edge(u, z)
            count += 1
    return _G


def link_mirage(G, M, t):
    # first compute the best partition
    """
    Link Mirage original algorithm
    :param G: graph to anonymize 
    :param M: Trials number 
    :param t: Most significant hop
    :return: Sanitized graph G' 
    """
    partition = community.best_partition(G)
    coms = set(partition.values())
    # independent randomized perturbation
    term_edges = set(G.edges())
    G_san = nx.Graph()
    for com_a in coms:
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com_a]
        G_ = nx.subgraph(G, list_nodes)
        term_edges = term_edges - set(G_.edges())
        G_anon_part = rand_pertub(G_, M, t)
        G_san = nx.compose(G_san, G_anon_part)

    # Get marginal nodes
    term_nodes = set()
    for e in term_edges:
        term_nodes.add(e[0])
        term_nodes.add(e[1])

    # Marginal nodes perturbation
    for com_a in coms:
        for com_b in coms:
            if com_a != com_b:
                V_a = term_nodes & set([nodes for nodes in partition.keys() if partition[nodes] == com_a])
                V_b = term_nodes & set([nodes for nodes in partition.keys() if partition[nodes] == com_b])

                V_a_marg = set([n for n in V_a if set(G.neighbors(n)) & V_b])
                V_b_marg = set([n for n in V_b if set(G.neighbors(n)) & V_a])

                edges_ab = set(itertools.product(V_a_marg, V_b_marg)) & term_edges

                for a, b in edges_ab:
                    deg_a = len(set(G.neighbors(a)) & V_b_marg)
                    deg_b = len(set(G.neighbors(b)) & V_a_marg)

                    prob = float(deg_a*deg_b*len(V_a_marg))/((len(V_a_marg) + len(V_b_marg))*len(edges_ab))
                    

                    if random() <= prob:
                        G_san.add_edge(a, b)

    return G_san

