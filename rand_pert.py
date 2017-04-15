#!/usr/bin/python
import networkx as nx
from random import choice
from random import random
import community
import itertools
from lru import LRU


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


def rand_pertub(G, M, t, p_val):
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

                    if random() <= (p_val * deg_u - 1) / (deg_u - 1):
                        _G.add_edge(u, z)
            count += 1
    return _G


def diff(f1, f2, w=None):
    if w is None:
        w = [1.0 for k in range(len(f1))]
    res = 0.0
    for k in range(len(f1)):
        assert (1.0 >= w[k] >= 0.0)
        res += w[k] * abs(f1[k] - f2[k])
    return res


def calc_features(G):
    # Average Neighbours degree
    av_deg = (k for k in nx.average_neighbor_degree(G).values())
    # Average degree centrality
    degs = (k for k in nx.degree_centrality(G).values())
    # Clustering coef
    centrs = (k for k in nx.clustering(G).values())

    return [(k, v, z) for k, v, z in zip(degs, centrs, av_deg)]


def smart_pertub(G, M, t, p_val, f, cache, w=None):
    """
    Random perturbation anonymization of a graph.
    Each edge is replaced with some similar edge

    :param p_val: Propability to add an edge
    :param w: Weights of the features
    :param cache:  LRU dict for shortest paths
    :param f: feature vector for each vertex
    :param G:
    :param M:
    :param t:
    :return:
    """
    _G = nx.Graph()
    for u in G.nodes():
        count = 1
        for v in G[u]:
            loop = 1
            z = u
            if v in cache.keys():
                sh_paths = cache[v]
            else:
                sh_paths = nx.single_source_shortest_path_length(G, source=v, cutoff=t - 1)
                cache[v] = sh_paths

            while (z == u or _G.has_edge(u, z)) and loop <= M:
                N = M
                z = choice(sorted(
                    ((k, diff(f[v], f[k], w))
                     for k in sh_paths.keys()),
                    reverse=False, key=lambda x: x[1]
                )[:N])
                loop += 1
                N += M
            if loop <= M:
                if count == 1:
                    _G.add_edge(u, z)
                else:
                    # Add with some probability ?
                    deg_u = G.degree(u)

                    if random() <= (p_val * deg_u - 1) / (deg_u - 1):
                        _G.add_edge(u, z)
            count += 1
    return _G


def link_mirage(G, M, t, p_val):
    """
    Link Mirage original algorithm
    :param p_val: Probability to add an edge
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
        G_anon_part = rand_pertub(G_, M, t, p_val)
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

                    prob = float(deg_a * deg_b * len(V_a_marg)) / ((len(V_a_marg) + len(V_b_marg)) * len(edges_ab))

                    if random() <= prob:
                        G_san.add_edge(a, b)

    return G_san


def smart_link_anon(G, M, t, p_val, f, w=None):
    """
      Randomized Link optimization with features and weights
      :param f: Feature vector for each vertex
      :param w: Weights for the feature vector
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
        cache = LRU(list_nodes)
        G_ = nx.subgraph(G, list_nodes)
        term_edges = term_edges - set(G_.edges())
        G_anon_part = smart_pertub(G_, M, t, p_val, f, cache, w)
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

                    prob = float(deg_a * deg_b * len(V_a_marg)) / ((len(V_a_marg) + len(V_b_marg)) * len(edges_ab))

                    if random() <= prob:
                        G_san.add_edge(a, b)

    return G_san
