#!usr/bin/python
import networkx as nx
import numpy as np

from emd import emd


def calc_cc(G_orig, G_anon):
    """
    Calculate mean difference in clustering coefficient G_orig vs G_anon
    :param G_orig: Original graph
    :param G_anon: Sanitized graph
    :return: Mean diff in cc
    """
    assert (G_orig.number_of_nodes() == G_anon.number_of_nodes())
    return np.mean(np.abs(np.array(nx.clustering(G_orig).values()) - np.array(nx.clustering(G_anon).values())))


def calc_emd(G_orig, G_anon):
    """
    Calc Earth Mover distance of degree distribution between original graph and sanitized
     PyEMD package required! https://github.com/garydoranjr/pyemd
    :param G_orig: 
    :param G_anon: 
    :return: 
    """
    assert (G_orig.number_of_nodes() == G_anon.number_of_nodes())
    return emd(G_orig.degree().items(), G_anon.degree().items())


def number_of_fake_edges(G_orig, G_san):
    """
    Get the percent of fake edges added  
    :param G_orig: 
    :param G_san: 
    :return: 
    """
    E1 = set(G_orig.edges())
    E2 = set(G_san.edges())
    return 1. - float(len(E1 & E2)) / len(E2)


def calc_dist(G_orig, G_san):
    """
    Calculate distortion value - number of noise added
    :param G_orig: 
    :param G_san: 
    :return: 
    """
    E1 = set(G_orig.edges())
    E2 = set(G_san.edges())
    return float(len(E1 | E2 - E1 & E2)) / len(E1)

def read_mtx_graph(filename):
    """
    Read from file graph in mtx format
    Return networkx graph
    """
    fh=open(filename, 'rb')
    G = nx.Graph()
    first_line = True
    for line in fh:
        if "%" in line:
            continue
        if first_line:
            first_line = False
            continue
        u, v = line.split()
        G.add_edge(int(u), int(v))
    fh.close()
    return G
