import numpy as np
import random
import networkx as nx
from time import time
import pickle

'''
Generate testset with two graphs G_aux and G_san 
'''
def gen_testset(G_aux, G_san, save_path):
    test_set = []
    test_labels = []
    s = time()
    print("Dataset is generating ... ")
    fill_set(G_aux, G_san, test_set, test_labels, n_max=None)
    e = time()
    print("Test set generated in  " + str(e-s) + " sec")
    pickle.dump( (test_set, test_labels), open( save_path, "wb" ) )
    return [test_set, test_labels]
'''
Generate trainset 
Two graphs with alpha_v rate of iterseaction 
Save trainset to save_path
'''
def gen_trainset(G_aux, G_san, alpha_v, save_path):
    G_a1, G_a2 = edge_split(G_aux, alpha_v) # aux 
    G_s1, G_s2 = edge_split(G_san, alpha_v) # sanitized
    
    # Generating train dataset: 
    train_set = []
    train_labels = []
    s = time()
    print("Dataset is generating ... ")
    fill_set(G_a1, G_a2, train_set, train_labels, n_max=None)
    e = time()
    print("Aux finished " + str(e-s) + " sec")
    s = time()
    fill_set(G_s1, G_s2, train_set, train_labels, n_max=None)
    e = time()
    print("San finished " + str(e-s) + " sec")
    pickle.dump( (train_set, train_labels), open( save_path, "wb" ) )
    return [train_set, train_labels]



'''
Split graph G into two subgraphs with intersection rate alpha_v (from 0. to 1)
'''
def edge_split(G, alpha_v, save_as=""):
    V = np.array(G.edges())
    random.shuffle(V)
    a = int((1-alpha_v)/2 * len(V))
    b = int(alpha_v * len(V))
    Va = V[:a]
    Vb =  V[a:a+b]
    Vc =  V[a+b:2*a+b]
    V1 = np.concatenate((Va, Vb))
    V2 = np.concatenate((Vb, Vc))
    G1 = nx.Graph()
    G1.add_edges_from(V1)
    G2 = nx.Graph()
    G2.add_edges_from(V2)
    if save_as != '': 
        nx.write_edgelist(G1, 'data/'+str(alpha_v)+save_as+'_aux.edgelist')
        nx.write_edgelist(G2, 'data/'+str(alpha_v)+save_as+'_san.edgelist')
    return G1, G2

'''
 Get the histogram of degree distribution for an node 'n' in a graph 'G' 
     bins - the number of bins in the deg. distribution 
     size - size of the historgram 
'''
def deg_dist(G, n, bins, size):
    feature_set = [0 for i in range(2*bins)]
    _1hop = G[n]
    _1hop = G.degree(_1hop).values()
    for h in _1hop:
        if h < bins*size:
            feature_set[int(h/size)] += 1
    _prev = set(nx.single_source_shortest_path_length(G, n, cutoff=1).keys())
    _2hop = set(nx.single_source_shortest_path_length(G, n, cutoff=2).keys())
    _2hop = _2hop - _prev
    _2hop = G.degree(_2hop).values()
    for h in _2hop:
        if h < bins*size:
            feature_set[bins+int(h/size)] += 1
    return feature_set


''' Get the feature vector for egde e1 in graph G1
     and e2 in graph G2.
     1hop, 2hop deg distributions
     + 4 Silhoutte coefficients
'''    
def e_feature(G1, e1, G2, e2,  bins = 21, size = 50):
    feature_set = [0 for i in range(8*bins)]
    for j in [0, 1]:
        f_set = deg_dist(G1, e1[j], bins, size)
        for i in range(len(f_set)):
            feature_set[2*bins*j + i] += f_set[i]
    for j in [0, 1]:
        f_set = deg_dist(G2, e2[j], bins, size)
        for i in range(len(f_set)):
            feature_set[4*bins+2*bins*j + i] += f_set[i]
    
    # 4 Silhouette Coefficients 
    e1_deg0 = G1.degree(e1[0])
    e1_deg1 = G1.degree(e1[1])
    
    e2_deg0 = G2.degree(e2[0])
    e2_deg1 = G2.degree(e2[1])
    
    feature_set.append(abs(e1_deg0-e2_deg0)/max(e1_deg0, e2_deg0, 1))
    feature_set.append(abs(e1_deg1-e2_deg1)/max(e1_deg1, e2_deg1, 1))
    feature_set.append(abs(e1_deg0-e2_deg1)/max(e1_deg0, e2_deg1, 1))
    feature_set.append(abs(e1_deg1-e2_deg0)/max(e1_deg1, e2_deg0, 1))
    
    return feature_set

'''
    Transform graph into feature vector dataset
    G1, G2 - graphs for comparison
'''
def fill_set(G1, G2, data, labels, n_max=None):
    indx = 0
    total = len(G1.nodes())
    # report every 10 % 
    part = total//10
    # Add all intersections first: 
    print('Start adding positive examples')
    common = set(G1.edges()) & set(G2.edges())
    if n_max is None: 
        n_max = len(common)
    common = random.sample(common, n_max)
    indx = 0
    for i in common: 
        data.append(e_feature(G1,i, G2, i))
        labels.append(1)
        indx += 1
        if indx % int(len(common)/10) ==0:
            print("Finished : " + str(indx/len(common)))
        
    g1 = set(G1.edges()) - set(common)
    g2 = set(G2.edges()) - set(common)
    
    # How many false results to add ? - 
    # The same size as common
    # TODO: 
    print('Start adding false examples')
    if len(g1) < n_max:
        n_max = min(len(g1), len(g2))
    g1 = random.sample(g1, n_max)
    g2 = random.sample(g2, n_max)
    for i in range(len(g1)):
        data.append(e_feature(G1, g1[i], G2, g2[i]))
        labels.append(0)
        if i % int(len(g1)/10) ==0:
            print("Finished : " + str(i/len(g1)))
