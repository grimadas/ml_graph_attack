import networkx as nx
from random import choice
from random import random

def random_walk(G, v, t):

    if t == 0:
        return v
    else:
        neighbours = G[v]
        z = choice(list(neighbours.keys()))
        return random_walk(G, z, t-1)
        
        
def rand_pertub(G, M, t):
    _G  = nx.Graph()
    for u in G.nodes():
        count = 1
        for v in G[u]:
            loop = 1
            z = u
            while (z == u or _G.has_edge(u, z)) and loop <= M :
                # Perform t-1 random walk 
                z = random_walk(G, v, t-1)
                loop += 1
            if loop <= M: 
                if count == 1:
                    _G.add_edge(u, z)
                else:
                    deg_u = G.degree(u)

                    if random() <= (0.5* deg_u-1)/(deg_u - 1):
                        _G.add_edge(u, z)
            count += 1
    return _G
