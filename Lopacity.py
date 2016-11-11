from collections import Counter
import pandas as pd
import networkx as nx
import random


def init(g, degs, verbose=False):
    """
    :bool verbose: Print graph stats
    :array degs: Degree distibution of a graph
    :Graph g: networkx graph
    """
    degree_count = Counter(degs.values())
    if verbose:
        print("Totally " + str(len(degree_count)) + " distinct degrees")
        print("Min degree " + str(min(degree_count)))
        print("Max degree " + str(max(degree_count)))
        print("Mode degree " + str(degree_count.most_common(1)[0][0]))
        print("Avg degree " + str(np.mean(list(degs.values()))))

    opacity = {}
    for k in degree_count.keys():
        opacity[k] = {}
        for i in degree_count.keys():
            opacity[k][i] = 0.0
    opacity = pd.DataFrame(opacity)

    return [degree_count, opacity]


def __opacity_calc(a1, a2, degs, deg_count, opacity, inv_opacity):
    """
    Add to opacity matrix edge a1, a2
    :param a1:
    :param a2:
    :param degs:
    :param deg_count:
    :param opacity:
    :param inv_opacity:
    """
    d1 = degs[a1]
    d2 = degs[a2]
    if d1 == d2:
        added_value = deg_count[d1] * (deg_count[d1] - 1) / 2
    else:
        added_value = deg_count[d1] * deg_count[d2]
    opacity[max(d1, d2)][min(d1, d2)] += 1.0 / added_value

    if opacity[max(d1, d2)][min(d1, d2)] > 1.0:
        opacity[max(d1, d2)][min(d1, d2)] = 1.0
    if (max(d1, d2), min(d1, d2)) not in inv_opacity.keys():
        inv_opacity[(max(d1, d2), min(d1, d2))] = [(a1, a2)]
    else:
        inv_opacity[(max(d1, d2), min(d1, d2))].append((a1, a2))


def calc_lopacity_matrix(g, L, degs, deg_count, opacity, inv_opacity):
    """
    Calucalte opacity matrix with shortest path with level L
    :param g: Graph
    :param L:
    :param degs:
    :param deg_count:
    :param opacity:
    :param inv_opacity:
    """
    opacity[opacity > 0.0] = 0.0
    lapsp = nx.all_pairs_shortest_path_length(g, cutoff=L)
    for i in lapsp:
        for z in lapsp[i].keys():
            if i < z:
                __opacity_calc(i, z, degs, deg_count, opacity, inv_opacity)


def get_max_violated(g, theta, opacity, inv_opacity):
    """
    Get the edes with maximum opacity value,  exceeding $theta$
    :param g:
    :param theta:
    :param opacity:
    :param inv_opacity:
    :return: Edges with maximum opacity
    """
    kf = opacity[(opacity > theta) & (opacity == opacity.max().max())]
    df_notnull = kf.notnull().unstack()
    deg_vals = df_notnull[df_notnull].keys()
    edges_viol = []
    for (d1, d2) in deg_vals:
        edges_viol.append(inv_opacity[(d1, d2)])
    return edges_viol


def get_all_violated(g, theta, opacity, inv_opacity):
    """
    Get all edges with with opacity value exceeding $theta$
    :param g:
    :param theta:
    :param opacity:
    :param inv_opacity:
    :return: Edges with maximum opacity
    """
    kf = opacity[opacity > theta]
    df_notnull = kf.notnull().unstack()
    deg_vals = df_notnull[df_notnull].keys()
    edges_viol = []
    for (d1, d2) in deg_vals:
        edges_viol.append(inv_opacity[(d1, d2)])
    return edges_viol


def edge_to_path(g, buckets):
    """
    Transform path to edges, calcaulte inverted index
    :param g:
    :param buckets:
    :return:
    """
    cand = {}
    inv_cand = {}
    bucket_id = 0
    for bucket in buckets:
        cand_id = 0
        cand[bucket_id] = {}
        for edge in bucket:
            # Get all shortest paths from A to B
            edge_paths = [p for p in nx.all_shortest_paths(g, source=edge[0], target=edge[1])]
            full_paths = []
            # Transfrom shortest path into edge list:
            # To remove the connection we need to remove everything
            # withing one vertex pair of one cell in opacity matrix
            for path in edge_paths:
                full_path = []
                for i in range(len(path) - 1):
                    full_path.append((path[i], path[i + 1]))
                full_paths.append(full_path)

            # Add the number of shortest path
            cand[bucket_id][cand_id] = len(full_paths)
            # Inverted candidates
            f_id = 0
            for path in full_paths:
                for e in path:
                    if e not in inv_cand.keys():
                        inv_cand[e] = {}
                    if bucket_id not in inv_cand[e].keys():
                        inv_cand[e][bucket_id] = {}
                    if cand_id not in inv_cand[e][bucket_id].keys():
                        inv_cand[e][bucket_id][cand_id] = []

                    inv_cand[e][bucket_id][cand_id].append(f_id)
                # Add some statistics maybe ?
                f_id += 1
            cand_id += 1
        bucket_id += 1

    return [cand, inv_cand]


def set_score_func(inv_cand, exceptSet=set([])):
    """
    Apply score function to inverted indexes with edges
    :param inv_cand:
    """
    for e in inv_cand:
        score_func = 1.0
        for b_id in inv_cand[e]:
            if b_id == 'score':
                continue
            for c_id in inv_cand[e][b_id]:
                if b_id in exceptSet:
                    score_func /= (len(inv_cand[e][b_id][c_id]) + 1)
                else:
                    score_func *= (len(inv_cand[e][b_id][c_id]) + 1)
            if b_id in exceptSet:
                score_func /= (len(inv_cand[e][b_id]) + 1) * (len(inv_cand[e][b_id]) + 1)
            else:
                score_func *= (len(inv_cand[e][b_id]) + 1) * (len(inv_cand[e][b_id]) + 1)
        if b_id in exceptSet:
            score_func /= (len(inv_cand[e]) + 1) * (len(inv_cand[e]) + 1) * (len(inv_cand[e]) + 1)
        else:
            score_func *= (len(inv_cand[e]) + 1) * (len(inv_cand[e]) + 1) * (len(inv_cand[e]) + 1)
        inv_cand[e]['score'] = score_func


def opacity_diff(op, new_op):
    pass


def remove_edge_lopacity_orig(g, L, degs, deg_count, opacity, edge_pairs_buckets, consider_all=False):
    """
    Original lopacity model
    :param consider_all:
    :param g:
    :param L:
    :param degs:
    :param deg_count:
    :param opacity:
    :param edge_pairs_buckets:
    :return:
    """
    cand, inv_cand = edge_to_path(g, edge_pairs_buckets)
    edges = list(inv_cand.keys())
    best_edge = edges[0]
    best_lo = 1.1
    best_pop = g.number_of_edges()
    t = 0
    rand_dec = False
    if consider_all:
        edges = g.edges()

    for edge in edges:
        g.remove_edge(edge[0], edge[1])
        new_opacity = opacity.copy()
        inv_op = {}
        calc_lopacity_matrix(g, L, degs, deg_count, new_opacity, inv_op)
        kf = new_opacity[(new_opacity == new_opacity.max().max())]
        df_notnull = kf.notnull().unstack()
        n_lo = len(df_notnull[df_notnull].keys())
        max_lo = new_opacity.max().max()
        if max_lo < best_lo:
            best_lo = max_lo
            t = 1
            best_pop = n_lo
            best_edge = edge
            rand_dec = False
        elif (max_lo == best_lo) and (n_lo < best_pop):
            best_pop = n_lo
            best_edge = edge
            rand_dec = False
            t = 1
        elif (max_lo == best_lo) and (n_lo == best_pop):
            rand = random.random()
            t += 1
            if rand < 1 / t:
                best_edge = edge
                rand_dec = True
        g.add_edge(edge[0], edge[1])
    # g.remove_edge(best_edge[0], best_edge[1])
    print("Best edge to remove: ", best_edge)
    print("Is the decision is random ? " + str(rand_dec))
    return best_edge


def get_edges_intersect(g, edge_pairs_buckets, verbose=False):
    """
    Inverted index model
    :param g:
    :param edge_pairs_buckets:
    :return:
    """
    cand, inv_cand = edge_to_path(g, edge_pairs_buckets)
    set_score_func(inv_cand)
    # get top candidates
    top_cand = sorted(inv_cand.items(), key=lambda x: (x[1]['score'], random.random()), reverse=True)
    if verbose:
        print("Top  candidates for removal: ")
        print(top_cand)
        print(" ")
        print(" In each cell how many candidates and how many shortest paths there are")
        print(cand)
    for k in cand:
        for k1 in cand[k]:
            cand[k][k1] = set(range(cand[k][k1]))

    rem_edges = []
    finished = False
    s_id = 0
    while not finished:
        # Take the first top candidate
        s = top_cand[s_id]
        # Get the edge id
        edge = s[0]
        # Get cell
        for b_id in s[1].keys():
            # Get candidate
            if b_id == 'score':
                continue
            for c_id in s[1][b_id].keys():
                # Remove from all that it has effect
                cand[b_id][c_id] = cand[b_id][c_id] - set(s[1][b_id][c_id])

        finished = True
        for b_id in cand.keys():
            completed = False
            for c_id in cand[b_id].keys():
                if len(cand[b_id][c_id]) == 0:
                    completed = True
            if not completed:
                finished = False
        rem_edges.append(edge)
        s_id += 1
    return rem_edges


def get_edges_intersect_dynamic(g, edge_pairs_buckets):
    """
    Inverted index model
    :param g:
    :param edge_pairs_buckets:
    :return:
    """
    exceptSet = set([])
    cand, inv_cand = edge_to_path(g, edge_pairs_buckets)
    set_score_func(inv_cand)
    # get top candidates
    top_cand = sorted(inv_cand.items(), key=lambda x: (x[1]['score'], random.random()), reverse=True)
    # print("Top 10 candidates for removal: ")
    # print(top_cand[:10])
    # print(" ")
    # print(" In each cell how many candidates and how many shortest paths there are")
    # print(cand)
    for k in cand:
        for k1 in cand[k]:
            cand[k][k1] = set(range(cand[k][k1]))

    rem_edges = set([])
    finished = False
    s_id = 0
    while not finished:
        # Take the first top candidate
        s = top_cand[s_id]
        # Get the edge id
        edge = s[0]
        # Get cell
        if edge in rem_edges:
            s_id += 1
            continue

        for b_id in s[1].keys():
            # Get candidate
            if b_id == 'score':
                continue
            for c_id in s[1][b_id].keys():
                # Remove from all that it has effect
                cand[b_id][c_id] = cand[b_id][c_id] - set(s[1][b_id][c_id])

        finished = True
        for b_id in cand.keys():
            completed = False
            for c_id in cand[b_id].keys():
                if len(cand[b_id][c_id]) == 0:
                    completed = True
            if not completed:
                finished = False
            else:
                exceptSet.add(b_id)
                set_score_func(inv_cand, exceptSet)
                top_cand = sorted(inv_cand.items(), key=lambda x: (x[1]['score'], random.random()), reverse=True)
                # print("Scores are updated: ")
                # print(top_cand)
                # print(" --------- ")
                s_id = -1

        rem_edges.add(edge)
        s_id += 1
    return rem_edges


def set_score_func_opt(cand, inv_cand):
    """
    Apply score function to inverted indexes with edges
    :param inv_cand:
    """
    for e in inv_cand:
        score_func = [0, 0]  # cells removed, cand removed, closer to remove
        for b_id in inv_cand[e]:
            if b_id == 'score':
                continue
            has_effect = False
            for c_id in inv_cand[e][b_id]:
                temp_set = cand[b_id][c_id] - set(inv_cand[e][b_id][c_id])
                # The removal of this edge leads to removal of all shortest path => decrease of cell b_id
                score_func[1] += -1
                if len(temp_set) == 0:
                    has_effect = True
            if has_effect:
                score_func[0] += 1  # number of cell it affects
        inv_cand[e]['score'] = score_func


def set_score_func_opt_2(cand, inv_cand):
    """
    Apply score function to inverted indexes with edges
    :param inv_cand:
    """
    for e in inv_cand:
        score_func = [0, 0, 0]  # cells removed, cand removed, closer to remove
        for b_id in inv_cand[e]:
            if b_id == 'score':
                continue
            has_effect = False
            for c_id in inv_cand[e][b_id]:
                temp_set = cand[b_id][c_id] - set(inv_cand[e][b_id][c_id])
                # The removal of this edge leads to removal of all shortest path => decrease of cell b_id
                score_func[1] += -1
                if len(temp_set) == 0:
                    has_effect = True
                else:
                    score_func[2] += -len(temp_set)
            if has_effect:
                score_func[0] += 1  # number of cell it affects
        inv_cand[e]['score'] = score_func


def get_edges_intersect_opt(g, edge_pairs_buckets, mode ='1', verbose=False):
    """
    Inverted index model
    :param verbose:
    :param g:
    :param edge_pairs_buckets:
    :return:
    """
    cand, inv_cand = edge_to_path(g, edge_pairs_buckets)

    for k in cand:
        for k1 in cand[k]:
            cand[k][k1] = set(range(cand[k][k1]))
    if mode == '1':
        set_score_func_opt(cand, inv_cand)
    else:
        set_score_func_opt_2(cand, inv_cand)

    top_cand = sorted(inv_cand.items(), key=lambda x: (x[1]['score'], random.random()), reverse=True)
    if verbose:
        print("Top candidates for removal: ")
        print(top_cand)
        print(" ")
        print(" In each cell how many candidates and how many shortest paths there are")
        print(cand)

    s = top_cand[0]  # get first top candidate
    return [s[0]]


def intersect_anonimize(g, L, theta, score='stat', mode='val'):
    """
     Anonimize graph g with intersect method and inverted index
     :param mode:  Mode to calculate :
                                     'max' - edges that contribute to maximum of the opacity matrix
                                     'val' - all violated edges
     :param score:
                  'stat' - simple static score function
                  'dyn' - dynamically changing score
                  'opt' - local optimally
     :param g: Graph for anonymization
     :param L: Maximum significant distance
     :param theta: The maximum allowed opacity value
     """
    degs = g.degree(g)

    deg_count, opacity = init(g, degs)
    finished = False
    g_hat = g.copy()

    total_removed = 0
    results = {}
    while not finished:
        # Main steps:
        # 1. Calculate L-APSP:
        inv_opacity = {}
        calc_lopacity_matrix(g_hat, L, degs, deg_count, opacity, inv_opacity)
        if total_removed == 0:
            results['old_opacity'] = opacity.copy()
        # 5. If the max is lower than theta - exit
        if opacity.max().max() <= theta:
            break

        # 2. Get all violated vertex-pairs that have the maximum opacity value
        if mode == 'val':
            edge_pairs_buckets = get_all_violated(g_hat, theta, opacity, inv_opacity)
        elif mode == 'max':
            edge_pairs_buckets = get_max_violated(g_hat, theta, opacity, inv_opacity)
        else:
            raise NameError('Not valid parameter mode')

        # 3. Get best edges for removal
        if score == 'dyn':
            rem_edges = get_edges_intersect_dynamic(g_hat, edge_pairs_buckets)
        elif score == 'stat':
            rem_edges = get_edges_intersect(g_hat, edge_pairs_buckets)
        elif score == 'opt':
            rem_edges = get_edges_intersect_opt(g_hat, edge_pairs_buckets, verbose=False)
        elif score == 'opt2':
            rem_edges = get_edges_intersect_opt(g_hat, edge_pairs_buckets, verbose=False, mode='2')
        else:
            raise NameError('Not valid parameter score')
        # 4. Remove edges
        g_hat.remove_edges_from(rem_edges)
        print("Edges removed: ")
        print(rem_edges)
        total_removed += len(rem_edges)

    #print("Total Removed edges : ")
    #print(total_removed)
    print('Finished')
    results['new_opacity'] = opacity
    results['new_graph'] = g_hat
    return results


def anonimize_lopacity(g, L, theta, mode='val'):
    """

    :param g: Graph to anonymize
    :param L: Maximum length
    :param theta:  maximum opacity value
    :param mode: Mode to calculate : 'all' - consider all possible edges,
                                     'max' - edges that contribute to maximum of the opacity matrix
                                     'val' - all violated edges
    :return:
    """
    degs = g.degree(g)

    inv_opacity = {}
    deg_count, opacity = init(g, degs)

    finished = False
    g_hat = g.copy()
    total_rem = 0
    results = {}
    while not finished:
        # 1. Calculate L-APSP:
        calc_lopacity_matrix(g_hat, L, degs, deg_count, opacity, inv_opacity)
        if total_rem == 0:
            results['old_opacity'] = opacity.copy()

        if opacity.max().max() <= theta:
            break

        consider_all = False
        if mode == 'val':
            edge_pairs_buckets = get_all_violated(g_hat, theta, opacity, inv_opacity)
        elif mode == 'max':
            edge_pairs_buckets = get_max_violated(g_hat, theta, opacity, inv_opacity)
        elif mode == 'all':
            edge_pairs_buckets = get_all_violated(g_hat, theta, opacity, inv_opacity)
            consider_all = True
        else:
            raise NameError('Not valid parameter mode')

        rem_edge = remove_edge_lopacity_orig(g_hat, L, degs, deg_count, opacity, edge_pairs_buckets,
                                             consider_all=consider_all)
        g_hat.remove_edges_from([rem_edge])
        total_rem += 1

    results['new_opacity'] = opacity
    results['new_graph'] = g_hat
    print("Total Removed edges : ")
    print(total_rem)
    return results


def lopacity_step(g, degs, L, theta, init=True, verbose=False):
    """

    :param g:
    :param degs:
    :param L:
    :param theta:
    :param init:
    :param verbose:
    :return:
    """
    deg_count, opacity = init(g, degs, verbose)
    inv_opacity = {}
    # 1. Calcaulte L-APSP:
    calc_lopacity_matrix(g, L, degs, deg_count, opacity, inv_opacity)
    if verbose:
        print(opacity)
    # 2. Get all violated vertex-pairs by the maximum
    edge_pairs_buckets = get_max_violated(g, theta, opacity, inv_opacity)
    return remove_edge_lopacity_orig(g, L, degs, deg_count, opacity, edge_pairs_buckets)
