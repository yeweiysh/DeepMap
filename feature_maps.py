import networkx as nx
from gensim import corpora
import gensim
import breadth_first_search as bfs
from collections import defaultdict
import numpy as np
import copy, pickle
import pynauty
from sklearn.preprocessing import normalize
from scipy.sparse import csc_matrix


def get_graphlet(window, nsize):
    """
    This function takes the upper triangle of a nxn matrix and computes its canonical map
    """
    adj_mat = {idx: [i for i in list(np.where(edge)[0]) if i!=idx] for idx, edge in enumerate(window)}

    g = pynauty.Graph(number_of_vertices=nsize, directed=False, adjacency_dict = adj_mat)
    cert = pynauty.certificate(g)
    return cert


def get_maps(n):
    # canonical_map -> {canonical string id: {"graph", "idx", "n"}}
    file_counter = open("canonical_maps/canonical_map_n%s.p"%n, "rb")
    canonical_map = pickle.load(file_counter, encoding='bytes')
    file_counter.close()
    # weight map -> {parent id: {child1: weight1, ...}}
    file_counter = open("graphlet_counter_maps/graphlet_counter_nodebased_n%s.p"%n, "rb")
    weight_map = pickle.load(file_counter, encoding='bytes')
    file_counter.close()
    weight_map = {parent: {child: weight/float(sum(children.values())) for child, weight in children.items()} for parent, children in weight_map.items()}
    child_map = {}
    for parent, children in weight_map.items():
        for k,v in children.items():
            if k not in child_map:
                child_map[k] = {}
            child_map[k][parent] = v
    weight_map = child_map
    return canonical_map, weight_map


def adj_wrapper(g):
    am_ = g["al"]
    size = max(np.shape(am_))
    am = np.zeros((size, size))
    for idx, i in enumerate(am_):
        for j in i:
            am[idx][j-1] = 1
    return am


def graphlet_feature_map(num_graphs, graphs, num_graphlets, samplesize):
    # if no graphlet is found in a graph, we will fall back to 0th graphlet of size k
    fallback_map = {1: 1, 2: 2, 3: 4, 4: 8, 5: 19, 6: 53, 7: 209, 8: 1253, 9: 13599}
    canonical_map, weight_map = get_maps(num_graphlets)
    canonical_map1, weight_map1 = get_maps(2)
    # randomly sample graphlets
    graph_map = {}
    graphlet_graph = []
    for gidx in range(num_graphs):
        #print(gidx)
        am = graphs[gidx]
        m = len(am)
        for node in range(m):
            graphlet_node = []
            for j in range(samplesize):
                rand = np.random.permutation(range(m))
                r = []
                r.append(node)
                for ele in rand:
                    if ele != node:
                        r.append(ele)

                for n in [num_graphlets]:
                #for n in range(3,6):
                    if m >= num_graphlets:
                        window = am[np.ix_(r[0:n], r[0:n])]
                        g_type = canonical_map[get_graphlet(window, n)]
                        #for key, value in g_type.items():
                        #    print(key.decode("utf-8"))
                        #    print(value)
                        graphlet_idx = str(g_type["idx".encode()])
                    else:
                        window = am[np.ix_(r[0:2], r[0:2])]
                        g_type = canonical_map1[get_graphlet(window, 2)]
                        graphlet_idx = str(g_type["idx".encode()])

                    graphlet_node.append(graphlet_idx)

            graphlet_graph.append(graphlet_node)

    dictionary = corpora.Dictionary(graphlet_graph)
    corpus = [dictionary.doc2bow(graphlet_node) for graphlet_node in graphlet_graph]
    M = gensim.matutils.corpus2csc(corpus, dtype=np.float32)
    M = normalize(M, norm='l1', axis=0)
    M = M.T

    allFeatures = {}
    index = 0
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        graphlet_feature = M[index:index + n, :]
        index += n
        allFeatures[gidx] = graphlet_feature

    return allFeatures


def wl_subtree_feature_map(num_graphs, graphs, labels, max_h):
    alllabels = {}
    label_lookup = {}
    label_counter = 0
    wl_graph_map = {it: {gidx: defaultdict(lambda: 0) for gidx in range(num_graphs)} for it in range(-1, max_h)}

    alllabels[0] = labels
    new_labels = {}
    # initial labeling
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        new_labels[gidx] = np.zeros(n, dtype=np.int32)
        label = labels[gidx]

        for node in range(n):
            la = label[node]
            if la not in label_lookup:
                label_lookup[la] = label_counter
                new_labels[gidx][node] = label_counter
                label_counter += 1
            else:
                new_labels[gidx][node] = label_lookup[la]
            wl_graph_map[-1][gidx][label_lookup[la]] = wl_graph_map[-1][gidx].get(label_lookup[la], 0) + 1
    compressed_labels = copy.deepcopy(new_labels)
    # WL iterations started
    for it in range(max_h - 1):
        label_lookup = {}
        label_counter = 0
        for gidx in range(num_graphs):
            adj = graphs[gidx]
            n = len(adj)
            nx_G = nx.from_numpy_matrix(adj)
            for node in range(n):
                node_label = tuple([new_labels[gidx][node]])
                neighbors = []
                edges = list(bfs.bfs_edges(nx_G, np.zeros(n), source=node, depth_limit=1))
                for u, v in edges:
                    neighbors.append(v)

                if len(neighbors) > 0:
                    neighbors_label = tuple([new_labels[gidx][i] for i in neighbors])
                    node_label = tuple(tuple(node_label) + tuple(sorted(neighbors_label)))
                if node_label not in label_lookup:
                    label_lookup[node_label] = str(label_counter)
                    compressed_labels[gidx][node] = str(label_counter)
                    label_counter += 1
                else:
                    compressed_labels[gidx][node] = label_lookup[node_label]
                wl_graph_map[it][gidx][label_lookup[node_label]] = wl_graph_map[it][gidx].get(label_lookup[node_label],
                                                                                              0) + 1
        # print("Number of compressed labels at iteration %s: %s"%(it, len(label_lookup)))
        new_labels = copy.deepcopy(compressed_labels)
        # print("labels")
        # print(labels)
        alllabels[it + 1] = new_labels

    subtrees_graph = []
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        for node in range(n):
            subtrees_node = []
            for it in range(max_h):
                graph_label = alllabels[it]
                label = graph_label[gidx]
                subtrees_node.append(str(label[node]))

            subtrees_graph.append(subtrees_node)

    dictionary = corpora.Dictionary(subtrees_graph)
    corpus = [dictionary.doc2bow(subtrees_node) for subtrees_node in subtrees_graph]
    M = gensim.matutils.corpus2csc(corpus, dtype=np.float32)
    M = M.T

    allFeatures = {}
    index = 0
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        subtree_feature = M[index:index + n, :]
        index += n
        allFeatures[gidx] = subtree_feature

    return allFeatures


def shortest_path_feature_map(num_graphs, graphs, labels):
    sp_graph = []
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        label = labels[gidx]
        nx_G = nx.from_numpy_matrix(adj)

        for i in range(n):
            sp_node = []
            for j in range(n):
                if i != j:
                    try:
                        path = list(nx.shortest_path(nx_G, i, j))
                    except nx.exception.NetworkXNoPath:
                        continue

                    if not path:
                        continue
                    if label[i] <=label[j]:
                        sp_label = str(int(label[i])) + ',' + str(int(label[j])) + ',' + str(len(path))
                    else:
                        sp_label = str(int(label[j])) + ',' + str(int(label[i])) + ',' + str(len(path))
                    sp_node.append(sp_label)
            sp_graph.append(sp_node)

    dictionary = corpora.Dictionary(sp_graph)
    corpus = [dictionary.doc2bow(sp_node) for sp_node in sp_graph]
    M = gensim.matutils.corpus2csc(corpus, dtype=np.float32)
    M = M.T

    allFeatures = {}
    index = 0
    for gidx in range(num_graphs):
        adj = graphs[gidx]
        n = len(adj)
        sp_feature = M[index:index + n, :]
        index += n
        allFeatures[gidx] = sp_feature

    return allFeatures
