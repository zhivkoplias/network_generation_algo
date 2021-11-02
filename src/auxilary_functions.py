"""

Auxiliary functions

"""

import warnings
warnings.filterwarnings("ignore")
import os
import json
import joblib
import numpy as np
import pandas as pd
from itertools import permutations, combinations, product
from numba import njit, prange
from tqdm import tqdm
import networkx as nx
import multiprocessing as mp
from math import factorial
from copy import copy
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import sparse
import resource
import csv

import networkx as nx
from networkx.algorithms.distance_measures import diameter
from networkx.algorithms.components import is_weakly_connected, is_strongly_connected, strongly_connected_components
from networkx.algorithms.centrality import degree_centrality, betweenness_centrality
from networkx.convert_matrix import to_numpy_array
from networkx.algorithms.swap import double_edge_swap
from collections import namedtuple

n_combs = lambda n, k: int(factorial(n)/factorial(n-k)/factorial(k))


def read_ecoli_network(path):
    f = open(path)
    line = f.readline()
    while line.startswith('#'):
        line = f.readline()
    df = pd.read_csv(f, sep="\t", header=None)
    df.loc[-1] = line.split("\t")
    df.index = df.index + 1
    df = df.sort_index()
    f.close()
    return df


def get_actual_parametrization(source, check_input=True, verbose=False):
        
    cfg = source if type(source) is dict else json.load(open(source, "r"))
    
    if check_input:
        assert cfg["NETWORK_TO_SEARCH_IN"] in ["ecoli", "test", "yeast", "ecoli", "gs0.01", "gs0.1", "gs1"]
    
    if verbose:
        for param, value in cfg.items():
            print(f"{param}: {value}")
    
    return cfg

def update_cfg(path, param, value, verbose=False):
    
    cfg = get_actual_parametrization(path, check_input=False, verbose=False)
    cfg[param] = value
    cfg = get_actual_parametrization(cfg, verbose=verbose)
    json.dump(cfg, open(path, "w"))
    
    return cfg


def get_interaction_matrix(config_file):
    
    cwd = os.path.abspath(os.path.join(os.getcwd(), os.pardir))    
    network = config_file["NETWORK_TO_SEARCH_IN"]
    interaction_matrix = joblib.load(
        os.path.join(cwd, "networks", network, f"interaction_matrix.gz")
    )
    
    return interaction_matrix


def build_motif_from_string(string):
    return np.array(list(map(int, string.split()))).reshape(3, 3)


def get_equivalents(core_pattern):
    pattern_variants = []
    for permutation in permutations(range(3)):
        variant = core_pattern[permutation, :]
        variant = variant[:, permutation]
        for prev_variant in pattern_variants:
            if (variant - prev_variant == np.zeros((3, 3))).all():
                break
        else:
            pattern_variants.append(variant)
    return pattern_variants


def print_equivalents(config_file):
    m = build_motif_from_string(json.load(open("./motifs_collection.json", "r"))[config_file["MOTIF_TO_SEARCH_FOR"]])
    if config_file["SELFLOOPS_INCLUDED"]: m += np.diag([1]*3)
    equivalents = get_equivalents(m)
    print(f"""Equivalent forms for {config_file["MOTIF_TO_SEARCH_FOR"]}{" with selfloops" if config_file["SELFLOOPS_INCLUDED"] else ""}\
    ({len(equivalents)} total):""")
    for x in equivalents:
        print(x)
        print()

        
def get_triad_codes(path=None):
    motifs = json.load(open("../motifs_collection.json", "r"))
    salt = np.array([2**i for i in range(6)])
    mapping = {x: i for i, x in enumerate(motifs.keys())}
    codes = {}
    for motif in motifs.keys():
        form = build_motif_from_string(motifs[motif])
        isoforms = get_equivalents(form)
        for isoform in isoforms:
            mask = np.concatenate([np.diag(isoform, k=i) for i in [-2, -1, 1, 2]])
            code =  mask @ np.array([2**i for i in range(6)])
            codes[code] = mapping[motif]
    xcodes = [-1 for _ in range(sum(salt)+1)]
    for code, motif in codes.items():
        xcodes[code] = motif
    xcodes
    return xcodes, {i: x for x, i in mapping.items()}


@njit(cache=True)
def get_motifs(interaction_matrix, combs, codes, n):
    triads = [[(-1, -1, -1)] for _ in range(n)]
    salt = np.array([2**i for i in range(6)]).astype(np.float64)
    n_combinations = len(combs)
    for i in prange(n_combinations):
        c = combs[i]
        cl = np.array(c)
        triad = interaction_matrix[cl, :][:, cl]
        mask = [0]
        for k in [-2, -1, 1, 2]:
            mask += list(np.diag(triad, k=k))
        mask = np.array(mask[1:]).astype(np.float64)
        code = int(mask @ salt)
        idx = codes[code]
        if idx == -1:
            pass
        else:
            triads[idx] += [c]
    return triads


def motif_search(config_file, interaction_matrix, batch_size, dump=False, verbose=False):
    
    network_name = config_file["NETWORK_TO_SEARCH_IN"]
    codes, mapping = get_triad_codes()
    N_CORES = mp.cpu_count() if config_file["N_CORES_TO_USE"] == -1 else config_file["N_CORES_TO_USE"]
    
    def connected_triads_generator(interaction_matrix):
        if type(interaction_matrix) == 'scipy.sparse.csr.csr_matrix':
            interaction_matrix = sparse.csr_matrix.toarray(interaction_matrix)
        interaction_matrix_adj = interaction_matrix - np.diag(np.diag(interaction_matrix))
        tg_idxs, tf_idxs = np.where(interaction_matrix_adj != 0)
        links = pd.DataFrame(index=range(len(tf_idxs)), columns=["tf", "tg"])
        links.tf = tf_idxs
        links.tg = tg_idxs
        links_tf = links.set_index("tf", drop=False)[["tg"]]

        cascades = links.join(links_tf[["tg"]], on="tg", how="inner", rsuffix="_final")
        cascades = cascades[cascades.tf != cascades.tg_final]

        for cascade in cascades.values:
            yield tuple(cascade)

        grouper = links.groupby("tg")
        counter = grouper["tf"].count()
        for tg in counter[counter > 1].index:
            tf_pairs = combinations(links[links.tg == tg].tf.values, 2)
            for tf_1, tf_2 in tf_pairs:
                yield tf_1, tf_2, tg

        grouper = links.groupby("tf")
        counter = grouper["tg"].count()
        for tf in counter[counter > 1].index:
            tg_pairs = combinations(links[links.tf == tf].tg.values, 2)
            for tg_1, tg_2 in tg_pairs:
                yield tf, tg_1, tg_2
    
    triads = connected_triads_generator(interaction_matrix)
    
    def batch_generator(triads):
        batch = []
        counter = 0
        for triad in triads:
            batch.append(triad)
            counter += 1
            if counter == batch_size:
                yield batch
                batch = []
                counter = 0
        yield batch    
    
    def processor(splitted_triads):
        
        def gen_to_queue(input_q, splitted_triads):
            for batch in splitted_triads:
                input_q.put(batch)
            for _ in range(N_CORES):
                input_q.put(None)

        def process(input_q, output_q):
            while True:
                batch = input_q.get()
                if batch is None:
                    output_q.put(None)
                    break
                output_q.put(get_motifs(interaction_matrix, batch, codes, len(mapping)))

        input_q = mp.Queue(maxsize = N_CORES * 2)
        output_q = mp.Queue(maxsize = N_CORES * 2)

        gen_pool = mp.Pool(1, initializer=gen_to_queue, initargs=(input_q, splitted_triads))
        pool = mp.Pool(N_CORES, initializer=process, initargs=(input_q, output_q))

        finished_workers = 0
        while True:
            result = output_q.get()
            if result is None:
                finished_workers += 1
                if finished_workers == N_CORES:
                    break
            else:
                yield result
        
        input_q = None
        output_q = None
        gen_pool.close()
        gen_pool.join()
        pool.close()
        pool.join()
    
    
    splitted_triads = batch_generator(triads)
    motifs_generator = processor(splitted_triads)

    motifs = [[] for _ in range(len(mapping))]
    for batch in tqdm(motifs_generator) if verbose else motifs_generator:
        for i in range(len(mapping)):
            if batch[i][1:] != []:
                for triad in batch[i][1:]:
                    motifs[i].append("_".join(map(str, sorted(triad))))
    motifs = {mapping[i]: list(set(motifs[i])) for i in range(len(mapping))}
    counter = {x: len(y) for x, y in motifs.items()}
    
    if dump:
        joblib.dump(motifs, f"./networks/{network_name}/motifs.gz")
        json.dump(counter, open(f"./networks/{network_name}/counter.json", "w"))
    
    return motifs, counter


def count_triads_nx(interaction_matrix):    
    G = nx.DiGraph(interaction_matrix.T)
    return nx.algorithms.triads.triadic_census(G)


def get_metrics_report(interaction_matrix):
    Report = namedtuple(
        "report",
        ["degree_seq", "avg_degree", "diameter_strong", "diameter_weak",
         "largest_component_frac", "degree_centrality", "betweenness_centrality"]
    )
    G = nx.DiGraph(interaction_matrix.T)
    degree_seq = pd.Series(np.array([x[1] for x in G.degree]))
    avg_degree = degree_seq.mean()
    diameter_weak = diameter(G.to_undirected()) if is_weakly_connected(G) else np.inf
    if is_strongly_connected(G):
        diameter_strong = diameter(G)
        largest_component_frac = 1
    else:
        diameter_strong = np.inf
        strong_components = [(c, len(c)) for c in strongly_connected_components(G)]
        strong_components = sorted(strong_components, key=lambda x: x[1], reverse=True)
        largest_component_frac = strong_components[0][1]/interaction_matrix.shape[0]
    dc = pd.Series(degree_centrality(G))
    bc = pd.Series(betweenness_centrality(G))
    report = Report(*[degree_seq, avg_degree, diameter_strong, diameter_weak, largest_component_frac, dc, bc])
    return report


def get_loops(matrix):
    m = matrix + matrix.T
    x = sorted([sorted([x, y]) for x, y in zip(*np.where(m == 2))])
    y = [x[k] for k in range(len(x)) if k % 2 == 0]
    return y


@njit
def get_shuffled_matrix(interaction_matrix, nswaps):
    shuffled = interaction_matrix.copy()
    tf_nodes = np.where(shuffled.sum(axis=0) != 0)[0]
    for i in range(nswaps):
        tf_1, tf_2 = np.random.choice(tf_nodes, size=2, replace=True)
        tg = shuffled[:, np.array([tf_1, tf_2])]
        x = np.where((tg[:, 0] == 1) & (tg[:, 1] == 0))[0]
        if x.shape[0] > 0:
            tg_1 = np.random.choice(x)
        else:
            continue
        y = np.where((tg[:, 1] == 1) & (tg[:, 0] == 0))[0]
        if y.shape[0] > 0:
            tg_2 = np.random.choice(y)
        else:
            continue
        s = shuffled[np.array([tg_1, tg_2]), :][:, np.array([tf_1, tf_2])]
        e1 = np.diag(np.array([1, 1]))
        e2 = e1[::-1]
        if (s == e1).all():
            shuffled[tg_1, tf_1] = 0
            shuffled[tg_1, tf_2] = 1
            shuffled[tg_2, tf_1] = 1
            shuffled[tg_2, tf_2] = 0
        else:
            shuffled[tg_1, tf_1] = 1
            shuffled[tg_1, tf_2] = 0
            shuffled[tg_2, tf_1] = 0
            shuffled[tg_2, tf_2] = 1
    return shuffled  


def corruption_score(shuffled_matrix, interaction_matrix):
    i, j = np.where(interaction_matrix == 1)
    return shuffled_matrix[i, j].sum()/interaction_matrix[i, j].sum()


def plot_distr(counters_shuffled, counter_orig, label, highlight):
    df = pd.DataFrame(columns=["motif", "abundance", "network"])
    df.motif = counter_orig.keys(); df.abundance = counter_orig.values(); df.network = "original"
    for counter_shuffled in tqdm(counters_shuffled):
        df2 = pd.DataFrame(columns=["motif", "abundance", "network"])
        df2.motif = counter_shuffled.keys(); df2.abundance = counter_shuffled.values(); df2.network = "shuffled"
        df = pd.concat([df, df2], axis=0)
    df.abundance = df.abundance/1000
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
    fig.suptitle(label, fontsize=30)
    for i in range(len(counter_orig.keys())):
        motif = list(counter_orig.keys())[i]
        b = sns.barplot(data=df[df["motif"]==motif], x="motif", y="abundance", hue="network", ax=ax[i], 
                        palette="Blues_r")
        if highlight and motif == highlight:
            b.set_facecolor('xkcd:wheat')
        b.legend_.remove()
#         else:
#             plt.setp(b.get_legend().get_texts(), fontsize='13')
#             plt.setp(b.get_legend().get_title(), fontsize='13')
        b.tick_params("x", labelsize=20)
        b.set_xlabel("",fontsize=0)
        b.set_ylabel("",fontsize=0);
    return df, fig


def get_shuffled_mp(params):
    matrix = params["matrix"]
    nswaps = params["nswaps"]
    return get_shuffled_matrix(matrix, nswaps)

def shuffle_network(matrix, threshold=0.75):
    """
    """
    complete = False
    swaps = (matrix.sum())*0.2
    while not complete:
        shuffled_matrix = get_shuffled_matrix(matrix, swaps)
        shuffled_score = 1-corruption_score(matrix, shuffled_matrix)
        #print(shiffled_score)
        swaps += (matrix.sum())*0.2
        if shuffled_score >= threshold:
            complete = True
    return shuffled_matrix

def generate_random_networks(config_file, interaction_matrix, nsims, nsteps, nswaps):
    counters = []
    for _ in range(nsteps):
        pool = mp.Pool(mp.cpu_count())
        params = {"matrix": interaction_matrix, "nswaps": nswaps}
        shuffled_arrays = pool.map(get_shuffled_mp, (params for _ in range(int(nsims/nsteps))))
        pool.close()
        pool.join()
        for arr in tqdm(shuffled_arrays):
            motifs, counter = motif_search(config_file, arr, batch_size=10000)
            counters.append(counter)
    return counters


def plot_distr_2(counters, counter_orig, ticks):
    distr = {triad: [] for triad in counters[0].keys()}
    for counter in counters:
        for triad, n in counter.items():
            distr[triad].append(n)
    distr = {x: np.array(y) for x, y in distr.items()}
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    for i, motif in enumerate(counters[0].keys()):
        ax = axes[i//3, i%3]
        ax.set_title(motif, fontsize=25)
        pd.Series(distr[motif]).hist(bins=50, ax=ax)
        ax.plot([counter_orig[motif]]*100, np.linspace(0, ticks[i], 100), "r")
        
        
def build_zscores_report(counters, counter_orig):
    distr = {triad: [] for triad in counters[0].keys()}
    for counter in counters:
        for triad, n in counter.items():
            distr[triad].append(n)
    distr = {x: np.array(y) for x, y in distr.items()}
    zscores_report = pd.DataFrame(
        index=["N_real", "mean(N_rand)", "sd(N_rand)", "Z-score", "P-value"]
    )
    for motif in counters[0].keys():
        n_hypothesis = len(counters[0].keys())
        d = distr[motif]
        zscore = (counter_orig[motif]-np.mean(distr[motif]))/np.std(distr[motif])
        pvalue = len(d[d <= counter_orig[motif]])/len(d)
        if pvalue > 0.5:
            pvalue = len(d[d >= counter_orig[motif]])/len(d)
        if pvalue < 0.01/n_hypothesis:
            result = " < 0.01"
        elif pvalue < 0.05/n_hypothesis:
            result = " < 0.05"
        else:
            result = "non-significant"
        result_list = [
            counter_orig[motif],
            np.mean(distr[motif]),
            np.std(distr[motif]),
            zscore,
            pvalue
        ]
        zscores_report[motif] = result_list
    return zscores_report.T


split_motif = lambda x: list(map(int, x.split("_")))


def build_vmn(motifs, verbose=False):
    motifs_network = np.zeros((len(motifs), len(motifs)))
    iterator = combinations(range(len(motifs)), 2)
    if verbose:
        iterator = tqdm(iterator, total=int(len(motifs)*(len(motifs)-1)/2))
    for i, j in iterator:
        m1, m2 = map(lambda x: set(map(int, x.split("_"))), [motifs[i], motifs[j]])
        motifs_network[i, j] = len(m1 & m2)
        motifs_network[j, i] = motifs_network[i, j]
    return motifs_network


def get_sparcity(matrix):
    return matrix.sum()/matrix.shape[0]


def get_tf_content(matrix):
    return len(np.where(matrix.sum(axis=0)!=0)[0])/matrix.shape[0]

def plot_motif_distr(counter):
    a = pd.Series(counter)
    a = a/sum(a)
    plt.title("Motifs distribution", fontsize=15)
    plt.ylim(0, 1)
    plt.bar(a.index, a.values)
    for key in a.keys():
        plt.text(x=key, y=a[key]+0.05,
                 s=f"{a[key]:.3f}", fontsize=10)
#     plt.savefig("./pics/motif_distr_art2.png")

def read_df_as_network(filename):
    with open(filename, 'rt') as f:
        network_df = pd.read_csv(f, sep=' ', header = None)
        network = nx.from_pandas_edgelist(network_df, source = 0, target = 1)
        return network

def out_prob_kernel(out_degree_arr, power_law_degree, random_node):
    out_prob = out_degree_arr[random_node]**power_law_degree
    out_prob /= sum(out_degree_arr**power_law_degree)
    return out_prob
    
def in_prob_kernel(in_degree_arr, power_law_degree, random_node):
    in_prob = in_degree_arr[random_node]**power_law_degree
    in_prob /= sum(in_degree_arr**power_law_degree)
    return in_prob
    
def repeats_density(x, f=0.25, a=3):
    return (f**(1/(1-a))-1)*f**(-x/(1-a))

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def build_Tnet(edges, n):
    """returns adjacency matrix
       requires adjacency list and matrix size
    """
    interaction_matrix = np.zeros((n, n))
    interaction_matrix[edges[:, 0], edges[:, 1]] = 1
    return interaction_matrix

def collect_topological_parameters(config_file, interaction_matrix, label):
    """returns ffl-node participation, sparsity, average in/out-degree
       requires adjacency matrix and config file
    """
    import statistics
    #ffl-part
    motifs, counter = motif_search(config_file, interaction_matrix, batch_size=10000)
    motifs = motifs["030T"]
    ffl_nodes = list(set(sum([list(map(int, x.split("_"))) for x in motifs], [])))
    p1 = len(ffl_nodes)/interaction_matrix.shape[0]
    p1 = len(motifs)

    #sparsity
    p2 = interaction_matrix.sum()/interaction_matrix.shape[0]

    #in-degree
    in_degree = []
    for i in range(interaction_matrix.shape[0]):
        in_degree.append(interaction_matrix[i:].sum()/interaction_matrix.shape[0])
    p3 = sum(in_degree)/len(in_degree)
    #p3 = statistics.median(in_degree)

    #out-degree
    out_degree = []
    for i in range(interaction_matrix.shape[0]):
        out_degree.append(interaction_matrix[:i].sum()/interaction_matrix.shape[0])
    p4 = sum(out_degree)/len(out_degree)
    #p4 = statistics.median(out_degree)
    
    params = list(map(lambda ids: round(ids, 3), [p1, p2, p3, p4]))
    params.append(label)
    
    return params

def collect_ffl_component(config_file, interaction_matrix):
    """returns ffl-node participation
       requires adjacency matrix and config file
    """
    import statistics
    #ffl-part
    motifs, counter = motif_search(config_file, interaction_matrix, batch_size=10000)
    motifs = motifs["030T"]
    ffl_nodes = list(set(sum([list(map(int, x.split("_"))) for x in motifs], [])))
    p1 = len(ffl_nodes)/interaction_matrix.shape[0]
    
    return p1


def get_free_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return (free_memory*1024)/1000000

def get_memory_usage():
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)/1000000

def limit_memory(maxsize):
    maxsize = (maxsize*1000000)/1024
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

def analyze_exctracted_network(config_file, path_to_tsv, network_label, network_rep, size, stability_motifs=False):
    """
    collect topological stats from extracted networks
    """
    import networkx as nx
    if network_label == 'randg' or network_label == 'dag':
        interaction_matrix = np.array(pd.read_csv(path_to_tsv, header = None, sep=','))
        interaction_matrix = np.apply_along_axis(list, 1, interaction_matrix)
        interaction_matrix = (interaction_matrix > 0).astype(np.int_)
        #print(interaction_matrix)
    else:
        edges = pd.read_csv(path_to_tsv, sep="\t")
        if network_label == 'gnw':
            edges.columns = ["tf", "tg"]
            #edges.columns = ["tf", "tg", "strength"]
            #edges = edges[["tf", "tg"]]
        else:
            edges.columns = ["tf", "tg"]
        edges['tf'].astype(str)
        edges['tg'].astype(str)
        edges.columns = ["tf", "tg"]
    
        nodes = sorted(np.unique(np.concatenate((edges.tf.unique(), edges.tg.unique()))))
        nodes = pd.DataFrame(data=range(len(nodes)), index=nodes, columns=["idx"])
        edges_ = edges.join(nodes, on="tf").join(nodes, on="tg", lsuffix="_tf", rsuffix="_tg")
        np_edges = edges_[["idx_tg", "idx_tf"]].values
        interaction_matrix = build_Tnet(np_edges, len(nodes))
    
    #if shuffled:
    #    interaction_matrix = shuffle_network(interaction_matrix)
    #print(interaction_matrix)
    topological_properties = collect_topological_parameters(config_file,interaction_matrix, network_label)
    topological_properties.append(size)
    topological_properties.append(network_rep)
    
    if stability_motifs:
        #ffl_counts = topological_properties[0]
        #graph_nx = nx.DiGraph(interaction_matrix)
        #cycles_counts = list(nx.algorithms.cycles.simple_cycles(graph_nx))
        #topological_properties = [ffl_counts, cycles_counts]
        motifs, counter = motif_search(config_file, interaction_matrix, batch_size=10000)
        shuffled_counters = generate_random_networks(config_file, interaction_matrix, 10, 10, 60000)
        #topological_properties = counter
        #topological_properties = {k:len(v) for k, v in counter.items()}
        topological_properties = build_zscores_report(shuffled_counters, counter)

    return topological_properties

def create_nx_network(n_trials,sparsity,size,out_dir):
    """
    requires number of networks, desired sparsity, desired network size, and output dir
    creates a set of networks (adjacency list format)
    """
    import numpy as np
    import networkx as nx
    import random
    import os

    for number in range(n_trials):
        test1 = nx.scale_free_graph(int(size), alpha=0.85, beta=0.1, gamma=0.05, delta_in=0.2, delta_out=0)
        edges1 = test1.number_of_edges()
        edge_list = list(set([e for e in test1.edges()]))
        edge_list = [list(ele) for ele in edge_list]
        nx_size = len(list(set([e for l in edge_list for e in l])))
        edges = pd.DataFrame(edge_list, columns=['tf', 'tg'])

        nodes = sorted(np.unique(np.concatenate((edges.tf.unique(), edges.tg.unique()))))
        nodes = pd.DataFrame(data=range(len(nodes)), index=nodes, columns=["idx"])
        edges_ = edges.join(nodes, on="tf").join(nodes, on="tg", lsuffix="_tf", rsuffix="_tg")
        np_edges = edges_[["idx_tg", "idx_tf"]].values

        interaction_matrix = build_Tnet(np_edges, len(nodes))
        interaction_matrix = interaction_matrix.T

        links_per_node = interaction_matrix.sum()/interaction_matrix.shape[0]
        nodes = list(range(0, len(interaction_matrix)))
        
        #sparsity = sparsity+(np.random.uniform(-2,2)*0.1)

        while links_per_node<sparsity:
            #print(links_per_node)
            #print(interaction_matrix)
            #calculate in, out degree
            out_degree = interaction_matrix.sum(axis=1)
            in_degree = interaction_matrix.sum(axis=0)

            #calculate probs
            in_probs = pd.Series(in_degree/sum(in_degree), nodes)
            out_probs = pd.Series(out_degree/sum(out_degree), nodes)

            #nodes that create edge
            regulator = np.random.choice(out_probs.index, p=out_probs.values)
            regulatee = np.random.choice(in_probs.index)
            if regulatee == regulator:
                while regulatee == regulator:
                    regulatee = np.random.choice(in_probs.index)
                    #print([regulator, regulatee])

            #add edge
            interaction_matrix[regulator,regulatee] = 1
            links_per_node = interaction_matrix.sum()/interaction_matrix.shape[0]
            #print(links_per_node)

        #create adj list
        adj_list = []
        for name_regulatee, i in enumerate(interaction_matrix.T):
            for name_regulator, j in enumerate(interaction_matrix):
                if interaction_matrix[name_regulatee][name_regulator] == 1:
                    adj_list.append([name_regulator, name_regulatee])
        
        #file name
        network_name = '_'.join(str(x) for x in ['scale_free_nx',number,'nodes',len(nodes)])
        
        #out dir
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        #store adjacency list of nx network:    
        with open(out_dir+'/'+network_name+'.tsv', "w", newline="") as file:
            writer = csv.writer(file, delimiter ='\t')
            writer.writerows(adj_list)
        
    return

def analyze_connectivity(path_to_tsv, network_label, network_rep, size):
    """
    collect topological stats from extracted networks
    """
    import networkx as nx
    if network_label == 'randg' or network_label == 'dag':
        interaction_matrix = np.array(pd.read_csv(path_to_tsv, header = None, sep=','))
        interaction_matrix = np.apply_along_axis(list, 1, interaction_matrix)
        interaction_matrix = (interaction_matrix > 0).astype(np.int_)
        #print(interaction_matrix)
    else:
        edges = pd.read_csv(path_to_tsv, sep="\t")
        if network_label == 'gnw':
            edges.columns = ["tf", "tg"]
            #edges.columns = ["tf", "tg", "strength"]
            #edges = edges[["tf", "tg"]]
        else:
            edges.columns = ["tf", "tg"]
        edges['tf'].astype(str)
        edges['tg'].astype(str)
        edges.columns = ["tf", "tg"]
    
        nodes = sorted(np.unique(np.concatenate((edges.tf.unique(), edges.tg.unique()))))
        nodes = pd.DataFrame(data=range(len(nodes)), index=nodes, columns=["idx"])
        edges_ = edges.join(nodes, on="tf").join(nodes, on="tg", lsuffix="_tf", rsuffix="_tg")
        np_edges = edges_[["idx_tg", "idx_tf"]].values
        interaction_matrix = build_Tnet(np_edges, len(nodes))
    
    interaction_matrix_graph = nx.DiGraph(interaction_matrix)
    degree_list_test = sorted([d for n, d in interaction_matrix_graph.degree])

    connectivity = {}
    for number, edges in enumerate(degree_list_test):
        av_nodes = sum(degree_list_test[number::])/len((degree_list_test[number::]))
        frequency = np.round(len((degree_list_test[number::]))/len((degree_list_test)), 4)
        connectivity[frequency] = np.round(np.log10(av_nodes), 4)
    df_connectivity = pd.DataFrame({'frequency': list(connectivity.keys()),'average degree':list(connectivity.values())})
    d = np.polyfit(df_connectivity['average degree'],df_connectivity['frequency'],1)
    f = np.poly1d(d)
    df_connectivity.insert(2,'Rfreq',f(df_connectivity['average degree']))
    df_connectivity['network'] = network_label
    df_connectivity['size'] = size
    df_connectivity['rep'] = network_rep

    return df_connectivity
