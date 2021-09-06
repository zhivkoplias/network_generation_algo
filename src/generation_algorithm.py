import auxilary_functions as f
cfg = f.get_actual_parametrization("../src/config.json")
#cfg = f.update_cfg("../src/config.json", "NETWORK_TO_SEARCH_IN", "yeast")
import psutil
import os
import numpy as np
import pandas as pd
import sys
import joblib
sys.path.insert(0, "../src")
ART_NET_PATH = "../networks"

import auxilary_functions as f
from generation_algorithm import *
from copy import deepcopy
import networkx as nx
from collections import namedtuple
from itertools import product, combinations
from matplotlib import pyplot as pltthe
from datetime import datetime
from tqdm import tqdm
from time import sleep
import multiprocessing as mp


def get_network_nucleus(
    interaction_matrix, motifs, motifs_network, min_size, random_seed=cfg["RANDOM_SEED"]
):
    """
    Getting subsample from real network as a nucleus for artificial network
    ________________________________________________________________________
    interaction_matrix (numpy.array) - binary interaction matrix for genes
    motifs (numpy.array) - list of unique identifiers for condidered motifs (FFL triads)
    motifs_network (numpy.array) - vertex-based motifs network (linkage by shared nodes)
    min_size (int) - minimal required size of resulting nucleus (may be slightly higher eventually)
    random_seed (int) - reproducibility parameter
    
    """
    np.random.seed(random_seed)
    substrate_motif_idxs = [np.random.randint(len(motifs))]
    substrate_motifs = np.array([motifs[i] for i in substrate_motif_idxs])
    substrate_size = len(set(sum([f.split_motif(motif) for motif in substrate_motifs], [])))

    # grow network nucleus while required size obtained
    while substrate_size < min_size:
        neighbors = np.where(motifs_network[:, substrate_motif_idxs].sum(axis=1) != 0)[0]
        neighbors = np.array(list(set(neighbors) - set(substrate_motif_idxs)))
        # assignment of weights to candidate motifs by their connectivity
        # with already selected motifs grown substrate network
        weights = motifs_network[neighbors, :][:, substrate_motif_idxs].sum(axis=1)
        weights /= sum(weights)
        substrate_motif_idxs.append(np.random.choice(neighbors, size=1, p=weights)[0])
        substrate_motifs = np.array([motifs[i] for i in substrate_motif_idxs])
        substrate_size = len(set(sum([f.split_motif(motif) for motif in substrate_motifs], [])))

    # interaction matrix building
    G = nx.DiGraph()
    for motif in substrate_motifs:
        nodes = f.split_motif(motif)
        M = nx.DiGraph(interaction_matrix[nodes, :][:, nodes])
        M = nx.relabel_nodes(M, mapping={i: node for i, node in enumerate(nodes)})
        G = nx.compose(G, M)
    substrate_matrix = nx.convert_matrix.to_numpy_array(G)
    return substrate_matrix

#### Parametrization
#Motif types and number of shared nodes distributions inference. 

#The support set for FFL motif type by TF/TG content is {TTT, TTG} where T and G are for TF and TG respectively.

#The support set for the number of shared nodes is {1, 2}. We are not considering 0 as we focus only on the largest connected component of FFL VMN which actually contains all of the FFLs in the yeast Tnet and nearly all (99%) in E.coli Tnet

def get_network_params(interaction_matrix, verbose=False,motif_search=True,known_motifs=False):
    # motif search
    if motif_search:
        motifs, counter = f.motif_search(
        cfg, interaction_matrix, batch_size=10000, verbose=False
    )
        motifs = motifs["030T"]
    else:
        motifs = known_motifs
    
    # TF/TG recognition
    tf_nodes = np.where(interaction_matrix.sum(axis=0) != 0)[0]
    tg_nodes = np.where(interaction_matrix.sum(axis=0) == 0)[0]
    # motif type distribution
    n_tg_nodes_list = np.array(
        [len(set(f.split_motif(motif)) - set(tf_nodes)) for motif in motifs]
    )
    mtype_probs = pd.Series(n_tg_nodes_list).value_counts(normalize=True).sort_index()
    if verbose:
        prob = len(tf_nodes)/interaction_matrix.shape[0]
        print(f"TF content: {prob}")
        print("Number of TG in motif distribution:")
        print(mtype_probs)
        print()
    # nodes participation in FFL
    node_part = np.zeros(interaction_matrix.shape[0])
    for triad in motifs:
        for x in map(int, triad.split("_")):
            node_part[x] += 1
    node_part = pd.Series(node_part)
    if verbose:
        print("Node patricipation distribution:")
        print(node_part.value_counts(normalize=True).head())
        print()
    # Distribution of X-unique nodes motifs number
    edges_1 = []
    motifs_0 = []
    types = {i: 0 for i in range(3)}
    for triad in motifs:
        res = 0
        i = 0
        for x in map(int, triad.split("_")):
            res += node_part[x] == 1
            if node_part[x] == 1:
                i = x
        types[res] += 1
        if res == 1:
            edges_1.append(set(f.split_motif(triad))-set([i]))
        if res == 0:
            motifs_0.append(triad)
    types = pd.Series(types)
    unique_nodes = types/sum(types)
    if verbose:
        print("Unique nodes number distribution")
        print(unique_nodes)
        print()
    #  Is edge unique? (for 1-unique node motifs)
    edges_1_part = {"_".join(map(str, sorted(edge))): 0 for edge in edges_1}
    for triad in motifs:
        for x in combinations(f.split_motif(triad), 2):
            edge_1 = "_".join(map(str, sorted(x)))
            try:
                edges_1_part[edge_1] += 1
            except KeyError:
                pass
    edges_1_part = pd.Series(edges_1_part)
    unique_edges_1 = (edges_1_part == 1).value_counts(normalize=True)
    if verbose:
        print("Is edge unique? (for 1-unique node motifs)")
        print(unique_edges_1)
        print()
    # Distribution of X-unique edges motifs (for 0-unique nodes motifs)
    edges_0_part = {}
    for triad in motifs_0:
        for x in combinations(f.split_motif(triad), 2):
            edge_0 = "_".join(map(str, sorted(x)))
            try:
                edges_0_part[edge_0] += 1
            except KeyError:
                edges_0_part[edge_0] = 1
    edges_0_part = pd.Series(edges_0_part)
    edge_types = {i: 0 for i in range(4)}
    for triad in motifs_0:
        res = 0
        for x in combinations(f.split_motif(triad), 2):
            edge_0 = "_".join(map(str, sorted(x)))
            res += edges_0_part[edge_0] == 1
        edge_types[res] += 1
    edge_types = pd.Series(edge_types)
    unique_edges_0 = edge_types/sum(edge_types)
    if verbose:
        print("Distribution of X-unique edges motifs (for 0-unique nodes motifs)")
        print(edge_types)
        print(unique_edges_0)
        print()
    Params = namedtuple(
        "Params", "substrate_motifs tf_nodes tg_nodes mtype_probs unique_nodes unique_edges_1 unique_edges_0"
    )
    params = Params(
        *[motifs, tf_nodes, tg_nodes, mtype_probs, unique_nodes, unique_edges_1, unique_edges_0]
    )
    return params

#### Single attachment step

#Random selection of the inner/outer motif types and number of shared nodes with probabilities from the previous step

def get_attachment_params(substrate_matrix, params, growth_pace):
    """
    Selection of inner/outer motifs and number of shared nodes
    ________________________________________________________________________
    params - network parameters from previous stage of analysis (see get_network_params)
    """
    # number of unique nodes in the outer motif
    normal_unique_nodes = params.unique_nodes.loc[[0, 1]]
    unique_nodes = normal_unique_nodes
    #print('uniq nodes')
    #print(unique_nodes)
    #unique_nodes = unique_nodes/sum(unique_nodes)
    unique_nodes[1] = (growth_pace)
    unique_nodes[0] = (1 - growth_pace)
    #print(unique_nodes.values)
	
    n_unique_nodes = np.random.choice(
        unique_nodes.index, p=unique_nodes.values
    )
#     n_unique_nodes = np.random.choice(
#         params.unique_nodes.index, p=params.unique_nodes.values
#     )
    # number of unique edges in the outer motif
    n_unique_edges = None
    if n_unique_nodes == 1:
        n_unique_edges = 0
#         n_unique_edges = int(np.random.choice(
#             params.unique_edges_1.index, p=params.unique_edges_1.values
#         ))
    elif n_unique_nodes == 0:
        unique_edges_0 = params.unique_edges_0.loc[[1, 2]]
        unique_edges_0 = unique_edges_0/sum(unique_edges_0)
        n_unique_edges = int(np.random.choice(
            unique_edges_0.index, p=unique_edges_0.values
        ))
#         n_unique_edges = int(np.random.choice(
#             params.unique_edges_0.index, p=params.unique_edges_0.values
#         ))
    # buiding VMN for motifs
    substrate_vmn = f.build_vmn(params.substrate_motifs)    
    
    # motif selection for 1 node/2 edges attachment case
    if n_unique_nodes == 1 and n_unique_edges == 0:
        substrate_type_idxs = list(range(len(params.substrate_motifs)))
        substrate_type_vmn = substrate_vmn
        weights = substrate_type_vmn.sum(axis=1)
        weights /= sum(weights)
        inner_motif_1_idx = np.random.choice(substrate_type_idxs, p=weights)
        inner_motif_2_idx = None
    # motif selection for 0 nodes/1 edges attachment case
    elif n_unique_nodes == 0 and n_unique_edges == 1:
        substrate_type_idxs = np.where((substrate_vmn==1)|(substrate_vmn==2))[0]
        substrate_type_vmn = substrate_vmn[substrate_type_idxs, :]
        weights = substrate_type_vmn.sum(axis=1)
        weights /= sum(weights)
        while True:
#             inner_motif_1_idx = np.random.choice(substrate_type_idxs, p=weights)
            inner_motif_1_idx = np.random.choice(substrate_type_idxs)
            line = substrate_vmn[inner_motif_1_idx, :]
            inner_motif_2_idx = np.random.choice(np.where((line==1)|(line==2))[0])
            # check if there at least one pair of nodes with no link between
            inner_motif_1 = f.split_motif(params.substrate_motifs[inner_motif_1_idx])
            inner_motif_2 = f.split_motif(params.substrate_motifs[inner_motif_2_idx])
            indecies = list(set(inner_motif_1) | set(inner_motif_2))
            a = substrate_matrix[indecies, :][:, indecies]
            if not (a + a.T + np.diag([1]*a.shape[0]) == 1).all():
                break

    # motif selection for 0 nodes/2 edges attachment case
    elif n_unique_nodes == 0 and n_unique_edges == 2:
        substrate_type_idxs = np.where((substrate_vmn==1)|(substrate_vmn==0))[0]
        substrate_type_vmn = substrate_vmn[substrate_type_idxs, :]
        weights = substrate_type_vmn.sum(axis=1)
        weights /= sum(weights)
        while True:
#             inner_motif_1_idx = np.random.choice(substrate_type_idxs, p=weights)
            inner_motif_1_idx = np.random.choice(substrate_type_idxs)
            line = substrate_vmn[inner_motif_1_idx, :]
            inner_motif_2_idx = np.random.choice(np.where((line==1)|(line==0))[0])
            # check if there at least one node with no links with two others
            inner_motif_1 = f.split_motif(params.substrate_motifs[inner_motif_1_idx])
            inner_motif_2 = f.split_motif(params.substrate_motifs[inner_motif_2_idx])
            indecies = list(set(inner_motif_1) | set(inner_motif_2))
            a = substrate_matrix[indecies, :][:, indecies]
            b = a + a.T; b
            if sum(np.array([b.shape[0] - np.count_nonzero(x) for x in b]) >= 3):
                break
        
    # determine types 
    intype_1 = len(set(f.split_motif(params.substrate_motifs[inner_motif_1_idx])) - set(params.tf_nodes))
    if inner_motif_2_idx is not None:
        intype_2 = len(set(f.split_motif(params.substrate_motifs[inner_motif_2_idx])) - set(params.tf_nodes))
    else:
        intype_2 = None
    # incoming motif type selection
    outtype = np.random.binomial(1, p=0.5)
#     outtype = np.random.binomial(1, p=len(params.tg_nodes)/substrate_matrix.shape[0])
    # results packing
    Params = namedtuple(
        "Params", "substrate_motifs inner_motif_1_idx inner_motif_2_idx intype_1 intype_2 outtype n_unique_nodes n_unique_edges"
    )
    params = Params(
        *[params.substrate_motifs, inner_motif_1_idx, inner_motif_2_idx, intype_1, intype_2, outtype, n_unique_nodes, n_unique_edges]
    )
    return params

#### Attach 1 node and 2 edges

#<img src="./pics/shared_edge_pattern.png" width=600 height=20/>

#Results of isoforms diversity analysis:

#(inner motif type / outer motif type / isoforms number)

#- TTT / TTT - 9 variants
#- TTT / TTG - 3 variants
#- TTG / TTT - 3 variants
#- TTG / TTG - 5 variants

def get_outer_motif_matrix(role_edge):
    """
    Selection of incoming motif matrix based on shared edge type
    """
    if role_edge == (2, 1):
        return f.build_motif_from_string("0 1 1 0 0 0 0 1 0")
    if role_edge == (2, 0):
        return f.build_motif_from_string("0 1 0 0 0 0 1 1 0")
    if role_edge == (1, 0):
        return f.build_motif_from_string("0 0 0 1 0 0 1 1 0")

def get_attachment_1n2e(substrate_matrix, params):
    """
    Attachment patterns constructing for shared edge case and random selection the particular one
    ________________________________________________________________________
    substrate_matrix - the netwotk we are growing
    params - attachment parameters from previous stage of analysis (see get_attachment_params)
    """
    try:
        tg_total = np.where(substrate_matrix.sum(axis=0) == 0)[0]
        #print('tg total: '+str(tg_total))
        inner_motif = f.split_motif(params.substrate_motifs[params.inner_motif_1_idx])
        #print('tg inner_motif: '+str(inner_motif))
        graph_nx = nx.DiGraph(substrate_matrix)
        #for node in graph_nx.nodes:
            #if node in inner_motif:
                #print('node: '+str(node))
                #print(list(nx.all_neighbors(graph_nx, node)))
        #print('size: '+str(substrate_matrix.shape[0]))


        inner_motif_matrix = substrate_matrix[inner_motif, :][:, inner_motif]
        #print("inner_motif_matrix: ")
        #print(inner_motif_matrix)
        inner_nodes_roles = list(inner_motif_matrix.sum(axis=0).astype(int))
        #print('inner nodes roles: '+str(inner_nodes_roles))

        # check if there is a target gene
        tg_in = inner_nodes_roles.index(0) if params.intype_1 == 1 else None
        #print(tg_in)
        # assignment roles and corresponding identifiers to the edges
        role_edges = list(combinations(range(2, -1, -1), 2))
        idx_by_role = lambda x: inner_nodes_roles.index(x)
        idx_edges_in = [(idx_by_role(source), idx_by_role(target)) for source, target in role_edges]
        #print('idx_edges_in:')
        #print(idx_edges_in)
        patterns = []

        prob = len(tg_total)/substrate_matrix.shape[0]
        if params.outtype == 1:
            append_gene = np.random.binomial(1, p=prob*(2-prob))
        else:
            append_gene = np.random.binomial(1, p=prob)

        for idx_edge_in, role_edge_out in product(idx_edges_in, role_edges):
            outer_motif_matrix = get_outer_motif_matrix(role_edge_out)
            outer_nodes_roles = list(outer_motif_matrix.sum(axis=0).astype(int))
            # reveal positions by known edges roles
            idx_by_role = lambda x: outer_nodes_roles.index(x)
            idx_edge_out = tuple(idx_by_role(v) for v in role_edge_out)
            # check if there is a target gene
            tg_out = outer_motif_matrix.sum(axis=0).argmin() if params.outtype == 1 else None

            if tg_out is not None and tg_out not in idx_edge_out and not append_gene:
                continue

            if append_gene and not(tg_out is not None and tg_out not in idx_edge_out):
                continue

            # filtering out inappropriate patterns (by target gene )
            if (idx_edge_in[0] != tg_in) & (idx_edge_out[0] == tg_out):
                continue
            if (idx_edge_in[0] == tg_in) & (idx_edge_out[0] != tg_out):
                continue
            if (idx_edge_in[1] != tg_in) & (idx_edge_out[1] == tg_out):
                continue
            if (idx_edge_in[1] == tg_in) & (idx_edge_out[1] != tg_out):
                continue
            I = nx.DiGraph(inner_motif_matrix)
            O = nx.DiGraph(outer_motif_matrix)
            mapping = {i: i+3 for i in range(3)}
            mapping[1], mapping[2] = idx_edge_in
            O = nx.relabel_nodes(O, mapping=mapping)
            C = nx.compose(I, O)
            compounded_matrix = nx.convert_matrix.to_numpy_array(C)
            patterns.append(compounded_matrix)
        if patterns:
            attachment_matrix = patterns[np.random.randint(len(patterns))]
        else:
            attachment_matrix = None
        #print(attachment_matrix)

        try:
            all_nodes = []
            for num, node in enumerate(inner_motif):
                #print(inner_nodes_roles[num])
                if inner_nodes_roles[num] == 2 or inner_nodes_roles[num] == 1:
                    all_nodes.append(node)
            all_nodes.append(int(substrate_matrix.shape[0])-1)
            all_nodes = sorted(all_nodes)

            key_to_update = '_'.join([str(i) for i in all_nodes])
            #print(key_to_update)
            return attachment_matrix, key_to_update

        except AttributeError or ValueError:
            return attachment_matrix
    except ValueError:
        attachment_matrix = None
        return attachment_matrix

#### Attach 0 nodes and 1 edge

def prepare_motif_libs():
    '''dl - downlink
    ul - uplink
    cs - cascade'''
    dl_lib = [
        "".join(map(str, x.flatten().astype(int))) for x in
        f.get_equivalents(f.build_motif_from_string("0 0 0 1 0 0 1 0 0"))
    ]
    ul_lib = [
        "".join(map(str, x.flatten().astype(int))) for x in
        f.get_equivalents(f.build_motif_from_string("0 1 1 0 0 0 0 0 0"))
    ]
    cs_lib = [
        "".join(map(str, x.flatten().astype(int))) for x in
        f.get_equivalents(f.build_motif_from_string("0 1 0 0 0 0 1 0 0"))
    ]
    return dl_lib, ul_lib, cs_lib

def get_attachment_0n1e(substrate_matrix, params):
    """
    Attachment patterns constructing for one edge attachment case 
    and random selection the particular one
    ________________________________________________________________________
    substrate_matrix - the netwotk we are growing
    params - attachment parameters from previous stage of analysis (see get_attachment_params)
    """
    dl_lib, ul_lib, cs_lib = prepare_motif_libs()
    inner_motif_1 = f.split_motif(params.substrate_motifs[params.inner_motif_1_idx])
    inner_motif_2 = f.split_motif(params.substrate_motifs[params.inner_motif_2_idx])
    tf_total = np.where(substrate_matrix.sum(axis=0) != 0)[0]
    tg_total = np.where(substrate_matrix.sum(axis=0) == 0)[0]
#     print("Motifs:", inner_motif_1, inner_motif_2)
    # if there tg nodes in this motif pair
    tg_nodes = (set(inner_motif_1) | set(inner_motif_2)) & set(tg_total)
    # separate nodes by motif they belong to (or both of them)
    shared_nodes = list(set(inner_motif_1) & set(inner_motif_2))
    unique_nodes_1 = list(set(inner_motif_1) - set(inner_motif_2))
    unique_nodes_2 = list(set(inner_motif_2) - set(inner_motif_1))
    # construct all possible triads (common vertex - node in 1st motif - node in 2nd motif)
    triads = product(shared_nodes, product(unique_nodes_1, unique_nodes_2))
    get_type = lambda triad: len(set(triad) - set(tf_total))
    triads = [(x, y, z) for x, (y, z) in triads if get_type((x, y, z))==params.outtype]
#     print(triads)
    # accumulate possible links upon base motif
    possible_links = []
    for triad in triads:
        triad_matrix = substrate_matrix[triad, :][:, triad]
        triad_str = "".join(map(str, triad_matrix.flatten().astype(int)))
        node_1, node_2 = triad[1:]
        if triad_str in dl_lib:
            if node_2 not in tg_nodes:
                possible_links.append((node_1, node_2))
            if node_1 not in tg_nodes:
                possible_links.append((node_2, node_1))
        elif triad_str in ul_lib:
            possible_links.append((node_1, node_2))
            possible_links.append((node_2, node_1))
        elif triad_str in cs_lib:

            if triad_str[3] == "1":
                possible_links.append((node_1, node_2))
            else:
                possible_links.append((node_2, node_1))
    possible_links = list(set(possible_links))
    
    if possible_links:
        link_to_attach = possible_links[np.random.choice(range(len(possible_links)))]
    else:
        link_to_attach = None

    if link_to_attach != None:
        graph_nx = nx.DiGraph(substrate_matrix)
        neighbours = []
        for node in graph_nx.nodes:
            if link_to_attach:
                if node in link_to_attach:
                    #print('node: '+str(node))
                    neighbours.append(list(nx.all_neighbors(graph_nx, node)))

        common_node = list(set(neighbours[0]).intersection(set(neighbours[1])))
        all_nodes = sorted([link_to_attach[0], link_to_attach[1], common_node[0]])
        key_to_update = '_'.join([str(i) for i in all_nodes])
        return link_to_attach, key_to_update
    else:
        return link_to_attach

#### Attach 0 nodes and 2 edge

def get_attachment_0n2e(substrate_matrix, params):
    """
    Attachment patterns constructing for two edges attachment case 
    and random selection the particular one
    ________________________________________________________________________
    substrate_matrix - the netwotk we are growing
    params - attachment parameters from previous stage of analysis (see get_attachment_params)
    """
    inner_motif_1 = f.split_motif(params.substrate_motifs[params.inner_motif_1_idx])
    inner_motif_2 = f.split_motif(params.substrate_motifs[params.inner_motif_2_idx])
    tf_total = np.where(substrate_matrix.sum(axis=0) != 0)[0]
    tg_total = np.where(substrate_matrix.sum(axis=0) == 0)[0]
#     print("Motifs:", inner_motif_1, inner_motif_2)
    # if there tg nodes in this motif pair
    tg_nodes = (set(inner_motif_1) | set(inner_motif_2)) & set(tg_total)
#     print("TG nodes", tg_nodes)
    # separate nodes by motif they belong to (or both of them)
    shared_nodes = list(set(inner_motif_1) & set(inner_motif_2))
#     print("Shared nodes:", shared_nodes)
    unique_nodes_1 = list(set(inner_motif_1) - set(inner_motif_2))
#     print("Unique nodes 1:", unique_nodes_1)
    unique_nodes_2 = list(set(inner_motif_2) - set(inner_motif_1))
#     print("Unique nodes 2:", unique_nodes_2)
    # triads construction
    triads = []
    if len(shared_nodes) == 1:
        unique_nodes = [unique_nodes_1, unique_nodes_2]
        for i in range(2):
            y, z =  unique_nodes[i-1]
            for x in unique_nodes[i]:
                # check if there is a place for two new links
                if substrate_matrix[(x, y, z), :][:, (x, y, z)].sum() == 1:
                    triads.append((x, y, z))
    else:
        edges_total = [list(combinations(inner_motif_1, 2)), list(combinations(inner_motif_2, 2))]
        nodes_total = [inner_motif_1, inner_motif_2]
        for i in range(2):
            edges = edges_total[i]
            nodes = nodes_total[i-1]
            for y, z in edges:
                for x in nodes:
                    # check if there is a place for two new links
                    if substrate_matrix[(x, y, z), :][:, (x, y, z)].sum() == 1:
                        triads.append((x, y, z))
    get_type = lambda triad: len(set(triad) - set(tf_total))
    triads = [triad for triad in triads if get_type(triad)==params.outtype]
    possible_link_pairs = []
    
    for triad in triads:
    #     print(triad)
        triad_matrix = substrate_matrix[triad, :][:, triad]
    #     print(triad_matrix)
        target, source = map(lambda x: x[0], np.where(triad_matrix == 1))
        outer = list(set(range(3)) - set([target, source]))[0]
        triad = tuple(triad[i] for i in [source, target, outer])
    #     print(triad)
        triad_matrix = substrate_matrix[triad, :][:, triad]
    #     print(triad_matrix)
        link_pairs = [
            [(2, 0), (2, 1)], [(2, 0), (1, 2)], [(0, 2), (1, 2)]
        ]
        link_pairs = [
            [(triad[i], triad[j]), (triad[k], triad[l])] for (i, j), (k, l) in link_pairs
            if triad[j] not in tg_nodes and triad[l] not in tg_nodes
        ]
        possible_link_pairs += link_pairs
        
    if possible_link_pairs:
        link_pair = possible_link_pairs[np.random.choice(range(len(possible_link_pairs)))]
    else:
        link_pair = None
    
    if link_pair != None:
        all_nodes = sorted(list(set([y for x in link_pair for y in x])))
        key_to_update = '_'.join([str(i) for i in all_nodes])
    
        return link_pair, key_to_update 
    else:
        return link_pair

#### Update
def update_edge_list(art_matrix, edge_list_update=None, num_of_nodes=1, power_law_degree=0.8):
    """Barabasi's node prefferential attachment algorithm with power law attachment kernel
    art_matrix - adjacency matrix
    power_law_degree - power for in/out degree parameter
    num_of_nodes - number of iterations"""
    #art_matrix = art_matrix.transpose()
    counter = 0
#     print(f"starting num of edges: {art_matrix.sum()}")
    
    #list of added nodes
    if edge_list_update==None:
        edge_list_ins = []
        edge_list_outs = []
    else:
        edge_list_ins = edge_list_update[0]
  #      print(f"edge_list_ins: {edge_list_ins}")
        edge_list_outs = edge_list_update[1]
  #      print(f"edge_list_outs: {edge_list_outs}")
    
 #   print(type(barabasi_list))
    while counter <= num_of_nodes:
        # calculate in/out degree
        out_degree_arr = art_matrix.sum(axis=0)
#         print(f"starting out_degree_arr: {out_degree_arr}")
        in_degree_arr = art_matrix.sum(axis=1)
#         print(f"starting in_degree_arr: {in_degree_arr}")

        # take candidate node randomly
        candidate = np.random.choice(range(art_matrix.shape[0]))
        random_node = candidate

        # calculate attachment kernel (probs)
        out_prob = f.out_prob_kernel(out_degree_arr, power_law_degree, random_node)
        in_prob = f.in_prob_kernel(in_degree_arr, power_law_degree, random_node)

        # drop number of repeats (from exp)
        variants = np.linspace(0, 100, 101).astype(int)
        probs = f.repeats_density(variants)/2
        n_repeats = np.random.choice(variants, size=1, p=probs)[0]
        #print(f"n_repeats: {n_repeats}")
        if n_repeats:
    #     print(candidate)
    #        print(out_prob, in_prob, n_repeats)
            for i in range(n_repeats):
                # drop random number from (0, 1)
                seed = np.random.rand()
#                 print(f"seed: {seed}")
#                 print(f"in-prob: {in_prob}")
#                 print(f"out-prob: {out_prob}")
            
                # compare with kernel value and add or do not add out node+link
                if seed < out_prob:
                    edge_list_outs.append([candidate, 'out'])
                    counter += 1
                
                # compare with kernel value and add or do not add in node+link
                if seed < in_prob:
                    edge_list_ins.append(['in',candidate])
                    counter += 1
        out_degree_arr = art_matrix.sum(axis=0)
#         print(f"final out_degree_arr: {out_degree_arr}")
#         in_degree_arr = art_matrix.sum(axis=1)
#         print(f"final in_degree_arr: {in_degree_arr}")
        
        
#     print(f"final num of edges: {art_matrix.sum()}")
    edge_list_update = [edge_list_ins, edge_list_outs]
 #   print(f"edge_list_update: {edge_list_update}")
 #   print(len(edge_list_ins)+len(edge_list_outs))
    #print(edge_list_update)
    return edge_list_update, len(edge_list_ins)+len(edge_list_outs)

def update_substrate_matrix(substrate_matrix, attachment_pattern, inner_motif, nodes_attach=False):
    """
    Substrate network update by selected attachment pattern
    ________________________________________________________________________
    substrate_matrix - the netwotk we are growing
    attachment_matrix - randomly selected attachment pattern compatible with chosen params
    inner_motif - inner triad we attach to
    """
    substrate_matrix_upd = deepcopy(substrate_matrix)
    if attachment_pattern is not None:
        if nodes_attach:
            n_nodes_to_join = int(attachment_pattern.shape[0] - 3)
            substrate_matrix_upd = np.concatenate(
                (substrate_matrix_upd, np.zeros((n_nodes_to_join, substrate_matrix_upd.shape[1]))), axis=0
            )
            substrate_matrix_upd = np.concatenate(
                (substrate_matrix_upd, np.zeros((substrate_matrix_upd.shape[0], n_nodes_to_join))), axis=1
            )
            shape = substrate_matrix_upd.shape[0]
            # interaction matrix update
            substrate_matrix_upd[
                np.ix_(range(shape-n_nodes_to_join, shape), range(shape-n_nodes_to_join, shape))
            ] = attachment_pattern[np.ix_(range(3, 3+n_nodes_to_join) ,range(3, 3+n_nodes_to_join))]
            substrate_matrix_upd[
                np.ix_(range(shape-n_nodes_to_join, shape), inner_motif)
            ] = attachment_pattern[np.ix_(range(3, 3+n_nodes_to_join) ,range(3))]
            substrate_matrix_upd[
                np.ix_(inner_motif, range(shape-n_nodes_to_join, shape))
            ] = attachment_pattern[np.ix_(range(3), range(3, 3+n_nodes_to_join))]
        else:
            try:
                (i, j), (k, l) = attachment_pattern
                substrate_matrix_upd[i, j] = 1
                substrate_matrix_upd[k, l] = 1
            except TypeError:
                i, j = attachment_pattern
                substrate_matrix_upd[i, j] = 1
    return substrate_matrix_upd

def contatenate_matrices(matrix, edge_list):
    """
    Substrate network update by selected attachment pattern
    ________________________________________________________________________
    matrix - the ffl-component of network we are growing
    edge_list - list with pairs of nodes that are not part of ffl-component // result of node-preferrential attachment model with power-law kernel
    """
    #retrieve node numbers from ffl-component network
    num = matrix.shape[0]
   #print(f"num start: {num}")
    
    #name new nodes that need to be attached
    for inner_list in edge_list:
        for inner_index, element in enumerate(inner_list):
            if element[0] == 'in':
                inner_list[inner_index][0] = num
                num+=1
            elif element[1] == 'out':
                inner_list[inner_index][1] = num
                num+=1
  
    #print(f"num end: {num}")
    edge_list_flatten = f.flatten(edge_list)
    f_list = list(f.flatten(edge_list_flatten))
    
    #print(f"max(f_list): {max(f_list)}")
    #print(f"f_list: {f_list}")

    #create template for resulting network
    final_mat = np.zeros((max(f_list)+1, max(f_list)+1))
    
    #add ffl-nodes
    final_mat[0:matrix.shape[0], 0:matrix.shape[0]] = matrix
    
    #add non-ffl nodes
    for pair in edge_list[0]:
        final_mat[pair[0], pair[1]] = 1
    for pair in edge_list[1]:
        final_mat[pair[1], pair[0]] = 1
    
    return final_mat

def filter_for_loops(substrate_matrix):
    matrix_motifs, motifs_stats = f.motif_search(cfg, substrate_matrix, batch_size=10000)
    loops_edges = 0
    
    #iterate over all 3-node loops in network
    for motif in matrix_motifs['030C']:
        #iterate over all nodes in motif
        nodes = motif.split('_')
        out_degrees = {}
        for node in nodes:
            #calculate out-degree
            out_degrees[node] = substrate_matrix[int(node),:].sum()
        
        #sort by out-degree
        out_degrees = {k: v for k, v in sorted(out_degrees.items(), key=lambda item: item[1])}
        nodes = list(out_degrees.keys())
    
        #delete out-degree for that node
        substrate_matrix[int(nodes[0]),int(nodes[1])] = 0
        substrate_matrix[int(nodes[0]),int(nodes[2])] = 0
    
        #counter
        loops_edges +=1
    return loops_edges, substrate_matrix

# Stack all in the pipeline

def generate_artificial_network(
    interaction_matrix,
    motifs=None, 
    motifs_network=None, 
    nucleus_size=30,
    growth_pace=0.4,
    network_size = 100,
    reference_matrix=None,
    random_seed=cfg["RANDOM_SEED"],
    growth_barabasi=cfg["GROWTH_BARABASI"],
    sparsity=cfg["SPARSITY"],
    shuffled=cfg["SHUFFLED"]
):
    
    """
    Aggragated pipeline of artificial network generation
    ________________________________________________________________________
    interaction_matrix (numpy.array) 
        Binary interaction matrix for genes
    motifs (numpy.array, default=None) 
        List of unique identifiers for condidered motifs (FFL triads). 
        If None motif counting is launched
    motifs_network (numpy.array, default=None) 
        Vertex-based motifs network (linkage by shared nodes)
        If None VMN buiding algorithm is launched
    nucleus_size (int, default=30)
        Minimal required size of initial nucleus. 
        The resulting size may be slightly higher as we may attach two nodes per time.
    network_size (int, default=100)
        Required resulting network size.
        The resulting size may be slightly higher as we may attach two nodes per time.
    random_seed (int, default=19)
        Reproducibility parameter
    growth_barabasi (int, default=0.2)
        Percentage of nodes that are part of FFL-component of resulting network.
    sparsity (int, default=3)
        Average number of links per node in resulting network
    """
    assert (motifs is None) & (motifs_network is None) | (motifs is not None)
    np.random.seed(random_seed)
    init_time = datetime.now()
    
    # check if motifs are provided and search them otherwise 
    if motifs is None:
        print("Motifs are not provided. Motif search is in progress...")
        motifs_orig, counter_orig = f.motif_search(
            cfg, interaction_matrix, batch_size=10000, verbose=False
        )
        motifs = motifs_orig["030T"]
        print()
    
    # check if motifs are provided and search them otherwise 
    if motifs_network is None:
        print("Vertex-based FFL net is not provided. VMN building is in progress...")
        motifs_network = f.build_vmn(motifs, verbose=False)
        print()
    
    # nucleus subsampling
    substrate_matrix = get_network_nucleus(
        interaction_matrix, motifs, motifs_network, min_size=nucleus_size
    )
    print(f"Nucleus matrix shape: {substrate_matrix.shape}")
    network_params = get_network_params(substrate_matrix, verbose=False)
    print()
    if reference_matrix is not None:
        #print("Reference matrix params")
        fix_network_params = get_network_params(reference_matrix, verbose=False)
        print()
    else:
        fix_network_params = None
    sleep(2)
    
    # preferencial attachment start
    substrate_size = substrate_matrix.shape[0]
    i = 0
    edges = 0
    barabasi_entering = 0

    N_CORES = mp.cpu_count() if cfg["N_CORES_TO_USE"] == -1 else cfg["N_CORES_TO_USE"]
    while substrate_matrix.shape[0]+edges < network_size:
        
     #   print(f"substrate_matrix.shape[0]: {substrate_matrix.shape[0]}")
      #  print(i)
        # Importing the library
        i += 1
        #print(network_params.substrate_motifs)
        
        #decide between node and motif preferrential attachment
        
        #p1 = np.random.uniform()
        #growth_barabasi = (0.1/growth_barabasi) - (nucleus_size/network_size)
        #print(f"growth_barabasi: {growth_barabasi}")
        ffl_desired = growth_barabasi
        #print(f"ffl_desired: {ffl_desired}")

        ffl_perc = ((ffl_desired*network_size)-nucleus_size)/(network_size-nucleus_size)
        if ffl_perc <= 0:
            ffl_perc = 0
        else:
            ffl_perc = ((ffl_desired*network_size)-nucleus_size)/(network_size-nucleus_size)
            scaling_factor = network_size/(nucleus_size*0.75)
            ffl_perc =(scaling_factor*ffl_perc)/((scaling_factor*ffl_perc) + (1-ffl_perc))
        #    print(f"ffl_perc: {ffl_perc}")

        your_choise = np.random.choice(['ffl', 'barabasi'], p=[ffl_perc,1-ffl_perc])
        if your_choise == 'barabasi':
            barabasi_entering += 1
            #print(f"edges: {edges}")
            try:
                out_edge_list, edge_counter = update_edge_list(substrate_matrix, edge_list)
                
            except NameError:
                out_edge_list, edge_counter = update_edge_list(substrate_matrix)
            edge_list = out_edge_list
            edges = edge_counter
        
        else:
            # Calling psutil.cpu_precent() for 2 seconds
            #print('The CPU usage is: ', psutil.cpu_percent(2))
            network_params = get_network_params(substrate_matrix, verbose=False, motif_search=False,
                                                known_motifs = network_params.substrate_motifs)
            if fix_network_params is not None: 
                Params = namedtuple(
                    "Params", "substrate_motifs tf_nodes tg_nodes mtype_probs unique_nodes unique_edges_1 unique_edges_0"
                )
                network_params = Params(
                    *[network_params.substrate_motifs,
                      network_params.tf_nodes,
                      network_params.tg_nodes,
                      fix_network_params.mtype_probs,
                      fix_network_params.unique_nodes,
                      fix_network_params.unique_edges_1,
                      fix_network_params.unique_edges_0]
                )
            params = get_attachment_params(substrate_matrix, network_params, growth_pace=growth_pace)
    #         print(params[-5:])
            if params.n_unique_nodes == 1:
                #print('1 node')
                attachment_pattern = get_attachment_1n2e(substrate_matrix, params)
    
                #             n_edges_to_join = 1
                if attachment_pattern is not None:
                    motif_to_add = attachment_pattern[1]
                    #print(motif_to_add)
                    attachment_pattern = attachment_pattern[0]
    
                    #print(network_params.substrate_motifs)
                    network_params.substrate_motifs.append(motif_to_add)        
                
            elif params.n_unique_nodes == 0 and params.n_unique_edges == 1:
                #print('0 node, 1 edges')
                attachment_pattern = get_attachment_0n1e(substrate_matrix, params)
                
                #update motifs
                if attachment_pattern is not None:
                    motif_to_add = attachment_pattern[1]
                    #print(motif_to_add)
                    attachment_pattern = attachment_pattern[0]
    
                    #print(network_params.substrate_motifs)
                    network_params.substrate_motifs.append(motif_to_add)
                
    #             n_edges_to_join = 2
            elif params.n_unique_nodes == 0 and params.n_unique_edges == 2:
                #print('0 nodes, 2 edges')
                attachment_pattern = get_attachment_0n2e(substrate_matrix, params)
                
                #update motifs
                if attachment_pattern is not None:
                    motif_to_add = attachment_pattern[1]
                    #print(motif_to_add)
                    attachment_pattern = attachment_pattern[0]
    
                    #print(network_params.substrate_motifs)
                    network_params.substrate_motifs.append(motif_to_add)
                    
    #             n_edges_to_join = 2
            inner_motif = f.split_motif(params.substrate_motifs[params.inner_motif_1_idx])
            nodes_attach = params.n_unique_nodes == 1
            substrate_matrix = update_substrate_matrix(
            substrate_matrix, attachment_pattern, inner_motif, nodes_attach
            )
            if nodes_attach and attachment_pattern is not None:
                substrate_size = substrate_matrix.shape[0]
                n_nodes_to_join = int(attachment_pattern.shape[0] - 3)
        #print(f"step: {i}\tnodes: {substrate_matrix.shape[0]}\tedeges: {substrate_matrix.sum()}")
    sleep(2)
    print()
    
    
    #print(edge_list)
    #print(substrate_matrix)
    nodes_in_ffl = substrate_matrix.shape[0]
    
    if edges > 0:
    	substrate_matrix = contatenate_matrices(substrate_matrix, edge_list)
    links_per_node = substrate_matrix.sum()/substrate_matrix.shape[0]
    #print(links_per_node)
    #print(substrate_matrix.shape)
    
    #check for sparsity
    loop_edges, substrate_matrix = filter_for_loops(substrate_matrix)
    compensated_edges = 0
    
    #check if loops were deleted
    #matrix_motifs, motifs_stats = f.motif_search(cfg, substrate_matrix, batch_size=10000)
    #print(motifs_stats)
    #print(matrix_motifs['030C'])

    while links_per_node < sparsity or compensated_edges < loop_edges:
        
        #matrix_motifs, motifs_stats = f.motif_search(cfg, substrate_matrix, batch_size=10000)
        #print(motifs_stats)
        #print(matrix_motifs['030C'])
        #print(substrate_matrix)
        substrate_matrix_upd = deepcopy(substrate_matrix)
        
        #calculate in, out degree
        out_degree = substrate_matrix.sum(axis=1)
        in_degree = substrate_matrix.sum(axis=0)
        nodes = list(range(0, len(substrate_matrix)))
    
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
        substrate_matrix_upd[regulator,regulatee] = 1
        
        #loop_edges, substrate_no_loops = filter_for_loops(substrate_matrix_upd)
        #print(f"loop_edges while sparsity: {loop_edges}")
        #if loop_edges == 0:
        graph_nx = nx.DiGraph(substrate_matrix_upd.T)
        link_to_attach = [regulator,regulatee]
        neighbours = []
        
        for node in graph_nx.nodes:
            if node in link_to_attach:
            #print('node: '+str(node))
                neighbours.append(list(nx.all_neighbors(graph_nx, node)))
        if len(neighbours) > 1:
         #   print(neighbours)
            common_node = list(set(neighbours[0]).intersection(set(neighbours[1])))
        #print(f"common_node: {common_node}")
        
        #check if 3-node loop is not created
        if len(common_node) > 0:
            safe_to_add = True
            for node in common_node:
                if substrate_matrix[regulatee,node] == 1 and substrate_matrix[node,regulator] == 1:
                    safe_to_add = False
                    
            if safe_to_add:
                #print(f"no loop created, your pair (and their neighbour): {link_to_attach, node}")
                compensated_edges += 1
                substrate_matrix = substrate_matrix_upd
                    
            else:
                #print(f"trouble! troubled pair: {link_to_attach}")
                substrate_matrix_upd = substrate_matrix
        else:
            #print(f"no common node, your pair: {link_to_attach}")
            compensated_edges += 1
            substrate_matrix = substrate_matrix_upd
            
        links_per_node = substrate_matrix.sum()/substrate_matrix.shape[0]
        #
        

    #return shuffled matrix
    if shuffled:
        complete = False
        swaps = (substrate_matrix.sum())*0.2
        while not complete:
            shuffled_matrix = f.get_shuffled_matrix(substrate_matrix, swaps)
            shiffled_score = 1-f.corruption_score(substrate_matrix, shuffled_matrix)
            print(shiffled_score)
            swaps += (substrate_matrix.sum())*0.2
            if shiffled_score >= 0.77:
                complete = True
                
        substrate_matrix = shuffled_matrix
                
    ffl_perc = nodes_in_ffl/substrate_matrix.shape[0]
    
    total_time_spent = str(f"{datetime.now() - init_time}")
    #calculate loops once again
    matrix_motifs, motifs_stats = f.motif_search(cfg, substrate_matrix, batch_size=10000)
    print(motifs_stats)
    print(matrix_motifs['030C'])
    #print(f"loop_edges: {loop_edges}")
    #print(f"num of barabasi entering: {barabasi_entering}")
    #print(f"num of edges: {edges}")
    print(f"growth_barabasi: {growth_barabasi}")
    print(f"ffl_perc: {ffl_perc}")
    print(f"links_per_node: {links_per_node}")
    print(f"Network has been successfully generated!\nTotal time spent: {datetime.now() - init_time}")
    
    return substrate_matrix, total_time_spent, links_per_node, ffl_perc
