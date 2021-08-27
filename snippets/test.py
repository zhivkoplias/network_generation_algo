#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import sys
import csv
sys.path.insert(0, "../src")
ART_NET_PATH = "../networks"

import auxilary_functions as f
from generation_algorithm import *
import joblib
import networkx as nx
from time import sleep
import statistics
from argparse import ArgumentParser


def load_ffl_based_component():
    interaction_matrix = f.get_interaction_matrix(cfg)
    motifs, counter = f.motif_search(cfg, interaction_matrix, batch_size=10000)
    motifs_orig = motifs["030T"]

    ffl_nodes = list(set(sum([list(map(int, x.split("_"))) for x in motifs_orig], [])))
    interaction_matrix_ffl = np.zeros((len(ffl_nodes), len(ffl_nodes)))
    for motif in motifs_orig:
        motif = f.split_motif(motif)
        motif_new = list(ffl_nodes.index(x) for x in motif)
        interaction_matrix_ffl[np.ix_(motif_new, motif_new)] = \
        interaction_matrix[np.ix_(motif, motif)]
    interaction_matrix_ffl.shape, interaction_matrix_ffl.sum()

    # Vertex-based motif network on FFL
    motifs_network = f.build_vmn(motifs_orig, verbose=True)
    V = nx.Graph(motifs_network)
    nx.is_connected(V)
    return interaction_matrix, motifs_orig, motifs_network, interaction_matrix_ffl

def main(args):
    artificial_matrix_ffl = generate_artificial_network(
                    yeast_matrix, motifs=ffl_motif, motifs_network=ffl_component,
                    nucleus_size=args.initial_size, network_size=args.final_size, growth_pace=args.growth_pace,
                    reference_matrix=ffl_matrix)
    
    ffl_perc = artificial_matrix_ffl[3]
    artificial_matrix_ffl = artificial_matrix_ffl[0]

    #GS-to-NetworkX format conversion
    artificial_matrix_ffl = artificial_matrix_ffl.transpose()

    if not os.path.exists(ART_NET_PATH):
       os.mkdir(ART_NET_PATH)
    
    network_name = '_'.join(str(x) for x in ['nodes',args.final_size,'core_size',args.initial_size,'growth',args.growth_pace,'ffl_perc',ffl_perc])
    joblib.dump(artificial_matrix_ffl, os.path.join(ART_NET_PATH,network_name+".gz"))

    return

if __name__ == '__main__':
    yeast_matrix, ffl_motif, ffl_component, ffl_matrix = load_ffl_based_component()
    parser = ArgumentParser()
    parser.add_argument("initial_size", type=int,
                            help="Number of nodex in nucleus")
    parser.add_argument("final_size", type=int,
                            help="Number of nodes in final network")
    parser.add_argument("growth_pace", type=float,
                            help="Probability to add one node over zero nodes")
    args = parser.parse_args()
    main(args)

#Command to execute the script:

#python3 test.py 30 100 0.4 

