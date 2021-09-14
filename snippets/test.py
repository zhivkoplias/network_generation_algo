#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import sys
import csv
sys.path.insert(0, "../src")
ART_NET_PATH = "../networks"

import auxilary_functions as ff
from generation_algorithm import *
import joblib
import networkx as nx
from time import sleep
import statistics
from argparse import ArgumentParser

cfg = f.get_actual_parametrization("../src/config.json")

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
    for rep in range(args.num_networks):
        yeast_matrix, ffl_motif, ffl_component, ffl_matrix = load_ffl_based_component()
        growth_rate = np.random.randint(1,6)*0.1
        growth_rate = 0.9
        core_size = np.random.randint(20,30)
        artificial_matrix_ffl = generate_artificial_network(
                        yeast_matrix, motifs=ffl_motif, motifs_network=ffl_component,
                        reference_matrix=ffl_matrix, growth_pace=growth_rate,
                        network_size=args.final_size, nucleus_size=core_size,
                        growth_barabasi=args.ffl_perc)
    
        #save output // if full stats
        #artificial_matrix_ffl_list = artificial_matrix_ffl[1]
        #artificial_matrix_ffl = artificial_matrix_ffl[0]

        #GS-to-NetworkX format conversion
        #artificial_matrix_ffl = artificial_matrix_ffl.transpose()
        #joblib.dump(artificial_matrix_ffl, os.path.join(ART_NET_PATH,network_name+".gz"))

        if not os.path.exists(ART_NET_PATH):
            os.mkdir(ART_NET_PATH)

        if not os.path.exists(args.out_dir):
           os.mkdir(args.out_dir)
    
    
        #save output // if adj list
        network_name = '_'.join(str(x) for x in ['fflatt_transcriptional_network',rep,'nodes',args.final_size,'ffl_perc',args.ffl_perc])
        with open(args.out_dir+'/'+network_name+'.tsv', "w", newline="") as f:
    	    writer = csv.writer(f, delimiter ='\t')
    	    writer.writerows(artificial_matrix_ffl)
        #print(ff.collect_topological_parameters(cfg, artificial_matrix_ffl, 'whatever'))
        #print('/n')
        print(ff.analyze_exctracted_network(cfg, args.out_dir+'/'+network_name+'.tsv', 'whatever', rep, args.final_size))

    return

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("final_size", type=int,
                            help="Number of nodes in final network")
    parser.add_argument("ffl_perc", type=float,
                            help="Percentage of FFL-nodes")
    parser.add_argument("num_networks", type=int, nargs='?', const=1, default='1',
                            help="Number of networks to generate")
    parser.add_argument("out_dir", type=str, nargs='?', const=1, default='.',
                            help="Output directory")
    args = parser.parse_args()
    main(args)


#Command to execute the script:
#python3 test.py 1000 0.13 1 test_networks/
#python3 test.py 1565 0.3 1 test_networks/
