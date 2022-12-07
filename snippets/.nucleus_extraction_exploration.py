#!/usr/bin/env python

import os
import numpy as np
import sys
import csv
sys.path.insert(0, "../src")
ART_NET_PATH = "../networks"

import auxilary_functions as functions
from generation_algorithm import *
import networkx as nx
from argparse import ArgumentParser
import json

def load_ffl_based_component():
    with open(args.config_file, 'r') as j:
        config_file = json.loads(j.read())
            
        interaction_matrix = functions.get_interaction_matrix(config_file)
        motifs, counter = functions.motif_search(config_file, interaction_matrix, batch_size=10000)
        motifs_orig = motifs["030T"]

        ffl_nodes = list(set(sum([list(map(int, x.split("_"))) for x in motifs_orig], [])))
        interaction_matrix_ffl = np.zeros((len(ffl_nodes), len(ffl_nodes)))
        for motif in motifs_orig:
            motif = functions.split_motif(motif)
            motif_new = list(ffl_nodes.index(x) for x in motif)
            interaction_matrix_ffl[np.ix_(motif_new, motif_new)] = \
            interaction_matrix[np.ix_(motif, motif)]
        interaction_matrix_ffl.shape, interaction_matrix_ffl.sum()

        # Vertex-based motif network on FFL
        motifs_network = functions.build_vmn(motifs_orig, verbose=True)
        V = nx.Graph(motifs_network)
        nx.is_connected(V)
        return interaction_matrix, motifs_orig, motifs_network, interaction_matrix_ffl

def main(args):
    for rep in range(args.num_networks):
        yeast_matrix, ffl_motif, ffl_component, ffl_matrix = load_ffl_based_component()
        #p2_rate = 0.5 #p2
        #p4_rate = 0.9 #p4
        #num_of_cascades = 3 #kN nb! save and store the number of the actual motif changes
        core_size = 25
        with open(args.config_file, 'r') as j:
            config_file = json.loads(j.read())
            artificial_matrix_ffl = generate_artificial_network(
                        yeast_matrix, config_file, random_seed = np.random.randint(1,100),
                        motifs=ffl_motif, motifs_network=ffl_component,
                        reference_matrix=ffl_matrix, p2_parameter=args.p2_rate,
                        p4_parameter=args.p4_rate, network_size=args.final_size,
                        nucleus_size=core_size, cascade_transformation_num=args.num_of_cascades)

            if not os.path.exists(ART_NET_PATH):
                os.mkdir(ART_NET_PATH)

            if not os.path.exists(args.out_dir):
                os.mkdir(args.out_dir)
    
    
            #save output // if adj list
            network_name = '_'.join(str(x) for x in ['fflatt_transcriptional_network',rep,'nodes',args.final_size])
            with open(args.out_dir+'/'+network_name+'.tsv', "w", newline="") as f:
    	        writer = csv.writer(f, delimiter ='\t')
    	        writer.writerows(artificial_matrix_ffl)
                
    return

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("config_file", type=str, nargs='?', const=1, default='.',
                            help="Config file")
    parser.add_argument("final_size", type=int,
                            help="Number of nodes in final network")
    parser.add_argument("num_networks", type=int, nargs='?', const=1, default='1',
                            help="Number of networks to generate")
    parser.add_argument("out_dir", type=str, nargs='?', const=1, default='.',
                            help="Output directory")
    parser.add_argument("p2_rate", type=str,
                            help="p2 probability")
    parser.add_argument("p4_rate", type=str,
                            help="p4 probability")
    parser.add_argument("num_of_cascades", type=str,
                            help="Number cascades to delete")
    args = parser.parse_args()
    main(args)


#Command example to execute the script:
#python3 test.py "../src/config.json" 1500 0.3 1 test_networks/
