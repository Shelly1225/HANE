#!/usr/bin/env python
# -*- coding: UTF-8 -*
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from lv import louvainModularityOptimization
from defs import Control
from embed import multilevel_embed
from refine_model import GCN, GraphSage
from utils import read_graph, setup_custom_logger,loadDataSet,normalized
import importlib
import logging
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from classification import node_classification_F1, read_label
import time
def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--data', required=False, default='/citeseer',
                        help='Input graph file')
    parser.add_argument('--format', required=False, default='edgelist', choices=['metis', 'edgelist'],
                        help='Format of the input graph file (metis/edgelist)')
    parser.add_argument('--no-eval', action='store_true',
                        help='Evaluate the embeddings.')
    parser.add_argument('--embed-dim', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--basic-embed', required=False, default='deepwalk',
                        choices=['deepwalk', 'grarep', 'netmf','stne'],
                        help='The basic embedding method. If you added a new embedding method, please add its name to choices')
    parser.add_argument('--refine-type', required=False, default='MD-gcn',
                        choices=['MD-gcn', 'MD-dumb', 'MD-gs'],
                        help='The method for refining embeddings.')
    parser.add_argument('--coarsen-level', default=2, type=int,
                        help='MAX number of levels of coarsening.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of workers.')
    parser.add_argument('--double-base', action='store_true',
                        help='Use double base for training')
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help='Learning rate of the refinement model')
    parser.add_argument('--self-weight', default=0.05, type=float,
                        help='Self-loop weight for GCN model.')  # usually in the range [0, 1]
    parser.add_argument('--directed',
                        default=False,
                        action='store_true',
                        help='treat graph as directed')
    args = parser.parse_args()
    return args


def set_control_params(ctrl, args, graph):
    ctrl.refine_model.double_base = args.double_base
    ctrl.refine_model.learning_rate = args.learning_rate
    ctrl.refine_model.self_weight = args.self_weight

    ctrl.coarsen_level = args.coarsen_level
    ctrl.coarsen_to = max(1, graph.node_num // (2 ** ctrl.coarsen_level))  # rough estimation.
    ctrl.embed_dim = args.embed_dim
    ctrl.basic_embed = args.basic_embed
    ctrl.refine_type = args.refine_type
    ctrl.data = args.data
    ctrl.workers = args.workers
    ctrl.max_node_wgt = int((5.0 * graph.node_num) / ctrl.coarsen_to)
    ctrl.logger = setup_custom_logger('HANE')

    if ctrl.debug_mode:
        ctrl.logger.setLevel(logging.DEBUG)
    else:
        ctrl.logger.setLevel(logging.INFO)
    ctrl.logger.info(args)


def read_data(ctrl, args):
    prefix = "./dataset" + args.data + args.data
    input_graph_path = prefix + ".edgelist"
    input_attr_path = prefix + ".features"
    input_label_path= prefix + ".label"
    label=read_label(input_label_path)
    ctrl.k=len(set(label))
    dataMat=loadDataSet(input_attr_path)
    #dataMat=np.load(input_attr_path)
    #dataMat = normalized(dataMat, per_feature=False) # for Flickr, BlogCatalog
    pca = PCA(n_components=ctrl.embed_dim)
    pca.fit(dataMat)
    lowDAttrMat=pca.fit_transform(dataMat)
    graph= read_graph(ctrl, input_graph_path, directed=args.directed)   #utils.py
    
    return input_graph_path, graph, lowDAttrMat,label


def select_base_embed(ctrl):
    mod_path = "base_embed_methods." + ctrl.basic_embed
    embed_mod = importlib.import_module(mod_path)
    embed_func = getattr(embed_mod, ctrl.basic_embed)
    return embed_func


def select_refine_model(ctrl):
    refine_model = None
    if ctrl.refine_type == 'MD-gcn':
        refine_model = GCN
    elif ctrl.refine_type == 'MD-gs':
        refine_model = GraphSage
    elif ctrl.refine_type == 'MD-dumb':
        refine_model = GCN
        ctrl.refine_model.untrained_model = True
    return refine_model


if __name__ == "__main__":
    seed = 2019
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    ctrl = Control()
    args = parse_args()
   
    # Read input graph
    input_graph_path, graph, lowDAttrMat, y= read_data(ctrl, args)
    print("read ok")
    set_control_params(ctrl, args, graph)
    print("set ctrl ok")
   
    # Coarsen method
    match_method = louvainModularityOptimization

    # Base embedding
    basic_embed = select_base_embed(ctrl)

    # Refinement model
    refine_model = select_refine_model(ctrl)

    # Generate embeddings
    start = time.time()
    embeddings = multilevel_embed(ctrl, graph, match_method=match_method, basic_embed=basic_embed,
                                  refine_model=refine_model,AttrMat=lowDAttrMat)
    
    end = time.time()
    print("times:", end-start)
    embeddings=np.concatenate((embeddings,lowDAttrMat),axis=1)
    pca = PCA(n_components=ctrl.embed_dim)
    pca.fit(embeddings)
    embeddings=pca.fit_transform(embeddings)
    #Evaluate embeddings
    print("type(embeddings)",type(embeddings))
    if not args.no_eval:
        
        for test_rio in [0.9,0.8,0.75,0.7,0.6,0.5,0.4,0.3,0.2,0.1]:
            print("train_rio",1-test_rio)
            node_classification_F1(embeddings, y, test_rio)
