#!/usr/bin/env python
# -*- coding: UTF-8 -*
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import logging
import networkx as nx
import numpy as np
import scipy.sparse as sp
import sys
from graph import Graph
import random

def read_graph(ctrl, file_path, directed=False):
    in_file = open(file_path)
    neigh_dict = defaultdict(list)
    edge_num = 0
    for line in in_file:
        eles = line.strip().split()
        n0, n1 = [int(ele) for ele in eles[:2]]
        if len(eles) == 3: # weighted graph
            wgt = float(eles[2])
            neigh_dict[n0].append((n1, wgt))
            if diercted ==False and n0 != n1:            
                neigh_dict[n1].append((n0, wgt))
        else:
            neigh_dict[n0].append(n1)
            if directed ==False and n0 != n1:
                neigh_dict[n1].append(n0)
        if directed ==False and n0 != n1:
            edge_num += 2
        else:
            edge_num += 1
    in_file.close()
    weighted = (len(eles) == 3)

    node_num = len(neigh_dict)
    graph = Graph(node_num, edge_num, weighted)
    edge_cnt = 0
    graph.adj_idx[0] = 0
    for idx in range(node_num):
        graph.node_wgt[idx] = 1 # default weight to nodes
        for neigh in neigh_dict[idx]:
            if graph.weighted:
                graph.adj_list[edge_cnt] = neigh[0]
                graph.adj_wgt[edge_cnt] = neigh[1]
            else:
                graph.adj_list[edge_cnt] = neigh
                graph.adj_wgt[edge_cnt] = 1.0
            edge_cnt += 1
        graph.adj_idx[idx+1] = edge_cnt
    print(" graph.adj_idx",len( graph.adj_idx))
    print(" graph.adj_wgt",len( graph.adj_wgt))
    if ctrl.debug_mode:
       assert nx.is_connected(graph2nx(graph)), "Only single connected component is allowed for embedding."
 
    graph.A = graph_to_adj(graph, self_loop=False)
    graph.G= graph2nx(graph)
    return graph

def loadDataSet(filename):
    fr=open(filename)
    stringArr=[]
    line=fr.readline()
    while line:
          items = line.strip().split(' ')
          stringArr.append(items[1:])
          line=fr.readline()
    datArr=[list(map(float,line))for line in stringArr]
    return np.array(datArr)

def graph2nx(graph): # mostly for debugging purpose. weights ignored.
    G=nx.DiGraph()
    for idx in range(graph.node_num):
        for neigh_idx in range(graph.adj_idx[idx], graph.adj_idx[idx+1]):
            neigh = graph.adj_list[neigh_idx]
            wgt= graph.adj_wgt[neigh_idx]
         
            if graph.weighted:
                G.add_edge(idx, neigh)
               
                G[idx][neigh]['weight'] = wgt
            else:
                G.add_edge(idx, neigh)
              
                G[idx][neigh]['weight'] = 1.0
  
    return G

def graph_to_adj(graph, self_loop=False):
    
    node_num = graph.node_num
    i_arr = []
    j_arr = []
    data_arr = []
    for i in range(0, node_num):
        for neigh_idx in range(graph.adj_idx[i], graph.adj_idx[i+1]):
            i_arr.append(i)
            j_arr.append(graph.adj_list[neigh_idx])
            data_arr.append(graph.adj_wgt[neigh_idx])
    adj = sp.csr_matrix((data_arr, (i_arr, j_arr)), shape=(node_num, node_num), dtype=np.float32)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    return adj


def cmap2C(cmap): # fine_graph to coarse_graph, matrix format of cmap: C: n x m, n>m.
    node_num = len(cmap)
    i_arr = []
    j_arr = []
    data_arr = []
    for i in range(node_num):
        i_arr.append(i)
        j_arr.append(cmap[i])
        data_arr.append(1)
    return sp.csr_matrix((data_arr, (i_arr, j_arr)))        

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(screen_handler)
    return logger

def normalized(embeddings, per_feature=True):
    if per_feature:
        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        scaler.fit(embeddings)
        return scaler.transform(embeddings)
    else:
        return normalize(embeddings, norm='l2')
