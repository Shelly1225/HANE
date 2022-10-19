# -*- coding: UTF-8 -*
from collections import defaultdict
from graph import Graph
import numpy as np
from utils import cmap2C ,graph2nx
import networkx as nx

def create_coarse_graph(stru_comms,attr_comms,attrmat,graph):
        group=0
        in_comm={}
        c_set=[]
       
        for stru_group in stru_comms.keys():
            s_set=set(stru_comms[stru_group])  
            for attr_group in range(len(attr_comms)): 
                a_set=set(attr_comms[attr_group])
                c_set= list(s_set.intersection(a_set))
                if len(c_set)>1: 
                   in_comm[group]=c_set
                   s_set=s_set.difference(in_comm[group]) 
                   group +=1
            #print("s_set:",list(s_set))
            if len(list(s_set))>0:
               in_comm[group]=list(s_set) 
               group += 1

        c_mat=[]
        for c_node in in_comm.keys():
            c1_mat=[]
            c3_mat=None
            for ch_node in in_comm[c_node]:
                c1_mat.append(attrmat[ch_node])   
            c2_mat=np.array(c1_mat)
            c3_mat=np.mean(c2_mat,axis=0)
            c_mat.append(c3_mat)
        Attrmat=np.array(c_mat)    
      
        NewGraph=create_NewGraph(in_comm,graph)
    
        return Attrmat,NewGraph,in_comm
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def create_NewGraph(in_comm, graph):
    
    cmap = graph.cmap   
    coarse_graph_size = 0                
    for inc_idx in in_comm.keys():
        for ele in in_comm[inc_idx]:
            cmap[ele] = coarse_graph_size   
        coarse_graph_size += 1 
    newGraph=Graph(coarse_graph_size, graph.edge_num, weighted=True)
    newGraph.finer = graph
    graph.coarser = newGraph
    
    adj_list = graph.adj_list
    adj_idx = graph.adj_idx
    adj_wgt = graph.adj_wgt
    node_wgt = graph.node_wgt
  
    coarse_adj_list = newGraph.adj_list
    
    coarse_adj_idx = newGraph.adj_idx
    coarse_adj_wgt = newGraph.adj_wgt
    coarse_node_wgt = newGraph.node_wgt
    coarse_degree = newGraph.degree
    coarse_adj_idx[0] = 0
    nedges = 0  # number of edges in the coarse graph
    idx=0

    for idx in range(len(in_comm)):  # idx in the graph
        coarse_node_idx = idx  
        neigh_dict = dict()  # coarser graph neighbor node --> its location idx in adj_list. 
        group = in_comm[idx]
        for i in range(len(group)):
            merged_node = group[i]
            if (i == 0):
                coarse_node_wgt[coarse_node_idx] = node_wgt[merged_node]
            else:
                coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]
            
            istart = adj_idx[merged_node]
            iend = adj_idx[merged_node + 1]   
            for j in range(istart, iend):
               
                k = cmap[adj_list[j]]  # adj_list[j] is the neigh of v; k is the new mapped id of adj_list[j] in coarse graph.
                if k not in neigh_dict:  # add new neigh
                    coarse_adj_list[nedges] = k
                    coarse_adj_wgt[nedges] = adj_wgt[j]
                    neigh_dict[k] = nedges
                    nedges += 1
                else:  # increase weight to the existing neigh
                    coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]
                # add weights to the degree. For now, we retain the loop. 

                coarse_degree[coarse_node_idx] += adj_wgt[j]
        
        coarse_node_idx += 1
        coarse_adj_idx[coarse_node_idx] = nedges
    
    
    newGraph.edge_num = nedges
    newGraph.G= graph2nx(newGraph)
    
    newGraph.resize_adj(nedges)
    #newGraph.G=newG
    C = cmap2C(cmap)  # construct the matching matrix.
    graph.C = C
    newGraph.A = C.transpose().dot(graph.A).dot(C)
    return newGraph
