# -*- coding: UTF-8 -*
import numpy as np
import networkx as nx
from graph import *
import random
import time
import math
import numpy
import scipy
import networkx as nx
import matplotlib
import pylab
from collections import defaultdict
import matplotlib.pyplot as plt
from utils import *
###############################################
# This section of code adapted from fuguoji/HSRL#
# https://github.com/fuguoji/HSRL #
###############################################

class Community:
    """ Data structure to hold community information and calculate modularity """
    def __init__(self, name, intraCommWeights, allWeights, nodeDict):
        self.name = name
        self.intraCommWeights = intraCommWeights
        self.allWeights = allWeights
        self.nodeDict = nodeDict
    def getModularity(self, graphWeight, graphWeight_x2):
        self.modularity = (self.intraCommWeights/graphWeight) - \
                          ((self.allWeights/graphWeight_x2)**2)

class PreCompute:
    """ Data structure to hold network information.
        Drastically reduces compute time by holding information that would otherwise
        need to be looked up over and over again..."""
    def __init__(self, nDegrees, nNeighbs, nComms, nSelfW, edgeDict, tW, tW_x2):
        self.nDegrees = nDegrees # {nodeID:degreeOfNode}
        self.nNeighbs = nNeighbs # {nodeID:set(nodeNeighborIDs)}
        self.nComms = nComms     # {nodeID:communityID_nodeBelongsTo}
        self.nSelfW = nSelfW     # {nodeID:selfLoopWeightOfNode}
        self.edgeDict = edgeDict # {(node1,node2):weight} {(node2,node1):weight}
        self.tW = tW             # totalWeight of network
        self.tW_x2 = tW_x2       # (totalWeight of network) x 2

def randomizeOrder(rand, items):
    if rand == True:
        randomOrder = list(items)
        random.shuffle(randomOrder)
        return iter(randomOrder)
    else:
        return items

def initializeCommunities(graph, weight='weight'):
    ### communityDict = {communityName:communityObject}
    ### preComputeObj = PreCompute Object that stores information so we only have to compute it once
    preComputeObj = PreCompute({}, {}, {}, {}, {}, 0.0, 0.0)
    communityDict = {}
    for node in graph.nodes():
        selfLoop = 0.0
        ### This could be replaced by a try statement. Would speed up for large networks ###
        if graph.has_edge(node, node):
            selfLoop = graph[node][node][weight]
        else:
            graph.add_edge(node,node, weight=selfLoop)
        nDegrees = graph.degree(node, weight=weight)
        preComputeObj.nDegrees.update({node:nDegrees})
        preComputeObj.nNeighbs.update({node:set(graph.neighbors(node))})
        preComputeObj.nComms.update({node:node})
        preComputeObj.nSelfW.update({node:selfLoop})
        communityDict.update({node:Community(node,selfLoop, nDegrees, {node:''})})
    
    for edge in graph.edges():
        edgeWeight = graph[edge[0]][edge[1]][weight]
        preComputeObj.edgeDict.update({edge:edgeWeight})
        preComputeObj.edgeDict.update({(edge[1], edge[0]):edgeWeight})
    
    preComputeObj.tW = graph.size(weight=weight)
    preComputeObj.tW_x2 = preComputeObj.tW * 2.0
    print('Initial communities found %d' % (len(communityDict)))

    return communityDict, preComputeObj

def initializeCommunities_priorComms(graph, levs, commKey='chrom', weight='weight'):
    ### communityDict = {communityName:communityObject}
    ### preComputeObj = PreCompute Object that stores information so we only have to compute it once
    preComputeObj = PreCompute({},{},{},{},{},0.0,0.0)
    communityDict = {}
    for node in graph.nodes():
        selfLoop = 0.0
        ### This could be replaced by a try statement. Would speed up for large networks ###
        if graph.has_edge(node, node):
            selfLoop = graph[node][node][weight]
        else:
            graph.add_edge(node,node, weight=selfLoop)
        nDegrees = graph.degree(node, weight=weight)
        priorComm = graph.node[node][commKey]
        if priorComm not in communityDict:
            communityDict.update({priorComm:Community(priorComm, selfLoop, nDegrees, {node:''})})
        else:
            comObj = communityDict[priorComm]
            comObj.intraCommWeights += selfLoop
            for comNode in comObj.nodeDict:
                if graph.has_edge(comNode, node):
                    comObj.intraCommWeights += graph[node][comNode][weight]
            
            comObj.allWeights += nDegrees
            comObj.nodeDict.update({node:''})

        preComputeObj.nDegrees.update({node:nDegrees})
        preComputeObj.nNeighbs.update({node:set(graph.neighbors(node))})
        preComputeObj.nComms.update({node:priorComm})
        preComputeObj.nSelfW.update({node:selfLoop})
    
    preComputeObj.tW = graph.size(weight=weight)
    preComputeObj.tW_x2 = preComputeObj.tW * 2.0
    nGraph = phaseTwo(graph, communityDict, preComputeObj, weight=weight)
    levs.append(preComputeObj.nComms.copy())
    communityDict, preComputeObj = initializeCommunities(nGraph, weight=weight)

    return nGraph, communityDict, preComputeObj, levs

def computeModularity(communities, pObj):
    mod = 0.0
    for comName,comObj in communities.items():
        if len(comObj.nodeDict) >= 1:
            comObj.getModularity(pObj.tW, pObj.tW_x2)
            mod += comObj.modularity
    ################################
    return mod

def calculateCost(comObj, node, nodeNeighbors, nodeSelfWeight, nodeDegree, eDict, w, w_x2, weight='weight'):
    intraEdges = [ eDict[(node,nei)] for nei in nodeNeighbors & set(comObj.nodeDict) - set([node]) ]
    intraEdgeWeights = float(sum(intraEdges))
    
    newIntraCommWeights = comObj.intraCommWeights - intraEdgeWeights - nodeSelfWeight
    newAllWeights = comObj.allWeights - nodeDegree
    newModScore = (newIntraCommWeights/w) - ((newAllWeights/w_x2)**2)

    return (newModScore-comObj.modularity), newIntraCommWeights, newAllWeights, newModScore

def calculateGain(comObj, node, nodeNeighbors, nodeSelfWeight, nodeDegree, eDict, w, w_x2, weight='weight'):
    newEdges = [ eDict[(node,nei)] for nei in nodeNeighbors & set(comObj.nodeDict) - set([node]) ]
    newEdgeWeights = float(sum(newEdges))

    newIntraCommWeights = comObj.intraCommWeights + newEdgeWeights + nodeSelfWeight
    newAllWeights = comObj.allWeights + nodeDegree
    newModScore = (newIntraCommWeights/w) - ((newAllWeights/w_x2)**2)
    
    return (newModScore-comObj.modularity), newIntraCommWeights, newAllWeights, newModScore
    
def phaseOne(graph, communityDict, pObj, random=False, weight='weight'):
    #endLoopMinimum = .0000001
    endLoopMinimum = 1.0
    modScores = []
    currentMod = computeModularity(communityDict, pObj)
    modScores.append(currentMod)
    
    while True:
        endLoopMod, newMod = currentMod, currentMod
        count = 0
        for node in randomizeOrder(random, graph.nodes()):
            nodeDegree = pObj.nDegrees[node]
            neighbs = pObj.nNeighbs[node]
            nodeSelfLoop = pObj.nSelfW[node]
            comObj1 = communityDict[pObj.nComms[node]]
            bestNeighb, bestIncrease = '!', 0.0
            bestIntraCommW, bestAllCommW = 0.0, 0.0
            bestModScore = 0.0
          
            cost1,intraCommW1,allCommW1,comObj1NewMod = calculateCost(comObj1,
                                                        node,
                                                        neighbs,
                                                        nodeSelfLoop,
                                                        nodeDegree,
                                                        pObj.edgeDict,
                                                        pObj.tW,
                                                        pObj.tW_x2,
                                                        weight=weight)
      
            neighborsToCalc = { pObj.nComms[n]:communityDict[pObj.nComms[n]] \
                                for n in neighbs if pObj.nComms[n] != comObj1.name }
            
            for neighCommName,comObj2 in randomizeOrder(random,neighborsToCalc.items()):
                cost2,intraCommW2,allCommW2,comObj2NewMod = calculateGain(comObj2,
                                                            node,
                                                            neighbs,
                                                            nodeSelfLoop,
                                                            nodeDegree,
                                                            pObj.edgeDict,
                                                            pObj.tW,
                                                            pObj.tW_x2,
                                                            weight=weight)
               
                increase = cost1 + cost2
                if increase > bestIncrease:
                    bestIncrease = increase
                    bestNeighb = neighCommName
                    bestComObj = comObj2
                    bestIntraCommW = intraCommW2
                    bestAllCommW = allCommW2
                    bestModScore = comObj2NewMod
                    
   
            if bestNeighb != '!':
                bestComObj.intraCommWeights = bestIntraCommW
                bestComObj.allWeights = bestAllCommW
                bestComObj.nodeDict.update({node:''})
                bestComObj.modularity = bestModScore
                pObj.nComms[node] = bestComObj.name
                comObj1.intraCommWeights = intraCommW1
                comObj1.allWeights = allCommW1
                comObj1.modularity = comObj1NewMod
                del comObj1.nodeDict[node]
                currentMod += bestIncrease
            
 
        if (currentMod - endLoopMod) < endLoopMinimum:
            killSignal = 0
            if len(modScores) == 1:
                killSignal = 1
            break
        else:
            print('Loop Completed')
            modScores.append(currentMod)
   
    return communityDict, pObj, modScores, killSignal
                                      
def louvainModularityOptimization(graph, priorCommunities=False, random=False, commKey='chrom', weight='weight'):
    levels = []
   
    if priorCommunities == False:
        cDict, pObj = initializeCommunities(graph.G, weight=weight)

    mScores = []
    roundNumber = 0

    if (priorCommunities == False):
        cDict, pObj, nmScores, killSig = phaseOne(graph.G, cDict, pObj, random, weight)
    ####################################################
    print('Round %s phase1 completed' % (roundNumber+1)) 
    mScores += nmScores
    roundNumber += 1
    levels.append(pObj.nComms.copy())   
    comms, inc_list, inc_nodes = giveFinalCommunities(levels)
    new_idx={}
    idx=0
    for key in inc_list.keys():
        new_idx[idx]=inc_list[key]
        idx += 1
    
    return new_idx

def giveFinalCommunities(partitionLevels, writeToFile=False):
    """ Retrieves the final community assignment for all nodes in the graph.
        Loops through a list of dictionaries. The first dictionary is { originalNodeName:communityID }
        Then each successive dictionary, the communityID from the previous one is now the key, and the
        value is the new communityID found at that level of the algorithm."""
    nodeComms = {}
    inc_list = {}
    comm_indx = -1
    comm_list = []
    for nodeName in partitionLevels[0]:
        i, comm = 1, partitionLevels[0][nodeName]
        while i < len(partitionLevels):
            comm = partitionLevels[i][comm]
            i += 1
        nodeComms.update({nodeName:comm})

        if comm not in comm_list:
            comm_list.append(comm)
            inc_list[comm] = []
            inc_list[comm].append(nodeName)
        else:
            inc_list[comm].append(nodeName)

    
    if writeToFile != False:
        outFile = open(writeToFile, 'w')
        outFile.write("#NodeID"+'\t'+"CommunityID")
        nodesToWrite = [ [node, comm] for node, comm in nodeComms.items() ]
        nodesToWrite = sorted(nodesToWrite, key=lambda x: x[0])
        for n in nodesToWrite:
            outFile.write(str(n[0])+'\t'+str(n[1])+'\n')
        outFile.close()
  
    return nodeComms, inc_list, comm_list
