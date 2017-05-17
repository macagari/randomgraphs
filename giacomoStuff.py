import sys
import numpy as np
import math
import datetime
from numpy import linalg as LA
randomGraph = __import__('ER-Random Graph')
regRandGraph = __import__('Regular Random Graph')


#function needed for bfs
def matrix_to_list(matrix):
    graph = {}
    for i, node in enumerate(matrix):
        adj = []
        for j, connected in enumerate(node):
            if connected:
                adj.append(j)
        graph[i] = adj
    return graph

def bfs(graph, v):
    all = []
    Q = []
    Q.append(v)
    while Q != []:
        v = Q.pop(0)
        all.append(v)
        for n in graph[v]:
            if n not in Q and \
                            n not in all:
                Q.append(n)
    return all


def method2(rand_graph):
    eigenValues = LA.eig(rand_graph)[0]
    secmin = sorted(eigenValues)[1]  ### sort and pick second
    if (secmin > 0):
        print("connesso")
    else: print("NON CONNESSO COGLIONE")


#def method3():

def main():
    n = int(sys.argv[1])
    r = int(sys.argv[2])
    p = float(sys.argv[3])

    rand_graph = randomGraph.buildRandomGraph(n, p)
    print(rand_graph,"\n", method2(rand_graph))

    rand_reg_graph = regRandGraph.RegularGraph(n, r, p)
    a = rand_reg_graph.build_regular_graph()
    print("\n", a,method2(a))

    print("method 2 \n ")
if __name__ == "__main__":
    main()