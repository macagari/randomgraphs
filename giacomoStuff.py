import sys
import numpy as np
import math
import datetime
import time
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


def method1(L):
    k = np.shape(L)[0]  # take the dimesion of Laplacian matrix
    sum_B = np.zeros((k, k))
    P = np.identity(k)
    A = np.diag(np.diag(L)) - L
    #print(A)
    sum_B = P
    for i in range(1, k):
        sum_B += LA.matrix_power(A, i)  # I+A+A^2....>0
        # print(i)
        # print(sum_B)
        # sum_B += P
        # P = P * A
    s = sum_B
    #print(s)
    if (s.all() > 0):
        print("Connected.M1")
    else:
        print("Not connected.M1")


def method2(rand_graph):
    eigenValues = LA.eig(rand_graph)[0]
    secmin = sorted(eigenValues)[1]  ### sort and pick second
    if (secmin > 0.001):
        print("Connected.M2")
    else: print("Not connected.M2",secmin)


def method3(rand_graph,n,start):
    start_time = time.time()
    lst = matrix_to_list(rand_graph)
    breads = bfs(lst, start)
    if n == len(breads):
        print("Connected.M3")
    else:
        print("Not connected.M3 ")

def main():
    n = int(sys.argv[1])
    r = int(sys.argv[2])
    p = float(sys.argv[3])
    startingNode =int(sys.argv[4])

    rand_graph = randomGraph.buildRandomGraph(n, p)
    print(rand_graph)
    print("\n",method1(rand_graph), method2(rand_graph),method3(rand_graph,n,startingNode))

    rand_reg_graph = regRandGraph.RegularGraph(n, r, p)
    a = rand_reg_graph.build_regular_graph()
    print(a)
    print(method1(a),method2(a),method3(a,n,startingNode))
if __name__ == "__main__":
    main()
    
    
    
#point 4 function 
                                    #n indicate which method complexity you want to check
def mean_complex(sim, nodes, prob,n_graph,n):
    
    v1 = []
    v2 = []
    v3 = []
    
    if(n == 1):
        for i in range(0,n_graph):
            g = buildRandomGraph(nodes, prob)
            for j in range(sim):
                t1 = time.time()
                m1 = method1(g)
                v1.append(time.time()-t1)
        print(v1)
        print("mean of method1:",np.mean(v1))
        plt.plot([i for i in v1], [v1[i] for i in v1], 'ro')
        plt.show()
    if(n == 2):
        for i in range(0,n_graph):
            g = buildRandomGraph(nodes, prob)
        
            for k in range(sim):
                t2 = time.time()
                m2 = method2(g)
                v2.append(time.time()-t2)
        print(v2)
        print("mean of method2:",np.mean(v2))
          
        
    if(n==3):
        for i in range(0,n_graph):
            g = buildRandomGraph(nodes, prob)
            
            for l in range(sim):
                t3 = time.time()
                m3 = method3(g,8,3)
                v3.append(time.time()-t3)
                
        print(v3)
        print("mean of method3:",np.mean(v3))
