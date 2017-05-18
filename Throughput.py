import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

randomGraph = __import__('ER-Random Graph')
regRandGraph = __import__('Regular Random Graph')
dp = __import__('DisjointPath')



def throughput(origG, l):
    n = len(origG)
    links = np.zeros((n, n,))
    # we use the fact that the graph is undirected
    for i in range(n - 1):
        j = i + 1
        while j < n:
            matrix = np.copy(origG)
            sel = dp.Graph(matrix)
            disj_path = sel.find_disjoint_paths(i, j, l)
            paths = disj_path[1]

            for path in paths:
                k = 0
                while k < len(path) - 1:
                    if (path[k] < path[k + 1]):
                        if links[path[k], path[k + 1]] == 0:
                            links[path[k], path[k + 1]] = 1
                        else:
                            links[path[k], path[k + 1]] = links[path[k], path[k + 1]] + 1
                    else:
                        if links[path[k + 1], path[k]] == 0:
                            links[path[k + 1], path[k]] = 1
                        else:
                            links[path[k + 1], path[k]] = links[path[k + 1], path[k]] + 1
                    k = k + 1

            j = j + 1
    m = links.max()
    thr = l / m
    return thr


def linkFail (mat, p, l, bloc):
    matrix = np.copy(mat)
    failure = 0
    if bloc != 0:
        if bloc == 'any':
            for i in range(len(mat)):
                for j in range(len(mat)):
                    if (matrix[i, j] == -1):
                        k = np.random.choice([0, 1], p=[1 - p, p])
                        if (k == 1):
                            failure = failure + 1
                            matrix[i, j] = 0
                            matrix[j, i] = 0
                            matrix[i, i] = matrix[i, i] - 1
                            matrix[j, j] = matrix[j, j] - 1
        else:
            while failure <= bloc:
                for i in range(len(mat)):
                    for j in range(len(mat)):
                        if (matrix[i,j] == -1):
                            k = np.random.choice([0, 1], p=[1 - p, p])
                            if (k == 1):
                                failure = failure + 1
                                matrix[i, j] = 0
                                matrix[j,i] = 0
                                matrix[i,i] = matrix[i,i] - 1
                                matrix[j,j] = matrix[j,j] - 1

    plotThroughput(matrix, l)


def plotThroughput( mat, l):
    thr = []
    for i in l:
        thr.append(throughput(mat,i))

    plt.plot(l, thr)
    plt.xlabel('l')
    plt.ylabel('Throughput')
    plt.axis([1, 4, 0, 1])
    plt.show()



def main():
   # l = int(sys.argv[1])
    g1 = randomGraph.buildRandomGraph(80, 0.7)
    #randGraph = regRandGraph.build_regular_graph(6, 2, 0.3)
    #randGraph = np.array([[3,-1,-1,-1],[-1,3,-1,-1],[-1,3,-1,-1],[0,-1,-1,2]])
    #print(randGraph)
    #G = nx.from_numpy_matrix(randGraph)
    #plt.clf()
    #nx.draw_networkx(G, pos=nx.spring_layout(G))
    #plt.show(block=True)


    #g1 = np.array([[3, -1, -1, -1, 0], [-1, 2, -1, 0, 0], [-1, -1, 3, 0, -1], [-1, 0, 0, 2, -1], [0, 0, -1, -1, 2]])
    G1 = nx.from_numpy_matrix(g1)
    plt.clf()
    nx.draw_networkx(G1, pos=nx.spring_layout(G1))


    plt.show(block=True)  #thr = g.throughput()
    #print(thr)
    # print("There are the following paths: %d from %d to %d" %
    #      (g.shortest_paths(0, np.shape(randGraph)[0] - 1, depth), 0, np.shape(randGraph)[0]))

    #g = Graph(G)
    #source = 0;
    #sink = np.shape(g1)[0] - 1

    #disj_path = g.find_disjoint_paths(source, sink, 2)
    #all_paths = disj_path[1]

    #print("These are the %d edge-disjoint paths from %d to %d:" %
    #     (disj_path[0], source, sink))
    #print(all_paths)


    #G2 = nx.from_numpy_matrix(mat)
    #plt.clf()
    #nx.draw_networkx(G2, pos=nx.spring_layout(G2))

    #G1 = nx.from_numpy_matrix(matrix)
    #plt.clf()
    #nx.draw_networkx(G1, pos=nx.spring_layout(G1))

    #g.plotThroughput(g1, [1, 2, 3, 4])
    l = [1,2,3,4]
    linkFail(g1,0.2, l)




    #plt.show(block=True)

if __name__ == "__main__":
    main()