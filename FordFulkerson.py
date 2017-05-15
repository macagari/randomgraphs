# Python program to find maximum number of edge disjoint paths
# Complexity : (E*(V^3))
# Total augmenting path = VE
# and BFS with adj matrix takes :V^2 times
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
randomGraph = __import__('ER-Random Graph')
regRandGraph = __import__('Regular Random Graph')
import sys


# This class represents a directed graph using
# adjacency matrix representation
class Graph:
    def __init__(self, graph):
        self.graph = graph  # residual graph
        self.ROW = len(graph)

    # Returns true if there is a path from source 's' to sink 't' in
    # residual graph. Also fills parent[] to store the path '''
    def BFS(self, s, t, parent):

        # Mark all the vertices as not visited
        visited = [False] * (self.ROW)

        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # Standard BFS Loop
        while queue:

            # Dequeue a vertex from queue and print it
            u = queue.pop(0)

            # Get all adjacent vertices of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val == -1:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        # If we reached sink in BFS starting from source, then return
        # true, else false
        return True if visited[t] else False

    # Returns tne maximum number of edge-disjoint paths from
    # s to t in the given graph
    def findDisjointPaths(self, source, sink, l):
        paths = []
        # This array is filled by BFS and to store path
        parent = [-1] * (self.ROW)

        max_flow = 0  # There is no flow initially

        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent):
            path = ""  # keep track of the nodes visited
            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = np.inf
            s = sink
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Add path flow to overall flow
            max_flow += path_flow

            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            path += str(sink)
            while v != source:
                u = parent[v]
                path += str(u)  # add the node to the path
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

                if u != source:
                    self.delete_edges(u)  # remove all the edges of the vertex u, so it cannot be used again
            paths.append(path[::-1])
            if l != 0 and l == len(paths):
                return -max_flow, paths
        return -max_flow, paths

    def delete_edges(self, u):
        for i in range(self.ROW):
            self.graph[u][i] = 0


def main():
    l = int(sys.argv[1])
    # randGraph = randomGraph.buildRandomGraph(10, 0.3)
    randGraph = regRandGraph.buildRegularGraph(6, 2, 0.3)
    print(randGraph)
    G = nx.from_numpy_matrix(randGraph)
    plt.clf()
    nx.draw_networkx(G, pos=nx.spring_layout(G))
    g = Graph(randGraph)
    source = 0;
    sink = np.shape(randGraph)[0] - 1
    disj_path = g.findDisjointPaths(source, sink, l)
    all_paths = disj_path[1]
    print("These are the %d edge-disjoint paths from %d to %d:" %
          (disj_path[0], source, sink))
    print(all_paths)
    plt.show(block=True)


if __name__ == "__main__":
    main()
