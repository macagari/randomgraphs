# Python program to find maximum number of edge disjoint paths
# Complexity : (E*(V^3))
# Total augmenting path = VE
# and BFS with adj matrix takes :V^2 times
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
randomGraph = __import__('ER-Random Graph')
regRandGraph = __import__('Regular Random Graph')


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
        visited = [False]*self.ROW

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
                    parent[ind] = u  # store path

        # If we reached sink in BFS starting from source, then return
        # true, else false
        return True if visited[t] else False

    # Returns the maximum number of edge-disjoint paths from s to t in the given graph
    # It was used the max-flow/residual capacity approach
    def find_disjoint_paths(self, source, sink, l):
        paths = []
        # This array is filled by BFS to store path
        parent = [-1]*self.ROW

        max_flow = 0  # There is no flow initially

        # Augment the flow while there is a path from source to sink
        while self.BFS(source, sink, parent):
            path = ""  # keep track of the nodes visited

            path_flow = np.inf
            s = sink
            while s != source:
                path += str(s)
                path_flow = min(path_flow, self.graph[parent[s]][s])
                if s != source and s != sink:
                    self.delete_edges(s)  # remove all the edges of the vertex s, so it cannot be used again
                elif parent[s] == source:
                    self.delete_edge(source, sink)
                s = parent[s]
            path += str(source)

            # just enter here in the first cicle or the path is also a shortest path
            if len(paths) == 0:
                max_flow += path_flow   # Add path flow to overall flow
                paths.append(path[::-1])
            elif len(path) == len(paths[0]):
                max_flow += path_flow  # Add path flow to overall flow
                paths.append(path[::-1])
            if l == len(paths):
                break
        return -max_flow, paths

    # delete all the edges of one node
    def delete_edges(self, u):
        for i in range(self.ROW):
            self.delete_edge(u,i)

    # delete a specific edge
    def delete_edge(self, u, v):
        self.graph[u][v] = 0


def main():
    l = int(sys.argv[1])
    RG = regRandGraph.RegularGraph(10, 3, 0.3)
    rand_graph = RG.build_regular_graph()
    print(rand_graph)
    G = nx.from_numpy_matrix(rand_graph)
    plt.clf()
    #nx.draw_networkx(G, pos=nx.spring_layout(G))
    #plt.show(block=True)

    g = Graph(rand_graph)
    source = 0;
    sink = np.shape(rand_graph)[0] - 1
    disj_path = g.find_disjoint_paths(source, sink, l)
    all_paths = disj_path[1]
    print("These are the %d edge-disjoint paths from %d to %d:" %
          (disj_path[0], source, sink))
    print(all_paths)


if __name__ == "__main__":
    main()
