import sys
import numpy as np
import math
import datetime
import random



def buildRandomGraph(n, p):
    mat = np.reshape(np.zeros(n*n), newshape=(n,n))
    for i in range(n):
        for j in range(i+1, n):
            rand = random.uniform(0, 1)
            if p >= rand:
                drawArch(mat,i,j)
    return mat


def drawArch(mat,i,j):
    mat[i, j] = -1  # create the arch
    mat[i, i] += 1  # update the node degree
    mat[j, i] = -1
    mat[j, j] += 1



class RegularGraph:
    def __init__(self, n, r, p):
        self.remaining_nodes = n
        self.mat = np.reshape(np.zeros(n * n), newshape=(n, n))
        self.n = n
        self.r = r
        self.p = p

    def build_regular_graph(self):
        restart = False
        # loop until all nodes are r-degree
        while self.remaining_nodes != 0:
            if restart:
                self.restart()
                restart = False

            # iterating through the lines creating the random arcs
            for i in range(self.n):

                # check if the degree of the node is already r or it has 0 arcs (it is still filled with nan)
                if self.mat[i,i] < self.r:
                    arcs_remaining = self.r - self.mat[i,i]

                    # if the quantity of arcs to be assigned is greater than the remaining nodes, then restart the graph
                    if arcs_remaining >= self.remaining_nodes:
                        restart = True
                        break

                    # empty_pos gives the arcs that could be drawn
                    empty_pos = self.get_empty_positions(i)
                    new_arcs = np.zeros(len(empty_pos))

                    # with prob p for each arc, repeat until it equals the quantity of the remaining arcs
                    while sum(new_arcs) != -arcs_remaining:
                        for j in range(len(empty_pos)):
                            new_arcs[j] = np.random.choice([0, -1], p=[1-self.p, self.p])

                    self.draw_arcs(i, empty_pos, new_arcs)
        where_are_NaNs = np.isnan(self.mat)
        self.mat[where_are_NaNs] = 0
        return self.mat

    def draw_arcs(self, i, empty_pos, new_arcs):
        for j in range(len(empty_pos)):

            # don't draw arcs to itself
            if i != empty_pos[j]:
                if new_arcs[j] == -1:
                    self.mat[i, empty_pos[j]] = new_arcs[j]  # draw the arc
                    self.mat[i, i] += 1  # update the node degree
                    self.mat[empty_pos[j], empty_pos[j]] += 1  # update the node degree in the other node

                    self.mat[empty_pos[j], i] = -1  # "reflect" the arc to the lower matrix

                    # check if the other nodes are full (node's degree == r)
                    self.check_degree(empty_pos[j])
        self.check_degree(i)

    def check_degree(self, i):
        if self.mat[i, i] == self.r:
            self.remaining_nodes -= 1
            self.make_node_unavailable(i)

    # in the line 'i', verify which arcs still need to be drawn
    def get_empty_positions(self, i):
        x = self.mat[i, ]  # get the row to be updated
        empty_pos = np.where(x == 0)[0]
        return np.delete(empty_pos, np.where(empty_pos == i)[0])  # don't generate arcs to itself (the diagonal)

    # guarantees that no other node will try to draw an arc to node U. It will be marked as "full"
    def make_node_unavailable(self, u):
        for j in range(0, self.mat.shape[0]):
            if j != u and self.mat[j, u] != -1:
                self.mat[j, u] = np.nan

    # reinitialize the matrix and the remaining nodes
    def restart(self):
        self.mat = np.reshape(np.zeros(self.n * self.n), newshape=(self.n, self.n))
        self.remaining_nodes = self.n


def main():
    n = int(sys.argv[1])
    r = int(sys.argv[2])
    p = float(sys.argv[3])

    rand_graph = buildRandomGraph(n, p)
    print(rand_graph)

    for i in range(30):
        t1 = datetime.datetime.now()
        rand_reg_graph = RegularGraph(n, r, p)
        a = rand_reg_graph.build_regular_graph()
        print(datetime.datetime.now() - t1)

if __name__ == "__main__":
    main()