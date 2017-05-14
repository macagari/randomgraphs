import sys
import numpy as np
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


def main():
    n = int(sys.argv[1])
    p = float(sys.argv[2])
    rand_graph = buildRandomGraph(n, p)
    print(rand_graph)

if __name__ == "__main__":
    main()