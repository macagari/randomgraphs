import sys
import numpy as np
import math
import datetime


def buildRegularGraph(n, r, p):
    while True:
        restart = False
        mat = np.reshape(np.zeros(n * n), newshape=(n, n))
        for i in range(n):
            x = mat[i,]
            if mat[i,i] < r:
                # the quantity of arches that needs to be assigned
                left = r - mat[i,i]
                empty_pos = np.where(x == 0)[0]
                new_arches = np.zeros(len(empty_pos))

                # with prob p for each arch, repeat until it equals the quantity of arches left
                nextLine = False
                while nextLine is False:
                    j = 0
                    while j < len(empty_pos):
                        new_arches[j] = np.random.choice([0, -1], p=[1-p, p])
                        j += 1
                    if sum(new_arches) == -left:
                        # when assigning a arch, the algorithm can break the regular condition on the other nodes!
                        if drawArches(mat, i, r, empty_pos, new_arches) is True:
                            nextLine = True
                        else:
                            restart = True
                            break
            if restart is True:
                break
        if mat.diagonal().prod() == math.pow(r, n):
            break

    return mat


def drawArches(mat, i, r, empty_pos, new_arches):
    isRegular = True  # it's guaranteed that the mat[i,] is Regular, but not the others that "receive" the arches
    for j in range(len(empty_pos)):
        if i != empty_pos[j]:
            mat[i, empty_pos[j]] = new_arches[j]
            if new_arches[j] == -1:
                mat[i, i] += 1  # update the node degree
                mat[empty_pos[j], i] = -1  # "reflect" the arch to the lower matrix
                mat[empty_pos[j], empty_pos[j]] += 1  # update the node degree in other node

                # check if the other node still keeps the regularity of the graph
                if mat[empty_pos[j], empty_pos[j]] > r:
                    isRegular = False
                    break
    return isRegular


def main():
    n = int(sys.argv[1])
    r = int(sys.argv[2])
    p = float(sys.argv[3])
    t1 = datetime.datetime.now()
    rand_graph = buildRegularGraph(n, r, p)
    print(datetime.datetime.now() - t1)
    print(rand_graph)

if __name__ == "__main__":
    main()