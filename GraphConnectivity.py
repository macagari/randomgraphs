import numpy as np
import matplotlib.pyplot as plt
randomGraph = __import__('ER-Random Graph')
regRandGraph = __import__('Regular Random Graph')


def plot_prob_connectivity(run=20):
    fig, all_plots = plt.subplots(3, 2)
    fig.suptitle("Connectivity probability of random graphs", fontsize=16)

    # generate the Y and X axis for the connectivity probability graph for the ER-Graph
    y_values = []
    prob_values = np.arange(0, 1, 0.01)
    for p in prob_values:
        number_connected = 0
        for i in range(run):
            g_er = randomGraph.buildRandomGraph(n=100, p=p)
            number_connected += test_connect(g_er)
        y_values.append(number_connected/run)

    all_plots[0, 0].plot(prob_values, y_values)
    all_plots[0, 0].set_title('Erdos-Renyi graph, n=100')

    # generate the Y and X axis for the connectivity probability graph for the R-Regular-Graph
    many_r_dict = {}
    for r in [2, 4, 8, 16]:
        y_values = []
        n_values = np.arange(5, 100, 5)
        for n in range(n_values):
            number_connected = 0
            for i in range(run):
                RG = regRandGraph.RegularGraph(n=n, r=r, p=0.3)
                g_reg = RG.build_regular_graph()
                number_connected += test_connect(g_reg)
            y_values.append(number_connected / run)
        many_r_dict[str(r)] = (n_values, y_values)

    all_plots[1, 0].plot(many_r_dict['2'][0], many_r_dict['2'][1])
    all_plots[1, 0].set_title('R-Regular graph, r=2, p=0.3')
    all_plots[1, 1].plot(many_r_dict['4'][0], many_r_dict['4'][1])
    all_plots[1, 1].set_title('R-Regular graph, r=4, p=0.3')
    all_plots[2, 0].plot(many_r_dict['8'][0], many_r_dict['8'][1])
    all_plots[2, 0].set_title('R-Regular graph, r=8, p=0.3')
    all_plots[2, 1].plot(many_r_dict['16'][0], many_r_dict['16'][1])
    all_plots[2, 1].set_title('R-Regular graph, r=16, p=0.3')

    plt.show(block=True)
def test_connect(a):
    return np.random.choice([0, 1], p=[0.5, 0.5])


def main():
    plot_prob_connectivity()



if __name__ == "__main__":
    main()