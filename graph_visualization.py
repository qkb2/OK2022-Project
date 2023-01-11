import networkx as nx
import matplotlib.pyplot as plt
from main import Graph


def visualize_graph(coloring: list, graph: Graph) -> None:
    # colors = [ "lightcoral", "gray", "lightgray", "firebrick", "red", "chocolate", "darkorange", "moccasin",
    # "gold", "yellow", "darkolivegreen", "chartreuse", "forestgreen", "lime", "mediumaquamarine", "turquoise",
    # "teal", "cadetblue", "dogerblue", "blue", "slateblue", "blueviolet", "magenta", "lightsteelblue"]

    colors = ['blue', 'yellow', 'green', 'gray', 'red']
    real_coloring = []

    for el in coloring:
        real_coloring.append(colors[el[1] - 1])

    # print(real_coloring)
    # remember use only after using the Coloring init
    network = nx.Graph()
    for i in range(1, graph.V + 1):
        network.add_node(i)

    for el in graph.edge_list:
        network.add_edge(el[0], el[1])

    # print(network)
    pos = nx.circular_layout(network)

    nx.draw_networkx(network, pos=pos, node_color=real_coloring)
    plt.show()
