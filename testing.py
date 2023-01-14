import main
import graphs_old

def test_random_const_dens(dens: float=0.6):
    with open("exp_data_d.txt", 'w') as f1:
        for i in range(1, 16):
            vertices = 20*i
            max_edges = vertices*(vertices-1)//2
            edges = int(max_edges * dens)
            graphs_old.generate_random(vertices, edges, "random_graph.txt")

            print("Iteration cycle {} of 16...\n".format(i))

            # greedy:
            greedy = graphs_old.Graph()
            greedy.create_graph("random_graph.txt")
            greedy.greedy_coloring()
            greedy_colors = greedy.get_greatest_color()+1

            problem = main.Coloring("random_graph.txt")
            problem.colorize()
            GA_colors = problem.best_color

            f1.write("{} {} {} {}\n".format(vertices, edges, greedy_colors, GA_colors))

def test_random_const_vert(v: int=100):
    with open("exp_data_v.txt", 'w') as f1:
        vertices = v
        max_edges = vertices*(vertices-1)//2
        for i in range(10, 90, 5):
            edges = int(max_edges * i//100)
            graphs_old.generate_random(vertices, edges, "random_graph.txt")

            print("Iteration cycle {} of 16...\n".format(i))

            # greedy:
            greedy = graphs_old.Graph()
            greedy.create_graph("random_graph.txt")
            greedy.greedy_coloring()
            greedy_colors = greedy.get_greatest_color()+1

            problem = main.Coloring("random_graph.txt")
            problem.colorize()
            GA_colors = problem.best_color

            f1.write("{} {} {} {}\n".format(i, edges, greedy_colors, GA_colors))

def test_benchmark():
    # names = [
    #     "le450_25a", "inithx", "fpsol2",
    #     "miles250", "miles500", "miles750", "miles1000", "miles1500",
    #     "queen6", "queen8_12", "queen11_11", "queen13_13"]
    # data = [25, 54, 65, 8, 20, 31, 42, 73, 7, 12, 11, 13]

    names = ["queen7_7", "queen8_8", "queen9_9"]
    data = [7, 9, 10]
    colorings = []
    errors = []

    for i in range(len(names)):
        adr = "graph_examples/"+names[i]+".txt"
        problem = main.Coloring(adr)
        problem.colorize()
        c = problem.best_color
        e = abs(data[i]-c)/data[i]
        colorings.append(c)
        errors.append(e)

    print(colorings)
    print(data)
    print(errors)

if __name__ == '__main__':
    test_random_const_vert()