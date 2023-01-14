import random
from time import perf_counter
from copy import copy

import networkx as nx
import matplotlib.pyplot as plt


class Vertex:
    def __init__(self, name) -> None:
        self.name = name
        self.color = -1
        self.is_visited = 0
        self.saturation = 0

    def __repr__(self) -> str:
        return str(self.color)


class Graph:
    def __init__(self, path: str = None) -> None:
        self.V = 0
        self.E = 0
        self.edge_list = []
        self.matrix = []
        self.vertices = []
        self.path = path
        self.create_edge_list_from_file(path)
        self.create_matrix_from_edge_list()

    def __repr__(self) -> str:
        s = ''
        for i in range(self.V):
            for j in range(self.V):
                s += str(self.matrix[i][j]) + ' '
            s += '\n'
        return s

    def create_edge_list_from_file(self, path: str = None):
        if path is None:
            path = input('Input the file address: ')

        self.path = path
        self.edge_list = []
        with open(path, 'r') as fread:
            x = fread.readline()
            self.V = int(x)

            while True:
                x = fread.readline()
                if x.strip() == '':
                    return

                array = x.split()
                array = list(map(int, array))
                self.edge_list.append(array)

    def create_matrix_from_edge_list(self) -> None:
        self.E = len(self.edge_list)
        self.matrix = [[0 for _ in range(self.V)] for _ in range(self.V)]

        for v in range(self.V):
            self.vertices.append(Vertex(v + 1))

        for el in self.edge_list:
            self.matrix[el[0] - 1][el[1] - 1] = 1
            self.matrix[el[1] - 1][el[0] - 1] = 1
            self.vertices[el[0] - 1].saturation += 1
            self.vertices[el[1] - 1].saturation += 1

    def reset_colors(self) -> None:
        for i in self.vertices:
            i.color = -1

    def get_dsatur_ordering(self) -> list:
        ordering = sorted(self.vertices, key=lambda x: x.saturation, reverse=True)
        perm = [v.name-1 for v in ordering]
        return perm

    def generate_random_graph(self, v: int, chromatic: float, safety_limit: int=100):
        # do not use
        self.V = v
        for v in range(self.V):
            self.vertices.append(Vertex(v + 1))
        self.matrix = [[0 for _ in range(v)] for _ in range(v)]
        c = 0
        safety_c = 0
        stop = int(chromatic*v)
        while c < stop and safety_c < safety_limit:
            a = random.randint(1, v)
            b = a
            while a == b:
                b = random.randint(1, v)
            if self.matrix[a - 1][b - 1] == 1:
                safety_c += 1
                continue
            self.matrix[a - 1][b - 1] = 1
            self.matrix[b - 1][a - 1] = 1
            self.vertices[a - 1].saturation += 1
            self.vertices[b - 1].saturation += 1
            c += 1

class Individual:
    def __init__(self, root, permutation) -> None:
        self.root = root
        self.permutation = copy(permutation)
        self.fitness = 0

    def __repr__(self) -> str:
        text = ''
        # text += "\nPermutation: " + str(self.permutation)
        text += "Fitness: " + str(self.fitness)
        return text

    def get_coloring(self) -> list:
        self.root.graph.reset_colors()
        coloring = []
        for i in self.permutation:
            colors = [0 for _ in range(self.root.graph.V)]
            for j in range(self.root.graph.V):
                if self.root.graph.matrix[i][j] == 1:
                    if self.root.graph.vertices[j].color == -1:
                        continue
                    colors[self.root.graph.vertices[j].color] = 1
            for j in range(self.root.graph.V):
                if colors[j] == 0:
                    self.root.graph.vertices[i].color = j
                    coloring.append((i + 1, j + 1))
                    break
        coloring.sort()

        c = 0
        for v in self.root.graph.vertices:
            if v.color > c:
                c = v.color

        return coloring

    def get_value(self) -> int:
        self.root.graph.reset_colors()
        for i in self.permutation:
            colors = [0 for _ in range(self.root.graph.V)]
            for j in range(self.root.graph.V):
                if self.root.graph.matrix[i][j] == 1:
                    if self.root.graph.vertices[j].color == -1:
                        continue
                    colors[self.root.graph.vertices[j].color] = 1
            for j in range(self.root.graph.V):
                if colors[j] == 0:
                    self.root.graph.vertices[i].color = j
                    break

        c = 0
        for v in self.root.graph.vertices:
            if v.color > c:
                c = v.color
        return c+1

    def set_fitness_from_greedy(self) -> None:
        self.fitness = self.get_value()

    def set_fitness_from_parents(self, parent1, parent2) -> None:
        self.fitness = (parent1.fitness + parent2.fitness) / 2

    def visualize_graph(self) -> None:
        colors = [ "lightcoral", "gray", "lightgray", "firebrick", "red", "chocolate", "darkorange", "moccasin",
        "gold", "yellow", "darkolivegreen", "chartreuse", "forestgreen", "lime", "mediumaquamarine", "turquoise",
        "teal", "cadetblue", "dogerblue", "blue", "slateblue", "blueviolet", "magenta", "lightsteelblue"]

        # colors = ['blue', 'yellow', 'green', 'gray', 'red']
        real_coloring = []

        coloring = self.get_coloring()

        for el in coloring:
            real_coloring.append(colors[el[1] - 1])

        # print(real_coloring)
        # remember use only after using the Coloring init
        network = nx.Graph()
        for i in range(1, self.root.graph.V + 1):
            network.add_node(i)

        for el in self.root.graph.edge_list:
            network.add_edge(el[0], el[1])

        # print(network)
        pos = nx.circular_layout(network)

        nx.draw_networkx(network, pos=pos, node_color=real_coloring)
        plt.show(block=False)
        plt.savefig("visualizations/graph{}.png".format(self.root.visualizations))
        self.root.visualizations += 1
        plt.close()


class Coloring:
    def __init__(self, path: str) -> None:
        self.best_individuals_list = []
        self.graph = Graph(path)
        self.population_size = 220
        self.genetic_iterations = 20
        self.fitness_check_iteration = 4
        self.selection_multiplier = 1
        self.dsatur_size = 10
        self.perf_time = 0
        self.max_time_minutes = 5
        self.safe_time = 20
        self.max_time = 60*self.max_time_minutes-self.safe_time
        self.visualize = False
        self.visualization_max = 2
        self.visualizations = 0
        self.best_color = 0

    def colorize(self, visualize=False) -> None:
        self.visualize = visualize
        time_start = perf_counter()
        population = self.initialize_population()

        for i in range(self.genetic_iterations):
            is_timeout = False
            print(f"Iteration {i + 1}...")
            if self.max_time > perf_counter():
                print("Timeout")
                is_timeout = True
                break
            if i % self.fitness_check_iteration == 0 or i == self.genetic_iterations - 1 or is_timeout:
                for individual in population:
                    individual.set_fitness_from_greedy()
                self.best_individuals_list.append(self.best_in_population(population))
                print(self.best_individuals_list)

                # for visualization
                if self.visualization_max > 0 and self.visualize:
                    self.best_individuals_list[-1].visualize_graph()
                    print(self.best_individuals_list[-1].fitness)
                    self.visualization_max -= 1

            if not is_timeout:
                population = self.selection(population)
                population = self.crossover(population)
                population = self.mutation(population)

        time_stop = perf_counter()
        self.perf_time = time_stop - time_start
        self.show_solution(self.best_individuals_list)

    def initialize_population(self) -> list:
        dsatur_perm = self.graph.get_dsatur_ordering()
        # print(dsatur_perm)
        population = [Individual(self, dsatur_perm) for _ in range(self.dsatur_size)]
        perm_of_colors = list(range(0, self.graph.V))
        for _ in range(self.population_size-self.dsatur_size):
            random.shuffle(perm_of_colors)
            population.append(Individual(self, perm_of_colors))
        return population

    @staticmethod
    def best_in_population(population: list) -> Individual:
        return min(population, key=lambda x: x.fitness)

    def selection(self, population: list) -> list:
        after_selection = []
        for i in range((self.population_size // 2)):
            multiplier_for_better = random.randint(1, 10)
            multiplier_for_worse = random.randint(1, int((1 + self.selection_multiplier) * 10))
            ind1 = population[i].fitness
            ind2 = population[i + self.population_size // 2].fitness
            if ind1 > ind2:
                ind1 *= multiplier_for_worse
                ind2 *= multiplier_for_better
            else:
                ind1 *= multiplier_for_better
                ind2 *= multiplier_for_worse

            if ind1 > ind2:
                after_selection.append(population[i + self.population_size // 2])
            else:
                after_selection.append(population[i])

        return after_selection

    def crossover(self, population: list) -> list:
        for i in range(self.population_size // 2):
            population.append(self.crossover_pair(random.choice(population[:self.population_size // 2]),
                                                  random.choice(population[:self.population_size // 2])))

        return population

    def crossover_pair(self, individual1: Individual, individual2: Individual) -> Individual:
        # counter visualizes the pair
        local_visualizer = False
        if self.visualization_max > 0 and self.visualize:
            local_visualizer = True
            individual1.visualize_graph()
            individual2.visualize_graph()
            self.visualization_max -= 1

        new_permutation = []
        crossover_point = random.randint(1, self.graph.V - 1)
        write_access = [x for x in range(self.graph.V)]

        if random.randint(0, 1):    # Adjusting to left  (Individual 1)
            for i in individual1.permutation[:crossover_point]:
                write_access[i] = -1
            for i in individual2.permutation[crossover_point:]:
                if write_access[i] != -1:
                    new_permutation.append(i)
                    write_access[i] = -1
                else:
                    new_permutation.append(-1)
            write_access = [i for i in write_access if i != -1]
            random.shuffle(write_access)

            iterator = 0
            for i in range(self.graph.V - crossover_point):
                if new_permutation[i] == -1:
                    new_permutation[i] = write_access[iterator]
                    iterator += 1
            new_permutation = individual1.permutation[:crossover_point] + new_permutation

        else:                       # Adjusting to right (Individual 2)
            for i in individual2.permutation[crossover_point:]:
                write_access[i] = -1
            for i in individual1.permutation[:crossover_point]:
                if write_access[i] != -1:
                    new_permutation.append(i)
                    write_access[i] = -1
                else:
                    new_permutation.append(-1)
            write_access = [i for i in write_access if i != -1]
            random.shuffle(write_access)
            iterator = 0

            for i in range(crossover_point):
                if new_permutation[i] == -1:
                    new_permutation[i] = write_access[iterator]
                    iterator += 1
            new_permutation = new_permutation + individual2.permutation[crossover_point:]

        new_individual = Individual(self, new_permutation)
        new_individual.set_fitness_from_parents(individual1, individual2)

        if local_visualizer:
            new_individual.visualize_graph()

        return new_individual

    def mutation(self, population: list) -> list:
        for i in population[:self.population_size // 2]:
            while random.randint(0, 1):
                gene1 = random.randint(0, self.graph.V - 1)
                gene2 = random.randint(0, self.graph.V - 1)
                i.permutation[gene1], i.permutation[gene2] = i.permutation[gene2], i.permutation[gene1]
        return population

    def show_solution(self, best_individuals_list: list) -> None:
        best_individual = self.best_in_population(best_individuals_list)
        self.best_color = best_individual.fitness
        if self.visualize:
            best_individual.visualize_graph()
        print(
            f'\n***** Best Solution: {best_individual.fitness} colors in {self.perf_time}s *****\n')
        with open('coloring_log.txt', 'a') as f:
            f.write(f'***** Best Solution for {self.graph.path}*****\n'
              f'Colors: {best_individual.fitness}\n'
              f'Time: {self.perf_time}\n'
              f'Permutation: {best_individual.permutation}\n'
              f'Coloring: {best_individual.get_coloring()}\n\n')


if __name__ == '__main__':
    problem = Coloring("graph_examples/le450_5a.txt")
    problem.colorize()
    # graph_visualization.visualize_graph(
    #     problem.best_in_population(problem.best_individuals_list).get_coloring(), problem.graph)
