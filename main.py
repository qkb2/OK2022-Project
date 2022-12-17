import random
from copy import copy


class Vertex:
    def __init__(self, name) -> None:
        self.name = name
        self.color = -1
        self.is_visited = 0

    def __repr__(self) -> str:
        return str(self.color)


class Graph:
    def __init__(self, path: str = None) -> None:
        self.V = 0
        self.E = 0
        self.edge_list = []
        self.matrix = []
        self.vertices = []
        self.create_edge_list_from_file(path)
        self.create_matrix_from_edge_list()

    def __repr__(self) -> str:
        s = ''
        for i in range(self.V):
            for j in range(self.V):
                s += str(self.matrix[i][j]) + ' '
            s += '\n'
        return s

    def create_edge_list_from_file(self, faddr: str = None):
        if faddr is None:
            faddr = input('Input the file address: ')

        self.edge_list = []
        with open(faddr, 'r') as fread:
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

    def reset_colors(self) -> None:
        for i in self.vertices:
            i.color = -1


class Individual:
    def __init__(self, root, permutation) -> None:
        self.root = root
        self.permutation = copy(permutation)
        self.fitness = 0

    def __repr__(self) -> str:
        text = ''
        text += "\nPermutation: " + str(self.permutation)
        text += "\nFitness: " + str(self.fitness)
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
        return c

    def set_fitness_from_greedy(self) -> None:
        self.fitness = self.get_value()

    def set_fitness_from_parents(self, parent1, parent2) -> None:
        self.fitness = (parent1.fitness + parent2.fitness) / 2


class Coloring:
    def __init__(self, path: str) -> None:
        self.graph = Graph(path)
        self.population_size = 200
        self.genetic_iterations = 20
        self.fitness_check_iteration = 4
        self.selection_multiplier = 1

    def colorize(self) -> None:
        population = self.initialize_population()
        best_individuals_list = []

        for i in range(self.genetic_iterations):
            print(f"Iteration {i + 1}...")
            if i % self.fitness_check_iteration == 0 or i == self.genetic_iterations - 1:
                for individual in population:
                    individual.set_fitness_from_greedy()
                best_individuals_list.append(self.best_in_population(population))

            population = self.selection(population)
            population = self.crossover(population)
            population = self.mutation(population)

        self.show_solution(best_individuals_list)

    def initialize_population(self) -> list:
        population = []
        perm_of_colors = list(range(0, self.graph.V))
        for _ in range(self.population_size):
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
        print(f'\n***** Best Solution *****\n'
              f'Colors: {best_individual.fitness + 1}\n'
              f'Permutation: {best_individual.permutation}\n'
              f'Coloring: {best_individual.get_coloring()}\n')


if __name__ == '__main__':
    problem = Coloring("gc500.txt")
    problem.colorize()
