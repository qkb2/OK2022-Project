import random
from copy import deepcopy

def generate_random(v: int, e: int, s: str):
    matrix = [[0 for _ in range(v)] for _ in range(v)]
    with open(s, 'w') as f1:
        f1.write(str(v) + '\n')
        c = 0
        while c < e:
            a = random.randint(1, v)
            b = a
            while a == b:
                b = random.randint(1, v)
            if matrix[a - 1][b - 1] == 1:
                continue
            matrix[a - 1][b - 1] = 1
            matrix[b - 1][a - 1] = 1
            c += 1
            f1.write(str(a) + ' ' + str(b) + '\n')


class Vertex:
    def __init__(self, name) -> None:
        self.name = name
        self.color = -1
        self.is_visited = 0

    def __repr__(self) -> str:
        return str(self.color)


class Graph:
    def __init__(self) -> None:
        self.V = 0
        self.E = 0
        self.edge_list = []
        self.matrix = []
        self.vertices = []

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

    def create_graph(self, faddr: str = None) -> None:
        self.create_edge_list_from_file(faddr)
        self.create_matrix_from_edge_list()

    def reset_colors(self):
        for i in self.vertices:
            i.color = -1



class Individual:
    def __init__(self, coloring):
        self.coloring = coloring
        self.colors = len(set([x for _, x in coloring]))

    def __repr__(self):
        # return "(" + str(self.coloring) + ", " + str(self.colors) + ")"
        return str(self.colors)

    def update_num_of_colors(self):
        self.colors = len(set([x for _, x in self.coloring]))


class Coloring:
    def __init__(self, path) -> None:
        self.graph = Graph()
        self.graph.create_graph(path)
        self.vertices_num = len(self.graph.vertices)

        self.greedies_on_start = 20
        self.population_size = 100
        self.max_number_of_populations = 10
        self.selection_chance = 0.5
        self.length_of_individual = self._bits_for_vertex(
            len(self.graph.vertices))
        self.crossover_probability = 0.8

        self.mutate_individual_probability = 0.5
        self.mutate_gene_probability = 0.05
        self.mutation_attempts = 20

    def greedy_coloring_perm(self, perm: list):
        self.graph.reset_colors()
        coloring = []
        for i in perm:
            colors = [0 for _ in range(self.graph.V)]
            for j in range(self.graph.V):
                if self.graph.matrix[i][j] == 1:
                    if self.graph.vertices[j].color == -1:
                        continue
                    colors[self.graph.vertices[j].color] = 1
            for j in range(self.graph.V):
                if colors[j] == 0:
                    self.graph.vertices[i].color = j
                    coloring.append((i + 1, j + 1))
                    break
        coloring.sort()
        return Individual(coloring)

    def greedy_coloring(self) -> None:
        for i in range(self.graph.V):
            colors = [0 for _ in range(self.graph.V)]
            for j in range(self.graph.V):
                if self.graph.matrix[i][j] == 1:
                    if self.graph.vertices[j].color == -1:
                        continue
                    colors[self.graph.vertices[j].color] = 1
            for j in range(self.graph.V):
                if colors[j] == 0:
                    self.graph.vertices[i].color = j
                    break

    def greedy_get_greatest_color(self) -> int:
        c = 0
        for v in self.graph.vertices:
            if v.color > c:
                c = v.color
        return c

    def greedy_get_color_str(self) -> str:
        s = ''
        for v in self.graph.vertices:
            s += str(v.name) + ' has color ' + str(v.color) + '\n'
        return s

    def genetic_coloring(self) -> None:
        population = self.first_pop()
        print(f'startowa populacja\t{population}')
        # solutions = [(self.best_in_pop(population))]
        print(f'Najlepszy w populacji\t {self.best_in_pop(population)}\t{self.best_in_pop(population).coloring}\n\n')
        for i in range(self.max_number_of_populations):
            population = self.selection(population)
            print(f'Po selekcji\t\t{population}')
            population = self._crossover(population)
            print(f'Po wymieszaniu\t{population}')
            population = self._mutation(population)
            print(f'Po mutacji\t\t{population}')
            # print(all([self.is_valid(x.coloring) for x in population]))
            # print(self.best_in_pop(population).coloring)
            print(f"Najlepszy w populacji: {self.best_in_pop(population)}\t{self.best_in_pop(population).coloring}")
            print('\n')



    def first_pop(self):
        pop = []
        perm_of_colors = list(range(0, self.vertices_num))
        for i in range(self.greedies_on_start):
            random.shuffle(perm_of_colors)
            pop.append(self.greedy_coloring_perm(perm_of_colors))

        while len(pop) < self.population_size:
            colors = [i for i in range(self.vertices_num)]
            coloring = []
            for i in range(self.vertices_num):
                vertex = i + 1
                chosen_color = random.choice(colors)
                coloring.append((vertex, chosen_color))
            if self.is_valid(coloring):
                # individual = self._encode(coloring)
                # pop.append(individual)
                pop.append(Individual(coloring))
        return pop

    def is_valid(self, coloring):
        e = self.graph.edge_list
        c = coloring
        return all(self.color(w1, c) != self.color(w2, c) for w1, w2 in e)

    @staticmethod
    def color(vertex, coloring):
        """Returns color in coloring palette given vertex number."""
        for v, c in coloring:
            if v == vertex:
                return c

    def best_in_pop(self, pop):
        best_individual = pop[0]

        for individual in pop:
            if individual.colors < best_individual.colors:
                best_individual = individual
        return best_individual

    def selection(self, population):
        after_selection = []
        for i in range((self.population_size // 2)):
            multiplier_for_better = random.randint(1, 10)
            multiplier_for_worse = random.randint(1, int((1 + self.selection_chance) * 10))
            ind1 = population[i].colors
            ind2 = population[i + self.population_size // 2].colors
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

    def _encode(self, coloring, individual_len=None):
        """Encode the vertices into one long binary sequence. The vertex
        number 1 is most righte number 2 second from right etc.
        """
        if individual_len is None:
            individual_len = self.length_of_individual

        individual = 0
        for w, k in coloring:
            individual |= k << (w * individual_len)
        return individual

    def _decode(self, individual, individual_len=None, num_of_vertices=None):
        """Reverse function to _encode. We first create mask to extract
        chosen color and then bitshift right to obtain an actual number.
        """
        if individual_len is None:
            individual_len = self.length_of_individual
        if num_of_vertices is None:
            num_of_vertices = self.vertices_num

        coloring = []
        mask = 0
        for i in range(individual_len):
            mask |= 1 << i

        for i in range(num_of_vertices):
            vertex = i + 1
            current_mask = mask << (vertex * individual_len)
            color = (individual & current_mask) >> (vertex * individual_len)
            coloring.append((vertex, color))

        return coloring

    def _bits_for_vertex(self, v):
        """Calculates number of bits necessary for decoding every vertex's
        color.
        """
        i = 1
        while 2 ** i <= v:
            i += 1
        return i

    def _crossover(self, population):
        pairs = []
        for i in range(self.population_size // 2):
            individual1 = random.choice(population)
            individual2 = random.choice(population)
            pairs.append((individual1, individual2))

        after_crossover = []
        for c1, c2 in pairs:
            after_crossover.extend(self._crossover_individuales(c1, c2))
        return after_crossover

    def _crossover_individuales(self, c1, c2):
        """Perform crossover with fixed probability. Then randomly choose
        point where two individuals will crossover.
        """
        c1 = self._encode(c1.coloring)
        c2 = self._encode(c2.coloring)

        if random.uniform(0.0, 1.0) < self.crossover_probability:
            crossover_point = random.randint(0, self.length_of_individual)
            mask = 0
            for i in range(crossover_point):
                mask |= 1 << i
            new_c1 = (c1 & mask) | (c2 & ~mask)
            new_c2 = (c1 & ~mask) | (c2 & mask)
            return Individual(self._decode(new_c1)), Individual(self._decode(new_c2))
        else:
            return Individual(self._decode(c1)), Individual(self._decode(c2))

    def _mutation(self, population):
        after_mutation = []
        for individual in population:
            if random.uniform(0.0, 1.0) < self.mutate_individual_probability:
                after_mutation.append(self._mutate_individual(individual))
                # print(f"po mutacji w _mutation {after_mutation[-1]}")
            else:
                after_mutation.append(individual)
        return after_mutation

    def _mutate_individual(self, individual):
        for _ in range(self.mutation_attempts):
            new_individual = self._change_color(deepcopy(individual))
            if self.is_valid(new_individual.coloring):
                # print(f"Nowy w mutate individual {new_individual}")
                new_individual.update_num_of_colors()
                # print(f"Dodaje do rozw {self.is_valid(new_individual.coloring)}")
                return new_individual
        return individual

    def _change_color(self, individual):
        for vert in range(self.vertices_num):
            if random.uniform(0.0, 1.0) < self.mutate_gene_probability:
                # print("Muruje koloj")
                individual.coloring[vert] = (vert + 1, random.randint(1, self.vertices_num))

        return individual


if __name__ == '__main__':
    Colors = Coloring("dane.txt")
    # Colors.greedy_coloring()
    # print(Colors.graph.vertices)
    Colors.genetic_coloring()
    # print(Colors.greedy_get_greatest_color())

# ([(1, 29), (2, 9), (3, 1), (4, 15), (5, 11), (6, 2), (7, 23), (8, 17), (9, 4), (10, 20), (11, 18), (12, 20), (13, 18), (14, 14), (15, 6), (16, 13), (17, 6), (18, 7), (19, 31), (20, 11), (21, 26), (22, 13), (23, 16), (24, 26), (25, 4), (26, 26), (27, 26), (28, 13), (29, 20), (30, 27), (31, 7), (32, 17), (33, 16)], 19)
