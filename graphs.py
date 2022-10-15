class Vertex:
    def __init__(self, name) -> None:
        self.name = name
        self.color = 0
        self.is_visited = 0

class Graph:
    def __init__(self) -> None:
        self.V = 0
        self.E = 0
        self.edge_list = []
        self.matrix = []
        self.vertices = []

    def __repr__(self) -> str:
        repr = ''
        for i in range(self.V):
            for j in range(self.V):
                repr += str(self.matrix[i][j])+' '
            repr += '\n'
        return repr

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
            self.vertices.append(Vertex(v+1))

        for el in self.edge_list:
            self.matrix[el[0]-1][el[1]-1] = 1
            self.matrix[el[1]-1][el[0]-1] = 1
    
    def create_graph(self, faddr: str = None) -> None:
        self.create_edge_list_from_file(faddr)
        self.create_matrix_from_edge_list()


if __name__ == '__main__':
    G = Graph()
    G.create_graph()
    print(G)