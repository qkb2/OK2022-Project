# OK2022-Project
This project solves the Graph Coloring Problem (GC) by using the Genetic Algorithm (GA) heuristics. 
Each individual is treated as a sequence of integers to feed into classic Greedy Algorithm. Solutions are provided as those sequences + a decoded sequence of pairs (vertex_name, vertex_color) for all vertices in graph with additional info. about the number of colors used.
This algorithm does NOT on average produce the optimal solution, but seems to get about 1,25χ on larger graphs. It is however pretty fast and reliable.

To use the program create an instance as a .txt file formatted as follows:
first line is the number of vertices in a graph,
each new line is a pair of vertices creating an edge.
Eg. graph G with vertices 1,2,3 and edges <1,2>, <2,3>, <3,1> would be written as:\
3\
1 2\
2 3\
3 1\
Note that the graphs are always assumed to be undirected.

Feel free to use our work if you want to. We'll of course appreciate a shoutout too, if you decide to do so.

Created by qkb2 and wylupek.
