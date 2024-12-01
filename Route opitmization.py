import numpy as np
import matplotlib.pyplot as plt
import heapq


class Graph:
    def __init__(self, vertices):
        self.V = vertices  
        self.graph = {}

    def add_edge(self, u, v, w):
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        self.graph[u].append((v, w))
        self.graph[v].append((u, w))  

    def get_vertices(self):
        return list(self.graph.keys())

    def dijkstra(self, start, end):
        distances = {vertex: float('infinity') for vertex in self.graph}
        distances[start] = 0
        priority_queue = [(0, start)]
        previous_vertices = {vertex: None for vertex in self.graph}

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_distance > distances[current_vertex]:
                continue

            for neighbor, weight in self.graph[current_vertex]:
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_vertices[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (distance, neighbor))

        path = self.reconstruct_path(previous_vertices, start, end)
        return path, distances[end] if path else float('infinity')

    def reconstruct_path(self, previous_vertices, start, end):
        path = []
        current_vertex = end
        while current_vertex is not None:
            path.append(current_vertex)
            current_vertex = previous_vertices[current_vertex]
            if current_vertex == start:
                path.append(start)
                break
        if not path or path[-1] != start:
            return []
        return path[::-1]

    def a_star(self, start, end, heuristic):
        open_set = {start}
        came_from = {}
        g_score = {vertex: float('infinity') for vertex in self.graph}
        g_score[start] = 0
        f_score = {vertex: float('infinity') for vertex in self.graph}
        f_score[start] = heuristic[start]

        while open_set:
            current = min(open_set, key=lambda vertex: f_score[vertex])

            if current == end:
                path = self.reconstruct_path(came_from, start, end)
                return path, g_score[end]

            open_set.remove(current)

            for neighbor, weight in self.graph[current]:
                tentative_g_score = g_score[current] + weight

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic[neighbor]
                    open_set.add(neighbor)

        return [], float('infinity')

    def greedy(self, start, end, heuristic):
        open_set = {start}
        came_from = {}
        g_score = {vertex: float('infinity') for vertex in self.graph}
        g_score[start] = 0

        while open_set:
            current = min(open_set, key=lambda vertex: heuristic[vertex])

            if current == end:
                path = self.reconstruct_path(came_from, start, end)
                return path, g_score[current]

            open_set.remove(current)

            for neighbor, weight in self.graph[current]:
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    g_score[neighbor] = g_score[current] + weight
                    open_set.add(neighbor)

        return [], float('infinity')

def visualize_path(graph, path, start, end, algorithm_name, path_cost, is_best=False):
    plt.figure(figsize=(12, 8))
    
    # Plot all edges in the graph
    for u in graph.graph:
        for v, w in graph.graph[u]:
            plt.plot([u[0], v[0]], [u[1], v[1]], 'gray', alpha=0.3, linestyle='--')

    # Plot vertices
    for vertex in graph.graph.keys():
        if vertex != start and vertex != end:
            plt.scatter(vertex[0], vertex[1], color='blue', s=60)
            plt.text(vertex[0] + 0.1, vertex[1] + 0.1, str(vertex), fontsize=10)

    # Highlight start and end points
    plt.scatter(start[0], start[1], color='green', label='Start', s=100 , zorder=5)
    plt.scatter(end[0], end[1], color='red', label='End', s=100, zorder=5)

    # Plot the path if it exists
    if path:
        path_color = 'purple' if is_best else 'red'
        # Draw the path with arrows
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            plt.arrow(p1[0], p1[1],
                     p2[0] - p1[0], p2[1] - p1[1],
                     head_width=0.1, head_length=0.2,
                     fc=path_color, ec=path_color,
                     length_includes_head=True,
                     zorder=4)

        # Add path sequence labels
        for i, point in enumerate(path):
            plt.text(point[0] - 0.2, point[1] - 0.3, 'Step %d' % (i+1),
                    fontsize=8, color='darkred')

    # Add title with algorithm name and path cost
    title = '%s Path Finding\nTotal Path Cost: %.2f' % (algorithm_name, path_cost)
    if is_best:
        title += ' (Best Path!)'
    plt.title(title, pad=20)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    # Add path sequence as text at the bottom
    if path:
        path_str = " -> ".join([str(p) for p in path])
        plt.figtext(0.5, 0.02, 'Path Sequence: %s' % path_str,
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                   wrap=True)

    plt.tight_layout()
    plt.show()

def print_available_vertices(graph):
    print("\nAvailable vertices:")
    vertices = graph.get_vertices()
    for i, vertex in enumerate(vertices):
        print(f"{i + 1}. {vertex}")
    return vertices

def get_vertex_input(prompt, vertices):
    while True:
        try:
            print(prompt)
            idx = int(input("Enter the number of your choice: ")) - 1
            if 0 <= idx < len(vertices):
                return vertices[idx]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

# Define the graph with coordinates
graph = Graph(vertices=[
    (0, 0), (1, 2), (2, 1), (3, 3), (4, 0),
    (5, 2), (6, 1), (7, 3), (8, 2)
])

# Define edges with weights (distances)
edges = [
    ((0, 0), (1, 2), 2.24),
    ((0, 0), (2, 1), 2.24),
    ((0, 0), (4, 0), 4.0),
    ((1, 2), (3, 3), 1.41),
    ((1, 2), (4, 0), 1.41),
    ((2, 1), (3, 3), 2.24),
    ((2, 1), (5, 2), 2.24),
    ((3, 3), (7, 3), 3.0),
    ((4, 0), (5, 2), 2.24),
    ((5, 2), (6, 1), 1.41),
    ((6, 1), (7, 3), 2.24),
    ((7, 3), (8, 2), 1.41)
]

# Add edges to the graph
for edge in edges:
    graph.add_edge(edge[0], edge[1], edge[2])

# Get user input for start and end points
vertices = print_available_vertices(graph)
start = get_vertex_input("\nSelect the starting point:", vertices)
end = get_vertex_input("Select the destination point:", vertices)

# Calculate heuristic (Euclidean distance to end)
heuristic = {}
for vertex in graph.get_vertices():
    dx = vertex[0] - end[0]
    dy = vertex[1] - end[1]
    heuristic[vertex] = np.sqrt(dx*dx + dy*dy)

# Run all algorithms and store results
results = []

# Dijkstra
dijkstra_path, dijkstra_cost = graph.dijkstra(start, end)
results.append(("Dijkstra's", dijkstra_path, dijkstra_cost))

# A*
a_star_path, a_star_cost = graph.a_star(start, end, heuristic)
results.append(("A*", a_star_path, a_star_cost))

# Greedy
greedy_path, greedy_cost = graph.greedy(start, end, heuristic)
results.append(("Greedy", greedy_path, greedy_cost))

# Find the best path (lowest cost)
best_algorithm = min(results, key=lambda x: x[2])
print("\nResults:")
print("-" * 50)
for algo, path, cost in results:
    print(f"{algo}: Cost = {cost:.2f}")
print(f"\nBest algorithm: {best_algorithm[0]} with cost {best_algorithm[2]:.2f}")

# Visualize the best path only
visualize_path(graph, best_algorithm[1], start, end, best_algorithm[0], best_algorithm[2], is_best=True)