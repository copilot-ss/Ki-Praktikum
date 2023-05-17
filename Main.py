import random
import numpy as np
import matplotlib.pyplot as plt

# EA Parameter
population_size = 100
mutation_rate = 0.1
crossover_rate = 0.5
num_generations = 100


# Klasse zur Repräsentation einer Route
class Route:
    def __init__(self, cities):
        self.cities = cities
        self.distance = self.calculate_distance()

    # Berechnung der Gesamtstrecke der Route
    def calculate_distance(self):
        total_distance = 0
        num_cities = len(self.cities)
        for i in range(num_cities - 1):
            city1 = self.cities[i]
            city2 = self.cities[i + 1]
            total_distance += np.linalg.norm(np.array(city1) - np.array(city2))
        return total_distance


# Erzeugung einer zufälligen Route
def generate_random_route(cities):
    route = Route(cities)
    random.shuffle(route.cities)
    return route


# Erzeugung einer zufälligen Population von Routen
def generate_random_population(cities, population_size):
    population = []
    for _ in range(population_size):
        route = generate_random_route(cities)
        population.append(route)
    return population


# Selektion von Eltern für die Rekombination
def selection(population, tournament_size):
    selected_parents = []
    for _ in range(2):
        tournament = random.sample(population, tournament_size)
        winner = min(tournament, key=lambda route: route.distance)
        selected_parents.append(winner)
    return selected_parents


# Rekombination zweier Routen
def crossover(parent1, parent2):
    cities = parent1.cities.copy()
    child1 = cities.copy()
    child2 = cities.copy()
    start_index = random.randint(0, len(cities) - 1)
    end_index = random.randint(start_index + 1, len(cities))
    child1[start_index:end_index] = parent1.cities[start_index:end_index]
    child2[start_index:end_index] = parent2.cities[start_index:end_index]
    index = end_index % len(cities)
    max_attempts = len(cities)

    for _ in range(max_attempts):
        if None not in child1 and None not in child2:
            break

        city1 = parent2.cities[index]
        city2 = parent1.cities[index]

        if None in child1:
            child1[child1.index(city1)] = None

        if None in child2:
            child2[child2.index(city2)] = None

        index = (index + 1) % len(cities)

    return Route(child1), Route(child2)





# Mutation einer Route durch Vertauschen zweier Städte
def mutation(route, mutation_rate):
    cities = route.cities
    if random.random() < mutation_rate and len(cities) >= 2:
        indices = random.sample(range(len(cities)), 2)
        cities[indices[0]], cities[indices[1]] = cities[indices[1]], cities[indices[0]]
        route.distance = route.calculate_distance()



# Evolutionärer Algorithmus für das TSP
def tsp_evolutionary_algorithm(cities, population_size, num_generations, tournament_size, mutation_rate):
    population = generate_random_population(cities, population_size)
    best_distances = []
    best_route = None

    for generation in range(num_generations):
        new_population = []

        while len(new_population) < population_size:
            parents = selection(population, tournament_size)
            child1, child2 = crossover(parents[0], parents[1])
            mutation(child1, mutation_rate)
            mutation(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population
        best_route = min(population, key=lambda route: route.distance)
        best_distances.append(best_route.distance)
        print("Generation:", generation + 1, "Beste Strecke:", best_route.distance)

    return best_route, best_distances


# Testinstanzen aus TSPLIB
# Beispielinstanz: berlin52
berlin52 = [
    (565, 575), (25, 185), (345, 750), (945, 685), (845, 655), (880, 660),
    (25, 230), (525, 1000), (580, 1175), (650, 1130), (1605, 620), (1220, 580),
    (1465, 200), (1530, 5), (845, 680), (725, 370), (145, 665), (415, 635),
    (510, 875), (560, 365), (300, 465), (520, 585), (480, 415), (835, 625),
    (975, 580), (1215, 245), (1320, 315), (1250, 400), (660, 180), (410, 250),
    (420, 555), (575, 665), (1150, 1160), (700, 580), (685, 595), (685, 610),
    (770, 610), (795, 645), (720, 635), (760, 650), (475, 960), (95, 260),
    (875, 920), (700, 500), (555, 815), (830, 485), (1170, 65), (830, 610),
    (605, 625), (595, 360), (1340, 725), (1740, 245)
]

# Beispielinstanz: st70
st70 = [
    (64, 96), (80, 39), (69, 23), (72, 42), (48, 67), (58, 43), (81, 60), (79, 10), (30, 39), (37, 72),
    (29, 10), (7, 23), (2, 64), (64, 96), (80, 39), (69, 23), (72, 42), (48, 67), (58, 43), (81, 60),
    (79, 10), (30, 39), (37, 72), (29, 10), (7, 23), (2, 64), (64, 96), (80, 39), (69, 23), (72, 42),
    (48, 67), (58, 43), (81, 60), (79, 10), (30, 39), (37, 72), (29, 10), (7, 23), (2, 64), (64, 96),
    (80, 39), (69, 23), (72, 42), (48, 67), (58, 43), (81, 60), (79, 10), (30, 39), (37, 72), (29, 10),
    (7, 23), (2, 64), (64, 96), (80, 39), (69, 23), (72, 42), (48, 67), (58, 43), (81, 60), (79, 10),
    (30, 39), (37, 72), (29, 10), (7, 23), (2, 64)
]

# Evolutionärer Algorithmus für Testinstanz berlin52
best_route_berlin52, best_distances_berlin52 = tsp_evolutionary_algorithm(berlin52, population_size, num_generations, 5,
                                                                          mutation_rate)

# Evolutionärer Algorithmus für Testinstanz st70
best_route_st70, best_distances_st70 = tsp_evolutionary_algorithm(st70, population_size, num_generations, 5,
                                                                  mutation_rate)

# Ausgabe der Ergebnisse
print("Beste Strecke für berlin52:", best_route_berlin52.distance)
print("Beste Strecke für st70:", best_route_st70.distance)

# Darstellung der Konvergenz der besten Strecke
plt.plot(best_distances_berlin52, label="berlin52")
plt.plot(best_distances_st70, label="st70")
plt.xlabel("Generation")
plt.ylabel("Beste Strecke")
plt.legend()
plt.show()
