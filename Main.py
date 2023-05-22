import sys

# population_size muss minimum tournament size sein
import numpy as np
import matplotlib.pyplot as plt
import random


# population_size muss minimum tournament size sein
population_size = 100
mutation_rate = 0.5
refraction_rate = 0.5
num_generations = 50
selection_size = 10

def generate_Berlin52_List():
    # input file --> Open
    infile = open('berlin52.tsp', 'r')

    # Aufteilen der Infozeilen von Berlin52
    name = infile.readline().strip().split()[1]
    filetype = infile.readline().strip().split()[1]
    comment = infile.readline().strip().split()[1]
    dimension = infile.readline().strip().split()[1]
    edgeweighttype = infile.readline().strip().split()[1]
    infile.readline()

    # Erstellen der Liste mit den Koordinaten Tupeln, HIER Tupel = float
    cities = []
    N = int(dimension)
    for i in range(0, N):
        x, y = infile.readline().strip().split()[1:]
        cities.append((float(x), float(y)))

    # input file --> Close
    infile.close()
    return cities


def generate_ST70_List():
    # input file --> Open
    infile = open('st70.tsp', 'r')

    # Aufteilen der Infozeilen von st70
    name = infile.readline().strip().split()[1]
    filetype = infile.readline().strip().split()[1]
    comment = infile.readline().strip().split()[1]
    dimension = infile.readline().strip().split()[1]
    edgeweighttype = infile.readline().strip().split()[1]
    infile.readline()

    # Erstellen der Liste mit den Koordinaten Tupeln, HIER Tupel = int
    cities = []
    N = int(dimension)
    for i in range(0, N):
        x, y = infile.readline().strip().split()[1:]
        cities.append((int(x), int(y)))

    # input file --> Close
    infile.close()
    return cities

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
            # Euklidischer Abstand
            total_distance += np.linalg.norm(np.array(city1) - np.array(city2))

        city1 = self.cities[0]
        city2 = self.cities[num_cities - 1]
        total_distance += np.linalg.norm(np.array(city2) - np.array(city1))
        return total_distance


# Erzeugen einer zufälligen Route bei einem zufälligem Startpunkt
def generate_random_route(cities):
    route = Route(cities)
    # Stätte in der Liste werden neu angeordnet
    # random.shuffle(route.cities)
    new_cities = random.sample(route.cities, len(cities))
    return Route(new_cities)


def generate_random_population(cities, population_size):
    population = []
    for _ in range(population_size):
        route = generate_random_route(cities)
        # append fügt einer Liste ein element hinzu, KEINE KONTROLLE
        population.append(route)
    return population


def selection(population, selection_size):
    selected_parents = []
    tournament = random.sample(population, selection_size)
    index = None
    best_route = None
    j = 0
    while j < 2:
        for i in range(len(tournament) - 1):
            if best_route is None:
                index = 0
                best_route = tournament[0]
            elif tournament[i].distance < best_route.distance:
                index = i
                best_route = tournament[i]
        selected_parents.append(best_route)
        del tournament[index]
        index = None
        best_route = None
        j += 1
    return selected_parents


from collections import deque

def refraction(parent1, parent2):
    size = len(parent1.cities)
    child1_cities = [None]*size
    child2_cities = [None]*size
    if random.random() > refraction_rate:
        return  parent1,parent2

    # Choose random start/end indices for crossover
    start, end = sorted(random.sample(range(size), 2))

    # Copy the segment from parent1 to child1 and vice versa
    child1_cities[start:end] = parent1.cities[start:end]
    child2_cities[start:end] = parent2.cities[start:end]

    # Create sets for fast lookup
    child1_set = set(child1_cities[start:end])
    child2_set = set(child2_cities[start:end])

    # Create queues of the remaining cities in the order they appear in the parents
    parent1_remaining = deque(x for x in parent2.cities if x not in child1_set)
    parent2_remaining = deque(x for x in parent1.cities if x not in child2_set)

    # Fill the remaining positions in the children’s route
    for i in list(range(start)) + list(range(end, size)):
        child1_cities[i] = parent1_remaining.popleft()
        child2_cities[i] = parent2_remaining.popleft()

    return Route(child1_cities), Route(child2_cities)

def mutation(route, mutation_rate):
    cities = route.cities
    if random.random() < mutation_rate and len(cities) >= 2:
        indices = random.sample(range(len(cities)), 2)
        cities[indices[0]], cities[indices[1]] = cities[indices[1]], cities[indices[0]]
        route.distance = route.calculate_distance()


def test_Route(route, original):
    cities = route.cities
    counter = len(original)
    test_list = []
    for i in cities:
        if i in original and i not in test_list:
            counter -= 1
            test_list.append(i)

    if counter == 0:
        return True
    else:
        return False


# Evolutionärer Algorithmus für das TSP
def tsp_evolutionary_algorithm(cities, population_size, num_generations, selection_size, mutation_rate):
    population = generate_random_population(cities, population_size)
    best_distances = []
    best_route = None

    for generation in range(num_generations):
        new_population = []

        while len(new_population) < population_size:
            parents = selection(population, selection_size)
            child1, child2 = refraction(parents[0], parents[1])
            mutation(child1, mutation_rate)
            mutation(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population
        best_route = min(population, key=lambda route: route.distance)
        best_distances.append(best_route.distance)
        print(best_route.cities)
        if test_Route(best_route, cities):
            print("Generation:", generation + 1, "Beste Strecke:", best_route.distance)
        else:
            print("Keine gültige Route wurde gebildet")

    return best_route, best_distances


berlin52 = generate_Berlin52_List()

st70 = generate_ST70_List()

# Evolutionärer Algorithmus für Testinstanz berlin52
#best_route_berlin52, best_distances_berlin52 =  tsp_evolutionary_algorithm(berlin52, population_size, num_generations, selection_size, mutation_rate)

# Evolutionärer Algorithmus für Testinstanz st70
best_route_st70, best_distances_st70 = tsp_evolutionary_algorithm(st70, population_size, num_generations,
                                                                  selection_size,
                                                                  mutation_rate)

# Ausgabe der Ergebnisse
# print("Beste Strecke für berlin52:", best_route_berlin52.distance)
# print("Beste Strecke für st70:", best_route_st70.distance)

# Darstellung der Konvergenz der besten Strecke
#plt.plot(best_distances_berlin52, label="berlin52")
plt.plot(best_distances_st70, label="st70")
plt.xlabel("Generation")
plt.ylabel("Beste Strecke")
plt.legend()
plt.show()
