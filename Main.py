# Beispielanwendung des Algorithmus

# Koordinaten der Städte (x, y)
import random
import numpy as np
import matplotlib.pyplot as plt


# Repräsentation einer Route
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
    cities = parent1.cities
    child1 = [None] * len(cities)
    child2 = [None] * len(cities)
    start_index = random.randint(0, len(cities) - 1)
    end_index = random.randint(start_index + 1, len(cities))
    child1[start_index:end_index] = parent1.cities[start_index:end_index]
    child2[start_index:end_index] = parent2.cities[start_index:end_index]
    index = end_index % len(cities)
    while None in child1:
        city = parent2.cities[index]
        if city not in child1:
            child1[child1.index(None)] = city
        index = (index + 1) % len(cities)
    index = end_index % len(cities)
    while None in child2:
        city = parent1.cities[index]
        if city not in child2:
            child2[child2.index(None)] = city
        index = (index + 1) % len(cities)
    return Route(child1), Route(child2)


# Mutation einer Route durch Vertauschen zweier Städte
def mutation(route, mutation_rate):
    cities = route.cities
    if random.random() < mutation_rate:
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


# Beispielanwendung des Algorithmus

# Koordinaten der Städte (x, y)
cities = [(60, 200), (180, 200), (80, 180), (140, 180), (20, 160), (100, 160),
          (200, 160), (140, 140), (40, 120)]
