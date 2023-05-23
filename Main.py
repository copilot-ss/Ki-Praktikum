import sys

# population_size muss minimum tournament size sein
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque


# population_size muss minimum tournament size sein
population_size = 50
mutation_rate = 0.2
refraction_rate = 0.8
num_generations = 100
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
def generate_random_route(cities, startingcity):
    # Shuffle nur die anderen Städte
    for i in range(len(cities)-1):
        if cities[i] == startingcity:
            cities.pop(i)
    new_cities = random.sample(cities, len(cities))

    # Fügen Sie den Startpunkt am Anfang hinzu
    new_cities.insert(0, startingcity)
    return Route(new_cities)




def generate_random_population(cities, population_size):
    population = []
    index=random.randint(0,len(cities)-1)
    startingcity = cities[index]
    for _ in range(population_size):
        route = generate_random_route(cities, startingcity)
        population.append(route)
    return population

def selection(population, selection_size):
    tournament = random.sample(population, selection_size)
    selected_parents = []

    for _ in range(2):
        best_route = min(tournament, key=lambda route: route.distance)
        selected_parents.append(best_route)
        tournament.remove(best_route)

    return selected_parents




def crossover(parent1, parent2):
    size = len(parent1.cities)
    child1_cities = [None]*size
    child2_cities = [None]*size
    if random.random() > refraction_rate:
        return  parent1,parent2

    # Random start und end wird gewählt um zufälligen teil aus Parent zu übernehmen
    start, end = sorted(random.sample(range(size), 2))

    child1_cities[start:end] = parent1.cities[start:end]
    child2_cities[start:end] = parent2.cities[start:end]


    # Sets erstellen zum prüfen
    child1_set = set(child1_cities[start:end])
    child2_set = set(child2_cities[start:end])


    # Queues erstellen in der selben reihenfolge wie das parent
    parent1_remaining = deque(x for x in parent2.cities if x not in child1_set )
    parent2_remaining = deque(x for x in parent1.cities if x not in child2_set )

    # Restlichen Cities auffüllen
    for i in list(range(start)) + list(range(end, size)):
        child1_cities[i] = parent1_remaining.popleft()
        child2_cities[i] = parent2_remaining.popleft()

    return Route(child1_cities), Route(child2_cities)

def mutation(route, mutation_rate):
    cities = route.cities[1:]  # schließen Sie den Startpunkt aus
    if random.random() < mutation_rate and len(cities) >= 2:
        indices = random.sample(range(len(cities)), 2)
        cities[indices[0]], cities[indices[1]] = cities[indices[1]], cities[indices[0]]
        route.cities[1:] = cities
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
    plot_route(population[0], "Erste Route") #Plotting
    print("Erste Strecke:", population[0].distance)
    best_distances = []
    best_route = None
    mu = population_size
    lambda_ = 2 * population_size  # Sie können diesen Wert entsprechend ändern

    for generation in range(num_generations):
        new_population = []

        while len(new_population) < lambda_:
            parents = selection(population, selection_size)
            child1, child2 = crossover(parents[0], parents[1])
            mutation(child1, mutation_rate)
            mutation(child2, mutation_rate)
            new_population.extend([child1, child2])

        # $(\mu + \lambda)$ Selektion: wähle die besten $\mu$ aus der Gesamtpopulation (Eltern + Nachkommen)
        combined_population = population + new_population
        combined_population.sort(key=lambda x: x.distance)
        population = combined_population[:mu]

        best_route = min(population, key=lambda route: route.distance)
        best_distances.append(best_route.distance)
        if test_Route(best_route, cities):
        # print("Generation:", generation + 1, "Beste Strecke:", best_route.distance)
            None
        else:
            print("Keine gültige Route wurde gebildet")

    return best_route, best_distances






def plot_route(route, title='Route'):
    # Route als Punkte darstellen
    cities = route.cities
    plt.scatter([city[0] for city in cities], [city[1] for city in cities], color='red')

    # Startpunkt besonders darstellen
    start_city = cities[0]
    plt.scatter(start_city[0], start_city[1], color='green', s=100)  # s kontrolliert die Größe des Punkts

    # Route durch Linien darstellen
    for i in range(-1, len(cities) - 1):
        plt.plot((cities[i][0], cities[i + 1][0]), (cities[i][1], cities[i + 1][1]), 'b-')

    # Titel setzen
    plt.title(title)

    # Anzeigen des Plots
    plt.show()


#testen:

berlin52 = generate_Berlin52_List()

st70 = generate_ST70_List()

# Evolutionärer Algorithmus für Testinstanz berlin52
best_route_berlin52, best_distances_berlin52 =  tsp_evolutionary_algorithm(berlin52, population_size, num_generations, selection_size, mutation_rate)

# Evolutionärer Algorithmus für Testinstanz st70
best_route_st70, best_distances_st70 = tsp_evolutionary_algorithm(st70, population_size, num_generations,
                                                                  selection_size,
                                                                  mutation_rate)

# Ausgabe der Ergebnisse
print("Beste Strecke für berlin52:", best_route_berlin52.distance)
print("Beste Strecke für st70:", best_route_st70.distance)

# Beste Routen grafisch darstellen
plot_route(best_route_berlin52,"Letzte Route")
plot_route(best_route_st70,"Letzte Route")
