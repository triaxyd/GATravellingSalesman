import math
import rustworkx as rx
import random
from collections import Counter

CITIES = 5
POPULATION = 10
MUTATION_RATE = 0.01
NUM_OF_GENERATIONS = 1000


def define_graph():
    #create graph
    G = rx.PyGraph()
    #create city nodes
    indices = G.add_nodes_from(range(CITIES))
    #create weighted edges
    G.add_edge(0,1,10)
    G.add_edge(0,2,20)
    G.add_edge(0,3,5)
    G.add_edge(0,4,10)
    G.add_edge(1,2,2)
    G.add_edge(1,3,10)
    G.add_edge(1,4,6)
    G.add_edge(2,3,7)
    G.add_edge(2,4,1)
    G.add_edge(3,4,20)
    return G


#create a random population as a list containing possible list solutions
def random_population(nodes):
    population = []
    for _ in range(POPULATION):
        middle_nodes = nodes[1 : CITIES]
        random.shuffle(middle_nodes)
        sequence = [0] + middle_nodes + [0]
        population.append(sequence)
    return population


#evaluate the fitness score of a chromosome
def fitness_function(chromosome):
    cost = 0
    for i in range(len(chromosome) - 1):
        edge_weight = G.get_edge_data(chromosome[i],chromosome[i + 1])
        if edge_weight is not None:
            cost += edge_weight
        else:
            cost += float(math.inf)
    return cost


def calculate_cumulative_probability(fitness_list):
    relative_fitness = []
    cumulative_probabilities = []
    previous_probability = 0.0
    sum_of_fitness = sum(fitness_list)
    relative_fitness = [fitness / sum_of_fitness for fitness in fitness_list]
    for fitness in relative_fitness:
        cumulative_probabilities.append(previous_probability+fitness)
        previous_probability+=fitness
    return cumulative_probabilities


def roulette_wheel_selection(population, fitness_list):
    cumulative_probabilities = calculate_cumulative_probability(fitness_list)
    mating_pool = []    

    for _ in range(len(population)):
        r = random.uniform(0, 1)
        for j, probability in enumerate(cumulative_probabilities):
            if r <= probability:
                mating_pool.append(population[j])
                break

    return mating_pool


    
def single_point_crossover(chromosome1,chromosome2):
    #middle of sequences without 0
    chromosome1_middle = [gene for gene in chromosome1][1:len(chromosome1)-1]
    chromosome2_middle = [gene for gene in chromosome2][1:len(chromosome2)-1]
    print(chromosome1)
    print(chromosome2)
    print(chromosome1_middle)
    print(chromosome2_middle)
    
    single_point = random.randint(0, len(chromosome1_middle) - 1)
    print(single_point)
    descendant1_middle = chromosome1_middle[:single_point] + chromosome2_middle[single_point:]
    descendant2_middle = chromosome2_middle[:single_point] + chromosome1_middle[single_point:]


    def repair(descendant_middle):
        cities = [i+1 for i in range(CITIES-1)]
        occurances = Counter(descendant_middle)        
        print(occurances)

    #repair possible duplicates, missing genes
    repaired_chromosome1 = repair(descendant1_middle)
    repaired_chromosome2 = repair(descendant2_middle)

    #reconstruct
    descendant1 = [0] + descendant1_middle + [0]
    descendant2 = [0] + descendant2_middle + [0]
    return descendant1,descendant2




def mutation(chromosome):
    return 0






def genetic_algorithm(population,mutation_rate,generations):
    fitness_values = [fitness_function(chrom) for chrom in population]
    selected_population = roulette_wheel_selection(population,fitness_values)

    

    
    


if __name__ == "__main__":

    G = define_graph()
    population = random_population(G.nodes())

    final = genetic_algorithm(population,MUTATION_RATE,NUM_OF_GENERATIONS)
    print(population)
    print(single_point_crossover(population[0],population[1]))
    
    
    
    
    


