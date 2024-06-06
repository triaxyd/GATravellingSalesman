import math
import rustworkx as rx
import random
from collections import Counter

CITIES = 5
POPULATION = 100
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
def random_population(nodes: list):
    population = []
    for _ in range(POPULATION):
        middle_nodes = nodes[1 : CITIES]
        random.shuffle(middle_nodes)
        sequence = [0] + middle_nodes + [0] #starting and ending node is always city 0
        population.append(sequence)
    return population


#evaluate the fitness score of a chromosome which is the total cost from the starting node 0 to the destination 0
def fitness_function(chromosome: list):
    cost = 0
    for gene in range(len(chromosome) - 1):
        edge_weight = G.get_edge_data(chromosome[gene],chromosome[gene + 1])
        if edge_weight is not None:
            cost += edge_weight
        else:
            cost += float(math.inf)
    return cost


#calculate the cumulative probability 
def calculate_cumulative_probability(fitness_list: list):
    relative_fitness = []
    cumulative_probabilities = []
    probability = 0.0
    sum_of_fitness = sum(fitness_list)
    relative_fitness = [fitness / sum_of_fitness for fitness in fitness_list]
    for fitness in relative_fitness:
        cumulative_probabilities.append(probability+fitness)
        probability+=fitness
    return cumulative_probabilities



#implementing roulette selection
#for the length of the population, we choose random number r and we select 
#the chromosome that has the next greater value than r
def roulette_wheel_selection(population: list, fitness_list: list):
    cumulative_probabilities = calculate_cumulative_probability(fitness_list)
    mating_pool = []    

    for _ in range(len(population)):
        r = random.uniform(0, 1)
        for j, probability in enumerate(cumulative_probabilities):
            if r <= probability:
                mating_pool.append(population[j])
                break

    return mating_pool


#in the tsp problem, offsprings generated might be invalid
#so we need to ensure that there are no duplicates
def crossover_function(parent1: list,parent2: list):
    #middle of sequences without 0
    parent1_middle = [gene for gene in parent1][1:len(parent1)-1]
    parent2_middle = [gene for gene in parent2][1:len(parent2)-1]
    
    print(parent1)
    print(parent2)
    print(parent1_middle)
    print(parent2_middle)

    
    #the random crossover point
    crossover_point = random.randint(1, len(parent1_middle) - 1)
    print(crossover_point)
    
    
    #create the first part of each offspring
    parent1_first_part = parent1_middle[:crossover_point]
    parent2_first_part = parent2_middle[:crossover_point]
    parent1_last_part = parent1_middle[crossover_point:]
    parent2_last_part = parent2_middle[crossover_point:]
    
    print(f"First part of 1: {parent1_first_part}")
    print(f"First part of 2: {parent2_first_part}")
    print(f"Last part of 1: {parent1_last_part}")
    print(f"Last part of 2: {parent2_last_part}")

    #first part of offspring1 -> cities in parent2 that are not in the last part of parent1
    offspring1_first_part = [gene for gene in parent2_middle if gene not in parent1_last_part]
    print(f"First part of offspring 1 : {offspring1_first_part}")
    #last part of offspring1 -> remaining cities from last part of parent1, but shuffled
    offspring1_last_part = parent1_last_part
    print(f"Last part of offspring 1 : {offspring1_last_part}")
    random.shuffle(offspring1_last_part)
    print(f"Last part of offspring 1 after shuffle: {offspring1_last_part}")
    offspring1 = offspring1_first_part + offspring1_last_part

    #first part of offspring2 -> cities in parent1 that are not in the last part of parent2
    offspring2_first_part = [gene for gene in parent1_middle if gene not in parent2_last_part]
    print(f"First part of offspring 2 : {offspring2_first_part}")
    #last part of offspring2 -> remaining cities from last part of parent2, but shuffled
    offspring2_last_part = parent2_last_part
    print(f"Last part of offspring 2 : {offspring2_last_part}")
    random.shuffle(offspring2_last_part)
    print(f"Last part of offspring 2 after shuffle: {offspring2_last_part}")
    offspring2 = offspring2_first_part + offspring2_last_part
    
    #reconstruct
    offspring1 = [0] + offspring1 + [0]
    offspring2 = [0] + offspring2 + [0]

    print(f"Offspring 1: {offspring1}")
    print(f"Offspring 2: {offspring2}")

    return offspring1,offspring2






def mutation(chromosome: list, mutation_rate: float):
    #remove first and final city
    chromosome_middle = [gene for gene in chromosome][1:len(chromosome)-1]
    print(f"chromosome_middle is {chromosome_middle}")

    #choose 2 random points in the chromosome
    idx1, idx2 = random.sample(range(len(chromosome_middle)), 2)
    print(f"idx1: {idx1} and idx2: {idx2}")

    #swap the values of the indexes
    chromosome_middle[idx1], chromosome_middle[idx2] = chromosome_middle[idx2], chromosome_middle[idx1]
    print(f"Swapped indices {idx1} and {idx2} in chromosome_middle: {chromosome_middle}")
    return 0






def genetic_algorithm(population: list, mutation_rate: float, generations: int):

    population = random_population(G.nodes())

    #for generation in range(NUM_OF_GENERATIONS):

    fitness_values = [fitness_function(chrom) for chrom in population]
    selected_population = roulette_wheel_selection(population,fitness_values)
    offspring1, offspring2 = crossover_function(population[0],population[1])
    mutation(offspring1,mutation_rate)
    



    

    
    


if __name__ == "__main__":

    G = define_graph()
    population = random_population(G.nodes())
    #final = genetic_algorithm(population,MUTATION_RATE,NUM_OF_GENERATIONS)

    #print(population)
    #print(crossover_point_crossover(population[0],population[1]))
    genetic_algorithm(population,MUTATION_RATE,NUM_OF_GENERATIONS)
    
    
    
    
    


