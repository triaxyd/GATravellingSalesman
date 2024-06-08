import rustworkx as rx
import matplotlib.pyplot as plt
import math
import random
import time

 
def define_graph(cities: int):
    #create graph
    G = rx.PyGraph()
    #create city nodes
    G.add_nodes_from(range(cities))
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
def random_population(nodes: list, population_size:int):
    population = []

    for _ in range(population_size):
        middle_nodes = nodes[1:]
        random.shuffle(middle_nodes)
        sequence = [0] + middle_nodes + [0] #starting and ending node is always city 0
        population.append(sequence)
    print(population)
    
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
    cumulative_probability = []
    probability = 0.0
    sum_of_fitness = sum(fitness_list)
    relative_fitness_list = [fitness / sum_of_fitness for fitness in fitness_list]

    for relative_fitness in relative_fitness_list:
        probability+=relative_fitness
        cumulative_probability.append(probability)

    return cumulative_probability



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


#we need crossover that will not generate invalid chromosomes because of nature of TSP
def crossover_function(parent1: list,parent2: list):
    #middle of sequences without 0
    parent1_middle = [gene for gene in parent1][1:len(parent1)-1]
    parent2_middle = [gene for gene in parent2][1:len(parent2)-1]
    #the random crossover point
    crossover_point = random.randint(1, len(parent1_middle) - 1)    
    
    #create the first part of each offspring
    #parent1_first_part = parent1_middle[:crossover_point]
    #parent2_first_part = parent2_middle[:crossover_point]
    parent1_last_part = parent1_middle[crossover_point:]
    parent2_last_part = parent2_middle[crossover_point:]


    #first part of offspring1 -> cities in parent2 that are not in the last part of parent1
    offspring1_first_part = [gene for gene in parent2_middle if gene not in parent1_last_part]

    #last part of offspring1 -> remaining cities from last part of parent1, but shuffled
    offspring1_last_part = parent1_last_part

    random.shuffle(offspring1_last_part)

    offspring1 = offspring1_first_part + offspring1_last_part

    #first part of offspring2 -> cities in parent1 that are not in the last part of parent2
    offspring2_first_part = [gene for gene in parent1_middle if gene not in parent2_last_part]

    #last part of offspring2 -> remaining cities from last part of parent2, but shuffled
    offspring2_last_part = parent2_last_part
    random.shuffle(offspring2_last_part)
    
    offspring2 = offspring2_first_part + offspring2_last_part
    
    #reconstruct
    offspring1 = [0] + offspring1 + [0]
    offspring2 = [0] + offspring2 + [0]

    return offspring1,offspring2



#mutation of a chromosome, change 2 genes in the chromosome with probability of mutation rate
def mutation(chromosome: list, mutation_rate: float):
    chromosome_middle = [gene for gene in chromosome][1:len(chromosome) - 1]

    #if random point < mutation rate then do mutation
    if random.random() < mutation_rate:
        #get the indexes of the 2 randomly selected positions that will be exchanged
        idx1, idx2 = random.sample(range(len(chromosome_middle)), 2)
        #change the values
        chromosome_middle[idx1], chromosome_middle[idx2] = chromosome_middle[idx2], chromosome_middle[idx1]

    return [0] + chromosome_middle + [0]





#genetic algorithm implementation
def genetic_algorithm(G, POPULATION_SIZE: int,  MUTATION_RATE: float, NUM_OF_GENERATIONS: int, ELITE_SIZE: int):
    #keep track of statistics
    fitness_over_time = {
        "best": [],
        "average": [],
        "worst": []
    }

    #keep track of time execution
    start_time = time.time()

    #initialize random population
    population = random_population(G.nodes(),POPULATION_SIZE)

    for generation in range(NUM_OF_GENERATIONS):

        fitness_values = [fitness_function(chrom) for chrom in population]
        
        best_fitness = min(fitness_values)
        average_fitness = sum(fitness_values) / len(fitness_values)
        worst_fitness = max(fitness_values)
        
        fitness_over_time["best"].append(best_fitness)
        fitness_over_time["average"].append(average_fitness)
        fitness_over_time["worst"].append(worst_fitness)

        #elitism, we choose which portion of the population will go directly to the next generation without processing
        elite_indices = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k])[:ELITE_SIZE]
        
        #adding fittest chromosomes to the next generation directly -> POPULATION_SIZE - ELITE_SIZE positions left for the population
        next_generation = [population[i] for i in elite_indices]

        #select chromosomes from the whole population
        selected_population = roulette_wheel_selection(population,fitness_values)

        #while the length of the next generation is less than the population size, do crossover and mutation 
        while len(next_generation) < POPULATION_SIZE:
            #randomly select 2 parents for crossover
            parent1, parent2 = random.sample(selected_population, 2)
            #get the 2 offsprings from the 2 parents
            offspring1, offspring2 = crossover_function(parent1, parent2)

            #mutate over the offsprings
            offspring1 = mutation(offspring1, MUTATION_RATE)
            offspring2 = mutation(offspring2, MUTATION_RATE)

            #add the offsprings to the next generation population
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(offspring1)

            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(offspring2)

        population = next_generation
    

    print("--- %s seconds ---" % (time.time() - start_time))

    plt.plot(fitness_over_time["best"], label='Best Fitness', color='green')
    plt.plot(fitness_over_time["average"], label='Average Fitness', color='blue')
    plt.plot(fitness_over_time["worst"], label='Worst Fitness', color='red')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

    return population
        



    



if __name__ == "__main__":

    CITIES = 5
    POPULATION_SIZE = 5
    ELITE_SIZE = 1
    MUTATION_RATE = 0.1
    NUM_OF_GENERATIONS = 50

    G = define_graph(CITIES)
    final_population = genetic_algorithm(G, POPULATION_SIZE, MUTATION_RATE, NUM_OF_GENERATIONS, ELITE_SIZE)
    print(f"Final Population: {final_population}")


    


    
    
    
    
    
    


