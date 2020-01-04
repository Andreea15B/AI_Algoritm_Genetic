import numpy as np
import copy as cp
import random as rand

POPULATION_SIZE = 1000
CHROMOSOME_SIZE = 10
MIN_ALLELE_VALUE = 99999
MAX_ALLELE_VALUE = -99999
MAX_NUM_OF_GENERATIONS = 100000
K_SELECTED_CHROMOSOMES = 1000
TARGET_TOTAL_FITNESS = 0

class Chromosome:
    def __init__(self, generation):
        self.generation = generation
        self.genes = []
        for _ in range(CHROMOSOME_SIZE):
            allele = np.random.uniform(MIN_ALLELE_VALUE, MAX_ALLELE_VALUE)
            self.genes.append(allele)
        self.fitness = fitness_function(self)

    def get_genes(self):
        return self.genes

    def set_genes(self, genes):
        self.genes = []
        self.genes = cp.deepcopy(genes)
        return None

    def get_fitness(self):
        return self.fitness

    def get_generation(self):
        return self.generation

class Population:
    def __init__(self, size, generation):
        self.size = size
        self.chromosomes = []
        self.total_fitness = 0
        for _ in range(self.size):
            chromosome = Chromosome(generation)
            self.chromosomes.append(chromosome)
            self.total_fitness = self.total_fitness + chromosome.get_fitness()

    def get_chromosomes(self):
        return self.chromosomes

    def set_chromosomes(self, chromosomes):
        self.chromosomes = []
        self.chromosomes = cp.deepcopy(chromosomes)
        return None

    def get_size(self):
        return self.size

    def set_size(self, size):
        self.size = size
        return None

    def get_total_fitness(self):
        return self.total_fitness

    def set_total_fitness(self, total_fitness):
        self.total_fitness = total_fitness
        return None

    def compute_total_fitness(self):
        total_fitness = 0
        for it in range(self.size):
            total_fitness += self.chromosomes[it].get_fitness()
        return total_fitness

class GeneticOperators:
    def __init__(self):
        pass

    # Selection
    def tournament_population_selection(self, population):
        J_SELECTED_CHROMOSOMES = np.random.randint(K_SELECTED_CHROMOSOMES / 4, K_SELECTED_CHROMOSOMES + 1)

        tournament_population = Population(J_SELECTED_CHROMOSOMES, 0)
        slected_chromosomes_indices = np.random.randint(0, population.get_size(), size = K_SELECTED_CHROMOSOMES)

        for it in range(K_SELECTED_CHROMOSOMES):
            tournament_population.get_chromosomes().append(population.get_chromosomes()[slected_chromosomes_indices[it]])

        tournament_population.get_chromosomes().sort(key = lambda x: x.get_fitness(), reverse = True)
        tournament_population.set_chromosomes(tournament_population.get_chromosomes()[0 : J_SELECTED_CHROMOSOMES])

        return tournament_population

    # Crossover
    def single_point_crossover(self, first_chromosome, second_chromosome, generation):
        if ((first_chromosome.get_generation() == second_chromosome.get_generation()) and
            (first_chromosome.get_generation() == generation - 1) and
            (second_chromosome.get_generation() == generation - 1)):

            first_offspring = Chromosome(generation)
            second_offspring = Chromosome(generation)

            cutting_point = rand.randrange(CHROMOSOME_SIZE)

            for it in range(CHROMOSOME_SIZE):
                if it <= cutting_point:
                    first_offspring.get_genes()[it] = first_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                else:
                    first_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = first_chromosome.get_genes()[it]

            return (first_offspring, second_offspring)
        else:
            return None

    # Mutation
    def uniform_mutation(self, chromosome):
        rand_gene = rand.randrange(CHROMOSOME_SIZE)
        chromosome.get_genes()[rand_gene] = np.random.uniform(MIN_ALLELE_VALUE, MAX_ALLELE_VALUE)
        return chromosome

def evolve(population, generation):
    new_population = Population(0, generation)
    genetic_operators = GeneticOperators()
    tournament_population = genetic_operators.tournament_population_selection(population)

    new_chromosomes = []
    new_size = 0
    for first_it in range(0, tournament_population.get_size()):
        for second_it in range(first_it + 1, tournament_population.get_size()):
            first_offspring, second_offspring = genetic_operators.single_point_crossover(
                                                    tournament_population.get_chromosomes()[first_it],
                                                    tournament_population.get_chromosomes()[second_it],
                                                    generation)
            
            first_offspring = genetic_operators.uniform_mutation(first_offspring)
            second_offspring = genetic_operators.uniform_mutation(second_offspring)

            if None != first_offspring:
                new_chromosomes.append(first_offspring)
                new_size += 1

            if None != second_offspring:
                new_chromosomes.append(second_offspring)
                new_size += 1

        if new_size > POPULATION_SIZE:
            break

    if new_size == 0:
        print("Err @ Population None!")

    new_population.set_chromosomes(new_chromosomes)
    new_population.set_size(new_size)
    new_population.set_total_fitness(new_population.compute_total_fitness())

    return new_population

def fitness_function(chromosome):
    fitness = 0
    for allele in chromosome.get_genes():
        fitness += allele
    return fitness

def print_population(population, generation):
    print("\n---------------------------------")
    print("Generation #{0} | Size: {1} | Total Fitness: {2} | Fittest chromosome fitness: {3}".format(
        generation,
        population.get_size(),
        population.get_total_fitness(),
        population.get_chromosomes()[0].get_fitness()))
    print("\n---------------------------------")

if __name__ == '__main__':
    LOGS_FILE_PATH = "logs.txt"

    with open(LOGS_FILE_PATH, "w+") as fd:
        population = Population(POPULATION_SIZE, 0)

        population.get_chromosomes().sort(key = lambda chromosome: chromosome.get_fitness(), reverse = True)
        print_population(population, 0)

        generation = 1
        while (generation < MAX_NUM_OF_GENERATIONS) or (population.get_chromosomes()[0].get_fitness() == MAX_ALLELE_VALUE):
            population = evolve(population, generation)
            population.get_chromosomes().sort(key = lambda chromosome: chromosome.get_fitness(), reverse = True)
            print_population(population, generation)
            fd.write(
                "Generation #{0} | Size: {1} | Total Fitness: {2} | Fittest chromosome fitness: {3}\n".format(
                    generation,
                    population.get_size(),
                    population.get_total_fitness(),
                    population.get_chromosomes()[0].get_fitness()
                )
            )
            generation += 1
