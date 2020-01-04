import numpy as np
import copy as cp
import random as rand

POPULATION_SIZE = 1000
CHROMOSOME_SIZE = 10
MIN_ALLELE_VALUE = -9999999
MAX_ALLELE_VALUE = 9999999
MAX_NUM_OF_GENERATIONS = 100000
K_SELECTED_CHROMOSOMES = 1000
BEST_FITNESS = 99999999

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

    def two_points_crossover(self, first_chromosome, second_chromosome, generation):
        if ((first_chromosome.get_generation() == second_chromosome.get_generation()) and
            (first_chromosome.get_generation() == generation - 1) and
            (second_chromosome.get_generation() == generation - 1)):

            first_offspring = Chromosome(generation)
            second_offspring = Chromosome(generation)

            first_cutting_point = rand.randrange(1, CHROMOSOME_SIZE - 3)
            second_cutting_point = rand.randrange(first_cutting_point + 2, CHROMOSOME_SIZE - 1)

            for it in range(CHROMOSOME_SIZE):
                if (it <= first_cutting_point) or (it >= second_cutting_point):
                    first_offspring.get_genes()[it] = first_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                else:
                    first_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = first_chromosome.get_genes()[it]

            return (first_offspring, second_offspring)
        else:
            return None

    def uniform_crossover(self, first_chromosome, second_chromosome, generation):
        if ((first_chromosome.get_generation() == second_chromosome.get_generation()) and
            (first_chromosome.get_generation() == generation - 1) and
            (second_chromosome.get_generation() == generation - 1)):

            first_offspring = Chromosome(generation)
            second_offspring = Chromosome(generation)

            for it in range(CHROMOSOME_SIZE):
                rand_value = rand.random()
                if rand_value < 0.5:
                    first_offspring.get_genes()[it] = first_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                else:
                    first_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = first_chromosome.get_genes()[it]

            return (first_offspring, second_offspring)
        else:
            return None

    def single_arithmetic_crossover(self,first_chromosome, second_chromosome,generation):
        if ((first_chromosome.get_generation() == second_chromosome.get_generation()) and
            (first_chromosome.get_generation() == generation - 1) and
            (second_chromosome.get_generation() == generation - 1)):

            first_offspring = Chromosome(generation)
            second_offspring = Chromosome(generation)

            alpha = rand.random()
            rand_gene = rand.randrange(CHROMOSOME_SIZE)

            for it in range(CHROMOSOME_SIZE):
                if it == rand_gene:
                    first_offspring.get_genes()[it] = (alpha * first_chromosome.get_genes()[it] + (1 - alpha) * second_chromosome.get_genes()[it])
                    second_offspring.get_genes()[it] = (alpha * second_chromosome.get_genes()[it] + (1 - alpha) * first_chromosome.get_genes()[it])
                else:
                    first_offspring.get_genes()[it] = first_chromosome.get_genes()[it]
                    second_offspring.get_genes()[it] = second_chromosome.get_genes()[it]

            return (first_offspring, second_offspring)
        else:
            return None

    def ring_crossover(self, first_chromosome, second_chromosome,generation):
        if ((first_chromosome.get_generation() == second_chromosome.get_generation()) and
            (first_chromosome.get_generation() == generation - 1) and
            (second_chromosome.get_generation() == generation - 1)):

            first_offspring = Chromosome(generation)
            second_offspring = Chromosome(generation)

            cutting_point = rand.randrange(CHROMOSOME_SIZE)

            for it in range(CHROMOSOME_SIZE):
                if it < cutting_point:
                    first_offspring.get_genes()[it] = first_chromosome.get_genes()[it]
                else:
                    second_offspring.get_genes()[it] = first_chromosome.get_genes()[it]

            for it in range(CHROMOSOME_SIZE):
                if it < CHROMOSOME_SIZE - cutting_point:
                    first_offspring.get_genes()[it] = second_chromosome.get_genes()[it]
                else:
                    second_offspring.get_genes()[it] = second_chromosome.get_genes()[it]

            return (first_offspring, second_offspring)
        else:
            return None

    # Mutation
    def uniform_mutation(self, chromosome):
        rand_gene = rand.randrange(CHROMOSOME_SIZE)
        chromosome.get_genes()[rand_gene] = np.random.uniform(MIN_ALLELE_VALUE, MAX_ALLELE_VALUE)
        return chromosome

    def swap_mutation(self, chromosome):
        first_rand_gene = rand.randrange(CHROMOSOME_SIZE)
        second_rand_gene = rand.randrange(CHROMOSOME_SIZE)

        while first_rand_gene == second_rand_gene:
            second_rand_gene = rand.randrange(CHROMOSOME_SIZE)

        aux = chromosome.get_genes()[first_rand_gene]
        chromosome.get_genes()[first_rand_gene] = chromosome.get_genes()[second_rand_gene]
        chromosome.get_genes()[second_rand_gene] = aux

        return chromosome

    def scramble_mutation(self, chromosome):
        first_rand_gene = rand.randrange(CHROMOSOME_SIZE)
        second_rand_gene = rand.randrange(CHROMOSOME_SIZE)

        while second_rand_gene == first_rand_gene:
            second_rand_gene = rand.randrange(CHROMOSOME_SIZE)

        if second_rand_gene < first_rand_gene:
            aux = first_rand_gene
            first_rand_gene = second_rand_gene
            second_rand_gene = aux

        genes = chromosome.get_genes()[first_rand_gene : second_rand_gene]
        rand.shuffle(genes)

        for it in range(first_rand_gene, second_rand_gene):
            chromosome.get_genes()[it] = genes[it - first_rand_gene]

        return chromosome

    def inversion_mutation(self, chromosome):
        first_rand_gene = rand.randrange(CHROMOSOME_SIZE)
        second_rand_gene = rand.randrange(CHROMOSOME_SIZE)

        while first_rand_gene == second_rand_gene:
            second_rand_gene = rand.randrange(CHROMOSOME_SIZE)

        if first_rand_gene > second_rand_gene:
            aux = first_rand_gene
            second_rand_gene = first_rand_gene
            second_rand_gene = aux

        genes = chromosome.get_genes()[first_rand_gene : second_rand_gene]
        genes = genes[::-1]

        for it in range(first_rand_gene, second_rand_gene):
            chromosome.get_genes()[it] = genes[it - first_rand_gene]

        return chromosome

    def gaussion_mutation(self, chromosome):
        mu, sigma = 0, 0.1  # mean and standard deviation
        s = np.random.normal(mu, sigma, 1)

        random_gene = rand.randrange(CHROMOSOME_SIZE)
        chromosome.get_genes()[random_gene] = chromosome.get_genes()[random_gene] + s[0]

        while chromosome.get_genes()[random_gene] > MAX_ALLELE_VALUE:
            s = np.random.normal(mu, sigma, 1)
            chromosome.get_genes()[random_gene] = chromosome.get_genes()[random_gene] + s[0]

        return chromosome

def evolve(population, generation):
    new_population = Population(0, generation)
    genetic_operators = GeneticOperators()
    tournament_population = genetic_operators.tournament_population_selection(population)

    new_chromosomes = []
    new_size = 0
    for first_it in range(0, tournament_population.get_size()):
        for second_it in range(first_it + 1, tournament_population.get_size()):
            offsprings = genetic_operators.ring_crossover(tournament_population.get_chromosomes()[first_it], tournament_population.get_chromosomes()[second_it], generation)

            if (None != offsprings) and (len(offsprings) == 2):
                first_offspring = genetic_operators.gaussion_mutation(offsprings[0])
                new_chromosomes.append(first_offspring)

                second_offspring = genetic_operators.gaussion_mutation(offsprings[1])
                new_chromosomes.append(second_offspring)

                new_size += 2

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

def print_population(population, generation, most_fittest_chromosome):
    print("\n---------------------------------")
    print("Generation #{0} | Size: {1} | Total Fitness: {2} | Fittest chromosome fitness: {3} | Most fittest chromosome generation #{4} and fitness: {5}\n".format(
        generation,
        population.get_size(),
        population.get_total_fitness(),
        population.get_chromosomes()[0].get_fitness(),
        most_fittest_chromosome.get_generation(),
        most_fittest_chromosome.get_fitness())
    )
    print("\n---------------------------------")

if __name__ == '__main__':
    LOGS_FILE_PATH = "logs.txt"

    with open(LOGS_FILE_PATH, "w+") as fd:
        population = Population(POPULATION_SIZE, 0)
        most_fittest_chromosome = Chromosome(0)

        population.get_chromosomes().sort(key = lambda chromosome: chromosome.get_fitness(), reverse = True)
        most_fittest_chromosome = cp.deepcopy(population.get_chromosomes()[0])

        print_population(population, 0, most_fittest_chromosome)
        fd.write(
            "Generation #{0} | Size: {1} | Total Fitness: {2} | Fittest chromosome fitness: {3} | Most fittest chromosome generation #{4} and fitness: {5}\n".format(
                0,
                population.get_size(),
                population.get_total_fitness(),
                population.get_chromosomes()[0].get_fitness(),
                most_fittest_chromosome.get_generation(),
                most_fittest_chromosome.get_fitness()
            )
        )

        generation = 1
        while (generation < MAX_NUM_OF_GENERATIONS) and (population.get_chromosomes()[0].get_fitness() < BEST_FITNESS):
            population = evolve(population, generation)
            population.get_chromosomes().sort(key = lambda chromosome: chromosome.get_fitness(), reverse = True)
            if population.get_chromosomes()[0].get_fitness() > most_fittest_chromosome.get_fitness():
                most_fittest_chromosome = cp.deepcopy(population.get_chromosomes()[0])

            print_population(population, generation, most_fittest_chromosome)
            fd.write(
                "Generation #{0} | Size: {1} | Total Fitness: {2} | Fittest chromosome fitness: {3} | Most fittest chromosome generation #{4} and fitness: {5}\n".format(
                    generation,
                    population.get_size(),
                    population.get_total_fitness(),
                    population.get_chromosomes()[0].get_fitness(),
                    most_fittest_chromosome.get_generation(),
                    most_fittest_chromosome.get_fitness()
                )
            )

            generation += 1
