import random

POPULATION_SIZE = 8
NR_ELITE_CHROMOSOMES = 1
TOURNAMENT_SELECTION_SIZE = 4
MUTATION_RATE = 0.25
TARGET_CHROMOSOME = [1,1,0,1,0,0,1,1,1,0]

class Chromosome:
    def __init__(self):
        self.genes = []
        self.fitness = 0
        i=0
        while i < TARGET_CHROMOSOME.__len__():
            if random.random() >= 0.5:
                self.genes.append(1)
            else:
                self.genes.append(0)
            i += 1

    def get_genes(self):
        return self.genes

    def get_fitness(self):
        self.fitness = 0
        for i in range(self.genes.__len__()):
            if self.genes[i] == TARGET_CHROMOSOME[i]:
                self.fitness += 1
        return self.fitness

    def __str__(self):
        return self.genes.__str__()

class Population:
    def __init__(self, size):
        self.chromosomes = []
        i = 0
        while i < size:
            self.chromosomes.append(Chromosome())
            i += 1

    def get_chromosomes(self):
        return self.chromosomes

class GeneticAlgorithm:
    def evolve(self,pop):
        return self.mutate_population(self.crossover_population(pop))

    def crossover_population(self,pop):
        crossover_pop = Population(0)
        for i in range(NR_ELITE_CHROMOSOMES):
            crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i])
        i = NR_ELITE_CHROMOSOMES
        while i < POPULATION_SIZE:
            chromosome1 = self.select_tournament_population(pop).get_chromosomes()[0]
            chromosome2 = self.select_tournament_population(pop).get_chromosomes()[0]
            crossover_pop.get_chromosomes().append(self.crossover_chromosomes(chromosome1, chromosome2))
            i += 1
        return crossover_pop

    def mutate_population(self,pop):
        for i in range(NR_ELITE_CHROMOSOMES, POPULATION_SIZE):
            self.mutate_chromosome(pop.get_chromosomes()[i])
        return pop

    def crossover_chromosomes(self,chromosome1, chromosome2):
        crossover_chrom = Chromosome()
        for i in range(TARGET_CHROMOSOME.__len__()):
            if random.random() >= 0.5:
                crossover_chrom.get_genes()[i] = chromosome1.get_genes()[i]
            else:
                crossover_chrom.get_genes()[i] = chromosome2.get_genes()[i]
        return crossover_chrom

    def mutate_chromosome(self,chromosome):
        for i in range(TARGET_CHROMOSOME.__len__()):
            if random.random() < 0.5:
                chromosome.get_genes()[i] = 1
            else:
                chromosome.get_genes()[i] = 0

    def select_tournament_population(self,pop):
        tournament_pop = Population(0)
        i = 0
        while i < TOURNAMENT_SELECTION_SIZE:
            tournament_pop.get_chromosomes().append(pop.get_chromosomes()[random.randrange(0, POPULATION_SIZE)])
            i += 1
        tournament_pop.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
        return tournament_pop

def print_population(pop, gen_number):
    print("\n---------------------------------")
    print("Generation #", gen_number, "| Fittest chromosome fitness: ", pop.get_chromosomes()[0].get_fitness())
    print("Target chromosome: ", TARGET_CHROMOSOME)
    print("\n---------------------------------")
    i = 0
    for x in pop.get_chromosomes():
        print("Chromosome #", i, ": ", x, " | Fitness: ", x.get_fitness())
        i += 1

if __name__ == '__main__':
    population = Population(POPULATION_SIZE)
    population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
    print_population(population, 0)
    generation_number = 1
    genetic = GeneticAlgorithm()
    while population.get_chromosomes()[0].get_fitness() < TARGET_CHROMOSOME.__len__():
        population = genetic.evolve(population)
        population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
        print_population(population, generation_number)
        generation_number += 1