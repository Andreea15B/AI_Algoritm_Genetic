import numpy as np
import random
import copy as cp
from math import sqrt

LENGTH_CHROMOSOME = 10
STOP_CONDITION = 100000
MAX = 99999
MIN = -99999
BEST_POSSIBLE_FITNESS = 0
MAX_POPULATION_SIZE = 36
K_INDIVIDUALS = int(sqrt(MAX_POPULATION_SIZE))

class Chromosome:
    def __init__(self):
        #TODO: Replace 'value' wih 'genes'
        self.value = []
        self.fitness = 0
        self.generation = 0

    def set_fitness(self):
        self.fitness = sum(self.value) / len(self.value)

    def get_fitness(self):
        return self.fitness

    def random_init(self):
        for _ in range(LENGTH_CHROMOSOME):
            number = np.random.uniform(MIN, MAX)
            self.value.append(number)

        self.set_fitness()

    def print_value(self):
        print(self.value)

    def uniform_mutation(self):
        length = len(self.value)
        random_position = random.randrange(length)
        self.value[random_position] = np.random.uniform(MIN, MAX)

    def swap_mutation(self):
        length = len(self.value)
        first_random_position = random.randrange(length)
        second_random_position = random.randrange(length)

        while(second_random_position == first_random_position):
            second_random_position = random.randrange(length)

        aux = self.value[first_random_position]
        self.value[first_random_position] = self.value[second_random_position]
        self.value[second_random_position] = aux

    def scramble_mutation(self):
        length = len(self.value)
        first_random_position = random.randrange(length)
        second_random_position = random.randrange(length)

        while(second_random_position == first_random_position):
            second_random_position = random.randrange(length)

        if(second_random_position < first_random_position):
            aux = first_random_position
            first_random_position = second_random_position
            second_random_position = aux

        new_array = self.value[first_random_position:second_random_position]
        random.shuffle(new_array)

        for i in range(first_random_position, second_random_position):
            self.value[i] = new_array[i - first_random_position]

    def inversion_mutation(self):
        length = len(self.value)
        first_random_position = random.randrange(length)
        second_random_position = random.randrange(length)

        while(second_random_position == first_random_position):
            second_random_position = random.randrange(length)

        if(second_random_position < first_random_position):
            aux = first_random_position
            first_random_position = second_random_position
            second_random_position = aux

        new_array = self.value[first_random_position:second_random_position]
        new_array = new_array[::-1]

        for i in range(first_random_position, second_random_position):
            self.value[i] = new_array[i - first_random_position]

    def gaussion_mutation(self):
        mu, sigma = 0, 0.1  # mean and standard deviation
        s = np.random.normal(mu, sigma, 1)

        length = len(self.value)
        random_position = random.randrange(length)
        self.value[random_position] = self.value[random_position] + s[0]

        while (self.value[random_position] > MAX):
            s = np.random.normal(mu, sigma, 1)
            self.value[random_position] = self.value[random_position] + s[0]

    def single_point_crossver(self,second_chromosome):
        length=len(second_chromosome.value)
        cutting_point=random.randrange(length)
        first_child=Chromosome()
        second_child=Chromosome()
        for i in range(length):
            if(i<=cutting_point):
                first_child.value.append(self.value[i])
                second_child.value.append(second_chromosome.value[i])
            else:
                first_child.value.append(second_chromosome.value[i])
                second_child.value.append(self.value[i])
        return (first_child,second_child)

    def two_points_crossver(self,second_chromosome):
        length=len(self.value)
        first_cutting_point=random.randrange(1,length-3)
        second_cutting_point=random.randrange(first_cutting_point+2,length-1)
        first_child=Chromosome()
        second_child=Chromosome()
        for i in range(length):
            if i<=first_cutting_point or i>=second_cutting_point:
                first_child.value.append(self.value[i])
                second_child.value.append(second_chromosome.value[i])
            else:
                first_child.value.append(second_chromosome.value[i])
                second_child.value.append(self.value[i])
        return (first_child,second_child)

    def uniform_crossver(self,second_chromosome):
        length=len(self.value)
        first_child=Chromosome()
        second_child=Chromosome()
        for i in range(length):
            value=random.random()
            if value<0.5:
                first_child.value.append(self.value[i])
                second_child.value.append(second_chromosome.value[i])
            else:
                first_child.value.append(second_chromosome.value[i])
                second_child.value.append(self.value[i])
        return (first_child,second_child)

    def single_arithmetic_crossover(self,second_chromosome):
        length=len(self.value)
        alpha=random.random()
        gene_position=random.randrange(length)
        first_child=Chromosome()
        second_child=Chromosome()
        for i in range(length):
            if i==gene_position:
                first_child.value.append(alpha*self.value[i]+(1-alpha)*second_chromosome.value[i])
                second_child.value.append(alpha*second_chromosome.value[i]+(1-alpha)*self.value[i])
            else:
                first_child.value.append(self.value[i])
                second_child.value.append(second_chromosome.value[i])
        return (first_child,second_child)

    def ring_crossover(self,second_chromosome):
        length=len(self.value)
        cutting_point=random.randrange(length)
        first_child=Chromosome()
        second_child=Chromosome()
        for i in range(length):
            if i<cutting_point:
                first_child.value.append(self.value[i])
            else:
                second_child.value.append(self.value[i])
        for i in range(length):
            if i <length-cutting_point:
                first_child.value.append(second_chromosome.value[i])
            else:
                second_child.value.append(second_chromosome.value[i])
        return (first_child,second_child)

class Population:
    def __init__(self, size):
        self.chromosomes = []
        self.size = size
        self.total_fitness = 0

        it = 0
        while it < size:
            chromosome = Chromosome()
            chromosome.random_init()
            self.chromosomes.append(chromosome)
            self.total_fitness = self.total_fitness + chromosome.get_fitness()
            it = it + 1

        self.chromosomes.sort(key = lambda x: x.get_fitness(), reverse = True)

    def get_size(self):
        return self.size

    def get_total_fitness(self):
        return self.total_fitness

    def get_chromosomes(self):
        return self.chromosomes

    def set_chromosomes(self, chromosomes):
        self.chromosomes = cp.deepcopy(chromosomes)
        self.size = len(chromosomes)

    def update_total_fitness(self):
        _total_fitness = 0

        for it in range(0, self.size):
            _total_fitness = _total_fitness + self.chromosomes[it].get_fitness()

        self.total_fitness = _total_fitness

    def tournament_population_selection(self):
        J_INDIVIDUALS = np.random.randint(2, K_INDIVIDUALS // 2)

        tournament_population = Population(0)

        indices = np.random.randint(0, self.get_size(), size = K_INDIVIDUALS)

        it = 0
        while it < K_INDIVIDUALS:
            tournament_population.get_chromosomes().append(self.get_chromosomes()[indices[it]])
            it = it + 1

        tournament_population.get_chromosomes().sort(key = lambda x: x.get_fitness(), reverse = True)
        tournament_population.set_chromosomes(tournament_population.get_chromosomes()[0:J_INDIVIDUALS])

        return tournament_population

class GeneticOperators:
    #TODO: Move here Crossover, Mutation and Selection
    pass

if __name__ == '__main__':
    population = Population(MAX_POPULATION_SIZE)

    generation = 0
    while (generation < STOP_CONDITION) or (population.get_chromosomes()[0] == MAX - 1) or (population.get_total_fitness() < 0):
        selected_individuals = population.tournament_population_selection().get_chromosomes()
        selected_individuals_len = len(selected_individuals)

        index = population.get_size() - 1

        for i in range(0, selected_individuals_len - 1):
            for j in range(i + 1, selected_individuals_len):
                if index > 1:
                    (offspring1, offspring2) = selected_individuals[i].single_point_crossver(selected_individuals[j])
                    offspring1.scramble_mutation()
                    offspring2.scramble_mutation()

                    print(offspring1.print_value())
                    print(offspring2.print_value())

                    print(population.get_chromosomes()[-1].print_value())
                    population.get_chromosomes()[index] = cp.deepcopy(offspring1)
                    print(population.get_chromosomes()[-1].print_value())
                    population.get_chromosomes()[index - 1] = cp.deepcopy(offspring2)

                    index = index - 2
                else:
                    break

        population.get_chromosomes().sort(key = lambda x: x.get_fitness(), reverse = True)
        population.update_total_fitness()

        print('Generation: ' + str(generation) + ' Total fittnes: ' + str(population.get_total_fitness()) + ' Best Chromosome Fitness: ' + str(population.get_chromosomes()[0].get_fitness()))

        print('Last Chromosome Genes:' + str(population.get_chromosomes()[-1].print_value()))

        generation = generation + 1

#TODO: Save Logs to Disk
