import numpy as np
import random
import copy as cp

LENGTH_CHROMOSOME = 10
STOP_CONDITION = 100000
MAX = 99999
MIN = -99999
MAX_POPULATION_SIZE = 1000

class Chromosome:
    def __init__(self):
        self.value = []
        self.fitness = 0
        self.generation = 0

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

    def set_fitness(self):
        self.fitness = sum(self.value) / len(self.value)

    def get_fitness(self):
        return self.fitness

    def single_point_crossver(self,second_chromosome):
        length=len(second_chromosome.value)
        cutting_point=random.randrange(length)
        print(cutting_point)
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
        print(cutting_point)
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
        i = 0
        while i < size:
            chromosome = Chromosome()
            self.chromosomes.append(chromosome.random_init())
            i += 1

    def get_size(self):
        return self.size

    def set_chromosomes(self, chromosomes):
        self.chromosomes = cp.deepcopy(chromosomes)
        self.size = len(chromosomes)

    def get_chromosomes(self):
        return self.chromosomes

    def tournament_population_selection(self):
        J_NUMBER = 25
        K_NUMBER = np.random.randint(J_NUMBER * 2, MAX_POPULATION_SIZE / 2)

        tournament_population = Population(0)

        indices = np.random.randint(0, MAX_POPULATION_SIZE, size = K_NUMBER)

        it = 0
        while it < K_NUMBER:
            tournament_population.get_chromosomes().append(self.get_chromosomes()[indices[it]])
            it = it + 1

        tournament_population.get_chromosomes().sort(key = lambda x: x.get_fitness(), reverse = True)
        tournament_population.set_chromosomes(tournament_population.get_chromosomes()[0:J_NUMBER])

        return tournament_population

if __name__ == '__main__':
    size_of_population = np.random.randint(2, MAX_POPULATION_SIZE // 100)
    population = Population(size_of_population)

    it = 0
    while (it < STOP_CONDITION) or (population.get_size() < MAX_POPULATION_SIZE):
        
