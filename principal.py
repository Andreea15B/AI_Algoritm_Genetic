import numpy as np
import random

LENGTH_CHROMOSOME = 10
STOP_CONDITION = 1000000
MAX = 99999
MIN = -99999


class Chromosome:
    def __init__(self):
        self.value = []
        self.fitness = 0
        self.generation = 0
        for _ in range(LENGTH_CHROMOSOME):
            number = np.random.uniform(MIN, MAX)
            self.value.append(number)

    def print_value(self):
        print(self.value)

    def crossover(self, other_chromosome):
        print('cross')

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

    def selection(self):
        print('selectia turneu')

    def fitness_(self):
        print('fitness')


c = Chromosome()
c.print_value()
c.gaussion_mutation()
c.print_value()
