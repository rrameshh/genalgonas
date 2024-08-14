# genetic_algorithm.py

import numpy as np
from config import (LAYER_TYPE_SPACE, KERNEL_SIZE_SPACE, STRIDE_SPACE, FILTERS_SPACE, RESIDUAL_SPACE,
                    NORMALIZATION_SPACE, ACTIVATION_SPACE, POOLING_TYPE_SPACE)

def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = {
            'layer_type': np.random.choice(LAYER_TYPE_SPACE),
            'filters': np.random.choice(FILTERS_SPACE),
            'kernel_size': np.random.choice(KERNEL_SIZE_SPACE),
            'activation': np.random.choice(ACTIVATION_SPACE),
            'normalization': np.random.choice(NORMALIZATION_SPACE),
            'residual': np.random.choice(RESIDUAL_SPACE),
            'pooling_type': np.random.choice(POOLING_TYPE_SPACE),
        }
        population.append(individual)
    return population

def select_mating_pool(population, scores, num_mating_individuals):
    sorted_indices = np.argsort(scores)
    top_indices = sorted_indices[:num_mating_individuals]
    return [population[i] for i in top_indices]

def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        child[key] = np.random.choice([parent1[key], parent2[key]])
    return child

def mutate(individual):
    mutation_rate = 0.1
    for key in individual.keys():
        if np.random.rand() < mutation_rate:
            if key == 'layer_type':
                individual[key] = np.random.choice(LAYER_TYPE_SPACE)
            elif key == 'filters':
                individual[key] = np.random.choice(FILTERS_SPACE)
            elif key == 'kernel_size':
                individual[key] = np.random.choice(KERNEL_SIZE_SPACE)
            elif key == 'activation':
                individual[key] = np.random.choice(ACTIVATION_SPACE)
            elif key == 'normalization':
                individual[key] = np.random.choice(NORMALIZATION_SPACE)
            elif key == 'residual':
                individual[key] = np.random.choice(RESIDUAL_SPACE)
            elif key == 'pooling_type':
                individual[key] = np.random.choice(POOLING_TYPE_SPACE)
    return individual
