# main.py

import numpy as np
from config import POPULATION_SIZE, NUM_GENERATIONS, MUTATION_RATE
from utils import evaluate_architecture
from genetic_algorithm import initialize_population, select_mating_pool, crossover, mutate
from models import model_to_json, json_to_model

def genetic_algorithm():
    population = initialize_population(POPULATION_SIZE)
    for generation in range(NUM_GENERATIONS):
        scores = []
        for individual in population:
            architecture = json_to_model(model_to_json(individual))
            accuracy, latency, model_size, flops = evaluate_architecture(architecture)
            scores.append(accuracy)
            print(f"Generation {generation+1}, Individual Score: {accuracy}")

        mating_pool = select_mating_pool(population, scores, num_mating_individuals=POPULATION_SIZE // 2)
        new_population = []

        for _ in range(POPULATION_SIZE):
            parent1 = np.random.choice(mating_pool)
            parent2 = np.random.choice(mating_pool)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    print("Genetic Algorithm finished.")

if __name__ == "__main__":
    genetic_algorithm()
