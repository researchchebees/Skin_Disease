import time

import numpy as np

def RDA(population,objective_function, lb,ub, iterations):
    population_size,dim = population.shape[0],population.shape[1]
    convergence = np.zeros((iterations))
    best_solution = np.zeros((dim))
    best_fitness = np.zeros((population_size))
    ct = time.time()
    for iteration in range(iterations):
        # Evaluate the objective function for each individual
        fitness = np.array([objective_function(individual) for individual in population])

        # Sort the population based on fitness
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]

        # Update the position of red deer (best individual)
        red_deer = population[0]

        # Update the positions of other individuals
        for i in range(1, population_size):
            alpha = np.random.rand(dim)
            population[i] = (1 - alpha) * population[i] + alpha * red_deer

            # Ensure the updated positions are within the bounds
            population[i] = np.clip(population[i], lb, ub)

        # Return the best individual and its fitness
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]
        convergence[iteration] = best_fitness
    ct = time.time()-ct

    return best_solution, convergence,best_fitness,ct
