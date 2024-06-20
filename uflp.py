import numpy as np
import time

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()

    num_facilities, num_customers = map(int, data[0].strip().split())
    facility_costs = np.zeros(num_facilities)
    service_costs = np.zeros((num_customers, num_facilities))

    for i in range(1, num_facilities + 1):
        facility_costs[i - 1] = float(data[i].strip().split()[1])

    current_line = num_facilities + 1
    for i in range(num_customers):
        costs = []
        while len(costs) < num_facilities + 1:
            line_costs = list(map(float, data[current_line].strip().split()))
            costs.extend(line_costs)
            current_line += 1
        service_costs[i, :] = costs[1:num_facilities + 1]

    return facility_costs, service_costs

def calculate_total_cost(facility_open, facility_costs, service_costs):
    total_cost = np.sum(facility_costs[facility_open])
    min_service_costs = np.min(service_costs[:, facility_open], axis=1) if np.any(facility_open) else np.zeros(service_costs.shape[0])
    total_cost += np.sum(min_service_costs)
    return total_cost

def local_search(facility_open, facility_costs, service_costs, max_local_iterations=100):
    current_cost = calculate_total_cost(facility_open, facility_costs, service_costs)
    num_facilities = len(facility_costs)
    for _ in range(max_local_iterations):
        facility_to_toggle = np.random.randint(0, num_facilities)
        facility_open[facility_to_toggle] = not facility_open[facility_to_toggle]
        new_cost = calculate_total_cost(facility_open, facility_costs, service_costs)
        if new_cost < current_cost:
            current_cost = new_cost
        else:
            facility_open[facility_to_toggle] = not facility_open[facility_to_toggle]
    return facility_open, current_cost

def initialize_population(pop_size, num_facilities):
    return [np.random.choice([True, False], size=num_facilities) for _ in range(pop_size)]

def evaluate_population(population, facility_costs, service_costs):
    return [calculate_total_cost(individual, facility_costs, service_costs) for individual in population]

def select_parents(population, fitness, num_parents):
    fitness = np.array(fitness)
    parents_indices = np.argsort(fitness)[:num_parents]
    return [population[i] for i in parents_indices]

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = not individual[i]
    return individual

def genetic_algorithm_with_local_search(facility_costs, service_costs, pop_size, num_generations, mutation_rate, num_parents, max_local_iterations):
    num_facilities = len(facility_costs)
    population = initialize_population(pop_size, num_facilities)
    for generation in range(num_generations):
        fitness = evaluate_population(population, facility_costs, service_costs)
        parents = select_parents(population, fitness, num_parents)
        new_population = parents.copy()
        while len(new_population) < pop_size:
            # Get indices for parents, instead of selecting parent objects directly
            parent_indices = np.random.choice(len(parents), 2, replace=False)
            parent1, parent2 = parents[parent_indices[0]], parents[parent_indices[1]]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            child1, _ = local_search(child1, facility_costs, service_costs, max_local_iterations)
            child2, _ = local_search(child2, facility_costs, service_costs, max_local_iterations)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        population = new_population

    final_fitness = evaluate_population(population, facility_costs, service_costs)
    best_index = np.argmin(final_fitness)
    return population[best_index], final_fitness[best_index]


def main():
    start_time = time.time()
    file_path = 'uflp_instances/capc.txt'  # Update this with the actual path to your data file
    facility_costs, service_costs = read_data(file_path)
    optimal_facilities, optimal_cost = genetic_algorithm_with_local_search(
        facility_costs, service_costs,
        pop_size=50,
        num_generations=100,
        mutation_rate=0.05,
        num_parents=10,
        max_local_iterations=50
    )
    print("Optimal Facilities Open:", optimal_facilities)
    print("Optimal Total Cost:", optimal_cost.round(2))
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime} seconds")

if __name__ == "__main__":
    main()
