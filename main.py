import random
import math
import numpy as np

# Визначення функції Сфери
def sphere_function(x):
  return sum(xi ** 2 for xi in x)

def start_point(bounds):
    res = []
    for bound in bounds:
        res.append(random.uniform(min(bound), max(bound)))
    return res

# Hill Climbing
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    step_size = 0.01
    current_point = start_point(bounds)
    current_value = func(current_point)

    for iteration in range(iterations):
        x, y  = current_point
        neighbors =  [
            (x + step_size, y),
            (x - step_size, y),
            (x, y + step_size),
            (x, y - step_size)
        ]

        next_point = None
        next_value = np.inf

        for neighbor in neighbors:
            value = func(neighbor)
            if value < next_value:
                next_point = neighbor
                next_value = value

        if next_value >= current_value:
            break
        if current_value - next_value <= epsilon:
            break
        current_point, current_value = next_point, next_value

    return current_point, current_value


# Random Local Search
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    step_size = 0.1
    probability = 0.2
    current_point = start_point(bounds)
    current_value = func(current_point)

    for iteration in range(iterations):
        x, y = current_point
        new_x = x + random.uniform(-step_size, step_size)
        new_y = y + random.uniform(-step_size, step_size)
        new_point = (new_x, new_y)
        new_value = func(new_point)

        if abs(current_value - new_value) <= epsilon:
            break
        if new_value < current_value or random.random() < probability:
            current_point, current_value = new_point, new_value

    return current_point, current_value

# Simulated Annealing
def simulated_annealing(func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6):
    current_point = start_point(bounds)
    current_energy = func(current_point)
    for _ in range(iterations):
        x, y = current_point
        new_x = x + random.uniform(-1, 1)
        new_y = y + random.uniform(-1, 1)
        new_point = (new_x, new_y)
        new_energy = func(new_point)
        delta_energy = current_energy - new_energy

        if temp <= epsilon:
            break

        if new_energy < current_energy or random.random() < math.exp(delta_energy / temp):
            current_point = new_point
            current_energy = new_energy

        temp *= cooling_rate

    return current_point, current_energy



if __name__ == "__main__":
    # Межі для функції
    bounds = [(-5, 5), (-5, 5)]

    # Виконання алгоритмів
    print("Hill Climbing:")
    hc_solution, hc_value = hill_climbing(sphere_function, bounds)
    print("Розв'язок:", hc_solution, "Значення:", hc_value)

    print("\nRandom Local Search:")
    rls_solution, rls_value = random_local_search(sphere_function, bounds)
    print("Розв'язок:", rls_solution, "Значення:", rls_value)

    print("\nSimulated Annealing:")
    sa_solution, sa_value = simulated_annealing(sphere_function, bounds)
    print("Розв'язок:", sa_solution, "Значення:", sa_value)