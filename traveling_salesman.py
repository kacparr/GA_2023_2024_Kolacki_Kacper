import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import pandas
from typing import List, Tuple
import threading
import time

#Parse the data from tsp file
def tsv_parser(a:list) -> pandas.DataFrame:
    s = a.index("NODE_COORD_SECTION\n") + 1
    e = a.index("EOF\n")
    a = [list(i.strip().split(" ")) for i in a[s:e]]
    df = pandas.DataFrame(a,columns=["Index","X","Y"])
    df = df.astype({"Index":int,"X":float,"Y":float})
    return df

#Find distance between two cities
def find_distance(a:pandas.Series, b:pandas.Series) -> float:
    x1,x2 = a.iloc[1],b.iloc[1]
    y1,y2 = a.iloc[2],b.iloc[2]
    distance = float(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    return distance

#Calculate total distance between every city
def calculate_fitness(coords:pandas.DataFrame,sol:list) -> float:
    total_distance = 0
    for i in range(len(sol)):        
        city_a = coords.loc[coords["Index"] == sol[i]].squeeze()
        city_b = coords.loc[coords["Index"] == sol[(i+1) % len(sol)]].squeeze()
        total_distance += find_distance(city_a,city_b)
    return total_distance

def fitness_info(fitness:float,sol:list):
        print(f"Solution: {sol}\nScore: {fitness}")


#Greedy algorithm (find the fastest route from start everytime) 
def tsp_greedy(coords:pandas.DataFrame, start_city:int) -> list:
    unvisited = set(coords["Index"].to_list())
    current = start_city
    route = [current]
    unvisited.remove(current)

    while unvisited:
        city_a = coords.loc[coords["Index"] == current].squeeze()
        city_b = min(unvisited, key=
                     lambda city_x: find_distance(
                         city_a, coords.loc[coords["Index"] ==city_x].squeeze()
                     ))
        route.append(city_b)
        unvisited.remove(city_b)
        current = city_b

    return route

# generate greedy solutions for mixing
def solutions_greedy(coords:pandas.DataFrame) ->List[Tuple[list, float]]:
    cities = coords["Index"].tolist()
    solutions = []
    for start in cities:
        route = tsp_greedy(coords,start)
        total = calculate_fitness(coords, route)
        solutions.append((route, total))
    return sorted(solutions, key=lambda x: x[1])

# generate random solutions for mixing
def solutions_random(coords:pandas.DataFrame, count:int =100)->list:
    cities = coords["Index"].tolist()
    solutions = []
    for i in range(count):
        sol = cities.copy()
        random.shuffle(sol)
        total = calculate_fitness(coords,sol)
        solutions.append([sol,total])
    return solutions

# create a population with mix of greedy and random solutions
def initialise_population(coords: pandas.DataFrame,individuals:int,greedy_count:int=0,) -> pandas.DataFrame:
    population = []
    greedy_count = min(greedy_count, len(coords))
    if greedy_count > 0:
        greedy_data = solutions_greedy(coords)[:greedy_count]
        population.extend(greedy_data)
    
    random_count = individuals - greedy_count
    if random_count > 0:
        random_data = solutions_random(coords,random_count)
        population.extend(random_data)
    
    population = pandas.DataFrame(population, columns=["Individual", "Fitness"])
    return population
    
# def population_data(p:pandas.DataFrame):
#     fitness = p["Fitness"].tolist()
#     best,worst = fitness.index(min(fitness)), fitness.index(max(fitness))
#     print(f"Population length: {p.shape[0]}\nMedian fitness: {np.median(fitness)}")
#     print(f"Best score: {min(fitness)} for individual {p.iloc[best,0]}")
#     print(f"Worst score: {max(fitness)} for individual {p.iloc[worst,0]}")


# return two individuals for crossover func
#ROULETTE - 
def selection_roulette(population:pandas.DataFrame, _) -> list:
    max_fitness = population["Fitness"].max()
    inverted_fitness = max_fitness - population["Fitness"] + 1
    fitness = inverted_fitness.sum()

    probabilities = inverted_fitness / fitness
    culminative = probabilities.cumsum()
    parents = []
    for i in range(2):
        roll = random.random()
        for i in range(len(culminative)):
            if culminative[i] >= roll:
                parents.append(population.iloc[i,0])
                break
    return parents


#TOURNAMENT
def selection_tournament(population:pandas.DataFrame, size:int=3) -> list:
    parents = []
    for i in range(2):
        tournament = population.sample(n=size)
        winner = tournament.loc[tournament["Fitness"].idxmin(), "Individual"]
        parents.append(winner)
    return parents

# cycle or ordered
def crossover(parents:list) ->list:
    parent1,parent2 = parents[0],parents[1]
    size = len(parent1)
    pos1, pos2 = sorted([random.randrange(size) for i in range(2)])

    def create_child(p1,p2):
        child = [None] * size
        child[pos1:pos2] = p1[pos1:pos2]
        p2_filtered = [x for x in p2 if x not in child]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_filtered[idx]
                idx+= 1
        return child
    
    child1 = create_child(parent1,parent2)
    child2 = create_child(parent2,parent1)
    children = [child1,child2]
    return children

def swap_mutation(child:list) -> list:
    a = child.copy()
    pos1, pos2 = [random.randrange(len(a)) for i in range(2)]
    a[pos1], a[pos2] = a[pos2], a[pos1]
    return a

def inverse_mutation(child:list):
    a = child.copy()
    pos1,pos2 = sorted([random.randrange(len(a)) for i in range(2)])
    a[pos1:pos2] = a[pos1:pos2][::-1]
    return a

    
def genetic_algorithm(file_path: str,
                     population_size=100,
                     generations=500, 
                     mutation_rate=0.05, 
                     greedy_count=0,
                     elitism_count:int=2,
                     selection_type:str="tournament",
                     tournament_size:int=3,
                     mutation_type:str="inversion",
                     seed:int=None) -> Tuple[list,float,list]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    with open(file_path, "r") as f:
        data = f.readlines()
    coords = tsv_parser(data)

    population = initialise_population(coords, population_size, greedy_count)

    history = []

    select_func = selection_tournament if selection_type == "tournament" else selection_roulette
    mutation_func = swap_mutation if mutation_type == "swap" else inverse_mutation

    for generation in range(generations):
        population = population.sort_values("Fitness").reset_index(drop=True)
        best_fitness = population.iloc[0,1]
        avg_fitness = population["Fitness"].mean()
        worst_fitness = population.iloc[-1, 1]
        history.append(best_fitness)
        # print(f"Generation {generation + 1:3d}: Best = {best_fitness:8.2f}, Avg = {avg_fitness:8.2f}, Worst = {worst_fitness:8.2f}")

        
        next_generation = []
        for i in range(elitism_count):
            next_generation.append([population.iloc[i,0],population.iloc[i,1]])

        while len(next_generation) < population_size:
            if selection_type == "tournament":
                parents = select_func(population,tournament_size)
            else:
                parents = select_func(population)
            
            children = crossover(parents)
            # mutate children
            for i in range(len(children)):
                
                if random.random() < mutation_rate:
                    children[i] = mutation_func(children[i])
            # fitness
            for child in children:
                if len(next_generation) < population_size:
                    fitness = calculate_fitness(coords, child)
                    next_generation.append([child, fitness])

        population = pandas.DataFrame(next_generation[:population_size], columns=["Individual", "Fitness"])


    best_index = population["Fitness"].idxmin()
    best_solution = population.loc[best_index, "Individual"]
    best_fitness = population.loc[best_index, "Fitness"]
    return best_solution, best_fitness, history


def run_genetic_tests(file_path: str, runs: int = 10, **ga_params) -> pandas.DataFrame:
    results = []
    for i in range(runs):
        rand = random.randint(1,99999999)
        print(f"\n--- RUN {i+1}/{runs}: ---")
        solution, fitness, history = genetic_algorithm(file_path, seed=rand, **ga_params)
        results.append({"Run": i+1, "Fitness": fitness, "Solution": solution})
        print(f"Final result: {fitness:2f}")

    print("\n1. Genetic test completed!")
    return pandas.DataFrame(results)

def run_greedy_tests(file_path:str) ->pandas.DataFrame:
    with open(file_path, "r") as f:
        data = f.readlines()
    coords = tsv_parser(data)
    
    solutions = solutions_greedy(coords)
    results = []
    for i, (route, fitness) in enumerate(solutions, 1):
        results.append({
            "Run": i,
            "Fitness": fitness,
            "Solution": route,
            "Start City": route[0],

        })
    df = pandas.DataFrame(results)
    print("\n2. Greedy tests completed!")
    return df

def run_random_tests(file_path:str, count:int=100)-> pandas.DataFrame:
    with open(file_path, "r") as f:
        data = f.readlines()
    coords = tsv_parser(data)
    
    solutions = solutions_random(coords, count)
    
    results = []
    for i, (sol, fitness) in enumerate(solutions, 1):
        results.append({
            "Run": i,
            "Fitness": fitness,
            "Solution": sol,
        })
    df = pandas.DataFrame(results)
    print(f"\n3.Random tests completed!")
    return df

best_runs = {"berlin11_modified": 4038,
             "berlin52": 7542,
             "kroA100": 21282,
             "kroA150": 26524,
             "kroA200": 29368
             }
  
def calculate_stats(df:pandas.DataFrame, file_path: str) -> dict:
    fitness = df["Fitness"]
    filename = os.path.basename(file_path)
    file = os.path.splitext(filename)[0]
    stats= {
        "File": file,
        "Best": fitness.min(),
        "Worst": fitness.max(),
        "Mean": fitness.mean(),
        "Std_Dev": fitness.std(),
        "Median": fitness.median(),
        "Variance": fitness.var(),    
        "Best_route": df.loc[fitness.idxmin(), "Solution"],

    }
    if file in best_runs:
        if int(fitness.min()) == math.floor(best_runs[file]):
            stats["Is_Best_Ever"] = "Yes"
            stats["First_appearance"] = df.loc[fitness.idxmin(), "Run"]
        else:
            stats["Is_Best_Ever"] = "No"
    
    return stats


def comparison_table(ga_results, greedy_results, random_results, file_path):
    print("\n========== ALGORITHM STATISTICS ==========")
    print(f"\n{'─'*80}")
    print("\n 1. GENETIC ALGORITHM")
    print(f"\n{'─'*80}")
    print("\nI Results:")
    print(ga_results[["Run", "Fitness"]].to_string(index=False))

    ga_stats = calculate_stats(ga_results, file_path)
    print(f"\nStatistics:")
    print(f"  Best:       {ga_stats['Best']:.2f}")
    print(f"  Worst:      {ga_stats['Worst']:.2f}")
    print(f"  Mean:       {ga_stats['Mean']:.2f}")
    print(f"  Std Dev:    {ga_stats['Std_Dev']:.2f}")
    print(f"  Median:     {ga_stats['Median']:.2f}")
    print(f"  Variance:   {ga_stats['Variance']:.2f}")
    print(f"Best route: {ga_stats['Best_route']}")
    print(f"Is best solution found: {ga_stats['Is_Best_Ever']}")
    if ga_stats["Is_Best_Ever"] == "Yes":
            print(f"First appearance in run: {ga_stats['First_appearance']}")


    print(f"\n{'─'*80}")
    print("\n 2. GREEDY ALGORITHM")
    print(f"\n{'─'*80}")
    print("\nBest 5 Results:")
    print(greedy_results.head(5)[["Start City", "Fitness"]].to_string(index=False))
    greedy_stats = calculate_stats(greedy_results, file_path)
    print(f"\nStatistics:")
    print(f"  Best:       {greedy_stats['Best']:.2f}")
    print(f"  Worst:      {greedy_stats['Worst']:.2f}")
    print(f"  Mean:       {greedy_stats['Mean']:.2f}")
    print(f"  Std Dev:    {greedy_stats['Std_Dev']:.2f}")
    print(f"  Median:     {greedy_stats['Median']:.2f}")
    print(f"  Variance:   {greedy_stats['Variance']:.2f}")
    print(f"Best route: {greedy_stats['Best_route']}")

    print(f"Is best solution found: {greedy_stats['Is_Best_Ever']}")
    if greedy_stats["Is_Best_Ever"] == "Yes":
            print(f"First appearance in run: {greedy_stats['First_appearance']}")

    print(f"\n{'─'*80}")
    print("\n 3. RANDOM ALGORITHM")
    print(f"\n{'─'*80}")
    random_stats = calculate_stats(random_results, file_path)
    print(f"\nStatistics:")
    print(f"  Best:       {random_stats['Best']:.2f}")
    print(f"  Worst:      {random_stats['Worst']:.2f}")
    print(f"  Mean:       {random_stats['Mean']:.2f}")
    print(f"  Std Dev:    {random_stats['Std_Dev']:.2f}")
    print(f"  Median:     {random_stats['Median']:.2f}")
    print(f"  Variance:   {random_stats['Variance']:.2f}")
    print(f"Best route: {random_stats['Best_route']}")

    print(f"Is best solution found: {random_stats['Is_Best_Ever']}")
    if random_stats["Is_Best_Ever"] == "Yes":
            print(f"First appearance in run: {random_stats['First_appearance']}")

    return ga_stats, greedy_stats, random_stats
  
def plot_route(coords:pandas.DataFrame, solution:list):
    plt.figure(figsize=(10,8))
    plt.scatter(coords["X"], coords["Y"], color='blue')

    for i in range(len(solution)):
        city_a = coords.loc[coords["Index"] == solution[i]].squeeze()
        city_b = coords.loc[coords["Index"] == solution[(i+1) % len(solution)]].squeeze()
        plt.plot([city_a["X"], city_b["X"]], 
                 [city_a["Y"], city_b["Y"]], 
                 'b-',alpha=0.7)
    for i, row in coords.iterrows():
        plt.annotate(str(row["Index"]), (row["X"], row["Y"]),
                     fontsize=9, ha='center')
        plt.title("Route and cities visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "src/berlin52.tsp"

    best_params = {
        'population_size': 100,
        'generations': 100,
        'mutation_rate': 0.02,
        'greedy_count': 10,
        'elitism_count': 2,
        'selection_type': 'tournament',
        'tournament_size': 3,
        'mutation_type': 'inversion'
    }
    print("Generating plot for the algorithm...")
    with open(file_path, "r") as f:
        data = f.readlines()
    coords = tsv_parser(data)
    solution, fitness, history = genetic_algorithm(
        file_path,
        **best_params
    )
    plot_route(coords, solution)

    is_running = True
    spin = ["-", "\\", "|", "/"]
    def loading():
        i = 0
        while is_running:
            print(f"\rThis can take a few minutes to complete {spin[i%4]}",end="",flush=True)
            i+=1
            time.sleep(0.25)

    print("Starting the algorithms for statistics")
    thread = threading.Thread(target=loading)
    thread.start()
    ga_results = run_genetic_tests(file_path, runs=10, **best_params)
    greedy_results = run_greedy_tests(file_path)
    random_results = run_random_tests(file_path, count=100)
    is_running = False
    thread.join()

    ga_stats, greedy_stats, random_stats = comparison_table(ga_results, greedy_results, random_results, file_path)  
