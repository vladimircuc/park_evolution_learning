# File: disney.py
# Description: Plan the most fun Walt Disney World trip itinerary using a genetic algorithm.
# Authors: Vladmir Cuc & Riley Sweeting

# Import required packages
import argparse
import datetime
from graphics import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
import sys
import time

# Root Directory
ROOT = os.path.dirname(os.path.abspath(__file__)) # path to source directory of this file

# Command Line Arguments
parser = argparse.ArgumentParser(description="Use a genetic algorithm to plan a trip to Disney World")
parser.add_argument('-p', '--park', help="which park to visit (default=magic-kingdom)", type=str, default='magic-kingdom')
parser.add_argument('-c', '--color', help="hex code for path color to show on map (default=#000000)", type=str, default="#000000")
parser.add_argument('-d', '--delay', help="time delay for visualization (default=0.05)", type=float, default=0.05)
parser.add_argument('-w', '--walk', help="walking speed in min/pixel (default=0.02)", type=float, default=0.02)
parser.add_argument('-t', '--time', help='time limit of minutes spent in park, excluding exit time', type=float, default=600.0)
parser.add_argument('-m', '--mutation', help='the mutation rate (between 0 and 1)', type=float, default=0.75)
parser.add_argument('-g', '--generations', help='the maximum number of generations', type=int, default=300)
parser.add_argument('-e', '--elitism', help='the proportion of elite parents to keep', type=float, default=0.2)
parser.add_argument('-db', '--debug', help='Set PDB trace for debugging (default=False)', type=bool, default=False )

# Class containing all global variables and dictionaries (data frames) for cleaner parameter passes through nested function calls
class Params:
    def __init__(self, rides, ride_times, wait_times, locations, fun_scores, **kwargs):
        self.rides = rides
        self.ride_times = ride_times
        self.wait_times = wait_times
        self.locations = locations
        self.fun_scores = fun_scores
        self.walk_rate = kwargs['walk']
        self.time_limit = kwargs['time']
        self.generations = kwargs['generations']
        self.mutation = kwargs['mutation']
        self.elitism = kwargs['elitism']
        self.debug = kwargs['debug']

def main(**kwargs):
    # Load data files for given park
    locations = pd.read_csv(os.path.join('parks', kwargs['park'], 'location.csv'), header=None)
    ridetime = pd.read_csv(os.path.join('parks', kwargs['park'], 'ridetime.csv'), header=None)
    wait = pd.read_csv(os.path.join('parks', kwargs['park'], 'wait.csv'), header=None)
    fun = pd.read_csv(os.path.join('parks', kwargs['park'], 'fun.csv'), header=None)
    
    # Convert data files to numpy arrays
    locs = locations.iloc[:, [1, 2]].to_numpy()
    ride_times = ridetime.iloc[:, 1].to_numpy()
    wait_times = wait.iloc[:, 1].to_numpy()
    rides = fun.iloc[:, 0].to_numpy(dtype='U50')
    fun_scores = fun.iloc[:, 1].to_numpy()
    
    # Initialize 'Params' object containing all global variables and dictionaries
    params = Params(rides, ride_times, wait_times, locs, fun_scores, **kwargs)

    # Initialize the GUI and circle park rides
    gui = make_gui(**kwargs)
    add_rides(gui, locations, **kwargs)

    # Run genetic algorithm
    solution = evolve(params)
    
    # Decode genotype to phenotype
    bestplan = rides[solution]

    # Plot route solution to map
    add_route(gui, bestplan, locations, fun, ridetime, wait, **kwargs)
    print('Select \'a\' to generate another solution...')

    # Infinite loop to allow for user controls
    while True:
        # If the user closed the gui, end the program
        if gui.isClosed():
            break
        
        # Otherwise, check for key presses
        key = gui.checkKey()
        if key == 'Escape':
            break
        elif key == 'd': # debug
            pdb.set_trace()
        elif key == 'a': # add another route to the map
            bestplan = rides[evolve(params)]
            clear_route(gui)
            add_route(gui, bestplan, locations, fun, ridetime, wait, **kwargs)
            print('Select \'a\' to generate another solution...')
        elif key == 'c': # clear route from the map
            clear_route(gui)
        elif key == 'r': # get a new randomized plan
            bestplan = np.random.permutation(rides)
            clear_route(gui)
            add_route(gui, bestplan, locations, fun, ridetime, wait, **kwargs)

        # *** The following code snippet may be helpful for locating rides on a new map ***
        # pt = gui.checkMouse()
        # if pt:
        #     print(pt)
        
def evolve(params):
    '''Using a Genetic Algorithm (GA), evolves a theme park itinerary that maximizes fun
    experienced while satisfying the timit limit constraint.'''
    
    # Initialize population of 50 individuals
    population = populate(50, params)
    
    # Compute initial fitness of population
    fitness_scores = compute_fitness(population, params)
    
    # Iterate over every generation
    for gen in range(params.generations):       
        # Initialize starting time
        start = time.time()
        
        # PARENT SELECTION - Select fittest parents to recombine using tournament
        parents, parent_scores = select(population, fitness_scores)
        
        # CROSSOVER - Recombine parents using modified 3-Point-Crossover
        offspring = recombine(parents, params)
        
        # MUTATION - Mutate offspring (in place)
        mutate(offspring, params)
        
        # Replace population with elite parents and offspring
        population, fitness_scores = replace(parents, offspring, parent_scores, params)
        
        # Display time elapsed for current generation
        end = time.time()
        scores = compute_fitness(population, params)
        print(f'GENERATION {gen + 1} WITH MAX FITNESS {np.max(scores)} COMPLETED IN {end - start:.2f} SECONDS\n')
        
    # Extract fittest itinerary solution
    fittest_idx = np.argmax(fitness_scores)
    solution = population[fittest_idx]
    
    return solution

def populate(size, params):
    '''Initialize population of specified size that does NOT need to satisfy the time limit constraint.
    The fitness computation handles candidates that exceed time limit. This approach allows every individual
    to be of the same size, allowing for efficient numpy array operations.'''
    # Initialize population with integer labels
    return np.random.randint(len(params.rides), size=(size,20))

def compute_fitness(population, params, debug=False):
    '''Computes an array of fitness scores for a given population. The fitness score is increased per gene
    until the time limit is exceeded.'''   
    # Initialize fitness scores array
    scores = np.zeros(population.shape[0], dtype='int32')
    
    # Iterate over each individual in population
    for chromosome in range(population.shape[0]):
        
        # Initialize starting time for individual
        total_time = 0
        
        # Initialize individual's fitness score
        fitness = 0
        
        # Fun scores dictionary to penalize repeat attraction scores
        fun_scores = params.fun_scores.copy()

        # Iterate over every gene
        for gene in range(0, population.shape[1]):           
            # Update total time
            if gene == 0:
                total_time += time_added(-1, population[chromosome, gene], params)
            else:
                total_time += time_added(population[chromosome, gene - 1], population[chromosome, gene], params)
            
            # Check if new gene exceeds itinerary time limit
            if total_time <= params.time_limit:
                # Increase individual's fitness score by current attraction's fun score
                fitness += fun_scores[population[chromosome, gene]]
                
                # Half the current attraction's fun score in fun scores dictionary
                fun_scores[population[chromosome, gene]] = fun_scores[population[chromosome, gene]] // 2
            else:
                break
                  
        # Add individual's fitness score 
        scores[chromosome] = fitness
        
    # Return fitness scores for each individual of population
    return scores

def select(population, scores):
    '''Randomly selects individuals from the population, with replacement, and pits
    them against other individuals 1v1, with the winner being selected. The number of winners
    is equal to the size of the population, keeping population size constant.'''
    # Randomly select individuals (with replacement)
    set1 = np.random.randint(population.shape[0], size=population.shape[0])
    set2 = np.random.randint(population.shape[0], size=population.shape[0])
    
    # Boolean mask where set1 beats set2
    mask = scores[set1] > scores[set2]
    
    # Compile winners
    winners = np.where(mask.reshape(population.shape[0], 1), population[set1], population[set2])
    
    # Return winner (parent) scores to reduce future fitness calculations during replacement
    parent_scores = np.where(mask, scores[set1], scores[set2])
                
    # Return array of parents that won the tournament and their scores
    return winners, parent_scores

def recombine(parents, params):
    '''Sends parents 2 at a time to be recombined and returned as offspring of equal size'''
    # Initialize offspring
    offspring = np.zeros(parents.shape, dtype='int32')
    
    # Iterate over parent population
    for idx in range(0, len(parents) - 1, 2):
        # Receive children from 3-Point-Crossover
        offspring[idx:idx + 2] = crossover(parents[idx:idx+2], params)
        
    # If the number of parents is odd, recombine last 2 parents and add only first child
    if len(parents) % 2 != 0:
        offspring[-1] = crossover(parents[-2:], params)[0]
        
    # Return recombined offspring
    return offspring

def crossover(parents, params):
    '''Recombines 2 parents using a modified 3-Point Crossover'''   
           
    # Initialize children
    children = parents.copy()
    
    # Generate 3 random crossover points
    k = sorted(np.random.permutation(np.arange(1, parents.shape[1]))[:3])
    
    # Perform 3-Point-Crossover for child 1 (skipping Entrance)
    children[:, k[0]:k[1]] = parents[::-1, k[0]:k[1]]
    children[:, k[2]:] = parents[::-1, k[2]:]
    
    # Return children
    return children

def mutate(population, params):
    '''Mutate each genotype in the population: We are using 2 types of mutation, Inversion Mutation 
    to keep some kind of order between rides and a random resetting to give the plan the
    chance of trying new rides. Therefore, the mutation rate for each type of mutation
    will be some percent of the overall mutation rate'''

    #setting the mutation rate for each mutation
    random_rate = params.mutation / 2
    inversion_rate = params.mutation / 2
    
    # Inversion mutation
    for i in range(len(population)):
        # If individual is selected for mutation
        if np.random.rand() <= inversion_rate:
            # Randomize inversion bounds (Start at index 1 to skip Entrance)
            j = sorted(np.random.randint(1, population.shape[1], size=(1, 2))[0])
            # Reverse slice
            population[i, j[0]:j[1]] = population[i, j[0]:j[1]][::-1]

    # Random resetting mutation (in place)
    mutants = np.random.randint(len(params.rides), size=population.shape)
    mask = np.random.rand(*population.shape) <= random_rate
    np.putmask(population, mask, mutants)
    
    # No return statement, population is mutated in place
    
def replace(parents, offspring, parent_scores, params):
    '''Replaces the population with a mixture of fittest mutated offspring and elite parents.
    Also returns fitness scores of new population to reduce later calculations.'''
    # Compute fitness scores of children
    child_scores = compute_fitness(offspring, params)
    
    # Determine sorted indices based on fitness scores (decreasing order)
    parent_indices = np.argsort(parent_scores)[::-1]
    child_indices = np.argsort(child_scores)[::-1]
    
    # Sort parents and offspring
    parents[:] = parents[parent_indices]
    offspring[:] = offspring[child_indices]
    
    # Sort parent and offspring fitness scores
    parent_scores[:] = parent_scores[parent_indices]
    child_scores[:] = child_scores[child_indices]
    
    # Compose new population with specified proportion of elite parents and offspring
    return np.concatenate((
        parents[:int(params.elitism * len(parents))],
        offspring[:int((1 - params.elitism) * len(offspring))]
    )), np.concatenate((
        parent_scores[:int(params.elitism * len(parents))],
        child_scores[:int((1 - params.elitism) * len(offspring))]
    ))   

def time_added(current, next, params):
    '''Given the latest (current) ride of an itinerary, the function returns how much time it
    would take to visit the given next ride. This time includes walking, wait, and ride time.'''
    # Determine (X,Y) locations of rides
    current_loc = params.locations[current + 1]
    next_loc = params.locations[next + 1]
    
    # Calculate walking time
    walk = walking_time(current_loc, next_loc, params)
    
    # Calculate ride and wait time
    ride = params.ride_times[next]
    wait = params.wait_times[next]
    
    # Return sum of individual times
    return walk + wait + ride           
            
def walking_time(a, b, params):
    '''Compute euclidean pixel distance between 2 attractions'''
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) * params.walk_rate        

def add_path(gui, a, b, **kwargs):
    '''Add a straight-line path from point a to point b on the map.'''
    line = Line(Point(a[0], a[1]), Point(b[0], b[1]))
    line.setWidth(3)
    line.setOutline(kwargs['color'])
    line.draw(gui)

def add_rides(gui, locations, **kwargs):
    '''Add colored marker for each ride on the map using ride locations.'''
    for ride, x, y in locations.itertuples(index=False):
        marker = Circle(Point(x, y), 8)
        marker.setOutline(kwargs['color'])
        marker.setWidth(3)
        marker.draw(gui)

def add_route(gui, plan, locations, fun, ridetime, wait, **kwargs):
    '''Display the planned route on the map, where the plan is a list of the rides to visit
    sequentially. The first and last location in the plan must be 'Entrance'; if it is not,
    the code adds it to the plan automatically (i.e. the GA doesn't need to include that).'''
    # Make sure the plan is a numpy array
    plan = np.array(plan)
    
    # Make sure the plan starts and ends at the 'Entrance'
    if plan[0] != 'Entrance':
        plan = np.insert(plan, 0, 'Entrance')
    if plan[-1] != 'Entrance':
        plan = np.append(plan, 'Entrance')
        
    # Copy of fun scores to penalize repeat attractions
    fun_scores = fun.copy()

    # Display the plan in the terminal and visually on the GUI
    print("Here's the plan...")
    total_time = 0
    total_fun = 0
    ride = plan[0]
    width_ride = max([len(ride) for ride in plan])
    width_fun = len(str(fun[1].max()))
    print(f'{0:>2} {convert_time(total_time)} {ride:<{width_ride}}')
    for i in range(1, plan.shape[0]):
        # Locations
        a = locations.loc[locations[0] == ride, [1, 2]].values[0]
        ride = plan[i]
        b = locations.loc[locations[0] == ride, [1, 2]].values[0]

        # Compute change in time
        travel_time = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) * kwargs['walk']
        total_time += travel_time
        if ride != "Entrance":
            wait_i = wait.loc[wait[0] == ride, 1].values[0]
            ridetime_i = ridetime.loc[ridetime[0] == ride, 1].values[0]
            total_time += wait_i + ridetime_i
        
        # Stop plotting route if time limit exceeded 
        if total_time > kwargs['time']:
            break
          
        # Update GUI  
        add_path(gui, a, b, **kwargs)
        time.sleep(kwargs['delay'])

        # Compute change in fun
        if ride != 'Entrance':
            fun_i = fun_scores.loc[fun[0] == ride, 1].values[0]
            fun_scores.loc[fun[0] == ride, 1] = fun_scores.loc[fun[0] == ride, 1] // 2
            total_fun += fun_i
        else:
            fun_i = 0

        # Update terminal
        print(f'{i:>2} {convert_time(total_time)} {ride:<{width_ride}} +{fun_i:>{width_fun}d} = {total_fun}')

    print()
    add_path(gui, a, b, **kwargs)

def clear_route(gui):
    '''Undraw any routes on the map.'''
    indices = np.where([isinstance(item, Line) for item in gui.items])[0][::-1]
    for i in indices:
        gui.items[i].undraw()

def convert_time(minutes):
    '''Helper function to convert time in minutes (as a float) to the format HH:MM:SS.'''
    total = datetime.timedelta(minutes=minutes)
    hours, remainder = divmod(total.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f'{hours:02}:{minutes:02}:{seconds:02}'

def make_gui(**kwargs):
    '''Helper function to initialize relevant graphics.'''
    # Read image from file
    file = os.path.join('parks', kwargs['park'], 'map.png')
    img = Image(Point(0, 0), file)
    hei = img.getHeight()
    wid = img.getWidth()
    img.move(wid / 2, hei / 2)

    # Create graphics window
    gui = GraphWin(f"Walt Disney World ({kwargs['park']})", wid, hei)
    img.draw(gui)

    return gui

if __name__ == '__main__':
    main(**vars(parser.parse_args()))
