<br />
<p align="center">
  <h1 align="center">Genetic Algorithm for Disney Trip Planning</h1>

  <p align="center">
This project uses a Genetic Algorithm (GA) to plan the most enjoyable Disney World trip itinerary, balancing fun and time constraints across theme park rides and attractions.</p>

## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
* [Running The Program](#running-the-program)
  * [Prerequisites](#prerequisites)
* [Evaluation](#evaluation)

## About The Project

### Data

For this project, the dataset consists of multiple CSV files for each theme park, located in a parks directory. These files contain data on ride locations, fun scores, wait times, and ride times:   

- map.png: A map of the park.   
- location.csv: A list of rides with their (x, y) pixel locations on the map, including the park entrance.   
- fun.csv: User-defined "fun" values for each ride, scaled for comparison.   
- wait.csv: Average wait times (in minutes) for each ride.   
- ridetime.csv: Average duration (in minutes) of each ride.   

For the Magic Kingdom, data was provided, and additional data was created for another theme park.   

### Steps

This project aims to optimize Disney park itineraries using a Genetic Algorithm, solving the following key problems:   

- Fitness Function: Calculate the total fun experienced within a time limit (default: 10 hours). This considers:     
  - Travel time between attractions (based on Euclidean distance and walking speed).   
  - Wait time in line for each attraction.   
  - Ride time for each attraction.  

- Dimensionality Reduction and Constraints: Adjust ride schedules dynamically:   
  - Penalize repeated attractions by reducing their fun scores each time.   
  - Stop computation if the time limit is exceeded.   
  - Begin and end each itinerary at the park entrance.   

- Genetic Algorithm Implementation:   
  - Population Initialization: Generate an initial population of 50 itineraries, each with up to 20 rides.  
  - Fitness Evaluation: Evaluate itineraries by balancing fun and time constraints.   
  - Selection: Use tournament selection to choose parent itineraries based on fitness.   
  - Crossover: Apply a modified 3-point crossover to combine parent itineraries.  
  - Mutation: Use inversion and random resetting mutations to explore new itineraries.   
  - Survivor Selection: Replace the population with elite parents and the fittest offspring.
 
### Results   
The algorithm produces optimized Disney park itineraries, maximizing fun while adhering to constraints. Improvements in the second submission reduced computational time, making the program faster and more efficient.  

## Getting Started 

To get a local copy up and running follow these simple steps.

### Prerequisites

* [Python 3.12.3](https://www.python.org/downloads/) or higher    
* Required Python libraries:     
- numpy   
- pandas   
- matplotlib   

## Running The Program

Run the program using the following command:    

- python disney.py -p magic-kingdom   

Additional command-line arguments allow customization:    
- --park: Specify the park folder (default: magic-kingdom).   
- --time: Set the time limit in minutes (default: 600).   
- --mutation: Set the mutation rate (default: 0.75).   
- --generations: Set the maximum number of generations (default: 300).    

## Evaluation

The analysis file in the repository provides detailed insights into the Genetic Algorithm's performance, runtime improvements, and challenges. The algorithm efficiently plans Disney itineraries, ensuring maximum enjoyment within constraints.   
<!-- If you want to provide some contact details, this is the place to do it -->

<!-- ## Acknowledgements  -->
