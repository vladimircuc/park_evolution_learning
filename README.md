[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zbf3PMZX)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=16891022&assignment_repo_type=AssignmentRepo)
# HW2: Theme Park Trip Planning

> "Sometimes the right path is not the easiest one"
>
> &mdash; <cite>Grandmother Willow in _Pocahontas_ (1995)</cite>

![alt text](https://github.com/FloridaSouthernCS/csc4510-f24-hw2/blob/main/sample_routes_slow.gif "Magic Kingdom Routes")

Let's take a trip to Disney World! Path planning is an excellent application for **_evolutionary learning_** techniques due to the adaptability of such algorithms for complex search spaces with many local optima. In this assignment, you will work in pairs to develop Python code that implements a genetic algorithm to plan a trip to various theme parks.

## Data

Theme park information is stored in separate folders in the `parks` directory. The required files for every theme park are listed below:

1. `map.png`: Image of the park map.
2. `location.csv`: Comma-separated list of ride names along with the corresponding (x, y) pixel location on the map. _The park entrance should be included in this file._
3. `fun.csv`: List of user-defined "fun" values for each ride in the park. The higher the value, the more fun a ride is. Scale is up to you, but 1-10 or 1-100 are recommended options.
4. `wait.csv`: List of average wait times (in minutes) for each ride in the park.
5. `ridetime.csv`: List of ride length (in minutes) for each ride in the park.

Some relevant notes:
* Park folder names should not include any spaces because a user will specify the park as a command-line argument.
* Each file must be named exactly as listed above in order for the default code to work.
* The term "ride" does not preclude using shows or other attractions at the park.
* With the exception of the _Entrance_ in the location file, the same set of rides must exist in each of the csv files.
* The order of rides does not matter in the csv files.
* __Please do not push changes to the Magic Kingdom data provided! This will allow some standardization across all student submissions.__

## Instructions

Starter code has been provided for you in `disney.py`. This code should run without error prior to editing. Study the code, identify possible command-line arguments, and explore various user interactions available (e.g. debugging, randomizing a route, clearing a route, ...). You DO NOT need to edit any of the provided helper functions. Instead, your programming tasks are as follows:

1. Create a function that implements a genetic algorithm for the trip planning problem. You are given considerable flexibility in designing the parameters of the algorithm (e.g. representation, selection, crossover, mutation) with the exception of fitness (see description below). You are also permitted (and even encouraged) to create additional functions as needed for the GA.

2. The fitness of a candidate solution should equal the total amount of fun experienced within a given time limit (add this limit as an optional command-line argument with a default value of 10 hours).

	* There are three types of time to consider:
		1. travel time - how long it takes to walk between attractions
		2. wait time - how long it takes to wait in line for an attraction
		3. ride time - how long it takes to ride an attraction
	* Travel time is computed via straight-line distance between two attractions on the map; assume a conversion rate of 0.02 minutes per pixel (e.g. if two rides are 200 pixels apart, then the travel time between them is 200 x 0.02 = 4 minutes).
	* You are allowed to ride an attraction more than once; if you ride Space Mountain twice in a row, for instance, you should double the wait time and ride time in the computation of fitness.
	* You are encouraged to creatively adjust the fun of each attraction after you ride it; that is, if The Hall of Presidents is the most fun attraction, we probably don't want to do it nonstop for the entire day!
	* Regardless of the length of your genotype, you should (computationally) stop considering rides after you reach the time limit.
	* Every planned route must start from the park entrance; you do not need to include the entrance as a gene in the genotype, but you should include the travel time from the entrance to the first ride in the computation of time spent in the park.
	* Every planned route will end by leaving the park through the park entrance, but this does not need to be included in the time limit.

3. Call the GA function in the main method of the program to run the algorithm and visualize the results using the existing helper functions.

In addition to writing code, you must also complete the following tasks:

1. Data for the Magic Kingdom has already been provided, but perhaps that is not the only theme park you want to visit. As a team, create the necessary data files for another theme park. Follow the conventions listed above in the Data section.

2. You are strongly encouraged to submit a short PDF that **_briefy_** and **_informally_** discusses the results of your program. Are you surprised by the performance of the GA? Did you encounter any issues or challenges? Did you integrate any particularly creative ideas? Add the PDF to your repo prior to submission.

## Submission Requirements

To earn credit for this assignment, you must commit and push any code changes to your group repository by the posted deadline (November 13, 2024 @ 9:25 AM EST). When you are ready for the instructor to review your code, please send a direct message on Slack.

Recall from the course syllabus that students are strongly encouraged to submit assignments well in advance of the deadline because they will be allowed to edit and resubmit each homework assignment for grade improvement, if they desire. Assignments can be resubmitted multiple times, as long as the grade improves each time.
