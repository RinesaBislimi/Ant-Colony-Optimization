import random
import copy
from typing import List, Dict
from models.instance_data import InstanceData
from models.solution import Solution
from models.library import Library
from models.solver_guided import Solver_Guided

class ACO_Solver:
    def __init__(self, num_ants: int = 10, evaporation_rate: float = 0.1, 
                 alpha: float = 1.0, beta: float = 2.0, max_iterations: int = 100):
        """
        Initialize the ACO solver with parameters:
        - num_ants: Number of ants in the colony
        - evaporation_rate: Rate at which pheromone evaporates
        - alpha: Importance of pheromone trail
        - beta: Importance of heuristic information
        - max_iterations: Maximum number of iterations
        """
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.pheromone = {}
        self.best_solution = None
        self.best_score = -1
        self.guided_solver = Solver_Guided()

    def initialize_pheromones(self, data: InstanceData):
        """Initialize pheromone trails for all libraries"""
        initial_pheromone = 1.0
        for lib in data.libs:
            self.pheromone[lib.id] = initial_pheromone

    def get_tweak_methods(self) -> list:
        """Return all available tweak methods from Solver_Guided"""
        return [
            self.guided_solver.tweak_solution_swap_signed,
            self.guided_solver.tweak_solution_swap_signed_with_unsigned,
            self.guided_solver.tweak_solution_swap_same_books,
            self.guided_solver.tweak_solution_insert_library,
            self.guided_solver.tweak_solution_swap_last_book,
            self.guided_solver.tweak_solution_swap_neighbor_libraries,
            self.guided_solver.tweak_solution_shuffle_books,
            self.guided_solver.tweak_solution_swap_signed_guided
        ]

    def select_random_tweak(self):
        """Randomly select one of the available tweak methods"""
        tweak_methods = self.get_tweak_methods()
        selected_method = random.choice(tweak_methods)
        self.last_tweak_method_name = selected_method.__name__ 
        return selected_method


    def heuristic_information(self, lib_id: int, data: InstanceData) -> float:
        """Calculate heuristic information for a library (score per signup day)"""
        lib = data.libs[lib_id]
        total_score = sum(data.scores[book.id] for book in lib.books)
        return total_score / lib.signup_days if lib.signup_days > 0 else 0

    def calculate_probability(self, lib_id: int, data: InstanceData, visited: set) -> float:
        """Calculate selection probability for a library"""
        if lib_id in visited:
            return 0.0
            
        tau = self.pheromone.get(lib_id, 1.0) ** self.alpha
        eta = self.heuristic_information(lib_id, data) ** self.beta
        return tau * eta

    def construct_solution(self, data: InstanceData) -> Solution:
        """Construct a solution using pheromone trails and heuristic information"""
        Library._id_counter = 0
        solution = Solution([], [], {}, set())
        curr_time = 0
        visited = set()

        while curr_time < data.num_days and len(visited) < len(data.libs):
            # Calculate probabilities for all unvisited libraries
            probabilities = []
            valid_libs = []
            
            for lib in data.libs:
                if lib.id not in visited:
                    prob = self.calculate_probability(lib.id, data, visited)
                    probabilities.append(prob)
                    valid_libs.append(lib.id)
            
            if not valid_libs:
                break
                
            # Normalize probabilities
            total = sum(probabilities)
            if total == 0:
                # If all probabilities are zero, select randomly
                selected_id = random.choice(valid_libs)
            else:
                # Select library based on probability
                probabilities = [p/total for p in probabilities]
                selected_id = random.choices(valid_libs, weights=probabilities, k=1)[0]
            
            visited.add(selected_id)
            library = data.libs[selected_id]
            
            if curr_time + library.signup_days >= data.num_days:
                solution.unsigned_libraries.append(selected_id)
                continue
                
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            
            available_books = sorted(
                {book.id for book in library.books} - solution.scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]
            
            if available_books:
                solution.signed_libraries.append(selected_id)
                solution.scanned_books_per_library[selected_id] = available_books
                solution.scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                solution.unsigned_libraries.append(selected_id)
        
        solution.calculate_fitness_score(data.scores)
        return solution

    def update_pheromones(self, solutions: List[Solution], data: InstanceData):
        """Update pheromone trails based on ant solutions"""
        # Evaporate pheromones
        for lib_id in self.pheromone:
            self.pheromone[lib_id] *= (1 - self.evaporation_rate)
        
        # Add new pheromones based on solutions
        for solution in solutions:
            delta = solution.fitness_score / sum(data.scores)  # Normalized delta
            for lib_id in solution.signed_libraries:
                self.pheromone[lib_id] += delta

    def run(self, data: InstanceData) -> Solution:
        """Run the ACO algorithm"""
        self.initialize_pheromones(data)
        
        for iteration in range(self.max_iterations):
            solutions = []
            
            # Let each ant construct a solution
            for _ in range(self.num_ants):
                solution = self.construct_solution(data)
                
                # Apply a random tweak to the solution
                tweak_method = self.select_random_tweak()
                tweaked_solution = tweak_method(copy.deepcopy(solution), data)
                
                # Only keep the tweaked solution if it's valid and better
                if tweaked_solution is not None and tweaked_solution.fitness_score > solution.fitness_score:
                    solution = tweaked_solution
                
                solutions.append(solution)
                
                # Update best solution found
                if solution.fitness_score > self.best_score:
                    self.best_score = solution.fitness_score
                    self.best_solution = copy.deepcopy(solution)
            
            # Update pheromones based on all solutions
            self.update_pheromones(solutions, data)
        
        return self.best_solution