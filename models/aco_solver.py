import random
import copy
from typing import List, Dict, Set
from models.instance_data import InstanceData
from models.solution import Solution
from models.new_solver import NewSolver

class ACO_Solver:
    def __init__(self, num_ants: int = 2, evaporation_rate: float = 0.1, 
                 alpha: float = 1.0, beta: float = 2.0, max_iterations: int = 10):
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.pheromone = {}
        self.best_solution = None
        self.best_score = -1
        self.new_solver = NewSolver()
        self.last_tweak_method_name = "None"
        self._tweak_methods = [
            self.new_solver.tweak_solution_swap_signed,
            self.new_solver.tweak_solution_swap_signed_with_unsigned,
            self.new_solver.tweak_solution_swap_same_books,
            self.new_solver.tweak_solution_swap_last_book,
            self.new_solver.tweak_solution_swap_neighbor_libraries,
            self.new_solver.tweak_solution_insert_library
        ]
        self._swap_methods = [
            self.new_solver.tweak_solution_swap_signed,
            self.new_solver.tweak_solution_swap_neighbor_libraries
        ]
        self._book_tweak_methods = [
            self.new_solver.tweak_solution_swap_last_book,
            self.new_solver.tweak_solution_swap_same_books
        ]

    def initialize_pheromones(self, data: InstanceData):
        """Initialize pheromone trails with default value"""
        self.pheromone = {lib.id: 1.0 for lib in data.libs}

    def select_guided_tweak(self, solution: Solution, data: InstanceData):
        """Select an appropriate tweak method based on solution state"""
        if solution.unsigned_libraries:
            self.last_tweak_method_name = "insert_library"
            return self.new_solver.tweak_solution_insert_library
        
        if len(solution.signed_libraries) >= 2:
            method = random.choice(self._swap_methods)
            self.last_tweak_method_name = method.__name__.split('.')[-1]
            return method
        
        method = random.choice(self._book_tweak_methods)
        self.last_tweak_method_name = method.__name__.split('.')[-1]
        return method

    def heuristic_information(self, lib_id: int, data: InstanceData) -> float:
        """Calculate heuristic information for a library (score per signup day)"""
        try:
            lib = next(lib for lib in data.libs if lib.id == lib_id)
        except StopIteration:
            return 0.0
        
        if lib.signup_days == 0:
            return 0.0
        
        # Calculate total score of books in the library
        total_score = sum(data.scores[book.id] for book in lib.books)
        return total_score / lib.signup_days

    def calculate_probability(self, lib_id: int, data: InstanceData, visited: Set[int]) -> float:
        """Calculate selection probability for a library"""
        if lib_id in visited:
            return 0.0
            
        tau = self.pheromone.get(lib_id, 1.0) ** self.alpha
        eta = self.heuristic_information(lib_id, data) ** self.beta
        return tau * eta

    def construct_solution(self, data: InstanceData) -> Solution:
        """Construct a solution using pheromone trails and heuristic information"""
        solution = Solution([], [], {}, set())
        curr_time = 0
        visited = set()
        lib_dict = {lib.id: lib for lib in data.libs}
        unvisited_libs = set(lib_dict.keys())

        while curr_time < data.num_days and unvisited_libs:
            # Calculate probabilities for unvisited libraries
            probabilities = []
            valid_libs = []

            for lib_id in unvisited_libs:
                prob = self.calculate_probability(lib_id, data, visited)
                if prob > 0:
                    probabilities.append(prob)
                    valid_libs.append(lib_id)

            if not valid_libs:
                break

            # Select library based on probability
            if sum(probabilities) == 0:
                selected_id = random.choice(valid_libs)
            else:
                selected_id = random.choices(valid_libs, weights=probabilities, k=1)[0]

            visited.add(selected_id)
            unvisited_libs.remove(selected_id)

            library = lib_dict[selected_id]

            if curr_time + library.signup_days >= data.num_days:
                solution.unsigned_libraries.append(selected_id)
                continue

            # Calculate available books
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books = time_left * library.books_per_day
            available_books = [book.id for book in library.books if book.id not in solution.scanned_books][:max_books]

            if available_books:
                solution.signed_libraries.append(selected_id)
                solution.scanned_books_per_library[selected_id] = available_books
                solution.scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                solution.unsigned_libraries.append(selected_id)

        solution.calculate_fitness_score(data.scores)
        return solution
                
    def run(self, data: InstanceData) -> Solution:
        """Run the ACO algorithm to find the best solution"""
        self.initialize_pheromones(data)
        
        for iteration in range(self.max_iterations):
            solutions = []
            
            for _ in range(self.num_ants):
                # Construct and tweak solution
                solution = self.construct_solution(data)
                tweak_method = self.select_guided_tweak(solution, data)
                tweaked_solution = tweak_method(solution, data)
                
                # Keep the better solution
                if tweaked_solution.fitness_score > solution.fitness_score:
                    solution = tweaked_solution
                
                solutions.append(solution)
                
                # Update best solution found so far
                if solution.fitness_score > self.best_score:
                    self.best_score = solution.fitness_score
                    self.best_solution = copy.deepcopy(solution)
            
            self.update_pheromones(solutions, data)
        
        return self.best_solution
    def update_pheromones(self, solutions, data):
        """Update pheromone trails based on the solutions found by the ants."""
        # Evaporate pheromones
        for lib_id in self.pheromone:
            self.pheromone[lib_id] *= (1 - self.evaporation_rate)
            # Prevent pheromone from dropping too low
            if self.pheromone[lib_id] < 1e-6:
                self.pheromone[lib_id] = 1e-6

        # Reinforce pheromones based on solution quality
        # Optionally, only reinforce the top N solutions for more exploitation
        top_solutions = sorted(solutions, key=lambda s: s.fitness_score, reverse=True)[:max(1, len(solutions)//3)]
        for solution in top_solutions:
            for lib_id in solution.signed_libraries:
                # The reinforcement can be proportional to the solution's fitness
                self.pheromone[lib_id] += solution.fitness_score / (1 + len(solution.signed_libraries))

        # Optionally normalize pheromones to avoid overflow
        max_pheromone = max(self.pheromone.values())
        if max_pheromone > 1e6:
            for lib_id in self.pheromone:
                self.pheromone[lib_id] /= max_pheromone