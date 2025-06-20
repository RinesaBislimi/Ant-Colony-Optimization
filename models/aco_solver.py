import random
import copy
from typing import List, Dict
from models.instance_data import InstanceData
from models.solution import Solution
from models.library import Library
from models.new_solver import NewSolver

class ACO_Solver:
    def __init__(self, num_ants: int = 10, evaporation_rate: float = 0.1, 
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

    def initialize_pheromones(self, data: InstanceData):
        for lib in data.libs:
            self.pheromone[lib.id] = 1.0

    def get_tweak_methods(self):
        return [
            self.new_solver.tweak_solution_swap_signed,
            self.new_solver.tweak_solution_swap_signed_with_unsigned,
            self.new_solver.tweak_solution_swap_same_books,
            self.new_solver.tweak_solution_swap_last_book,
            self.new_solver.tweak_solution_swap_neighbor_libraries,
            self.new_solver.tweak_solution_insert_library
        ]

    def select_guided_tweak(self, solution: Solution, data: InstanceData):
            """Try all tweak methods and pick the one with the best fitness score."""
            tweaks = [
                ("swap_signed", self.new_solver.tweak_solution_swap_signed),
                ("swap_signed_with_unsigned", self.new_solver.tweak_solution_swap_signed_with_unsigned),
                ("swap_same_books", self.new_solver.tweak_solution_swap_same_books),
                ("swap_last_book", self.new_solver.tweak_solution_swap_last_book),
                ("swap_neighbor_libraries", self.new_solver.tweak_solution_swap_neighbor_libraries),
                ("insert_library", self.new_solver.tweak_solution_insert_library),
            ]
            best_score = float('-inf')
            best_solution = None
            best_name = None
            for name, tweak in tweaks:
                candidate = tweak(solution, data)
                if candidate and hasattr(candidate, "fitness_score") and candidate.fitness_score > best_score:
                    best_score = candidate.fitness_score
                    best_solution = candidate
                    best_name = name
            self.last_tweak_method_name = best_name
            # Return a function that, when called, returns the best solution
            return lambda sol, dat: best_solution

    def heuristic_information(self, lib_id: int, data: InstanceData) -> float:
        lib = data.libs[lib_id]
        total_score = sum(data.scores[book.id] for book in lib.books)
        return total_score / lib.signup_days if lib.signup_days > 0 else 0

    def calculate_probability(self, lib_id: int, data: InstanceData, visited: set) -> float:
        if lib_id in visited:
            return 0.0
        tau = self.pheromone.get(lib_id, 1.0) ** self.alpha
        eta = self.heuristic_information(lib_id, data) ** self.beta
        return tau * eta

    def construct_solution(self, data: InstanceData) -> Solution:
        Library._id_counter = 0
        solution = Solution([], [], {}, set())
        curr_time = 0
        visited = set()

        while curr_time < data.num_days and len(visited) < len(data.libs):
            valid_libs = []
            probabilities = []

            for lib in data.libs:
                if lib.id not in visited:
                    if curr_time + lib.signup_days < data.num_days:
                        prob = self.calculate_probability(lib.id, data, visited)
                        probabilities.append(prob)
                        valid_libs.append(lib.id)
                    
            if not valid_libs:
                break

            total = sum(probabilities)
            if total == 0:
                selected_id = random.choice(valid_libs)
            else:
                probabilities = [p / total for p in probabilities]
                selected_id = random.choices(valid_libs, weights=probabilities, k=1)[0]

            visited.add(selected_id)
            library = data.libs[selected_id]

            time_after_signup = curr_time + library.signup_days
            time_left = data.num_days - time_after_signup

            # Skip library if no time left for scanning after signup
            if time_left <= 0:
                solution.unsigned_libraries.append(selected_id)
                continue

            max_books_scanned = time_left * library.books_per_day

            # Select books sorted by descending score
            available_books_all = sorted(
                {book.id for book in library.books} - solution.scanned_books,
                key=lambda b: -data.scores[b]
            )

            available_books = available_books_all[:max_books_scanned]

            # If no books available to scan, mark unsigned
            if len(available_books) == 0:
                solution.unsigned_libraries.append(selected_id)
                continue

            # Otherwise, sign the library and update solution
            solution.signed_libraries.append(selected_id)
            solution.scanned_books_per_library[selected_id] = available_books
            solution.scanned_books.update(available_books)
            curr_time = time_after_signup

        solution.calculate_fitness_score(data.scores)
        return solution


    def update_pheromones(self, solutions: List[Solution], data: InstanceData):
        for lib_id in self.pheromone:
            self.pheromone[lib_id] *= (1 - self.evaporation_rate)

        for solution in solutions:
            delta = solution.fitness_score / sum(data.scores)
            for lib_id in solution.signed_libraries:
                self.pheromone[lib_id] += delta

    def run(self, data: InstanceData) -> Solution:
        self.initialize_pheromones(data)

        for _ in range(self.max_iterations):
            solutions = []

            for _ in range(self.num_ants):
                solution = self.construct_solution(data)
                tweak_method = self.select_guided_tweak(solution, data)
                tweaked = tweak_method(copy.deepcopy(solution), data)

                if tweaked and tweaked.fitness_score > solution.fitness_score:
                    solution = tweaked

                solutions.append(solution)

                if solution.fitness_score > self.best_score:
                    self.best_score = solution.fitness_score
                    self.best_solution = copy.deepcopy(solution)

            self.update_pheromones(solutions, data)

        return self.best_solution
