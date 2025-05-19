from collections import defaultdict, deque
import threading
import time
from models.library import Library
import os
from models.solution import Solution
from models.solver import Solver
import copy
import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
from typing import Tuple
from models.instance_data import InstanceData
import random

# Deterministic Cooling Functions
def cooling_exponential(temp, cooling_rate=0.003):
    return temp * (1 - cooling_rate)

def cooling_geometric(temp, alpha=0.95):
    return temp * alpha

def cooling_lundy_mees(temp, beta=0.001):
    return temp / (1 + beta * temp)

def _pool_init(instance_data: InstanceData, hc_steps: int, mutation_prob: float):
    global INSTANCE, HC_STEPS, MUT_PROB, SOLVER
    INSTANCE    = instance_data
    HC_STEPS    = hc_steps
    MUT_PROB    = mutation_prob
    SOLVER      = Solver()

def _process_offspring(sol: Solution) -> Solution:
    """Deterministic mutation + hill-climb on one offspring."""
    if sol.fitness_score % 2 == 0:  # Deterministic condition instead of random
        _, sol = SOLVER.hill_climbing_combined_w_initial_solution(sol, INSTANCE, iterations=HC_STEPS)
    return sol

class Solver_Guided:
    def __init__(self):
        self.iteration_counter = 0
        self.last_improvement = 0
        self.tweak_weights = [1.0, 1.0, 1.0]  # For deterministic tweak selection
        
    def generate_initial_solution(self, data):
        Library._id_counter = 0
        
        # Sort libraries by signup time and total score (deterministic)
        sorted_libs = sorted(data.libs, key=lambda l: (l.signup_days, -sum(data.scores[b.id] for b in l.books)))

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        for library in sorted_libs:
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(library.id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books, 
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                signed_libraries.append(library.id)
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days

        solution = Solution(signed_libraries, unsigned_libraries, scanned_books_per_library, scanned_books)
        solution.calculate_fitness_score(data.scores)
        return solution

    def deterministic_crossover(self, solution, data):
        """Deterministic crossover using library scoring."""
        new_solution = copy.deepcopy(solution) 

        # Sort libraries by their total book score (deterministic)
        library_scores = []
        for lib_id in solution.signed_libraries:
            lib = data.libs[lib_id]
            total_score = sum(data.scores[book.id] for book in lib.books)
            library_scores.append((lib_id, total_score))
        
        # Sort by score descending
        library_scores.sort(key=lambda x: -x[1])
        new_order = [lib_id for lib_id, _ in library_scores]

        new_scanned_books_per_library = {}
        lib_lookup = {lib.id: lib for lib in data.libs}

        # Process libraries in new order
        curr_time = 0
        scanned_books = set()
        new_signed_libraries = []
        
        for lib_id in new_order:
            library = lib_lookup[lib_id]
            
            if curr_time + library.signup_days >= data.num_days:
                new_solution.unsigned_libraries.append(lib_id)
                continue
                
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days

        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = scanned_books
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def deterministic_tweak_selection(self):
        """Deterministically select tweak method based on iteration count."""
        methods = [
            self.tweak_solution_swap_signed_with_unsigned,
            self.tweak_solution_swap_same_books,
            self.tweak_solution_swap_signed
        ]
        return methods[self.iteration_counter % len(methods)]

    def tweak_solution_swap_signed(self, solution, data):
        """Deterministic swap of libraries based on their scores."""
        if len(solution.signed_libraries) < 2:
            return solution

        new_solution = copy.deepcopy(solution)

        # Find the two libraries with smallest score difference
        lib_scores = []
        for lib_id in solution.signed_libraries:
            lib = data.libs[lib_id]
            total_score = sum(data.scores[book.id] for book in lib.books)
            lib_scores.append((lib_id, total_score))
        
        # Sort by score to find adjacent pairs
        lib_scores.sort(key=lambda x: x[1])
        
        # Select pair with smallest difference
        min_diff = float('inf')
        idx1, idx2 = 0, 1
        for i in range(len(lib_scores)-1):
            diff = abs(lib_scores[i][1] - lib_scores[i+1][1])
            if diff < min_diff:
                min_diff = diff
                idx1, idx2 = i, i+1

        lib_id1, lib_id2 = lib_scores[idx1][0], lib_scores[idx2][0]

        # Find their positions in the original list
        pos1 = solution.signed_libraries.index(lib_id1)
        pos2 = solution.signed_libraries.index(lib_id2)

        new_signed_libraries = solution.signed_libraries.copy()
        new_signed_libraries[pos1] = lib_id2
        new_signed_libraries[pos2] = lib_id1

        # Rebuild solution with new order
        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}

        for lib_id in new_signed_libraries:
            library = data.libs[lib_id]

            if curr_time + library.signup_days >= data.num_days:
                new_solution.unsigned_libraries.append(lib_id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                new_solution.unsigned_libraries.append(lib_id)

        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = scanned_books
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def tweak_solution_swap_signed_with_unsigned(self, solution, data):
        """Deterministic swap between signed and unsigned libraries."""
        if not solution.signed_libraries or not solution.unsigned_libraries:
            return solution

        # Find the worst signed library (lowest total score)
        signed_scores = []
        for lib_id in solution.signed_libraries:
            lib = data.libs[lib_id]
            total_score = sum(data.scores[book.id] for book in lib.books)
            signed_scores.append((lib_id, total_score))
        worst_signed = min(signed_scores, key=lambda x: x[1])[0]

        # Find the best unsigned library (highest total score)
        unsigned_scores = []
        for lib_id in solution.unsigned_libraries:
            lib = data.libs[lib_id]
            total_score = sum(data.scores[book.id] for book in lib.books)
            unsigned_scores.append((lib_id, total_score))
        best_unsigned = max(unsigned_scores, key=lambda x: x[1])[0]

        # Create new solution with the swap
        new_solution = copy.deepcopy(solution)
        signed_pos = new_solution.signed_libraries.index(worst_signed)
        unsigned_pos = new_solution.unsigned_libraries.index(best_unsigned)

        new_solution.signed_libraries[signed_pos] = best_unsigned
        new_solution.unsigned_libraries[unsigned_pos] = worst_signed

        # Rebuild the solution from the swap point
        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}

        # Process libraries before swap point
        for i in range(signed_pos):
            lib_id = new_solution.signed_libraries[i]
            if lib_id in solution.scanned_books_per_library:
                books = solution.scanned_books_per_library[lib_id]
                new_scanned_books_per_library[lib_id] = books
                scanned_books.update(books)
                curr_time += data.libs[lib_id].signup_days

        # Re-process from swap point
        new_signed_libraries = new_solution.signed_libraries[:signed_pos]
        
        for i in range(signed_pos, len(new_solution.signed_libraries)):
            lib_id = new_solution.signed_libraries[i]
            library = data.libs[lib_id]

            if curr_time + library.signup_days >= data.num_days:
                new_solution.unsigned_libraries.append(lib_id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                new_solution.unsigned_libraries.append(lib_id)

        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = scanned_books
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def tweak_solution_swap_same_books(self, solution, data):
        """Deterministic swap of libraries that share books."""
        library_ids = [lib for lib in solution.signed_libraries if lib < len(data.libs)]
        if len(library_ids) < 2:
            return solution

        # Find libraries with overlapping books
        book_to_libs = defaultdict(list)
        for lib_id in library_ids:
            for book in data.libs[lib_id].books:
                book_to_libs[book.id].append(lib_id)

        # Find pair with most overlapping books
        max_overlap = 0
        best_pair = (0, 1)
        for book, libs in book_to_libs.items():
            if len(libs) > 1:
                for i in range(len(libs)):
                    for j in range(i+1, len(libs)):
                        if (libs[i], libs[j]) in book_to_libs:
                            overlap = len(set(data.libs[libs[i]].books) & set(data.libs[libs[j]].books))
                            if overlap > max_overlap:
                                max_overlap = overlap
                                best_pair = (libs[i], libs[j])

        if max_overlap == 0:
            return solution

        lib1, lib2 = best_pair
        pos1 = library_ids.index(lib1)
        pos2 = library_ids.index(lib2)
        library_ids[pos1], library_ids[pos2] = library_ids[pos2], library_ids[pos1]

        # Rebuild solution with new order
        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        for lib_id in library_ids:
            library = data.libs[lib_id]
            
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(library.id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                signed_libraries.append(library.id)
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                unsigned_libraries.append(library.id)

        new_solution = Solution(
            signed_libraries,
            unsigned_libraries,
            scanned_books_per_library,
            scanned_books
        )
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    def deterministic_hill_climbing(self, data, iterations=1000):
        """Deterministic hill climbing with guided tweak selection."""
        solution = self.generate_initial_solution(data)
        
        for i in range(iterations):
            # Select tweak method based on iteration count
            tweak_method = self.deterministic_tweak_selection()
            new_solution = tweak_method(copy.deepcopy(solution), data)
            
            if new_solution.fitness_score > solution.fitness_score:
                solution = new_solution
                self.last_improvement = i

        return solution.fitness_score, solution

    def deterministic_simulated_annealing(self, data, iterations=1000):
        """Deterministic simulated annealing with guided moves."""
        current = self.generate_initial_solution(data)
        best = copy.deepcopy(current)
        temp = 1000.0
        
        for i in range(iterations):
            # Select tweak based on iteration count
            tweak_idx = i % 3
            tweak_method = [
                self.tweak_solution_swap_signed,
                self.tweak_solution_swap_signed_with_unsigned,
                self.tweak_solution_swap_same_books
            ][tweak_idx]
            
            neighbor = tweak_method(copy.deepcopy(current), data)
            delta = neighbor.fitness_score - current.fitness_score
            
            # Deterministic acceptance based on temperature and delta
            if delta > 0 or (i % 10 == 0 and delta > -temp):
                current = neighbor
                if current.fitness_score > best.fitness_score:
                    best = copy.deepcopy(current)
            
            # Cool temperature deterministically
            temp = cooling_geometric(temp)
            
        return best.fitness_score, best

    # Other methods can be similarly modified to remove randomness
    # ...
    def tweak_solution_insert_library(self, solution, data, target_lib=None):
        """Deterministic library insertion based on scores."""
        # Return original solution if no unsigned libraries
        if not solution.unsigned_libraries:
            return solution

        new_solution = copy.deepcopy(solution)
        
        # If no target specified, find the best unsigned library
        if target_lib is None:
            # Filter valid unsigned libraries (those that exist in data.libs)
            valid_unsigned = [lib_id for lib_id in solution.unsigned_libraries 
                            if lib_id < len(data.libs)]
            if not valid_unsigned:
                return solution
                
            best_unsigned = max(
                valid_unsigned,
                key=lambda lib_id: sum(data.scores[b.id] for b in data.libs[lib_id].books)
            )
        else:
            # Verify target library exists and is unsigned
            if (target_lib >= len(data.libs) or 
                target_lib not in solution.unsigned_libraries):
                return solution
            best_unsigned = target_lib

        # Find position to insert (based on signup time)
        insert_pos = 0
        for i, lib_id in enumerate(solution.signed_libraries):
            if (lib_id < len(data.libs) and (best_unsigned < len(data.libs))):
                if data.libs[lib_id].signup_days > data.libs[best_unsigned].signup_days:
                    insert_pos = i
                    break
            insert_pos = i + 1

        # Insert the library (only if it's not already signed)
        if best_unsigned not in new_solution.signed_libraries:
            new_solution.signed_libraries.insert(insert_pos, best_unsigned)
            
            # Remove from unsigned only if it exists there
            if best_unsigned in new_solution.unsigned_libraries:
                new_solution.unsigned_libraries.remove(best_unsigned)

        # Rebuild solution from insertion point
        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}

        # Process libraries before insertion point
        for i in range(insert_pos):
            lib_id = new_solution.signed_libraries[i]
            if (lib_id in solution.scanned_books_per_library and 
                lib_id < len(data.libs)):
                books = solution.scanned_books_per_library[lib_id]
                new_scanned_books_per_library[lib_id] = books
                scanned_books.update(books)
                curr_time += data.libs[lib_id].signup_days

        # Re-process from insertion point
        new_signed_libraries = new_solution.signed_libraries[:insert_pos]
        
        for i in range(insert_pos, len(new_solution.signed_libraries)):
            lib_id = new_solution.signed_libraries[i]
            if lib_id >= len(data.libs):
                continue
                
            library = data.libs[lib_id]

            if curr_time + library.signup_days >= data.num_days:
                if lib_id not in new_solution.unsigned_libraries:
                    new_solution.unsigned_libraries.append(lib_id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                if lib_id not in new_solution.unsigned_libraries:
                    new_solution.unsigned_libraries.append(lib_id)

        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = scanned_books
        new_solution.calculate_fitness_score(data.scores)

        return new_solution
    
    def deterministic_guided_local_search(self, data, max_time=300):
        """Deterministic version of guided local search."""
        C = set(range(len(data.libs)))
        component_utilities = {
            i: sum(data.scores[book.id] for book in data.libs[i].books)
            for i in C if i < len(data.libs)  # Ensure valid library IDs
        }
        
        S = self.generate_initial_solution(data)
        Best = copy.deepcopy(S)
        p = [0] * len(data.libs)
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < max_time:
            iteration += 1
            
            # Get current components that are actually in the solution and valid
            current_components = {c for c in S.signed_libraries 
                                if c in C and c < len(data.libs)}
            if not current_components:
                break
                
            # Find component with highest utility/penalty ratio
            try:
                selected = max(
                    current_components,
                    key=lambda c: component_utilities.get(c, 0) / (1 + p[c])
                )
            except ValueError:
                break
            
            # Apply tweak only if selected is valid
            if selected < len(data.libs):
                R = self.tweak_solution_insert_library(S, data, target_lib=selected)
                
                if R.fitness_score > Best.fitness_score:
                    Best = copy.deepcopy(R)
                    
                # Update penalties
                if selected < len(p):
                    p[selected] += 1
                S = R
            
        return Best

    def construct_solution_grasp(self, data, alpha=0.3):
        """
        Construct a greedy randomized solution (GRASP approach).
        Alpha controls the greediness/randomness (0 = greedy, 1 = random).
        """
        Library._id_counter = 0

        # Step 1: Compute score per signup day for each library
        lib_scores = []
        for lib in data.libs:
            total_score = sum(data.scores[book.id] for book in lib.books)
            score_per_day = total_score / lib.signup_days if lib.signup_days > 0 else 0
            lib_scores.append((lib.id, score_per_day))

        # Step 2: Sort libraries by greedy metric
        lib_scores.sort(key=lambda x: -x[1])

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        available_libs = [lib_id for lib_id, _ in lib_scores]

        while curr_time < data.num_days and available_libs:
            # Step 3: Create a restricted candidate list (RCL)
            rcl_size = max(1, int(len(available_libs) * alpha))
            rcl = available_libs[:rcl_size]

            # Step 4: Select a library from RCL randomly
            chosen_id = random.choice(rcl)
            available_libs.remove(chosen_id)

            library = data.libs[chosen_id]
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(chosen_id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                signed_libraries.append(chosen_id)
                scanned_books_per_library[chosen_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                unsigned_libraries.append(chosen_id)

        # Construct the solution object
        solution = Solution(signed_libraries, unsigned_libraries, scanned_books_per_library, scanned_books)
        solution.calculate_fitness_score(data.scores)
        return solution
    
    def tweak_solution_swap_last_book(self, solution, data):
        """Deterministic version of swapping last book from a signed library with best available book from unsigned libraries."""
        if not solution.scanned_books_per_library or not solution.unsigned_libraries:
            return solution

        new_solution = copy.deepcopy(solution)
        
        # Find library with lowest score last book
        worst_lib = None
        worst_book_score = float('inf')
        worst_book = None
        
        for lib_id, books in solution.scanned_books_per_library.items():
            if not books:
                continue
            last_book = books[-1]
            book_score = data.scores[last_book]
            if book_score < worst_book_score:
                worst_book_score = book_score
                worst_book = last_book
                worst_lib = lib_id

        if worst_lib is None:
            return solution

        # Find best available book from unsigned libraries
        best_book = None
        best_score = -1
        best_lib = None
        
        for lib_id in solution.unsigned_libraries:
            if lib_id >= len(data.libs):
                continue
            library = data.libs[lib_id]
            for book in library.books:
                if book.id not in solution.scanned_books:
                    if data.scores[book.id] > best_score:
                        best_score = data.scores[book.id]
                        best_book = book.id
                        best_lib = lib_id
                    break  # Just check first available book per library

        if best_book is None:
            return solution

        # Perform the swap
        new_books = new_solution.scanned_books_per_library[worst_lib].copy()
        new_books.remove(worst_book)
        new_books.append(best_book)
        new_solution.scanned_books_per_library[worst_lib] = new_books
        
        new_solution.scanned_books.remove(worst_book)
        new_solution.scanned_books.add(best_book)
        
        new_solution.calculate_fitness_score(data.scores)
        
        return new_solution

    def tweak_solution_swap_neighbor_libraries(self, solution, data):
        """Deterministic swap of adjacent libraries with smallest score difference."""
        if len(solution.signed_libraries) < 2:
            return solution

        new_solution = copy.deepcopy(solution)
        
        # Calculate library scores
        lib_scores = []
        for lib_id in solution.signed_libraries:
            lib = data.libs[lib_id]
            total_score = sum(data.scores[book.id] for book in lib.books)
            lib_scores.append((lib_id, total_score))
        
        # Find adjacent pair with smallest score difference
        min_diff = float('inf')
        swap_pos = 0
        for i in range(len(lib_scores)-1):
            diff = abs(lib_scores[i][1] - lib_scores[i+1][1])
            if diff < min_diff:
                min_diff = diff
                swap_pos = i

        # Swap the libraries
        new_solution.signed_libraries[swap_pos], new_solution.signed_libraries[swap_pos+1] = \
            new_solution.signed_libraries[swap_pos+1], new_solution.signed_libraries[swap_pos]
        
        # Rebuild solution from swap point
        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}
        
        # Process libraries before swap point
        for i in range(swap_pos):
            lib_id = new_solution.signed_libraries[i]
            if lib_id >= len(data.libs):
                continue
            library = data.libs[lib_id]
            curr_time += library.signup_days
            
            if lib_id in solution.scanned_books_per_library:
                books = solution.scanned_books_per_library[lib_id]
                new_scanned_books_per_library[lib_id] = books
                scanned_books.update(books)
        
        # Re-process from swap point
        new_signed_libraries = new_solution.signed_libraries[:swap_pos]
        
        for i in range(swap_pos, len(new_solution.signed_libraries)):
            lib_id = new_solution.signed_libraries[i]
            if lib_id >= len(data.libs):
                continue
                
            library = data.libs[lib_id]
            
            if curr_time + library.signup_days >= data.num_days:
                new_solution.unsigned_libraries.append(lib_id)
                continue
                
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            
            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]
            
            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                new_solution.unsigned_libraries.append(lib_id)

        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = scanned_books
        new_solution.calculate_fitness_score(data.scores)
        
        return new_solution

    def tweak_solution_shuffle_books(self, solution, data):
        """Deterministic book shuffling - moves books from lower scoring to higher scoring libraries."""
        if len(solution.signed_libraries) < 2:
            return solution

        new_solution = copy.deepcopy(solution)
        
        # Sort libraries by their total score (best first)
        lib_scores = []
        for lib_id in solution.signed_libraries:
            lib = data.libs[lib_id]
            total_score = sum(data.scores[book.id] for book in lib.books)
            lib_scores.append((lib_id, total_score))
        lib_scores.sort(key=lambda x: -x[1])
        
        # Find a book to move from a lower scoring library to higher scoring one
        for i in range(len(lib_scores)-1):
            lower_lib_id = lib_scores[i+1][0]
            lower_lib = data.libs[lower_lib_id]
            
            if lower_lib_id not in new_solution.scanned_books_per_library:
                continue
                
            # Find lowest scoring book in this library
            books = new_solution.scanned_books_per_library[lower_lib_id]
            if not books:
                continue
                
            min_book = min(books, key=lambda b: data.scores[b])
            min_score = data.scores[min_book]
            
            # Try to find a higher scoring library that can take this book
            for j in range(i+1):
                higher_lib_id = lib_scores[j][0]
                higher_lib = data.libs[higher_lib_id]
                
                # Check if book exists in higher library
                if min_book not in {book.id for book in higher_lib.books}:
                    continue
                    
                # Check if higher library has capacity
                curr_books = new_solution.scanned_books_per_library.get(higher_lib_id, [])
                max_books = higher_lib.books_per_day * (data.num_days - higher_lib.signup_days)
                if len(curr_books) >= max_books:
                    continue
                    
                # Perform the move
                new_solution.scanned_books_per_library[lower_lib_id].remove(min_book)
                if higher_lib_id in new_solution.scanned_books_per_library:
                    new_solution.scanned_books_per_library[higher_lib_id].append(min_book)
                else:
                    new_solution.scanned_books_per_library[higher_lib_id] = [min_book]
                
                new_solution.calculate_fitness_score(data.scores)
                return new_solution
        
        return new_solution

    def tweak_solution_swap_signed_guided(self, solution, data):
        """Guided version of swap_signed that considers library scores and signup times."""
        if len(solution.signed_libraries) < 2:
            return solution

        new_solution = copy.deepcopy(solution)
        
        # Calculate library scores and signup times
        lib_info = []
        for lib_id in solution.signed_libraries:
            lib = data.libs[lib_id]
            total_score = sum(data.scores[book.id] for book in lib.books)
            lib_info.append({
                'id': lib_id,
                'score': total_score,
                'signup': lib.signup_days,
                'score_per_day': total_score / lib.signup_days if lib.signup_days > 0 else 0
            })
        
        # Find worst pair to swap based on score per day
        worst_diff = float('inf')
        swap_pair = (0, 1)
        
        for i in range(len(lib_info)):
            for j in range(i+1, len(lib_info)):
                diff = abs(lib_info[i]['score_per_day'] - lib_info[j]['score_per_day'])
                if diff < worst_diff:
                    worst_diff = diff
                    swap_pair = (i, j)
        
        # Perform the swap
        i, j = swap_pair
        new_solution.signed_libraries[i], new_solution.signed_libraries[j] = \
            new_solution.signed_libraries[j], new_solution.signed_libraries[i]
        
        # Rebuild solution from earlier of the two positions
        rebuild_pos = min(i, j)
        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}
        
        # Process libraries before rebuild point
        for k in range(rebuild_pos):
            lib_id = new_solution.signed_libraries[k]
            if lib_id in solution.scanned_books_per_library:
                books = solution.scanned_books_per_library[lib_id]
                new_scanned_books_per_library[lib_id] = books
                scanned_books.update(books)
                curr_time += data.libs[lib_id].signup_days
        
        # Re-process from rebuild point
        new_signed_libraries = new_solution.signed_libraries[:rebuild_pos]
        
        for k in range(rebuild_pos, len(new_solution.signed_libraries)):
            lib_id = new_solution.signed_libraries[k]
            library = data.libs[lib_id]
            
            if curr_time + library.signup_days >= data.num_days:
                new_solution.unsigned_libraries.append(lib_id)
                continue
                
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            
            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]
            
            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                new_solution.unsigned_libraries.append(lib_id)

        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = scanned_books
        new_solution.calculate_fitness_score(data.scores)
        
        return new_solution

        # Additional helper methods would follow the same pattern
        # of replacing random selections with deterministic choices
        # based on scores, iteration counts, or other deterministic criteria