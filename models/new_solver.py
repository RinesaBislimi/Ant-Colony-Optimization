import random
from models.solution import Solution
from models.instance_data import InstanceData   
import heapq
import time

class NewSolver:
    def __init__(self):
        pass

    def get_available_books(self, library, scanned_books, scores, max_books):
        available = [(scores[book.id], book.id) for book in library.books if book.id not in scanned_books]
        top_books = heapq.nlargest(max_books, available)  
        return [book_id for _, book_id in top_books]

    def evaluate_library(self, lib, scanned_books, scores, days_left):
        max_books = (days_left - lib.signup_days) * lib.books_per_day
        if max_books <= 0:
            return 0
        available_scores = [scores[book.id] for book in lib.books if book.id not in scanned_books]
        top_scores = heapq.nlargest(max_books, available_scores)
        return sum(top_scores) / lib.signup_days

    def tweak_solution_swap_signed(self, solution, data):
        start = time.time()

        if len(solution.signed_libraries) < 2:
            return solution

        lib_dict = {lib.id: lib for lib in data.libs}
        scanned_books = set(solution.scanned_books)

        # Evaluate potential of signed libraries
        potentials = []
        curr_time = 0
        for lib_id in solution.signed_libraries:
            if lib_id not in lib_dict:
                continue
            lib = lib_dict[lib_id]
            days_left = data.num_days - curr_time - lib.signup_days
            potential = self.evaluate_library(lib, scanned_books, data.scores, days_left)
            potentials.append((lib_id, potential))
            curr_time += lib.signup_days

        if len(potentials) < 2:
            return solution

        # Find lowest and highest potential libraries
        potentials.sort(key=lambda x: x[1])
        low_id = potentials[0][0]
        high_id = potentials[-1][0]

        if low_id == high_id:
            return solution

        # Swap them in signed_libraries
        idx_low = solution.signed_libraries.index(low_id)
        idx_high = solution.signed_libraries.index(high_id)
        new_order = solution.signed_libraries.copy()
        new_order[idx_low], new_order[idx_high] = new_order[idx_high], new_order[idx_low]
        
        print(f"[tweak_solution_swap_signed] Time: {time.time() - start:.4f}s")

        return self.build_solution_from_order(new_order, data)

    def build_solution_from_order(self, signed_order, data, unsigned_libraries=None):
        lib_dict = {lib.id: lib for lib in data.libs}
        scanned_books = set()
        new_scanned_books_per_library = {}
        new_signed = []
        curr_time = 0

        for lib_id in signed_order:
            if lib_id not in lib_dict:
                continue
            lib = lib_dict[lib_id]
            if curr_time + lib.signup_days >= data.num_days:
                continue
            time_left = data.num_days - (curr_time + lib.signup_days)
            max_books = time_left * lib.books_per_day
            books = self.get_available_books(lib, scanned_books, data.scores, max_books)
            if books:
                new_signed.append(lib_id)
                new_scanned_books_per_library[lib_id] = books
                scanned_books.update(books)
                curr_time += lib.signup_days

        if unsigned_libraries is None:
            unsigned_libraries = [lib.id for lib in data.libs if lib.id not in new_signed]
        else:
            unsigned_libraries = [lid for lid in unsigned_libraries if lid not in new_signed]

        
        new_solution = Solution(
            new_signed,
            unsigned_libraries,
            new_scanned_books_per_library,
            scanned_books
        )
        new_solution.calculate_fitness_score(data.scores)
        
        return new_solution

    def tweak_solution_swap_signed_with_unsigned(self, solution, data):
        start = time.time()
        if not solution.signed_libraries or not solution.unsigned_libraries:
            return solution

        lib_dict = {lib.id: lib for lib in data.libs}
        scanned_books = set(solution.scanned_books)

        # Evaluate signed libraries
        curr_time = 0
        signed_scores = []
        for lib_id in solution.signed_libraries:
            if lib_id not in lib_dict:
                continue
            lib = lib_dict[lib_id]
            days_left = data.num_days - curr_time - lib.signup_days
            score = self.evaluate_library(lib, scanned_books, data.scores, days_left)
            signed_scores.append((lib_id, score))
            curr_time += lib.signup_days

        if not signed_scores:
            return solution

        # Worst signed library
        signed_scores.sort(key=lambda x: x[1])
        worst_signed_id = signed_scores[0][0]

        # Evaluate unsigned libraries
        unsigned_scores = []
        for lib_id in solution.unsigned_libraries:
            if lib_id not in lib_dict:
                continue
            lib = lib_dict[lib_id]
            days_left = data.num_days  # Estimate as if inserted early
            score = self.evaluate_library(lib, scanned_books, data.scores, days_left)
            unsigned_scores.append((lib_id, score))

        if not unsigned_scores:
            return solution

        # Best unsigned library
        unsigned_scores.sort(key=lambda x: -x[1])
        best_unsigned_id = unsigned_scores[0][0]

        # Swap
        new_signed = [best_unsigned_id if lid == worst_signed_id else lid for lid in solution.signed_libraries]
        new_unsigned = [lid for lid in solution.unsigned_libraries if lid != best_unsigned_id] + [worst_signed_id]

        print(f"[tweak_solution_swap_signed_with_unsigned] Time: {time.time() - start:.4f}s")
        return self.build_solution_from_order(new_signed, data, new_unsigned)


    def tweak_solution_swap_same_books(self, solution, data):
        start = time.time()
        if not solution.signed_libraries or not solution.unsigned_libraries:
            return solution

        lib_dict = {lib.id: lib for lib in data.libs}
        if len(solution.signed_libraries) < 2:
            return solution

        # Use faster random sampling
        idx1, idx2 = random.sample(range(len(solution.signed_libraries)), 2)
        new_order = solution.signed_libraries.copy()
        new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]

        curr_time = 0
        scanned_books = set()
        scanned_books_per_library = {}
        signed_libraries = []
        unsigned_libraries = []

        for lib_id in new_order:
            if lib_id not in lib_dict:
                unsigned_libraries.append(lib_id)
                continue
            library = lib_dict[lib_id]
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(lib_id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            # Pre-allocate list for better performance
            available_books = []
            for book in library.books:
                if book.id not in scanned_books:
                    available_books.append(book.id)
                    if len(available_books) == max_books_scanned:
                        break

            if available_books:
                signed_libraries.append(lib_id)
                scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                unsigned_libraries.append(lib_id)

        # Only include libraries not in signed_libraries
        unsigned_libraries.extend(lib.id for lib in data.libs if lib.id not in signed_libraries)

        new_solution = Solution(
            signed_libraries,
            unsigned_libraries,
            scanned_books_per_library,
            scanned_books
        )
        print(f"[tweak_solution_swap_same_books] Time: {time.time() - start:.4f}s")
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def tweak_solution_swap_last_book(self, solution, data):
        start = time.time()
        if not solution.scanned_books_per_library or not solution.unsigned_libraries:
            return solution

        lib_dict = {lib.id: lib for lib in data.libs}  # Safe lookup

        # Choose a random library with books
        valid_libs = [lib_id for lib_id, books in solution.scanned_books_per_library.items() if books]
        if not valid_libs:
            return solution
            
        chosen_lib_id = random.choice(valid_libs)
        scanned_books = solution.scanned_books_per_library[chosen_lib_id]
        last_scanned_book = scanned_books[-1]

        # Find the best available book in one pass
        best_book = None
        best_score = -1
        scanned_set = solution.scanned_books
        max_score = max(data.scores) if data.scores else -1

        for lib_id in solution.unsigned_libraries:
            if lib_id not in lib_dict:
                continue
            library = lib_dict[lib_id]
            for book in library.books:
                book_id = book.id
                if book_id not in scanned_set and data.scores[book_id] > best_score:
                    best_book = book_id
                    best_score = data.scores[book_id]
                    if best_score == max_score:  # Early exit if max possible score found
                        break
            if best_score == max_score:
                break

        if best_book is None:
            return solution

        # Update the solution
        new_scanned_books_per_library = solution.scanned_books_per_library.copy()
        new_scanned_books_per_library[chosen_lib_id] = scanned_books[:-1] + [best_book]
        
        new_scanned_books = set(solution.scanned_books)
        new_scanned_books.remove(last_scanned_book)
        new_scanned_books.add(best_book)

        new_solution = Solution(
            solution.signed_libraries,
            solution.unsigned_libraries,
            new_scanned_books_per_library,
            new_scanned_books
        )
        
        print(f"[tweak_solution_swap_last_book] Time: {time.time() - start:.4f}s")
        new_solution.calculate_fitness_score(data.scores)
        return new_solution
    
    def crossover(self, solution, data):
        # Create a new solution with shuffled libraries
        lib_dict = {lib.id: lib for lib in data.libs}
        library_ids = list(lib_dict.keys())
        random.shuffle(library_ids)

        new_scanned_books_per_library = {}
        scanned_books = set()
        signed_libraries = []
        curr_time = 0

        for lib_id in library_ids:
            if lib_id not in lib_dict:
                continue
            if curr_time >= data.num_days:
                break

            lib = lib_dict[lib_id]
            if curr_time + lib.signup_days >= data.num_days:
                continue

            time_left = data.num_days - (curr_time + lib.signup_days)
            max_books_scanned = time_left * lib.books_per_day

            # Pre-allocate list for better performance
            available_books = []
            for book in lib.books:
                if book.id not in scanned_books:
                    available_books.append(book.id)
                    if len(available_books) == max_books_scanned:
                        break

            if available_books:
                signed_libraries.append(lib.id)
                new_scanned_books_per_library[lib.id] = available_books
                scanned_books.update(available_books)
                curr_time += lib.signup_days

        unsigned_libraries = [lib_id for lib_id in lib_dict if lib_id not in signed_libraries]

        new_solution = Solution(
            signed_libraries,
            unsigned_libraries,
            new_scanned_books_per_library,
            scanned_books
        )
        new_solution.calculate_fitness_score(data.scores)
        return new_solution
    
    def optimize_remaining_time(self, solution, data):
        # After initial solution, fill remaining time with small libraries
        used_time = sum(data.libs[lib_id].signup_days for lib_id in solution.signed_libraries)
        remaining_time = data.num_days - used_time
        
        # Try to find libraries that can fit in remaining time
        for lib in data.libs:
            if lib.id not in solution.signed_libraries and lib.signup_days <= remaining_time:
                # Evaluate adding this library at the end
                new_order = solution.signed_libraries + [lib.id]
                new_sol = self.build_solution_from_order(new_order, data)
                if new_sol.fitness > solution.fitness:
                    return new_sol
        return solution
        
    def tweak_solution_swap_neighbor_libraries(self, solution: Solution, data: InstanceData) -> Solution:
        start = time.time()
        if len(solution.signed_libraries) < 2:
            return solution

        lib_dict = {lib.id: lib for lib in data.libs}  # Safe lookup

        swap_pos = random.randint(0, len(solution.signed_libraries) - 2)
        new_signed = solution.signed_libraries[:]
        new_signed[swap_pos], new_signed[swap_pos + 1] = new_signed[swap_pos + 1], new_signed[swap_pos]

        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}
        signed_libraries = []
        # Use a set for faster membership checks and to avoid duplicates
        unsigned_libraries_set = set(solution.unsigned_libraries)

        for lib_id in new_signed:
            if lib_id not in lib_dict:
                unsigned_libraries_set.add(lib_id)
                continue

            library = lib_dict[lib_id]
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries_set.add(lib_id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            # Use list comprehension for available books, limited by max_books_scanned
            available_books = [book.id for book in library.books if book.id not in scanned_books]
            if max_books_scanned > 0:
                available_books = available_books[:max_books_scanned]
            else:
                available_books = []

            if available_books:
                signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                unsigned_libraries_set.add(lib_id)

        # Ensure unsigned_libraries contains only libraries not in signed_libraries
        unsigned_libraries = [lib_id for lib_id in unsigned_libraries_set if lib_id not in signed_libraries]

        new_solution = Solution(
            signed_libraries,
            unsigned_libraries,
            new_scanned_books_per_library,
            scanned_books
        )
        
        print(f"[tweak_solution_swap_neighbor_libraries] Time: {time.time() - start:.4f}s")
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def tweak_solution_insert_library(self, solution: Solution, data: InstanceData) -> Solution:
        start = time.time()
        if not solution.unsigned_libraries:
            return solution

        lib_dict = {lib.id: lib for lib in data.libs}  # Safe lookup

        # Filter valid unsigned libraries
        valid_unsigned = [lib_id for lib_id in solution.unsigned_libraries if lib_id in lib_dict]
        if not valid_unsigned:
            return solution

        curr_total_time = sum(lib_dict[lib_id].signup_days for lib_id in solution.signed_libraries if lib_id in lib_dict)
        num_days_left = data.num_days - curr_total_time

        # Precompute candidate libraries and their potential scores
        candidate_libs = []
        scanned_books_set = solution.scanned_books
        for lib_id in valid_unsigned:
            lib = lib_dict[lib_id]
            signup_left = num_days_left - lib.signup_days
            if signup_left <= 0:
                continue
            max_books = signup_left * lib.books_per_day
            if max_books <= 0:
                continue
            # Use list comprehension for available books and sum scores directly
            available_books = [book.id for book in lib.books if book.id not in scanned_books_set]
            potential_books = available_books[:max_books]
            potential_score = sum(data.scores[bid] for bid in potential_books)
            if potential_score > 0:
                candidate_libs.append((lib_id, potential_score, potential_books, lib.signup_days))

        if not candidate_libs:
            return solution

        # Select top candidates (limit to 3 for performance)
        candidate_libs.sort(key=lambda x: -x[1])
        selected = candidate_libs[:3]
        lib_to_insert, _, available_books, insert_signup_days = random.choice(selected)
        insert_lib = lib_dict[lib_to_insert]

        # Test insertion positions (limit to 3 for performance)
        test_positions = sorted({
            0,
            len(solution.signed_libraries),
            random.randint(0, len(solution.signed_libraries))
        })

        best_pos = 0
        best_score = -1
        best_final_books = []

        for pos in test_positions:
            # Calculate time used up to insertion point
            time_before = sum(lib_dict[lib_id].signup_days for lib_id in solution.signed_libraries[:pos] if lib_id in lib_dict)
            total_time = time_before + insert_signup_days
            if total_time >= data.num_days:
                continue
            time_left = data.num_days - total_time
            max_books = time_left * insert_lib.books_per_day
            final_books = available_books[:max_books]
            added_score = sum(data.scores[b] for b in final_books)
            if added_score > best_score:
                best_score = added_score
                best_pos = pos
                best_final_books = final_books

        if best_score <= 0 or not best_final_books:
            return solution

        # Apply the insertion
        new_signed = solution.signed_libraries[:best_pos] + [lib_to_insert] + solution.signed_libraries[best_pos:]
        new_unsigned = [lib for lib in solution.unsigned_libraries if lib != lib_to_insert]
        new_scanned = solution.scanned_books_per_library.copy()
        new_scanned[lib_to_insert] = best_final_books
        new_books = set(solution.scanned_books)
        new_books.update(best_final_books)

        new_solution = Solution(
            new_signed,
            new_unsigned,
            new_scanned,
            new_books
        )
        
        print(f"[tweak_solution_insert_library] Time: {time.time() - start:.4f}s")
        new_solution.calculate_fitness_score(data.scores)
        return new_solution