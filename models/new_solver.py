import random
import copy
from models.solution import Solution
from models.book import Book
from models.library import Library
from models.instance_data import InstanceData   

class NewSolver:
    def __init__(self):
        pass

    def tweak_solution_swap_signed(self, solution, data):
        if len(solution.signed_libraries) < 2:
            return solution

        idx1, idx2 = random.sample(range(len(solution.signed_libraries)), 2)
        new_signed_libraries = solution.signed_libraries.copy()
        new_signed_libraries[idx1], new_signed_libraries[idx2] = new_signed_libraries[idx2], new_signed_libraries[idx1]

        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}
        for lib_id in new_signed_libraries:
            library = data.libs[lib_id]
            if curr_time + library.signup_days >= data.num_days:
                break
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            # Only allow books that belong to this library and are not already scanned
            available_books = []
            for book in library.books:
                if book.id not in scanned_books:
                    available_books.append(book.id)
                    if len(available_books) == max_books_scanned:
                        break
            if available_books:
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                break

        # All remaining libraries are unsigned
        new_unsigned_libraries = [lib.id for lib in data.libs if lib.id not in new_scanned_books_per_library]

        new_solution = Solution(
            list(new_scanned_books_per_library.keys()),
            new_unsigned_libraries,
            new_scanned_books_per_library,
            scanned_books
        )
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def tweak_solution_swap_signed_with_unsigned(self, solution, data, bias_type=None, bias_ratio=2/3):
        if not solution.signed_libraries or not solution.unsigned_libraries:
            return solution

        local_signed_libs = solution.signed_libraries.copy()
        local_unsigned_libs = solution.unsigned_libraries.copy()
        total_signed = len(local_signed_libs)

        # Bias
        if bias_type == "favor_first_half" and total_signed > 1:
            if random.random() < bias_ratio:
                signed_idx = random.randint(0, total_signed // 2 - 1)
            else:
                signed_idx = random.randint(0, total_signed - 1)
        elif bias_type == "favor_second_half" and total_signed > 1:
            if random.random() < bias_ratio:
                signed_idx = random.randint(total_signed // 2, total_signed - 1)
            else:
                signed_idx = random.randint(0, total_signed - 1)
        else:
            signed_idx = random.randint(0, total_signed - 1)

        unsigned_idx = random.randint(0, len(local_unsigned_libs) - 1)
        signed_lib_id = local_signed_libs[signed_idx]
        unsigned_lib_id = local_unsigned_libs[unsigned_idx]

        # Swap
        local_signed_libs[signed_idx] = unsigned_lib_id
        local_unsigned_libs[unsigned_idx] = signed_lib_id

        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}
        new_signed_libraries = []
        new_unsigned_libraries = []

        for lib_id in local_signed_libs:
            library = data.libs[lib_id]
            if curr_time + library.signup_days >= data.num_days:
                new_unsigned_libraries.append(lib_id)
                continue
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            available_books = []
            for book in library.books:
                if book.id not in scanned_books:
                    available_books.append(book.id)
                    if len(available_books) == max_books_scanned:
                        break
            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                new_unsigned_libraries.append(lib_id)

        # Add any unsigned not already present
        for lib_id in local_unsigned_libs:
            if lib_id not in new_signed_libraries and lib_id not in new_unsigned_libraries:
                new_unsigned_libraries.append(lib_id)

        new_solution = Solution(
            new_signed_libraries,
            new_unsigned_libraries,
            new_scanned_books_per_library,
            scanned_books
        )
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def tweak_solution_swap_same_books(self, solution, data):
            
        library_ids = [lib for lib in solution.signed_libraries if lib < len(data.libs)]
        if len(library_ids) < 2:
            return solution

        idx1, idx2 = random.sample(range(len(library_ids)), 2)
        library_ids[idx1], library_ids[idx2] = library_ids[idx2], library_ids[idx1]

        curr_time = 0
        scanned_books = set()
        scanned_books_per_library = {}
        signed_libraries = []
        unsigned_libraries = []

        for lib_id in library_ids:
            library = data.libs[lib_id]
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(lib_id)
                continue
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
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

        # Add any remaining unsigned libraries
        all_lib_ids = set(range(len(data.libs)))
        for lib_id in all_lib_ids - set(library_ids):
            unsigned_libraries.append(lib_id)

        new_solution = Solution(
            signed_libraries,
            unsigned_libraries,
            scanned_books_per_library,
            scanned_books
        )
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def tweak_solution_swap_last_book(self, solution, data):
        if not solution.scanned_books_per_library or not solution.unsigned_libraries:
            return solution

        chosen_lib_id = random.choice(list(solution.scanned_books_per_library.keys()))
        scanned_books = solution.scanned_books_per_library[chosen_lib_id]
        if not scanned_books:
            return solution

        last_scanned_book = scanned_books[-1]
        new_scanned_books_per_library = solution.scanned_books_per_library.copy()
        new_scanned_books = solution.scanned_books.copy()

        # Find a replacement book that is valid and not already scanned
        best_book = None
        best_score = -1
        for unsigned_lib in solution.unsigned_libraries:
            library = data.libs[unsigned_lib]
            for book in library.books:
                if book.id not in new_scanned_books and data.scores[book.id] > best_score:
                    best_book = book.id
                    best_score = data.scores[book.id]

        if best_book is None:
            return solution

        # Update only the affected library and scanned_books set
        new_scanned_books_per_library[chosen_lib_id] = scanned_books[:-1] + [best_book]
        new_scanned_books.remove(last_scanned_book)
        new_scanned_books.add(best_book)

        new_solution = Solution(
            solution.signed_libraries,
            solution.unsigned_libraries,
            new_scanned_books_per_library,
            new_scanned_books
        )
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def crossover(self, solution, data):
        new_solution = copy.copy(solution)
        old_order = new_solution.signed_libraries[:]
        library_indices = list(range(len(data.libs)))
        random.shuffle(library_indices)

        new_scanned_books_per_library = {}
        scanned_books = set()
        signed_libraries = []
        curr_time = 0

        for new_idx, new_lib_idx in enumerate(library_indices):
            if new_idx >= len(old_order):
                break
            lib = data.libs[new_lib_idx]
            if curr_time + lib.signup_days >= data.num_days:
                continue
            time_left = data.num_days - (curr_time + lib.signup_days)
            max_books_scanned = time_left * lib.books_per_day
            available_books = [book.id for book in lib.books if book.id not in scanned_books][:max_books_scanned]
            if available_books:
                signed_libraries.append(lib.id)
                new_scanned_books_per_library[lib.id] = available_books
                scanned_books.update(available_books)
                curr_time += lib.signup_days

        unsigned_libraries = [lib.id for lib in data.libs if lib.id not in signed_libraries]
        new_solution = Solution(
            signed_libraries,
            unsigned_libraries,
            new_scanned_books_per_library,
            scanned_books
        )
        new_solution.calculate_fitness_score(data.scores)
        return new_solution