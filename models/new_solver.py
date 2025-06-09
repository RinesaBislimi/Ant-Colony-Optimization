import random
from models.solution import Solution
from models.instance_data import InstanceData   

class NewSolver:
    def __init__(self):
        pass

    def tweak_solution_swap_signed(self, solution, data):
        if len(solution.signed_libraries) < 2:
            return solution

        # Use faster random sampling
        idx1, idx2 = random.sample(range(len(solution.signed_libraries)), 2)
        new_signed_libraries = solution.signed_libraries.copy()
        new_signed_libraries[idx1], new_signed_libraries[idx2] = new_signed_libraries[idx2], new_signed_libraries[idx1]

        curr_time = 0
        scanned_books = set(solution.scanned_books)  # Start with existing books
        new_scanned_books_per_library = {}
        signed_libraries = []

        for lib_id in new_signed_libraries:
            library = data.libs[lib_id]
            if curr_time + library.signup_days >= data.num_days:
                break
                
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            
            # Pre-filter books that aren't already scanned
            available_books = []
            for book in library.books:
                if book.id not in scanned_books:
                    available_books.append(book.id)
                    if len(available_books) == max_books_scanned:
                        break
            
            if available_books:
                signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                break

        # Only include libraries we actually processed
        new_unsigned_libraries = [lib.id for lib in data.libs if lib.id not in signed_libraries]

        new_solution = Solution(
            signed_libraries,
            new_unsigned_libraries,
            new_scanned_books_per_library,
            scanned_books
        )
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def tweak_solution_swap_signed_with_unsigned(self, solution, data, bias_type=None, bias_ratio=2/3):
        if not solution.signed_libraries or not solution.unsigned_libraries:
            return solution

        total_signed = len(solution.signed_libraries)
        
        # Optimized bias selection
        if bias_type == "favor_first_half" and total_signed > 1:
            signed_idx = random.randint(0, total_signed // 2 - 1) if random.random() < bias_ratio else random.randint(0, total_signed - 1)
        elif bias_type == "favor_second_half" and total_signed > 1:
            signed_idx = random.randint(total_signed // 2, total_signed - 1) if random.random() < bias_ratio else random.randint(0, total_signed - 1)
        else:
            signed_idx = random.randint(0, total_signed - 1)

        unsigned_idx = random.randint(0, len(solution.unsigned_libraries) - 1)
        signed_lib_id = solution.signed_libraries[signed_idx]
        unsigned_lib_id = solution.unsigned_libraries[unsigned_idx]

        # Create new lists with the swap
        new_signed = solution.signed_libraries[:signed_idx] + [unsigned_lib_id] + solution.signed_libraries[signed_idx+1:]
        new_unsigned = solution.unsigned_libraries[:unsigned_idx] + [signed_lib_id] + solution.unsigned_libraries[unsigned_idx+1:]

        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}
        final_signed = []
        final_unsigned = new_unsigned.copy()

        for lib_id in new_signed:
            library = data.libs[lib_id]
            if curr_time + library.signup_days >= data.num_days:
                final_unsigned.append(lib_id)
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
                final_signed.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                final_unsigned.append(lib_id)

        new_solution = Solution(
            final_signed,
            final_unsigned,
            new_scanned_books_per_library,
            scanned_books
        )
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def tweak_solution_swap_same_books(self, solution, data):
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
            library = data.libs[lib_id]
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
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def tweak_solution_swap_last_book(self, solution, data):
        if not solution.scanned_books_per_library or not solution.unsigned_libraries:
            return solution

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
        
        for lib_id in solution.unsigned_libraries:
            library = data.libs[lib_id]
            for book in library.books:
                book_id = book.id
                if book_id not in scanned_set and data.scores[book_id] > best_score:
                    best_book = book_id
                    best_score = data.scores[book_id]
                    if best_score == max(data.scores):  # Early exit if max possible score found
                        break
            if best_score == max(data.scores):
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
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def crossover(self, solution, data):
        # Create a new solution with shuffled libraries
        library_indices = list(range(len(data.libs)))
        random.shuffle(library_indices)

        new_scanned_books_per_library = {}
        scanned_books = set()
        signed_libraries = []
        curr_time = 0

        for lib_idx in library_indices:
            if curr_time >= data.num_days:
                break
                
            lib = data.libs[lib_idx]
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

        unsigned_libraries = [lib.id for lib in data.libs if lib.id not in signed_libraries]
        
        new_solution = Solution(
            signed_libraries,
            unsigned_libraries,
            new_scanned_books_per_library,
            scanned_books
        )
        new_solution.calculate_fitness_score(data.scores)
        return new_solution
    
    def tweak_solution_swap_neighbor_libraries(self, solution: Solution, data: InstanceData) -> Solution:
        if len(solution.signed_libraries) < 2:
            return solution

        swap_pos = random.randint(0, len(solution.signed_libraries) - 2)
        new_signed = solution.signed_libraries.copy()
        new_signed[swap_pos], new_signed[swap_pos + 1] = new_signed[swap_pos + 1], new_signed[swap_pos]
        
        curr_time = 0
        scanned_books = set()
        new_scanned_books_per_library = {}
        signed_libraries = []
        unsigned_libraries = solution.unsigned_libraries.copy()
        
        for lib_id in new_signed:
            if lib_id >= len(data.libs):
                unsigned_libraries.append(lib_id)
                continue
                
            library = data.libs[lib_id]
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(lib_id)
                continue
                
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            
            # Optimized book selection with pre-allocation
            available_books = []
            for book in library.books:
                if book.id not in scanned_books:
                    available_books.append(book.id)
                    if len(available_books) == max_books_scanned:
                        break
            
            if available_books:
                signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                unsigned_libraries.append(lib_id)
        
        new_solution = Solution(
            signed_libraries,
            unsigned_libraries,
            new_scanned_books_per_library,
            scanned_books
        )
        new_solution.calculate_fitness_score(data.scores)
        return new_solution

    def tweak_solution_insert_library(self, solution: Solution, data: InstanceData) -> Solution:
        if not solution.unsigned_libraries:
            return solution
            
        # First filter out any invalid library IDs from unsigned libraries
        valid_unsigned = [lib_id for lib_id in solution.unsigned_libraries 
                        if lib_id < len(data.libs)]
        
        if not valid_unsigned:
            return solution
            
        # Calculate current time used
        curr_total_time = sum(data.libs[lib_id].signup_days 
                            for lib_id in solution.signed_libraries 
                            if lib_id < len(data.libs))
        
        # Pre-compute potential libraries and scores
        candidate_libs = []
        for lib_id in valid_unsigned:
            lib = data.libs[lib_id]
            if curr_total_time + lib.signup_days >= data.num_days:
                continue
                
            # Calculate potential score
            potential_score = 0
            count = 0
            max_possible = (data.num_days - curr_total_time - lib.signup_days) * lib.books_per_day
            for book in lib.books:
                if book.id not in solution.scanned_books:
                    potential_score += data.scores[book.id]
                    count += 1
                    if count >= max_possible:
                        break
                        
            if potential_score > 0:
                candidate_libs.append((lib_id, potential_score))
        
        if not candidate_libs:
            return solution
            
        # Select top candidates (limited to 3 for performance)
        candidate_libs.sort(key=lambda x: -x[1])
        selected = candidate_libs[:min(3, len(candidate_libs))]
        lib_to_insert = random.choice(selected)[0]
        insert_lib = data.libs[lib_to_insert]
        
        # Find available books
        available_books = []
        max_possible = (data.num_days - curr_total_time - insert_lib.signup_days) * insert_lib.books_per_day
        for book in insert_lib.books:
            if book.id not in solution.scanned_books:
                available_books.append(book.id)
                if len(available_books) >= max_possible:
                    break
        
        if not available_books:
            return solution
            
        # Test insertion positions (limited to 3 for performance)
        test_positions = sorted({
            0,
            len(solution.signed_libraries),
            random.randint(0, len(solution.signed_libraries))
        })
        
        best_pos = 0
        best_score = -1
        
        for pos in test_positions:
            # Calculate approximate time at insertion point
            time_before = sum(data.libs[lib_id].signup_days 
                            for lib_id in solution.signed_libraries[:pos] 
                            if lib_id < len(data.libs))
            total_time = time_before + insert_lib.signup_days
            if total_time >= data.num_days:
                continue
                
            # Calculate approximate score
            time_left = data.num_days - total_time
            max_books = time_left * insert_lib.books_per_day
            added_score = sum(data.scores[b] for b in available_books[:max_books])
            
            if added_score > best_score:
                best_score = added_score
                best_pos = pos
        
        if best_score <= 0:
            return solution
            
        # Apply the insertion
        new_signed = solution.signed_libraries[:best_pos] + [lib_to_insert] + solution.signed_libraries[best_pos:]
        new_unsigned = [lib for lib in solution.unsigned_libraries if lib != lib_to_insert]
        
        # Get the actual books that can be scanned
        time_used = sum(data.libs[lib_id].signup_days 
                    for lib_id in new_signed[:best_pos+1] 
                    if lib_id < len(data.libs))
        max_books = (data.num_days - time_used) * insert_lib.books_per_day
        final_books = available_books[:max_books]
        
        new_scanned = solution.scanned_books_per_library.copy()
        new_scanned[lib_to_insert] = final_books
        new_books = set(solution.scanned_books)
        new_books.update(final_books)
        
        new_solution = Solution(
            new_signed,
            new_unsigned,
            new_scanned,
            new_books
        )
        new_solution.calculate_fitness_score(data.scores)
        return new_solution