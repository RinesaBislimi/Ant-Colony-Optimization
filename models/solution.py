class Solution:
    signed_libraries = []
    unsigned_libraries = []
    scanned_books_per_library = {}
    scanned_books = set()
    fitness_score = -1

    def __init__(self, signed_libs, unsigned_libs, scanned_books_per_library, scanned_books):
        self.signed_libraries = signed_libs
        self.unsigned_libraries = unsigned_libs
        self.scanned_books_per_library = scanned_books_per_library
        self.scanned_books = scanned_books

    def export(self, file_path):
        with open(file_path, "w+") as ofp:
            ofp.write(f"{len(self.signed_libraries)}\n")
            for library in self.signed_libraries:
                books = self.scanned_books_per_library.get(library, [])
                ofp.write(f"{library} {len(books)}\n")
                ofp.write(" ".join(map(str, books)) + "\n")

        print(f"Processing complete! Output written to: {file_path}")

    def describe(self, file_path="./output/output.txt"):
        with open(file_path, "w+") as lofp:
            lofp.write("Signed libraries: " + ", ".join(self.signed_libraries) + "\n")
            lofp.write("Unsigned libraries: " + ", ".join(self.unsigned_libraries) + "\n")
            lofp.write("\nScanned books per library:\n")
            for library_id, books in self.scanned_books_per_library.items():
                lofp.write(f"Library {library_id}: " + ", ".join(map(str, books)) + "\n")
            lofp.write("\nOverall scanned books: " + ", ".join(map(str, sorted(self.scanned_books))) + "\n")

    def calculate_fitness_score(self, scores):
        score = 0
        for book in self.scanned_books:
            score += scores[book]
        self.fitness_score = score

    def calculate_delta_fitness(self, data, new_book_id, removed_book_id=None):
        """
        Updates the fitness score after swapping a book between libraries.

        :param data: The instance data containing book scores.
        :param new_book_id: The ID of the newly scanned book in one library.
        :param removed_book_id: The ID of the book removed from the other library (if any).
        """
        current_fitness = self.fitness_score

        new_book_score = data.scores[new_book_id]

        if removed_book_id is not None:
            removed_book_score = data.scores[removed_book_id]
        else:
            removed_book_score = 0

        delta_fitness = new_book_score - removed_book_score
        updated_fitness = current_fitness + delta_fitness

        self.fitness_score = updated_fitness
    def clean(self, data):
        """Ensure all scanned books are valid and within scan limits."""
        valid_books_per_lib = {lib.id: set(book.id for book in lib.books) for lib in data.libs}
        seen_books = set()
        for lib_id in self.signed_libraries:
            library = data.libs[lib_id]
            books = self.scanned_books_per_library.get(lib_id, [])
            # Only keep books that belong to the library and are not already scanned
            cleaned_books = []
            for book_id in books:
                if book_id in valid_books_per_lib[lib_id] and book_id not in seen_books:
                    cleaned_books.append(book_id)
                    seen_books.add(book_id)
            # Enforce scan limit
            signup_days = library.signup_days
            idx = self.signed_libraries.index(lib_id)
            days_left = data.num_days - sum(data.libs[lid].signup_days for lid in self.signed_libraries[:idx+1])
            max_books = max(0, days_left) * library.books_per_day
            cleaned_books = cleaned_books[:max_books]
            self.scanned_books_per_library[lib_id] = cleaned_books
        # Update the global scanned_books set
        self.scanned_books = set()
        for books in self.scanned_books_per_library.values():
            self.scanned_books.update(books)