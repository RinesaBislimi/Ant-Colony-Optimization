
import tkinter as tk
from tkinter import messagebox
import os

# from models.parser import Parser
# from models.solver_guided import Solver_Guided
# import os
# import time
# import multiprocessing

# directory = './input'  # Define your input directory here

# def run_guided_solver_methods():
#     print("---------- GUIDED SOLVER EXECUTION ----------")
    
#     for file in os.listdir(directory):
#         if file.endswith('.txt'):
#             print(f'\nProcessing file: ./input/{file}')
#             start_time = time.time()
#             parser = Parser(f'./input/{file}')
#             data = parser.parse()
#             solver = Solver_Guided()

#             # Run and show initial solution
#             print("\n---- Initial Solution ----")
#             initial = solver.generate_initial_solution(data)
#             print(f"Initial Solution Score: {initial.fitness_score:,}")

#             # Run deterministic hill climbing
#             print("\n---- Deterministic Hill Climbing ----")
#             hc_score, hc_solution = solver.deterministic_hill_climbing(data, iterations=1000)
#             print(f"Hill Climbing Score: {hc_score:,}")
#             print(f"Improvement: {hc_score - initial.fitness_score:,}")

#             # Run deterministic simulated annealing
#             print("\n---- Deterministic Simulated Annealing ----")
#             sa_score, sa_solution = solver.deterministic_simulated_annealing(data, iterations=1000)
#             print(f"Simulated Annealing Score: {sa_score:,}")
#             print(f"Improvement: {sa_score - hc_score:,}")

#             # Run deterministic guided local search
#             print("\n---- Deterministic Guided Local Search ----")
#             gls_solution = solver.deterministic_guided_local_search(data, max_time=60)
#             print(f"Guided Local Search Score: {gls_solution.fitness_score:,}")
#             print(f"Improvement: {gls_solution.fitness_score - sa_score:,}")

#             # Run all tweak methods and compare results
#             print("\n---- Testing All Tweak Methods ----")
#             tweak_methods = [
#                 ("Swap Signed Libraries", solver.tweak_solution_swap_signed),
#                 ("Swap Signed with Unsigned", solver.tweak_solution_swap_signed_with_unsigned),
#                 ("Swap Same Books", solver.tweak_solution_swap_same_books),
#                 ("Insert Library", solver.tweak_solution_insert_library),
#                 ("Swap Last Book", solver.tweak_solution_swap_last_book),
#                 ("Swap Neighbor Libraries", solver.tweak_solution_swap_neighbor_libraries),
#                 ("Shuffle Books", solver.tweak_solution_shuffle_books),
#                 ("Guided Swap Signed", solver.tweak_solution_swap_signed_guided)
#             ]
            
#             best_tweak_score = initial.fitness_score
#             best_tweak_name = "Initial"
#             best_tweak_solution = initial
            
#             for name, method in tweak_methods:
#                 initial_clone = solver.generate_initial_solution(data)
#                 tweaked = method(initial_clone, data)
#                 improvement = tweaked.fitness_score - initial.fitness_score
#                 print(f"{name}: {tweaked.fitness_score:,} (Improvement: {improvement:,})")

#                 if tweaked.fitness_score > best_tweak_score:
#                     best_tweak_score = tweaked.fitness_score
#                     best_tweak_name = name
#                     best_tweak_solution = tweaked


#             print(f"\nBest tweak method: {best_tweak_name} with score {best_tweak_score:,}")

#             # Select best solution found from all methods
#             solutions = {
#                 "Initial": initial,
#                 "Hill Climbing": hc_solution,
#                 "Simulated Annealing": sa_solution,
#                 "Guided Local Search": gls_solution,
#                 "Best Tweak": best_tweak_solution
#             }
#             best_name = min(solutions.keys(), key=lambda k: solutions[k].fitness_score)
#             best_solution = solutions[best_name]
            
#             print(f"\nOverall best solution found: {best_name} with score {best_solution.fitness_score:,}")

#             # Export the final solution
#             if not os.path.exists('./output'):
#                 os.makedirs('./output')
#             output_file = f'./output/guided_{file}'
#             best_solution.export(output_file)
#             print(f"Output written to: {output_file}")
            
#             elapsed_time = time.time() - start_time
#             print(f"Total processing time: {elapsed_time:.2f} seconds")


# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     run_guided_solver_methods()from models.parser import Parser
from models.parser import Parser
from models.aco_solver import ACO_Solver
import os
import time
import multiprocessing

directory = './input'  # Input directory path

def run_aco_solver_methods():
    print("---------- ACO SOLVER EXECUTION ----------")

    for file in os.listdir(directory):
        if file.endswith('.txt'):
            print(f'\nProcessing file: {file}')
            start_time = time.time()
            
            # Parse input file
            file_path = os.path.join(directory, file)
            parser = Parser(file_path)
            data = parser.parse()

            # Initialize and run ACO Solver
            solver = ACO_Solver()
            best_solution = solver.run(data)

            if best_solution is None:
                print("ACO failed to find a solution.")
                continue

            # Output only the tweak method used and the score
            print(f"Tweak used: {solver.last_tweak_method_name}")
            print(f"ACO Solution Score: {best_solution.fitness_score:,}")

            elapsed_time = time.time() - start_time
            print(f"Processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_aco_solver_methods()
