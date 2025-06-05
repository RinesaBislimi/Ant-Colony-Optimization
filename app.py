import os
import time
import multiprocessing
from models.parser import Parser
from models.aco_solver import ACO_Solver
from validator import validator

INPUT_DIR = './input'
OUTPUT_DIR = './output'

def run_aco_solver_methods(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    print("========== ACO SOLVER EXECUTION ==========")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        # Only process files that are .txt and not directories
        if not file_name.endswith(".txt") or not os.path.isfile(input_path):
            continue

        print(f"\nProcessing: {file_name}", flush=True)
        start_time = time.time()

        try:
            # Parse input
            parser = Parser(input_path)
            data = parser.parse()
        except Exception as e:
            print(f" Error parsing {file_name}: {e}")
            continue

        try:
            # Run solver
           if file_name == "c_incunabula.txt":
            solver = ACO_Solver(num_ants=1, max_iterations=2)
           else:
            solver = ACO_Solver(num_ants=3, max_iterations=10)
            
            best_solution = solver.run(data)
        except Exception as e:
            print(f" Error running ACO solver on {file_name}: {e}")
            continue

        if best_solution is None:
            print(" ACO failed to find a solution.")
            continue

        try:
            # Export directly to file
            best_solution.export(output_path)
            print(f" Output written to: {output_path}")
        except Exception as e:
            print(f" Error exporting solution for {file_name}: {e}")
            continue

        print(f" Tweak used: {getattr(solver, 'last_tweak_method_name', 'N/A')}")
        print(f"  ACO Solution Score: {best_solution.fitness_score:,}")

        elapsed_time = time.time() - start_time
        print(f" Processing time: {elapsed_time:.2f} seconds")

        # Validate the solution
        try:
            # Clean the solution before exporting
            best_solution.clean(data)
            # Export directly to file
            best_solution.export(output_path)
            print(f" Output written to: {output_path}")
        except Exception as e:
            print(f" Error exporting solution for {file_name}: {e}")
            continue

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_aco_solver_methods()