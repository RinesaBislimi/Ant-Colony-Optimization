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
        if not file_name.endswith(".txt"):
            continue

        print(f"\nProcessing: {file_name}")
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        start_time = time.time()

        # Parse input
        parser = Parser(input_path)
        data = parser.parse()

        # Run solver
        solver = ACO_Solver()
        best_solution = solver.run(data)

        if best_solution is None:
            print(" ACO failed to find a solution.")
            continue

        # Export directly to file
        best_solution.export(output_path)
        print(f" Output written to: {output_path}")

        print(f" Tweak used: {solver.last_tweak_method_name}")
        print(f"  ACO Solution Score: {best_solution.fitness_score:,}")

        elapsed_time = time.time() - start_time
        print(f" Processing time: {elapsed_time:.2f} seconds")

        # Validate the solution
        validation_result = validator.validate_solution(input_path, output_path, isConsoleApplication=True)
        print("\n Validation result:\n" + validation_result)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_aco_solver_methods()
