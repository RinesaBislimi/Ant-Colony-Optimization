import os
import time
from models.parser import Parser
from models.aco_solver import ACO_Solver

INPUT_DIR = './input'
OUTPUT_DIR = './output/guided/'

def run_aco_solver_methods(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    print("========== ACO SOLVER EXECUTION ==========")

    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        if not file_name.endswith(".txt") or not os.path.isfile(input_path):
            continue

        print(f"\nProcessing: {file_name}", flush=True)
        start_time = time.time()

        try:
            parser = Parser(input_path)
            data = parser.parse()
        except Exception as e:
            print(f" Error parsing {file_name}: {e}")
            continue

        solver = ACO_Solver()
        best_solution = solver.run(data)

        if best_solution is None:
            print(" ACO failed to find a solution.")
            continue

        try:

            best_solution.clean(data)
            best_solution.export(output_path)
            print(f" Output written to: {output_path}")
        except Exception as e:
            print(f" Error exporting solution for {file_name}: {e}")
            continue

        print(f" Tweak used: {solver.last_tweak_method_name if hasattr(solver, 'last_tweak_method_name') else 'None'}")
        print(f"  ACO Solution Score: {best_solution.fitness_score:,}")
        elapsed_time = time.time() - start_time
        print(f" Processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    run_aco_solver_methods()