# University of Prishtina “Hasan Prishtina” <img src="https://upload.wikimedia.org/wikipedia/commons/e/e1/University_of_Prishtina_logo.svg" width="100" align="right">

*Faculty of Electrical and Computer Engineering*  
**Level:** Master  
**Course:** Nature-Inspired Algorithms  
**Project Title:** Ant Colony Optimization Solver  
**Professor:** Dr. Techn. Kadri Sylejmani, prof. ass.

---

## Ant Colony Optimization Solver

This project implements a guided **Ant Colony Optimization (ACO)** algorithm to solve library scanning problems, such as those in book digitization or resource scheduling tasks. It constructs candidate solutions using probabilistic selection and then applies guided local search (tweaking) to refine results over multiple iterations.

---

##  How It Works

The algorithm simulates a colony of virtual ants building candidate solutions by selecting libraries to sign up, based on a combination of:

- **Pheromone Trails** — storing historical success of certain decisions.
- **Heuristic Information** — prioritizing libraries with high book scores per signup time.

After construction, each solution is locally optimized using **tweak methods**. The best solutions in each iteration reinforce the pheromone trails to guide future decision-making, leading to convergence toward optimal or near-optimal solutions.

---

## ACO Process Overview

-  **Pheromone trails** influence which libraries are selected.
- **Heuristic values** favor high-efficiency libraries.
-  **Tweak methods** locally optimize candidate solutions.
- **Best solutions** reinforce the trails for future iterations.

---

##  Tweak Methods Used

- Swap signed with unsigned libraries
- Swap signed libraries only
- Swap books between libraries
- Insert new libraries into the schedule
- Swap neighbor libraries
- Swap last-scanned books

These methods are modular and tested individually to ensure improvement or stability in fitness scores.

---

##  Running the Solver

```bash
python main.py
```
The program will:

- Read all .txt files from the input/ folder.

- Parse input files using the Parser class.

- Construct and refine solutions via the ACO_Solver.

- Save the best solutions to the output/guided/ directory.

 ## Output Format
Each input file produces a corresponding output file in:
```bash
output/guided/
```
Each solution is stored in the same format as the input problem, ready for further evaluation or scoring.

