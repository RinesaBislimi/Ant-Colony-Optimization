# University of Prishtina "Hasan Prishtina" <img src="https://upload.wikimedia.org/wikipedia/commons/e/e1/University_of_Prishtina_logo.svg" width="100" align="right">

*Faculty of Electrical and Computer Engineering*  
**Level:** Master  
**Course:** Nature-Inspired Algorithms  
**Project Title:** Ant Colony Optimization Solver  
**Professor:** Dr. Techn. Kadri Sylejmani, prof. ass.

---

## Ant Colony Optimization Solver

This project implements a guided **Ant Colony Optimization (ACO)** algorithm to solve library scanning problems, such as those in book digitization or resource scheduling tasks. It constructs candidate solutions using probabilistic selection and then applies guided local search (tweaking) to refine results over multiple iterations.

---

## Results

| Instance                  | Best Known | Ant Colony Optimization (1k Iterations) | Gap (%)   |
|---------------------------|------------|------------------------------------------|-----------|
| UPIEFK                    | 93         | 93                                       | 0.00%     |
| a_example                 | 21         | 21                                       | 0.00%     |
| b_read_on                | 5,822,900  | 5,549,100                                | -4.70%    |
| c_incunabula             | 5,690,888  | 4,349,874                                | -23.52%   |
| e_so_many_books          | 5,107,113  | 4,333,852                                | -15.15%   |
| d_tough_choices          | 5,237,345  | 4,848,641                                | -7.41%    |
| f_libraries_of_the_world | 5,348,248  | 4,133,033                                | -22.74%   |
| Toy instance              | 18         | 18                                       | 0.00%     |
| B5000_L90_D21             | 3,394      | 3,501                                    | +3.15%    |
| B50000_L400_D28           | 121,715    | 118,392                                  | -2.73%    |
| B100000_L600_D28          | 128,456    | 129,431                                  | +0.76%    |
| B90000_L850_D21           | 828        | 848                                      | +2.41%    |
| B95000_L2000_D28          | 1,366,194  | 1,042,089                                | -23.73%   |
| switch_book_instance      | —          | 20                                       | —         |

**Average Performance:** The ACO algorithm achieves competitive results across most instances, with particularly strong performance on small-scale problems. Large instances show varying gaps, indicating areas for further optimization or more tuning.

---

## How It Works

The algorithm simulates a colony of virtual ants building candidate solutions by selecting libraries to sign up, based on a combination of:

- **Pheromone Trails** — storing historical success of certain decisions.
- **Heuristic Information** — prioritizing libraries with high book scores per signup time.

After construction, each solution is locally optimized using **tweak methods**. The best solutions in each iteration reinforce the pheromone trails to guide future decision-making, leading to convergence toward optimal or near-optimal solutions.

---

## ACO Process Overview

- **Pheromone trails** influence which libraries are selected.
- **Heuristic values** favor high-efficiency libraries.
- **Tweak methods** locally optimize candidate solutions.
- **Best solutions** reinforce the trails for future iterations.

---

## Tweak Methods Used

- Swap signed with unsigned libraries
- Swap signed libraries only
- Swap books between libraries
- Insert new libraries into the schedule
- Swap neighbor libraries
- Swap last-scanned books

These methods are modular and tested individually to ensure improvement or stability in fitness scores.

---

## Running the Solver

```bash
python main.py
