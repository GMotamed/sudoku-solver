import numpy as np
import random
from copy import deepcopy
import time

# کلاس الگوریتم ژنتیک
class GeneticAlgorithmSolver:
    def __init__(self, initial_board, population_size=100, generations=1000, mutation_rate=0.1, patience=50):
        self.initial_board = initial_board
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.patience = patience

    def generate_random_sudoku(self):
        board = deepcopy(self.initial_board)
        for i in range(9):
            missing_indices = [idx for idx, val in enumerate(board[i]) if val == 0]
            missing_values = [val for val in range(1, 10) if val not in board[i]]
            random.shuffle(missing_values)
            for idx, val in zip(missing_indices, missing_values):
                board[i][idx] = val
        return np.array(board)

    def generate_initial_population(self):
        return [self.generate_random_sudoku() for _ in range(self.population_size)]

    def fitness(self, board):
        score = 0
        for i in range(9):
            score += len(np.unique(board[i, :]))  # امتیاز برای سطرها
            score += len(np.unique(board[:, i]))  # امتیاز برای ستون‌ها
        
        for i in range(3):
            for j in range(3):
                sub_grid = board[i*3:(i+1)*3, j*3:(j+1)*3]
                score += len(np.unique(sub_grid))  # امتیاز برای زیرماتریس‌ها

        return score

    def crossover(self, parent1, parent2):
        point = np.random.randint(1, 8)
        child1 = np.vstack((parent1[:point, :], parent2[point:, :]))
        child2 = np.vstack((parent2[:point, :], parent1[point:, :]))
        return child1, child2

    def mutate(self, board):
        row = np.random.randint(0, 9)
        col1, col2 = np.random.choice([c for c in range(9) if self.initial_board[row][c] == 0], 2, replace=False)
        board[row, col1], board[row, col2] = board[row, col2], board[row, col1]
        return board

    def print_sudoku(self, board):
        for row in board:
            print(" ".join(str(num) for num in row))
        print("\n")

    def solve(self):
        population = self.generate_initial_population()
        best_fitness = 0
        no_improvement_generations = 0
        best_solution = None
        
        start_time = time.time()
        
        for gen in range(self.generations):
            population = sorted(population, key=lambda x: -self.fitness(x))
            new_population = population[:2]
            
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(population[:10], 2)
                child1, child2 = self.crossover(parent1, parent2)
                
                if np.random.rand() < self.mutation_rate:
                    child1 = self.mutate(child1)
                if np.random.rand() < self.mutation_rate:
                    child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population
            current_best_fitness = self.fitness(population[0])
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = deepcopy(population[0])
                no_improvement_generations = 0
            else:
                no_improvement_generations += 1
            
            if best_fitness == 243 or no_improvement_generations >= self.patience:
                break
        
        end_time = time.time()
        elapsed_time = end_time - start_time
                
        return best_solution, best_fitness, gen + 1, elapsed_time


# کلاس الگوریتم CSP
class CSPSolver:
    def __init__(self, board):
        self.board = board
        self.assignment_count = 0

    def print_sudoku(self, board):
        for row in board:
            print(" ".join(str(num) if num != 0 else '.' for num in row))
        print("\n")

    def find_empty_location(self):
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    return i, j
        return None, None

    def is_safe(self, row, col, num):
        if num in self.board[row]:
            return False

        if num in self.board[:, col]:
            return False

        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if self.board[start_row + i][start_col + j] == num:
                    return False

        return True

    def order_domain_values(self, row, col):
        return [num for num in range(1, 10) if self.is_safe(row, col, num)]

    def select_unassigned_variable(self):
        min_rem_val = float('inf')
        selected_row, selected_col = None, None
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    num_possibilities = len(self.order_domain_values(i, j))
                    if num_possibilities < min_rem_val:
                        min_rem_val = num_possibilities
                        selected_row, selected_col = i, j
        return selected_row, selected_col

    def solve_sudoku(self):
        row, col = self.find_empty_location()
        if row is None and col is None:
            return True

        for num in self.order_domain_values(row, col):
            if self.is_safe(row, col, num):
                self.board[row][col] = num
                self.assignment_count += 1
                if self.solve_sudoku():
                    return True
                self.board[row][col] = 0

        return False

    def solve(self):
        start_time = time.time()

        if self.solve_sudoku():
            end_time = time.time()
            elapsed_time = end_time - start_time
            return self.board, self.assignment_count, elapsed_time
        else:
            return None, self.assignment_count, None


# منطق اصلی برای انتخاب الگوریتم و حل سودوکو

def main():
    sudoku_board = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])

    print("Choose the algorithm to solve Sudoku:")
    print("1. Genetic Algorithm")
    print("2. CSP (Constraint Satisfaction Problem)")

    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        ga_solver = GeneticAlgorithmSolver(sudoku_board)
        best_solution, best_fitness, num_generations, elapsed_time = ga_solver.solve()

        print("Best solution found using Genetic Algorithm:")
        ga_solver.print_sudoku(best_solution)
        print(f"Best Fitness: {best_fitness}")
        print(f"Generations: {num_generations}")
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")

    elif choice == '2':
        csp_solver = CSPSolver(sudoku_board)
        solved_board, assignment_count, elapsed_time = csp_solver.solve()

        if solved_board is not None:
            print("Solved Sudoku board using CSP:")
            csp_solver.print_sudoku(solved_board)
            print(f"Number of assignments: {assignment_count}")
            print(f"Time taken: {elapsed_time:.4f} seconds")
        else:
            print("No solution exists using CSP.")

    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()