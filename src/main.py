import os
import sys
import json
import argparse
try:
    from model.area51 import Area51Solver as GurobiSolver
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from model.pulp_area51 import Area51Solver as PuLPSolver


def parse_args():
    parser = argparse.ArgumentParser(description='Area51 Fence Puzzle Solver')
    parser.add_argument('--puzzle', '-p', type=str, required=False, default='data/puzzle_2.json',
                      help='Path to the puzzle file (JSON format)')
    parser.add_argument('--output', '-o', type=str, required=False,
                      help='Path to save the solution (JSON format)')
    parser.add_argument('--time-limit', '-t', type=int, required=False, default=None,
                      help='Time limit for solving (in seconds)')
    parser.add_argument('--visualize', '-v', action='store_true',
                      help='Visualize the solution')
    parser.add_argument('--debug', '-d', action='store_true',
                      help='Enable debug mode')
    parser.add_argument('--solver', '-s', type=str, choices=['gurobi', 'pulp'], default='pulp',
                      help='Choose the solver to use (gurobi or pulp)')
    return parser.parse_args()


def print_puzzle(puzzle_data):
    """Print the puzzle in a readable format"""
    matrix_1 = puzzle_data.get('matrix_1', [])
    matrix_2 = puzzle_data.get('matrix_2', [])
    
    if not matrix_1 or not matrix_2:
        print("Invalid puzzle data")
        return
    
    rows = len(matrix_1)
    cols = len(matrix_1[0])
    
    print("\nPuzzle:")
    for i in range(rows + 1):
        # Print vertices
        line = ""
        for j in range(cols + 1):
            if i < len(matrix_2) and j < len(matrix_2[i]) and matrix_2[i][j] is not None:
                if matrix_2[i][j] == "F":
                    line += "● "  # Black circle
                elif matrix_2[i][j] == "E":
                    line += "○ "  # White circle
                else:
                    line += matrix_2[i][j] + " "
            else:
                line += "+ "
        print(line)
        
        # Print cells
        if i < rows:
            line = ""
            for j in range(cols):
                if matrix_1[i][j] is not None:
                    if isinstance(matrix_1[i][j], dict) and matrix_1[i][j].get("circled"):
                        line += str(matrix_1[i][j]["value"]) + "○ "
                    else:
                        line += str(matrix_1[i][j]) + " "
                else:
                    line += "  "
            print(line)
    print("")


def main():
    args = parse_args()
    
    # Check if the puzzle file exists
    if not os.path.exists(args.puzzle):
        print(f"Error: Puzzle file '{args.puzzle}' not found.")
        sys.exit(1)
    
    # Create and run the solver
    try:
        print(f"Loading puzzle from {args.puzzle}...")
    
        # Make sure this script can be run from the project root or src directory
        base_dir = os.getcwd()
        puzzle_path = args.puzzle
        if not os.path.isabs(puzzle_path) and not os.path.exists(puzzle_path):
            # Try relative to src directory
            puzzle_path = os.path.join('src', puzzle_path)
            if not os.path.exists(puzzle_path):
                print(f"Error: Could not find puzzle file at {args.puzzle} or {puzzle_path}")
                sys.exit(1)
                
        print(f"Reading puzzle file from: {puzzle_path}")
        
        # Load puzzle data
        with open(puzzle_path, 'r') as f:
                puzzle_data = json.load(f)
                
        # Print puzzle in readable format
        if args.debug:
            print_puzzle(puzzle_data)
        
        # Choose and create solver
        if args.solver == 'gurobi' and GUROBI_AVAILABLE:
            print("Using Gurobi solver")
            solver = GurobiSolver(puzzle_path)
        else:
            if args.solver == 'gurobi' and not GUROBI_AVAILABLE:
                print("Gurobi không khả dụng. Sử dụng PuLP thay thế.")
            else:
                print("Using PuLP solver")
            solver = PuLPSolver(puzzle_path)
            
        print(f"Puzzle loaded. Grid size: {solver.rows}x{solver.cols}")
            
        # Enable debug logging if in debug mode
        if args.debug and hasattr(solver, 'model') and hasattr(solver.model, 'Params'):
            # Only for Gurobi
            solver.model.Params.LogToConsole = 1
            solver.model.Params.OutputFlag = 1
        
        # Solve the puzzle
        print("Solving the puzzle...")
        output_path = args.output
        result = solver.solve(time_limit=args.time_limit, output_file=output_path)
        
        # Check and report the result
        if isinstance(result, dict):
            status = result.get("status")
            status_str = result.get("status_str", "UNKNOWN")
            
            if (GUROBI_AVAILABLE and status == 2) or (not GUROBI_AVAILABLE and status == 1):  # Optimal
                print(f"Optimal solution found!")
                
                if output_path:
                    print(f"Solution saved to {output_path}")
                    
                # Display solution
                solver.display_solution()
                    
            else:
                print(f"No optimal solution found. Status: {status_str}")
                
                # Print any additional information
                if "message" in result:
                    print(f"Message: {result['message']}")
        else:
            print("Solver returned an invalid result.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 
