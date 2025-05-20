#!/usr/bin/env python3
"""
Area51 Fence Puzzle Solver

This script solves and visualizes Area51 fence puzzles from JSON files.
"""

import os
import sys
import argparse
from model.area51 import solve_area51_puzzle

def main():
    """Main function to run the solver"""
    parser = argparse.ArgumentParser(description='Area51 Fence Puzzle Solver')
    parser.add_argument('--puzzle', '-p', type=str, required=False,
                        help='Path to a specific puzzle file (JSON format)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Solve all puzzles in the data directory')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Save the visualization to an image file')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Output directory for saved images')
    
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Solve all puzzles
    if args.all:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
        
        # Create output directory if it doesn't exist
        if args.save and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get all puzzle files
        puzzle_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        
        for puzzle_file in puzzle_files:
            puzzle_path = os.path.join(data_dir, puzzle_file)
            print(f"Solving puzzle: {puzzle_file}")
            
            # Solve the puzzle
            solver = solve_area51_puzzle(puzzle_path)
            
            if solver:
                if args.save:
                    output_path = os.path.join(output_dir, f"{os.path.splitext(puzzle_file)[0]}_solution.png")
                    solver.visualize_solution(output_path)
                    print(f"Solution saved to: {output_path}")
                else:
                    solver.visualize_solution()
            else:
                print(f"No solution found for puzzle: {puzzle_file}")
        
        return
    
    # Solve a specific puzzle
    if args.puzzle:
        # Check if the puzzle file exists
        if not os.path.exists(args.puzzle):
            print(f"Error: Puzzle file '{args.puzzle}' not found.")
            return
        
        # Solve the puzzle
        solver = solve_area51_puzzle(args.puzzle)
        
        if solver:
            if args.save:
                # Create output directory if it doesn't exist
                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Get puzzle filename
                puzzle_file = os.path.basename(args.puzzle)
                output_path = os.path.join(output_dir, f"{os.path.splitext(puzzle_file)[0]}_solution.png")
                
                solver.visualize_solution(output_path)
                print(f"Solution saved to: {output_path}")
            else:
                solver.visualize_solution()
        else:
            print("No solution found for the puzzle.")
        
        return

if __name__ == "__main__":
    main()
