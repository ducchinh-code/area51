#!/usr/bin/env python3
"""
Area51 Fence Puzzle Visualizer

This script visualizes Area51 fence puzzles from JSON files.
"""

import os
import sys
import glob
import json
import argparse
from visualize.vs_data import visualize_puzzle, load_puzzle, create_puzzle_figure, save_puzzle_image

def visualize_puzzles():
    """Visualize all puzzle files in the data directory"""
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to data directory
    data_dir = os.path.join(script_dir, "data")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files in the data directory
    puzzle_files = glob.glob(os.path.join(data_dir, "puzzle_*.json"))
    
    if not puzzle_files:
        print(f"No puzzle files found in {data_dir}")
        return []
    
    print(f"Found {len(puzzle_files)} puzzle files")
    
    # List to store output paths
    output_paths = []
    
    # Process each puzzle file
    for file_path in puzzle_files:
        try:
            # Check if the file is a valid puzzle file
            with open(file_path, 'r') as f:
                data = json.load(f)
                if "matrix_1" not in data or "matrix_2" not in data:
                    print(f"Skipping {file_path} - not a valid puzzle file")
                    continue
            
            # Get puzzle name
            puzzle_name = os.path.basename(file_path).split('.')[0]
            print(f"Visualizing {puzzle_name}...")
            
            # Create the figure
            title = f"Area51 Fence Puzzle: {puzzle_name}"
            fig = create_puzzle_figure(data, title)
            
            # Save the image
            output_path = os.path.join(output_dir, f"{puzzle_name}.png")
            save_puzzle_image(fig, output_path)
            
            print(f"Image saved to {output_path}")
            output_paths.append(output_path)
            
        except json.JSONDecodeError as e:
            print(f"Error: {file_path} contains invalid JSON: {e}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("\nVisualization complete. Images saved to the output directory.")
    return output_paths

def main():
    """Main function to run the visualizer"""
    parser = argparse.ArgumentParser(description='Area51 Fence Puzzle Visualizer')
    parser.add_argument('--puzzle', '-p', type=str, required=False,
                        help='Path to a specific puzzle file (JSON format)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Visualize all puzzles in the data directory')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Save the visualization to an image file')
    parser.add_argument('--output', '-o', type=str, required=False,
                        help='Output directory for saved images')
    
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Visualize all puzzles
    if args.all:
        output_paths = visualize_puzzles()
        print(f"Visualized {len(output_paths)} puzzles")
        return
    
    # Visualize a specific puzzle
    if args.puzzle:
        # Check if the puzzle file exists
        if not os.path.exists(args.puzzle):
            print(f"Error: Puzzle file '{args.puzzle}' not found.")
            return
        
        # Visualize the puzzle
        output_path = visualize_puzzle(args.puzzle, save_image=args.save)
        
        if args.save:
            print(f"Puzzle visualization saved to {output_path}")
        
        return

if __name__ == "__main__":
    main() 