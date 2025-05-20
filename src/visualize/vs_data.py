import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
from matplotlib.patches import Circle, Rectangle, Polygon, Ellipse, PathPatch
from matplotlib.path import Path

def load_puzzle(file_path):
    """Load puzzle data from a JSON file"""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error loading puzzle file {file_path}: {e}")
        return None

def draw_alien(ax, x, y, size=0.25):
    """Draw a better looking alien symbol"""
    # Main body (head) - blue oval
    head = Ellipse((x, y), size*1.2, size*1.5, color='royalblue', alpha=0.9)
    ax.add_patch(head)
    
    # Eyes - two small black ovals
    eye_size = size * 0.25
    left_eye = Ellipse((x-size*0.3, y+size*0.3), eye_size, eye_size*1.5, color='black')
    right_eye = Ellipse((x+size*0.3, y+size*0.3), eye_size, eye_size*1.5, color='black')
    ax.add_patch(left_eye)
    ax.add_patch(right_eye)
    
    # Antenna
    ax.plot([x, x], [y+size*0.75, y+size*1.2], color='black', linewidth=1.5)
    antenna_top = Circle((x, y+size*1.2), size*0.15, color='black')
    ax.add_patch(antenna_top)

def draw_cactus(ax, x, y, size=0.25):
    """Draw a better looking cactus symbol"""
    # Main stem
    stem = Rectangle((x-size*0.15, y-size*0.6), size*0.3, size*1.2, 
                     facecolor='green', edgecolor='darkgreen', linewidth=1)
    ax.add_patch(stem)
    
    # Left arm
    left_arm_main = Rectangle((x-size*0.5, y), size*0.35, size*0.3, 
                         facecolor='green', edgecolor='darkgreen', linewidth=1)
    ax.add_patch(left_arm_main)
    left_arm_up = Rectangle((x-size*0.35, y), size*0.2, size*0.5, 
                       facecolor='green', edgecolor='darkgreen', linewidth=1)
    ax.add_patch(left_arm_up)
    
    # Right arm
    right_arm_main = Rectangle((x+size*0.15, y-size*0.3), size*0.35, size*0.3, 
                          facecolor='green', edgecolor='darkgreen', linewidth=1)
    ax.add_patch(right_arm_main)
    right_arm_up = Rectangle((x+size*0.3, y-size*0.3), size*0.2, size*0.6, 
                        facecolor='green', edgecolor='darkgreen', linewidth=1)
    ax.add_patch(right_arm_up)

def draw_grid_lines(ax, rows, cols):
    """Draw light grid lines for better clarity"""
    # Draw horizontal grid lines
    for i in range(rows + 1):
        ax.plot([0, cols], [rows - i, rows - i], color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    
    # Draw vertical grid lines
    for j in range(cols + 1):
        ax.plot([j, j], [0, rows], color='lightgray', linestyle='-', linewidth=0.5, zorder=0)

def draw_matrix(matrix_1, matrix_2, title, fig, ax, show_grid=True):
    """Draw a puzzle with proper formatting showing both matrices"""
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    rows = len(matrix_1)
    cols = len(matrix_1[0])
    
    # Set background color to white (no background)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Draw grid lines if requested
    if show_grid:
        draw_grid_lines(ax, rows, cols)
    
    # Draw grid points
    for i in range(rows + 1):
        for j in range(cols + 1):
            ax.plot(j, rows - i, 'k.', markersize=2)
    
    # Draw cell contents (matrix_1)
    for i in range(rows):
        for j in range(cols):
            cell = matrix_1[i][j]
            if cell is not None:
                if cell == "A":
                    # Alien - use improved alien drawing
                    draw_alien(ax, j + 0.5, rows - i - 0.5)
                elif cell == "C":
                    # Cactus - use improved cactus drawing
                    draw_cactus(ax, j + 0.5, rows - i - 0.5)
                elif isinstance(cell, dict) and cell.get("circled"):
                    # Circled number - draw a dotted circle around the number
                    circle = Circle((j + 0.5, rows - i - 0.5), 0.3, fill=False, color='black', 
                                   linestyle='dotted', linewidth=1.5)
                    ax.add_patch(circle)
                    ax.text(j + 0.5, rows - i - 0.5, str(cell["value"]), 
                           ha='center', va='center', fontsize=12, fontweight='bold')
                else:
                    # Regular number
                    ax.text(j + 0.5, rows - i - 0.5, str(cell), 
                           ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw vertex contents (matrix_2)
    for i in range(rows + 1):
        for j in range(cols + 1):
            if i < len(matrix_2) and j < len(matrix_2[i]) and matrix_2[i][j] is not None:
                if matrix_2[i][j] == "F":
                    # Black circle - Masyu black circle
                    circle = Circle((j, rows - i), 0.12, facecolor='black', edgecolor='black')
                    ax.add_patch(circle)
                elif matrix_2[i][j] == "E":
                    # White circle - Masyu white circle with better appearance
                    outer_circle = Circle((j, rows - i), 0.12, facecolor='white', 
                                        edgecolor='black', linewidth=1.2, zorder=2)
                    inner_circle = Circle((j, rows - i), 0.09, facecolor='white', 
                                        edgecolor='white', linewidth=0, zorder=3)
                    ax.add_patch(outer_circle)
                    ax.add_patch(inner_circle)
    
    # Set limits to show the entire grid with some padding
    ax.set_xlim(-0.5, cols + 0.5)
    ax.set_ylim(-0.5, rows + 0.5)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')

def create_puzzle_figure(data, title="", figsize=(7, 7), show_grid=True):
    """Create a figure with the puzzle visualization"""
    fig, ax = plt.subplots(figsize=figsize)
    draw_matrix(data["matrix_1"], data["matrix_2"], title, fig, ax, show_grid)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    return fig

def save_puzzle_image(fig, output_path, format='png', dpi=150):
    """Save the puzzle figure to an image file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the figure in the specified format
    fig.savefig(output_path, bbox_inches='tight', dpi=dpi, format=format)
    plt.close(fig)

def visualize_puzzle(file_path, save_image=False, output_format='png', show_grid=True):
    """Visualize a puzzle from a JSON file"""
    data = load_puzzle(file_path)
    if not data:
        return
    
    puzzle_name = os.path.basename(file_path).split('.')[0]
    title = f"Area51 Fence Puzzle: {puzzle_name}" if save_image else ""
    
    # Create the figure
    fig = create_puzzle_figure(data, title, show_grid=show_grid)
    
    if save_image:
        # Determine output path
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        
        # Set file extension based on format
        file_ext = f".{output_format}"
        output_path = os.path.join(output_dir, f"{puzzle_name}{file_ext}")
        
        # Save the image
        save_puzzle_image(fig, output_path, format=output_format)
        return output_path
    else:
        plt.show()
        return fig

def visualize_solution(puzzle_file, solution_file, save_image=False, output_format='png'):
    """Visualize a puzzle solution alongside the original puzzle"""
    # Load puzzle and solution data
    puzzle_data = load_puzzle(puzzle_file)
    solution_data = load_puzzle(solution_file)
    
    if not puzzle_data or not solution_data:
        return
    
    puzzle_name = os.path.basename(puzzle_file).split('.')[0]
    title = f"Area51 Fence Puzzle: {puzzle_name}"
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Draw the original puzzle
    draw_matrix(puzzle_data["matrix_1"], puzzle_data["matrix_2"], "Original Puzzle", fig, ax1)
    
    # Draw the solution
    draw_matrix(solution_data["matrix_1"], solution_data["matrix_2"], "Solution", fig, ax2)
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.1)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_image:
        # Determine output path
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        
        # Set file extension based on format
        file_ext = f".{output_format}"
        output_path = os.path.join(output_dir, f"{puzzle_name}_solution{file_ext}")
        
        # Save the image
        save_puzzle_image(fig, output_path, format=output_format)
        return output_path
    else:
        plt.show()
        return fig

if __name__ == "__main__":
    # If a file path is provided as an argument, use it
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        visualize_puzzle(file_path)
    else:
        # Otherwise, use a default puzzle
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "puzzle_1.json")
        visualize_puzzle(default_path)