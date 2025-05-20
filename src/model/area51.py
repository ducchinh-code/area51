import json
import numpy as np
import pulp as pl
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Set, Optional

class Area51Solver:
    def __init__(self, puzzle_path: str):
        """Initialize the Area51 puzzle solver."""
        self.puzzle_path = puzzle_path
        self.load_puzzle()
        self.solution = None
        
    def load_puzzle(self):
        """Load puzzle data from JSON file."""
        try:
            with open(self.puzzle_path, 'r') as file:
                puzzle_data = json.load(file)
            
            # Verify required keys exist
            if "matrix_1" not in puzzle_data or "matrix_2" not in puzzle_data:
                raise ValueError("Puzzle data must contain 'matrix_1' and 'matrix_2'")
            
            # Process matrix data
            self.matrix_1 = puzzle_data["matrix_1"]  # Cells matrix
            self.matrix_2 = puzzle_data["matrix_2"]  # Points matrix
            
            # Validate matrix data
            if not self.matrix_1 or not isinstance(self.matrix_1, list):
                raise ValueError("matrix_1 must be a non-empty list")
            if not self.matrix_2 or not isinstance(self.matrix_2, list):
                raise ValueError("matrix_2 must be a non-empty list")
            
            # Set dimensions
            self.n_r = len(self.matrix_1)       # Number of rows
            self.n_c = len(self.matrix_1[0])    # Number of columns
            
            # Validate matrix dimensions
            if self.n_r < 1 or self.n_c < 1:
                raise ValueError("Puzzle dimensions must be at least 1x1")
            
            print(f"Puzzle dimensions: {self.n_r} rows x {self.n_c} columns")
            print(f"Point matrix dimensions: {len(self.matrix_2)} rows x {len(self.matrix_2[0])} columns")
            
            # Validate that matrix_2 has the correct dimensions (matrix_1 rows+1 x matrix_1 columns+1)
            if len(self.matrix_2) != self.n_r + 1 or len(self.matrix_2[0]) != self.n_c + 1:
                print(f"Warning: matrix_2 dimensions ({len(self.matrix_2)}x{len(self.matrix_2[0])}) don't match expected size ({self.n_r+1}x{self.n_c+1})")
            
            # Extract constraint locations
            self.F = []  # Set of cells with uncircled numbers
            self.K = []  # Set of cells with circled numbers
            self.A = []  # Set of cells with aliens
            self.T = []  # Set of cells with triffids (cactus)
            self.W = []  # Set of white circles
            self.B = []  # Set of black circles
            
            # Process matrix_1 (cell data)
            for i in range(self.n_r):
                for j in range(self.n_c):
                    cell = self.matrix_1[i][j]
                    if cell == "A":
                        self.A.append((i+1, j+1))  # 1-indexed for consistency with constraints
                        print(f"Found alien at ({i+1},{j+1})")
                    elif cell == "C":
                        self.T.append((i+1, j+1))
                        print(f"Found triffid/cactus at ({i+1},{j+1})")
                    elif isinstance(cell, dict) and cell.get("circled"):
                        if "value" not in cell:
                            print(f"Warning: Circled number at ({i+1},{j+1}) missing 'value' key")
                            continue
                        self.K.append((i+1, j+1))
                        print(f"Found circled number {cell['value']} at ({i+1},{j+1})")
                    elif isinstance(cell, int):
                        self.F.append((i+1, j+1))
                        print(f"Found uncircled number {cell} at ({i+1},{j+1})")
            
            # Process matrix_2 (point data)
            for i in range(len(self.matrix_2)):
                for j in range(len(self.matrix_2[0])):
                    point = self.matrix_2[i][j]
                    if point == "E":  # White circles
                        self.W.append((i, j))
                        print(f"Found white circle at ({i},{j})")
                    elif point == "F":  # Black circles
                        self.B.append((i, j))
                        print(f"Found black circle at ({i},{j})")
            
            # Validate that we have at least one alien and one triffid
            if not self.A:
                print("Warning: No aliens found in puzzle")
            if not self.T:
                print("Warning: No triffids found in puzzle")
            
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading puzzle: {e}")
            raise
        except Exception as e:
            print(f"Error processing puzzle data: {e}")
            raise
    
    def solve(self):
        """Solve the Area51 puzzle."""
        # Initialize model
        self.model = pl.LpProblem("Area51_Fence_Puzzle", pl.LpMinimize)
        
        # Initialize decision variables
        self._init_variables()
        
        # Add constraints
        self._add_constraints()
        
        print(f"Model has {len(self.model.constraints)} constraints")
        print(f"Model has {self.model.numVariables()} variables")
        
        # Solve iteratively to ensure single loop
        iterations = 0
        max_iterations = 100  # Prevent infinite loops
        
        while iterations < max_iterations:
            # Solve the model
            self.model.solve(pl.PULP_CBC_CMD(msg=True))
            
            # Check if we have a valid solution
            if self.model.status != pl.LpStatusOptimal:
                print(f"Solver status: {pl.LpStatus[self.model.status]}")
                return False
            
            # Extract current solution
            h_sol = {(i, j): pl.value(self.h[i, j]) for i in range(self.n_r+1) for j in range(self.n_c)}
            v_sol = {(i, j): pl.value(self.v[i, j]) for i in range(self.n_r) for j in range(self.n_c+1)}
            
            # Find loops in current solution
            loops = self._find_loops(h_sol, v_sol)
            
            print(f"Found {len(loops)} loops in iteration {iterations+1}")
            
            if len(loops) == 1:
                # We have found a valid solution with a single loop
                print(f"Solution found after {iterations+1} iterations.")
                self.solution = {"h": h_sol, "v": v_sol}
                return True
            
            # Add constraints to eliminate multiple loops
            for loop_idx, loop in enumerate(loops):
                h_loop = []
                v_loop = []
                
                # Separate horizontal and vertical segments
                for i in range(len(loop)-1):
                    curr = loop[i]
                    next_node = loop[i+1]
                    
                    # Check if segment is horizontal or vertical
                    if curr[0] == next_node[0]:  # Horizontal
                        if curr[1] < next_node[1]:
                            h_loop.append((curr[0], curr[1]))
                        else:
                            h_loop.append((curr[0], next_node[1]))
                    else:  # Vertical
                        if curr[0] < next_node[0]:
                            v_loop.append((curr[0], curr[1]))
                        else:
                            v_loop.append((next_node[0], curr[1]))
                
                # Add constraint to prevent this loop
                if h_loop or v_loop:
                    print(f"Adding constraint to eliminate loop {loop_idx+1} with {len(h_loop)} horizontal and {len(v_loop)} vertical segments")
                    self.model += (
                        pl.lpSum(self.h[i, j] for i, j in h_loop) + 
                        pl.lpSum(self.v[i, j] for i, j in v_loop) <= 
                        len(h_loop) + len(v_loop) - 1
                    )
            
            iterations += 1
        
        print(f"Maximum iterations ({max_iterations}) reached without finding a valid solution.")
        return False
    
    def _init_variables(self):
        """Initialize decision variables."""
        # Horizontal fence segments
        self.h = {}
        for i in range(self.n_r+1):
            for j in range(self.n_c):
                self.h[i, j] = pl.LpVariable(f"h_{i}_{j}", 0, 1, pl.LpBinary)
        
        # Vertical fence segments
        self.v = {}
        for i in range(self.n_r):
            for j in range(self.n_c+1):
                self.v[i, j] = pl.LpVariable(f"v_{i}_{j}", 0, 1, pl.LpBinary)
        
        # Points on the fence
        self.p = {}
        for i in range(self.n_r+1):
            for j in range(self.n_c+1):
                self.p[i, j] = pl.LpVariable(f"p_{i}_{j}", 0, 1, pl.LpBinary)
        
        # Inside/outside cells
        self.inside = {}
        for i in range(self.n_r):
            for j in range(self.n_c):
                self.inside[i, j] = pl.LpVariable(f"inside_{i}_{j}", 0, 1, pl.LpBinary)
    
    def _add_constraints(self):
        """Add all constraints to the model."""
        # Objective function (any valid solution is acceptable)
        self.model += 0, "Empty objective"
        
        # Add constraints gradually to identify the source of infeasibility
        print("Adding cycle constraints...")
        self._add_cycle_constraints()
        
        # First try with just cycle constraints
        result = self.model.solve(pl.PULP_CBC_CMD(msg=False))
        if self.model.status != pl.LpStatusOptimal:
            print("Model is infeasible with just cycle constraints!")
            return
        print("Model is feasible with cycle constraints.")
        
        # Add uncircled number constraints
        print("Adding uncircled number constraints...")
        self._add_uncircled_number_constraints()
        
        # Try solving with these constraints
        result = self.model.solve(pl.PULP_CBC_CMD(msg=False))
        if self.model.status != pl.LpStatusOptimal:
            print("Model became infeasible after adding uncircled number constraints!")
            return
        print("Model is feasible with uncircled number constraints.")
        
        # Initialize inside/outside variables early
        print("Adding inside/outside cell constraints...")
        
        # Choose a reference point known to be outside (we'll fix a point at the edge as "outside")
        if self.n_r > 0 and self.n_c > 0:
            # Fix the top-left cell as "outside" (reference point)
            self.model += self.inside[0, 0] == 0, "Outside_Reference"
        
        # For each pair of adjacent cells, they must have the same inside/outside status if there's no fence between them
        for i in range(self.n_r):
            for j in range(self.n_c):
                # Connect with cell to the right (if in bounds)
                if j < self.n_c - 1:
                    # If no vertical fence between cells (i,j) and (i,j+1), they have the same inside/outside status
                    self.model += self.inside[i, j] - self.inside[i, j+1] <= self.v[i, j+1], f"Inside_Right_{i}_{j}"
                    self.model += self.inside[i, j+1] - self.inside[i, j] <= self.v[i, j+1], f"Inside_Left_{i}_{j+1}"
                
                # Connect with cell below (if in bounds)
                if i < self.n_r - 1:
                    # If no horizontal fence between cells (i,j) and (i+1,j), they have the same inside/outside status
                    self.model += self.inside[i, j] - self.inside[i+1, j] <= self.h[i+1, j], f"Inside_Down_{i}_{j}" 
                    self.model += self.inside[i+1, j] - self.inside[i, j] <= self.h[i+1, j], f"Inside_Up_{i+1}_{j}"
        
        # Try solving with inside/outside variables
        result = self.model.solve(pl.PULP_CBC_CMD(msg=False))
        if self.model.status != pl.LpStatusOptimal:
            print("Model became infeasible after adding inside/outside variables!")
            return
        print("Model is feasible with inside/outside variables.")
        
        # Add alien and triffid constraints - THIS IS THE KEY PART FOR DEFINING INSIDE VS OUTSIDE
        print("Adding alien and triffid constraints...")
        # Aliens must be inside the fence
        for p, q in self.A:
            # Values are 1-indexed in A
            i, j = p-1, q-1
            # Aliens MUST be INSIDE the fence (inside=1)
            self.model += self.inside[i, j] == 1, f"Alien_Inside_{p}_{q}"
        
        # Triffids must be outside the fence
        for p, q in self.T:
            # Values are 1-indexed in T
            i, j = p-1, q-1
            # Triffids MUST be OUTSIDE the fence (inside=0)
            self.model += self.inside[i, j] == 0, f"Triffid_Outside_{p}_{q}"
        
        # Check feasibility after alien/triffid constraints
        result = self.model.solve(pl.PULP_CBC_CMD(msg=False))
        if self.model.status != pl.LpStatusOptimal:
            print("Model became infeasible after adding alien and triffid constraints!")
            return
        print("Model is feasible with alien and triffid constraints.")
            
        # Add circled number constraints
        print("Adding circled number constraints...")
        self._add_circled_number_constraints()
        
        # Try solving after circled number constraints
        result = self.model.solve(pl.PULP_CBC_CMD(msg=False))
        if self.model.status != pl.LpStatusOptimal:
            print("Model became infeasible after adding circled number constraints!")
            return
        print("Model is feasible with circled number constraints.")
        
        # Try adding black circle constraints separately
        print("Adding black circle constraints...")
        self._add_black_circle_constraints()
        
        # Check feasibility after black circle constraints
        result = self.model.solve(pl.PULP_CBC_CMD(msg=False))
        if self.model.status != pl.LpStatusOptimal:
            print("Model became infeasible after adding black circle constraints!")
            return
        print("Model is feasible with black circle constraints.")
        
        # Try adding white circle constraints separately
        print("Adding white circle constraints...")
        self._add_white_circle_constraints()
        
        # Final check
        result = self.model.solve(pl.PULP_CBC_CMD(msg=False))
        if self.model.status != pl.LpStatusOptimal:
            print("Model became infeasible after adding white circle constraints!")
            return
        
        print("All constraints added successfully. Model is feasible.")
    
    def _add_uncircled_number_constraints(self):
        """Add constraints for uncircled numbers."""
        for p, q in self.F:
            # Values are 1-indexed in F
            i, j = p-1, q-1
            value = self.matrix_1[i][j]
            
            # The sum of fence segments around this square must equal the value
            self.model += (
                self.h[i, j] + self.h[i+1, j] + 
                self.v[i, j] + self.v[i, j+1] == value,
                f"Uncircled_Number_{p}_{q}"
            )
    
    def _add_circled_number_constraints(self):
        """Add constraints for circled numbers."""
        for p, q in self.K:
            # Values are 1-indexed in K
            i, j = p-1, q-1
            value = self.matrix_1[i][j]["value"]
            
            # Circled numbers MUST be inside the fence (inside=1)
            self.model += self.inside[i, j] == 1, f"Circled_Number_Inside_{p}_{q}"
            
            # Add visibility constraints - count visible squares in the 4 cardinal directions
            # Create variables to count visible squares in each direction
            visible_squares = []

            # The circled number square itself is always visible (counted in the value)
            visible_squares.append(1)  # Always count the center square itself
            
            # Check visibility to the north (negative i direction)
            for di in range(i-1, -1, -1):
                # Create a variable for visibility of this square
                vis_var = pl.LpVariable(f"vis_N_{i}_{j}_{di}", 0, 1, pl.LpBinary)
                
                # Check for fence crossings between the circled number and this cell
                fence_crossings = []
                for k in range(di, i):
                    fence_crossings.append(self.h[k, j])  # Horizontal fence segment crossing the line of sight
                
                # Square is visible if it's inside AND there are no fence crossings
                max_crossings = pl.LpVariable(f"cross_N_{i}_{j}_{di}", 0, 1, pl.LpBinary)
                if fence_crossings:
                    self.model += pl.lpSum(fence_crossings) <= len(fence_crossings) * (1 - max_crossings), f"Cross_N_{i}_{j}_{di}"
                    self.model += max_crossings <= 1 - (pl.lpSum(fence_crossings) / len(fence_crossings)), f"Cross_N2_{i}_{j}_{di}"
                else:
                    self.model += max_crossings == 1, f"Cross_N3_{i}_{j}_{di}"  # No fence crossings possible
                    
                # Square is visible if inside AND no fence crossings
                self.model += vis_var <= self.inside[di, j], f"Vis_N_Inside_{i}_{j}_{di}"
                self.model += vis_var <= max_crossings, f"Vis_N_Fence_{i}_{j}_{di}"
                self.model += vis_var >= self.inside[di, j] + max_crossings - 1, f"Vis_N_Both_{i}_{j}_{di}"
                
                visible_squares.append(vis_var)
            
            # Check visibility to the south (positive i direction)
            for di in range(i+1, self.n_r):
                # Create a variable for visibility of this square
                vis_var = pl.LpVariable(f"vis_S_{i}_{j}_{di}", 0, 1, pl.LpBinary)
                
                # Check for fence crossings between the circled number and this cell
                fence_crossings = []
                for k in range(i+1, di+1):
                    fence_crossings.append(self.h[k, j])  # Horizontal fence segment crossing the line of sight
                
                # Square is visible if it's inside AND there are no fence crossings
                max_crossings = pl.LpVariable(f"cross_S_{i}_{j}_{di}", 0, 1, pl.LpBinary)
                if fence_crossings:
                    self.model += pl.lpSum(fence_crossings) <= len(fence_crossings) * (1 - max_crossings), f"Cross_S_{i}_{j}_{di}"
                    self.model += max_crossings <= 1 - (pl.lpSum(fence_crossings) / len(fence_crossings)), f"Cross_S2_{i}_{j}_{di}"
                else:
                    self.model += max_crossings == 1, f"Cross_S3_{i}_{j}_{di}"  # No fence crossings possible
                    
                # Square is visible if inside AND no fence crossings
                self.model += vis_var <= self.inside[di, j], f"Vis_S_Inside_{i}_{j}_{di}"
                self.model += vis_var <= max_crossings, f"Vis_S_Fence_{i}_{j}_{di}"
                self.model += vis_var >= self.inside[di, j] + max_crossings - 1, f"Vis_S_Both_{i}_{j}_{di}"
                
                visible_squares.append(vis_var)
            
            # Check visibility to the east (positive j direction)
            for dj in range(j+1, self.n_c):
                # Create a variable for visibility of this square
                vis_var = pl.LpVariable(f"vis_E_{i}_{j}_{dj}", 0, 1, pl.LpBinary)
                
                # Check for fence crossings between the circled number and this cell
                fence_crossings = []
                for k in range(j+1, dj+1):
                    fence_crossings.append(self.v[i, k])  # Vertical fence segment crossing the line of sight
                
                # Square is visible if it's inside AND there are no fence crossings
                max_crossings = pl.LpVariable(f"cross_E_{i}_{j}_{dj}", 0, 1, pl.LpBinary)
                if fence_crossings:
                    self.model += pl.lpSum(fence_crossings) <= len(fence_crossings) * (1 - max_crossings), f"Cross_E_{i}_{j}_{dj}"
                    self.model += max_crossings <= 1 - (pl.lpSum(fence_crossings) / len(fence_crossings)), f"Cross_E2_{i}_{j}_{dj}"
                else:
                    self.model += max_crossings == 1, f"Cross_E3_{i}_{j}_{dj}"  # No fence crossings possible
                    
                # Square is visible if inside AND no fence crossings
                self.model += vis_var <= self.inside[i, dj], f"Vis_E_Inside_{i}_{j}_{dj}"
                self.model += vis_var <= max_crossings, f"Vis_E_Fence_{i}_{j}_{dj}"
                self.model += vis_var >= self.inside[i, dj] + max_crossings - 1, f"Vis_E_Both_{i}_{j}_{dj}"
                
                visible_squares.append(vis_var)
            
            # Check visibility to the west (negative j direction)
            for dj in range(j-1, -1, -1):
                # Create a variable for visibility of this square
                vis_var = pl.LpVariable(f"vis_W_{i}_{j}_{dj}", 0, 1, pl.LpBinary)
                
                # Check for fence crossings between the circled number and this cell
                fence_crossings = []
                for k in range(dj, j):
                    fence_crossings.append(self.v[i, k])  # Vertical fence segment crossing the line of sight
                
                # Square is visible if it's inside AND there are no fence crossings
                max_crossings = pl.LpVariable(f"cross_W_{i}_{j}_{dj}", 0, 1, pl.LpBinary)
                if fence_crossings:
                    self.model += pl.lpSum(fence_crossings) <= len(fence_crossings) * (1 - max_crossings), f"Cross_W_{i}_{j}_{dj}"
                    self.model += max_crossings <= 1 - (pl.lpSum(fence_crossings) / len(fence_crossings)), f"Cross_W2_{i}_{j}_{dj}"
                else:
                    self.model += max_crossings == 1, f"Cross_W3_{i}_{j}_{dj}"  # No fence crossings possible
                    
                # Square is visible if inside AND no fence crossings
                self.model += vis_var <= self.inside[i, dj], f"Vis_W_Inside_{i}_{j}_{dj}"
                self.model += vis_var <= max_crossings, f"Vis_W_Fence_{i}_{j}_{dj}"
                self.model += vis_var >= self.inside[i, dj] + max_crossings - 1, f"Vis_W_Both_{i}_{j}_{dj}"
                
                visible_squares.append(vis_var)
            
            # Total visibility is the sum of visible squares (including the circled number itself)
            self.model += pl.lpSum(visible_squares) == value, f"Circled_Number_Visibility_{p}_{q}"
            
            print(f"Added visibility constraints for circled number {value} at ({p},{q})")
    
    def _add_cycle_constraints(self):
        """Add constraints to ensure a valid cycle."""
        # Each point on the fence has exactly 2 connected segments
        for i in range(self.n_r+1):
            for j in range(self.n_c+1):
                # Left, right, up, down segments
                segments = []
                if j > 0:
                    segments.append(self.h[i, j-1])  # Left
                if j < self.n_c:
                    segments.append(self.h[i, j])    # Right
                if i > 0:
                    segments.append(self.v[i-1, j])  # Up
                if i < self.n_r:
                    segments.append(self.v[i, j])    # Down
                
                # Either 0 or 2 segments must be connected to this point
                self.model += (pl.lpSum(segments) == 2*self.p[i, j], f"Point_{i}_{j}")
    
    def _add_black_circle_constraints(self):
        """Add constraints for black circles."""
        print(f"Number of black circles: {len(self.B)}")
        for i, j in self.B:
            # Black circle requires a 90° turn and extends at least one segment in both directions
            print(f"Adding strict constraints for black circle at ({i}, {j})")
            
            # For each black circle, identify possible segments
            h_left = (i < self.n_r+1 and j > 0, self.h[i, j-1] if i < self.n_r+1 and j > 0 else None)
            h_right = (i < self.n_r+1 and j < self.n_c, self.h[i, j] if i < self.n_r+1 and j < self.n_c else None)
            v_up = (i > 0 and j < self.n_c+1, self.v[i-1, j] if i > 0 and j < self.n_c+1 else None)
            v_down = (i < self.n_r and j < self.n_c+1, self.v[i, j] if i < self.n_r and j < self.n_c+1 else None)
            
            # Simplified approach: count available segments first
            available_segments = []
            if h_left[0]: available_segments.append(h_left[1])
            if h_right[0]: available_segments.append(h_right[1])
            if v_up[0]: available_segments.append(v_up[1])
            if v_down[0]: available_segments.append(v_down[1])
            
            # The point must be on the fence
            if available_segments:
                self.model += pl.lpSum(available_segments) >= 2*self.p[i, j], f"Black_FencePoint_{i}_{j}"
                # Either 0 or 2 segments must be connected (consistent with cycle constraints)
                self.model += pl.lpSum(available_segments) <= 2, f"Black_Max2_{i}_{j}"
            else:
                print(f"Warning: Black circle at ({i}, {j}) has no available segments!")
                continue
            
            # Check horizontal-vertical pairs for 90° turns
            if h_left[0] and v_up[0]:
                # If both segments used, extensions must exist
                hv_corner = pl.LpVariable(f"hv1_{i}_{j}", 0, 1, pl.LpBinary)
                self.model += h_left[1] + v_up[1] - 1 <= hv_corner, f"Black_HV1_Use_{i}_{j}"
                
                # If this corner is used, check for available extensions
                if j > 1:  # Can extend left
                    self.model += self.h[i, j-2] >= hv_corner, f"Black_HV1_ExtL_{i}_{j}"
                if i > 1:  # Can extend up
                    self.model += self.v[i-2, j] >= hv_corner, f"Black_HV1_ExtU_{i}_{j}"
            
            if h_left[0] and v_down[0]:
                # If both segments used, extensions must exist
                hv_corner = pl.LpVariable(f"hv2_{i}_{j}", 0, 1, pl.LpBinary)
                self.model += h_left[1] + v_down[1] - 1 <= hv_corner, f"Black_HV2_Use_{i}_{j}"
                
                # If this corner is used, check for available extensions
                if j > 1:  # Can extend left
                    self.model += self.h[i, j-2] >= hv_corner, f"Black_HV2_ExtL_{i}_{j}"
                if i+1 < self.n_r:  # Can extend down
                    self.model += self.v[i+1, j] >= hv_corner, f"Black_HV2_ExtD_{i}_{j}"
            
            if h_right[0] and v_up[0]:
                # If both segments used, extensions must exist
                hv_corner = pl.LpVariable(f"hv3_{i}_{j}", 0, 1, pl.LpBinary)
                self.model += h_right[1] + v_up[1] - 1 <= hv_corner, f"Black_HV3_Use_{i}_{j}"
                
                # If this corner is used, check for available extensions
                if j+1 < self.n_c:  # Can extend right
                    self.model += self.h[i, j+1] >= hv_corner, f"Black_HV3_ExtR_{i}_{j}"
                if i > 1:  # Can extend up
                    self.model += self.v[i-2, j] >= hv_corner, f"Black_HV3_ExtU_{i}_{j}"
            
            if h_right[0] and v_down[0]:
                # If both segments used, extensions must exist
                hv_corner = pl.LpVariable(f"hv4_{i}_{j}", 0, 1, pl.LpBinary)
                self.model += h_right[1] + v_down[1] - 1 <= hv_corner, f"Black_HV4_Use_{i}_{j}"
                
                # If this corner is used, check for available extensions
                if j+1 < self.n_c:  # Can extend right
                    self.model += self.h[i, j+1] >= hv_corner, f"Black_HV4_ExtR_{i}_{j}"
                if i+1 < self.n_r:  # Can extend down
                    self.model += self.v[i+1, j] >= hv_corner, f"Black_HV4_ExtD_{i}_{j}"
    
    def _add_white_circle_constraints(self):
        """Add constraints for white circles."""
        print(f"Number of white circles: {len(self.W)}")
        for i, j in self.W:
            # White circle requires a straight passage with turn immediately after on at least one end
            print(f"Adding strict constraints for white circle at ({i}, {j})")
            
            # Identify possible straight passages through this point
            h_passage = (i < self.n_r+1 and j > 0 and j < self.n_c, 
                        (self.h[i, j-1], self.h[i, j]) if i < self.n_r+1 and j > 0 and j < self.n_c else (None, None))
            
            v_passage = (i > 0 and i < self.n_r and j < self.n_c+1,
                        (self.v[i-1, j], self.v[i, j]) if i > 0 and i < self.n_r and j < self.n_c+1 else (None, None))
            
            # Create binary variables for the two possible passage types
            horiz = pl.LpVariable(f"w_horiz_{i}_{j}", 0, 1, pl.LpBinary)
            vert = pl.LpVariable(f"w_vert_{i}_{j}", 0, 1, pl.LpBinary)
            
            # The point must be on the fence
            self.model += horiz + vert == self.p[i, j], f"White_OnFence_{i}_{j}"
            
            # Set up constraints for horizontal passage
            if h_passage[0]:
                # Both horizontal segments must be used if horiz is 1
                self.model += h_passage[1][0] + h_passage[1][1] >= 2 * horiz, f"White_H_Pass1_{i}_{j}"
                self.model += h_passage[1][0] <= horiz, f"White_H_Pass2_{i}_{j}"
                self.model += h_passage[1][1] <= horiz, f"White_H_Pass3_{i}_{j}"
                
                # Check for possible turns at left end (j-1)
                left_turns = []
                if j-1 >= 0:
                    if i > 0:  # Can turn up
                        left_turns.append(self.v[i-1, j-1])
                    if i < self.n_r:  # Can turn down
                        left_turns.append(self.v[i, j-1])
                
                # Check for turns at right end (j+1)
                right_turns = []
                if j+1 <= self.n_c:
                    if i > 0:  # Can turn up
                        right_turns.append(self.v[i-1, j+1])
                    if i < self.n_r:  # Can turn down
                        right_turns.append(self.v[i, j+1])
                
                # At least one turn must exist on either end if horizontal passage is used
                if left_turns or right_turns:
                    turns = left_turns + right_turns
                    if turns:
                        # If horizontal passage is used, at least one turn must exist
                        self.model += pl.lpSum(turns) >= horiz, f"White_H_Turns_{i}_{j}"
                else:
                    # No turns possible for horizontal passage - can't use this orientation
                    self.model += horiz == 0, f"White_H_Disabled_{i}_{j}"
            else:
                # Horizontal passage not available
                self.model += horiz == 0, f"White_H_NotAvail_{i}_{j}"
            
            # Set up constraints for vertical passage
            if v_passage[0]:
                # Both vertical segments must be used if vert is 1
                self.model += v_passage[1][0] + v_passage[1][1] >= 2 * vert, f"White_V_Pass1_{i}_{j}"
                self.model += v_passage[1][0] <= vert, f"White_V_Pass2_{i}_{j}"
                self.model += v_passage[1][1] <= vert, f"White_V_Pass3_{i}_{j}"
                
                # Check for turns at top end (i-1)
                top_turns = []
                if i-1 >= 0:
                    if j > 0:  # Can turn left
                        top_turns.append(self.h[i-1, j-1])
                    if j < self.n_c:  # Can turn right
                        top_turns.append(self.h[i-1, j])
                
                # Check for turns at bottom end (i+1)
                bottom_turns = []
                if i+1 <= self.n_r:
                    if j > 0:  # Can turn left
                        bottom_turns.append(self.h[i+1, j-1])
                    if j < self.n_c:  # Can turn right
                        bottom_turns.append(self.h[i+1, j])
                
                # At least one turn must exist on either end if vertical passage is used
                if top_turns or bottom_turns:
                    turns = top_turns + bottom_turns
                    if turns:
                        # If vertical passage is used, at least one turn must exist
                        self.model += pl.lpSum(turns) >= vert, f"White_V_Turns_{i}_{j}"
                else:
                    # No turns possible for vertical passage - can't use this orientation
                    self.model += vert == 0, f"White_V_Disabled_{i}_{j}"
            else:
                # Vertical passage not available
                self.model += vert == 0, f"White_V_NotAvail_{i}_{j}"
    
    def _find_loops(self, h_sol, v_sol):
        """Find all loops in the current solution."""
        # Create a graph from the solution
        graph = {}
        for i in range(self.n_r+1):
            for j in range(self.n_c+1):
                graph[(i, j)] = []
        
        # Add horizontal edges
        for i in range(self.n_r+1):
            for j in range(self.n_c):
                if h_sol.get((i, j), 0) > 0.5:
                    graph[(i, j)].append((i, j+1))
                    graph[(i, j+1)].append((i, j))
        
        # Add vertical edges
        for i in range(self.n_r):
            for j in range(self.n_c+1):
                if v_sol.get((i, j), 0) > 0.5:
                    graph[(i, j)].append((i+1, j))
                    graph[(i+1, j)].append((i, j))
        
        # Find all loops using DFS
        visited = set()
        loops = []
        
        # Find all nodes that are part of the graph (have edges)
        active_nodes = []
        for node, neighbors in graph.items():
            if neighbors:
                active_nodes.append(node)
        
        # For each unvisited node with edges, try to find a loop
        for start_node in active_nodes:
            if start_node in visited:
                continue
            
            # Start DFS from this node
            stack = [(start_node, None)]  # (node, parent)
            path = {}  # Maps node to its parent in the DFS tree
            
            while stack:
                node, parent = stack.pop()
                
                if node in path:
                    # We found a cycle
                    # Reconstruct the cycle
                    cycle = []
                    
                    # From the current node back to where it was first visited
                    current = parent
                    while current != node and current is not None:
                        cycle.append(current)
                        current = path.get(current)
                        if current is None:
                            # This shouldn't happen in a valid cycle
                            break
                    
                    # Only add valid cycles
                    if current == node and len(cycle) >= 3:
                        cycle.append(node)
                        cycle.reverse()  # Get the correct order
                        loops.append(cycle)
                    
                    # Mark all nodes in the cycle as visited
                    for n in cycle:
                        visited.add(n)
                    
                    continue
                
                # Mark node as visited in the current DFS path
                path[node] = parent
                
                # Explore neighbors
                for neighbor in graph[node]:
                    if neighbor != parent:  # Avoid going back to parent
                        stack.append((neighbor, node))
        
        return loops
    
    def visualize_solution(self, output_path=None):
        """Visualize the solution."""
        if self.solution is None:
            print("No solution to visualize.")
            return
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw the grid
        for i in range(self.n_r+1):
            ax.axhline(i, color='gray', linestyle=':')
        for j in range(self.n_c+1):
            ax.axvline(j, color='gray', linestyle=':')
        
        # Get inside/outside status for each cell
        inside_status = {}
        for i in range(self.n_r):
            for j in range(self.n_c):
                inside_status[(i, j)] = pl.value(self.inside.get((i, j), 0)) > 0.5
        
        # Color regions - Use distinct colors to clearly identify inside vs outside
        for i in range(self.n_r):
            for j in range(self.n_c):
                if inside_status.get((i, j), False):
                    # Inside (value=1) - BLUE color
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='lightblue', alpha=0.4))
                else:
                    # Outside (value=0) - YELLOW color
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='lightyellow', alpha=0.3))
        
        # Draw the fence
        h_sol = self.solution["h"]
        v_sol = self.solution["v"]
        
        # Horizontal segments
        for (i, j), val in h_sol.items():
            if val > 0.5:
                ax.plot([j, j+1], [i, i], 'k-', linewidth=3)
        
        # Vertical segments
        for (i, j), val in v_sol.items():
            if val > 0.5:
                ax.plot([j, j], [i, i+1], 'k-', linewidth=3)
        
        # Draw special cells
        for i in range(self.n_r):
            for j in range(self.n_c):
                cell = self.matrix_1[i][j]
                if cell == "A":  # Alien
                    is_inside = inside_status.get((i, j), False)
                    status_text = "IN" if is_inside else "OUT"
                    ax.text(j+0.5, i+0.5, f"👽\n{status_text}", fontsize=12, ha='center', va='center')
                    # Add highlight for aliens
                    ax.add_patch(plt.Circle((j+0.5, i+0.5), 0.4, fill=False, edgecolor='blue', linestyle='--', linewidth=1.5))
                elif cell == "C":  # Triffid (Cactus)
                    is_inside = inside_status.get((i, j), False)
                    status_text = "IN" if is_inside else "OUT"
                    ax.text(j+0.5, i+0.5, f"🌵\n{status_text}", fontsize=12, ha='center', va='center')
                    # Add highlight for triffids
                    ax.add_patch(plt.Circle((j+0.5, i+0.5), 0.4, fill=False, edgecolor='green', linestyle='--', linewidth=1.5))
                elif isinstance(cell, dict) and cell.get("circled"):  # Circled number
                    is_inside = inside_status.get((i, j), False)
                    status_text = "IN" if is_inside else "OUT"
                    ax.add_patch(plt.Circle((j+0.5, i+0.5), 0.3, fill=False, edgecolor='red', linewidth=1.5))
                    ax.text(j+0.5, i+0.5, f"{cell['value']}\n{status_text}", fontsize=10, ha='center', va='center')
                elif isinstance(cell, int):  # Uncircled number
                    ax.text(j+0.5, i+0.5, str(cell), fontsize=10, ha='center', va='center')
        
        # Draw special points with enhanced visibility
        for i in range(len(self.matrix_2)):
            for j in range(len(self.matrix_2[0])):
                point = self.matrix_2[i][j]
                if point == "E":  # White circle
                    ax.add_patch(plt.Circle((j, i), 0.2, facecolor='white', edgecolor='black', linewidth=2, zorder=10))
                elif point == "F":  # Black circle
                    ax.add_patch(plt.Circle((j, i), 0.2, facecolor='black', edgecolor='white', linewidth=1, zorder=10))
        
        # Add legend
        ax.add_patch(plt.Rectangle((0, -0.5), 0.3, 0.3, facecolor='lightblue', alpha=0.4))
        ax.text(0.4, -0.35, "Inside fence", fontsize=10, va='center')
        ax.add_patch(plt.Rectangle((2, -0.5), 0.3, 0.3, facecolor='lightyellow', alpha=0.3))
        ax.text(2.4, -0.35, "Outside fence", fontsize=10, va='center')
        
        # Set limits and invert y-axis
        ax.set_xlim(-0.5, self.n_c+0.5)
        ax.set_ylim(self.n_r+0.5, -0.8)  # Extended to show legend
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title with puzzle info
        ax.set_title(f"Area51 Puzzle: {os.path.basename(self.puzzle_path)}\nAliens must be INSIDE (blue), Triffids must be OUTSIDE (yellow)", 
                    fontsize=12, pad=10)
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

def solve_area51_puzzle(puzzle_path):
    """Solve an Area51 puzzle from a JSON file."""
    solver = Area51Solver(puzzle_path)
    if solver.solve():
        return solver
    return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        puzzle_path = sys.argv[1]
        solver = solve_area51_puzzle(puzzle_path)
        if solver:
            solver.visualize_solution()
    else:
        print("Usage: python area51.py <puzzle_json_path>")
