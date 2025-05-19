import json
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, LpInteger, lpSum, value, PULP_CBC_CMD, LpStatusOptimal, LpStatusInfeasible

class Area51Solver:
    def __init__(self, puzzle_input):
        if isinstance(puzzle_input, tuple):
            # Initialize with grid size
            self.rows, self.cols = puzzle_input
            self.matrix_1 = [[None] * self.cols for _ in range(self.rows)]
            self.matrix_2 = [[None] * (self.cols + 1) for _ in range(self.rows + 1)]
        else:
            # Initialize from file
            self.load_puzzle(puzzle_input)
            self.rows = len(self.matrix_1)
            self.cols = len(self.matrix_1[0])
        self.model = LpProblem("Area51", LpMinimize)
        self.h_fence = {}
        self.v_fence = {}
        self.inside = {}
        
    def load_puzzle(self, puzzle_file):
        try:
            with open(puzzle_file, 'r') as f:
                data = json.load(f)
                if 'matrix_1' not in data or 'matrix_2' not in data:
                    raise ValueError("Puzzle file must contain matrix_1 and matrix_2")
                self.matrix_1 = data['matrix_1']
                self.matrix_2 = data['matrix_2']
        except Exception as e:
            print(f"Error loading puzzle file: {e}")
            raise

    def create_variables(self):
        # Create binary variables for fence segments
        for i in range(self.rows + 1):
            for j in range(self.cols):
                self.h_fence[i, j] = LpVariable(f'h_{i}_{j}', cat=LpBinary)
        
        for i in range(self.rows):
            for j in range(self.cols + 1):
                self.v_fence[i, j] = LpVariable(f'v_{i}_{j}', cat=LpBinary)

        # Create binary variables for cell containment (inside/outside fence)
        for i in range(self.rows):
            for j in range(self.cols):
                self.inside[i, j] = LpVariable(f'in_{i}_{j}', cat=LpBinary)

    def add_constraints(self):
        """Add all constraints to the model"""
        self.add_fence_constraints()
        self.add_inside_outside_constraints()
        self.add_special_constraints()
        self.add_masyu_constraints()

    def add_fence_constraints(self):
        """Add constraints for a valid fence"""
        
        # Fence must form a loop (even number of fence segments at each vertex)
        for i in range(self.rows + 1):
            for j in range(self.cols + 1):
                edges = []
                
                # Check all four possible edges incident to this vertex
                if j > 0:  # Left horizontal edge
                    edges.append(self.h_fence[i, j-1])
                if j < self.cols:  # Right horizontal edge
                    edges.append(self.h_fence[i, j])
                if i > 0:  # Top vertical edge
                    edges.append(self.v_fence[i-1, j])
                if i < self.rows:  # Bottom vertical edge
                    edges.append(self.v_fence[i, j])
                
                # Each vertex must have 0 or 2 incident fence segments (mod 2 = 0)
                # PuLP version: we need an auxiliary variable for mod constraint
                k = LpVariable(f'k_{i}_{j}', cat=LpInteger, lowBound=0, upBound=2)
                self.model += lpSum(edges) == 2 * k, f'vertex_{i}_{j}'
        
        # Make sure there's at least one fence segment (to avoid trivial solution)
        total_fence = lpSum([self.h_fence[i, j] for i in range(self.rows + 1) for j in range(self.cols)]) + \
                     lpSum([self.v_fence[i, j] for i in range(self.rows) for j in range(self.cols + 1)])
        self.model += total_fence >= 4, "min_fence"  # At least 4 fence segments
        
        # Objective: Minimize the total fence length 
        self.model += total_fence, "objective"

    def add_inside_outside_constraints(self):
        """Add constraints for inside/outside determination"""
        # Ray shooting to determine inside/outside
        for i in range(self.rows):
            for j in range(self.cols):
                # Ray shooting to the right (East)
                # Count vertical fence segments to the right
                crossings = []
                for k in range(j + 1, self.cols + 1):
                    crossings.append(self.v_fence[i, k])
                
                # Cell is inside if number of crossings is odd
                # Create an integer variable to represent (sum_crossings - inside[i,j]) / 2
                k = LpVariable(f'k_inside_{i}_{j}', cat=LpInteger, lowBound=0)
                self.model += lpSum(crossings) == 2 * k + self.inside[i, j], f'inside_{i}_{j}'
        
        # At least one cell must be inside
        self.model += lpSum(self.inside[i, j] for i in range(self.rows) for j in range(self.cols)) >= 1, "min_inside"
        
        # There must be some cells outside as well
        outside_sum = lpSum(1 - self.inside[i, j] for i in range(self.rows) for j in range(self.cols))
        self.model += outside_sum >= 1, "min_outside"
        
        # Add inside/outside relationship to fence
        for i in range(self.rows):
            for j in range(self.cols):
                # Horizontal fence below this cell
                if i < self.rows - 1:
                    # Two cells with the same inside/outside status cannot have a fence between them
                    self.model += self.h_fence[i+1, j] <= self.inside[i, j] + self.inside[i+1, j], f'h_fence_same1_{i}_{j}'
                    self.model += self.h_fence[i+1, j] <= 2 - self.inside[i, j] - self.inside[i+1, j], f'h_fence_same2_{i}_{j}'
                    
                    # If cells have different inside/outside status, must have a fence between them
                    self.model += self.h_fence[i+1, j] >= self.inside[i, j] - self.inside[i+1, j], f'h_fence_diff1_{i}_{j}'
                    self.model += self.h_fence[i+1, j] >= self.inside[i+1, j] - self.inside[i, j], f'h_fence_diff2_{i}_{j}'
                
                # Vertical fence to the right
                if j < self.cols - 1:
                    # Two cells with the same inside/outside status cannot have a fence between them
                    self.model += self.v_fence[i, j+1] <= self.inside[i, j] + self.inside[i, j+1], f'v_fence_same1_{i}_{j}'
                    self.model += self.v_fence[i, j+1] <= 2 - self.inside[i, j] - self.inside[i, j+1], f'v_fence_same2_{i}_{j}'
                    
                    # If cells have different inside/outside status, must have a fence between them
                    self.model += self.v_fence[i, j+1] >= self.inside[i, j] - self.inside[i, j+1], f'v_fence_diff1_{i}_{j}'
                    self.model += self.v_fence[i, j+1] >= self.inside[i, j+1] - self.inside[i, j], f'v_fence_diff2_{i}_{j}'

    def add_special_constraints(self):
        """Add constraints for special cells in the puzzle"""
        
        # Process matrix_1 constraints (cell-based)
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.matrix_1[i][j]
                
                if cell is None:
                    continue
                    
                # Process different special cell types
                if cell == "A":  # Alien (must be inside)
                    self.model += self.inside[i, j] == 1, f'alien_{i}_{j}'
                    
                elif cell == "C":  # Cactus (must be outside)
                    self.model += self.inside[i, j] == 0, f'cactus_{i}_{j}'
                    
                elif isinstance(cell, int) or (isinstance(cell, dict) and "value" in cell):
                    # Number indicates fence segments around the cell
                    value = cell if isinstance(cell, int) else cell["value"]
                    
                    # Count the fence segments around this cell
                    fence_segments = []
                    
                    # Top edge
                    fence_segments.append(self.h_fence[i, j])
                    
                    # Right edge
                    fence_segments.append(self.v_fence[i, j+1])
                    
                    # Bottom edge
                    fence_segments.append(self.h_fence[i+1, j])
                    
                    # Left edge
                    fence_segments.append(self.v_fence[i, j])
                    
                    # Add constraint for the exact number of fence segments
                    self.model += lpSum(fence_segments) == value, f'numbered_{i}_{j}'

    def add_masyu_constraints(self):
        """Add constraints specific to Masyu puzzles"""
        
        # Process matrix_2 (vertex-based) constraints for Masyu
        for i in range(self.rows + 1):
            for j in range(self.cols + 1):
                vertex = self.matrix_2[i][j] if i < len(self.matrix_2) and j < len(self.matrix_2[i]) else None
                
                if vertex is None:
                    continue
                
                # Black circle: Must have a straight line passing through
                if vertex == "F":
                    # Create auxiliary variables for each possible orientation
                    horiz = LpVariable(f'black_horiz_{i}_{j}', cat=LpBinary)
                    vert = LpVariable(f'black_vert_{i}_{j}', cat=LpBinary)
                    
                    # Must have exactly one orientation
                    self.model += horiz + vert == 1, f'black_orientation_{i}_{j}'
                    
                    # Check if horizontal orientation is possible
                    if j > 0 and j < self.cols:
                        # If horizontal orientation is chosen, both left and right must be fence
                        self.model += self.h_fence[i, j-1] >= horiz, f'black_left_{i}_{j}'
                        self.model += self.h_fence[i, j] >= horiz, f'black_right_{i}_{j}'
                        
                        # If both left and right are fence, then horizontal orientation must be chosen
                        self.model += horiz >= self.h_fence[i, j-1] + self.h_fence[i, j] - 1, f'black_horiz_force_{i}_{j}'
                    else:
                        # If not possible horizontally, force vertical
                        self.model += horiz == 0, f'black_horiz_impossible_{i}_{j}'
                    
                    # Check if vertical orientation is possible
                    if i > 0 and i < self.rows:
                        # If vertical orientation is chosen, both up and down must be fence
                        self.model += self.v_fence[i-1, j] >= vert, f'black_up_{i}_{j}'
                        self.model += self.v_fence[i, j] >= vert, f'black_down_{i}_{j}'
                        
                        # If both up and down are fence, then vertical orientation must be chosen
                        self.model += vert >= self.v_fence[i-1, j] + self.v_fence[i, j] - 1, f'black_vert_force_{i}_{j}'
                    else:
                        # If not possible vertically, force horizontal
                        self.model += vert == 0, f'black_vert_impossible_{i}_{j}'
                
                # White circle: Must have a turn at this point
                elif vertex == "E":
                    # Count the potential fence segments in each direction
                    left = self.h_fence[i, j-1] if j > 0 else 0
                    right = self.h_fence[i, j] if j < self.cols else 0
                    up = self.v_fence[i-1, j] if i > 0 else 0 
                    down = self.v_fence[i, j] if i < self.rows else 0
                    
                    # Create variables for each possible turn configuration
                    turn_left_up = LpVariable(f'white_turn_left_up_{i}_{j}', cat=LpBinary)
                    turn_left_down = LpVariable(f'white_turn_left_down_{i}_{j}', cat=LpBinary)
                    turn_right_up = LpVariable(f'white_turn_right_up_{i}_{j}', cat=LpBinary)
                    turn_right_down = LpVariable(f'white_turn_right_down_{i}_{j}', cat=LpBinary)
                    
                    # Must have exactly one turn configuration
                    self.model += turn_left_up + turn_left_down + turn_right_up + turn_right_down == 1, f'white_turn_config_{i}_{j}'
                    
                    # For each configuration, enforce the corresponding fence segments
                    
                    # Left-Up turn
                    if j > 0 and i > 0:
                        self.model += left >= turn_left_up, f'white_left_up_1_{i}_{j}'
                        self.model += up >= turn_left_up, f'white_left_up_2_{i}_{j}'
                        # Force this configuration if both left and up are fence
                        self.model += turn_left_up >= left + up - 1, f'white_left_up_force_{i}_{j}'
                    else:
                        self.model += turn_left_up == 0, f'white_left_up_impossible_{i}_{j}'
                    
                    # Left-Down turn
                    if j > 0 and i < self.rows:
                        self.model += left >= turn_left_down, f'white_left_down_1_{i}_{j}'
                        self.model += down >= turn_left_down, f'white_left_down_2_{i}_{j}'
                        # Force this configuration if both left and down are fence
                        self.model += turn_left_down >= left + down - 1, f'white_left_down_force_{i}_{j}'
                    else:
                        self.model += turn_left_down == 0, f'white_left_down_impossible_{i}_{j}'
                    
                    # Right-Up turn
                    if j < self.cols and i > 0:
                        self.model += right >= turn_right_up, f'white_right_up_1_{i}_{j}'
                        self.model += up >= turn_right_up, f'white_right_up_2_{i}_{j}'
                        # Force this configuration if both right and up are fence
                        self.model += turn_right_up >= right + up - 1, f'white_right_up_force_{i}_{j}'
                    else:
                        self.model += turn_right_up == 0, f'white_right_up_impossible_{i}_{j}'
                    
                    # Right-Down turn
                    if j < self.cols and i < self.rows:
                        self.model += right >= turn_right_down, f'white_right_down_1_{i}_{j}'
                        self.model += down >= turn_right_down, f'white_right_down_2_{i}_{j}'
                        # Force this configuration if both right and down are fence
                        self.model += turn_right_down >= right + down - 1, f'white_right_down_force_{i}_{j}'
                    else:
                        self.model += turn_right_down == 0, f'white_right_down_impossible_{i}_{j}'
                    
                    # Prevent straight lines (no horizontal or vertical straight lines)
                    if j > 0 and j < self.cols:
                        self.model += left + right <= 1, f'white_no_horiz_line_{i}_{j}'
                    
                    if i > 0 and i < self.rows:
                        self.model += up + down <= 1, f'white_no_vert_line_{i}_{j}'
                    
                    # Total fence segments at this vertex must be exactly 2
                    fence_segments = []
                    if j > 0:
                        fence_segments.append(left)
                    if j < self.cols:
                        fence_segments.append(right)
                    if i > 0:
                        fence_segments.append(up)
                    if i < self.rows:
                        fence_segments.append(down)
                    
                    self.model += lpSum(fence_segments) == 2, f'white_count_{i}_{j}'

    def solve(self, time_limit=None, output_file=None):
        """Solve the Area51 puzzle using PuLP
        
        Args:
            time_limit: Maximum time to spend solving (in seconds)
            output_file: Optional file to save the solution to
            
        Returns:
            A dictionary with the solution status and solution (if found)
        """
        try:
            # Create variables and constraints if not already done
            if not self.h_fence:
                self.create_variables()
                self.add_constraints()
            
            # Set time limit if specified
            if time_limit is not None:
                # PuLP doesn't have a direct time limit parameter, 
                # so we'll specify it for the solver
                solver = PULP_CBC_CMD(timeLimit=time_limit)
                status = self.model.solve(solver)
            else:
                status = self.model.solve()
            
            # Check solution status
            solution = {
                "status": self.model.status
            }
            
            # PuLP status codes: 1 = Optimal, 0 = Not Solved, -1 = Infeasible
            if self.model.status == LpStatusOptimal:  # Optimal solution found
                solution["status_str"] = "OPTIMAL"
                solution["objective"] = value(self.model.objective)
                
                # Extract fence solution
                h_fence_sol = {}
                for (i, j), var in self.h_fence.items():
                    if value(var) > 0.5:
                        h_fence_sol[f"{i},{j}"] = 1
                
                v_fence_sol = {}
                for (i, j), var in self.v_fence.items():
                    if value(var) > 0.5:
                        v_fence_sol[f"{i},{j}"] = 1
                
                # Extract inside/outside regions
                inside_sol = {}
                for (i, j), var in self.inside.items():
                    if value(var) > 0.5:
                        inside_sol[f"{i},{j}"] = 1
                
                solution["h_fence"] = h_fence_sol
                solution["v_fence"] = v_fence_sol
                solution["inside"] = inside_sol
                
                # Save solution to file if requested
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(solution, f, indent=2)
                
                return solution
            
            elif self.model.status == LpStatusInfeasible:  # Infeasible
                solution["status_str"] = "INFEASIBLE"
                print("Model is infeasible")
                return solution
            
            else:
                solution["status_str"] = f"OTHER ({self.model.status})"
                print(f"Solver status: {self.model.status}")
                return solution
                
        except Exception as e:
            print(f"Error solving model: {e}")
            return {"status": "ERROR", "message": str(e)}

    def display_solution(self):
        """Display the solution in a readable format"""
        if self.model.status != LpStatusOptimal:  # Not optimal
            print("No solution to display!")
            return
        
        # Extract solution
        h_fence = {}
        v_fence = {}
        inside = {}
        
        for (i, j), var in self.h_fence.items():
            if value(var) > 0.5:
                h_fence[i, j] = 1
        
        for (i, j), var in self.v_fence.items():
            if value(var) > 0.5:
                v_fence[i, j] = 1
        
        for (i, j), var in self.inside.items():
            if value(var) > 0.5:
                inside[i, j] = 1
        
        # Display the grid
        print("\nSolution:")
        print("-" * (self.cols * 4 + 2))
        
        for i in range(self.rows + 1):
            # Print horizontal fences
            line = "|"
            for j in range(self.cols):
                if (i, j) in h_fence:
                    line += "---+"
                else:
                    line += "   +"
            print(line)
            
            # Print vertical fences and inside/outside status
            if i < self.rows:
                line = "|"
                for j in range(self.cols):
                    cell = " "
                    # Special cell content
                    if self.matrix_1[i][j] is not None:
                        if isinstance(self.matrix_1[i][j], dict) and self.matrix_1[i][j].get("circled"):
                            cell = str(self.matrix_1[i][j]["value"]) + "○"
                        elif self.matrix_1[i][j] == "A":
                            cell = "A"
                        elif self.matrix_1[i][j] == "C":
                            cell = "C"
                        elif isinstance(self.matrix_1[i][j], (int, float)):
                            cell = str(self.matrix_1[i][j])
                    else:
                        # Inside/outside status
                        if (i, j) in inside:
                            cell = "■"  # Inside
                        else:
                            cell = "□"  # Outside
                    
                    if (i, j+1) in v_fence:
                        line += f" {cell} |"
                    else:
                        line += f" {cell}  "
                print(line)
        
        print("-" * (self.cols * 4 + 2)) 