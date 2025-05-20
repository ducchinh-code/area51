import json
import numpy as np
import pulp as pl
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager
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
            
            # Validate rectangularity (all rows have equal length)
            if any(len(row) != self.n_c for row in self.matrix_1):
                raise ValueError("matrix_1 rows must have equal length")
            if any(len(row) != self.n_c + 1 for row in self.matrix_2):
                raise ValueError("matrix_2 rows must have equal length")
            
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
            self.W = []  # Set of white dots
            self.B = []  # Set of black dots
            
            # Process matrix_1 (cell data)
            for i in range(self.n_r):
                for j in range(self.n_c):
                    cell = self.matrix_1[i][j]
                    if cell == "A":
                        self.A.append((i, j))  # Now using 0-indexed coordinates
                        print(f"Found alien at ({i},{j})")
                    elif cell == "C":
                        self.T.append((i, j))
                        print(f"Found triffid/cactus at ({i},{j})")
                    elif isinstance(cell, dict) and cell.get("circled"):
                        if "value" not in cell:
                            print(f"Warning: Circled number at ({i},{j}) missing 'value' key")
                            continue
                        self.K.append((i, j))
                        print(f"Found circled number {cell['value']} at ({i},{j})")
                        
                        # Check if circled number might be infeasible
                        if cell["value"] > self.n_r + self.n_c + 1:
                            print(f"Warning: Circled number {cell['value']} at ({i},{j}) may be infeasible in {self.n_r}x{self.n_c} grid")
                    elif isinstance(cell, int):
                        # Validate uncircled numbers (must be 0-3)
                        if cell < 0 or cell > 3:
                            raise ValueError(f"Uncircled number at ({i},{j}) has invalid value {cell}; must be 0-3")
                        self.F.append((i, j))
                        print(f"Found uncircled number {cell} at ({i},{j})")
            
            # Process matrix_2 (point data)
            for i in range(len(self.matrix_2)):
                for j in range(len(self.matrix_2[0])):
                    point = self.matrix_2[i][j]
                    if point == "E":  # White dots
                        self.W.append((i, j))
                        print(f"Found white dots at ({i},{j})")
                        
                        # Check if white dot is at a boundary or corner where constraints may be infeasible
                        if i == 0 or i == self.n_r or j == 0 or j == self.n_c:
                            print(f"Warning: White dot at boundary position ({i},{j}) may have infeasible constraints")
                            if (i == 0 and j == 0) or (i == 0 and j == self.n_c) or (i == self.n_r and j == 0) or (i == self.n_r and j == self.n_c):
                                print(f"Warning: White dot at corner position ({i},{j}) may have severely limited constraints")
                    
                    elif point == "F":  # Black dots
                        self.B.append((i, j))
                        print(f"Found black dots at ({i},{j})")
                        
                        # Check if black dot is at a boundary or corner where constraints may be infeasible
                        if i == 0 or i == self.n_r or j == 0 or j == self.n_c:
                            print(f"Warning: Black dot at boundary position ({i},{j}) may have infeasible constraints")
                            if (i == 0 and j == 0) or (i == 0 and j == self.n_c) or (i == self.n_r and j == 0) or (i == self.n_r and j == self.n_c):
                                print(f"Warning: Black dot at corner position ({i},{j}) may have severely limited constraints")
            
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
            
            # Check if the solution has exactly one cycle
            has_single_cycle = self._find_loops(h_sol, v_sol)
            
            print(f"Single cycle check in iteration {iterations+1}: {has_single_cycle}")
            
            if has_single_cycle:
                # We have found a valid solution with a single loop
                print(f"Solution found after {iterations+1} iterations.")
                self.solution = {"h": h_sol, "v": v_sol}
                return True
            
            # Add a constraint to eliminate the current solution
            # We need to ensure that at least one edge variable gets flipped
            active_edges = []
            
            # Collect active horizontal edges
            for i in range(self.n_r+1):
                for j in range(self.n_c):
                    if h_sol.get((i, j), 0) > 0.5:
                        active_edges.append(self.h[i, j])
            
            # Collect active vertical edges
            for i in range(self.n_r):
                for j in range(self.n_c+1):
                    if v_sol.get((i, j), 0) > 0.5:
                        active_edges.append(self.v[i, j])
            
            # The sum of active edges in the next solution must be less than the total number
            # of active edges in the current solution
            if active_edges:
                self.model += (pl.lpSum(active_edges) <= len(active_edges) - 1)
            
            iterations += 1
        
        print(f"Maximum iterations ({max_iterations}) reached without finding a valid solution.")
        return False
    
    def _init_variables(self):
        """Initialize decision variables."""
        print("Initializing decision variables...")
        
        # Horizontal fence segments: h[i,j] = 1 if there is a fence segment at row i, between columns j and j+1
        self.h = {}
        for i in range(self.n_r+1):
            for j in range(self.n_c):
                self.h[i, j] = pl.LpVariable(f"h_{i}_{j}", 0, 1, pl.LpBinary)
        
        # Vertical fence segments: v[i,j] = 1 if there is a fence segment at column j, between rows i and i+1
        self.v = {}
        for i in range(self.n_r):
            for j in range(self.n_c+1):
                self.v[i, j] = pl.LpVariable(f"v_{i}_{j}", 0, 1, pl.LpBinary)
        
        # Points on the fence: p[i,j] = 1 if the grid point at (i,j) is on the fence
        self.p = {}
        for i in range(self.n_r+1):
            for j in range(self.n_c+1):
                self.p[i, j] = pl.LpVariable(f"p_{i}_{j}", 0, 1, pl.LpBinary)
        
        # Inside/outside status for each cell: inside[i,j] = 1 if cell (i,j) is inside the fence
        self.inside = {}
        for i in range(self.n_r):
            for j in range(self.n_c):
                self.inside[i, j] = pl.LpVariable(f"inside_{i}_{j}", 0, 1, pl.LpBinary)
        
        print(f"Created {len(self.h)} horizontal segments, {len(self.v)} vertical segments, {len(self.p)} points, and {len(self.inside)} inside/outside variables")
    
    def _add_constraints(self):
        """Add all constraints to the model."""
        # Dummy objective: maximize 0 (constraint satisfaction only)
        self.model += 0, "Dummy_Objective"
        
        # Step 1: Add flow-based constraints to determine inside/outside
        print("Adding inside/outside constraints...")
        
        # Choose a source cell that must be inside (use first circled number or alien)
        source = None
        if self.K:
            source = self.K[0]
        elif self.A:
            source = self.A[0]
        else:
            source = (0, 0)  # Fallback to top-left cell
        
        print(f"Using source cell {source} for inside/outside determination")
        
        # Define flow variables between cells
        flow = {}
        for i1 in range(self.n_r):
            for j1 in range(self.n_c):
                for i2 in range(self.n_r):
                    for j2 in range(self.n_c):
                        if (i1, j1) != (i2, j2) and (abs(i1 - i2) + abs(j1 - j2) == 1):  # Adjacent cells
                            flow[(i1, j1), (i2, j2)] = pl.LpVariable(f"flow_{i1}_{j1}_{i2}_{j2}", 0, 1, pl.LpBinary)
        
        # Flow can only pass between cells if there is no fence between them
        for i1 in range(self.n_r):
            for j1 in range(self.n_c):
                # Right neighbor
                if j1 + 1 < self.n_c:
                    i2, j2 = i1, j1 + 1
                    self.model += flow[(i1, j1), (i2, j2)] <= 1 - self.v[i1, j1 + 1], f"NoFenceFlow_R_{i1}_{j1}_{i2}_{j2}"
                    self.model += flow[(i2, j2), (i1, j1)] <= 1 - self.v[i1, j1 + 1], f"NoFenceFlow_L_{i2}_{j2}_{i1}_{j1}"
                # Down neighbor
                if i1 + 1 < self.n_r:
                    i2, j2 = i1 + 1, j1
                    self.model += flow[(i1, j1), (i2, j2)] <= 1 - self.h[i1 + 1, j1], f"NoFenceFlow_D_{i1}_{j1}_{i2}_{j2}"
                    self.model += flow[(i2, j2), (i1, j1)] <= 1 - self.h[i1 + 1, j1], f"NoFenceFlow_U_{i2}_{j2}_{i1}_{j1}"
        
        # Flow conservation: incoming flow = outgoing flow (except source)
        for i in range(self.n_r):
            for j in range(self.n_c):
                if (i, j) == source:
                    continue
                incoming = []
                outgoing = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.n_r and 0 <= nj < self.n_c:
                        incoming.append(flow[(ni, nj), (i, j)])
                        outgoing.append(flow[(i, j), (ni, nj)])
                self.model += pl.lpSum(incoming) == pl.lpSum(outgoing), f"FlowConserve_{i}_{j}"
        
        # Source cell has outgoing flow to at least one neighbor
        source_outgoing = []
        si, sj = source
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = si + di, sj + dj
            if 0 <= ni < self.n_r and 0 <= nj < self.n_c:
                source_outgoing.append(flow[(si, sj), (ni, nj)])
        self.model += pl.lpSum(source_outgoing) >= 1, f"SourceFlow_{si}_{sj}"
        
        # Cells reachable by flow are inside
        for i in range(self.n_r):
            for j in range(self.n_c):
                if (i, j) == source:
                    self.model += self.inside[i, j] == 1, f"SourceInside_{i}_{j}"
                    continue
                incoming = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.n_r and 0 <= nj < self.n_c:
                        incoming.append(flow[(ni, nj), (i, j)])
                self.model += self.inside[i, j] >= pl.lpSum(incoming) / 4, f"InsideIfFlow_{i}_{j}"
        
        # Enforce inside/outside constraints
        for i, j in self.A:
            self.model += self.inside[i, j] == 1, f"AlienInside_{i}_{j}"
        for i, j in self.T:
            self.model += self.inside[i, j] == 0, f"TriffidOutside_{i}_{j}"
        for i, j in self.K:
            self.model += self.inside[i, j] == 1, f"CircledInside_{i}_{j}"
        
        # Add other constraints
        print("Adding cycle constraints...")
        self._add_cycle_constraints()
        
        # Check feasibility after cycle constraints
        result = self.model.solve(pl.PULP_CBC_CMD(msg=False))
        if self.model.status != pl.LpStatusOptimal:
            print("Model is infeasible with cycle constraints!")
            return
        print("Model is feasible with cycle constraints.")
        
        print("Adding uncircled number constraints...")
        self._add_uncircled_number_constraints()
        
        result = self.model.solve(pl.PULP_CBC_CMD(msg=False))
        if self.model.status != pl.LpStatusOptimal:
            print("Model became infeasible after adding uncircled number constraints!")
            return
        print("Model is feasible with uncircled number constraints.")
        
        print("Adding circled number constraints...")
        self._add_circled_number_constraints()
        
        result = self.model.solve(pl.PULP_CBC_CMD(msg=False))
        if self.model.status != pl.LpStatusOptimal:
            print("Model became infeasible after adding circled number constraints!")
            return
        print("Model is feasible with circled number constraints.")
        
        print("Adding black dot constraints...")
        self._add_black_dots_constraints()
        
        result = self.model.solve(pl.PULP_CBC_CMD(msg=False))
        if self.model.status != pl.LpStatusOptimal:
            print("Model became infeasible after adding black dot constraints!")
            return
        print("Model is feasible with black dot constraints.")
        
        print("Adding white dot constraints...")
        self._add_white_dots_constraints()
        
        result = self.model.solve(pl.PULP_CBC_CMD(msg=False))
        if self.model.status != pl.LpStatusOptimal:
            print("Model became infeasible after adding white dot constraints!")
            return
        
        print("All constraints added successfully. Model is feasible.")
    
    def _add_uncircled_number_constraints(self):
        """Add constraints for uncircled numbers.
        
        Mỗi số không khoanh tròn yêu cầu tổng số đoạn thẳng xung quanh ô đó bằng giá trị của số.
        Có 4 đoạn có thể có: trên, dưới, trái, phải
        """
        print(f"Adding constraints for {len(self.F)} uncircled numbers...")
        for i, j in self.F:
            value = self.matrix_1[i][j]
            
            # Thu thập các đoạn thẳng xung quanh ô (i,j)
            segments = []
            
            # Đoạn phía trên
            if i >= 0 and i <= self.n_r and j >= 0 and j < self.n_c:
                segments.append(self.h[i, j])
            
            # Đoạn phía dưới
            if i+1 <= self.n_r and j >= 0 and j < self.n_c:
                segments.append(self.h[i+1, j])
            
            # Đoạn bên trái
            if i >= 0 and i < self.n_r and j >= 0 and j <= self.n_c:
                segments.append(self.v[i, j])
            
            # Đoạn bên phải
            if i >= 0 and i < self.n_r and j+1 <= self.n_c:
                segments.append(self.v[i, j+1])
            
            # Kiểm tra xem có đủ đoạn không
            if len(segments) < 4:
                print(f"Warning: Uncircled number at ({i},{j}) has fewer than 4 segments: {len(segments)}")
            
            # Thêm ràng buộc
            self.model += (
                pl.lpSum(segments) == value,
                f"Uncircled_Number_{i}_{j}"
            )
    
    def _add_circled_number_constraints(self):
        """Add constraints for circled numbers in the Area51 puzzle.

        Each circled number must:
        1. Be inside the fence (already enforced by self.inside[i, j] == 1).
        2. See exactly 'value' cells (including itself) in the 4 cardinal directions.
           - A cell is visible if it is inside the fence AND no fences block the line of sight.
        """
        for i, j in self.K:
            value = self.matrix_1[i][j]["value"]
            print(f"Processing circled number {value} at ({i}, {j})")

            # Count visible cells in all 4 directions (including the cell itself)
            visible_cells = [1]  # The cell (i, j) counts as 1 visible cell

            # Helper function to add visibility constraints in one direction
            def add_visibility_in_direction(start, end, step, is_horizontal, coord_idx):
                dir_label = "N" if (step == -1 and coord_idx == 0) else \
                            "S" if (step == 1 and coord_idx == 0) else \
                            "E" if (step == 1 and coord_idx == 1) else "W"
                cells_seen_in_direction = 0

                for k in range(start, end, step):
                    # Determine the coordinates of the target cell
                    if coord_idx == 0:  # North or South
                        cell = (k, j)
                    else:  # East or West
                        cell = (i, k)

                    # Check if the target cell is inside the fence
                    cell_is_inside = self.inside[cell]

                    # Check for fence crossings between (i, j) and the target cell
                    fences = []
                    if is_horizontal:  # North or South: check horizontal fences
                        fence_start = min(i, k) + 1
                        fence_end = max(i, k) + 1
                        for fence_idx in range(fence_start, fence_end):
                            if 0 <= fence_idx <= self.n_r:
                                fences.append(self.h[fence_idx, j])
                    else:  # East or West: check vertical fences
                        fence_start = min(j, k) + 1
                        fence_end = max(j, k) + 1
                        for fence_idx in range(fence_start, fence_end):
                            if 0 <= fence_idx <= self.n_c:
                                fences.append(self.v[i, fence_idx])

                    # Determine if the line of sight is clear
                    clear = pl.LpVariable(f"clear_{dir_label}_{i}_{j}_{k}", 0, 1, pl.LpBinary)
                    if fences:
                        self.model += pl.lpSum(fences) <= len(fences) * (1 - clear), f"Clear_{dir_label}_1_{i}_{j}_{k}"
                        self.model += clear <= 1 - (pl.lpSum(fences) / len(fences)), f"Clear_{dir_label}_2_{i}_{j}_{k}"
                    else:
                        self.model += clear == 1, f"Clear_{dir_label}_3_{i}_{j}_{k}"

                    # The cell is visible if it is inside AND the line of sight is clear
                    vis_var = pl.LpVariable(f"vis_{dir_label}_{i}_{j}_{k}", 0, 1, pl.LpBinary)
                    self.model += vis_var <= cell_is_inside, f"Vis_{dir_label}_Inside_{i}_{j}_{k}"
                    self.model += vis_var <= clear, f"Vis_{dir_label}_Clear_{i}_{j}_{k}"
                    self.model += vis_var >= cell_is_inside + clear - 1, f"Vis_{dir_label}_Both_{i}_{j}_{k}"

                    visible_cells.append(vis_var)
                    cells_seen_in_direction += 1

                print(f"Direction {dir_label} from ({i}, {j}): checked {cells_seen_in_direction} cells")

            # Check visibility in all 4 directions
            add_visibility_in_direction(i - 1, -1, -1, True, 0)  # North
            add_visibility_in_direction(i + 1, self.n_r, 1, True, 0)  # South
            add_visibility_in_direction(j + 1, self.n_c, 1, False, 1)  # East
            add_visibility_in_direction(j - 1, -1, -1, False, 1)  # West

            # Ensure the total number of visible cells equals the circled number's value
            self.model += pl.lpSum(visible_cells) == value, f"Circled_Number_Visibility_{i}_{j}"
            print(f"Total visible cells for circled number {value} at ({i}, {j}): {len(visible_cells)} (including self)")
    
    def _add_cycle_constraints(self):
        """Add constraints to ensure a valid cycle.
        
        Tại mỗi điểm lưới (grid point), số đoạn thẳng kết nối đến điểm đó phải là 0 hoặc 2.
        - Nếu là 0: điểm không nằm trên chu trình
        - Nếu là 2: điểm nằm trên chu trình và có chính xác 2 đoạn kết nối (để tạo thành một mạch kín)
        """
        print("Adding cycle constraints for each grid point...")
        for i in range(self.n_r+1):
            for j in range(self.n_c+1):
                segments = []
                
                # Kiểm tra điều kiện biên cho mỗi phương hướng
                # Cạnh bên trái (Left)
                if j > 0 and i <= self.n_r:
                    segments.append(self.h[i, j-1])
                
                # Cạnh bên phải (Right)
                if j < self.n_c and i <= self.n_r:
                    segments.append(self.h[i, j])
                
                # Cạnh phía trên (Up)
                if i > 0 and j <= self.n_c:
                    segments.append(self.v[i-1, j])
                
                # Cạnh phía dưới (Down)
                if i < self.n_r and j <= self.n_c:
                    segments.append(self.v[i, j])
                
                # Ràng buộc: tại mỗi đỉnh, số cạnh kết nối phải bằng 2 lần giá trị của biến điểm p
                if segments:  # Chỉ thêm ràng buộc nếu có ít nhất một đoạn thẳng hợp lệ
                    self.model += (pl.lpSum(segments) == 2*self.p[i, j], f"Point_{i}_{j}")
                else:
                    # Nếu không có đoạn thẳng hợp lệ, buộc p[i,j] = 0
                    self.model += (self.p[i, j] == 0, f"PointZero_{i}_{j}")
    
    def _add_black_dots_constraints(self):
        """Add constraints for black dots.
        
        Ràng buộc cho chấm đen:
        1. h(i, j-2) + h(i, j-1) + h(i, j) + h(i, j+1) = 2
        2. (h(i, j-2) + h(i, j-1) - 2)(h(i, j) + h(i, j+1) - 2) = 0
        3. v(i-2, j) + v(i-1, j) + v(i, j) + v(i+1, j) = 2
        4. (v(i-2, j) + v(i-1, j) - 2)(v(i, j) + v(i+1, j) - 2) = 0
        
        Các segment nằm ngoài biên được coi là bằng 0.
        """
        print(f"Number of black dots: {len(self.B)}")
        for i, j in self.B:
            print(f"Adding constraints for black dot at ({i}, {j})")
            
            # Helper function to get segment value or 0 if out of bounds
            def get_h_segment(i, j):
                if 0 <= i <= self.n_r and 0 <= j < self.n_c:
                    return self.h[i, j]
                return 0
                
            def get_v_segment(i, j):
                if 0 <= i < self.n_r and 0 <= j <= self.n_c:
                    return self.v[i, j]
                return 0
            
            # Get horizontal segments
            h_left2 = get_h_segment(i, j-2)
            h_left1 = get_h_segment(i, j-1)
            h_right1 = get_h_segment(i, j)
            h_right2 = get_h_segment(i, j+1)
            
            # Get vertical segments
            v_up2 = get_v_segment(i-2, j)
            v_up1 = get_v_segment(i-1, j)
            v_down1 = get_v_segment(i, j)
            v_down2 = get_v_segment(i+1, j)
            
            # Constraint 1: Tổng các cạnh ngang = 2
            h_segments = []
            if isinstance(h_left2, pl.LpVariable): h_segments.append(h_left2)
            if isinstance(h_left1, pl.LpVariable): h_segments.append(h_left1)
            if isinstance(h_right1, pl.LpVariable): h_segments.append(h_right1)
            if isinstance(h_right2, pl.LpVariable): h_segments.append(h_right2)
            
            if h_segments:
                self.model += pl.lpSum(h_segments) == 2, f"BlackDot_Horizontal_Sum_{i}_{j}"
            
            # Constraint 2: Tổng các cạnh dọc = 2
            v_segments = []
            if isinstance(v_up2, pl.LpVariable): v_segments.append(v_up2)
            if isinstance(v_up1, pl.LpVariable): v_segments.append(v_up1)
            if isinstance(v_down1, pl.LpVariable): v_segments.append(v_down1)
            if isinstance(v_down2, pl.LpVariable): v_segments.append(v_down2)
            
            if v_segments:
                self.model += pl.lpSum(v_segments) == 2, f"BlackDot_Vertical_Sum_{i}_{j}"
            
            # Constraint 3: (h(i,j-2) + h(i,j-1) - 2)(h(i,j) + h(i,j+1) - 2) = 0
            # Áp dụng ràng buộc phi tuyến bằng biến nhị phân
            
            # Tạo biến cho tổng nửa trái và phải
            left_sum_nonzero = pl.LpVariable(f"BlackDot_Left_Sum_Nonzero_{i}_{j}", 0, 1, pl.LpBinary)
            right_sum_nonzero = pl.LpVariable(f"BlackDot_Right_Sum_Nonzero_{i}_{j}", 0, 1, pl.LpBinary)
            
            # Tổng bên trái: h_left2 + h_left1
            left_sum = 0
            if isinstance(h_left2, pl.LpVariable): left_sum += h_left2
            if isinstance(h_left1, pl.LpVariable): left_sum += h_left1
            
            # Tổng bên phải: h_right1 + h_right2
            right_sum = 0
            if isinstance(h_right1, pl.LpVariable): right_sum += h_right1
            if isinstance(h_right2, pl.LpVariable): right_sum += h_right2
            
            # Constraint: Ít nhất một trong hai tổng phải bằng 2
            # Kiểm tra các segment có tồn tại trước khi áp dụng ràng buộc
            if isinstance(left_sum, pl.LpAffineExpression):
                self.model += left_sum <= 2, f"BlackDot_Left_Sum_Max_{i}_{j}"
                self.model += left_sum >= 2 * left_sum_nonzero, f"BlackDot_Left_Sum_Min_{i}_{j}"
            
            if isinstance(right_sum, pl.LpAffineExpression):
                self.model += right_sum <= 2, f"BlackDot_Right_Sum_Max_{i}_{j}"
                self.model += right_sum >= 2 * right_sum_nonzero, f"BlackDot_Right_Sum_Min_{i}_{j}"
            
            # Ít nhất một trong hai phải bằng 2
            if isinstance(left_sum, pl.LpAffineExpression) and isinstance(right_sum, pl.LpAffineExpression):
                self.model += left_sum_nonzero + right_sum_nonzero >= 1, f"BlackDot_Horizontal_Either_Two_{i}_{j}"
                # Và cả hai không thể cùng bằng 2, vì tổng tất cả đã là 2
                self.model += left_sum_nonzero + right_sum_nonzero <= 1, f"BlackDot_Horizontal_NotBoth_Two_{i}_{j}"
            
            # Tương tự cho các cạnh dọc
            top_sum_nonzero = pl.LpVariable(f"BlackDot_Top_Sum_Nonzero_{i}_{j}", 0, 1, pl.LpBinary)
            bottom_sum_nonzero = pl.LpVariable(f"BlackDot_Bottom_Sum_Nonzero_{i}_{j}", 0, 1, pl.LpBinary)
            
            # Tổng phía trên: v_up2 + v_up1
            top_sum = 0
            if isinstance(v_up2, pl.LpVariable): top_sum += v_up2
            if isinstance(v_up1, pl.LpVariable): top_sum += v_up1
            
            # Tổng phía dưới: v_down1 + v_down2
            bottom_sum = 0
            if isinstance(v_down1, pl.LpVariable): bottom_sum += v_down1
            if isinstance(v_down2, pl.LpVariable): bottom_sum += v_down2
            
            # Kiểm tra các segment có tồn tại trước khi áp dụng ràng buộc
            if isinstance(top_sum, pl.LpAffineExpression):
                self.model += top_sum <= 2, f"BlackDot_Top_Sum_Max_{i}_{j}"
                self.model += top_sum >= 2 * top_sum_nonzero, f"BlackDot_Top_Sum_Min_{i}_{j}"
            
            if isinstance(bottom_sum, pl.LpAffineExpression):
                self.model += bottom_sum <= 2, f"BlackDot_Bottom_Sum_Max_{i}_{j}"
                self.model += bottom_sum >= 2 * bottom_sum_nonzero, f"BlackDot_Bottom_Sum_Min_{i}_{j}"
            
            # Ít nhất một trong hai phải bằng 2
            if isinstance(top_sum, pl.LpAffineExpression) and isinstance(bottom_sum, pl.LpAffineExpression):
                self.model += top_sum_nonzero + bottom_sum_nonzero >= 1, f"BlackDot_Vertical_Either_Two_{i}_{j}"
                # Và cả hai không thể cùng bằng 2, vì tổng tất cả đã là 2
                self.model += top_sum_nonzero + bottom_sum_nonzero <= 1, f"BlackDot_Vertical_NotBoth_Two_{i}_{j}"
            
            print(f"  Added constraints for black dot at ({i}, {j})")
    
    def _add_white_dots_constraints(self):
        """Add constraints for white dots.
        
        Mỗi white dot yêu cầu:
        1. Phải có đúng 2 đoạn thẳng đi qua và tạo một góc 90 độ
        2. Phải là một trong hai cấu hình: h(i,j-1) + h(i,j) = 2 hoặc v(i-1,j) + v(i,j) = 2
        3. Phải thỏa mãn các ràng buộc cạnh vuông góc bổ sung:
           - v(i-1,j-1) + v(i,j-1) + v(i-1,j+1) + v(i,j+1) ≥ h(i,j-1) + h(i,j) - 1
           - h(i-1,j-1) + h(i-1,j) + h(i+1,j-1) + h(i+1,j) ≥ v(i-1,j) + v(i,j) - 1
        
        Đối với white dots ở biên, cần xử lý đặc biệt vì một số đoạn thẳng không tồn tại.
        """
        print(f"Adding constraints for {len(self.W)} white dots...")
        for i, j in self.W:
            print(f"Adding constraints for white dots at ({i}, {j})")
            
            # Kiểm tra xem white dot có nằm ở biên không
            is_boundary = (i == 0 or i == self.n_r or j == 0 or j == self.n_c)
            if is_boundary:
                print(f"  White dot at ({i},{j}) is at boundary")
            
            # Initialize values for segments that might not exist
            h_i_jm1 = 0  # h(i,j-1)
            h_i_j = 0    # h(i,j)
            v_im1_j = 0  # v(i-1,j)
            v_i_j = 0    # v(i,j)
            
            # Assign actual variables for segments that exist
            if j > 0 and i <= self.n_r:
                h_i_jm1 = self.h[i, j-1]
            if j < self.n_c and i <= self.n_r:
                h_i_j = self.h[i, j]
            if i > 0 and j <= self.n_c:
                v_im1_j = self.v[i-1, j]
            if i < self.n_r and j <= self.n_c:
                v_i_j = self.v[i, j]
            
            # Kiểm tra xem có đủ đoạn không
            h_valid = (h_i_jm1 != 0 and h_i_j != 0)
            v_valid = (v_im1_j != 0 and v_i_j != 0)
            
            if not (h_valid or v_valid):
                print(f"  Warning: White dot at ({i},{j}) doesn't have valid horizontal or vertical segments.")
                continue
            
            # Create binary indicators for horizontal and vertical path configurations
            h_path_active = pl.LpVariable(f"h_path_active_{i}_{j}", 0, 1, pl.LpBinary)
            v_path_active = pl.LpVariable(f"v_path_active_{i}_{j}", 0, 1, pl.LpBinary)
            
            # Main constraint: exactly one of horizontal or vertical path must be active
            self.model += h_path_active + v_path_active == 1, f"WhiteDot_OnePath_{i}_{j}"
            
            # If h_valid is false, force h_path_active to be 0
            if not h_valid:
                self.model += h_path_active == 0, f"WhiteDot_H_Impossible_{i}_{j}"
            else:
                # If horizontal path is active, then h(i,j-1) + h(i,j) = 2
                self.model += h_i_jm1 + h_i_j >= 2 * h_path_active, f"WhiteDot_H_Min_{i}_{j}"
                self.model += h_i_jm1 + h_i_j <= 2 * h_path_active + 2 * (1 - h_path_active), f"WhiteDot_H_Max_{i}_{j}"
            
            # If v_valid is false, force v_path_active to be 0
            if not v_valid:
                self.model += v_path_active == 0, f"WhiteDot_V_Impossible_{i}_{j}"
            else:
                # If vertical path is active, then v(i-1,j) + v(i,j) = 2
                self.model += v_im1_j + v_i_j >= 2 * v_path_active, f"WhiteDot_V_Min_{i}_{j}"
                self.model += v_im1_j + v_i_j <= 2 * v_path_active + 2 * (1 - v_path_active), f"WhiteDot_V_Max_{i}_{j}"
            
            # Các ràng buộc bổ sung chỉ áp dụng cho điểm không nằm ở biên
            if not is_boundary:
                # Initialize perpendicular segments for additional constraints
                v_im1_jm1 = 0  # v(i-1,j-1)
                v_i_jm1 = 0    # v(i,j-1)
                v_im1_jp1 = 0  # v(i-1,j+1)
                v_i_jp1 = 0    # v(i,j+1)
                
                h_im1_jm1 = 0  # h(i-1,j-1)
                h_im1_j = 0    # h(i-1,j)
                h_ip1_jm1 = 0  # h(i+1,j-1)
                h_ip1_j = 0    # h(i+1,j)
                
                # Assign actual variables for perpendicular segments that exist
                if i > 0 and j > 0 and j <= self.n_c:
                    v_im1_jm1 = self.v[i-1, j-1]
                if i < self.n_r and j > 0 and j <= self.n_c:
                    v_i_jm1 = self.v[i, j-1]
                if i > 0 and j + 1 <= self.n_c:
                    v_im1_jp1 = self.v[i-1, j+1]
                if i < self.n_r and j + 1 <= self.n_c:
                    v_i_jp1 = self.v[i, j+1]
                    
                if i > 0 and j > 0 and i <= self.n_r:
                    h_im1_jm1 = self.h[i-1, j-1]
                if i > 0 and j < self.n_c and i <= self.n_r:
                    h_im1_j = self.h[i-1, j]
                if i + 1 <= self.n_r and j > 0:
                    h_ip1_jm1 = self.h[i+1, j-1]
                if i + 1 <= self.n_r and j < self.n_c:
                    h_ip1_j = self.h[i+1, j]
                
                # Constraint 1: v(i-1,j-1) + v(i,j-1) + v(i-1,j+1) + v(i,j+1) ≥ h(i,j-1) + h(i,j) - 1
                # Chỉ áp dụng nếu h_path_active có thể = 1
                if h_valid:
                    vertical_segments = v_im1_jm1 + v_i_jm1 + v_im1_jp1 + v_i_jp1
                    horizontal_path = h_i_jm1 + h_i_j
                    
                    self.model += vertical_segments >= horizontal_path - 1, f"WhiteDot_H_Perpendicular_{i}_{j}"
                
                # Constraint 2: h(i-1,j-1) + h(i-1,j) + h(i+1,j-1) + h(i+1,j) ≥ v(i-1,j) + v(i,j) - 1
                # Chỉ áp dụng nếu v_path_active có thể = 1
                if v_valid:
                    horizontal_segments = h_im1_jm1 + h_im1_j + h_ip1_jm1 + h_ip1_j
                    vertical_path = v_im1_j + v_i_j
                    
                    self.model += horizontal_segments >= vertical_path - 1, f"WhiteDot_V_Perpendicular_{i}_{j}"
    
    def _find_loops(self, h_sol, v_sol):
        """Check if the solution contains exactly one cycle.
        
        Phương pháp:
        1. Đếm tổng số cạnh trong giải pháp
        2. Tìm một chu trình và đếm số cạnh trong chu trình
        3. Nếu số cạnh trong chu trình = tổng số cạnh, thì chỉ có một chu trình duy nhất
        """
        # Đếm tổng số cạnh trong giải pháp
        total_edges = 0
        for i in range(self.n_r+1):
            for j in range(self.n_c):
                if h_sol.get((i, j), 0) > 0.5:
                    total_edges += 1
        
        for i in range(self.n_r):
            for j in range(self.n_c+1):
                if v_sol.get((i, j), 0) > 0.5:
                    total_edges += 1
        
        if total_edges == 0:
            print("No edges found in solution")
            return False
        
        # Xây dựng đồ thị từ các cạnh hoạt động
        graph = {}
        # Khởi tạo đồ thị với tất cả các đỉnh (lưới điểm)
        for i in range(self.n_r+1):
            for j in range(self.n_c+1):
                graph[(i, j)] = []
        
        # Thêm các cạnh ngang
        for i in range(self.n_r+1):
            for j in range(self.n_c):
                if h_sol.get((i, j), 0) > 0.5:
                    # Kiểm tra biên - đảm bảo cả hai đầu mút nằm trong lưới
                    if 0 <= i <= self.n_r and 0 <= j < self.n_c and 0 <= j+1 <= self.n_c:
                        graph[(i, j)].append((i, j+1))
                        graph[(i, j+1)].append((i, j))
                    else:
                        print(f"Warning: Horizontal edge at ({i},{j}) is out of bounds!")
        
        # Thêm các cạnh dọc
        for i in range(self.n_r):
            for j in range(self.n_c+1):
                if v_sol.get((i, j), 0) > 0.5:
                    # Kiểm tra biên - đảm bảo cả hai đầu mút nằm trong lưới
                    if 0 <= i < self.n_r and 0 <= j <= self.n_c and 0 <= i+1 <= self.n_r:
                        graph[(i, j)].append((i+1, j))
                        graph[(i+1, j)].append((i, j))
                    else:
                        print(f"Warning: Vertical edge at ({i},{j}) is out of bounds!")
        
        # Tìm một đỉnh hoạt động để bắt đầu DFS
        active_nodes = []
        for node, neighbors in graph.items():
            if neighbors:
                active_nodes.append(node)
                break
        
        if not active_nodes:
            print("No active nodes found in graph")
            return False
        
        start_node = active_nodes[0]
        visited = set()
        cycle_edges = []
        
        # Tìm chu trình bằng DFS cải tiến
        def find_cycle(current, parent, path):
            if current in path:
                # Tìm thấy chu trình
                cycle_start_idx = path.index(current)
                cycle = path[cycle_start_idx:]
                edges = []
                for i in range(len(cycle)-1):
                    edges.append((cycle[i], cycle[i+1]))
                edges.append((cycle[-1], cycle[0]))  # Đóng chu trình
                return edges
            
            visited.add(current)
            path.append(current)
            
            for neighbor in graph[current]:
                if neighbor == parent:
                    continue  # Bỏ qua node cha
                
                if neighbor not in visited:
                    result = find_cycle(neighbor, current, path)
                    if result:
                        return result
                elif neighbor in path:
                    # Tìm thấy chu trình
                    cycle_start_idx = path.index(neighbor)
                    cycle = path[cycle_start_idx:]
                    edges = []
                    for i in range(len(cycle)-1):
                        edges.append((cycle[i], cycle[i+1]))
                    edges.append((cycle[-1], neighbor))  # Đóng chu trình
                    return edges
            
            path.pop()  # Quay lui
            return None
        
        cycle_edges = find_cycle(start_node, None, [])
        
        if not cycle_edges:
            print("No cycle found in graph")
            return False
        
        # Nếu số cạnh trong chu trình = tổng số cạnh -> chỉ có một chu trình
        print(f"Cycle has {len(cycle_edges)} edges, total edges: {total_edges}")
        return len(cycle_edges) == total_edges
    
    def _flood_fill(self, h_sol, v_sol):
        """Determine which cells are inside and outside the fence using flood fill."""
        fence_h = {}
        fence_v = {}
        
        # Tạo bản đồ hàng rào ngang
        for i in range(self.n_r+1):
            for j in range(self.n_c):
                fence_h[(i, j)] = h_sol.get((i, j), 0) > 0.5
        
        # Tạo bản đồ hàng rào dọc
        for i in range(self.n_r):
            for j in range(self.n_c+1):
                fence_v[(i, j)] = v_sol.get((i, j), 0) > 0.5
        
        # Khởi tạo trạng thái trong/ngoài cho mỗi ô
        inside_status = {}
        for i in range(self.n_r):
            for j in range(self.n_c):
                inside_status[(i, j)] = None
        
        # Chọn điểm bắt đầu (thường là một ô chắc chắn nằm trong hàng rào)
        start_point = None
        if self.K:  # Số được khoanh tròn phải nằm bên trong
            start_point = self.K[0]
        elif self.A:  # Người ngoài hành tinh phải nằm bên trong
            start_point = self.A[0]
        else:
            # Không có gợi ý, sử dụng điểm trung tâm
            mid_i = self.n_r // 2
            mid_j = self.n_c // 2
            start_point = (mid_i, mid_j)
        
        print(f"Flood fill starting from {start_point}")
        
        # Thực hiện BFS để lan tỏa trạng thái trong/ngoài
        queue = [start_point]
        inside_status[start_point] = True
        
        while queue:
            i, j = queue.pop(0)
            neighbors = []
            
            # Kiểm tra hàng xóm bên phải
            if j < self.n_c - 1:
                # Chỉ đi qua được nếu không có hàng rào dọc chặn
                if not fence_v.get((i, j+1), False):
                    neighbors.append((i, j+1))
            
            # Kiểm tra hàng xóm bên trái
            if j > 0:
                # Chỉ đi qua được nếu không có hàng rào dọc chặn
                if not fence_v.get((i, j), False):
                    neighbors.append((i, j-1))
            
            # Kiểm tra hàng xóm bên dưới
            if i < self.n_r - 1:
                # Chỉ đi qua được nếu không có hàng rào ngang chặn
                if not fence_h.get((i+1, j), False):
                    neighbors.append((i+1, j))
            
            # Kiểm tra hàng xóm bên trên
            if i > 0:
                # Chỉ đi qua được nếu không có hàng rào ngang chặn
                if not fence_h.get((i, j), False):
                    neighbors.append((i-1, j))
            
            for ni, nj in neighbors:
                # Chỉ xét các ô chưa được đánh dấu
                if inside_status.get((ni, nj)) is None:
                    inside_status[(ni, nj)] = inside_status[(i, j)]
                    queue.append((ni, nj))
        
        # Các ô không thể đến được từ điểm bắt đầu nằm ở bên ngoài hàng rào
        for i in range(self.n_r):
            for j in range(self.n_c):
                if inside_status.get((i, j)) is None:
                    inside_status[(i, j)] = False
        
        # Kiểm tra ràng buộc trong/ngoài
        for i, j in self.A:
            if not inside_status.get((i, j), False):
                print(f"Warning: Alien at ({i},{j}) is not inside the fence!")
        
        for i, j in self.T:
            if inside_status.get((i, j), False):
                print(f"Warning: Triffid at ({i},{j}) is not outside the fence!")
        
        for i, j in self.K:
            if not inside_status.get((i, j), False):
                print(f"Warning: Circled number at ({i},{j}) is not inside the fence!")
        
        return inside_status
    
    def visualize_solution(self, output_path=None):
        """Visualize the solution with corrected alien/triffid labels and font."""
        if self.solution is None:
            print("No solution to visualize.")
            return
        
        # Set a basic font to avoid display issues
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
        except:
            try:
                plt.rcParams['font.family'] = 'Arial'
            except:
                print("Warning: Could not set font to DejaVu Sans or Arial. Using default font.")

        fig, ax = plt.subplots(figsize=(10, 10))
        
        for i in range(self.n_r+1):
            ax.axhline(i, color='gray', linestyle=':')
        for j in range(self.n_c+1):
            ax.axvline(j, color='gray', linestyle=':')
        
        h_sol = self.solution["h"]
        v_sol = self.solution["v"]
        inside_status = self._flood_fill(h_sol, v_sol)
        
        for i in range(self.n_r):
            for j in range(self.n_c):
                if inside_status.get((i, j), False):
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='lightblue', alpha=0.4))
                else:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='lightyellow', alpha=0.3))
        
        for (i, j), val in h_sol.items():
            if val > 0.5:
                ax.plot([j, j+1], [i, i], 'k-', linewidth=3)
        
        for (i, j), val in v_sol.items():
            if val > 0.5:
                ax.plot([j, j], [i, i+1], 'k-', linewidth=3)
        
        for i in range(self.n_r):
            for j in range(self.n_c):
                cell = self.matrix_1[i][j]
                if cell == "A":
                    is_inside = inside_status.get((i, j), False)
                    status_text = "IN" if is_inside else "OUT"
                    correctness = "correct" if is_inside else "incorrect"
                    ax.text(j+0.5, i+0.5, f"A\n{status_text}\n({correctness})", fontsize=10, ha='center', va='center')
                    ax.add_patch(plt.Circle((j+0.5, i+0.5), 0.4, fill=False, edgecolor='blue', linestyle='--', linewidth=1.5))
                elif cell == "C":
                    is_inside = inside_status.get((i, j), False)
                    status_text = "IN" if is_inside else "OUT"
                    correctness = "correct" if not is_inside else "incorrect"
                    ax.text(j+0.5, i+0.5, f"C\n{status_text}\n({correctness})", fontsize=10, ha='center', va='center')
                    ax.add_patch(plt.Circle((j+0.5, i+0.5), 0.4, fill=False, edgecolor='green', linestyle='--', linewidth=1.5))
                elif isinstance(cell, dict) and cell.get("circled"):
                    is_inside = inside_status.get((i, j), False)
                    status_text = "IN" if is_inside else "OUT"
                    ax.add_patch(plt.Circle((j+0.5, i+0.5), 0.3, fill=False, edgecolor='red', linewidth=1.5))
                    ax.text(j+0.5, i+0.5, f"{cell['value']}\n{status_text}", fontsize=10, ha='center', va='center')
                elif isinstance(cell, int):
                    ax.text(j+0.5, i+0.5, str(cell), fontsize=10, ha='center', va='center')
        
        for i in range(len(self.matrix_2)):
            for j in range(len(self.matrix_2[0])):
                point = self.matrix_2[i][j]
                if point == "E":
                    ax.add_patch(plt.Circle((j, i), 0.2, facecolor='white', edgecolor='black', linewidth=2, zorder=10))
                elif point == "F":
                    ax.add_patch(plt.Circle((j, i), 0.2, facecolor='black', edgecolor='white', linewidth=1, zorder=10))
        
        ax.add_patch(plt.Rectangle((0, -0.5), 0.3, 0.3, facecolor='lightblue', alpha=0.4))
        ax.text(0.4, -0.35, "Inside fence", fontsize=10, va='center')
        ax.add_patch(plt.Rectangle((2, -0.5), 0.3, 0.3, facecolor='lightyellow', alpha=0.3))
        ax.text(2.4, -0.35, "Outside fence", fontsize=10, va='center')
        
        ax.set_xlim(-0.5, self.n_c+0.5)
        ax.set_ylim(self.n_r+0.5, -0.8)
        ax.set_aspect('equal')
        ax.axis('off')
        
        ax.set_title(f"Area51 Puzzle: {os.path.basename(self.puzzle_path)}\nAliens must be INSIDE (blue), Triffids must be OUTSIDE (yellow)", 
                    fontsize=12, pad=10)
        
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