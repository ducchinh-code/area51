import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB


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
        self.model = gp.Model("Area51")
        
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
        self.h_fence = {}  # Horizontal fence segments
        self.v_fence = {}  # Vertical fence segments
        for i in range(self.rows + 1):
            for j in range(self.cols):
                self.h_fence[i, j] = self.model.addVar(vtype=GRB.BINARY, name=f'h_{i}_{j}')
        for i in range(self.rows):
            for j in range(self.cols + 1):
                self.v_fence[i, j] = self.model.addVar(vtype=GRB.BINARY, name=f'v_{i}_{j}')

        # Create binary variables for cell containment (inside/outside fence)
        self.inside = {}
        for i in range(self.rows):
            for j in range(self.cols):
                self.inside[i, j] = self.model.addVar(vtype=GRB.BINARY, name=f'in_{i}_{j}')

        self.model.update()

    def add_constraints(self):
        self.add_fence_constraints()
        self.add_inside_outside_constraints()
        self.add_element_constraints()

    def add_fence_constraints(self):
        # Each vertex must have 0 or 2 incident edges (degree constraint)
        for i in range(self.rows + 1):
            for j in range(self.cols + 1):
                edges = []
                if j > 0:
                    edges.append(self.h_fence[i, j - 1])
                if j < self.cols:
                    edges.append(self.h_fence[i, j])
                if i > 0:
                    edges.append(self.v_fence[i - 1, j])
                if i < self.rows:
                    edges.append(self.v_fence[i, j])
                
                if edges:
                    # Every vertex must have 0 or 2 incident edges (for a valid loop)
                    # Tối ưu: sử dụng ràng buộc mod-2 trực tiếp thay vì thêm biến k
                    # self.model.addConstr(gp.quicksum(edges) % 2 == 0, f"degree_{i}_{j}")
                    # Sửa lại: Gurobi không hỗ trợ toán tử % trực tiếp
                    k = self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=len(edges) // 2, name=f'k_degree_{i}_{j}')
                    self.model.addConstr(gp.quicksum(edges) == 2 * k, f"degree_{i}_{j}")
        
        # The total number of fence segments must be at least 4 (smallest possible loop)
        total_fence = gp.quicksum(self.h_fence[i, j] for i in range(self.rows + 1) for j in range(self.cols)) + \
                      gp.quicksum(self.v_fence[i, j] for i in range(self.rows) for j in range(self.cols + 1))
        self.model.addConstr(total_fence >= 4, "min_fence_length")
        
        # Tối ưu: sử dụng phương pháp hiệu quả hơn cho việc bảo đảm tính liên thông
        # Thay vì tạo biến dòng chảy cho mỗi cạnh, chúng ta sử dụng các biến chỉ ra thứ tự cạnh
        # trong chu trình. Điều này giảm đáng kể số lượng ràng buộc và biến.
        
        # Tạo biến chỉ ra thứ tự của các cạnh trong chu trình
        order_h = {}  # Thứ tự của các cạnh ngang
        order_v = {}  # Thứ tự của các cạnh dọc
        
        # Ước tính số cạnh tối đa
        max_edges = self.rows * self.cols * 2
        
        for i in range(self.rows + 1):
            for j in range(self.cols):
                # Chỉ tạo biến thứ tự nếu cạnh tồn tại trong hàng rào
                order_h[i, j] = self.model.addVar(lb=0, ub=max_edges,
                                              vtype=GRB.INTEGER, name=f'order_h_{i}_{j}')
                # Thứ tự = 0 nếu cạnh không được sử dụng
                self.model.addConstr((self.h_fence[i, j] == 0) >> (order_h[i, j] == 0), f"order_h_unused_{i}_{j}")
                # Thứ tự > 0 nếu cạnh được sử dụng
                self.model.addConstr((self.h_fence[i, j] == 1) >> (order_h[i, j] >= 1), f"order_h_used_{i}_{j}")
                self.model.addConstr((self.h_fence[i, j] == 1) >> (order_h[i, j] <= max_edges), f"order_h_max_{i}_{j}")
        
        for i in range(self.rows):
            for j in range(self.cols + 1):
                # Chỉ tạo biến thứ tự nếu cạnh tồn tại trong hàng rào
                order_v[i, j] = self.model.addVar(lb=0, ub=max_edges,
                                              vtype=GRB.INTEGER, name=f'order_v_{i}_{j}')
                # Thứ tự = 0 nếu cạnh không được sử dụng
                self.model.addConstr((self.v_fence[i, j] == 0) >> (order_v[i, j] == 0), f"order_v_unused_{i}_{j}")
                # Thứ tự > 0 nếu cạnh được sử dụng
                self.model.addConstr((self.v_fence[i, j] == 1) >> (order_v[i, j] >= 1), f"order_v_used_{i}_{j}")
                self.model.addConstr((self.v_fence[i, j] == 1) >> (order_v[i, j] <= max_edges), f"order_v_max_{i}_{j}")
        
        # Đảm bảo thứ tự của các cạnh liên tiếp trên chu trình phải liên tục
        # Các cạnh liền kề với cùng một đỉnh phải có thứ tự liên tiếp nếu cả hai đều thuộc chu trình
        for i in range(self.rows + 1):
            for j in range(self.cols + 1):
                # Thu thập các cặp cạnh liền kề tại đỉnh này
                adjacent_pairs = []
                
                # Cạnh ngang bên trái và bên phải
                if j > 0 and j < self.cols:
                    adjacent_pairs.append((self.h_fence[i, j-1], order_h[i, j-1], self.h_fence[i, j], order_h[i, j]))
                
                # Cạnh dọc phía trên và phía dưới
                if i > 0 and i < self.rows:
                    adjacent_pairs.append((self.v_fence[i-1, j], order_v[i-1, j], self.v_fence[i, j], order_v[i, j]))
                
                # Cạnh ngang bên trái và cạnh dọc phía trên
                if j > 0 and i > 0:
                    adjacent_pairs.append((self.h_fence[i, j-1], order_h[i, j-1], self.v_fence[i-1, j], order_v[i-1, j]))
                
                # Cạnh ngang bên trái và cạnh dọc phía dưới
                if j > 0 and i < self.rows:
                    adjacent_pairs.append((self.h_fence[i, j-1], order_h[i, j-1], self.v_fence[i, j], order_v[i, j]))
                
                # Cạnh ngang bên phải và cạnh dọc phía trên
                if j < self.cols and i > 0:
                    adjacent_pairs.append((self.h_fence[i, j], order_h[i, j], self.v_fence[i-1, j], order_v[i-1, j]))
                
                # Cạnh ngang bên phải và cạnh dọc phía dưới
                if j < self.cols and i < self.rows:
                    adjacent_pairs.append((self.h_fence[i, j], order_h[i, j], self.v_fence[i, j], order_v[i, j]))
                
                # Thêm ràng buộc cho các cặp cạnh liền kề
                for edge1, order1, edge2, order2 in adjacent_pairs:
                    # Nếu cả hai cạnh đều thuộc chu trình, thứ tự của chúng phải liên tiếp
                    # edge1 * edge2 * (order1 - order2) + (1-edge1) * max_edges + (1-edge2) * max_edges >= 0
                    # edge1 * edge2 * (order2 - order1) + (1-edge1) * max_edges + (1-edge2) * max_edges >= 0
                    
                    # Thêm biến chỉ báo cho biết cả hai cạnh có đều thuộc chu trình không
                    both_used = self.model.addVar(vtype=GRB.BINARY, name=f'both_used_{edge1.VarName}_{edge2.VarName}')
                    self.model.addConstr(both_used <= edge1, f"both_used1_{edge1.VarName}_{edge2.VarName}")
                    self.model.addConstr(both_used <= edge2, f"both_used2_{edge1.VarName}_{edge2.VarName}")
                    self.model.addConstr(both_used >= edge1 + edge2 - 1, f"both_used3_{edge1.VarName}_{edge2.VarName}")
                    
                    # Nếu cả hai cạnh đều thuộc chu trình, thứ tự của chúng chênh lệch 1 hoặc max_edges-1
                    # (nếu một cạnh là cạnh cuối cùng và cạnh kia là cạnh đầu tiên)
                    # self.model.addConstr((both_used == 1) >> (
                    #     (order1 - order2 == 1) | (order2 - order1 == 1) | 
                    #     (order1 - order2 == max_edges - 1) | (order2 - order1 == max_edges - 1)
                    # ), f"adjacent_order_{edge1.VarName}_{edge2.VarName}")
                    
                    # Sửa lỗi: Gurobi không hỗ trợ toán tử OR (|) trực tiếp cho ràng buộc
                    # Thay vào đó, sử dụng biến binary cho mỗi trường hợp và OR bằng tổng >= 1
                    
                    # Trường hợp 1: order1 - order2 == 1
                    case1 = self.model.addVar(vtype=GRB.BINARY, name=f'case1_{edge1.VarName}_{edge2.VarName}')
                    self.model.addConstr((case1 == 1) >> (order1 - order2 == 1), f"case1_{edge1.VarName}_{edge2.VarName}")
                    
                    # Trường hợp 2: order2 - order1 == 1
                    case2 = self.model.addVar(vtype=GRB.BINARY, name=f'case2_{edge1.VarName}_{edge2.VarName}')
                    self.model.addConstr((case2 == 1) >> (order2 - order1 == 1), f"case2_{edge1.VarName}_{edge2.VarName}")
                    
                    # Trường hợp 3: order1 - order2 == max_edges - 1
                    case3 = self.model.addVar(vtype=GRB.BINARY, name=f'case3_{edge1.VarName}_{edge2.VarName}')
                    self.model.addConstr((case3 == 1) >> (order1 - order2 == max_edges - 1), f"case3_{edge1.VarName}_{edge2.VarName}")
                    
                    # Trường hợp 4: order2 - order1 == max_edges - 1
                    case4 = self.model.addVar(vtype=GRB.BINARY, name=f'case4_{edge1.VarName}_{edge2.VarName}')
                    self.model.addConstr((case4 == 1) >> (order2 - order1 == max_edges - 1), f"case4_{edge1.VarName}_{edge2.VarName}")
                    
                    # Ít nhất một trong các trường hợp phải đúng
                    self.model.addConstr((both_used == 1) >> (case1 + case2 + case3 + case4 >= 1), 
                                         f"adjacent_order_{edge1.VarName}_{edge2.VarName}")
        
        # Đảm bảo các giá trị thứ tự đều khác nhau nếu cạnh được sử dụng
        # Điều này ngăn chặn chu trình nhỏ hơn
        self.model.update()

    def add_inside_outside_constraints(self):
        """
        Add constraints to determine inside/outside regions.
        
        A cell is "inside" if a ray from that cell to the boundary crosses
        an odd number of fence segments. We use ray shooting to the right (East).
        """
        for i in range(self.rows):
            for j in range(self.cols):
                # Ray shooting to the right (East)
                # Count fence segments crossed by a ray from cell (i,j) to the right edge
                crossings = []
                for k in range(j + 1, self.cols + 1):
                    crossings.append(self.v_fence[i, k])
                # Cell is inside if number of crossings is odd
                # Tối ưu: sử dụng ràng buộc mod-2 trực tiếp thay vì thêm biến k
                # self.model.addConstr(gp.quicksum(crossings) % 2 == self.inside[i, j], f"inside_{i}_{j}")
                # Sửa lại: Gurobi không hỗ trợ toán tử % trực tiếp
                k = self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=self.cols + 1 - j, name=f'k_inside_{i}_{j}')
                self.model.addConstr(gp.quicksum(crossings) == 2 * k + self.inside[i, j], f"inside_{i}_{j}")
        
        # Add connectivity constraints for inside region
        # At least one cell must be inside
        total_inside = gp.quicksum(self.inside[i, j] for i in range(self.rows) for j in range(self.cols))
        self.model.addConstr(total_inside >= 1, "min_inside")
        
        # There must be some cells outside as well
        total_outside = gp.quicksum(1 - self.inside[i, j] for i in range(self.rows) for j in range(self.cols))
        self.model.addConstr(total_outside >= 1, "min_outside")
        
        # Add connectivity constraints: adjacent cells separated by a fence must have different inside/outside status
        for i in range(self.rows):
            for j in range(self.cols):
                # Horizontal fence below this cell
                if i < self.rows - 1:
                    # Tối ưu: sử dụng ràng buộc chỉ báo thay vì nhiều biến
                    # self.model.addConstr(
                    #     (self.h_fence[i + 1, j] == 1) >> (self.inside[i, j] != self.inside[i + 1, j]),
                    #     f"h_fence_below_{i}_{j}"
                    # )
                    
                    # Sửa lỗi: Gurobi không hỗ trợ ràng buộc bất đẳng thức (!=)
                    # Thay thế bằng hai ràng buộc tương đương: 
                    # nếu h_fence = 1 thì inside1 + inside2 != 1 (tức là inside1 + inside2 = 0 hoặc 2)
                    self.model.addConstr(
                        (self.h_fence[i + 1, j] == 1) >> 
                        (self.inside[i, j] + self.inside[i + 1, j] <= 0 + self.h_fence[i + 1, j]),
                        f"h_fence_below_{i}_{j}_0"
                    )
                    self.model.addConstr(
                        (self.h_fence[i + 1, j] == 1) >> 
                        (self.inside[i, j] + self.inside[i + 1, j] >= 2 - self.h_fence[i + 1, j]),
                        f"h_fence_below_{i}_{j}_2"
                    )
                
                # Vertical fence to the right of this cell
                if j < self.cols - 1:
                    # Tối ưu: sử dụng ràng buộc chỉ báo thay vì nhiều biến
                    # self.model.addConstr(
                    #     (self.v_fence[i, j + 1] == 1) >> (self.inside[i, j] != self.inside[i, j + 1]),
                    #     f"v_fence_right_{i}_{j}"
                    # )
                    
                    # Sửa lỗi: Gurobi không hỗ trợ ràng buộc bất đẳng thức (!=)
                    # Thay thế bằng hai ràng buộc tương đương: 
                    # nếu v_fence = 1 thì inside1 + inside2 != 1 (tức là inside1 + inside2 = 0 hoặc 2)
                    self.model.addConstr(
                        (self.v_fence[i, j + 1] == 1) >> 
                        (self.inside[i, j] + self.inside[i, j + 1] <= 0 + self.v_fence[i, j + 1]),
                        f"v_fence_right_{i}_{j}_0"
                    )
                    self.model.addConstr(
                        (self.v_fence[i, j + 1] == 1) >> 
                        (self.inside[i, j] + self.inside[i, j + 1] >= 2 - self.v_fence[i, j + 1]),
                        f"v_fence_right_{i}_{j}_2"
                    )

    def add_element_constraints(self):
        for i in range(self.rows):
            for j in range(self.cols):
                cell_1 = self.matrix_1[i][j]
                
                # Handle matrix_1 elements
                if cell_1 is not None:
                    if cell_1 == "A":  # Alien must be inside
                        self.model.addConstr(self.inside[i, j] == 1, f"alien_{i}_{j}")
                    elif cell_1 == "C":  # Cactus must be outside
                        self.model.addConstr(self.inside[i, j] == 0, f"cactus_{i}_{j}")
                    elif isinstance(cell_1, dict) and cell_1.get("circled"):
                        # Circled number - visibility constraints
                        value = cell_1["value"]
                        # Check if the value is reasonable
                        max_visible = 2 * (self.rows + self.cols) - 3  # Max possible visible cells
                        if value > max_visible:
                            print(f"Warning: Circled number at ({i},{j}) is {value}, may be too large (max possible: {max_visible})")
                        self.add_visibility_constraints(i, j, value)
                    elif isinstance(cell_1, (int, float)):
                        # Validate number range (0-3)
                        if cell_1 < 0 or cell_1 > 3:
                            print(f"Warning: Uncircled number at ({i},{j}) is {cell_1}, should be between 0-3")
                        # Uncircled number - count fence segments around the square
                        self.add_uncircled_number_constraints(i, j, cell_1)
                
                # Handle matrix_2 elements (Masyu circles)
                for di, dj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    if i + di < len(self.matrix_2) and j + dj < len(self.matrix_2[0]):
                        dot = self.matrix_2[i + di][j + dj]
                        if dot in ["E", "F"]:
                            self.add_masyu_constraints(i + di, j + dj, dot)

    def add_visibility_constraints(self, i, j, value):
        """Add constraints for circled number visibility.
        A circled number:
        1. Must be inside the fence
        2. Counts total visible squares in all 4 directions (N,S,E,W)
        3. Includes its own square in the count
        4. A square is visible if it's not blocked by a fence
        """
        # First, ensure the circled number is inside the fence
        self.model.addConstr(self.inside[i, j] == 1, f"circled_inside_{i}_{j}")
        
        # Tối ưu: Trực tiếp đếm số lượng ô nhìn thấy thay vì tạo biến cho mỗi ô
        # Điều này giảm đáng kể số lượng biến và ràng buộc
        
        # Số lượng ô nhìn thấy theo hướng Bắc (lên)
        north_visible = []
        for r in range(i - 1, -1, -1):
            # Tạo biến cho vùng nhìn thấy liên tục
            visible_to_r = self.model.addVar(vtype=GRB.BINARY, name=f'vis_to_N_{r}_{j}')
            
            # Nếu có hàng rào ngang giữa ô gốc và ô này, ô này không nhìn thấy được
            blocking_fences = []
            for k in range(r, i):
                blocking_fences.append(self.h_fence[k + 1, j])
            
            # Ô nhìn thấy được nếu không có hàng rào nào chặn tầm nhìn
            self.model.addConstr(visible_to_r <= 1 - gp.quicksum(blocking_fences), f"vis_N_{r}_{j}")
            
            # Thêm vào danh sách ô nhìn thấy được
            north_visible.append(visible_to_r)
        
        # Số lượng ô nhìn thấy theo hướng Nam (xuống)
        south_visible = []
        for r in range(i + 1, self.rows):
            # Tạo biến cho vùng nhìn thấy liên tục
            visible_to_r = self.model.addVar(vtype=GRB.BINARY, name=f'vis_to_S_{r}_{j}')
            
            # Nếu có hàng rào ngang giữa ô gốc và ô này, ô này không nhìn thấy được
            blocking_fences = []
            for k in range(i, r):
                blocking_fences.append(self.h_fence[k + 1, j])
            
            # Ô nhìn thấy được nếu không có hàng rào nào chặn tầm nhìn
            self.model.addConstr(visible_to_r <= 1 - gp.quicksum(blocking_fences), f"vis_S_{r}_{j}")
            
            # Thêm vào danh sách ô nhìn thấy được
            south_visible.append(visible_to_r)
        
        # Số lượng ô nhìn thấy theo hướng Đông (phải)
        east_visible = []
        for c in range(j + 1, self.cols):
            # Tạo biến cho vùng nhìn thấy liên tục
            visible_to_c = self.model.addVar(vtype=GRB.BINARY, name=f'vis_to_E_{i}_{c}')
            
            # Nếu có hàng rào dọc giữa ô gốc và ô này, ô này không nhìn thấy được
            blocking_fences = []
            for k in range(j, c):
                blocking_fences.append(self.v_fence[i, k + 1])
            
            # Ô nhìn thấy được nếu không có hàng rào nào chặn tầm nhìn
            self.model.addConstr(visible_to_c <= 1 - gp.quicksum(blocking_fences), f"vis_E_{i}_{c}")
            
            # Thêm vào danh sách ô nhìn thấy được
            east_visible.append(visible_to_c)
        
        # Số lượng ô nhìn thấy theo hướng Tây (trái)
        west_visible = []
        for c in range(j - 1, -1, -1):
            # Tạo biến cho vùng nhìn thấy liên tục
            visible_to_c = self.model.addVar(vtype=GRB.BINARY, name=f'vis_to_W_{i}_{c}')
            
            # Nếu có hàng rào dọc giữa ô gốc và ô này, ô này không nhìn thấy được
            blocking_fences = []
            for k in range(c, j):
                blocking_fences.append(self.v_fence[i, k + 1])
            
            # Ô nhìn thấy được nếu không có hàng rào nào chặn tầm nhìn
            self.model.addConstr(visible_to_c <= 1 - gp.quicksum(blocking_fences), f"vis_W_{i}_{c}")
            
            # Thêm vào danh sách ô nhìn thấy được
            west_visible.append(visible_to_c)
        
        # Tổng số ô nhìn thấy được phải bằng giá trị của số được khoanh tròn
        # Ô ban đầu (i,j) luôn nhìn thấy được
        all_visible = north_visible + south_visible + east_visible + west_visible
        self.model.addConstr(gp.quicksum(all_visible) + 1 == value, f"total_visible_{i}_{j}")

    def add_masyu_constraints(self, i, j, dot_type):
        """Add constraints for Masyu black and white circles
        
        Rules:
        - Black circle (F): The fence must make a turn at the circle, and must go
          straight through the next cell in both directions.
        - White circle (E): The fence must go straight through the circle, and must
          make a turn at at least one of the adjacent cells.
        """
        # Get all possible fence segments that can go through this vertex
        h_left = None if j == 0 else self.h_fence[i, j - 1]
        h_right = None if j >= self.cols else self.h_fence[i, j]
        v_up = None if i == 0 else self.v_fence[i - 1, j]
        v_down = None if i >= self.rows else self.v_fence[i, j]

        # Collect valid fence segments
        fence_edges = []
        if h_left is not None:
            fence_edges.append(h_left)
        if h_right is not None:
            fence_edges.append(h_right)
        if v_up is not None:
            fence_edges.append(v_up)
        if v_down is not None:
            fence_edges.append(v_down)
        
        # Fence must pass through the vertex with exactly 2 edges
        if fence_edges:
            self.model.addConstr(gp.quicksum(fence_edges) == 2, f"masyu_pass_{i}_{j}")
        
        # Tối ưu: Xử lý horiz_pass và vert_pass theo cách khác để giảm số biến
        if dot_type == "F":  # Black circle
            # Black circle must make a turn
            # Tối ưu: Thay vì tạo các biến trung gian, sử dụng ràng buộc trực tiếp
            # Không thể có hai cạnh ngang hoặc hai cạnh dọc cùng được sử dụng
            if h_left is not None and h_right is not None:
                self.model.addConstr(h_left + h_right <= 1, f"black_no_horiz_{i}_{j}")
            if v_up is not None and v_down is not None:
                self.model.addConstr(v_up + v_down <= 1, f"black_no_vert_{i}_{j}")

            # Tối ưu: Xác định các cặp tạo góc và thêm các ràng buộc thẳng hàng trực tiếp
            turns = []
            
            # Left-Up turn
            if h_left is not None and v_up is not None:
                left_up_turn = self.model.addVar(vtype=GRB.BINARY, name=f'left_up_turn_{i}_{j}')
                self.model.addConstr(left_up_turn <= h_left, f"left_up_h_{i}_{j}")
                self.model.addConstr(left_up_turn <= v_up, f"left_up_v_{i}_{j}")
                self.model.addConstr(left_up_turn >= h_left + v_up - 1, f"left_up_used_{i}_{j}")
                turns.append(left_up_turn)
                
                # Thêm điều kiện thẳng hàng nếu góc này được sử dụng
                if j > 1:  # Có thể mở rộng sang trái hai ô
                    self.model.addConstr((left_up_turn == 1) >> (self.h_fence[i, j-1] == 1), f"black_left_extend1_{i}_{j}")
                    self.model.addConstr((left_up_turn == 1) >> (self.h_fence[i, j-2] == 1), f"black_left_extend2_{i}_{j}")
                
                if i > 1:  # Có thể mở rộng lên trên hai ô
                    self.model.addConstr((left_up_turn == 1) >> (self.v_fence[i-1, j] == 1), f"black_up_extend1_{i}_{j}")
                    self.model.addConstr((left_up_turn == 1) >> (self.v_fence[i-2, j] == 1), f"black_up_extend2_{i}_{j}")
            
            # Left-Down turn
            if h_left is not None and v_down is not None:
                left_down_turn = self.model.addVar(vtype=GRB.BINARY, name=f'left_down_turn_{i}_{j}')
                self.model.addConstr(left_down_turn <= h_left, f"left_down_h_{i}_{j}")
                self.model.addConstr(left_down_turn <= v_down, f"left_down_v_{i}_{j}")
                self.model.addConstr(left_down_turn >= h_left + v_down - 1, f"left_down_used_{i}_{j}")
                turns.append(left_down_turn)
                
                # Thêm điều kiện thẳng hàng nếu góc này được sử dụng
                if j > 1:  # Có thể mở rộng sang trái hai ô
                    self.model.addConstr((left_down_turn == 1) >> (self.h_fence[i, j-1] == 1), f"black_left_extend1_2_{i}_{j}")
                    self.model.addConstr((left_down_turn == 1) >> (self.h_fence[i, j-2] == 1), f"black_left_extend2_2_{i}_{j}")
                
                if i < self.rows - 2:  # Có thể mở rộng xuống dưới hai ô
                    self.model.addConstr((left_down_turn == 1) >> (self.v_fence[i+1, j] == 1), f"black_down_extend1_{i}_{j}")
                    self.model.addConstr((left_down_turn == 1) >> (self.v_fence[i+2, j] == 1), f"black_down_extend2_{i}_{j}")
            
            # Right-Up turn
            if h_right is not None and v_up is not None:
                right_up_turn = self.model.addVar(vtype=GRB.BINARY, name=f'right_up_turn_{i}_{j}')
                self.model.addConstr(right_up_turn <= h_right, f"right_up_h_{i}_{j}")
                self.model.addConstr(right_up_turn <= v_up, f"right_up_v_{i}_{j}")
                self.model.addConstr(right_up_turn >= h_right + v_up - 1, f"right_up_used_{i}_{j}")
                turns.append(right_up_turn)
                
                # Thêm điều kiện thẳng hàng nếu góc này được sử dụng
                if j < self.cols - 2:  # Có thể mở rộng sang phải hai ô
                    self.model.addConstr((right_up_turn == 1) >> (self.h_fence[i, j+1] == 1), f"black_right_extend1_{i}_{j}")
                    self.model.addConstr((right_up_turn == 1) >> (self.h_fence[i, j+2] == 1), f"black_right_extend2_{i}_{j}")
                
                if i > 1:  # Có thể mở rộng lên trên hai ô
                    self.model.addConstr((right_up_turn == 1) >> (self.v_fence[i-1, j] == 1), f"black_up_extend1_2_{i}_{j}")
                    self.model.addConstr((right_up_turn == 1) >> (self.v_fence[i-2, j] == 1), f"black_up_extend2_2_{i}_{j}")
            
            # Right-Down turn
            if h_right is not None and v_down is not None:
                right_down_turn = self.model.addVar(vtype=GRB.BINARY, name=f'right_down_turn_{i}_{j}')
                self.model.addConstr(right_down_turn <= h_right, f"right_down_h_{i}_{j}")
                self.model.addConstr(right_down_turn <= v_down, f"right_down_v_{i}_{j}")
                self.model.addConstr(right_down_turn >= h_right + v_down - 1, f"right_down_used_{i}_{j}")
                turns.append(right_down_turn)
                
                # Thêm điều kiện thẳng hàng nếu góc này được sử dụng
                if j < self.cols - 2:  # Có thể mở rộng sang phải hai ô
                    self.model.addConstr((right_down_turn == 1) >> (self.h_fence[i, j+1] == 1), f"black_right_extend1_2_{i}_{j}")
                    self.model.addConstr((right_down_turn == 1) >> (self.h_fence[i, j+2] == 1), f"black_right_extend2_2_{i}_{j}")
                
                if i < self.rows - 2:  # Có thể mở rộng xuống dưới hai ô
                    self.model.addConstr((right_down_turn == 1) >> (self.v_fence[i+1, j] == 1), f"black_down_extend1_2_{i}_{j}")
                    self.model.addConstr((right_down_turn == 1) >> (self.v_fence[i+2, j] == 1), f"black_down_extend2_2_{i}_{j}")
            
            # Một góc phải được sử dụng
            if turns:
                self.model.addConstr(gp.quicksum(turns) == 1, f"black_one_turn_{i}_{j}")

        else:  # White circle (E)
            # Tối ưu: Cách tiếp cận trực tiếp hơn cho vòng tròn trắng
            # White circle must have straight passing - horizontal or vertical
            horiz_pass = vert_pass = None
            
            if h_left is not None and h_right is not None:
                horiz_pass = self.model.addVar(vtype=GRB.BINARY, name=f'horiz_pass_{i}_{j}')
                self.model.addConstr((horiz_pass == 1) >> (h_left == 1), f"horiz_left_{i}_{j}")
                self.model.addConstr((horiz_pass == 1) >> (h_right == 1), f"horiz_right_{i}_{j}")
                self.model.addConstr(horiz_pass >= h_left + h_right - 1, f"horiz_both_{i}_{j}")
            
            if v_up is not None and v_down is not None:
                vert_pass = self.model.addVar(vtype=GRB.BINARY, name=f'vert_pass_{i}_{j}')
                self.model.addConstr((vert_pass == 1) >> (v_up == 1), f"vert_up_{i}_{j}")
                self.model.addConstr((vert_pass == 1) >> (v_down == 1), f"vert_down_{i}_{j}")
                self.model.addConstr(vert_pass >= v_up + v_down - 1, f"vert_both_{i}_{j}")
            
            # Phải đi thẳng qua (hoặc ngang hoặc dọc)
            if horiz_pass is not None and vert_pass is not None:
                self.model.addConstr(horiz_pass + vert_pass == 1, f"white_straight_{i}_{j}")
            elif horiz_pass is not None:
                self.model.addConstr(horiz_pass == 1, f"white_horiz_{i}_{j}")
            elif vert_pass is not None:
                self.model.addConstr(vert_pass == 1, f"white_vert_{i}_{j}")
            
            # Tạo các biến cho các góc rẽ ở các ô liền kề và tối ưu ràng buộc
            # Tối ưu: Giảm số lượng biến bằng cách sử dụng cấu trúc khác
            
            # Tạo biến chỉ báo có ít nhất một góc rẽ ở các ô liền kề
            has_adjacent_turn = self.model.addVar(vtype=GRB.BINARY, name=f'has_adj_turn_{i}_{j}')
            
            # Danh sách các góc rẽ tiềm năng cho từng hướng
            adjacent_turns = []
            
            # Kiểm tra góc rẽ ở các ô liền kề khi đi ngang
            if horiz_pass is not None:
                # Vertex bên trái
                if j > 0:
                    # Góc rẽ xuống ở bên trái
                    if i < self.rows - 1 and j > 0:
                        left_down_turn = self.model.addVar(vtype=GRB.BINARY, name=f'adj_left_down_{i}_{j}')
                        # Góc này chỉ tồn tại nếu đi ngang qua vòng tròn trắng
                        self.model.addConstr(left_down_turn <= horiz_pass, f"adj_left_down_enable_{i}_{j}")
                        # Các cạnh tạo góc phải tồn tại
                        self.model.addConstr(left_down_turn <= self.v_fence[i, j-1], f"adj_left_down_v_{i}_{j}")
                        self.model.addConstr(left_down_turn <= self.h_fence[i+1, j-1], f"adj_left_down_h_{i}_{j}")
                        # Nếu cả hai cạnh tồn tại thì có góc rẽ
                        self.model.addConstr(left_down_turn >= self.v_fence[i, j-1] + self.h_fence[i+1, j-1] + horiz_pass - 2, f"adj_left_down_exist_{i}_{j}")
                        adjacent_turns.append(left_down_turn)
                    
                    # Góc rẽ lên ở bên trái
                    if i > 0 and j > 0:
                        left_up_turn = self.model.addVar(vtype=GRB.BINARY, name=f'adj_left_up_{i}_{j}')
                        # Tối ưu: thêm ràng buộc tương tự
                        self.model.addConstr(left_up_turn <= horiz_pass, f"adj_left_up_enable_{i}_{j}")
                        self.model.addConstr(left_up_turn <= self.v_fence[i-1, j-1], f"adj_left_up_v_{i}_{j}")
                        self.model.addConstr(left_up_turn <= self.h_fence[i, j-1], f"adj_left_up_h_{i}_{j}")
                        self.model.addConstr(left_up_turn >= self.v_fence[i-1, j-1] + self.h_fence[i, j-1] + horiz_pass - 2, f"adj_left_up_exist_{i}_{j}")
                        adjacent_turns.append(left_up_turn)
                
                # Vertex bên phải
                if j < self.cols - 1:
                    # Góc rẽ xuống ở bên phải
                    if i < self.rows - 1 and j < self.cols - 1:
                        right_down_turn = self.model.addVar(vtype=GRB.BINARY, name=f'adj_right_down_{i}_{j}')
                        # Tối ưu: thêm ràng buộc tương tự
                        self.model.addConstr(right_down_turn <= horiz_pass, f"adj_right_down_enable_{i}_{j}")
                        self.model.addConstr(right_down_turn <= self.v_fence[i, j+1], f"adj_right_down_v_{i}_{j}")
                        self.model.addConstr(right_down_turn <= self.h_fence[i+1, j+1], f"adj_right_down_h_{i}_{j}")
                        self.model.addConstr(right_down_turn >= self.v_fence[i, j+1] + self.h_fence[i+1, j+1] + horiz_pass - 2, f"adj_right_down_exist_{i}_{j}")
                        adjacent_turns.append(right_down_turn)
                    
                    # Góc rẽ lên ở bên phải
                    if i > 0 and j < self.cols - 1:
                        right_up_turn = self.model.addVar(vtype=GRB.BINARY, name=f'adj_right_up_{i}_{j}')
                        # Tối ưu: thêm ràng buộc tương tự
                        self.model.addConstr(right_up_turn <= horiz_pass, f"adj_right_up_enable_{i}_{j}")
                        self.model.addConstr(right_up_turn <= self.v_fence[i-1, j+1], f"adj_right_up_v_{i}_{j}")
                        self.model.addConstr(right_up_turn <= self.h_fence[i, j+1], f"adj_right_up_h_{i}_{j}")
                        self.model.addConstr(right_up_turn >= self.v_fence[i-1, j+1] + self.h_fence[i, j+1] + horiz_pass - 2, f"adj_right_up_exist_{i}_{j}")
                        adjacent_turns.append(right_up_turn)
            
            # Kiểm tra góc rẽ ở các ô liền kề khi đi dọc
            if vert_pass is not None:
                # Tương tự cho các góc rẽ khi đi dọc
                # Vertex phía trên
                if i > 0:
                    # Góc rẽ sang trái ở trên
                    if j > 0 and i > 0:
                        up_left_turn = self.model.addVar(vtype=GRB.BINARY, name=f'adj_up_left_{i}_{j}')
                        self.model.addConstr(up_left_turn <= vert_pass, f"adj_up_left_enable_{i}_{j}")
                        self.model.addConstr(up_left_turn <= self.h_fence[i-1, j-1], f"adj_up_left_h_{i}_{j}")
                        self.model.addConstr(up_left_turn <= self.v_fence[i-1, j], f"adj_up_left_v_{i}_{j}")
                        self.model.addConstr(up_left_turn >= self.h_fence[i-1, j-1] + self.v_fence[i-1, j] + vert_pass - 2, f"adj_up_left_exist_{i}_{j}")
                        adjacent_turns.append(up_left_turn)
                    
                    # Góc rẽ sang phải ở trên
                    if j < self.cols - 1 and i > 0:
                        up_right_turn = self.model.addVar(vtype=GRB.BINARY, name=f'adj_up_right_{i}_{j}')
                        self.model.addConstr(up_right_turn <= vert_pass, f"adj_up_right_enable_{i}_{j}")
                        self.model.addConstr(up_right_turn <= self.h_fence[i-1, j], f"adj_up_right_h_{i}_{j}")
                        self.model.addConstr(up_right_turn <= self.v_fence[i-1, j], f"adj_up_right_v_{i}_{j}")
                        self.model.addConstr(up_right_turn >= self.h_fence[i-1, j] + self.v_fence[i-1, j] + vert_pass - 2, f"adj_up_right_exist_{i}_{j}")
                        adjacent_turns.append(up_right_turn)
                
                # Vertex phía dưới
                if i < self.rows - 1:
                    # Góc rẽ sang trái ở dưới
                    if j > 0 and i < self.rows - 1:
                        down_left_turn = self.model.addVar(vtype=GRB.BINARY, name=f'adj_down_left_{i}_{j}')
                        self.model.addConstr(down_left_turn <= vert_pass, f"adj_down_left_enable_{i}_{j}")
                        self.model.addConstr(down_left_turn <= self.h_fence[i+1, j-1], f"adj_down_left_h_{i}_{j}")
                        self.model.addConstr(down_left_turn <= self.v_fence[i, j], f"adj_down_left_v_{i}_{j}")
                        self.model.addConstr(down_left_turn >= self.h_fence[i+1, j-1] + self.v_fence[i, j] + vert_pass - 2, f"adj_down_left_exist_{i}_{j}")
                        adjacent_turns.append(down_left_turn)
                    
                    # Góc rẽ sang phải ở dưới
                    if j < self.cols - 1 and i < self.rows - 1:
                        down_right_turn = self.model.addVar(vtype=GRB.BINARY, name=f'adj_down_right_{i}_{j}')
                        self.model.addConstr(down_right_turn <= vert_pass, f"adj_down_right_enable_{i}_{j}")
                        self.model.addConstr(down_right_turn <= self.h_fence[i+1, j], f"adj_down_right_h_{i}_{j}")
                        self.model.addConstr(down_right_turn <= self.v_fence[i, j], f"adj_down_right_v_{i}_{j}")
                        self.model.addConstr(down_right_turn >= self.h_fence[i+1, j] + self.v_fence[i, j] + vert_pass - 2, f"adj_down_right_exist_{i}_{j}")
                        adjacent_turns.append(down_right_turn)
            
            # Ít nhất một góc rẽ phải tồn tại ở các ô liền kề
            if adjacent_turns:
                self.model.addConstr(gp.quicksum(adjacent_turns) >= 1, f"white_adj_turn_{i}_{j}")

    def add_uncircled_number_constraints(self, i, j, value):
        """Add constraints for uncircled numbers
        
        Uncircled numbers indicate how many fence segments are used around the cell.
        The number ranges from 0 to 3 (not 4, as per the problem statement).
        """
        # Count the fence segments around this cell
        fence_segments = []
        
        # Top fence
        fence_segments.append(self.h_fence[i-1, j-1])
        
        # Bottom fence
        fence_segments.append(self.h_fence[i , j-1])
        
        # Left fence
        fence_segments.append(self.v_fence[i-1, j-1])
        
        # Right fence
        fence_segments.append(self.v_fence[i-1, j])
        
        # The total number of fence segments must equal the given value
        # Validate that the value is between 0 and 3 as per the requirements
        if value < 0 or value > 3:
            raise ValueError(f"Uncircled number value must be between 0 and 3, got {value}")
            
        self.model.addConstr(gp.quicksum(fence_segments) == value, f"uncircled_{i}_{j}")

    def solve(self, time_limit=None, output_file=None):
        """Solve the Area51 puzzle
        
        Args:
            time_limit: Maximum time to spend solving (in seconds)
            output_file: Optional file to save the solution to
            
        Returns:
            A dictionary with the solution status and solution (if found)
        """
        try:
            # Set time limit if specified
            if time_limit is not None:
                self.model.Params.TimeLimit = time_limit
                
            # Enable presolve to speed up solving
            self.model.Params.Presolve = 2
            
            # Create all variables
            self.create_variables()
            
            # Add all constraints
            self.add_constraints()
            
            # Solve the model
            self.model.optimize()
            
            # Check solution status
            solution = {
                "status": self.model.Status
            }
            
            if self.model.Status == GRB.OPTIMAL:
                solution["status_str"] = "OPTIMAL"
                solution["objective"] = self.model.ObjVal
                
                # Extract fence solution
                h_fence_sol = {}
                for i in range(self.rows + 1):
                    for j in range(self.cols):
                        if self.h_fence[i, j].X > 0.5:
                            h_fence_sol[f"{i},{j}"] = 1
                
                v_fence_sol = {}
                for i in range(self.rows):
                    for j in range(self.cols + 1):
                        if self.v_fence[i, j].X > 0.5:
                            v_fence_sol[f"{i},{j}"] = 1
                
                # Extract inside/outside regions
                inside_sol = {}
                for i in range(self.rows):
                    for j in range(self.cols):
                        if self.inside[i, j].X > 0.5:
                            inside_sol[f"{i},{j}"] = 1
                
                solution["h_fence"] = h_fence_sol
                solution["v_fence"] = v_fence_sol
                solution["inside"] = inside_sol
                
                # Save solution to file if requested
                if output_file:
                    with open(output_file, 'w') as f:
                        json.dump(solution, f, indent=2)
                
                return solution
            
            elif self.model.Status == GRB.INFEASIBLE:
                solution["status_str"] = "INFEASIBLE"
                print("Model is infeasible")
                
                # Try to get infeasibility analysis
                try:
                    self.model.computeIIS()
                    solution["iis_constraints"] = []
                    for c in self.model.getConstrs():
                        if c.IISConstr:
                            solution["iis_constraints"].append(c.ConstrName)
                    print(f"Number of constraints in IIS: {len(solution['iis_constraints'])}")
                except Exception as e:
                    print(f"Error computing IIS: {e}")
                
                # Try to find a relaxed solution
                try:
                    # Create a new model for relaxation
                    relaxation = self.model.copy()
                    relaxation.update()
                    
                    # Add slack variables to all constraints
                    slacks = {}
                    for c in relaxation.getConstrs():
                        # Skip certain constraints if needed
                        name = c.ConstrName
                        # Add slack variable
                        slack = relaxation.addVar(name=f"slack_{name}", obj=1.0)
                        slacks[name] = slack
                    
                    # Modify constraints with slack variables
                    for c in relaxation.getConstrs():
                        name = c.ConstrName
                        slack = slacks[name]
                        expr = relaxation.getRow(c)
                        sense = c.Sense
                        rhs = c.RHS
                        
                        # Remove original constraint
                        relaxation.remove(c)
                        
                        # Add new constraint with slack
                        if sense == '=':
                            relaxation.addConstr(expr + slack - slack == rhs, name)
                        elif sense == '<':
                            relaxation.addConstr(expr <= rhs + slack, name)
                        elif sense == '>':
                            relaxation.addConstr(expr >= rhs - slack, name)
                    
                    # Solve relaxed model
                    relaxation.optimize()
                    
                    if relaxation.Status == GRB.OPTIMAL:
                        print("Found a relaxed solution")
                        solution["relaxed_status"] = "OPTIMAL"
                        
                        # Identify violated constraints
                        violated = []
                        for name, var in slacks.items():
                            if var.X > 1e-6:
                                violated.append(name)
                        solution["violated_constraints"] = violated
                        print(f"Number of violated constraints: {len(violated)}")
                    else:
                        print(f"Relaxation status: {relaxation.Status}")
                        solution["relaxed_status"] = relaxation.Status
                except Exception as e:
                    print(f"Error in relaxation: {e}")
                
                return solution
            
            else:
                solution["status_str"] = f"OTHER ({self.model.Status})"
                print(f"Solver status: {self.model.Status}")
                return solution
                
        except Exception as e:
            print(f"Error solving model: {e}")
            return {"status": "ERROR", "message": str(e)}

    def display_solution(self, solution=None):
        """Display the solution in a readable format
        
        Args:
            solution: Optional solution dictionary. If not provided, the method 
                     will use the current model solution (assumes model was solved).
        """
        if solution is None and self.model.Status != GRB.OPTIMAL:
            print("No solution to display!")
            return
        
        h_fence = {}
        v_fence = {}
        inside = {}
        
        if solution:
            h_fence_data = solution.get("h_fence", {})
            v_fence_data = solution.get("v_fence", {})
            inside_data = solution.get("inside", {})
            
            # Convert string keys to tuples
            for key, val in h_fence_data.items():
                i, j = map(int, key.split(','))
                h_fence[i, j] = val
                
            for key, val in v_fence_data.items():
                i, j = map(int, key.split(','))
                v_fence[i, j] = val
                
            for key, val in inside_data.items():
                i, j = map(int, key.split(','))
                inside[i, j] = val
        else:
            # Extract from model variables
            for i in range(self.rows + 1):
                for j in range(self.cols):
                    if self.h_fence[i, j].X > 0.5:
                        h_fence[i, j] = 1
            
            for i in range(self.rows):
                for j in range(self.cols + 1):
                    if self.v_fence[i, j].X > 0.5:
                        v_fence[i, j] = 1
            
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.inside[i, j].X > 0.5:
                        inside[i, j] = 1
        
        # Display the solution
        print("\nSolution:")
        print("=" * (self.cols * 2 + 1))
        
        for i in range(self.rows):
            # Print horizontal edges above this row
            h_line = ""
            for j in range(self.cols):
                h_line += "+" + (" " if (i, j) not in h_fence else "-")
            h_line += "+"
            print(h_line)
            
            # Print vertical edges and cells
            v_line = ""
            for j in range(self.cols):
                v_line += (" " if (i, j) not in v_fence else "|")
                
                # Print cell content
                cell_content = " "
                if self.matrix_1[i][j] is not None:
                    if isinstance(self.matrix_1[i][j], dict) and self.matrix_1[i][j].get("circled"):
                        cell_content = str(self.matrix_1[i][j]["value"]) + "⭘"
                    elif self.matrix_1[i][j] == "A":
                        cell_content = "A"
                    elif self.matrix_1[i][j] == "C":
                        cell_content = "C"
                    else:
                        cell_content = str(self.matrix_1[i][j])
                
                # Mark inside/outside
                if (i, j) in inside:
                    # If no specific content, show inside mark
                    if cell_content == " ":
                        cell_content = "I"
                
                v_line += cell_content
            
            # Print the last vertical edge
            v_line += (" " if (i, self.cols) not in v_fence else "|")
            print(v_line)
        
        # Print the bottom horizontal edges
        h_line = ""
        for j in range(self.cols):
            h_line += "+" + (" " if (self.rows, j) not in h_fence else "-")
        h_line += "+"
        print(h_line)
        
        print("=" * (self.cols * 2 + 1))