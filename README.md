# Area51 Fence Puzzle Solver

Đây là mã nguồn để giải bài toán "Area 51" về việc tạo một hàng rào khép kín trên một lưới ô vuông, tuân theo các quy tắc và ràng buộc.

## Quy tắc của bài toán

- **Mục tiêu**: Xây dựng một hàng rào đóng kín (vòng kín không tự cắt) để giữ người ngoài hành tinh (Aliens) ở bên trong và ngăn cây xương rồng (Triffids) không vào bên trong.
- **Các ký hiệu**:
  - **A** (Aliens): Phải nằm bên trong hàng rào.
  - **C** (Cactus/Triffids): Phải nằm bên ngoài hàng rào.
  - **Số không được khoanh tròn** (1, 2, 3, 4): Chỉ ra số đoạn hàng rào được sử dụng xung quanh ô đó.
  - **Số được khoanh tròn**: Luôn ở bên trong hàng rào; chỉ ra tổng số ô có thể nhìn thấy từ ô đó theo 4 hướng (Bắc, Nam, Đông, Tây), bao gồm cả ô đó.
  - **Chấm đen** (F): Hàng rào đi qua chấm đen phải rẽ góc 90° và kéo dài thẳng hai ô ở cả hai hướng.
  - **Chấm trắng** (E): Hàng rào đi qua chấm trắng phải đi thẳng, nhưng phải rẽ góc 90° ở ít nhất một trong hai ô kề bên.

## Cấu trúc dữ liệu

Dữ liệu được lưu trữ trong file JSON với hai ma trận:

1. **matrix_1**: Ma trận chứa thông tin về các ô, với giá trị:
   - `null`: Ô trống
   - Số nguyên (1, 2, 3): Số không khoanh tròn
   - `{ "value": số, "circled": true }`: Số được khoanh tròn
   - `"A"`: Người ngoài hành tinh (Aliens)
   - `"C"`: Cây xương rồng (Triffids)

2. **matrix_2**: Ma trận chứa thông tin về các điểm giao của lưới, với giá trị:
   - `null`: Điểm trống
   - `"F"`: Chấm đen (black circle)
   - `"E"`: Chấm trắng (white circle)

## Cài đặt

```bash
# Tạo và kích hoạt môi trường ảo (tùy chọn)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

## Sử dụng

### Chạy giải bài toán

```bash
# Chạy với file puzzle mặc định (data/puzzle_1.json)
python src/main.py

# Chỉ định file puzzle
python src/main.py --puzzle data/puzzle_2.json

# Lưu kết quả vào file
python src/main.py --output solution.json

# Hiển thị kết quả
python src/main.py --visualize

# Đặt giới hạn thời gian giải (giây)
python src/main.py --time-limit 60
```

### Tạo file puzzle mới

Để tạo một file puzzle mới, bạn cần tạo một file JSON theo cấu trúc sau:

```json
{
  "matrix_1": [
    [null, 2, null, null, ...],
    ...
  ],
  "matrix_2": [
    [null, null, "E", null, ...],
    ...
  ]
}
```

## Phương pháp giải

Bài toán được mô hình hóa dưới dạng bài toán quy hoạch tuyến tính nguyên (Integer Linear Programming) và giải bằng thư viện Gurobi:

1. **Biến quyết định**:
   - Biến nhị phân cho hàng rào ngang và dọc
   - Biến nhị phân xác định một ô nằm bên trong hay bên ngoài hàng rào

2. **Ràng buộc**:
   - Ràng buộc hàng rào tạo thành một vòng kín (mỗi đỉnh có 0 hoặc 2 cạnh kề)
   - Ràng buộc liên thông của vòng hàng rào (không có các vòng tách rời)
   - Ràng buộc xác định bên trong/ngoài hàng rào dựa vào số lần giao cắt với tia 
   - Ràng buộc liên thông của vùng bên trong và bên ngoài
   - Ràng buộc cho các ký hiệu đặc biệt (A, C, số, chấm đen, chấm trắng)

## Kết quả

Kết quả giải sẽ được hiển thị dưới dạng lưới ô vuông với hàng rào được biểu diễn bằng các ký tự `-` (ngang) và `|` (dọc).

## Yêu cầu

- Python 3.6+
- Gurobi Optimizer với license hợp lệ
- NumPy

## Tác giả

Dự án được phát triển để giải bài toán Area51 Fence Puzzle.
