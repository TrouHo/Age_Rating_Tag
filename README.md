# 🎬 Automated Film Censorship & Age Rating

Hệ thống AI tự động gắn nhãn kiểm duyệt và đề xuất độ tuổi phù hợp cho phim dựa trên trailer và metadata.

---

## 1. Mục tiêu dự án

Dự án xây dựng hệ thống gồm hai thành phần chính:

### Nhận diện hành vi gây hại trong video

- Violence (bạo lực)
- Nudity (cảnh khỏa thân / nhạy cảm)
- Crime (hành vi phạm pháp)

→ Giúp backbone học được “ý thức kiểm duyệt”.

### Dự đoán độ tuổi phù hợp cho phim

- Sử dụng trailer (vision branch)
- Sử dụng metadata (metadata branch)

### Late Fusion

Kết hợp kết quả từ hai nhánh để đưa ra phân loại cuối cùng chính xác hơn.

---

## 2. Pipeline tổng quan

### 🔹 Phase 1 — Harm-Aware Pretraining (VideoMAE Multi-task)

**Input:** các clip ngắn đã được cắt từ dataset:

- Violence: https://www.kaggle.com/datasets/frostedpilot/violence-clips-extracted
- Nudity: https://www.kaggle.com/datasets/trouho/porn-dataset
- Crime: https://www.kaggle.com/datasets/frostedpilot/ucf-crime-extracted

**Model:**

- Backbone: `VideoMAE`
- 3 classification heads tương ứng với 3 hành vi

**Output:**

- Checkpoint harm-aware VideoMAE đã biết nhận dạng hành vi gây hại

---

### 🔹 Phase 2 — Age Rating Prediction

Gồm hai nhánh độc lập:

#### 1. Vision Branch (Trailer → Age Rating)

- Input: Trailer phim, cắt thành nhiều clips
- Sử dụng backbone từ Phase 1
- Trích xuất vector đại diện cho trailer
- Dự đoán nhãn độ tuổi (P, T13, T16, T18…)

#### 2. Metadata Branch

- Input: CSV metadata của từng phim (thể loại, quốc gia, keywords…)
- Model: MLP hoặc mô hình tabular (XGBoost, Logistic Regression…)
- Dự đoán độ tuổi dựa trên đặc trưng phi hình ảnh

#### 3. Late Fusion

```bash
p_final = α * p_vision + (1 - α) * p_metadata
```

Trong đó `α` được tinh chỉnh trên validation set.

---

## 3. Mô tả vai trò từng phần

### 🔹 data/

Chỉ chứa dữ liệu local, không commit.  
`data/README.md` hướng dẫn cấu trúc folder và format clip.

---

### 🔹 scripts/

Các file chạy trực tiếp bằng terminal (train / evaluate / inference).

- `train_phase1_vision.py` → train VideoMAE multitask
- `train_phase2_vision.py` → train age rating từ trailer
- `train_metadata.py` → train meta_model từ CSV
- `eval_fusion.py` → evaluate late fusion
- `infer_single_movie.py` → chạy demo 1 phim
- `split_dataset_clips.py` → script tách clip (đã có sẵn)

---

### 🔹 src/vision/

Toàn bộ logic liên quan đến video.

**datasets.py**

- Load clip dataset phase 1
- Load trailer dataset phase 2

**videomae_backbone.py**

- Load VideoMAE pretrained
- Chuẩn hoá input và xử lý tensor
---

### 🔹 src/metadata/

Code cho nhánh metadata.

- `dataset.py` → Load CSV, xử lý one-hot / normalization
- `model.py` → Meta-model đơn giản (MLP hoặc sklearn)

---

### 🔹 src/fusion/

- `late_fusion.py` → implement công thức:  
  `p = α * p_vision + (1-α) * p_meta`

---

### 🔹 configs/

Lưu cấu hình `.yaml` để training reproducible:

- Learning rate
- Batch size
- Đường dẫn data
- Checkpoint để load

---
