# Báo cáo Ngày 3 — Tracking Annotation

Chép file này thành `reports/REPORT.md` rồi điền. Giữ nguyên các tiêu đề.

Họ tên / nhóm: Tưởng Đức Tâm
Ngày: 15/09/2026

---

## 1. Quá trình gán nhãn

| Mục | Giá trị |
| --- | --- |
| Công cụ | CVAT |
| Thời gian gán `clip_02` (warm-up) | 15 phút |
| Thời gian gán `clip_01` | 30 phút |
| Số track đã vẽ trong `clip_01` | 8 |
| Số keyframe trung bình mỗi track | 68 |

Ba tình huống khó nhất khi gán clip này, và bạn xử lý thế nào:

1. `...`
2. `...`
3. `...`

## 2. Tự kiểm và kiểm chéo

Ba lượt tua bắt được gì (lượt 1 nhìn ID, lượt 2 frame đầu/cuối, lượt 3 frame giữa):

- Lượt 1: Không có lỗi IDSW
- Lượt 2: Không có lỗi FN, FP
- Lượt 3: BBox không lỗi

Kiểm chéo với: `...`. Chi tiết ở `reports/review_partner.md`.
Số lỗi bạn tìm được trong bản của bạn ấy: `...`. Số lỗi bạn ấy tìm được trong bản của bạn: `...`.

Ca nào hai người quyết khác nhau, và luật nào còn thiếu trong `GUIDELINE_MINI.md`?

`...`

## 3. Chấm với gold — trước và sau rework
==========================================================================
  nhãn của bạn  vs  gold clip_01
  clip: clip_01 · 190 frame · IoU ngưỡng 0.5
==========================================================================
  HOTA 0.808   = sqrt(DetA x AssA)   -- điểm tổng của tracking
    DetA 0.795  tìm đúng vật thể chưa (giống bài toán Ngày 2)
    AssA 0.822  giữ đúng ID chưa      (phần riêng của tracking)
    LocA 0.880  bbox khít đến đâu
--------------------------------------------------------------------------
  IDF1 0.957   MOTA 0.916   MOTP 0.869
  FP 11  FN 37  ID switch 0   |  bbox gold 573  bbox của bạn 547  |  track gold 8  track của bạn 8
==========================================================================
  CỔNG QUA BÀI
    [ĐẠT ] IDF1  0.957  (cần >= 0.80)
    [ĐẠT ] MOTA  0.916  (cần >= 0.75)
    [ĐẠT ] MOTP  0.869  (cần >= 0.70)
  => ĐẠT — sang bước chạy model
==========================================================================

  3. BBOX TREO / BBOX THỪA — ID của bạn tồn tại ở nơi không có vật thể  (2)
    - ID 4: còn bbox sau khi track tham chiếu 4 đã rời khung (frame 149-151, 3 frame) -> bấm outside đúng frame xe rời khung
    - ID 8: còn bbox sau khi track tham chiếu 8 đã rời khung (frame 169-171, 3 frame) -> bấm outside đúng frame xe rời khung

  4. BBOX TRÔI — bbox lệch khỏi vật thể, thường ở giữa hai keyframe  (2)
    - frame 96: track gold 5 chỉ còn IoU 0.59 -> thêm keyframe quanh đây
    - frame 86: track gold 5 chỉ còn IoU 0.59 -> thêm keyframe quanh đây

  6. THIẾU ĐOẠN — track có nhãn nhưng không phủ hết quãng đời  (1)
    - track gold 6: mới phủ 44/56 frame (79%)

  Đã ghi /content/Day3-Lab/outputs/eval_vs_gold.json

Qua cổng (`IDF1 >= 0.80`, `MOTA >= 0.75`, `MOTP >= 0.70`): có

Sau khi đọc danh sách lỗi, bạn đã sửa cụ thể những gì? Ghi theo frame và ID:

| Loại lỗi | Frame | ID | Đã sửa thế nào |
| --- | --- | --- | --- |
| | | | |
| | | | |
| | | | |

## 4. Kết quả model và so sánh ba chiều

Cấu hình: model yolo26n.pt, tracker bytetrack.yaml, conf 0.25, imgsz 190 frame

                    HOTA    DetA    AssA    LocA    IDF1    MOTA    MOTP      FP      FN    IDSW
------------------------------------------------------------------------------------------------
ban_vs_gold        0.808   0.795   0.822   0.880   0.957   0.916   0.869      11      37       0
model_vs_gold      0.709   0.649   0.776   0.846   0.875   0.749   0.823      88      54       2
model_vs_ban       0.751   0.684   0.828   0.880   0.879   0.746   0.864      99      39       1

Cổng annotation: ĐẠT {'IDF1': 0.957, 'MOTA': 0.916, 'MOTP': 0.869}

## 5. Phân tích — năm câu hỏi

**1. MOTA của bạn cao hơn hay thấp hơn IDF1? Nếu MOTA cao mà IDF1 thấp thì điều đó nói gì, và vì sao MOTA không phạt nặng lỗi ID?**

1. MOTA (0.746) thấp hơn IDF1 (0.879), chênh 0.133.
Nếu một model có MOTA cao nhưng IDF1 thấp, nghĩa là nó thường phát hiện xe đúng (ít FP/FN), nhưng không duy trì danh tính xe ổn định xuyên suốt video — ví dụ cùng một xe bị gán ID mới sau khi bị che khuất.
MOTA không phạt nặng lỗi ID vì công thức của nó chủ yếu cộng lỗi theo số lần:
MOTA = 1 -(FN + FP + IDSW) / GT
Mỗi lần đổi ID chỉ đóng góp một IDSW; trong khi IDF1 đánh giá mức độ khớp danh tính trên toàn bộ các frame/tracklet, nên một ID bị phân mảnh có thể làm giảm IDF1 đáng kể dù số IDSW không nhiều. Ở đây chỉ có 1 IDSW, nên IDF1 rất tốt.

**2. DetA và AssA của model lệch nhau bao nhiêu? Cái nào kéo HOTA xuống — model không tìm ra xe, hay tìm ra rồi nhưng đánh mất ID?**

2. DetA = 0.684, AssA = 0.828, lệch:
0.828 - 0.684 = 0.144
DetA thấp hơn AssA 0.144, nên phần kéo HOTA (0.751) xuống chủ yếu là detection: model bỏ sót xe hoặc có phát hiện thừa, đặc biệt có 39 FN và 99 FP.
Nói ngắn gọn: model đã tìm được xe thì giữ ID khá tốt; điểm cần cải thiện là khả năng phát hiện xe ổn định và chính xác hơn.

**3. Một chỗ bạn đúng và model sai (frame, ID, vì sao):**

 frame  chỉ model có    chỉ bạn có   khác ID
   111             3             1         0
   139             2             2         0
   140             1             2         0
   149             1             2         0
   150             1             2         0
   151             1             2         0
   167             2             1         0
    56             2             0         0

**4. Một chỗ model đúng và bạn sai (frame, ID, vì sao):**

 frame  chỉ model có    chỉ bạn có   khác ID
   111             3             1         0
   139             2             2         0
   140             1             2         0
   149             1             2         0
   150             1             2         0
   151             1             2         0
   167             2             1         0
    56             2             0         0

**5. Trong ba loại bất đồng giữa bạn và model, loại nào nhiều nhất? Nó nói gì về clip này?**

Tôi vẫn track vật thể khi chỉ còn 1 phần nhỏ chưa ra khỏi khung hình, còn model thì đánh dấu vật thể đã outside khi không còn nhận dạng được vật thể nữa

## 6. Nếu phải gán thêm 10 clip nữa

Bạn sẽ sửa gì trong `GUIDELINE_MINI.md`, và đổi gì trong quy trình làm việc của mình?

Quy trình: Chọn vật thể, tua nhanh 10 Frame, hoặc 5 nếu clip ngắn, chọn track, tua ngược lại 10 frame để kiểm tra

## 7. Tệp đã nộp

- [x] `annotations/clip_01/gt.txt`
- [x] `annotations/clip_02/gt.txt`
- [x] `GUIDELINE_MINI.md` đã điền
- [x] `outputs/eval_vs_gold.json`
- [x] `outputs/model_clip_01.txt`
- [x] `outputs/eval_model_vs_gold.json`, `outputs/eval_model_vs_me.json`
- [x] `reports/review_partner.md`
- [x] `reports/REPORT.md` (file này)
