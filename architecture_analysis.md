# Phân Tích Kiến Trúc Transformer

## 1. Tổng Quan Kiến Trúc

Transformer là một kiến trúc mạng neural được giới thiệu trong paper "Attention Is All You Need" (Vaswani et al., 2017). Kiến trúc này dựa hoàn toàn trên cơ chế attention, loại bỏ recurrence và convolution.

### Cấu Trúc Tổng Thể:
```
Input Embeddings + Positional Encoding
           ↓
    Encoder Stack (N layers)
           ↓ 
    Decoder Stack (N layers)
           ↓
    Linear + Softmax
           ↓
        Output
```

## 2. Các Thành Phần Chính

### 2.1 Positional Encoding
**Mục đích**: Cung cấp thông tin về vị trí của các token trong sequence vì Transformer không có cơ chế tuần tự như RNN.

**Công thức**:
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

**Ưu điểm**:
- Cho phép model học được vị trí tương đối
- Có thể xử lý sequences dài hơn so với training
- Không cần học thêm parameters

### 2.2 Multi-Head Attention
**Ý tưởng cốt lõi**: Thay vì thực hiện một attention function với d_model dimensions, ta thực hiện h attention functions song song với d_model/h dimensions.

**Công thức**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Ưu điểm**:
- Cho phép model attend đến thông tin từ nhiều representation subspaces khác nhau
- Mỗi head có thể học các loại relationships khác nhau
- Tính song song cao

### 2.3 Scaled Dot-Product Attention
**Công thức**: Attention(Q, K, V) = softmax(QK^T / √d_k)V

**Tại sao chia cho √d_k?**
- Khi d_k lớn, dot products có magnitude lớn
- Đẩy softmax vào vùng có gradients nhỏ
- Scaling giúp ổn định gradients

### 2.4 Feed-Forward Networks
**Cấu trúc**: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

**Đặc điểm**:
- Áp dụng cho mỗi position độc lập
- Thường d_ff = 4 × d_model
- Sử dụng ReLU activation

## 3. Encoder Stack

### Cấu Trúc Mỗi Encoder Layer:
1. **Multi-Head Self-Attention**
   - Input: (Q=K=V=input)
   - Cho phép mỗi position attend đến tất cả positions trong input sequence
   
2. **Residual Connection + Layer Normalization**
   - output = LayerNorm(input + Sublayer(input))
   - Giúp training ổn định và sâu hơn

3. **Position-wise Feed-Forward**
   - Áp dụng transformation phi tuyến
   - Tăng khả năng biểu diễn của model

4. **Residual Connection + Layer Normalization**

### Vai Trò:
- Encode input sequence thành representations
- Mỗi layer học các patterns phức tạp hơn
- Output được sử dụng làm K, V cho decoder

## 4. Decoder Stack

### Cấu Trúc Mỗi Decoder Layer:
1. **Masked Multi-Head Self-Attention**
   - Prevent attention to future positions
   - Đảm bảo tính autoregressive

2. **Residual Connection + Layer Normalization**

3. **Multi-Head Cross-Attention**
   - Q từ decoder, K,V từ encoder
   - Cho phép decoder attend đến encoder output

4. **Residual Connection + Layer Normalization**

5. **Position-wise Feed-Forward**

6. **Residual Connection + Layer Normalization**

### Masking:
- **Padding Mask**: Ignore padding tokens
- **Look-ahead Mask**: Prevent looking at future tokens

## 5. Ưu Điểm Của Kiến Trúc

### 5.1 Parallelization
- Tất cả positions được xử lý song song
- Không cần sequential computation như RNN
- Training nhanh hơn đáng kể

### 5.2 Long-range Dependencies
- Direct connections giữa bất kỳ hai positions nào
- Path length = 1 (so với O(n) trong RNN)
- Giải quyết vanishing gradient problem

### 5.3 Interpretability
- Attention weights có thể visualize
- Hiểu được model attend vào đâu
- Debugging và analysis dễ dàng

## 6. Computational Complexity

### Self-Attention: O(n²·d)
- n: sequence length
- d: model dimension
- Quadratic với sequence length

### RNN: O(n·d²)
- Linear với sequence length
- Quadratic với model dimension

### Trade-off:
- Transformer nhanh hơn khi n < d
- RNN có thể hiệu quả hơn với sequences rất dài

## 7. Biến Thể và Cải Tiến

### 7.1 Layer Normalization Placement
- **Post-norm**: Norm sau residual (original paper)
- **Pre-norm**: Norm trước sublayer (training ổn định hơn)

### 7.2 Attention Optimizations
- **Sparse Attention**: Giảm complexity từ O(n²) xuống O(n√n)
- **Linear Attention**: Approximation để đạt O(n)
- **Local Attention**: Chỉ attend trong window cố định

### 7.3 Positional Encoding Variants
- **Learned Positional Embeddings**
- **Relative Positional Encoding**
- **RoPE (Rotary Position Embedding)**

## 8. Ứng Dụng

### 8.1 Natural Language Processing
- Machine Translation
- Text Summarization
- Question Answering
- Language Modeling

### 8.2 Computer Vision
- Vision Transformer (ViT)
- Object Detection
- Image Classification

### 8.3 Multimodal
- CLIP (Text-Image)
- DALL-E (Text-to-Image)
- GPT-4V (Vision-Language)

## 9. Hạn Chế

### 9.1 Computational Cost
- Quadratic complexity với sequence length
- Memory intensive cho long sequences

### 9.2 Lack of Inductive Bias
- Không có built-in understanding về sequence order
- Cần nhiều data để học structural patterns

### 9.3 Training Instability
- Có thể khó train với very deep architectures
- Sensitive to hyperparameter choices

## 10. Kết Luận

Transformer đã cách mạng hóa deep learning với:
- **Hiệu quả**: Parallelization và faster training
- **Hiệu suất**: State-of-the-art results trên nhiều tasks
- **Linh hoạt**: Có thể áp dụng cho nhiều domains khác nhau
- **Khả năng mở rộng**: Scale tốt với data và compute

Kiến trúc này đã trở thành foundation cho các models lớn hiện đại như GPT, BERT, T5, và nhiều breakthrough khác trong AI.