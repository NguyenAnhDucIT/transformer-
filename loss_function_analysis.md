# Phân Tích Hàm Mất Mát (Loss Functions) trong Transformer

## 1. Tổng Quan về Loss Functions

Trong Transformer, việc lựa chọn và thiết kế hàm mất mát là yếu tố quan trọng quyết định hiệu suất training và chất lượng model. Chúng ta sẽ phân tích các hàm loss chính được sử dụng.

## 2. Cross-Entropy Loss (Cơ Bản)

### 2.1 Công Thức
```
L = -∑(i=1 to N) y_i * log(p_i)
```
Trong đó:
- y_i: ground truth (one-hot vector)
- p_i: predicted probability
- N: vocabulary size

### 2.2 Ưu Điểm
- Đơn giản và hiệu quả
- Gradient tốt cho classification
- Phù hợp với softmax activation

### 2.3 Nhược Điểm
- Có thể dẫn đến overconfidence
- Không xử lý được uncertainty
- Sensitive với outliers

### 2.4 Implementation
```python
def cross_entropy_loss(pred, target):
    """
    pred: [batch_size, seq_len, vocab_size]
    target: [batch_size, seq_len]
    """
    return F.cross_entropy(pred.view(-1, pred.size(-1)), 
                          target.view(-1), 
                          ignore_index=0)
```

## 3. Label Smoothing Loss

### 3.1 Motivation
Cross-entropy loss có thể khiến model quá tự tin, dẫn đến:
- Overfitting
- Poor calibration
- Reduced generalization

### 3.2 Công Thức
```
y_smooth = (1 - α) * y_true + α/K
```
Trong đó:
- α: smoothing parameter (thường 0.1)
- K: số classes
- y_true: ground truth labels

### 3.3 Loss Calculation
```
L_smooth = -∑(i=1 to K) y_smooth_i * log(p_i)
```

### 3.4 Ưu Điểm
- **Regularization Effect**: Giảm overfitting
- **Better Calibration**: Predictions ít overconfident hơn
- **Improved Generalization**: Performance tốt hơn trên test set
- **Smoother Training**: Gradients ổn định hơn

### 3.5 Implementation Detail
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        # Tạo smooth distribution
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Mask padding tokens
        mask = (target != self.ignore_index).unsqueeze(1).float()
        true_dist = true_dist * mask
        
        # Calculate loss
        log_pred = F.log_softmax(pred, dim=1)
        loss = -torch.sum(true_dist * log_pred, dim=1)
        
        return loss.sum() / mask.sum()
```

## 4. Focal Loss

### 4.1 Motivation
- Giải quyết class imbalance
- Focus vào hard examples
- Giảm weight của easy examples

### 4.2 Công Thức
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```
Trong đó:
- p_t: predicted probability cho true class
- α_t: balancing factor
- γ: focusing parameter (thường 2)

### 4.3 Ưu Điểm
- **Hard Example Mining**: Tự động focus vào difficult cases
- **Class Imbalance**: Xử lý tốt unbalanced data
- **Better Convergence**: Training ổn định hơn

### 4.4 Implementation
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, 
                                 ignore_index=self.ignore_index, 
                                 reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

## 5. Sequence-Level Loss Functions

### 5.1 BLEU-based Loss
**Mục đích**: Optimize directly cho evaluation metric

```python
def bleu_loss(pred_tokens, target_tokens):
    # Approximate BLEU score
    bleu_score = calculate_bleu(pred_tokens, target_tokens)
    return 1.0 - bleu_score
```

### 5.2 Minimum Risk Training (MRT)
**Ý tưởng**: Minimize expected risk under model distribution

```python
def mrt_loss(model, src, references):
    # Sample multiple outputs
    samples = model.sample(src, num_samples=10)
    
    # Calculate risk for each sample
    risks = []
    for sample in samples:
        risk = 1.0 - bleu_score(sample, references)
        risks.append(risk)
    
    # Weight by model probability
    log_probs = model.log_prob(samples)
    weights = F.softmax(log_probs, dim=0)
    
    return torch.sum(weights * torch.tensor(risks))
```

## 6. Masking và Padding Handling

### 6.1 Padding Mask
```python
def create_padding_mask(seq, pad_token=0):
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)
```

### 6.2 Loss Masking
```python
def masked_loss(pred, target, mask):
    loss = F.cross_entropy(pred.view(-1, pred.size(-1)), 
                          target.view(-1), 
                          reduction='none')
    masked_loss = loss * mask.view(-1).float()
    return masked_loss.sum() / mask.sum().float()
```

## 7. Advanced Loss Techniques

### 7.1 Scheduled Sampling
**Problem**: Exposure bias - model chỉ thấy ground truth during training

**Solution**: Thỉnh thoảng sử dụng model's own predictions

```python
def scheduled_sampling_loss(model, src, tgt, teacher_forcing_ratio):
    outputs = []
    input_token = tgt[:, 0]  # Start token
    
    for t in range(1, tgt.size(1)):
        output = model.decode_step(input_token, src)
        outputs.append(output)
        
        # Decide whether to use teacher forcing
        if random.random() < teacher_forcing_ratio:
            input_token = tgt[:, t]  # Ground truth
        else:
            input_token = output.argmax(dim=-1)  # Model prediction
    
    outputs = torch.stack(outputs, dim=1)
    return F.cross_entropy(outputs.view(-1, outputs.size(-1)), 
                          tgt[:, 1:].contiguous().view(-1))
```

### 7.2 Contrastive Loss
**Ý tưởng**: Học representations tốt hơn bằng cách contrast positive và negative pairs

```python
def contrastive_loss(embeddings, labels, temperature=0.1):
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Create positive mask
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    # Calculate loss
    exp_sim = torch.exp(similarity_matrix)
    sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
    log_prob = similarity_matrix - torch.log(sum_exp_sim)
    
    loss = -torch.sum(positive_mask * log_prob) / positive_mask.sum()
    return loss
```

## 8. Loss Function Selection Guidelines

### 8.1 Task-Specific Considerations

**Machine Translation:**
- Label Smoothing Loss (primary)
- BLEU-based objectives (fine-tuning)

**Language Modeling:**
- Cross-Entropy với perplexity tracking
- Focal Loss cho rare tokens

**Classification:**
- Focal Loss cho imbalanced data
- Label Smoothing cho better calibration

### 8.2 Training Stage Considerations

**Early Training:**
- Simpler losses (Cross-Entropy)
- Higher learning rates

**Fine-tuning:**
- Task-specific losses
- Lower learning rates
- Sequence-level objectives

## 9. Loss Monitoring và Debugging

### 9.1 Key Metrics
```python
def compute_training_metrics(pred, target, loss):
    # Perplexity
    perplexity = torch.exp(loss)
    
    # Accuracy
    pred_tokens = pred.argmax(dim=-1)
    mask = (target != 0)
    accuracy = (pred_tokens == target)[mask].float().mean()
    
    # Top-k accuracy
    _, top_k_pred = pred.topk(k=5, dim=-1)
    top_k_acc = (top_k_pred == target.unsqueeze(-1)).any(dim=-1)[mask].float().mean()
    
    return {
        'loss': loss.item(),
        'perplexity': perplexity.item(),
        'accuracy': accuracy.item(),
        'top_5_accuracy': top_k_acc.item()
    }
```

### 9.2 Loss Visualization
```python
import matplotlib.pyplot as plt

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
```

## 10. Best Practices

### 10.1 Loss Function Design
1. **Start Simple**: Begin với Cross-Entropy
2. **Add Regularization**: Label Smoothing cho better generalization
3. **Task-Specific**: Customize cho specific requirements
4. **Monitor Carefully**: Track multiple metrics

### 10.2 Hyperparameter Tuning
- **Smoothing Factor**: 0.1 là good starting point
- **Focal Loss γ**: 2.0 cho most cases
- **Temperature**: 0.1 cho contrastive learning

### 10.3 Common Pitfalls
- **Ignoring Padding**: Always mask padding tokens
- **Wrong Reduction**: Pay attention to reduction method
- **Scale Issues**: Normalize losses appropriately
- **Gradient Explosion**: Use gradient clipping

## 11. Kết Luận

Hàm mất mát trong Transformer không chỉ là tool để optimize model mà còn là:

- **Regularization Mechanism**: Label smoothing, focal loss
- **Training Stabilizer**: Proper masking, gradient clipping
- **Performance Booster**: Task-specific objectives
- **Debugging Tool**: Monitoring training progress

Việc lựa chọn và thiết kế loss function phù hợp là critical cho thành công của Transformer models, đòi hỏi understanding sâu về task requirements và model behavior.