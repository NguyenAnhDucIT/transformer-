import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from transformer_model import Transformer
import time
import math


class SyntheticTranslationDataset(Dataset):
    
    def __init__(self, num_samples=10000, seq_len=10, vocab_size=100):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        self.src_data = []
        self.tgt_data = []
        
        for _ in range(num_samples):
            src_seq = torch.randint(1, vocab_size-1, (seq_len,))
            
            tgt_seq = (src_seq + 1) % vocab_size
            
            src_seq = torch.cat([src_seq, torch.tensor([vocab_size])])
            tgt_seq = torch.cat([torch.tensor([vocab_size]), tgt_seq, torch.tensor([vocab_size])])
            
            self.src_data.append(src_seq)
            self.tgt_data.append(tgt_seq)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_batch, tgt_batch


class LabelSmoothingLoss(nn.Module):
    
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=0):
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        """
        Args:
            pred: [batch_size, seq_len, vocab_size] - predicted logits
            target: [batch_size, seq_len] - ground truth labels
        """
        batch_size, seq_len, vocab_size = pred.shape
        
        # Reshape for cross entropy calculation
        pred = pred.view(-1, vocab_size)
        target = target.view(-1)
        
        # Create smoothed target distribution
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Mask padding tokens
        mask = (target != self.ignore_index).unsqueeze(1).float()
        true_dist = true_dist * mask
        
        # Calculate loss
        log_pred = F.log_softmax(pred, dim=1)
        loss = -torch.sum(true_dist * log_pred, dim=1)
        
        # Average over non-padding tokens
        return loss.sum() / mask.sum()


class TransformerTrainer:
    """
    Trainer class for Transformer model
    """
    
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function with label smoothing
        self.criterion = LabelSmoothingLoss(vocab_size=model.tgt_embedding.num_embeddings, 
                                          smoothing=0.1)
        
        # Optimizer with learning rate scheduling
        self.optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (src, tgt) in enumerate(self.train_loader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # Prepare decoder input and target
            tgt_input = tgt[:, :-1]  # Remove last token for input
            tgt_output = tgt[:, 1:]  # Remove first token for target
            
            # Create target mask
            tgt_mask = self.model.generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input, tgt_mask=tgt_mask)
            
            # Calculate loss
            loss = self.criterion(output, tgt_output)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for src, tgt in self.val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                tgt_mask = self.model.generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)
                
                output = self.model(src, tgt_input, tgt_mask=tgt_mask)
                loss = self.criterion(output, tgt_output)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs):
        """Train the model for specified number of epochs"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)
            
            # Print epoch results
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Learning Rate: {current_lr:.6f}')
            print(f'  Time: {epoch_time:.2f}s')
            print('-' * 50)
    
    def plot_training_progress(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot
        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()


def calculate_bleu_score(predictions, targets, vocab_size):
    """
    Simple BLEU score calculation for evaluation
    """
    # Convert predictions to token indices
    pred_tokens = torch.argmax(predictions, dim=-1)
    
    # Calculate exact match accuracy
    mask = (targets != 0)  # Ignore padding tokens
    correct = (pred_tokens == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def inference_example(model, dataset, device, num_examples=5):
    """
    Show inference examples
    """
    model.eval()
    
    print("\n" + "="*60)
    print("INFERENCE EXAMPLES")
    print("="*60)
    
    with torch.no_grad():
        for i in range(num_examples):
            src, tgt = dataset[i]
            src = src.unsqueeze(0).to(device)  # Add batch dimension
            tgt_input = tgt[:-1].unsqueeze(0).to(device)  # Remove EOS, add batch dim
            tgt_output = tgt[1:].to(device)  # Remove BOS
            
            # Create mask
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            # Forward pass
            output = model(src, tgt_input, tgt_mask=tgt_mask)
            predictions = torch.argmax(output, dim=-1).squeeze(0)
            
            # Calculate accuracy
            mask = (tgt_output != 0)
            correct = (predictions == tgt_output) & mask
            accuracy = correct.sum().float() / mask.sum().float()
            
            print(f"\nExample {i+1}:")
            print(f"Source:     {src.squeeze().cpu().numpy()}")
            print(f"Target:     {tgt_output.cpu().numpy()}")
            print(f"Predicted:  {predictions.cpu().numpy()}")
            print(f"Accuracy:   {accuracy:.2%}")


def main():
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    vocab_size = 102  # 100 tokens + PAD + EOS
    d_model = 256
    num_heads = 8
    num_layers = 4
    d_ff = 1024
    dropout = 0.1
    
    # Create model
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create datasets
    train_dataset = SyntheticTranslationDataset(num_samples=8000, seq_len=10, vocab_size=100)
    val_dataset = SyntheticTranslationDataset(num_samples=2000, seq_len=10, vocab_size=100)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Create trainer
    trainer = TransformerTrainer(model, train_loader, val_loader, device)
    
    # Train model
    trainer.train(num_epochs=5)
    
    # Plot results
    trainer.plot_training_progress()
    
    # Show inference examples
    inference_example(model, val_dataset, device)
    
    # Save model
    torch.save(model.state_dict(), 'transformer_model.pth')
    print("\nModel saved as 'transformer_model.pth'")


if __name__ == "__main__":
    main()