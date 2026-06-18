"""
MatRIS trainer class.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from onescience.models.matris import MatRIS
from onescience.datapipes.materials.matris import RadiusGraph
from onescience.utils.matris.loss import MatrisLoss
from onescience.utils.matris.metrics import compute_metrics, Metrics


class MatrisTrainer:
    """Trainer for MatRIS model."""
    
    def __init__(
        self,
        model: MatRIS,
        optimizer: Optimizer,
        loss_fn: MatrisLoss | None = None,
        scheduler: LRScheduler | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_grad_norm: float | None = None,
        checkpoint_dir: str | None = None,
        log_interval: int = 10,
        task: str = "ef",
        is_distributed: bool = False,
    ):
        """
        Args:
            model: MatRIS model
            optimizer: Optimizer
            loss_fn: Loss function (if None, creates default)
            scheduler: Learning rate scheduler
            device: Device to use
            max_grad_norm: Max gradient norm for clipping
            checkpoint_dir: Directory to save checkpoints
            log_interval: Log every N batches
            task: Task type
            is_distributed: Whether using distributed training
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn or MatrisLoss(task=task)
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.task = task
        self.is_distributed = is_distributed
        
        # Create checkpoint directory
        if checkpoint_dir and (not is_distributed or dist.get_rank() == 0):
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Wrap model for distributed training
        if is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[torch.cuda.current_device()],
            )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dict of average losses
        """
        self.model.train()
        total_losses = {}
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (graphs, targets) in enumerate(train_loader):
            # Move targets to device
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in targets.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(graphs, task=self.task, is_training=True)
            
            # Compute loss
            losses = self.loss_fn(predictions, targets)
            loss = losses["total"]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
            
            self.optimizer.step()
            
            # Update scheduler if per-step
            if self.scheduler is not None and hasattr(self.scheduler, 'step_per_batch'):
                self.scheduler.step()
            
            # Accumulate losses
            for key, val in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += val.item()
            num_batches += 1
            self.global_step += 1
            
            # Log
            if batch_idx % self.log_interval == 0:
                logging.info(
                    f"Epoch {self.epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}"
                )
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        avg_losses['epoch_time'] = time.time() - start_time
        
        return avg_losses
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> tuple[dict[str, float], dict[str, Metrics]]:
        """
        Evaluate on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (losses dict, metrics dict)
        """
        self.model.eval()
        total_losses = {}
        all_metrics = {}
        num_batches = 0
        
        for graphs, targets in val_loader:
            # Move targets to device
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in targets.items()}
            
            # Forward pass
            predictions = self.model(graphs, task=self.task, is_training=False)
            
            # Compute loss
            losses = self.loss_fn(predictions, targets)
            
            # Accumulate losses
            for key, val in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += val.item()
            
            # Compute metrics
            batch_metrics = compute_metrics(predictions, targets, self.task)
            for key, metrics in batch_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = Metrics()
                all_metrics[key].mae = (all_metrics[key].mae * all_metrics[key].count + 
                                        metrics.mae * metrics.count) / (all_metrics[key].count + metrics.count)
                all_metrics[key].rmse = (all_metrics[key].rmse * all_metrics[key].count + 
                                         metrics.rmse * metrics.count) / (all_metrics[key].count + metrics.count)
                all_metrics[key].count += metrics.count
            
            num_batches += 1
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses, all_metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_epochs: int = 100,
        patience: int | None = None,
    ) -> dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            max_epochs: Maximum number of epochs
            patience: Early stopping patience (optional)
            
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        patience_counter = 0
        
        for epoch in range(max_epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch(train_loader)
            history['train_loss'].append(train_losses)
            
            log_msg = f"Epoch {epoch}: Train Loss = {train_losses.get('total', 0):.6f}"
            
            # Validate
            if val_loader is not None:
                val_losses, val_metrics = self.evaluate(val_loader)
                history['val_loss'].append(val_losses)
                log_msg += f", Val Loss = {val_losses.get('total', 0):.6f}"
                
                # Log metrics
                for metric_name, metrics in val_metrics.items():
                    metric_dict = metrics.to_dict(f"val_{metric_name}_")
                    log_msg += f", {metric_name}_MAE={metric_dict[f'val_{metric_name}_mae']:.6f}"
                
                # Early stopping
                val_total = val_losses.get('total', float('inf'))
                if val_total < self.best_val_loss:
                    self.best_val_loss = val_total
                    patience_counter = 0
                    self.save_checkpoint('best_model.pt')
                else:
                    patience_counter += 1
                    if patience is not None and patience_counter >= patience:
                        logging.info(f"Early stopping at epoch {epoch}")
                        break
            
            logging.info(log_msg)
            
            # Update scheduler if per-epoch
            if self.scheduler is not None and not hasattr(self.scheduler, 'step_per_batch'):
                self.scheduler.step()
            
            # Save checkpoint
            if self.checkpoint_dir and epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        return history
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        if self.is_distributed and dist.get_rank() != 0:
            return
        
        checkpoint_path = Path(self.checkpoint_dir) / filename
        
        # Get model state (unwrap if DDP)
        model = self.model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': model.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Get model (unwrap if DDP)
        model = self.model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
