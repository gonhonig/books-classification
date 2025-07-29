"""
Visualization utilities for training and model analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import json

def plot_training_curves(phase_results: List[Dict], save_path: Optional[str] = None):
    """Plot training curves for all phases."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Constructive Learning Training Curves', fontsize=16)
    
    # Plot 1: Training Loss by Phase
    ax1 = axes[0, 0]
    for i, phase_result in enumerate(phase_results):
        losses = [epoch['train'] for epoch in phase_result['losses']]
        epochs = range(1, len(losses) + 1)
        ax1.plot(epochs, losses, marker='o', label=f"Phase {i}: {phase_result['name']}")
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss by Phase')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss by Phase
    ax2 = axes[0, 1]
    for i, phase_result in enumerate(phase_results):
        losses = [epoch['val'] for epoch in phase_result['losses']]
        epochs = range(1, len(losses) + 1)
        ax2.plot(epochs, losses, marker='s', label=f"Phase {i}: {phase_result['name']}")
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss by Phase')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Best Validation Loss by Phase
    ax3 = axes[1, 0]
    phases = [f"Phase {i}" for i in range(len(phase_results))]
    best_losses = [phase_result['best_val_loss'] for phase_result in phase_results]
    
    bars = ax3.bar(phases, best_losses, color='skyblue', alpha=0.7)
    ax3.set_xlabel('Phase')
    ax3.set_ylabel('Best Validation Loss')
    ax3.set_title('Best Validation Loss by Phase')
    
    # Add value labels on bars
    for bar, loss in zip(bars, best_losses):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{loss:.4f}', ha='center', va='bottom')
    
    # Plot 4: Loss Improvement Across Phases
    ax4 = axes[1, 1]
    phase_names = [phase_result['name'] for phase_result in phase_results]
    initial_losses = [phase_result['losses'][0]['train'] for phase_result in phase_results]
    final_losses = [phase_result['losses'][-1]['train'] for phase_result in phase_results]
    
    x = np.arange(len(phase_names))
    width = 0.35
    
    ax4.bar(x - width/2, initial_losses, width, label='Initial Loss', alpha=0.7)
    ax4.bar(x + width/2, final_losses, width, label='Final Loss', alpha=0.7)
    
    ax4.set_xlabel('Phase')
    ax4.set_ylabel('Loss')
    ax4.set_title('Loss Improvement Across Phases')
    ax4.set_xticks(x)
    ax4.set_xticklabels(phase_names, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()

def plot_phase_knowledge_evolution(phase_knowledge_paths: List[str], save_path: Optional[str] = None):
    """Plot the evolution of knowledge across phases."""
    
    # Load knowledge vectors
    knowledge_vectors = []
    for path in phase_knowledge_paths:
        if Path(path).exists():
            knowledge = torch.load(path)
            knowledge_vectors.append(knowledge.numpy())
    
    if not knowledge_vectors:
        print("No knowledge vectors found.")
        return
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Knowledge Evolution Across Phases', fontsize=16)
    
    # Plot 1: Knowledge vector magnitude
    ax1 = axes[0, 0]
    magnitudes = [np.linalg.norm(kv) for kv in knowledge_vectors]
    phases = range(1, len(magnitudes) + 1)
    
    ax1.plot(phases, magnitudes, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Knowledge Magnitude')
    ax1.set_title('Knowledge Magnitude by Phase')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Knowledge similarity between consecutive phases
    ax2 = axes[0, 1]
    similarities = []
    for i in range(len(knowledge_vectors) - 1):
        similarity = np.dot(knowledge_vectors[i], knowledge_vectors[i + 1]) / \
                    (np.linalg.norm(knowledge_vectors[i]) * np.linalg.norm(knowledge_vectors[i + 1]))
        similarities.append(similarity)
    
    phases_sim = range(1, len(similarities) + 1)
    ax2.plot(phases_sim, similarities, marker='s', linewidth=2, markersize=8)
    ax2.set_xlabel('Phase Transition')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Knowledge Similarity Between Consecutive Phases')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Knowledge distribution
    ax3 = axes[1, 0]
    for i, kv in enumerate(knowledge_vectors):
        ax3.hist(kv, alpha=0.5, label=f'Phase {i+1}', bins=30)
    
    ax3.set_xlabel('Knowledge Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Knowledge Distribution by Phase')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Knowledge heatmap
    ax4 = axes[1, 1]
    knowledge_matrix = np.array(knowledge_vectors)
    im = ax4.imshow(knowledge_matrix, cmap='viridis', aspect='auto')
    ax4.set_xlabel('Knowledge Dimension')
    ax4.set_ylabel('Phase')
    ax4.set_title('Knowledge Heatmap')
    ax4.set_yticks(range(len(knowledge_vectors)))
    ax4.set_yticklabels([f'Phase {i+1}' for i in range(len(knowledge_vectors))])
    
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Knowledge evolution plot saved to {save_path}")
    
    plt.show()

def plot_model_architecture(model, save_path: Optional[str] = None):
    """Plot model architecture visualization."""
    
    # Get model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Parameter distribution
    param_counts = []
    param_names = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_counts.append(param.numel())
            param_names.append(name.split('.')[0])  # Get module name
    
    # Group by module
    unique_modules = list(set(param_names))
    module_counts = []
    for module in unique_modules:
        count = sum(param_counts[i] for i, name in enumerate(param_names) if name == module)
        module_counts.append(count)
    
    ax1.pie(module_counts, labels=unique_modules, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Parameter Distribution\n(Total: {total_params:,})')
    
    # Plot 2: Layer-wise parameter count
    ax2.bar(range(len(unique_modules)), module_counts, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Module')
    ax2.set_ylabel('Number of Parameters')
    ax2.set_title('Parameters by Module')
    ax2.set_xticks(range(len(unique_modules)))
    ax2.set_xticklabels(unique_modules, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model architecture plot saved to {save_path}")
    
    plt.show()

def plot_self_supervised_tasks_performance(ss_metrics: Dict, save_path: Optional[str] = None):
    """Plot performance of self-supervised tasks."""
    
    tasks = list(ss_metrics.keys())
    metrics = ['accuracy', 'loss']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Task accuracy
    ax1 = axes[0]
    accuracies = [ss_metrics[task]['accuracy'] for task in tasks]
    bars1 = ax1.bar(tasks, accuracies, color='lightblue', alpha=0.7)
    ax1.set_xlabel('Self-Supervised Task')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Self-Supervised Tasks Accuracy')
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Plot 2: Task loss
    ax2 = axes[1]
    losses = [ss_metrics[task]['loss'] for task in tasks]
    bars2 = ax2.bar(tasks, losses, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Self-Supervised Task')
    ax2.set_ylabel('Loss')
    ax2.set_title('Self-Supervised Tasks Loss')
    
    # Add value labels
    for bar, loss in zip(bars2, losses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{loss:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Self-supervised tasks performance plot saved to {save_path}")
    
    plt.show()

def plot_dataset_statistics(dataset_stats: Dict, save_path: Optional[str] = None):
    """Plot dataset statistics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Dataset Statistics', fontsize=16)
    
    splits = list(dataset_stats.keys())
    
    # Plot 1: Sample count by split
    ax1 = axes[0, 0]
    sample_counts = [dataset_stats[split]['total_samples'] for split in splits]
    bars1 = ax1.bar(splits, sample_counts, color='lightgreen', alpha=0.7)
    ax1.set_xlabel('Dataset Split')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Sample Count by Split')
    
    # Add value labels
    for bar, count in zip(bars1, sample_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_counts)*0.01,
                f'{count}', ha='center', va='bottom')
    
    # Plot 2: Class distribution
    ax2 = axes[0, 1]
    if 'train' in dataset_stats and 'class_distribution' in dataset_stats['train']:
        class_dist = dataset_stats['train']['class_distribution']
        classes = list(class_dist.keys())
        counts = list(class_dist.values())
        
        bars2 = ax2.bar(classes, counts, color='lightblue', alpha=0.7)
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Class Distribution (Train Set)')
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Sentence length distribution
    ax3 = axes[1, 0]
    if 'train' in dataset_stats and 'avg_sentence_length' in dataset_stats['train']:
        avg_lengths = [dataset_stats[split].get('avg_sentence_length', 0) for split in splits]
        bars3 = ax3.bar(splits, avg_lengths, color='lightcoral', alpha=0.7)
        ax3.set_xlabel('Dataset Split')
        ax3.set_ylabel('Average Sentence Length')
        ax3.set_title('Average Sentence Length by Split')
    
    # Plot 4: Sentence length range
    ax4 = axes[1, 1]
    if 'train' in dataset_stats and 'min_sentence_length' in dataset_stats['train']:
        min_lengths = [dataset_stats[split].get('min_sentence_length', 0) for split in splits]
        max_lengths = [dataset_stats[split].get('max_sentence_length', 0) for split in splits]
        
        x = np.arange(len(splits))
        width = 0.35
        
        ax4.bar(x - width/2, min_lengths, width, label='Min Length', alpha=0.7)
        ax4.bar(x + width/2, max_lengths, width, label='Max Length', alpha=0.7)
        ax4.set_xlabel('Dataset Split')
        ax4.set_ylabel('Sentence Length')
        ax4.set_title('Sentence Length Range by Split')
        ax4.set_xticks(x)
        ax4.set_xticklabels(splits)
        ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dataset statistics plot saved to {save_path}")
    
    plt.show()

def create_training_dashboard(phase_results: List[Dict], 
                            final_metrics: Dict,
                            dataset_stats: Dict,
                            save_path: Optional[str] = None):
    """Create a comprehensive training dashboard."""
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Book Classification Training Dashboard', fontsize=16, y=0.95)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Plot 1: Training loss curves
    ax1 = fig.add_subplot(gs[0, :2])
    for i, phase_result in enumerate(phase_results):
        losses = [epoch['train'] for epoch in phase_result['losses']]
        epochs = range(1, len(losses) + 1)
        ax1.plot(epochs, losses, marker='o', label=f"Phase {i}: {phase_result['name']}")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss by Phase')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation loss curves
    ax2 = fig.add_subplot(gs[0, 2:])
    for i, phase_result in enumerate(phase_results):
        losses = [epoch['val'] for epoch in phase_result['losses']]
        epochs = range(1, len(losses) + 1)
        ax2.plot(epochs, losses, marker='s', label=f"Phase {i}: {phase_result['name']}")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss by Phase')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final metrics
    ax3 = fig.add_subplot(gs[1, 0])
    if 'overall' in final_metrics:
        metrics = ['accuracy', 'macro_f1', 'weighted_f1']
        values = [final_metrics['overall'][m] for m in metrics]
        bars = ax3.bar(metrics, values, color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.7)
        ax3.set_ylabel('Score')
        ax3.set_title('Final Model Performance')
        ax3.set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
    
    # Plot 4: Per-class performance
    ax4 = fig.add_subplot(gs[1, 1])
    if 'per_class' in final_metrics:
        classes = list(final_metrics['per_class'].keys())
        f1_scores = [final_metrics['per_class'][c]['f1'] for c in classes]
        bars = ax4.bar(classes, f1_scores, color='lightblue', alpha=0.7)
        ax4.set_ylabel('F1-Score')
        ax4.set_title('Per-Class F1-Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 1)
    
    # Plot 5: Dataset statistics
    ax5 = fig.add_subplot(gs[1, 2])
    if 'train' in dataset_stats and 'class_distribution' in dataset_stats['train']:
        class_dist = dataset_stats['train']['class_distribution']
        classes = list(class_dist.keys())
        counts = list(class_dist.values())
        
        bars = ax5.bar(classes, counts, color='lightgreen', alpha=0.7)
        ax5.set_xlabel('Class')
        ax5.set_ylabel('Number of Samples')
        ax5.set_title('Training Set Class Distribution')
        ax5.tick_params(axis='x', rotation=45)
    
    # Plot 6: Confidence analysis
    ax6 = fig.add_subplot(gs[1, 3])
    if 'confidence_analysis' in final_metrics:
        conf_analysis = final_metrics['confidence_analysis']
        metrics = ['mean_confidence', 'std_confidence']
        values = [conf_analysis[m] for m in metrics]
        bars = ax6.bar(metrics, values, color=['lightcoral', 'lightblue'], alpha=0.7)
        ax6.set_ylabel('Value')
        ax6.set_title('Confidence Analysis')
    
    # Plot 7: Phase comparison
    ax7 = fig.add_subplot(gs[2, :])
    phases = [f"Phase {i}" for i in range(len(phase_results))]
    best_losses = [phase_result['best_val_loss'] for phase_result in phase_results]
    
    bars = ax7.bar(phases, best_losses, color='skyblue', alpha=0.7)
    ax7.set_xlabel('Phase')
    ax7.set_ylabel('Best Validation Loss')
    ax7.set_title('Best Validation Loss by Phase')
    
    # Add value labels
    for bar, loss in zip(bars, best_losses):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{loss:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training dashboard saved to {save_path}")
    
    plt.show() 