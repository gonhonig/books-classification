import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create output directory for plots
output_dir = Path(__file__).parent
output_dir.mkdir(exist_ok=True)

def load_model_metrics(model_path):
    """Load metrics from a model's detailed metrics JSON file."""
    # Define the correct file names for each model
    file_mapping = {
        'model_per_book': 'per_book_detailed_metrics.json',
        'multi_label_model': 'multi_label_detailed_metrics.json',
        'naive_model_per_book': 'naive_per_book_detailed_metrics.json',
        'naive_multi_label_model': 'naive_multi_label_detailed_metrics.json'
    }
    
    # Construct path relative to the script's parent directory
    script_parent = Path(__file__).parent.parent
    metrics_file = script_parent / model_path / file_mapping[Path(model_path).name]
    with open(metrics_file, 'r') as f:
        return json.load(f)

def extract_accuracy_metrics(metrics_data):
    """Extract accuracy metrics for overall, single_label, and multi_label."""
    results = {
        'overall': [],
        'single_label': [],
        'multi_label': []
    }
    
    for book, book_metrics in metrics_data.items():
        for metric_type in ['overall', 'single_label', 'multi_label']:
            if metric_type in book_metrics:
                results[metric_type].append(book_metrics[metric_type]['accuracy'])
    
    return results

def calculate_average_metrics(accuracy_metrics):
    """Calculate average accuracy for each metric type."""
    return {
        'overall': np.mean(accuracy_metrics['overall']),
        'single_label': np.mean(accuracy_metrics['single_label']),
        'multi_label': np.mean(accuracy_metrics['multi_label'])
    }

def create_comparison_plots():
    """Create comprehensive comparison plots for all models."""
    
    # Model configurations
    models = {
        'model_per_book': 'Per-Book Model',
        'multi_label_model': 'Multi-Label Model', 
        'naive_model_per_book': 'Naive Per-Book Model',
        'naive_multi_label_model': 'Naive Multi-Label Model'
    }
    
    # Load metrics for all models
    model_metrics = {}
    for model_path, model_name in models.items():
        try:
            metrics_data = load_model_metrics(model_path)
            accuracy_metrics = extract_accuracy_metrics(metrics_data)
            avg_metrics = calculate_average_metrics(accuracy_metrics)
            model_metrics[model_name] = avg_metrics
            print(f"Successfully loaded metrics for {model_name}")
        except Exception as e:
            print(f"Error loading metrics for {model_path}: {e}")
            continue
    
    if not model_metrics:
        print("No model metrics loaded. Exiting.")
        return
    
    # Create comparison plots
    create_accuracy_comparison_plot(model_metrics)
    create_detailed_comparison_plot(model_metrics)
    create_per_book_comparison_plot(models)
    create_metric_breakdown_plot(model_metrics)
    create_summary_table(model_metrics)
    save_summary_csv(model_metrics)

def create_accuracy_comparison_plot(model_metrics):
    """Create main accuracy comparison bar plot."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    model_names = list(model_metrics.keys())
    metric_types = ['overall', 'single_label', 'multi_label']
    
    # Set up bar positions
    x = np.arange(len(model_names))
    width = 0.25
    multiplier = 0
    
    # Create bars for each metric type
    for metric_type in metric_types:
        values = [model_metrics[model][metric_type] for model in model_names]
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=metric_type.replace('_', ' ').title())
        ax.bar_label(rects, fmt='%.3f', padding=3)
        multiplier += 1
    
    # Customize plot
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison: Overall vs Single-Label vs Multi-Label', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add value annotations
    for i, model in enumerate(model_names):
        for j, metric_type in enumerate(metric_types):
            value = model_metrics[model][metric_type]
            ax.text(i + j*width, value + 0.01, f'{value:.3f}', 
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_comparison_plot(model_metrics):
    """Create detailed comparison with subplots for each metric type."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metric_types = ['overall', 'single_label', 'multi_label']
    
    for i, metric_type in enumerate(metric_types):
        ax = axes[i]
        model_names = list(model_metrics.keys())
        values = [model_metrics[model][metric_type] for model in model_names]
        
        # Create bars
        bars = ax.bar(model_names, values, color=sns.color_palette("husl", len(model_names)))
        ax.bar_label(bars, fmt='%.3f', padding=3)
        
        # Customize subplot
        ax.set_title(f'{metric_type.replace("_", " ").title()} Accuracy', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
        
        # Add value annotations
        for j, (bar, value) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Detailed Model Comparison by Metric Type', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_per_book_comparison_plot(models):
    """Create per-book comparison plot showing how each model performs on each book."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    books = ["Anna Karenina", "Frankenstein", "The Adventures of Alice in Wonderland", "Wuthering Heights"]
    
    # Define the correct file names for each model
    file_mapping = {
        'model_per_book': 'per_book_detailed_metrics.json',
        'multi_label_model': 'multi_label_detailed_metrics.json',
        'naive_model_per_book': 'naive_per_book_detailed_metrics.json',
        'naive_multi_label_model': 'naive_multi_label_detailed_metrics.json'
    }
    
    for book_idx, book in enumerate(books):
        ax = axes[book_idx]
        
        # Collect data for this book across all models
        model_names = []
        overall_accuracies = []
        single_label_accuracies = []
        multi_label_accuracies = []
        
        for model_path, model_name in models.items():
            try:
                script_parent = Path(__file__).parent.parent
                metrics_file = script_parent / model_path / file_mapping[model_path]
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                if book in metrics_data:
                    model_names.append(model_name)
                    overall_accuracies.append(metrics_data[book]['overall']['accuracy'])
                    single_label_accuracies.append(metrics_data[book]['single_label']['accuracy'])
                    multi_label_accuracies.append(metrics_data[book]['multi_label']['accuracy'])
            except Exception as e:
                print(f"Error loading data for {model_name} on {book}: {e}")
                continue
        
        # Create grouped bar chart
        x = np.arange(len(model_names))
        width = 0.25
        
        ax.bar(x - width, overall_accuracies, width, label='Overall', alpha=0.8)
        ax.bar(x, single_label_accuracies, width, label='Single-Label', alpha=0.8)
        ax.bar(x + width, multi_label_accuracies, width, label='Multi-Label', alpha=0.8)
        
        ax.set_title(f'{book}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.suptitle('Per-Book Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_book_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_metric_breakdown_plot(model_metrics):
    """Create a heatmap-style visualization of all metrics."""
    # Prepare data for heatmap
    metric_types = ['overall', 'single_label', 'multi_label']
    model_names = list(model_metrics.keys())
    
    # Create data matrix
    data_matrix = []
    for model in model_names:
        row = [model_metrics[model][metric] for metric in metric_types]
        data_matrix.append(row)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data_matrix, cmap='RdYlBu', aspect='auto')
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(metric_types)):
            text = ax.text(j, i, f'{data_matrix[i][j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Customize plot
    ax.set_xticks(range(len(metric_types)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metric_types])
    ax.set_yticklabels(model_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', rotation=270, labelpad=20)
    
    plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(model_metrics):
    """Create a summary table of all model performances."""
    # Convert to DataFrame for easier manipulation
    data = []
    for model_name, metrics in model_metrics.items():
        data.append({
            'Model': model_name,
            'Overall Accuracy': f"{metrics['overall']:.3f}",
            'Single-Label Accuracy': f"{metrics['single_label']:.3f}",
            'Multi-Label Accuracy': f"{metrics['multi_label']:.3f}"
        })
    
    df = pd.DataFrame(data)
    
    # Create a formatted table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def save_summary_csv(model_metrics):
    """Save the model performance summary as a CSV file."""
    # Convert to DataFrame for easier manipulation
    data = []
    for model_name, metrics in model_metrics.items():
        data.append({
            'Model': model_name,
            'Overall Accuracy': metrics['overall'],
            'Single-Label Accuracy': metrics['single_label'],
            'Multi-Label Accuracy': metrics['multi_label']
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = output_dir / 'model_performance_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Model performance summary saved to: {csv_path}")
    
    # Also save with formatted values (3 decimal places)
    df_formatted = df.copy()
    for col in ['Overall Accuracy', 'Single-Label Accuracy', 'Multi-Label Accuracy']:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.3f}")
    
    csv_formatted_path = output_dir / 'model_performance_summary_formatted.csv'
    df_formatted.to_csv(csv_formatted_path, index=False)
    print(f"Formatted model performance summary saved to: {csv_formatted_path}")
    
    return df

if __name__ == "__main__":
    print("Creating comprehensive model comparison plots...")
    create_comparison_plots()
    print("Plots saved successfully!") 