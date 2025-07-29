# 🚀 Cloud GPU Training Guide

This guide helps you set up and run training on cloud GPU platforms for the books classification project.

## 🌟 Quick Start Options

### 1. **Google Colab (Recommended for beginners)**
- **Free**: Tesla T4 GPU (limited hours)
- **Pro**: V100/A100 GPUs, more hours
- **Setup**: Upload `colab_setup.ipynb` to Colab

### 2. **Google Cloud Platform (GCP)**
- **GPU Options**: T4, V100, A100
- **Cost**: ~$0.50-2.50/hour depending on GPU
- **Setup**: Use `train_cloud.py`

### 3. **AWS EC2**
- **GPU Options**: P3/P4 instances with V100/A100
- **Cost**: Spot instances for savings
- **Setup**: Use `train_cloud.py`

## 📋 Prerequisites

### For Google Colab:
1. Go to [Google Colab](https://colab.research.google.com)
2. Upload `colab_setup.ipynb`
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run the notebook cells

### For GCP/AWS:
1. Install cloud CLI tools
2. Set up authentication
3. Create GPU instance
4. Upload project files

## 🛠️ Setup Instructions

### Google Colab Setup

1. **Upload the notebook**:
   ```bash
   # Download colab_setup.ipynb to your computer
   # Then upload to Google Colab
   ```

2. **Enable GPU**:
   - Runtime → Change runtime type
   - Hardware accelerator: GPU
   - GPU type: T4 (free) or V100/A100 (Pro)

3. **Run the notebook**:
   - Execute cells in order
   - Upload your project files when prompted
   - Monitor training progress

### GCP Setup

1. **Create GPU instance**:
   ```bash
   gcloud compute instances create books-training \
     --zone=us-central1-a \
     --machine-type=n1-standard-4 \
     --accelerator="type=nvidia-tesla-t4,count=1" \
     --image-family=debian-11-gpu \
     --image-project=debian-cloud
   ```

2. **Connect and setup**:
   ```bash
   gcloud compute ssh books-training --zone=us-central1-a
   
   # Install dependencies
   sudo apt-get update
   sudo apt-get install -y python3-pip
   pip3 install -r requirements-cloud.txt
   ```

3. **Upload project**:
   ```bash
   # Upload your project files
   gcloud compute scp --recurse ./books-classification books-training:~/ --zone=us-central1-a
   ```

4. **Run training**:
   ```bash
   cd books-classification
   python3 train_cloud.py --epochs 10
   ```

### AWS Setup

1. **Launch GPU instance**:
   ```bash
   aws ec2 run-instances \
     --image-id ami-0c02fb55956c7d316 \
     --count 1 \
     --instance-type p3.2xlarge \
     --key-name your-key-pair
   ```

2. **Connect and setup**:
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   
   # Install dependencies
   sudo apt-get update
   sudo apt-get install -y python3-pip
   pip3 install -r requirements-cloud.txt
   ```

3. **Upload and run**:
   ```bash
   # Upload project files
   scp -r ./books-classification ubuntu@your-instance-ip:~/
   
   # Run training
   ssh ubuntu@your-instance-ip
   cd books-classification
   python3 train_cloud.py --epochs 10
   ```

## ⚙️ Configuration

### GPU Memory Optimization

The `train_cloud.py` script automatically adjusts batch size based on GPU memory:

- **< 8GB**: Batch size = 8
- **8-16GB**: Batch size = 16  
- **> 16GB**: Batch size = 32

### Training Parameters

Edit `configs/config.yaml` to adjust:

```yaml
training:
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 10
  weight_decay: 0.01
```

## 📊 Monitoring

### Weights & Biases Integration

1. **Setup WandB**:
   ```bash
   pip install wandb
   wandb login
   ```

2. **Monitor training**:
   - Visit wandb.ai to see real-time metrics
   - Track loss, accuracy, learning rate
   - Compare different runs

### TensorBoard

1. **Start TensorBoard**:
   ```bash
   tensorboard --logdir experiments/logs
   ```

2. **View metrics**:
   - Open browser to `http://localhost:6006`
   - Monitor training progress

## 💾 Checkpointing

The cloud training script automatically saves:

- **Regular checkpoints**: Every epoch
- **Best model**: Lowest validation loss
- **Metadata**: Training configuration and metrics

### Resume Training

```bash
# Resume from checkpoint
python3 train_cloud.py --resume experiments/checkpoints/checkpoint_epoch_5.pt
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce batch size in config
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow Training**:
   - Check GPU utilization: `nvidia-smi`
   - Increase batch size if memory allows
   - Use multiple GPUs if available

3. **Connection Issues**:
   - Use `tmux` for persistent sessions
   - Set up automatic reconnection
   - Use spot instances for cost savings

### Performance Tips

1. **Data Loading**:
   - Use multiple workers for DataLoader
   - Enable pin_memory for GPU
   - Preprocess data on CPU

2. **Model Optimization**:
   - Use mixed precision training
   - Enable gradient checkpointing
   - Optimize model architecture

## 💰 Cost Optimization

### GCP Cost Saving Tips

1. **Use Preemptible Instances**:
   ```bash
   gcloud compute instances create books-training \
     --preemptible \
     --zone=us-central1-a
   ```

2. **Spot Instances (AWS)**:
   - Save up to 90% on compute costs
   - Use for non-critical training runs

3. **Auto-shutdown**:
   ```bash
   # Set up auto-shutdown after training
   timeout 3600 python3 train_cloud.py
   ```

## 📈 Expected Performance

### Training Times (Estimated)

| GPU | Batch Size | Epochs | Time |
|-----|------------|--------|------|
| T4 | 16 | 10 | ~2 hours |
| V100 | 32 | 10 | ~1 hour |
| A100 | 64 | 10 | ~30 minutes |

### Memory Requirements

- **Model**: ~138M parameters
- **GPU Memory**: 4-8GB minimum
- **RAM**: 8GB minimum
- **Storage**: 2GB for dataset

## 🎯 Next Steps

1. **Start with Colab**: Use the notebook for quick testing
2. **Scale to GCP/AWS**: For longer training runs
3. **Monitor metrics**: Use WandB for experiment tracking
4. **Optimize hyperparameters**: Based on validation results
5. **Deploy model**: Save best checkpoint for inference

## 📞 Support

- **Colab Issues**: Check GPU availability and runtime settings
- **GCP Issues**: Check instance quotas and billing
- **AWS Issues**: Verify instance limits and key pairs
- **Training Issues**: Check logs in `experiments/logs/`

Happy training! 🚀 