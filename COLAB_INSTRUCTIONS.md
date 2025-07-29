# 🚀 Google Colab Setup Instructions

## Quick Start Guide

### Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com)
2. Sign in with your Google account

### Step 2: Enable GPU
1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Click **Save**

### Step 3: Upload Files
1. Download `books_classification_colab.zip` from your computer
2. In Colab, click **Files** → **Upload to session storage**
3. Upload the `books_classification_colab.zip` file

### Step 4: Run the Notebook
1. Copy the contents of `colab_setup.ipynb` into a new Colab notebook
2. Run each cell in order
3. When prompted, upload the ZIP file

## Alternative: Quick Start Script

If you prefer a simpler approach:

1. **Upload the ZIP file** to Colab
2. **Run this in a cell**:
   ```python
   # Extract the ZIP file
   import zipfile
   with zipfile.ZipFile('books_classification_colab.zip', 'r') as zip_ref:
       zip_ref.extractall('.')
   
   # Run the quick start script
   !python colab_quick_start.py
   ```

## What You'll Get

✅ **GPU-accelerated training** (much faster than CPU)  
✅ **Automatic data preparation** (downloads and processes the 4 books)  
✅ **Model training** (5 epochs by default)  
✅ **Results download** (checkpoints, logs, and trained model)  

## Expected Timeline

- **Data preparation**: ~5 minutes
- **Training (5 epochs)**: ~30-60 minutes (depending on GPU)
- **Total time**: ~1-2 hours

## Troubleshooting

### No GPU Available
- Make sure you enabled GPU in Runtime settings
- Free Colab has limited GPU hours
- Consider Colab Pro for more hours

### Out of Memory
- The script automatically adjusts batch size
- If still having issues, reduce epochs in the config

### File Upload Issues
- Make sure the ZIP file is complete (2.2 MB)
- Try uploading individual files if needed

## Files Included

- `configs/config.yaml` - Training configuration
- `data/prepare_data.py` - Data preparation script
- `models/constructive_model.py` - The neural network model
- `train_cloud.py` - Cloud-optimized training script
- `test_prediction.py` - Model testing script
- All utility functions and dependencies

## Next Steps

After training completes:
1. **Download results** using the download cell
2. **Analyze performance** in the logs
3. **Test the model** with new sentences
4. **Fine-tune** hyperparameters if needed

Happy training! 🎉 