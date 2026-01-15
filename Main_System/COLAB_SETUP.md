# Google Colab Training Setup Guide

## 🚀 Quick Start

### Step 1: Open Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `BDH_Training_Colab.ipynb` from your project

OR

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → New notebook**
3. Copy and paste the code from `BDH_Training_Colab.ipynb`

### Step 2: Prepare Files to Upload

You need to upload these files to Colab:

**Required:**
- `core/bdh.py` - Core BDH model
- `utils/bdh_narrative_builder.py` - Narrative builder utility
- `Books/In search of the castaways.txt` - Your novel
- `Books/The Count of Monte Cristo.txt` - Your novel

**Optional (for testing):**
- `test.csv` - Test cases
- `train.csv` - Training labels

### Step 3: Run the Notebook

1. **Cell 1**: Install dependencies (runs automatically)
2. **Cell 2**: Upload files (click "Choose Files" and select all)
3. **Cell 3**: Import and setup (runs automatically)
4. **Cell 4**: Load novels (runs automatically)
5. **Cell 5**: **Train model** (30-60 minutes, grab coffee ☕)
6. **Cell 6**: Save model (runs automatically)
7. **Cell 7**: Download model to your Mac

### Step 4: Use Trained Model Locally

1. Download `bdh_trained.pt` from Colab
2. Move it to: `Main_System/scripts/models/bdh_trained.pt`
3. Run: `cd Main_System/scripts && python3 bdh_quickstart.py`
4. The script will automatically detect and use the trained model!

---

## 📋 Detailed Instructions

### Uploading Files

When you run Cell 2, you'll see a "Choose Files" button. Click it and select:

```
✅ core/bdh.py
✅ utils/bdh_narrative_builder.py
✅ Books/In search of the castaways.txt
✅ Books/The Count of Monte Cristo.txt
✅ test.csv (optional)
✅ train.csv (optional)
```

The notebook will automatically organize them into the correct folders.

### Training Progress

During training (Cell 5), you'll see:
```
Epoch 1/50: loss = 4.5234
Epoch 10/50: loss = 3.2145
Epoch 20/50: loss = 2.8765
...
```

**Expected timeline:**
- Epochs 1-10: Loss drops quickly (4.5 → 3.0)
- Epochs 10-30: Gradual improvement (3.0 → 2.5)
- Epochs 30-50: Fine-tuning (2.5 → 2.0)

### Downloading the Model

After training completes:
1. Run Cell 7
2. A download will start automatically
3. Save `bdh_trained.pt` to your Downloads folder
4. Move it to: `Main_System/scripts/models/bdh_trained.pt`

---

## ⚙️ Configuration

### Model Config (Already Set)

The notebook uses **FULL model config** optimized for GPU:

```python
config = BDHConfig(
    n_layer=6,        # Full size
    n_embd=256,       # Full size
    n_head=4,
    mlp_internal_dim_multiplier=128,
    vocab_size=256,
    dropout=0.1
)
```

### Training Parameters

- **Epochs**: 50 (can reduce to 30 for faster training)
- **Batch size**: 8 (optimal for GPU)
- **Learning rate**: 3e-4 (default, works well)

---

## 🐛 Troubleshooting

### "CUDA out of memory"

**Solution**: Reduce batch size
```python
model = train_model_on_books(books, config, epochs=50, batch_size=4)  # Changed from 8 to 4
```

### "File not found" errors

**Solution**: Make sure you uploaded all files in Cell 2. Check the file list printed after upload.

### Training is too slow

**Solution**: Reduce epochs
```python
model = train_model_on_books(books, config, epochs=30, batch_size=8)  # Changed from 50 to 30
```

### Model download fails

**Solution**: 
1. Check file size (should be ~50-100MB)
2. Try downloading again
3. If still fails, use this code:
```python
from google.colab import files
files.download('models/bdh_trained.pt')
```

---

## 💡 Tips

1. **Keep Colab tab open** during training (sessions timeout after 90 min of inactivity)
2. **Monitor GPU usage**: Runtime → Change runtime type → GPU (should be selected)
3. **Save progress**: Colab auto-saves, but you can also download intermediate checkpoints
4. **Use GPU**: Make sure Runtime → Change runtime type → GPU is selected

---

## ✅ After Training

Once you have `bdh_trained.pt` on your local machine:

1. Place it in: `Main_System/scripts/models/bdh_trained.pt`
2. Run: `cd Main_System/scripts && python3 bdh_quickstart.py`
3. The script will:
   - Detect the trained model
   - Skip training
   - Build narrative states
   - Test on real claims
   - Show accuracy results

**Expected accuracy**: 70-85% with a fully trained model!

---

## 📞 Need Help?

- Check `TRAINING_GUIDE.md` for detailed explanations
- Check Colab console for error messages
- Make sure all files uploaded correctly in Cell 2

