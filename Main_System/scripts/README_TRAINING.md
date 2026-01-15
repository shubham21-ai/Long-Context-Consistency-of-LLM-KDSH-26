# Quick Training Guide

## 🚀 Start Training (MacBook Air 8GB)

```bash
cd Main_System/scripts
python3 bdh_quickstart.py
```

**What happens:**
1. ✅ Loads novels from `../../Books/`
2. ✅ Trains BDH model (30 epochs, ~2-3 hours)
3. ✅ Saves model to `models/bdh_trained.pt`
4. ✅ Tests on real claims from `test.csv` + `train.csv`
5. ✅ Shows accuracy results

## ⚙️ Configuration

Edit `bdh_quickstart.py` line ~250:

```python
USE_SMALL_MODEL = True   # For MacBook Air 8GB
USE_SMALL_MODEL = False  # For Google Colab/GPU
```

## 📊 Expected Timeline

- **Training**: 2-4 hours (MacBook Air) or 30-60 min (Colab)
- **State Building**: 5-10 minutes per book
- **Verification**: 1-2 seconds per claim

## 🎯 After Training

The script automatically:
- Tests on 5 test cases
- Shows accuracy
- Displays detailed results

See `TRAINING_GUIDE.md` for full details!

