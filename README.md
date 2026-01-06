# 🎨 CycleGAN Face-Sketch Converter

Convert face photos to sketches and vice versa using CycleGAN.

## 🚀 Deployment on Hugging Face Spaces

### Step 1: Prepare Your Models

After training, run this cell in your notebook:

```python
# Run this in your training notebook
export_for_huggingface(checkpoint_epoch=100)  # Replace 100 with your best epoch
```

This creates two files in `deployment_models/`:
- `photo_to_sketch.pth` (~44 MB)
- `sketch_to_photo.pth` (~44 MB)

### Step 2: Create Hugging Face Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Configure:
   - **Space name**: `cyclegan-face-sketch`
   - **License**: Choose appropriate license
   - **Select SDK**: **Streamlit**
   - **Space hardware**: CPU Basic (free) or GPU for faster inference

### Step 3: Upload Files

Upload these files to your Space:

```
your-space/
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── photo_to_sketch.pth       # Model file (Photo → Sketch)
├── sketch_to_photo.pth       # Model file (Sketch → Photo)
└── README.md                 # This file (optional)
```

### Step 4: Your Space will Auto-Build

Hugging Face will automatically:
1. Install dependencies from `requirements.txt`
2. Run your Streamlit app
3. Provide a public URL

## 📦 File Sizes

- `photo_to_sketch.pth`: ~44 MB
- `sketch_to_photo.pth`: ~44 MB
- **Total**: ~88 MB (well within free tier limits)

## 🎯 Features

- ✅ Photo to Sketch conversion
- ✅ Sketch to Photo conversion  
- ✅ Auto-detection of input type
- ✅ Camera input support
- ✅ Download results
- ✅ Responsive UI

## 💡 Usage Tips

### For Best Results:
- Use clear, front-facing photos
- Ensure good lighting
- Images are automatically resized to 256x256

### Hardware:
- **CPU Basic (Free)**: ~3-5 seconds per image
- **GPU T4 (Paid)**: ~0.5 seconds per image

## 🔧 Local Testing

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

Then open http://localhost:8501

## 📊 Model Details

- **Architecture**: CycleGAN with ResNet-based generators
- **Input Size**: 256x256 RGB images
- **Training**: Unpaired face-sketch dataset
- **Loss Functions**: 
  - Adversarial loss (LSGAN)
  - Cycle consistency loss
  - Identity loss

## 🐛 Troubleshooting

### Space doesn't start:
- Check all files are uploaded correctly
- Verify file names match exactly in `app.py`
- Check Space logs for errors

### Out of memory:
- Use CPU Basic hardware
- Models are optimized for CPU inference

### Slow inference:
- Upgrade to GPU hardware in Space settings
- Or use batch processing

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

- CycleGAN paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- Person Face Sketches dataset: [Kaggle](https://www.kaggle.com/datasets/almightyj/person-face-sketches)