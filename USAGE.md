# Usage Guide - WAN 2.2 Image-to-Video LoRA Demo

## Quick Start

### 1. Deploying to Hugging Face Spaces

To deploy this demo to Hugging Face Spaces:

```bash
# Install git-lfs if not already installed
git lfs install

# Create a new Space on huggingface.co
# Then clone your space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy all files from this demo
cp -r * YOUR_SPACE_NAME/

# Commit and push
git add .
git commit -m "Initial commit: WAN 2.2 Image-to-Video LoRA Demo"
git push
```

### 2. Running Locally

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will be available at `http://localhost:7860`

## Using the Demo

### Basic Usage

1. **Upload Image**: Click the image upload area and select an image file
2. **Enter Prompt**: Type a description of the motion you want (e.g., "A person walking forward, cinematic")
3. **Click Generate**: Wait for the video to be generated (first run will download the model)
4. **View Result**: The generated video will appear in the output area

### Advanced Settings

Expand the "Advanced Settings" accordion to access:

- **Inference Steps** (20-100): More steps = higher quality but slower generation
  - 20-30: Fast, lower quality
  - 50: Balanced (recommended)
  - 80-100: Slow, highest quality

- **Guidance Scale** (1.0-15.0): How closely to follow the prompt
  - 1.0-3.0: More creative, less faithful to prompt
  - 6.0: Balanced (recommended)
  - 10.0-15.0: Very faithful to prompt, less creative

- **Use LoRA**: Enable/disable LoRA fine-tuning

- **LoRA Type**:
  - **High-Noise**: Best for dynamic, action-heavy scenes
  - **Low-Noise**: Best for subtle, smooth motions

## Example Prompts

### Good Prompts

- "A cat walking through a garden, sunny day, high quality"
- "Waves crashing on a beach, sunset lighting, cinematic"
- "A car driving down a highway, fast motion, 4k"
- "Smoke rising from a campfire, slow motion"

### Tips for Better Results

1. **Be Specific**: Include details about motion, lighting, and quality
2. **Use Keywords**: "cinematic", "high quality", "4k", "smooth"
3. **Describe Motion**: Clearly state what should move and how
4. **Consider Style**: Add style descriptors like "photorealistic" or "animated"

## Troubleshooting

### Out of Memory Error

If you encounter OOM errors:

1. The model requires significant VRAM (16GB+ recommended)
2. On Hugging Face Spaces, ensure you're using at least `gpu-medium` hardware
3. For local runs, try reducing the number of frames or using CPU offloading

### Slow Generation

- First generation will be slower (model downloads)
- Reduce inference steps for faster results
- Ensure GPU is being used (check logs for "Loading model on cuda")

### Model Not Loading

If the model fails to load:

1. Check your internet connection (model is ~20GB)
2. Ensure sufficient disk space
3. For Hugging Face Spaces, check your Space's logs

## Customization

### Using Your Own LoRA Files

To use your own LoRA weights:

1. Upload LoRA `.safetensors` files to Hugging Face
2. Update the URLs in `app.py`:

```python
HIGH_NOISE_LORA_URL = "https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/your_lora.safetensors"
```

3. Uncomment and implement the LoRA loading code in the `generate_video` function

### Changing the Model

To use a different model:

1. Update `MODEL_ID` in `app.py`
2. Ensure the model is compatible with `CogVideoXImageToVideoPipeline`
3. Adjust memory optimizations if needed

## Performance Notes

- **GPU (A10G/T4)**: ~2-3 minutes per video
- **GPU (A100)**: ~1-2 minutes per video
- **CPU**: Not recommended (20+ minutes)

## API Access

For programmatic access, you can use the Gradio Client:

```python
from gradio_client import Client

client = Client("YOUR_USERNAME/YOUR_SPACE_NAME")
result = client.predict(
    image="path/to/image.jpg",
    prompt="A cat walking",
    api_name="/predict"
)
```

## Credits

- Model: CogVideoX by THUDM
- Framework: Hugging Face Diffusers
- Interface: Gradio

## License

Apache 2.0 - See LICENSE file for details


