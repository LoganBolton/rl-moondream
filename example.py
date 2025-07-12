import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from moondream import MoondreamModel
from config import MoondreamConfig
from simple_weights import simple_load_weights

print("Creating model with local architecture...")
config = MoondreamConfig()  # Uses default config values
model = MoondreamModel(config, dtype=torch.float16)  # Changed to float16 to match weights

# Download weights from HuggingFace
print("Downloading model weights...")
weights_path = hf_hub_download(
    repo_id="vikhyatk/moondream2", 
    filename="model.safetensors",
    revision="2025-06-21"
)

# Load weights into the model
print("Loading weights into model...")
simple_load_weights(weights_path, model)

# Move model to GPU if available
device = "cuda"
model = model.to(device)
print(f"Model loaded on {device}")

# Load and process image
image = Image.open("images/test.png")

print("\nVisual query: 'How many people are in the image?'")
print(model.query(image, "How many people are in the image?")["answer"])