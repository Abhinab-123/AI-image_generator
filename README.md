📦 Requirements

Make sure you have Python 3.8+ and pip installed.
Then, install or upgrade the required libraries:

!pip install --upgrade diffusers transformers accelerate torch bitsandbytes scipy safetensors xformers

Dependencies

diffusers – for diffusion model pipelines

transformers – Hugging Face models and tokenizers

accelerate – for optimized inference and distributed compute

torch – PyTorch (deep learning framework)

bitsandbytes – 8-bit inference/training (memory-efficient)

scipy – general scientific computing utilities

safetensors – safe and fast model checkpoints

xformers – memory-efficient transformer ops

🚀 Usage
1. Import libraries and load the model
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt

2. Free up GPU memory (optional)
torch.cuda.empty_cache()

3. Load Stable Diffusion 2.1
model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

4. Generate an image
prompt = "Ocean beach"
image = pipe(prompt, width=1000, height=1000).images[0]

5. Display the image
plt.imshow(image)
plt.axis('off')
plt.show()

🧠 Tips

Change the prompt string to generate different scenes.
Example: "A futuristic city at sunset" or "A fantasy castle in the clouds".

Adjust width and height for output resolution (use multiples of 8).

Make sure your GPU has enough VRAM (at least 6–8 GB recommended).

💾 Output Example
Prompt	Result
Ocean beach
