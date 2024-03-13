import sys

# Add the directory containing your module to the Python path
sys.path.append('/home/riccardo/Sources/GitHub/torch.TensorRef')

import torchTensorRef
torchTensorRef.injectTo.append('transformers') # add TensorRef support also to transformers library

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, RePaintScheduler

torchTensorRef.tensorsManager.device = torch.device("hip")

model_id = "/media/riccardo/M2/Shared-M2/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
#pipe = pipe.to("cuda")

prompt = "vintage corset muglier style"
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")