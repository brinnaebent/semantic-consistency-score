import os
from io import BytesIO
import requests
from PIL import Image
import replicate
from tqdm import tqdm

def run_pixart(prompt, seed, path_to_save):
  output = replicate.run(
    "lucataco/pixart-xl-2:816c99673841b9448bc2539834c16d40e0315bbf92fef0317b57a226727409bb",
    input={
        "style": "None",
        "width": 768,
        "height": 768,
        "prompt": prompt,
        "scheduler": "K_EULER",
        "num_outputs": 1,
        "guidance_scale": 7.5,
        "num_inference_steps": 20,
        "seed": seed
    }
  )
  
  image_url = output[0]
  response = requests.get(image_url)
  image_data = response.content
  image = Image.open(BytesIO(image_data))
  image_jpeg = image.convert('RGB')

  image_jpeg.save(f'{path_to_save}/{seed}.jpg', 'JPEG')
  print(f"Run complete. Seed:{seed}")

def run_sdxl(prompt, seed, path_to_save):
  output = replicate.run(
      "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
      input={
          "width": 768,
          "height": 768,
          "prompt": prompt,
          "refine": "no_refiner",
          "scheduler": "K_EULER",
          "lora_scale": 0.6,
          "num_outputs": 1,
          "guidance_scale": 7.5,
          "apply_watermark": False,
          "negative_prompt": "",
          "prompt_strength": 0.8,
          "num_inference_steps": 20,
          "seed": seed
      }
  )

  image_url = output[0]
  response = requests.get(image_url)
  image_data = response.content
  image = Image.open(BytesIO(image_data))
  image_jpeg = image.convert('RGB')

  image_jpeg.save(f'{path_to_save}/{seed}.jpg', 'JPEG')
  print(f"Run complete. Seed:{seed}")


def run_sdxl_lora(prompt, seed, path_to_save, lora_model="brinnaebent/sdxl-lora-monet:7f5d28ab8ed4de56770de0402ac91f1d6ffe3caff39c4b979ed3ed0779a452ed"):
  output = replicate.run(
      lora_model,
      input={
          "width": 768,
          "height": 768,
          "prompt": prompt,
          "refine": "no_refiner",
          "scheduler": "K_EULER",
          "lora_scale": 0.6,
          "num_outputs": 1,
          "guidance_scale": 7.5,
          "apply_watermark": False,
          "negative_prompt": "",
          "prompt_strength": 0.8,
          "num_inference_steps": 20,
          "seed": seed
      }
  )

  image_url = output[0]
  response = requests.get(image_url)
  image_data = response.content
  image = Image.open(BytesIO(image_data))
  image_jpeg = image.convert('RGB')

  image_jpeg.save(f'{path_to_save}/{seed}.jpg', 'JPEG')
  print(f"Run complete. Seed:{seed}")

