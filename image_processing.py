import threading
import numpy as np
from PIL import Image
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from diffusers import StableDiffusionImg2ImgPipeline

class ImageProcessor:
    def __init__(self, device):
        self.device = device
        self.diffusion_model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.diffusion_model_id).to(device)
        self.pipe.safety_checker = None
        self.img2img_model_lock = threading.Lock()
        self.fastsam = FastSAM('models/FastSAM-x.pt')
        self.fastsam_model_lock = threading.Lock()

    def process_image(self, image_file, prompt, hex_color):
        init_image = Image.open(image_file.stream).convert("RGB")

        with self.fastsam_model_lock:
            everything_results = self.fastsam(init_image, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
            prompt_process = FastSAMPrompt(init_image, everything_results, device=self.device)
            ann = prompt_process.text_prompt(text=prompt)
        
        changed_image = self.change_image_color_with_mask(init_image, ann, hex_color)

        with self.img2img_model_lock:
            images = self.pipe(prompt=prompt, image=changed_image, strength=0.75, guidance_scale=7.5).images
            return images[0]

    def change_image_color_with_mask(self, image, mask_tensor, hex_color):
        mask = Image.fromarray(mask_tensor.squeeze().astype(np.uint8) * 255)

        color = hex_color.replace("#", "")+'ff'  # Add alpha value
        color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4, 6))

        colored_image = Image.new('RGBA', image.size, color)
        masked_colored_image = Image.composite(colored_image, image, mask)
        return masked_colored_image.convert('RGB')