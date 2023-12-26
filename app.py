import os
os.environ['PATH'] = os.getcwd() + '\libs;' + os.environ['PATH']
print(os.environ['PATH'])


from flask import Flask, request, send_file
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
from FastSAM.fastsam import FastSAM
from FastSAM.fastsam import FastSAMPrompt
import threading
import numpy as np
from ad_creator import generate_pdf, pdf_to_high_res_image_with_crop

app = Flask(__name__)

device = "cpu"
diffusion_model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(diffusion_model_id).to(device)
pipe.safety_checker = None

img2img_model_lock = threading.Lock()

fastsam = FastSAM('models/FastSAM-x.pt')

fastsam_model_lock = threading.Lock()

@app.route('/generate-image', methods=['POST'])
def generate_image():
    if 'image' not in request.files:
        return "No image file provided", 400
    if 'prompt' not in request.form or 'color' not in request.form:
        return "Missing prompt or color", 400

    image_file = request.files['image']
    prompt = request.form['prompt']
    color_hex = request.form['color']

    response_image = process_image(image_file, prompt, color_hex)

    img_io = BytesIO()
    response_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


def process_image(image_file, prompt, color_hex):
    init_image = Image.open(image_file.stream).convert("RGB")

    with fastsam_model_lock:
        everything_results = fastsam(init_image, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        prompt_process = FastSAMPrompt(init_image, everything_results, device=device)
        ann = prompt_process.text_prompt(text=prompt)
    
    changed_image = change_image_color_with_mask(init_image, ann, color_hex)

    with img2img_model_lock:
        images = pipe(prompt=prompt, image=changed_image, strength=0.75, guidance_scale=7.5).images
        return images[0]
    
def change_image_color_with_mask(image, mask_tensor, hex_color):
    mask = Image.fromarray(mask_tensor.squeeze().astype(np.uint8) * 255)

    color = hex_color.replace("#", "")+'ff'  # Add alpha value
    color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4, 6))

    colored_image = Image.new('RGBA', image.size, color)
    masked_colored_image = Image.composite(colored_image, image, mask)
    return masked_colored_image.convert('RGB')


@app.route('/generate-ad', methods=['POST'])
def generate_ad():
    if 'image' not in request.files or 'logo' not in request.files:
        return "No image file provided", 400
    if 'punchline' not in request.form or 'color' not in request.form or 'button_text' not in request.form:
        return "Missing prompt or color", 400

    image_file = request.files['image']
    logo = request.files['logo']
    punchline = request.form['punchline']
    button_text = request.form['button_text']
    color_hex = request.form['color']

    pdf = generate_pdf(logo, image_file, color_hex, punchline, button_text)
    response_image = pdf_to_high_res_image_with_crop(pdf)

    img_io = BytesIO()
    response_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)