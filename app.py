import os
os.environ['PATH'] = os.getcwd() + '\libs;' + os.environ['PATH']

import torch
from io import BytesIO
from flask import Flask, request, send_file

from ad_creator import AdCreator
from image_processing import ImageProcessor

app = Flask(__name__)


image_processor = ImageProcessor(device='cuda' if torch.cuda.is_available() else 'cpu')
ad_creator = AdCreator()

@app.route('/generate-image', methods=['POST'])
def generate_image():
    if 'image' not in request.files:
        return "No image file provided", 400
    if 'prompt' not in request.form or 'color' not in request.form:
        return "Missing prompt or color", 400

    image_file = request.files['image']
    prompt = request.form['prompt']
    hex_color = request.form['color']

    response_image = image_processor.process_image(image_file, prompt, hex_color)

    img_io = BytesIO()
    response_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


@app.route('/generate-ad', methods=['POST'])
def generate_ad():
    if 'image' not in request.files or 'logo' not in request.files:
            return "No image or logo file provided", 400
    if 'punchline' not in request.form or 'color' not in request.form or 'button_text' not in request.form:
        return "Missing punchline, color, or button text", 400

    logo_image = request.files['logo']
    main_image = request.files['image']
    hex_color = request.form['color']
    punchline = request.form['punchline']
    button_text = request.form['button_text']

    ad_image = ad_creator.generate_ad(logo_image, main_image, hex_color, punchline, button_text)
    
    img_io = BytesIO()
    ad_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True, threaded=True)
