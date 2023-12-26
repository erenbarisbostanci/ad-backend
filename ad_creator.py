from weasyprint import HTML
from PIL import Image
import base64
from io import BytesIO
import io
import numpy as np
import fitz

def image_to_data_uri(file_storage):
    image = Image.open(file_storage.stream)
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def generate_pdf(logo_image, main_image, hex_color, punchline, button_text):
    logo_data_uri = image_to_data_uri(logo_image)
    main_image_data_uri = image_to_data_uri(main_image)

    
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                background-color: #f8eddb;
                width: auto;
                height: auto;
                padding: 20px;
            }}
            .logo {{
                max-width: 200px;
                height: auto;
                margin: 20px auto;
                display: block;
            }}
            .main-image {{
                max-width: 500px;
                height: auto;
                margin: 20px auto;
                display: block;
                border-radius: 20px;
            }}
            .punchline {{
                color: {hex_color};
                font-size: 24px;
                margin: 20px 0;
            }}
            .button {{
                background-color: {hex_color};
                border: none;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: block;
                font-size: 16px;
                margin: 20px auto;
                cursor: pointer;
                border-radius: 15px;
                box-sizing: border-box;
                max-width: 90%;
                overflow-wrap: break-word;
                white-space: normal;
                width: 500px;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <img src="{logo_data_uri}" class="logo">
            <img src="{main_image_data_uri}" class="main-image">
            <div class="punchline">{punchline}</div>
            <div class="button-container">
                <button class="button">{button_text}</button>
            </div>
        </div>
    </body>
    </html>
    '''
    return HTML(string=html_content).write_pdf()

def pdf_to_high_res_image_with_crop(pdf, dpi=300):
    doc = fitz.open("pdf", pdf)
    if len(doc) == 0:
        raise ValueError("The PDF file has no pages.")

    page = doc.load_page(0)

    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    img_np = np.array(img)

    rows = np.any(img_np != [255, 255, 255], axis=1)
    cols = np.any(img_np != [255, 255, 255], axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    cropped_img = img.crop((xmin, ymin, xmax, ymax))

    doc.close()
    return cropped_img