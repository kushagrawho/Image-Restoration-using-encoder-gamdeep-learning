import os
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from inference import load_model, predict, generate_mask, apply_mask

app = Flask(__name__)

# Setup folder to store the images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained model
model = load_model()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'orig_' + filename)
            file.save(original_path)

            # 1. Prepare image
            image = Image.open(original_path).convert("RGB").resize((128, 128))
            
            # 2. Create damage (Mask)
            mask_np = generate_mask(size=128)
            masked_pil = apply_mask(image, mask_np)
            masked_path = os.path.join(app.config['UPLOAD_FOLDER'], 'masked_' + filename)
            masked_pil.save(masked_path)

            # 3. AI Restoration
            restored_np = predict(image, mask_np, model)
            restored_pil = Image.fromarray(restored_np)
            restored_path = os.path.join(app.config['UPLOAD_FOLDER'], 'restored_' + filename)
            restored_pil.save(restored_path)

            return render_template('index.html', 
                                   orig='orig_'+filename, 
                                   masked='masked_'+filename, 
                                   restored='restored_'+filename)

    return render_template('index.html')

if __name__ == '__main__':
    print("Starting server... Go to http://127.0.0.1:5000")
    app.run(debug=True, port=5000)