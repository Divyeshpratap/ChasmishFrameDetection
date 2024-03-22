from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import model_from_json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load and prepare datasets
df = pd.read_csv('Datasets/chasmish_products.csv').drop(columns=['Unnamed: 0'])
df['Image_Front'] = df['Image_Front'].str.replace(' ', '%20', regex=True)
y_set = np.load('product_id_array.npy')
features = np.load('vgg_features_array.npy').reshape(len(y_set), -1)
feature_dict = dict(zip(y_set, features))

# Load model
with open('model.json', 'r') as json_file:
    loaded_model = model_from_json(json_file.read())
loaded_model.load_weights("model.h5")

# VGG16 model for feature extraction
modelv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
modelv.trainable = False  # Freezing layers

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('predict', filename=filename))
    
    return render_template('upload.html')

@app.route('/predict/<filename>')
def predict(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    feat_upld = modelv.predict(preprocess_input(np.expand_dims(img_array, axis=0))).flatten()
    
    prediction = loaded_model.predict(feat_upld.reshape(1, 7, 7, 512))
    cat_dict = {0: 'Non-Power Reading', 1: 'eyeframe', 2: 'sunglasses'}
    shape_dict = {0: 'Aviator', 1: 'Oval', 2: 'Rectangle', 3: 'Wayfarer'}
    category_pred, shape_pred = np.argmax(prediction[0]), np.argmax(prediction[1])

    filtered_ids = df[(df.parent_category == cat_dict[category_pred]) & (df.frame_shape == shape_dict[shape_pred])]['product_id']
    df_cos = pd.DataFrame({'product_id': filtered_ids, 'score': [cosine_similarity([feat_upld], [feature_dict[id]])[0][0] for id in filtered_ids]})
    df_cos_sim = df_cos.sort_values(by='score', ascending=False).head(10).merge(df[['product_id', 'Image_Front']], on='product_id')

    similar_images = [{'url': row['Image_Front'], 'product_id': row['product_id']} for _, row in df_cos_sim.iterrows()]
    return render_template('display_images.html', uploaded_image_url=url_for('static', filename=f'uploads/{filename}'), similar_images=similar_images)

if __name__ == '__main__':
    app.run(debug=True)
