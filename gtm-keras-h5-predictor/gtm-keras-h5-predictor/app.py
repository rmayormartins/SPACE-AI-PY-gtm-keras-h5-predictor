import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import zipfile
import tempfile
import os
#######
def classify_images(uploaded_file, images):
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        model_path = os.path.join(temp_dir, 'keras_model.h5')
        labels_path = os.path.join(temp_dir, 'labels.txt')

        
        loaded_model = load_model(model_path)

        
        with open(labels_path, 'r') as file:
            class_names = file.read().splitlines()

        
        predictions = []
        for img in images:
            with Image.open(img) as pil_image:
                image = pil_image.resize((224, 224))
                image = np.array(image)
                image = image / 255.0
                image = np.expand_dims(image, axis=0)

                prediction = loaded_model.predict(image)
                predicted_class = class_names[np.argmax(prediction)]
                predictions.append(predicted_class)

        return predictions

# 

# Gradio interface
iface = gr.Interface(
    fn=classify_images,
    inputs=[
        gr.File(label="Upload do Modelo (.h5 or .zip (with .h5))"),
        gr.Files(label="Upload of Images")
    ],
    outputs="text",
    title="GTM-Keras-h5-Predictor",
    description="In Google Teachable Machine, after training, under 'Export Model', go to 'Tensorflow', click on 'Keras' and then 'Download my model' (wait a moment). The zip will contain the Keras .h5 model.",
    examples=[["converted_keras.zip", ["example1.jpg", "example2.jpg"]]]
)

if __name__ == "__main__":
    iface.launch(debug=True)
