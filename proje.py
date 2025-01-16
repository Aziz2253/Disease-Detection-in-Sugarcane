import gradio as gr

import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

# Modeli yükleme
model = load_model("model2.1_transfer.sugarcane.keras")

# Görüntü sınıflandırma fonksiyonu
def classify_image(img):
    img = image.load_img(img, target_size=(128, 128))  # Görüntü boyutunu modele uygun şekilde ayarlama
    img_array = image.img_to_array(img) / 255.0  # Görüntüyü normalize etme
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutunu ekleme

    prediction = model.predict(img_array)
    class_names = ["Healthy", "Mosaic", "Redrot", "Rust", "Yellow"]  # Sınıf isimleri
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return f"Tahmini Sınıf : {predicted_class}  \nDoğruluk Oranı: {confidence:.2f}"

# Gradio arayüzü oluşturma
interface = gr.Interface(
    fn=classify_image, 
    inputs=gr.Image(type="filepath"), 
    outputs="text",
    title="Sugarcane Disease Classifier",
    description="Upload an image of a sugarcane leaf to classify its condition."
)

# Arayüzü başlatma
interface.launch()
