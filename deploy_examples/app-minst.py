from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo entrenado
model = keras.models.load_model("mnist_model.h5")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de reconocimiento de dígitos MNIST funcionando 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Verificar si se envió un archivo
        if "image" not in request.files:
            return jsonify({"error": "No se envió ninguna imagen"}), 400

        # Leer la imagen del request
        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read())).convert("L")  # Escala de grises

        # Redimensionar a 28x28
        img = img.resize((28, 28))

        # Convertir a arreglo NumPy
        img_array = np.array(img)

        # Invertir colores si el fondo es blanco y el número negro
        # (MNIST tiene fondo negro y número blanco)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array

        # Normalizar y dar forma al arreglo para el modelo
        img_array = img_array.reshape(1, 28 * 28) / 255.0

        # Realizar la predicción
        prediction = model.predict(img_array)
        digit = int(np.argmax(prediction))

        return jsonify({
            "prediction": digit,
            "confidence": float(np.max(prediction))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)