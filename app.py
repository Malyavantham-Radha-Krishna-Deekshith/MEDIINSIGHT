from flask import Flask, render_template, request, flash, redirect, url_for
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import random

app = Flask(__name__)

# Load the validity classifier
validity_model = load_model("models/validity_classifier.h5")

def classify_validity(image):
    try:
        img = Image.open(image).convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        prediction = validity_model.predict(img_array)[0][0]
        return prediction >= 0.5  # True if valid, False if invalid
    except Exception as e:
        print("Validity classification error:", e)
        return False

# Dummy advanced validation (you should define this properly)
def advanced_validate_image(image_file, is_grayscale=False):
    try:
        image = Image.open(image_file)
        if is_grayscale and image.mode != 'L':
            return False, "Image is not grayscale as expected."
        if not is_grayscale and image.mode == 'L':
            return False, "Expected a color image, but got grayscale."
        return True, "Valid image"
    except Exception as e:
        print("Advanced validation error:", e)
        return False, "Image validation failed."

def predict(values, dic):
    values = np.asarray(values)
    try:
        if len(values) == 8:
            model = pickle.load(open('Python Notebooks/diabetes.pkl', 'rb'))
        elif len(values) == 26:
            model = pickle.load(open('models/breast_cancer.pkl', 'rb'))
        elif len(values) == 13:
            model = pickle.load(open('models/heart.pkl', 'rb'))
        elif len(values) == 18:
            model = pickle.load(open('D:/Multi_Disease_Predictor/Python Notebooks/kidney.pkl', 'rb'))
        elif len(values) == 10:
            model = pickle.load(open('D:/Multi_Disease_Predictor/Python Notebooks/liver.pkl', 'rb'))
        else:
            return "Invalid Input"
        return model.predict(values.reshape(1, -1))[0]
    except Exception as e:
        print("Prediction error:", e)
        return "Error in prediction"

@app.route("/")
def home():
    tips = [
        "🚶‍♂️ Take a 30-minute walk daily for heart health.",
        "💧 Stay hydrated—drink at least 2 liters of water daily.",
        "🧘 Practice deep breathing for 5 minutes to reduce stress.",
        "🍎 Eat a variety of colorful fruits and vegetables.",
        "😴 Get 7-8 hours of quality sleep every night.",
        "🦷 Brush and floss regularly for oral and heart health.",
        "🛡️ Wash hands often to prevent infections.",
        "☀️ Spend 10–15 mins in sunlight for vitamin D."
    ]
    tip = random.choice(tips)
    return render_template('home.html', tip=tip)

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods=['POST'])
def predictPage():
    try:
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))

        disease_map = {
            8: "Diabetes",
            15: "Breast Cancer",
            13: "Heart Disease",
            18: "Kidney Disease",
            10: "Liver Disease"
        }

        input_length = len(to_predict_list)
        disease = disease_map.get(input_length, "Unknown Disease")
        pred = predict(to_predict_list, to_predict_dict)

        return render_template('predict.html', pred=pred, disease=disease)

    except Exception as e:
        print("General prediction error:", e)
        message = "Please enter valid data"
        tip = random.choice([
            "🧼 Wash hands often to avoid infections.",
            "🥗 Include fiber-rich foods to improve digestion."
        ])
        return render_template("home.html", message=message, tip=tip)

@app.route("/malariapredict", methods=['POST'])
def malariapredictPage():
    try:
        if 'image' in request.files:
            file = request.files['image']
            print("🖼️ Malaria - Step 1: File received")

            file.seek(0)
            if not classify_validity(file):
                print("❌ Malaria - Step 2: Image failed validity classification")
                return render_template('malaria.html', message="Image classified as INVALID for malaria diagnosis.")

            file.seek(0)
            is_valid, message = advanced_validate_image(file, is_grayscale=False)
            if not is_valid:
                print("❌ Malaria - Step 3: Advanced validation failed:", message)
                return render_template('malaria.html', message=message)

            file.seek(0)
            img = Image.open(file).resize((36, 36))
            img = np.asarray(img).reshape((1, 36, 36, 3)).astype(np.float64)

            print("📦 Malaria - Step 4: Predicting with model")
            model = load_model("models/malaria.h5")
            pred = np.argmax(model.predict(img)[0])
            return render_template('malaria_predict.html', pred=pred)
        else:
            print("⚠️ Malaria - No image uploaded")
            return render_template('malaria.html', message="Please upload an image.")
    except Exception as e:
        print("⚠️ Malaria prediction error:", e)
        return render_template('malaria.html', message="Something went wrong.")

@app.route("/pneumoniapredict", methods=['POST'])
def pneumoniapredictPage():
    try:
        if 'image' in request.files:
            file = request.files['image']
            print("🖼️ Pneumonia - Step 1: File received")

            file.seek(0)
            if not classify_validity(file):
                print("❌ Pneumonia - Step 2: Image failed validity classification")
                return render_template('pneumonia.html', message="Image classified as INVALID for pneumonia diagnosis.")

            file.seek(0)
            is_valid, message = advanced_validate_image(file, is_grayscale=True)
            if not is_valid:
                print("❌ Pneumonia - Step 3: Advanced validation failed:", message)
                return render_template('pneumonia.html', message=message)

            file.seek(0)
            img = Image.open(file).convert('L').resize((36, 36))
            img = np.asarray(img).reshape((1, 36, 36, 1)) / 255.0

            print("📦 Pneumonia - Step 4: Predicting with model")
            model = load_model("models/pneumonia.h5")
            pred = np.argmax(model.predict(img)[0])
            return render_template('pneumonia_predict.html', pred=pred)
        else:
            print("⚠️ Pneumonia - No image uploaded")
            return render_template('pneumonia.html', message="Please upload an image.")
    except Exception as e:
        print("⚠️ Pneumonia prediction error:", e)
        return render_template('pneumonia.html', message="Something went wrong.")

if __name__ == '__main__':
    app.run(debug=True)
