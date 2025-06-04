# MEDIINSIGHT - Multi Disease Predictor

Mediinsight is a Flask-based web application that leverages machine learning and deep learning models to predict the likelihood of several diseases based on user input or medical images. The application provides an easy-to-use interface for both tabular data (for diseases like diabetes, heart, kidney, liver, and breast cancer) and image-based predictions (for malaria and pneumonia).

## Features

- **Predicts Multiple Diseases:**  
  Supports prediction for Diabetes, Heart Disease, Kidney Disease, Liver Disease, Breast Cancer, Malaria, and Pneumonia.

- **Tabular & Image Input:**  
  Accepts both form-based (tabular) data and medical images (cell and X-ray images) for disease prediction.

- **Deep Learning Integration:**  
  Uses pre-trained Keras/TensorFlow models for image-based disease detection.

- **User-Friendly Interface:**  
  Built with Flask and Jinja2 templates for a simple and interactive web experience.

- **Health Tips:**  
  Displays random health tips on the homepage to encourage healthy habits.

## How It Works

1. **User selects a disease** and provides the required input (form data or image upload).
2. **The app preprocesses the input** and feeds it to the corresponding trained model.
3. **Prediction results** are displayed along with relevant health information.

## Technologies Used

- Python 3.6+
- Flask
- TensorFlow / Keras
- scikit-learn
- Pillow (for image processing)
- HTML/CSS (Jinja2 templates)
- Docker support for easy deployment

## Getting Started

1. Clone the repository.
2. Install dependencies from requirements.txt.
3. (Optional) Use Docker for containerized deployment.
4. Run app.py and open the web interface in your browser.

## Note

- Large datasets and model files (such as validator and chest_xray) are **not included** in the repository.
- Please add your own trained models and datasets as needed.

---

**Empower your healthcare predictions with Multi Disease Predictor!**
