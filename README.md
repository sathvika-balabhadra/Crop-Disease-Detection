## 🪴 Plant Disease Detection

The Plant Disease Detection system is an AI-powered application that helps farmers and agronomists identify crop diseases in real time. Users upload an image of a plant leaf, which is then analyzed by a Convolutional Neural Network (CNN) model hosted on a Streamlit web interface. The model classifies the type of plant, detects any disease, and provides recommended treatments or preventive actions.

The application is designed for accessibility and efficiency which enabling farmers to get accurate diagnostic results without specialized equipment, supporting better decision-making and early intervention.

# Features

1. **Image-based Disease Detection**: Upload plant images for instant analysis and disease identification.
2. **Multi-Class Classification**: Detects both plant species and corresponding diseases from a diverse dataset.
3. **AI-Powered Predictions**: Uses a CNN trained on thousands of labeled plant images from Kaggle.
4. **Treatment Recommendations**: Displays actionable suggestions based on disease type.
5. **Streamlit Web Interface**: Simple and interactive UI for real-time results.
6. **Visualization Tools**: Shows prediction confidence and visual analysis using Matplotlib.

# Model

The CNN model is trained to classify:
- 5 classes for disease detection.
- 24 classes for detailed disease classification.

Dataset can be downloaded form [kaggle](https://www.kaggle.com/abdallahalidev/plantvillage-dataset)

- Disease Classification Classes

                       - Apple___Apple_scab
                       - Apple___Black_rot
			   - Apple___Cedar_apple_rust
			   - Apple___healthy
			   - Blueberry___healthy
			   - Cherry___healthy
			   - Cherry___Powdery_mildew
			   - Grape___Black_rot
			   - Grape___Esca_Black_Measles
			   - Grape___healthy
			   - Grape___Leaf_blight_Isariopsis_Leaf_Spot
			   - Orange___Haunglongbing
			   - Peach___Bacterial_spot
			   - Peach___healthy
			   - Pepper,_bell___Bacterial_spot
			   - Pepper,_bell___healthy
			   - Potato___Early_blight
			   - Potato___healthy
			   - Raspberry___healthy
			   - Soybean___healthy
			   - Squash___Powdery_mildew
			   - Strawberry___healthy
			   - Strawberry___Leaf_scorch
			
            - Disease Detection Classes
            
			   - Cherry___healthy
			   - Cherry___Powdery_mildew
			   - Grape___Black_rot
			   - Grape___Esca_Black_Measles
			   - Grape___healthy
			   - Grape___Leaf_blight_Isariopsis_Leaf_Spot
  
# Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy & Pandas
- Matplotlib
- Streamlit

# How It Works

- User uploads an image of a plant leaf through the Streamlit interface.
- The image is preprocessed using OpenCV (resizing, normalization, noise reduction).
- The CNN model classifies the image into one of the predefined disease categories.
- The system returns the predicted disease, confidence score, and recommended treatment.
- Results are visualized and displayed instantly to the user.

# Required Libraries
- opencv-contrib-python-headless
- tensorflow-cpu
- streamlit
- numpy
- pandas  
pillow  
keras  
matplotlib  
