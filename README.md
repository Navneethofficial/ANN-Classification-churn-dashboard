# 📊 Customer Churn Prediction Dashboard

An interactive dashboard powered by **Artificial Neural Networks (ANNs)** for predicting customer churn.  
This project demonstrates end-to-end machine learning, including **data preprocessing, deep learning model training**, and a **Streamlit-based web application** for interactive predictions and insights.

🔗 **Live Demo**: [Customer Churn Dashboard](https://ann-classification-churn-dashboard-jn4qsgybr5bswajttbm9lu.streamlit.app/)

---

## 🔍 Project Structure

ann-classification-churn-dashboard:
  - app.py                # Main Streamlit app
  - requirements.txt      # Python dependencies
  
  data:                   # Dataset folder
    - churn.csv           # Sample customer churn data

  models:                 # Trained model artifacts
    - ann_model.h5        # Saved ANN model

  notebooks:              # Jupyter notebooks
    - training.ipynb      # Model training & experiments

  - README.md             # Project documentation


---

## 🚀 Features

- 📊 **Interactive Streamlit dashboard** for churn prediction  
- 🤖 **ANN Model** trained on structured customer data  
- 📈 Model evaluation with **Accuracy, ROC-AUC, and Classification Report**  
- 🔍 **Explainability with SHAP** for feature importance and interpretability  
- ⚡ **Real-time predictions** with custom user input  
- ☁️ Deployed seamlessly on **Streamlit Cloud**  

---

## 📂 Dataset

The dataset consists of customer details such as:

- 👤 Demographic information  
- 📦 Subscription details  
- 📡 Account activity  
- 🎯 Target column: **Churn** (Yes/No)  

Data is preprocessed with **encoding, scaling, and balancing techniques** before training.

---

## 🤖 Model

The model is a **deep learning Artificial Neural Network** built with **TensorFlow/Keras**.  

### 🔧 Training Highlights:
- Input preprocessing with **scikit-learn**  
- Multiple hidden layers with **ReLU activation**  
- Output layer with **sigmoid activation** for binary classification  
- Optimized with **Adam optimizer** and **binary cross-entropy loss**  

### 📊 Performance:
- ✅ Accuracy: ~85%  
- 📉 ROC-AUC: ~0.90  
- ⚖️ Balanced metrics across churn/no-churn classes  

**Artifacts:**
- `ann_model.h5` → Trained ANN model  
- `training.ipynb` → Notebook with full training pipeline  

---

## 💻 Tech Stack

- **Frontend & Deployment**: Streamlit, Streamlit Cloud  
- **Machine Learning**: TensorFlow/Keras, Scikit-learn  
- **Data Processing**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn, SHAP  

---

## 🧪 How to Run Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/ann-classification-churn-dashboard.git
   cd ann-classification-churn-dashboard
Create a Virtual Environment

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install Requirements

bash
Copy code
pip install -r requirements.txt
Run the App

bash
Copy code
streamlit run app.py
✅ Deployment
The project is deployed on Streamlit Cloud.
👉 Live Dashboard

🛠 Future Improvements
🔮 Add more advanced deep learning models (CNNs, Transformers for embeddings)

🔗 Integrate with live customer data APIs

📊 Enhance visualization with Plotly for interactive charts

🔒 Add user authentication for secure access

📜 License
This project is open-source and available under the MIT License.

✨ Made with ❤️ by Navneeth
