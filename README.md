[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-orange?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/your-username/ann-classification-churn-dashboard)](https://github.com/your-username/ann-classification-churn-dashboard)

# 📊 Customer Churn Prediction Dashboard

An interactive dashboard powered by **Artificial Neural Networks (ANNs)** for accurate, real-time customer churn prediction. Explore the entire ML lifecycle—data preprocessing, model training, and live deployment—with a feature-rich **Streamlit** web application.

***

## 🔗 Live Demo

👉 [Customer Churn Dashboard](https://ann-classification-churn-dashboard-jn4qsgybr5bswajttbm9lu.streamlit.app/)

***

## 📂 Project Structure

```plaintext
ann-classification-churn-dashboard/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── data/
│   └── churn.csv        # Sample customer churn dataset
├── models/
│   └── ann_model.h5     # Trained ANN model (TensorFlow/Keras)
├── notebooks/
│   └── training.ipynb   # Model training & experiments
└── README.md            # Project documentation
```


***

## 🚀 Features

- Interactive **Streamlit dashboard** for live churn prediction and feature analysis
- **ANN Model** trained using structured, preprocessed customer data
- Robust model evaluation: **accuracy**, **ROC-AUC**, **classification report**
- **Model Explainability** with integrated **SHAP** visualizations
- Real-time predictions with custom user input
- End-to-end deployment: Cloud-hosted via Streamlit Cloud

***

## 📊 Dataset

- Includes customer **demographics**, subscription \& account activity
- **Target column**: `Churn` (Yes/No)
- Preprocessing:
    - Encoding of categorical features
    - Scaling of numeric features
    - Data balancing for class distribution

***

## 🤖 Model

- **Framework**: TensorFlow/Keras (Sequential ANN)
- **Preprocessing**: Via scikit-learn pipelines
- **Architecture**:
    - Multiple dense hidden layers (`ReLU` activations)
    - Output layer: `Sigmoid` for binary classification
    - Loss: `Binary Cross-Entropy`
    - Optimizer: `Adam`
- **Performance**:
    - Accuracy: ~85%
    - ROC-AUC: ~0.90
    - Balanced precision/recall on churn/no-churn
- **Artifacts**:
    - `models/ann_model.h5` (trained model)
    - `notebooks/training.ipynb` (all code \& analysis)

***

## 📈 Tech Stack

- **Frontend \& Deployment**: Streamlit, Streamlit Cloud
- **Machine Learning \& Model**: TensorFlow/Keras, scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, SHAP

***

## 🧪 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ann-classification-churn-dashboard.git
cd ann-classification-churn-dashboard
```


### 2. Create a Virtual Environment

```bash
python -m venv venv
# On Unix/macOS
source venv/bin/activate
# On Windows
venv\Scripts\activate
```


### 3. Install Requirements

```bash
pip install -r requirements.txt
```


### 4. Run the App Locally

```bash
streamlit run app.py
```


***

## ✅ Deployment

The app is deployed on Streamlit Cloud for quick and easy access:

👉 [Customer Churn Dashboard](https://ann-classification-churn-dashboard-jn4qsgybr5bswajttbm9lu.streamlit.app/)

***

## 📝 File Descriptions

| File/Directory | Description |
| :-- | :-- |
| app.py | Main Streamlit web app |
| requirements.txt | Required Python packages |
| data/churn.csv | Sample dataset for churn prediction |
| models/ann_model.h5 | Trained Keras/TensorFlow ANN model |
| notebooks/training.ipynb | Jupyter notebook with full pipeline |


***

## 🔍 Model Explainability

- **SHAP** integration provides feature importance and individualized explanations for every prediction.
- Visualizes how each feature impacts churn risk, improving model transparency.

***

## 🛠 Future Improvements

- Explore advanced models (e.g., CNNs for embeddings, transformers)
- Live customer data API integration
- Interactive charts with Plotly
- User authentication for secure access

***

## 📜 License

This project is open source, available under the [MIT License](LICENSE).

***

## 🤝 Acknowledgements

- Built using open-source frameworks and inspired by leading churn prediction projects.

***

**Made with ❤️ by Navneeth**

***
