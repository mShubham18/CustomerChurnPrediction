# Customer Churn Prediction System 🔄

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-prediction-system.streamlit.app/)

A Deep-learning application that predicts customer churn probability using Artificial Neural Networks (ANN). This system helps businesses identify customers who are likely to discontinue their services, enabling proactive retention strategies.

## 🌟 Live Demo
Try out the live application: [Customer Churn Predictor](https://customer-churn-prediction-system.streamlit.app/)

## 🎯 Features

- Real-time churn prediction
- User-friendly web interface built with Streamlit
- Comprehensive customer attribute analysis
- Instant probability scores and interpretations
- Support for multiple geographical regions

## 📊 Input Features

The model considers the following customer attributes:
- Credit Score
- Geography (France, Germany, Spain)
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

## 🛠️ Technology Stack

- **Python 3.x**
- **TensorFlow 2.15.0** - Deep Learning Framework
- **Streamlit** - Web Interface
- **Scikit-learn** - Data Preprocessing
- **Pandas** - Data Manipulation
- **NumPy** - Numerical Operations
- **Matplotlib/Seaborn** - Data Visualization

## 🚀 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/mShubham18/CustomerChurnPrediction.git
   cd CustomerChurnPrediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run CustomerChurnPrediction.py
   ```

## 📁 Project Structure

```
CustomerChurnPrediction/
├── .streamlit/          # Streamlit configuration
├── Notebooks/           # Jupyter notebooks for development
├── models/             # Saved model files
│   ├── model.h5        # Trained ANN model
│   ├── std_scaler.pkl  # Standard scaler
│   ├── label_encoder_gender.pkl
│   └── one_hot_encoder_geography.pkl
├── CustomerChurnPrediction.py  # Main Streamlit application
├── prediction.ipynb    # Model development notebook
└── requirements.txt    # Project dependencies
```

## 🤖 Model Architecture

The system uses an Artificial Neural Network (ANN) with:
- Input layer for customer features
- Multiple hidden layers with ReLU activation
- Output layer with sigmoid activation for binary classification
- Binary cross-entropy loss function
- Adam optimizer

## 🔧 Usage

1. Navigate to the web interface
2. Input customer details in the provided fields
3. Click "Predict Churn" to get the prediction
4. View the churn probability and interpretation
5. Use "Reset" to make another prediction

Contributions are welcome! Please feel free to submit a Pull Request.

## 🔗 Links
- [GitHub Repository](https://github.com/mShubham18/CustomerChurnPrediction)
- [Live Application](https://customer-churn-prediction-system.streamlit.app/) 