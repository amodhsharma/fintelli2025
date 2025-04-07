# Fintelli 
📈 Stock Price Prediction with Multiple ML Models


This project uses 8 machine learning models to predict future stock prices based on historical data. The final output includes predicted prices, model evaluation metrics, and aggregated **BUY/SELL** recommendations.

## 🧠 Machine Learning Models Used
- Linear Regression
- Exponential Smoothening
- ARIMA 
- SARIMA 
- Random Forest 
- XG Boost
- Prophet 
- LSTM 

## 📏 Evaluation Metrics

Each model is evaluated using the following metrics:

- **RMSE (Root Mean Squared Error)** – Measures the average magnitude of prediction errors.
- **MAE (Mean Absolute Error)** – Average of absolute differences between predictions and actual values.
- **MAPE (Mean Absolute Percentage Error)** – Expresses error as a percentage.
- **R² Score (Coefficient of Determination)** – Indicates how well data fits the model (closer to 1 is better).

## 🚀 Future Scope

- ✅ Integration with **real-time stock APIs** for live predictions 
- 🤖 Incorporation of **ensemble models** for improved performance  
- 🔐 User authentication and portfolio tracking features  
- 🧠 Use of **transformers** or more advanced DL models for long-term forecasting

## 🛠️ Installation and execution
Clone the [repo](https://github.com/amodhsharma/fintelli2025)

Extract the content and cd into the directory 

Install Dependencies 

```bash
pip install -r requirements.txt
```

As the application is based on Streamlit, in the directory of the project, type 
```bash
streamlit run main.py
```

The application should open in your browser automatically, if it doesnt, paste - 
```bash 
http://localhost:8000
```

Notes

- Last Stable commit - 8th April 2025 
- Due to limitations of local machine, a few models are capping at certain points
- In the main.py file, one can configure the length of historical data ot be used for training, in this repo- it was taken to be of 10 years
- [Link for the hosted app](https://fintelli2025.streamlit.app)

Hope you have as much fun exploring the app as much as we had fun creating it!
