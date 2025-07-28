# Stock-Volatility-Forecasting-using-GARCH-Models

## 📊 Project Overview

This project implements a comprehensive system for forecasting financial market volatility using GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models. The system helps capture time-varying volatility patterns in stock returns, which is crucial for risk management, algorithmic trading, and financial forecasting.

## 🎯 Use Cases

- **Portfolio Risk Assessment**: Quantify uncertainty around returns
- **Options Pricing**: Feed into Black-Scholes-like models for implied volatility
- **Market Stress Prediction**: Anticipate future periods of turbulence
- **Trading Signals**: Identify potential high-volatility days for strategic entry/exit

## 📈 Statistical Foundation

Traditional time series models assume constant variance (homoscedasticity). However, stock return volatility is time-varying with clusters of calm vs. turbulent days.

GARCH models the conditional variance σ²ₜ as a function of past squared returns and past variance:

**σ²ₜ = α₀ + α₁ε²ₜ₋₁ + β₁σ²ₜ₋₁**

The project supports multiple GARCH variants:
- **GARCH(1,1)**: Standard model
- **EGARCH**: Exponential GARCH for asymmetric effects
- **GJR-GARCH**: Glosten-Jagannathan-Runkle GARCH for leverage effects

## 🚀 Features

- 📥 **Data Loading**: Fetch stock data via Yahoo Finance API
- 📊 **Visualization**: Plot daily returns and volatility clustering
- ⚙️ **Model Fitting**: Multiple GARCH models (GARCH(1,1), EGARCH, GJR-GARCH)
- 📈 **Forecasting**: Future volatility with confidence intervals
- 🧪 **Model Comparison**: AIC/BIC metrics and backtesting
- 📉 **Interactive Dashboard**: Streamlit-based UI
- ✅ **Reporting**: Save model metrics and charts

## 🛠️ Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Command Line Interface
```bash
python main.py --ticker AAPL --period 5y --model garch
```

### Streamlit Dashboard
```bash
streamlit run dashboard.py
```

## 📊 Example Output

The system generates:
- Volatility clustering plots
- Model comparison metrics
- Forecast charts with confidence intervals
- Backtesting results
- Comprehensive reports

### Example Dashboard Images 

<img width="2446" height="1419" alt="stat 1" src="https://github.com/user-attachments/assets/0c97f756-1e4a-46b4-96a4-035c152b68a2" />


<img width="2108" height="1373" alt="stat 3" src="https://github.com/user-attachments/assets/2eceef0f-4331-47e0-8b82-45dda2f978ca" />


<img width="2119" height="983" alt="stat 2" src="https://github.com/user-attachments/assets/e28ba260-bbf2-4708-8ba6-f83961b57bf7" />


## 🔧 Configuration

Key parameters can be adjusted in `config.py`:
- Default ticker symbols
- Time periods
- Model parameters
- Confidence intervals

## 📚 References

- Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity"
- Engle, R.F. (1982). "Autoregressive conditional heteroskedasticity"
- Glosten, L.R., Jagannathan, R., & Runkle, D.E. (1993). "On the relation between the expected value and the volatility of the nominal excess return on stocks"

