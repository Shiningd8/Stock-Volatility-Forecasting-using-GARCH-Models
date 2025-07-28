#!/usr/bin/env python3
"""
Streamlit Dashboard for Stock Volatility Forecasting using GARCH Models.
Provides interactive web interface for data analysis, model fitting, and forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import StockDataLoader
from src.garch_models import GARCHModel, EGARCHModel, GJR_GARCHModel, GARCHModelComparison
from src.visualization import VolatilityVisualizer
from src.forecasting import VolatilityForecaster
from src.utils import (
    validate_ticker, validate_period, validate_model_type,
    create_directories, save_json, generate_html_report, load_config,
    print_banner, print_section, print_success, print_error, print_info
)


def main():
    """
    🎯 Main Streamlit dashboard function - the interactive web interface for all your volatility needs
    """
    
    # Page configuration
    st.set_page_config(
        page_title="Stock Volatility Forecasting",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("📊 Stock Volatility Forecasting using GARCH Models")
    st.markdown("""
    🔥 This application provides comprehensive volatility analysis and forecasting using GARCH models.
    Upload stock data or fetch from Yahoo Finance to analyze volatility patterns and generate forecasts.
    """)
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Yahoo Finance", "Upload CSV"]
    )
    
    if data_source == "Yahoo Finance":
        # Yahoo Finance parameters
        ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
        period = st.sidebar.selectbox(
            "Time Period",
            ["1y", "2y", "5y", "10y", "max"],
            index=2
        )
        
        # Validate ticker
        if not validate_ticker(ticker):
            st.sidebar.error("Invalid ticker symbol")
            return
        
        # Load data button
        if st.sidebar.button("📥 Load Data"):
            with st.spinner("Loading data..."):
                try:
                    loader = StockDataLoader()
                    data, returns, volatility = loader.get_data(ticker, period)
                    
                    # Store in session state
                    st.session_state['data'] = data
                    st.session_state['returns'] = returns
                    st.session_state['volatility'] = volatility
                    st.session_state['ticker'] = ticker
                    st.session_state['period'] = period
                    
                    st.success(f"✅ Data loaded successfully for {ticker}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
                    return
    
    else:
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file with columns: Date, Open, High, Low, Close, Volume"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
                
                # Calculate returns and volatility
                returns = data['Close'].pct_change().dropna()
                volatility = returns.rolling(window=20).std() * np.sqrt(252)
                
                # Store in session state
                st.session_state['data'] = data
                st.session_state['returns'] = returns
                st.session_state['volatility'] = volatility
                st.session_state['ticker'] = "UPLOADED"
                st.session_state['period'] = "CUSTOM"
                
                st.success("✅ Data uploaded successfully")
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                return
    
    # Model configuration
    st.sidebar.header("🔧 Model Settings")
    
    model_type = st.sidebar.selectbox(
        "GARCH Model Type",
        ["GARCH", "EGARCH", "GJR-GARCH"],
        help="Select the type of GARCH model to use"
    )
    
    compare_models = st.sidebar.checkbox(
        "Compare Multiple Models",
        help="Fit and compare multiple GARCH models"
    )
    
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=1,
        max_value=30,
        value=10,
        help="Number of days to forecast"
    )
    
    perform_backtest = st.sidebar.checkbox(
        "Perform Backtesting",
        help="Perform backtesting to validate model performance"
    )
    
    # Main content area
    if 'data' in st.session_state and 'returns' in st.session_state and 'volatility' in st.session_state:
        data = st.session_state['data']
        returns = st.session_state['returns']
        volatility = st.session_state['volatility']
        ticker = st.session_state['ticker']
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", 
            "🔧 Model Fitting", 
            "📈 Forecasting", 
            "🧪 Backtesting",
            "📋 Reports"
        ])
        
        with tab1:
            st.header("📊 Data Overview")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Data Points", len(data))
                st.metric("Date Range", f"{data.index[0].date()} to {data.index[-1].date()}")
            
            with col2:
                st.metric("Mean Return", f"{returns.mean()*100:.4f}%")
                st.metric("Return Std", f"{returns.std()*100:.4f}%")
            
            with col3:
                st.metric("Skewness", f"{returns.skew():.4f}")
                st.metric("Kurtosis", f"{returns.kurtosis():.4f}")
            
            with col4:
                st.metric("Min Return", f"{returns.min()*100:.4f}%")
                st.metric("Max Return", f"{returns.max()*100:.4f}%")
            
            # Data visualization
            st.subheader("📈 Data Visualization")
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Stock Price', 'Daily Returns', 'Rolling Volatility'),
                vertical_spacing=0.08,
                shared_xaxes=True
            )
            
            # Stock price
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ),
                row=1, col=1
            )
            
            # Returns
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=returns * 100,
                    mode='lines',
                    name='Returns (%)',
                    line=dict(color='#2ca02c', width=1)
                ),
                row=2, col=1
            )
            
            # Volatility
            fig.add_trace(
                go.Scatter(
                    x=volatility.index,
                    y=volatility * 100,
                    mode='lines',
                    name='Volatility (%)',
                    line=dict(color='#d62728', width=2)
                ),
                row=3, col=1
            )
            
            fig.update_layout(
                title=f'{ticker} - Data Overview',
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("🔧 Model Fitting")
            
            if st.button("🚀 Fit Model"):
                with st.spinner("Fitting model..."):
                    try:
                        if compare_models:
                            # Compare multiple models
                            comparison = GARCHModelComparison()
                            comparison_results = comparison.fit_models(returns)
                            
                            # Display comparison results
                            st.subheader("🏆 Model Comparison Results")
                            
                            # Create comparison table
                            comparison_data = []
                            for model_name, metrics in comparison_results.items():
                                if model_name not in ['best_aic', 'best_bic']:
                                    comparison_data.append({
                                        'Model': model_name,
                                        'AIC': f"{metrics['aic']:.4f}",
                                        'BIC': f"{metrics['bic']:.4f}",
                                        'Log-Likelihood': f"{metrics['log_likelihood']:.4f}"
                                    })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Get best model
                            best_model_name, best_model = comparison.get_best_model('aic')
                            st.success(f"🏆 Best model by AIC: {best_model_name}")
                            
                            # Store results
                            st.session_state['model_results'] = comparison.results[best_model_name]
                            st.session_state['model_results']['model_object'] = best_model
                            st.session_state['comparison_results'] = comparison_results
                            
                        else:
                            # Fit single model
                            if model_type == 'GARCH':
                                model = GARCHModel()
                            elif model_type == 'EGARCH':
                                model = EGARCHModel()
                            elif model_type == 'GJR-GARCH':
                                model = GJR_GARCHModel()
                            
                            model_results = model.fit(returns)
                            model_results['model_object'] = model
                            
                            st.session_state['model_results'] = model_results
                            
                            # Display model results
                            st.subheader("📊 Model Results")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("AIC", f"{model_results['aic']:.4f}")
                            with col2:
                                st.metric("BIC", f"{model_results['bic']:.4f}")
                            with col3:
                                st.metric("Log-Likelihood", f"{model_results['log_likelihood']:.4f}")
                            
                            # Model parameters
                            st.subheader("🔧 Model Parameters")
                            params_df = pd.DataFrame([
                                {'Parameter': k, 'Value': f"{v:.6f}"} 
                                for k, v in model_results['params'].items()
                            ])
                            st.dataframe(params_df, use_container_width=True)
                        
                        st.success("✅ Model fitted successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error fitting model: {str(e)}")
        
        with tab3:
            st.header("📈 Forecasting")
            
            if 'model_results' in st.session_state:
                if st.button("🔮 Generate Forecast"):
                    with st.spinner("Generating forecast..."):
                        try:
                            forecaster = VolatilityForecaster()
                            forecast_results = forecaster.generate_forecast(
                                st.session_state['model_results'],
                                horizon=forecast_horizon
                            )
                            
                            st.session_state['forecast_results'] = forecast_results
                            
                            # Display forecast results
                            st.subheader("📊 Forecast Results")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Forecast Volatility", 
                                    f"{forecast_results['volatility_forecast']*100:.4f}%"
                                )
                            with col2:
                                st.metric(
                                    "Forecast Variance", 
                                    f"{forecast_results['variance_forecast']:.6f}"
                                )
                            with col3:
                                st.metric(
                                    "Horizon", 
                                    f"{forecast_results['horizon']} days"
                                )
                            
                            # Forecast plot
                            st.subheader("📈 Forecast Visualization")
                            
                            # Create forecast plot
                            fig = go.Figure()
                            
                            # Historical volatility
                            historical_vol = volatility.dropna()
                            fig.add_trace(
                                go.Scatter(
                                    x=historical_vol.index,
                                    y=historical_vol * 100,
                                    mode='lines',
                                    name='Historical Volatility (%)',
                                    line=dict(color='#1f77b4', width=2)
                                )
                            )
                            
                            # Forecast
                            forecast_dates = forecast_results['forecast_dates']
                            forecast_vol = forecast_results['volatility_forecast']
                            fig.add_trace(
                                go.Scatter(
                                    x=forecast_dates,
                                    y=[forecast_vol * 100] * len(forecast_dates),
                                    mode='lines+markers',
                                    name='Forecast Volatility (%)',
                                    line=dict(color='red', width=3, dash='dash'),
                                    marker=dict(size=8)
                                )
                            )
                            
                            fig.update_layout(
                                title=f'{ticker} - Volatility Forecast',
                                xaxis_title='Date',
                                yaxis_title='Volatility (%)',
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.success("✅ Forecast generated successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Error generating forecast: {str(e)}")
            else:
                st.warning("⚠️ Please fit a model first in the 'Model Fitting' tab.")
        
        with tab4:
            st.header("🧪 Backtesting")
            
            if 'model_results' in st.session_state and perform_backtest:
                if st.button("🧪 Run Backtest"):
                    with st.spinner("Running backtest..."):
                        try:
                            forecaster = VolatilityForecaster()
                            backtest_results = forecaster.backtest_model(
                                returns, 
                                st.session_state['model_results']
                            )
                            
                            st.session_state['backtest_results'] = backtest_results
                            
                            # Display backtest results
                            st.subheader("📊 Backtest Results")
                            
                            metrics = backtest_results['performance_metrics']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("RMSE", f"{metrics['rmse']:.6f}")
                            with col2:
                                st.metric("MAE", f"{metrics['mae']:.6f}")
                            with col3:
                                st.metric("R²", f"{metrics['r_squared']:.4f}")
                            with col4:
                                st.metric("Directional Accuracy", f"{metrics['directional_accuracy']:.4f}")
                            
                            # Backtest plot
                            st.subheader("📈 Backtest Visualization")
                            
                            fig = go.Figure()
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=backtest_results['forecast_dates'],
                                    y=[v * 100 for v in backtest_results['actual_volatility']],
                                    mode='lines',
                                    name='Actual Volatility (%)',
                                    line=dict(color='blue', width=2)
                                )
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=backtest_results['forecast_dates'],
                                    y=[v * 100 for v in backtest_results['predicted_volatility']],
                                    mode='lines',
                                    name='Predicted Volatility (%)',
                                    line=dict(color='red', width=2)
                                )
                            )
                            
                            fig.update_layout(
                                title=f'{ticker} - Backtest Results',
                                xaxis_title='Date',
                                yaxis_title='Volatility (%)',
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.success("✅ Backtest completed successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Error in backtesting: {str(e)}")
            else:
                st.warning("⚠️ Please fit a model first and enable backtesting.")
        
        with tab5:
            st.header("📋 Reports")
            
            if 'model_results' in st.session_state and 'forecast_results' in st.session_state:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Save Results"):
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            results = {
                                'ticker': ticker,
                                'model_results': st.session_state['model_results'],
                                'forecast_results': st.session_state['forecast_results'],
                                'timestamp': timestamp
                            }
                            
                            if 'backtest_results' in st.session_state:
                                results['backtest_results'] = st.session_state['backtest_results']
                            
                            if 'comparison_results' in st.session_state:
                                results['comparison_results'] = st.session_state['comparison_results']
                            
                            save_json(results, f"reports/{ticker}_analysis_{timestamp}.json")
                            st.success("✅ Results saved successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Error saving results: {str(e)}")
                
                with col2:
                    if st.button("📄 Generate HTML Report"):
                        try:
                            # Create a simple HTML report content
                            html_content = f"""
                            <h2>📊 Volatility Analysis Report for {ticker}</h2>
                            <div class="metric">
                                <h3>Model Results</h3>
                                <p>AIC: {st.session_state['model_results']['aic']:.4f}</p>
                                <p>BIC: {st.session_state['model_results']['bic']:.4f}</p>
                                <p>Log-Likelihood: {st.session_state['model_results']['log_likelihood']:.4f}</p>
                            </div>
                            <div class="metric">
                                <h3>Forecast Results</h3>
                                <p>Forecast Volatility: {st.session_state['forecast_results']['volatility_forecast']*100:.4f}%</p>
                                <p>Forecast Variance: {st.session_state['forecast_results']['variance_forecast']:.6f}</p>
                                <p>Horizon: {st.session_state['forecast_results']['horizon']} days</p>
                            </div>
                            """
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            generate_html_report(f"Volatility Report - {ticker}", html_content, f"reports/{ticker}_report_{timestamp}.html")
                            st.success("✅ HTML report generated successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Error generating report: {str(e)}")
            else:
                st.warning("⚠️ Please fit a model and generate forecast first.")
    
    else:
        st.info("👆 Please load data using the sidebar to begin analysis.")


if __name__ == "__main__":
    main() 