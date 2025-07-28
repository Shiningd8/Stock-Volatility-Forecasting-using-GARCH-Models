"""
🔮 Volatility Forecasting - Crystal ball for your portfolio
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class VolatilityForecaster:
    """
    🔮 The ultimate volatility forecaster - predicts market chaos like a boss
    """
    
    def __init__(self):
        self.forecasts = {}
        self.backtest_results = {}
    
    def generate_forecast(self, model_results: Dict[str, Any], 
                         horizon: int = 5, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        🔮 Generates volatility forecasts with confidence intervals (the real deal)
        
        Args:
            model_results (Dict): Model fitting results with diagnostics
            horizon (int): How many days ahead to predict
            confidence_level (float): Confidence level for intervals (0.95 = 95%)
            
        Returns:
            Dict: Complete forecast with all the juicy details
        """
        try:
            # Check if we have a model object to work with
            if 'model_object' in model_results:
                model = model_results['model_object']
                # Use the model's built-in forecast method
                forecast = model.forecast(horizon=horizon)
            else:
                # Fallback to simple forecast if no model object
                forecast = self._create_simple_forecast(model_results, horizon)
            
            # Add confidence intervals for extra swag
            forecast = self._add_confidence_intervals(forecast, confidence_level)
            
            # Store the forecast
            self.forecasts[f"forecast_{len(self.forecasts)}"] = forecast
            
            return forecast
            
        except Exception as e:
            print(f"❌ Error in forecasting: {str(e)}")
            raise
    
    def _create_simple_forecast(self, model_results: Dict[str, Any], horizon: int) -> Dict[str, Any]:
        """
        🛠️ Creates a simple forecast when model object isn't available (backup plan)
        
        Args:
            model_results (Dict): Model results
            horizon (int): Forecast horizon
            
        Returns:
            Dict: Simple forecast
        """
        # Extract conditional volatility from model results
        if 'conditional_volatility' in model_results:
            last_vol = model_results['conditional_volatility'].iloc[-1]
        else:
            # Fallback to a reasonable volatility estimate
            last_vol = 0.2  # 20% annualized volatility
        
        # Simple forecast: use the last volatility as a baseline
        forecast = {
            'horizon': horizon,
            'volatility_forecast': last_vol,
            'mean_forecast': 0.0,  # Assume zero mean returns
            'variance_forecast': (last_vol / np.sqrt(252)) ** 2,
            'forecast_dates': pd.date_range(
                start=pd.Timestamp.now(),
                periods=horizon,
                freq='D'
            ),
            'method': 'simple_fallback'
        }
        
        return forecast
    
    def _add_confidence_intervals(self, forecast: Dict[str, Any], confidence_level: float) -> Dict[str, Any]:
        """
        📊 Adds confidence intervals to your forecast (the uncertainty swag)
        
        Args:
            forecast (Dict): Base forecast
            confidence_level (float): Confidence level
            
        Returns:
            Dict: Forecast with confidence intervals
        """
        volatility = forecast['volatility_forecast']
        
        # Calculate confidence intervals using normal distribution
        z_score = norm.ppf((1 + confidence_level) / 2)
        
        # Standard error (simplified)
        se = volatility * 0.1  # Assume 10% standard error
        
        forecast['confidence_intervals'] = {
            'lower': max(0, volatility - z_score * se),
            'upper': volatility + z_score * se,
            'confidence_level': confidence_level
        }
        
        return forecast
    
    def backtest_model(self, returns: pd.Series, model_results: Dict[str, Any], 
                      window: int = 252, horizon: int = 5) -> Dict[str, Any]:
        """
        🧪 Backtests your model to see how good it really is (the truth test)
        
        Args:
            returns (pd.Series): Historical returns
            model_results (Dict): Model results
            window (int): Rolling window size
            horizon (int): Forecast horizon
            
        Returns:
            Dict: Backtesting results with performance metrics
        """
        try:
            actual_volatilities = []
            predicted_volatilities = []
            
            # Rolling window backtest
            for i in range(window, len(returns) - horizon):
                # Get training data
                train_returns = returns.iloc[i-window:i]
                
                # Fit model on training data
                if 'model_object' in model_results:
                    model = model_results['model_object']
                    # Create a new model instance for this window
                    new_model = type(model)()
                    new_model.fit(train_returns)
                    
                    # Generate forecast
                    forecast = new_model.forecast(horizon=horizon)
                    predicted_vol = forecast['volatility_forecast']
                else:
                    # Simple volatility estimate
                    predicted_vol = train_returns.std() * np.sqrt(252)
                
                # Actual volatility (realized)
                actual_vol = returns.iloc[i:i+horizon].std() * np.sqrt(252)
                
                actual_volatilities.append(actual_vol)
                predicted_volatilities.append(predicted_vol)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(
                actual_volatilities, predicted_volatilities
            )
            
            backtest_results = {
                'actual_volatilities': actual_volatilities,
                'predicted_volatilities': predicted_volatilities,
                'metrics': metrics,
                'window_size': window,
                'horizon': horizon
            }
            
            self.backtest_results[f"backtest_{len(self.backtest_results)}"] = backtest_results
            
            return backtest_results
            
        except Exception as e:
            print(f"❌ Error in backtesting: {str(e)}")
            raise
    
    def _calculate_performance_metrics(self, actual: List[float], predicted: List[float]) -> Dict[str, float]:
        """
        📊 Calculates how good your forecasts really are (the performance tea)
        
        Args:
            actual (List): Actual values
            predicted (List): Predicted values
            
        Returns:
            Dict: Performance metrics
        """
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {'error': 'No valid data for metrics calculation'}
        
        # Calculate metrics
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Directional accuracy
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared,
            'directional_accuracy': directional_accuracy,
            'mean_actual': np.mean(actual),
            'mean_predicted': np.mean(predicted),
            'correlation': np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else 0
        }
        
        return metrics
    
    def generate_rolling_forecast(self, returns: pd.Series, model_results: Dict[str, Any],
                                 window: int = 252, horizon: int = 5) -> pd.DataFrame:
        """
        🔄 Generates rolling forecasts (the continuous crystal ball)
        
        Args:
            returns (pd.Series): Historical returns
            model_results (Dict): Model results
            window (int): Rolling window size
            horizon (int): Forecast horizon
            
        Returns:
            pd.DataFrame: Rolling forecasts with dates
        """
        forecasts = []
        
        for i in range(window, len(returns) - horizon):
            # Get training data
            train_returns = returns.iloc[i-window:i]
            
            # Generate forecast
            if 'model_object' in model_results:
                model = model_results['model_object']
                new_model = type(model)()
                new_model.fit(train_returns)
                forecast = new_model.forecast(horizon=horizon)
                predicted_vol = forecast['volatility_forecast']
            else:
                predicted_vol = train_returns.std() * np.sqrt(252)
            
            # Store forecast
            forecast_date = returns.index[i]
            forecasts.append({
                'date': forecast_date,
                'predicted_volatility': predicted_vol,
                'actual_volatility': returns.iloc[i:i+horizon].std() * np.sqrt(252)
            })
        
        return pd.DataFrame(forecasts)
    
    def calculate_value_at_risk(self, returns: pd.Series, volatility: float, 
                               confidence_level: float = 0.95, horizon: int = 1) -> Dict[str, float]:
        """
        💰 Calculates Value at Risk (the risk management swag)
        
        Args:
            returns (pd.Series): Historical returns
            volatility (float): Forecasted volatility
            confidence_level (float): VaR confidence level
            horizon (int): Time horizon in days
            
        Returns:
            Dict: VaR calculations
        """
        # Calculate VaR using normal distribution assumption
        z_score = norm.ppf(1 - confidence_level)
        
        # Annualized volatility to daily
        daily_vol = volatility / np.sqrt(252)
        
        # VaR calculation
        var = z_score * daily_vol * np.sqrt(horizon)
        
        # Expected shortfall (conditional VaR)
        es = norm.pdf(z_score) / (1 - confidence_level) * daily_vol * np.sqrt(horizon)
        
        return {
            'var': abs(var),
            'expected_shortfall': abs(es),
            'confidence_level': confidence_level,
            'horizon': horizon,
            'volatility': volatility
        }
    
    def generate_forecast_report(self, forecast: Dict[str, Any], 
                                backtest_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        📋 Generates a comprehensive forecast report (all the deets)
        
        Args:
            forecast (Dict): Forecast results
            backtest_results (Dict): Backtesting results
            
        Returns:
            Dict: Complete forecast report
        """
        report = {
            'forecast_summary': {
                'horizon': forecast['horizon'],
                'volatility_forecast': forecast['volatility_forecast'],
                'mean_forecast': forecast['mean_forecast'],
                'method': forecast.get('method', 'unknown')
            },
            'confidence_intervals': forecast.get('confidence_intervals', {}),
            'forecast_dates': forecast['forecast_dates'].tolist() if isinstance(forecast['forecast_dates'], pd.DatetimeIndex) else forecast['forecast_dates']
        }
        
        # Add backtesting results if available
        if backtest_results:
            report['backtest_metrics'] = backtest_results['metrics']
            report['backtest_summary'] = {
                'window_size': backtest_results['window_size'],
                'horizon': backtest_results['horizon'],
                'n_forecasts': len(backtest_results['actual_volatilities'])
            }
        
        return report
    
    def create_forecast_summary(self, forecasts: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        📊 Creates a summary table of all forecasts (the comparison chart)
        
        Args:
            forecasts (List): List of forecast dictionaries
            
        Returns:
            pd.DataFrame: Summary table
        """
        summary_data = []
        
        for i, forecast in enumerate(forecasts):
            summary_data.append({
                'forecast_id': i,
                'horizon': forecast['horizon'],
                'volatility_forecast': forecast['volatility_forecast'],
                'mean_forecast': forecast['mean_forecast'],
                'method': forecast.get('method', 'unknown'),
                'has_confidence_intervals': 'confidence_intervals' in forecast
            })
        
        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Example usage
    from data_loader import StockDataLoader
    from garch_models import GARCHModel
    
    # Load data
    loader = StockDataLoader()
    data = loader.fetch_data("AAPL", "2y")
    returns = loader.get_returns()
    
    # Fit model
    garch_model = GARCHModel()
    diagnostics = garch_model.fit(returns)
    
    # Create forecaster
    forecaster = VolatilityForecaster()
    
    # Generate forecast
    forecast = forecaster.generate_forecast(diagnostics, horizon=10)
    print(f"Forecast volatility: {forecast['volatility_forecast']:.4f}")
    
    # Backtest
    backtest = forecaster.backtest_model(returns, diagnostics)
    
    # Generate report
    report = forecaster.generate_forecast_report(forecast, backtest)
    print(report) 