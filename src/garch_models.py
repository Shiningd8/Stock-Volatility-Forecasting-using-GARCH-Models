"""
🔥 GARCH Models - The real MVPs of volatility forecasting
"""

import pandas as pd
import numpy as np
from arch import arch_model
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class GARCHModel:
    """
    🚀 The OG GARCH model - captures volatility like a boss
    """
    
    def __init__(self, model_type: str = "GARCH"):
        """
        Initialize the GARCH model with swag
        
        Args:
            model_type (str): Type of GARCH model (GARCH, EGARCH, GJR-GARCH)
        """
        self.model_type = model_type
        self.model = None
        self.results = None
        self.fitted = False
    
    def fit(self, returns: pd.Series, p: int = 1, q: int = 1, 
            mean: str = 'Zero', vol: str = 'GARCH', dist: str = 'normal') -> Dict[str, Any]:
        """
        🔥 Fits the GARCH model to your data (the magic happens here)
        
        Args:
            returns (pd.Series): Your return series (the spicy data)
            p (int): GARCH lag order (how many past variances to consider)
            q (int): ARCH lag order (how many past shocks to consider)
            mean (str): Mean model ('Zero', 'ARX', 'Constant')
            vol (str): Volatility model ('GARCH', 'EGARCH', 'GJR-GARCH')
            dist (str): Error distribution ('normal', 't', 'skewt')
            
        Returns:
            Dict: All the juicy model diagnostics
        """
        try:
            # Create the model with all the right vibes
            self.model = arch_model(
                returns * 100,  # Convert to percentage (ARCH library likes it this way)
                mean=mean,
                vol=vol,
                p=p,
                q=q,
                dist=dist
            )
            
            # Fit that bad boy
            self.results = self.model.fit(disp='off', show_warning=False)
            self.fitted = True
            
            # Extract the tea (diagnostics)
            diagnostics = self._extract_diagnostics()
            
            print(f"✅ Successfully fitted {self.model_type} model")
            print(f"📊 AIC: {diagnostics['aic']:.4f}")
            print(f"📊 BIC: {diagnostics['bic']:.4f}")
            print(f"📊 Log-Likelihood: {diagnostics['log_likelihood']:.4f}")
            
            return diagnostics
            
        except Exception as e:
            print(f"❌ Oops! Failed to fit {self.model_type} model: {str(e)}")
            raise
    
    def _extract_diagnostics(self) -> Dict[str, Any]:
        """
        📊 Extracts all the model diagnostics (the real tea)
        
        Returns:
            Dict: Model diagnostics that tell you if it's good or not
        """
        if not self.fitted:
            raise ValueError("😅 Model not fitted yet! Call fit() first")
        
        # Get the basic stats
        diagnostics = {
            'aic': self.results.aic,
            'bic': self.results.bic,
            'log_likelihood': self.results.loglikelihood,
            'model_type': self.model_type,
            'fitted': True,
            'params': self.results.params.to_dict(),
            'conditional_volatility': self.results.conditional_volatility,
            'residuals': self.results.resid,
            'model_object': self  # Keep the whole model for forecasting
        }
        
        return diagnostics
    
    def forecast(self, horizon: int = 5, method: str = 'analytic') -> Dict[str, Any]:
        """
        🔮 Forecasts volatility into the future (crystal ball stuff)
        
        Args:
            horizon (int): How many days ahead to predict
            method (str): Forecast method ('analytic', 'simulation')
            
        Returns:
            Dict: Your volatility forecast with all the deets
        """
        if not self.fitted:
            raise ValueError("😅 Model must be fitted before forecasting.")
        
        try:
            # Generate forecast
            forecast = self.results.forecast(horizon=horizon, method=method)
            
            # Extract forecast components - handle different forecast structures
            try:
                # Try the standard structure with h.01, h.02, etc.
                mean_forecast = forecast.mean['h.01'].iloc[-1] / 100  # Convert from percentage
                variance_forecast = forecast.variance['h.01'].iloc[-1] / 10000  # Convert from percentage squared
            except (KeyError, AttributeError):
                # Fallback to alternative structure
                try:
                    mean_forecast = forecast.mean.iloc[-1] / 100
                    variance_forecast = forecast.variance.iloc[-1] / 10000
                except (KeyError, AttributeError):
                    # Use conditional volatility as fallback
                    mean_forecast = 0.0
                    variance_forecast = (self.results.conditional_volatility.iloc[-1] / 100) ** 2
            
            volatility_forecast = np.sqrt(variance_forecast * 252)  # Annualized volatility
            
            # Get the last date from the model's index
            try:
                last_date = self.results.data.index[-1]
            except AttributeError:
                # Fallback: use current date
                last_date = pd.Timestamp.now()
            
            # Confidence intervals (simplified)
            forecast_results = {
                'horizon': horizon,
                'mean_forecast': mean_forecast,
                'variance_forecast': variance_forecast,
                'volatility_forecast': volatility_forecast,
                'forecast_dates': pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=horizon,
                    freq='D'
                ),
                'method': method
            }
            
            return forecast_results
            
        except Exception as e:
            print(f"❌ Error in forecasting: {str(e)}")
            raise
    
    def get_model_summary(self) -> str:
        """Get model summary as string."""
        if not self.fitted:
            return "Model not fitted yet."
        
        return str(self.results.summary())


class EGARCHModel(GARCHModel):
    """
    🔥 Exponential GARCH model - handles asymmetric volatility like a pro
    """
    
    def __init__(self):
        super().__init__("EGARCH")
    
    def fit(self, returns: pd.Series, p: int = 1, q: int = 1, 
            mean: str = 'Zero', vol: str = 'EGARCH', dist: str = 'normal') -> Dict[str, Any]:
        """Fit EGARCH model with asymmetric effects."""
        return super().fit(returns, p, q, mean, vol, dist)


class GJR_GARCHModel(GARCHModel):
    """
    🚀 Glosten-Jagannathan-Runkle GARCH model - the leverage effect specialist
    """
    
    def __init__(self):
        super().__init__("GJR-GARCH")
    
    def fit(self, returns: pd.Series, p: int = 1, q: int = 1, 
            mean: str = 'Zero', vol: str = 'GARCH', dist: str = 'normal') -> Dict[str, Any]:
        """Fit GJR-GARCH model with leverage effects."""
        return super().fit(returns, p, q, mean, vol, dist)


class GARCHModelComparison:
    """
    🏆 Model comparison - finds the best GARCH model for your data
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def fit_models(self, returns: pd.Series, models: list = None) -> Dict[str, Any]:
        """
        🔥 Fits multiple GARCH models and compares them (model battle royale)
        
        Args:
            returns (pd.Series): Your return series
            models (list): List of model types to fit (default: all three)
            
        Returns:
            Dict: Comparison results with rankings
        """
        if models is None:
            models = ['GARCH', 'EGARCH', 'GJR-GARCH']
        
        comparison_results = {}
        
        for model_type in models:
            try:
                # Create and fit the model
                if model_type == 'GARCH':
                    model = GARCHModel()
                elif model_type == 'EGARCH':
                    model = EGARCHModel()
                elif model_type == 'GJR-GARCH':
                    model = GJR_GARCHModel()
                else:
                    print(f"⚠️ Unknown model type: {model_type}")
                    continue
                
                # Fit that model
                diagnostics = model.fit(returns)
                
                # Store results
                self.models[model_type] = model
                comparison_results[model_type] = diagnostics
                
            except Exception as e:
                print(f"❌ Failed to fit {model_type}: {str(e)}")
                continue
        
        self.results = comparison_results
        return comparison_results
    
    def get_best_model(self, criterion: str = 'aic') -> Tuple[str, GARCHModel]:
        """
        🏆 Finds the best model based on your favorite criterion
        
        Args:
            criterion (str): 'aic', 'bic', or 'log_likelihood'
            
        Returns:
            Tuple: (best_model_name, best_model_object)
        """
        if not self.results:
            raise ValueError("😅 No models fitted yet! Call fit_models() first")
        
        # Find the best model (lower is better for AIC/BIC, higher for log-likelihood)
        if criterion in ['aic', 'bic']:
            best_model = min(self.results.keys(), 
                           key=lambda x: self.results[x][criterion])
        else:
            best_model = max(self.results.keys(), 
                           key=lambda x: self.results[x][criterion])
        
        return best_model, self.models[best_model]
    
    def forecast_all_models(self, horizon: int = 5) -> Dict[str, Any]:
        """
        🔮 Forecasts with all fitted models (the full crystal ball experience)
        
        Args:
            horizon (int): Forecast horizon
            
        Returns:
            Dict: Forecasts from all models
        """
        if not self.results:
            raise ValueError("😅 No models fitted yet! Call fit_models() first")
        
        forecasts = {}
        
        for model_name, model in self.models.items():
            try:
                forecast = model.forecast(horizon=horizon)
                forecasts[model_name] = forecast
            except Exception as e:
                print(f"❌ Failed to forecast with {model_name}: {str(e)}")
                continue
        
        return forecasts


def create_garch_model(model_type: str = "GARCH") -> GARCHModel:
    """
    🏭 Factory function to create GARCH models (the model maker)
    
    Args:
        model_type (str): Type of model to create
        
    Returns:
        GARCHModel: Your new model ready to rock
    """
    if model_type.upper() == "GARCH":
        return GARCHModel()
    elif model_type.upper() == "EGARCH":
        return EGARCHModel()
    elif model_type.upper() == "GJR-GARCH":
        return GJR_GARCHModel()
    else:
        raise ValueError(f"😵‍💫 Unknown model type: {model_type}")


# Test the models if run directly
if __name__ == "__main__":
    from data_loader import StockDataLoader
    
    # Load some data
    loader = StockDataLoader()
    data, returns, volatility = loader.get_data("AAPL", "1y")
    
    # Test model fitting
    garch_model = GARCHModel()
    diagnostics = garch_model.fit(returns)
    
    # Test forecasting
    forecast = garch_model.forecast(horizon=5)
    print(f"🔮 Forecast volatility: {forecast['volatility_forecast']:.4f}") 