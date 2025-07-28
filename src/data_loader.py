"""
🔥 Stock Data Loader - Grabs all the juicy stock data you need
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class StockDataLoader:
    """
    🚀 The ultimate stock data grabber - gets you all the deets on any ticker
    """
    
    def __init__(self):
        self.data = None
        self.returns = None
        self.volatility = None
    
    def fetch_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """
        🔥 Snags fresh stock data from Yahoo Finance
        
        Args:
            ticker (str): The stock ticker (like AAPL, TSLA, etc.)
            period (str): How far back to go (1y, 2y, 5y, max, etc.)
            
        Returns:
            pd.DataFrame: All the stock data you could ever want
        """
        try:
            # Grab that data like a pro
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                raise ValueError(f"😵‍💫 No data found for {ticker}")
            
            # Clean it up real nice
            data = data.dropna()
            self.data = data
            
            print(f"✅ Successfully loaded {len(data)} days of data for {ticker}")
            print(f"📊 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            
            return data
            
        except Exception as e:
            print(f"❌ Oops! Failed to grab data for {ticker}: {str(e)}")
            raise
    
    def get_returns(self, data: pd.DataFrame = None) -> pd.Series:
        """
        📈 Calculates those sweet returns (percentage changes)
        
        Args:
            data (pd.DataFrame): Stock data (uses self.data if not provided)
            
        Returns:
            pd.Series: The returns that make or break your portfolio
        """
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("😅 No data to work with! Call fetch_data() first")
        
        # Calculate returns like a boss
        returns = data['Close'].pct_change().dropna()
        self.returns = returns
        
        return returns
    
    def get_volatility(self, data: pd.DataFrame = None, window: int = 30) -> pd.Series:
        """
        📊 Calculates rolling volatility (the spicy stuff)
        
        Args:
            data (pd.DataFrame): Stock data
            window (int): How many days to look back (default: 30)
            
        Returns:
            pd.Series: Rolling volatility that shows the market's mood swings
        """
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("😅 No data to work with! Call fetch_data() first")
        
        # Get those returns first
        returns = self.get_returns(data)
        
        # Calculate rolling volatility (the real tea)
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        self.volatility = volatility
        
        avg_vol = volatility.mean()
        print(f"📉 Average volatility: {avg_vol:.4f}")
        
        return volatility
    
    def get_data(self, ticker: str, period: str = "2y") -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        🎯 One-stop shop: grab data, calculate returns, and get volatility
        
        Args:
            ticker (str): Stock ticker
            period (str): Time period
            
        Returns:
            Tuple: (data, returns, volatility) - everything you need
        """
        # Grab the goods
        data = self.fetch_data(ticker, period)
        returns = self.get_returns(data)
        volatility = self.get_volatility(data)
        
        print(f"✅ Loaded {len(data)} days of data")
        print(f"📊 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        return data, returns, volatility
    
    def get_summary_stats(self, returns: pd.Series = None) -> Dict[str, float]:
        """
        📊 Gives you the tea on your returns (stats that matter)
        
        Args:
            returns (pd.Series): Return series
            
        Returns:
            Dict: All the juicy stats you need to know
        """
        if returns is None:
            returns = self.returns
        
        if returns is None:
            raise ValueError("😅 No returns to analyze! Call get_returns() first")
        
        # Calculate the stats that actually matter
        stats = {
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min_return': returns.min(),
            'max_return': returns.max(),
            'var_95': returns.quantile(0.05),  # 95% VaR
            'var_99': returns.quantile(0.01),  # 99% VaR
        }
        
        print(f"📈 Mean return: {stats['mean_return']:.4f}")
        print(f"📉 Return std: {stats['std_return']:.4f}")
        print(f"📊 Skewness: {stats['skewness']:.4f}")
        print(f"📊 Kurtosis: {stats['kurtosis']:.4f}")
        
        return stats
    
    def check_stationarity(self, returns: pd.Series = None) -> Dict[str, any]:
        """
        🔍 Checks if your data is stationary (no trends, just vibes)
        
        Args:
            returns (pd.Series): Return series
            
        Returns:
            Dict: Stationarity test results
        """
        if returns is None:
            returns = self.returns
        
        if returns is None:
            raise ValueError("😅 No returns to test! Call get_returns() first")
        
        from statsmodels.tsa.stattools import adfuller
        
        # Run the ADF test (the real deal)
        adf_result = adfuller(returns.dropna())
        
        results = {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }
        
        if results['is_stationary']:
            print("✅ Data is stationary (no trends, just pure chaos)")
        else:
            print("⚠️ Data might have trends (not ideal for GARCH)")
        
        return results
    
    def detect_volatility_clustering(self, returns: pd.Series = None) -> Dict[str, any]:
        """
        🔥 Detects volatility clustering (when things get spicy)
        
        Args:
            returns (pd.Series): Return series
            
        Returns:
            Dict: Volatility clustering analysis
        """
        if returns is None:
            returns = self.returns
        
        if returns is None:
            raise ValueError("😅 No returns to analyze! Call get_returns() first")
        
        # Calculate squared returns (the volatility proxy)
        squared_returns = returns ** 2
        
        # Check for autocorrelation in squared returns
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        # Ljung-Box test for autocorrelation
        lb_test = acorr_ljungbox(squared_returns.dropna(), lags=10, return_df=True)
        
        results = {
            'ljung_box_stat': lb_test['lb_stat'].iloc[-1],
            'ljung_box_pvalue': lb_test['lb_pvalue'].iloc[-1],
            'has_clustering': lb_test['lb_pvalue'].iloc[-1] < 0.05
        }
        
        if results['has_clustering']:
            print("🔥 Volatility clustering detected (perfect for GARCH!)")
        else:
            print("😐 No significant volatility clustering found")
        
        return results


def load_multiple_tickers(tickers: List[str], period: str = "2y") -> Dict[str, Tuple]:
    """
    🚀 Load data for multiple tickers at once (because why not?)
    
    Args:
        tickers (List[str]): List of stock tickers
        period (str): Time period
        
    Returns:
        Dict: Data for all tickers
    """
    loader = StockDataLoader()
    results = {}
    
    for ticker in tickers:
        try:
            print(f"📥 Loading stock data for {ticker}...")
            data, returns, volatility = loader.get_data(ticker, period)
            results[ticker] = (data, returns, volatility)
        except Exception as e:
            print(f"❌ Failed to load {ticker}: {str(e)}")
            continue
    
    return results


if __name__ == "__main__":
    # Example usage
    loader = StockDataLoader()
    data = loader.fetch_data("AAPL", "2y")
    
    print("\n📊 Summary Statistics:")
    stats = loader.get_summary_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n🔍 Stationarity Tests:")
    stationarity = loader.check_stationarity()
    for key, value in stationarity.items():
        print(f"{key}: {value}")
    
    print("\n📈 Volatility Clustering Analysis:")
    clustering = loader.detect_volatility_clustering()
    for key, value in clustering.items():
        print(f"{key}: {value}") 