"""
📊 Visualization Module - Makes your data look absolutely fire
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class VolatilityVisualizer:
    """
    🎨 The ultimate data visualizer - turns boring numbers into stunning charts
    """
    
    def __init__(self):
        self.plots = {}
    
    def plot_returns_and_volatility(self, returns: pd.Series, volatility: pd.Series, 
                                   title: str = "Returns & Volatility") -> go.Figure:
        """
        📈 Creates a sick chart showing returns and volatility over time
        
        Args:
            returns (pd.Series): Your return series
            volatility (pd.Series): Your volatility series
            title (str): Chart title
            
        Returns:
            go.Figure: A beautiful plotly figure
        """
        # Create subplots (returns on top, volatility on bottom)
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Returns', 'Volatility'),
            vertical_spacing=0.1
        )
        
        # Add returns trace
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns.values,
                mode='lines',
                name='Returns',
                line=dict(color='#1f77b4', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add volatility trace
        fig.add_trace(
            go.Scatter(
                x=volatility.index,
                y=volatility.values,
                mode='lines',
                name='Volatility',
                line=dict(color='#ff7f0e', width=2),
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.1)'
            ),
            row=2, col=1
        )
        
        # Update layout for maximum swag
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Returns", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (Annualized)", row=2, col=1)
        
        return fig
    
    def plot_volatility_clustering(self, returns: pd.Series, window: int = 30) -> go.Figure:
        """
        🔥 Shows volatility clustering (when things get spicy)
        
        Args:
            returns (pd.Series): Return series
            window (int): Rolling window size
            
        Returns:
            go.Figure: Volatility clustering visualization
        """
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Create the plot
        fig = go.Figure()
        
        # Add volatility trace
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name=f'{window}-Day Rolling Volatility',
                line=dict(color='#d62728', width=2),
                fill='tonexty',
                fillcolor='rgba(214, 39, 40, 0.1)'
            )
        )
        
        # Add mean line
        mean_vol = rolling_vol.mean()
        fig.add_hline(
            y=mean_vol,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Mean: {mean_vol:.4f}"
        )
        
        # Update layout
        fig.update_layout(
            title="🔥 Volatility Clustering Analysis",
            xaxis_title="Date",
            yaxis_title="Volatility (Annualized)",
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_garch_diagnostics(self, model_results: Dict[str, Any]) -> go.Figure:
        """
        🔍 Shows GARCH model diagnostics (the model health check)
        
        Args:
            model_results (Dict): GARCH model results
            
        Returns:
            go.Figure: Diagnostic plots
        """
        # Create subplots for different diagnostics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals', 'Squared Residuals', 'Q-Q Plot', 'ACF'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        residuals = model_results['residuals']
        
        # Residuals plot
        fig.add_trace(
            go.Scatter(
                x=residuals.index,
                y=residuals.values,
                mode='lines',
                name='Residuals',
                line=dict(color='#1f77b4', width=1)
            ),
            row=1, col=1
        )
        
        # Squared residuals plot
        squared_residuals = residuals ** 2
        fig.add_trace(
            go.Scatter(
                x=squared_residuals.index,
                y=squared_residuals.values,
                mode='lines',
                name='Squared Residuals',
                line=dict(color='#ff7f0e', width=1)
            ),
            row=1, col=2
        )
        
        # Q-Q plot (simplified)
        sorted_residuals = np.sort(residuals.dropna())
        theoretical_quantiles = np.quantile(np.random.normal(0, 1, len(sorted_residuals)), 
                                          np.linspace(0, 1, len(sorted_residuals)))
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_residuals,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color='#2ca02c', size=4)
            ),
            row=2, col=1
        )
        
        # Add diagonal line for Q-Q plot
        min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
        max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Fit',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # ACF plot (simplified)
        from statsmodels.tsa.stattools import acf
        acf_values = acf(residuals.dropna(), nlags=20)
        
        fig.add_trace(
            go.Bar(
                x=list(range(len(acf_values))),
                y=acf_values,
                name='ACF',
                marker_color='#9467bd'
            ),
            row=2, col=2
        )
        
        # Add confidence bands for ACF
        confidence_interval = 1.96 / np.sqrt(len(residuals.dropna()))
        fig.add_hline(y=confidence_interval, line_dash="dash", line_color="red", 
                     annotation_text="95% CI", row=2, col=2)
        fig.add_hline(y=-confidence_interval, line_dash="dash", line_color="red", 
                     annotation_text="", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title="🔍 GARCH Model Diagnostics",
            height=600,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def plot_forecast(self, forecast: Dict[str, Any], 
                     historical_volatility: pd.Series = None) -> go.Figure:
        """
        🔮 Shows your volatility forecast (the crystal ball visualization)
        
        Args:
            forecast (Dict): Forecast results
            historical_volatility (pd.Series): Historical volatility for context
            
        Returns:
            go.Figure: Forecast visualization
        """
        fig = go.Figure()
        
        # Add historical volatility if provided
        if historical_volatility is not None:
            fig.add_trace(
                go.Scatter(
                    x=historical_volatility.index,
                    y=historical_volatility.values,
                    mode='lines',
                    name='Historical Volatility',
                    line=dict(color='#1f77b4', width=2),
                    opacity=0.7
                )
            )
        
        # Add forecast
        forecast_dates = forecast['forecast_dates']
        forecast_vol = forecast['volatility_forecast']
        
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=[forecast_vol] * len(forecast_dates),
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=8)
            )
        )
        
        # Add confidence intervals if available
        if 'confidence_intervals' in forecast:
            ci = forecast['confidence_intervals']
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=[ci['upper']] * len(forecast_dates),
                    mode='lines',
                    name=f"{int(ci['confidence_level']*100)}% Upper CI",
                    line=dict(color='rgba(255, 127, 14, 0.3)', width=1),
                    showlegend=False
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=[ci['lower']] * len(forecast_dates),
                    mode='lines',
                    name=f"{int(ci['confidence_level']*100)}% Lower CI",
                    line=dict(color='rgba(255, 127, 14, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255, 127, 14, 0.1)',
                    showlegend=False
                )
            )
        
        # Update layout
        fig.update_layout(
            title="🔮 Volatility Forecast",
            xaxis_title="Date",
            yaxis_title="Volatility (Annualized)",
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def plot_model_comparison(self, comparison_results: Dict[str, Any]) -> go.Figure:
        """
        🏆 Compares different GARCH models (the model battle royale)
        
        Args:
            comparison_results (Dict): Model comparison results
            
        Returns:
            go.Figure: Model comparison visualization
        """
        # Extract model names and metrics
        models = list(comparison_results.keys())
        aic_values = [comparison_results[model]['aic'] for model in models]
        bic_values = [comparison_results[model]['bic'] for model in models]
        ll_values = [comparison_results[model]['log_likelihood'] for model in models]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('AIC (Lower is Better)', 'BIC (Lower is Better)', 'Log-Likelihood (Higher is Better)')
        )
        
        # AIC comparison
        fig.add_trace(
            go.Bar(
                x=models,
                y=aic_values,
                name='AIC',
                marker_color='#1f77b4'
            ),
            row=1, col=1
        )
        
        # BIC comparison
        fig.add_trace(
            go.Bar(
                x=models,
                y=bic_values,
                name='BIC',
                marker_color='#ff7f0e'
            ),
            row=1, col=2
        )
        
        # Log-likelihood comparison
        fig.add_trace(
            go.Bar(
                x=models,
                y=ll_values,
                name='Log-Likelihood',
                marker_color='#2ca02c'
            ),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            title="🏆 Model Comparison",
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_summary_dashboard(self, data: pd.DataFrame, returns: pd.Series, 
                               volatility: pd.Series, model_results: Dict[str, Any],
                               forecast: Dict[str, Any] = None) -> go.Figure:
        """
        📊 Creates a comprehensive dashboard (the ultimate data overview)
        
        Args:
            data (pd.DataFrame): Stock data
            returns (pd.Series): Return series
            volatility (pd.Series): Volatility series
            model_results (Dict): GARCH model results
            forecast (Dict): Forecast results
            
        Returns:
            go.Figure: Comprehensive dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Stock Price', 'Returns', 'Volatility', 'Model Diagnostics', 'Forecast', 'Summary Stats'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
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
                y=returns.values,
                mode='lines',
                name='Returns',
                line=dict(color='#ff7f0e', width=1)
            ),
            row=1, col=2
        )
        
        # Volatility
        fig.add_trace(
            go.Scatter(
                x=volatility.index,
                y=volatility.values,
                mode='lines',
                name='Volatility',
                line=dict(color='#2ca02c', width=2),
                fill='tonexty',
                fillcolor='rgba(44, 160, 44, 0.1)'
            ),
            row=2, col=1
        )
        
        # Model residuals
        residuals = model_results['residuals']
        fig.add_trace(
            go.Scatter(
                x=residuals.index,
                y=residuals.values,
                mode='lines',
                name='Residuals',
                line=dict(color='#d62728', width=1)
            ),
            row=2, col=2
        )
        
        # Forecast (if available)
        if forecast:
            forecast_dates = forecast['forecast_dates']
            forecast_vol = forecast['volatility_forecast']
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=[forecast_vol] * len(forecast_dates),
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#9467bd', width=3),
                    marker=dict(size=8)
                ),
                row=3, col=1
            )
        
        # Summary statistics table
        stats_text = f"""
        <b>Model Summary</b><br>
        AIC: {model_results['aic']:.2f}<br>
        BIC: {model_results['bic']:.2f}<br>
        Log-Likelihood: {model_results['log_likelihood']:.2f}<br>
        Mean Return: {returns.mean():.4f}<br>
        Volatility: {volatility.mean():.4f}
        """
        
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode='text',
                text=[stats_text],
                textposition='middle center',
                showlegend=False
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="📊 Volatility Analysis Dashboard",
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def save_plot(self, fig: go.Figure, filename: str, format: str = 'html') -> None:
        """
        💾 Saves your plot to a file (preserve the beauty)
        
        Args:
            fig (go.Figure): Plotly figure
            filename (str): Output filename
            format (str): Output format ('html', 'png', 'pdf')
        """
        if format == 'html':
            fig.write_html(filename)
        elif format == 'png':
            fig.write_image(filename)
        elif format == 'pdf':
            fig.write_image(filename)
        else:
            raise ValueError("😵‍💫 Unsupported format. Use 'html', 'png', or 'pdf'")
        
        print(f"✅ Plot saved as {filename}")


def save_plot(fig: go.Figure, filename: str, format: str = 'html') -> None:
    """
    💾 Utility function to save plots (the quick save)
    
    Args:
        fig (go.Figure): Plotly figure
        filename (str): Output filename
        format (str): Output format
    """
    visualizer = VolatilityVisualizer()
    visualizer.save_plot(fig, filename, format) 