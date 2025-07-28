#!/usr/bin/env python3
"""
🚀 Stock Volatility Forecasting Example - Shows you how to use all the cool features
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import StockDataLoader
from src.garch_models import GARCHModel, EGARCHModel, GJR_GARCHModel, GARCHModelComparison
from src.forecasting import VolatilityForecaster
from src.visualization import VolatilityVisualizer
from src.utils import *


def main():
    """
    🎯 Main example function - shows off all the sick features
    """
    print_banner("Stock Volatility Forecasting Example")
    
    try:
        # Load some data (AAPL is always a good choice)
        print_section("Data Loading")
        loader = StockDataLoader()
        data, returns, volatility = loader.get_data("AAPL", "2y")
        
        # Analyze the data
        print_section("Data Analysis")
        stats = loader.get_summary_stats(returns)
        stationarity = loader.check_stationarity(returns)
        clustering = loader.detect_volatility_clustering(returns)
        
        # Fit a single GARCH model
        print_section("Single Model Fitting")
        garch_model = GARCHModel()
        diagnostics = garch_model.fit(returns)
        
        # Add model object to diagnostics for forecasting
        diagnostics['model_object'] = garch_model
        
        # Generate forecast
        print_section("Forecasting")
        forecaster = VolatilityForecaster()
        forecast = forecaster.generate_forecast(diagnostics, horizon=5)
        
        print_success(f"🔮 Forecast volatility: {forecast['volatility_forecast']:.4f}")
        
        # Compare multiple models (the model battle royale)
        print_section("Model Comparison")
        comparison = GARCHModelComparison()
        comparison_results = comparison.fit_models(returns)
        
        # Get the best model
        best_model_name, best_model = comparison.get_best_model('aic')
        print_success(f"🏆 Best model by AIC: {best_model_name}")
        
        # Backtesting (see how good your model really is)
        print_section("Backtesting")
        backtest_results = forecaster.backtest_model(returns, diagnostics)
        
        if 'metrics' in backtest_results:
            metrics = backtest_results['metrics']
            print_info(f"📊 RMSE: {metrics.get('rmse', 'N/A'):.6f}")
            print_info(f"📊 MAE: {metrics.get('mae', 'N/A'):.6f}")
            print_info(f"📊 R²: {metrics.get('r_squared', 'N/A'):.4f}")
        
        # Create some sick visualizations
        print_section("Visualization")
        visualizer = VolatilityVisualizer()
        
        # Create all the plots
        plots = {}
        
        # Returns and volatility plot
        plots['returns_volatility'] = visualizer.plot_returns_and_volatility(
            returns, volatility, "AAPL - Returns & Volatility"
        )
        
        # Volatility clustering plot
        plots['clustering'] = visualizer.plot_volatility_clustering(returns)
        
        # GARCH diagnostics plot
        plots['diagnostics'] = visualizer.plot_garch_diagnostics(diagnostics)
        
        # Forecast plot
        plots['forecast'] = visualizer.plot_forecast(forecast, volatility)
        
        # Model comparison plot
        plots['comparison'] = visualizer.plot_model_comparison(comparison_results)
        
        # Dashboard (the ultimate overview)
        plots['dashboard'] = visualizer.create_summary_dashboard(
            data, returns, volatility, diagnostics, forecast
        )
        
        # Save all the plots
        print_info("Saving plots...")
        timestamp = generate_timestamp()
        
        for plot_name, fig in plots.items():
            filename = f"reports/AAPL_{plot_name}_{timestamp}.html"
            visualizer.save_plot(fig, filename, 'html')
        
        # Generate reports
        print_section("Report Generation")
        
        # Save results as JSON
        results_file = f"reports/AAPL_results_{timestamp}.json"
        save_json(diagnostics, results_file)
        
        forecast_file = f"reports/AAPL_forecast_{timestamp}.json"
        save_json(forecast, forecast_file)
        
        # Generate HTML report
        report_content = f"""
        <h2>📈 Summary Statistics</h2>
        <div class="metric">
            <strong>Data Points:</strong> {len(data)}<br>
            <strong>Date Range:</strong> {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}<br>
            <strong>Mean Return:</strong> {stats['mean_return']:.4f}<br>
            <strong>Return Std:</strong> {stats['std_return']:.4f}<br>
            <strong>Skewness:</strong> {stats['skewness']:.4f}<br>
            <strong>Kurtosis:</strong> {stats['kurtosis']:.4f}
        </div>
        
        <h2>🔧 Model Results</h2>
        <div class="metric">
            <strong>Model Type:</strong> {diagnostics['model_type']}<br>
            <strong>AIC:</strong> {diagnostics['aic']:.4f}<br>
            <strong>BIC:</strong> {diagnostics['bic']:.4f}<br>
            <strong>Log-Likelihood:</strong> {diagnostics['log_likelihood']:.4f}
        </div>
        
        <h2>🔮 Forecast Results</h2>
        <div class="metric">
            <strong>Forecast Horizon:</strong> {forecast['horizon']} days<br>
            <strong>Forecast Volatility:</strong> {forecast['volatility_forecast']:.4f}<br>
            <strong>Method:</strong> {forecast.get('method', 'unknown')}
        </div>
        """
        
        if 'metrics' in backtest_results:
            metrics = backtest_results['metrics']
            report_content += f"""
            <h2>🧪 Backtesting Results</h2>
            <div class="metric">
                <strong>RMSE:</strong> {metrics.get('rmse', 'N/A'):.6f}<br>
                <strong>MAE:</strong> {metrics.get('mae', 'N/A'):.6f}<br>
                <strong>R²:</strong> {metrics.get('r_squared', 'N/A'):.4f}<br>
                <strong>Directional Accuracy:</strong> {metrics.get('directional_accuracy', 'N/A'):.4f}
            </div>
            """
        
        # Save HTML report
        report_file = f"reports/AAPL_report_{timestamp}.html"
        generate_html_report(
            "Volatility Forecast Report - AAPL",
            report_content,
            report_file
        )
        
        # Final summary
        print_section("Summary")
        print_success("✅ Example completed successfully!")
        print_info("📁 Check the 'reports/' directory for generated files")
        print_info("🎮 Run 'streamlit run dashboard.py' for interactive analysis")
        print_info("💻 Run 'python main.py --help' for CLI options")
        
        return 0
        
    except Exception as e:
        print_error(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 