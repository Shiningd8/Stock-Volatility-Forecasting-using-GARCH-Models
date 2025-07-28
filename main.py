#!/usr/bin/env python3
"""
🚀 Stock Volatility Forecasting CLI - The command line interface for all your volatility needs
"""

import argparse
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import StockDataLoader
from src.garch_models import GARCHModel, EGARCHModel, GJR_GARCHModel, GARCHModelComparison
from src.forecasting import VolatilityForecaster
from src.visualization import VolatilityVisualizer
from src.utils import (
    validate_ticker, validate_period, validate_model_type,
    create_directories, save_json, generate_html_report, load_config,
    print_banner, print_section, print_success, print_error, print_info,
    print_warning, check_data_quality, generate_timestamp
)


def main():
    """
    🎯 Main function - runs the whole volatility forecasting show
    """
    parser = argparse.ArgumentParser(
        description="🔥 Stock Volatility Forecasting using GARCH Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ticker AAPL --period 2y
  python main.py --ticker TSLA --model EGARCH --forecast-horizon 10
  python main.py --ticker SPY --compare-models --backtest
        """
    )
    
    # Add all the cool arguments
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--period', type=str, default='2y',
                       help='Time period for data (default: 2y)')
    parser.add_argument('--model', type=str, default='GARCH',
                       choices=['GARCH', 'EGARCH', 'GJR-GARCH'],
                       help='GARCH model type (default: GARCH)')
    parser.add_argument('--forecast-horizon', type=int, default=5,
                       help='Forecast horizon in days (default: 5)')
    parser.add_argument('--compare-models', action='store_true',
                       help='Compare multiple GARCH models')
    parser.add_argument('--backtest', action='store_true',
                       help='Run backtesting analysis')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory for reports (default: reports)')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                       help='Confidence level for intervals (default: 0.95)')
    
    args = parser.parse_args()
    
    # Validate inputs like a pro
    if not validate_ticker(args.ticker):
        print_error(f"Invalid ticker: {args.ticker}")
        return 1
    
    if not validate_period(args.period):
        print_error(f"Invalid period: {args.period}")
        return 1
    
    if not validate_model_type(args.model):
        print_error(f"Invalid model type: {args.model}")
        return 1
    
    # Print the cool banner
    print_banner("Stock Volatility Forecasting")
    
    try:
        # Create directories for all the good stuff
        create_directories()
        
        # Load that data
        print_section("Data Loading")
        loader = StockDataLoader()
        data, returns, volatility = loader.get_data(args.ticker, args.period)
        
        # Check data quality (no garbage data allowed)
        quality_report = check_data_quality(data)
        if quality_report['issues']:
            print_warning("Data quality issues detected:")
            for issue in quality_report['issues']:
                print_warning(f"  {issue}")
        
        # Analyze the data
        print_section("Data Analysis")
        stats = loader.get_summary_stats(returns)
        stationarity = loader.check_stationarity(returns)
        clustering = loader.detect_volatility_clustering(returns)
        
        # Model fitting time
        print_section("Model Fitting")
        
        if args.compare_models:
            # Compare all the models (model battle royale)
            print_info("Comparing multiple GARCH models...")
            comparison = GARCHModelComparison()
            comparison_results = comparison.fit_models(returns)
            
            # Get the best model
            best_model_name, best_model = comparison.get_best_model('aic')
            print_success(f"🏆 Best model by AIC: {best_model_name}")
            
            # Use the best model for forecasting
            model_results = comparison.results[best_model_name]
            model_object = comparison.models[best_model_name]
            
        else:
            # Fit single model
            print_info(f"Fitting {args.model} model...")
            
            if args.model == 'GARCH':
                model = GARCHModel()
            elif args.model == 'EGARCH':
                model = EGARCHModel()
            elif args.model == 'GJR-GARCH':
                model = GJR_GARCHModel()
            
            model_results = model.fit(returns)
            model_object = model
        
        # Add model object to results for forecasting
        model_results['model_object'] = model_object
        
        # Forecasting time
        print_section("Forecasting")
        forecaster = VolatilityForecaster()
        forecast = forecaster.generate_forecast(
            model_results, 
            horizon=args.forecast_horizon,
            confidence_level=args.confidence_level
        )
        
        print_success(f"🔮 Forecast volatility: {forecast['volatility_forecast']:.4f}")
        
        # Backtesting if requested
        if args.backtest:
            print_section("Backtesting")
            backtest_results = forecaster.backtest_model(returns, model_results)
            
            if 'metrics' in backtest_results:
                metrics = backtest_results['metrics']
                print_info(f"📊 RMSE: {metrics.get('rmse', 'N/A'):.6f}")
                print_info(f"📊 MAE: {metrics.get('mae', 'N/A'):.6f}")
                print_info(f"📊 R²: {metrics.get('r_squared', 'N/A'):.4f}")
        
        # Visualization time
        print_section("Visualization")
        visualizer = VolatilityVisualizer()
        
        # Create all the sick plots
        plots = {}
        
        # Returns and volatility plot
        plots['returns_volatility'] = visualizer.plot_returns_and_volatility(
            returns, volatility, f"{args.ticker} - Returns & Volatility"
        )
        
        # Volatility clustering plot
        plots['clustering'] = visualizer.plot_volatility_clustering(returns)
        
        # GARCH diagnostics plot
        plots['diagnostics'] = visualizer.plot_garch_diagnostics(model_results)
        
        # Forecast plot
        plots['forecast'] = visualizer.plot_forecast(forecast, volatility)
        
        # Model comparison plot (if comparing models)
        if args.compare_models:
            plots['comparison'] = visualizer.plot_model_comparison(comparison_results)
        
        # Dashboard (the ultimate overview)
        plots['dashboard'] = visualizer.create_summary_dashboard(
            data, returns, volatility, model_results, forecast
        )
        
        # Save plots if requested
        if args.save_plots:
            print_info("Saving plots...")
            timestamp = generate_timestamp()
            
            for plot_name, fig in plots.items():
                filename = f"{args.output_dir}/{args.ticker}_{plot_name}_{timestamp}.html"
                visualizer.save_plot(fig, filename, 'html')
        
        # Generate reports
        print_section("Report Generation")
        timestamp = generate_timestamp()
        
        # Save model results
        results_file = f"{args.output_dir}/{args.ticker}_results_{timestamp}.json"
        save_json(model_results, results_file)
        
        # Save forecast results
        forecast_file = f"{args.output_dir}/{args.ticker}_forecast_{timestamp}.json"
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
            <strong>Model Type:</strong> {model_results['model_type']}<br>
            <strong>AIC:</strong> {model_results['aic']:.4f}<br>
            <strong>BIC:</strong> {model_results['bic']:.4f}<br>
            <strong>Log-Likelihood:</strong> {model_results['log_likelihood']:.4f}
        </div>
        
        <h2>🔮 Forecast Results</h2>
        <div class="metric">
            <strong>Forecast Horizon:</strong> {forecast['horizon']} days<br>
            <strong>Forecast Volatility:</strong> {forecast['volatility_forecast']:.4f}<br>
            <strong>Method:</strong> {forecast.get('method', 'unknown')}
        </div>
        """
        
        if args.backtest and 'metrics' in backtest_results:
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
        report_file = f"{args.output_dir}/{args.ticker}_report_{timestamp}.html"
        generate_html_report(
            f"Volatility Forecast Report - {args.ticker}",
            report_content,
            report_file
        )
        
        # Final summary
        print_section("Summary")
        print_success(f"✅ Analysis completed for {args.ticker}")
        print_info(f"📁 Reports saved in: {args.output_dir}")
        print_info(f"🔮 Forecast volatility: {forecast['volatility_forecast']:.4f}")
        
        if args.compare_models:
            print_info(f"🏆 Best model: {best_model_name}")
        
        return 0
        
    except Exception as e:
        print_error(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    main() 