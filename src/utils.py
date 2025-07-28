"""
🛠️ Utility Functions - All the helper functions you need to make things work
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def create_directories() -> None:
    """
    📁 Creates all the directories you need (keeps things organized)
    """
    directories = ['reports', 'data', 'models', 'plots']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")


def validate_ticker(ticker: str) -> bool:
    """
    ✅ Validates if a ticker symbol is legit (no fake tickers allowed)
    
    Args:
        ticker (str): Stock ticker to validate
        
    Returns:
        bool: True if valid, False if not
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Basic validation (you can make this more sophisticated)
    ticker = ticker.upper().strip()
    
    # Check if it's a reasonable length
    if len(ticker) < 1 or len(ticker) > 10:
        return False
    
    # Check if it contains only letters and maybe numbers
    if not ticker.replace('.', '').isalnum():
        return False
    
    return True


def validate_period(period: str) -> bool:
    """
    ✅ Validates if a time period is valid (no time travel allowed)
    
    Args:
        period (str): Time period to validate
        
    Returns:
        bool: True if valid, False if not
    """
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    
    if not period or not isinstance(period, str):
        return False
    
    return period.lower() in valid_periods


def validate_model_type(model_type: str) -> bool:
    """
    ✅ Validates if a model type is supported (only the good models)
    
    Args:
        model_type (str): Model type to validate
        
    Returns:
        bool: True if valid, False if not
    """
    valid_models = ['GARCH', 'EGARCH', 'GJR-GARCH']
    
    if not model_type or not isinstance(model_type, str):
        return False
    
    return model_type.upper() in valid_models


def format_number(value: float, decimals: int = 4) -> str:
    """
    🔢 Formats numbers to look nice (no ugly decimals)
    
    Args:
        value (float): Number to format
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted number string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    return f"{value:.{decimals}f}"


def generate_timestamp() -> str:
    """
    ⏰ Generates a timestamp for file naming (no more boring names)
    
    Returns:
        str: Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data: Dict[str, Any], filename: str) -> None:
    """
    💾 Saves data as JSON (the universal data format)
    
    Args:
        data (Dict): Data to save
        filename (str): Output filename
    """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"✅ JSON saved: {filename}")
    except Exception as e:
        print(f"❌ Failed to save JSON: {str(e)}")


def load_json(filename: str) -> Dict[str, Any]:
    """
    📂 Loads data from JSON (the universal data format)
    
    Args:
        filename (str): Input filename
        
    Returns:
        Dict: Loaded data
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"✅ JSON loaded: {filename}")
        return data
    except Exception as e:
        print(f"❌ Failed to load JSON: {str(e)}")
        return {}


def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """
    ⚙️ Loads configuration from file (all your settings in one place)
    
    Args:
        config_file (str): Configuration file path
        
    Returns:
        Dict: Configuration settings
    """
    if os.path.exists(config_file):
        return load_json(config_file)
    else:
        # Default configuration
        default_config = {
            "default_ticker": "AAPL",
            "default_period": "2y",
            "default_model": "GARCH",
            "forecast_horizon": 5,
            "confidence_level": 0.95,
            "rolling_window": 252,
            "output_directory": "reports"
        }
        
        # Save default config
        save_json(default_config, config_file)
        print(f"✅ Created default config: {config_file}")
        
        return default_config


def generate_html_report(title: str, content: str, filename: str) -> None:
    """
    📄 Generates a beautiful HTML report (because plain text is boring)
    
    Args:
        title (str): Report title
        content (str): Report content
        filename (str): Output filename
    """
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
            }}
            .metric {{
                background-color: #ecf0f1;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #3498db;
            }}
            .highlight {{
                background-color: #e8f5e8;
                padding: 10px;
                border-radius: 5px;
                border-left: 4px solid #27ae60;
            }}
            .warning {{
                background-color: #fff3cd;
                padding: 10px;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 {title}</h1>
            <div class="highlight">
                <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>
            {content}
        </div>
    </body>
    </html>
    """
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"✅ HTML report saved: {filename}")
    except Exception as e:
        print(f"❌ Failed to save HTML report: {str(e)}")


def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    🔍 Checks if your data is good quality (no garbage data allowed)
    
    Args:
        data (pd.DataFrame): Data to check
        
    Returns:
        Dict: Data quality report
    """
    quality_report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().to_dict(),
        'duplicate_rows': data.duplicated().sum(),
        'data_types': data.dtypes.to_dict(),
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'date_columns': data.select_dtypes(include=['datetime64']).columns.tolist(),
        'issues': []
    }
    
    # Check for common issues
    if len(data) < 100:
        quality_report['issues'].append("⚠️ Less than 100 observations (might be insufficient)")
    
    if data.isnull().sum().sum() > 0:
        quality_report['issues'].append("⚠️ Missing values detected")
    
    if data.duplicated().sum() > 0:
        quality_report['issues'].append("⚠️ Duplicate rows found")
    
    # Check for required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        quality_report['issues'].append(f"⚠️ Missing required columns: {missing_columns}")
    
    # Check for reasonable price ranges
    if 'Close' in data.columns:
        close_prices = data['Close'].dropna()
        if len(close_prices) > 0:
            if close_prices.min() <= 0:
                quality_report['issues'].append("⚠️ Negative or zero prices detected")
            if close_prices.max() > 10000:
                quality_report['issues'].append("⚠️ Unusually high prices detected")
    
    return quality_report


def calculate_summary_stats(data: pd.Series) -> Dict[str, float]:
    """
    📊 Calculates summary statistics (the basic stats that matter)
    
    Args:
        data (pd.Series): Data series
        
    Returns:
        Dict: Summary statistics
    """
    if len(data) == 0:
        return {}
    
    stats = {
        'count': len(data),
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'median': data.median(),
        'skewness': data.skew(),
        'kurtosis': data.kurtosis(),
        'q25': data.quantile(0.25),
        'q75': data.quantile(0.75)
    }
    
    return stats


def format_duration(seconds: float) -> str:
    """
    ⏱️ Formats duration in a human-readable way (no more confusing seconds)
    
    Args:
        seconds (float): Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_banner(title: str) -> None:
    """
    🎨 Prints a cool banner (makes your output look professional)
    
    Args:
        title (str): Banner title
    """
    width = 60
    print("=" * width)
    print(f"🚀 {title}")
    print("=" * width)


def print_section(section_name: str) -> None:
    """
    📋 Prints a section header (organizes your output)
    
    Args:
        section_name (str): Section name
    """
    print(f"\n📋 {section_name}")
    print("-" * 40)


def print_success(message: str) -> None:
    """
    ✅ Prints a success message (the good vibes)
    
    Args:
        message (str): Success message
    """
    print(f"✅ {message}")


def print_error(message: str) -> None:
    """
    ❌ Prints an error message (the bad vibes)
    
    Args:
        message (str): Error message
    """
    print(f"❌ {message}")


def print_warning(message: str) -> None:
    """
    ⚠️ Prints a warning message (the caution vibes)
    
    Args:
        message (str): Warning message
    """
    print(f"⚠️ {message}")


def print_info(message: str) -> None:
    """
    ℹ️ Prints an info message (the neutral vibes)
    
    Args:
        message (str): Info message
    """
    print(f"ℹ️ {message}")


# Test the utilities if run directly
if __name__ == "__main__":
    print_banner("Utility Functions Test")
    
    # Test validation functions
    print_section("Validation Tests")
    print_success(f"Valid ticker: {validate_ticker('AAPL')}")
    print_success(f"Valid period: {validate_period('2y')}")
    print_success(f"Valid model: {validate_model_type('GARCH')}")
    
    # Test formatting
    print_section("Formatting Tests")
    print_info(f"Formatted number: {format_number(123.456789, 2)}")
    print_info(f"Timestamp: {generate_timestamp()}")
    print_info(f"Duration: {format_duration(3661.5)}")
    
    # Test configuration
    print_section("Configuration Test")
    config = load_config()
    print_info(f"Default ticker: {config.get('default_ticker', 'N/A')}")
    
    print_success("All utility tests completed!") 