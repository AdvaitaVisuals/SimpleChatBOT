import yfinance as yf
import pandas as pd
from langchain.tools import tool

@tool
def yfinance_stock_analysis(ticker: str) -> str:
    """
    Analyze stock data using yfinance for the given ticker symbol.

    Args:
        ticker: Ticker symbol for the company (e.g., AAPL for Apple Inc.)

    Returns:
        Analysis report as string containing key financial metrics.
    """
    try:
        # Fetch the stock data
        stock = yf.Ticker(ticker)

        # Get the info dictionary
        info = stock.info

        # Check if info is empty or invalid
        if not info or 'longName' not in info:
            return f"Error: Unable to fetch data for ticker '{ticker}'. Please check if the ticker symbol is correct."

        # Get historical market data
        history = stock.history(period="5y")

        if history.empty:
            return f"Error: No historical data available for ticker '{ticker}'. The stock may be delisted or the ticker may be incorrect."

        # Calculate 52-week high and low
        week_52_high = history['High'].tail(252).max() if len(history) >= 252 else history['High'].max()
        week_52_low = history['Low'].tail(252).min() if len(history) >= 252 else history['Low'].min()

        # Calculate 5-year revenue growth rate
        financials = stock.financials
        revenue_growth = None
        if not financials.empty and 'Total Revenue' in financials.index:
            revenue_5y = financials.loc['Total Revenue'].iloc[:5]
            if len(revenue_5y) >= 2:
                revenue_growth = ((revenue_5y.iloc[0] / revenue_5y.iloc[-1]) ** (1/len(revenue_5y)) - 1)

        # Prepare the analysis report
        analysis = {
            'Ticker Symbol': ticker,
            'Company Name': info.get('longName', 'N/A'),
            'Current Stock Price': info.get('currentPrice', 'N/A'),
            '52-Week High': round(week_52_high, 2) if not pd.isna(week_52_high) else 'N/A',
            '52-Week Low': round(week_52_low, 2) if not pd.isna(week_52_low) else 'N/A',
            'Market Capitalization': info.get('marketCap', 'N/A'),
            'PE Ratio': info.get('trailingPE', 'N/A'),
            'P/B Ratio': info.get('priceToBook', 'N/A'),
            'Debt to Equity Ratio': info.get('debtToEquity', 'N/A'),
            'Current Ratio': info.get('currentRatio', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            '5-Year Revenue Growth Rate(%)': revenue_growth,
            'Free Cash Flow': info.get('freeCashflow', 'N/A'),
            'Profit Margins': info.get('profitMargins', 'N/A'),
            'Operating Margin': info.get('operatingMargins', 'N/A'),
            'Earnings Growth': info.get('earningsGrowth', 'N/A'),
            'Revenue Growth': info.get('revenueGrowth', 'N/A'),
            'Analyst target Price': info.get('targetMedianPrice', 'N/A'),
            'Beta': info.get('beta', 'N/A'),
            '5-year Average Return on Equity(ROE)(%)': info.get('returnOnEquity', 'N/A')
        }

        # Convert percentage values
        for key in ['Dividend Yield', '5-Year Revenue Growth Rate(%)', 'Profit Margins', 'Operating Margin', 'Earnings Growth', 'Revenue Growth', '5-year Average Return on Equity(ROE)(%)']:
            if analysis[key] is not None and analysis[key] != 'N/A' and isinstance(analysis[key], (int, float)):
                analysis[key] = round(analysis[key] * 100, 2)

        return str(analysis)

    except Exception as e:
        return f"Error: Failed to fetch data for ticker '{ticker}'. Details: {str(e)}"
