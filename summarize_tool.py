from langchain.tools import tool
import ast
import re

@tool
def summarize_last_message(message: str) -> str:
    """
    Summarize the content of the last message, such as stock analysis results or any text.
    This tool intelligently parses and summarizes different types of content.

    Args:
        message: The message content to summarize

    Returns:
        A well-formatted, concise summary of the message
    """
    try:
        # Clean up the message first
        cleaned_message = message.strip()

        # Check if this looks like a stock analysis result (dictionary format)
        if cleaned_message.startswith("{") and cleaned_message.endswith("}"):
            try:
                # Handle numpy types and special values
                cleaned_message = re.sub(r'np\.float64\(([^)]+)\)', r'\1', cleaned_message)
                cleaned_message = re.sub(r"'N/A'", 'None', cleaned_message)
                cleaned_message = re.sub(r'None', 'None', cleaned_message)

                # Parse the dictionary
                data = ast.literal_eval(cleaned_message)

                if isinstance(data, dict) and 'Ticker Symbol' in data:
                    return _summarize_stock_data(data)

            except Exception as e:
                # If parsing fails, try alternative approaches
                pass

        # Check if it contains stock data in a different format
        if 'Ticker Symbol' in cleaned_message or 'Company Name' in cleaned_message:
            return _summarize_stock_text(cleaned_message)

        # General text summarization
        return _summarize_general_text(cleaned_message)

    except Exception as e:
        return f"❌ Unable to summarize due to error: {str(e)}. Original message preview: {message[:150]}..."

def _summarize_stock_data(data: dict) -> str:
    """Summarize structured stock data dictionary"""
    ticker = data.get('Ticker Symbol', 'Unknown')
    company = data.get('Company Name', 'Unknown Company')
    price = data.get('Current Stock Price', 'N/A')
    market_cap = data.get('Market Capitalization', 'N/A')
    pe_ratio = data.get('PE Ratio', 'N/A')
    profit_margin = data.get('Profit Margins', 'N/A')

    # Format market cap nicely
    if isinstance(market_cap, (int, float)) and market_cap and market_cap != 'N/A':
        if market_cap >= 1e12:
            market_cap_str = f"${market_cap/1e12:.1f}T"
        elif market_cap >= 1e9:
            market_cap_str = f"${market_cap/1e9:.1f}B"
        elif market_cap >= 1e6:
            market_cap_str = f"${market_cap/1e6:.1f}M"
        else:
            market_cap_str = f"${market_cap:,.0f}"
    else:
        market_cap_str = 'N/A'

    # Format other metrics
    week_high = data.get('52-Week High', 'N/A')
    week_low = data.get('52-Week Low', 'N/A')
    debt_equity = data.get('Debt to Equity Ratio', 'N/A')
    revenue_growth = data.get('Revenue Growth', 'N/A')
    analyst_target = data.get('Analyst target Price', 'N/A')
    beta = data.get('Beta', 'N/A')

    summary = f"""📊 **Stock Analysis Summary: {ticker}**

🏢 **Company:** {company}
💰 **Current Price:** ${price if price != 'N/A' else 'N/A'}
🏛️ **Market Cap:** {market_cap_str}

📈 **Valuation Metrics:**
• P/E Ratio: {pe_ratio if pe_ratio != 'N/A' else 'N/A'}
• Profit Margin: {profit_margin}%{'' if profit_margin == 'N/A' else ''}

📊 **Price Range (52-week):**
• High: ${week_high if week_high != 'N/A' else 'N/A'}
• Low: ${week_low if week_low != 'N/A' else 'N/A'}

💼 **Financial Health:**
• Debt-to-Equity: {debt_equity if debt_equity != 'N/A' else 'N/A'}
• Revenue Growth: {revenue_growth}%{'' if revenue_growth == 'N/A' else ''}
• Beta (Volatility): {beta if beta != 'N/A' else 'N/A'}

🎯 **Analyst Target:** ${analyst_target if analyst_target != 'N/A' else 'N/A'}"""

    return summary

def _summarize_stock_text(text: str) -> str:
    """Summarize stock information from text format"""
    # Extract key information using regex
    ticker_match = re.search(r"Ticker Symbol[':]*\s*([^,\n]+)", text, re.IGNORECASE)
    company_match = re.search(r"Company Name[':]*\s*([^,\n]+)", text, re.IGNORECASE)
    price_match = re.search(r"Current Stock Price[':]*\s*([^\n,]+)", text, re.IGNORECASE)

    ticker = ticker_match.group(1).strip() if ticker_match else "Unknown"
    company = company_match.group(1).strip() if company_match else "Unknown Company"
    price = price_match.group(1).strip() if price_match else "N/A"

    return f"📊 **Stock Summary:** {ticker} ({company}) - Current Price: ${price}"

def _summarize_general_text(text: str) -> str:
    """Summarize general text content"""
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', text.strip())

    if len(cleaned) <= 300:
        return f"📝 **Summary:** {cleaned}"

    # For longer text, extract key sentences
    sentences = re.split(r'[.!?]+', cleaned)
    important_sentences = []

    # Look for sentences with key financial/stock terms
    key_terms = ['stock', 'price', 'market', 'company', 'analysis', 'growth', 'profit', 'revenue']
    total_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Prioritize sentences with financial terms
        has_key_term = any(term in sentence.lower() for term in key_terms)

        if has_key_term and total_length + len(sentence) <= 250:
            important_sentences.append(sentence)
            total_length += len(sentence)
        elif not has_key_term and total_length + len(sentence) <= 200 and len(important_sentences) < 2:
            important_sentences.append(sentence)
            total_length += len(sentence)

    if important_sentences:
        summary = '. '.join(important_sentences)
        return f"📝 **Summary:** {summary}."
    else:
        return f"📝 **Summary:** {cleaned[:250]}..."
