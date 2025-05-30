# Standard library imports
import argparse
import re

# Third-party library imports
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# --- Configuration ---
# List of stock tickers to analyze
DEFAULT_STOCK_LIST = ["AAPL", "BRK-B", "NVDA", "MSFT", "GOOGL", "AMZN", "V", "PLTR", "OKLO", "BABA", "BIDU", "QCOM", "JD"]

# Alternative ticker mapping for ValueInvesting.io
STOCK_REPLACEMENT_MAP = {
    'BRK-B': 'BRK.A',  # Berkshire Hathaway Class B to Class A for valueinvesting.io
    'JD': '9618.HK'    # JD.com ADR to Hong Kong listing for valueinvesting.io
}

# Web scraping configuration
BASE_URL_TEMPLATE_VI = "https://valueinvesting.io/{TICKER}/valuation/dcf-growth-exit-5y"
# Configuration for extracting DCF value from valueinvesting.io
# These specify the HTML structure to navigate to find the desired value.
# Grandparent element:
GP_TAG_CONFIG = 'div'
GP_CLASS_CONFIG = 'fs col-lg-2' # CSS class of the grandparent div
GP_INSTANCE_CONFIG = 1          # First instance of this grandparent div

# Parent element (within the grandparent):
P_TAG_CONFIG = 'div'
P_CLASS_CONFIG = 'price_square' # CSS class of the parent div
P_INSTANCE_CONFIG = 2           # Second instance of this parent div

# Child element (within the parent):
C_TAG_CONFIG = 'div'
C_CLASS_CONFIG = 'norm'         # CSS class of the child div (containing the value)
C_INSTANCE_CONFIG = 1           # First instance of this child div

# --- yfinance Data Retrieval ---

def get_stock_financial_metrics(ticker_symbol):
    """
    Retrieves key financial metrics for a given stock ticker using yfinance.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., "AAPL", "MSFT").

    Returns:
        dict: A dictionary containing the financial metrics.
              Returns None for metrics not available.
              Returns a dictionary with an error message if data cannot be fetched.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        # Check if essential data is missing, indicating an invalid or delisted ticker
        if not info or 'symbol' not in info or info.get('symbol', '').lower() != ticker_symbol.lower():
            if info.get('regularMarketPrice') is None and info.get('logo_url') == '':
                return {
                    "ticker": ticker_symbol, "error_message": f"Invalid/delisted ticker: {ticker_symbol}"
                }
            if 'symbol' not in info:
                 return {
                    "ticker": ticker_symbol, "error_message": f"Essential 'symbol' info missing for: {ticker_symbol}"
                }

        metrics = {
            "ticker": ticker_symbol,
            "price": info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose'))),
            "pe_ratio": info.get('trailingPE', info.get('forwardPE')),
            "eps": info.get('trailingEps', info.get('forwardEps')),
            "roe": info.get('returnOnEquity'),
            "roa": info.get('returnOnAssets'),
            "profit_margin": info.get('profitMargins'),
            "book_value_per_share": info.get('bookValue'),
            "shares_outstanding": info.get('sharesOutstanding'),
            "price_to_book": info.get('priceToBook'),
            "shortName": info.get('shortName', 'N/A')
        }
        return metrics

    except Exception as e:
        # Catch-all for other yfinance issues
        return {
            "ticker": ticker_symbol,
            "shortName": f"Error: {str(e)}",
            "error_message": str(e)
        }

def get_financials_for_stock_list(ticker_list):
    """
    Fetches financial metrics for a list of stock tickers.

    Args:
        ticker_list (list): A list of stock ticker symbols.

    Returns:
        pandas.DataFrame: A DataFrame containing the financial metrics.
    """
    all_metrics_data = []
    print(f"Fetching financial data for {len(ticker_list)} stocks from Yahoo Finance...")
    for i, ticker in enumerate(ticker_list):
        print(f"  Processing {ticker} ({i+1}/{len(ticker_list)})...")
        data = get_stock_financial_metrics(ticker)
        # Ensure all expected keys are present, filling with None if missing (except ticker and error_message)
        default_keys = {"price": None, "pe_ratio": None, "eps": None, "roe": None, "roa": None,
                        "profit_margin": None, "book_value_per_share": None,
                        "shares_outstanding": None, "price_to_book": None, "shortName": "N/A"}
        
        # Start with default_keys and update with actual data
        # This ensures DataFrame consistency even if some data points are missing for a ticker
        complete_data = {"ticker": ticker, "error_message": data.get("error_message")}
        for key, default_value in default_keys.items():
            complete_data[key] = data.get(key, default_value)
        
        all_metrics_data.append(complete_data)

    df = pd.DataFrame(all_metrics_data)

    # Define standard column order
    cols_order = ["ticker", "shortName", "price", "pe_ratio", "eps", "roe", "roa", "profit_margin",
                  "book_value_per_share", "shares_outstanding", "price_to_book", "error_message"]
    
    # Ensure all columns in cols_order exist in df, add if missing (e.g., if all tickers failed)
    for col in cols_order:
        if col not in df.columns:
            df[col] = None # Or np.nan if preferred for numeric types later

    df = df[cols_order] # Reorder
    return df

# --- Web Scraping for ValueInvesting.io ---

def extract_value_from_deeply_nested_div(url, gp_tag, gp_class, gp_instance,
                                         p_tag, p_class, p_instance,
                                         c_tag, c_class, c_instance):
    """
    Extracts text from a deeply nested HTML element based on tag names,
    CSS classes, and instance numbers for grandparent, parent, and child.

    Args:
        url (str): The URL to scrape.
        gp_tag, p_tag, c_tag (str): HTML tag names for grandparent, parent, child.
        gp_class, p_class, c_class (str): CSS class names.
        gp_instance, p_instance, c_instance (int): 1-based instance number.

    Returns:
        str or None: The extracted text, or None if not found or an error occurs.
    """
    if not all(isinstance(i, int) and i >= 1 for i in [gp_instance, p_instance, c_instance]):
        # print(f"Debug: Invalid instance numbers for {url}")
        return None
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15) # Increased timeout
        response.raise_for_status() # Will raise an HTTPError for bad responses (4XX or 5XX)
        soup = BeautifulSoup(response.content, 'html.parser')

        # 1. Find Grandparent
        all_grandparent_elements = soup.find_all(gp_tag, class_=gp_class)
        if len(all_grandparent_elements) < gp_instance:
            # print(f"Debug: Grandparent not found for {url}. Expected {gp_tag}.{gp_class} instance {gp_instance}, found {len(all_grandparent_elements)}")
            return None
        grandparent_element = all_grandparent_elements[gp_instance - 1]

        # 2. Find Parent within Grandparent
        all_parent_elements = grandparent_element.find_all(p_tag, class_=p_class)
        if len(all_parent_elements) < p_instance:
            # print(f"Debug: Parent not found for {url}. Expected {p_tag}.{p_class} instance {p_instance}, found {len(all_parent_elements)}")
            return None
        parent_element = all_parent_elements[p_instance - 1]

        # 3. Find Child within Parent
        all_child_elements = parent_element.find_all(c_tag, class_=c_class)
        if len(all_child_elements) < c_instance:
            # print(f"Debug: Child not found for {url}. Expected {c_tag}.{c_class} instance {c_instance}, found {len(all_child_elements)}")
            return None
        child_element = all_child_elements[c_instance - 1]
        
        return child_element.get_text(strip=True)

    except requests.exceptions.Timeout:
        print(f"Warning: Timeout while fetching {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Warning: Request failed for {url}: {e}")
        return None
    except Exception as e:
        print(f"Warning: An unexpected error occurred while scraping {url}: {e}")
        return None

def scrape_dcf_values(stock_list_vi):
    """
    Scrapes DCF values from valueinvesting.io for a list of tickers.

    Args:
        stock_list_vi (list): List of tickers (possibly modified for valueinvesting.io).

    Returns:
        pandas.DataFrame: DataFrame with scraped tickers and their DCF values.
    """
    urls_to_scrape = [BASE_URL_TEMPLATE_VI.format(TICKER=value) for value in stock_list_vi]
    results_data = []
    print(f"\nScraping DCF values for {len(urls_to_scrape)} URLs from valueinvesting.io...")

    for i, current_url in enumerate(urls_to_scrape):
        # Extract original ticker from URL for mapping back if needed
        # This assumes the ticker is the part of the path after the domain and before '/valuation'
        try:
            # A more robust way to get the ticker from the URL
            # Example: "https://valueinvesting.io/AAPL/valuation/dcf-growth-exit-5y" -> "AAPL"
            # Example: "https://valueinvesting.io/9618.HK/valuation/dcf-growth-exit-5y" -> "9618.HK"
            match = re.search(r"valueinvesting\.io/([^/]+)/valuation", current_url)
            ticker_from_url = match.group(1) if match else "N/A_URL_Parse_Error"
        except Exception:
            ticker_from_url = "N/A_URL_Parse_Error"
        
        print(f"  Scraping {current_url} for ticker {ticker_from_url} ({i+1}/{len(urls_to_scrape)})...")

        raw_extracted_value = extract_value_from_deeply_nested_div(
            current_url,
            GP_TAG_CONFIG, GP_CLASS_CONFIG, GP_INSTANCE_CONFIG,
            P_TAG_CONFIG, P_CLASS_CONFIG, P_INSTANCE_CONFIG,
            C_TAG_CONFIG, C_CLASS_CONFIG, C_INSTANCE_CONFIG
        )
        
        results_data.append({
            "ticker_vi": ticker_from_url, # Ticker used for scraping (e.g. BRK.A)
            "dcf_raw_value": raw_extracted_value
        })
    
    results_df = pd.DataFrame(results_data)
    return results_df

# --- Data Processing and Merging ---

def get_fx_rate(currency_code):
    """
    Fetches the FX rate to USD for a given currency code using yfinance.

    Args:
        currency_code (str): The 3-letter currency code (e.g., "HKD", "EUR").

    Returns:
        float or np.nan: The exchange rate, or np.nan if fetching fails or currency is USD.
    """
    if pd.isna(currency_code) or currency_code.upper() == "USD":
        return 1.0 # USD to USD is 1
    
    ticker_symbol = f"{currency_code.upper()}USD=X" # e.g., HKDUSD=X
    try:
        data = yf.Ticker(ticker_symbol).history(period="5d", interval="1d") # Get a few days for robustness
        if not data.empty and 'Close' in data.columns:
            # Use the most recent closing price
            return data['Close'].iloc[-1]
        print(f"Warning: No data found for FX rate {ticker_symbol}")
        return np.nan
    except Exception as e:
        print(f"Warning: Error fetching FX rate for {ticker_symbol}: {e}")
        return np.nan

def process_scraped_values(scraped_df):
    """
    Processes the raw scraped DCF values:
    - Extracts numeric value and currency.
    - Fetches FX rates for non-USD currencies.
    - Converts values to USD.

    Args:
        scraped_df (pandas.DataFrame): DataFrame with 'dcf_raw_value'.

    Returns:
        pandas.DataFrame: DataFrame with added 'dcf_currency', 'dcf_value_original_currency', and 'dcf_usd'.
    """
    if 'dcf_raw_value' not in scraped_df.columns:
        print("Error: 'dcf_raw_value' column missing in scraped_df.")
        scraped_df['dcf_currency'] = pd.NA
        scraped_df['dcf_value_original_currency'] = pd.NA
        scraped_df['dcf_usd'] = pd.NA
        return scraped_df

    # Extract numeric value and currency (e.g., "123.45 USD", "56.78 HKD")
    # Regex tries to find:
    # 1. Optional currency symbol like $
    # 2. Number with optional commas and decimal
    # 3. Optional space
    # 4. 3-letter currency code
    # If no currency code, assumes USD if it looks like a price.
    extracted_data = scraped_df['dcf_raw_value'].str.extract(r'([\$€£]?\s*[\d,\.]+)\s*([A-Z]{3})?', expand=True)
    extracted_data.columns = ['value_str', 'currency']

    scraped_df['dcf_value_original_currency'] = extracted_data['value_str'].str.replace(r'[^\d\.]', '', regex=True).astype(float, errors='ignore')
    scraped_df['dcf_currency'] = extracted_data['currency'].fillna('USD') # Default to USD if no currency detected

    # Get unique currencies to fetch FX rates
    unique_currencies = scraped_df['dcf_currency'].dropna().unique()
    fx_rates_map = {}
    print("\nFetching FX rates for detected currencies...")
    for curr in unique_currencies:
        if curr != "USD": # No need to fetch for USD
            print(f"  Fetching FX rate for {curr} to USD...")
            fx_rates_map[curr] = get_fx_rate(curr)
        else:
            fx_rates_map[curr] = 1.0 # USD to USD is 1
    
    scraped_df['fx_rate_to_usd'] = scraped_df['dcf_currency'].map(fx_rates_map)
    
    # Convert to USD
    scraped_df['dcf_usd'] = scraped_df['dcf_value_original_currency'] * scraped_df['fx_rate_to_usd']
    
    # Select and rename columns for clarity
    processed_df = scraped_df[['ticker_vi', 'dcf_raw_value', 'dcf_currency', 'dcf_value_original_currency', 'fx_rate_to_usd', 'dcf_usd']].copy()
    return processed_df

def merge_and_analyze_data(yfinance_df, processed_scraped_df, stock_replacement_map):
    """
    Merges Yahoo Finance data with processed scraped DCF data.
    Handles alternative tickers for merging.
    Calculates the 'opportunity' metric.

    Args:
        yfinance_df (pandas.DataFrame): DataFrame from get_financials_for_stock_list.
        processed_scraped_df (pandas.DataFrame): DataFrame from process_scraped_values.
        stock_replacement_map (dict): Mapping from yfinance ticker to valueinvesting.io ticker.

    Returns:
        pandas.DataFrame: The final merged and analyzed DataFrame.
    """
    # Prepare yfinance_df: add 'alt_ticker' for joining
    yfinance_df_copy = yfinance_df.copy()
    yfinance_df_copy['alt_ticker'] = yfinance_df_copy['ticker'].map(stock_replacement_map)
    # If no alt_ticker, use original ticker for matching with scraped data that might use original ticker
    yfinance_df_copy['join_key_vi'] = yfinance_df_copy['alt_ticker'].fillna(yfinance_df_copy['ticker'])


    # Prepare scraped_df: rename 'ticker_vi' to 'join_key_vi' for the merge
    processed_scraped_df_copy = processed_scraped_df.copy()
    processed_scraped_df_copy = processed_scraped_df_copy.rename(columns={'ticker_vi': 'join_key_vi'})

    # Merge yfinance data with scraped DCF data
    # We use 'join_key_vi' which is either the alt_ticker or the original yfinance ticker
    merged_df = pd.merge(
        yfinance_df_copy,
        processed_scraped_df_copy[['join_key_vi', 'dcf_usd', 'dcf_raw_value', 'dcf_currency']], # Select columns from scraped data
        on='join_key_vi',
        how='left'
    )

    # Calculate opportunity: (Current Price - DCF USD Value) / DCF USD Value * 100
    # This shows how much the current price is above (positive) or below (negative) the DCF value.
    # A negative opportunity means the stock is potentially undervalued according to DCF.
    merged_df['opportunity'] = np.where(
        (merged_df['dcf_usd'].notna()) & (merged_df['dcf_usd'] != 0) & (merged_df['price'].notna()),
        ((merged_df['price'] - merged_df['dcf_usd']) / merged_df['dcf_usd']) * 100,
        np.nan
    )
    
    # Rename columns for final presentation
    merged_df = merged_df.rename(columns={
        'price_to_book': 'pb',
        'pe_ratio': 'pe',
        'dcf_usd': 'dcf_5y_usd', # DCF 5-year growth exit model in USD
        'dcf_raw_value': 'dcf_5y_raw',
        'dcf_currency': 'dcf_5y_currency'
    })
    
    # Select and order final columns
    final_cols = ['ticker', 'shortName', 'price', 'pe', 'roe', 'profit_margin', 'pb',
                  'dcf_5y_raw', 'dcf_5y_currency', 'dcf_5y_usd', 'opportunity', 'error_message', 'alt_ticker']
    
    # Ensure all final_cols exist, adding them with None if not
    for col in final_cols:
        if col not in merged_df.columns:
            merged_df[col] = None
            
    return merged_df[final_cols]

# --- Main Execution ---
def main(stock_list_arg):
    """
    Main function to run the stock analysis pipeline.
    """
    print("Starting stock analysis application...")
    print("-" * 30)

    # 1. Get financial data from Yahoo Finance
    financials_df = get_financials_for_stock_list(stock_list_arg)
    
    # Filter out rows with critical errors from yfinance before proceeding
    # We keep rows even if some metrics are None, but drop if 'error_message' indicates a fundamental issue
    # A more nuanced approach might be needed depending on how strictly to treat missing yfinance data
    successful_yfinance_df = financials_df[financials_df['error_message'].isnull()].copy()
    if successful_yfinance_df.empty and not financials_df.empty :
        print("\nWarning: No stocks retrieved successfully from Yahoo Finance. Cannot proceed with valueinvesting.io scraping or full analysis.")
        # Display errors from yfinance
        print("\n--- Yahoo Finance Data Retrieval Summary (Errors) ---")
        print(financials_df[['ticker', 'shortName', 'error_message']].to_string())
        return
    elif financials_df.empty:
        print("\nError: Yahoo Finance returned no data at all. Exiting.")
        return

    # Convert relevant yfinance columns to numeric
    numeric_cols_yf = ["price", "pe_ratio", "eps", "roe", "roa", "profit_margin",
                       "book_value_per_share", "shares_outstanding", "price_to_book"]
    for col in numeric_cols_yf:
        if col in successful_yfinance_df.columns:
            successful_yfinance_df[col] = pd.to_numeric(successful_yfinance_df[col], errors='coerce')
    
    print(f"\nSuccessfully fetched initial data for {len(successful_yfinance_df)} stocks from Yahoo Finance.")
    if len(successful_yfinance_df) < len(stock_list_arg):
        print(f"Could not fetch initial data for {len(stock_list_arg) - len(successful_yfinance_df)} stocks.")
        errors_df = financials_df[financials_df['error_message'].notnull()]
        if not errors_df.empty:
            print("Stocks with yfinance errors:")
            print(errors_df[['ticker', 'shortName', 'error_message']].to_string())


    # 2. Prepare stock list for valueinvesting.io (using alternative tickers)
    # We use the original stock_list_arg to create stock_list_vi to ensure all intended stocks are attempted for scraping
    stock_list_vi = [STOCK_REPLACEMENT_MAP.get(ticker, ticker) for ticker in stock_list_arg]

    # 3. Scrape DCF values from valueinvesting.io
    scraped_dcf_df = scrape_dcf_values(stock_list_vi)
    if scraped_dcf_df.empty:
        print("\nWarning: No data scraped from valueinvesting.io. DCF analysis will be incomplete.")
        # If scraping fails entirely, create an empty df with expected columns for merging
        processed_scraped_df = pd.DataFrame(columns=['ticker_vi', 'dcf_usd', 'dcf_raw_value', 'dcf_currency'])
    else:
        # 4. Process scraped DCF values (extract currency, convert to USD)
        processed_scraped_df = process_scraped_values(scraped_dcf_df)
        #print("\n--- Processed DCF Data (Sample) ---")
        #print(processed_scraped_df.head().to_string())


    # 5. Merge yfinance data with processed scraped data and calculate opportunity
    # We use 'financials_df' here (which includes rows with yfinance errors)
    # to ensure all original tickers are present in the final output,
    # even if yfinance or VI scraping failed for them.
    # The 'successful_yfinance_df' was used for numeric conversions needed before this step.
    # The merge function itself will handle missing data gracefully.
    final_analysis_df = merge_and_analyze_data(financials_df, processed_scraped_df, STOCK_REPLACEMENT_MAP)

    # 6. Display final results
    # Sort by 'opportunity' (lower is better, meaning more undervalued)
    # Stocks with NaN opportunity (e.g. due to missing DCF) will be at the bottom if na_position='last'
    sorted_df = final_analysis_df.sort_values(by='opportunity', ascending=True, na_position='last')

    print("\n--- Final Stock Analysis (Sorted by Opportunity) ---")
    # Select key columns for the final display
    display_cols = ['ticker', 'shortName', 'price', 'pe', 'roe', 'dcf_5y_usd', 'opportunity']
    
    # Create a view for display, handling cases where some columns might be missing
    # (though merge_and_analyze_data tries to ensure they exist)
    display_df_view = pd.DataFrame()
    for col in display_cols:
        if col in sorted_df.columns:
            display_df_view[col] = sorted_df[col]
        else:
            display_df_view[col] = pd.NA # Or np.nan

    # Format opportunity as percentage for display
    if 'opportunity' in display_df_view.columns:
        display_df_view['opportunity'] = display_df_view['opportunity'].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    if 'dcf_5y_usd' in display_df_view.columns:
         display_df_view['dcf_5y_usd'] = display_df_view['dcf_5y_usd'].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    if 'price' in display_df_view.columns:
         display_df_view['price'] = display_df_view['price'].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

    # Show analysis table view.
    print(display_df_view.to_string(index=False))

    # Print complete message.
    print("\nAnalysis complete.")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Valuation Analysis Tool")
    parser.add_argument(
        "--stocks",
        nargs="+", # Allows multiple stock tickers
        default=DEFAULT_STOCK_LIST,
        help=f"List of stock tickers to analyze (e.g., AAPL MSFT GOOG). Defaults to: {', '.join(DEFAULT_STOCK_LIST)}"
    )
    args = parser.parse_args()
    
    main(args.stocks)
