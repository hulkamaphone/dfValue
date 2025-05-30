# Overview
dfValue is a python notebook for aggregating and providing financial stock information.

#### Key Features
* Pulls financial metrics from Yahoo Finance (Price, P/E, ROE, etc.).
* Scrapes DCF valuations from ValueInvesting.io.
* Converts DCF values to USD if needed.
* Calculates an "opportunity %" = how under- or overvalued a stock is.
* Outputs a sorted table of analyzed stocks with key metrics.

#### Capabilities
* CLI: Analyze custom stock tickers (```--stocks AAPL MSFT```).
* Currency handling: Auto FX conversion from Yahoo Finance via ```yfinance```.
* Ticker mapping: Resolves symbol mismatches (e.g., BRK.B ↔ BRK-B).
* Robust error handling for invalid data or network issues.

# Installation
The notebook is built on Jupyter notebook with the [Anaconda](https://anaconda.org/) python distribution in mind.

# Requirements
Some key packages will need to be installed that may not already exist in the default environment (especially a base Docker image like Jupyter's [minimal-notebook](https://hub.docker.com/r/jupyter/minimal-notebook)). Most of the packages will be available in Anaconda environment and above step likely not required.

```python
pip install yfinance pandas numpy requests beautifulsoup4
```
Also uses Python 3 (f-strings, type hints) and standard libraries like argparse, re.

# Usage
```python
python df_value.py --stocks AAPL MSFT
```
Or run without arguments to use a default ticker list.

# Limitations
For the most part, the process is `not` idempotent. Future versions should resolve this as it gets migrated into an application-type workflow.
* Fragile Web Scraping: HTML structure on ValueInvesting.io can break the script if changed.
* Live Data: Results vary based on current stock data, FX rates, and scraping success.
* Rate Limits: Excessive scraping may trigger blocks.

# License
[MIT](https://choosealicense.com/licenses/mit/)



