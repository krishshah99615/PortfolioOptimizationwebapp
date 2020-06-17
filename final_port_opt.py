# LIBRARIES
import pandas as pd
import numpy as np
import pandas_datareader as web
import datetime
import seaborn as sns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Function for data


def get_data(tickers, start, end):

    date_index = pd.date_range(start, end)
    data = pd.DataFrame(index=date_index)
    print(" \nDownloading Data...")
    for sym in tickers:
        df = web.DataReader(sym, 'yahoo', start, end)
        data = data.join(df['Open'])
        data.rename(columns={'Open': f'{sym}'}, inplace=True)
    data.dropna(inplace=True)

    print("Downloaded data and saved")
    print(data.head())
    return data


def opt_data(ticker, start, end, amt):
    weight = np.array([1/len(ticker)]*len(ticker))

    hist_data = get_data(ticker, start, end)

    daily_returns = hist_data.pct_change()
    cov_annual_mat = daily_returns.cov()*255

    port_variance = np.dot(weight.T, np.dot(cov_annual_mat, weight))
    port_volatility = np.sqrt(port_variance).round(4)
    port_simple_annual_return = np.sum(daily_returns.mean()*weight)*255

    mu = expected_returns.mean_historical_return(hist_data)
    S = risk_models.sample_cov(hist_data)
    ef = EfficientFrontier(mu, S)
    w = ef.max_sharpe()
    cw = ef.clean_weights()

    e = ef.portfolio_performance(verbose=False)
    latest_prices = get_latest_prices(hist_data)
    da = DiscreteAllocation(cw, latest_prices, total_portfolio_value=amt)

    allocation, leftover = da.lp_portfolio()

    return {
        "orignal_port_volatility": str(round(port_volatility, 3)*100)+"%",
        "orignal_annual_return": str(round(port_simple_annual_return, 3)*100)+"%",
        "new_port_volatility": str(round(e[1], 3)*100)+"%",
        "new_annual_return": str(round(e[0], 3)*100)+"%",
        "Allocation": cw,
        "AllocationNo": allocation,
        "Left_Amount": str(round(leftover, 2))

    }
