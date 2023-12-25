import pandas as pd
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

def multiplicative_weights_direction_based(df: pd.DataFrame, initial_investment: float, learning_rate: float) -> tuple:
    """
    Calculates portfolio values using the Multiplicative Weights algorithm, considering stock price movements.
    It returns the portfolio values over time and the count of days a random stock pick had a positive movement.

    :param df: DataFrame with stock data (open and close prices).
    :param initial_investment: Initial amount invested in the portfolio.
    :param learning_rate: Learning rate for weight adjustments.
    :return: Tuple containing the list of portfolio values and the count of positive movement days for a random pick.
    """

    num_stocks = df.shape[1]
    weights = np.full(num_stocks, 1 / num_stocks)
    portfolio_values = [initial_investment]
    random_pick_positive_days = 0

    for index in range(len(df) - 1):
        row = df.iloc[index]
        open_prices = np.array([data[0] for data in row])
        close_prices = np.array([data[1] for data in row])
        daily_portfolio_value = np.dot(weights, close_prices / open_prices) * portfolio_values[-1]
        portfolio_values.append(daily_portfolio_value)
        stock_loss = np.where(close_prices > open_prices, 0, 1)
        weights *= (1 - learning_rate * stock_loss)
        weights /= np.sum(weights)
        random_pick = np.random.choice(num_stocks, p=weights)
        if close_prices[random_pick] > open_prices[random_pick]:
            random_pick_positive_days += 1

    final_day_value = np.dot(weights, df.iloc[-1].apply(lambda x: x[1] / x[0])) * portfolio_values[-1]
    portfolio_values.append(final_day_value)

    return portfolio_values, random_pick_positive_days

def multiplicative_weights_update_same_day_close(df: pd.DataFrame, initial_investment: float, learning_rate: float) -> np.ndarray:
    """
    Calculates portfolio values based on same-day stock price movements using the Multiplicative Weights algorithm.
    It updates weights based on daily stock returns and computes the portfolio value over time.

    :param df: DataFrame with stock data (open and close prices).
    :param initial_investment: Initial amount invested in the portfolio.
    :param learning_rate: Learning rate for weight adjustments.
    :return: Numpy array containing the portfolio values over time.
    """

    num_stocks = df.shape[1]
    weights = np.full(num_stocks, 1 / num_stocks)
    portfolio_values = [initial_investment]

    for _, row in df.iterrows():
        open_prices = np.array([data[0] for data in row])
        close_prices = np.array([data[1] for data in row])
        daily_returns = close_prices / open_prices
        daily_portfolio_value = np.dot(weights, daily_returns) * portfolio_values[-1]
        portfolio_values.append(daily_portfolio_value)
        portfolio_return = weights @ daily_returns
        weights *= np.exp(learning_rate * daily_returns / portfolio_return)
        weights /= np.sum(weights)

    final_day_value = np.dot(weights, df.iloc[-1].apply(lambda x: x[1] / x[0])) * portfolio_values[-1]
    portfolio_values.append(final_day_value)

    return np.array(portfolio_values)

def multiplicative_weights_update(df: pd.DataFrame, initial_investment: float, learning_rate: float) -> (np.ndarray, np.ndarray):
    """
    Calculates portfolio values and expected values using the Multiplicative Weights algorithm, 
    based on the next day's open prices. This version updates weights based on expected returns 
    from the next day's open prices and calculates expected values at each step.

    :param df: DataFrame with stock data (open and close prices).
    :param initial_investment: Initial amount invested in the portfolio.
    :param learning_rate: Learning rate for weight adjustments.
    :return: Tuple of two numpy arrays containing portfolio values and expected values over time.
    """

    num_stocks = df.shape[1]
    weights = np.full(num_stocks, 1 / num_stocks)
    portfolio_values = [initial_investment]
    expected_values = []

    for index in range(len(df) - 1):
        row = df.iloc[index]
        next_day_open_prices = df.iloc[index + 1].apply(lambda x: x[0])
        open_prices = np.array([data[0] for data in row])
        daily_returns = next_day_open_prices / open_prices
        daily_portfolio_value = np.dot(weights, daily_returns) * portfolio_values[-1]
        portfolio_values.append(daily_portfolio_value)
        weights *= np.exp(learning_rate * daily_returns)
        weights /= np.sum(weights)
        expected_value = np.sum(weights * daily_returns) * daily_portfolio_value
        expected_values.append(expected_value)

    final_day_value = np.dot(weights, df.iloc[-1].apply(lambda x: x[1] / x[0])) * portfolio_values[-1]
    portfolio_values.append(final_day_value)

    return np.array(portfolio_values), np.array(expected_values)

def load_and_transform_data(folder_path: str) -> pd.DataFrame:
    """
    Loads stock data from CSV files in the specified folder, transforming it into a DataFrame.
    Each CSV file represents a stock, with the file name as the stock symbol. The resulting DataFrame
    has dates as indices and each stock's open and close prices as columns.

    :param folder_path: Path to the folder containing the stock data CSV files.
    :return: DataFrame with combined stock data.
    """

    all_data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            symbol = filename.split('.')[0]
            df = pd.read_csv(os.path.join(folder_path, filename))

            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df[symbol] = df[['Open', 'Close']].values.tolist()
                all_data[symbol] = df.set_index('Date')[symbol]
            else:
                print(f"Date column missing in file {filename}, skipping...")

    combined_df = pd.DataFrame(all_data)
    return combined_df

def get_stock_window(folder_path: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Retrieves stock data within a specified time window.
    
    :param folder_path: Path to the folder containing stock data.
    :param start_date: Start date of the data window.
    :param end_date: End date of the data window.
    :return: DataFrame containing stock data within the specified window.
    """
    print(f"Getting stock data from {start_date} to {end_date}...")
    stock_data = load_and_transform_data(folder_path)
    window_data = stock_data[(stock_data.index >= start_date) & (stock_data.index <= end_date)].dropna(axis=1)
    return window_data

def equal_investment_simulation(df: pd.DataFrame, initial_investment: float) -> np.ndarray:
    """
    Simulates an equal investment strategy across all stocks in the DataFrame.
    
    :param df: DataFrame containing stock prices.
    :param initial_investment: Initial amount invested.
    :return: Array of portfolio values over time.
    """
    current_investment = initial_investment
    portfolio_values = [initial_investment]
    
    for _, row in df.iterrows():
        daily_value = sum((close_price / open_price) * (current_investment / len(df.columns)) 
                          for open_price, close_price in row)
        portfolio_values.append(daily_value)
        current_investment = daily_value

    return np.array(portfolio_values)

def multiplicative_weights_update_with_momentum(df: pd.DataFrame, initial_investment: float, learning_rate: float = 0.2, transaction_cost: float = 0.0, momentum_window: int = 30) -> np.ndarray:
    """
    Applies the Multiplicative Weights algorithm with momentum to update portfolio values.

    :param df: DataFrame containing stock prices.
    :param initial_investment: Initial investment amount.
    :param learning_rate: Learning rate for the algorithm.
    :param transaction_cost: Transaction costs for portfolio rebalancing.
    :param momentum_window: Window size for calculating momentum.
    :return: Array of updated portfolio values over time.
    """
    num_stocks = df.shape[1]
    weights = np.full(num_stocks, 1 / num_stocks)
    portfolio_values = [initial_investment]

    open_prices_df = df.applymap(lambda x: x[0])
    close_prices_df = df.applymap(lambda x: x[1])
    momentum = close_prices_df.pct_change(periods=momentum_window).mean()

    for _, row in df.iterrows():
        open_prices = np.array([data[0] for data in row])
        close_prices = np.array([data[1] for data in row])
        daily_returns = close_prices / open_prices
        portfolio_value = np.dot(weights, daily_returns) * (portfolio_values[-1] - np.sum(transaction_cost * weights))
        portfolio_values.append(portfolio_value)
        
        current_momentum = momentum.loc[row.name] if row.name in momentum.index else np.zeros(num_stocks)
        adjusted_returns = daily_returns * (1 + current_momentum)
        weights *= np.exp(learning_rate * adjusted_returns / (weights @ daily_returns))
        weights /= np.sum(weights)

    return np.array(portfolio_values)

def best_stock_by_positive_days(df: pd.DataFrame) -> tuple:
    """
    Identifies the best stock based on the number of positive days.

    :param df: DataFrame containing stock prices.
    :return: Tuple containing the best stock's symbol and its count of positive days.
    """
    positive_days_count = [(symbol, (df[symbol].apply(lambda x: x[1] > x[0])).sum()) for symbol in df.columns]
    best_stock, best_positive_days = max(positive_days_count, key=lambda x: x[1])
    return best_stock, best_positive_days

def best_single_stock_performance(df: pd.DataFrame, initial_investment: float) -> np.ndarray:
    """
    Calculates the best single stock performance over time.

    :param df: DataFrame containing stock prices.
    :param initial_investment: Initial amount invested.
    :return: Cumulative product of stock prices over time for the best-performing stock.
    """
    final_values = {symbol: (df[symbol].apply(lambda x: x[1] / x[0]).cumprod()[-1] * initial_investment) for symbol in df.columns}
    best_stock = max(final_values, key=final_values.get)
    return df[best_stock].apply(lambda x: x[1] / x[0]).cumprod() * initial_investment

def average_stock_performance(df: pd.DataFrame, initial_investment: float) -> np.ndarray:
    """
    Calculates the average performance of all stocks.

    :param df: DataFrame containing stock prices.
    :param initial_investment: Initial amount invested.
    :return: Cumulative average return of all stocks over time.
    """
    daily_avg_returns = df.applymap(lambda x: x[1] / x[0]).mean(axis=1)
    return daily_avg_returns.cumprod() * initial_investment

def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates daily returns for each stock in the DataFrame.

    :param df: DataFrame containing stock prices.
    :return: DataFrame of daily returns for each stock.
    """
    daily_returns = pd.concat([df[col].apply(lambda x: x[1] / x[0] - 1).rename(col) for col in df.columns], axis=1)
    return daily_returns

def calculate_rolling_volatility(daily_returns_df: pd.DataFrame, window_size: int) -> pd.Series:
    """
    Calculates the rolling volatility of daily returns.

    :param daily_returns_df: DataFrame of daily returns.
    :param window_size: Window size for rolling calculation.
    :return: Series of rolling volatility values.
    """
    rolling_volatility = daily_returns_df.rolling(window=window_size).std().mean(axis=1)
    return rolling_volatility

def plot_stock_progression(df: pd.DataFrame) -> None:
    """
    Plots the progression of stock prices over time.

    :param df: DataFrame containing stock prices with open and close values.
    """
    cumulative_returns = pd.DataFrame({symbol: df[symbol].apply(lambda x: x[1] / x[0]).cumprod() for symbol in df.columns}, index=df.index)
    cumulative_returns.plot(figsize=(12, 6), title="Stock Price Progression Over Time", xlabel="Date", ylabel="Cumulative Returns", grid=True, legend=True)

def calculate_stock_values(stock_data: pd.Series, initial_investment: float) -> np.ndarray:
    """
    Calculates the value of a stock over time based on open prices.

    :param stock_data: Series containing open and close prices of a stock.
    :param initial_investment: Initial investment amount.
    :return: Array of stock values over time.
    """
    stock_values = [initial_investment * (close_price / open_price) for open_price, close_price in stock_data]
    return np.array(stock_values)

def annualized_return(portfolio_values: np.ndarray, years: float) -> float:
    """
    Calculates the annualized return from a series of portfolio values.

    :param portfolio_values: Array of portfolio values over time.
    :param years: Number of years over which the return is calculated.
    :return: Annualized return value.
    """
    return ((portfolio_values[-1] / portfolio_values[0]) ** (1 / years)) - 1 if len(portfolio_values) > 1 else 0

def get_six_month_df(folder_path: str) -> pd.DataFrame:
    """
    Generates a DataFrame with six-month interval performance metrics for various stock strategies.

    :param folder_path: Path to the folder containing stock data.
    :return: DataFrame with performance metrics.
    """
    start_year, end_year = 2013, 2022
    results_df = pd.DataFrame()

    current_date = datetime(start_year, 1, 1)
    while current_date.year < end_year:
        end_date = current_date + timedelta(days=182) if current_date.year < end_year else datetime(end_year, 12, 31)
        stock_window = get_stock_window(folder_path, current_date, end_date)
        num_stocks = len(stock_window.columns)

        mw_direction_returns, mw_direction_positive_days = multiplicative_weights_direction_based(stock_window, 100)
        best_stock, best_stock_positive_days = best_stock_by_positive_days(stock_window)
        best_stock_returns = calculate_stock_values(stock_window[best_stock], 100)
        mw_returns = multiplicative_weights_update_same_day_close(stock_window, 100)
        average_stocks = average_stock_performance(stock_window, 100)

        half_year = 0.5
        results_df = results_df.append({
            'Start Date': current_date.strftime('%Y-%m-%d'), 
            'End Date': end_date.strftime('%Y-%m-%d'),
            'MW Direction-Based Return': annualized_return(mw_direction_returns, half_year),
            'Average Return': annualized_return(average_stocks, half_year),
            'Best Stock Return': annualized_return(best_stock_returns, half_year),
            'MW Return': annualized_return(mw_returns, half_year),
            'Best Positive Days Stock Return': annualized_return(best_stock_returns, half_year),
            'Error Term': 2 * np.sqrt(len(best_stock_returns) * np.log(num_stocks)),
            'Best Stock Positive Days': best_stock_positive_days,
            'MW Direction-Based Positive Days': mw_direction_positive_days
        }, ignore_index=True)

        current_date = end_date

    results_df['MW Direction-Based Return'] = results_df['MW Direction-Based Return'].apply(lambda x: f"{x * 100:.2f}%")
    # Repeat for other percentage columns

    results_df.to_csv('six_month_results.csv')
    return results_df

def grid_search_learning_rates(folder_path: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Performs a grid search over learning rates to analyze their impact on annualized returns.

    :param folder_path: Path to the folder containing stock data.
    :param start_year: Start year for the data window.
    :param end_year: End year for the data window.
    :return: DataFrame with learning rates and corresponding returns.
    """
    learning_rates = np.arange(0.0, 0.5, 0.05)
    results = []

    stock_window = get_stock_window(folder_path, datetime(start_year, 1, 1), datetime(end_year, 12, 31))
    equal_investment_values = equal_investment_simulation(stock_window, 100)
    equal_investment_return = annualized_return(equal_investment_values, end_year - start_year)

    for lr in learning_rates:
        directional_values, _ = multiplicative_weights_direction_based(stock_window, 100, lr)
        general_values = multiplicative_weights_update_same_day_close(stock_window, 100, lr)

        min_length = min(len(directional_values), len(general_values))
        results.append({
            'Learning Rate': lr,
            'Directional Return': annualized_return(directional_values[:min_length], end_year - start_year),
            'General Return': annualized_return(general_values[:min_length], end_year - start_year),
            'Equal Investment Return': equal_investment_return,
            'Best Stock Return': annualized_return(calculate_stock_values(stock_window[best_stock_by_positive_days(stock_window)[0]], 100), end_year - start_year)
        })

    results_df = pd.DataFrame(results)
    results_df.plot(x='Learning Rate', y=['Directional Return', 'General Return'], figsize=(12, 6), grid=True)
    plt.axhline(equal_investment_return, color='black', linestyle='--', label='Equal Investment Return')
    plt.savefig('grid_search_learning_rates.jpeg')
    results_df.to_csv('grid_search_learning_rates.csv')

    return results_df

def plot_stock_performance(filename: str) -> None:
    """
    Plots stock performance based on data from a given CSV file.

    :param filename: Path to the CSV file containing stock performance data.
    """
    data = pd.read_csv(filename)
    data = data.replace({'%': ''}, regex=True).astype(float)
    
    plt.figure(figsize=(14, 7))
    for col in ['MW Direction-Based Return', 'Average Return', 'Best Stock Return', 'MW Return', 'Best Positive Days Stock Return']:
        plt.plot(data['Start Date'], data[col] / 100.0, label=col)
    
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.title('Stock Performance Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stock_performance.jpeg')

def simulate_stock_selection_and_performance(stock_data: pd.DataFrame, initial_investment: float, start_year: int, end_year: int) -> dict:
    """
    Simulates stock selection performance based on a range of stock numbers.

    :param stock_data: DataFrame containing stock data.
    :param initial_investment: Initial amount invested.
    :param start_year: Start year of the analysis.
    :param end_year: End year of the analysis.
    :return: Dictionary containing performance data.
    """
    num_stocks_range = range(5, 405, 5)
    performances = {'Best Positive Stock': [], 'MW Direction-Based': [], 'MW General': [], 'Average Stock': [], 'Number of Stocks': []}

    for num_stocks in num_stocks_range:
        selected_stocks = random.sample(stock_data.columns.tolist(), num_stocks)
        selected_data = stock_data[selected_stocks]

        best_stock, _ = best_stock_by_positive_days(selected_data)
        mw_direction_values, _ = multiplicative_weights_direction_based(selected_data, initial_investment)
        mw_general_values = multiplicative_weights_update_same_day_close(selected_data, initial_investment)

        performances['Best Positive Stock'].append(annualized_return(calculate_stock_values(selected_data[best_stock], initial_investment), end_year - start_year))
        performances['MW Direction-Based'].append(annualized_return(mw_direction_values, end_year - start_year))
        performances['MW General'].append(annualized_return(mw_general_values, end_year - start_year))
        performances['Average Stock'].append(annualized_return(average_stock_performance(selected_data, initial_investment), end_year - start_year))
        performances['Number of Stocks'].append(num_stocks)

    plt.figure(figsize=(14, 7))
    for key in ['Best Positive Stock', 'MW Direction-Based', 'MW General', 'Average Stock']:
        plt.plot(performances['Number of Stocks'], performances[key], label=key)

    plt.xlabel('Number of Stocks Selected')
    plt.ylabel('Annualized Return')
    plt.title('Performance based on Number of Stocks Selected')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_based_on_stock_selection.jpeg')

    return performances
