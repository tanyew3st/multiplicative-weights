import pandas as pd
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random


def multiplicative_weights_direction_based(df, initial_investment, learning_rate=0.2):
    num_stocks = df.shape[1]
    weights = np.full(num_stocks, 1 / num_stocks)
    portfolio_values = [initial_investment]
    random_pick_positive_days = 0

    for index in range(len(df) - 1):
        row = df.iloc[index]
        next_day_row = df.iloc[index + 1]
        
        open_prices = np.array([data[0] for data in row])
        close_prices = np.array([data[1] for data in row])
        
        # Calculate portfolio value based on current weights and close prices
        daily_portfolio_value = np.dot(weights, close_prices / open_prices) * portfolio_values[-1]
        portfolio_values.append(daily_portfolio_value)

        # Determine if the stock went up (loss = 0) or not (loss = 1)
        stock_loss = np.where(close_prices > open_prices, 0, 1)

        # Update weights based on stock loss
        weights *= (1 - learning_rate * stock_loss)

        weights /= np.sum(weights)  # Normalize the weights

        # Randomly pick a stock based on weights
        random_pick = np.random.choice(num_stocks, p=weights)
        if close_prices[random_pick] > open_prices[random_pick]:
            random_pick_positive_days += 1

    portfolio_values.append(np.dot(weights, df.iloc[-1].apply(lambda x: x[1] / x[0])) * portfolio_values[-1])

    return portfolio_values, random_pick_positive_days




def multiplicative_weights_update_same_day_close(df, initial_investment, learning_rate=0.2):
    num_stocks = df.shape[1]
    weights = np.full(num_stocks, 1 / num_stocks)
    portfolio_values = [initial_investment]

    for _, row in df.iterrows():
        open_prices = np.array([data[0] for data in row])
        close_prices = np.array([data[1] for data in row])

        # Calculate returns as the ratio of close to open prices
        daily_returns = close_prices / open_prices

        # Update portfolio value based on current weights and returns
        daily_portfolio_value = np.dot(weights, daily_returns) * portfolio_values[-1]
        portfolio_values.append(daily_portfolio_value)

        # Calculate the portfolio's return for normalization
        portfolio_return = weights @ daily_returns

        # Update the weights using the multiplicative weights algorithm as per the formula
        weights *= np.exp(learning_rate * daily_returns / portfolio_return)
        weights /= np.sum(weights)  # Normalize the weights to sum to 1
    
    # print the weights & stocks
    for symbol, weight in zip(df.columns, weights):
        print(f"{symbol}: {weight}")

    portfolio_values.append(np.dot(weights, df.iloc[-1].apply(lambda x: x[1] / x[0])) * portfolio_values[-1])

    return np.array(portfolio_values)



def multiplicative_weights_update(df, initial_investment, learning_rate=0.2):
    num_stocks = df.shape[1]
    weights = np.full(num_stocks, 1 / num_stocks)
    portfolio_values = [initial_investment]
    expected_values = []

    # Iterate over the DataFrame, except for the last row
    for index in range(len(df) - 1):
        row = df.iloc[index]
        next_day_open_prices = df.iloc[index + 1].apply(lambda x: x[0])  # Get open prices for the next day

        open_prices = np.array([data[0] for data in row])
        daily_returns = next_day_open_prices / open_prices  # Calculate returns based on next day's open prices

        daily_portfolio_value = np.dot(weights, daily_returns) * portfolio_values[-1]
        portfolio_values.append(daily_portfolio_value)

        weights *= np.exp(learning_rate * daily_returns)
        weights /= np.sum(weights)

        expected_value = np.sum(weights * daily_returns) * daily_portfolio_value
        expected_values.append(expected_value)

    # print the stock symbol with their weights at the end of the window
    for symbol, weight in zip(df.columns, weights):
        print(f"{symbol}: {weight}")
    return np.array(portfolio_values), np.array(expected_values)


def load_and_transform_data(folder_path):
    print(f"Loading data from {folder_path}...")

    # Dictionary to hold dataframes for each stock
    all_data = {}

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Extract the symbol from the filename
            symbol = filename.split('.')[0]

            # Read the CSV file
            df = pd.read_csv(os.path.join(folder_path, filename))

            # Check if 'Date' column exists and is in the correct format
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                print(f"Date column missing in file {filename}, skipping...")
                continue

            # Create a new column with Open and Close prices as a list
            df[symbol] = df[['Open', 'Close']].values.tolist()

            # Add the dataframe to the dictionary
            all_data[symbol] = df.set_index('Date')[symbol]

    # Combine all dataframes into a single dataframe
    combined_df = pd.DataFrame(all_data)

    return combined_df

def get_stock_window(folder_path, start_date, end_date):
    print(f"Getting stock data for the window {start_date} to {end_date}...")

    # Load and transform data
    stock_data = load_and_transform_data(folder_path)

    # Filter data based on the date window
    window_data = stock_data[(stock_data.index >= start_date) & (stock_data.index <= end_date)]

    # Drop columns with any NaN values in the window
    window_data = window_data.dropna(axis=1)

    return window_data

def equal_investment_simulation(df, initial_investment):
    # Initialize the current investment to the initial investment
    current_investment = initial_investment

    # Initialize an array to store portfolio value for each day
    portfolio_values = [initial_investment]

    # Loop over each day in the DataFrame
    for index, row in df.iterrows():
        daily_value = 0
        investment_per_stock = current_investment / df.shape[1]

        # Calculate the value for each stock
        for symbol in df.columns:
            open_price, close_price = row[symbol]
            # Calculate the proportion of change in price
            price_change_ratio = close_price / open_price
            # Update daily value
            daily_value += investment_per_stock * price_change_ratio

        # Update the current investment to the new daily value
        current_investment = daily_value

        # Append the daily portfolio value
        portfolio_values.append(daily_value)

    return np.array(portfolio_values)

def multiplicative_weights_update_with_momentum(df, initial_investment, learning_rate=0.2, transaction_cost=0.000, momentum_window=30):
    num_stocks = df.shape[1]
    weights = np.full(num_stocks, 1 / num_stocks)
    portfolio_values = [initial_investment]
    
    # To calculate momentum, we first need to transform the list of prices into separate columns
    open_prices_df = df.applymap(lambda x: x[0])  # Extract open prices
    close_prices_df = df.applymap(lambda x: x[1])  # Extract close prices

    # Now we can calculate the percentage change on the close prices
    momentum = close_prices_df.pct_change(periods=momentum_window).mean()

    for _, row in df.iterrows():
        open_prices = np.array([data[0] for data in row])
        close_prices = np.array([data[1] for data in row])

        # Calculate returns as the ratio of close to open prices
        daily_returns = close_prices / open_prices

        # Update portfolio value based on current weights and returns
        portfolio_value = np.dot(weights, daily_returns) * (portfolio_values[-1] - np.sum(transaction_cost * weights))
        portfolio_values.append(portfolio_value)

        # Calculate the portfolio's return for normalization
        portfolio_return = weights @ daily_returns

        # We need to ensure that the momentum values used are aligned with the current row
        # So we need to select the row from the momentum DataFrame that corresponds to the current date
        current_date = row.name  # assuming the index of df is a datetime index
        current_momentum = momentum.loc[current_date] if current_date in momentum.index else np.zeros(num_stocks)

        # Include momentum in the weight update
        adjusted_returns = daily_returns * (1 + current_momentum)

        # Update the weights using the multiplicative weights algorithm as per the formula
        weights *= np.exp(learning_rate * adjusted_returns / portfolio_return)
        weights /= np.sum(weights)  # Normalize the weights to sum to 1
    
    return np.array(portfolio_values)  

def best_stock_by_positive_days(df):
    positive_days_count = np.array([(df[symbol].apply(lambda x: x[1]) > df[symbol].apply(lambda x: x[0])).sum() for symbol in df.columns])
    best_stock_index = np.argmax(positive_days_count)
    best_stock = df.columns[best_stock_index]
    best_stock_positive_days = positive_days_count[best_stock_index]
    return best_stock, best_stock_positive_days

def best_single_stock_performance(df, initial_investment):
    final_values = {}
    for symbol in df.columns:
        open_prices = df[symbol].apply(lambda x: x[0])
        close_prices = df[symbol].apply(lambda x: x[1])
        final_values[symbol] = (close_prices / open_prices).cumprod()[-1] * initial_investment
    best_stock = max(final_values, key=final_values.get)

    # print out the best single stock
    print(f"Best single stock: {best_stock}")

    return df[best_stock].apply(lambda x: x[1] / x[0]).cumprod() * initial_investment

def average_stock_performance(df, initial_investment):
    daily_avg_returns = df.applymap(lambda x: x[1] / x[0]).mean(axis=1)
    return (daily_avg_returns).cumprod() * initial_investment

def calculate_daily_returns(df):
    returns_list = []
    for col in df.columns:
        # Extract open and close prices and calculate returns
        open_prices = df[col].apply(lambda x: x[0])
        close_prices = df[col].apply(lambda x: x[1])
        returns = close_prices / open_prices - 1
        returns_list.append(returns.rename(col))
    
    # Concatenate all returns series into a single DataFrame
    daily_returns = pd.concat(returns_list, axis=1)
    return daily_returns

def calculate_rolling_volatility(daily_returns_df, window_size):
    # Calculate rolling standard deviation (volatility) of returns
    rolling_volatility = daily_returns_df.rolling(window=window_size).std().mean(axis=1)
    return rolling_volatility

def plot_stock_progression(df):
    """
    Plots the progression of all stocks in the DataFrame.
    Each column in the DataFrame should represent a stock with each cell containing a tuple or list of (open, close) prices.
    """
    # Initialize a DataFrame to hold the cumulative returns
    cumulative_returns = pd.DataFrame(index=df.index)

    # Calculate cumulative returns for each stock
    for symbol in df.columns:
        # Extract open and close prices
        open_prices = df[symbol].apply(lambda x: x[0])
        close_prices = df[symbol].apply(lambda x: x[1])

        # Calculate daily returns and cumulative product
        daily_returns = close_prices / open_prices
        cumulative_returns[symbol] = daily_returns.cumprod()

    # Plotting
    plt.figure(figsize=(12, 6))
    for symbol in df.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[symbol], label=symbol)

    plt.title("Stock Price Progression Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_stock_values(stock_data, initial_investment):
    """ Calculate the values of a stock over time based on open prices. """
    stock_values = [initial_investment]
    for _, (open_price, close_price) in stock_data.iteritems():
        new_value = stock_values[-1] * (close_price / open_price)
        stock_values.append(new_value)
    return np.array(stock_values)

def annualized_return(portfolio_values, years):
    """Calculate the annualized return from a series of portfolio values."""
    if len(portfolio_values) > 1:
        return ((portfolio_values[-1] / portfolio_values[0]) ** (1 / years)) - 1
    else:
        return 0
    
def get_six_month_df():
    folder_path = 'data'  # Replace with your folder path
    
    # Define the start and end years
    start_year = 2013
    end_year = 2022

    # DataFrame to store the results
    results_df = pd.DataFrame(columns=[
        'Start Date', 'End Date', 
        'MW Direction-Based Return', 'Average Return',
        'Best Stock Return', 
        'MW Return', 'Best Positive Days Stock Return', 
        'Error Term', 'Best Stock Positive Days', 'MW Direction-Based Positive Days'
    ])

    # Iterate over 6-month intervals
    current_date = datetime(start_year, 1, 1)
    while current_date.year < end_year:
        end_date = current_date + timedelta(days=182)  # Approximately 6 months
        if end_date.year > end_year:  # Adjust for the final interval
            end_date = datetime(end_year, 12, 31)

        # Load the data for the current interval
        stock_window = get_stock_window(folder_path, current_date, end_date)
        num_stocks = len(stock_window.columns)

        # Calculate returns and positive days for each strategy
        mw_direction_returns, mw_direction_positive_days = multiplicative_weights_direction_based(stock_window, 100)
        best_stock, best_stock_positive_days = best_stock_by_positive_days(stock_window)
        print(best_stock)
        best_stock_returns = calculate_stock_values(stock_window[best_stock], 100)
        mw_returns = multiplicative_weights_update_same_day_close(stock_window, 100)
        average_stocks = average_stock_performance(stock_window, 100)

        # Calculate annualized returns
        half_year = 0.5  # 6 months as a fraction of a year
        mw_direction_annual_return = annualized_return(mw_direction_returns, half_year)
        best_stock_annual_return = annualized_return(best_stock_returns, half_year)
        mw_annual_return = annualized_return(mw_returns, half_year)
        average_annual_return = annualized_return(average_stocks, half_year)

        # Calculate error term
        half_year_days = len(best_stock_returns)
        error_term = 2 * np.sqrt(half_year_days * np.log(num_stocks))

        # Store the results
        results_df = results_df.append({
            'Start Date': current_date.strftime('%Y-%m-%d'), 
            'End Date': end_date.strftime('%Y-%m-%d'),
            'MW Direction-Based Return': mw_direction_annual_return,
            'Average Return': average_annual_return,
            'Best Stock Return': best_stock_annual_return,
            'MW Return': mw_annual_return,
            'Best Positive Days Stock Return': annualized_return(best_stock_returns, half_year),
            'Error Term': error_term,
            'Best Stock Positive Days': best_stock_positive_days,
            'MW Direction-Based Positive Days': mw_direction_positive_days
        }, ignore_index=True)

        print(results_df)

        # Move to the next interval
        current_date = end_date

    # Convert returns to percentages
    for col in ['MW Direction-Based Return', 'Best Stock Return', 'MW Return', 'Best Positive Days Stock Return', 'Average Return']:
        results_df[col] = results_df[col].apply(lambda x: f"{x * 100:.2f}%")

    # Print the DataFrame
    # save to csv
    results_df.to_csv('six_month_results.csv')

    return results_df

def grid_search_learning_rates(folder_path, start_year=2013, end_year=2022):
    learning_rates = np.arange(0.0, 0.5, 0.05)
    results = []

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    stock_window = get_stock_window(folder_path, start_date, end_date)
    
    # do the equal investment simulation once
    equal_investment_values = equal_investment_simulation(stock_window, 100)
    equal_investment_return = annualized_return(equal_investment_values, end_year - start_year)

    # best single stock based on number of positive days
    best_stock, best_stock_positive_days = best_stock_by_positive_days(stock_window)
    best_stock_values = calculate_stock_values(stock_window[best_stock], 100)
    best_stock_return = annualized_return(best_stock_values, end_year - start_year)


    for lr in learning_rates:
        print(f"Testing with learning rate: {lr}")

        # Test the directional based multiplicative weights
        directional_values, _ = multiplicative_weights_direction_based(stock_window, 100, lr)
        
        # Test the general multiplicative weights
        general_values = multiplicative_weights_update_same_day_close(stock_window, 100, lr)

        # Ensure both arrays are of the same length
        min_length = min(len(directional_values), len(general_values))
        directional_values = directional_values[:min_length]
        general_values = general_values[:min_length]

        # Calculate annualized returns
        directional_return = annualized_return(directional_values, end_year - start_year)
        general_return = annualized_return(general_values, end_year - start_year)

        results.append((lr, directional_return, general_return, equal_investment_return, best_stock_return))

    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=['Learning Rate', 'Directional Return', 'General Return', 'Equal Investment Return', 'Best Stock Return'])
    
    # Graph the results
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Learning Rate'], results_df['Directional Return'], label='Directional Return')
    plt.plot(results_df['Learning Rate'], results_df['General Return'], label='General Return')
    plt.axhline(equal_investment_return, color='black', linestyle='--', label='Equal Investment Return')
    plt.axhline(best_stock_return, color='red', linestyle='--', label='Best Stock Return')
    plt.xlabel('Learning Rate')
    plt.ylabel('Annualized Return')
    plt.title('Grid Search for Learning Rates (2013-2022)')
    plt.legend()
    plt.grid(True)

    # print it to jpeg
    plt.savefig('grid_search_learning_rates.jpeg')

    # save df to csv
    results_df.to_csv('grid_search_learning_rates.csv')

    return results_df

def plot_stock_performance(filename):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(filename)

    # Convert percentage strings to float values
    for col in ['MW Direction-Based Return', 'Average Return', 'Best Stock Return', 'MW Return', 'Best Positive Days Stock Return']:
        data[col] = data[col].str.rstrip('%').astype('float') / 100.0

    # Set the plot size
    plt.figure(figsize=(14, 7))
    
    # Plot each of the required columns
    plt.plot(data['Start Date'], data['MW Direction-Based Return'], label='MW Direction-Based Return')
    plt.plot(data['Start Date'], data['Average Return'], label='Average Return')
    plt.plot(data['Start Date'], data['Best Stock Return'], label='Best Stock Return')
    plt.plot(data['Start Date'], data['MW Return'], label='MW Return')
    plt.plot(data['Start Date'], data['Best Positive Days Stock Return'], label='Best Positive Days Stock Return')

    # Formatting the plot
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.title('Stock Performance Over Time')
    plt.legend()
    plt.xticks(rotation=45) # Rotate the x-axis labels to make them readable
    plt.tight_layout() # Adjust layout to prevent clipping of labels

    # Show the plot
    plt.savefig('stock_performance.jpeg')

def simulate_stock_selection_and_performance(stock_data, initial_investment, start_year=2013, end_year=2022):
    num_stocks_range = range(5, 405, 5)
    performances = {'Best Positive Stock': [], 'MW Direction-Based': [], 'MW General': [], 'Average Stock': [], 'Number of Stocks': []}

    for num_stocks in num_stocks_range:
        selected_stocks = random.sample(stock_data.columns.tolist(), num_stocks)
        selected_data = stock_data[selected_stocks]

        # Calculate performances
        best_stock, _ = best_stock_by_positive_days(selected_data)
        best_stock_values = calculate_stock_values(selected_data[best_stock], initial_investment)
        best_stock_return = annualized_return(best_stock_values, end_year - start_year)

        mw_direction_values, _ = multiplicative_weights_direction_based(selected_data, initial_investment)
        mw_direction_return = annualized_return(mw_direction_values, end_year - start_year)

        mw_general_values = multiplicative_weights_update_same_day_close(selected_data, initial_investment)
        mw_general_return = annualized_return(mw_general_values, end_year - start_year)

        average_stock_values = average_stock_performance(selected_data, initial_investment)
        average_stock_return = annualized_return(average_stock_values, end_year - start_year)

        # Store the results
        performances['Best Positive Stock'].append(best_stock_return)
        performances['MW Direction-Based'].append(mw_direction_return)
        performances['MW General'].append(mw_general_return)
        performances['Average Stock'].append(average_stock_return)
        performances['Number of Stocks'].append(num_stocks)

    # Plotting the performances
    plt.figure(figsize=(14, 7))
    plt.plot(performances['Number of Stocks'], performances['Best Positive Stock'], label='Best Positive Stock Return')
    plt.plot(performances['Number of Stocks'], performances['MW Direction-Based'], label='MW Direction-Based Return')
    plt.plot(performances['Number of Stocks'], performances['MW General'], label='MW General Return')
    plt.plot(performances['Number of Stocks'], performances['Average Stock'], label='Average Stock Return')

    plt.xlabel('Number of Stocks Selected')
    plt.ylabel('Annualized Return')
    plt.title('Performance based on Number of Stocks Selected')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig('performance_based_on_stock_selection.jpeg')
    return performances

# main method
if __name__ == '__main__':
    folder_path = 'data' 
    start_date = datetime(2013, 1, 1)
    end_date = datetime(2022, 12, 31)
    stock_window = get_stock_window(folder_path, start_date, end_date)

    # getting stock performances based on number of stocks selected
    performances = simulate_stock_selection_and_performance(stock_window, 100)
    print(performances)

    # grid searching learning rate
    performances_df = pd.DataFrame(performances)
    performances_df.to_csv('performance_based_on_stock_selection.csv', index=False)
    grid_search_results = grid_search_learning_rates(folder_path)
    print(grid_search_results)

    # plotting stock performance on six month intervals
    six_month_df = get_six_month_df()
    print(six_month_df)