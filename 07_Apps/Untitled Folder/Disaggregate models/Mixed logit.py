#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 22:41:35 2023

@author: patricksweeney
"""

#%% Import
def import_data():
    import pandas as pd
    file_path = '/Users/patricksweeney/growth/07_Apps/Untitled Folder/Disaggregate models/Choice data 2.xlsx'
    data = pd.read_excel(file_path)
    return data

data = import_data()


#%% Date to index

def date_to_time(data):
    import pandas as pd

    # Function to check if a column is date-formatted
    def is_date_column(col):
        try:
            pd.to_datetime(data[col])
            return True
        except:
            return False

    # Find the first date-formatted column
    date_column = None
    for col in data.columns:
        if is_date_column(col):
            date_column = col
            break

    if date_column is None:
        raise ValueError("No date-formatted column found in the dataframe.")

    # Sort the dataframe by the date column
    data_sorted = data.sort_values(by=date_column)

    # Map each unique date to an ascending integer
    unique_dates = pd.to_datetime(data_sorted[date_column]).dt.date.unique()
    date_to_integer = {date: i+1 for i, date in enumerate(unique_dates)}

    # Create a new 'time' column based on the date-to-integer mapping
    data_sorted['time'] = data_sorted[date_column].apply(lambda x: date_to_integer[pd.to_datetime(x).date()])

    return data_sorted



data = date_to_time(data)

#%% Create available column for choice of product

def add_availability(data):
    # Create a new column 'available' and set all its values to 1
    data['available'] = 1


#%% Double up volume as repeated choices with unavailable other choices

import pandas as pd

def double_up_volume(data, volume):
    # Ensure the original data has the 'available' column set to 1
    data['available'] = 1

    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame()

    # Iterate over each row
    for index, row in data.iterrows():
        row_volume = row[volume]
        
        # Check if volume is greater than 1
        if row_volume > 1:
            # Duplicate the row volume - 1 times and set 'available' to 0
            row_copy = row.copy()
            row_copy['available'] = 0
            duplicates = pd.DataFrame([row_copy] * (int(row_volume) - 1))
            result_df = pd.concat([result_df, duplicates], ignore_index=True)
    
    # Combine the original data with the duplicated rows
    return pd.concat([data, result_df], ignore_index=True)

# Example usage
# data = ... (Your DataFrame with 'seats_latest' column)
# data = double_up_volume(data, 'seats_latest')


data = double_up_volume(data = data, volume = 'seats_latest')


#%% Create alternatives

    
def create_alternatives(data, choice, price, volume):
    import pandas as pd
    
    # Rename the column specified in the choice argument to 'choice'
    data = data.rename(columns={choice: 'choice'})
    
    # Calculate the average price for each choice
    avg_price = data.groupby('choice')[price].mean()
    
    # Get the unique choices
    choices = data['choice'].unique()
    
    # Create a price and volume column for each choice
    for c in choices:
        data[f'price_{c}'] = data.apply(lambda x: x[price] if x['choice'] == c else avg_price[c], axis=1)
        data[f'volume_{c}'] = data[volume]
        
    # Drop the original price and volume columns
    data = data.drop(columns=[volume])
    
    
    return data

data = create_alternatives(data = data, choice = 'products_latest', price = 'seat_price', volume = 'seats_latest')



#%% Transform

#Each price metric gets its own price and volume leading into the choice

def transform(data, id_col, choice):
    from xlogit.utils import wide_to_long
    
    # Get unique alternatives from the choice column
    alt_list = data[choice].unique().tolist()

    # Convert the data from wide to long format
    data_long = wide_to_long(data, id_col=id_col, alt_list=alt_list,
                             varying=['price', 'volume'], alt_name='alt', sep='_', alt_is_prefix=False)

    return data_long

data_long =  transform(data = data, id_col = 'workspace_id', choice = 'choice')



#%% Check for constants

import pandas as pd
import numpy as np

def constant_check(data_long, choice):
    # Define the variables to check
    variables_to_check = ['price', 'volume']

    # Group by the 'choice' variable
    grouped = data_long.groupby(choice)

    for var in variables_to_check:
        print(f"Checking variability for {var}:")
        for choice_val, group in grouped:
            std_dev = group[var].std()
            mean = group[var].mean()

            if std_dev == 0 or pd.isna(std_dev):
                print(f"  - {choice_val}: No variability in {var}")
            else:
                # Coefficient of variation = standard deviation / mean
                coef_of_variation = std_dev / mean
                print(f"  - {choice_val}: Coefficient of Variation for {var} is {coef_of_variation}")

    # Check for collinearity
    correlation = data_long['price'].corr(data_long['volume'])
    print(f"\nCollinearity Check (Correlation between price and volume): {correlation}")

    # Check for outliers
    for var in variables_to_check:
        q1, q3 = np.percentile(data_long[var], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers = data_long[(data_long[var] < lower_bound) | (data_long[var] > upper_bound)]
        print(f"\nOutliers in {var}: {len(outliers)}")

    # Check for matrix invertibility
    cov_matrix = data_long[variables_to_check].cov()
    det = np.linalg.det(cov_matrix)
    if det == 0 or np.isclose(det, 0):
        print("\nThe covariance matrix is not invertible or poorly conditioned.")
    else:
        print("\nThe covariance matrix is invertible.")

# Example usage
constant_check(data_long, choice='choice')






def transform_log_variables(data_long):
    import numpy as np

    # Function to add a tiny noise to a variable
    def add_noise(series):
        noise = np.random.uniform(-0.0001, 0.0001, series.shape)
        return series + noise

    # Add noise and then apply natural log transformation to the 'price' column
    data_long['price'] = np.log(add_noise(data_long['price']))

    # Add noise and then apply natural log transformation to the 'volume' column
    data_long['volume'] = np.log(add_noise(data_long['volume']))

    return data_long




data_long = transform_log_variables(data_long)

#%%
#Estimate model

def estimate_mixed_logit_model(data_long):
    from xlogit import MixedLogit
    varnames = ['price']

    model = MixedLogit()

    model.fit(X = data_long[varnames], y = data_long['choice'], varnames=varnames,
              alts=data_long['alt'], ids=data_long['workspace_id'], avail=data_long['available'],
              panels=None, randvars={'price': 'n'}, 
              #maxiter = 1000, n_draws=3500, 
              num_hess = True,
              optim_method='L-BFGS-B')

    model.summary()

estimate_mixed_logit_model(data_long)



#%%

def estimate_mnl_model(data_long):
    from xlogit import MultinomialLogit
    varnames = ['price']

    model = MultinomialLogit()

    model.fit(X = data_long[varnames], y = data_long['choice'], varnames=varnames,
              alts=data_long['alt'], ids=data_long['workspace_id'])

    model.summary()
    
    return model

model = estimate_mnl_model(data_long)


#%% Simulator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def simulator(data, choice, model, market_size):
    # Extract price coefficient from the xlogit model
    price_coef = model.coeff_[0]  # Adjust index as needed

    # Calculate average prices for each choice
    avg_prices = data.groupby(choice)['seat_price'].mean()
    
    print("Average Prices for Each Choice:")
    print(avg_prices)

    # Calculate the utility for each choice
    utilities = np.exp(price_coef * avg_prices)

    # Calculate the sum of all utilities
    sum_of_utilities = utilities.sum()

    # Calculate probabilities for each choice
    initial_probabilities = (utilities / sum_of_utilities)
    # Sort probabilities in descending order for plotting
    initial_probabilities = initial_probabilities.sort_values(ascending=False)

    # Plot initial probabilities
    ax = initial_probabilities.plot(kind='bar')
    plt.title('Initial Probabilities at Average Prices')
    plt.xlabel('Choice')
    plt.ylabel('Probability')

    # Annotate each bar with percentage or thousand separated integers
    for p in ax.patches:
        if market_size > 1:
            # Format as thousand separated integers
            annotation = f"{p.get_height() * market_size:,.0f}"
        else:
            # Format as percentage
            annotation = f"{p.get_height() * 100:.1f}%"
        ax.annotate(annotation, (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', xytext=(0, 10), textcoords='offset points')

    plt.show()

    # Create a range of prices around the average to simulate demand
    price_range = {c: np.linspace(p * 0.8, p * 1.2, 100) for c, p in avg_prices.items()}

    # Simulate demand probabilities for each choice
    demand_probabilities = {}
    for c, prices in price_range.items():
        probabilities = market_size * np.exp(price_coef * prices) / sum_of_utilities
        demand_probabilities[c] = probabilities

        # Plotting the demand curve for each choice
        rounded_price = round(avg_prices[c])
        plt.plot(prices, probabilities, label=f'Choice {c} (${rounded_price})')

    plt.title('Simulated Demand Curves')
    plt.xlabel('Price')
    plt.ylabel('Probability of Choosing')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    ######
    # Define a broad range of prices
    min_price, max_price = min(avg_prices), max(avg_prices)
    price_domain = np.linspace(min_price * 0.8, max_price * 1.2, 100)

    plt.figure(figsize=(10, 6))
    for c in avg_prices.index:
        # Create a DataFrame for different price scenarios
        temp_prices_df = pd.DataFrame({col: np.repeat(avg, len(price_domain)) for col, avg in avg_prices.items()})
        temp_prices_df[c] = price_domain  # Vary the price for the current choice
    
        # Compute utilities and probabilities for each scenario
        temp_utilities = market_size * np.exp(price_coef * temp_prices_df)
        temp_sum_of_utilities = temp_utilities.sum(axis=1)
        temp_probabilities = temp_utilities.div(temp_sum_of_utilities, axis=0)
    
        # Plot the demand curve for the choice
        plt.plot(price_domain, temp_probabilities[c], label=f'Choice {c}')
    
    plt.title('Fuller Demand Curves')
    plt.xlabel('Price')
    plt.ylabel('Percentage Demanded')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Return the original demand probabilities
    #return demand_probabilities
    
    # Compute total revenue
    total_revenue = 0
    for c, prob in initial_probabilities.items():
        revenue_for_choice = market_size * prob * avg_prices[c]
        total_revenue += revenue_for_choice

    # Format and print the total revenue
    formatted_revenue = "${:,.0f}".format(total_revenue)
    print("Total Revenue:", formatted_revenue)




simulator(data, choice = 'choice', model = model, market_size = 15186)


#%% Compare with distribution of seat choices

import pandas as pd
import matplotlib.pyplot as plt

def test_model(data, choice):
    # Count occurrences of each choice and convert to percentage
    total_count = len(data)
    choice_percentages = (data[choice].value_counts() / total_count) * 100

    # Plot a bar plot
    ax = choice_percentages.plot(kind='bar', figsize=(8, 6))
    
    # Annotate each bar with the percentage
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', xytext=(0, 10), textcoords='offset points')

    # Set labels and title
    plt.xlabel('Choices')
    plt.ylabel('Percentage (%)')
    plt.title('Bar Plot of Choice Distribution (Percentage)')
    plt.grid(axis='y')
    plt.show()


test_model(data, choice = 'choice')

