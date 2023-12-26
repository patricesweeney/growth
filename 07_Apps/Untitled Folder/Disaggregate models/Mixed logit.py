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

#%% Add 'other' choices


def add_other_choices(data, paid_conversion_rate):
    import pandas as pd
    import numpy as np
    import uuid

    """
    Add rows representing 'Other' choices to the dataset. Assign unique UUIDs
    as workspace_id to the new rows while keeping the original workspace_id for existing rows.

    :param data: pandas DataFrame containing the original data
    :param paid_conversion_rate: float, the paid conversion rate (e.g., 0.04 for 4%)
    :return: pandas DataFrame with additional rows for 'Other' choices
    """
    # Calculate the number of 'Other' rows to add
    current_count = len(data)
    total_count_needed = current_count / paid_conversion_rate
    other_count = int(total_count_needed - current_count)

    # Create a DataFrame for 'Other' choices
    other_data = pd.DataFrame({
        'products_latest': ['Other'] * other_count,
        'seat_price': [0] * other_count,
        'seats_latest': [1] * other_count,
        # Add other columns as None
    })

    # Add other columns as None
    for col in data.columns:
        if col not in other_data:
            other_data[col] = np.nan

    # Generate unique UUIDs for the new rows
    other_data['workspace_id'] = [str(uuid.uuid4()) for _ in range(other_count)]

    # Append the 'Other' data to the original data
    updated_data = pd.concat([data, other_data], ignore_index=True)

    return updated_data

data = add_other_choices(data, 0.04)




#%% Double up volume as repeated choices with unavailable other choices



def double_up_volume(data, volume):
    import pandas as pd
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



    
simulator(data, choice = 'choice', model = model, market_size = 74350)


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


#%% Revenue optimal prices

import pandas as pd
import numpy as np
from scipy.optimize import minimize

def optimise(data, choice, model, market_size):
    # Extract price coefficient from the xlogit model
    price_coef = model.coeff_[0]
    
    # Calculate average prices for each choice
    avg_prices = data.groupby(choice)['seat_price'].mean()
    
    # Define a broad range of prices for optimization
    price_ranges = {c: (p * 0.5, p * 2) for c, p in avg_prices.items()}  # Price range for each product
    
    def demand_function(prices):
        utilities = np.exp(price_coef * pd.Series(prices))
        probabilities = utilities / utilities.sum()
        return probabilities * market_size
    
    def revenue_function(prices):
        demands = demand_function(prices)
        revenue = np.sum([prices[i] * demands[i] for i in range(len(prices))])
        return -revenue  # Negative revenue for maximization
    
    def jacobian(prices):
        demands = demand_function(prices)
        jacobian = np.zeros(len(prices))
        for i in range(len(prices)):
            partial_sum = np.sum([prices[j] * (np.exp(price_coef * prices[j]) / np.exp(price_coef * prices).sum()) * market_size for j in range(len(prices)) if j != i])
            jacobian[i] = demands[i] + prices[i] * price_coef * demands[i] + partial_sum
        return -jacobian  # Negative for maximization
    
    # Initial guess for prices - starting with average prices
    initial_prices = avg_prices.values
    bounds = list(price_ranges.values())  # Bounds for prices
    
    # Perform optimization
    result = minimize(revenue_function, initial_prices, jac=jacobian, bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        optimal_prices = result.x
        optimal_revenue = -revenue_function(optimal_prices)
        optimal_demands = demand_function(optimal_prices)
        optimal_probabilities = optimal_demands / market_size

        print("Optimal Prices:")
        for c, price in zip(avg_prices.index, optimal_prices):
            print(f"{c}: ${price:.2f}")
        print("Total Optimal Revenue: ${:,.2f}".format(optimal_revenue))
        
        print("\nChoice Probabilities at Optimal Prices:")
        for c, prob in zip(avg_prices.index, optimal_probabilities):
            print(f"{c}: {prob:.2f}")
        print("Sum of Probabilities: {:.2f}".format(optimal_probabilities.sum()))
    else:
        print("Optimization failed.")
        optimal_prices = None
        optimal_revenue = None
        optimal_probabilities = None

# Example usage
optimise(data, choice='choice', model=model, market_size=20000)


#%% Do it in torch coice


# =============================================================================
# Import shit
# =============================================================================

import torch
import numpy as np
import pandas as pd
from torch_choice.data import ChoiceDataset, JointDataset, utils
from torch_choice.model.nested_logit_model import NestedLogitModel
from torch_choice import run

# Ignore warnings for cleaner outputs
import warnings
warnings.filterwarnings("ignore")

# Print torch version
print(torch.__version__)

# Set device based on CUDA availability
if torch.cuda.is_available():
    print(f'CUDA device used: {torch.cuda.get_device_name()}')
    DEVICE = 'cuda'
else:
    print('Running tutorial on CPU')
    DEVICE = 'cpu'



#%% Create a chosen column

import pandas as pd

def create_chosen_column(data_long):
    # Ensure the necessary columns exist
    if 'choice' not in data_long.columns or 'alt' not in data_long.columns:
        raise ValueError("Required columns 'choice' or 'alt' are missing")

    # Rename columns
    data_long = data_long.rename(columns={'choice': 'temp_chosen', 'alt': 'choice'})

    # Create the 'chosen' column with True if 'choice' equals 'temp_chosen', else False
    data_long['chosen'] = data_long['choice'] == data_long['temp_chosen']

    # Drop the temporary column
    data_long = data_long.drop(columns=['temp_chosen'])

    return data_long

data_long = create_chosen_column(data_long)


#%%


# =============================================================================
# Count of choices
# =============================================================================

data['choice'].value_counts()


# =============================================================================
# Choice information
# =============================================================================

item_index = data_long[data_long['chosen'] == True].sort_values(by='workspace_id')['choice'].reset_index(drop=True)

item_names = data_long['choice'].unique().tolist()

num_items = data_long['choice'].nunique()


# =============================================================================
# Encode choices
# =============================================================================

encoder = dict(zip(item_names, range(num_items)))
item_index = item_index.map(lambda x: encoder[x])
item_index = torch.LongTensor(item_index)


# =============================================================================
# Nesting 
# =============================================================================

# nest feature: no nest feature, all features are item-level.
nest_dataset = ChoiceDataset(item_index=item_index.clone()).to(DEVICE)


# =============================================================================
# Set up regressors / X variables
# =============================================================================

item_feat_cols = ['price']

price_obs = utils.pivot3d(data_long, dim0='workspace_id', dim1='choice', values=item_feat_cols)

price_obs.shape

item_dataset = ChoiceDataset(item_index=item_index, price_obs=price_obs).to(DEVICE)


# =============================================================================
# Create final dataset
# =============================================================================

dataset = JointDataset(nest=nest_dataset, item=item_dataset)
print(dataset)


# =============================================================================
# Declare nesting
# =============================================================================

nest_to_item = {0: ['gcc', 'ecc', 'erc', 'hpc'],
                1: ['gc', 'ec', 'er']}

# encode items to integers.
for k, v in nest_to_item.items():
    v = [encoder[item] for item in v]
    nest_to_item[k] = sorted(v)

print(nest_to_item)


# =============================================================================
# Model
# =============================================================================

model = NestedLogitModel(nest_to_item=nest_to_item,
                         nest_coef_variation_dict={},
                         nest_num_param_dict={},
                         item_coef_variation_dict={'price_obs': 'constant'},
                         item_num_param_dict={'price_obs': 7},
                         shared_lambda=True)

model = NestedLogitModel(nest_to_item=nest_to_item,
                         nest_formula='',
                         item_formula='(price_obs|constant)',
                         dataset=dataset,
                         shared_lambda=True)

model = model.to(DEVICE)

print(model)

run(model, dataset, num_epochs=1000, model_optimizer="LBFGS")


#%% 
