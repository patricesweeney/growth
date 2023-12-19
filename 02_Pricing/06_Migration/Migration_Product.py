#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:14:29 2023

@author: patricksweeney
"""


#%% Import
def import_data():
    import pandas as pd
    file_path = '/Users/patricksweeney/growth/02_Pricing/06_Migration/Migration.xlsx'
    data = pd.read_excel(file_path)
    return data

data = import_data()


#%% Data checks
def data_checks(data):
    import pandas as pd

    # Initialize message list
    messages = []

    # Check for missing values and list columns with missing values
    for col in data.columns:
        if data[col].isnull().any():
            messages.append(f"Variable '{col}': Contains missing values")

    # Automatically set variables to int, float, or category
    for col in data.columns:
        original_dtype = data[col].dtype
        try:
            # Try converting to int
            data[col] = pd.to_numeric(data[col], downcast='integer')
        except ValueError:
            try:
                # Try converting to float
                data[col] = pd.to_numeric(data[col], downcast='float')
            except ValueError:
                # Convert to category if other conversions fail
                if original_dtype == object:
                    data[col] = data[col].astype('category')
                else:
                    messages.append(f"Variable '{col}': Cannot be converted to int, float, or category")

    # Return message or a success message if no issues are found
    return "\n".join(messages) if messages else "Data passes all checks."


data_checks(data)


#%% Choose varibales

def declare_variables(data, package, p1, v1, p2):
    # Select the specified columns and create a new DataFrame
    # Rename the columns to p1, v1, and v2
    new_data = data[[package, p1, v1, p2]].rename(columns={p1: 'p1', v1: 'v1', p2: 'p2'})

    return new_data

# Example usage
data = declare_variables(data, 'product', 'arpu', 'seats_latest', 'new_arpu')




#%% Summarise

def summarise_numbers(data):
    """
    Summarize key financial metrics from the dataset, with results rounded to the nearest integer.

    Parameters:
    - data (DataFrame): The dataset containing the variables p1, v1.

    Returns:
    - Dictionary containing the summary statistics, rounded to the nearest integer.
    """

    # Calculate Total Revenue
    total_revenue = round((data['p1'] * data['v1']).sum())

    # Count Total Customers
    total_customers = data.shape[0]

    # Calculate ARPA (Average Revenue Per Account)
    arpa = round(total_revenue / total_customers) if total_customers > 0 else 0

    # Sum Total Seats
    total_seats = round(data['v1'].sum())

    # Calculate ARPU (Average Revenue Per Unit)
    arpu = round(total_revenue / total_seats) if total_seats > 0 else 0

    return {
        'Total Revenue': total_revenue,
        'Total Customers': total_customers,
        'ARPA': arpa,
        'Total Seats': total_seats,
        'ARPU': arpu
    }
# Example usage:
summarise_numbers(data)




#%% Empirical survival function

def plot_empirical_survival_function_with_volume(data, product_column):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    """
    Plot the survival function of price for each product, taking into account the volume (v1) for each price (p1).
    If a product column is specified, creates a separate subplot for each product.

    Parameters:
    - data (DataFrame): The dataset containing the variables p1 (price) and v1 (volume), and optionally a product column.
    - product_column (str): The name of the column containing product identifiers. If None, plots for the entire dataset.

    Returns:
    - Subplots of the price survival function for each product.
    """

    if product_column and product_column in data.columns:
        products = data[product_column].unique()
        num_products = len(products)
        fig, axes = plt.subplots(num_products, 1, figsize=(10, 6 * num_products))

        for i, product in enumerate(products):
            product_data = data[data[product_column] == product]
            expanded_prices = np.repeat(product_data['p1'], product_data['v1'])
            sorted_prices = np.sort(expanded_prices)[::-1]
            survival_rate = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)

            ax = axes[i] if num_products > 1 else axes
            ax.plot(sorted_prices, survival_rate, label=f'Price Survival Function for {product}')
            ax.set_xlabel('Price (p1)')
            ax.set_ylabel('Survival Rate')
            ax.set_title(f'Survival Function of Price for {product}')
            ax.legend()
            ax.grid(True)
    else:
        # Expand the price data by volume
        expanded_prices = np.repeat(data['p1'], data['v1'])
        sorted_prices = np.sort(expanded_prices)[::-1]
        survival_rate = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices)

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_prices, survival_rate, label='Price Survival Function with Volume')
        plt.xlabel('Price (p1)')
        plt.ylabel('Survival Rate')
        plt.title('Survival Function of Price with Volume Consideration')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage:
plot_empirical_survival_function_with_volume(data, product_column='product')


#%%

def generalized_gamma_prf(data, product_column):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gengamma

    """
    Estimate and plot a Generalized Gamma survival function, price elasticity, and revenue curve for price data with volume consideration,
    adjusting for non-positive values, for each product, and print the model parameters.
    """

    if product_column and product_column in data.columns:
        products = data[product_column].unique()
        num_products = len(products)
        fig, axes = plt.subplots(num_products, 3, figsize=(45, 8 * num_products))

        for i, product in enumerate(products):
            product_data = data[data[product_column] == product]
            expanded_prices = np.repeat(product_data['p1'], product_data['v1'])
            df_for_gengamma = pd.DataFrame(expanded_prices, columns=['p1'])
            df_for_gengamma['p1'] = df_for_gengamma['p1'].clip(lower=0.001)

            # Fit a Generalized Gamma model
            params = gengamma.fit(df_for_gengamma['p1'])
            a, c, loc, scale = params

            prices = np.linspace(df_for_gengamma['p1'].min(), df_for_gengamma['p1'].max(), 100)
            survival_function = gengamma.sf(prices, a, c, loc=loc, scale=scale)

            # Subplot for survival function
            ax1 = axes[i, 0]
            ax1.plot(prices, survival_function, label=f'GG Survival Function for {product}')
            ax1.set_xlabel('Price (p1)')
            ax1.set_ylabel('Survival Probability')
            ax1.legend(loc='upper right')
            ax1.set_title(f'Generalized Gamma Survival Function for {product}')

            # Calculate and plot price elasticity (based on hazard rate)
            ax2 = axes[i, 1]
            hazard_rate = gengamma.pdf(prices, a, c, loc=loc, scale=scale) / survival_function
            elasticity = hazard_rate * prices
            ax2.plot(prices, elasticity, label=f'Price Elasticity for {product}', color='red')
            ax2.set_xlabel('Price (p1)')
            ax2.set_ylabel('Price Elasticity')
            ax2.legend(loc='upper right')
            ax2.set_title(f'Price Elasticity for {product}')

            # Subplot for revenue curve
            ax3 = axes[i, 2]
            revenue = prices * (1 - survival_function)
            ax3.plot(prices, revenue, label=f'Revenue for {product}', color='blue')
            ax3.set_xlabel('Price (p1)')
            ax3.set_ylabel('Revenue')
            ax3.legend(loc='upper right')
            ax3.set_title(f'Revenue Curve for {product}')

            print(f"Product: {product}")
            print("Generalized Gamma Model Parameters:")
            print(f"Shape parameter (a): {a:.4f}")
            print(f"Scale parameter (c): {c:.4f}")
            print(f"Location parameter: {loc:.4f}")
            print(f"Scale parameter: {scale:.4f}")
            print("\n")

        plt.tight_layout()
        plt.show()

    else:
        # Expand the price data by volume and ensure all prices are positive
        expanded_prices = np.repeat(data['p1'], data['v1'])
        df_for_gengamma = pd.DataFrame(expanded_prices, columns=['p1'])
        df_for_gengamma['p1'] = df_for_gengamma['p1'].clip(lower=0.001)

        # Fit a Generalized Gamma model
        params = gengamma.fit(df_for_gengamma['p1'])
        a, c, loc, scale = params

        prices = np.linspace(df_for_gengamma['p1'].min(), df_for_gengamma['p1'].max(), 100)
        survival_function = gengamma.sf(prices, a, c, loc=loc, scale=scale)

        # Plotting
        plt.figure(figsize=(15, 6))

        # Plot survival function
        ax1 = plt.subplot(1, 3, 1)
        ax1.plot(prices, survival_function, label='GG Survival Function with Volume')
        ax1.set_xlabel('Price (p1)')
        ax1.set_ylabel('Survival Probability')
        ax1.legend(loc='upper left')

        # Calculate and plot price elasticity (based on hazard rate)
        ax2 = plt.subplot(1, 3, 2)
        hazard_rate = gengamma.pdf(prices, a, c, loc=loc, scale=scale) / survival_function
        elasticity = hazard_rate * prices
        ax2.plot(prices, elasticity, label='Price Elasticity', color='red')
        ax2.set_xlabel('Price (p1)')
        ax2.set_ylabel('Price Elasticity')
        ax2.legend(loc='upper right')

        # Plot revenue curve
        ax3 = plt.subplot(1, 3, 3)
        revenue = prices * (1 - survival_function)
        ax3.plot(prices, revenue, label='Revenue Curve', color='blue')
        ax3.set_xlabel('Price (p1)')
        ax3.set_ylabel('Revenue')
        ax3.legend(loc='upper right')

        # Print model parameters
        print("Generalized Gamma Model Parameters:")
        print(f"Shape parameter (a): {a:.4f}")
        print(f"Scale parameter (c): {c:.4f}")
        print(f"Location parameter: {loc:.4f}")
        print(f"Scale parameter: {scale:.4f}")

        plt.tight_layout()
        plt.show()

# Example usage
generalized_gamma_prf(data, product_column='product')




#%% Fit an exponential function with lifelines
def exponential_prf(data, product_column):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from lifelines import ExponentialFitter

    """
    Estimate and plot an exponential survival function, log-elasticity, hazard rate, and revenue curve for price data with volume consideration,
    adjusting for non-positive values, for each product, and print the model parameters.

    Parameters:
    - data (DataFrame): The dataset containing the variables p1 (price), v1 (volume), and optionally a product column.
    - product_column (str): The name of the column containing product identifiers. If None, plots for the entire dataset.

    Returns:
    - Subplots of the estimated exponential survival function, log-elasticity, hazard rate, and revenue curve for each product.
    - Prints the model parameters for each product.
    """
    
    

    if product_column and product_column in data.columns:
        products = data[product_column].unique()
        num_products = len(products)
        fig, axes = plt.subplots(num_products, 4, figsize=(50, 8 * num_products))  # Increase number of subplots to 5

        for i, product in enumerate(products):
            product_data = data[data[product_column] == product]
            expanded_prices = np.repeat(product_data['p1'], product_data['v1'])
            df_for_lifelines = pd.DataFrame(expanded_prices, columns=['p1'])
            df_for_lifelines['p1'] = df_for_lifelines['p1'].clip(lower=0.001)
            df_for_lifelines['event_observed'] = 1

            # Calculate empirical survival function
            df_sorted = df_for_lifelines.sort_values(by='p1', ascending=False)
            empirical_survival = np.cumsum(df_sorted['event_observed']) / len(df_sorted)

            # Plot empirical survival function
            ax0 = axes[i, 0]
            ax0.step(df_sorted['p1'], empirical_survival, where="post", label=f'Empirical Survival for {product}')
            ax0.set_xlabel('Price (p1)')
            ax0.set_ylabel('Empirical Survival Probability')
            ax0.legend(loc='upper right')
            ax0.set_title(f'Empirical Survival Function for {product}')


            exp_fit = ExponentialFitter()
            exp_fit.fit(df_for_lifelines['p1'], event_observed=df_for_lifelines['event_observed'])
            lambda_value = 1 / exp_fit.lambda_

            # Subplot for survival function
            ax1 = axes[i, 0]
            exp_fit.survival_function_.plot(ax=ax1, label=f'Exp Survival Function for {product}')
            ax1.set_xlabel('Price (p1)')
            ax1.set_ylabel('Survival Probability')
            ax1.legend(loc='upper right')
            ax1.set_title(f'Exponential Survival Function for {product}')

            # Subplot for log-elasticity
            ax2 = axes[i, 1]
            prices = np.linspace(df_for_lifelines['p1'].min(), df_for_lifelines['p1'].max(), 100)
            log_elasticity = abs(-lambda_value * (prices))
            ax2.plot(prices, log_elasticity, label=f'Log-Elasticity for {product}', color='red', linestyle='--')
            ax2.set_xlabel('Price (p1)')
            ax2.set_ylabel('Log-Elasticity')
            ax2.legend(loc='upper right')
            ax2.set_title(f'Log-Elasticity for {product}')

            # Calculate and plot the distribution of WTP
            ax3 = axes[i, 2]
            wtp_distribution = lambda_value * np.exp(-lambda_value * prices)
            ax3.plot(prices, wtp_distribution, label=f'WTP Distribution for {product}', color='green')
            ax3.set_xlabel('Price (p1)')
            ax3.set_ylabel('Density')
            ax3.legend(loc='upper right')
            ax3.set_title(f'WTP Distribution for {product}')

            # Subplot for revenue curve
            ax4 = axes[i, 3]
            revenue = prices * np.exp(-lambda_value * prices)  # Revenue based on exponential survival function
            ax4.plot(prices, revenue, label=f'Revenue for {product}', color='blue')
            ax4.set_xlabel('Price (p1)')
            ax4.set_ylabel('Revenue')
            ax4.legend(loc='upper right')
            ax4.set_title(f'Revenue Curve for {product}')

            print(f"Product: {product}")
            print("Exponential Model Parameters:")
            print(f"Lambda (rate parameter): {lambda_value:.4f}")
            print(f"Log-likelihood: {exp_fit.log_likelihood_:.4f}")
            print("\n")
            
        plt.tight_layout()
        plt.show()

    else:
        # Expand the price data by volume and ensure all prices are positive
        expanded_prices = np.repeat(data['p1'], data['v1'])
        df_for_lifelines = pd.DataFrame(expanded_prices, columns=['p1'])
        df_for_lifelines['p1'] = df_for_lifelines['p1'].clip(lower=0.001)
        df_for_lifelines['event_observed'] = 1

        # Fit an exponential model
        exp_fit = ExponentialFitter()
        exp_fit.fit(df_for_lifelines['p1'], event_observed=df_for_lifelines['event_observed'])
        lambda_value = 1 / exp_fit.lambda_

        # Plotting
        plt.figure(figsize=(15, 6))

        # Plot survival function
        ax1 = plt.subplot(1, 4, 1)
        exp_fit.survival_function_.plot(ax=ax1, label='Exponential Survival Function with Volume')
        ax1.set_xlabel('Price (p1)')
        ax1.set_ylabel('Survival Probability')
        ax1.legend(loc='upper left')

        # Calculate and plot log-elasticity
        ax2 = plt.subplot(1, 4, 2)
        prices = np.linspace(df_for_lifelines['p1'].min(), df_for_lifelines['p1'].max(), 100)
        log_elasticity = -lambda_value * np.log(prices)
        ax2.plot(prices, log_elasticity, label='Log-Elasticity', color='red', linestyle='--')
        ax2.set_xlabel('Price (p1)')
        ax2.set_ylabel('Log-Elasticity')
        ax2.legend(loc='upper right')

        # Plot hazard rate
        ax3 = plt.subplot(1, 4, 3)
        ax3.axhline(y=lambda_value, color='green', linestyle='-.')
        ax3.set_xlabel('Price (p1)')
        ax3.set_ylabel('Hazard Rate')
        ax3.set_title('Constant Hazard Rate')
        ax3.legend(['Constant Hazard Rate'], loc='upper right')

        # Plot revenue curve
        ax4 = plt.subplot(1, 4, 4)
        revenue = prices * np.exp(-lambda_value * prices)
        ax4.plot(prices, revenue, label='Revenue Curve', color='blue')
        ax4.set_xlabel('Price (p1)')
        ax4.set_ylabel('Revenue')
        ax4.legend(loc='upper right')

        # Print model parameters
        print("Exponential Model Parameters:")
        print(f"Lambda (rate parameter): {lambda_value:.4f}")
        print(f"Log-likelihood: {exp_fit.log_likelihood_:.4f}")

        plt.tight_layout()
        plt.show()

# Example usage
exponential_prf(data, product_column='product')




#%% Weibull with distribution of WTP
def weibull_prf(data, product_column=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from lifelines import WeibullFitter

    """
    Estimate and plot a Weibull survival function, elasticity, hazard rate, and revenue curve for price data with volume consideration,
    adjusting for non-positive values, for each product, and print the model parameters.

    Parameters:
    - data (DataFrame): The dataset containing the variables p1 (price), v1 (volume), and optionally a product column.
    - product_column (str, optional): The name of the column containing product identifiers.

    Returns:
    - Subplots of the estimated Weibull survival function, elasticity, hazard rate, and revenue curve for each product.
    - Prints the model parameters for each product.
    """

    if product_column and product_column in data.columns:
        products = data[product_column].unique()
        num_products = len(products)
        fig, axes = plt.subplots(num_products, 4, figsize=(40, 8 * num_products))  # Increased number of subplots to 4
        
        weibull_params = {}
        for i, product in enumerate(products):
            product_data = data[data[product_column] == product]
            expanded_prices = np.repeat(product_data['p1'], product_data['v1'])
            df_for_lifelines = pd.DataFrame({'price': expanded_prices})
            df_for_lifelines['price'] = df_for_lifelines['price'].clip(lower=0.001)
            df_for_lifelines['event_observed'] = 1

            weibull_fit = WeibullFitter()
            weibull_fit.fit(df_for_lifelines['price'], event_observed=df_for_lifelines['event_observed'])
            lambda_param = weibull_fit.lambda_
            rho_param = weibull_fit.rho_
            
            weibull_params[product] = (weibull_fit.lambda_, weibull_fit.rho_)
            
            # Calculate empirical survival function
            df_sorted = df_for_lifelines.sort_values(by='price', ascending=False)
            empirical_survival = np.cumsum(df_sorted['event_observed']) / len(df_sorted)
            
            # Subplot for Weibull survival function with empirical overlay
            ax1 = axes[i, 0]
            weibull_fit.survival_function_.plot(ax=ax1, label=f'Weibull Survival Function for {product}')
            ax1.step(df_sorted['price'], empirical_survival, where="post", label=f'Empirical Survival for {product}', linestyle='--', color='orange')
            ax1.set_xlabel('Price (p1)')
            ax1.set_ylabel('Survival Probability')
            ax1.legend(loc='upper right')
            ax1.set_title(f'Weibull Survival Function for {product}')

            # Subplot for survival function
            ax1 = axes[i, 0]
            weibull_fit.survival_function_.plot(ax=ax1, label=f'Weibull Survival Function for {product}')
            ax1.set_xlabel('Price (p1)')
            ax1.set_ylabel('Survival Probability')
            ax1.legend(loc='upper right')
            ax1.set_title(f'Weibull Survival Function for {product}')

            # Subplot for elasticity
            ax2 = axes[i, 1]
            prices = np.linspace(df_for_lifelines['price'].min(), df_for_lifelines['price'].max(), 100)
            absolute_elasticity = abs(-rho_param * (prices / lambda_param) ** (rho_param))
            ax2.plot(prices, np.abs(absolute_elasticity), label=f'Absolute Elasticity for {product}', color='red', linestyle='--')
            ax2.set_xlabel('Price (p1)')
            ax2.set_ylabel('Absolute Elasticity')
            ax2.legend(loc='upper right')
            ax2.set_title(f'Absolute Elasticity for {product}')

            # Subplot for WTP distribution
            ax3 = axes[i, 2]
            prices_pdf = np.linspace(df_for_lifelines['price'].min(), df_for_lifelines['price'].max(), 100)
            weibull_pdf = (rho_param / lambda_param) * (prices_pdf / lambda_param)**(rho_param - 1) * np.exp(-(prices_pdf / lambda_param)**rho_param)
            ax3.plot(prices_pdf, weibull_pdf, label=f'WTP Distribution for {product}', color='green')
            ax3.set_xlabel('Price (p1)')
            ax3.set_ylabel('Density')
            ax3.legend(loc='upper right')
            ax3.set_title(f'WTP Distribution for {product}')


            # Find the price where absolute elasticity is approximately 1
            unit_elasticity_price = prices[np.argmin(np.abs(absolute_elasticity - 1))]
            
            # Subplot for revenue curve
            ax4 = axes[i, 3]
            revenue = prices * weibull_fit.survival_function_at_times(prices).values
            ax4.plot(prices, revenue, label=f'Revenue for {product}', color='blue')
            ax4.axvline(x=unit_elasticity_price, color='grey', linestyle='--', label='Unit Elasticity Price')
            ax4.set_xlabel('Price (p1)')
            ax4.set_ylabel('Revenue')
            ax4.legend(loc='upper right')
            ax4.set_title(f'Revenue Curve for {product}')


            print(f"Product: {product}")
            print("Weibull Model Parameters:")
            print(f"Lambda (scale parameter): {lambda_param:.4f}")
            print(f"Rho (shape parameter): {rho_param:.4f}")
            print(f"Log-likelihood: {weibull_fit.log_likelihood_:.4f}")
            print("\n")
            

        plt.tight_layout()
        plt.show()

    else:
        # Handling single product case
        expanded_prices = np.repeat(data['p1'], data['v1'])
        df_for_lifelines = pd.DataFrame({'price': expanded_prices})
        df_for_lifelines['price'] = df_for_lifelines['price'].clip(lower=0.001)
        df_for_lifelines['event_observed'] = 1

        weibull_fit = WeibullFitter()
        weibull_fit.fit(df_for_lifelines['price'], event_observed=df_for_lifelines['event_observed'])
        lambda_param = weibull_fit.lambda_
        rho_param = weibull_fit.rho_

        # Plotting
        plt.figure(figsize=(20, 6))

        # Plot survival function
        ax1 = plt.subplot(141)
        weibull_fit.survival_function_.plot(ax=ax1, label='Weibull Survival Function with Volume')
        ax1.set_xlabel('Price (p1)')
        ax1.set_ylabel('Survival Probability')
        ax1.legend(loc='upper left')

        # Calculate and plot elasticity
        ax2 = plt.subplot(142)
        prices = np.linspace(df_for_lifelines['price'].min(), df_for_lifelines['price'].max(), 100)
        elasticity = rho_param * (prices / lambda_param) ** (rho_param - 1)
        ax2.plot(prices, elasticity, label='Absolute Elasticity', color='red', linestyle='--')
        ax2.set_xlabel('Price (p1)')
        ax2.set_ylabel('Elasticity')
        ax2.legend(loc='upper right')

        # Plot hazard rate
        ax3 = plt.subplot(143)
        hazard_rate = weibull_fit.hazard_.values.flatten()
        ax3.plot(weibull_fit.hazard_.index, hazard_rate, label='Hazard Rate', color='green')
        ax3.set_xlabel('Price (p1)')
        ax3.set_ylabel('Hazard Rate')
        ax3.legend(loc='upper right')

        # Plot revenue curve
        ax4 = plt.subplot(144)
        revenue = prices * weibull_fit.survival_function_.values.flatten()[:100]
        ax4.plot(prices, revenue, label='Revenue Curve', color='blue')
        ax4.set_xlabel('Price (p1)')
        ax4.set_ylabel('Revenue')
        ax4.legend(loc='upper right')

        # Print model parameters
        print("Weibull Model Parameters:")
        print(f"Lambda (scale parameter): {lambda_param:.4f}")
        print(f"Rho (shape parameter): {rho_param:.4f}")
        print(f"Log-likelihood: {weibull_fit.log_likelihood_:.4f}")

        plt.tight_layout()
        plt.show()
        
    return weibull_params

# Example usage
weibull_params = weibull_prf(data, product_column='product')



#%% 



#%% Exponential survival fucntion


def calculate_new_revenue(data, p1, v1, p2, weibull_params, product_column):
    import numpy as np
    # Initialize new columns
    data['v2'] = np.nan
    data['direction'] = 'No change'
    data['elasticity'] = np.nan
    data['r1'] = data[p1] * data[v1]
    data['r2'] = np.nan
    data['lift'] = np.nan

    for product in data[product_column].unique():
        lambda_param, rho_param = weibull_params.get(product, (None, None))
        if lambda_param and rho_param:
            product_data = data[data[product_column] == product]
            new_volume = product_data[v1] * np.exp(-((product_data[p2] / lambda_param) ** rho_param - (product_data[p1] / lambda_param) ** rho_param))
            data.loc[data[product_column] == product, 'v2'] = new_volume

            # Calculate direction
            data.loc[data[product_column] == product, 'direction'] = np.where(product_data[p2] > product_data[p1], 'Increase', 
                                                                               np.where(product_data[p2] < product_data[p1], 'Decrease', 'No change'))

            # Calculate elasticity
            price_ratio = np.log(product_data[p2] / product_data[p1])
            volume_ratio = np.log(new_volume / product_data[v1])
            data.loc[data[product_column] == product, 'elasticity'] = np.abs(volume_ratio / price_ratio)

            # Calculate new revenue and lift
            data.loc[data[product_column] == product, 'r2'] = product_data[p2] * new_volume
            # Calculate arithmetic lift instead of log lift
            data.loc[data[product_column] == product, 'lift'] = (data.loc[data[product_column] == product, 'r2'] - data.loc[data[product_column] == product, 'r1']) / data.loc[data[product_column] == product, 'r1'].replace(0, np.nan)

    # Handle cases where price or revenue doesn't change to avoid division by zero
    data.loc[data['direction'] == 'No change', 'elasticity'] = np.nan
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    return data

# Example usage
# Assuming 'data' is a predefined DataFrame and 'weibull_params' is a predefined dictionary
new_data = calculate_new_revenue(data, 'p1', 'v1', 'p2', weibull_params, 'product')


#%% Save




#%%

def calculate_impact(new_data):
    import matplotlib.pyplot as plt
    import pandas as pd 
    import numpy as np
    
    # Ensure the data types are correct for calculations
    new_data[['p1', 'v1', 'p2', 'v2']] = new_data[['p1', 'v1', 'p2', 'v2']].apply(pd.to_numeric, errors='coerce')
    
    # Replace all zeros in p1, p2, v1, and v2 with a small value
    small_value = 0.01
    new_data['p1'] = new_data['p1'].replace(0, small_value)
    new_data['p2'] = new_data['p2'].replace(0, small_value)
    new_data['v1'] = new_data['v1'].replace(0, small_value)
    new_data['v2'] = new_data['v2'].replace(0, small_value)
    
    # Initial Calculations
    new_data['r1'] = new_data['p1'] * new_data['v1']
    new_data['r2'] = new_data['p2'] * new_data['v2']
    new_data['lift'] = (new_data['r2'] - new_data['r1']) / new_data['r1'].replace(0, np.nan)
    new_data['elasticity'] = np.abs((new_data['v2'] - new_data['v1']) / new_data['v1']) / np.abs((new_data['p2'] - new_data['p1']) / new_data['p1'])
    
    # Revenue Impact
    total_revenue_diff = new_data['r2'].sum() - new_data['r1'].sum()
    percent_revenue_diff = total_revenue_diff / new_data['r1'].sum()
    
    
    new_data['elasticity'] = np.abs((new_data['v2'] - new_data['v1']) / new_data['v1'].replace(0, np.nan)) / np.abs((new_data['p2'] - new_data['p1']) / new_data['p1'].replace(0, np.nan))
    new_data['elasticity'].replace([np.inf, -np.inf], np.nan, inplace=True)

    
    # Portfolio Lift and Volatility
    total_revenue = new_data['r1'].sum()
    new_data['weight'] = new_data['r1'] / total_revenue
    portfolio_lift = (new_data['lift'] * new_data['weight']).sum()
    portfolio_volatility = np.sqrt((new_data['weight'] * (new_data['lift'] - portfolio_lift)**2).sum())
    sharpe_ratio = portfolio_lift / portfolio_volatility if portfolio_volatility != 0 else np.nan
    
    # APRA Impact
    apra_diff = new_data['r2'].mean() - new_data['r1'].mean()
    percent_apra_diff = apra_diff / new_data['r1'].mean()
    
    # Volume Impact
    volume_diff = (new_data['v2'] > 1).sum() - (new_data['v1'] > 1).sum()
    percent_volume_diff = volume_diff / (new_data['v1'] > 1).sum()
    
    # Seats Impact
    seats_diff = volume_diff
    percent_seats_diff = percent_volume_diff
    
    # ARPU Impact
    arpu_diff = new_data['r2'].mean() - new_data['r1'].mean()
    percent_arpu_diff = percent_apra_diff

    # Plot PMF of direction column
    if 'direction' in new_data.columns:
        direction_pmf = new_data['direction'].value_counts(normalize=True)
        plt.figure()
        direction_pmf.plot(kind='bar')
        plt.title('Direction Probability Mass Function')
        plt.ylabel('Percentage')
        plt.xlabel('Direction')
    
    # Creating subplots for distributions
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of subplots

    # Distribution of Revenue Lift
    axs[0, 0].hist(new_data['lift'].replace([np.inf, -np.inf], np.nan).dropna(), bins=20, edgecolor='black', color='blue')
    axs[0, 0].set_title('Distribution of Revenue Lift')
    axs[0, 0].set_xlabel('Lift')
    axs[0, 0].set_ylabel('Frequency')

    # Distribution of Elasticity
    axs[0, 1].hist(new_data['elasticity'].dropna(), bins=20, edgecolor='black', color='green')
    axs[0, 1].set_title('Distribution of Elasticity')
    axs[0, 1].set_xlabel('Elasticity')
    axs[0, 1].set_ylabel('Frequency')

    # Distribution of Price Change (expressed as a percentage)
    price_change = 100 * np.abs((new_data['p2'] - new_data['p1']) / new_data['p1'])
    axs[1, 0].hist(price_change.dropna(), bins=20, edgecolor='black', color='red')
    axs[1, 0].set_title('Distribution of Price Change (%)')
    axs[1, 0].set_xlabel('Price Change (%)')
    axs[1, 0].set_ylabel('Frequency')


    # Distribution of Volume Change (expressed as a percentage)
    volume_change = 100 * np.abs((new_data['v2'] - new_data['v1']) / new_data['v1'])
    axs[1, 1].hist(volume_change.dropna(), bins=20, edgecolor='black', color='purple')
    axs[1, 1].set_title('Distribution of Volume Change (%)')
    axs[1, 1].set_xlabel('Volume Change (%)')
    axs[1, 1].set_ylabel('Frequency')


    plt.tight_layout()  # Adjusts the subplots to fit into the figure area.
    plt.show()

    # Recommendation: Go / No-Go
    recommendation = 'Go' if total_revenue_diff > 0 else 'No-Go'

    # Formatting results
    results = {
        'Total Revenue Difference': f"${total_revenue_diff:,.0f}",
        'Percent Revenue Difference': f"{percent_revenue_diff:.1%}",
        '\nPortfolio Lift': f"{portfolio_lift * 100:.1f}%",
        'Portfolio Volatility': f"{portfolio_volatility * 100:.1f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        '\nAPRA Difference': f"${apra_diff:,.0f}",
        'Percent APRA Difference': f"{percent_apra_diff:.1%}",
        '\nVolume Difference': f"{volume_diff:,d}",
        'Percent Volume Difference': f"{percent_volume_diff:.1%}",
        '\nSeats Difference': f"{int(seats_diff):,d}",
        'Percent Seats Difference': f"{percent_seats_diff:.1%}",
        '\nARPU Difference': f"${arpu_diff:,.0f}",
        'Percent ARPU Difference': f"{percent_arpu_diff:.1%}",
        '\nRecommendation': recommendation
    }


    return results


# Example usage
impact_results = calculate_impact(new_data)
for key, value in impact_results.items():
    print(f"{key}: {value}")



#%%

def find_risks(data):
    import pandas as pd

    # Filter for rows with price increases
    price_increase_data = data[data['p2'] > data['p1']]

    # Calculate percentage changes
    price_increase_data['price_change_pct'] = 100 * (price_increase_data['p2'] - price_increase_data['p1']) / price_increase_data['p1']
    price_increase_data['volume_change_pct'] = 100 * (price_increase_data['v2'] - price_increase_data['v1']) / price_increase_data['v1']
    price_increase_data['revenue_change_pct'] = 100 * (price_increase_data['r2'] - price_increase_data['r1']) / price_increase_data['r1']

    # Sort by price change percentage in descending order
    sorted_data = price_increase_data.sort_values(by='price_change_pct', ascending=False)

    # Format the results
    formatted_results = sorted_data.apply(
        lambda row: f"${row['p1']} -> ${row['p2']} (+{row['price_change_pct']:.1f}%): ({row['volume_change_pct']:.1f}% volume, {row['revenue_change_pct']:.1f}% revenue)", 
        axis=1
    )

    return formatted_results

# Example usage
risk_list = find_risks(new_data)
print(risk_list)

#%%


def save_excel(data, filename='new_data.xlsx'):
    import pandas as pd

    # Save the DataFrame to an Excel file
    data.to_excel(filename, index=False)

# Example usage
new_data = calculate_new_revenue(data, 'p1', 'v1', 'p2', weibull_params, 'product')
save_excel(new_data, 'new_revenue_data.xlsx')

