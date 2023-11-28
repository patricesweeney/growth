#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:10:21 2023

@author: patricksweeney
"""
def import_data_local():
    import pandas as pd
    
    file_path = '/Users/patricksweeney/growth/03_Product/02_Effective network/0901-0909.csv'
    data = pd.read_csv(file_path)
    return data

data = import_data_local()


def fix_time(data, time_variable):
    import pandas as pd
    
    # Convert the time variable to datetime, allowing pandas to infer the format
    # and coerce any errors, which will turn unparseable strings into NaT
    data[time_variable] = pd.to_datetime(data[time_variable], errors='coerce', utc=True)
    
    # Round the datetime to the nearest second
    data[time_variable] = data[time_variable].dt.round('1s')
    
    # Drop rows with NaT in the time_variable column if any
    data = data.dropna(subset=[time_variable])
    
    # Infer the granularity by checking the smallest time delta
    time_deltas = data[time_variable].sort_values().diff().dropna()
    min_delta = time_deltas.min()
    
    # Determine granularity
    granularity = 'seconds'  # Default to seconds
    if min_delta >= pd.Timedelta(1, 'D'):
        granularity = 'days'
    elif min_delta >= pd.Timedelta(1, 'H'):
        granularity = 'hours'
    elif min_delta >= pd.Timedelta(1, 'T'):
        granularity = 'minutes'
    
    # The datetime objects are already in a nice format as pandas Timestamps
    # If you need to format them as strings in a specific format, you can do so:
    # data[time_variable] = data[time_variable].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return data, granularity

# Usage
data, granularity = fix_time(data, 'timestamp')


def daily_counts(data, time_variable, id_variable, event_variable):
    import pandas as pd
    
    # Convert the event variable to datetime without timezone information
    data[time_variable] = pd.to_datetime(data[time_variable]).dt.tz_localize(None)
    
    # Extract date from time_variable
    data['date'] = data[time_variable].dt.date
    
    # Group by id_variable, date, and event_variable, then count the occurrences
    group_vars = ['date', event_variable]
    if id_variable:
        group_vars.insert(0, id_variable)
    
    daily_data = data.groupby(group_vars).size().reset_index(name='count')
    
    return daily_data

# Usage
daily_data = daily_counts(data, 'timestamp', 'user_id', 'event')



def fix_time_series(data, time_variable, id_variable=None, event_variable=None):
    import pandas as pd
    import numpy as np
    from itertools import product
    
    # Convert the time variable to datetime without timezone information
    data[time_variable] = pd.to_datetime(data[time_variable]).dt.tz_localize(None)
    
    # Find the maximum time value
    max_time = data[time_variable].max()
    
    # Initialize the complete time series DataFrame
    complete_time_series = pd.DataFrame()
    
    if id_variable and event_variable:
        # Get unique IDs and events
        all_ids = data[id_variable].unique()
        all_events = data[event_variable].unique()
        
        # Create a date range for the time series
        all_times = pd.date_range(start=data[time_variable].min(), end=max_time, freq='D')
        
        # Create a product of all combinations
        all_combinations = list(product(all_ids, all_events, all_times))
        
        # Create a DataFrame from all combinations
        complete_time_series = pd.DataFrame(all_combinations, columns=[id_variable, event_variable, time_variable])
        
        # Merge to create a complete time series with all events
        merged_data = pd.merge(complete_time_series, data, on=[id_variable, event_variable, time_variable], how='left')
        
    elif id_variable:
        # Group by id_variable and get the min time for each group
        first_obs = data.groupby(id_variable)[time_variable].min().reset_index()
        all_ids = first_obs[id_variable].unique()
        
        # Create a date range for the time series
        all_times = pd.date_range(start=first_obs[time_variable].min(), end=max_time, freq='D')
        complete_time_series = pd.DataFrame(list(product(all_ids, all_times)), columns=[id_variable, time_variable])
        
        # Merge to create a complete time series
        merged_data = pd.merge(complete_time_series, data, on=[id_variable, time_variable], how='left')
        
    else:
        # Create a date range for the time series
        all_times = pd.date_range(start=data[time_variable].min(), end=max_time, freq='D')
        complete_time_series = pd.DataFrame({time_variable: all_times})
        
        # Merge to create a complete time series
        merged_data = pd.merge(complete_time_series, data, on=time_variable, how='left')
    
    # Fill the missing values
    for column in merged_data.columns:
        if column not in [time_variable, id_variable, event_variable]:
            dtype = merged_data[column].dtype
            # Check if the column is numeric
            if np.issubdtype(dtype, np.number):
                # For numeric variables
                merged_data[column].fillna(0, inplace=True)
            else:
                # For non-numeric and non-time variables
                fill_value = merged_data[column].mode()[0] if not merged_data[column].mode().empty else 'Unknown'
                merged_data[column].fillna(fill_value, inplace=True)
    
    return merged_data

# Usage
fixed_data = fix_time_series(daily_data, 'date', 'user_id', 'event')


#%% Impute missing


def calculate_sparsity(data):
    import pandas as pd
    import numpy as np
    
    # If data is a DataFrame
    if isinstance(data, pd.DataFrame):
        total_elements = data.size
        zero_elements = (data == 0).sum().sum()
    # If data is a NumPy Array
    elif isinstance(data, np.ndarray):
        total_elements = data.size
        zero_elements = np.count_nonzero(data == 0)
    else:
        return "Invalid data type. Please provide a Pandas DataFrame or a NumPy Array."
    
    sparsity = zero_elements / total_elements
    return sparsity

calculate_sparsity(fixed_data)


def find_missing_values(data):
    missing_values = data.isnull().sum()
    print("Features with missing values are...")
    print(missing_values)
    
find_missing_values(fixed_data)

    
def impute_missing_values(data, impute_value):
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    
    # Drop columns with all missing values
    data.dropna(axis=1, how='all', inplace=True)
    
    # Replace infinities with NaNs
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Create an imputer instance based on the impute_value argument
    if impute_value == 'median':
        imputer = SimpleImputer(strategy='median')
    elif impute_value == 'zero':
        imputer = SimpleImputer(strategy='constant', fill_value=0)
    else:
        raise ValueError("Invalid impute_value. Choose either 'median' or 'zero'")
    
    # Extract numerical columns
    data_num = data.select_dtypes(include=[np.number])
    
    # Fit the imputer to the numerical data
    imputer.fit(data_num)
    
    # Transform the numerical data with the imputer
    data_num_imputed = imputer.transform(data_num)
    
    # Convert the imputed data back to a DataFrame
    data_num_imputed_df = pd.DataFrame(data_num_imputed, columns=data_num.columns)
    
    # Replace the original numerical columns with the imputed ones
    data[data_num.columns] = data_num_imputed_df
    
    return data




data = impute_missing_values(data, impute_value = 'zero')
calculate_sparsity(data)
find_missing_values(data)



#%% Reshape

def long_to_wide(data, event_column, date_column, id_column, numeric_column):
    import pandas as pd
    
    # Filter the data for the specific id
    specific_id = '40a0dc7c-1e1c-4df2-a284-fdac9a861fca'
    data_filtered = data[data[id_column] == specific_id]
    
    # Pivot the table to wide format with the numeric column as values
    data_wide = data_filtered.pivot(index=date_column, columns=event_column, values=numeric_column)
    
    # Reset the index to make sure the date_column becomes a column again
    data_wide.reset_index(inplace=True)
    
    # Rename the columns to remove the original event column name from the multi-index
    data_wide.columns.name = None
    
    return data_wide

# Usage
# Replace 'numeric_column_name' with the actual name of your numeric column
wide_data = long_to_wide(fixed_data, 'event', 'date', 'user_id', 'count')



#%% Effective network


from jpype import *
jarLocation = "/Users/patricksweeney/JIDT/infodynamics.jar"
startJVM(getDefaultJVMPath(), "-ea", "-Xmx4024M", "-Djava.class.path=" + jarLocation, convertStrings=True)



def effective_network(data, date_col):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    from idtxl.multivariate_te import MultivariateTE
    from idtxl.data import Data
    from idtxl.visualise_graph import plot_network
    # Remove columns with all NaN values
    data.dropna(axis=1, how='all', inplace=True)
    
    # Remove rows with any NaN values
    data.dropna(axis=0, inplace=True)
    
    # Sort data by date column
    data.sort_values(by=date_col, inplace=True)
    
    # Drop the date column for further analysis
    data.drop(columns=[date_col], inplace=True)
    
    # Keep only numeric columns that vary
    var_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col]) and data[col].var() != 0]
    data = data[var_cols]

    if data.empty:
        print("Data is empty after preprocessing.")
        return

    # Generate node to variable mapping
    node_mapping = {i: var for i, var in enumerate(var_cols)}
    print("Node to variable mapping:", node_mapping)

    # Convert to numpy array and reshape for IDTxl
    np_data = np.array(data).T.reshape(data.shape[1], data.shape[0], 1)
    
    # Convert to IDTxl Data object
    idtxl_data = Data(np_data, dim_order='psr')
    
    # Initialize MultivariateTE and set settings
    network_analysis = MultivariateTE()
    settings = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1}

    # Run analysis
    results = network_analysis.analyse_network(settings=settings, data=idtxl_data)

    # Create IDTxl plot
    results.print_edge_list(weights='max_te_lag', fdr=False)
    plot_network(results=results, weights='max_te_lag', fdr=False)
    plt.show()

    # Create a NetworkX graph
    G = nx.DiGraph()
    
    # Fill the graph based on results
    for target in results.targets_analysed:
        sources = results.get_target_sources(target=target, fdr=False)
        
        for source in sources:
            G.add_edge(node_mapping[source], node_mapping[target])
    
    # Create a PGN (Probabilistic Graphical Network)
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos, node_color='lightblue', node_size=1700, font_size=12,
                    arrowsize=20, arrowstyle='->,head_width=0.6,head_length=1', edgecolors='k', linewidths=1)
    plt.title("Effective Network (Transfer Entropy)")
    plt.show()

     # Print the edge list with variable names
    for target in results.targets_analysed:
        sources = results.get_target_sources(target=target, fdr=False)
        
        for source in sources:
            print(f"{node_mapping[source]} -> {node_mapping[target]}")
    
    return results, node_mapping

# Example usage
results, node_mapping = effective_network(wide_data, 'date')

