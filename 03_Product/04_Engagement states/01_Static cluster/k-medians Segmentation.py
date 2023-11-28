"""
Created on Tue Sep 12 19:34:20 2023

@author: patricksweeney
"""



#%% Import

import pandas as pd
import numpy as np

def import_user_data_local():
    file_path = '/Users/patricksweeney/growth/03_Product/04_Engagement states/01_Static cluster/User data.xlsx'
    data = pd.read_excel(file_path)
    return data

data = import_user_data_local()

# Remove the 'shared_object_count' variable
if 'shared_object_count' in data.columns:
    data.drop('shared_object_count', axis=1, inplace=True)

# Filter out rows where 'role' is equal to 'VIEWER'
data = data[data['role'] != 'VIEWER']





def import_workspace_data_local():
    file_path = '/Users/patricksweeney/growth/03_Product/04_Engagement states/01_Static cluster/Workspace data.xlsx'
    data = pd.read_excel(file_path)
    return data

workspace_data = import_workspace_data_local()

# # Filter out rows where 'role' is equal to 'VIEWER'
data = data[data['role'] != 'VIEWER']



def calculate_sparsity(data):
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

calculate_sparsity(data)


def join(left, right, on):
    import pandas as pd
    
    # Perform left join
    joined_data = pd.merge(left, right, how='left', on=on)

    
    return joined_data

data = join(data, workspace_data, 'workspace_id')


#%%Clean
def find_missing_values(data):
    missing_values = data.isnull().sum()
    print("Features with missing values are...")
    print(missing_values)
    
    
find_missing_values(data)



def impute_missing_values(data):
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    
    # Create a copy to avoid in-place modifications
    data_copy = data.copy()
    
    # Drop columns with all missing values
    data_copy.dropna(axis=1, how='all', inplace=True)
    
    # Replace infinities with NaNs
    data_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Extract numerical columns
    data_num = data_copy.select_dtypes(include=[np.number])
    
    # Create an imputer instance with median strategy for numerical columns
    imputer = SimpleImputer(strategy='median')
    
    # Fit and transform the numerical data
    imputer.fit(data_num)
    data_num_imputed = imputer.transform(data_num)
    
    # Convert the imputed data back to a DataFrame
    data_num_imputed_df = pd.DataFrame(data_num_imputed, columns=data_num.columns, index=data_num.index)
    
    # Replace only the NaNs in the numerical columns with the imputed values
    for col in data_num.columns:
        data_copy.loc[data_num[col].isna(), col] = data_num_imputed_df.loc[data_num[col].isna(), col]
        
    # Handle categorical or string columns
    data_cat = data_copy.select_dtypes(include=['object'])
    data_copy[data_cat.columns] = data_cat.fillna('Unknown')
    
    return data_copy


# Example usage (replace 'data' with your DataFrame)
# imputed_data = impute_missing_values(data)

find_missing_values(data)

data = impute_missing_values(data)

find_missing_values(data)

calculate_sparsity(data)



def eda_fivenum(data):
    import pandas as pd
    
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=['number'])
    
    # Get the summary statistics including mean
    summary = numeric_data.describe(percentiles=[.25, .5, .75]).loc[['min', '25%', '50%', 'mean', '75%', 'max']]
    
    # Add the standard deviation as a new row
    summary.loc['std'] = numeric_data.std()
    
    return summary

summary_original = eda_fivenum(data)



#%% Feature engineering
def usage_lambdas(data):
    import pandas as pd
    
    # Identify columns that end with '_count'
    count_cols = [col for col in data.columns if col.endswith('_count')]
    
    # Check if 'duration' column exists
    if 'duration' not in data.columns:
        return "The dataset does not have a 'duration' column."
    
    # Calculate the rates by dividing each '_count' by 'duration'
    for col in count_cols:
        rate_col_name = col.replace('_count', '_rate')
        data[rate_col_name] = data[col] / data['duration']
    
    # Drop the original '_count' columns
    data.drop(columns=count_cols, inplace=True)
    
    return data


data = usage_lambdas(data)
find_missing_values(data)
data = impute_missing_values(data)
summary_lambdas = eda_fivenum(data)


#%% Ising shit
def decimate(data):
    import pandas as pd
    
    # Identify columns that end with '_rate'
    count_cols = [col for col in data.columns if col.endswith('_rate')]
    
    # Create new DataFrame to store the binary columns
    new_data = data.copy()
    
    # Store the average of each new binary column
    avg_values = {}
    
    for col in count_cols:
        # Calculate the mean of the column
        median_value = data[col].median()
        
        # Create new binary column with values set to 1 if above the mean, else 0
        new_col_name = col.replace('_rate', '_binary')
        new_data[new_col_name] = (data[col] > median_value).astype(int)
        
        # Store the average of the new binary column
        avg_values[new_col_name] = new_data[new_col_name].mean() * 100
        
        # Drop the old '_rate' column
        new_data.drop(columns=[col], inplace=True)
    
    # Count the number of new binary columns
    num_binary_cols = len(avg_values)
    
    # Calculate the number of potential microstates
    num_microstates = 2 ** num_binary_cols
    
    # Print the average of each new binary column in percentage with no decimal places
    for col, avg in avg_values.items():
        print(f"The average of {col} is {int(avg)}%")
    
    print() 
    # Print the number of binary columns and potential microstates
    print(f"You have {num_binary_cols} variables with 2 states each. Therefore, the number of potential microstates are 2^{num_binary_cols} = {num_microstates}.")
    
    return new_data

    
decimated_data = decimate(data)




find_missing_values(decimated_data)

data = impute_missing_values(decimated_data)



def join(left, right, on):
    import pandas as pd
    
    # Perform left join
    joined_data = pd.merge(left, right, how='left', on=on)

    
    return joined_data

joined_decimated_data = join(decimated_data, workspace_data, 'workspace_id')



from scipy.stats import entropy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def total_correlation(group, binary_cols):
    import numpy as np
    from scipy.stats import entropy
    
    # Filter and transpose data
    group = group[binary_cols]
    transposed_group = group.transpose()
    
    # Joint distribution
    joint_distribution = transposed_group.apply(lambda x: tuple(x), axis=1).value_counts(normalize=True)
    joint_probs = joint_distribution.values
    joint_entropy = -np.sum(joint_probs * np.log2(joint_probs))
    
    # Marginal distributions and entropy
    marginal_entropies = []
    for col in binary_cols:
        marginal_distribution = group[col].value_counts(normalize=True)
        marginal_entropies.append(entropy(marginal_distribution, base=2))
    marginal_entropy_sum = np.sum(marginal_entropies)
    
    # Total correlation
    tc = marginal_entropy_sum - joint_entropy
    
    return tc, joint_entropy, marginal_entropy_sum


import seaborn as sns
import matplotlib.pyplot as plt

# Set font to Helvetica
plt.rcParams['font.family'] = 'Helvetica'

def ising_summary_with_total_correlation_and_plots(data):
    import pandas as pd
    import numpy as np
    
    binary_cols = [col for col in data.columns if data[col].dtype == 'bool']
    grouped = data.groupby(['workspace_id', 'stripe_customer_name_x', 'mrr_latest'])
    
    summary_list = []

    for name, group in grouped:
        workspace_id, stripe_customer_name, mrr_latest = name
        n_rows = len(group)
        magnetization = group[binary_cols].mean().mean()
        tc, joint_entropy_sum, marginal_entropy_sum = total_correlation(group, binary_cols)
        
        summary_list.append({
            'workspace_id': workspace_id,
            'stripe_customer_name': stripe_customer_name,
            'mrr_latest': mrr_latest,
            'magnetization': magnetization,
            'total_correlation': tc,
            'joint_entropy_sum': joint_entropy_sum,
            'marginal_entropy_sum': marginal_entropy_sum,
            'n_rows': n_rows
        })
        
    summary_df = pd.DataFrame(summary_list)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # All plots involving mrr_latest now log-scaled
    for ax in axes:
        ax.set_xscale('log')
    
    sns.scatterplot(x='mrr_latest', y='total_correlation', data=summary_df, ax=axes[0], palette='RdBu_r')
    axes[0].set_xlabel('mrr_latest')
    axes[0].set_ylabel('Total Correlation')
    
    sns.scatterplot(x='mrr_latest', y='magnetization', data=summary_df, ax=axes[1], palette='RdBu_r')
    axes[1].set_xlabel('mrr_latest')
    axes[1].set_ylabel('Magnetization')
    
    sns.scatterplot(x='magnetization', y='total_correlation', data=summary_df, ax=axes[2], palette='RdBu_r')
    axes[2].set_xlabel('Magnetization')
    axes[2].set_ylabel('Total Correlation')
    
    plt.show()
    
    return summary_df

# Assuming joined_decimated_data is already defined
ising = ising_summary_with_total_correlation_and_plots(joined_decimated_data)








def sequence_plot(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Filter for specific workspace_id and sample 100 random user_ids
    filtered_data = data[data['workspace_id'] == 'ba9d0519-6ac7-4f8a-99b1-1576e927df26'].sample(500, random_state=42)
    
    # Filter columns that end with '_binary'
    binary_data = filtered_data.filter(like='_binary')
    
    # Create two-letter feature names
    two_letter_names = {col: col[:2].title() for col in binary_data.columns}
    renamed_data = binary_data.rename(columns=two_letter_names)
    
    # Create heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(renamed_data, cmap=["blue", "red"], cbar=False)
    
    plt.xlabel('Features')
    plt.ylabel('User IDs')
    plt.title('Feature Sequence Heatmap')
    plt.show()


sequence_plot(decimated_data)

#%% Distribution (log log plots)

def adoption(data, class_var=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter
    
    def percentage_formatter(x, pos):
        return f'{x:.0f}%'
        
    def format_label(label):
        return label.replace('_', ' ').title().replace(' Rate', '')
    
    class_data = None
    if class_var and class_var in data.columns:
        class_data = data[class_var].copy()
    
    filtered_data = data[data['role'] != 'VIEWER']
    rate_cols = [col for col in filtered_data.columns if col.endswith('_rate')]
    filtered_data = filtered_data[rate_cols]
    filtered_data = filtered_data.select_dtypes(include=['number']).loc[:, filtered_data.nunique() > 1]
    
    if class_data is not None:
        filtered_data[class_var] = class_data
    
    # Exclude 'Unknown' from class_var
    if class_var:
        filtered_data = filtered_data[filtered_data[class_var] != 'Unknown']
    
    plt.rcParams['font.family'] = 'Helvetica'
    
    if class_var and class_var in filtered_data.columns:
        classes = filtered_data[class_var].unique()
        n_classes = len(classes)
        n_cols = 2
        n_rows = int(np.ceil(n_classes / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 10), squeeze=False)
        axes = axes.flatten()
        
        for i, cls in enumerate(classes):
            ax = axes[i]
            group_data = filtered_data[filtered_data[class_var] == cls]
            numeric_group_data = group_data.select_dtypes(include=['number'])
            
            percentages = (numeric_group_data > 0).mean() * 100
            percentages.index = percentages.index.map(format_label)
            percentages = percentages.sort_values(ascending=False).drop(class_var, errors='ignore')
            
            sns.barplot(x=percentages.values, y=percentages.index, palette="RdBu_r", ax=ax)
            ax.set_xlabel('Adoption Percentage')
            ax.set_ylabel('Feature')
            ax.set_title(f'Feature Adoption Percentage for {cls}')
            ax.xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        
        for i in range(n_classes, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
    else:
        percentages = (filtered_data > 0).mean() * 100
        percentages.index = percentages.index.map(format_label)
        percentages = percentages.sort_values(ascending=False)
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=percentages.values, y=percentages.index, palette="RdBu_r")
        plt.xlabel('Adoption Percentage')
        plt.ylabel('Feature')
        plt.title('Feature Adoption Percentage')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        plt.show()


# Running the function again with the sample data
adoption(data, class_var='products_segment')


def feature_based_adoption(data, class_var=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.ticker import FuncFormatter
    
    def percentage_formatter(x, pos):
        return f'{x:.0f}%'
        
    def format_label(label):
        return label.replace('_', ' ').title().replace(' Rate', '')
    
    class_data = None
    if class_var and class_var in data.columns:
        class_data = data[class_var].copy()
    
    filtered_data = data[data['role'] != 'VIEWER']
    rate_cols = [col for col in filtered_data.columns if col.endswith('_rate')]
    filtered_data = filtered_data[rate_cols]
    filtered_data = filtered_data.select_dtypes(include=['number']).loc[:, filtered_data.nunique() > 1]
    
    if class_data is not None:
        filtered_data[class_var] = class_data
    
    # Exclude 'Unknown' from class_var
    if class_var:
        filtered_data = filtered_data[filtered_data[class_var] != 'Unknown']
    
    # Remove classes with n < 50
    class_counts = filtered_data[class_var].value_counts()
    valid_classes = class_counts[class_counts >= 50].index.tolist()
    filtered_data = filtered_data[filtered_data[class_var].isin(valid_classes)]
    
    plt.rcParams['font.family'] = 'Helvetica'
    
    if class_var and class_var in filtered_data.columns:
        features = rate_cols
        n_features = len(features)
        n_cols = 2
        n_rows = int(np.ceil(n_features / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 10), squeeze=False)
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i]
            # Calculate the proportion of adoption where value > 0 for each class category
            adoption_proportion = filtered_data.groupby(class_var)[feature].apply(lambda x: (x > 0).mean() * 100).reset_index()
            adoption_proportion = adoption_proportion.sort_values(by=feature, ascending=False)
            sns.barplot(data=adoption_proportion, x=feature, y=class_var, palette="RdBu_r", ax=ax)
            ax.set_xlabel('Adoption Percentage')
            ax.set_ylabel('Class Category')
            ax.set_title(f'Feature Adoption Percentage for {format_label(feature)}')
            ax.xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
            
            # Remove the lines across bars
            sns.despine(left=True, bottom=True)
        
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
    else:
        print("No class_var specified or not found in data columns.")




feature_based_adoption(data, class_var='company_vertical')



def loglog_plot(data, class_var=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import FuncFormatter

    def real_numbers(x, pos):
        return str(int(10 ** x))

    # Drop constant and non-numeric columns
    filtered_data = data.select_dtypes(include=['number']).loc[:, data.nunique() > 1]
    
    if class_var and class_var in data.columns:
        filtered_data[class_var] = data[class_var]

    n_cols = 4  # Adjust as needed
    n = len(filtered_data.columns) - int(class_var in filtered_data.columns)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
    axes = axes.flatten()
    
    for i, col in enumerate(filtered_data.drop(columns=[class_var], errors='ignore').columns):
        ax = axes[i]
        non_zero_data = filtered_data.loc[filtered_data[col] > 0, [col, class_var]]

        # Scale the y-variable by 7
        scaled_y = np.log10(1 + 7 * np.sort(non_zero_data[col])[::-1])
        
        sns.scatterplot(x=np.log10(1 + np.arange(len(non_zero_data))), 
                        y=scaled_y, 
                        hue=non_zero_data[class_var] if class_var else None,
                        ax=ax, edgecolor='none', palette="RdBu_r", legend='full')

        mean_value = np.log10(1 + 7 * non_zero_data[col].mean())
        median_value = np.log10(1 + 7 * np.median(non_zero_data[col]))

        ax.axhline(mean_value, color='red', linestyle='-', linewidth=1)
        ax.axhline(median_value, color='red', linestyle='--', linewidth=1)

        ax.set_title(col)
        ax.set_xlabel('User Rank')
        ax.set_ylabel('Weekly Usage Rate')

        ax.xaxis.set_major_locator(plt.FixedLocator(np.log10([1, 10, 100, 1000, 10000, 100000])))
        ax.xaxis.set_major_formatter(FuncFormatter(real_numbers))
        ax.yaxis.set_major_locator(plt.FixedLocator(np.log10([1, 10, 100, 1000, 10000, 100000])))
        ax.yaxis.set_major_formatter(FuncFormatter(real_numbers))

    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# Call the function
loglog_plot(data, class_var='products_latest')


def loglog_plot_with_kde(data, class_var=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    def real_numbers(x, pos):
        return str(int(10 ** x))

    filtered_data = data.select_dtypes(include=['number']).loc[:, data.nunique() > 4]
    filtered_data = filtered_data.drop(columns=['mrr_latest', 'seats_latest', 'duration'], errors='ignore')

    if class_var and class_var in data.columns:
        filtered_data[class_var] = data[class_var]
        filtered_data = filtered_data[filtered_data[class_var] != 'Unknown']

    n_cols = 4
    n = len(filtered_data.columns) - int(class_var in filtered_data.columns)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
    axes = axes.flatten()

    handles, labels = None, None  # Initialize variables to hold legend info

    for i, col in enumerate(filtered_data.drop(columns=[class_var], errors='ignore').columns):
        ax = axes[i]
        non_zero_data = filtered_data.loc[filtered_data[col] > 0, [col, class_var]]

        if non_zero_data[col].var() == 0:
            ax.set_title(f"{col} (Zero variance)")
            continue

        scaled_y = np.log10(1 + 7 * non_zero_data[col])
        sns.kdeplot(x=scaled_y, hue=non_zero_data[class_var] if class_var else None, ax=ax, palette="RdBu_r", common_norm=False, legend=False)

        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel('Weekly Usage Rate')
        ax.set_ylabel('Density')
        ax.xaxis.set_major_locator(plt.FixedLocator(np.log10([1, 10, 100, 1000, 10000, 100000])))
        ax.xaxis.set_major_formatter(FuncFormatter(real_numbers))

        # Capture legend handles and labels from the first subplot that has them
        if class_var and handles is None:
            handles, labels = ax.get_legend_handles_labels()

    # Remove extra axes
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    # Create a single legend for the entire figure
    if class_var and handles:
        fig.legend(handles[1:], labels[1:], title='Class', bbox_to_anchor=(1.05, 0.5), loc='center left')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


loglog_plot_with_kde(data, class_var='company_vertical')


def calculate_and_plot_gini(data):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Set font to Helvetica
    plt.rcParams['font.family'] = 'Helvetica'
    
    def gini(array):
        """Calculate the Gini coefficient of a numpy array."""
        array = array.astype(float).flatten()
        if np.amin(array) < 0:
            array -= np.amin(array)
        array += 0.0000001
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))
    
    # Drop constant and non-numeric columns
    filtered_data = data.select_dtypes(include=['number']).loc[:, data.nunique() > 1]
    
    # Remove 'duration' if it exists
    if 'duration' in filtered_data.columns:
        filtered_data.drop(columns=['duration'], inplace=True, errors='ignore')
    
    # Dictionary to store Gini coefficients
    gini_coefficients = {}
    
    # Calculate Gini coefficient for each numeric column
    for column in filtered_data.columns:
        gini_coefficients[column] = gini(filtered_data[column].values)
    
    # Sort the Gini coefficients in descending order
    sorted_result = {k: v for k, v in sorted(gini_coefficients.items(), key=lambda item: item[1], reverse=True)}
    
    # Replace underscores with spaces and use title case
    formatted_labels = [label.replace('_', ' ').title() for label in sorted_result.keys()]
    
    # Plot
    plt.figure(figsize=(10, 5))
    bars = plt.barh(formatted_labels, list(sorted_result.values()), color='darkred')

    # Colormap
    cm = plt.cm.get_cmap('Reds_r')
    for i, bar in enumerate(bars):
        bar.set_color(cm(i / len(sorted_result)))

    plt.xlabel('Gini Coefficient')
    plt.ylabel('Feature')
    plt.title('Gini Coefficients For Each Variable')
    plt.gca().invert_yaxis()

    # Font and other formatting
    plt.rcParams['font.family'] = 'Helvetica'
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Gini Coefficients For Each Feature', fontsize=12)

    plt.show()



calculate_and_plot_gini(data)

#%% Preprocessing

def log_transform(data, leave):
    import pandas as pd
    import numpy as np

    # Make a copy of the original data
    data_original = data.copy()

    # Identify numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove columns that shouldn't be transformed from the list
    transform_cols = [col for col in numeric_cols if col not in leave]
    
    # Apply log(x+1) transformation
    data[transform_cols] = data[transform_cols].apply(lambda x: np.log1p(x))
    
    return data, data_original



data, data_original = log_transform(data, ['seats_latest', 'viewers_latest', 'duration', 'mrr_latest'] )
find_missing_values(data)
data = impute_missing_values(data)
summary_seats_log = eda_fivenum(data)




# def boxcox_transform(data, leave):
#     from scipy.stats import boxcox
#     import pandas as pd
#     import numpy as np

#     # Make a copy of the original data
#     data_original = data.copy()

#     # Identify numeric columns
#     numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

#     # Remove columns that shouldn't be transformed from the list
#     transform_cols = [col for col in numeric_cols if col not in leave]

#     # Apply Box-Cox transformation
#     for col in transform_cols:
#         # Adding 1 to handle zero values as boxcox requires strictly positive input
#         data[col], _ = boxcox(data[col] + 1)
    
#     return data, data_original

# # Example usage
# data, data_original = boxcox_transform(data, ['seats_latest', 'viewers_latest', 'duration', 'mrr_latest'])
# find_missing_values(data)
# data = impute_missing_values(data)
# summary_seats_boxcox = eda_fivenum(data)




def standardization(data, leave=[]):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Identify numerical columns to standardize, excluding those in 'leave'
    numerical_cols = [col for col in data.select_dtypes(include=['float64', 'int64']).columns 
                      if data[col].nunique() > 2 and col not in leave]

    # Standardize the selected numerical columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    return data


data = standardization(data, leave = ['duration', 'seats_latest', 'viewers_latest', 'mrr_latest'])
find_missing_values(data)
data = impute_missing_values(data)
summary_seats_log_standardized = eda_fivenum(data)


#%% EDA


def eda_pair_plot(data):
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Set font to Helvetica
    plt.rcParams['font.family'] = 'Helvetica'
    
    # Replace infinite values with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[float, int])
    
    # Rename columns to 'Title Case' without underscores
    renamed_columns = {col: col.replace('_', ' ').title() for col in numeric_data.columns}
    numeric_data.rename(columns=renamed_columns, inplace=True)
    
    # Generate pairplot with different colors and linear regression line
    g = sns.pairplot(numeric_data, kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.02}})
    
    # Update plot labels to 'Title Case'
    for ax in plt.gcf().axes:
        ax.set_xlabel(ax.get_xlabel().replace('_', ' ').title())
        ax.set_ylabel(ax.get_ylabel().replace('_', ' ').title())
        
    # Add a title to the plot
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Distributions and Relationship Between Feature Usage', fontsize=16)
    
    plt.show()


eda_pair_plot(data)



def plot_elasticities(data):
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Preprocess the data
    plt.rcParams['font.family'] = 'Helvetica'
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    if 'duration' in data.columns:
        data.drop('duration', axis=1, inplace=True)
    numeric_data = data.select_dtypes(include=[float, int])
    renamed_columns = {col: col.replace('_', ' ').title() for col in numeric_data.columns}
    numeric_data.rename(columns=renamed_columns, inplace=True)

    # Prepare for plotting scatterplots
    num_vars = len(numeric_data.columns)
    num_plots = num_vars * (num_vars - 1)
    num_rows = -(-num_plots // 4)  # Ceiling division
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))
    axes = axes.flatten()
    plot_idx = 0
    
    # Initialize list to store elasticity info
    elasticity_list = []

    # Loop through each pair of features, including reverse pairs
    for i, col1 in enumerate(numeric_data.columns):
        for j, col2 in enumerate(numeric_data.columns):
            if i == j:
                continue
            
            # Prepare data and fit the model with robust standard errors
            X = add_constant(numeric_data[col1])
            y = numeric_data[col2]
            model = OLS(y, X).fit(cov_type='HC3')

            # Extract elasticity, p-value, and R-squared
            elasticity = model.params[1]
            p_value = model.pvalues[1]
            r_squared = model.rsquared

            # Plot scatter and regression line
            sns.regplot(x=col1, y=col2, data=numeric_data, ax=axes[plot_idx], line_kws={"color": "red"})
            axes[plot_idx].set_title(f"{col1} vs {col2}\nElasticity: {elasticity:.1f}, P-value: {p_value:.2f}, R^2: {r_squared:.2f}")

            # Store elasticity info if R-squared > 0.3 and p-value < 0.01
            if r_squared > 0.3 and p_value < 0.01:
                elasticity_list.append((f"{col1} <- {col2}", elasticity))
            
            plot_idx += 1

    # Remove unused subplots
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    

# Call the function with robust standard errors
plot_elasticities(data)


def pair_elasticities_matrix(data, color_map_upper_bound=1):
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'Helvetica'
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    if 'duration' in data.columns:
        data.drop('duration', axis=1, inplace=True)

    numeric_data = data.select_dtypes(include=[float, int])
    renamed_columns = {col: col.replace('_', ' ').title() for col in numeric_data.columns}
    numeric_data.rename(columns=renamed_columns, inplace=True)

    elasticity_matrix = pd.DataFrame(index=numeric_data.columns, columns=numeric_data.columns)
    rsquared_matrix = pd.DataFrame(index=numeric_data.columns, columns=numeric_data.columns)

    for col1 in numeric_data.columns:
        for col2 in numeric_data.columns:
            if col1 == col2:
                continue

            X, y = numeric_data[col2], numeric_data[col1]
            X = add_constant(X)
            model = OLS(y, X).fit(cov_type='HC3')
            
            if model.pvalues[1] < 0.01 and model.rsquared >= 0.0:
                elasticity_matrix.at[col1, col2] = model.params[1]
                rsquared_matrix.at[col1, col2] = model.rsquared
            else:
                elasticity_matrix.at[col1, col2] = np.nan
                rsquared_matrix.at[col1, col2] = np.nan

    sorted_columns = np.abs(elasticity_matrix).mean().sort_values(ascending=False).index
    sorted_elasticity_matrix = elasticity_matrix.loc[sorted_columns, sorted_columns]
    sorted_rsquared_matrix = rsquared_matrix.loc[sorted_columns, sorted_columns]

    plt.figure(figsize=(14, 10))
    sns.heatmap(sorted_rsquared_matrix.astype(float), annot=sorted_elasticity_matrix.astype(float), fmt=".1f", cmap="Blues",
                center=0.5, mask=sorted_elasticity_matrix.isna(), vmax=color_map_upper_bound)
    plt.title('Feature Elasticity Matrix', fontsize=16)
    plt.show()



# Call the modified function with robust standard errors
pair_elasticities_matrix(data)




def eda_correlation_matrix(data):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    # Set font to Helvetica
    plt.rcParams['font.family'] = 'Helvetica'
    
    # Drop constant and non-numeric columns
    filtered_data = data.select_dtypes(include=['number']).loc[:, data.nunique() > 1]
    
    # Rename columns to 'Title Case' without underscores
    renamed_columns = {col: col.replace('_', ' ').title() for col in filtered_data.columns}
    filtered_data.rename(columns=renamed_columns, inplace=True)
    
    # Compute the correlation matrix
    corr = filtered_data.corr()
    
    # Create a copy of the correlation matrix to sort it
    corr_copy = corr.copy()
    corr_copy['sum'] = corr_copy.abs().sum()
    sorted_columns = corr_copy.sort_values(by='sum', ascending=False).index
    
    # Fetch sorted columns and rows without 'sum'
    sorted_corr = corr.loc[sorted_columns, sorted_columns]
    
    # Set up the matplotlib figure for the heatmap
    plt.figure(figsize=(10, 8))
    
    # Draw the heatmap
    sns.heatmap(sorted_corr, cmap="RdBu", vmin=-1, vmax=1, annot=True, fmt=".1f")
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    # Show the heatmap
    plt.show()
    
    # Flatten the upper triangle of the sorted_corr matrix
    upper_triangle = np.triu(sorted_corr, k=1)
    upper_triangle_flat = upper_triangle.flatten()
    
    # Remove zeros and sort
    non_zero = upper_triangle_flat[upper_triangle_flat != 0]
    sorted_indices = np.argsort(np.abs(non_zero))[::-1]
    
    # Find the top 10 pairs
    top_10_pairs = [(np.unravel_index(sorted_indices[i], upper_triangle.shape), non_zero[sorted_indices[i]]) for i in range(min(10, len(sorted_indices)))]
    
    # Prepare data for the bar plot
    labels = [f"{sorted_columns[pair[0][0]]}, {sorted_columns[pair[0][1]]}" for pair in top_10_pairs]
    values = [pair[1] for pair in top_10_pairs]
    
    # Create bar plot with Helvetica font and RdBu color map
    plt.figure(figsize=(10, 8))
    sns.barplot(x=values, y=labels, palette="RdBu_r")
    plt.xlabel('Correlation')
    plt.title('Top 10 Correlated Feature Pairs')
    
    # Show the bar plot
    plt.show()

    # Standardize the filtered_data for variance check
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(filtered_data)
    
    # Create boxplot to check variance stability
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=pd.DataFrame(standardized_data, columns=filtered_data.columns))
    plt.xticks(rotation=90)
    plt.title('Variance Stability Check')
    
    # Show the boxplot
    plt.show()





eda_correlation_matrix(data)


# Updated pca function incorporating color and size based on feature's contribution to total variance in the scatter plot
# Final pca function with zoomed-in loading plot
# Updating pca function to remove grid lines and adjust axis zoom
# Updating pca function to set major gridlines at the middle of each axis range
def pca(data):
    from sklearn.decomposition import PCA
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Set font to Helvetica
    plt.rcParams['font.family'] = 'Helvetica'
    
    # Drop constant and non-numeric columns
    filtered_data = data.select_dtypes(include=['number']).loc[:, data.nunique() > 1]
    
    # Remove 'duration' if it exists
    filtered_data = filtered_data.drop(columns=['duration'], errors='ignore')
    
    # If no suitable columns, exit
    if filtered_data.shape[1] == 0:
        print("No suitable numeric columns found that vary. Exiting.")
        return
    
    # Rename columns to 'Title Case' without underscores
    renamed_columns = {col: col.replace('_', ' ').title() for col in filtered_data.columns}
    filtered_data.rename(columns=renamed_columns, inplace=True)

    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(filtered_data)
    
    # Explained Variance Bar Plot
    plt.figure(figsize=(10, 5))
    sns.barplot(x=np.arange(1, len(pca.explained_variance_ratio_)+1), y=pca.explained_variance_ratio_ * 100, palette="RdBu_r")
    plt.xlabel('Superfeatures')
    plt.ylabel('Explained Variance (%)')
    plt.title('Explained Variance By Each Superfeature')
    plt.show()
    
    # Scree Plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=np.arange(1, len(pca.explained_variance_)+1), y=pca.explained_variance_, marker='o', palette="RdBu_r")
    plt.xlabel('Superfeature')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot')
    plt.show()
    
    # Loading Plot as Dots with Contribution to Total Variance
    plt.figure(figsize=(10, 10))
    components = pd.DataFrame(pca.components_, columns=filtered_data.columns)
    contributions = components.iloc[0, :]**2 * pca.explained_variance_ratio_[0] + components.iloc[1, :]**2 * pca.explained_variance_ratio_[1]
    normalized_contributions = contributions / np.max(contributions)
    
    # Dynamic range for x and y axes, with a small padding to make sure dots are fully visible
    x_max, x_min = components.iloc[0, :].max() + 0.1, components.iloc[0, :].min() - 0.1
    y_max, y_min = components.iloc[1, :].max() + 0.1, components.iloc[1, :].min() - 0.1
    
    for i, (x, y, contrib) in enumerate(zip(components.iloc[0, :], components.iloc[1, :], normalized_contributions)):
        plt.scatter(x, y, s=contrib * 500, c=[plt.cm.RdBu_r(1 - contrib)])  # Size and color by normalized contribution, color inverted
        plt.annotate(filtered_data.columns[i], (x, y), textcoords="offset points", xytext=(0,10), ha='center', color='k')
    
    # Add lines to divide plot into half for each axis
    plt.axhline(y=(y_max + y_min) / 2, color='black', linewidth=0.8)
    plt.axvline(x=(x_max + x_min) / 2, color='black', linewidth=0.8)
        
    plt.xlim([x_min, x_max])  # Dynamic zoom-in with padding
    plt.ylim([y_min, y_max])  # Dynamic zoom-in with padding
    plt.xlabel('Superfeature 1 Loading')
    plt.ylabel('Superfeature 2 Loading')
    plt.title('Superfeature Loadings')
    plt.grid(False)  # Removing other grid lines
    plt.show()

    # Other plots and outputs would follow here
    
    # Compute the contributions to PC1, PC2, and total variance
    pc1_contributions = components.iloc[0, :] ** 2 * pca.explained_variance_ratio_[0]
    pc2_contributions = components.iloc[1, :] ** 2 * pca.explained_variance_ratio_[1]
    total_contributions = pc1_contributions + pc2_contributions

    # Contribution to PC1
    plt.figure(figsize=(15, 5))
    sns.barplot(x=filtered_data.columns, y=pc1_contributions, palette="RdBu_r", order=pc1_contributions.sort_values(ascending=False).index)
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Contribution to Superfeature 1 (%)')
    plt.title('Contribution to Superfeature 1')
    plt.show()
    
    # Contribution to PC2
    plt.figure(figsize=(15, 5))
    sns.barplot(x=filtered_data.columns, y=pc2_contributions, palette="RdBu_r", order=pc2_contributions.sort_values(ascending=False).index)
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Contribution to Superfeature 2 (%)')
    plt.title('Contribution to Superfeature 2')
    plt.show()
    
    # Contribution to Total Variance
    plt.figure(figsize=(15, 5))
    sns.barplot(x=filtered_data.columns, y=total_contributions, palette="RdBu_r", order=total_contributions.sort_values(ascending=False).index)
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Contribution to Total Variance (%)')
    plt.title('Contribution to Total Variance')
    plt.show()

# The loading plot is now divided into half for each axis by a single line, placed at the middle of each axis range.
# Other grid lines are removed and axis limits adjusted for better dot visibility.


pca(data)

#%% Factor analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis

# Function to format label
def format_label(label):
    return ' '.join([word.capitalize() for word in label.split('_')])

def factor_analysis(data):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import FactorAnalysis
    
    # Drop specified columns
    drop_columns = ['mrr_latest', 'transcription_hours_total', 'seats_latest', 'duration']
    filtered_data = data.drop(columns=drop_columns, errors='ignore')
    
    # Drop constant and non-numeric columns
    filtered_data = filtered_data.select_dtypes(include=['number']).loc[:, filtered_data.nunique() > 1]
    
    # Rename columns to 'Title Case' without underscores
    renamed_columns = {col: format_label(col) for col in filtered_data.columns}
    filtered_data.rename(columns=renamed_columns, inplace=True)
    
    # Apply Factor Analysis
    fa = FactorAnalysis(n_components=10, rotation = "varimax")
    fa_result = fa.fit_transform(filtered_data)
    
    # Loadings Matrix
    loadings = pd.DataFrame(fa.components_, columns=filtered_data.columns)
    
    # Scree Plot
    plt.figure(figsize=(10, 5))
    explained_variance = np.var(fa_result, axis=0)
    sns.lineplot(x=np.arange(1, len(explained_variance)+1), y=explained_variance, marker='o')
    plt.xlabel('Factor')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot')
    plt.show()
    
    # Heatmap of Loadings
    plt.figure(figsize=(20, 10))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt='.2f')
    plt.xlabel('Features')
    plt.ylabel('Factors')
    plt.title('Loadings Matrix')
    plt.show()

    # Contributions of Each Factor
    contributions = pd.DataFrame(fa_result, columns=[f'Factor_{i}' for i in range(fa_result.shape[1])]).apply(np.var)
    contributions = contributions.sort_values(ascending=False)
    
    # Bar Plot of Contributions
    plt.figure(figsize=(10, 5))
    sns.barplot(x=contributions.index, y=contributions.values, palette="RdBu_r")
    plt.xlabel('Factors')
    plt.ylabel('Contribution (Variance)')
    plt.title('Contributions of Each Factor')
    plt.show()
    
    return loadings, explained_variance, contributions


# Call the function
loadings, explained_variance, contributions = factor_analysis(data)



#%% Feature selection



def correlation_selector(data, corr_threshold):
    import numpy as np
    
    # Make a copy of the original data
    data_original = data.copy()
    
    # Filter numeric columns with variance
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_data = numeric_data.loc[:, numeric_data.var() > 0]
    
    # Apply correlation threshold
    corr_matrix = numeric_data.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]
    
    # Drop correlated columns from the original data
    if len(to_drop) > 0:
        data.drop(columns=to_drop, inplace=True)
    
    return data


data = correlation_selector(data, corr_threshold = 0.75)

#%% Model selection


def find_optimal_parameters_kmedoids(data, max_pca_components=10, max_clusters=10, do_scaling=True):
    from sklearn_extra.cluster import KMedoids
    from sklearn.metrics import silhouette_samples
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    """
    Finds the optimal number of Principal Components and clusters to maximize the medoid-based silhouette score.
    
    Parameters:
    - data: DataFrame, the data to cluster.
    - max_pca_components: int, the maximum number of Principal Components to consider.
    - max_clusters: int, the maximum number of clusters to consider.
    - do_scaling: bool, whether to scale the data or not.
    
    Returns:
    - best_n_pca: int, the optimal number of Principal Components.
    - best_n_clusters: int, the optimal number of clusters.
    - best_silhouette: float, the best silhouette score achieved.
    """
    
    # Filter only numeric and varying columns, excluding 'Duration'
    numeric_data = data.select_dtypes(include=[np.number]).loc[:, data.nunique() > 1]
    if 'Duration' in numeric_data.columns:
        numeric_data.drop('Duration', axis=1, inplace=True)
    
    if numeric_data.empty:
        print("No suitable columns for clustering.")
        return None, None, None
    
    # Data scaling
    if do_scaling:
        scaler = StandardScaler()
        numeric_data = scaler.fit_transform(numeric_data)
    
    best_n_pca = None
    best_n_clusters = None
    best_silhouette = -1  # Initialize with -1 as silhouette score ranges from -1 to 1
    
    for n_pca in range(1, max_pca_components + 1):
        pca = PCA(n_components=n_pca)
        reduced_data = pca.fit_transform(numeric_data)
        
        for n_clusters in range(2, max_clusters + 1):  # Starting from 2 as minimum number of clusters
            kmedoids = KMedoids(n_clusters=n_clusters, method='pam')
            kmedoids.fit(reduced_data)
            labels = kmedoids.labels_
            medoid_indices = kmedoids.medoid_indices_
            
            # Calculate medoid-based silhouette score
            silhouette_values = silhouette_samples(reduced_data, labels)
            medoid_silhouette_values = silhouette_values[medoid_indices]
            silhouette_avg = np.mean(medoid_silhouette_values)
            
            # Update best parameters if current silhouette score is greater
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_n_pca = n_pca
                best_n_clusters = n_clusters
    
    return best_n_pca, best_n_clusters, best_silhouette

best_n_pca, best_n_clusters, best_silhouette = find_optimal_parameters_kmedoids(data, max_pca_components=10, max_clusters=10)


#%% Train kmeans

def perform_clustering(data, pca, n_clusters, algorithm='KMeans', do_scaling=True, do_pca=True, plot=True, **kwargs):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, MeanShift, SpectralClustering, DBSCAN, Birch
    from sklearn_extra.cluster import KMedoids
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Importing this for 3D plotting
    from hdbscan import HDBSCAN
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    plt.rcParams['font.family'] = 'Helvetica'
    
    # Filter only numeric and varying columns, excluding 'Duration'
    numeric_data = data.select_dtypes(include=[np.number]).loc[:, data.nunique() > 1]
    if 'Duration' in numeric_data.columns:
        numeric_data.drop('Duration', axis=1, inplace=True)
    
    if numeric_data.empty:
        print("No suitable columns for clustering.")
        return None, None, None
    
    # Data scaling
    if do_scaling:
        scaler = StandardScaler()
        numeric_data = scaler.fit_transform(numeric_data)
    
    # Dimensionality reduction
    if do_pca:
        if isinstance(pca, int):
            pca_obj = PCA(n_components=pca)
        elif isinstance(pca, PCA):
            pca_obj = pca
        else:
            raise TypeError("Invalid type for pca. Must be an integer or an instance of sklearn.decomposition.PCA.")
        
        reduced_data = pca_obj.fit_transform(numeric_data)
    else:
        reduced_data = numeric_data
    
    # Algorithm selection
    algorithms = {
        'KMeans': KMeans,
        'KMedoids': KMedoids,
        'Agglomerative': AgglomerativeClustering,
        'AffinityPropagation': AffinityPropagation,
        'MeanShift': MeanShift,
        'SpectralClustering': SpectralClustering,
        'DBSCAN': DBSCAN,
        'HDBSCAN': HDBSCAN,
        'GaussianMixture': GaussianMixture,
        'Birch': Birch
    }
    
    if algorithm not in algorithms:
        raise ValueError("Invalid clustering algorithm")
    
    if algorithm == 'HDBSCAN':
        model = algorithms[algorithm](min_cluster_size=n_clusters, **kwargs)
    else:
        model = algorithms[algorithm](n_clusters=n_clusters, **kwargs)
    
    model.fit(reduced_data)
    
    labels = model.labels_ if hasattr(model, 'labels_') else model.predict(reduced_data)
    
    # Metrics
    if len(set(labels)) > 1:
        silhouette_avg = round(silhouette_score(reduced_data, labels), 2)
    else:
        silhouette_avg = "N/A"
    
    # Add cluster labels to data
    clustered_data = data.copy()
    clustered_data['Cluster label'] = labels
    
    # Print the number of points/rows in each segment
    cluster_counts = clustered_data['Cluster label'].value_counts()
    print("Number of points/rows in each segment:")
    print(cluster_counts)
    
    # Plotting
    if plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.8)
        plt.title(f'{algorithm} Clustering ({n_clusters} Segments)\nSilhouette Score: {silhouette_avg}')
        plt.xlabel('Superfeature 1')
        plt.ylabel('Superfeature 2')
        
        # Set custom axis limits
        plt.xlim([-5, 15])  # Set x-axis limits
        plt.ylim([-10, 20])  # Set y-axis limits

        plt.show()
    
    # Cluster summary
    cluster_summary = clustered_data.groupby('Cluster label').median(numeric_only=True)
    
    return clustered_data, silhouette_avg, cluster_summary

# Example usage for testing (replace 'data' with your DataFrame)
clustered_data, silhouette_avg, cluster_summary = perform_clustering(data=data, pca=10, n_clusters=3, algorithm='KMedoids', do_pca=True)


#%% Interpretation

def join_and_rename_clusters(left, right, on):
    import pandas as pd
    
    # Perform left join
    joined_data = pd.merge(left, right, how='left', on=on)

    
    return joined_data

joined_data = join_and_rename_clusters(data, workspace_data, 'workspace_id', cluster_column='Cluster label')
decimated_data = join_and_rename_clusters(decimated_data, workspace_data, 'workspace_id')


import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters_rotated(data, cluster_variable):
    
    # Configure plot style
    sns.set(style="white")
    plt.rcParams['font.family'] = 'Helvetica'
    
    # Create a figure and a 4x1 grid of subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 16))
    
    # Remove grid lines
    for ax in axes:
        ax.grid(False)
    
    # Plot 'mrr_latest' boxplots
    sns.boxplot(x=np.log10(data['mrr_latest']), y=cluster_variable, data=data, ax=axes[0], orient='h', palette='viridis')
    axes[0].set_title('Distribution of Workspace MRR by User Segment')
    axes[0].set_ylabel('Segment')
    axes[0].set_xlabel('MRR Latest')
    axes[0].set_xticklabels([f'${10 ** x:.0f}' for x in axes[0].get_xticks()])
    
    # Plot 'seats_latest' boxplots
    sns.boxplot(x=np.log10(data['seats_latest']), y=cluster_variable, data=data, ax=axes[1], orient='h', palette='viridis')
    axes[1].set_title('Distribution of Workspace Seats by User Segment')
    axes[1].set_ylabel('Segment')
    axes[1].set_xlabel('Seats Latest')
    axes[1].set_xticklabels([f'{10 ** x:.0f}' for x in axes[1].get_xticks()])
    
    # Calculate 'Seat Price' as mrr_latest / seats_latest
    data['seat_price'] = data['mrr_latest'] / data['seats_latest']
    
    # Plot 'Seat Price' boxplots
    sns.boxplot(x=np.log10(data['seat_price']), y=cluster_variable, data=data, ax=axes[2], orient='h', palette='viridis')
    axes[2].set_title('Distribution of User Seat Price by User Segment')
    axes[2].set_ylabel('Segment')
    axes[2].set_xlabel('Seat Price')
    axes[2].set_xticklabels([f'${10 ** x:.0f}' for x in axes[2].get_xticks()])
    
    # Plot 'Product Latest' as normalized bar plot
    sns.countplot(y=cluster_variable, hue='products_latest', data=data, ax=axes[3], palette='plasma')
    axes[3].set_title('Distribution of Products Latest by Segment')
    axes[3].set_ylabel('Segment')
    axes[3].set_xlabel('Count')
        
    # Normalize the bars within each segment
    for p in axes[3].patches:
        total = len(data[data['Cluster label'] == int(p.get_y() + p.get_height() / 2)])
        percentage = f'{100 * p.get_width() / total:.1f}%'
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height() / 2
        axes[3].annotate(percentage, (x, y))
    
    plt.tight_layout()
    plt.show()

# Call the function with example data
plot_clusters_rotated(joined, 'Cluster label')

#%% Hierarchical

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

def hierarchical_model(data):
    # Make a deep copy to avoid warnings
    data = data.copy(deep=True)
    
    # Drop rows with missing values in key columns
    data.dropna(subset=['mrr_latest', 'seats_latest'] + [col for col in data.columns if col.endswith('_rate')], inplace=True)
    
    # Create the dependent variable 'mrr_user'
    data['mrr_user'] = data['mrr_latest'] / data['seats_latest']
    
    # Identify variables ending in '_rate'
    rate_vars = [col for col in data.columns if col.endswith('_rate')]
    
    # Create the formula for the mixed linear model
    formula = 'mrr_user ~ ' + ' + '.join(rate_vars)
    
    try:
        # Fit the model
        model = smf.mixedlm(formula, data, groups=data['workspace_id'])
        result = model.fit()
        
        # Diagnostic plots
        # 1. Residuals vs Fitted values
        plt.figure(figsize=(10, 6))
        sns.residplot(x=result.fittedvalues, y=result.resid, lowess=True)
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted')
        plt.show()
        
        # 2. Q-Q Plot
        sm.qqplot(result.resid, fit=True, line='45')
        plt.title('Normal Q-Q')
        plt.show()

        print(result.summary())
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Assume joined_data is your DataFrame
# Call the function
model_result = hierarchical_model(joined_data)

#%% Adoption

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def adoption_by_segment(data, segment_variable):
    # Function to format label
    def format_label(label):
        return ' '.join([word.capitalize() for word in label.split('_')])
    
    # Function to format percentage on x-axis
    def percentage_formatter(x, pos):
        return f"{x:.0f}%"

    # Filter out rows where 'role' is equal to 'VIEWER'
    filtered_data = data[data['role'] != 'VIEWER']
    
    # Drop constant and non-numeric columns
    filtered_data = filtered_data.select_dtypes(include=['number']).loc[:, filtered_data.nunique() > 1]

    # Remove 'duration' if it exists from filtered_data
    if 'duration' in filtered_data:
        filtered_data = filtered_data.drop(columns=['duration'], errors='ignore')
    
    # Get unique segments
    unique_segments = data[segment_variable].unique()

    # Create subplots for each segment
    fig, axes = plt.subplots(len(unique_segments), 1, figsize=(10, 5 * len(unique_segments)))

    # Set font to Helvetica
    plt.rcParams['font.family'] = 'Helvetica'
    
    for i, segment in enumerate(unique_segments):
        segment_data = filtered_data[data[segment_variable] == segment]
        percentages = (segment_data > 0).mean() * 100

        # Rename columns using the custom label formatting function
        percentages.index = percentages.index.map(format_label)

        # Sort the percentages in descending order
        percentages = percentages.sort_values(ascending=False)
        
        # Create a bar plot of adoption percentages
        sns.barplot(x=percentages.values, y=percentages.index, palette="RdBu_r", order=percentages.index, ax=axes[i])
        axes[i].set_xlabel('Adoption Percentage')
        axes[i].set_ylabel('Feature')
        axes[i].set_title(f'Adoption Percentage for Each Feature in Segment {segment}')
        axes[i].xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        
    plt.tight_layout()
    plt.show()


# Call the function with example data
adoption_by_segment(joined, 'Cluster label')


#%% Save

def save_model():
    return

def package_distribution(clustered_data):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set aesthetic style
    sns.set(style="white")
    
    # Define a color palette
    color_palette = sns.color_palette("husl", 10)

    # Filter out only the rows where 'live_flag' is True
    filtered_data = clustered_data[clustered_data['live_flag'] == True]
    
    # Create FacetGrid to show distribution of 'products_latest' for each cluster
    g = sns.FacetGrid(filtered_data, col='cluster_label', col_wrap=3)
    
    # Set the binwidth to a different size, for example, 5
    g.map_dataframe(sns.histplot, x='products_latest', binwidth=5, palette=color_palette)
    
    # Label the bars with percentages
    for ax in g.axes.flat:
        cluster_label = ax.get_title().split()[-1]
        cluster_data = filtered_data[filtered_data['cluster_label'] == int(cluster_label)]
        total_count = len(cluster_data)

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height / total_count * 100:.0f}%", 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='baseline',
                        fontsize=10, fontweight='bold')

        # Remove grid lines
        ax.grid(False)
        
        # Outline subplot with a black border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

    g.set_axis_labels('Products Latest', fontsize=10)
    g.set_titles("Segment {col_name}", fontsize=10)
    g.set(xticks=range(0, int(filtered_data['products_latest'].max()) + 1, 5))

    plt.tight_layout()
    plt.show()

# Assuming 'clustered_data' is the DataFrame containing your data with 'cluster_levels' as cluster labels
package_distribution(clustered_data)


def save_to_excel():
    import pandas as pd

    # Assuming clustered_data is your DataFrame
    clustered_data.to_excel("clustered_data.xlsx", index=False)

save_to_excel()

def get_working_directory():
    import os
    return os.getcwd()

current_dir = get_working_directory()
print("Current working directory:", current_dir)





def main():
    load()
    preprocess()
    split()
    train()
    test()
    evaluate()
    interpret()
    save()
    



