"""
Created on Tue Sep 12 19:34:20 2023

@author: patricksweeney
"""


#%% Import
import pandas as pd
import numpy as np

def import_data_local():
    file_path = '/Users/patricksweeney/growth/03_Product/04_Engagement states/01_Static cluster/User data.xlsx'
    data = pd.read_excel(file_path)
    return data

data = import_data_local()

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
    
    # Create an imputer instance with median strategy
    imputer = SimpleImputer(strategy='median')
    
    # Extract numerical columns
    data_num = data_copy.select_dtypes(include=[np.number])
    
    # Fit the imputer to the numerical data with NaNs
    imputer.fit(data_num)
    
    # Transform the numerical data with the imputer
    data_num_imputed = imputer.transform(data_num)
    
    # Convert the imputed data back to a DataFrame
    data_num_imputed_df = pd.DataFrame(data_num_imputed, columns=data_num.columns, index=data_num.index)
    
    # Replace only the NaNs in the original DataFrame with the imputed values
    for col in data_num.columns:
        data_copy.loc[data_num[col].isna(), col] = data_num_imputed_df.loc[data_num[col].isna(), col]
    
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


#%% Distribution (log log plots)


# Define a custom tick formatter to display percentages
def percentage_formatter(x, pos):
    return f'{x:.0f}%'

# Custom label formatting function
def format_label(label):
    # Replace underscores with spaces, apply title case, and remove ' Rate' from the end
    return label.replace('_', ' ').title().replace(' Rate', '')

def adoption(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import FuncFormatter
    
    filtered_data = data

    # Filter out rows where 'role' is equal to 'VIEWER'
    filtered_data = data[data['role'] != 'VIEWER']
    


    # Drop constant and non-numeric columns
    filtered_data = filtered_data.select_dtypes(include=['number']).loc[:, filtered_data.nunique() > 1]

    # Remove 'duration' if it exists from filtered_data
    if 'duration' in filtered_data:
        filtered_data = filtered_data.drop(columns=['duration'], errors='ignore')

    # For each column, calculate the percentage of rows where value > 0
    # Store and plot in a bar plot
    percentages = (filtered_data > 0).mean() * 100

    # Set font to Helvetica
    plt.rcParams['font.family'] = 'Helvetica'

    # Rename columns using the custom label formatting function
    percentages.index = percentages.index.map(format_label)

    # Sort the percentages in descending order
    percentages = percentages.sort_values(ascending=False)

    # Create a bar plot of adoption percentages
    plt.figure(figsize=(10, 5))
    sns.barplot(x=percentages.values, y=percentages.index, palette="RdBu_r", order=percentages.index)
    plt.xlabel('Adoption Percentage')
    plt.ylabel('Features')
    plt.title('Adoption Percentage for Each Feature')

    # Use the custom percentage formatter for x-axis
    plt.gca().xaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    plt.show()

# Call the function
adoption(data)





# Define a custom tick formatter to display real numbers
def real_numbers(x, pos):
    return str(int(10**x))

def loglog(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import FuncFormatter
    
    # Drop constant and non-numeric columns
    filtered_data = data.select_dtypes(include=['number']).loc[:, data.nunique() > 1]
    
    # Extract the 'duration' column for coloring
    duration_values = data['duration']
    
    # Remove 'duration' if it exists from filtered_data
    if 'duration' in filtered_data:
        filtered_data = filtered_data.drop(columns=['duration'], errors='ignore')
    
    # Rename columns to 'Title Case' without underscores
    renamed_columns = {col: col.replace('_', ' ').title() for col in filtered_data.columns}
    filtered_data.rename(columns=renamed_columns, inplace=True)
    
    # Number of rows and columns for subplots
    n = len(filtered_data.columns)
    n_cols = 4  # You can adjust the number of columns here
    n_rows = int(np.ceil(n / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), squeeze=False)
    axes = axes.flatten()
    
    # Loop through each variable for log-log plot
    for i, col in enumerate(filtered_data.columns):
        ax = axes[i]
        non_zero_values = filtered_data[col][filtered_data[col] > 0]
        colors = duration_values[filtered_data[col] > 0]
        
        # Add jitter to x and y values
        jitter = 0  # You can adjust the amount of jitter
        x_jitter = np.random.uniform(-jitter, jitter, len(non_zero_values))
        y_jitter = np.random.uniform(-jitter, jitter, len(non_zero_values))
        
        sns.scatterplot(x=np.log10(1 + np.arange(len(non_zero_values))) + x_jitter, 
                        y=np.log10(1 + np.sort(non_zero_values)[::-1]) + y_jitter, 
                        ax=ax, edgecolor='none', hue=colors, palette="RdBu_r", legend=False)
        ax.set_title(col)
        ax.set_xlabel('User rank')
        ax.set_ylabel('Usage rate')
        
        # Set x and y axis tick locators and formatters in log10 format
        ax.xaxis.set_major_locator(plt.FixedLocator(np.log10([1, 10, 100, 1000, 10000])))
        ax.xaxis.set_major_formatter(FuncFormatter(real_numbers))
        ax.yaxis.set_major_locator(plt.FixedLocator(np.log10([1, 10, 100, 1000, 10000])))
        ax.yaxis.set_major_formatter(FuncFormatter(real_numbers))
        
    # Remove extra subplots
    for i in range(len(filtered_data.columns), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

# Call the function
loglog(data)




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
    g = sns.pairplot(numeric_data, kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}})
    
    # Update plot labels to 'Title Case'
    for ax in plt.gcf().axes:
        ax.set_xlabel(ax.get_xlabel().replace('_', ' ').title())
        ax.set_ylabel(ax.get_ylabel().replace('_', ' ').title())
        
    # Add a title to the plot
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Distributions and Relationship Between Feature Usage', fontsize=16)
    
    plt.show()


eda_pair_plot(data)

def eda_correlation_matrix(data):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
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




#%% Train kmeans

# Adding axis labels to the clustering plot for Principal Component 1 and Principal Component 2.

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
        pca = PCA(n_components=pca)
        reduced_data = pca.fit_transform(numeric_data)
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

    # Plotting
    if plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.title(f'{algorithm} Clustering ({n_clusters} Segments)\nSilhouette Score: {silhouette_avg}')
        plt.xlabel('Superfeature 1')  # Labeling the x-axis
        plt.ylabel('Superfeature 2')  # Labeling the y-axis
        plt.show()

        # Plot dendrogram for hierarchical clustering methods
        if algorithm in ['Agglomerative', 'HDBSCAN']:
            linked = linkage(reduced_data, 'single')
            plt.figure(figsize=(15, 7))
            dendrogram(linked, truncate_mode='lastp', p=12, color_threshold=1, orientation='top', distance_sort='descending', show_leaf_counts=True, cmap='RdBu')
            plt.title(f'{algorithm} Dendrogram')
            plt.xlabel('Sample index or (cluster size)')
            plt.ylabel('Distance')
            plt.show()

    # Cluster summary
    cluster_summary = clustered_data.groupby('Cluster label').median(numeric_only=True)

    return clustered_data, silhouette_avg, cluster_summary

# Example usage for testing (replace 'data' with your DataFrame)
clustered_data, silhouette_avg, cluster_summary = perform_clustering(data=data, pca=2, n_clusters=3, algorithm='Agglomerative', do_pca=True)


#%% Interpretation

# Modified function to plot clusters
def plot_clusters(clustered_data, plot_type='scatter'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    plt.rcParams['font.family'] = 'Helvetica'
    
    # Identify numeric variables
    numeric_vars = clustered_data.select_dtypes(include=['number']).columns.tolist()
    
    # Make sure only numeric variables are used for median calculation
    if 'Cluster label' in numeric_vars:
        numeric_vars.remove('Cluster label')
    
    # Rename clusters based on their median values
    try:
        cluster_summary = clustered_data.groupby('Cluster label').median(numeric_only=True)
        sorted_clusters = cluster_summary.apply(lambda x: x.median(), axis=1).sort_values(ascending=False).index
        rename_dict = {old: f'Segment {chr(65 + new)}' for new, old in enumerate(sorted_clusters)}
        clustered_data['Cluster label'] = clustered_data['Cluster label'].map(rename_dict)
    except Exception as e:
        print(f"An error occurred while renaming clusters: {e}")
    
    if plot_type == 'scatter':
        sns.pairplot(data=clustered_data, hue='Cluster label', vars=numeric_vars, diag_kind='kde', palette="viridis", plot_kws={'alpha': 0.5})
        plt.suptitle('Scatter Plots by Segment', y=1.02)
    
    elif plot_type == 'heatmap':
        cluster_summary.rename(index=rename_dict, inplace=True)
        sns.heatmap(cluster_summary.T, annot=True, cmap='RdBu_r', fmt='.2f')
        plt.title('Average Metrics by Segment')
        plt.xlabel('Segment')
        plt.ylabel('Feature')
    
    else:
        print("Invalid plot_type. Supported types are 'scatter' and 'heatmap'.")



# Uncomment below lines to test the function with your 'clustered_data' DataFrame
plot_clusters(clustered_data, plot_type='scatter')
plot_clusters(clustered_data, plot_type='heatmap')
plot_clusters(clustered_data, plot_type='bar')









def bin_rate_inverse(rate_per_30_days):
    if rate_per_30_days == 0:
        return "Never"
    
    inverse_rate_30_days = 30 / rate_per_30_days
    inverse_bins = {
        'Daily': (0, 1),
        'Weekly': (1, 7),
        'Fortnightly': (7, 14),
        'Monthly': (14, 30),
        'Quarterly': (30, 90),
        'Annually': (90, 365),
        'Rarely': (365, float('inf'))
    }

    for time_frame, (lower, upper) in inverse_bins.items():
        if lower <= inverse_rate_30_days < upper:
            return time_frame
    return "Out of Range"



def cluster_summary(data, data_original, join_on):
    import pandas as pd

    joined_data = pd.merge(data[['cluster_label', join_on]], data_original, on=join_on, how='inner')
    
    numeric_cols = joined_data.select_dtypes(include=['number']).columns
    numeric_cols = [col for col in numeric_cols if col != 'cluster_label']

    summary = joined_data.groupby('cluster_label')[numeric_cols].median().reset_index()

    rate_cols = [col for col in numeric_cols if 'rate' in col]
    summary[rate_cols] = (summary[rate_cols] * 30).round(1)
    
    cluster_counts = joined_data['cluster_label'].value_counts().reset_index()
    cluster_counts.columns = ['cluster_label', 'n']
    total_count = len(joined_data)
    cluster_counts['% of total'] = (cluster_counts['n'] / total_count * 100).round(1)

    summary = pd.merge(summary, cluster_counts, on='cluster_label', how='inner')

    total_mrr = joined_data['mrr_latest'].sum()
    mrr_by_cluster = joined_data.groupby('cluster_label')['mrr_latest'].sum().reset_index()
    mrr_by_cluster['% mrr_latest'] = (mrr_by_cluster['mrr_latest'] / total_mrr * 100).round(1)

    summary = pd.merge(summary, mrr_by_cluster[['cluster_label', '% mrr_latest']], on='cluster_label', how='inner')

    summary_row = pd.DataFrame(summary.mean(numeric_only=True)).T
    summary_row['cluster_label'] = 'Summary'
    summary = pd.concat([summary, summary_row], ignore_index=True)
    
    print("Cluster Summary:")
    print(summary)
    
    for rate_col in rate_cols:
        binned_col = f"{rate_col}_bin"
        summary[binned_col] = summary[rate_col].apply(bin_rate_inverse)
    
    return summary


# Sample usage
summary = cluster_summary(clustered_data, data_original, 'workspace_id')



def plot_cluster_summary(summary):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    sns.set(style="white")
    color_palette = sns.color_palette("husl", 10)

    numeric_cols = summary.select_dtypes(include=['number']).columns.tolist()
    for col in ['cluster_label', 'n', '% of total']:
        if col in numeric_cols:
            numeric_cols.remove(col)
    
    summary_for_plot = summary[summary['cluster_label'] != 'Summary']
    scaled_summary = summary_for_plot[numeric_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    scaled_summary['cluster_label'] = summary_for_plot['cluster_label']
    
    melted = pd.melt(scaled_summary, id_vars=['cluster_label'], value_vars=numeric_cols)
    unique_clusters = summary_for_plot['cluster_label'].unique()
    
    fig, axes = plt.subplots(1, len(unique_clusters), figsize=(15, 8), sharey=True)
    
    for ax, cluster in zip(axes, unique_clusters):
        cluster_data = melted[melted['cluster_label'] == cluster]
        sns.barplot(x='value', y='variable', data=cluster_data, ax=ax, palette=color_palette)
        
        pct_total = int(summary_for_plot.loc[summary_for_plot['cluster_label'] == cluster, '% of total'].values[0])
        ax.set_title(f"Segment {cluster} ({pct_total}%)", fontsize=14)
        
        ax.grid(False)
        
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        
        for index, p in enumerate(ax.patches):
            width = p.get_width()
            y, height = p.get_y(), p.get_height()
            actual_value = summary_for_plot.loc[summary_for_plot['cluster_label'] == cluster, numeric_cols[index]].values[0]
            
            binned_col_name = f"{numeric_cols[index]}_bin"
            if binned_col_name in summary_for_plot.columns:
                binned_value = summary_for_plot.loc[summary_for_plot['cluster_label'] == cluster, binned_col_name].values[0]
                ax.text(width + 0.01, y + height / 2, f"{actual_value:.1f} ({binned_value})", va='center', fontsize=10)
            else:
                ax.text(width + 0.01, y + height / 2, f"{actual_value:.1f}", va='center', fontsize=10)
        
        ax.set_xlim(0, 1.1)
        ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
    plt.tight_layout()
    plt.show()

plot_cluster_summary(summary)


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
    



