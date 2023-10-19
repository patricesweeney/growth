"""
Created on Tue Sep 12 19:34:20 2023

@author: patricksweeney
"""


#%% Import
import pandas as pd
import numpy as np


def import_data_local():
    import pandas as pd
    file_path = '/Users/patricksweeney/growth/03_Product/04_Engagement states/01_Static cluster/User data.xlsx'
    data = pd.read_excel(file_path)
    return data

data = import_data_local()



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
    
    


def impute_missing_values(data):
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    
    # Drop columns with all missing values
    data.dropna(axis=1, how='all', inplace=True)
    
    # Replace infinities with NaNs
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Create an imputer instance with median strategy
    imputer = SimpleImputer(strategy='median')
    
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

find_missing_values(data)

data = impute_missing_values(data)
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



#%% Preprocess
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
    
    # Replace infinite values with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[float, int])
    
    # Generate pairplot with red LOESS fit
    sns.pairplot(numeric_data)

# Run the function
eda_pair_plot(data)



def eda_correlation_matrix(data):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Drop constant and non-numeric columns
    filtered_data = data.select_dtypes(include=['number']).loc[:, data.nunique() > 1]
    
    # Compute the correlation matrix
    corr = filtered_data.corr()
    
    # Create a copy of the correlation matrix to sort it
    corr_copy = corr.copy()
    corr_copy['sum'] = corr_copy.abs().sum()
    sorted_columns = corr_copy.sort_values(by='sum', ascending=False).index
    
    # Now fetch the sorted columns and rows without 'sum'
    sorted_corr = corr.loc[sorted_columns, sorted_columns]
    
    # Set up the matplotlib figure with a reasonable size
    plt.figure(figsize=(10, 8))
    
    # Draw the heatmap with a custom color map and annotations rounded to 1 decimal place
    sns.heatmap(sorted_corr, cmap="RdBu", vmin=-1, vmax=1, annot=True, fmt=".1f")
    
    # Show the plot
    plt.show()


eda_correlation_matrix(data)



def pca(data):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    numeric_data = data.select_dtypes(include=[np.number])
    normalized_cols = [col for col in numeric_data.columns 
                        if np.isclose(numeric_data[col].mean(), 0, atol=1e-2) 
                        and np.isclose(numeric_data[col].std(), 1, atol=1e-2)]
    
    if not normalized_cols:
        print("No normalized columns found. Check the data.")
        return
    
    filtered_data = numeric_data[normalized_cols]
    
    pca = PCA()
    pca_result = pca.fit_transform(filtered_data)
    
    # Explained Variance
    explained_var_ratio = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(explained_var_ratio)), explained_var_ratio)
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance by Each Component')
    plt.show()
    
    # Scree Plot
    plt.figure(figsize=(10, 5))
    plt.plot(pca.explained_variance_, 'o-')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.show()
    
    # Loading Plot (as vectors)
    plt.figure(figsize=(10, 10))
    components = pd.DataFrame(pca.components_, columns=filtered_data.columns)
    for i, (x, y) in enumerate(zip(components.iloc[0, :], components.iloc[1, :])):
        plt.arrow(0, 0, x, y, head_width=0.05, head_length=0.1, fc='k', ec='k')
        plt.text(x * 1.2, y * 1.2, filtered_data.columns[i], color='r')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Loading Plot as Vectors')
    plt.grid()
    plt.show()
    
    # Print top features by loading for the first two principal components
    print("Top Features by Loading for Principal Component 1:")
    print(components.iloc[0, :].sort_values(ascending=False)[:5])
    print("Top Features by Loading for Principal Component 2:")
    print(components.iloc[1, :].sort_values(ascending=False)[:5])


pca(data = data)



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

def find_optimal_clusters_kmedians(data, max_k):
    from sklearn_extra.cluster import KMedoids
    from sklearn.metrics import silhouette_score
    import numpy as np
    import matplotlib.pyplot as plt
    
    """
    Find the optimal number of clusters using KMedians, Elbow Method, and Silhouette Method.
    """
    # Filter for numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Further filter for normalized columns
    normalized_cols = [col for col in numeric_data.columns 
                       if np.isclose(numeric_data[col].mean(), 0, atol=1e-2) 
                       and np.isclose(numeric_data[col].std(), 1, atol=1e-2)]
    
    if not normalized_cols:
        print("No suitable columns for clustering.")
        return
    
    # Filter data to include only normalized columns
    filtered_data = numeric_data[normalized_cols]
    
    # Elbow Method
    inertia = []
    for k in range(1, max_k+1):
        kmedians = KMedoids(n_clusters=k, random_state=1).fit(filtered_data)
        inertia.append(kmedians.inertia_)
        
    plt.figure()
    plt.plot(range(1, max_k+1), inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Silhouette Method
    sil_scores = []
    for k in range(2, max_k+1):  # Silhouette analysis starts with at least 2 clusters
        kmedians = KMedoids(n_clusters=k, random_state=1).fit(filtered_data)
        score = silhouette_score(filtered_data, kmedians.labels_)
        sil_scores.append(score)
        
    plt.figure()
    plt.plot(range(2, max_k+1), sil_scores, marker='o')
    plt.title('Silhouette Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()



# Call the function
find_optimal_clusters_kmedians(data, max_k=10)


#%% Train kmeans

def perform_clustering(data, variables, algorithm='KMeans', n_clusters=3, do_scaling=True, do_pca=True, plot=True, **kwargs):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, MeanShift, SpectralClustering, DBSCAN, Birch
    from sklearn_extra.cluster import KMedoids
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    from hdbscan import HDBSCAN
    import matplotlib.pyplot as plt
    
    # Filter only numeric columns and convert to matrix
    numeric_data = data[variables].select_dtypes(include=['number']).values
    
    # Data scaling
    if do_scaling:
        scaler = StandardScaler()
        numeric_data = scaler.fit_transform(numeric_data)
    
    # Dimensionality reduction
    if do_pca:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(numeric_data)
    else:
        reduced_data = numeric_data
    
    # Clustering
    if algorithm == 'KMeans':
        model = KMeans(n_clusters=n_clusters, **kwargs)
    elif algorithm == 'KMedoids':
        model = KMedoids(n_clusters=n_clusters, **kwargs)
    elif algorithm == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
    elif algorithm == 'AffinityPropagation':
        model = AffinityPropagation(**kwargs)
    elif algorithm == 'MeanShift':
        model = MeanShift(**kwargs)
    elif algorithm == 'SpectralClustering':
        model = SpectralClustering(n_clusters=n_clusters, **kwargs)
    elif algorithm == 'DBSCAN':
        model = DBSCAN(**kwargs)
    elif algorithm == 'HDBSCAN':
        model = HDBSCAN(**kwargs)
    elif algorithm == 'GaussianMixture':
        model = GaussianMixture(n_components=n_clusters, **kwargs)
    elif algorithm == 'Birch':
        model = Birch(n_clusters=n_clusters, **kwargs)
    else:
        raise ValueError("Invalid clustering algorithm")
    
    model.fit(reduced_data)
    labels = model.labels_ if hasattr(model, 'labels_') else model.predict(reduced_data)
    
    # Metrics
    silhouette_avg = silhouette_score(reduced_data, labels) if len(set(labels)) > 1 else "N/A"
    
    # Add cluster labels to data
    clustered_data = data.copy()
    clustered_data['Cluster label'] = labels  # Rename the cluster label column
    
    # Plotting
    if plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='rainbow', alpha=0.5)
        plt.title(f'{algorithm} Clustering (k={n_clusters})\nSilhouette Score: {silhouette_avg}')
        plt.show()
    
    # Cluster summary
    cluster_summary = clustered_data.groupby('cluster_label').median(numeric_only=True)
    
    return clustered_data, silhouette_avg, cluster_summary


variables = ['note_rate', 'highlight_rate', 'insight_rate', 'invite_rate', 'shared_object_rate']
clustered_data, silhouette_avg, cluster_summary = perform_clustering(data, variables, do_pca=True, algorithm='Agglomerative', n_clusters=3)





#%% Interpretation


# Function to plot clusters with user-specified figure size and DPI settings
def plot_clusters(clustered_data, plot_type='scatter', variables=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Function to format labels
    def format_label(label):
        return label.replace('_', ' ').capitalize()
    
    # Set the color palette and font style
    sns.set_palette("viridis")
    plt.rcParams['font.family'] = 'Helvetica'
    sns.set_style("whitegrid", {'axes.grid' : False})
    
    # Format variable names
    if variables is None:
        variables = clustered_data.select_dtypes(include=['number']).columns.tolist()
    formatted_vars = [format_label(var) for var in variables]
    
    # Rename columns in DataFrame for plotting
    clustered_data.rename(columns={old: new for old, new in zip(variables, formatted_vars)}, inplace=True)
    
    plt.figure(1, figsize=(20, 20), dpi=72)  # Set figure size and DPI
    
    if plot_type == 'scatter':
        sns.pairplot(data=clustered_data, hue='Cluster label', vars=formatted_vars, diag_kind='kde', palette="viridis", plot_kws={'alpha': 0.5})
        plt.suptitle('Customer Behavior by Segment', fontsize=20, fontweight='bold', y=1.04)
    
    elif plot_type == 'heatmap':
        cluster_means = clustered_data.groupby('Cluster label').mean()[formatted_vars]
        sns.heatmap(cluster_means, annot=True, cmap='viridis')
        plt.title('Average Customer Metrics by Segment', fontsize=20, fontweight='bold', pad=20)
        
    elif plot_type == 'box':
        for var in formatted_vars:
            sns.boxplot(x='Cluster label', y=var, data=clustered_data, palette="viridis")
            plt.xlabel('Cluster Label')
            plt.ylabel(var)
            plt.title(f'Distribution of {var} by Customer Segment', fontsize=20, fontweight='bold', pad=20)
            plt.show()
            
    elif plot_type == 'bar':
        cluster_counts = clustered_data['Cluster label'].value_counts(normalize=True).reset_index()
        print("Debug: Columns in cluster_counts:", cluster_counts.columns)  # Debug line
        # cluster_counts.columns = [format_label(c) for c in cluster_counts.columns]  # remove this line
        sns.barplot(x='Cluster label', y='Cluster label', data=cluster_counts, palette="viridis")  # use 'Cluster label' directly here
            
    else:
        print("Invalid plot_type. Supported types are 'scatter', 'heatmap', 'box', and 'bar'.")



plot_clusters(clustered_data, plot_type='box', variables=['note_rate', 'highlight_rate'])


plot_clusters(clustered_data, plot_type='scatter', variables=['note_rate', 'highlight_rate', 'insight_rate', 'invite_rate', 'shared_object_rate'])

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
    



