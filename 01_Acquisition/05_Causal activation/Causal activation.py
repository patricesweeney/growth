#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:09:41 2023

@author: patricksweeney
"""


#%% Import

def import_data_local():
    import pandas as pd
    file_path = '/Users/patricksweeney/growth/01_Acquisition/05_Causal activation/F7 days.xlsx'
    data = pd.read_excel(file_path)
    
    columns_to_remove = ['paid_signup', 'converted']  # Replace with your column names
    data = data.drop(columns=columns_to_remove, axis=1)
    
    # columns_to_remove = ['role', 'seniority']  # Replace with your column names
    # data = data.drop(columns=columns_to_remove, axis=1)

    return data

data = import_data_local()


def one_hot_encode(data, variables):
    import pandas as pd
    
    # Ensure the variables are in the DataFrame
    for var in variables:
        if var not in data.columns:
            raise ValueError(f"Variable '{var}' not found in the DataFrame")

    # One-hot encode the specified variables
    for var in variables:
        # Get one-hot encoding
        one_hot = pd.get_dummies(data[var], prefix=var, dtype=bool)
        # Convert True/False to 1/0
        one_hot = one_hot.astype(int)
        # Drop the original column
        data = data.drop(var, axis=1)
        # Join the encoded DataFrame
        data = data.join(one_hot)
    
    return data

data = one_hot_encode(data, ['seniority', 'role'])


def print_variables(data):
    for column in data.columns:
        print(column)

print_variables(data)


#%%  Correlations and PID



def eda_correlation_matrix(data, log):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Drop constant and non-numeric columns
    filtered_data = data.select_dtypes(include=['number']).loc[:, data.nunique() > 1]
    
    # Apply log+1 transformation if log is True
    if log:
        filtered_data = np.log1p(filtered_data)
        
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
    
    # Draw the heatmap with a custom color map
    sns.heatmap(sorted_corr, cmap="RdBu", vmin=-1, vmax=1)
    
    # Show the plot
    plt.show()

eda_correlation_matrix(data, log = True)





def eda_mutual_information_matrix(data, log, k=3):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.feature_selection import mutual_info_regression

    # Drop constant and non-numeric columns
    filtered_data = data.select_dtypes(include=['number']).loc[:, data.nunique() > 1]

    # Apply log+1 transformation if log is True
    if log:
        filtered_data = np.log1p(filtered_data)

    # Initialize matrix for mutual information
    n_features = filtered_data.shape[1]
    mi_matrix = np.zeros((n_features, n_features))

    # Compute mutual information for each pair of features
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                mi_matrix[i, j] = mutual_info_regression(
                    filtered_data.iloc[:, [i]], 
                    filtered_data.iloc[:, j], 
                    discrete_features=False, 
                    n_neighbors=k
                )[0]

    # Set diagonal elements to NaN or zero
    np.fill_diagonal(mi_matrix, 0)

    # Convert to DataFrame for easier plotting
    mi_df = pd.DataFrame(mi_matrix, index=filtered_data.columns, columns=filtered_data.columns)

    # Set up the matplotlib figure with a reasonable size
    plt.figure(figsize=(10, 8))

    # Draw the heatmap
    sns.heatmap(mi_df, cmap="plasma", annot=False, vmin=0, vmax = 0.1)

    # Show the plot
    plt.show()

# Example usage
eda_mutual_information_matrix(data, log=True, k=3)



def mi_regression(data, target_column):
    from sklearn.feature_selection import mutual_info_regression
    import pandas as pd
    from scipy.stats import entropy
    from pycit.markov_blanket import MarkovBlanket

    # Separate the target variable from the source variables
    target = data[target_column]
    source_data = data.drop(target_column, axis=1)
    
    # Calculate the entropy of the target variable
    target_entropy = entropy(target.value_counts(normalize=True))

    # Calculate mutual information
    mi_scores = mutual_info_regression(source_data, target)

    # Pair each variable name with its MI score and round the MI scores
    mi_scores_with_names = [(name, round(score, 2)) for name, score in zip(source_data.columns, mi_scores)]

    # Sort the pairs by MI score, descending
    sorted_mi_scores = sorted(mi_scores_with_names, key=lambda x: x[1], reverse=True)

    # Print the entropy and the results
    print("Entropy of target:", round(target_entropy,2))
    for name, score in sorted_mi_scores:
        print(f"{name}: {score}")
        
    
    source_data_3d = np.expand_dims(source_data, axis=-1)
    target_3d = np.expand_dims(target, axis=-1)

    # Markov Blanket computation
    cit_funcs = {
        'it_args': {
            'test_args': {
                'statistic': 'ksg_mi',
                'n_jobs': 2
            }
        },
        'cit_args': {
            'test_args': {
                'statistic': 'ksg_cmi',
                'n_jobs': 2
            }
        }
    }

    # Using reshaped data
    mb = MarkovBlanket(source_data_3d, target_3d, cit_funcs)
    markov_blanket = mb.find_markov_blanket(verbose=True)

    # Print the Markov Blanket
    print("Markov Blanket of the target:", markov_blanket)

    return target_3d, source_data_3d

# Usage example
target, source_data = mi_regression(data, 'mrr_converted')  # Uncomment and use this line with your data




def pid(data, target):
    import dit
    import pandas as pd
    from dit.pid import PID_WB

    # Ensure the target variable is in the data
    if target not in data.columns:
        raise ValueError("Target variable not found in data.")

    # Prepare the data for dit
    # Convert DataFrame to list of tuples (each tuple is a row in the DataFrame)
    data_tuples = list(data.itertuples(index=False, name=None))

    # Create a dit Distribution from the data
    distribution = dit.Distribution(data_tuples)
    distribution.normalize()  # Normalize to ensure it's a valid probability distribution

    # Identify sources (all columns except the target)
    sources = [col for col in data.columns if col != target]

    # Create the PID object using the chosen measure (e.g., Williams and Beer)
    pid_obj = PID_WB(distribution, sources, target)

    return pid_obj

# Example usage:
# Assuming 'df' is your DataFrame and 'target_column' is your target variable
pid_result = pid(data, 'mrr_converted')
print(pid_result)

    


#%% EDA

def eda_pair_plot(data, transform, regression):
    import pandas as pd
    import seaborn as sns
    import numpy as np
    from scipy.stats.mstats import winsorize
    
    # Replace infinite values with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    # Data Transformation for Continuous Variables
    for col in numeric_data.columns:
        if numeric_data[col].nunique() > 2:  # Check if column is continuous
            if transform == 'log':
                # Shift the data if necessary to handle zero values
                shift = numeric_data[col].min() - 1 if numeric_data[col].min() <= 0 else 0
                numeric_data[col] = np.log1p(numeric_data[col] + shift)
            elif transform == 'winsorize':
                # Winsorize the right tail at the 99.9th percentile
                numeric_data[col] = winsorize(numeric_data[col], limits=[0, 0.001])

    # Generate pairplot with optional regression lines
    scatter_kws = {"alpha": 0.6, "s": 20}  # Adjust alpha and point size here
    if regression:
        sns.pairplot(numeric_data, kind='reg', plot_kws={'scatter_kws': {'color': 'blue', **scatter_kws, 'edgecolor': 'none'}, 'line_kws': {'color': 'red'}})
    else:
        sns.pairplot(numeric_data, plot_kws={'color': 'blue', **scatter_kws, 'edgecolor': 'none'})

# Usage example
eda_pair_plot(data = data, transform = 'winsorize', regression = True)




def eda_interaction_plots(data, outcome, transform=None):
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    # Replace infinite values with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Ensure the outcome variable is in the dataframe
    if outcome not in data.columns:
        raise ValueError(f"Outcome variable '{outcome}' not found in the data.")

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    # Data Transformation for all Variables
    if transform:
        numeric_data = numeric_data.apply(lambda x: np.log1p(x - x.min() + 1e-10) if x.nunique() > 2 else x)

    # Exclude the outcome variable from interaction variables
    interaction_vars = numeric_data.columns.drop(outcome)

    # Number of rows and columns for the subplot grid
    n_rows = n_cols = len(numeric_data.columns)  # square layout

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    for i, interaction_var in enumerate(numeric_data.columns):
        for j, col in enumerate(numeric_data.columns):
            ax = axes[i, j]
            sns.scatterplot(x=numeric_data[col], y=numeric_data[outcome], hue=numeric_data[interaction_var],
                            palette="coolwarm", ax=ax, legend=False)

            # Color scale normalization
            if transform:
                norm = plt.Normalize(numeric_data[interaction_var].min(), numeric_data[interaction_var].max())
            else:
                norm = plt.Normalize(numeric_data[interaction_var].min(), numeric_data[interaction_var].max())

            sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax)

            ax.set_title(f'{col} vs {outcome}\nInteraction: {interaction_var}')
            ax.xaxis.label.set_visible(False)
            ax.yaxis.label.set_visible(False)

    plt.tight_layout()
    plt.show()



# Usage example
eda_interaction_plots(data, 'mrr_converted', 'log')


def node_names(data):
    return {index: column_name for index, column_name in enumerate(data.columns)}

node_names(data)


#%% Casual priors

def causal_priors(reverse):
    import numpy as np
    import networkx as nx
    from io import StringIO

    GML_priors = """
    graph [
      directed 1
    node [id 0 label "mrr_converted"]
    node [id 1 label "project_count_f7d"]
    node [id 2 label "transcription_count_f7d"]
    node [id 3 label "highlight_count_f7d"]
    node [id 4 label "tag_count_f7d"]
    node [id 5 label "insight_count_f7d"]
    node [id 6 label "reel_created_count_f7d"]
    node [id 7 label "invite_count_f7d"]
    node [id 8 label "shared_object_note_count_f7d"]
    node [id 9 label "shared_object_insight_count_f7d"]
    node [id 10 label "note_viewed_user_count_f7d"]
    node [id 11 label "tag_viewed_user_count_f7d"]
    node [id 12 label "insight_viewed_user_count_f7d"]
    node [id 13 label "seniority_Department lead"]
    node [id 14 label "seniority_Executive"]
    node [id 15 label "seniority_Individual contributor"]
    node [id 16 label "seniority_Team lead"]
    node [id 17 label "role_CUSTOMER_SUCCESS"]
    node [id 18 label "role_DESIGN"]
    node [id 19 label "role_ENGINEERING"]
    node [id 20 label "role_FINANCE"]
    node [id 21 label "role_LEGAL"]
    node [id 22 label "role_MANAGEMENT"]
    node [id 23 label "role_MARKETING"]
    node [id 24 label "role_OPERATIONS"]
    node [id 25 label "role_OTHER"]
    node [id 26 label "role_PRODUCT_MANAGEMENT"]
    node [id 27 label "role_RESEARCH"]
    node [id 28 label "role_SALES"]
    node [id 29 label "role_STUDENT"]
    node [id 30 label "role_SUPPORT"]

          
    
      edge [source 1 target 2 label "project_count_f7d → transcription_count_f7d"]
      edge [source 2 target 3 label "transcription_count_f7d → highlight_count_f7d"]
      edge [source 2 target 4 label "transcription_count_f7d → tag_count_f7d"]
      edge [source 4 target 3 label "tag_count_f7d → highlight_count_f7d"]
      edge [source 3 target 6 label "highlight_count_f7d → reel_created_count_f7d"]
      edge [source 3 target 5 label "highlight_count_f7d → insight_count_f7d"]
      edge [source 6 target 7 label "reel_created_count_f7d → invite_count_f7d"]
      edge [source 5 target 7 label "insight_count_f7d → invite_count_f7d"]
      edge [source 6 target 8 label "reel_created_count_f7d → shared_object_note_count_f7d"]
      edge [source 5 target 9 label "insight_count_f7d → shared_object_insight_count_f7d"]
      edge [source 6 target 0 label "reel_created_count_f7d → mrr_converted"]
      edge [source 7 target 0 label "invite_count_f7d → mrr_converted"]
      edge [source 8 target 0 label "shared_object_note_count_f7d → mrr_converted"]
      edge [source 9 target 0 label "shared_object_insight_count_f7d → mrr_converted"]
      edge [source 4 target 0 label "tag_count_f7d → mrr_converted"]
      edge [source 3 target 0 label "highlight_count_f7d → mrr_converted"]
      edge [source 8 target 10 label "shared_object_note_count_f7d → note_viewed_user_count_f7d"]
      edge [source 9 target 12 label "shared_object_insight_count → note_viewed_user_count_f7d"]

      edge [source 14 target 0 label "seniority_Executive → mrr_converted"]
    edge [source 15 target 0 label "seniority_Individual contributor → mrr_converted"]
    edge [source 16 target 0 label "seniority_Team lead → mrr_converted"]
    edge [source 17 target 0 label "role_CUSTOMER_SUCCESS → mrr_converted"]
    edge [source 18 target 0 label "role_DESIGN → mrr_converted"]
    edge [source 19 target 0 label "role_ENGINEERING → mrr_converted"]
    edge [source 20 target 0 label "role_FINANCE → mrr_converted"]
    edge [source 21 target 0 label "role_LEGAL → mrr_converted"]
    edge [source 22 target 0 label "role_MANAGEMENT → mrr_converted"]
    edge [source 23 target 0 label "role_MARKETING → mrr_converted"]
    edge [source 24 target 0 label "role_OPERATIONS → mrr_converted"]
    edge [source 25 target 0 label "role_OTHER → mrr_converted"]
    edge [source 26 target 0 label "role_PRODUCT_MANAGEMENT → mrr_converted"]
    edge [source 27 target 0 label "role_RESEARCH → mrr_converted"]
    edge [source 28 target 0 label "role_SALES → mrr_converted"]
    edge [source 29 target 0 label "role_STUDENT → mrr_converted"]
    edge [source 30 target 0 label "role_SUPPORT → mrr_converted"]
      
      ] """
      
    
    # node [id 13 label "role"]
    # node [id 14 label "seniority"]
    
    # edge [source 13 target 3 label "role → highlight_count_f7d"]
    # edge [source 13 target 4 label "role → tag_count_f7d"]            
    # edge [source 13 target 5 label "role → insight_count_f7d"]
    # edge [source 13 target 6 label "role → reel_created_count_f7d"]
    # edge [source 13 target 7 label "role → invite_count_f7d"]
    # edge [source 13 target 0 label "role → mrr_converted"]
  
    # edge [source 14 target 3 label "seniority → highlight_count_f7d"]
    # edge [source 14 target 4 label "seniority → tag_count_f7d"]            
    # edge [source 14 target 5 label "seniority → insight_count_f7d"]
    # edge [source 14 target 6 label "seniority → reel_created_count_f7d"]
    # edge [source 14 target 7 label "seniority → invite_count_f7d"]
    # edge [source 14 target 0 label "seniority → mrr_converted"]
        
      
    # ] """

    # Use StringIO to treat string as a file for parsing
    GML_buffer = StringIO(GML_priors)
    G_priors = nx.parse_gml(GML_buffer, label='label')
    
    # Initialize a matrix with -1 indicating no prior knowledge
    n_features = len(G_priors.nodes())
    prior_knowledge = np.full((n_features, n_features), -1)

    # Update the matrix with 0s and 1s based on the edges in the graph
    for i, node_i in enumerate(G_priors.nodes()):
        for j, node_j in enumerate(G_priors.nodes()):
            if i != j:
                edge_exists = G_priors.has_edge(node_i, node_j) if not reverse else G_priors.has_edge(node_j, node_i)
                if edge_exists:
                    prior_knowledge[i, j] = 1
                else:
                    prior_knowledge[i, j] = 0

    return G_priors, prior_knowledge

# Call the function to get the NetworkX graph
G_priors, prior_knowledge = causal_priors(reverse = True)

# You now have the graph as a NetworkX object with correct labels


#%% Causal discovery
def causal_discovery(data, algorithm, prior_knowledge, transform):
    import networkx as nx
    import matplotlib.pyplot as plt
    from scipy.stats.mstats import winsorize
    import numpy as np
    from castle.common import GraphDAG
    from castle.common.independence_tests import CITest
    from castle.common.priori_knowledge import PrioriKnowledge
    from scipy import stats
    from castle.algorithms import PC, GES, ICALiNGAM, DirectLiNGAM, NotearsNonlinear, GOLEM, GAE, DAG_GNN, RL, ANMNonlinear, GraNDAG, Notears, MCSL, NotearsLowRank, PNL, GraNDAG

    # Make a copy of the data to avoid modifying the original DataFrame
    data_copy = data.copy()    
    

    # Select the algorithm based on the 'algorithm' argument
    if algorithm == 'PC': #fast
        algo = PC(variant='stable', alpha=0.01, ci_test = CITest.hsic_test)
    elif algorithm == 'GES': #medium
        algo = GES(criterion='bic')
    elif algorithm == 'GAE': #very slow
        algo = GAE(input_dim = len(data_copy.columns))
    elif algorithm == 'ANMNonlinear':
        algo = ANMNonlinear() #broken
    elif algorithm == 'DirectLiNGAM':
        if not prior_knowledge is not None:
            algo = DirectLiNGAM() #fast
        else:
            algo = DirectLiNGAM(prior_knowledge = prior_knowledge) #fast
    elif algorithm == 'ICALiNGAM':
        algo = ICALiNGAM() #fast
    elif algorithm == 'Notears':
        algo = Notears() #medium
    elif algorithm == 'NotearsLowRank':
        algo = NotearsLowRank() #medium
    elif algorithm == 'NotearsNonlinear':
        algo = NotearsNonlinear()
    elif algorithm == 'GOLEM':
        algo = GOLEM(num_iter=2e4) #medium
    elif algorithm == 'DAG_GNN':
        algo = DAG_GNN()
    elif algorithm == 'PNL':
        algo = PNL(device_type='cpu') #broken
    elif algorithm == 'GRAN':
        d = {'model_name': 'NonLinGauss', 'nonlinear': 'leaky-relu', 'optimizer': 'sgd', 'norm_prod': 'paths', 'device_type': 'cpu'}
        algo = GraNDAG(input_dim = len(data_copy.columns))
    elif algorithm == 'RL':
        algo = RL(nb_epoch=2000) #Slow
    elif algorithm == 'MCSL':
        algo = MCSL(model_type='nn',
          iter_step=100,
          rho_thresh=1e20,
          init_rho=1e-5,
          rho_multiply=10,
          graph_thresh=0.5,
          l1_graph_penalty=2e-3) #slow
    else:
        raise ValueError("Invalid algorithm specified")

# Select only numeric columns for transformation
    numeric_columns = data_copy.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        if data_copy[col].nunique() > 2:  # Check if column is continuous
            if transform == 'log':
                data_copy[col] = np.log1p(data_copy[col])  # Log transformation
            elif transform == 'loghalf':
                data_copy[col] = np.log1p(data_copy[col]) + 0.5  # Log transformation plus 0.5
            elif transform == 'boxcox':
                # Adding a small constant to avoid issues with zero or negative values
                data_copy[col], _ = stats.boxcox(data_copy[col] + 0.01)  # Box-Cox transformation
            elif transform == 'winsorize':
                data_copy[col] = winsorize(data_copy[col], limits=[0.00, 0.001])  # Winsorizing the data

    
# Check for near-zero variance in numeric columns only
    near_zero_variance_cols = data_copy[numeric_columns].var() <= 1e-8
    if near_zero_variance_cols.any():
        print("Warning: Columns with near-zero variance detected:", near_zero_variance_cols[near_zero_variance_cols].index.tolist())

    
    
    #Transform
    if algorithm == 'GAE':
        data_copy = data_copy.to_numpy()
    
    #Learn
    algo.learn(data_copy)

    # Extract the causal matrix
    causal_matrix = algo.causal_matrix
    if algorithm == 'DirectLiNGAM':
        weighted_causal_matrix = algo.weight_causal_matrix

    def print_rounded_matrix(matrix):
        # Round the matrix values to one decimal place
        rounded_matrix = [[round(value, 1) for value in row] for row in matrix]
    
        # Print the matrix in a square format
        for row in rounded_matrix:
            print(" ".join(f"{value:5}" for value in row))



        # Extract column names as variable names for labels
    variable_names = data.columns

    # Create a directed graph
    G = nx.DiGraph()
    
    # Add all variables as nodes to the graph
    for variable in variable_names:
        G.add_node(variable)
    
    # Add edges based on the causal matrix without weights
    for i, row in enumerate(causal_matrix):
        for j, col in enumerate(row):
            if col != 0:  # Nonzero entries indicate edges
                G.add_edge(variable_names[i], variable_names[j])


    # Create an adjacency matrix using networkx
    adjacency_matrix = nx.adjacency_matrix(G, nodelist=variable_names).toarray()

    # Create a heatmap of the adjacency matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(adjacency_matrix, cmap='Greens', interpolation='nearest', aspect='auto')

    # Add labels to the x and y axes
    plt.xticks(np.arange(len(variable_names)), variable_names, rotation=90)
    plt.yticks(np.arange(len(variable_names)), variable_names)

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Edge Weight')

    # Show the heatmap
    plt.title(f'Causal DAG ({algorithm} Algorithm): Left Causes Right')
    plt.show()


    if algorithm == 'DirectLiNGAM':
        print_rounded_matrix(weighted_causal_matrix)
        print(causal_matrix)
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    # Example list of variable_names
    # variable_names = ['var_one', 'var_two', 'var_three', ...]
    
    # Process variable_names to replace underscores with spaces and convert to sentence case
    formatted_variable_names = [name.replace('_', ' ').capitalize() for name in variable_names]
    
    # Additional heatmap for DirectLiNGAM using weighted causal matrix
    if weighted_causal_matrix is not None:
        # Define a custom colormap
        cmap = plt.cm.Blues
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[0, :3] = 1  # Set the color for zero values to white (RGB: 1, 1, 1)
        my_cmap = ListedColormap(my_cmap)
    
        # Create a heatmap of the weighted adjacency matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(weighted_causal_matrix, cmap=my_cmap, interpolation='none', aspect='auto')
    
        # Add labels to the x and y axes with formatted names
        plt.xticks(np.arange(len(formatted_variable_names)), formatted_variable_names, rotation=90)
        plt.yticks(np.arange(len(formatted_variable_names)), formatted_variable_names)
    
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Edge Weight')
    
        # Show the heatmap
        plt.title(f'Weighted Causal DAG ({algorithm} Algorithm): Edge Weights')
        plt.show()

        
        
    return G

# Example usage
G = causal_discovery(data, 'DirectLiNGAM', prior_knowledge = prior_knowledge, transform = 'loghalf')



#%% Causal priors





#%% Graph DAG
def graph_dag(G, outcome):
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import igraph as ig

    # Calculating various centrality measures
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Calculating DAG-specific measures (ancestral and descendant sets)
    ancestral_set_size = {node: len(nx.ancestors(G, node)) for node in G.nodes()}
    descendant_set_size = {node: len(nx.descendants(G, node)) for node in G.nodes()}

    # Calculating Hub and Authority Scores
    hubs, authorities = nx.hits(G, max_iter=1000)

    # Creating subplots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 18))
    centrality_measures = [in_degree_centrality, out_degree_centrality, closeness_centrality, betweenness_centrality, hubs, authorities]
    titles = ['In-Degree Centrality', 'Out-Degree Centrality', 'Closeness Centrality', 'Betweenness Centrality', 'Hub Scores', 'Authority Scores']

    # Plotting each centrality measure as a horizontal bar plot
    for i, ax in enumerate(axes.flatten()):
        if i >= len(centrality_measures):
            break  # Avoiding index error due to an extra subplot
        
        centrality = centrality_measures[i]
        # Sorting the nodes based on centrality values
        nodes_sorted = sorted(centrality.items(), key=lambda x: x[1], reverse=False)
        nodes, values = zip(*nodes_sorted)

        # Creating the bar plot
        cmap = plt.get_cmap('RdBu')
        colors = cmap(np.linspace(0, 1, len(nodes)))
        ax.barh(nodes, values, color=colors)
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()
        
        # Function to split label into two lines if needed
    def split_label_function(label, max_chars_per_line=15):
        if len(label) <= max_chars_per_line:
            return label
        split_index = label.rfind(' ', 0, max_chars_per_line)
        if split_index == -1:
            return label  # No space found, return original label
        return label[:split_index] + '\n' + label[split_index + 1:]
    
    # Apply topological layout and format labels
    for layer, nodes in enumerate(nx.topological_generations(G)):
        for node in nodes:
            G.nodes[node]["layer"] = layer
    
            # Reformatting and splitting labels
            formatted_label = node.replace('_', ' ').title()
            split_label = split_label_function(formatted_label)  # Use the function correctly
            G.nodes[node]["label"] = split_label
    
    pos = nx.multipartite_layout(G, subset_key="layer")
    
    # Draw the graph using the formatted and split labels
    plt.figure(figsize=(16, 8))
    nx.draw(G, pos, labels=nx.get_node_attributes(G, 'label'), with_labels=True, 
            node_size=4000, node_color='white', font_size=8, font_weight='regular', 
            edgecolors='black', linewidths=4, arrowstyle='-|>', arrowsize=20)
    plt.title('Estimated Causal Graph (Data + Prior Information)')
    plt.show()
    
    
    # Draw the graph using the spring layout
    pos_spring = nx.spring_layout(G)
    plt.figure(figsize=(16, 8))
    nx.draw(G, pos_spring, labels=nx.get_node_attributes(G, 'label'), with_labels=True, 
            node_size=4000, node_color='lightblue', font_size=8, font_weight='regular', 
            edgecolors='black', linewidths=4, arrowstyle='-|>', arrowsize=20)
    plt.title('Estimated Causal Graph (Data + Prior Information) - Spring Layout')
    plt.show()

    # Print Clustering Coefficient and Assortativity
    clustering_coefficient = nx.average_clustering(G)
    assortativity = nx.degree_assortativity_coefficient(G)
    print(f"Average Clustering Coefficient: {clustering_coefficient:.2f}")
    print(f"Assortativity Coefficient: {assortativity:.2f}")

    # Check if G is a DAG and print the result
    is_dag = nx.is_directed_acyclic_graph(G)
    print(f"Is the Graph a DAG?: {is_dag}")

    # If not a DAG, count and print the number of cycles
    if not nx.is_directed_acyclic_graph(G):
        cycle_count = len(list(nx.simple_cycles(G)))
        print(f"Number of Cycles: {cycle_count}")

        # Convert networkx graph to igraph
        ig_G = ig.Graph.from_networkx(G)

        # Store node names as attributes in igraph
        ig_G.vs["name"] = list(G.nodes())

        # Find the Feedback Arc Set using igraph
        fas = ig_G.feedback_arc_set()
        print(f"Minimum number of edges to remove to make G a DAG: {len(fas)}")

        # Print the edges that need to be removed
        print("Edges to be removed:")
        for edge in fas:
            source, target = ig_G.es[edge].tuple
            source_name = ig_G.vs[source]["name"]
            target_name = ig_G.vs[target]["name"]
            print(f"({source_name}, {target_name})")
    
    print()
    
    parents = list(G.predecessors(outcome))
    print(f"Parents of {outcome}:")
    for parent in parents:
        print(f"  {parent}")
    
    print()

    # Print all ancestors of the outcome node, each on a new line
    ancestors = list(nx.ancestors(G, outcome))
    print(f"Ancestors of {outcome}:")
    for ancestor in ancestors:
        print(f"  {ancestor}")


graph_dag(G, 'mrr_converted')


#%% Feature relevanced
def feature_relevance(data):
    import numpy as np, pandas as pd, networkx as nx
    from dowhy import gcm
    
    causal_model = gcm.InvertibleStructuralCausalModel(G_priors)
    gcm.auto.assign_causal_mechanisms(causal_model, data)
    gcm.fit(causal_model, data)
    
    parent_relevance, noise_relevance = gcm.parent_relevance(causal_model, target_node="mrr_latest")


feature_relevance(data)


#%% DoWhy Inference


def dowhy_inference(data, outcome, treatment, G, transform):
    import numpy as np
    import scipy.stats as stats  
    import networkx as nx
    from io import StringIO
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib.pyplot as plt

    from dowhy import CausalModel
    from econml.dml import SparseLinearDML, DML, CausalForestDML, NonParamDML, KernelDML
    from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
    from econml.inference import BootstrapInference
    
    from sklearn.linear_model import LinearRegression, LassoCV
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import PolynomialFeatures

    

# =============================================================================
# Make NetworkX graph a GML    
# =============================================================================
    
    G_gml = "\n".join(nx.generate_gml(G))
    
    # Create a copy of the data to avoid modifying the original DataFrame
    data_copy = data.copy()
    
    
    # Data Transformation with Log or LogHalf
    if transform == 'log':
        data_copy[outcome] = np.log1p(data_copy[outcome])
        data_copy[treatment] = np.log1p(data_copy[treatment])
    elif transform == 'loghalf':
        data_copy[outcome] = np.log(data_copy[outcome] + 0.5)
        data_copy[treatment] = np.log(data_copy[treatment] + 0.5)
    
    for col in data_copy.columns:
        if col not in [outcome, treatment] and data_copy[col].nunique() > 2:
            if transform == 'log':
                data_copy[col] = np.log1p(data_copy[col])
            elif transform == 'loghalf':
                data_copy[col] = np.log(data_copy[col] + 0.5)


# =============================================================================
# Make model
# =============================================================================
    
    model = CausalModel(
    data = data_copy,
    treatment = treatment,
    outcome = outcome,
    graph = G_gml
    )       

    model.view_model()

# =============================================================================
# Refute model
# =============================================================================

    # refuter_object = model.refute_graph(k=1, independence_test = 
    #                                     {'test_for_continuous': 'partial_correlation', 
    #                                       'test_for_discrete' : 'conditional_mutual_information'})
    # print(refuter_object)

# =============================================================================
# Identify estimand
# =============================================================================
    identified_estimand = model.identify_effect()
    print(identified_estimand)
 

# =============================================================================
# Estimate    
# =============================================================================

    estimate = model.estimate_effect(identified_estimand,
                                          method_name="backdoor.econml.dml.CausalForestDML",
                                          target_units = 'ate', 
                                          confidence_intervals=True,
                                          effect_modifiers =   [# Define the specific covariates you want to include in X
                                                'seniority_Department lead', 'seniority_Executive', 'seniority_Individual contributor', 
                                                'seniority_Team lead', 'role_CUSTOMER_SUCCESS', 'role_DESIGN', 'role_ENGINEERING', 
                                                'role_FINANCE', 'role_LEGAL', 'role_MANAGEMENT', 'role_MARKETING', 'role_OPERATIONS', 
                                                'role_OTHER', 'role_PRODUCT_MANAGEMENT', 'role_RESEARCH', 'role_SALES', 
                                                'role_STUDENT', 'role_SUPPORT', 'tag_count_f7d', 'reel_created_count_f7d'],
                                          method_params={"init_params":{'model_y':GradientBoostingRegressor(),
                                                                  'model_t': GradientBoostingRegressor(),
                                                                  'featurizer':PolynomialFeatures(degree=1, include_bias=True),
                                                                  'cv': 4},
                                                    "fit_params":{
                                                                    'inference': BootstrapInference(n_bootstrap_samples=100, n_jobs=-1),
                                                                }
                                                  })
    cate = estimate.cate_estimates
    print(estimate)
    print(cate)
    

# =============================================================================
# Refute estimates
# =============================================================================
    res_placebo = model.refute_estimate(identified_estimand, estimate,
            method_name="placebo_treatment_refuter", show_progress_bar=True, n_jobs = -1, placebo_type="permute")
    
    
    print(res_placebo)
    
    
    # res_subset = model.refute_estimate(
    # identified_estimand, estimate,
    # method_name="data_subset_refuter",
    # n_jobs=-1)
    
    # print(res_subset)
    
    
    # res_random = model.refute_estimate(
    # identified_estimand, estimate,
    # method_name="random_common_cause",
    # n_jobs=-1
    # )
    # print(res_random)
    
    # res_bootstrap = model.refute_estimate(
    # identified_estimand, estimate,
    # method_name="bootstrap_refuter",
    # num_simulations=1000,  # Number of bootstrap simulations
    # n_jobs=-1
    # )
    # print(res_bootstrap)
    
    
    print(res_placebo)
    # print(res_subset)
    # print(res_random)
    # print(res_bootstrap)




# =============================================================================
#   Return
# =============================================================================
    return estimate, identified_estimand


# Example usage of the function
estimate, identified_estimand = dowhy_inference(data, 'mrr_converted', 'invite_count_f7d', G_priors, transform='loghalf')






#%% EconML Inference

def econml_dml_inference(data, outcome, treatment, estimator,  transform):
    import numpy as np
    import scipy.stats as stats
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LassoCV
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.model_selection import GridSearchCV
    
    from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML
    # from econml.cate_interpreter import SingleTreeCateInterpreter
    # from econml.cate_interpreter import SingleTreePolicyInterpreter
    # import shap
    
    
# =============================================================================
#     Data preprocessing
# =============================================================================
    
    # Create a copy of the data to avoid modifying the original DataFrame
    data_copy = data.copy()
    
    # Data Transformation with Log
    if transform == 'log':
        data_copy[outcome] = np.log1p(data_copy[outcome])
        data_copy[treatment] = np.log1p(data_copy[treatment])
        for col in data_copy.columns:
            if col not in [outcome, treatment] and data_copy[col].nunique() > 2:
                data_copy[col] = np.log1p(data_copy[col])
    
    # New Transformation with Loghalf
    elif transform == 'loghalf':
        data_copy[outcome] = np.log1p(data_copy[outcome]) + 0.5
        data_copy[treatment] = np.log1p(data_copy[treatment]) + 0.5
        for col in data_copy.columns:
            if col not in [outcome, treatment] and data_copy[col].nunique() > 2:
                data_copy[col] = np.log1p(data_copy[col]) + 0.5

                

# =============================================================================
# Create train and test data
# =============================================================================
    
    # Define the specific covariates you want to include in X
    interaction_variables = [
        'seniority_Department lead', 'seniority_Executive', 'seniority_Individual contributor', 
        'seniority_Team lead', 'role_CUSTOMER_SUCCESS', 'role_DESIGN', 'role_ENGINEERING', 
        'role_FINANCE', 'role_LEGAL', 'role_MANAGEMENT', 'role_MARKETING', 'role_OPERATIONS', 
        'role_OTHER', 'role_PRODUCT_MANAGEMENT', 'role_RESEARCH', 'role_SALES', 
        'role_STUDENT', 'role_SUPPORT', 'tag_count_f7d', 'reel_created_count_f7d'
    ]
    
    # interaction_variables = [
    #     'highlight_count_f7d', 'tag_count_f7d', 'insight_count_f7d', 'reel_created_count_f7d', 'shared_object_note_count_f7d',
    #     'shared_object_insight_count_f7d'
    # ]
    
    # Extracting Y, T, and X with only the specified covariates
    Y = data_copy[outcome]
    T = data_copy[treatment]
    X = data_copy[interaction_variables]  # Only include the desired covariates in X
    
    # List all columns in data_copy
    all_columns = data_copy.columns.tolist()
    
    # Remove the columns that are in outcome, treatment, and confounders
    W_columns = [col for col in all_columns if col not in [outcome, treatment] + interaction_variables]
    
    # Create W with the remaining columns
    W = data_copy[W_columns]
    
    # Split data into train-validation
    from sklearn.model_selection import train_test_split
    X_train, X_test, T_train, T_test, Y_train, Y_test, W_train, W_test = train_test_split(X, T, Y, W, test_size=0.5)



# =============================================================================
# Reverse engineer graph representation
# =============================================================================


    import networkx as nx
    
    # Initialize the directed graph
    G_priors = nx.DiGraph()
    
    # Add nodes for each type of variable
    G_priors.add_node(outcome)  # Outcome variable
    G_priors.add_node(treatment)  # Treatment variable
    
    # Add nodes for covariates with their actual names
    for var in interaction_variables:
        G_priors.add_node(var)
    
    # Add nodes for confounders/other variables with their actual names
    for var in W_columns:
        G_priors.add_node(var)
    
    # Add edges based on the relationships
    G_priors.add_edge(treatment, outcome)  # Treatment affects outcome
    
    # Add edges for covariates affecting outcome and treatment
    for var in interaction_variables:
        G_priors.add_edge(var, outcome)  # Covariate affects outcome
        G_priors.add_edge(var, treatment)  # Covariate may affect treatment
    
    # Add edges for confounders affecting outcome and treatment
    for var in W_columns:
        G_priors.add_edge(var, outcome)  # Confounder affects outcome
        G_priors.add_edge(var, treatment)  # Confounder may affect treatment
    


# =============================================================================
# Set up estimators
# =============================================================================
    
#Cross validation
    rf_reg = lambda: GridSearchCV(
                estimator=RandomForestRegressor(),
                param_grid={
                        'max_depth': [5, 10, 15, None],
                        'n_estimators': (10, 30, 50, 100, 200),
                        'max_features': (1,2,3)
                    }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                )





    gb_reg = lambda: GridSearchCV(
                estimator=GradientBoostingRegressor(),
                param_grid={
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 4, 5],
                    'min_samples_split': [2, 4, 6],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'subsample': [0.8, 0.9, 1.0]
                },
                cv=5,
                n_jobs=-1,
                scoring='neg_mean_squared_error',
                verbose=1
            )


#Train
    if estimator == 'linear':
        est = LinearDML(model_y = GradientBoostingRegressor(),
                        model_t = GradientBoostingRegressor())
    elif estimator == 'sparse':
        est = SparseLinearDML(model_y = GradientBoostingRegressor(),
                              model_t = GradientBoostingRegressor(),
                              featurizer = PolynomialFeatures(degree=2),  # Modify degree as needed
                              random_state = 123)
    elif estimator == 'forest':
        est = CausalForestDML(model_y = GradientBoostingRegressor(),
                              model_t = GradientBoostingRegressor(),
                              featurizer = PolynomialFeatures(degree=2),  # Modify degree as needed
                              criterion ='mse', n_estimators = 1000,
                              min_impurity_decrease = 0.001, random_state = 0)
    else:
        raise ValueError("Invalid estimator. Choose 'linear', 'sparse', or 'forest'.")


# =============================================================================
# Train model on estimator
# =============================================================================

    # Fit the model
    est.fit(Y_train, T_train, X=X_train, W = W_train)


    # Get CATE estimates and confidence intervals
    cate_estimates = est.effect(X_test)
    lb, ub = est.effect_interval(X_test, alpha=0.05)
    
    # Assuming X_test is a NumPy array or similar, convert it to a DataFrame
    results = pd.DataFrame(X_test)
    
        # Add the CATE estimates and the bounds as new columns
    results['CATE_Estimates'] = cate_estimates
    results['Lower_Bound'] = lb
    results['Upper_Bound'] = ub
    results['invite_count_f7d'] = T_test
    results['mrr_converted'] = Y_test
    
    
# =============================================================================
# Tree interpreter
# =============================================================================


    # est.fit(Y, T, X=X, W=W)
    # intrp = SingleTreeCateInterpreter(include_model_uncertainty=True, max_depth=2, min_samples_leaf=10)
    
    # # We interpret the CATE model's behavior based on the features used for heterogeneity
    # intrp.interpret(est, X)
    
    # # Plot the tree
    # plt.figure(figsize=(25, 5))
    # intrp.plot(feature_names = interaction_variables, fontsize=12)
    # plt.show()


# =============================================================================
# Policy interepreter
# =============================================================================
    

    # # We find a tree-based treatment policy based on the CATE model
    # # sample_treatment_costs is the cost of treatment. Policy will treat if effect is above this cost.
    # intrp = SingleTreePolicyInterpreter(risk_level=None, max_depth=2, min_samples_leaf=1,min_impurity_decrease=.001)
    # intrp.interpret(est, X, sample_treatment_costs=0.02)
    # # Plot the tree
    # intrp.plot(feature_names=[interaction_variables], fontsize=12)



# =============================================================================
# SHAP values
# =============================================================================


    # shap_values = est.shap_values(X)
    # shap.summary_plot(shap_values['Y0']['T0'])


# =============================================================================
# Plotting    
# =============================================================================

# Assuming 'results' DataFrame and necessary variables (treatment, outcome) are already defined

    sns.set(style="darkgrid")

# =============================================================================
# Line Plot for CATE Estimates
# =============================================================================
    plt.figure(figsize=(12, 6))
    for column in results.columns:
        if column != 'invite_count_f7d' and results[column].nunique() == 2:
            subset = results[results[column] == 1]
            sns.lineplot(x='invite_count_f7d', y='CATE_Estimates', data=subset, label=column)
    
    plt.xlabel('Invite Count (7 days)')
    plt.ylabel('$ Uplift per Invite')
    plt.title('CATE Estimates by Treatment and Dummy Variables')
    plt.legend()
    plt.show()

# =============================================================================
# Box Plot for CATE Estimates by Segment
# =============================================================================

# Reshape for Box Plot
    melted_results = pd.melt(results, id_vars=['CATE_Estimates'], value_vars=[col for col in results.columns if col not in ['CATE_Estimates', 'invite_count_f7d', 'Lower_Bound', 'Upper_Bound']], 
                                 var_name='Dummy_Variable', value_name='Value')
    
    # Filter out rows where Value is 0
    melted_results = melted_results[melted_results['Value'] == 1]
    
    # Calculate medians and sort
    medians = melted_results.groupby('Dummy_Variable')['CATE_Estimates'].median().sort_values(ascending=False)
    sorted_dummies = medians.index.tolist()
    
    # Calculate the Average Treatment Effect (ATE)
    ATE = results['CATE_Estimates'].mean() / 10  # Dividing by 10 for elasticity
    
    # Format the ATE as a percentage with no decimal places
    ATE_percent = f"{ATE:.0%}"
    
    # Format treatment and outcome names
    formatted_treatment = treatment.replace('_', ' ').title()
    formatted_outcome = outcome.replace('_', ' ').title()
    
    # New plot title
    new_title = f"A 10% increase in {formatted_treatment} drives a {ATE_percent} uplift in converted revenue"
    
    # Creating the box plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(y='Dummy_Variable', x='CATE_Estimates', data=melted_results, palette="Set2", orient='h', order=sorted_dummies)
    
    plt.ylabel('Segment')
    plt.xlabel('$ Revenue Uplift (Elasticity)')
    plt.title(new_title)
    plt.show()


    
# =============================================================================
#  Return graph and results   
# =============================================================================

    return G_priors,  results



G_priors, results = econml_dml_inference(data, 'mrr_converted', 'invite_count_f7d', estimator = 'forest', transform = 'loghalf')





#%

#%%

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def interpret(results, treatment, outcome, total_treatments=6000):
    # Ensure the necessary columns are present
    required_columns = ['CATE_Estimates', treatment, outcome]
    if not all(col in results.columns for col in required_columns):
        raise ValueError("Missing required columns in the results dataframe.")

    # Histogram of CATE_Estimates
    plt.figure(figsize=(10, 6))
    sns.histplot(results['CATE_Estimates'], kde=True)
    plt.title("Histogram of CATE Estimates")
    plt.xlabel("CATE Estimates")
    plt.ylabel("Frequency")
    plt.show()
    
        # CDF of CATE_Estimates
    plt.figure(figsize=(10, 6))
    sns.histplot(results['CATE_Estimates'], kde=True, cumulative=True, stat="density")
    plt.title("CDF of CATE Estimates")
    plt.xlabel("CATE Estimates")
    plt.ylabel("Cumulative Density")
    plt.show()

    # Scatter Plot - Treatment vs CATE Estimates
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=results[treatment], y=results['CATE_Estimates'])
    plt.title("Scatter Plot of Treatment vs CATE Estimates")
    plt.xlabel("Treatment")
    plt.ylabel("CATE Estimates")
    plt.show()

    # Scatter Plot - Outcome vs CATE Estimates
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=results[outcome], y=results['CATE_Estimates'])
    plt.title("Scatter Plot of Outcome vs CATE Estimates")
    plt.xlabel("Outcome")
    plt.ylabel("CATE Estimates")
    plt.show()

    # Qini Curve
    plt.figure(figsize=(10, 6))
    qini_curve(results, treatment, outcome, total_treatments)
    plt.title("Qini Curve")
    plt.xlabel("Proportion Targeted")
    plt.ylabel("Cumulative Uplift")
    plt.show()

def qini_curve(df, treatment, outcome, total_treatments):
    df_sorted = df.sort_values(by='CATE_Estimates', ascending=False)
    df_sorted['cumulative_treatment'] = df_sorted[treatment].cumsum()
    df_sorted['cumulative_control'] = (1 - df_sorted[treatment]).cumsum()
    df_sorted['cumulative_outcome_treatment'] = df_sorted[outcome].cumsum()
    df_sorted['cumulative_outcome_control'] = (1 - df_sorted[outcome]).cumsum()

    df_sorted['uplift'] = df_sorted['cumulative_outcome_treatment'] - (df_sorted['cumulative_outcome_control'] * df_sorted['cumulative_treatment'] / df_sorted['cumulative_control'])
    df_sorted['uplift'] = df_sorted['uplift'] * total_treatments / len(df_sorted)
    df_sorted['proportion'] = np.arange(1, len(df_sorted) + 1) / len(df_sorted)

    plt.plot(df_sorted['proportion'], df_sorted['uplift'], label="Qini Curve")
    plt.plot([0, 1], [0, df_sorted['uplift'].max()], linestyle='--', color='red', label="Random")

# Example usage
# interpret(results, 'invite_count_f7d', 'mrr_converted', 6000)

interpret(results, 'invite_count_f7d', 'mrr_converted')


