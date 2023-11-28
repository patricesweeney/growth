#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:09:41 2023

@author: patricksweeney
"""


#%%

def import_data_local():
    import pandas as pd
    file_path = '/Users/patricksweeney/growth/01_Acquisition/05_Causal activation/F7 Days.xlsx'
    data = pd.read_excel(file_path)
    
    columns_to_remove = ['paid_signup', 'converted']  # Replace with your column names
    data = data.drop(columns=columns_to_remove, axis=1)

    return data

data = import_data_local()



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
eda_pair_plot(data, 'winsorize', True)




#%%

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


#%%
def causal_discovery(data, algorithm, transform):
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    from castle.common import GraphDAG
    from castle.common.priori_knowledge import PrioriKnowledge
    from scipy import stats
    from castle.algorithms import PC, GES, ICALiNGAM, DirectLiNGAM, NotearsNonlinear, GOLEM, GAE, DAG_GNN, RL, ANMNonlinear, GraNDAG, Notears, MCSL, NotearsLowRank, PNL, GraNDAG

    # Make a copy of the data to avoid modifying the original DataFrame
    data_copy = data.copy()    
    
  #   # #Add priors
  #   priori_knowledge = PrioriKnowledge(n_nodes = len(data_copy.columns))
  # #  priori_knowledge.add_required_edges([(3, 4), (4,5), (5,7)])
  #   priori_knowledge.add_forbidden_edges([(0, 1), (0, 2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10), (0,11), (0,12)])
    
    # Select the algorithm based on the 'algorithm' argument
    if algorithm == 'PC': #fast
        algo = PC(variant='stable', alpha=0.01)
    elif algorithm == 'GES': #medium
        algo = GES(criterion='bic')
    elif algorithm == 'GAE': #very slow
        algo = GAE(input_dim = len(data_copy.columns))
    elif algorithm == 'ANMNonlinear':
        algo = ANMNonlinear() #broken
    elif algorithm == 'DirectLiNGAM':
        algo = DirectLiNGAM() #fast
    elif algorithm == 'ICALiNGAM':
        algo = ICALiNGAM()
    elif algorithm == 'Notears':
        algo = Notears()
    elif algorithm == 'NotearsLowRank':
        algo = NotearsLowRank()
    elif algorithm == 'NotearsNonlinear':
        algo = NotearsNonlinear()
    elif algorithm == 'GOLEM':
        algo = GOLEM(num_iter=2e4)
    elif algorithm == 'DAG_GNN':
        algo = DAG_GNN()
    elif algorithm == 'PNL':
        algo = PNL(device_type='cpu')
    elif algorithm == 'GRAN':
        d = {'model_name': 'NonLinGauss', 'nonlinear': 'leaky-relu', 'optimizer': 'sgd', 'norm_prod': 'paths', 'device_type': 'cpu'}
        algo = GraNDAG(input_dim = len(data_copy.columns))
    elif algorithm == 'RL':
        algo = RL(nb_epoch=2000)
    elif algorithm == 'MCSL':
        algo = MCSL(model_type='nn',
          iter_step=100,
          rho_thresh=1e20,
          init_rho=1e-5,
          rho_multiply=10,
          graph_thresh=0.5,
          l1_graph_penalty=2e-3)


    else:
        raise ValueError("Invalid algorithm specified")
    
    for col in data_copy.columns:
        if data_copy[col].nunique() > 2:  # Check if column is continuous
            if transform == 'log':
                data_copy[col] = np.log1p(data_copy[col])  # Log transformation
            elif transform == 'boxcox':
                data_copy[col] = stats.boxcox(data_copy[col] + 0.01)  # Box-Cox transformation
    
    
    
    # Check for columns with near-zero variance
    near_zero_variance_cols = data.columns[data.var() <= 1e-8]
    if not near_zero_variance_cols.empty:
        print("Warning: Columns with near-zero variance detected:", near_zero_variance_cols)
        # Consider handling these columns here
    
    
    #Transform
    if algorithm == 'GAE':
        data_copy = data_copy.to_numpy()
    
    #Learn
    algo.learn(data_copy)

    # Extract the causal matrix
    causal_matrix = algo.causal_matrix
    #weighted_causal_matrix = algo.weight_causal_matrix

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

        # Add edges with weights based on the causal matrix
    for i, row in enumerate(causal_matrix):
        for j, col in enumerate(row):
            if col != 0:  # Nonzero entries indicate edges
                weight = round(col, 3)  # Round the weight for readability
                G.add_edge(variable_names[i], variable_names[j], weight=weight)

    # Create an adjacency matrix using networkx
    adjacency_matrix = nx.adjacency_matrix(G, nodelist=variable_names).toarray()

    # Create a heatmap of the adjacency matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(adjacency_matrix, cmap='Greens', interpolation='none', aspect='auto')

    # Add labels to the x and y axes
    plt.xticks(np.arange(len(variable_names)), variable_names, rotation=90)
    plt.yticks(np.arange(len(variable_names)), variable_names)

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Edge Weight')

    # Show the heatmap
    plt.title(f'Causal DAG ({algorithm} Algorithm): Left Causes Right')
    plt.show()


    # if algorithm == 'DirectLiNGAM':
    #     print_rounded_matrix(weighted_causal_matrix)
    #     print(causal_matrix)
        
    return G

# Example usage
G = causal_discovery(data, 'RL', 'log')




#%%
def graph_dag(G):
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
    plt.title('Estimated Causal Graph')
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


graph_dag(G)





def gml(G):
    import networkx as nx
    G_gml = "\n".join(nx.generate_gml(G))
    print(G_gml)
    return G_gml

G_gml = gml(G)

#%%


def dowhy_inference(data, outcome, treatment, G, transform):
    import numpy as np
    import scipy.stats as stats
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    import networkx as nx
    
    from dowhy import CausalModel
    
    from econml.dml import SparseLinearDML
    from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
    from econml.inference import BootstrapInference
    
    from sklearn.linear_model import LinearRegression, LassoCV
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import PolynomialFeatures
    
    import warnings
    warnings.filterwarnings('ignore')
  
    import matplotlib.pyplot as plt
    
    
    G_gml = "\n".join(nx.generate_gml(G))

    # Create a copy of the data to avoid modifying the original DataFrame
    data_copy = data.copy()
    

    # Data Transformation with Log
    if transform == 'log':
        data_copy[outcome] = np.log1p(data_copy[outcome])
        data_copy[treatment] = np.log1p(data_copy[treatment])
        for col in data_copy.columns:
            if col not in [outcome, treatment] and data_copy[col].nunique() > 2:
                data_copy[col] = np.log1p(data_copy[col])

    # # Extracting Y, T, and X
    # Y = data_copy[outcome]
    # T = data_copy[treatment]
    # X = data_copy.drop(columns=[outcome, treatment])

    # # Split data into train-validation
    # X_train, X_test, T_train, T_test, Y_train, Y_test, data_train, data_test = train_test_split(X, T, Y, data_copy, test_size=0.3)

    
    #1. Create model
    model = CausalModel(
    data = data_copy,
    treatment = treatment,
    outcome = outcome,
    graph = G_gml
    )       

    model.view_model()

    
    #2. Identify estimand (Backdoor / Frontdoor / IV)
    identified_estimand = model.identify_effect()
    print(identified_estimand)
    

    #3. Estimate effect

    estimate = model.estimate_effect(identified_estimand,
                                         method_name="backdoor.econml.dml.DML",
                                         target_units = 'ate',
                                         confidence_intervals=True,
                                         method_params={"init_params":{'model_y':GradientBoostingRegressor(),
                                                                  'model_t': GradientBoostingRegressor(),
                                                                  "model_final": LassoCV(fit_intercept=False),
                                                                  'featurizer':PolynomialFeatures(degree=1, include_bias=True)},
                                                   "fit_params":{
                                                                   'inference': BootstrapInference(n_bootstrap_samples=100, n_jobs=-1),
                                                                }
                                                  })
    print(estimate)
    
    
    #4 Refute
    res_placebo=model.refute_estimate(identified_estimand, estimate,
            method_name="placebo_treatment_refuter", show_progress_bar=True, placebo_type="permute")
    print(res_placebo)
    
    return estimate, identified_estimand


# Example usage of the function
estimate, identified_estimand = dowhy_inference(data, 'mrr_converted', 'invite_count_f7d', G, transform='log')




#%%





#%%

def econml_dml_inference(data, outcome, treatment, estimator, G,  transform):
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LassoCV
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import PolynomialFeatures
    
    from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML

    # Create a copy of the data to avoid modifying the original DataFrame
    data_copy = data.copy()
    
    import networkx as nx
    G_gml = "\n".join(nx.generate_gml(G))

    # Data Transformation with Log
    if transform == 'log':
        data_copy[outcome] = np.log1p(data_copy[outcome])
        data_copy[treatment] = np.log1p(data_copy[treatment])
        for col in data_copy.columns:
            if col not in [outcome, treatment] and data_copy[col].nunique() > 2:
                data_copy[col] = np.log1p(data_copy[col])

    # Extracting Y, T, and X
    Y = data_copy[outcome]
    T = data_copy[treatment]
    X = data_copy.drop(columns=[outcome, treatment])

    # Split data into train-validation
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.4)
    
    
        # Selecting the estimator based on the argument
    if estimator == 'linear':
        est = LinearDML(model_y = GradientBoostingRegressor(),
                        model_t = GradientBoostingRegressor())
    elif estimator == 'sparse':
        est = SparseLinearDML(model_y = GradientBoostingRegressor(),
                              model_t = GradientBoostingRegressor(),
                              featurizer = PolynomialFeatures(degree=1),  # Modify degree as needed
                              random_state = 123)
    elif estimator == 'forest':
        est = CausalForestDML(model_y = GradientBoostingRegressor(),
                              model_t = GradientBoostingRegressor(),
                              criterion ='mse', n_estimators = 1000,
                              min_impurity_decrease = 0.001, random_state = 0)
    else:
        raise ValueError("Invalid estimator. Choose 'linear', 'sparse', or 'forest'.")

    
    est.fit(Y_train, T_train, X = X_train, W = None)
    
    
    
    #Get results
    results = est.effect(X_test)
    lb, ub = est.effect_interval(X_test, alpha=0.01)
    print(results)
    

    # Adding jitter to the treatment variable
    jitter = np.random.normal(scale=0.01, size=T_test.shape)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    # Plot the estimated treatment effects against the treatment variable with jitter
    plt.scatter(T_test + jitter, results, label='Estimated Treatment Effects', alpha=0.6)
    plt.fill_between(np.sort(T_test), lb[np.argsort(T_test)], ub[np.argsort(T_test)], alpha=0.1)
    
    # Label the axes with actual variable names
    plt.xlabel('Treatment: {}'.format(treatment))
    plt.ylabel('Estimated Effect on {}'.format(outcome))
    
    # Add a legend
    plt.legend()
    
    # Show the plot
    plt.show()


# Example usage of the function
econml_dml_inference(data, 'mrr_converted', 'invite_count_f7d', estimator = 'forest', G = G, transform ='log')



#%%

def econml_dr_inference(data, outcome, treatment, estimator, G,  transform):
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression, LassoCV
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor,  GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_absolute_percentage_error
    
    from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML
    from econml.dr import DRLearner, LinearDRLearner, SparseLinearDRLearner, ForestDRLearner


    # Create a copy of the data to avoid modifying the original DataFrame
    data_copy = data.copy()
    
    import networkx as nx
    G_gml = "\n".join(nx.generate_gml(G))

    # Data Transformation with Log
    if transform == 'log':
        data_copy[outcome] = np.log1p(data_copy[outcome])
        data_copy[treatment] = np.log1p(data_copy[treatment])
        for col in data_copy.columns:
            if col not in [outcome, treatment] and data_copy[col].nunique() > 2:
                data_copy[col] = np.log1p(data_copy[col])

    # Extracting Y, T, and X
    Y = data_copy[outcome]
    T = data_copy[treatment]
    X = data_copy.drop(columns=[outcome, treatment])

    # Split data into train-validation
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.4)
    
    
    #Lambda for cross-validation
    model_reg = lambda: GridSearchCV(
                estimator=RandomForestRegressor(),
                param_grid={
                        'max_depth': [3, None],
                        'n_estimators': (10, 50, 100)
                    }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                )
    model_clf = lambda: GridSearchCV(
                    estimator=RandomForestClassifier(min_samples_leaf=10),
                    param_grid={
                            'max_depth': [3, None],
                            'n_estimators': (10, 50, 100)
                        }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                    )
    
    
    # Selecting the estimator based on the argument
    if estimator == 'DR':
        est = DRLearner(model_regression = model_reg,
                        model_propensity = model_clf,
                        model_final = model_reg
                        )
    elif estimator == 'LinearDR':
        est = LinearDRLearner(model_regression = GradientBoostingRegressor(),
                        model_propensity = GradientBoostingClassifier(),
                        model_final = GradientBoostingRegressor()
                      )
    elif estimator == 'SparseLinearDR':
        est = SparseLinearDRLearner(model_regression = GradientBoostingRegressor(),
                        model_propensity = GradientBoostingClassifier(),
                       # model_final = GradientBoostingRegressor()
                        )
    elif estimator == 'ForestDR':
        est = ForestDRLearner(model_regression = GradientBoostingRegressor(),
                              model_propensity = GradientBoostingClassifier())
    else:
        raise ValueError("Invalid estimator. Choose 'DR', 'LinearDR', or 'SparselinearDR' or 'ForestDR.")
    
    est.fit(Y_train, T_train, X = X_train)
    
    #Get results
    results = est.effect(X_test)
    lb, ub = est.effect_interval(X_test, alpha = 0.01)
    

    # Adding jitter to the treatment variable
    jitter = np.random.normal(scale = 0.01, size = T_test.shape)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    # Plot the estimated treatment effects against the treatment variable with jitter
    plt.scatter(T_test + jitter, results, label='Estimated Treatment Effects', alpha=0.6)
    plt.fill_between(np.sort(T_test), lb[np.argsort(T_test)], ub[np.argsort(T_test)], alpha=0.1)
    
    # Label the axes with actual variable names
    plt.xlabel('Treatment: {}'.format(treatment))
    plt.ylabel('Estimated Effect on {}'.format(outcome))
    
    # Add a legend
    plt.legend()
    
    # Show the plot
    plt.show()

# Example usage of the function
econml_dr_inference(data, 'mrr_converted', 'invite_count_f7d', estimator = 'DR', G = G, transform = 'log')



