import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, lognorm, beta

# Parameters for the channels - replace with your actual parameters
channels = {
    'channel_1': {
        'C1_mean': 0.1, 'C1_std': 0.02,
        'C2_mean': 0.2, 'C2_std': 0.03,
        'MRR_shape': 0.5, 'MRR_scale': 1000, 'MRR_loc': 0,
        'Retention_alpha': 2, 'Retention_beta': 5,
        'CAC_mean': 500, 'CAC_std': 100,
        'max_traffic': 1000
    },
    
    'channel_2': {
        'C1_mean': 0.1, 'C1_std': 0.02,
        'C2_mean': 0.15, 'C2_std': 0.03,
        'MRR_shape': 0.5, 'MRR_scale': 1000, 'MRR_loc': 0,
        'Retention_alpha': 2, 'Retention_beta': 5,
        'CAC_mean': 500, 'CAC_std': 100,
        'max_traffic': 1000
    },
    # Add more channels with their parameters here
}

# Total budget
budget = 1000000  # Replace with your actual budget

# Function to sample from the distributions
def sample_distributions(channel_params):
    C1 = norm.rvs(channel_params['C1_mean'], channel_params['C1_std'])
    C2 = norm.rvs(channel_params['C2_mean'], channel_params['C2_std'])
    MRR = lognorm.rvs(channel_params['MRR_shape'], scale=channel_params['MRR_scale'])
    Retention = beta.rvs(channel_params['Retention_alpha'], channel_params['Retention_beta'])
    CAC = norm.rvs(channel_params['CAC_mean'], channel_params['CAC_std'])
    return C1, C2, MRR, Retention, CAC

# Objective function
def objective_function(allocations, channels):
    clv_sum = 0
    for i, allocation in enumerate(allocations):
        channel = channels['channel_' + str(i+1)]
        C1, C2, MRR, Retention, CAC = sample_distributions(channel)
        WACC = 0.1  # Replace with your actual WACC
        clv = (C1 * C2 * (MRR * Retention / (1 + WACC - Retention))) - CAC
        clv_sum += allocation * clv
    return -clv_sum  # Negative because we want to maximize

# Constraints
def budget_constraint(allocations):
    return budget - np.dot(allocations, [channel['CAC_mean'] for channel in channels.values()])

def traffic_constraints(allocations):
    return [channel['max_traffic'] - allocation for allocation, channel in zip(allocations, channels.values())]

# Initial guess for the allocations
initial_allocations = np.full(len(channels), budget / len(channels))

# Optimization constraints
constraints = [{'type': 'eq', 'fun': budget_constraint}] + \
              [{'type': 'ineq', 'fun': lambda x, i=i: traffic_constraints(x)[i]} for i in range(len(channels))]

# Run the optimization
result = minimize(objective_function, initial_allocations, args=(channels,), constraints=constraints)

# Check if the optimization was successful
if result.success:
    optimal_allocations = result.x
    print("Optimal Allocation:", optimal_allocations)
else:
    print("Optimization failed:", result.message)
