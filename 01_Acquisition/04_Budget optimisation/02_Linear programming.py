import numpy as np
from scipy.optimize import linprog

# Parameters for the channels and other constants
channels = {
    'SEO': {
        'C1_mean': 0.015,
        'C2_mean': 0.04,
        'MRR': 300,
        'Retention': 0.95,
        'CPC_mean': 1.30,  # CPC instead of CAC
        'max_traffic': 2000000
    },
    'Performance': {
        'C1_mean': 0.28,
        'C2_mean': 0.01,
        'MRR': 300,
        'Retention': 0.95,
        'CPC_mean': 6,  # CPC instead of CAC
        'max_traffic': 5000000
    }
}

budget = 5000000
WACC = 0.01

# Objective function coefficients
# We want to maximize the (LTV - CPC * Traffic) for all channels
# Since linprog minimizes the function, we use negative values to represent maximization
c = [-(
        channel['C1_mean'] * channel['C2_mean'] *
        (channel['MRR'] * channel['Retention'] / (1 + WACC - channel['Retention'])) -
        channel['CPC_mean']
    ) for channel in channels.values()]

# Constraints for the upper bound on traffic based on max traffic
A_ub = np.eye(len(channels))  # Identity matrix for individual traffic constraints
b_ub = [channel['max_traffic'] for channel in channels.values()]

# Budget constraint (sum of all traffic * CPC should be less than or equal to the budget)
A_eq = [[channel['CPC_mean'] for channel in channels.values()]]
b_eq = [budget]

# Bounds for each traffic variable must be >= 0
bounds = [(0, None) for _ in channels]

# Optimization
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

if result.success:
    optimal_traffic = result.x
    print(f"Optimal traffic allocations: {optimal_traffic}")

    # Calculating total CPC, total expected acquisitions, and expected LTV per channel
    total_CPC_per_channel = {}
    total_expected_acquisitions_per_channel = {}
    expected_LTV_per_channel = {}

    for i, (channel_name, channel_data) in enumerate(channels.items()):
        traffic = optimal_traffic[i]
        total_CPC_per_channel[channel_name] = traffic * channel_data['CPC_mean']
        total_expected_acquisitions_per_channel[channel_name] = traffic * channel_data['C1_mean'] * channel_data['C2_mean']
        expected_LTV_per_channel[channel_name] = traffic * (
            channel_data['C1_mean'] * channel_data['C2_mean'] *
            (channel_data['MRR'] * channel_data['Retention'] /
            (1 + WACC - channel_data['Retention']))
        ) - total_CPC_per_channel[channel_name]

    # Print the results
    print("Total Cost per Channel:")
    for channel, cost in total_CPC_per_channel.items():
        print(f"{channel}: {cost}")

    print("\nTotal Expected Acquisitions per Channel:")
    for channel, acquisitions in total_expected_acquisitions_per_channel.items():
        print(f"{channel}: {acquisitions}")

    print("\nExpected LTV per Channel (before conversion):")
    for channel, ltv in expected_LTV_per_channel.items():
        print(f"{channel}: {ltv}")

    # Check if total budget is exceeded
    total_spent = sum(total_CPC_per_channel.values())
    print(f"\nTotal Budget Spent: {total_spent}")
    print("Budget Exceeded!" if total_spent > budget else "Within Budget.")
else:
    print("Optimization was not successful. Please check the inputs and constraints.")
