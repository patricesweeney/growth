#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:25:31 2023

@author: patricksweeney
"""

import numpy as np

def impact(elasticity_matrix, quantity_vector, price_vector, price_change_vector):
    def calculate_quantity_change(elasticity_matrix, quantity_vector, price_change_vector):
        """
        Calculate the change in quantities demanded based on elasticity matrix, initial quantity vector, 
        and price change vector.

        Parameters:
        elasticity_matrix (numpy.ndarray): A square matrix where diagonal elements are own-price elasticities 
                                           and off-diagonal elements are cross-price elasticities.
        quantity_vector (numpy.ndarray): A column vector representing the initial quantities of products.
        price_change_vector (numpy.ndarray): A column vector representing the percentage change in prices of products.

        Returns:
        numpy.ndarray: A column vector representing the changes in quantities for each product.
        """
        E = np.array(elasticity_matrix)
        Q = np.array(quantity_vector)
        ΔP = np.array(price_change_vector)

        intermediate_result = np.dot(E, ΔP)
        return np.multiply(intermediate_result, Q)

    # Calculate the change in quantities
    dv = calculate_quantity_change(elasticity_matrix, quantity_vector, price_change_vector)

    # Calculating revenues and changes
    original_revenue = np.dot(price_vector, quantity_vector)
    new_quantity_vector = np.add(quantity_vector, dv)
    new_revenue = np.dot(price_vector, new_quantity_vector)
    revenue_change_percentage = ((new_revenue - original_revenue) / original_revenue) * 100

    # Calculating average prices and changes
    original_average_price = np.mean(price_vector)
    new_price_vector = np.add(price_vector, np.multiply(price_vector, price_change_vector))
    new_average_price = np.mean(new_price_vector)
    average_price_change_percentage = ((new_average_price - original_average_price) / original_average_price) * 100

    # Calculating volumes and changes
    original_volume = np.sum(quantity_vector)
    new_volume = np.sum(new_quantity_vector)
    volume_change_percentage = ((new_volume - original_volume) / original_volume) * 100

    # Printing the results
    print("Original Revenue: ${:.2f}".format(original_revenue))
    print("New Revenue: ${:.2f}".format(new_revenue))
    print("% Change in Revenue: {:.2f}%".format(revenue_change_percentage))
    print()
    
    print("Original Average Price: ${:.2f}".format(original_average_price))
    print("New Average Price: ${:.2f}".format(new_average_price))
    print("% Change in Average Price: {:.2f}%".format(average_price_change_percentage))
    print()
    
    print("Original Volume: {:.2f} units".format(original_volume))
    print("New Volume: {:.2f} units".format(new_volume))
    print("% Change in Volume: {:.2f}%".format(volume_change_percentage))

# Example usage of the function
elasticity_matrix_example = [
    [-1.9, 1.2, 0.9],
    [0.5, -0.8, 0.3],
    [0.5, 0.4, -1.2]
]
quantity_vector_example = [100, 50, 30]
price_vector_example = [30, 50, 100]
price_change_vector_example = [0.5, 0, 0]

impact(elasticity_matrix_example, quantity_vector_example, price_vector_example, price_change_vector_example)
