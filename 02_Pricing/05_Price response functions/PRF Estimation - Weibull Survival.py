#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 20:26:39 2023
@author: patricksweeney
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# Set Streamlit app title and page style
st.set_page_config(
    page_title="Price Response Function Estimator",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

#CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');
        
        body, .css-17eq0hr, .css-qbe2hs {
            font-family: 'Poppins', sans-serif;
            font-weight: 400;
        }
        
        h1, .css-5q7aqi {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
        }
    </style>
    """, unsafe_allow_html=True)

# Define Streamlit app header
st.title("Price Response Function Estimator")
st.write("The Price Response Function estimates how demand for a product changes as its price changes. It helps in optimizing pricing strategies.")

# Create a file uploader widget
uploaded_file = st.file_uploader("Upload a file (CSV or Excel)")

def to_percent(y, position):
    return f"{100 * y:.0f}%"
    
formatter = FuncFormatter(to_percent)

def plot_data_and_elasticity(sorted_data, survival_function):
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Set background color to white
    fig.patch.set_facecolor('white')

    
    # Plot Survival Function
    ax1.plot(sorted_data, survival_function, marker="o", linestyle="-", color="black")
    ax1.set_xlabel("Willingness to Pay", fontsize=12)
    ax1.set_ylabel("% of Market", fontsize=12)
    ax1.yaxis.set_major_formatter(formatter)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Plot Elasticity
    elasticity = np.abs(np.diff(np.log(survival_function)) / np.diff(np.log(sorted_data)))
    ax2.plot(sorted_data[:-1], elasticity, marker="o", linestyle="-", color="black")
    ax2.set_xlabel("Willingness to Pay", fontsize=12)
    ax2.set_ylabel("Absolute Elasticity", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Display the subplots
    st.pyplot(fig)

from scipy.optimize import minimize
import numpy as np

# Negative log-likelihood function for the Weibull distribution
def neg_log_likelihood(params, data):
    k, lambda_ = params
    n = len(data)
    log_lik = -n * np.log(k) - n * k * np.log(lambda_) + (k - 1) * np.sum(np.log(data)) - (1 / lambda_) * np.sum(data ** k)
    return -log_lik

# Function to fit Weibull parameters using MLE
def fit_weibull(data):
    initial_guess = [1, 1]
    result = minimize(neg_log_likelihood, initial_guess, args=(data), method='L-BFGS-B', bounds=[(0.01, 10), (0.01, 10)])
    k, lambda_ = result.x
    return k, lambda_

# Function to calculate Weibull survival function
def weibull_survival(data, k, lambda_):
    return np.exp(-((data / lambda_) ** k))

# Function to calculate Weibull elasticity
def weibull_elasticity(data, k, lambda_):
    return k / data


# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded file into a DataFrame
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        df = pd.read_excel(uploaded_file)

    # Check if the DataFrame has at least two columns
    if df.shape[1] < 2:
        st.error("Please ensure the uploaded file contains at least two columns.")
    else:
        # Find the first numeric column
        numeric_column = None
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_column = col
                break

        # Check if a numeric column was found
        if numeric_column is None:
            st.error("No numeric column found in the uploaded file.")
        else:
            # Data preprocessing
            sorted_data = np.sort(df[numeric_column])
            num_samples = len(sorted_data)

            # Calculate the survival function
            survival_function = np.arange(num_samples, 0, -1) / num_samples

             # Create and display the initial subplots
            plot_data_and_elasticity(sorted_data, survival_function)
        
            # Fit Weibull parameters using MLE
            k, lambda_ = fit_weibull(sorted_data)
            
            # Calculate Weibull survival function and elasticity
            weibull_survival_function = weibull_survival(sorted_data, k, lambda_)
            weibull_elasticity = weibull_elasticity(sorted_data, k, lambda_)
        
            # Create additional subplots for Weibull survival function and its elasticity
            fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
            
            # Plot Weibull Survival Function
            ax3.plot(sorted_data, weibull_survival_function, marker='o', linestyle='-', color='red')
            ax3.set_xlabel('Willingness to Pay', fontsize=12)
            ax3.set_ylabel('% of Market (Weibull)', fontsize=12)
            ax3.yaxis.set_major_formatter(formatter)  # Format as percentage
            ax3.set_ylim(0, 1)  # Set y-axis limits to 0 to 1
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # Plot Weibull Elasticity
            ax4.plot(sorted_data, weibull_elasticity, marker='o', linestyle='-', color='red')
            ax4.set_xlabel('Willingness to Pay', fontsize=12)
            ax4.set_ylabel('Absolute Elasticity (Weibull)', fontsize=12)
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            # Display the additional subplots
            st.pyplot(fig)