#!/usr/bin/env python
# coding: utf-8

# ##### ONE SAMPLE t-TEST ANALYZER

# In[1]:


import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

def main():
    st.title("One-Sample t-Test Analyzer")
    st.write("This app performs a one-sample t-test with options to upload data or enter statistics manually.")
    
    # User inputs - data source selection
    st.sidebar.header("Data Source")
    data_source = st.sidebar.radio("Choose data input method:", 
                                 ["Upload Dataset", "Enter Statistics Manually"])
    
    sample_mean = None
    sample_std = None
    sample_size = None
    
    if data_source == "Upload Dataset":
        st.sidebar.header("Dataset Parameters")
        uploaded_file = st.sidebar.file_uploader("Upload your data file (CSV or Excel)", 
                                               type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                # Read the file based on extension
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:  # Excel
                    df = pd.read_excel(uploaded_file)
                
                # Let user select column
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    selected_col = st.sidebar.selectbox("Select column to analyze", numeric_cols)
                    
                    # Calculate statistics from data
                    sample_data = df[selected_col].dropna()
                    sample_mean = np.mean(sample_data)
                    sample_std = np.std(sample_data, ddof=1)  # sample std dev
                    sample_size = len(sample_data)
                    
                    # Display data summary
                    st.subheader("Data Summary")
                    st.write(f"Selected column: **{selected_col}**")
                    st.write(f"Sample size: {sample_size}")
                    st.write(f"Calculated mean: {sample_mean:.4f}")
                    st.write(f"Calculated standard deviation: {sample_std:.4f}")
                    
                    # Show data preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head())
                    
                    # Show distribution plot
                    fig, ax = plt.subplots()
                    ax.hist(sample_data, bins='auto', edgecolor='black')
                    ax.set_title(f"Distribution of {selected_col}")
                    ax.set_xlabel("Value")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                else:
                    st.error("The uploaded file contains no numeric columns.")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            st.info("Please upload a data file to proceed.")
    else:  # Manual entry
        st.sidebar.header("Manual Input Parameters")
        sample_mean = st.sidebar.number_input("Sample Mean", value=0.0)
        sample_std = st.sidebar.number_input("Sample Standard Deviation", value=1.0, min_value=0.01)
        sample_size = st.sidebar.number_input("Sample Size (n)", value=30, min_value=2, step=1)
    
    # Common test parameters (only shown if we have data)
    if sample_mean is not None and sample_std is not None and sample_size is not None:
        st.sidebar.header("Test Parameters")
        hypothesized_mean = st.sidebar.number_input("Hypothesized Population Mean (μ₀)", 
                                                 value=0.0)
        alpha = st.sidebar.number_input("Significance Level (α)", 
                                      value=0.05, min_value=0.001, max_value=0.5, step=0.01)
        test_type = st.sidebar.radio("Test Type", 
                                   ["Two-tailed (μ ≠ μ₀)", 
                                    "Left-tailed (μ < μ₀)", 
                                    "Right-tailed (μ > μ₀)"])
        
        # Calculate test statistics
        st.header("Test Results")
        df = sample_size - 1  # degrees of freedom
        sem = sample_std / np.sqrt(sample_size)  # standard error of the mean
        t_stat = (sample_mean - hypothesized_mean) / sem
        
        # Calculate critical values and p-value based on test type
        if test_type == "Two-tailed (μ ≠ μ₀)":
            critical_lower = stats.t.ppf(alpha/2, df)
            critical_upper = stats.t.ppf(1 - alpha/2, df)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            reject = (t_stat < critical_lower) or (t_stat > critical_upper)
        elif test_type == "Left-tailed (μ < μ₀)":
            critical_lower = stats.t.ppf(alpha, df)
            p_value = stats.t.cdf(t_stat, df)
            reject = t_stat < critical_lower
            critical_upper = None
        else:  # Right-tailed
            critical_upper = stats.t.ppf(1 - alpha, df)
            p_value = 1 - stats.t.cdf(t_stat, df)
            reject = t_stat > critical_upper
            critical_lower = None
        
        # Display test results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("t-statistic", f"{t_stat:.4f}")
            st.metric("Degrees of Freedom", df)
            st.metric("Standard Error", f"{sem:.4f}")
        with col2:
            st.metric("p-value", f"{p_value:.4f}")
            st.metric("Significance Level", f"{alpha:.3f}")
            st.metric("Conclusion", "Reject H₀" if reject else "Fail to reject H₀")
        
        # Display critical values
        st.subheader("Critical Values")
        if test_type == "Two-tailed (μ ≠ μ₀)":
            st.write(f"Lower critical value: {critical_lower:.4f}")
            st.write(f"Upper critical value: {critical_upper:.4f}")
        elif test_type == "Left-tailed (μ < μ₀)":
            st.write(f"Critical value: {critical_lower:.4f}")
        else:
            st.write(f"Critical value: {critical_upper:.4f}")
        
        # Visualization
        st.header("Distribution Visualization")
        
        # Create a range of t-values for plotting
        t_values = np.linspace(-4, 4, 500) if df > 0 else np.linspace(-4, 4, 500)
        pdf = stats.t.pdf(t_values, df)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the t-distribution
        ax.plot(t_values, pdf, label=f"t-distribution (df={df})", color='blue')
        
        # Fill the rejection regions
        if test_type == "Two-tailed (μ ≠ μ₀)":
            ax.fill_between(t_values, pdf, where=(t_values <= critical_lower), 
                          color='red', alpha=0.5, label='Rejection Region')
            ax.fill_between(t_values, pdf, where=(t_values >= critical_upper), 
                          color='red', alpha=0.5)
        elif test_type == "Left-tailed (μ < μ₀)":
            ax.fill_between(t_values, pdf, where=(t_values <= critical_lower), 
                          color='red', alpha=0.5, label='Rejection Region')
        else:
            ax.fill_between(t_values, pdf, where=(t_values >= critical_upper), 
                          color='red', alpha=0.5, label='Rejection Region')
        
        # Add the t-statistic
        ax.axvline(x=t_stat, color='green', linestyle='--', 
                  label=f't-statistic = {t_stat:.2f}')
        
        # Add critical values
        if critical_lower is not None:
            ax.axvline(x=critical_lower, color='black', linestyle=':', 
                      label=f'Critical value = {critical_lower:.2f}')
        if critical_upper is not None:
            ax.axvline(x=critical_upper, color='black', linestyle=':', 
                      label=f'Critical value = {critical_upper:.2f}')
        
        ax.set_title(f"t-Distribution with {test_type.split(' ')[0]} Test")
        ax.set_xlabel("t-value")
        ax.set_ylabel("Probability Density")
        ax.legend()
        
        st.pyplot(fig)
        
        # Interpretation
        st.header("Interpretation")
#         st.write(f"**Null Hypothesis (H₀):** The population mean is equal to {hypothesized_mean:.2f}")
        
        if test_type == "Two-tailed (μ ≠ μ₀)":
            st.write(f"**Null Hypothesis (H₀):** The population mean is equal to {hypothesized_mean:.2f}")
            st.write(f"**Alternative Hypothesis (H₁):** The population mean is not equal to {hypothesized_mean:.2f}")
        elif test_type == "Left-tailed (μ < μ₀)":
            st.write(f"**Null Hypothesis (H₀):** The population mean = or > {hypothesized_mean:.2f}")
            st.write(f"**Alternative Hypothesis (H₁):** The population mean < {hypothesized_mean:.2f}")
        else:
            st.write(f"**Null Hypothesis (H₀):** The population mean = or < {hypothesized_mean:.2f}")
            st.write(f"**Alternative Hypothesis (H₁):** The population mean > {hypothesized_mean:.2f}")
        
        st.write(f"With a significance level of α = {alpha:.3f}, we {'reject' if reject else 'fail to reject'} the null hypothesis.")
        st.write(f"The p-value of {p_value:.4f} is {'less' if p_value < alpha else 'greater'} than α = {alpha:.3f}.")
        
        if reject:
            st.success("Conclusion: There is statistically significant evidence to reject the null hypothesis in favor of the alternative hypothesis.")
        else:
            st.info("Conclusion: There is not enough evidence to reject the null hypothesis.")

if __name__ == "__main__":
    main()

