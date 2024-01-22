# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 23:21:59 2024

@author: Kazim
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(file_path):
    
    """
    Load a dataset from a CSV file.
    
    """
    return pd.read_csv(file_path)


def remove_null_values(df):
    
    """
    Remove null values from a DataFrame.

    """
    return df.dropna(axis=1, how="all").dropna()

def apply_clustering(df, n_clusters=3):
    
    """
    Apply KMeans clustering to a DataFrame.

    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_features = df.drop(columns=["Country Name"])
    labels = kmeans.fit_predict(df_features)
    return labels

def plot_clusters(df, labels, 
                  title="Clusters Based on GDP per Capita and CO2 Emissions"):
    """
    Plot clusters in a scatter plot.

    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df.iloc[:, 1], 
                          df.iloc[:, 2], c=labels, cmap='viridis', 
                          s=50, label='Clusters')
    plt.title(title, fontweight='bold')
    plt.xlabel("GDP per Capita", fontweight='bold')
    plt.ylabel("CO2 Emissions", fontweight='bold')
    legend_labels = [f'Cluster {i}' for i in range(len(np.unique(labels)))]
    plt.legend(handles=scatter.legend_elements()[0], title='Clusters', 
               labels=legend_labels)
    plt.show()
    

def fit_curve_to_cluster(df, labels, cluster_id, curve_function):
    
    """
    Fit a curve to a specific cluster and plot the result.

    """
    cluster_indices = np.where(labels == cluster_id)[0]
      
# finding enough data points for the curve fitting
    if len(cluster_indices) >= 3:
        cluster_data = df.iloc[cluster_indices]

        x_data = cluster_data.iloc[:, 1].values  # GDP per Capita
        y_data = cluster_data.iloc[:, 2].values  # CO2 Emissions

        # Fit a curve using curve_fit
        params, _ = curve_fit(curve_function, x_data, y_data)

        # Generate y values using the fitted parameters
        y_fit = curve_function(x_data, *params)

        # Plot the original cluster data
        plt.scatter(x_data, y_data, label=f'Cluster {cluster_id}', s=50)

        # Plot the fitted curve
        plt.plot(x_data, y_fit, color='red', linestyle='--', linewidth=2)

        plt.title(f'Curve Fit for Cluster {cluster_id}')
        plt.xlabel("GDP per Capita", fontweight='bold')
        plt.ylabel("CO2 Emissions", fontweight='bold')
        plt.legend()
        plt.show()
    else:
        print(f"Not enough data points in Cluster {cluster_id} for curve fitting")

# Define a curve function (example: quadratic function)
def quadratic_function(x, a, b, c):
    
    """
    Quadratic function for curve fitting.

    Parameters:
    - x (np.ndarray): Independent variable.
    - a, b, c (float): Parameters of the quadratic function.

    Returns:
    - np.ndarray: The computed values of the quadratic function.
    """
    return a * x**2 + b * x + c

# Load CO2 emissions dataset
co2_emissions_df = load_dataset("co2_emissions.csv")

# Load GDP per capita dataset
gdp_per_capita_df = load_dataset("gdp_per_capita.csv")

# Remove null values from both datasets
co2_emissions_df_cleaned = remove_null_values(co2_emissions_df)
gdp_per_capita_df_cleaned = remove_null_values(gdp_per_capita_df)

# Transpose the cleaned CO2 emissions dataset
co2_emissions_df_transposed = co2_emissions_df_cleaned.transpose()

# Display the transposed CO2 emissions dataset
print("Transposed CO2 Emissions Dataset:")
print(co2_emissions_df_transposed)

# Transpose the cleaned GDP per capita dataset
gdp_per_capita_df_transposed = gdp_per_capita_df_cleaned.transpose()

# Display the transposed GDP per capita dataset
print("\nTransposed GDP per Capita Dataset:")
print(gdp_per_capita_df_transposed)

# Apply clustering on both datasets
co2_emissions_labels = apply_clustering(co2_emissions_df_cleaned)
gdp_per_capita_labels = apply_clustering(gdp_per_capita_df_cleaned)

# Plot clusters for CO2 emissions
plot_clusters(co2_emissions_df_cleaned, 
              co2_emissions_labels, title = 'Cluster based on GDP Per Capita')

# Plot clusters for GDP per capita
plot_clusters(gdp_per_capita_df_cleaned, 
              gdp_per_capita_labels, title = 'Cluster based on C02 Emissions')

# Apply clustering on the combined dataset
combined_df = pd.merge(co2_emissions_df_cleaned, 
                       gdp_per_capita_df_cleaned, on="Country Name")
combined_labels = apply_clustering(combined_df)

# Plot clusters for combined data
plot_clusters(combined_df, combined_labels)

# Fit curves to clusters (example: quadratic function)
for cluster_id in np.unique(combined_labels):
    fit_curve_to_cluster(combined_df, combined_labels, cluster_id, 
                         quadratic_function)


"""
Generating a heatmap to visualize the correlation matrix of CO2 emissions data.
This function will select the relevant columns for the desired years and 
will calculate the correlation matrix. The correlation matrix is then displayed 
as a heatmap using Seaborn.

"""

# Select the relevant columns for the desired years
emissions_data = co2_emissions_df_cleaned[["1990", "1995", 
                                           "2000", "2005", "2010", "2015"]]

# Calculate the correlation matrix
correlation_matrix = emissions_data.corr()

# Create the heatmap using Seaborn
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap="viridis",
                      linewidths=0.5, cbar_kws={'orientation': 'vertical'})

# Move the y-axis labels to the top
heatmap.xaxis.tick_top()

# Customize the plot foe Heatmap
plt.title("Correlation Heatmap of CO2 Emissions (1990-2015)", y=1.08, fontweight='bold') 
plt.xlabel("Year", fontweight='bold')
plt.ylabel("Year", fontweight='bold')
plt.show()


"""Fitting a quadratic curve to GDP per capita data and visualize the forecast.

This function will takes GDP per capita data for China and Malaysia, fits a 
quadratic curve to the historical data, generates future years until 2040, and 
predicts GDP per capita using the fitted curve. The function then plots the 
actual data and forecast for both China and Malaysia.

"""

# Select GDP data for China
gdp_china_df = gdp_per_capita_df[gdp_per_capita_df['Country Name'] 
                                 == 'China']

# Select GDP data for Malaysia
gdp_malaysia_df = gdp_per_capita_df[gdp_per_capita_df['Country Name'] 
                                    == 'Malaysia']




# Extract years and GDP per capita values for China
years_china = gdp_china_df.columns[1:].astype(int)
gdp_per_capita_values_china = gdp_china_df.iloc[:, 1:].values.flatten()

# Extract years and GDP per capita values for Malaysia
years_malaysia = gdp_malaysia_df.columns[1:].astype(int)
gdp_per_capita_values_malaysia = gdp_malaysia_df.iloc[:, 1:].values.flatten()


#Fit the quadratic curve to the data for China and Malaysia
params_china, _ = curve_fit(quadratic_function, years_china, 
                            gdp_per_capita_values_china)
params_malaysia, _ = curve_fit(quadratic_function, years_malaysia, 
                               gdp_per_capita_values_malaysia)

# Generate future years until 2040
future_years = np.arange(1960, 2041)

# Predict GDP per capita using the fitted curve for China and Malaysia
predicted_gdp_per_capita_china = quadratic_function(future_years,
                                                    *params_china)
predicted_gdp_per_capita_malaysia = quadratic_function(future_years,
                                                       *params_malaysia)


'''
    Plotting GDP per capita data and forecast for China from the year 1960
till 2040
    
'''

plt.figure(figsize=(12, 6))
plt.plot(years_china, gdp_per_capita_values_china, label='Actual Data')
plt.plot(future_years, predicted_gdp_per_capita_china, linestyle='--', 
         label='Forecast')
plt.title('GDP per Capita Forecast China (1960-2040)', fontweight='bold')
plt.xlabel('Year', fontweight='bold')
plt.ylabel('GDP per Capita', fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()


'''
    Plotting GDP per capita data and forecast for Malaysia from the year 1960
till 2040
    
'''
plt.figure(figsize=(12, 6))
plt.plot(years_malaysia, gdp_per_capita_values_malaysia, label='Actual Data')
plt.plot(future_years, predicted_gdp_per_capita_malaysia, linestyle='--', 
         label='Forecast')
plt.title('GDP per Capita Forecast for Malaysia (1960-2040)', fontweight='bold')
plt.xlabel('Year', fontweight='bold')
plt.ylabel('GDP per Capita', fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()

"""
Fitting a quadratic curve to CO2 emissions data and visualize the forecast.

This function takes CO2 emissions data for China and Malaysia, fits a 
quadratic curve to the historical data, generates future years until 2040, 
and predicts CO2 emissions using the fitted curve. The function then plots 
the original data and forecast for both China and Malaysia.

"""

# Select CO2 emissions data for China
co2_emissions_china_df = co2_emissions_df[co2_emissions_df['Country Name'] 
                                          == 'China']

# Select CO2 emissions data for Malaysia
co2_emissions_malay_df = co2_emissions_df[co2_emissions_df['Country Name'] 
                                             == 'Malaysia']

# Extract years for CO2 emissions data for China
emissions_years_china = np.array([int(year) for year 
                                  in co2_emissions_china_df.columns[1:]])

# Extract CO2 values for China
co2_values_china = co2_emissions_china_df.iloc[0, 1:].values.astype(float)

# Extract years for CO2 emissions data for Malaysia
emissions_years_malaysia = np.array([int(year) for year 
                                     in co2_emissions_malay_df.columns[1:]])
# Extract CO2 values for Malaysia
co2_values_china = co2_emissions_china_df.iloc[0, 1:].values.astype(float)
co2_values_malaysia = co2_emissions_malay_df.iloc[0, 1:].values.astype(float)



# Check for NaN and replace them with zeros for China and Malaysia
co2_values_china[np.isnan(co2_values_china)] = 0

co2_values_malaysia[np.isnan(co2_values_malaysia)] = 0


# Fit the curve to the CO2 emissions data for China
params_china, covariance = curve_fit(quadratic_function, 
                                     emissions_years_china, co2_values_china)

# Fit the curve to the CO2 emissions data for Malaysia
params_malaysia, covariance = curve_fit(quadratic_function, 
                                        emissions_years_malaysia, 
                                        co2_values_malaysia)


''' Prediction of  CO2 emissions using the fitted curve and applying 
the time frame of the forecast'''

# Generating the time frame of forecast
future_years = np.arange(1990, 2041)


forecast_china = quadratic_function(future_years, *params_china)
forecast_malaysia = quadratic_function(future_years, *params_malaysia)


''' Plotting the original data for CO2 Emission from 1990 to 2021 and the 
forecast for china'''

plt.figure(figsize=(12, 6))
plt.plot(emissions_years_china[(emissions_years_china >= 1990) & 
                               (emissions_years_china <= 2019)], 
         co2_values_china[(emissions_years_china >= 1990) & 
                          (emissions_years_china <= 2019)], 
         label='Original Data (1990-2019)')
plt.plot(future_years, forecast_china, linestyle='--', 
         color='red', label='Forecast (1990-2040)')
plt.title('CO2 Emissions Forecast for China', fontweight='bold')
plt.xlabel('Year', fontweight='bold')
plt.ylabel('CO2 Emissions', fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()


''' Plotting the original data of CO2 Emission from 1990 to 2021 and 
the forecast for Malaysia'''

plt.figure(figsize=(12, 6))
plt.plot(emissions_years_malaysia[(emissions_years_malaysia >= 1990) & 
                                  (emissions_years_malaysia <= 2019)], 
         co2_values_malaysia[(emissions_years_malaysia >= 1990) & 
                             (emissions_years_malaysia <= 2019)], 
         label='Original Data (1990-2019)')
plt.plot(future_years, forecast_malaysia, linestyle='--', 
         color='red', label='Forecast (1990-2040)')
plt.title('CO2 Emissions Forecast for Malaysia', fontweight='bold')
plt.xlabel('Year', fontweight='bold')
plt.ylabel('CO2 Emissions', fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()
