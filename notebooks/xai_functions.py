import random
import numpy as np
import pandas as pd

def calculate_pdp_ice(data, model, feature_name, n_ice_samples=50):
    """
    Calculate the Partial Dependence Plot (PDP) curve and Individual Conditional Expectation (ICE) curves for a given feature.

    Parameters:
    - data (pandas.DataFrame): The dataset containing the feature and target values.
    - model: The trained machine learning model used for prediction.
    - feature_name (str): The name of the feature for which to calculate the PDP and ICE curves.
    - n_ice_samples (int): The number of samples to use for generating ICE curves. Default is 50.

    Returns:
    - pdp_curve (list): The PDP curve, representing the average predicted target values for each unique value of the feature.
    - centered_ice_curves (numpy.ndarray): The ICE curves, representing the predicted target values for each instance at each unique value of the feature.
    - xax (numpy.ndarray): The x-axis values for the PDP curve, corresponding to the unique values of the feature.

    """

    # Prepare data
    n_samples = len(data) # Get the number of instances in the dataset
    data_sorted = data.sort_values(by=feature_name) # Order the rows of the data according to the value of the target feature for montonic plots
    feature_values = data_sorted[feature_name] # Get the target feature values
    feature_values_unique = feature_values.unique() # Get the unique values of the target feature 
    xax = feature_values_unique.copy() # These are the x-axis values for the PDP curve
    n_unique_values = len(feature_values_unique) # Get the number of unique values of the target feature

    # Calculate PDP curve
    pdp_curve = [] # List to store the PDP curve
    for value in feature_values_unique: # Loop over the unique values of the target feature
        data_sorted[feature_name] = value # Replace the entire column with the target feature value for a fast vectorized operation
        predictions = model.predict(data_sorted) # Predict the target values for all instances at once
        pdp_curve.append(np.mean(predictions)) # Calculate the average of the output list representing a single point on the PDP curve

    # Calculate ICE curves
    idx = np.random.choice(n_samples, n_ice_samples, replace=False) # Random sample of indices
    instance_samples = data.iloc[idx] # Get the randomly selected instances
    anchor_value = feature_values_unique[0] # Determine the anchor point (lower end of the feature) to center the ICE curves
    ice_curves = [] # List to store the ICE curves
    for i, instance_sample in instance_samples.iterrows(): # Loop over the randomly selected instances
        temp_data = pd.concat([instance_sample.to_frame().T]*n_unique_values, ignore_index=True) # Make a temporary dataframe with the instance sample and the original data dimensions
        temp_data[feature_name] = feature_values_unique # Replace the entire column with the target feature values
        predictions = model.predict(temp_data) # Predict the target values for all instances at once
        ice_curves.append(predictions) # Append the ICE curve to the ICE curves list
    ice_curves = np.array(ice_curves) # Convert the ICE curves list to a NumPy array

    # Calculate the anchor predictions
    anchor_value = feature_values_unique[0] # Determine the anchor point (lower end of the feature)
    anchor_predictions = [] # List to store the anchor predictions
    for i in range(n_ice_samples): # Loop over the randomly selected instances
        instance_sample = instance_samples.iloc[i].copy()  # Get the instance sample
        instance_sample[feature_name] = anchor_value # Create a single-row DataFrame with the anchor value for the feature
        instance_sample_df = instance_sample.to_frame().T # Create a single-row DataFrame with the anchor value for the feature
        anchor_prediction = model.predict(instance_sample_df)[0] # Predict the target value at the anchor point
        anchor_predictions.append(anchor_prediction) # Append the anchor prediction to the list
    anchor_predictions = np.array(anchor_predictions) # Convert the list to a NumPy array

    # Center the ICE curves
    centered_ice_curves = ice_curves - anchor_predictions[:, np.newaxis]

    return pdp_curve, centered_ice_curves, xax

def monte_carlo_shapley_values(data: pd.DataFrame, model: object, M: int, sample_idx: int) -> list:
    """
    Calculate the Shapley values for all features using the Monte Carlo approximation.
    Parameters:
    data (pd.DataFrame): The dataset.
    model (sklearn-like model): The model to evaluate.
    M (int): The number of Monte Carlo simulations (evaluated coalitions).
    Returns:
    list: Shapley values for each feature.
    """
    shapley_feature_values = []  # Initialize the list to store the Shapley values
    x = data.iloc[sample_idx].copy() # Extract the instance to explain
    n_features = data.shape[1]  # Number of features
    for feature_index in range(n_features): # Iterate over all features
        shapley_value = 0  # Initialize the Shapley value
        for _ in range(M): # Perform M Monte Carlo simulations
            # Randomly generate a subset S not including the feature_index
            S = np.random.choice([i for i in range(n_features) if i != feature_index], size=random.randint(0, n_features - 1), replace=False)
            # Construct x+ and x- based on S
            x_plus = x.copy() # x_plus is a copy of the instance x
            x_minus = x.copy() # x_minus is a copy of the instance x
            zm = data.iloc[random.randint(0, data.shape[0] - 1)] # Randomly select an instance from the dataset
            # Replace values of x_plus and x_minus with the values of zm for the indices in S
            for i in S: # Iterate over the indices in S
                x_plus[i] = zm[i] 
                x_minus[i] = zm[i] 
            x_minus[feature_index] = zm[feature_index]  # x_minus has the same values as x_plus except that the feature_index is replaced with the value of zm
            # Ensure x_plus and x_minus are DataFrames with a single row
            x_plus = pd.DataFrame([x_plus], columns=data.columns)
            x_minus = pd.DataFrame([x_minus], columns=data.columns)
            # Calculate the marginal contribution
            phi_m_j = model.predict(x_plus) - model.predict(x_minus)
            # Accumulate the Shapley value
            shapley_value += phi_m_j
        # Average the accumulated Shapley values
        shapley_value /= M
        shapley_feature_values.append(shapley_value[0])  # Extract value from array
    return shapley_feature_values