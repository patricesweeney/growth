import pandas as pd
from lightweight_mmm import preprocessing, lightweight_mmm, LightweightMMM, plot, optimize_media
import jax.numpy as jnp
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np



def import_data_local():
    import pandas as pd
    file_path = '/Users/patricksweeney/growth/01_Acquisition/02_Media mix modelling/Sr Advertising Analyst Work Sample (1).xlsx'
    data = pd.read_excel(file_path)
    return data

data = import_data_local()


agg_data = data.groupby(["Date", "Ad group alias"])[["Impressions","Spend", "Sales",]].sum()
agg_data = agg_data.drop(["Brand 1 Ad Group 12"], axis=0, level=1) # zero cost train

media_data_raw = agg_data['Impressions'].unstack().fillna(0)
costs_raw = agg_data['Spend'].unstack()
sales_raw = agg_data['Sales'].reset_index().groupby("Date").sum()


#%% Train and test sets

split_point = pd.Timestamp("2021-12-15") # 28 days to end of data

media_data_train = media_data_raw.loc[:split_point - pd.Timedelta(1,'D')]
media_data_test = media_data_raw.loc[split_point:]

target_train = sales_raw.loc[:split_point - pd.Timedelta(1,'D')]
target_test = sales_raw.loc[split_point:]

costs_train = costs_raw.loc[:split_point - pd.Timedelta(1,'D')].sum(axis=0).loc[media_data_train.columns]


#%% Add organic

import pandas as pd
import numpy as np

# Assuming media_data_raw and sales_raw are defined elsewhere and sales_raw is a pandas DataFrame
# Convert the 'Sales' column of sales_raw values to numeric
sales_raw_numeric = pd.to_numeric(sales_raw['Sales'], errors='coerce')

# Create a DataFrame for organic_raw with the same index as media_data_raw
organic_raw = pd.DataFrame({'organic_search': 0, 'organic_social': 0}, index=media_data_raw.index)

# Update organic_raw with calculated values
organic_raw['organic_search'] = sales_raw_numeric / 10 + np.random.randint(10000, 100001, size=organic_raw.shape[0])
organic_raw['organic_social'] = sales_raw_numeric / 10 + np.random.randint(10000, 100001, size=organic_raw.shape[0])



#%% Scaling

from lightweight_mmm.preprocessing import CustomScaler
import jax.numpy as jnp
import pandas as pd

# Your existing setup code
split_point = pd.Timestamp("2021-12-15")  # 28 days to end of data

media_data_train = media_data_raw.loc[:split_point - pd.Timedelta(1, 'D')]
media_data_test = media_data_raw.loc[split_point:]

organic_data_train = organic_raw.loc[:split_point - pd.Timedelta(1, 'D')]
organic_data_test = organic_raw.loc[split_point:]

# Exclude the non-numeric 'Ad group alias' column from target_train
target_train_numeric = sales_raw.loc[:split_point - pd.Timedelta(1, 'D'), 'Sales']

costs_train = costs_raw.loc[:split_point - pd.Timedelta(1, 'D')].sum(axis=0).loc[media_data_train.columns]

# Your existing scaler setup code
media_scaler = CustomScaler(divide_operation=jnp.mean)
organic_scaler = CustomScaler(divide_operation=jnp.mean)
target_scaler = CustomScaler(divide_operation=jnp.mean)
cost_scaler = CustomScaler(divide_operation=jnp.mean)

# Perform scaling on the numeric part of target_train
target_train_scaled = target_scaler.fit_transform(target_train_numeric.values.squeeze())

# Your existing scaling code for other data
media_data_train_scaled = media_scaler.fit_transform(media_data_train.values)
organic_data_train_scaled = organic_scaler.fit_transform(organic_data_train.values)
costs_scaled = cost_scaler.fit_transform(costs_train.values)

# Your existing code to scale the test data
media_data_test_scaled = media_scaler.transform(media_data_test.values)
organic_data_test_scaled = organic_scaler.transform(organic_data_test.values)

# Your existing code to retrieve media names
media_names = media_data_raw.columns



#%% Hyperparameter search

from lightweight_mmm.lightweight_mmm import LightweightMMM
from sklearn.metrics import mean_absolute_percentage_error


target_train_numeric = target_train['Sales'].values
target_test_numeric = target_test['Sales'].values


adstock_models = ["adstock", "hill_adstock", "carryover"]
degrees_season = [1,2,3]


for model_name in adstock_models:
    for degrees in degrees_season:
        mmm = LightweightMMM(model_name=model_name)
        mmm.fit(
            media=media_data_train_scaled,
            media_prior=costs_scaled,
            target=target_train_numeric,
            extra_features=organic_data_train_scaled,
            number_warmup=1000,
            number_samples=1000,
            number_chains=1,
            degrees_seasonality=degrees,
            weekday_seasonality=True,
            seasonality_frequency=365,
            seed=1
        )
        
        prediction = mmm.predict(
            media=media_data_test_scaled,
            extra_features=organic_data_test_scaled,
            target_scaler=target_scaler
        )
        p = prediction.mean(axis=0)
        
        # Use the numeric target values for MAPE calculation
        mape = mean_absolute_percentage_error(target_test_numeric, p)
        print(f"model_name={model_name} degrees={degrees} MAPE={mape} samples={p[:3]}")

#%% Retrain with new hyperparameters

import pandas as pd
import jax.numpy as jnp
from lightweight_mmm.preprocessing import CustomScaler
from lightweight_mmm.lightweight_mmm import LightweightMMM

# If sales_raw is a DataFrame, select the column with sales data, e.g., 'Sales'
if isinstance(sales_raw, pd.DataFrame):
    sales_column = 'Sales'  # replace with your actual sales column name
    sales_raw_series = sales_raw[sales_column]
else:
    sales_raw_series = sales_raw  # if sales_raw is already a Series

# Convert sales data to numeric, coercing errors to NaN, then drop NaNs
sales_raw_numeric = pd.to_numeric(sales_raw_series, errors='coerce').dropna()

# Apply the same for media_data_raw and organic_raw if they are not numeric
media_data_raw_numeric = media_data_raw.apply(pd.to_numeric, errors='coerce').dropna()
organic_raw_numeric = organic_raw.apply(pd.to_numeric, errors='coerce').dropna()

# Ensure that costs are numeric and correspond to the media names
costs = costs_raw.sum(axis=0)
costs_numeric = pd.to_numeric(costs.loc[media_data_raw_numeric.columns], errors='coerce').dropna()

# Initialize scalers
media_scaler2 = CustomScaler(divide_operation=jnp.mean)
organic_scaler2 = CustomScaler(divide_operation=jnp.mean)
target_scaler2 = CustomScaler(divide_operation=jnp.mean)
cost_scaler2 = CustomScaler(divide_operation=jnp.mean)

# Scale the data
media_data_scaled = media_scaler2.fit_transform(media_data_raw_numeric.values)
organic_data_scaled = organic_scaler2.fit_transform(organic_raw_numeric.values)
target_scaled = target_scaler2.fit_transform(sales_raw_numeric.values.reshape(-1, 1))
costs_scaled2 = cost_scaler2.fit_transform(costs_numeric.values.reshape(-1, 1))

# Fit the model
mmm = LightweightMMM(model_name="hill_adstock")
mmm.fit(
    media=media_data_scaled,
    media_prior=costs_scaled2,
    extra_features=organic_data_scaled,
    target=target_scaled,
    number_warmup=1000,
    number_samples=1000,
    number_chains=1,
    degrees_seasonality=1,
    weekday_seasonality=True,
    seasonality_frequency=365,
    seed=1
)



#%% Plotting

plot.plot_media_channel_posteriors(media_mix_model=mmm, channel_names=media_names)

plot.plot_bars_media_metrics(metric=roi_hat, channel_names=media_names)

!pip uninstall -y matplotlib
!pip install matplotlib==3.1.3

