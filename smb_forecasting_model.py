import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from datetime import date, timedelta
import random
import holidays
import joblib
from data_ingestion import fetch_pos_data, fetch_weather_data, fetch_event_data

# --- 1. DATA INGESTION ---
# This section has been updated to use the data_ingestion.py module.

# --- 2. FEATURE ENGINEERING ---
# This section creates the predictive features defined in the spec.

def create_features(df):
    """
    Engineers features for the model from the raw data.
    """
    print("Engineering features...")
    # Retain original product_name before one-hot encoding
    df['product_name_raw'] = df['product_name']

    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Add holiday feature
    us_holidays = holidays.US()
    df['is_holiday'] = df['date'].apply(lambda x: x in us_holidays).astype(int)

    # Create lag features (sales from the recent past)
    # We sort to ensure lags are calculated correctly
    df = df.sort_values(by=['product_name', 'date'])
    df['sales_lag_1_day'] = df.groupby('product_name')['quantity_sold'].shift(1)
    df['sales_lag_7_day'] = df.groupby('product_name')['quantity_sold'].shift(7)

    # Create rolling average features
    df['sales_rolling_7_day_avg'] = df.groupby('product_name')['quantity_sold'].shift(1).rolling(window=7).mean()

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['weather_condition', 'product_name', 'category'], drop_first=True)

    # Fill any NaNs created by lags/rolling averages
    df = df.fillna(0)
    print("...Feature engineering complete.")
    return df

# --- 3. MODEL TRAINING ---

class SalesForecaster:
    def __init__(self, model_params=None):
        if model_params is None:
            self.model_params = {
                'objective': 'regression_l1', # L1 loss is Mean Absolute Error
                'metric': 'mape', # Mean Absolute Percentage Error
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'verbose': -1,
                'n_jobs': -1,
                'seed': 42
            }
        self.model = lgb.LGBMRegressor(**self.model_params)
        self.trained_features = []

    def train(self, df):
        """
        Trains the LightGBM model on the feature-engineered data.
        """
        print("Training the forecasting model...")

        # Define features (X) and target (y)
        self.target = 'quantity_sold'
        self.features = [col for col in df.columns if col not in ['date', 'timestamp', self.target, 'price_per_unit', 'total_value', 'product_name_raw']]

        X = df[self.features]
        y = df[self.target]

        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

        self.model.fit(X_train, y_train,
                       eval_set=[(X_test, y_test)],
                       eval_metric='mape',
                       callbacks=[lgb.early_stopping(100, verbose=False)])

        # Evaluate model performance
        preds = self.model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, preds)
        print(f"...Training complete. Model MAPE on test set: {mape:.2%}")

        # Store feature names for prediction
        self.trained_features = X_train.columns.tolist()

    def save_model(self, filepath="sales_forecaster.joblib"):
        """Saves the trained model to a file."""
        print(f"Saving model to {filepath}...")
        joblib.dump(self, filepath)
        print("...Model saved.")

    @classmethod
    def load_model(cls, filepath="sales_forecaster.joblib"):
        """Loads a trained model from a file."""
        print(f"Loading model from {filepath}...")
        model = joblib.load(filepath)
        print("...Model loaded.")
        return model

    def predict_future(self, future_df):
        """
        Makes predictions on a future-dated, feature-engineered dataframe.
        """
        print("Generating future forecast...")
        # Ensure future dataframe has the same columns as training data
        future_X = future_df[self.trained_features]

        predictions = self.model.predict(future_X)
        future_df['forecasted_sales'] = np.round(predictions).astype(int)
        # Ensure forecast is not negative
        future_df['forecasted_sales'] = future_df['forecasted_sales'].clip(lower=0)

        return future_df

def retrain_model(new_sales_data_path, model_path="sales_forecaster.joblib"):
    """
    Loads an existing model, retrains it with new data, and saves it.
    """
    print(f"--- Retraining Model with data from {new_sales_data_path} ---")

    # Load the existing model
    forecaster = SalesForecaster.load_model(model_path)

    # Load new sales data
    new_sales_df = pd.read_csv(new_sales_data_path)
    # Basic preprocessing, assuming CSV has 'timestamp' and 'product_id', etc.
    new_sales_df['date'] = pd.to_datetime(new_sales_df['timestamp']).dt.date
    new_sales_df.rename(columns={'product_id': 'product_name'}, inplace=True)

    # Fetch corresponding external data
    unique_dates = new_sales_df['date'].unique()
    weather_df = fetch_weather_data(unique_dates)
    event_df = fetch_event_data(unique_dates)

    # Merge data sources
    new_sales_df['date'] = pd.to_datetime(new_sales_df['date'])
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    event_df['date'] = pd.to_datetime(event_df['date'])

    full_new_df = pd.merge(new_sales_df, weather_df, on='date')
    full_new_df = pd.merge(full_new_df, event_df, on='date')

    # Engineer features
    featured_new_df = create_features(full_new_df)

    # Retrain the model
    forecaster.train(featured_new_df)

    # Save the updated model
    forecaster.save_model(model_path)

    print("--- Retraining complete ---")
    return forecaster

def log_waste(product_name, quantity_wasted, log_file="waste_log.csv"):
    """
    Logs unsold items to a CSV file.
    """
    print(f"Logging {quantity_wasted} units of {product_name} as waste...")

    # Create a dataframe with the new waste data
    waste_df = pd.DataFrame({
        'date': [date.today()],
        'product_name': [product_name],
        'quantity_wasted': [quantity_wasted]
    })

    # Append to the CSV file, creating it if it doesn't exist
    try:
        with open(log_file, 'a') as f:
            waste_df.to_csv(f, header=f.tell()==0, index=False)
        print("...Waste logged successfully.")
    except IOError as e:
        print(f"Error logging waste: {e}")

def log_event_feedback(event_name, log_file="event_feedback.csv"):
    """
    Prompts the user for qualitative feedback on an event and logs it.
    """
    print(f"\n--- Logging Feedback for Event: {event_name} ---")
    feedback = input("How was the event? (e.g., 'Busier than expected', 'No impact', 'Slower than expected'): ")

    feedback_df = pd.DataFrame({
        'date': [date.today()],
        'event_name': [event_name],
        'feedback': [feedback]
    })

    try:
        with open(log_file, 'a') as f:
            feedback_df.to_csv(f, header=f.tell()==0, index=False)
        print("...Feedback logged successfully.")
    except IOError as e:
        print(f"Error logging feedback: {e}")

# --- 4. DECISION SUPPORT & APPLICATION ---

def generate_purchase_recommendation(forecast_df, current_inventory):
    """
    Generates a simple purchase order based on the forecast.
    """
    print("\n--- Automated Purchasing Report ---")
    recommendations = []
    total_cost = 0
    for _, row in forecast_df.iterrows():
        product = row['product_name_raw']
        forecast = row['forecasted_sales']
        on_hand = current_inventory.get(product, 0)
        cost_per_unit = row['cost_per_unit']

        needed = forecast - on_hand
        if needed > 0:
            purchase_cost = needed * cost_per_unit
            total_cost += purchase_cost
            print(f"Product: {product}")
            print(f"  > Forecasted Sales: {forecast} units")
            print(f"  > Currently On Hand: {on_hand} units")
            print(f"  > RECOMMENDATION: Purchase at least {needed} units.")
            print(f"  > Estimated Cost: ${purchase_cost:.2f}")
            recommendations.append({'product': product, 'purchase_qty': needed, 'cost': purchase_cost})

    print("-" * 20)
    print(f"Total Estimated Purchase Cost: ${total_cost:.2f}")
    return recommendations

def recommend_staff_levels(forecast_df, revenue_per_staff=250):
    """
    Recommends staffing levels based on projected revenue.
    """
    print("\n--- Staffing Recommendation ---")
    projected_revenue = (forecast_df['forecasted_sales'] * forecast_df['price_per_unit']).sum()

    recommended_staff = int(projected_revenue / revenue_per_staff)

    print(f"Projected Revenue: ${projected_revenue:.2f}")
    print(f"Heuristic (Revenue per Staff): ${revenue_per_staff}")
    print(f"RECOMMENDATION: Schedule {recommended_staff} staff members.")
    return recommended_staff

def build_cash_flow_projector(forecast_df):
    """
    Creates a 7-day cash flow projection.
    """
    print("\n--- 7-Day Cash Flow Projection ---")

    # Group by day
    daily_projection = forecast_df.groupby(forecast_df['date'].dt.date).apply(lambda x: pd.Series({
        'projected_revenue': (x['forecasted_sales'] * x['price_per_unit']).sum(),
        'projected_cogs': (x['forecasted_sales'] * x['cost_per_unit']).sum()
    })).reset_index()

    daily_projection['projected_gross_profit'] = daily_projection['projected_revenue'] - daily_projection['projected_cogs']

    print(daily_projection.to_string(index=False))

    total_revenue = daily_projection['projected_revenue'].sum()
    total_profit = daily_projection['projected_gross_profit'].sum()

    print("-" * 20)
    print(f"Total Projected 7-Day Revenue: ${total_revenue:,.2f}")
    print(f"Total Projected 7-Day Gross Profit: ${total_profit:,.2f}")

    return daily_projection

def run_event_roi_calculator(event_name, costs, opportunity_cost, brand_lift):
    """
    Runs the Event ROI analysis as defined in the spec.
    """
    print(f"\n--- Event ROI Calculator: '{event_name}' ---")
    projected_profit = costs['projected_sales'] - (costs['inventory'] + costs['staff'] + costs['fee'])
    net_financial = projected_profit - opportunity_cost

    print(f"Projected Event Profit: ${projected_profit:,.2f}")
    print(f"Opportunity Cost (Lost Sales at Store): ${opportunity_cost:,.2f}")
    print(f"Projected Brand Lift (Followers): {brand_lift:.0%}")
    print("-" * 20)
    print(f"Net Financial Impact: ${net_financial:,.2f}")

    if net_financial < 0:
        print(f"Conclusion: The event represents a net financial loss of ${abs(net_financial):,.2f} but offers a significant brand visibility boost.")
    else:
        print("Conclusion: The event is projected to be financially positive and offers a brand visibility boost.")

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # 1. Ingest Data from APIs
    sales_df = fetch_pos_data()
    sales_df['date'] = sales_df['timestamp'].dt.date
    sales_df.rename(columns={'product_id': 'product_name'}, inplace=True)


    unique_dates = sales_df['date'].unique()
    weather_df = fetch_weather_data(unique_dates)
    event_df = fetch_event_data(unique_dates)

    # Merge data sources
    # To avoid issues with data types, convert date columns to datetime
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    event_df['date'] = pd.to_datetime(event_df['date'])

    full_df = pd.merge(sales_df, weather_df, on='date')
    full_df = pd.merge(full_df, event_df, on='date')

    # 2. Engineer Features
    featured_df = create_features(full_df.copy())

    # 3. Train Model
    forecaster = SalesForecaster()
    forecaster.train(featured_df)

    # 4. Application: Generate a 7-day forecast
    print("\n--- Generating 7-Day Sales Forecast ---")

    # Start with the most recent data
    last_day_data = featured_df[featured_df['date'] == featured_df['date'].max()].copy()

    # Store future predictions
    future_predictions = []

    # Loop for the next 7 days
    for i in range(1, 8):
        future_date = date.today() + timedelta(days=i)
        print(f"Forecasting for: {future_date}")

        # Create future data for prediction
        future_data_for_prediction = last_day_data.copy()
        future_data_for_prediction['date'] = pd.to_datetime(future_date)

        # Get external data for the future date
        future_weather = fetch_weather_data([future_date]).iloc[0]
        future_event = fetch_event_data([future_date]).iloc[0]

        # Update external features
        weather_conditions = ['Sunny', 'Cloudy', 'Rainy']
        for condition in weather_conditions:
            col_name = f'weather_condition_{condition}'
            if col_name in future_data_for_prediction.columns:
                future_data_for_prediction[col_name] = 1 if future_weather['weather_condition'] == condition else 0

        future_data_for_prediction['daily_high_temperature'] = future_weather['daily_high_temperature']
        future_data_for_prediction['is_local_event_day'] = future_event['is_local_event_day']

        # Re-calculate date-related features
        future_data_for_prediction['day_of_week'] = future_data_for_prediction['date'].dt.dayofweek
        future_data_for_prediction['month'] = future_data_for_prediction['date'].dt.month
        future_data_for_prediction['week_of_year'] = future_data_for_prediction['date'].dt.isocalendar().week.astype(int)
        future_data_for_prediction['is_weekend'] = future_data_for_prediction['day_of_week'].isin([5, 6]).astype(int)
        us_holidays = holidays.US()
        future_data_for_prediction['is_holiday'] = future_data_for_prediction['date'].apply(lambda x: x in us_holidays).astype(int)

        # Update lag features based on the previous day's forecast
        sales_map = last_day_data.set_index('product_name_raw')['quantity_sold']
        future_data_for_prediction['sales_lag_1_day'] = future_data_for_prediction['product_name_raw'].map(sales_map)
        future_data_for_prediction['sales_lag_1_day'] = future_data_for_prediction['sales_lag_1_day'].fillna(0)

        # Note: For simplicity, lag_7 and rolling_avg are not updated dynamically in this loop.
        # A more robust solution would require maintaining a longer history.

        # Make prediction
        daily_forecast = forecaster.predict_future(future_data_for_prediction)
        future_predictions.append(daily_forecast)

        # Update last_day_data for the next iteration
        daily_forecast['quantity_sold'] = daily_forecast['forecasted_sales']
        last_day_data = daily_forecast


    # Combine all future predictions
    full_forecast_df = pd.concat(future_predictions)

    print("\n--- 7-Day Sales Forecast ---")
    print(full_forecast_df[['date', 'product_name_raw', 'forecasted_sales']])

    # Run Decision Support Tools for the first day of the forecast
    print("\n--- Decision Support for Tomorrow ---")
    tomorrow_forecast = full_forecast_df[full_forecast_df['date'] == pd.to_datetime(date.today() + timedelta(days=1))]
    current_inventory = {'Coffee': 20, 'Sandwich': 10, 'Pastry': 15, 'Branded Mug': 3}
    generate_purchase_recommendation(tomorrow_forecast, current_inventory)
    recommend_staff_levels(tomorrow_forecast)
    build_cash_flow_projector(full_forecast_df)

    event_costs = {'projected_sales': 1200, 'inventory': 300, 'staff': 400, 'fee': 100}
    run_event_roi_calculator(
        event_name="Dallas Summerfest",
        costs=event_costs,
        opportunity_cost=500,
        brand_lift=0.08
    )

    # --- 5. Feedback and Improvement Loop Examples ---
    print("\n--- Feedback and Improvement Loop Examples ---")

    # Save the initial model
    forecaster.save_model()

    # Example of retraining the model with new data
    # In a real scenario, this would be new, actual sales data.
    retrain_model("new_sales.csv")

    # Example of logging waste
    log_waste("Pastry", 5)

    # Example of logging event feedback (commented out to avoid interactive input)
    # log_event_feedback("Dallas Summerfest")
