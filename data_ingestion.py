import pandas as pd
from datetime import date, timedelta
import random
import numpy as np

def fetch_pos_data(days=365 * 2):
    """
    Simulates fetching transaction data from a POS system API.
    In a real implementation, this would connect to Square, Toast, etc.
    """
    print(f"Fetching {days} days of POS data...")
    start_date = date.today() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    products = {
        'Coffee': {'base_price': 3.50, 'cost_per_unit': 1.00, 'category': 'Beverage', 'base_units': 50},
        'Sandwich': {'base_price': 8.00, 'cost_per_unit': 3.50, 'category': 'Food', 'base_units': 30},
        'Pastry': {'base_price': 4.00, 'cost_per_unit': 1.50, 'category': 'Food', 'base_units': 40},
        'Branded Mug': {'base_price': 12.00, 'cost_per_unit': 5.00, 'category': 'Merchandise', 'base_units': 5}
    }

    data = []
    for dt in dates:
        day_of_week = dt.weekday()  # Monday=0, Sunday=6
        seasonality_factor = 1.4 if day_of_week in [5, 6] else (0.8 if day_of_week == 0 else 1.0)

        for product_id, props in products.items():
            noise = random.uniform(0.85, 1.15)
            quantity_sold = int(props['base_units'] * seasonality_factor * noise)

            data.append({
                'timestamp': pd.to_datetime(dt),
                'product_id': product_id,
                'quantity_sold': quantity_sold,
                'price_per_unit': props['base_price'],
                'cost_per_unit': props['cost_per_unit'],
                'category': props['category'],
                'total_value': quantity_sold * props['base_price']
            })

    df = pd.DataFrame(data)
    print("...POS data fetch complete.")
    return df

def fetch_weather_data(dates):
    """
    Simulates fetching weather data from a weather API like OpenWeatherMap.
    """
    print("Fetching weather data...")
    weather_data = []
    for dt in dates:
        temp = 60 + np.sin(pd.to_datetime(dt).month * 2 * np.pi / 12) * 20 + random.uniform(-5, 5)
        condition = random.choice(['Sunny', 'Cloudy', 'Rainy', 'Sunny'])
        weather_data.append({
            'date': pd.to_datetime(dt),
            'daily_high_temperature': round(temp, 1),
            'weather_condition': condition
        })
    df = pd.DataFrame(weather_data)
    print("...Weather data fetch complete.")
    return df

def fetch_event_data(dates):
    """
    Simulates fetching local event data from a platform like Eventbrite.
    """
    print("Fetching local event data...")
    event_data = []
    for dt in dates:
        is_event = True if (pd.to_datetime(dt) - pd.to_datetime(dates[0])).days % 90 < 2 else False
        event_data.append({
            'date': pd.to_datetime(dt),
            'is_local_event_day': is_event
        })
    df = pd.DataFrame(event_data)
    print("...Event data fetch complete.")
    return df
