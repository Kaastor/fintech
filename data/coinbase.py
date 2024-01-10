import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt

def get_bitcoin_hourly_data():
    # Coinbase API endpoint for historical data
    url = "https://api.coinbase.com/v2/prices/BTC-USD/historic"

    # Calculate the start and end time for the last 24 hours
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(hours=24)

    # Format times in ISO format
    start_time_iso = start_time.isoformat()
    end_time_iso = end_time.isoformat()

    # Parameters for the API request
    params = {
        'start': start_time_iso,
        'end': end_time_iso,
        'granularity': 3600  # 3600 seconds = 1 hour
    }

    # Make the request to Coinbase API
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()['data']['prices']
    else:
        return f"Error: Unable to fetch data, status code {response.status_code}"

def plot_bitcoin_data(data):
    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)

    # Convert epoch time to datetime
    df['time'] = pd.to_datetime(df['time'], unit='h')

    # Convert price to numeric
    df['price'] = pd.to_numeric(df['price'])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['price'], marker='o')
    plt.title('Bitcoin Price (Last 24 hours)')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Call the function and plot the result
bitcoin_data = get_bitcoin_hourly_data()
plot_bitcoin_data(bitcoin_data)
