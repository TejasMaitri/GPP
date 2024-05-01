import streamlit as st
import yfinance as yf
from datetime import date, timedelta, datetime
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import base64
from io import StringIO
import requests

# Set constants
START = "2000-08-01"
GOLD_TICKER = 'GC=F'

# Function to load data with error handling


@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start_date, end_date)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return None


# Functions to plot raw data with column selection

def plot_line_chart(data, selected_columns):
    fig = go.Figure()
    for column in selected_columns:
        fig.add_trace(go.Scatter(x=data['Date'], y=data[column],
                                 name=f"Gold {column} Price", line=dict(color='blue' if column == "Open" else 'red')))
    fig.update_layout(title='Gold Price Time Series Data',
                      xaxis_title='Date', yaxis_title='Price (USD)')
    fig.update_layout(xaxis_rangeslider_visible=True)
    return fig


def plot_candlestick_chart(data):
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title='Gold Price Candlestick Chart',
                      xaxis_title='Date', yaxis_title='Price (USD)')
    return fig


def plot_histogram(data, selected_columns):
    fig = go.Figure()
    for column in selected_columns:
        fig.add_trace(go.Histogram(
            x=data[column], name=f"Gold {column} Price"))
    fig.update_layout(title='Gold Price Histogram',
                      xaxis_title='Price (USD)', yaxis_title='Frequency')
    return fig


def plot_mountain_plot(data):
    fig = go.Figure(data=[go.Surface(z=data.values)])
    fig.update_layout(scene=dict(xaxis_title='Date',
                                 yaxis_title='Price (USD)',
                                 zaxis_title='Volume'),
                      title='Gold Price Mountain Plot')
    return fig


def plot_raw_data(data):
    selected_columns = st.multiselect(
        "Select columns to plot", data.columns.tolist(), default=["Open", "Close"]
    )

    plot_types = ['Line', 'Candle', 'Histogram', 'Mountain', 'Baseline']
    selected_plot_type = st.selectbox('Select plot type:', plot_types)

    if not selected_columns:
        st.error("Please select at least one column to plot.")
    else:
        if selected_plot_type == 'Line':
            fig = plot_line_chart(data, selected_columns)
        elif selected_plot_type == 'Candle':
            fig = plot_candlestick_chart(data)
        elif selected_plot_type == 'Histogram':
            fig = plot_histogram(data, selected_columns)
        elif selected_plot_type == 'Mountain':
            fig = plot_mountain_plot(data)

    st.plotly_chart(fig)

# Function to forecast using ARIMA with dynamic period calculation


def forecast_arima(data, n_years, n_days, p, d, q):
    period = n_years * 365 + n_days
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    model = ARIMA(df_train['y'], order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=period)
    forecast_dates = pd.date_range(
        start=data['Date'].iloc[-1], periods=period+1)[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
    return forecast_df, model_fit

# Function to evaluate forecast accuracy including MAPE and DA


def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)

    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    da = np.mean(np.sign(actual.values[1:] - actual.values[:-1])
                 == np.sign(forecast.values[1:] - forecast.values[:-1])) * 100
    return mae, mse, rmse, mape, da

# Function to download forecasted data as CSV


def download_csv(data):
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    b64 = base64.b64encode(csv_buffer.read().encode()).decode()
    href = f'<a href="data:file/csv;base64,{
        b64}" download="forecasted_data.csv">Download Forecasted Data as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)


# Define function to get currency exchange rates

def get_currency_exchange_rates(base_currency='USD'):
    api_key = '36d170118bce46a187e7b60c89b394a3'
    url = f'https://open.er-api.com/v6/latest/{base_currency}?apikey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['rates']
    else:
        # Handle API request error
        return None


# Main function


def main():
    # Get current date
    TODAY = date.today().strftime("%Y-%m-%d")

    # Page layout
    st.set_page_config(
        page_title="Gold Price Forecast App",
        page_icon=":bar_chart:",
        layout="wide"
    )

    # Define static price for the currency (USD)
    static_price_usd = 1.0

    # Use the function to get current currency exchange rates
    currency_exchange_rates = get_currency_exchange_rates()

    # Create dropdown for selecting country prices
    country_prices = {
        'United States ($)': currency_exchange_rates.get('USD', 1.0),
        'United Kingdom (£)': currency_exchange_rates.get('GBP', 0.73),
        'Eurozone (€)': currency_exchange_rates.get('EUR', 0.83),
        'Australia (A$)': currency_exchange_rates.get('AUD', 1.29),
        'Japan (¥)': currency_exchange_rates.get('JPY', 110.09),
        'India (₹)': currency_exchange_rates.get('INR', 74.35),
        'Canada (C$)': currency_exchange_rates.get('CAD', 1.25),
        'Switzerland (Fr)': currency_exchange_rates.get('CHF', 0.91),
        'China (¥)': currency_exchange_rates.get('CNY', 6.43),
        'South Korea (₩)': currency_exchange_rates.get('KRW', 1151.41),
        'Brazil (R$)': currency_exchange_rates.get('BRL', 5.64),
        'Russia (₽)': currency_exchange_rates.get('RUB', 74.19),
        'South Africa (R)': currency_exchange_rates.get('ZAR', 15.27),
        'Mexico (Mex$)': currency_exchange_rates.get('MXN', 20.18),
        'Singapore (S$)': currency_exchange_rates.get('SGD', 1.34),
        'Hong Kong (HK$)': currency_exchange_rates.get('HKD', 7.78),
        'New Zealand (NZ$)': currency_exchange_rates.get('NZD', 1.37),
        'Sweden (kr)': currency_exchange_rates.get('SEK', 8.66),
        'Norway (kr)': currency_exchange_rates.get('NOK', 8.46),
    }
# Sidebar
    st.sidebar.title("Gold Jun 24 (GC=F) Price Forecast App")
    st.sidebar.markdown("### Model Parameters")
    n_years = st.sidebar.slider('Years of prediction:', 0, 10)
    n_days = st.sidebar.slider('Days of prediction:', 1, 365)
    selected_country = st.sidebar.selectbox(
        'Select Country Prices:', list(country_prices.keys()))

    p = st.sidebar.slider('ARIMA parameter p:', 0, 10, 5)
    d = st.sidebar.slider('ARIMA parameter d:', 0, 2, 1)
    q = st.sidebar.slider('ARIMA parameter q:', 0, 10, 5)

 # Main content
    st.title('Gold Price Forecast App')
    data = load_data(GOLD_TICKER, START, TODAY)
    if data is not None:
        st.subheader('Raw Data')
        plot_raw_data(data)

        st.subheader('Forecast Data')
        forecast_df, model_fit = forecast_arima(data, n_years, n_days, p, d, q)
        st.write(forecast_df)

        st.subheader('Forecast Plot')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'],
                                 mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'],
                                 mode='lines', name='Forecast', line=dict(color='red')))
        fig.update_layout(title='Gold Price Forecast', xaxis_title='Date',
                          yaxis_title='Price (USD)', plot_bgcolor='#FFFFFF')
        st.plotly_chart(fig)

        # Download forecasted data as CSV
    if st.sidebar.checkbox('Download Forecasted Data', True):
        download_csv(forecast_df)

# Calculate forecasted prices for the selected country
    forecast_df[selected_country] = forecast_df['Forecast'] * \
        country_prices[selected_country] / static_price_usd

    # Show forecast data table for the selected country
    st.subheader(f'Forecast data in {selected_country}')
    st.write(forecast_df[['Date', selected_country]])

    # Evaluation
    actual = data['Close'].iloc[-len(forecast_df):]
    forecast = forecast_df['Forecast']
    mae, mse, rmse, mape, da = evaluate_forecast(actual, forecast)
    st.subheader('Forecast Evaluation Metrics')
    st.write(f'Mean Absolute Error (MAE): {mae}')
    st.write(f'Mean Squared Error (MSE): {mse}')
    st.write(f'Root Mean Squared Error (RMSE): {rmse}')
    st.write(f'Mean Absolute Percentage Error (MAPE): {mape}')
    st.write(f'Directional Accuracy (DA): {da}%')


if __name__ == '__main__':
    main()
