import streamlit as st
import yfinance as yf
from datetime import date, timedelta, datetime
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import base64

# Set constants
START = "2000-08-01"
GOLD_TICKER = 'GC=F'

# Function to load data


def load_data(ticker, end_date):
    data = yf.download(ticker, START, end_date)
    data.reset_index(inplace=True)
    return data

# Function to plot raw data


def plot_raw_data(data):
    selected_columns = st.multiselect(
        "Select columns to plot", data.columns.tolist(), default=["Open", "Close"]
    )

    plot_types = ['Line Plot', 'Candlestick Chart', 'Histogram', '3D Plot']
    selected_plot_type = st.selectbox('Select plot type:', plot_types)

    if not selected_columns:
        st.error("Please select at least one column to plot.")
    else:
        if selected_plot_type == 'Line Plot':
            fig = go.Figure()
            for column in selected_columns:
                fig.add_trace(go.Scatter(x=data['Date'], y=data[column],
                                         name=f"Gold {column} Price", line=dict(color='blue' if column == "Open" else 'red')))
            fig.update_layout(title='Gold Price Time Series Data',
                              xaxis_title='Date', yaxis_title='Price (USD)')
            fig.update_layout(xaxis_rangeslider_visible=True)
        elif selected_plot_type == 'Candlestick Chart':
            fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                                 open=data['Open'],
                                                 high=data['High'],
                                                 low=data['Low'],
                                                 close=data['Close'])])
            fig.update_layout(title='Gold Price Candlestick Chart',
                              xaxis_title='Date', yaxis_title='Price (USD)')

        elif selected_plot_type == 'Histogram':
            fig = go.Figure()
            for column in selected_columns:
                fig.add_trace(go.Histogram(
                    x=data[column], name=f"Gold {column} Price"))
            fig.update_layout(title='Gold Price Histogram',
                              xaxis_title='Price (USD)', yaxis_title='Frequency')
        elif selected_plot_type == '3D Plot':
            fig = go.Figure(data=[go.Scatter3d(x=data['Date'],
                                               y=data['Close'],
                                               z=data['Volume'],
                                               mode='markers')])
            fig.update_layout(scene=dict(xaxis_title='Date',
                                         yaxis_title='Price (USD)',
                                         zaxis_title='Volume'),
                              title='Gold Price 3D Scatter Plot')

        # Auto-adjusting y-axis
        fig.update_layout(yaxis=dict(type='linear'))

    st.plotly_chart(fig)
# Function to forecast using ARIMA


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

# Function to evaluate forecast accuracy


def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# Function to download forecasted data as CSV


def download_csv(data):
    csv_file = data.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()
    href = f'<a href="data:file/csv;base64,{
        b64}" download="forecasted_data.csv">Download Forecasted Data as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

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

    # Sidebar
    st.sidebar.title("Gold Price Forecast App")
    st.sidebar.markdown("### Model Parameters")
    n_years = st.sidebar.slider('Years of prediction:', 0, 20)
    n_days = st.sidebar.slider('Days of prediction:', 1, 365)
    p = st.sidebar.slider('ARIMA parameter p:', 0, 10, 5)
    d = st.sidebar.slider('ARIMA parameter d:', 0, 2, 1)
    q = st.sidebar.slider('ARIMA parameter q:', 0, 10, 5)

    # Define static price for the currency (USD)
    static_price_usd = 1.0

    # Create dropdown for selecting country prices
    country_prices = {
        'United States ($)': 1.0,
        'United Kingdom (£)': 0.73,
        'Eurozone (€)': 0.83,
        'Australia (A$)': 1.29,
        'Japan (¥)': 110.09,
        'India (₹)': 74.35,
        'Canada (C$)': 1.25,
        'Switzerland (Fr)': 0.91,
        'China (¥)': 6.43,
        'South Korea (₩)': 1151.41,
        'Brazil (R$)': 5.64,
        'Russia (₽)': 74.19,
        'South Africa (R)': 15.27,
        'Mexico (Mex$)': 20.18,
        'Singapore (S$)': 1.34,
        'Hong Kong (HK$)': 7.78,
        'New Zealand (NZ$)': 1.37,
        'Sweden (kr)': 8.66,
        'Norway (kr)': 8.46,
        # Add more countries and their corresponding exchange rates
    }

    selected_country = st.sidebar.selectbox(
        'Select Country Prices:', list(country_prices.keys()))

    # Main content
    st.title('Gold Price Forecast App')
    data = load_data(GOLD_TICKER, TODAY)
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

    # Evaluation
    actual = data['Close'].iloc[-len(forecast_df):]
    forecast = forecast_df['Forecast']
    mae, mse, rmse = evaluate_forecast(actual, forecast)
    st.subheader('Forecast Evaluation Metrics')
    st.write(f'Mean Absolute Error (MAE): {mae}')
    st.write(f'Mean Squared Error (MSE): {mse}')
    st.write(f'Root Mean Squared Error (RMSE): {rmse}')

    # Calculate forecasted prices for the selected country
    forecast_df[selected_country] = forecast_df['Forecast'] * \
        country_prices[selected_country] / static_price_usd

    # Show forecast data table for the selected country
    st.subheader(f'Forecast data in {selected_country}')
    st.write(forecast_df[['Date', selected_country]])

    # Download forecasted data as CSV
    if st.sidebar.checkbox('Download Forecasted Data'):
        download_csv(forecast_df)


# Run the app
if __name__ == '__main__':
    main()
