import pytest
from main import load_data, forecast_arima


def test_load_data():
    # Test loading data for a valid ticker and date range
    data = load_data("GC=F", "2000-01-01", "2000-01-10")
    assert data is not None
    assert len(data) > 0

    # Test loading data for an invalid ticker
    invalid_data = load_data("INVALID_TICKER", "2000-01-01", "2000-01-10")
    assert invalid_data is None

    # Test loading data for an invalid date range
    invalid_date_range_data = load_data("GC=F", "2000-01-10", "2000-01-01")
    assert invalid_date_range_data is None

    # Test loading data for a future end date
    future_date_data = load_data("GC=F", "2000-01-01", "2030-01-01")
    assert future_date_data is not None
    assert len(future_date_data) == 0


def test_forecast_arima():
    # Test forecasting using ARIMA for a valid dataset
    data = load_data("GC=F", "2000-01-01", "2000-01-10")
    forecast_df, model_fit = forecast_arima(
        data, n_years=1, n_days=30, p=1, d=1, q=1)
    assert forecast_df is not None
    assert model_fit is not None

    # Test forecasting using ARIMA for an invalid dataset
    invalid_data = pd.DataFrame()  # Empty DataFrame
    with pytest.raises(Exception):
        forecast_arima(invalid_data, n_years=1, n_days=30, p=1, d=1, q=1)

    # Test forecasting using ARIMA for a dataset with insufficient data
    insufficient_data = data.head(2)  # DataFrame with only 2 rows
    with pytest.raises(Exception):
        forecast_arima(insufficient_data, n_years=1, n_days=30, p=1, d=1, q=1)

    # Test forecasting using ARIMA with different parameter values
    data = load_data("GC=F", "2000-01-01", "2000-01-30")
    forecast_df, model_fit = forecast_arima(
        data, n_years=1, n_days=30, p=1, d=1, q=1)
    assert forecast_df is not None
    assert model_fit is not None

    forecast_df, model_fit = forecast_arima(
        data, n_years=2, n_days=0, p=2, d=1, q=2)
    assert forecast_df is not None
    assert model_fit is not None

    # Test forecasting using ARIMA with different combinations of parameters
    forecast_df, model_fit = forecast_arima(
        data, n_years=1, n_days=30, p=0, d=0, q=0)
    assert forecast_df is not None
    assert model_fit is not None

    forecast_df, model_fit = forecast_arima(
        data, n_years=1, n_days=30, p=3, d=2, q=2)
    assert forecast_df is not None
    assert model_fit is not None

    # Test forecasting using ARIMA for longer prediction periods
    forecast_df, model_fit = forecast_arima(
        data, n_years=5, n_days=0, p=1, d=1, q=1)
    assert forecast_df is not None
    assert model_fit is not None

    # Test forecasting using ARIMA for shorter prediction periods
    forecast_df, model_fit = forecast_arima(
        data, n_years=0, n_days=30, p=1, d=1, q=1)
    assert forecast_df is not None
    assert model_fit is not None
