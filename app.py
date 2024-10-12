import pandas as pd
from prophet import Prophet
import streamlit as st

# File uploader to allow users to upload their stock price data
# uploaded_file = st.file_uploader('/Users/saikrishna/Desktop/Stock Price Predeictions/stock_prices.csv', type='csv')

# Streamlit app title
st.title('Stock Price Prediction')

# df = pd.DataFrame({'ds':['2021-01-01','2021-02-01','2021-03-01','2021-04-01','2021-05-01','2021-06-01','2021-07-01','2021-08-01'],
#                    'y':[10,20,30,35,35,40,70,100]})

df = pd.read_csv('stock_prices.csv')

# Display the data
# print(df.head())
# Display the data
st.write('Stock Price Data', df.head())

# Fit the Prophet model
model = Prophet()
model.fit(df)

# Forecast for the next month
future = model.make_future_dataframe(periods=1, freq='M')
forecast = model.predict(future)

st.write('Next Month Forecast', forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the forecast
fig = model.plot(forecast)
st.pyplot(fig)
