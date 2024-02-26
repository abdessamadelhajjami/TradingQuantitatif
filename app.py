import yfinance as yf
import streamlit as st
import datetime
import pandas as pd
import cufflinks as cf
from plotly.offline import iplot
cf.go_offline()


@st.cache
def get_sp500_components():
  df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
  df = df[0]
  tickers = df["Symbol"].to_list()
  tickers_companies_dict = dict(
    zip(df["Symbol"], df["Security"])
  )
  return tickers, tickers_companies_dict

@st.cache
def load_data(symbol, start, end):
  return yf.download(symbol, start, end)

@st.cache
def convert_df_to_csv(df):
  return df.to_csv().encode("utf-8")

st.sidebar.header("Stock Parameters")
available_tickers, tickers_companies_dict = get_sp500_components()


ticker = st.sidebar.selectbox(
  "Ticker", 
  available_tickers, 
  format_func=tickers_companies_dict.get
)
start_date = st.sidebar.date_input(
  "Start date", 
  datetime.date(2019, 1, 1)
)
end_date = st.sidebar.date_input(
  "End date", 
  datetime.date.today()
)
if start_date > end_date:
  st.sidebar.error("The end date must fall after the start date")

st.sidebar.header("Technical Analysis Parameters")
volume_flag = st.sidebar.checkbox(label="Add volume")



st.title("A Simple web app for technical analysis")
st.write("""
   ### User manual
   * you can select any company from the S&P 500 constituents
  """)

df = load_data(ticker, start_date, end_date)

data_exp = st.expander("Preview data")
available_cols = df.columns.tolist()
columns_to_show = data_exp.multiselect(
   "Columns", 
   available_cols, 
   default=available_cols
)

data_exp.dataframe(df[columns_to_show])
csv_file = convert_df_to_csv(df[columns_to_show])



title_str = f"{tickers_companies_dict[ticker]}'s stock price"
qf = cf.QuantFig(df, title=title_str)
if volume_flag:
   qf.add_volume()

fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)










