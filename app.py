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

@st.cache
def detect_pivots(df, window=5):
    """
    Détecte les pivots hauts et bas dans un DataFrame de prix.
    
    :param df: DataFrame contenant les données de prix.
    :param window: La fenêtre pour détecter les pivots. Doit être un nombre impair.
    :return: DataFrame avec des colonnes supplémentaires 'PivotHigh' et 'PivotLow' marquant les pivots.
    """
    from scipy.signal import argrelextrema
    df['PivotHigh'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=window)[0]]['Close']
    df['PivotLow'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=window)[0]]['Close']
    return df


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

exp_sma = st.sidebar.expander("SMA")
sma_flag = exp_sma.checkbox(label="Add SMA")
sma_periods= exp_sma.number_input(
  label="SMA Periods", 
  min_value=1, 
  max_value=50, 
  value=20, 
  step=1
)

exp_bb = st.sidebar.expander("Bollinger Bands")
bb_flag = exp_bb.checkbox(label="Add Bollinger Bands")
bb_periods= exp_bb.number_input(label="BB Periods", 
                             min_value=1, max_value=50, 
                             value=20, step=1)
bb_std= exp_bb.number_input(label="# of standard deviations", 
                             min_value=1, max_value=4, 
                             value=2, step=1)


exp_rsi = st.sidebar.expander("Relative Strength Index")
rsi_flag = exp_rsi.checkbox(label="Add RSI")
rsi_periods= exp_rsi.number_input(
    label="RSI Periods", 
    min_value=1, 
    max_value=50, 
    value=20, 
    step=1
)
rsi_upper= exp_rsi.number_input(label="RSI Upper", 
    min_value=50, 
    max_value=90, value=70, 
    step=1)
rsi_lower= exp_rsi.number_input(label="RSI Lower", 
   min_value=10, 
   max_value=50, value=30, 
   step=1)

# Ajouter dans la partie de configuration des paramètres techniques dans la sidebar
st.sidebar.header("Pivot Parameters")
pivot_flag = st.sidebar.checkbox("Add Pivot Points")
pivot_window = 5  # Une valeur par défaut

if pivot_flag:
    pivot_window = st.sidebar.number_input("Pivot Detection Window",
                                           min_value=3,
                                           max_value=25,
                                           value=5,
                                           step=2,
                                           help="Choose an odd number for the best results")


st.title("A Simple web app for technical analysis")
st.write("""
   ### User manual
   * you can select any company from the S&P 500 constituents
  """)

df = load_data(ticker, start_date, end_date)
window_size = 5  # Vous pouvez ajuster cette valeur selon vos besoins
df = detect_pivots(df, window=window_size)

data_exp = st.expander("Preview data")
available_cols = df.columns.tolist()
columns_to_show = data_exp.multiselect(
   "Columns", 
   available_cols, 
   default=available_cols
)

data_exp.dataframe(df[columns_to_show])
csv_file = convert_df_to_csv(df[columns_to_show])
data_exp.download_button(
  label="Download selected as CSV",
  data=csv_file,
  file_name=f"{ticker}_stock_prices.csv",
  mime="text/csv",
)

title_str = f"{tickers_companies_dict[ticker]}'s stock price"
qf = cf.QuantFig(df, title=title_str)
if volume_flag:
   qf.add_volume()
if sma_flag:
   qf.add_sma(periods=sma_periods)
if bb_flag:
   qf.add_bollinger_bands(periods=bb_periods,
                           boll_std=bb_std)
if rsi_flag:
   qf.add_rsi(periods=rsi_periods,
               rsi_upper=rsi_upper,
               rsi_lower=rsi_lower,
             showbands=True)

# Ajout des pivots hauts et bas au graphique
if 'PivotHigh' in df.columns:
    qf.add_scatter(x=df.dropna(subset=['PivotHigh']).index, y=df.dropna(subset=['PivotHigh'])['PivotHigh'], name='Pivot High', mode='markers', marker=dict(color='green', size=5))
if 'PivotLow' in df.columns:
    qf.add_scatter(x=df.dropna(subset=['PivotLow']).index, y=df.dropna(subset=['PivotLow'])['PivotLow'], name='Pivot Low', mode='markers', marker=dict(color='red', size=5))
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)







