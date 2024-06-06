import yfinance as yf
import streamlit as st
import datetime
import numpy as np
import pandas as pd
import cufflinks as cf
from plotly.offline import iplot
import plotly.graph_objs as go

from scipy.signal import argrelextrema

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
    high_pivots_idx = argrelextrema(df['Close'].values, np.greater_equal, order=window)[0]
    low_pivots_idx = argrelextrema(df['Close'].values, np.less_equal, order=window)[0]
    # Création de Series pour les valeurs de pivot
    pivot_highs = pd.Series(df['Close'].iloc[high_pivots_idx], index=df.index[high_pivots_idx])
    pivot_lows = pd.Series(df['Close'].iloc[low_pivots_idx], index=df.index[low_pivots_idx])
    return pivot_highs, pivot_lows

@st.cache
def is_line_valid(df, start_index, end_index, start_value, end_value):
    """
    Vérifie si la ligne de tendance entre deux points de pivot ne coupe aucun prix entre eux.
    
    :param df: DataFrame contenant les données de prix.
    :param start_index: L'indice du premier point de pivot.
    :param end_index: L'indice du second point de pivot.
    :param start_value: La valeur du prix au premier point de pivot.
    :param end_value: La valeur du prix au second point de pivot.
    :return: Booléen indiquant si la ligne est valide ou non.
    """
    # Obtenez les prix correspondant aux dates des points de pivot
    start_price = df.loc[start_index, 'Close']
    end_price = df.loc[end_index, 'Close']
    
    # Calculez la pente de la ligne
    slope = (end_price - start_price) / (end_index - start_index)
    
    # Pour chaque point entre start_index et end_index, vérifiez si le prix est sous la ligne
    for i in range(start_index + 1, end_index):
        # Prix prédit par la ligne de tendance à l'indice i
        predicted_price = start_price + slope * (i - start_index)
        
        # Si un prix réel dépasse la ligne prédite, la ligne n'est pas valide
        if df.loc[i, 'Close'] > predicted_price:
            return False
    
    return True


@st.cache
def measure_trendline_strength(df, start_index, end_index, start_value, end_value, threshold=0.02):
    """
    Mesure la force d'une ligne de tendance en comptant le nombre de fois où le prix
    s'approche de la ligne sans la couper, basé sur un seuil de proximité.

    :param df: DataFrame contenant les données de prix.
    :param start_index: L'indice du premier point de pivot.
    :param end_index: L'indice du second point de pivot.
    :param start_value: La valeur du prix au premier point de pivot.
    :param end_value: La valeur du prix au second point de pivot.
    :param threshold: Le seuil de proximité en pourcentage du prix.
    :return: Le nombre de fois où les prix sont proches de la ligne sans la couper.
    """
    # Calculez la pente de la ligne
    slope = (end_value - start_value) / (end_index - start_index)
    # Initialisez le compteur de proximité
    proximity_count = 0
    
    for i in range(start_index + 1, end_index):
        # Prix prédit par la ligne de tendance à l'indice i
        predicted_price = start_value + slope * (i - start_index)
        actual_price = df.iloc[i]['Close']
        
        # Calculez la distance absolue entre le prix prédit et le prix réel
        distance = abs(predicted_price - actual_price)
        
        # Calculez le seuil de proximité en termes de prix
        price_threshold = actual_price * threshold
        
        # Si la distance est inférieure au seuil de proximité et que le prix ne coupe pas la ligne, augmentez le compteur
        if distance <= price_threshold:
            proximity_count += 1

    return proximity_count




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

pivot_flag = st.sidebar.checkbox("Show Pivot Points", value=False)
pivot_window = 5
if pivot_flag:
    pivot_window = st.sidebar.number_input("Pivot Detection Window Size", min_value=1, max_value=50, value=5, step=1)

# Dans la partie de la sidebar pour les paramètres de l'analyse technique
st.sidebar.header("Trend Lines Parameters")
# Option pour activer/désactiver l'affichage des lignes de tendance
show_trend_lines = st.sidebar.checkbox("Show Trend Lines", value=False)

# Si les lignes de tendance sont activées, afficher plus d'options
if show_trend_lines:
    # Option pour choisir le nombre minimum de contacts pour une ligne de tendance significative
    min_contacts = st.sidebar.number_input("Minimum Contacts for Significance", min_value=2, max_value=10, value=2, step=1)







st.title("A Simple web app for technical analysis")
st.write("""
  
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
    qf.add_bollinger_bands(periods=bb_periods, boll_std=bb_std)
if rsi_flag:
    qf.add_rsi(periods=rsi_periods, rsi_upper=rsi_upper, rsi_lower=rsi_lower, showbands=True)

# Préparez le graphique pour l'ajout des annotations si l'option de pivot est sélectionnée
if pivot_flag:
    pivot_highs, pivot_lows = detect_pivots(df, pivot_window)
    fig = qf.iplot(asFigure=True)

    # Ajoutez des annotations pour chaque pivot haut et bas
    for date, value in pivot_highs.items():
        fig.add_annotation(x=date, y=value, text="PH", showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=2, arrowcolor="green")
    for date, value in pivot_lows.items():
        fig.add_annotation(x=date, y=value, text="PL", showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=2, arrowcolor="red")
else:
    # Si les pivots ne sont pas sélectionnés, générez simplement le graphique sans annotations supplémentaires
    fig = qf.iplot(asFigure=True)

# Affichez le graphique final avec toutes les modifications appliquées
# Assurez-vous que cette logique s'exécute seulement si show_trend_lines est activé
if show_trend_lines and pivot_flag:
    pivot_highs, pivot_lows = detect_pivots(df, pivot_window)
    fig = qf.iplot(asFigure=True)

    # Exemple simplifié avec les points de pivot hauts
    for start_idx in range(len(pivot_highs) - 1):
        for end_idx in range(start_idx + 1, len(pivot_highs)):
            start_date, start_value = pivot_highs.index[start_idx], pivot_highs.iloc[start_idx]
            end_date, end_value = pivot_highs.index[end_idx], pivot_highs.iloc[end_idx]
            
            if is_line_valid(df, start_date, end_date, start_value, end_value):
                line_strength = measure_trendline_strength(df, start_date, end_date, start_value, end_value)
                
                # Tracez la ligne si elle a une force supérieure à un seuil, par exemple 3
                if line_strength > min_contacts:  # Utilisez min_contacts comme seuil de force
                    fig.add_shape(type="line", x0=start_date, y0=start_value, x1=end_date, y1=end_value,
                                  line=dict(color="RoyalBlue", width=2))

    st.plotly_chart(fig)

st.plotly_chart(fig)




