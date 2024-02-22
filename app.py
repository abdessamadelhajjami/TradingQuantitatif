import streamlit as st
import yfinance as yf
import datetime

# Télécharger les données du CAC40 des 6 derniers mois
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=6*30)
cac40_data = yf.download("AAPL", start=30/12/2023, end=10/01/2024, progress=False)

# Créer l'application Streamlit
st.title('Analyse du CAC40')

# Afficher les données dans un tableau
st.write(cac40_data)
