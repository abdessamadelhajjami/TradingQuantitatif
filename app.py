import streamlit as st
import yfinance as yf
import datetime

# Déterminer les dates de début et de fin pour les 6 derniers mois
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=6*30)

# Convertir les dates en chaînes de caractères dans le format 'YYYY-MM-DD'
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Télécharger les données du CAC40 des 6 derniers mois
cac40_data = yf.download("AAPL", start=2024/01/01, end=2024/01/10)

# Créer l'application Streamlit
st.title('Analyse du CAC40')

# Afficher les données dans un tableau
st.write(cac40_data)
