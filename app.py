"""
Streamlit — Voiture venante ou d'occasion ? (TP1, Car_Data)
Construite sur le modèle d'application fourni par le prof : les objets sauvegardés
dans le notebook sont rechargés, puis le meilleur des sept classifieurs fait la prédiction.

En local :  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st

st.set_page_config(page_title="Venante ou d'occasion", page_icon="🚘", layout="wide")


# ---------- Objets issus du notebook (chargés une seule fois) ----------
@st.cache_resource
def charger_objets():
    encoders = jb.load("encoders.joblib")   # Marque, Transmission, Quartier, Etat
    uniques = jb.load("uniques.joblib")     # modalités de chaque variable texte
    scaler = jb.load("scaler.joblib")       # StandardScaler
    modele = jb.load("best_model.joblib")   # classifieur retenu
    return encoders, uniques, scaler, modele


encoders, uniques, scaler, modele = charger_objets()
etats = uniques[-1]  # D'occasion / Venant


# ---------- Prédiction pour une voiture ----------
def Pred_func(marque, annee, transmission, quartier, prix):
    code_marque = encoders[0].transform([marque])[0]
    code_transmission = encoders[1].transform([transmission])[0]
    code_quartier = encoders[2].transform([quartier])[0]
    # même ordre de colonnes que dans le notebook : Marque, Année, Transmission, Quartier, Prix
    vecteur = np.array([code_marque, annee, code_transmission, code_quartier, prix]).reshape(1, -1)
    vecteur_norm = scaler.transform(vecteur)
    classe = modele.predict(vecteur_norm)[0]
    return etats[classe]


# ---------- Prédiction pour un fichier ----------
def Pred_func_csv(fichier):
    tableau = pd.read_csv(fichier)
    resultats = []
    for ligne in tableau.values:
        resultats.append(Pred_func(ligne[0], ligne[1], ligne[2], ligne[3], ligne[4]))
    tableau["Etat prédit"] = resultats
    return tableau


# ---------- Barre latérale ----------
with st.sidebar:
    st.header("À propos")
    st.write("Modèle entraîné sur des annonces de voitures à Dakar (expat-dakar.com). "
             "Sept algorithmes ont été comparés ; le meilleur sur la validation est utilisé ici.")
    st.caption("Projet Machine Learning — classification (TP1)")

st.title("🚘 Cette voiture est-elle venante ou d'occasion ?")
choix = st.radio("Mode", ["Une voiture", "Un fichier CSV"], horizontal=True)

if choix == "Une voiture":
    gauche, droite = st.columns(2)
    with gauche:
        marque = st.selectbox("Marque", list(uniques[0]))
        transmission = st.selectbox("Boîte de vitesses", list(uniques[1]))
        quartier = st.selectbox("Quartier de vente", list(uniques[2]))
    with droite:
        annee = st.slider("Année", 1990, 2024, 2015)
        prix = st.number_input("Prix demandé (FCFA)", min_value=0, value=8_000_000, step=250_000)

    if st.button("Lancer la prédiction", type="primary"):
        try:
            st.metric("État estimé", Pred_func(marque, annee, transmission, quartier, prix))
        except Exception as erreur:
            st.error(f"Prédiction impossible : {erreur}")
else:
    st.info("Colonnes attendues, dans cet ordre : Marque, Année, Transmission, Quartier, Prix.")
    fichier = st.file_uploader("Choisir un fichier CSV", type="csv")
    if fichier is not None:
        try:
            tableau = Pred_func_csv(fichier)
            st.dataframe(tableau, use_container_width=True)
            st.download_button("Récupérer les résultats (CSV)", tableau.to_csv(index=False).encode("utf-8"),
                               "resultats_voitures.csv", "text/csv")
        except Exception as erreur:
            st.error(f"Fichier non traité : {erreur}")
