import streamlit as st
import requests
import pandas as pd
import json
from streamlit_lottie import st_lottie
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Scoring Crédit", layout="wide", page_icon="📊")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# --- En-tête ---
with st.container():
    col_icon, col_title = st.columns([1, 10])
    with col_icon:
        st.image("https://img.icons8.com/ios-filled/100/1e293b/combo-chart--v1.png", width=60)  # Icône en noir élégant
    with col_title:
        st.markdown("""
            <style>
                .main-title {
                    font-size: 42px;
                    font-weight: 800;
                    color: #1e293b; /* Bleu foncé / noir bleuté */
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin-bottom: 5px;
                }
                .subtitle {
                    font-size: 17px;
                    color: #4b5563; /* Gris foncé */
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin-top: 0;
                }
            </style>
            <div>
                <div class="main-title">Tableau de bord – Scoring de Crédit</div>
                <div class="subtitle">Évaluez une demande ou explorez les données clients.</div>
            </div>
        """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/train.csv")
    except:
        return pd.DataFrame()

df = load_data()

# Seuil fixé à 10% (pas de slider)
SEUIL = 10

# Tabs
tab1, tab2 = st.tabs(["📈 Prédiction", "🔍 Infos Client"])

with tab1:
    with st.form("formulaire_credit"):
        st.subheader("📝 Informations du Demandeur")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Genre", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
            married = st.selectbox("État civil", [0, 1], format_func=lambda x: "Non Marié(e)" if x == 0 else "Marié(e)")
            dependents = st.number_input("Personnes à charge", min_value=0, step=1)
            education = st.selectbox("Éducation", [0, 1], format_func=lambda x: "Supérieur" if x == 0 else "Non Supérieur")
            self_employed = st.selectbox("Indépendant", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
        with col2:
            applicant_income = st.number_input("Revenu Demandeur", min_value=0)
            coapplicant_income = st.number_input("Revenu Co-demandeur", min_value=0)
            loan_amount = st.number_input("Montant du prêt (en milliers)", min_value=0)
            loan_term = st.number_input("Durée du prêt (mois)", min_value=1)
            credit_history = st.selectbox("Historique de crédit", [0.0, 1.0], format_func=lambda x: "Mauvais" if x == 0.0 else "Bon")
            property_area = st.selectbox("Zone", [0, 1, 2], format_func=lambda x: ["Rurale", "Urbaine", "Semi-urbaine"][x])

        submitted = st.form_submit_button("Évaluer ")

    if submitted:
        with st.spinner("Analyse en cours..."):
            data = {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": credit_history,
                "Property_Area": property_area
            }

            try:
                response = requests.post("https://api-scoring-c8xa.onrender.com/predict", json=data)
                if response.status_code == 200:
                    result = response.json()
                    proba = float(result['Probabilité de défaut']) * 100

                    if proba <= SEUIL:
                        statut_affiche = "Approuvé"
                        st.success(f"✅ Crédit Approuvé (risque {proba:.2f}%) – Seuil fixé à {SEUIL}%")
                    else:
                        statut_affiche = "Refusé"
                        st.error(f"❌ Crédit Refusé (risque {proba:.2f}%) – Seuil fixé à {SEUIL}%")

                    # Jauge Plotly
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba,
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "red" if statut_affiche == "Refusé" else "green"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgreen"},
                                {'range': [50, 100], 'color': "salmon"}
                            ]
                        },
                        title={'text': "Probabilité de Défaut (%)"}
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                    # Camembert
                    st.markdown("#### 🔍 Répartition de la décision")
                    labels = ['Approuvé', 'Refusé']
                    values = [100 - proba, proba]
                    colors = ['#27ae60', '#e74c3c']
                    fig1, ax1 = plt.subplots(figsize=(4, 4))
                    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                    ax1.axis('equal')
                    st.pyplot(fig1)
                    plt.close(fig1)

                    # Histogramme revenus
                    if not df.empty:
                        st.markdown("#### 📊 Comparaison du revenu avec les autres clients")
                        plt.figure(figsize=(8, 3))
                        sns.histplot(df["ApplicantIncome"], color="lightblue", label="Population")
                        plt.axvline(applicant_income, color="red", linestyle="--", label="Client actuel")
                        plt.legend()
                        st.pyplot(plt.gcf())
                        plt.clf()
                else:
                    st.error("❌ Erreur lors de la prédiction.")
            except Exception as e:
                st.error(f"🚨 Erreur de connexion à l'API : {e}")

with tab2:
    st.subheader("📇 Sélectionner un client pour afficher ses informations")

    SEUIL = 10  # Seuil fixe

    try:
        df_test = pd.read_csv("data/test.csv")
        id_col = [col for col in df_test.columns if "id" in col.lower()]
        if id_col:
            id_col = id_col[0]
            id_list = df_test[id_col].dropna().unique().tolist()
            selected_id = st.selectbox("🔍 Sélectionnez un ID Client", id_list)

            client_data = df_test[df_test[id_col] == selected_id]
            if not client_data.empty:
                st.success(f"✅ Données du client {selected_id}")

                infos = client_data.iloc[0].to_dict()

                # ✅ Nouveau CSS pour mise en forme élégante
                st.markdown("""
                <style>
                .card-client {
                    background-color: #ffffff;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                    padding: 20px 25px;
                    margin-bottom: 15px;
                    font-family: 'Segoe UI', sans-serif;
                    transition: 0.3s;
                }
                .card-client:hover {
                    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
                }
                .card-title {
                    font-size: 18px;
                    font-weight: 600;
                    color: #2563eb;
                    margin-bottom: 10px;
                }
                .card-value {
                    font-size: 16px;
                    font-weight: 500;
                    color: #1e293b;
                }
                .card-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
                    gap: 20px;
                }
                </style>
                """, unsafe_allow_html=True)

                # 🔄 Mapping valeurs binaires
                st.markdown('<div class="card-grid">', unsafe_allow_html=True)
                for key, value in infos.items():
                    if isinstance(value, (int, float)) and value in [0, 1]:
                        mapping = {
                            "gender": {0: "Femme", 1: "Homme"},
                            "married": {0: "Non Marié(e)", 1: "Marié(e)"},
                            "education": {0: "Supérieur", 1: "Non Supérieur"},
                            "self_employed": {0: "Non", 1: "Oui"},
                            "credit_history": {0: "Mauvais", 1: "Bon"},
                        }
                        key_lower = key.lower()
                        value = mapping.get(key_lower, {}).get(value, value)

                    key_clean = key.replace("_", " ").capitalize()
                    st.markdown(f"""
                    <div class="card-client">
                        <div class="card-title">{key_clean}</div>
                        <div class="card-value">{value}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # 🔘 Prédiction
                if st.button("🔍 Prédire ce client"):
                    with st.spinner("Analyse en cours..."):

                        def text_to_int(key, val):
                            mappings = {
                                "Gender": {"Femme": 0, "Homme": 1, "Female": 0, "Male": 1},
                                "Married": {"Non Marié(e)": 0, "Marié(e)": 1, "No": 0, "Yes": 1},
                                "Education": {"Supérieur": 0, "Non Supérieur": 1, "Graduate": 0, "Not Graduate": 1},
                                "Self_Employed": {"Non": 0, "Oui": 1, "No": 0, "Yes": 1},
                                "Credit_History": {"Mauvais": 0, "Bon": 1, 0.0: 0, 1.0: 1, 0: 0, 1: 1},
                                "Property_Area": {
                                    "Rurale": 0, "Urbaine": 1, "Semi-urbaine": 2,
                                    "Rural": 0, "Urban": 1, "Semiurban": 2,
                                    "RURAL": 0, "URBAN": 1, "SEMIURBAN": 2
                                }
                            }
                            return mappings.get(key, {}).get(val, val)

                            

                        prediction_data = {
                            "Gender": infos.get("Gender"),
                            "Married": infos.get("Married"),
                            "Dependents": infos.get("Dependents"),
                            "Education": infos.get("Education"),
                            "Self_Employed": infos.get("Self_Employed"),
                            "ApplicantIncome": infos.get("ApplicantIncome"),
                            "CoapplicantIncome": infos.get("CoapplicantIncome"),
                            "LoanAmount": infos.get("LoanAmount"),
                            "Loan_Amount_Term": infos.get("Loan_Amount_Term"),
                            "Credit_History": infos.get("Credit_History"),
                            "Property_Area": infos.get("Property_Area")
                        }

                        prediction_data_mapped = {k: text_to_int(k, v) for k, v in prediction_data.items()}

                        try:
                            response = requests.post("https://api-scoring-c8xa.onrender.com/predict", json=prediction_data_mapped)
                            if response.status_code == 200:
                                result = response.json()
                                proba = float(result['Probabilité de défaut']) * 100

                                if proba <= SEUIL:
                                    st.success(f"✅ Crédit Approuvé (risque {proba:.2f}%) – Seuil fixé à {SEUIL}%")
                                else:
                                    st.error(f"❌ Crédit Refusé (risque {proba:.2f}%) – Seuil fixé à {SEUIL}%")

                                # Jauge
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=proba,
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "red" if proba > SEUIL else "green"},
                                        'steps': [
                                            {'range': [0, 50], 'color': "lightgreen"},
                                            {'range': [50, 100], 'color': "salmon"}
                                        ]
                                    },
                                    title={'text': "Probabilité de Défaut (%)"}
                                ))
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)

                            else:
                                st.error(f"❌ Erreur lors de la prédiction. Code: {response.status_code} – Détail: {response.text}")
                        except Exception as e:
                            st.error(f"🚨 Erreur de connexion à l'API : {e}")

            else:
                st.warning("⚠️ Client introuvable.")
        else:
            st.error("❌ Colonne ID non trouvée.")
    except FileNotFoundError:
        st.error("❌ Fichier test.csv introuvable.")
    except Exception as e:
        st.error(f"🚨 Erreur : {e}")

