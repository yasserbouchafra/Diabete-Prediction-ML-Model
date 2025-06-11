import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix
from fpdf import FPDF
from datetime import datetime
import io
import warnings
import streamlit.components.v1 as components

st.set_page_config(
    page_title="GlucoGuard",
    page_icon="⚕️",
    layout="wide"
)

warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use('seaborn-v0_8-whitegrid')

LANGUAGES = {
    "Français": {
        "app_title": "GlucoGuard", "app_subtitle": "Plateforme d'Analyse Prédictive du Risque de Diabète",
        "analysis_mode": "Mode d'Analyse", "individual_analysis": "Analyse Individuelle", "batch_analysis": "Analyse par Lot",
        "model_config": "Configuration du Modèle", "model_choice": "Modèle d'IA", "preprocessing": "Gestion des Données Anormales (0)",
        "median": "Remplacer par la Médiane", "mean": "Remplacer par la Moyenne", "model_performance": "Performance du Modèle Actif",
        "auc_score": "Score de Performance (AUC)", "confusion_matrix": "Matrice de Confusion", "patient_data": "Données du Patient",
        "patient_id": "Identifiant du Patient", "risk_summary": "Synthèse du Risque", "risk_score_label": "Score de Risque de Diabète",
        "risk_high": "Élevé", "risk_low": "Faible", "risk_factors": "Facteurs Influents", "recommendations_tab": "Informations Clés",
        "xai_tab": "Analyse Détaillée (XAI)", "simulation_tab": "Simulation 'What-If'", "generate_pdf": "Générer un Rapport PDF",
        "download_report": "Télécharger le Rapport", "report_for": "Rapport pour le patient", "disclaimer": "Ce rapport est généré par une IA et ne remplace pas un avis médical professionnel.",
        "disclaimer_app": "Application développée à des fins d'illustration. Ne pas utiliser pour un auto-diagnostic.",
        "analyze_button": "Analyser le Patient", "upload_prompt": "Importez un fichier CSV. Colonnes requises :", "batch_spinner": "Analyse de {n} patients en cours...",
        "batch_success": "Analyse terminée !", "batch_error": "Erreur de traitement du fichier", "download_results": "Télécharger les Résultats",
        "cohort_dashboard": "Dashboard de Cohorte", "risk_dist": "Distribution des Scores de Risque", "pred_dist": "Répartition des Prédictions",
        "global_factors": "Facteurs d'Influence Globaux", "waterfall_info": "Ce graphique décompose la prédiction, montrant la contribution de chaque facteur."
    },
    "English": {
        "app_title": "GlucoGuard", "app_subtitle": "Predictive Diabetes Risk Analysis Platform",
        "analysis_mode": "Analysis Mode", "individual_analysis": "Individual Analysis", "batch_analysis": "Batch Analysis",
        "model_config": "Model Configuration", "model_choice": "AI Model", "preprocessing": "Abnormal Data Handling (0)",
        "median": "Replace with Median", "mean": "Replace with Mean", "model_performance": "Active Model Performance",
        "auc_score": "Performance Score (AUC)", "confusion_matrix": "Confusion Matrix", "patient_data": "Patient Data",
        "patient_id": "Patient Identifier", "risk_summary": "Risk Summary", "risk_score_label": "Diabetes Risk Score",
        "risk_high": "High", "risk_low": "Low", "risk_factors": "Influential Factors", "recommendations_tab": "Key Insights",
        "xai_tab": "Detailed Analysis (XAI)", "simulation_tab": "What-If Simulation", "generate_pdf": "Generate PDF Report",
        "download_report": "Download Report", "report_for": "Report for patient", "disclaimer": "This report is AI-generated and does not substitute for professional medical advice.",
        "disclaimer_app": "Application developed for illustrative purposes. Do not use for self-diagnosis.",
        "analyze_button": "Analyze Patient", "upload_prompt": "Upload a CSV file. Required columns:", "batch_spinner": "Analyzing {n} patients...",
        "batch_success": "Analysis complete!", "batch_error": "File processing error", "download_results": "Download Results",
        "cohort_dashboard": "Cohort Dashboard", "risk_dist": "Risk Score Distribution", "pred_dist": "Prediction Distribution",
        "global_factors": "Global Influential Factors", "waterfall_info": "This waterfall plot breaks down the prediction, showing each factor's contribution."
    }
}
if 'lang' not in st.session_state: st.session_state.lang = "Français"
def t(key): return LANGUAGES[st.session_state.lang].get(key, key)

class PDFReport(FPDF):
    def header(self): self.set_font('Arial', 'B', 16); self.cell(0, 10, 'GlucoGuard - Risk Analysis Report', 0, 1, 'C'); self.ln(5)
    def footer(self): self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C'); self.ln(4); self.set_font('Arial', 'I', 6); self.multi_cell(0, 3, t('disclaimer'), 0, 'C')
    def add_section_title(self, title): self.set_font('Arial', 'B', 12); self.set_fill_color(230, 230, 230); self.cell(0, 8, title, 0, 1, 'L', fill=True); self.ln(4)
    def add_summary(self, patient_id, data, risk_score, prediction):
        self.set_font('Arial', '', 10); self.cell(0, 6, f"{t('report_for')}: {patient_id}", 0, 1); self.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1); self.ln(5)
        self.set_font('Arial', 'B', 14); risk_level = t('risk_high') if prediction == 1 else t('risk_low'); self.cell(0, 10, f"{t('risk_score_label')}: {risk_score}% ({risk_level})", 0, 1); self.ln(5)
        self.set_font('Arial', '', 10)
        for key, value in data.to_dict('records')[0].items(): self.cell(95, 6, f"{key.replace('_', ' ').title()}:", 0, 0); self.cell(95, 6, f"{value:.2f}" if isinstance(value, float) else str(value), 0, 1)
        self.ln(5)
    def add_waterfall_plot(self, shap_values):
        try:
            fig, ax = plt.subplots(figsize=(8, 4)); shap.plots.waterfall(shap_values, max_display=10, show=False); plt.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150); buf.seek(0)
            self.image(buf, x=10, w=190); plt.close(fig); self.ln(5)
        except Exception as e:
            self.set_font('Arial', 'I', 10); self.set_text_color(255, 0, 0); self.cell(0, 6, f"Error generating plot: {e}", 0, 1); self.set_text_color(0, 0, 0)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, scrolling=False)

@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv('diabetes.csv'); df.columns = [col.lower() for col in df.columns]; return df
    except FileNotFoundError: return None

def preprocess_data(df, strategy):
    df_processed = df.copy(); cols_to_clean = ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi']
    for col in cols_to_clean:
        df_processed[col] = df_processed[col].replace(0, np.nan)
        impute_value = df_processed[col].median() if strategy == 'median' else df_processed[col].mean()
        df_processed[col].fillna(impute_value, inplace=True)
    return df_processed

@st.cache_resource(show_spinner=False)
def get_trained_model(model_name, prep_strat_key):
    df_raw = load_data()
    if df_raw is None: return None, None, None, None
    prep_strat = 'median' if prep_strat_key == t("median") else 'mean'
    df = preprocess_data(df_raw, prep_strat); features = [c for c in df.columns if c != 'outcome']; target = 'outcome'
    X, y = df[features], df[target]; X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {"Random Forest": RandomForestClassifier(n_estimators=200, random_state=42), "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), "Régression Logistique": LogisticRegression(solver='liblinear', random_state=42)}
    model = models[model_name]; model.fit(X_train, y_train)
    explainer = shap.Explainer(model, X_train)
    y_pred = model.predict(X_test); y_proba = model.predict_proba(X_test)[:, 1]
    performance = {'auc': roc_auc_score(y_test, y_proba), 'cm': confusion_matrix(y_test, y_pred)}
    return model, explainer, performance, features

def render_sidebar():
    st.sidebar.title("GlucoGuard"); st.sidebar.markdown(f"*{t('app_subtitle')}*")
    lang_choice = st.sidebar.radio("Language / Langue", list(LANGUAGES.keys()), horizontal=True, index=list(LANGUAGES.keys()).index(st.session_state.lang))
    if lang_choice != st.session_state.lang: st.session_state.lang = lang_choice; st.rerun()
    
    app_mode = st.sidebar.selectbox(t("analysis_mode"), [t("individual_analysis"), t("batch_analysis")])
    with st.sidebar.expander(t("model_config"), expanded=False):
        model_choice = st.selectbox(t("model_choice"), ["Random Forest", "XGBoost", "Régression Logistique"])
        prep_choice = st.radio(t("preprocessing"), (t("median"), t("mean")), horizontal=True)
    
    with st.spinner(t('model_config')):
        model, explainer, perf, feats = get_trained_model(model_choice, prep_choice)
    
    with st.sidebar.expander(t("model_performance"), expanded=True):
        if perf:
            st.metric(t("auc_score"), f"{perf['auc']:.3f}")
            fig = px.imshow(perf['cm'], text_auto=True, labels=dict(x="Prédit", y="Réel"), color_continuous_scale='Blues')
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), title_text=t('confusion_matrix'), title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        else: st.error("Modèle non chargé.")
    st.sidebar.markdown("---"); st.sidebar.caption(t("disclaimer_app"))
    
    return app_mode, model, explainer, feats, prep_choice, perf

def render_individual_analysis(model, explainer, features):
    st.title(t("individual_analysis"))
    with st.form("patient_data_form"):
        st.subheader(t("patient_data")); patient_id = st.text_input(t("patient_id"), "Patient-001"); cols = st.columns(2)
        input_data = {
            'pregnancies': cols[0].slider('Pregnancies', 0, 17, 3), 'glucose': cols[0].slider('Glucose (mg/dL)', 40, 200, 110),
            'bloodpressure': cols[0].slider('Blood Pressure (mm Hg)', 40, 122, 72), 'skinthickness': cols[0].slider('Skin Thickness (mm)', 7, 99, 20),
            'insulin': cols[1].slider('Insulin (mu U/ml)', 14, 846, 79), 'bmi': cols[1].slider('BMI', 18.0, 67.0, 32.0, 0.1),
            'diabetespedigreefunction': cols[1].slider('Diabetes Pedigree Function', 0.078, 2.42, 0.372, 0.001), 'age': cols[1].slider('Age', 21, 81, 29)
        }
        submitted = st.form_submit_button(t('analyze_button'), type="primary")
    
    if not submitted: st.info("Veuillez remplir les données du patient et cliquer sur 'Analyser'."); st.stop()

    input_df = pd.DataFrame(input_data, index=[0]); pred_proba = model.predict_proba(input_df)[0, 1]; pred = model.predict(input_df)[0]; risk_score = round(pred_proba * 100); shap_values = explainer(input_df)
    
    st.subheader(t("risk_summary")); c1, c2 = st.columns([1, 2])
    c1.metric(t("risk_score_label"), f"{risk_score}%", f"{t('risk_high')} Risk" if risk_score >= 50 else f"{t('risk_low')} Risk", "inverse"); c1.progress(float(pred_proba))
    with c2: st.markdown(f"**{t('risk_factors')}**"); st_shap(shap.plots.force(shap_values[0, :, 1]), 160)
    
    tab1, tab2, tab3 = st.tabs([f"ℹ️ {t('recommendations_tab')}", f"📊 {t('xai_tab')}", f"⚙️ {t('simulation_tab')}"])
    with tab1: st.info(f"**IMC:** {input_df['bmi'].values[0]:.1f} (Healthy: 18.5-24.9). **Glucose:** {input_df['glucose'].values[0]} mg/dL (Normal Fasting: <100).")
    with tab2: st.info(t('waterfall_info')); fig, ax = plt.subplots(); shap.plots.waterfall(shap_values[0,:,1], max_display=10, show=False); st.pyplot(fig, use_container_width=True)
    with tab3:
        sim_df = input_df.copy(); c1, c2 = st.columns(2); sim_df['bmi'] = c1.slider("Simulate BMI", 18.0, 67.0, float(sim_df['bmi']), 0.1, key="sim_bmi"); sim_df['glucose'] = c2.slider("Simulate Glucose", 40, 200, int(sim_df['glucose']), key="sim_glucose"); sim_proba = model.predict_proba(sim_df)[0, 1]; sim_risk_score = round(sim_proba * 100); st.metric("Simulated Risk Score", f"{sim_risk_score}%", f"{sim_risk_score - risk_score:+d} pts", "normal")
    
    with st.expander(t('generate_pdf')):
        pdf = PDFReport('P', 'mm', 'A4'); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page(); pdf.add_section_title("Patient & Risk Summary"); pdf.add_summary(patient_id, input_df, risk_score, pred)
        pdf.add_section_title("Risk Factor Analysis (Waterfall Plot)"); pdf.add_waterfall_plot(shap_values[0,:,1])
        pdf_output = pdf.output()
        st.download_button(t("download_report"), data=bytes(pdf_output), file_name=f"Report_GlucoGuard_{patient_id}.pdf", mime="application/pdf")

def render_batch_analysis(model, explainer, features, prep_strat_key):
    st.title(t("batch_analysis")); st.markdown(f"{t('upload_prompt')} `{', '.join(features)}`")
    uploaded_file = st.file_uploader("Upload", type="csv", label_visibility="collapsed")
    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file); batch_df.columns = [c.lower() for c in batch_df.columns]
            if not set(features).issubset(batch_df.columns): st.error(f"Invalid file. Missing columns: `{set(features) - set(batch_df.columns)}`")
            else:
                prep_strat = 'median' if prep_strat_key == t("median") else 'mean'
                with st.spinner(t('batch_spinner').format(n=len(batch_df))):
                    processed = preprocess_data(batch_df[features], prep_strat)
                    batch_df['prediction'] = ["High Risk" if p == 1 else "Low Risk" for p in model.predict(processed)]
                    batch_df['risk_score_%'] = np.round(model.predict_proba(processed)[:, 1] * 100).astype(int)
                    st.success(t('batch_success'))
                    tab1, tab2 = st.tabs(["📑 Detailed Results", f"📈 {t('cohort_dashboard')}"])
                    with tab1: st.dataframe(batch_df); st.download_button(t('download_results'), batch_df.to_csv(index=False).encode('utf-8'), "batch_results.csv", "text/csv")
                    with tab2:
                        c1, c2 = st.columns(2); fig = px.histogram(batch_df, x='risk_score_%', nbins=20, title=t('risk_dist')); c1.plotly_chart(fig, use_container_width=True); pie_data = batch_df['prediction'].value_counts().reset_index(); fig = px.pie(pie_data, values='count', names='prediction', title=t('pred_dist'), color_discrete_map={'High Risk':'#FF4B4B', 'Low Risk':'#28A745'}); c2.plotly_chart(fig, use_container_width=True)
                        st.subheader(t('global_factors')); shap_values = explainer(processed); fig, ax = plt.subplots(); shap.summary_plot(shap_values[:,:,1], processed, plot_type="bar", show=False); st.pyplot(fig, use_container_width=True)
        except Exception as e: st.error(f"{t('batch_error')}: {e}")

def main():
    if load_data() is None: st.stop()
    app_mode, model, explainer, features, prep_strat, perf = render_sidebar()
    if model:
        if app_mode == t("individual_analysis"): render_individual_analysis(model, explainer, features)
        else: render_batch_analysis(model, explainer, features, prep_strat)

if __name__ == "__main__":
    main()