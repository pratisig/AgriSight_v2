# ============================================================
# AgriSight Pro v3.0 — PARTIE 1 / 3
# ANALYSE AGRONOMIQUE AVEC DONNÉES RÉELLES
# ============================================================
# - Chargement zone (coords / future extension GeoJSON)
# - Données climatiques réelles (CHIRPS, NASA POWER)
# - NDVI réel Sentinel-2 (Google Earth Engine)
# - Persistance SQLite
# - UI Wizard robuste
# ============================================================

# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import ee
import requests
from datetime import date, datetime, timedelta
from shapely.geometry import Polygon

# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="AgriSight Pro v3.0 – Analyse",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 AgriSight Pro – Analyse agronomique")
st.caption("Données réelles satellitaires & climatiques")

# ============================================================
# INITIALISATION GOOGLE EARTH ENGINE
# ============================================================
try:
    ee.Initialize()
    GEE_READY = True
except Exception:
    GEE_READY = False

if not GEE_READY:
    st.error("Google Earth Engine non initialisé. Lancez ee.Authenticate() en local.")
    st.stop()

# ============================================================
# BASE DE DONNÉES SQLITE
# ============================================================
DB_PATH = "agrisight.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_name TEXT,
    latitude REAL,
    longitude REAL,
    surface_ha REAL,
    culture TEXT,
    soil_type TEXT,
    agro_zone TEXT,
    irrigation INTEGER,
    start_date TEXT,
    end_date TEXT,
    ndvi_mean REAL,
    rain_total REAL,
    temp_mean REAL,
    created_at TEXT
)
""")

conn.commit()
conn.close()

# ============================================================
# FONCTIONS DONNÉES RÉELLES
# ============================================================

def get_ndvi_sentinel2(aoi, start, end):
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
    )

    ndvi = collection.mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,
        maxPixels=1e9
    ).get("NDVI")

    return ndvi.getInfo()


def get_rain_chirps(aoi, start, end):
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(aoi)
        .filterDate(start, end)
    )

    rain = chirps.sum().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=5000,
        maxPixels=1e9
    ).get("precipitation")

    return rain.getInfo()


def get_temperature_nasa(lat, lon, start, end):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M"
        f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}"
        f"&latitude={lat}&longitude={lon}"
        f"&community=AG&format=JSON"
    )

    r = requests.get(url, timeout=30)
    data = r.json()["properties"]["parameter"]["T2M"]
    return float(np.mean(list(data.values())))

# ============================================================
# SIDEBAR — WIZARD PARAMÉTRAGE
# ============================================================
st.sidebar.header("🧭 Paramétrage de l'analyse")

zone_name = st.sidebar.text_input("Nom de la parcelle", "Parcelle test")
lat = st.sidebar.number_input("Latitude", value=14.7, format="%.6f")
lon = st.sidebar.number_input("Longitude", value=-17.4, format="%.6f")
surface = st.sidebar.number_input("Surface (ha)", min_value=0.1, value=1.0)

culture = st.sidebar.selectbox("Culture", ["Mil", "Maïs", "Riz", "Sorgho"])
soil_type = st.sidebar.selectbox("Type de sol", ["Sableux", "Argileux", "Limon"])
agro_zone = st.sidebar.selectbox("Zone agroécologique", ["Sahel", "Savane", "Zone irriguée"])
irrigation = st.sidebar.checkbox("Irrigation disponible")

start_date = st.sidebar.date_input("Date début", date.today() - timedelta(days=90))
end_date = st.sidebar.date_input("Date fin", date.today() - timedelta(days=10))

if start_date >= end_date:
    st.sidebar.error("La date de début doit être antérieure à la date de fin")

# ============================================================
# BOUTON ANALYSE
# ============================================================
if st.sidebar.button("🚀 Lancer l'analyse"):

    aoi = ee.Geometry.Polygon([[
        [lon - 0.01, lat - 0.01],
        [lon + 0.01, lat - 0.01],
        [lon + 0.01, lat + 0.01],
        [lon - 0.01, lat + 0.01],
        [lon - 0.01, lat - 0.01]
    ]])

    with st.spinner("Analyse en cours avec données réelles..."):
        ndvi = get_ndvi_sentinel2(aoi, start_date, end_date)
        rain = get_rain_chirps(aoi, start_date, end_date)
        temp = get_temperature_nasa(lat, lon, start_date, end_date)

    st.subheader("📊 Indicateurs agro-climatiques")
    st.metric("NDVI moyen (Sentinel-2)", round(ndvi, 3))
    st.metric("Pluie cumulée (mm)", round(rain, 1))
    st.metric("Température moyenne (°C)", round(temp, 1))

    conn = sqlite3.connect(DB_PATH)
    pd.DataFrame([{
        "zone_name": zone_name,
        "latitude": lat,
        "longitude": lon,
        "surface_ha": surface,
        "culture": culture,
        "soil_type": soil_type,
        "agro_zone": agro_zone,
        "irrigation": int(irrigation),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "ndvi_mean": ndvi,
        "rain_total": rain,
        "temp_mean": temp,
        "created_at": datetime.utcnow().isoformat()
    }]).to_sql("analyses", conn, if_exists="append", index=False)
    conn.close()

# ============================================================
# HISTORIQUE
# ============================================================
st.markdown("---")
st.subheader("📚 Historique des analyses")

conn = sqlite3.connect(DB_PATH)
hist = pd.read_sql("SELECT * FROM analyses ORDER BY created_at DESC", conn)
conn.close()

st.dataframe(hist, use_container_width=True)

# ============================================================
# AgriSight Pro v3.0 — PARTIE 2 / 3
# MOTEUR DE RECOMMANDATIONS AGRONOMIQUES CONTEXTUALISÉES
# ============================================================
# Cette partie complète la PARTIE 1 en ajoutant :
# - Base de connaissances agronomiques (SQLite)
# - Recommandations NPK NON génériques
# - Prise en compte du sol, zone, irrigation, climat
# - Scoring de confiance et justification
# ============================================================

# =========================
# IMPORTS
# =========================
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "agrisight.db"

# ============================================================
# INITIALISATION TABLE RECOMMANDATIONS
# ============================================================

def init_recommendations_table():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            culture TEXT NOT NULL,
            soil_type TEXT NOT NULL,
            agro_zone TEXT NOT NULL,
            irrigation INTEGER NOT NULL,
            n REAL NOT NULL,
            p REAL NOT NULL,
            k REAL NOT NULL,
            organic_advice TEXT,
            confidence_base REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_recommendations_table()

# ============================================================
# ALIMENTATION INITIALE DE LA BASE AGRONOMIQUE
# ============================================================

def seed_recommendations():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM recommendations")
    if cur.fetchone()[0] > 0:
        conn.close()
        return

    # Base réaliste adaptée Afrique de l'Ouest / Sahel
    data = [
        # culture, sol, zone, irrigation, N, P, K, conseil, confiance
        ("Mil", "Sableux", "Sahel", 0, 40, 20, 20, "Apport fumier bien décomposé", 0.85),
        ("Mil", "Argileux", "Sahel", 0, 30, 15, 15, "Fractionner l'azote", 0.80),
        ("Mil", "Limon", "Savane", 0, 45, 25, 25, "Gestion des résidus conseillée", 0.82),

        ("Maïs", "Limon", "Savane", 1, 120, 60, 60, "Compost + NPK minéral", 0.90),
        ("Maïs", "Argileux", "Savane", 0, 90, 45, 45, "Labour léger recommandé", 0.83),
        ("Maïs", "Sableux", "Savane", 1, 110, 55, 55, "Irrigation régulière", 0.86),

        ("Riz", "Argileux", "Zone irriguée", 1, 100, 50, 50, "Gestion stricte de l'eau", 0.88),
        ("Riz", "Limon", "Zone irriguée", 1, 95, 45, 45, "Nivellement du sol conseillé", 0.84),

        ("Sorgho", "Sableux", "Sahel", 0, 50, 25, 25, "Paillage pour conserver humidité", 0.83),
        ("Sorgho", "Argileux", "Savane", 0, 55, 30, 30, "Rotation culturale recommandée", 0.85)
    ]

    cur.executemany("""
        INSERT INTO recommendations (
            culture, soil_type, agro_zone, irrigation,
            n, p, k, organic_advice, confidence_base
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    conn.close()

seed_recommendations()

# ============================================================
# MOTEUR DE MATCHING ET DE SCORING
# ============================================================

def recommend(
    culture: str,
    soil_type: str,
    agro_zone: str,
    irrigation: int,
    ndvi: float,
    rainfall: float
):
    """
    Génère une recommandation agronomique contextualisée
    avec score de confiance et justification.
    """

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM recommendations WHERE culture = ?",
        conn,
        params=(culture,)
    )
    conn.close()

    if df.empty:
        return None

    scores = []

    for _, row in df.iterrows():
        score = row["confidence_base"]

        # Matching contexte
        if row["soil_type"] == soil_type:
            score += 0.05
        if row["agro_zone"] == agro_zone:
            score += 0.05
        if row["irrigation"] == irrigation:
            score += 0.05

        # Ajustements dynamiques basés sur observations
        if ndvi < 0.3:
            score -= 0.05
        if rainfall < 300 and irrigation == 0:
            score -= 0.05
        if rainfall > 800:
            score -= 0.03  # risque lessivage

        scores.append(score)

    df["final_score"] = scores
    best = df.sort_values("final_score", ascending=False).iloc[0]

    justification = (
        f"Recommandation basée sur :\n"
        f"- Culture : {culture}\n"
        f"- Type de sol : {soil_type}\n"
        f"- Zone agroécologique : {agro_zone}\n"
        f"- Irrigation : {'Oui' if irrigation else 'Non'}\n"
        f"- NDVI moyen observé : {ndvi:.2f}\n"
        f"- Pluviométrie cumulée : {rainfall:.0f} mm"
    )

    return {
        "Azote (N) kg/ha": best["n"],
        "Phosphore (P) kg/ha": best["p"],
        "Potassium (K) kg/ha": best["k"],
        "Conseil organique": best["organic_advice"],
        "Score de confiance": round(best["final_score"], 2),
        "Justification": justification
    }

# ============================================================
# TEST LOCAL (OPTIONNEL)
# ============================================================
if __name__ == "__main__":
    test = recommend(
        culture="Mil",
        soil_type="Sableux",
        agro_zone="Sahel",
        irrigation=0,
        ndvi=0.28,
        rainfall=220
    )
    print(test)

# ============================================================
# AgriSight Pro v3.0 — PARTIE 3 / 3
# VALIDATION, BENCHMARK, EXPORT & CRÉDIBILITÉ SCIENTIFIQUE
# ============================================================
# Cette partie apporte :
# - Contrôle qualité (QC) des données
# - Indicateurs d'incertitude
# - Benchmark rendement (proxy NDVI / pluie)
# - Export PDF & Excel
# - Préparation multi-langue
# ============================================================

# =========================
# IMPORTS
# =========================
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

DB_PATH = "agrisight.db"

# ============================================================
# CONTRÔLE QUALITÉ (QC)
# ============================================================

def quality_control(ndvi, rainfall, temperature):
    flags = []

    if ndvi is None or ndvi < 0 or ndvi > 1:
        flags.append("NDVI hors plage valide")

    if rainfall < 0:
        flags.append("Pluviométrie négative")

    if temperature < 5 or temperature > 45:
        flags.append("Température atypique")

    status = "OK" if not flags else "À vérifier"

    return {
        "status": status,
        "flags": flags
    }

# ============================================================
# INCERTITUDE SIMPLIFIÉE
# ============================================================

def uncertainty_index(ndvi, rainfall):
    uncertainty = 0.0

    if ndvi < 0.3:
        uncertainty += 0.2
    if rainfall < 300:
        uncertainty += 0.2
    if rainfall > 900:
        uncertainty += 0.15

    return min(1.0, uncertainty)

# ============================================================
# BENCHMARK RENDEMENT (PROXY)
# ============================================================

def yield_benchmark(culture, ndvi, rainfall):
    base_yield = {
        "Mil": 1.2,
        "Sorgho": 1.3,
        "Maïs": 3.5,
        "Riz": 4.0
    }

    y = base_yield.get(culture, 1.0)
    y *= (ndvi / 0.5)
    y *= min(1.2, rainfall / 600)

    return round(max(0.3, y), 2)

# ============================================================
# EXPORT EXCEL
# ============================================================

def export_excel(analysis_row: dict, recommendation: dict):
    output = BytesIO()

    df_analysis = pd.DataFrame([analysis_row])
    df_reco = pd.DataFrame([recommendation])

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_analysis.to_excel(writer, sheet_name="Analyse", index=False)
        df_reco.to_excel(writer, sheet_name="Recommandation", index=False)

    output.seek(0)
    return output

# ============================================================
# EXPORT PDF
# ============================================================

def export_pdf(analysis_row: dict, recommendation: dict, qc: dict, benchmark: float):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("AgriSight Pro – Rapport agronomique", styles['Title']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Données analysées</b>", styles['Heading2']))
    table_data = [[k, str(v)] for k, v in analysis_row.items()]
    elements.append(Table(table_data))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Recommandations</b>", styles['Heading2']))
    reco_data = [[k, str(v)] for k, v in recommendation.items()]
    elements.append(Table(reco_data))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"<b>Contrôle qualité :</b> {qc['status']}", styles['Normal']))
    if qc['flags']:
        for f in qc['flags']:
            elements.append(Paragraph(f"- {f}", styles['Normal']))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Rendement estimé :</b> {benchmark} t/ha", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ============================================================
# HISTORIQUE MULTI-SAISON
# ============================================================

def load_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM analyses ORDER BY created_at DESC", conn)
    conn.close()
    return df

# ============================================================
# MULTI-LANGUE (PRÉPARATION)
# ============================================================
TRANSLATIONS = {
    "fr": {
        "report_title": "Rapport agronomique",
        "quality_ok": "Qualité des données satisfaisante"
    },
    "en": {
        "report_title": "Agronomic report",
        "quality_ok": "Data quality acceptable"
    }
}

# ============================================================
# TEST LOCAL
# ============================================================
if __name__ == "__main__":
    print("PARTIE 3 prête : QC, benchmark et export.")

# ============================================================
# FIN PARTIE 3
# ============================================================


