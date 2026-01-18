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
# FIN PARTIE 1
# ============================================================
