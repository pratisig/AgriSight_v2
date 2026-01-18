app_part1 = """
# ============================================================
# AgriSight Pro v3.0 — PARTIE 1/3
# Données réelles + refactor architecture
# Backend satellite : Google Earth Engine (Sentinel-2, CHIRPS)
# ============================================================
# PRÉREQUIS :
# - Compte Google Earth Engine validé
# - pip install earthengine-api geemap
# - ee.Authenticate() puis ee.Initialize()
#
# LANCEMENT :
# streamlit run app_part1.py
# ============================================================

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import ee
from shapely.geometry import Polygon
from datetime import date, datetime, timedelta
import sqlite3
import requests

# ============================================================
# INITIALISATION GEE
# ============================================================
try:
    ee.Initialize()
    GEE_STATUS = True
except Exception:
    GEE_STATUS = False

# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="AgriSight Pro v3.0 (Partie 1)",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AgriSight Pro v3.0 — Données Réelles (Partie 1)")

# ============================================================
# BASE DE DONNÉES SQLITE
# ============================================================
DB_PATH = "agrisight.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(\"\"\"
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_name TEXT,
            culture TEXT,
            surface_ha REAL,
            start_date TEXT,
            end_date TEXT,
            ndvi_mean REAL,
            rain_total REAL,
            temp_mean REAL,
            created_at TEXT
        )
    \"\"\")
    conn.commit()
    conn.close()

init_db()

def save_analysis(row):
    conn = sqlite3.connect(DB_PATH)
    pd.DataFrame([row]).to_sql("analyses", conn, if_exists="append", index=False)
    conn.close()

# ============================================================
# FONCTIONS DONNÉES RÉELLES
# ============================================================
def get_ndvi_sentinel2(aoi, start, end):
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
    )
    ndvi = s2.mean().reduceRegion(
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

def get_temp_nasa(lat, lon, start, end):
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M"
        f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}"
        f"&latitude={lat}&longitude={lon}&format=JSON&community=AG"
    )
    r = requests.get(url, timeout=30)
    data = r.json()["properties"]["parameter"]["T2M"]
    return np.mean(list(data.values()))

# ============================================================
# WIZARD UI
# ============================================================
st.sidebar.header("🧭 Wizard d'analyse")
step = st.sidebar.radio(
    "Étapes",
    ["1️⃣ Zone", "2️⃣ Culture", "3️⃣ Période", "4️⃣ Analyse"]
)

if step == "1️⃣ Zone":
    zone_name = st.text_input("Nom de la parcelle", "Parcelle test")
    lat = st.number_input("Latitude", value=14.7)
    lon = st.number_input("Longitude", value=-17.4)
    surface = st.number_input("Surface (ha)", min_value=0.1, value=1.0)
    st.session_state["zone"] = dict(name=zone_name, lat=lat, lon=lon, surface=surface)

elif step == "2️⃣ Culture":
    culture = st.selectbox("Culture", ["Mil", "Maïs", "Riz", "Sorgho"])
    st.session_state["culture"] = culture

elif step == "3️⃣ Période":
    start = st.date_input("Début", date.today() - timedelta(days=90))
    end = st.date_input("Fin", date.today() - timedelta(days=10))
    st.session_state["dates"] = dict(start=start, end=end)

elif step == "4️⃣ Analyse":
    if not GEE_STATUS:
        st.error("Google Earth Engine non initialisé")
        st.stop()

    if st.button("Lancer l'analyse"):
        z = st.session_state["zone"]
        d = st.session_state["dates"]

        aoi = ee.Geometry.Polygon([[
            [z["lon"]-0.01, z["lat"]-0.01],
            [z["lon"]+0.01, z["lat"]-0.01],
            [z["lon"]+0.01, z["lat"]+0.01],
            [z["lon"]-0.01, z["lat"]+0.01],
            [z["lon"]-0.01, z["lat"]-0.01]
        ]])

        ndvi = get_ndvi_sentinel2(aoi, d["start"], d["end"])
        rain = get_rain_chirps(aoi, d["start"], d["end"])
        temp = get_temp_nasa(z["lat"], z["lon"], d["start"], d["end"])

        row = dict(
            zone_name=z["name"],
            culture=st.session_state["culture"],
            surface_ha=z["surface"],
            start_date=d["start"].isoformat(),
            end_date=d["end"].isoformat(),
            ndvi_mean=ndvi,
            rain_total=rain,
            temp_mean=temp,
            created_at=datetime.utcnow().isoformat()
        )
        save_analysis(row)
        st.success("Analyse terminée et enregistrée")
        st.json(row)

st.markdown("---")
st.subheader("📚 Historique")
conn = sqlite3.connect(DB_PATH)
hist = pd.read_sql("SELECT * FROM analyses ORDER BY created_at DESC", conn)
conn.close()
st.dataframe(hist)
"""

path = "/mnt/data/app_part1.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(app_part1)

path

Result

'/mnt/data/app_part1.py'
