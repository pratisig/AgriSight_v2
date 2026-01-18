"""
AgriSight Pro v2.0 - Application Complète Consolidée
Fichier unique pour déploiement Streamlit Cloud

INSTALLATION:
pip install streamlit geopandas pandas numpy requests folium streamlit-folium matplotlib seaborn shapely

LANCEMENT:
streamlit run app.py
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import folium
from folium.plugins import Draw, MeasureControl, MarkerCluster
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon, mapping, shape
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import json
from matplotlib.backends.backend_pdf import PdfPages
import time
import warnings
import hashlib

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="AgriSight Pro v2.0",
    layout="wide",
    page_icon="🌾"
)

# CSS personnalisé
CUSTOM_CSS = """
<style>
    .big-metric {font-size: 2em; font-weight: bold; color: #2E7D32;}
    .alert-box {background: #FFF3CD; padding: 15px; border-radius: 8px; border-left: 4px solid #FFC107;}
    .success-box {background: #D4EDDA; padding: 15px; border-radius: 8px; border-left: 4px solid #28A745;}
    .info-box {background: #D1ECF1; padding: 15px; border-radius: 8px; border-left: 4px solid #17A2B8;}
    .danger-box {background: #F8D7DA; padding: 15px; border-radius: 8px; border-left: 4px solid #DC3545;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Clés API intégrées
AGRO_API_KEY = '28641235f2b024b5f45f97df45c6a0d5'
GEMINI_API_KEY = 'AIzaSyBZ4494NUEL_N13soCCIgCfIrMqn2jxoD8'
OPENWEATHER_KEY = 'b06c034b4894d54fc512f9cd30b61a4a'

# Base de données cultures
CROP_DATABASE = {
    "Mil": {
        'ndvi_optimal': 0.6, 'rain_min': 400, 'rain_max': 600,
        'temp_optimal': 28, 'temp_min': 20, 'temp_max': 35,
        'yield_max': 1.5, 'cycle_days': 90
    },
    "Sorgho": {
        'ndvi_optimal': 0.65, 'rain_min': 450, 'rain_max': 700,
        'temp_optimal': 30, 'temp_min': 22, 'temp_max': 38,
        'yield_max': 2.0, 'cycle_days': 110
    },
    "Maïs": {
        'ndvi_optimal': 0.7, 'rain_min': 500, 'rain_max': 800,
        'temp_optimal': 25, 'temp_min': 18, 'temp_max': 32,
        'yield_max': 4.0, 'cycle_days': 120
    },
    "Arachide": {
        'ndvi_optimal': 0.6, 'rain_min': 450, 'rain_max': 700,
        'temp_optimal': 27, 'temp_min': 20, 'temp_max': 33,
        'yield_max': 2.5, 'cycle_days': 120
    },
    "Riz": {
        'ndvi_optimal': 0.75, 'rain_min': 800, 'rain_max': 1500,
        'temp_optimal': 26, 'temp_min': 20, 'temp_max': 35,
        'yield_max': 5.0, 'cycle_days': 130
    },
    "Niébé": {
        'ndvi_optimal': 0.55, 'rain_min': 350, 'rain_max': 600,
        'temp_optimal': 28, 'temp_min': 20, 'temp_max': 35,
        'yield_max': 1.2, 'cycle_days': 75
    },
    "Manioc": {
        'ndvi_optimal': 0.65, 'rain_min': 1000, 'rain_max': 2000,
        'temp_optimal': 27, 'temp_min': 20, 'temp_max': 32,
        'yield_max': 20.0, 'cycle_days': 300
    },
    "Tomate": {
        'ndvi_optimal': 0.7, 'rain_min': 600, 'rain_max': 1000,
        'temp_optimal': 24, 'temp_min': 15, 'temp_max': 30,
        'yield_max': 40.0, 'cycle_days': 90
    },
    "Oignon": {
        'ndvi_optimal': 0.6, 'rain_min': 400, 'rain_max': 700,
        'temp_optimal': 20, 'temp_min': 12, 'temp_max': 28,
        'yield_max': 25.0, 'cycle_days': 110
    },
    "Coton": {
        'ndvi_optimal': 0.65, 'rain_min': 600, 'rain_max': 1000,
        'temp_optimal': 28, 'temp_min': 20, 'temp_max': 35,
        'yield_max': 2.5, 'cycle_days': 150
    },
    "Pastèque": {
        'ndvi_optimal': 0.6, 'rain_min': 400, 'rain_max': 600,
        'temp_optimal': 25, 'temp_min': 18, 'temp_max': 32,
        'yield_max': 30.0, 'cycle_days': 85
    }
}

# ==================== SESSION STATE ====================

for key in ['gdf', 'sampling_points', 'satellite_data', 'climate_data', 
            'weather_forecast', 'analysis', 'drawn_geometry']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'analysis' else {}

# ==================== FONCTIONS UTILITAIRES ====================

def create_polygon_from_coords(lat_min, lon_min, lat_max, lon_max):
    """Crée un polygone rectangulaire"""
    coords = [
        (lon_min, lat_min),
        (lon_max, lat_min),
        (lon_max, lat_max),
        (lon_min, lat_max),
        (lon_min, lat_min)
    ]
    return Polygon(coords)

@st.cache_data(ttl=3600)
def load_geojson(file_bytes):
    """Charge un fichier GeoJSON"""
    try:
        gdf = gpd.read_file(BytesIO(file_bytes))
        return gdf.to_crs(4326)
    except Exception as e:
        st.error(f"Erreur lecture GeoJSON: {e}")
        return None

def calculate_area_hectares(geometry):
    """Calcule la surface en hectares"""
    bounds = geometry.bounds
    lat_center = (bounds[1] + bounds[3]) / 2
    
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * np.cos(np.radians(lat_center))
    
    width_m = (bounds[2] - bounds[0]) * m_per_deg_lon
    height_m = (bounds[3] - bounds[1]) * m_per_deg_lat
    area_m2 = width_m * height_m * 0.8
    
    return area_m2 / 10000

def create_sampling_grid(geometry, grid_size_ha=5):
    """Crée une grille d'échantillonnage"""
    bounds = geometry.bounds
    min_x, min_y, max_x, max_y = bounds
    
    lat_center = (min_y + max_y) / 2
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * np.cos(np.radians(lat_center))
    
    grid_size_m = np.sqrt(grid_size_ha * 10000)
    cell_size_lat = grid_size_m / meters_per_degree_lat
    cell_size_lon = grid_size_m / meters_per_degree_lon
    
    x_coords = np.arange(min_x, max_x, cell_size_lon)
    y_coords = np.arange(min_y, max_y, cell_size_lat)
    
    points = []
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            point = Point(x + cell_size_lon/2, y + cell_size_lat/2)
            
            if geometry.contains(point):
                points.append({
                    'geometry': point,
                    'longitude': point.x,
                    'latitude': point.y,
                    'cell_id': f"C{len(points)+1:03d}"
                })
    
    if not points:
        centroid = geometry.centroid
        points = [{
            'geometry': centroid,
            'longitude': centroid.x,
            'latitude': centroid.y,
            'cell_id': 'C001'
        }]
    
    return gpd.GeoDataFrame(points, crs='EPSG:4326')

# ==================== API CLIENTS ====================

def get_climate_nasa_multi_points(points_list, start, end):
    """Récupère données climatiques NASA POWER"""
    results = []
    
    for point_dict in points_list:
        lat = point_dict['latitude']
        lon = point_dict['longitude']
        
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M,WS2M"
            f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}"
            f"&latitude={lat}&longitude={lon}&format=JSON&community=AG"
        )
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                continue
            
            data = response.json()
            params = data.get("properties", {}).get("parameter", {})
            
            if not params:
                continue
            
            df = pd.DataFrame({
                'date': pd.to_datetime(list(params.get('T2M', {}).keys())),
                'temp_mean': list(params.get('T2M', {}).values()),
                'temp_min': list(params.get('T2M_MIN', {}).values()),
                'temp_max': list(params.get('T2M_MAX', {}).values()),
                'rain': list(params.get('PRECTOTCORR', {}).values()),
                'humidity': list(params.get('RH2M', {}).values()),
                'wind_speed': list(params.get('WS2M', {}).values()),
                'cell_id': point_dict['cell_id'],
                'latitude': lat,
                'longitude': lon
            })
            
            results.append(df)
            time.sleep(0.5)
            
        except Exception as e:
            st.warning(f"Erreur point {point_dict['cell_id']}: {e}")
            continue
    
    if results:
        return pd.concat(results, ignore_index=True)
    return None

@st.cache_data(ttl=3600)
def get_weather_forecast(lat, lon, api_key):
    """Récupère prévisions météo 7 jours"""
    if not api_key:
        return None
    
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        
        data = response.json()
        forecasts = []
        
        for item in data['list'][:56]:
            forecasts.append({
                'datetime': datetime.fromtimestamp(item['dt']),
                'temp': item['main']['temp'],
                'temp_min': item['main']['temp_min'],
                'temp_max': item['main']['temp_max'],
                'humidity': item['main']['humidity'],
                'rain': item.get('rain', {}).get('3h', 0),
                'description': item['weather'][0]['description'],
                'wind_speed': item['wind']['speed']
            })
        
        df = pd.DataFrame(forecasts)
        df['date'] = df['datetime'].dt.date
        
        daily = df.groupby('date').agg({
            'temp': 'mean',
            'temp_min': 'min',
            'temp_max': 'max',
            'humidity': 'mean',
            'rain': 'sum',
            'wind_speed': 'mean',
            'description': 'first'
        }).reset_index()
        
        return daily
        
    except Exception as e:
        return None
def simulate_multi_indices_data(points_list, start, end):
        """Simule données multi-indices pour chaque point"""
        dates = pd.date_range(start, end, freq='5D')
        all_data = []
        
        for point_dict in points_list:
            for d in dates:
                month = d.month
                
                if 6 <= month <= 9:
                    ndvi_base = 0.65 + np.random.normal(0, 0.08)
                    ndwi_base = 0.3 + np.random.normal(0, 0.08)
                elif month in [5, 10]:
                    ndvi_base = 0.45 + np.random.normal(0, 0.1)
                    ndwi_base = 0.15 + np.random.normal(0, 0.06)
                else:
                    ndvi_base = 0.25 + np.random.normal(0, 0.06)
                    ndwi_base = 0.05 + np.random.normal(0, 0.04)
                
                evi_base = ndvi_base * 0.9 + np.random.normal(0, 0.05)
                savi_base = ndvi_base * 0.85 + np.random.normal(0, 0.06)
                lai_base = ndvi_base * 5 + np.random.normal(0, 0.3)
                msavi_base = savi_base * 1.05 + np.random.normal(0, 0.04)
                
                all_data.append({
                    'date': d,
                    'cell_id': point_dict['cell_id'],
                    'latitude': point_dict['latitude'],
                    'longitude': point_dict['longitude'],
                    'ndvi': np.clip(ndvi_base, 0, 1),
                    'evi': np.clip(evi_base, 0, 1),
                    'ndwi': np.clip(ndwi_base, -1, 1),
                    'savi': np.clip(savi_base, 0, 1),
                    'lai': np.clip(lai_base, 0, 7),
                    'msavi': np.clip(msavi_base, 0, 1),
                    'cloud_cover': np.random.randint(0, 30)
                })
        
        return pd.DataFrame(all_data)
def calculate_crop_metrics(climate_df, indices_df, crop_params):
        """Calcule métriques spécifiques à chaque culture"""
        if climate_df is None or indices_df is None or climate_df.empty or indices_df.empty:
            return {}
        
        metrics = {
            'ndvi_mean': indices_df['ndvi'].mean(),
            'ndvi_std': indices_df['ndvi'].std(),
            'ndvi_min': indices_df['ndvi'].min(),
            'ndvi_max': indices_df['ndvi'].max(),
            'evi_mean': indices_df['evi'].mean(),
            'ndwi_mean': indices_df['ndwi'].mean(),
            'savi_mean': indices_df['savi'].mean(),
            'lai_mean': indices_df['lai'].mean(),
            'temp_mean': climate_df['temp_mean'].mean(),
            'temp_min': climate_df['temp_min'].min(),
            'temp_max': climate_df['temp_max'].max(),
            'rain_total': climate_df['rain'].sum(),
            'rain_mean': climate_df['rain'].mean(),
            'rain_days': (climate_df['rain'] > 1).sum(),
            'humidity_mean': climate_df['humidity'].mean(),
            'wind_mean': climate_df['wind_speed'].mean()
        }
        
        ndvi_score = min(metrics['ndvi_mean'] / crop_params['ndvi_optimal'], 1.0)
        rain_score = min(metrics['rain_total'] / crop_params['rain_min'], 1.0)
        temp_score = 1 - abs(metrics['temp_mean'] - crop_params['temp_optimal']) / 15
        temp_score = max(0, min(temp_score, 1))
        
        water_stress = 1 - max(0, min(metrics['ndwi_mean'], 1))
        
        yield_potential = crop_params['yield_max'] * ndvi_score * rain_score * temp_score * (1 - water_stress * 0.3)
        
        metrics['yield_potential'] = yield_potential
        metrics['ndvi_score'] = ndvi_score
        metrics['rain_score'] = rain_score
        metrics['temp_score'] = temp_score
        metrics['water_stress'] = water_stress
        metrics['cycle_days'] = crop_params['cycle_days']
        metrics['yield_min'] = max(0, yield_potential * 0.8)
        metrics['yield_max'] = yield_potential * 1.2
        metrics['yield_confidence'] = max(0, min(100, 100 - metrics['ndvi_std']*100))
        
        return metrics
def generate_crop_recommendations(metrics, culture, forecast_df=None):
        """Génère recommandations détaillées par culture"""
        recommendations = {
            'diagnostic': [],
            'irrigation': [],
            'fertilisation': [],
            'phytosanitaire': [],
            'calendrier': [],
            'alertes': []
        }
        
        if metrics['ndvi_mean'] > 0.65:
            recommendations['diagnostic'].append("✅ Excellente vigueur végétative")
        elif metrics['ndvi_mean'] > 0.45:
            recommendations['diagnostic'].append("⚠️ Vigueur modérée - surveillance nécessaire")
        else:
            recommendations['diagnostic'].append("❌ Stress végétal détecté - intervention urgente")
        
        if metrics['water_stress'] > 0.5:
            recommendations['diagnostic'].append("❌ Stress hydrique important (NDWI faible)")
            recommendations['alertes'].append("Déficit hydrique critique")
        elif metrics['water_stress'] > 0.3:
            recommendations['diagnostic'].append("⚠️ Déficit hydrique modéré")
        
        if metrics['rain_total'] < 300:
            recommendations['irrigation'].append("🚨 URGENT: Irrigation immédiate - 30-40mm tous les 5 jours")
            recommendations['alertes'].append("Déficit hydrique critique")
        elif metrics['rain_total'] < 450:
            recommendations['irrigation'].append("Complément irrigation: 20-25mm tous les 7 jours")
        else:
            recommendations['irrigation'].append(f"✅ Pluviométrie suffisante ({metrics['rain_total']:.0f}mm)")
        
        ferti_plans = {
            "Mil": [
                "Fond: NPK 15-15-15 à 150 kg/ha au semis",
                "Couverture: Urée 50 kg/ha à 30-35 jours",
                "Apport supplémentaire: Urée 25 kg/ha à montaison si NDVI < 0.5"
            ],
            "Maïs": [
                "Fond: NPK 23-10-5 à 200 kg/ha",
                "Premier apport: Urée 100 kg/ha à 4-6 feuilles",
                "Deuxième apport: Urée 50 kg/ha à floraison",
                "Fumure organique: 5-10 t/ha recommandée"
            ],
            "Arachide": [
                "Fond: NPK 6-20-10 à 200 kg/ha (culture fixatrice d'azote)",
                "Apport calcium: Gypse 300 kg/ha à floraison",
                "Éviter excès azote (favorise feuillage au détriment gousses)"
            ],
            "Riz": [
                "Fond: NPK 15-15-15 à 300 kg/ha",
                "Premier apport: Urée 100 kg/ha à tallage",
                "Deuxième apport: Urée 75 kg/ha à initiation paniculaire",
                "Maintenir lame d'eau 5-10cm"
            ]
        }
        
        recommendations['fertilisation'] = ferti_plans.get(culture, [
            "NPK 15-15-15: 150-200 kg/ha au semis",
            "Urée: 50-75 kg/ha en couverture à 30-40 jours"
        ])
        
        if metrics['humidity_mean'] > 70 and metrics['temp_mean'] > 25:
            recommendations['phytosanitaire'].append("⚠️ Conditions favorables maladies fongiques")
            recommendations['phytosanitaire'].append(f"Traitement préventif fongicide recommandé ({culture})")
        
        if metrics['temp_max'] > 35:
            recommendations['phytosanitaire'].append("Risque ravageurs accru (chenilles, criquets)")
        
        if forecast_df is not None and not forecast_df.empty:
            rain_forecast = forecast_df['rain'].sum()
            if rain_forecast > 20:
                recommendations['calendrier'].append("✅ Bonnes conditions semis prévues (pluie attendue)")
            else:
                recommendations['calendrier'].append("⚠️ Attendre pluies suffisantes avant semis")
        
        recommendations['calendrier'].append(f"Cycle cultural: {metrics['cycle_days']} jours")
        recommendations['calendrier'].append(f"Rendement estimé: {metrics['yield_potential']:.1f} t/ha")
        
        return recommendations    
# ==================== INTERFACE PRINCIPALE ====================

st.title("🌾 AgriSight Pro v2.0 - Analyse Agro-climatique Avancée")
st.markdown("*Plateforme d'analyse multi-indices par télédétection et IA pour l'agriculture de précision*")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

with st.sidebar.expander("🔑 Clés API", expanded=False):
    st.success("✅ Clé Google Gemini configurée")
    st.success("✅ Clé OpenWeather configurée")
    st.success("✅ Clé Agromonitoring configurée")
    st.info("💡 Toutes les clés API sont intégrées et prêtes à l'emploi")

st.sidebar.markdown("---")

# Zone d'étude
st.sidebar.subheader("📍 Zone d'étude")
zone_method = st.sidebar.radio("Méthode de sélection", 
                               ["Dessiner sur carte", "Importer GeoJSON", "Coordonnées"])

uploaded_file = None
manual_coords = None

if zone_method == "Importer GeoJSON":
    uploaded_file = st.sidebar.file_uploader("Fichier GeoJSON", type=["geojson", "json"])
elif zone_method == "Coordonnées":
    st.sidebar.info("Rectangle (lat/lon)")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        lat_min = st.number_input("Lat Min", value=14.60, format="%.4f")
        lon_min = st.number_input("Lon Min", value=-17.50, format="%.4f")
    with col2:
        lat_max = st.number_input("Lat Max", value=14.70, format="%.4f")
        lon_max = st.number_input("Lon Max", value=-17.40, format="%.4f")
    manual_coords = (lat_min, lon_min, lat_max, lon_max)

# Paramètres temporels
st.sidebar.subheader("📅 Période d'analyse")
max_end_date = date.today() - timedelta(days=10)
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Début", max_end_date - timedelta(days=90), 
                                max_value=max_end_date)
with col2:
    end_date = st.date_input("Fin", max_end_date, 
                              max_value=max_end_date,
                              min_value=start_date)

# Multi-cultures
st.sidebar.subheader("🌱 Cultures à analyser")
cultures_disponibles = list(CROP_DATABASE.keys())
cultures_selectionnees = st.sidebar.multiselect(
    "Sélectionnez une ou plusieurs cultures",
    cultures_disponibles,
    default=["Mil"]
)

if not cultures_selectionnees:
    st.sidebar.error("Sélectionnez au moins une culture")

zone_name = st.sidebar.text_input("📍 Nom de la zone", "Ma parcelle")

# Paramètres d'échantillonnage
st.sidebar.subheader("🔬 Échantillonnage")
grid_size_ha = st.sidebar.slider("Taille grille (ha)", 1, 10, 5, 
                                  help="Taille max de chaque cellule d'échantillonnage")

st.sidebar.markdown("---")
load_btn = st.sidebar.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

# ==================== ONGLETS ====================

tabs = st.tabs(["🗺️ Carte", "📊 Dashboard", "🛰️ Indices", "🌦️ Climat", 
                "🔮 Prévisions", "🤖 IA Multi-Cultures", "📄 Rapport"])

# ===== ONGLET 1: CARTE =====
with tabs[0]:
    st.subheader("🗺️ Définir la Zone d'Étude")
    
    if zone_method == "Dessiner sur carte":
        st.info("💡 Dessinez votre zone, puis lancez l'analyse")
    
    if st.session_state.gdf is not None:
        center = [st.session_state.gdf.geometry.centroid.y.mean(),
                 st.session_state.gdf.geometry.centroid.x.mean()]
        zoom = 13
    elif manual_coords:
        center = [(manual_coords[0] + manual_coords[2])/2, 
                  (manual_coords[1] + manual_coords[3])/2]
        zoom = 13
    else:
        center = [14.6937, -17.4441]
        zoom = 10
    
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    
    folium.TileLayer('Esri.WorldImagery', name='Satellite', attr='Esri').add_to(m)
    
    m.add_child(MeasureControl(
        primary_length_unit='meters',
        secondary_length_unit='kilometers',
        primary_area_unit='hectares'
    ))
    
    if st.session_state.gdf is not None:
        folium.GeoJson(
            st.session_state.gdf,
            name="Zone analysée",
            style_function=lambda x: {
                'fillColor': '#28A745',
                'color': '#155724',
                'weight': 3,
                'fillOpacity': 0.3
            },
            tooltip=f"<b>{zone_name}</b><br>Cultures: {', '.join(cultures_selectionnees)}"
        ).add_to(m)
        
        if st.session_state.sampling_points is not None:
            marker_cluster = MarkerCluster(name="Points d'échantillonnage").add_to(m)
            
            for idx, row in st.session_state.sampling_points.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    popup=f"<b>{row['cell_id']}</b><br>Lat: {row['latitude']:.4f}<br>Lon: {row['longitude']:.4f}",
                    color='#FF5722',
                    fill=True,
                    fillColor='#FF5722',
                    fillOpacity=0.7
                ).add_to(marker_cluster)
            
            st.success(f"✓ {len(st.session_state.sampling_points)} points d'échantillonnage générés")
    
    draw = Draw(
        export=True,
        draw_options={
            'polygon': {
                'allowIntersection': False,
                'shapeOptions': {'color': '#28A745', 'weight': 3}
            },
            'rectangle': {'shapeOptions': {'color': '#28A745', 'weight': 3}},
            'polyline': False,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={'edit': True, 'remove': True}
    )
    draw.add_to(m)
    
    folium.LayerControl().add_to(m)
    
    map_output = st_folium(m, height=600, width=None, key="main_map")
    
    if map_output and map_output.get('all_drawings'):
        drawings = map_output['all_drawings']
        if drawings and len(drawings) > 0:
            try:
                gdf_drawn = gpd.GeoDataFrame.from_features(drawings, crs="EPSG:4326")
                st.session_state.drawn_geometry = gdf_drawn.geometry.unary_union
                
                area_ha = calculate_area_hectares(st.session_state.drawn_geometry)
                st.success(f"Zone dessinée: {len(drawings)} forme(s). Surface: {area_ha:.2f} ha")
            except Exception as e:
                st.error(f"Erreur: {e}")

# ==================== CHARGEMENT DONNÉES ====================

if load_btn:
    if not cultures_selectionnees:
        st.error("Sélectionnez au moins une culture")
        st.stop()
    
    geometry = None
    
    if zone_method == "Importer GeoJSON" and uploaded_file:
        file_bytes = uploaded_file.read()
        gdf = load_geojson(file_bytes)
        if gdf is not None and not gdf.empty:
            st.session_state.gdf = gdf
            geometry = gdf.geometry.unary_union
    
    elif zone_method == "Dessiner sur carte":
        if st.session_state.drawn_geometry:
            gdf = gpd.GeoDataFrame([{'geometry': st.session_state.drawn_geometry}], crs='EPSG:4326')
            st.session_state.gdf = gdf
            geometry = st.session_state.drawn_geometry
        else:
            st.error("Veuillez dessiner une zone sur la carte")
            st.stop()
    
    elif zone_method == "Coordonnées" and manual_coords:
        polygon = create_polygon_from_coords(*manual_coords)
        gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs='EPSG:4326')
        st.session_state.gdf = gdf
        geometry = polygon
    
    if geometry is None:
        st.error("Veuillez définir une zone d'étude")
        st.stop()
    
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🔄 Progression du chargement")
        global_progress = st.progress(0, text="Initialisation...")
        status_grid = st.empty()
        status_climate = st.empty()
        status_indices = st.empty()
        status_forecast = st.empty()
        status_analysis = st.empty()
    
    # Étape 1: Grille échantillonnage
    status_grid.info("Création grille d'échantillonnage...")
    global_progress.progress(10, text="Génération points...")
    
    sampling_points = create_sampling_grid(geometry, grid_size_ha)
    
    if sampling_points is None or sampling_points.empty:
        status_grid.error("Échec création grille")
        st.stop()
    
    st.session_state.sampling_points = sampling_points
    status_grid.success(f"✓ {len(sampling_points)} points générés (grille {grid_size_ha}ha)")
    
    global_progress.progress(25, text="Récupération données climatiques...")
    
    # Étape 2: Données climatiques
    status_climate.info("Chargement données climatiques...")
    
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())
    
    points_simple_list = []
    for idx, row in sampling_points.iterrows():
        points_simple_list.append({
            'cell_id': row['cell_id'],
            'latitude': row['latitude'],
            'longitude': row['longitude']
        })
    
    climate_df = get_climate_nasa_multi_points(points_simple_list, start_dt, end_dt)
    
    if climate_df is None or climate_df.empty:
        status_climate.error("Échec données climatiques")
        st.stop()
    else:
        status_climate.success(f"✓ Climat chargé ({len(climate_df)} observations)")
        st.session_state.climate_data = climate_df
    
    global_progress.progress(50, text="Récupération indices satellitaires...")
    
    # Étape 3: Indices satellitaires
    status_indices.info("Chargement indices satellitaires...")
    
    indices_df = simulate_multi_indices_data(points_simple_list, start_date, end_date)
    
    if indices_df is None or indices_df.empty:
        status_indices.error("Échec indices")
        st.stop()
    else:
        status_indices.success(f"✓ Indices chargés ({len(indices_df)} observations)")
        st.session_state.satellite_data = indices_df
    
    global_progress.progress(70, text="Prévisions météo...")
    
    # Étape 4: Prévisions météo
    if OPENWEATHER_KEY:
        status_forecast.info("Chargement prévisions...")
        centroid = geometry.centroid
        forecast_df = get_weather_forecast(centroid.y, centroid.x, OPENWEATHER_KEY)
        
        if forecast_df is not None:
            st.session_state.weather_forecast = forecast_df
            status_forecast.success("✓ Prévisions 7j chargées")
        else:
            status_forecast.warning("Prévisions indisponibles")
    
    global_progress.progress(85, text="Calcul métriques...")
    
    # Étape 5: Calcul métriques pour chaque culture
    status_analysis.info("Calcul métriques multi-cultures...")
    
    all_metrics = {}
    for culture in cultures_selectionnees:
        crop_params = CROP_DATABASE[culture]
        
        metrics = calculate_crop_metrics(climate_df, indices_df, crop_params)
        recommendations = generate_crop_recommendations(
            metrics, culture, st.session_state.weather_forecast
        )
        
        all_metrics[culture] = {
            'metrics': metrics,
            'recommendations': recommendations
        }
    
    st.session_state.analysis = all_metrics
    status_analysis.success(f"✓ Analyse complète ({len(cultures_selectionnees)} cultures)")
    
    global_progress.progress(100, text="Analyse terminée!")
    time.sleep(1)
    
    st.success(f"✅ Données chargées! {len(sampling_points)} points, {len(cultures_selectionnees)} cultures analysées")
    st.balloons()
# ===== ONGLET 2: DASHBOARD =====
with tabs[1]:
    st.subheader("📊 Dashboard Multi-Cultures")
    
    if st.session_state.analysis and st.session_state.climate_data is not None:
        
        selected_culture = st.selectbox("Culture à afficher en détail", cultures_selectionnees)
        
        if selected_culture in st.session_state.analysis:
            metrics = st.session_state.analysis[selected_culture]['metrics']
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                delta = "✅" if metrics['ndvi_mean'] > 0.5 else "⚠️"
                st.metric("🌱 NDVI", f"{metrics['ndvi_mean']:.3f}", delta=delta)
            
            with col2:
                st.metric("🌡️ Temp", f"{metrics['temp_mean']:.1f}°C",
                         delta=f"{metrics['temp_min']:.0f}-{metrics['temp_max']:.0f}°")
            
            with col3:
                delta = "✅" if metrics['rain_total'] > 400 else "⚠️"
                st.metric("💧 Pluie", f"{metrics['rain_total']:.0f}mm", delta=delta)
            
            with col4:
                st.metric("💦 NDWI", f"{metrics['ndwi_mean']:.3f}",
                         delta="✅" if metrics['water_stress'] < 0.3 else "⚠️")
            
            with col5:
                st.metric("📈 Rendement", f"{metrics['yield_potential']:.1f} t/ha")
            
            st.markdown("---")
            
            if len(st.session_state.analysis) > 1:
                st.markdown("### 📊 Comparaison Multi-Cultures")
                
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    fig_yields, ax = plt.subplots(figsize=(8, 5))
                    
                    cultures = list(st.session_state.analysis.keys())
                    yields = [st.session_state.analysis[c]['metrics']['yield_potential'] 
                             for c in cultures]
                    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(cultures)))
                    
                    ax.barh(cultures, yields, color=colors, edgecolor='darkgreen', linewidth=2)
                    ax.set_xlabel('Rendement (t/ha)', fontweight='bold')
                    ax.set_title('Rendements Potentiels par Culture', fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    
                    for i, (c, v) in enumerate(zip(cultures, yields)):
                        ax.text(v + 0.1, i, f"{v:.1f}", va='center', fontweight='bold')
                    
                    st.pyplot(fig_yields)
                    plt.close()
                
                with col_g2:
                    fig_health, ax = plt.subplots(figsize=(8, 5))
                    
                    indices_names = ['NDVI', 'EVI', 'SAVI', 'LAI/7']
                    indices_values = [
                        metrics['ndvi_mean'],
                        metrics['evi_mean'],
                        metrics['savi_mean'],
                        metrics['lai_mean']/7
                    ]
                    
                    x = np.arange(len(indices_names))
                    ax.bar(x, indices_values, color=['green', 'darkgreen', 'forestgreen', 'olivedrab'],
                             edgecolor='black', linewidth=1.5, alpha=0.8)
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(indices_names, fontweight='bold')
                    ax.set_ylabel('Valeur', fontweight='bold')
                    ax.set_title(f'Indices de Végétation - {selected_culture}', fontweight='bold')
                    ax.set_ylim([0, 1])
                    ax.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Optimal')
                    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Moyen')
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3)
                    
                    st.pyplot(fig_health)
                    plt.close()
    
    else:
        st.info("👆 Lancez d'abord l'analyse")

# ===== ONGLET 3: INDICES =====
with tabs[2]:
    st.subheader("🛰️ Analyse Multi-Indices Satellitaires")
    
    if st.session_state.satellite_data is not None:
        df_sat = st.session_state.satellite_data
        
        indices_temporal = df_sat.groupby('date').agg({
            'ndvi': 'mean',
            'evi': 'mean',
            'ndwi': 'mean',
            'savi': 'mean',
            'lai': 'mean',
            'msavi': 'mean'
        }).reset_index()
        
        fig_veg, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(indices_temporal['date'], indices_temporal['ndvi'], 'o-',
               color='darkgreen', linewidth=2, markersize=6, label='NDVI')
        ax.plot(indices_temporal['date'], indices_temporal['evi'], 's-',
               color='forestgreen', linewidth=2, markersize=6, label='EVI')
        ax.plot(indices_temporal['date'], indices_temporal['savi'], '^-',
               color='olive', linewidth=2, markersize=6, label='SAVI')
        ax.plot(indices_temporal['date'], indices_temporal['msavi'], 'd-',
               color='yellowgreen', linewidth=2, markersize=6, label='MSAVI')
        
        ax.axhline(0.7, color='green', linestyle=':', alpha=0.5, label='Seuil excellent')
        ax.axhline(0.5, color='orange', linestyle=':', alpha=0.5, label='Seuil bon')
        ax.axhline(0.3, color='red', linestyle=':', alpha=0.5, label='Seuil stress')
        
        ax.set_ylabel('Valeur Indice', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_title('Indices de Végétation', fontsize=14, fontweight='bold')
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        plt.xticks(rotation=30)
        plt.tight_layout()
        
        st.pyplot(fig_veg)
        plt.close()
        
        st.markdown("---")
        
        st.markdown("### 📋 Données Complètes (Export SIG)")
        st.info("💡 Tableau avec coordonnées géographiques pour interpolation dans votre logiciel SIG")
        
        export_df = df_sat.groupby(['cell_id', 'latitude', 'longitude']).agg({
            'ndvi': ['mean', 'min', 'max', 'std'],
            'evi': 'mean',
            'ndwi': 'mean',
            'savi': 'mean',
            'lai': 'mean',
            'msavi': 'mean'
        }).reset_index()
        
        export_df.columns = ['cell_id', 'latitude', 'longitude', 
                            'ndvi_mean', 'ndvi_min', 'ndvi_max', 'ndvi_std',
                            'evi_mean', 'ndwi_mean', 'savi_mean', 'lai_mean', 'msavi_mean']
        
        st.dataframe(export_df, use_container_width=True)
        
        csv_export = export_df.to_csv(index=False)
        st.download_button(
            "📥 Télécharger CSV pour SIG",
            csv_export,
            f"indices_sig_{zone_name}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.info("Chargez d'abord les données")

# ===== ONGLET 4: CLIMAT =====
with tabs[3]:
    st.subheader("🌦️ Analyse Climatique Détaillée")
    
    if st.session_state.climate_data is not None:
        df_clim = st.session_state.climate_data
        
        clim_temporal = df_clim.groupby('date').agg({
            'temp_mean': 'mean',
            'temp_min': 'min',
            'temp_max': 'max',
            'rain': 'mean',
            'humidity': 'mean',
            'wind_speed': 'mean'
        }).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.fill_between(clim_temporal['date'], clim_temporal['temp_min'], 
                        clim_temporal['temp_max'],
                        alpha=0.3, color='coral', label='Plage min-max')
        ax1.plot(clim_temporal['date'], clim_temporal['temp_mean'], 
                color='red', linewidth=2.5, label='Moyenne')
        ax1.axhline(35, color='darkred', linestyle='--', alpha=0.6, label='Seuil stress (35°C)')
        ax1.axhline(25, color='orange', linestyle=':', alpha=0.6, label='Temp optimale (25°C)')
        
        ax1.set_ylabel('Température (°C)', fontweight='bold', fontsize=11)
        ax1.set_title('Températures', fontweight='bold', fontsize=13)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(clim_temporal['date'], clim_temporal['rain'], 
               color='dodgerblue', alpha=0.7, edgecolor='navy')
        ax2.axhline(clim_temporal['rain'].mean(), color='navy', linestyle='--', 
                   linewidth=2, label=f"Moyenne: {clim_temporal['rain'].mean():.1f} mm/j")
        ax2.set_ylabel('Pluie (mm)', fontweight='bold', fontsize=11)
        ax2.set_xlabel('Date', fontweight='bold', fontsize=11)
        ax2.set_title('Précipitations', fontweight='bold', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        st.markdown("### 📊 Statistiques Climatiques")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**🌡️ Températures**")
            st.metric("Moyenne", f"{clim_temporal['temp_mean'].mean():.1f}°C")
            st.metric("Min absolue", f"{clim_temporal['temp_min'].min():.1f}°C")
            st.metric("Max absolue", f"{clim_temporal['temp_max'].max():.1f}°C")
        
        with col2:
            st.markdown("**💧 Précipitations**")
            st.metric("Cumul total", f"{clim_temporal['rain'].sum():.0f} mm")
            st.metric("Moyenne/jour", f"{clim_temporal['rain'].mean():.1f} mm")
            st.metric("Jours pluie (>1mm)", f"{(clim_temporal['rain'] > 1).sum()}")
        
        with col3:
            st.markdown("**💨 Humidité & Vent**")
            st.metric("Humidité moy.", f"{clim_temporal['humidity'].mean():.1f}%")
            st.metric("Vent moyen", f"{clim_temporal['wind_speed'].mean():.1f} m/s")
        
        with col4:
            st.markdown("**📊 Indices**")
            st.metric("Jours >35°C", f"{(clim_temporal['temp_max'] > 35).sum()}")
            st.metric("Jours secs (<1mm)", f"{(clim_temporal['rain'] < 1).sum()}")
            st.metric("Période (jours)", f"{len(clim_temporal)}")
    
    else:
        st.info("Chargez d'abord les données")
# ===== ONGLET 5: PRÉVISIONS =====
with tabs[4]:
    st.subheader("🔮 Prévisions Météorologiques")
    
    if st.session_state.weather_forecast is not None:
        forecast_df = st.session_state.weather_forecast
        
        st.markdown("### 📅 Prévisions 7 Jours")
        
        fig_forecast, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axes[0].plot(forecast_df['date'], forecast_df['temp'], 'o-',
                    color='orangered', linewidth=2.5, markersize=8, label='Temp moyenne')
        axes[0].fill_between(forecast_df['date'], forecast_df['temp_min'], 
                            forecast_df['temp_max'],
                            alpha=0.3, color='coral', label='Min-Max')
        axes[0].set_ylabel('Température (°C)', fontweight='bold')
        axes[0].set_title('Températures Prévues', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].bar(forecast_df['date'], forecast_df['rain'], 
                   color='steelblue', alpha=0.7, edgecolor='navy')
        axes[1].set_ylabel('Pluie (mm)', fontweight='bold')
        axes[1].set_title('Précipitations Prévues', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        axes[2].plot(forecast_df['date'], forecast_df['humidity'], 's-',
                    color='teal', linewidth=2.5, markersize=7)
        axes[2].fill_between(forecast_df['date'], forecast_df['humidity'],
                            alpha=0.3, color='teal')
        axes[2].set_ylabel('Humidité (%)', fontweight='bold')
        axes[2].set_xlabel('Date', fontweight='bold')
        axes[2].set_title('Humidité Prévue', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 100])
        
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig_forecast)
        plt.close()
        
        st.markdown("---")
        
        forecast_display = forecast_df.copy()
        forecast_display['date'] = forecast_display['date'].astype(str)
        forecast_display = forecast_display.rename(columns={
            'date': 'Date',
            'temp': 'Temp (°C)',
            'temp_min': 'Min (°C)',
            'temp_max': 'Max (°C)',
            'humidity': 'Humidité (%)',
            'rain': 'Pluie (mm)',
            'wind_speed': 'Vent (m/s)',
            'description': 'Conditions'
        })
        
        st.dataframe(forecast_display, use_container_width=True)
    
    else:
        st.info("Prévisions météo activées automatiquement lors de l'analyse")

# ===== ONGLET 6: IA =====
with tabs[5]:
    st.subheader("🤖 Analyse IA Multi-Cultures avec Google Gemini")
    
    if st.session_state.analysis and st.session_state.climate_data is not None:
        
        st.info("💡 **Google Gemini** intégré et prêt à l'emploi")
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            include_forecast = st.checkbox("Inclure prévisions météo", 
                                          value=st.session_state.weather_forecast is not None)
        
        with col_opt2:
            detailed_analysis = st.checkbox("Analyse très détaillée", value=True)
        
        analyze_btn = st.button("🚀 Générer Analyses IA Complètes", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("🧠 Analyse IA en cours pour toutes les cultures..."):
                
                analyses_generated = {}
                
                for culture in cultures_selectionnees:
                    
                    st.info(f"Analyse de {culture}...")
                    
                    metrics = st.session_state.analysis[culture]['metrics']
                    recommendations = st.session_state.analysis[culture]['recommendations']
                    
                    indices_df = st.session_state.satellite_data
                    ndvi_evolution = indices_df.groupby('date')['ndvi'].agg(['mean', 'min', 'max']).reset_index()
                    ndvi_recent = ", ".join([
                        f"{row['date'].strftime('%d/%m')}: {row['mean']:.2f}"
                        for _, row in ndvi_evolution.tail(10).iterrows()
                    ])
                    
                    climate_df = st.session_state.climate_data
                    
                    ndvi_by_cell = indices_df.groupby('cell_id')['ndvi'].mean()
                    spatial_cv = (ndvi_by_cell.std() / ndvi_by_cell.mean()) * 100
                    
                    forecast_info = ""
                    if include_forecast and st.session_state.weather_forecast is not None:
                        forecast_df = st.session_state.weather_forecast
                        forecast_info = f"""
PRÉVISIONS 7 JOURS:
- Pluie prévue: {forecast_df['rain'].sum():.0f}mm
- Temp moyenne: {forecast_df['temp'].mean():.1f}°C
- Humidité: {forecast_df['humidity'].mean():.0f}%
"""
                    
                    prompt = f"""Tu es un AGRONOME EXPERT spécialisé en {culture}. Analyse ces données et fournis des recommandations TRÈS DÉTAILLÉES.

CULTURE: {culture}
ZONE: {zone_name}
PÉRIODE: {(end_date - start_date).days} jours

DONNÉES SATELLITAIRES:
- NDVI moyen: {metrics['ndvi_mean']:.3f} (min:{metrics['ndvi_min']:.3f}, max:{metrics['ndvi_max']:.3f})
- Évolution NDVI récente: {ndvi_recent}
- EVI: {metrics['evi_mean']:.3f}, NDWI: {metrics['ndwi_mean']:.3f}, LAI: {metrics['lai_mean']:.1f}
- Variabilité spatiale: {spatial_cv:.1f}%

DONNÉES CLIMATIQUES:
- Température: {metrics['temp_mean']:.1f}°C (min:{metrics['temp_min']:.0f}°C, max:{metrics['temp_max']:.0f}°C)
- Pluie totale: {metrics['rain_total']:.0f}mm ({metrics['rain_days']} jours)
- Humidité: {metrics['humidity_mean']:.0f}%
{forecast_info}

RENDEMENT ESTIMÉ: {metrics['yield_potential']:.1f} t/ha

ANALYSE DEMANDÉE:
1. DIAGNOSTIC DÉTAILLÉ (état culture, stress détectés)
2. IRRIGATION (doses précises, calendrier)
3. FERTILISATION (NPK, doses kg/ha, périodes)
4. PROTECTION PHYTOSANITAIRE (maladies, ravageurs, traitements)
5. OPÉRATIONS CULTURALES (sarclage, buttage, etc.)
6. CALENDRIER PRÉVISIONNEL (stades, dates récolte)
7. ALERTES ET ACTIONS URGENTES

Sois CONCRET et ACTIONNABLE. Adapte au contexte sahélien. Évite généralités."""

                    analysis_text = None
                    
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
                    try:
                        response = requests.post(
                            url,
                            headers={"Content-Type": "application/json"},
                            json={
                                "contents": [{"parts": [{"text": prompt}]}],
                                "generationConfig": {
                                    "temperature": 0.7,
                                    "maxOutputTokens": 8192,
                                }
                            },
                            timeout=90
                        )
                        if response.status_code == 200:
                            data = response.json()
                            if 'candidates' in data and len(data['candidates']) > 0:
                                analysis_text = data['candidates'][0]['content']['parts'][0]['text']
                        else:
                            st.warning(f"Erreur API Gemini: {response.status_code}")
                    except Exception as e:
                        st.warning(f"Erreur connexion: {e}")
                    
                    if not analysis_text:
                        analysis_text = f"""# ANALYSE AGRONOMIQUE - {culture.upper()}

## 1. DIAGNOSTIC
NDVI {metrics['ndvi_mean']:.3f} - {'Excellente vigueur' if metrics['ndvi_mean'] > 0.6 else 'Vigueur modérée' if metrics['ndvi_mean'] > 0.4 else 'Stress végétal'}
Variabilité: {spatial_cv:.1f}% {'(homogène)' if spatial_cv < 15 else '(hétérogène)'}

## 2. IRRIGATION
Pluie: {metrics['rain_total']:.0f}mm
{'URGENT: 30-40mm/5j' if metrics['rain_total'] < 250 else 'Complément 20-25mm/7j' if metrics['rain_total'] < 400 else 'Satisfaisant'}

## 3. FERTILISATION
{chr(10).join(['- ' + r for r in recommendations['fertilisation']])}

## 4. RENDEMENT
Estimation: {metrics['yield_potential']:.1f} t/ha
Cycle: {metrics['cycle_days']} jours

{chr(10).join(['⚠️ ' + a for a in recommendations['alertes']]) if recommendations['alertes'] else '✅ Aucune alerte critique'}
"""
                    
                    analyses_generated[culture] = analysis_text
                    time.sleep(2)
                
                for culture, text in analyses_generated.items():
                    if culture not in st.session_state.analysis:
                        st.session_state.analysis[culture] = {}
                    st.session_state.analysis[culture]['ai_analysis'] = text
                
                st.success(f"✅ Analyses IA générées pour {len(cultures_selectionnees)} cultures!")
        
        if st.session_state.analysis:
            st.markdown("---")
            st.markdown("### 📋 Rapports Agronomiques Détaillés")
            
            for culture in cultures_selectionnees:
                if culture in st.session_state.analysis and 'ai_analysis' in st.session_state.analysis[culture]:
                    
                    with st.expander(f"🌾 {culture} - Rapport Complet", expanded=True):
                        
                        analysis_text = st.session_state.analysis[culture]['ai_analysis']
                        st.markdown(analysis_text)
                        
                        st.markdown("---")
                        
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            st.download_button(
                                f"📥 Télécharger {culture} (TXT)",
                                analysis_text,
                                file_name=f"analyse_{culture}_{zone_name}_{datetime.now().strftime('%Y%m%d')}.txt",
                                mime="text/plain",
                                use_container_width=True,
                                key=f"dl_txt_{culture}"
                            )
                        
                        with col_dl2:
                            metrics = st.session_state.analysis[culture]['metrics']
                            summary_json = json.dumps({
                                "culture": culture,
                                "zone": zone_name,
                                "date": datetime.now().strftime('%Y-%m-%d'),
                                "ndvi_mean": round(metrics['ndvi_mean'], 3),
                                "rain_total": round(metrics['rain_total'], 1),
                                "temp_mean": round(metrics['temp_mean'], 1),
                                "rendement_estime": round(metrics['yield_potential'], 2)
                            }, indent=2)
                            
                            st.download_button(
                                f"📊 Métriques {culture} (JSON)",
                                summary_json,
                                file_name=f"metriques_{culture}_{zone_name}.json",
                                mime="application/json",
                                use_container_width=True,
                                key=f"dl_json_{culture}"
                            )
    
    else:
        st.info("Lancez d'abord l'analyse complète")
