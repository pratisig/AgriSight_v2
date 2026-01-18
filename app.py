"""
AgriSight Pro v4.0 - Application Principale
Fichier: app.py

INSTALLATION:
pip install streamlit geopandas pandas numpy requests folium streamlit-folium matplotlib seaborn shapely

LANCEMENT:
streamlit run app.py
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import Draw, MeasureControl, MarkerCluster
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import time
import warnings
import logging
from io import BytesIO
import sys
import os

warnings.filterwarnings('ignore')

# Ajouter le répertoire courant au path pour imports locaux
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports modules locaux avec gestion d'erreurs
try:
    from config import (
        CUSTOM_CSS, CROP_DATABASE, OPENWEATHER_KEY, 
        GEMINI_API_KEY, SoilType, AgroZone, ProductionLevel
    )
except ImportError as e:
    st.error(f"❌ Erreur import config.py: {e}")
    st.info("Vérifiez que config.py est dans le même dossier que app.py")
    st.stop()

try:
    from database import DatabaseManager
except ImportError as e:
    st.error(f"❌ Erreur import database.py: {e}")
    st.stop()

try:
    from api_clients import NASAPowerClient, SentinelHubClient, OpenWeatherClient, DataValidator
except ImportError as e:
    st.error(f"❌ Erreur import api_clients.py: {e}")
    st.info("Vérifiez que api_clients.py est présent et que config.py est valide")
    st.stop()

try:
    from analytics import (
        generate_zone_id, calculate_area_hectares, create_sampling_grid,
        simulate_indices_fallback, calculate_advanced_metrics, generate_recommendations,
        compare_crops_performance
    )
except ImportError as e:
    st.error(f"❌ Erreur import analytics.py: {e}")
    st.stop()

try:
    from ui_components import (
        init_wizard_state, render_wizard_progress,
        wizard_step_1_zone, wizard_step_2_context, wizard_step_3_crops,
        wizard_step_4_advanced, wizard_step_5_summary,
        render_metric_card, render_data_quality_badge, render_alert_box
    )
except ImportError as e:
    st.error(f"❌ Erreur import ui_components.py: {e}")
    st.stop()

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION STREAMLIT ====================

st.set_page_config(
    page_title="AgriSight Pro v4.0",
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialiser DB
@st.cache_resource
def get_database():
    return DatabaseManager()

db = get_database()

# ==================== HEADER ====================

st.title("🌾 AgriSight Pro v4.0")
st.markdown("*Plateforme d'Analyse Agro-climatique par Télédétection et Intelligence Artificielle*")

# Afficher stats DB
db_stats = db.get_database_stats()
with st.expander("📊 Statistiques Base de Données", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Zones", db_stats['zones'])
    col2.metric("Analyses", db_stats['analyses'])
    col3.metric("Recommandations", db_stats['recommendations'])
    col4.metric("Observations", db_stats['field_observations'])

st.markdown("---")

# ==================== MODE SÉLECTION ====================

mode = st.sidebar.radio(
    "🎯 Mode d'utilisation",
    ["🧙‍♂️ Assistant Guidé (Recommandé)", "⚙️ Configuration Avancée"],
    help="Mode Assistant: guidage étape par étape pour débutants. Mode Avancé: contrôle total pour utilisateurs expérimentés"
)

# ==================== INITIALISATION SESSION STATE ====================

for key in ['gdf', 'sampling_points', 'satellite_data', 'climate_data', 
            'weather_forecast', 'analysis', 'drawn_geometry', 'zone_id']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'analysis' else {}

# ==================== MODE WIZARD ====================

if mode == "🧙‍♂️ Assistant Guidé (Recommandé)":
    
    init_wizard_state()
    
    # Afficher progression
    render_wizard_progress(st.session_state.wizard_step)
    
    st.markdown("---")
    
    # Rendu étape courante
    if st.session_state.wizard_step == 1:
        wizard_step_1_zone()
        
    elif st.session_state.wizard_step == 2:
        wizard_step_2_context()
        
    elif st.session_state.wizard_step == 3:
        wizard_step_3_crops()
        
    elif st.session_state.wizard_step == 4:
        wizard_step_4_advanced()
        
    elif st.session_state.wizard_step == 5:
        wizard_step_5_summary()
    
    # Si wizard complété, préparer lancement
    if st.session_state.get('wizard_completed', False):
        st.session_state.wizard_completed = False  # Reset
        
        # Transférer données wizard vers session_state principal
        data = st.session_state.wizard_data
        
        # Géométrie
        if data.get('zone_method') == "📐 Coordonnées manuelles" and 'manual_coords' in data:
            coords = data['manual_coords']
            polygon = Polygon([
                (coords[1], coords[0]),  # lon_min, lat_min
                (coords[3], coords[0]),  # lon_max, lat_min
                (coords[3], coords[2]),  # lon_max, lat_max
                (coords[1], coords[2]),  # lon_min, lat_max
                (coords[1], coords[0])   # close
            ])
            st.session_state.gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs='EPSG:4326')
        
        # Lancer analyse
        st.session_state.launch_analysis = True

# ==================== MODE AVANCÉ ====================

else:  # Mode Avancé
    
    st.sidebar.markdown("### ⚙️ Configuration Manuelle")
    st.sidebar.markdown("---")
    
    # Zone
    st.sidebar.subheader("📍 Zone d'étude")
    zone_method = st.sidebar.radio("Méthode", ["Dessiner", "Importer GeoJSON", "Coordonnées"])
    
    zone_name = st.sidebar.text_input("📍 Nom zone", "Ma parcelle")
    
    uploaded_file = None
    manual_coords = None
    
    if zone_method == "Importer GeoJSON":
        uploaded_file = st.sidebar.file_uploader("Fichier GeoJSON", type=["geojson", "json"])
        
    elif zone_method == "Coordonnées":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            lat_min = st.number_input("Lat Min", value=14.60, format="%.4f")
            lon_min = st.number_input("Lon Min", value=-17.50, format="%.4f")
        with col2:
            lat_max = st.number_input("Lat Max", value=14.70, format="%.4f")
            lon_max = st.number_input("Lon Max", value=-17.40, format="%.4f")
        manual_coords = (lat_min, lon_min, lat_max, lon_max)
    
    # Contexte
    st.sidebar.subheader("🌱 Contexte")
    soil_type = st.sidebar.selectbox("Type sol", [s.value for s in SoilType])
    agro_zone = st.sidebar.selectbox("Zone agro", [z.value for z in AgroZone])
    prod_level = st.sidebar.selectbox("Niveau prod", [p.value for p in ProductionLevel])
    
    # Période
    st.sidebar.subheader("📅 Période")
    max_end_date = date.today() - timedelta(days=10)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Début", max_end_date - timedelta(days=90), max_value=max_end_date)
    with col2:
        end_date = st.date_input("Fin", max_end_date, max_value=max_end_date, min_value=start_date)
    
    # Cultures
    st.sidebar.subheader("🌾 Cultures")
    cultures_selectionnees = st.sidebar.multiselect(
        "Sélectionnez",
        list(CROP_DATABASE.keys()),
        default=["Mil"]
    )
    
    # Paramètres
    st.sidebar.subheader("⚙️ Paramètres")
    grid_size_ha = st.sidebar.slider("Grille (ha)", 1, 20, 5)
    
    # Lancement
    st.sidebar.markdown("---")
    load_btn = st.sidebar.button("🚀 Lancer Analyse", type="primary", use_container_width=True)
    
    if load_btn:
        st.session_state.launch_analysis = True
        st.session_state.wizard_data = {
            'zone_name': zone_name,
            'zone_method': zone_method,
            'soil_type': soil_type,
            'agro_zone': agro_zone,
            'production_level': prod_level,
            'cultures': cultures_selectionnees,
            'start_date': start_date,
            'end_date': end_date,
            'grid_size': grid_size_ha,
            'manual_coords': manual_coords
        }

# ==================== ONGLETS PRINCIPAUX ====================

tabs = st.tabs(["🗺️ Carte", "📊 Dashboard", "🛰️ Indices", "🌦️ Climat", 
                "🔮 Prévisions", "🤖 IA", "📄 Rapport", "📚 Historique"])

# ===== ONGLET CARTE =====
with tabs[0]:
    st.subheader("🗺️ Carte Interactive")
    
    # Centrage
    if st.session_state.gdf is not None:
        center = [
            st.session_state.gdf.geometry.centroid.y.mean(),
            st.session_state.gdf.geometry.centroid.x.mean()
        ]
        zoom = 13
    else:
        center = [14.6937, -17.4441]  # Dakar par défaut
        zoom = 10
    
    # Carte Folium
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
    folium.TileLayer('Esri.WorldImagery', name='Satellite').add_to(m)
    
    # Zone
    if st.session_state.gdf is not None:
        folium.GeoJson(
            st.session_state.gdf,
            name="Zone d'étude",
            style_function=lambda x: {
                'fillColor': '#28A745',
                'color': '#155724',
                'weight': 3,
                'fillOpacity': 0.3
            }
        ).add_to(m)
        
        # Points échantillonnage
        if st.session_state.sampling_points is not None:
            mc = MarkerCluster(name="Points").add_to(m)
            
            for idx, row in st.session_state.sampling_points.head(50).iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    popup=f"<b>{row['cell_id']}</b>",
                    color='red',
                    fill=True,
                    fillOpacity=0.7
                ).add_to(mc)
    
    # Outils dessin
    draw = Draw(
        export=True,
        draw_options={
            'polygon': {'allowIntersection': False},
            'rectangle': True,
            'polyline': False,
            'circle': False,
            'marker': False
        }
    )
    draw.add_to(m)
    
    folium.LayerControl().add_to(m)
    
    map_output = st_folium(m, height=600, width=None)
    
    # Capturer dessin
    if map_output and map_output.get('all_drawings'):
        try:
            gdf_drawn = gpd.GeoDataFrame.from_features(map_output['all_drawings'], crs="EPSG:4326")
            st.session_state.drawn_geometry = gdf_drawn.geometry.unary_union
            
            area_ha = calculate_area_hectares(st.session_state.drawn_geometry)
            st.success(f"✅ Zone capturée: {area_ha:.2f} ha")
        except Exception as e:
            st.error(f"Erreur capture: {e}")

# ===== ONGLET DASHBOARD =====
with tabs[1]:
    st.subheader("📊 Dashboard Multi-Cultures")
    
    if st.session_state.analysis and st.session_state.climate_data is not None:
        
        # Sélection culture
        selected_culture = st.selectbox(
            "Culture",
            list(st.session_state.analysis.keys())
        )
        
        if selected_culture in st.session_state.analysis:
            metrics = st.session_state.analysis[selected_culture]['metrics']
            
            # Métriques en cards
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "🌱 NDVI Moyen",
                    f"{metrics['ndvi_mean']:.3f}",
                    delta=f"Min: {metrics['ndvi_min']:.2f}"
                )
            
            with col2:
                st.metric(
                    "🌡️ Température",
                    f"{metrics['temp_mean']:.1f}°C",
                    delta=f"{metrics['temp_min']:.0f}-{metrics['temp_max']:.0f}°C"
                )
            
            with col3:
                st.metric(
                    "💧 Pluie Totale",
                    f"{metrics['rain_total']:.0f} mm",
                    delta=f"{metrics['rain_days']} jours"
                )
            
            with col4:
                st.metric(
                    "💦 NDWI",
                    f"{metrics['ndwi_mean']:.3f}",
                    delta=f"Stress: {metrics['water_stress']*100:.0f}%"
                )
            
            with col5:
                st.metric(
                    "📈 Rendement",
                    f"{metrics['yield_potential']:.1f} t/ha",
                    delta=f"±{(metrics['yield_max']-metrics['yield_min'])/2:.1f}"
                )
            
            # Badge qualité données
            st.markdown("---")
            render_data_quality_badge(metrics.get('data_source', 'simulated'))
            
            # Graphique comparatif
            if len(st.session_state.analysis) > 1:
                st.markdown("### 📊 Comparaison Cultures")
                
                comp_df = compare_crops_performance(
                    {k: v['metrics'] for k, v in st.session_state.analysis.items()}
                )
                
                fig, ax = plt.subplots(figsize=(10, 5))
                comp_df.plot(
                    x='Culture',
                    y='Rendement (t/ha)',
                    kind='barh',
                    ax=ax,
                    color='green',
                    alpha=0.7
                )
                ax.set_xlabel('Rendement (t/ha)', fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
    else:
        st.info("💡 Lancez une analyse pour voir le dashboard")

# ===== ONGLET HISTORIQUE =====
with tabs[7]:
    st.subheader("📚 Historique des Analyses")
    
    if st.session_state.zone_id:
        hist_df = db.get_historical_analyses(st.session_state.zone_id, limit=20)
        
        if not hist_df.empty:
            st.dataframe(
                hist_df[['culture', 'start_date', 'end_date', 'ndvi_mean', 
                        'rain_total', 'yield_potential', 'created_at']],
                use_container_width=True
            )
            
            # Graphique évolution
            st.markdown("### 📈 Évolution Rendements")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            for culture in hist_df['culture'].unique():
                data = hist_df[hist_df['culture'] == culture]
                ax.plot(
                    pd.to_datetime(data['created_at']),
                    data['yield_potential'],
                    marker='o',
                    label=culture
                )
            
            ax.set_xlabel('Date', fontweight='bold')
            ax.set_ylabel('Rendement (t/ha)', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Aucune analyse historique pour cette zone")
    else:
        st.info("Sélectionnez une zone pour voir l'historique")

# ==================== PROCESSUS ANALYSE ====================

if st.session_state.get('launch_analysis', False):
    st.session_state.launch_analysis = False  # Reset
    
    data = st.session_state.wizard_data
    
    # Préparer géométrie
    geometry = None
    
    if data.get('zone_method') in ["Coordonnées", "📐 Coordonnées manuelles"] and data.get('manual_coords'):
        coords = data['manual_coords']
        polygon = Polygon([
            (coords[1], coords[0]),
            (coords[3], coords[0]),
            (coords[3], coords[2]),
            (coords[1], coords[2]),
            (coords[1], coords[0])
        ])
        gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs='EPSG:4326')
        st.session_state.gdf = gdf
        geometry = polygon
        
    elif st.session_state.drawn_geometry:
        geometry = st.session_state.drawn_geometry
        st.session_state.gdf = gpd.GeoDataFrame([{'geometry': geometry}], crs='EPSG:4326')
    
    if geometry is None:
        st.error("❌ Aucune zone définie")
        st.stop()
    
    # Validation géométrie
    is_valid, error_msg = DataValidator.validate_geometry(geometry)
    if not is_valid:
        st.error(error_msg)
        st.stop()
    
    # Progression
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🔄 Analyse en Cours")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Grille échantillonnage
        status_text.text("📍 Création grille échantillonnage...")
        progress_bar.progress(10)
        
        grid_size = data.get('grid_size', 5)
        sampling_points = create_sampling_grid(geometry, grid_size)
        st.session_state.sampling_points = sampling_points
        
        status_text.text(f"✅ {len(sampling_points)} points créés")
        time.sleep(0.5)
        
        # 2. Données climatiques
        status_text.text("🌦️ Récupération données NASA POWER...")
        progress_bar.progress(30)
        
        points_list = sampling_points[['latitude', 'longitude', 'cell_id']].to_dict('records')
        
        climate_df = None
        for point in points_list[:min(5, len(points_list))]:  # Limiter à 5 pour démo
            df_point = NASAPowerClient.get_climate_data(
                point['latitude'],
                point['longitude'],
                data['start_date'],
                data['end_date']
            )
            
            if df_point is not None:
                df_point['cell_id'] = point['cell_id']
                df_point['latitude'] = point['latitude']
                df_point['longitude'] = point['longitude']
                
                if climate_df is None:
                    climate_df = df_point
                else:
                    climate_df = pd.concat([climate_df, df_point])
            
            time.sleep(0.5)
        
        if climate_df is None or climate_df.empty:
            st.error("❌ Échec récupération données climatiques")
            st.stop()
        
        st.session_state.climate_data = climate_df
        status_text.text(f"✅ Climat: {len(climate_df)} observations")
        progress_bar.progress(50)
        
        # 3. Indices satellitaires
        status_text.text("🛰️ Récupération indices satellitaires...")
        progress_bar.progress(60)
        
        # Pour démo, utiliser simulation
        indices_df = simulate_indices_fallback(
            sampling_points,
            data['start_date'],
            data['end_date']
        )
        
        st.session_state.satellite_data = indices_df
        status_text.text(f"✅ Indices: {len(indices_df)} observations")
        progress_bar.progress(75)
        
        # 4. Prévisions
        if data.get('use_forecast', True):
            status_text.text("🔮 Récupération prévisions...")
            
            centroid = geometry.centroid
            forecast_df = OpenWeatherClient.get_weather_forecast(
                centroid.y,
                centroid.x,
                OPENWEATHER_KEY
            )
            
            st.session_state.weather_forecast = forecast_df
            status_text.text("✅ Prévisions chargées")
        
        progress_bar.progress(85)
        
        # 5. Analyse multi-cultures
        status_text.text("📊 Calcul métriques...")
        
        all_analysis = {}
        
        for culture in data.get('cultures', ['Mil']):
            crop_params = CROP_DATABASE[culture]
            
            metrics = calculate_advanced_metrics(
                climate_df,
                indices_df,
                crop_params
            )
            
            # Recommandations contextuelles
            contextual_reco = db.get_contextual_recommendation(
                culture,
                data.get('soil_type', 'Argilo-sableux'),
                data.get('agro_zone', 'Sahélo-soudanien (400-600mm)'),
                data.get('production_level', 'Petit exploitant (0.5-2 ha)')
            )
            
            recommendations = generate_recommendations(
                metrics,
                culture,
                contextual_reco,
                data.get('has_irrigation', False),
                st.session_state.weather_forecast
            )
            
            all_analysis[culture] = {
                'metrics': metrics,
                'recommendations': recommendations,
                'contextual_reco': contextual_reco
            }
            
            # Sauvegarde DB
            if data.get('save_to_db', True):
                zone_id = generate_zone_id(geometry, data.get('zone_name', 'zone'))
                st.session_state.zone_id = zone_id
                
                # Sauvegarder zone
                db.save_zone(
                    zone_id,
                    data.get('zone_name', 'Ma zone'),
                    st.session_state.gdf.to_json(),
                    calculate_area_hectares(geometry),
                    data.get('soil_type'),
                    data.get('agro_zone'),
                    data.get('production_level')
                )
                
                # Sauvegarder analyse
                analysis_id = f"{zone_id}_{culture}_{datetime.now().strftime('%Y%m%d%H%M')}"
                
                db.save_analysis({
                    'id': analysis_id,
                    'zone_id': zone_id,
                    'culture': culture,
                    'start_date': data['start_date'].strftime('%Y-%m-%d'),
                    'end_date': data['end_date'].strftime('%Y-%m-%d'),
                    'ndvi_mean': metrics['ndvi_mean'],
                    'ndvi_min': metrics['ndvi_min'],
                    'ndvi_max': metrics['ndvi_max'],
                    'ndvi_std': metrics['ndvi_std'],
                    'evi_mean': metrics['evi_mean'],
                    'ndwi_mean': metrics['ndwi_mean'],
                    'lai_mean': metrics['lai_mean'],
                    'rain_total': metrics['rain_total'],
                    'rain_days': metrics['rain_days'],
                    'temp_mean': metrics['temp_mean'],
                    'temp_min': metrics['temp_min'],
                    'temp_max': metrics['temp_max'],
                    'humidity_mean': metrics['humidity_mean'],
                    'yield_potential': metrics['yield_potential'],
                    'yield_min': metrics['yield_min'],
                    'yield_max': metrics['yield_max'],
                    'yield_confidence': metrics['yield_confidence'],
                    'water_stress': metrics['water_stress'],
                    'heat_stress_days': metrics['heat_stress_days'],
                    'data_source': metrics.get('data_source', 'simulated'),
                    'soil_type': data.get('soil_type'),
                    'agro_zone': data.get('agro_zone'),
                    'production_level': data.get('production_level')
                })
        
        st.session_state.analysis = all_analysis
        
        progress_bar.progress(100)
        status_text.text("✅ Analyse terminée!")
        
        time.sleep(1)
        progress_container.empty()
        
        st.success(f"✅ Analyse complète: {len(all_analysis)} culture(s)")
        st.balloons()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <b>🌾 AgriSight Pro v4.0</b> - Analyse Agricole Avancée<br>
    <small>Télédétection • IA • Agriculture de Précision • Recommandations Contextuelles</small>
</div>
""", unsafe_allow_html=True)
