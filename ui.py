"""
AgriSight Pro v4.0 - Composants UI
Fichier: ui_components.py
Wizard étape par étape, composants réutilisables
"""

import streamlit as st
from datetime import date, timedelta
from typing import Optional, List, Dict
from config import (
    SoilType, AgroZone, ProductionLevel, 
    CROP_DATABASE, CUSTOM_CSS
)

# ==================== WIZARD NAVIGATION ====================

def init_wizard_state():
    """Initialise l'état du wizard"""
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1
    
    if 'wizard_data' not in st.session_state:
        st.session_state.wizard_data = {}

def render_wizard_progress(current_step: int, total_steps: int = 5):
    """Affiche barre de progression wizard"""
    progress = current_step / total_steps
    st.progress(progress, text=f"Étape {current_step}/{total_steps}")
    
    # Indicateurs visuels
    cols = st.columns(total_steps)
    for i, col in enumerate(cols, 1):
        with col:
            if i < current_step:
                st.markdown("✅")
            elif i == current_step:
                st.markdown("▶️")
            else:
                st.markdown("⏸️")

# ==================== ÉTAPES WIZARD ====================

def wizard_step_1_zone():
    """Étape 1: Définir zone d'étude"""
    st.markdown("## 📍 Étape 1: Définir la Zone d'Étude")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        zone_name = st.text_input(
            "Nom de la zone *",
            value=st.session_state.wizard_data.get('zone_name', 'Ma parcelle'),
            key="wiz_zone_name",
            help="Ex: Parcelle Nord, Champ Mil 2025"
        )
        
        zone_method = st.radio(
            "Méthode de sélection *",
            ["🗺️ Dessiner sur carte", "📤 Importer GeoJSON", "📐 Coordonnées manuelles"],
            key="wiz_zone_method",
            help="Choisissez comment définir votre zone"
        )
        
        uploaded_file = None
        manual_coords = None
        
        if "Dessiner" in zone_method:
            st.info("💡 **Instructions:** Utilisez les outils de dessin sur la carte principale (onglet Carte) pour délimiter votre zone, puis revenez ici.")
            
        elif "Importer" in zone_method:
            uploaded_file = st.file_uploader(
                "Sélectionnez votre fichier GeoJSON",
                type=["geojson", "json"],
                help="Format: GeoJSON avec géométrie Polygon"
            )
            
            if uploaded_file:
                st.success(f"✅ Fichier chargé: {uploaded_file.name}")
                st.session_state.wizard_data['geojson_file'] = uploaded_file
                
        elif "Coordonnées" in zone_method:
            st.markdown("**Définir un rectangle (latitude/longitude)**")
            
            col_a, col_b = st.columns(2)
            with col_a:
                lat_min = st.number_input(
                    "Latitude Min", 
                    value=st.session_state.wizard_data.get('lat_min', 14.60),
                    format="%.4f",
                    key="wiz_lat_min"
                )
                lon_min = st.number_input(
                    "Longitude Min", 
                    value=st.session_state.wizard_data.get('lon_min', -17.50),
                    format="%.4f",
                    key="wiz_lon_min"
                )
            with col_b:
                lat_max = st.number_input(
                    "Latitude Max", 
                    value=st.session_state.wizard_data.get('lat_max', 14.70),
                    format="%.4f",
                    key="wiz_lat_max"
                )
                lon_max = st.number_input(
                    "Longitude Max", 
                    value=st.session_state.wizard_data.get('lon_max', -17.40),
                    format="%.4f",
                    key="wiz_lon_max"
                )
            
            manual_coords = (lat_min, lon_min, lat_max, lon_max)
            st.session_state.wizard_data['manual_coords'] = manual_coords
            
            # Estimation surface
            approx_area = abs(lat_max - lat_min) * abs(lon_max - lon_min) * 111 * 111 / 10000
            st.info(f"📏 Surface approximative: {approx_area:.1f} ha")
    
    with col2:
        st.markdown("### ℹ️ Conseils")
        st.markdown("""
        **Recommandations:**
        - Surface optimale: 1-100 ha
        - Évitez zones hétérogènes
        - Privilégiez parcelles homogènes
        
        **Qualité données:**
        - Plus petite zone = meilleure précision
        - Évitez obstacles (bâtiments, routes)
        """)
    
    # Sauvegarde
    st.session_state.wizard_data['zone_name'] = zone_name
    st.session_state.wizard_data['zone_method'] = zone_method
    
    # Navigation
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav3:
        if st.button("Suivant ➡️", type="primary", use_container_width=True):
            if not zone_name:
                st.error("❌ Veuillez saisir un nom de zone")
            else:
                st.session_state.wizard_step = 2
                st.rerun()

def wizard_step_2_context():
    """Étape 2: Contexte agronomique"""
    st.markdown("## 🌱 Étape 2: Contexte Agronomique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Caractéristiques Sol et Zone")
        
        soil_type = st.selectbox(
            "Type de sol dominant *",
            [s.value for s in SoilType],
            index=[s.value for s in SoilType].index(
                st.session_state.wizard_data.get('soil_type', SoilType.ARGILO_SABLEUX.value)
            ),
            key="wiz_soil",
            help="Type de sol majoritaire sur la parcelle"
        )
        
        agro_zone = st.selectbox(
            "Zone agro-écologique *",
            [z.value for z in AgroZone],
            index=[z.value for z in AgroZone].index(
                st.session_state.wizard_data.get('agro_zone', AgroZone.SAHELO_SOUDANIEN.value)
            ),
            key="wiz_zone",
            help="Basé sur pluviométrie annuelle moyenne"
        )
        
        st.markdown("### Type d'Exploitation")
        
        prod_level = st.selectbox(
            "Niveau de production *",
            [p.value for p in ProductionLevel],
            index=[p.value for p in ProductionLevel].index(
                st.session_state.wizard_data.get('production_level', ProductionLevel.PETIT_EXPLOITANT.value)
            ),
            key="wiz_prod",
            help="Détermine recommandations adaptées"
        )
    
    with col2:
        st.markdown("### Ressources Disponibles")
        
        has_irrigation = st.checkbox(
            "Irrigation disponible",
            value=st.session_state.wizard_data.get('has_irrigation', False),
            key="wiz_irrig",
            help="Accès à système d'irrigation (gravitaire, aspersion, goutte-à-goutte)"
        )
        
        if has_irrigation:
            irrigation_type = st.selectbox(
                "Type d'irrigation",
                ["Gravitaire", "Aspersion", "Goutte-à-goutte", "Autre"],
                key="wiz_irrig_type"
            )
            st.session_state.wizard_data['irrigation_type'] = irrigation_type
        
        has_inputs = st.checkbox(
            "Accès intrants améliorés",
            value=st.session_state.wizard_data.get('has_inputs', True),
            key="wiz_inputs",
            help="Engrais chimiques, semences améliorées, produits phyto"
        )
        
        has_mechanization = st.checkbox(
            "Mécanisation disponible",
            value=st.session_state.wizard_data.get('has_mechanization', False),
            key="wiz_mech",
            help="Tracteur, motoculteur, batteuse"
        )
        
        st.markdown("### 💡 Impact Contexte")
        st.info(f"""
        **Sol {soil_type}:**
        Certaines cultures seront plus/moins adaptées.
        
        **Zone {agro_zone}:**
        Recommandations pluviométrie ajustées.
        
        **Niveau {prod_level}:**
        Intensité recommandations adaptée.
        """)
    
    # Sauvegarde
    st.session_state.wizard_data.update({
        'soil_type': soil_type,
        'agro_zone': agro_zone,
        'production_level': prod_level,
        'has_irrigation': has_irrigation,
        'has_inputs': has_inputs,
        'has_mechanization': has_mechanization
    })
    
    # Navigation
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav1:
        if st.button("⬅️ Précédent", use_container_width=True):
            st.session_state.wizard_step = 1
            st.rerun()
    with col_nav3:
        if st.button("Suivant ➡️", type="primary", use_container_width=True):
            st.session_state.wizard_step = 3
            st.rerun()

def wizard_step_3_crops():
    """Étape 3: Cultures et période"""
    st.markdown("## 🌾 Étape 3: Cultures et Période d'Analyse")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cultures à Analyser")
        
        cultures_disponibles = list(CROP_DATABASE.keys())
        
        cultures_selectionnees = st.multiselect(
            "Sélectionnez une ou plusieurs cultures *",
            cultures_disponibles,
            default=st.session_state.wizard_data.get('cultures', ["Mil"]),
            key="wiz_cultures",
            help="Plusieurs cultures = analyse comparative"
        )
        
        if cultures_selectionnees:
            st.success(f"✅ {len(cultures_selectionnees)} culture(s) sélectionnée(s)")
            
            # Compatibilité sol
            st.markdown("#### 🔍 Compatibilité Sol")
            
            soil_type = st.session_state.wizard_data.get('soil_type', 'Argilo-sableux')
            
            for culture in cultures_selectionnees:
                crop_params = CROP_DATABASE[culture]
                
                # Vérifier compatibilité
                soil_match = any(
                    s.value == soil_type 
                    for s in crop_params.soil_preferences
                )
                
                if soil_match:
                    st.markdown(f"✅ **{culture}:** Sol optimal")
                else:
                    preferred = ", ".join([s.value for s in crop_params.soil_preferences])
                    st.markdown(f"⚠️ **{culture}:** Sol sous-optimal (préfère: {preferred})")
        else:
            st.warning("⚠️ Sélectionnez au moins une culture")
    
    with col2:
        st.markdown("### Période d'Analyse")
        
        max_end = date.today() - timedelta(days=10)
        
        start_date = st.date_input(
            "Date de début *",
            value=st.session_state.wizard_data.get('start_date', max_end - timedelta(days=90)),
            max_value=max_end,
            key="wiz_start",
            help="Début période analyse (max aujourd'hui - 10j)"
        )
        
        end_date = st.date_input(
            "Date de fin *",
            value=st.session_state.wizard_data.get('end_date', max_end),
            max_value=max_end,
            min_value=start_date,
            key="wiz_end",
            help="Fin période analyse"
        )
        
        duration = (end_date - start_date).days
        
        st.metric("Durée de l'analyse", f"{duration} jours")
        
        if duration < 30:
            st.warning("⚠️ Période courte - résultats limités")
        elif duration > 180:
            st.info("ℹ️ Longue période - peut couvrir plusieurs cycles")
        
        st.markdown("#### 💡 Recommandations Période")
        st.info("""
        **Optimale:** 60-120 jours
        - Minimum: 30 jours
        - Couvre cycle cultural complet
        - Évite saison sèche seule
        """)
    
    # Sauvegarde
    st.session_state.wizard_data.update({
        'cultures': cultures_selectionnees,
        'start_date': start_date,
        'end_date': end_date
    })
    
    # Navigation
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav1:
        if st.button("⬅️ Précédent", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()
    with col_nav3:
        can_continue = len(cultures_selectionnees) > 0
        if st.button("Suivant ➡️", type="primary", use_container_width=True, disabled=not can_continue):
            if can_continue:
                st.session_state.wizard_step = 4
                st.rerun()

def wizard_step_4_advanced():
    """Étape 4: Paramètres avancés"""
    st.markdown("## ⚙️ Étape 4: Paramètres Avancés")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Échantillonnage Spatial")
        
        grid_size = st.slider(
            "Taille grille (hectares)",
            min_value=1,
            max_value=20,
            value=st.session_state.wizard_data.get('grid_size', 5),
            help="Taille cellules échantillonnage. Plus petit = plus précis mais plus lent",
            key="wiz_grid"
        )
        
        st.info(f"Grille {grid_size} ha = cellule ~{int(np.sqrt(grid_size*10000))}m × {int(np.sqrt(grid_size*10000))}m")
        
        st.markdown("### Seuils Qualité")
        
        max_cloud = st.slider(
            "Couverture nuageuse max (%)",
            0, 50, 
            st.session_state.wizard_data.get('max_cloud', 30),
            help="Images avec plus de nuages seront exclues",
            key="wiz_cloud"
        )
    
    with col2:
        st.markdown("### Options Analyse")
        
        use_sentinel = st.checkbox(
            "🛰️ Utiliser données Sentinel-2 réelles",
            value=st.session_state.wizard_data.get('use_sentinel', False),
            help="Nécessite clés API Sentinel Hub (gratuit 3 req/mois)",
            key="wiz_sentinel"
        )
        
        if use_sentinel:
            st.markdown("**Identifiants Sentinel Hub:**")
            sentinel_id = st.text_input(
                "Client ID",
                type="password",
                key="wiz_sentinel_id",
                help="Depuis https://apps.sentinel-hub.com"
            )
            sentinel_secret = st.text_input(
                "Client Secret",
                type="password",
                key="wiz_sentinel_secret"
            )
            
            st.session_state.wizard_data['sentinel_creds'] = (sentinel_id, sentinel_secret)
        
        use_forecast = st.checkbox(
            "🔮 Inclure prévisions météo 7j",
            value=st.session_state.wizard_data.get('use_forecast', True),
            help="Prévisions OpenWeather",
            key="wiz_forecast"
        )
        
        detailed_report = st.checkbox(
            "📄 Générer rapport PDF détaillé",
            value=st.session_state.wizard_data.get('detailed_report', True),
            key="wiz_pdf"
        )
        
        save_to_db = st.checkbox(
            "💾 Sauvegarder analyse en base",
            value=st.session_state.wizard_data.get('save_to_db', True),
            help="Permet suivi historique",
            key="wiz_save"
        )
    
    # Sauvegarde
    st.session_state.wizard_data.update({
        'grid_size': grid_size,
        'max_cloud': max_cloud,
        'use_sentinel': use_sentinel,
        'use_forecast': use_forecast,
        'detailed_report': detailed_report,
        'save_to_db': save_to_db
    })
    
    # Navigation
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
    with col_nav1:
        if st.button("⬅️ Précédent", use_container_width=True):
            st.session_state.wizard_step = 3
            st.rerun()
    with col_nav3:
        if st.button("Suivant ➡️", type="primary", use_container_width=True):
            st.session_state.wizard_step = 5
            st.rerun()

def wizard_step_5_summary():
    """Étape 5: Récapitulatif"""
    st.markdown("## ✅ Étape 5: Récapitulatif et Lancement")
    
    st.markdown("### 📋 Configuration Complète")
    
    data = st.session_state.wizard_data
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🗺️ Zone**")
        st.write(f"📍 {data.get('zone_name', 'N/A')}")
        st.write(f"📐 {data.get('zone_method', 'N/A')}")
        
        st.markdown("**🌾 Cultures**")
        for cult in data.get('cultures', []):
            st.write(f"• {cult}")
    
    with col2:
        st.markdown("**🌱 Contexte**")
        st.write(f"Sol: {data.get('soil_type', 'N/A')}")
        st.write(f"Zone: {data.get('agro_zone', 'N/A')}")
        st.write(f"Niveau: {data.get('production_level', 'N/A')}")
        
        st.markdown("**🔧 Ressources**")
        st.write(f"💧 Irrigation: {'Oui' if data.get('has_irrigation') else 'Non'}")
        st.write(f"🌱 Intrants: {'Oui' if data.get('has_inputs') else 'Non'}")
    
    with col3:
        st.markdown("**📅 Période**")
        start = data.get('start_date')
        end = data.get('end_date')
        if start and end:
            st.write(f"Du: {start.strftime('%d/%m/%Y')}")
            st.write(f"Au: {end.strftime('%d/%m/%Y')}")
            st.write(f"Durée: {(end - start).days} jours")
        
        st.markdown("**⚙️ Paramètres**")
        st.write(f"Grille: {data.get('grid_size', 5)} ha")
        st.write(f"Sentinel: {'Oui' if data.get('use_sentinel') else 'Non'}")
        st.write(f"Prévisions: {'Oui' if data.get('use_forecast') else 'Non'}")
    
    st.markdown("---")
    
    # Estimation temps
    n_cultures = len(data.get('cultures', []))
    duration_days = (data.get('end_date', date.today()) - data.get('start_date', date.today())).days
    estimated_time = 2 + n_cultures * 0.5 + duration_days * 0.01
    
    st.info(f"⏱️ Temps estimé: {estimated_time:.0f}-{estimated_time*1.5:.0f} minutes")
    
    # Navigation
    col_nav1, col_nav2 = st.columns([1, 2])
    with col_nav1:
        if st.button("⬅️ Modifier", use_container_width=True):
            st.session_state.wizard_step = 1
            st.rerun()
    with col_nav2:
        if st.button("🚀 LANCER L'ANALYSE COMPLÈTE", type="primary", use_container_width=True):
            st.session_state.wizard_completed = True
            st.session_state.launch_analysis = True
            st.rerun()

# ==================== COMPOSANTS RÉUTILISABLES ====================

def render_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Carte métrique stylisée"""
    delta_html = ""
    if delta:
        color = "green" if delta_color == "normal" else "red"
        delta_html = f'<div style="color: {color}; font-size: 0.9em;">{delta}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #666; font-size: 0.85em; margin-bottom: 5px;">{title}</div>
        <div style="font-size: 1.8em; font-weight: bold; margin-bottom: 5px;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def render_data_quality_badge(data_source: str):
    """Badge qualité données"""
    if data_source == "real":
        badge_class = "quality-real"
        text = "✅ Données Réelles"
    else:
        badge_class = "quality-simulated"
        text = "⚠️ Données Simulées"
    
    st.markdown(f"""
    <span class="data-quality-badge {badge_class}">{text}</span>
    """, unsafe_allow_html=True)

def render_alert_box(message: str, alert_type: str = "info"):
    """Boîte d'alerte stylisée"""
    box_class = f"{alert_type}-box"
    st.markdown(f"""
    <div class="{box_class}">
        {message}
    </div>
    """, unsafe_allow_html=True)

# Import numpy pour calculs
import numpy as np
