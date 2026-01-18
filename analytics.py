"""
AgriSight Pro v4.0 - Moteur d'Analyse
Fichier: analytics.py
Calculs métriques, prévisions rendement, recommandations
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import logging
import hashlib
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from config import CROP_DATABASE, CropParameters, YIELD_UNCERTAINTY_BASE

logger = logging.getLogger(__name__)

# ==================== UTILITAIRES GÉOMÉTRIE ====================

def generate_zone_id(geometry, zone_name: str) -> str:
    """Génère ID unique pour une zone"""
    geom_str = str(geometry.bounds) + zone_name + str(datetime.now().timestamp())
    return hashlib.md5(geom_str.encode()).hexdigest()[:16]

def calculate_area_hectares(geometry) -> float:
    """Calcule surface en hectares (conversion degrés -> m²)"""
    # Méthode approximative rapide
    bounds = geometry.bounds
    lat_center = (bounds[1] + bounds[3]) / 2
    
    # Conversion degrés -> mètres à cette latitude
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * np.cos(np.radians(lat_center))
    
    # Surface approximative
    width_m = (bounds[2] - bounds[0]) * m_per_deg_lon
    height_m = (bounds[3] - bounds[1]) * m_per_deg_lat
    area_m2 = width_m * height_m * 0.8  # Facteur correction forme
    
    return area_m2 / 10000

def create_sampling_grid(geometry, grid_size_ha: int = 5) -> gpd.GeoDataFrame:
    """
    Crée grille d'échantillonnage optimisée
    
    Args:
        geometry: Géométrie de la zone (Polygon/MultiPolygon)
        grid_size_ha: Taille cellule en hectares
        
    Returns:
        GeoDataFrame avec points d'échantillonnage
    """
    bounds = geometry.bounds
    min_x, min_y, max_x, max_y = bounds
    
    # Conversion ha -> degrés (ajusté à la latitude)
    lat_center = (min_y + max_y) / 2
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * np.cos(np.radians(lat_center))
    
    grid_size_m = np.sqrt(grid_size_ha * 10000)
    cell_size_lat = grid_size_m / meters_per_degree_lat
    cell_size_lon = grid_size_m / meters_per_degree_lon
    
    # Générer grille
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
                    'cell_id': f"C{len(points)+1:03d}",
                    'grid_x': i,
                    'grid_y': j
                })
    
    # Si zone trop petite, utiliser centroïde
    if not points:
        centroid = geometry.centroid
        points = [{
            'geometry': centroid,
            'longitude': centroid.x,
            'latitude': centroid.y,
            'cell_id': 'C001',
            'grid_x': 0,
            'grid_y': 0
        }]
        logger.warning("⚠️ Zone petite, 1 seul point échantillonnage")
    
    geometries = [p['geometry'] for p in points]
    data = [{k: v for k, v in p.items() if k != 'geometry'} for p in points]
    
    gdf = gpd.GeoDataFrame(data, geometry=geometries, crs='EPSG:4326')
    
    logger.info(f"✅ Grille: {len(gdf)} points ({grid_size_ha} ha/cellule)")
    return gdf

# ==================== SIMULATION INDICES (FALLBACK) ====================

def simulate_indices_fallback(points_gdf: gpd.GeoDataFrame, 
                              start_date: date, 
                              end_date: date) -> pd.DataFrame:
    """
    Simule indices satellitaires si API Sentinel indisponible
    Basé sur climatologie sahélienne
    
    ATTENTION: Données simulées, résultats à valider sur terrain
    """
    logger.warning("⚠️ UTILISATION DONNÉES SIMULÉES (Sentinel Hub non disponible)")
    
    dates = pd.date_range(start_date, end_date, freq='5D')
    all_data = []
    
    for idx, row in points_gdf.iterrows():
        # Variabilité spatiale légère
        spatial_factor = 1 + np.random.normal(0, 0.05)
        
        for d in dates:
            month = d.month
            
            # Modèle saisonnier Sahel/Soudanien
            if 6 <= month <= 9:  # Saison des pluies
                ndvi_base = 0.65 + np.random.normal(0, 0.08)
                ndwi_base = 0.3 + np.random.normal(0, 0.08)
            elif month in [5, 10]:  # Transition
                ndvi_base = 0.45 + np.random.normal(0, 0.1)
                ndwi_base = 0.15 + np.random.normal(0, 0.06)
            else:  # Saison sèche
                ndvi_base = 0.25 + np.random.normal(0, 0.06)
                ndwi_base = 0.05 + np.random.normal(0, 0.04)
            
            # Appliquer variabilité spatiale
            ndvi_base *= spatial_factor
            
            # Autres indices dérivés
            evi_base = ndvi_base * 0.9 + np.random.normal(0, 0.05)
            savi_base = ndvi_base * 0.85 + np.random.normal(0, 0.06)
            lai_base = ndvi_base * 5 + np.random.normal(0, 0.3)
            msavi_base = savi_base * 1.05 + np.random.normal(0, 0.04)
            
            all_data.append({
                'date': d,
                'cell_id': row['cell_id'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'ndvi': np.clip(ndvi_base, 0, 1),
                'evi': np.clip(evi_base, 0, 1),
                'ndwi': np.clip(ndwi_base, -1, 1),
                'savi': np.clip(savi_base, 0, 1),
                'lai': np.clip(lai_base, 0, 7),
                'msavi': np.clip(msavi_base, 0, 1),
                'cloud_cover': np.random.randint(0, 30),
                'data_quality': 'simulated'
            })
    
    df = pd.DataFrame(all_data)
    logger.info(f"✅ Indices simulés: {len(df)} observations")
    
    return df

# ==================== CALCUL MÉTRIQUES AVANCÉES ====================

def calculate_advanced_metrics(climate_df: pd.DataFrame, 
                               indices_df: pd.DataFrame,
                               crop_params: CropParameters) -> Dict:
    """
    Calcule métriques avancées avec intervalles de confiance
    
    Args:
        climate_df: Données climatiques
        indices_df: Indices végétation
        crop_params: Paramètres culture
        
    Returns:
        Dictionnaire métriques complètes
    """
    
    # ===== INDICES VÉGÉTATION =====
    metrics = {
        'ndvi_mean': indices_df['ndvi'].mean(),
        'ndvi_std': indices_df['ndvi'].std(),
        'ndvi_min': indices_df['ndvi'].min(),
        'ndvi_max': indices_df['ndvi'].max(),
        'ndvi_cv': (indices_df['ndvi'].std() / (indices_df['ndvi'].mean() + 0.001)) * 100,
        'ndvi_trend': _calculate_trend(indices_df, 'ndvi'),
        
        'evi_mean': indices_df['evi'].mean(),
        'evi_std': indices_df['evi'].std(),
        
        'ndwi_mean': indices_df['ndwi'].mean(),
        'ndwi_std': indices_df['ndwi'].std(),
        
        'savi_mean': indices_df['savi'].mean(),
        'lai_mean': indices_df['lai'].mean(),
        'lai_max': indices_df['lai'].max(),
        'msavi_mean': indices_df['msavi'].mean(),
    }
    
    # ===== CLIMAT =====
    metrics.update({
        'temp_mean': climate_df['temp_mean'].mean(),
        'temp_std': climate_df['temp_mean'].std(),
        'temp_min': climate_df['temp_min'].min(),
        'temp_max': climate_df['temp_max'].max(),
        
        'rain_total': climate_df['rain'].sum(),
        'rain_mean': climate_df['rain'].mean(),
        'rain_std': climate_df['rain'].std(),
        'rain_days': (climate_df['rain'] > 1).sum(),
        'rain_cv': (climate_df['rain'].std() / (climate_df['rain'].mean() + 0.01)) * 100,
        'rain_intensity': climate_df[climate_df['rain'] > 0]['rain'].mean() if (climate_df['rain'] > 0).any() else 0,
        
        'humidity_mean': climate_df['humidity'].mean(),
        'humidity_std': climate_df['humidity'].std(),
        
        'wind_mean': climate_df['wind_speed'].mean(),
        'wind_max': climate_df['wind_speed'].max(),
    })
    
    # ===== STRESS ENVIRONNEMENTAUX =====
    metrics['heat_stress_days'] = (climate_df['temp_max'] > crop_params.temp_max).sum()
    metrics['cold_stress_days'] = (climate_df['temp_min'] < crop_params.temp_min).sum()
    metrics['water_stress'] = max(0, 1 - (metrics['ndwi_mean'] + 1) / 2)
    
    # Sécheresse (jours consécutifs sans pluie)
    dry_spells = _calculate_dry_spells(climate_df)
    metrics['max_dry_spell'] = max(dry_spells) if dry_spells else 0
    metrics['avg_dry_spell'] = np.mean(dry_spells) if dry_spells else 0
    
    # ===== SCORES NORMALISÉS =====
    metrics['ndvi_score'] = min(metrics['ndvi_mean'] / crop_params.ndvi_optimal, 1.0)
    
    # Score pluie (sigmoïde entre rain_min et rain_max)
    rain_ratio = (metrics['rain_total'] - crop_params.rain_min) / (crop_params.rain_max - crop_params.rain_min)
    metrics['rain_score'] = np.clip(rain_ratio, 0, 1)
    
    # Score température (gaussienne centrée sur optimal)
    temp_diff = abs(metrics['temp_mean'] - crop_params.temp_optimal)
    metrics['temp_score'] = max(0, 1 - (temp_diff / 10))
    
    # ===== RENDEMENT POTENTIEL =====
    # Modèle multiplicatif avec pénalités stress
    base_yield = crop_params.yield_max * \
                 metrics['ndvi_score'] * \
                 metrics['rain_score'] * \
                 metrics['temp_score']
    
    # Pénalités
    water_penalty = base_yield * metrics['water_stress'] * 0.3
    heat_penalty = base_yield * min(metrics['heat_stress_days'] / 10, 0.2)
    cold_penalty = base_yield * min(metrics['cold_stress_days'] / 5, 0.15)
    
    metrics['yield_potential'] = max(0, base_yield - water_penalty - heat_penalty - cold_penalty)
    
    # Intervalle de confiance (basé sur variabilité NDVI)
    uncertainty_factor = YIELD_UNCERTAINTY_BASE * (1 + metrics['ndvi_cv'] / 100)
    uncertainty = metrics['yield_potential'] * uncertainty_factor
    
    metrics['yield_min'] = max(0, metrics['yield_potential'] - uncertainty)
    metrics['yield_max'] = metrics['yield_potential'] + uncertainty
    metrics['yield_confidence'] = max(0, min(100, 100 - metrics['ndvi_cv']))
    
    # ===== MÉTADONNÉES =====
    metrics['cycle_days'] = crop_params.cycle_days
    metrics['data_source'] = indices_df['data_quality'].iloc[0] if 'data_quality' in indices_df else 'real'
    
    return metrics

def _calculate_trend(df: pd.DataFrame, column: str) -> str:
    """Calcule tendance temporelle (hausse/baisse/stable)"""
    if len(df) < 3:
        return "insufficient_data"
    
    # Régression linéaire simple
    x = np.arange(len(df))
    y = df[column].values
    
    # Enlever NaN
    mask = ~np.isnan(y)
    if np.sum(mask) < 3:
        return "insufficient_data"
    
    x = x[mask]
    y = y[mask]
    
    # Pente
    slope = np.polyfit(x, y, 1)[0]
    
    if slope > 0.01:
        return "increasing"
    elif slope < -0.01:
        return "decreasing"
    else:
        return "stable"

def _calculate_dry_spells(climate_df: pd.DataFrame, threshold: float = 1.0) -> List[int]:
    """Calcule longueurs des périodes sèches consécutives"""
    is_dry = (climate_df['rain'] < threshold).astype(int)
    
    dry_spells = []
    current_spell = 0
    
    for dry in is_dry:
        if dry:
            current_spell += 1
        else:
            if current_spell > 0:
                dry_spells.append(current_spell)
                current_spell = 0
    
    if current_spell > 0:
        dry_spells.append(current_spell)
    
    return dry_spells

# ==================== RECOMMANDATIONS ====================

def generate_recommendations(metrics: Dict, 
                            crop_name: str,
                            contextual_reco: Optional[Dict] = None,
                            has_irrigation: bool = False,
                            forecast_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Génère recommandations agronomiques détaillées
    
    Args:
        metrics: Métriques calculées
        crop_name: Nom culture
        contextual_reco: Recommandations contextuelles DB
        has_irrigation: Irrigation disponible
        forecast_df: Prévisions météo
        
    Returns:
        Dictionnaire recommandations par catégorie
    """
    
    recommendations = {
        'diagnostic': [],
        'irrigation': [],
        'fertilisation': [],
        'phytosanitaire': [],
        'operations': [],
        'calendrier': [],
        'alertes': []
    }
    
    # ===== DIAGNOSTIC =====
    if metrics['ndvi_mean'] > 0.65:
        recommendations['diagnostic'].append("✅ Excellente vigueur végétative")
    elif metrics['ndvi_mean'] > 0.45:
        recommendations['diagnostic'].append("⚠️ Vigueur modérée - marge amélioration")
    else:
        recommendations['diagnostic'].append("❌ Stress végétatif détecté")
        recommendations['alertes'].append("Intervention urgente requise")
    
    # Variabilité spatiale
    if metrics['ndvi_cv'] > 20:
        recommendations['diagnostic'].append(
            f"⚠️ Forte hétérogénéité parcellaire (CV={metrics['ndvi_cv']:.1f}%) "
            "→ Envisager gestion différenciée"
        )
    
    # Stress hydrique
    if metrics['water_stress'] > 0.5:
        recommendations['diagnostic'].append("❌ Stress hydrique important (NDWI faible)")
        recommendations['alertes'].append("Déficit hydrique critique")
    elif metrics['water_stress'] > 0.3:
        recommendations['diagnostic'].append("⚠️ Déficit hydrique modéré")
    
    # Stress thermique
    if metrics['heat_stress_days'] > 5:
        recommendations['diagnostic'].append(
            f"⚠️ {metrics['heat_stress_days']} jours de stress thermique détectés"
        )
    
    # ===== IRRIGATION =====
    if contextual_reco and 'irrigation_strategy' in contextual_reco:
        recommendations['irrigation'].append(contextual_reco['irrigation_strategy'])
    else:
        # Recommandations génériques
        if metrics['rain_total'] < 300:
            if has_irrigation:
                recommendations['irrigation'].append("🚨 Irrigation 30-40mm tous les 5 jours")
            else:
                recommendations['irrigation'].append("🚨 Déficit critique - irrigation urgente si possible")
            recommendations['alertes'].append("Pluviométrie très insuffisante")
            
        elif metrics['rain_total'] < 450:
            if has_irrigation:
                recommendations['irrigation'].append("Irrigation complémentaire 20-25mm/semaine")
            else:
                recommendations['irrigation'].append("Surveillance hydrique renforcée")
        else:
            recommendations['irrigation'].append(f"✅ Pluviométrie satisfaisante ({metrics['rain_total']:.0f}mm)")
    
    # Ajustement prévisions
    if forecast_df is not None and not forecast_df.empty:
        rain_forecast = forecast_df['rain'].sum()
        if rain_forecast < 10:
            recommendations['irrigation'].append(
                f"⚠️ Prévision 7j: {rain_forecast:.0f}mm seulement → anticiper irrigation"
            )
    
    # ===== FERTILISATION =====
    if contextual_reco:
        if 'fertilizer_base' in contextual_reco:
            recommendations['fertilisation'].append(f"**Fond:** {contextual_reco['fertilizer_base']}")
        if 'fertilizer_cover' in contextual_reco:
            recommendations['fertilisation'].append(f"**Couverture:** {contextual_reco['fertilizer_cover']}")
    else:
        # Recommandations génériques
        recommendations['fertilisation'].extend([
            f"NPK 15-15-15: 150kg/ha au semis",
            f"Urée 50kg/ha en couverture (stade végétatif)"
        ])
    
    # Ajustement selon NDVI
    if metrics['ndvi_mean'] < 0.4:
        recommendations['fertilisation'].append(
            "⚠️ NDVI faible → envisager apport azoté supplémentaire"
        )
    
    # ===== PHYTOSANITAIRE =====
    if contextual_reco and 'pest_control' in contextual_reco:
        recommendations['phytosanitaire'].append(contextual_reco['pest_control'])
    
    # Conditions favorables maladies
    if metrics['humidity_mean'] > 70 and metrics['temp_mean'] > 25:
        recommendations['phytosanitaire'].append(
            "⚠️ Conditions favorables maladies fongiques (humidité + chaleur)"
        )
        recommendations['phytosanitaire'].append("Surveillance renforcée + traitement préventif")
    
    if metrics['temp_max'] > 35:
        recommendations['phytosanitaire'].append(
            "⚠️ Chaleur élevée → risque ravageurs (chenilles, pucerons)"
        )
    
    # ===== OPÉRATIONS CULTURALES =====
    recommendations['operations'].extend([
        "Sarclage régulier (2-3 passages selon enherbement)",
        "Maintien sol meuble pour infiltration eau",
        "Surveillance continue état culture"
    ])
    
    if metrics['ndvi_cv'] > 20:
        recommendations['operations'].append(
            "Cartographie variabilité pour interventions ciblées"
        )
    
    # ===== CALENDRIER =====
    recommendations['calendrier'].append(f"**Cycle culture:** {metrics['cycle_days']} jours")
    recommendations['calendrier'].append(
        f"**Rendement estimé:** {metrics['yield_potential']:.1f} t/ha "
        f"(IC 95%: {metrics['yield_min']:.1f}-{metrics['yield_max']:.1f})"
    )
    recommendations['calendrier'].append(
        f"**Confiance prévision:** {metrics['yield_confidence']:.0f}%"
    )
    
    # Prévisions
    if forecast_df is not None and not forecast_df.empty:
        days_good = (forecast_df['rain'] > 5).sum()
        if days_good >= 3:
            recommendations['calendrier'].append("✅ Conditions favorables semaine prochaine")
        else:
            recommendations['calendrier'].append("⚠️ Période sèche prévue")
    
    return recommendations

# ==================== ANALYSE COMPARATIVE ====================

def compare_crops_performance(all_metrics: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare performances de plusieurs cultures
    
    Args:
        all_metrics: {culture: metrics_dict}
        
    Returns:
        DataFrame comparatif
    """
    data = []
    
    for culture, metrics in all_metrics.items():
        data.append({
            'Culture': culture,
            'NDVI': metrics['ndvi_mean'],
            'Pluie (mm)': metrics['rain_total'],
            'Temp (°C)': metrics['temp_mean'],
            'Rendement (t/ha)': metrics['yield_potential'],
            'Confiance (%)': metrics['yield_confidence'],
            'Stress Hydrique': metrics['water_stress'],
            'Score Global': (
                metrics['ndvi_score'] * 0.4 +
                metrics['rain_score'] * 0.3 +
                metrics['temp_score'] * 0.3
            ) * 100
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Rendement (t/ha)', ascending=False)
    
    return df
