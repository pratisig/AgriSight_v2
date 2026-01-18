"""
AgriSight Pro v4.0 - Clients API
Fichier: api_clients.py
Gère les connexions aux APIs externes (NASA POWER, Sentinel Hub, OpenWeather)
"""

import requests
import pandas as pd
import numpy as np
import logging
from datetime import date, datetime
from typing import Optional, Dict, List, Tuple
from config import (
    OPENWEATHER_KEY, API_DELAY, 
    TEMP_MIN, TEMP_MAX, RAIN_MIN, RAIN_MAX
)
import time

logger = logging.getLogger(__name__)

# ==================== NASA POWER CLIENT ====================

class NASAPowerClient:
    """Client pour NASA POWER API - Données climatiques réelles"""
    
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    @staticmethod
    def get_climate_data(lat: float, lon: float, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        Récupère données climatiques réelles depuis NASA POWER
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            DataFrame avec données climatiques ou None si erreur
        """
        params = {
            'parameters': 'T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN',
            'community': 'AG',
            'longitude': lon,
            'latitude': lat,
            'start': start_date.strftime('%Y%m%d'),
            'end': end_date.strftime('%Y%m%d'),
            'format': 'JSON'
        }
        
        try:
            logger.info(f"📡 Requête NASA POWER: ({lat:.4f}, {lon:.4f})")
            
            response = requests.get(
                NASAPowerClient.BASE_URL, 
                params=params, 
                timeout=45
            )
            
            if response.status_code == 200:
                data = response.json()
                parameters = data.get('properties', {}).get('parameter', {})
                
                if not parameters:
                    logger.warning("Aucune donnée NASA POWER retournée")
                    return None
                
                dates = list(parameters.get('T2M', {}).keys())
                
                if not dates:
                    logger.warning("Aucune date dans réponse NASA POWER")
                    return None
                
                df = pd.DataFrame({
                    'date': pd.to_datetime(dates),
                    'temp_mean': list(parameters.get('T2M', {}).values()),
                    'temp_min': list(parameters.get('T2M_MIN', {}).values()),
                    'temp_max': list(parameters.get('T2M_MAX', {}).values()),
                    'rain': list(parameters.get('PRECTOTCORR', {}).values()),
                    'humidity': list(parameters.get('RH2M', {}).values()),
                    'wind_speed': list(parameters.get('WS2M', {}).values()),
                    'solar_radiation': list(parameters.get('ALLSKY_SFC_SW_DWN', {}).values())
                })
                
                # Validation et nettoyage
                df = NASAPowerClient._validate_climate_data(df)
                
                logger.info(f"✅ NASA POWER: {len(df)} jours récupérés")
                return df
            
            else:
                logger.error(f"❌ NASA POWER erreur HTTP {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("⏱️ NASA POWER timeout")
            return None
        except Exception as e:
            logger.error(f"❌ Erreur NASA POWER: {e}")
            return None
    
    @staticmethod
    def _validate_climate_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validation et nettoyage des données climatiques
        
        - Supprime valeurs aberrantes
        - Remplace -999 (valeur manquante NASA) par NaN
        - Interpole valeurs manquantes
        """
        # Remplacer -999 par NaN
        df = df.replace(-999, np.nan)
        df = df.replace(-999.0, np.nan)
        
        # Valider plages de valeurs
        df.loc[(df['temp_mean'] < TEMP_MIN) | (df['temp_mean'] > TEMP_MAX), 'temp_mean'] = np.nan
        df.loc[(df['temp_min'] < TEMP_MIN - 10) | (df['temp_min'] > TEMP_MAX), 'temp_min'] = np.nan
        df.loc[(df['temp_max'] < TEMP_MIN) | (df['temp_max'] > TEMP_MAX + 5), 'temp_max'] = np.nan
        df.loc[(df['rain'] < RAIN_MIN) | (df['rain'] > RAIN_MAX), 'rain'] = np.nan
        df.loc[(df['humidity'] < 0) | (df['humidity'] > 100), 'humidity'] = np.nan
        df.loc[(df['wind_speed'] < 0) | (df['wind_speed'] > 50), 'wind_speed'] = np.nan
        
        # Interpolation linéaire (max 3 jours consécutifs)
        df = df.interpolate(method='linear', limit=3, limit_direction='both')
        
        # Remplacer NaN restants par médiane
        for col in df.columns:
            if col != 'date' and df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        return df

# ==================== SENTINEL HUB CLIENT ====================

class SentinelHubClient:
    """
    Client pour Sentinel Hub API - Données Sentinel-2 réelles
    Nécessite compte Sentinel Hub (gratuit jusqu'à 3 requêtes/mois)
    """
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.base_url = "https://services.sentinel-hub.com"
        
        if client_id and client_secret:
            self.authenticate()
    
    def authenticate(self) -> bool:
        """Authentification OAuth2"""
        if not self.client_id or not self.client_secret:
            logger.warning("⚠️ Sentinel Hub: identifiants manquants")
            return False
        
        try:
            logger.info("🔐 Authentification Sentinel Hub...")
            
            response = requests.post(
                f"{self.base_url}/oauth/token",
                data={
                    'grant_type': 'client_credentials',
                    'client_id': self.client_id,
                    'client_secret': self.client_secret
                },
                timeout=15
            )
            
            if response.status_code == 200:
                self.token = response.json()['access_token']
                logger.info("✅ Sentinel Hub authentifié")
                return True
            else:
                logger.error(f"❌ Authentification échouée: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur auth Sentinel Hub: {e}")
            return False
    
    def get_indices_data(self, bbox: List[float], start_date: str, end_date: str,
                        max_cloud_cover: int = 30) -> Optional[pd.DataFrame]:
        """
        Récupère indices Sentinel-2 réels
        
        Args:
            bbox: [lon_min, lat_min, lon_max, lat_max]
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            max_cloud_cover: Couverture nuageuse max (%)
            
        Returns:
            DataFrame avec indices ou None
        """
        if not self.token:
            logger.warning("⚠️ Sentinel Hub non authentifié")
            return None
        
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"],
                    units: "DN"
                }],
                output: {
                    bands: 7,
                    sampleType: "FLOAT32"
                }
            };
        }

        function evaluatePixel(sample) {
            // Masque nuages (SCL)
            if (sample.SCL == 3 || sample.SCL == 8 || sample.SCL == 9 || sample.SCL == 10) {
                return [NaN, NaN, NaN, NaN, NaN, NaN, 1];
            }
            
            // NDVI
            let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 0.0001);
            
            // EVI
            let evi = 2.5 * ((sample.B08 - sample.B04) / (sample.B08 + 6*sample.B04 - 7.5*sample.B02 + 1));
            
            // NDWI
            let ndwi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11 + 0.0001);
            
            // SAVI (L=0.5)
            let savi = 1.5 * ((sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 0.5));
            
            // LAI (empirique)
            let lai = 3.618 * evi - 0.118;
            lai = lai < 0 ? 0 : (lai > 7 ? 7 : lai);
            
            // MSAVI
            let msavi = (2*sample.B08 + 1 - Math.sqrt(Math.pow(2*sample.B08+1, 2) - 8*(sample.B08-sample.B04)))/2;
            
            return [ndvi, evi, ndwi, savi, lai, msavi, 0];
        }
        """
        
        request_payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{start_date}T00:00:00Z",
                            "to": f"{end_date}T23:59:59Z"
                        },
                        "maxCloudCoverage": max_cloud_cover
                    }
                }]
            },
            "output": {
                "width": 512,
                "height": 512,
                "responses": [{
                    "identifier": "default",
                    "format": {"type": "application/json"}
                }]
            },
            "evalscript": evalscript
        }
        
        try:
            logger.info("📡 Requête Sentinel Hub...")
            
            headers = {
                'Authorization': f'Bearer {self.token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/process",
                json=request_payload,
                headers=headers,
                timeout=90
            )
            
            if response.status_code == 200:
                data = response.json()
                df = self._parse_sentinel_response(data)
                
                if df is not None:
                    logger.info(f"✅ Sentinel-2: {len(df)} observations")
                return df
            else:
                logger.error(f"❌ Sentinel Hub erreur: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur Sentinel Hub: {e}")
            return None
    
    def _parse_sentinel_response(self, data: dict) -> Optional[pd.DataFrame]:
        """Parse la réponse Sentinel Hub"""
        # Note: Implémentation simplifiée
        # Dans la vraie version, parser le JSON complexe retourné
        # Format dépend de la config (GeoTIFF, JSON, etc.)
        
        # Pour l'instant, retourne None (sera simulé en fallback)
        logger.warning("⚠️ Parsing Sentinel Hub non implémenté (use fallback)")
        return None

# ==================== OPENWEATHER CLIENT ====================

class OpenWeatherClient:
    """Client pour OpenWeather API - Prévisions météo"""
    
    BASE_URL = "http://api.openweathermap.org/data/2.5"
    
    @staticmethod
    def get_weather_forecast(lat: float, lon: float, api_key: str) -> Optional[pd.DataFrame]:
        """
        Récupère prévisions météo 7 jours
        
        Args:
            lat: Latitude
            lon: Longitude
            api_key: Clé API OpenWeather
            
        Returns:
            DataFrame avec prévisions ou None
        """
        if not api_key:
            logger.warning("⚠️ OpenWeather: clé API manquante")
            return None
        
        url = f"{OpenWeatherClient.BASE_URL}/forecast"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'metric',
            'cnt': 56  # 7 jours × 8 prévisions/jour (3h)
        }
        
        try:
            logger.info(f"📡 Requête OpenWeather: ({lat:.4f}, {lon:.4f})")
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                forecasts = []
                
                for item in data['list']:
                    forecasts.append({
                        'datetime': datetime.fromtimestamp(item['dt']),
                        'temp': item['main']['temp'],
                        'temp_min': item['main']['temp_min'],
                        'temp_max': item['main']['temp_max'],
                        'humidity': item['main']['humidity'],
                        'rain': item.get('rain', {}).get('3h', 0),
                        'description': item['weather'][0]['description'],
                        'wind_speed': item['wind']['speed'],
                        'clouds': item['clouds']['all']
                    })
                
                df = pd.DataFrame(forecasts)
                df['date'] = df['datetime'].dt.date
                
                # Agrégation journalière
                daily = df.groupby('date').agg({
                    'temp': 'mean',
                    'temp_min': 'min',
                    'temp_max': 'max',
                    'humidity': 'mean',
                    'rain': 'sum',
                    'wind_speed': 'mean',
                    'clouds': 'mean',
                    'description': 'first'
                }).reset_index()
                
                logger.info(f"✅ OpenWeather: {len(daily)} jours prévus")
                return daily
                
            else:
                logger.error(f"❌ OpenWeather erreur: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur OpenWeather: {e}")
            return None

# ==================== DATA VALIDATOR ====================

class DataValidator:
    """Validation et contrôle qualité des données"""
    
    @staticmethod
    def validate_ndvi(ndvi_values: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Valide les valeurs NDVI
        
        Returns:
            (ndvi_corrigé, liste_warnings)
        """
        warnings = []
        
        # Plage théorique: -1 à 1
        if np.any(ndvi_values < -1) or np.any(ndvi_values > 1):
            n_invalid = np.sum((ndvi_values < -1) | (ndvi_values > 1))
            warnings.append(f"⚠️ {n_invalid} valeurs NDVI hors plage [-1, 1] → clippées")
            ndvi_values = np.clip(ndvi_values, -1, 1)
        
        # Détection outliers (méthode IQR)
        q1, q3 = np.percentile(ndvi_values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = (ndvi_values < lower_bound) | (ndvi_values > upper_bound)
        if np.any(outliers):
            n_outliers = np.sum(outliers)
            warnings.append(f"⚠️ {n_outliers} valeurs NDVI aberrantes détectées (IQR)")
        
        # Avertissement si beaucoup de valeurs négatives (eau/sol nu)
        neg_ratio = np.sum(ndvi_values < 0) / len(ndvi_values)
        if neg_ratio > 0.3:
            warnings.append(f"⚠️ {neg_ratio*100:.0f}% valeurs NDVI négatives (eau/sol nu)")
        
        return ndvi_values, warnings
    
    @staticmethod
    def validate_geometry(geometry) -> Tuple[bool, Optional[str]]:
        """Valide une géométrie Shapely"""
        from config import MIN_AREA_HA, MAX_AREA_HA
        
        if not geometry.is_valid:
            return False, "❌ Géométrie invalide (auto-intersection?)"
        
        # Calcul surface approximatif
        area_deg2 = geometry.area
        area_ha = area_deg2 * 111 * 111 / 10000  # Approximation
        
        if area_ha < MIN_AREA_HA:
            return False, f"❌ Zone trop petite ({area_ha:.3f} ha < {MIN_AREA_HA} ha)"
        
        if area_ha > MAX_AREA_HA:
            return False, f"❌ Zone trop grande ({area_ha:.0f} ha > {MAX_AREA_HA} ha)"
        
        return True, None
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame, required_cols: List[str]) -> Dict:
        """Vérifie qualité d'un DataFrame"""
        quality_report = {
            'is_valid': True,
            'warnings': [],
            'missing_data': {},
            'completeness': 100.0
        }
        
        # Colonnes manquantes
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            quality_report['is_valid'] = False
            quality_report['warnings'].append(f"❌ Colonnes manquantes: {missing_cols}")
            return quality_report
        
        # Données manquantes
        for col in required_cols:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > 0:
                quality_report['missing_data'][col] = missing_pct
                
                if missing_pct > 20:
                    quality_report['warnings'].append(
                        f"⚠️ {col}: {missing_pct:.1f}% données manquantes"
                    )
        
        # Taux complétude global
        total_values = len(df) * len(required_cols)
        missing_values = df[required_cols].isna().sum().sum()
        quality_report['completeness'] = ((total_values - missing_values) / total_values) * 100
        
        if quality_report['completeness'] < 80:
            quality_report['is_valid'] = False
            quality_report['warnings'].append(
                f"❌ Complétude insuffisante: {quality_report['completeness']:.1f}%"
            )
        
        return quality_report
