"""
AgriSight Pro v4.0 - Gestionnaire Base de Données
Fichier: database.py
"""

import sqlite3
import pandas as pd
import logging
from typing import Optional, List, Dict
from config import DATABASE_PATH, DEFAULT_RECOMMENDATIONS_DATA

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Gestionnaire de base de données SQLite pour persistance"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialise les tables de la base de données"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table zones d'étude
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zones (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                geometry TEXT NOT NULL,
                area_ha REAL,
                soil_type TEXT,
                agro_zone TEXT,
                production_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table analyses
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                zone_id TEXT NOT NULL,
                culture TEXT NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                ndvi_mean REAL,
                ndvi_min REAL,
                ndvi_max REAL,
                ndvi_std REAL,
                evi_mean REAL,
                ndwi_mean REAL,
                lai_mean REAL,
                rain_total REAL,
                rain_days INTEGER,
                temp_mean REAL,
                temp_min REAL,
                temp_max REAL,
                humidity_mean REAL,
                yield_potential REAL,
                yield_min REAL,
                yield_max REAL,
                yield_confidence REAL,
                water_stress REAL,
                heat_stress_days INTEGER,
                data_source TEXT,
                soil_type TEXT,
                agro_zone TEXT,
                production_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (zone_id) REFERENCES zones(id)
            )
        """)
        
        # Table recommandations contextuelles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                culture TEXT NOT NULL,
                soil_type TEXT NOT NULL,
                agro_zone TEXT NOT NULL,
                production_level TEXT NOT NULL,
                rain_range TEXT,
                fertilizer_base TEXT,
                fertilizer_cover TEXT,
                irrigation_strategy TEXT,
                pest_control TEXT,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table observations terrain (pour validation)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS field_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_id TEXT NOT NULL,
                culture TEXT NOT NULL,
                observation_date DATE NOT NULL,
                actual_yield REAL,
                phenological_stage TEXT,
                pest_presence TEXT,
                disease_presence TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (zone_id) REFERENCES zones(id)
            )
        """)
        
        # Table historique prévisions vs réel (benchmarking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS yield_validation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                predicted_yield REAL NOT NULL,
                actual_yield REAL,
                error_percentage REAL,
                validation_date DATE,
                notes TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id)
            )
        """)
        
        # Index pour performances
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_zones_name ON zones(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_zone ON analyses(zone_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_date ON analyses(start_date, end_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_culture ON recommendations(culture)")
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Base de données initialisée: {self.db_path}")
        
        # Peupler recommandations si vide
        self.populate_recommendations()
    
    def populate_recommendations(self):
        """Peuple la table recommandations si vide"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Vérifier si déjà peuplée
        cursor.execute("SELECT COUNT(*) FROM recommendations")
        count = cursor.fetchone()[0]
        
        if count > 0:
            conn.close()
            logger.info(f"📋 Recommandations déjà présentes ({count} entrées)")
            return
        
        # Insérer recommandations par défaut
        cursor.executemany("""
            INSERT INTO recommendations 
            (culture, soil_type, agro_zone, production_level, rain_range, 
             fertilizer_base, fertilizer_cover, irrigation_strategy, pest_control, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, DEFAULT_RECOMMENDATIONS_DATA)
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ {len(DEFAULT_RECOMMENDATIONS_DATA)} recommandations contextuelles ajoutées")
    
    # ==================== ZONES ====================
    
    def save_zone(self, zone_id: str, name: str, geometry_json: str, area_ha: float,
                  soil_type: str = None, agro_zone: str = None, production_level: str = None):
        """Sauvegarde ou met à jour une zone"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO zones 
                (id, name, geometry, area_ha, soil_type, agro_zone, production_level, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (zone_id, name, geometry_json, area_ha, soil_type, agro_zone, production_level))
            
            conn.commit()
            logger.info(f"✅ Zone sauvegardée: {name} ({zone_id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde zone: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_zone(self, zone_id: str) -> Optional[Dict]:
        """Récupère une zone par son ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM zones WHERE id = ?", (zone_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'name': row[1],
                'geometry': row[2],
                'area_ha': row[3],
                'soil_type': row[4],
                'agro_zone': row[5],
                'production_level': row[6],
                'created_at': row[7],
                'updated_at': row[8]
            }
        return None
    
    def list_zones(self, limit: int = 50) -> List[Dict]:
        """Liste toutes les zones"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, area_ha, soil_type, agro_zone, created_at 
            FROM zones 
            ORDER BY updated_at DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': r[0],
                'name': r[1],
                'area_ha': r[2],
                'soil_type': r[3],
                'agro_zone': r[4],
                'created_at': r[5]
            }
            for r in rows
        ]
    
    # ==================== ANALYSES ====================
    
    def save_analysis(self, analysis_data: Dict) -> bool:
        """Sauvegarde une analyse complète"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO analyses 
                (id, zone_id, culture, start_date, end_date, 
                 ndvi_mean, ndvi_min, ndvi_max, ndvi_std,
                 evi_mean, ndwi_mean, lai_mean,
                 rain_total, rain_days, temp_mean, temp_min, temp_max, humidity_mean,
                 yield_potential, yield_min, yield_max, yield_confidence,
                 water_stress, heat_stress_days, data_source,
                 soil_type, agro_zone, production_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_data['id'],
                analysis_data['zone_id'],
                analysis_data['culture'],
                analysis_data['start_date'],
                analysis_data['end_date'],
                analysis_data['ndvi_mean'],
                analysis_data['ndvi_min'],
                analysis_data['ndvi_max'],
                analysis_data['ndvi_std'],
                analysis_data['evi_mean'],
                analysis_data['ndwi_mean'],
                analysis_data['lai_mean'],
                analysis_data['rain_total'],
                analysis_data['rain_days'],
                analysis_data['temp_mean'],
                analysis_data['temp_min'],
                analysis_data['temp_max'],
                analysis_data['humidity_mean'],
                analysis_data['yield_potential'],
                analysis_data['yield_min'],
                analysis_data['yield_max'],
                analysis_data['yield_confidence'],
                analysis_data['water_stress'],
                analysis_data['heat_stress_days'],
                analysis_data['data_source'],
                analysis_data['soil_type'],
                analysis_data['agro_zone'],
                analysis_data['production_level']
            ))
            
            conn.commit()
            logger.info(f"✅ Analyse sauvegardée: {analysis_data['culture']} ({analysis_data['id']})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde analyse: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_historical_analyses(self, zone_id: str, culture: str = None, limit: int = 10) -> pd.DataFrame:
        """Récupère l'historique des analyses pour une zone"""
        conn = sqlite3.connect(self.db_path)
        
        if culture:
            query = """
                SELECT * FROM analyses 
                WHERE zone_id = ? AND culture = ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            params = (zone_id, culture, limit)
        else:
            query = """
                SELECT * FROM analyses 
                WHERE zone_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """
            params = (zone_id, limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict]:
        """Récupère une analyse par son ID"""
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query(
            "SELECT * FROM analyses WHERE id = ?",
            conn,
            params=(analysis_id,)
        )
        
        conn.close()
        
        if not df.empty:
            return df.iloc[0].to_dict()
        return None
    
    # ==================== RECOMMANDATIONS ====================
    
    def get_contextual_recommendation(self, culture: str, soil_type: str, 
                                     agro_zone: str, production_level: str) -> Optional[Dict]:
        """Récupère la meilleure recommandation contextualisée"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Recherche exacte
        cursor.execute("""
            SELECT fertilizer_base, fertilizer_cover, irrigation_strategy, 
                   pest_control, confidence_score
            FROM recommendations
            WHERE culture = ? AND soil_type = ? AND agro_zone = ? AND production_level = ?
            ORDER BY confidence_score DESC
            LIMIT 1
        """, (culture, soil_type, agro_zone, production_level))
        
        result = cursor.fetchone()
        
        # Si pas de correspondance exacte, recherche élargie
        if not result:
            cursor.execute("""
                SELECT fertilizer_base, fertilizer_cover, irrigation_strategy, 
                       pest_control, confidence_score
                FROM recommendations
                WHERE culture = ? AND (soil_type = ? OR soil_type = 'Tous')
                ORDER BY confidence_score DESC
                LIMIT 1
            """, (culture, soil_type))
            
            result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return {
                'fertilizer_base': result[0],
                'fertilizer_cover': result[1],
                'irrigation_strategy': result[2],
                'pest_control': result[3],
                'confidence_score': result[4]
            }
        
        return None
    
    def add_custom_recommendation(self, culture: str, soil_type: str, agro_zone: str,
                                 production_level: str, rain_range: str,
                                 fertilizer_base: str, fertilizer_cover: str,
                                 irrigation_strategy: str, pest_control: str,
                                 confidence_score: float = 0.8) -> bool:
        """Ajoute une recommandation personnalisée"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO recommendations 
                (culture, soil_type, agro_zone, production_level, rain_range,
                 fertilizer_base, fertilizer_cover, irrigation_strategy, pest_control, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (culture, soil_type, agro_zone, production_level, rain_range,
                  fertilizer_base, fertilizer_cover, irrigation_strategy, pest_control, confidence_score))
            
            conn.commit()
            logger.info(f"✅ Recommandation personnalisée ajoutée: {culture}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur ajout recommandation: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    # ==================== OBSERVATIONS TERRAIN ====================
    
    def add_field_observation(self, zone_id: str, culture: str, observation_date: str,
                             actual_yield: float = None, phenological_stage: str = None,
                             pest_presence: str = None, disease_presence: str = None,
                             notes: str = None) -> bool:
        """Ajoute une observation terrain"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO field_observations 
                (zone_id, culture, observation_date, actual_yield, phenological_stage,
                 pest_presence, disease_presence, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (zone_id, culture, observation_date, actual_yield, phenological_stage,
                  pest_presence, disease_presence, notes))
            
            conn.commit()
            logger.info(f"✅ Observation terrain ajoutée: {zone_id} - {culture}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur ajout observation: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_field_observations(self, zone_id: str, culture: str = None) -> pd.DataFrame:
        """Récupère les observations terrain"""
        conn = sqlite3.connect(self.db_path)
        
        if culture:
            query = """
                SELECT * FROM field_observations 
                WHERE zone_id = ? AND culture = ?
                ORDER BY observation_date DESC
            """
            params = (zone_id, culture)
        else:
            query = """
                SELECT * FROM field_observations 
                WHERE zone_id = ?
                ORDER BY observation_date DESC
            """
            params = (zone_id,)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    # ==================== VALIDATION RENDEMENTS ====================
    
    def add_yield_validation(self, analysis_id: str, predicted_yield: float,
                           actual_yield: float, validation_date: str,
                           notes: str = None) -> bool:
        """Ajoute une validation de rendement (prédiction vs réel)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            error_pct = ((actual_yield - predicted_yield) / predicted_yield) * 100
            
            cursor.execute("""
                INSERT INTO yield_validation 
                (analysis_id, predicted_yield, actual_yield, error_percentage, validation_date, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (analysis_id, predicted_yield, actual_yield, error_pct, validation_date, notes))
            
            conn.commit()
            logger.info(f"✅ Validation rendement ajoutée: erreur {error_pct:.1f}%")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur validation rendement: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_validation_stats(self) -> Dict:
        """Calcule les statistiques de validation du modèle"""
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query("""
            SELECT error_percentage, predicted_yield, actual_yield
            FROM yield_validation
            WHERE actual_yield IS NOT NULL
        """, conn)
        
        conn.close()
        
        if df.empty:
            return {'count': 0, 'mean_error': None, 'rmse': None}
        
        return {
            'count': len(df),
            'mean_error': df['error_percentage'].mean(),
            'median_error': df['error_percentage'].median(),
            'std_error': df['error_percentage'].std(),
            'rmse': np.sqrt(((df['predicted_yield'] - df['actual_yield']) ** 2).mean())
        }
    
    # ==================== UTILITAIRES ====================
    
    def export_to_csv(self, table_name: str, output_path: str) -> bool:
        """Exporte une table en CSV"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            
            df.to_csv(output_path, index=False)
            logger.info(f"✅ Table {table_name} exportée: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur export CSV: {e}")
            return False
    
    def get_database_stats(self) -> Dict:
        """Retourne statistiques de la base de données"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        for table in ['zones', 'analyses', 'recommendations', 'field_observations', 'yield_validation']:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]
        
        conn.close()
        
        return stats
