"""
AgriSight Pro v4.0 - Configuration
Fichier: config.py
"""

from enum import Enum
from dataclasses import dataclass
from typing import List

# ==================== ENUMS ====================

class SoilType(Enum):
    """Types de sol"""
    ARGILE = "Argileux"
    SABLE = "Sableux"
    LIMON = "Limoneux"
    ARGILO_SABLEUX = "Argilo-sableux"
    LIMONO_SABLEUX = "Limono-sableux"

class AgroZone(Enum):
    """Zones agro-écologiques"""
    SAHEL = "Sahel (< 400mm/an)"
    SAHELO_SOUDANIEN = "Sahélo-soudanien (400-600mm)"
    SOUDANIEN = "Soudanien (600-900mm)"
    SOUDANO_GUINEEN = "Soudano-guinéen (900-1200mm)"
    GUINEEN = "Guinéen (> 1200mm)"

class ProductionLevel(Enum):
    """Niveau de production"""
    SUBSISTANCE = "Subsistance (< 0.5 ha)"
    PETIT_EXPLOITANT = "Petit exploitant (0.5-2 ha)"
    MOYEN = "Moyen (2-10 ha)"
    INTENSIF = "Intensif (> 10 ha)"

# ==================== DATACLASSES ====================

@dataclass
class CropParameters:
    """Paramètres optimaux par culture"""
    name: str
    ndvi_optimal: float
    rain_min: float
    rain_max: float
    temp_optimal: float
    temp_min: float
    temp_max: float
    yield_max: float
    cycle_days: int
    soil_preferences: List[SoilType]
    water_requirement: str  # faible, moyen, élevé, très élevé

# ==================== BASE DE DONNÉES CULTURES ====================

CROP_DATABASE = {
    "Mil": CropParameters(
        name="Mil",
        ndvi_optimal=0.6,
        rain_min=400,
        rain_max=600,
        temp_optimal=28,
        temp_min=20,
        temp_max=35,
        yield_max=1.5,
        cycle_days=90,
        soil_preferences=[SoilType.SABLE, SoilType.ARGILO_SABLEUX],
        water_requirement="faible"
    ),
    
    "Sorgho": CropParameters(
        name="Sorgho",
        ndvi_optimal=0.65,
        rain_min=450,
        rain_max=700,
        temp_optimal=30,
        temp_min=22,
        temp_max=38,
        yield_max=2.0,
        cycle_days=110,
        soil_preferences=[SoilType.ARGILO_SABLEUX, SoilType.LIMON],
        water_requirement="moyen"
    ),
    
    "Maïs": CropParameters(
        name="Maïs",
        ndvi_optimal=0.7,
        rain_min=500,
        rain_max=800,
        temp_optimal=25,
        temp_min=18,
        temp_max=32,
        yield_max=4.0,
        cycle_days=120,
        soil_preferences=[SoilType.LIMON, SoilType.ARGILE],
        water_requirement="élevé"
    ),
    
    "Arachide": CropParameters(
        name="Arachide",
        ndvi_optimal=0.6,
        rain_min=450,
        rain_max=700,
        temp_optimal=27,
        temp_min=20,
        temp_max=33,
        yield_max=2.5,
        cycle_days=120,
        soil_preferences=[SoilType.SABLE, SoilType.LIMONO_SABLEUX],
        water_requirement="moyen"
    ),
    
    "Riz": CropParameters(
        name="Riz",
        ndvi_optimal=0.75,
        rain_min=800,
        rain_max=1500,
        temp_optimal=26,
        temp_min=20,
        temp_max=35,
        yield_max=5.0,
        cycle_days=130,
        soil_preferences=[SoilType.ARGILE],
        water_requirement="très élevé"
    ),
    
    "Niébé": CropParameters(
        name="Niébé",
        ndvi_optimal=0.55,
        rain_min=350,
        rain_max=600,
        temp_optimal=28,
        temp_min=20,
        temp_max=35,
        yield_max=1.2,
        cycle_days=75,
        soil_preferences=[SoilType.SABLE, SoilType.ARGILO_SABLEUX],
        water_requirement="faible"
    ),
    
    "Manioc": CropParameters(
        name="Manioc",
        ndvi_optimal=0.65,
        rain_min=1000,
        rain_max=2000,
        temp_optimal=27,
        temp_min=20,
        temp_max=32,
        yield_max=20.0,
        cycle_days=300,
        soil_preferences=[SoilType.SABLE, SoilType.LIMONO_SABLEUX],
        water_requirement="moyen"
    ),
    
    "Tomate": CropParameters(
        name="Tomate",
        ndvi_optimal=0.7,
        rain_min=600,
        rain_max=1000,
        temp_optimal=24,
        temp_min=15,
        temp_max=30,
        yield_max=40.0,
        cycle_days=90,
        soil_preferences=[SoilType.LIMON],
        water_requirement="élevé"
    ),
    
    "Oignon": CropParameters(
        name="Oignon",
        ndvi_optimal=0.6,
        rain_min=400,
        rain_max=700,
        temp_optimal=20,
        temp_min=12,
        temp_max=28,
        yield_max=25.0,
        cycle_days=110,
        soil_preferences=[SoilType.LIMON, SoilType.ARGILO_SABLEUX],
        water_requirement="élevé"
    ),
    
    "Coton": CropParameters(
        name="Coton",
        ndvi_optimal=0.65,
        rain_min=600,
        rain_max=1000,
        temp_optimal=28,
        temp_min=20,
        temp_max=35,
        yield_max=2.5,
        cycle_days=150,
        soil_preferences=[SoilType.ARGILO_SABLEUX, SoilType.LIMON],
        water_requirement="moyen"
    ),
    
    "Pastèque": CropParameters(
        name="Pastèque",
        ndvi_optimal=0.6,
        rain_min=400,
        rain_max=600,
        temp_optimal=25,
        temp_min=18,
        temp_max=32,
        yield_max=30.0,
        cycle_days=85,
        soil_preferences=[SoilType.SABLE, SoilType.LIMONO_SABLEUX],
        water_requirement="élevé"
    )
}

# ==================== CLÉS API (À CONFIGURER) ====================

# Agromonitoring (OpenWeather Agro)
AGRO_API_KEY = '28641235f2b024b5f45f97df45c6a0d5'

# OpenWeather (Prévisions)
OPENWEATHER_KEY = 'b06c034b4894d54fc512f9cd30b61a4a'

# Google Gemini (IA)
GEMINI_API_KEY = 'AIzaSyBZ4494NUEL_N13soCCIgCfIrMqn2jxoD8'

# Sentinel Hub (Optionnel - données Sentinel-2 réelles)
SENTINEL_CLIENT_ID = None  # À configurer si disponible
SENTINEL_CLIENT_SECRET = None

# ==================== PARAMÈTRES APPLICATION ====================

# Base de données
DATABASE_PATH = "agrisight_data.db"

# Limites géométrie
MIN_AREA_HA = 0.01
MAX_AREA_HA = 10000

# Qualité données
MAX_CLOUD_COVER = 30  # %
MIN_DATA_POINTS = 3

# Cache
CACHE_TTL = 3600  # 1 heure

# Rate limiting API
API_DELAY = 0.5  # secondes entre requêtes

# Validation NDVI
NDVI_MIN = -1
NDVI_MAX = 1
NDVI_VEGETATION_MIN = 0  # Végétation active

# Validation température
TEMP_MIN = -20
TEMP_MAX = 55

# Validation pluie
RAIN_MIN = 0
RAIN_MAX = 500  # mm/jour

# Incertitude rendement
YIELD_UNCERTAINTY_BASE = 0.20  # ±20%

# ==================== STYLES CSS ====================

CUSTOM_CSS = """
<style>
    .success-box {
        background: linear-gradient(135deg, #D4EDDA 0%, #C3E6CB 100%); 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 5px solid #28A745; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .alert-box {
        background: linear-gradient(135deg, #FFF3CD 0%, #FFE69C 100%); 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 5px solid #FFC107; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .info-box {
        background: linear-gradient(135deg, #D1ECF1 0%, #BEE5EB 100%); 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 5px solid #17A2B8; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .danger-box {
        background: linear-gradient(135deg, #F8D7DA 0%, #F5C6CB 100%); 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 5px solid #DC3545; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .metric-card {
        background: white; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); 
        margin: 10px 0;
        border-top: 3px solid #28A745;
    }
    .stButton>button {
        border-radius: 8px; 
        font-weight: 600; 
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .wizard-step {
        padding: 15px; 
        border-left: 4px solid #007bff; 
        margin: 10px 0; 
        background: #f8f9fa; 
        border-radius: 5px;
    }
    .wizard-step.active {
        background: #e7f3ff; 
        border-left-color: #28a745;
    }
    .wizard-step.completed {
        background: #d4edda; 
        border-left-color: #28a745;
    }
    .data-quality-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.85em;
        font-weight: 600;
    }
    .quality-real {
        background: #28a745;
        color: white;
    }
    .quality-simulated {
        background: #ffc107;
        color: #333;
    }
</style>
"""

# ==================== RECOMMANDATIONS PAR DÉFAUT ====================

DEFAULT_RECOMMENDATIONS_DATA = [
    # Mil - Sahel - Sableux - Subsistance
    ("Mil", "Sableux", "Sahel (< 400mm/an)", "Subsistance (< 0.5 ha)", "250-400mm",
     "NPK 15-15-15: 100kg/ha au semis + Compost 2t/ha",
     "Urée 25kg/ha à 25-30j (si pluie suffisante)",
     "Irrigation d'appoint: 15-20mm/semaine en déficit hydrique",
     "Traitement semences (Apron Star) + Surveillance chenilles mineuses",
     0.85),
    
    # Mil - Sahelo-soudanien - Argilo-sableux - Petit exploitant
    ("Mil", "Argilo-sableux", "Sahélo-soudanien (400-600mm)", "Petit exploitant (0.5-2 ha)", "400-600mm",
     "NPK 15-15-15: 150kg/ha + DAP 50kg/ha au semis",
     "Urée 50kg/ha à 30-35j (fractionnée si >50kg)",
     "Irrigation: 20-25mm/semaine si cumul pluie <15mm/semaine",
     "Semences traitées + Décis 12.5EC contre foreurs à montaison",
     0.90),
    
    # Maïs - Soudanien - Limoneux - Moyen
    ("Maïs", "Limoneux", "Soudanien (600-900mm)", "Moyen (2-10 ha)", "600-900mm",
     "NPK 23-10-5: 250kg/ha + Urée 100kg/ha (50kg au semis)",
     "Urée 100kg/ha stade 4-6 feuilles + 50kg/ha floraison",
     "Irrigation gravitaire/aspersion: 30-40mm/semaine en période critique",
     "Cypermethrine 5% contre foreurs + Mancozèbe contre helminthosporiose",
     0.92),
    
    # Arachide - Sahelo-soudanien - Sableux - Petit exploitant
    ("Arachide", "Sableux", "Sahélo-soudanien (400-600mm)", "Petit exploitant (0.5-2 ha)", "450-700mm",
     "NPK 6-20-10: 200kg/ha au semis (phosphore essentiel)",
     "Gypse agricole 300kg/ha début floraison (apport Ca pour gousses)",
     "Irrigation: 25mm/semaine floraison-remplissage gousses",
     "Traitement semences (Thirame) + Chlorpyrifos contre termites",
     0.88),
    
    # Riz - Soudano-guinéen - Argileux - Intensif
    ("Riz", "Argileux", "Soudano-guinéen (900-1200mm)", "Intensif (> 10 ha)", "900-1200mm",
     "NPK 15-15-15: 300kg/ha + Urée 150kg/ha (50kg épandage)",
     "Urée 100kg/ha tallage (20-25j) + 75kg/ha initiation paniculaire (45-50j)",
     "Irrigation submersion: maintien lame d'eau 5-10cm (tallage à maturation)",
     "Herbicide post-levée + Carbofuran contre foreurs + Thiodicarbe nématodes",
     0.95),
    
    # Sorgho - Soudanien - Argilo-sableux - Moyen
    ("Sorgho", "Argilo-sableux", "Soudanien (600-900mm)", "Moyen (2-10 ha)", "600-900mm",
     "NPK 15-15-15: 200kg/ha + DAP 75kg/ha au semis",
     "Urée 75kg/ha montaison (35-40j) + 50kg/ha épiaison",
     "Irrigation complémentaire: 25-30mm/semaine floraison-grain laiteux",
     "Lambda-cyhalothrine contre pucerons + Propiconazole ergot/charbon",
     0.89),
    
    # Niébé - Sahel - Sableux - Subsistance
    ("Niébé", "Sableux", "Sahel (< 400mm/an)", "Subsistance (< 0.5 ha)", "300-500mm",
     "NPK 10-20-20: 100kg/ha (privilégier P-K, légumineuse fixe N)",
     "Apport phosphate naturel 150kg/ha si disponible",
     "Irrigation d'appoint: 15mm/semaine floraison-formation gousses",
     "Deltaméthrine contre thrips/pucerons + Surveillance bruches stockage",
     0.82),
    
    # Tomate - Guinéen - Limoneux - Intensif
    ("Tomate", "Limoneux", "Guinéen (> 1200mm)", "Intensif (> 10 ha)", ">1000mm",
     "NPK 15-15-15: 400kg/ha (200kg plantation + 100kg 30j + 100kg 60j)",
     "Engrais foliaire NPK 20-20-20 + oligoéléments (Mg, Ca, B) 3-4 applications",
     "Irrigation goutte-à-goutte: 30-50mm/semaine selon stade",
     "Fongicide systémique mildiou + Abamectine contre aleurodes/acariens + Tuteurage",
     0.93),
    
    # Oignon - Sahelo-soudanien - Limoneux - Moyen
    ("Oignon", "Limoneux", "Sahélo-soudanien (400-600mm)", "Moyen (2-10 ha)", "400-700mm",
     "NPK 10-20-20: 300kg/ha (100kg repiquage + 100kg 30j + 100kg 60j)",
     "Sulfate potassium 150kg/ha grossissement bulbes",
     "Irrigation aspersion/goutte-à-goutte: 20-30mm/semaine (arrêt 15j avant récolte)",
     "Mancozèbe mildiou + Diméthoate thrips + Désherbage manuel soigné",
     0.87),
    
    # Coton - Soudanien - Argilo-sableux - Intensif
    ("Coton", "Argilo-sableux", "Soudanien (600-900mm)", "Intensif (> 10 ha)", "600-1000mm",
     "NPK 15-15-15: 300kg/ha semis + Urée 100kg/ha",
     "Urée 100kg/ha boutons floraux (50-60j) + KCl 50kg/ha ouverture capsules",
     "Irrigation aspersion: 30-40mm/semaine floraison-maturation capsules",
     "Programme insecticide (6-8 traitements): Pyréthrinoïdes + Néonicotinoïdes chenilles/punaises",
     0.91),
    
    # Générique - Tous sols - Toutes zones - Subsistance (fallback)
    ("Mil", "Tous", "Toutes zones", "Subsistance (< 0.5 ha)", "variable",
     "NPK 15-15-15: 100kg/ha + Compost disponible",
     "Urée 30kg/ha si ressources disponibles",
     "Irrigation selon disponibilité eau",
     "Pratiques culturales locales + Rotation cultures",
     0.70)
]
