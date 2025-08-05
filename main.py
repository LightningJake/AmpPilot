import os
import googlemaps
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional
import uvicorn
from math import radians, cos, sin, sqrt, atan2
import math
import logging
import joblib
from datetime import datetime
import pytz
import json
from shapely.geometry import shape, Point
import concurrent.futures
import threading
import requests
import requests.adapters
import pandas as pd
from ev_data_loader import ev_data_loader
import warnings
import uuid
warnings.filterwarnings('ignore')

app = FastAPI()

# CORS (for frontend connection) - Enhanced for browser compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "https://localhost:3000",
        "https://127.0.0.1:3000",
        "*"  # Fallback for any origin
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=[
        "accept",
        "accept-encoding", 
        "authorization",
        "content-type",
        "dnt",
        "origin",
        "user-agent",
        "x-csrftoken",
        "x-requested-with",
    ],
    expose_headers=["*"],
)

# Google Maps API Key with optimized connection pool
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY") or ""

# Create Google Maps client with optimized settings for production
session = requests.Session()
# Increase connection pool size to handle concurrent requests
adapter = requests.adapters.HTTPAdapter(
    pool_connections=20,  # Increased from default 10
    pool_maxsize=20,      # Increased from default 10
    max_retries=3
)
session.mount('https://', adapter)
session.mount('http://', adapter)

gmaps = googlemaps.Client(
    key=GOOGLE_MAPS_API_KEY,
    requests_session=session,
    timeout=10  # Add timeout for reliability
)

# Constants for production optimization
MAX_VEHICLE_RANGE_KM = 200  # Maximum range at 100% charge
GEOFENCING_RADIUS_KM = 3  # 3km radius for better station detection
MAX_THREADS = 6  # Reduced from 8 to prevent connection pool exhaustion
THREAD_LOCK = threading.Lock()  # For thread-safe operations
API_DELAY = 0.1  # Small delay between API calls to prevent rate limiting
# Semaphore to limit concurrent API calls and prevent connection pool exhaustion
API_SEMAPHORE = threading.Semaphore(10)  # Max 10 concurrent API calls

# Request Models
class EVRouteRequest(BaseModel):
    origin: str
    destination: str
    current_range_km: float  # Current remaining range shown on dashboard
    battery_data: Optional[dict] = None  # Optional battery health data

class BreakdownRequest(BaseModel):
    latitude: float
    longitude: float

class ProducerAnalysisRequest(BaseModel):
    state_name: str
    district_name: Optional[str] = None  # Optional district for enhanced analysis
    analysis_type: str = "urban"  # urban, highway, heavy_duty
    fast_mode: bool = True  # Enable fast mode for presentations

class OccupancyPredictionRequest(BaseModel):
    latitude: float
    longitude: float
    station_name: Optional[str] = "Unknown Station"
    place_id: Optional[str] = None
    prediction_time: Optional[str] = None  # ISO format datetime string
    
    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v
    
    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v

class BulkOccupancyRequest(BaseModel):
    stations: List[dict]  # List of station dictionaries with lat, lng, name, place_id
    prediction_time: Optional[str] = None

class BatteryHealthRequest(BaseModel):
    user_id: str
    battery_capacity_kwh: float  # Total battery capacity
    current_charge_percent: float  # Current charge level
    charging_cycles: Optional[int] = 300  # Number of charging cycles (default: 300)
    average_temperature: Optional[float] = 25.0  # Average operating temperature (°C, default: 25°C)
    fast_charge_frequency: Optional[float] = 0.3  # Percentage of fast charges vs slow charges (default: 30%)
    driving_patterns: Optional[dict] = None  # Aggressive, eco, moderate driving percentages
    vehicle_age_months: Optional[int] = 12  # Age of the vehicle in months (default: 12 months)
    manufacturer: Optional[str] = "Tata"  # Tata, MG, Hyundai, etc. (default: Tata - India's leading EV brand)
    model: Optional[str] = "Nexon EV"  # Nexon EV, ZS EV, etc. (default: Nexon EV - India's bestseller)
    auto_detect_specs: Optional[bool] = True  # Whether to auto-detect vehicle specs from database
    
    @validator('battery_capacity_kwh', pre=True)
    def validate_battery_capacity(cls, v):
        try:
            capacity = float(v)
            return max(10.0, min(200.0, capacity))  # Between 10kWh and 200kWh
        except (ValueError, TypeError):
            raise ValueError("Battery capacity must be a valid number")
    
    @validator('current_charge_percent', pre=True)
    def validate_current_charge(cls, v):
        try:
            charge = float(v)
            return max(0.0, min(100.0, charge))  # Between 0% and 100%
        except (ValueError, TypeError):
            raise ValueError("Current charge percent must be a valid number")
    
    @validator('charging_cycles', pre=True)
    def validate_charging_cycles(cls, v):
        if v == '' or v is None:
            return 300  # Default value
        try:
            cycles = int(v)
            if cycles < 0:
                return 300  # Default for negative values
            return max(0, cycles)  # Ensure non-negative
        except (ValueError, TypeError):
            return 300  # Default on invalid type
    
    @validator('average_temperature', pre=True)
    def validate_temperature(cls, v):
        if v == '' or v is None:
            return 25.0  # Default value
        try:
            temp = float(v)
            # Reasonable temperature range: -40°C to 80°C
            return max(-40.0, min(80.0, temp))
        except (ValueError, TypeError):
            return 25.0  # Default on invalid type
    
    @validator('fast_charge_frequency', pre=True)
    def validate_fast_charge_frequency(cls, v):
        if v == '' or v is None:
            return 0.3  # Default value
        try:
            freq = float(v)
            # Frequency should be between 0 and 1 (0% to 100%)
            return max(0.0, min(1.0, freq))
        except (ValueError, TypeError):
            return 0.3  # Default on invalid type
    
    @validator('vehicle_age_months', pre=True)
    def validate_vehicle_age(cls, v):
        if v == '' or v is None:
            return 12  # Default value
        try:
            age = int(v)
            if age < 0:
                return 12  # Default for negative values
            return max(0, age)  # Ensure non-negative
        except (ValueError, TypeError):
            return 12  # Default on invalid type
    
    @validator('manufacturer', pre=True)
    def validate_manufacturer(cls, v):
        if v == '' or v is None:
            return "Tata"  # Default to India's leading EV brand
        return str(v)
    
    @validator('model', pre=True)
    def validate_model(cls, v):
        if v == '' or v is None:
            return "Nexon EV"  # Default to India's bestselling EV
        return str(v)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Load the trained occupancy model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_model.pkl")
try:
    occupancy_model = joblib.load(MODEL_PATH)
    logging.info(f"Occupancy prediction model loaded successfully from {MODEL_PATH}")
    logging.info(f"Model type: {type(occupancy_model)}")
    logging.info(f"Model expects {occupancy_model.n_features_in_} features")
except Exception as e:
    occupancy_model = None
    logging.error(f"Failed to load occupancy model from {MODEL_PATH}: {e}")

# Weather API for enhanced occupancy prediction
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") or "e9a670988a29f296d925ade0692dc7a2"

# Helper: Haversine distance calculator
def haversine(coord1, coord2):
    R = 6371  # Earth radius in km
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# Helper: Get weather data for enhanced occupancy prediction
def get_weather_data(lat, lng):
    """Get current weather data for occupancy prediction"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        weather_main = data["weather"][0]["main"]
        temperature = data["main"]["temp"]
        
        # Convert weather to numeric (based on model training data)
        weather_numeric = {
            'Clear': 1, 'Clouds': 2, 'Rain': 3, 'Thunderstorm': 4, 
            'Drizzle': 3, 'Snow': 5, 'Mist': 2, 'Haze': 2
        }.get(weather_main, 2)  # Default to 2 (Clouds)
        
        return {
            'temperature': temperature,
            'weather_numeric': weather_numeric,
            'weather_description': weather_main
        }
    except Exception as e:
        logging.warning(f"Weather API failed: {e}, using defaults")
        return {
            'temperature': 30.0,  # Default Indian temperature
            'weather_numeric': 2,  # Default to cloudy
            'weather_description': 'Unknown'
        }

# Helper: Get traffic data estimation
def get_traffic_estimation(station_lat, station_lng, dt=None):
    """Estimate traffic level based on time and location"""
    try:
        if dt is None:
            dt = datetime.now(pytz.timezone("Asia/Kolkata"))
        
        hour = dt.hour
        is_weekend = dt.weekday() >= 5
        
        # Traffic level estimation (1=Low, 2=Medium, 3=High)
        if is_weekend:
            if 10 <= hour <= 20:
                traffic_level = 2  # Medium on weekends during day
            else:
                traffic_level = 1  # Low otherwise
        else:
            if (7 <= hour <= 10) or (17 <= hour <= 20):
                traffic_level = 3  # High during rush hours
            elif 10 <= hour <= 17:
                traffic_level = 2  # Medium during work hours
            else:
                traffic_level = 1  # Low during night/early morning
        
        return traffic_level
    except Exception as e:
        logging.warning(f"Traffic estimation failed: {e}")
        return 2  # Default to medium traffic

# Helper: Predict occupancy for a charging station using trained ML model
def predict_station_occupancy(station, dt=None):
    """
    Predict occupancy for a given station using the trained ML model.
    Returns occupancy probability (0.0 to 1.0) or None if prediction fails.
    The model expects exactly 30 features as trained.
    """
    if occupancy_model is None:
        logging.warning("Occupancy model not loaded, returning default occupancy")
        return 0.5  # Default occupancy

    try:
        # Use IST timezone for India
        if dt is None:
            dt = datetime.now(pytz.timezone("Asia/Kolkata"))
        
        # Extract temporal features
        hour = dt.hour
        day = dt.weekday()  # 0=Monday, 6=Sunday
        is_weekend = int(day >= 5)
        
        # Extract station location
        station_lat = station["location"]["lat"]
        station_lng = station["location"]["lng"]
        
        # Get weather data
        weather_data = get_weather_data(station_lat, station_lng)
        temperature = weather_data['temperature']
        
        # Get traffic estimation
        traffic_level = get_traffic_estimation(station_lat, station_lng, dt)
        traffic_delay_min = traffic_level * 5.0  # Convert level to minutes
        
        # Generate station ID (consistent hash from place_id or coordinates)
        station_id = hash(station.get('place_id', f"{station_lat}_{station_lng}")) % 100000
        
        # Station quality features
        rating = station.get('rating')
        if not isinstance(rating, (int, float)):
            rating = 4.0  # Assign a neutral default rating
        user_ratings_total = station.get('user_ratings_total', 0)
        open_now = 1 if station.get('open_now', True) else 0
        
        # City mapping (based on training data)
        city_mapping = {
            'Mumbai': 0, 'Pune': 1, 'Nagpur': 2, 'Nashik': 3, 
            'Aurangabad': 4, 'Hyderabad': 5, 'Nizamabad': 6
        }
        
        # Determine city based on coordinates (approximate)
        if 18.5 <= station_lat <= 19.3 and 72.7 <= station_lng <= 73.2:
            city_encoded = city_mapping['Mumbai']
        elif 18.4 <= station_lat <= 18.7 and 73.7 <= station_lng <= 74.0:
            city_encoded = city_mapping['Pune']
        elif 20.9 <= station_lat <= 21.3 and 78.8 <= station_lng <= 79.3:
            city_encoded = city_mapping['Nagpur']
        elif 19.8 <= station_lat <= 20.2 and 73.6 <= station_lng <= 74.0:
            city_encoded = city_mapping['Nashik']
        elif 19.7 <= station_lat <= 20.0 and 75.2 <= station_lng <= 75.5:
            city_encoded = city_mapping['Aurangabad']
        elif 17.2 <= station_lat <= 17.6 and 78.2 <= station_lng <= 78.7:
            city_encoded = city_mapping['Hyderabad']
        else:
            city_encoded = 0  # Default to Mumbai
        
        # Weather encoding (based on training data)
        weather_main = weather_data.get('weather_description', 'Clear')
        weather_encoding = {
            'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Thunderstorm': 3,
            'Drizzle': 2, 'Snow': 4, 'Mist': 1, 'Haze': 1, 'Unknown': 1
        }
        weather_encoded = weather_encoding.get(weather_main, 1)
        
        # Create the exact 30 features that the model expects
        # Based on the notebook's feature engineering
        
        # 1. Basic features
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        
        # 2. Time period encoding (0=night, 1=morning, 2=afternoon, 3=evening)
        if 0 <= hour <= 5:
            time_period_encoded = 0  # night
        elif 6 <= hour <= 11:
            time_period_encoded = 1  # morning
        elif 12 <= hour <= 17:
            time_period_encoded = 2  # afternoon
        else:
            time_period_encoded = 3  # evening
        
        # 3. Weekend pattern encoding
        if is_weekend:
            if 10 <= hour <= 16:
                weekend_pattern_encoded = 1  # weekend_afternoon
            else:
                weekend_pattern_encoded = 0  # weekend_other
        else:
            if 7 <= hour <= 10 or 17 <= hour <= 20:
                weekend_pattern_encoded = 2  # weekday_rush
            else:
                weekend_pattern_encoded = 3  # weekday_other
        
        # 4. Temperature features
        if temperature < 22:
            temperature_category_encoded = 0  # cold
        elif temperature <= 30:
            temperature_category_encoded = 1  # moderate
        elif temperature <= 35:
            temperature_category_encoded = 2  # warm
        else:
            temperature_category_encoded = 3  # hot
        
        temperature_squared = temperature ** 2
        temperature_normalized = min(1.0, max(0.0, (temperature - 15) / 30))  # 15-45°C range
        
        # 5. Traffic features
        if traffic_delay_min < 2:
            traffic_category_encoded = 0  # low
        elif traffic_delay_min < 5:
            traffic_category_encoded = 1  # medium
        else:
            traffic_category_encoded = 2  # high
        
        traffic_delay_squared = traffic_delay_min ** 2
        
        # 6. Station quality features
        if rating < 3.5:
            rating_category_encoded = 0  # poor
        elif rating < 4.0:
            rating_category_encoded = 1  # average
        elif rating < 4.5:
            rating_category_encoded = 2  # good
        else:
            rating_category_encoded = 3  # excellent
        
        if user_ratings_total < 5:
            popularity_category_encoded = 0  # new
        elif user_ratings_total < 20:
            popularity_category_encoded = 1  # moderate
        else:
            popularity_category_encoded = 2  # popular
        
        rating_normalized = min(1.0, max(0.0, (rating - 1) / 4))  # 1-5 rating range
        quality_score = rating_normalized * 0.7 + (min(1.0, user_ratings_total / 50)) * 0.3
        
        # 7. Location features
        city_size_encoded = 0 if city_encoded in [0, 1, 5] else 1  # large vs small cities
        # Distance from city center (approximation)
        city_centers = {
            0: (19.0760, 72.8777),  # Mumbai
            1: (18.5204, 73.8567),  # Pune
            5: (17.4065, 78.4772),  # Hyderabad
        }
        center_lat, center_lng = city_centers.get(city_encoded, (19.0760, 72.8777))
        distance_from_center = ((station_lat - center_lat) ** 2 + (station_lng - center_lng) ** 2) ** 0.5
        
        # 8. Interaction features (encoded as integers for simplicity)
        weather_temp_encoded = weather_encoded * 10 + temperature_category_encoded
        time_weather_encoded = time_period_encoded * 10 + weather_encoded
        quality_location_encoded = rating_category_encoded * 10 + city_size_encoded
        traffic_time_interaction = traffic_category_encoded * 10 + time_period_encoded
        
        # Create the exact 30-feature vector that matches the trained model
        features = [
            city_encoded,                   # "city_encoded"
            city_size_encoded,              # "city_size_encoded"
            day,                            # "day"
            distance_from_center,           # "distance_from_center"
            hour,                           # "hour"
            hour_cos,                       # "hour_cos"
            hour_sin,                       # "hour_sin"
            station_lat,                    # "lat"
            station_lng,                    # "lng"
            open_now,                       # "open_now"
            popularity_category_encoded,    # "popularity_category_encoded"
            quality_location_encoded,       # "quality_location_encoded"
            quality_score,                  # "quality_score"
            rating,                         # "rating"
            rating_category_encoded,        # "rating_category_encoded"
            rating_normalized,              # "rating_normalized"
            temperature,                    # "temperature"
            temperature_category_encoded,   # "temperature_category_encoded"
            temperature_normalized,         # "temperature_normalized"
            temperature_squared,            # "temperature_squared"
            time_period_encoded,            # "time_period_encoded"
            time_weather_encoded,           # "time_weather_encoded"
            traffic_category_encoded,       # "traffic_category_encoded"
            traffic_delay_min,              # "traffic_delay_min"
            traffic_delay_squared,          # "traffic_delay_squared"
            traffic_time_interaction,       # "traffic_time_interaction"
            user_ratings_total,             # "user_ratings_total"
            weather_encoded,                # "weather_encoded"
            weather_temp_encoded,           # "weather_temp_encoded"
            weekend_pattern_encoded         # "weekend_pattern_encoded"
        ]

        
        # Verify we have exactly 30 features
        if len(features) != 30:
            logging.error(f"Feature vector has {len(features)} features, expected 30")
            return 0.5
        
        # Make prediction
        prediction = occupancy_model.predict([features])[0]
        
        # Ensure prediction is within valid range (0.0 to 1.0)
        prediction = max(0.0, min(1.0, prediction))
        
        logging.info(f"Occupancy prediction for '{station.get('name', 'Unknown')}': {prediction:.2f} "
                    f"(hour={hour}, day={day}, weekend={is_weekend}, temp={temperature:.1f}°C, "
                    f"weather={weather_main}, traffic={traffic_level}, city={city_encoded})")
        
        return round(prediction, 3)
        
    except Exception as e:
        logging.error(f"Occupancy prediction failed for station {station.get('name', 'Unknown')}: {e}")
        logging.error(f"Error details: {str(e)}")
        return 0.5  # Return default occupancy on error

# Battery Health Prediction Engine
def predict_battery_health(battery_request: BatteryHealthRequest):
    """
    Enhanced battery health prediction using actual EV database.
    Returns State of Health (SoH) percentage and degradation predictions.
    """
    try:
        # Extract battery parameters with defaults for optional fields
        capacity_kwh = battery_request.battery_capacity_kwh
        cycles = battery_request.charging_cycles or 300
        avg_temp = battery_request.average_temperature or 25.0
        fast_charge_freq = battery_request.fast_charge_frequency or 0.3
        age_months = battery_request.vehicle_age_months or 12
        current_charge = battery_request.current_charge_percent
        manufacturer = battery_request.manufacturer or "Tata"
        model = battery_request.model or "Nexon EV"
        auto_detect = battery_request.auto_detect_specs if battery_request.auto_detect_specs is not None else True
        
        # ENHANCED: Get actual vehicle specifications from database
        vehicle_specs = None
        if auto_detect:
            vehicle_specs = ev_data_loader.get_vehicle_specs(manufacturer, model)
            logging.info(f"Vehicle lookup: {manufacturer} {model} - Found: {vehicle_specs.get('found', False)}")
        
        # Use actual vehicle data if available
        if vehicle_specs and vehicle_specs.get('found'):
            # Use actual battery capacity if available
            actual_min_capacity = vehicle_specs.get('battery_capacity_min_kwh')
            actual_max_capacity = vehicle_specs.get('battery_capacity_max_kwh')
            
            if actual_min_capacity and actual_max_capacity:
                # Use average of min and max as typical capacity
                actual_capacity = (actual_min_capacity + actual_max_capacity) / 2
                # Validate user input against actual specs (within 20% tolerance)
                if abs(capacity_kwh - actual_capacity) / actual_capacity > 0.2:
                    logging.warning(f"User input capacity ({capacity_kwh}kWh) differs significantly from actual vehicle capacity ({actual_capacity}kWh)")
                    # Use actual capacity for more accurate predictions
                    capacity_kwh = actual_capacity
            
            # Get actual range specifications
            actual_arai_range = vehicle_specs.get('arai_range_max_km')
            actual_real_range = vehicle_specs.get('real_world_range_km')
            
            # Get manufacturer from database (corrected spelling/casing)
            manufacturer = vehicle_specs.get('brand', manufacturer)
            model = vehicle_specs.get('model', model)
        
        # Handle driving patterns with defaults
        if battery_request.driving_patterns:
            driving_patterns = battery_request.driving_patterns
        else:
            # Default driving pattern: moderate usage
            driving_patterns = {
                'aggressive': 20,
                'eco': 50,
                'moderate': 30
            }
        
        aggressive_score = driving_patterns.get('aggressive', 20) / 100.0
        eco_score = driving_patterns.get('eco', 50) / 100.0
        moderate_score = driving_patterns.get('moderate', 30) / 100.0
        
        # ENHANCED: Updated manufacturer factors based on CSV data analysis
        manufacturer_factors = {
            # Premium International Brands (High-end EVs with advanced BMS)
            'tesla': 0.95,        # Model Y - Premium BMS
            'bmw': 0.93,          # iX1, iX, i7 - German engineering
            'mercedes-benz': 0.92, # EQB - Premium quality
            'audi': 0.91,         # e-tron GT - Premium segment
            'porsche': 0.94,      # Taycan - Top-tier engineering
            'rolls-royce': 0.96,  # Spectre - Ultra premium
            
            # Established International Brands in India
            'hyundai': 0.90,      # Creta Electric, Ioniq 5 - Solid battery tech
            'kia': 0.89,          # EV6 - Sister company to Hyundai
            
            # Leading Indian Market Players
            'tata': 0.87,         # Nexon EV, Punch EV, Curvv EV - Market leader, proven reliability
            'mahindra': 0.85,     # XUV400, BE 6, XEV 9e - Improving technology, focus on SUVs
            
            # Chinese Brands with Strong Battery Tech
            'byd': 0.88,          # Atto 3, Seal - Blade battery technology
            'mg': 0.84,           # Comet EV, ZS EV - Decent but volume focused
            
            # European Budget/Mid-range
            'citroen': 0.82,      # eC3 - Entry-level focus
            
            # Default for unknown manufacturers
            'default': 0.83
        }
        
        # Normalize manufacturer name for lookup
        manufacturer_normalized = manufacturer.lower().replace('-', '').replace(' ', '')
        manufacturer_factor = manufacturer_factors.get(
            manufacturer_normalized, 
            manufacturer_factors.get(manufacturer.lower(), manufacturer_factors['default'])
        )
        
        # ENHANCED: Battery chemistry and technology factor based on vehicle data
        battery_tech_factor = 1.0
        if vehicle_specs and vehicle_specs.get('found'):
            # Determine battery technology based on specs and year
            price_min = vehicle_specs.get('price_min_lakh', 0)
            dc_charge_time = vehicle_specs.get('dc_charge_time_min')
            
            # Premium vehicles (>40 lakh) likely have better battery tech
            if price_min > 40:
                battery_tech_factor = 0.95  # 5% better degradation
            elif price_min > 20:
                battery_tech_factor = 0.98  # 2% better degradation
            
            # Fast charging capability indicates better thermal management
            if dc_charge_time and dc_charge_time < 40:
                battery_tech_factor *= 0.97  # Better thermal management
        
        # 1. CALENDAR AGING (time-based degradation)
        # Enhanced based on battery chemistry and technology
        base_calendar_rate = 0.025 * battery_tech_factor  # 2.5% per year base
        calendar_degradation = (age_months / 12) * base_calendar_rate
        
        # 2. CYCLE AGING (usage-based degradation)
        # Enhanced based on actual battery capacity and technology
        cycle_capacity_factor = 1.0
        if capacity_kwh > 70:  # Large batteries typically have better longevity
            cycle_capacity_factor = 0.9
        elif capacity_kwh < 30:  # Small batteries may degrade faster
            cycle_capacity_factor = 1.1
        
        # Typical EV battery: ~80% health after 1000-1500 cycles
        cycle_degradation = min(0.20, cycles / 1500 * 0.20 * cycle_capacity_factor * battery_tech_factor)
        
        # 3. TEMPERATURE STRESS FACTOR (Enhanced for Indian conditions)
        # Indian climate considerations
        if 20 <= avg_temp <= 28:
            temp_factor = 1.0  # Optimal for Indian conditions
        elif avg_temp < 20:
            temp_factor = 1.0 + (20 - avg_temp) * 0.004  # Cold stress (mild in India)
        else:
            # Enhanced heat stress for Indian summers
            if avg_temp > 40:
                temp_factor = 1.0 + (avg_temp - 28) * 0.012  # Severe heat stress
            else:
                temp_factor = 1.0 + (avg_temp - 28) * 0.008  # Moderate heat stress
        
        temp_degradation = (temp_factor - 1.0) * 0.25
        
        # 4. FAST CHARGING STRESS (Enhanced based on actual DC charging capabilities)
        base_fast_charge_stress = max(0, (fast_charge_freq - 0.3) * 0.05)
        
        # Adjust based on vehicle's DC charging capability
        if vehicle_specs and vehicle_specs.get('found'):
            dc_charge_time = vehicle_specs.get('dc_charge_time_min')
            if dc_charge_time:
                if dc_charge_time < 30:  # Very fast charging (>100kW)
                    fast_charge_stress = base_fast_charge_stress * 0.8  # Better thermal management
                elif dc_charge_time > 60:  # Slower charging (<50kW)
                    fast_charge_stress = base_fast_charge_stress * 1.2  # May indicate older tech
                else:
                    fast_charge_stress = base_fast_charge_stress
            else:
                fast_charge_stress = base_fast_charge_stress
        else:
            fast_charge_stress = base_fast_charge_stress
        
        # 5. DRIVING PATTERN IMPACT
        driving_stress = (aggressive_score * 0.03) - (eco_score * 0.01)
        
        # 6. DEPTH OF DISCHARGE (DoD) impact - Enhanced
        # Estimate DoD based on battery size and usage patterns
        if capacity_kwh > 60:  # Large battery - users typically charge less frequently
            estimated_dod = 0.6
        elif capacity_kwh < 30:  # Small battery - users charge more frequently
            estimated_dod = 0.4
        else:
            estimated_dod = 0.5
        
        dod_factor = 1.0 + (estimated_dod - 0.3) * 0.02
        
        # CALCULATE TOTAL DEGRADATION
        total_degradation = (
            calendar_degradation + 
            cycle_degradation * temp_factor * dod_factor +
            temp_degradation +
            fast_charge_stress +
            driving_stress
        )
        
        # Apply manufacturer quality factor
        total_degradation *= (2.0 - manufacturer_factor)
        
        # Ensure total degradation is never negative
        total_degradation = max(0.0, total_degradation)
        
        # Current State of Health (SoH) - Enhanced limits
        current_soh = max(0.65, min(1.0, 1.0 - total_degradation))  # Between 65% and 100%
        
        # FUTURE PREDICTIONS (Enhanced with technology factors)
        def predict_future_health(years_ahead):
            future_calendar = calendar_degradation + (years_ahead * base_calendar_rate)
            future_cycles = cycles + (years_ahead * 365 * 1.0)  # ~1 cycle per day average
            future_cycle_degradation = min(0.20, future_cycles / 1500 * 0.20 * cycle_capacity_factor * battery_tech_factor)
            
            future_total = (
                future_calendar + 
                future_cycle_degradation * temp_factor * dod_factor +
                temp_degradation +
                fast_charge_stress +
                driving_stress
            ) * (2.0 - manufacturer_factor)
            
            future_total = max(0.0, future_total)
            return max(0.65, min(1.0, 1.0 - future_total))
        
        # Generate predictions
        predictions = {
            '1_year': predict_future_health(1),
            '2_years': predict_future_health(2),
            '5_years': predict_future_health(5)
        }
        
        # ENHANCED EFFECTIVE RANGE CALCULATION
        # Use actual vehicle range data if available
        if vehicle_specs and vehicle_specs.get('found'):
            actual_real_range = vehicle_specs.get('real_world_range_km')
            actual_arai_range = vehicle_specs.get('arai_range_max_km')
            
            if actual_real_range:
                nominal_range = actual_real_range
                range_source = "real_world_data"
            elif actual_arai_range:
                nominal_range = actual_arai_range * 0.85  # ARAI is typically optimistic
                range_source = "arai_adjusted"
            else:
                nominal_range = MAX_VEHICLE_RANGE_KM
                range_source = "default"
        else:
            nominal_range = MAX_VEHICLE_RANGE_KM
            range_source = "default"
        
        effective_range = nominal_range * current_soh
        
        # ENHANCED RECOMMENDATIONS with vehicle-specific insights
        recommendations = []
        
        if fast_charge_freq > 0.5:
            recommendations.append("Reduce fast charging frequency to preserve battery life")
        
        if avg_temp > 35:
            recommendations.append("Consider covered parking - high temperatures significantly accelerate aging")
        elif avg_temp > 30:
            recommendations.append("Park in shade when possible - heat stress affects battery longevity")
        
        if aggressive_score > 0.3:
            recommendations.append("Consider eco-driving mode to reduce battery stress")
        
        if current_soh < 0.80:
            recommendations.append("Battery showing significant aging - consider professional assessment")
        elif current_soh < 0.85:
            recommendations.append("Monitor range closely - battery showing signs of aging")
        
        if cycles > 1200:
            recommendations.append("High cycle count - excellent usage! Consider battery health checkup")
        
        # Vehicle-specific recommendations
        if vehicle_specs and vehicle_specs.get('found'):
            if vehicle_specs.get('dc_charge_time_min', 0) < 40:
                recommendations.append(f"Your {manufacturer} {model} has excellent fast charging - use it strategically for long trips")
            
            if capacity_kwh > 60:
                recommendations.append("Large battery pack - optimize by charging to 80% for daily use")
        
        # ENHANCED CHARGING STRATEGY OPTIMIZATION
        optimal_charge_range = {
            'daily_minimum': 20,
            'daily_maximum': 80,
            'trip_maximum': 90,
            'storage_level': 50
        }
        
        # Adjust based on battery size
        if capacity_kwh > 60:  # Large battery
            optimal_charge_range['daily_maximum'] = 75  # Can afford to charge less
        elif capacity_kwh < 30:  # Small battery
            optimal_charge_range['daily_maximum'] = 85  # May need higher daily charge
        
        # Confidence score based on data availability
        confidence_score = 0.70  # Base confidence
        if cycles > 100:
            confidence_score += 0.10
        if vehicle_specs and vehicle_specs.get('found'):
            confidence_score += 0.15  # Higher confidence with actual vehicle data
        if age_months > 6:
            confidence_score += 0.05  # More data points
        
        return {
            'current_state_of_health_percent': round(current_soh * 100, 1),
            'effective_range_km': round(effective_range, 1),
            'nominal_range_km': round(nominal_range, 1),
            'range_data_source': range_source,
            'vehicle_database_match': vehicle_specs.get('found', False) if vehicle_specs else False,
            'verified_vehicle_specs': vehicle_specs if vehicle_specs and vehicle_specs.get('found') else None,
            'degradation_factors': {
                'calendar_aging_percent': round(calendar_degradation * 100, 1),
                'cycle_aging_percent': round(cycle_degradation * 100, 1),
                'temperature_stress_percent': round(temp_degradation * 100, 1),
                'fast_charge_stress_percent': round(fast_charge_stress * 100, 1),
                'driving_pattern_impact_percent': round(driving_stress * 100, 1)
            },
            'future_predictions': {
                '1_year_health_percent': round(predictions['1_year'] * 100, 1),
                '2_years_health_percent': round(predictions['2_years'] * 100, 1),
                '5_years_health_percent': round(predictions['5_years'] * 100, 1)
            },
            'manufacturer_factor': manufacturer_factor,
            'battery_technology_factor': battery_tech_factor,
            'recommendations': recommendations,
            'optimal_charging_strategy': optimal_charge_range,
            'analysis_timestamp': datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
            'confidence_score': round(min(0.95, confidence_score), 2)
        }
        
    except Exception as e:
        logging.error(f"Battery health prediction failed: {e}")
        return {
            'error': f'Prediction failed: {str(e)}',
            'current_state_of_health_percent': 85.0,
            'effective_range_km': MAX_VEHICLE_RANGE_KM * 0.85,
            'vehicle_database_match': False
        }
        
        # Ensure total degradation is never negative
        total_degradation = max(0.0, total_degradation)
        
        # Current State of Health (SoH) - cap at 100% maximum
        current_soh = max(0.6, min(1.0, 1.0 - total_degradation))  # Between 60% and 100%
        
        # FUTURE PREDICTIONS (1, 2, 5 years)
        def predict_future_health(years_ahead):
            future_calendar = calendar_degradation + (years_ahead * 0.025)
            future_cycles = cycles + (years_ahead * 365 * 1.2)  # ~1.2 cycles per day
            future_cycle_degradation = min(0.20, future_cycles / 1500 * 0.20)
            
            future_total = (
                future_calendar + 
                future_cycle_degradation * temp_factor * dod_factor +
                temp_degradation +
                fast_charge_stress +
                driving_stress
            ) * (2.0 - manufacturer_factor)
            
            # Ensure future degradation is never negative
            future_total = max(0.0, future_total)
            
            return max(0.6, min(1.0, 1.0 - future_total))  # Between 60% and 100%
        
        # Generate predictions
        predictions = {
            '1_year': predict_future_health(1),
            '2_years': predict_future_health(2),
            '5_years': predict_future_health(5)
        }
        
        # EFFECTIVE RANGE CALCULATION
        # Current effective range considering battery health
        nominal_range = MAX_VEHICLE_RANGE_KM
        effective_range = nominal_range * current_soh
        
        # RECOMMENDATIONS
        recommendations = []
        
        if fast_charge_freq > 0.5:
            recommendations.append("Reduce fast charging frequency to preserve battery life")
        
        if avg_temp > 30:
            recommendations.append("Park in shade when possible - high temperatures accelerate aging")
        
        if aggressive_score > 0.3:
            recommendations.append("Consider eco-driving mode to reduce battery stress")
        
        if current_soh < 0.85:
            recommendations.append("Battery showing signs of aging - monitor range closely")
        
        if cycles > 1000:
            recommendations.append("High cycle count - consider professional battery assessment")
        
        # CHARGING STRATEGY OPTIMIZATION
        optimal_charge_range = {
            'daily_minimum': 20,  # Don't go below 20%
            'daily_maximum': 80,  # Don't charge above 80% daily
            'trip_maximum': 90,   # Only charge to 90% for long trips
            'storage_level': 50   # Store at 50% for long periods
        }
        
        return {
            'current_state_of_health_percent': round(current_soh * 100, 1),
            'effective_range_km': round(effective_range, 1),
            'degradation_factors': {
                'calendar_aging_percent': round(calendar_degradation * 100, 1),
                'cycle_aging_percent': round(cycle_degradation * 100, 1),
                'temperature_stress_percent': round(temp_degradation * 100, 1),
                'fast_charge_stress_percent': round(fast_charge_stress * 100, 1),
                'driving_pattern_impact_percent': round(driving_stress * 100, 1)
            },
            'future_predictions': {
                '1_year_health_percent': round(predictions['1_year'] * 100, 1),
                '2_years_health_percent': round(predictions['2_years'] * 100, 1),
                '5_years_health_percent': round(predictions['5_years'] * 100, 1)
            },
            'manufacturer_factor': manufacturer_factor,
            'recommendations': recommendations,
            'optimal_charging_strategy': optimal_charge_range,
            'analysis_timestamp': datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
            'confidence_score': 0.85 if cycles > 100 else 0.70  # Higher confidence with more data
        }
        
    except Exception as e:
        logging.error(f"Battery health prediction failed: {e}")
        return {
            'error': f'Prediction failed: {str(e)}',
            'current_state_of_health_percent': 85.0,  # Default assumption
            'effective_range_km': MAX_VEHICLE_RANGE_KM * 0.85
        }

def calculate_range_adjustment(battery_health_percent, temperature, driving_style='moderate', manufacturer='Tata', model='Nexon EV'):
    """
    Enhanced real-time range adjustment using vehicle database and battery health.
    """
    try:
        # Base range from battery health
        health_factor = battery_health_percent / 100.0
        
        # ENHANCED: Get actual vehicle range data
        vehicle_specs = ev_data_loader.get_vehicle_specs(manufacturer, model)
        
        if vehicle_specs and vehicle_specs.get('found'):
            # Use real-world range if available, otherwise ARAI adjusted
            base_range = vehicle_specs.get('real_world_range_km')
            if not base_range:
                arai_range = vehicle_specs.get('arai_range_max_km')
                if arai_range:
                    base_range = arai_range * 0.85  # ARAI is typically optimistic
                else:
                    base_range = MAX_VEHICLE_RANGE_KM
            
            range_source = "database_real_world" if vehicle_specs.get('real_world_range_km') else "database_arai_adjusted"
        else:
            base_range = MAX_VEHICLE_RANGE_KM
            range_source = "default"
        
        # Enhanced temperature impact based on Indian conditions
        temp_range_factor = 1.0
        if temperature < 15:  # Cold weather (rare in India but affects some regions)
            temp_range_factor = 0.75  # 25% reduction
        elif temperature > 40:  # Extreme Indian summer
            temp_range_factor = 0.70  # 30% reduction - AC usage, battery thermal management
        elif temperature > 35:  # Hot Indian weather
            temp_range_factor = 0.80  # 20% reduction - increased AC usage
        elif temperature > 30:  # Warm Indian weather
            temp_range_factor = 0.90  # 10% reduction - moderate AC usage
        
        # Enhanced driving style factors
        style_factors = {
            'eco': 1.25,      # 25% increase - optimized driving
            'moderate': 1.0,  # No change
            'aggressive': 0.65, # 35% decrease - Indian traffic conditions
            'city_traffic': 0.70,  # 30% decrease - stop-and-go traffic
            'highway': 1.10   # 10% increase - efficient cruising
        }
        
        style_factor = style_factors.get(driving_style, 1.0)
        
        # Additional adjustment based on vehicle type and drivetrain
        drivetrain_factor = 1.0
        if vehicle_specs and vehicle_specs.get('found'):
            drivetrain = vehicle_specs.get('drivetrain', '').upper()
            if 'AWD' in drivetrain:
                drivetrain_factor = 0.90  # AWD consumes more power
            elif 'RWD' in drivetrain:
                drivetrain_factor = 0.95  # RWD slightly less efficient than FWD
        
        # Calculate final adjusted range
        adjusted_range = base_range * health_factor * temp_range_factor * style_factor * drivetrain_factor
        
        return {
            'adjusted_range_km': round(adjusted_range, 1),
            'base_range_km': round(base_range, 1),
            'range_source': range_source,
            'adjustment_factors': {
                'health_factor': round(health_factor, 3),
                'temperature_factor': round(temp_range_factor, 3),
                'driving_style_factor': round(style_factor, 3),
                'drivetrain_factor': round(drivetrain_factor, 3)
            },
            'vehicle_database_match': vehicle_specs.get('found', False) if vehicle_specs else False,
            'conditions': {
                'temperature_c': temperature,
                'driving_style': driving_style,
                'battery_health_percent': battery_health_percent
            }
        }
        
    except Exception as e:
        logging.error(f"Enhanced range adjustment calculation failed: {e}")
        return {
            'adjusted_range_km': MAX_VEHICLE_RANGE_KM * 0.85,
            'base_range_km': MAX_VEHICLE_RANGE_KM,
            'range_source': 'error_fallback',
            'error': str(e)
        }

def find_better_stations_nearby(original_station_location, original_rating=None, original_occupancy=None):
    """Find better charging stations within 3km radius - now includes custom stations"""
    try:
        # Get combined stations from both Google Maps and custom stations
        original_location = (original_station_location['lat'], original_station_location['lng'])
        all_stations = get_combined_stations_nearby(original_location, radius=3000, keyword="EV charging")

        better_stations = []
        # Use original rating for comparison, default to 3.0 if not available
        original_rating_value = original_rating if original_rating else 3.0

        for idx, station in enumerate(all_stations):
            station_rating = station.get('rating', 4.0)  # Default rating for custom stations

            distance = haversine(
                original_location,
                (station['location']['lat'], station['location']['lng'])
            )
            
            if distance <= GEOFENCING_RADIUS_KM and distance > 0.1:
                # For Google Maps stations, get additional details
                if not station.get('is_custom_station', False):
                    try:
                        details = gmaps.place(place_id=station['place_id']).get('result', {})
                        open_now = details.get('opening_hours', {}).get('open_now')
                    except:
                        open_now = True  # Default assumption
                else:
                    open_now = station.get('open_now', True)
                
                station_info = {
                    "name": station['name'],
                    "address": station.get('address', station.get('vicinity', '')),
                    "rating": station_rating,
                    "location": station['location'],
                    "place_id": station['place_id'],
                    "station_id": station.get('station_id', idx + 2),
                    "distance_from_original": round(distance, 2),
                    "rating_improvement": round(station_rating - original_rating_value, 1),
                    "maps_link": station.get('maps_link', f"https://www.google.com/maps/place/?q=place_id:{station['place_id']}"),
                    "open_now": open_now,
                    "is_custom_station": station.get('is_custom_station', False)
                }
                
                # Add custom station specific info if applicable
                if station.get('is_custom_station'):
                    station_info.update({
                        "station_type": station.get('station_type'),
                        "charger_types": station.get('charger_types', []),
                        "power_output_kw": station.get('power_output_kw'),
                        "number_of_charging_points": station.get('number_of_charging_points', 1),
                        "operating_hours": station.get('operating_hours', '24/7'),
                        "amenities": station.get('amenities', []),
                        "pricing_info": station.get('pricing_info')
                    })
                
                # Predict occupancy
                occupancy = predict_station_occupancy(station_info)
                station_info["predicted_occupancy"] = occupancy

                # FIXED LOGIC: Look for stations that are meaningfully better than the original
                rating_improvement = station_rating - original_rating_value
                occupancy_improvement = (original_occupancy or 0.5) - (occupancy or 0.5)
                
                # Consider a station better if:
                # 1. Rating is at least 0.2 stars higher AND occupancy is not worse by more than 10%
                # 2. OR rating is at least 0.5 stars higher (regardless of occupancy)
                # 3. OR occupancy is significantly lower (20%+ improvement) and rating is not worse
                is_better_station = (
                    (rating_improvement >= 0.2 and occupancy_improvement >= -0.1) or
                    (rating_improvement >= 0.5) or
                    (occupancy_improvement >= 0.2 and rating_improvement >= 0)
                )
                
                if is_better_station and open_now:
                    better_stations.append(station_info)

        better_stations.sort(key=lambda x: (-x['rating'], x['distance_from_original']))
        return better_stations

    except Exception as e:
        logging.error(f"Error finding better stations: {e}")
        return []

def find_nearby_services(location, keyword, service_type, max_results=10):
    """
    IMPROVED: Find nearby services with better search strategy and distance-based filtering
    
    Key improvements:
    1. Multiple search strategies with fallback options
    2. Calculate distances for ALL results before limiting
    3. Use appropriate place types for different services  
    4. Quality filtering to remove irrelevant/distant results
    """
    try:
        logging.info(f"Searching for {service_type} near {location} with keyword '{keyword}'")
        
        # Define search configurations based on service type
        search_configs = []
        
        if service_type == "car_repair":
            search_configs = [
                {"keyword": "car repair", "type": "car_repair", "radius": 8000},
                {"keyword": "auto repair", "type": "establishment", "radius": 10000},
                {"keyword": "mechanic", "type": "establishment", "radius": 8000},
                {"keyword": "garage", "type": "establishment", "radius": 10000},
                {"keyword": "service center", "type": "establishment", "radius": 12000}
            ]
        elif service_type == "towing":
            # FIXED: Use 'establishment' instead of invalid 'towing' type
            search_configs = [
                {"keyword": "towing service", "type": "establishment", "radius": 15000},
                {"keyword": "tow truck", "type": "establishment", "radius": 15000},
                {"keyword": "roadside assistance", "type": "establishment", "radius": 20000},
                {"keyword": "breakdown service", "type": "establishment", "radius": 15000}
            ]
        else:
            # Fallback for any other service type
            search_configs = [
                {"keyword": keyword, "type": "establishment", "radius": 10000}
            ]
        
        all_places = []
        seen_place_ids = set()
        
        # Try each search configuration until we have enough results
        for config in search_configs:
            try:
                response = gmaps.places_nearby(
                    location=location,
                    radius=config["radius"],
                    keyword=config["keyword"],
                    type=config["type"]
                )
                
                places = response.get('results', [])
                logging.info(f"Search '{config['keyword']}' returned {len(places)} results")
                
                # Add unique places to our collection (avoid duplicates)
                for place in places:
                    if place['place_id'] not in seen_place_ids:
                        seen_place_ids.add(place['place_id'])
                        all_places.append(place)
                
                # If we have enough results for processing, we can stop early
                if len(all_places) >= 25:  # Buffer for distance sorting
                    break
                    
            except Exception as e:
                logging.error(f"Search failed for '{config['keyword']}': {e}")
                continue
        
        logging.info(f"Total unique places found: {len(all_places)}")
        
        if not all_places:
            logging.warning("No places found with any search strategy")
            return []
        
        # CRITICAL FIX: Calculate distances for ALL results FIRST
        services_with_distance = []
        for place in all_places:
            try:
                service = {
                    "name": place['name'],
                    "address": place.get('vicinity', 'Address not available'),
                    "rating": place.get('rating'),
                    "location": place['geometry']['location'],
                    "place_id": place['place_id'],
                    "maps_link": f"https://www.google.com/maps/place/?q=place_id:{place['place_id']}",
                    "service_type": service_type,
                    "phone": None 
                }
                
                # Calculate distance from breakdown location
                distance = haversine(
                    location, 
                    (place['geometry']['location']['lat'], place['geometry']['location']['lng'])
                )
                service["distance_km"] = round(distance, 2)
                
                # Quality filter: exclude results that are clearly too far
                max_distance = 25 if service_type == "towing" else 20  # Towing can be further
                if distance <= max_distance:
                    services_with_distance.append(service)
                
            except Exception as e:
                logging.error(f"Error processing place {place.get('name', 'Unknown')}: {e}")
                continue
        
        # CRITICAL FIX: Sort by distance BEFORE limiting results
        services_with_distance.sort(key=lambda x: x["distance_km"])
        
        # Return the closest services up to max_results
        final_services = services_with_distance[:max_results]
        
        if final_services:
            logging.info(f"Returning {len(final_services)} {service_type} services")
            logging.info(f"Closest: {final_services[0]['name']} at {final_services[0]['distance_km']}km")
            logging.info(f"Furthest: {final_services[-1]['name']} at {final_services[-1]['distance_km']}km")
        else:
            logging.warning(f"No {service_type} services found within reasonable distance")
        
        return final_services
        
    except Exception as e:
        logging.error(f"Error finding {service_type}: {e}")
        return []


def find_nearest_charger(location):
    """Enhanced charging station finder with ML-based occupancy prediction - now includes custom stations"""
    try:
        # Get combined stations from both Google Maps and custom stations
        all_stations = get_combined_stations_nearby(location, radius=5000, keyword="EV charging")

        if not all_stations:
            return None

        # Find the closest station
        closest_station = None
        min_distance = float('inf')
        
        for station in all_stations:
            station_location = (station['location']['lat'], station['location']['lng'])
            distance = haversine(location, station_location)
            
            if distance < min_distance:
                min_distance = distance
                closest_station = station

        if not closest_station:
            return None
        
        # For Google Maps stations, fetch additional details
        if not closest_station.get('is_custom_station', False):
            try:
                details = gmaps.place(place_id=closest_station['place_id']).get('result', {})
                open_now = details.get('opening_hours', {}).get('open_now')
            except:
                open_now = True  # Default assumption
        else:
            open_now = closest_station.get('open_now', True)
        
        station_info = {
            "name": closest_station['name'],
            "address": closest_station.get('address', closest_station.get('vicinity', '')),
            "rating": closest_station.get('rating'),
            "location": closest_station['location'],
            "place_id": closest_station['place_id'],
            "station_id": closest_station.get('station_id', hash(closest_station['place_id']) % 100000),
            "maps_link": closest_station.get('maps_link', f"https://www.google.com/maps/place/?q=place_id:{closest_station['place_id']}"),
            "open_now": open_now,
            "is_custom_station": closest_station.get('is_custom_station', False),
            "distance_km": round(min_distance, 2)
        }
        
        # Add custom station specific info if applicable
        if closest_station.get('is_custom_station'):
            station_info.update({
                "station_type": closest_station.get('station_type'),
                "charger_types": closest_station.get('charger_types', []),
                "power_output_kw": closest_station.get('power_output_kw'),
                "number_of_charging_points": closest_station.get('number_of_charging_points', 1),
                "operating_hours": closest_station.get('operating_hours', '24/7'),
                "amenities": closest_station.get('amenities', []),
                "pricing_info": closest_station.get('pricing_info')
            })
        
        # Enhanced occupancy prediction with detailed context
        occupancy = predict_station_occupancy(station_info)
        station_info["predicted_occupancy"] = occupancy
        
        # Add occupancy context and recommendation
        if occupancy is not None:
            if occupancy <= 0.3:
                station_info["occupancy_level"] = "Low"
                station_info["wait_time_estimate"] = "0-5 minutes"
            elif occupancy <= 0.7:
                station_info["occupancy_level"] = "Moderate"
                station_info["wait_time_estimate"] = "5-15 minutes"
            else:
                station_info["occupancy_level"] = "High"
                station_info["wait_time_estimate"] = "15+ minutes"
        else:
            station_info["occupancy_level"] = "Unknown"
            station_info["wait_time_estimate"] = "Unknown"
            
        return station_info
        
    except Exception as e:
        logging.error(f"Error finding nearest charger: {e}")
        return None

@app.get("/station-status/{place_id}")
def get_station_status(place_id: str):
    """
    Get real-time status and occupancy prediction for a specific charging station.
    Useful for monitoring individual stations and providing detailed user information.
    """
    try:
        # Get station details from Google Places
        details = gmaps.place(place_id=place_id).get('result', {})
        
        if not details:
            return {"error": f"Station with place_id {place_id} not found"}
        
        # Extract station information
        station_info = {
            "name": details.get('name', 'Unknown Station'),
            "address": details.get('formatted_address', details.get('vicinity', 'Address not available')),
            "location": details.get('geometry', {}).get('location', {}),
            "rating": details.get('rating'),
            "user_ratings_total": details.get('user_ratings_total', 0),
            "place_id": place_id,
            "station_id": hash(place_id) % 100000,
            "phone": details.get('formatted_phone_number'),
            "website": details.get('website'),
            "maps_link": f"https://www.google.com/maps/place/?q=place_id:{place_id}"
        }
        
        # Check if station is currently open
        opening_hours = details.get('opening_hours', {})
        station_info["open_now"] = opening_hours.get('open_now')
        station_info["opening_hours"] = opening_hours.get('weekday_text', [])
        
        # Get occupancy prediction
        occupancy = predict_station_occupancy(station_info)
        
        # Enhanced status information
        current_time = datetime.now(pytz.timezone("Asia/Kolkata"))
        
        status_info = {
            "occupancy_prediction": {
                "probability": occupancy,
                "level": "Unknown" if occupancy is None else (
                    "Low" if occupancy <= 0.3 else 
                    "Moderate" if occupancy <= 0.7 else "High"
                ),
                "wait_time_estimate": "Unknown" if occupancy is None else (
                    "0-5 minutes" if occupancy <= 0.3 else
                    "5-15 minutes" if occupancy <= 0.7 else "15+ minutes"
                ),
                "recommendation": "Unknown" if occupancy is None else (
                    "Good time to charge" if occupancy <= 0.3 else
                    "Moderate wait expected" if occupancy <= 0.7 else
                    "Consider alternative stations"
                )
            },
            "operational_status": {
                "is_open": station_info["open_now"],
                "last_updated": current_time.isoformat()
            },
            "context": {
                "current_time": current_time.strftime("%I:%M %p"),
                "day_of_week": current_time.strftime("%A"),
                "is_weekend": current_time.weekday() >= 5
            }
        }
        
        # Get weather data for context
        if station_info.get("location"):
            weather_data = get_weather_data(
                station_info["location"]["lat"], 
                station_info["location"]["lng"]
            )
            status_info["weather"] = {
                "temperature_celsius": weather_data["temperature"],
                "condition": weather_data["weather_description"]
            }
        
        return {
            "station": station_info,
            "status": status_info,
            "success": True
        }
        
    except Exception as e:
        logging.error(f"Failed to get station status for {place_id}: {e}")
        return {
            "error": f"Failed to retrieve station status: {str(e)}",
            "place_id": place_id,
            "success": False
        }

@app.get("/nearby-stations")
def get_nearby_stations(lat: float, lng: float, radius: int = 5000):
    """
    Get all nearby charging stations with real-time occupancy predictions.
    Enhanced version of station search with ML predictions for better user experience.
    """
    try:
        # Validate coordinates
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return {"error": "Invalid coordinates provided"}
        
        # Search for combined nearby charging stations (Google Maps + Custom)
        location = (lat, lng)
        all_stations = get_combined_stations_nearby(location, radius=min(radius, 10000), keyword="EV charging")
        
        enhanced_stations = []
        
        for station in all_stations:
            try:
                # Build station information
                station_info = {
                    "name": station['name'],
                    "address": station.get('address', station.get('vicinity', '')),
                    "location": station['location'],
                    "rating": station.get('rating', 4.0),  # Default for custom stations
                    "user_ratings_total": station.get('user_ratings_total', 0),
                    "place_id": station['place_id'],
                    "station_id": station.get('station_id', hash(station['place_id']) % 100000),
                    "maps_link": station.get('maps_link', f"https://www.google.com/maps/place/?q=place_id:{station['place_id']}"),
                    "is_custom_station": station.get('is_custom_station', False)
                }
                
                # Calculate distance from user location
                distance = haversine(
                    location,
                    (station['location']['lat'], station['location']['lng'])
                )
                station_info["distance_km"] = round(distance, 2)
                
                # Get detailed info for operational status
                if not station.get('is_custom_station', False):
                    # Google Maps station - fetch additional details
                    try:
                        details = gmaps.place(place_id=station['place_id']).get('result', {})
                        station_info["open_now"] = details.get('opening_hours', {}).get('open_now')
                        station_info["phone"] = details.get('formatted_phone_number')
                    except Exception as e:
                        logging.warning(f"Failed to get details for station {station['name']}: {e}")
                        station_info["open_now"] = None
                else:
                    # Custom station - use provided data
                    station_info["open_now"] = station.get('open_now', True)
                    station_info["phone"] = None  # Custom stations don't have phone from Google
                    # Add custom station specific information
                    station_info.update({
                        "station_type": station.get('station_type'),
                        "charger_types": station.get('charger_types', []),
                        "power_output_kw": station.get('power_output_kw'),
                        "number_of_charging_points": station.get('number_of_charging_points', 1),
                        "operating_hours": station.get('operating_hours', '24/7'),
                        "amenities": station.get('amenities', []),
                        "pricing_info": station.get('pricing_info')
                    })
                
                # Get occupancy prediction
                occupancy = predict_station_occupancy(station_info)
                station_info["predicted_occupancy"] = occupancy
                
                # Add user-friendly occupancy information
                if occupancy is not None:
                    if occupancy <= 0.3:
                        station_info["occupancy_level"] = "Low"
                        base_recommendation = "Recommended - low wait time"
                        station_info["priority"] = 1
                    elif occupancy <= 0.7:
                        station_info["occupancy_level"] = "Moderate"
                        base_recommendation = "Available - moderate wait time"
                        station_info["priority"] = 2
                    else:
                        station_info["occupancy_level"] = "High"
                        base_recommendation = "Busy - consider alternatives"
                        station_info["priority"] = 3
                else:
                    station_info["occupancy_level"] = "Unknown"
                    base_recommendation = "Status unavailable"
                    station_info["priority"] = 2
                
                # Enhanced recommendation for custom stations
                if station.get('is_custom_station'):
                    station_info["recommendation"] = f"🏪 Community Station - {base_recommendation}"
                    station_info["station_source"] = "Community Contributed"
                else:
                    station_info["recommendation"] = base_recommendation
                    station_info["station_source"] = "Google Maps"
                
                enhanced_stations.append(station_info)
                
            except Exception as e:
                logging.error(f"Error processing station {station.get('name', 'Unknown')}: {e}")
                continue
        
        # Sort stations by priority (occupancy) and then by distance
        enhanced_stations.sort(key=lambda x: (x.get("priority", 2), x.get("distance_km", 999)))
        
        # Generate summary statistics
        total_stations = len(enhanced_stations)
        available_stations = len([s for s in enhanced_stations if s.get("occupancy_level") in ["Low", "Moderate"]])
        avg_distance = sum(s.get("distance_km", 0) for s in enhanced_stations) / total_stations if total_stations > 0 else 0
        
        return {
            "user_location": {"lat": lat, "lng": lng},
            "search_radius_km": radius / 1000,
            "stations": enhanced_stations,
            "summary": {
                "total_stations_found": total_stations,
                "available_stations": available_stations,
                "busy_stations": total_stations - available_stations,
                "average_distance_km": round(avg_distance, 2),
                "closest_station_km": enhanced_stations[0].get("distance_km") if enhanced_stations else None,
                "search_timestamp": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()
            },
            "model_info": {
                "occupancy_predictions_available": occupancy_model is not None,
                "prediction_accuracy": "High" if occupancy_model else "Fallback model used"
            }
        }
        
    except Exception as e:
        logging.error(f"Failed to get nearby stations: {e}")
        return {
            "error": f"Search failed: {str(e)}",
            "user_location": {"lat": lat, "lng": lng},
            "stations": []
        }

@app.get("/model-info")
def get_model_info():
    """
    Get information about the ML model used for occupancy prediction.
    Useful for monitoring model health and understanding prediction capabilities.
    """
    try:
        model_info = {
            "model_status": {
                "loaded": occupancy_model is not None,
                "model_type": "Random Forest Regressor" if occupancy_model else "Not Available",
                "model_path": MODEL_PATH
            },
            "features": {
                "total_features": occupancy_model.n_features_in_ if occupancy_model else 30,
                "feature_description": [
                    "station_id - Unique identifier for the charging station",
                    "latitude - Station geographical latitude", 
                    "longitude - Station geographical longitude",
                    "hour - Hour of day (0-23)",
                    "day - Day of week (0=Monday, 6=Sunday)",
                    "is_weekend - Weekend flag (0/1)",
                    "temperature - Current temperature in Celsius",
                    "traffic_delay_min - Traffic delay in minutes",
                    "rating - Station rating (1-5)",
                    "user_ratings_total - Number of user ratings",
                    "open_now - Station open status (0/1)",
                    "hour_sin - Sine encoding of hour (cyclical)",
                    "hour_cos - Cosine encoding of hour (cyclical)",
                    "time_period_encoded - Time period category (0-3)",
                    "weekend_pattern_encoded - Weekend pattern category (0-3)",
                    "weather_encoded - Weather condition category (0-4)",
                    "temperature_category_encoded - Temperature category (0-3)",
                    "weather_temp_encoded - Weather-temperature interaction",
                    "temperature_squared - Temperature squared for non-linearity",
                    "traffic_category_encoded - Traffic category (0-2)",
                    "traffic_delay_squared - Traffic delay squared",
                    "rating_category_encoded - Rating category (0-3)",
                    "popularity_category_encoded - Popularity category (0-2)",
                    "city_encoded - City identifier (0-6)",
                    "city_size_encoded - City size category (0-1)",
                    "distance_from_center - Distance from city center",
                    "time_weather_encoded - Time-weather interaction",
                    "quality_location_encoded - Quality-location interaction",
                    "rating_normalized - Normalized rating (0-1)",
                    "temperature_normalized - Normalized temperature (0-1)"
                ] if occupancy_model else []
            },
            "prediction_details": {
                "output_range": "0.0 to 1.0 (probability of being occupied)",
                "update_frequency": "Real-time with current conditions",
                "accuracy_metrics": "R² Score: ~0.42 (based on training data)",
                "confidence_level": "High for urban areas, Moderate for rural areas"
            },
            "data_sources": {
                "weather": f"OpenWeatherMap API (Key: {'***' + OPENWEATHER_API_KEY[-4:] if OPENWEATHER_API_KEY else 'Not configured'})",
                "traffic": "Time-based estimation algorithm",
                "station_data": "Google Places API",
                "historical_patterns": "Trained on Maharashtra EV charging data"
            },
            "api_capabilities": [
                "/predict-occupancy - Single station prediction",
                "/predict-occupancy-bulk - Multiple station predictions",
                "/station-status/{place_id} - Detailed station status",
                "/nearby-stations - Area-based station search with predictions"
            ],
            "last_updated": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()
        }
        
        # Add model performance info if available
        if occupancy_model:
            try:
                # Test prediction to ensure model is working (30 features as required)
                test_features = [[
                    1, 19.0760, 72.8777, 14, 1, 0, 30.0, 2.0, 4.0, 50, 1,  # Basic features (11)
                    0.8, 0.6, 2, 1, 1, 2, 12, 900, 1, 4.0,  # Extended features (10) 
                    3, 2, 0, 0, 0.1, 21, 13, 0.8, 0.5  # Final features (9)
                ]]
                test_prediction = occupancy_model.predict(test_features)[0]
                model_info["health_check"] = {
                    "status": "Healthy",
                    "test_prediction": round(test_prediction, 3),
                    "feature_count_verified": len(test_features[0]),
                    "last_test": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()
                }
            except Exception as e:
                model_info["health_check"] = {
                    "status": "Error",
                    "error": str(e),
                    "last_test": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()
                }
        else:
            model_info["health_check"] = {
                "status": "Model not loaded",
                "fallback_mode": True
            }
            
        return model_info
        
    except Exception as e:
        logging.error(f"Model info endpoint failed: {e}")
        return {
            "error": f"Failed to retrieve model information: {str(e)}",
            "model_status": {"loaded": False, "error": True}
        }

@app.get("/health")
def health_check():
    """Quick health check for the API and ML model"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
            "ml_model": "loaded" if occupancy_model else "not_loaded",
            "apis": {
                "google_maps": "configured" if GOOGLE_MAPS_API_KEY else "not_configured",
                "openweather": "configured" if OPENWEATHER_API_KEY else "not_configured"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()
        }

@app.get("/test-cors")
def test_cors_endpoint():
    """Test endpoint for debugging CORS and browser connectivity issues"""
    return {
        "status": "success",
        "message": "CORS test successful - browser can connect to API",
        "timestamp": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
        "server": "main_api_server",
        "port": 8000,
        "headers_info": "CORS configured for browser access"
    }

@app.post("/predict-occupancy")
def predict_station_occupancy_endpoint(request: OccupancyPredictionRequest):
    """
    Predict occupancy for a single charging station using ML model.
    Provides real-time occupancy prediction based on location, time, weather, and traffic.
    """
    try:
        # Parse prediction time if provided
        prediction_dt = None
        if request.prediction_time:
            try:
                prediction_dt = datetime.fromisoformat(request.prediction_time.replace('Z', '+00:00'))
                # Convert to IST
                prediction_dt = prediction_dt.astimezone(pytz.timezone("Asia/Kolkata"))
            except Exception as e:
                logging.warning(f"Invalid prediction_time format: {e}, using current time")
        
        if prediction_dt is None:
            prediction_dt = datetime.now(pytz.timezone("Asia/Kolkata"))
        
        # Create station object for prediction
        station = {
            "name": request.station_name,
            "location": {"lat": request.latitude, "lng": request.longitude},
            "place_id": request.place_id or f"lat_{request.latitude}_lng_{request.longitude}"
        }
        
        # Get occupancy prediction
        occupancy = predict_station_occupancy(station, prediction_dt)
        
        # Get weather data for additional context
        weather_data = get_weather_data(request.latitude, request.longitude)
        
        # Get traffic estimation
        traffic_level = get_traffic_estimation(request.latitude, request.longitude, prediction_dt)
        
        # Determine occupancy level description
        if occupancy is None:
            occupancy_level = "Unknown"
            recommendation = "Model not available"
        elif occupancy <= 0.3:
            occupancy_level = "Low"
            recommendation = "Good time to charge - low congestion expected"
        elif occupancy <= 0.7:
            occupancy_level = "Moderate"
            recommendation = "Average waiting time expected"
        else:
            occupancy_level = "High"
            recommendation = "Consider alternative stations or wait - high congestion expected"
        
        return {
            "station": {
                "name": request.station_name,
                "latitude": request.latitude,
                "longitude": request.longitude,
                "place_id": request.place_id
            },
            "prediction": {
                "occupancy_probability": occupancy,
                "occupancy_level": occupancy_level,
                "recommendation": recommendation,
                "confidence": "High" if occupancy_model else "Low (fallback model)"
            },
            "context": {
                "prediction_time": prediction_dt.isoformat(),
                "hour": prediction_dt.hour,
                "day_of_week": prediction_dt.strftime("%A"),
                "is_weekend": prediction_dt.weekday() >= 5,
                "weather": {
                    "temperature_celsius": weather_data['temperature'],
                    "condition": weather_data['weather_description']
                },
                "traffic_level": {
                    "level": traffic_level,
                    "description": {1: "Low", 2: "Moderate", 3: "High"}[traffic_level]
                }
            },
            "model_info": {
                "model_available": occupancy_model is not None,
                "model_type": "Random Forest Regressor" if occupancy_model else "Fallback",
                "features_used": [
                    "station_location", "hour", "day_of_week", "weekend_flag",
                    "temperature", "weather_condition", "traffic_level", "historical_patterns"
                ]
            }
        }
        
    except Exception as e:
        logging.error(f"Occupancy prediction endpoint failed: {e}")
        return {
            "error": f"Prediction failed: {str(e)}",
            "station": {
                "name": request.station_name,
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "prediction": {
                "occupancy_probability": 0.5,
                "occupancy_level": "Unknown",
                "recommendation": "Unable to predict - use alternative assessment"
            }
        }

@app.post("/predict-occupancy-bulk")
def predict_bulk_occupancy(request: BulkOccupancyRequest):
    """
    Predict occupancy for multiple charging stations efficiently.
    Useful for real-time dashboard updates and route optimization.
    """
    try:
        # Parse prediction time if provided
        prediction_dt = None
        if request.prediction_time:
            try:
                prediction_dt = datetime.fromisoformat(request.prediction_time.replace('Z', '+00:00'))
                prediction_dt = prediction_dt.astimezone(pytz.timezone("Asia/Kolkata"))
            except Exception as e:
                logging.warning(f"Invalid prediction_time format: {e}, using current time")
        
        if prediction_dt is None:
            prediction_dt = datetime.now(pytz.timezone("Asia/Kolkata"))
        
        predictions = []
        
        for station_data in request.stations:
            try:
                # Validate station data
                if 'lat' not in station_data or 'lng' not in station_data:
                    predictions.append({
                        "station": station_data,
                        "error": "Missing latitude or longitude",
                        "occupancy_probability": None
                    })
                    continue
                
                # Create station object
                station = {
                    "name": station_data.get("name", "Unknown Station"),
                    "location": {"lat": station_data["lat"], "lng": station_data["lng"]},
                    "place_id": station_data.get("place_id", f"lat_{station_data['lat']}_lng_{station_data['lng']}")
                }
                
                # Get prediction
                occupancy = predict_station_occupancy(station, prediction_dt)
                
                # Determine occupancy level
                if occupancy is None:
                    occupancy_level = "Unknown"
                elif occupancy <= 0.3:
                    occupancy_level = "Low"
                elif occupancy <= 0.7:
                    occupancy_level = "Moderate"
                else:
                    occupancy_level = "High"
                
                predictions.append({
                    "station": {
                        "name": station["name"],
                        "latitude": station_data["lat"],
                        "longitude": station_data["lng"],
                        "place_id": station.get("place_id")
                    },
                    "occupancy_probability": occupancy,
                    "occupancy_level": occupancy_level,
                    "prediction_successful": True
                })
                
            except Exception as e:
                logging.error(f"Failed to predict occupancy for station {station_data}: {e}")
                predictions.append({
                    "station": station_data,
                    "error": str(e),
                    "occupancy_probability": 0.5,
                    "occupancy_level": "Unknown",
                    "prediction_successful": False
                })
        
        # Calculate summary statistics
        successful_predictions = [p for p in predictions if p.get("prediction_successful", False)]
        
        if successful_predictions:
            occupancies = [p["occupancy_probability"] for p in successful_predictions if p["occupancy_probability"] is not None]
            avg_occupancy = sum(occupancies) / len(occupancies) if occupancies else 0.5
            
            # Count by level
            levels = [p["occupancy_level"] for p in successful_predictions]
            level_counts = {
                "Low": levels.count("Low"),
                "Moderate": levels.count("Moderate"), 
                "High": levels.count("High"),
                "Unknown": levels.count("Unknown")
            }
        else:
            avg_occupancy = 0.5
            level_counts = {"Low": 0, "Moderate": 0, "High": 0, "Unknown": len(predictions)}
        
        return {
            "predictions": predictions,
            "summary": {
                "total_stations": len(request.stations),
                "successful_predictions": len(successful_predictions),
                "failed_predictions": len(request.stations) - len(successful_predictions),
                "average_occupancy": round(avg_occupancy, 3),
                "occupancy_distribution": level_counts,
                "prediction_time": prediction_dt.isoformat()
            },
            "model_info": {
                "model_available": occupancy_model is not None,
                "model_type": "Random Forest Regressor" if occupancy_model else "Fallback"
            }
        }
        
    except Exception as e:
        logging.error(f"Bulk occupancy prediction failed: {e}")
        return {
            "error": f"Bulk prediction failed: {str(e)}",
            "predictions": [],
            "summary": {
                "total_stations": len(request.stations) if hasattr(request, 'stations') else 0,
                "successful_predictions": 0,
                "failed_predictions": len(request.stations) if hasattr(request, 'stations') else 0
            }
        }

@app.post("/ev-route")
def generate_ev_route(request: EVRouteRequest):
    try:
        # Validate and log the request parameters
        logging.info(f"Route request - Origin: {request.origin}, Destination: {request.destination}")
        
        # Validate origin coordinates format
        if ',' in request.origin:
            try:
                lat, lng = request.origin.split(',')
                lat, lng = float(lat.strip()), float(lng.strip())
                if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                    return {"error": f"Invalid origin coordinates: {request.origin}. Latitude must be between -90 and 90, longitude between -180 and 180."}
            except ValueError:
                return {"error": f"Invalid origin coordinate format: {request.origin}. Expected 'latitude,longitude'"}
        
        # Step 1: Get driving directions with error handling
        try:
            directions = gmaps.directions(
                request.origin,
                request.destination,
                mode="driving"
            )
        except Exception as e:
            logging.error(f"Google Maps API error: {str(e)}")
            if "NOT_FOUND" in str(e):
                return {"error": f"Route not found. Please check if the destination '{request.destination}' is valid and accessible by road from your location."}
            elif "INVALID_REQUEST" in str(e):
                return {"error": f"Invalid request parameters. Origin: '{request.origin}', Destination: '{request.destination}'"}
            elif "OVER_QUERY_LIMIT" in str(e):
                return {"error": "Google Maps API quota exceeded. Please try again later."}
            else:
                return {"error": f"Route planning failed: {str(e)}"}

        if not directions:
            return {"error": f"No route found between '{request.origin}' and '{request.destination}'. Please verify both locations are accessible by road."}

        route_legs = directions[0]['legs'][0]
        steps = route_legs['steps']

        total_route_distance_km = route_legs['distance']['value'] / 1000  # in km
        
        # NEW: Battery health integration
        battery_health_data = None
        effective_range = request.current_range_km
        
        if request.battery_data:
            # If battery data is provided, calculate health-adjusted range
            try:
                battery_request = BatteryHealthRequest(**request.battery_data)
                battery_health_data = predict_battery_health(battery_request)
                effective_range = battery_health_data.get('effective_range_km', request.current_range_km)
                
                # Use the minimum of reported range and health-adjusted range for safety
                effective_range = min(request.current_range_km, effective_range)
                
                logging.info(f"Battery health analysis: SoH {battery_health_data.get('current_state_of_health_percent', 'N/A')}%, "
                            f"Effective range: {effective_range}km vs reported: {request.current_range_km}km")
            except Exception as e:
                logging.warning(f"Battery health analysis failed, using reported range: {e}")
                effective_range = request.current_range_km

    except Exception as e:
        logging.error(f"Unexpected error in generate_ev_route: {str(e)}")
        return {"error": f"Route generation failed: {str(e)}"}

    # Check if current battery can complete the trip
    if total_route_distance_km <= effective_range:
        return {
            "message": "Route is within current EV range, no charging needed.",
            "total_distance_km": total_route_distance_km,
            "current_battery_sufficient": True,
            "charging_stops": [],
            "geofencing_opportunities": [],
            "waypoints": [],  # Add waypoints for route
            "battery_health_analysis": battery_health_data,  # Include battery analysis
            "effective_range_km": effective_range,
            "reported_range_km": request.current_range_km
        }

    # Step 2: Walk through the route step-by-step
    charging_stops = []
    geofencing_opportunities = []
    waypoints = []  # Add waypoints list
    distance_tracker = 0.0
    current_battery_range = effective_range  # Use effective range instead of reported range
    last_charge_location = steps[0]['start_location']
    last_latlng = (last_charge_location['lat'], last_charge_location['lng'])

    for step in steps:
        current_latlng = (step['end_location']['lat'], step['end_location']['lng'])
        segment_distance = haversine(last_latlng, current_latlng)
        distance_tracker += segment_distance

        # If battery will die before next step
        if distance_tracker >= current_battery_range:
            # Find charging station near the last valid point
            charger_location = find_nearest_charger(last_latlng)
            if charger_location:
                # Predict occupancy for original station
                original_occupancy = charger_location.get("predicted_occupancy")
                better_stations = find_better_stations_nearby(
                    charger_location['location'],
                    charger_location.get('rating'),
                    original_occupancy
                )
                
                # DEBUG: Log geofencing opportunity analysis
                logging.info(f"Geofencing analysis for {charger_location.get('name', 'Unknown Station')}:")
                logging.info(f"  Original rating: {charger_location.get('rating')}")
                logging.info(f"  Original occupancy: {original_occupancy}")
                logging.info(f"  Better stations found: {len(better_stations)}")
                
                if better_stations:
                    for idx, station in enumerate(better_stations[:3]):  # Log top 3
                        logging.info(f"    Option {idx+1}: {station['name']} - "
                                   f"Rating: {station['rating']} (+{station['rating_improvement']}), "
                                   f"Distance: {station['distance_from_original']}km, "
                                   f"Occupancy: {station.get('predicted_occupancy', 'N/A')}")

                # --- NEW LOGIC: Calculate required charge percentage ---
                # Calculate remaining distance from this charging stop to destination
                remaining_distance_to_destination = haversine(
                    (charger_location['location']['lat'], charger_location['location']['lng']),
                    (steps[-1]['end_location']['lat'], steps[-1]['end_location']['lng'])
                )
                
                # Factor in battery health for charging recommendations
                if battery_health_data:
                    health_factor = battery_health_data.get('current_state_of_health_percent', 85) / 100.0
                    # Recommend slightly higher charge percentage for degraded batteries
                    safety_margin = 1.3 if health_factor < 0.8 else 1.2
                else:
                    safety_margin = 1.2
                
                total_needed_km = remaining_distance_to_destination * safety_margin
                required_charge_percent = min(100, round((total_needed_km / MAX_VEHICLE_RANGE_KM) * 100))
                
                charging_stop = {
                    **charger_location,
                    "distance_from_start": distance_tracker - segment_distance,
                    "battery_level_on_arrival": f"{max(0, current_battery_range - (distance_tracker - segment_distance)):.1f} km remaining",
                    "stop_index": len(charging_stops),
                    "required_charge_percent": required_charge_percent,
                    "battery_health_adjusted": battery_health_data is not None,
                    "safety_margin_applied": safety_margin
                }
                
                charging_stops.append(charging_stop)
                
                # Add waypoint for route generation
                waypoints.append({
                    "lat": charger_location['location']['lat'],
                    "lng": charger_location['location']['lng']
                })
                
                # If better stations found, add geofencing opportunity
                if better_stations:
                    best_alternative = better_stations[0]
                    geofencing_opportunity = {
                        "original_station": charging_stop,
                        "better_alternative": best_alternative,
                        "improvement_reason": (
                            f"Better rating ({best_alternative['rating']}⭐ vs {charger_location.get('rating', 'N/A')}⭐) "
                            f"and occupancy ({best_alternative.get('predicted_occupancy', 'N/A')} vs {original_occupancy})"
                        ),
                        "distance_detour": best_alternative['distance_from_original'],
                        "better_predicted_occupancy": best_alternative.get('predicted_occupancy')
                    }
                    geofencing_opportunities.append(geofencing_opportunity)
                    
                    # DEBUG: Log the geofencing opportunity creation
                    logging.info(f"✅ GEOFENCING OPPORTUNITY CREATED:")
                    logging.info(f"  Original: {charger_location.get('name')} (Rating: {charger_location.get('rating')}, Occupancy: {original_occupancy})")
                    logging.info(f"  Better: {best_alternative['name']} (Rating: {best_alternative['rating']}, Occupancy: {best_alternative.get('predicted_occupancy')}, Distance: +{best_alternative['distance_from_original']}km)")
                else:
                    logging.info(f"❌ No better stations found for geofencing opportunity")
                
                # Reset battery to full capacity after charging
                current_battery_range = MAX_VEHICLE_RANGE_KM
                distance_tracker = segment_distance
                last_latlng = (charger_location['location']['lat'], charger_location['location']['lng'])
            else:
                return {
                    "error": "No charging station found before battery depletion.",
                    "last_known_location": last_latlng,
                    "distance_traveled": distance_tracker - segment_distance
                }

        last_latlng = current_latlng

    # DEBUG: Log final route generation results
    logging.info(f"🔋 ROUTE GENERATION COMPLETE:")
    logging.info(f"  Total distance: {total_route_distance_km:.2f}km")
    logging.info(f"  Charging stops: {len(charging_stops)}")
    logging.info(f"  Geofencing opportunities: {len(geofencing_opportunities)}")
    
    if geofencing_opportunities:
        logging.info(f"📍 GEOFENCING OPPORTUNITIES TO FRONTEND:")
        for idx, opp in enumerate(geofencing_opportunities):
            logging.info(f"  {idx+1}. {opp['original_station']['name']} → {opp['better_alternative']['name']} (+{opp['distance_detour']}km)")

    return {
        "origin": request.origin,
        "destination": request.destination,
        "total_distance_km": total_route_distance_km,
        "initial_range_km": request.current_range_km,
        "effective_range_km": effective_range,  # NEW: Include effective range
        "max_vehicle_range_km": MAX_VEHICLE_RANGE_KM,
        "charging_stops": charging_stops,
        "geofencing_opportunities": geofencing_opportunities,
        "total_charging_stops": len(charging_stops),
        "waypoints": waypoints,  # Include waypoints in response
        "battery_health_analysis": battery_health_data,  # NEW: Include battery analysis
        "range_safety_applied": effective_range < request.current_range_km  # NEW: Flag if range was adjusted
    }

@app.post("/car-breakdown")
def handle_car_breakdown(request: BreakdownRequest):
    """
    IMPROVED: Handle car breakdown with better error handling, validation, and logging
    """
    try:
        # Validate coordinates
        if not (-90 <= request.latitude <= 90) or not (-180 <= request.longitude <= 180):
            logging.warning(f"Invalid coordinates provided: {request.latitude}, {request.longitude}")
            return {
                "error": "Invalid coordinates provided. Latitude must be between -90 and 90, longitude between -180 and 180.",
                "user_location": {"lat": request.latitude, "lng": request.longitude},
                "garages": [],
                "tow_services": [],
                "total_services": 0
            }
        
        location = (request.latitude, request.longitude)
        logging.info(f"Processing car breakdown request at: {request.latitude}, {request.longitude}")
        
        # Find nearby garages with improved search
        garages = find_nearby_services(location, "car repair", "car_repair", max_results=8)
        
        # Find nearby tow services with improved search
        tow_services = find_nearby_services(location, "towing service", "towing", max_results=7)
        
        # Calculate and log statistics for monitoring
        all_services = garages + tow_services
        total_services = len(all_services)
        
        if all_services:
            distances = [s['distance_km'] for s in all_services]
            min_distance = min(distances)
            max_distance = max(distances)
            avg_distance = sum(distances) / len(distances)
            
            logging.info(f"Breakdown service statistics:")
            logging.info(f"  Total services found: {total_services}")
            logging.info(f"  Garages: {len(garages)}, Towing: {len(tow_services)}")
            logging.info(f"  Distance range: {min_distance:.2f}km - {max_distance:.2f}km")
            logging.info(f"  Average distance: {avg_distance:.2f}km")
            
            # Flag potential issues for monitoring
            if min_distance > 15:
                logging.warning(f"Closest service is {min_distance:.2f}km away - potentially too far")
            if avg_distance > 20:
                logging.warning(f"Average service distance is {avg_distance:.2f}km - services seem distant")
        else:
            logging.warning(f"No breakdown services found near {request.latitude}, {request.longitude}")
        
        return {
            "user_location": {
                "lat": request.latitude,
                "lng": request.longitude
            },
            "garages": garages,
            "tow_services": tow_services,
            "total_services": total_services,
            "search_metadata": {
                "search_strategy": "multi_keyword_fallback",
                "max_search_radius_km": 25,
                "distance_sorted": True,
                "quality_filtered": True,
                "garages_searched": len(garages),
                "towing_searched": len(tow_services)
            }
        }
        
    except Exception as e:
        logging.error(f"Car breakdown handler failed: {e}")
        return {
            "error": f"Failed to find breakdown services: {str(e)}",
            "user_location": {"lat": request.latitude, "lng": request.longitude},
            "garages": [],
            "tow_services": [],
            "total_services": 0
        }

# Helper functions for Producer Analysis
def load_geojson_data(file_path):
    """Load and parse GeoJSON data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load GeoJSON from {file_path}: {e}")
        return None

def get_state_boundary(state_name):
    """Get the boundary polygon for a specific state"""
    try:
        states_data = load_geojson_data("india_state.geojson")
        if not states_data:
            return None
            
        for feature in states_data['features']:
            if feature['properties']['NAME_1'].lower() == state_name.lower():
                return shape(feature['geometry'])
        return None
    except Exception as e:
        logging.error(f"Error getting state boundary: {e}")
        return None

def get_district_boundary(state_name, district_name):
    """Get the boundary polygon for a specific district within a state"""
    try:
        districts_data = load_geojson_data("india_district.geojson")
        if not districts_data:
            return None
            
        for feature in districts_data['features']:
            if (feature['properties']['NAME_1'].lower() == state_name.lower() and 
                feature['properties']['NAME_2'].lower() == district_name.lower()):
                return shape(feature['geometry'])
        return None
    except Exception as e:
        logging.error(f"Error getting district boundary: {e}")
        return None

def get_districts_in_state(state_name):
    """Get all districts within a state"""
    try:
        districts_data = load_geojson_data("india_district.geojson")
        if not districts_data:
            return []
            
        state_districts = []
        for feature in districts_data['features']:
            if feature['properties']['NAME_1'].lower() == state_name.lower():
                district_info = {
                    'name': feature['properties']['NAME_2'],
                    'geometry': shape(feature['geometry']),
                    'bounds': shape(feature['geometry']).bounds
                }
                state_districts.append(district_info)
        return state_districts
    except Exception as e:
        logging.error(f"Error getting districts: {e}")
        return []

def get_specific_district_info(state_name, district_name):
    """Get information for a specific district within a state"""
    try:
        districts_data = load_geojson_data("india_district.geojson")
        if not districts_data:
            return None
            
        for feature in districts_data['features']:
            if (feature['properties']['NAME_1'].lower() == state_name.lower() and 
                feature['properties']['NAME_2'].lower() == district_name.lower()):
                district_info = {
                    'name': feature['properties']['NAME_2'],
                    'state': feature['properties']['NAME_1'],
                    'geometry': shape(feature['geometry']),
                    'bounds': shape(feature['geometry']).bounds
                }
                return district_info
        return None
    except Exception as e:
        logging.error(f"Error getting specific district info: {e}")
        return None

def find_charging_stations_in_bounds_concurrent(bounds, max_results=None, fast_mode=True, boundary_filter=None):
    """Enhanced concurrent charging station search with optional boundary filtering for district analysis"""
    try:
        # bounds format: (min_lon, min_lat, max_lon, max_lat)
        min_lon, min_lat, max_lon, max_lat = bounds
        
        area_width = max_lon - min_lon
        area_height = max_lat - min_lat
        area_size = area_width * area_height
        
        # Enhanced search grid strategy - more precise for district analysis
        if boundary_filter:
            # For district analysis, use more search points for better coverage
            if area_size > 2.0:  # Very large districts
                grid_size = 5  # 5x5 = 25 search points for precise district coverage
            elif area_size > 0.5:  # Large districts
                grid_size = 4  # 4x4 = 16 search points
            elif area_size > 0.1:  # Medium districts
                grid_size = 3  # 3x3 = 9 search points
            else:  # Small districts
                grid_size = 3  # 3x3 for precise coverage even in small areas
        else:
            # Original state-level search grid
            if area_size > 2.0:  # Very large districts (e.g., Rajasthan, Maharashtra districts)
                grid_size = 4  # 4x4 = 16 search points
            elif area_size > 0.5:  # Large districts (most state districts)
                grid_size = 3  # 3x3 = 9 search points
            elif area_size > 0.1:  # Medium districts (urban areas, small states)
                grid_size = 3  # 3x3 = 9 search points (uniform approach)
            else:  # Small districts/areas
                grid_size = 2  # 2x2 = 4 search points
        
        # Generate enhanced search grid
        search_centers = []
        for i in range(grid_size):
            for j in range(grid_size):
                lat = min_lat + (i + 0.5) * area_height / grid_size
                lng = min_lon + (j + 0.5) * area_width / grid_size
                
                # For district analysis, ensure search points are within boundary
                if boundary_filter:
                    search_point = Point(lng, lat)
                    if boundary_filter.contains(search_point) or boundary_filter.touches(search_point):
                        search_centers.append((lat, lng))
                else:
                    search_centers.append((lat, lng))
        
        # Add center point for additional coverage
        center_lat = (min_lat + max_lat) / 2
        center_lng = (min_lon + max_lon) / 2
        
        if boundary_filter:
            center_point = Point(center_lng, center_lat)
            if boundary_filter.contains(center_point) or boundary_filter.touches(center_point):
                search_centers.append((center_lat, center_lng))
        else:
            search_centers.append((center_lat, center_lng))
        
        logging.info(f"Using {len(search_centers)} search points for {'district' if boundary_filter else 'state'} analysis")
        
        def search_single_location(search_point):
            """Enhanced search function with boundary filtering"""
            search_lat, search_lng = search_point
            local_stations = []
            
            try:
                # Dynamic radius - smaller for district analysis for precision
                if boundary_filter:
                    # More precise radii for district analysis
                    if area_size > 2.0:
                        radius = 20000  # 20km for very large districts
                    elif area_size > 0.5:
                        radius = 15000  # 15km for large districts
                    elif area_size > 0.1:
                        radius = 10000  # 10km for medium districts
                    else:
                        radius = 7000   # 7km for small districts
                else:
                    # Original radii for state analysis
                    if area_size > 2.0:
                        radius = 25000  # 25km for very large areas
                    elif area_size > 0.5:
                        radius = 18000  # 18km for large areas
                    elif area_size > 0.1:
                        radius = 12000  # 12km for medium areas
                    else:
                        radius = 8000   # 8km for small areas
                
                # Enhanced search strategies for better coverage
                search_strategies = [
                    # Primary charging infrastructure
                    {"keyword": "EV charging", "type": "charging_station"},
                    {"keyword": "electric vehicle charging", "type": "charging_station"},
                    {"keyword": "charging station", "type": "charging_station"},
                    
                    # Major Indian charging networks
                    {"keyword": "tata power charging", "type": "charging_station"},
                    {"keyword": "statiq charging", "type": "charging_station"},
                    {"keyword": "ather charging", "type": "charging_station"},
                    {"keyword": "magenta charging", "type": "charging_station"},
                    
                    # Additional charging types
                    {"keyword": "fast charging", "type": "charging_station"},
                    {"keyword": "DC charging", "type": "charging_station"}
                ]
                
                for strategy in search_strategies:
                    # Use semaphore to limit concurrent API calls
                    with API_SEMAPHORE:
                        try:
                            # Add small delay to prevent connection pool exhaustion
                            time.sleep(API_DELAY)
                            
                            # Single API call per strategy
                            response = gmaps.places_nearby(
                                location=(search_lat, search_lng),
                                radius=radius,
                                **strategy
                            )
                            
                            # Process results with optional boundary filtering
                            for place in response.get('results', []):
                                station_lat = place['geometry']['location']['lat']
                                station_lng = place['geometry']['location']['lng']
                                
                                # Apply boundary filter if provided (for district analysis)
                                if boundary_filter:
                                    station_point = Point(station_lng, station_lat)
                                    if not (boundary_filter.contains(station_point) or boundary_filter.touches(station_point)):
                                        continue  # Skip stations outside district boundary
                                
                                station = {
                                    'name': place['name'],
                                    'location': {
                                        'lat': station_lat,
                                        'lng': station_lng
                                    },
                                    'place_id': place['place_id'],
                                    'rating': place.get('rating', 0),
                                    'address': place.get('vicinity', ''),
                                    'operational_status': place.get('business_status', 'OPERATIONAL')
                                }
                                local_stations.append(station)
                            
                            # Get next page if available (with boundary filtering)
                            next_page_token = response.get('next_page_token')
                            if next_page_token:
                                time.sleep(2.5)  # Delay for token activation
                                try:
                                    response = gmaps.places_nearby(page_token=next_page_token)
                                    for place in response.get('results', []):
                                        station_lat = place['geometry']['location']['lat']
                                        station_lng = place['geometry']['location']['lng']
                                        
                                        # Apply boundary filter if provided
                                        if boundary_filter:
                                            station_point = Point(station_lng, station_lat)
                                            if not (boundary_filter.contains(station_point) or boundary_filter.touches(station_point)):
                                                continue
                                        
                                        station = {
                                            'name': place['name'],
                                            'location': {
                                                'lat': station_lat,
                                                'lng': station_lng
                                            },
                                            'place_id': place['place_id'],
                                            'rating': place.get('rating', 0),
                                            'address': place.get('vicinity', ''),
                                            'operational_status': place.get('business_status', 'OPERATIONAL')
                                        }
                                        local_stations.append(station)
                                except Exception as page_error:
                                    logging.warning(f"Error getting pagination: {page_error}")
                        
                        except Exception as strategy_error:
                            logging.warning(f"Error with search strategy {strategy}: {strategy_error}")
                            time.sleep(0.5)
                            continue
                
                return local_stations
                
            except Exception as e:
                logging.error(f"Error in location search: {e}")
                return []
        
        # Execute concurrent searches with optimized threading
        all_stations = []
        seen_place_ids = set()
        
        # Optimize thread count for district vs state analysis
        max_workers = min(MAX_THREADS, len(search_centers), 6 if boundary_filter else MAX_THREADS)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all search tasks
            future_to_point = {executor.submit(search_single_location, point): point for point in search_centers}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_point):
                try:
                    stations = future.result()
                    for station in stations:
                        place_id = station['place_id']
                        if place_id not in seen_place_ids:
                            all_stations.append(station)
                            seen_place_ids.add(place_id)
                except Exception as e:
                    logging.error(f"Error processing search result: {e}")
        
        # Apply result limit if specified
        if max_results:
            all_stations = all_stations[:max_results]
            
        logging.info(f"Found {len(all_stations)} unique charging stations")
        return all_stations
        
    except Exception as e:
        logging.error(f"Error in concurrent charging station search: {e}")
        return []

def generate_analysis_grid_concurrent(state_boundary, analysis_type="urban", fast_mode=True):
    """Generate grid points with concurrent processing for speed"""
    try:
        # Ultra-fast grid generation for sub-10-second target
        spacing_map = {
            "urban": 0.05,      # ~5.5km (optimized for speed while maintaining coverage)
            "highway": 0.4,     # ~45km  
            "heavy_duty": 1.3   # ~145km
        }
        
        step = spacing_map.get(analysis_type, 0.05)
        bounds = state_boundary.bounds
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Dynamic adjustment for very large states
        expected_points = ((max_lat - min_lat) / step) * ((max_lon - min_lon) / step)
        if expected_points > 1500:  # Reduced from 2000 for faster processing
            step = step * (expected_points / 1500) ** 0.5
        
        logging.info(f"Generating concurrent grid for bounds: {bounds} with step: {step}")
        
        # Generate grid points in parallel chunks
        grid_points = []
        max_points = 1500  # Hard limit for sub-10-second performance
        
        def generate_chunk(lat_range):
            """Generate grid points for a latitude range"""
            chunk_points = []
            lat_start, lat_end = lat_range
            
            lat = lat_start
            while lat <= lat_end and len(chunk_points) < max_points // 4:
                lon = min_lon
                while lon <= max_lon and len(chunk_points) < max_points // 4:
                    point = Point(lon, lat)
                    if state_boundary.contains(point) or state_boundary.touches(point):
                        chunk_points.append((lat, lon))
                    lon += step
                lat += step
            return chunk_points
        
        # Divide latitude range into chunks for parallel processing
        lat_ranges = [
            (min_lat, min_lat + (max_lat - min_lat) * 0.25),
            (min_lat + (max_lat - min_lat) * 0.25, min_lat + (max_lat - min_lat) * 0.5),
            (min_lat + (max_lat - min_lat) * 0.5, min_lat + (max_lat - min_lat) * 0.75),
            (min_lat + (max_lat - min_lat) * 0.75, max_lat)
        ]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            chunk_futures = [executor.submit(generate_chunk, lat_range) for lat_range in lat_ranges]
            
            for future in concurrent.futures.as_completed(chunk_futures):
                chunk_points = future.result()
                grid_points.extend(chunk_points)
                if len(grid_points) >= max_points:
                    break
        
        # Limit to max_points for performance
        grid_points = grid_points[:max_points]
        
        logging.info(f"Generated {len(grid_points)} grid points using concurrent processing")
        return grid_points
        
    except Exception as e:
        logging.error(f"Error in concurrent grid generation: {e}")
        return []

def find_underserved_areas_concurrent(grid_points, existing_stations, min_distance_km=3, fast_mode=True):
    """Concurrent processing of underserved areas for maximum speed"""
    try:
        total_points = len(grid_points)
        
        # Pre-calculate station coordinates for faster distance computation
        station_coords = [(station['location']['lat'], station['location']['lng']) for station in existing_stations]
        
        logging.info(f"Concurrent analysis of {total_points} grid points against {len(existing_stations)} stations")
        
        def process_point_batch(point_batch):
            """Process a batch of points in parallel"""
            batch_opportunities = []
            
            for point in point_batch:
                lat, lng = point
                
                # Fast distance calculation - find minimum distance to any station
                min_distance = float('inf')
                for station_coord in station_coords:
                    # Use Manhattan distance for speed (approximation)
                    distance = abs(lat - station_coord[0]) + abs(lng - station_coord[1])
                    distance = distance * 111.32  # Convert to km
                    
                    if distance < min_distance:
                        min_distance = distance
                        # Early exit if within range
                        if distance <= min_distance_km:
                            break
                
                # If underserved, add to opportunities
                if min_distance > min_distance_km:
                    # Simplified opportunity score for speed
                    opportunity_score = min(10, max(1, (min_distance - min_distance_km) * 1.5 + 5))
                    
                    batch_opportunities.append({
                        'location': {'lat': lat, 'lng': lng},
                        'nearest_station_distance_km': round(min_distance, 2),
                        'opportunity_score': round(opportunity_score, 1),
                        'area_type': 'underserved',
                        'government_compliance': 'helps_achieve_mandate'
                    })
            
            return batch_opportunities
        
        # Divide points into batches for concurrent processing
        batch_size = max(50, total_points // MAX_THREADS)
        point_batches = [grid_points[i:i + batch_size] for i in range(0, total_points, batch_size)]
        
        all_opportunities = []
        
        # Process batches concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            batch_futures = [executor.submit(process_point_batch, batch) for batch in point_batches]
            
            for future in concurrent.futures.as_completed(batch_futures):
                batch_opportunities = future.result()
                all_opportunities.extend(batch_opportunities)
        
        # Sort by opportunity score and limit results
        all_opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        # Return top opportunities (small red circles like Delhi)
        max_opportunities = 60  # Optimized for clean visualization like Delhi
        result = all_opportunities[:max_opportunities]
        
        logging.info(f"Concurrent processing found {len(result)} top opportunities")
        return result
        
    except Exception as e:
        logging.error(f"Error in concurrent underserved area analysis: {e}")
        return []

def calculate_opportunity_score(point, existing_stations, min_distance, fast_mode=True):
    """Calculate business opportunity score for a location with performance optimization"""
    try:
        lat, lng = point
        
        # Base score from distance to nearest competitor (0-5 points)
        distance_score = min(5, (min_distance - 3) / 2)  # 3km+ gives score, max at 7km+
        distance_score = max(0, distance_score)
        
        # Accessibility factor based on road network proximity (simplified)
        accessibility_score = 2.0  # Default moderate accessibility
        
        # Population density factor (simplified using major city proximity)
        population_score = 1.0  # Default
        
        if fast_mode:
            # FAST MODE: Use simplified city proximity calculation
            # Pre-calculated major cities for faster lookup
            major_cities = [
                (28.6139, 77.2090),  # Delhi
                (19.0760, 72.8777),  # Mumbai  
                (12.9716, 77.5946),  # Bangalore
                (13.0827, 80.2707),  # Chennai
                (22.5726, 88.3639),  # Kolkata
                (26.9124, 75.7873),  # Jaipur
                (23.0225, 72.5714),  # Ahmedabad
                (17.3850, 78.4867),  # Hyderabad
                (18.5204, 73.8567),  # Pune
                (15.2993, 74.1240),  # Goa
            ]
            
            # Find distance to nearest major city (simplified calculation)
            min_city_distance = min(abs(lat - city[0]) + abs(lng - city[1]) for city in major_cities) * 111.32
        else:
            # THOROUGH MODE: Use accurate haversine distance
            major_cities = [
                (28.6139, 77.2090),  # Delhi
                (19.0760, 72.8777),  # Mumbai  
                (12.9716, 77.5946),  # Bangalore
                (13.0827, 80.2707),  # Chennai
                (22.5726, 88.3639),  # Kolkata
                (26.9124, 75.7873),  # Jaipur
                (23.0225, 72.5714),  # Ahmedabad
                (17.3850, 78.4867),  # Hyderabad
                (18.5204, 73.8567),  # Pune
                (15.2993, 74.1240),  # Goa
                (9.9312, 76.2673),   # Kochi
                (34.0837, 74.7973),  # Srinagar
            ]
            min_city_distance = min(haversine(point, city) for city in major_cities)
        
        # Urban proximity bonus (closer to cities = higher score)
        if min_city_distance < 50:  # Within 50km of major city
            population_score = 2.5 - (min_city_distance / 50)
        elif min_city_distance < 100:  # 50-100km from city
            population_score = 1.5
        else:  # Rural areas
            population_score = 1.0
            
        # Competition density factor (simplified for speed)
        if fast_mode:
            # Count nearby competitors using simplified distance
            nearby_competitors = sum(1 for station in existing_stations 
                                   if abs(lat - station['location']['lat']) + abs(lng - station['location']['lng']) < 0.135)  # ~15km
        else:
            # Use accurate distance calculation
            nearby_competitors = sum(1 for station in existing_stations 
                                   if haversine(point, (station['location']['lat'], station['location']['lng'])) < 15)
        
        competition_factor = max(0.3, 1.2 - (nearby_competitors * 0.15))
        
        # Government mandate urgency factor
        mandate_factor = 1.5 if min_distance > 5 else 1.0  # Higher urgency for very underserved areas
        
        # Calculate final score (max 10)
        final_score = (distance_score + accessibility_score + population_score) * competition_factor * mandate_factor
        final_score = min(10, max(1, final_score))  # Ensure score is between 1-10
        
        return round(final_score, 1)
        
    except Exception as e:
        logging.error(f"Error calculating opportunity score: {e}")
        return 5.0  # Default score

@app.post("/analyze-charging-gaps")
def analyze_charging_gaps(request: ProducerAnalysisRequest):
    """Enhanced charging infrastructure analysis with district-level support and optimized multithreading"""
    try:
        state_name = request.state_name
        district_name = request.district_name
        analysis_type = request.analysis_type
        fast_mode = request.fast_mode
        
        start_time = time.time()
        
        # Determine analysis scope
        if district_name:
            logging.info(f"Starting district-level analysis for {district_name}, {state_name} - {analysis_type}")
            analysis_scope = "district"
        else:
            logging.info(f"Starting state-level analysis for {state_name} - {analysis_type}")
            analysis_scope = "state"
        
        # Get appropriate boundary based on analysis scope
        if analysis_scope == "district":
            # District-level analysis
            target_boundary = get_district_boundary(state_name, district_name)
            if not target_boundary:
                return {
                    "error": f"District '{district_name}' not found in state '{state_name}'",
                    "available_districts": "Use /get-districts-by-state/{state_name} to get available districts"
                }
            
            # For district analysis, we focus only on the selected district
            district_info = get_specific_district_info(state_name, district_name)
            if not district_info:
                return {"error": f"Could not get information for district '{district_name}' in state '{state_name}'"}
            
            districts_to_process = [district_info]
            target_name = f"{district_name}, {state_name}"
            
        else:
            # State-level analysis (existing logic)
            target_boundary = get_state_boundary(state_name)
            if not target_boundary:
                return {
                    "error": f"State '{state_name}' not found in database",
                    "available_states": ["Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat", "Rajasthan", "Uttar Pradesh", "Delhi", "West Bengal", "Madhya Pradesh", "Andhra Pradesh"]
                }
            
            # Get all districts in the state
            districts_to_process = get_districts_in_state(state_name)
            if not districts_to_process:
                return {"error": f"No districts found for state '{state_name}'"}
            
            target_name = state_name
        
        logging.info(f"Processing {len(districts_to_process)} district(s) for {analysis_scope}-level analysis")
        
        # ENHANCED CONCURRENT DISTRICT PROCESSING
        def process_district_concurrent(district):
            """Process a single district with concurrent station search and boundary filtering"""
            try:
                district_start = time.time()
                logging.info(f"Processing district: {district['name']}")
                
                # For district-level analysis, we want more precise boundary filtering
                if analysis_scope == "district":
                    # Use more precise search for district analysis
                    stations = find_charging_stations_in_bounds_concurrent(
                        district['bounds'], 
                        max_results=None, 
                        fast_mode=fast_mode,
                        boundary_filter=district['geometry']  # Add boundary filtering
                    )
                else:
                    # Use existing logic for state analysis
                    stations = find_charging_stations_in_bounds_concurrent(
                        district['bounds'], 
                        max_results=None, 
                        fast_mode=fast_mode
                    )
                
                district_time = time.time() - district_start
                logging.info(f"District {district['name']} completed in {district_time:.2f}s with {len(stations)} stations")
                
                return stations
                
            except Exception as e:
                logging.error(f"Error processing district {district['name']}: {e}")
                return []
        
        # Execute district processing with optimized threading
        all_stations = []
        max_workers = min(MAX_THREADS, len(districts_to_process), 4)  # Optimize for district analysis
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all district processing tasks
            district_futures = [executor.submit(process_district_concurrent, district) for district in districts_to_process]
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(district_futures):
                try:
                    district_stations = future.result()
                    all_stations.extend(district_stations)
                except Exception as e:
                    logging.error(f"Error collecting district results: {e}")
        
        # Fast deduplication using set for O(1) lookups
        unique_stations = []
        seen_place_ids = set()
        
        for station in all_stations:
            place_id = station['place_id']
            if place_id not in seen_place_ids:
                # Additional boundary check for district analysis
                if analysis_scope == "district":
                    station_point = Point(station['location']['lng'], station['location']['lat'])
                    if target_boundary.contains(station_point):
                        unique_stations.append(station)
                        seen_place_ids.add(place_id)
                else:
                    unique_stations.append(station)
                    seen_place_ids.add(place_id)
        
        logging.info(f"Found {len(unique_stations)} unique stations within {analysis_scope} boundary")
        
        # ENHANCED CONCURRENT GRID GENERATION
        # Generate grid points within the target boundary (state or district)
        grid_points = generate_analysis_grid_concurrent(target_boundary, analysis_type, fast_mode)
        
        # CONCURRENT OPPORTUNITY ANALYSIS
        min_distance_map = {
            "urban": 3,      # 3km for urban areas
            "highway": 25,   # 25km for highways
            "heavy_duty": 100  # 100km for heavy duty
        }
        min_distance = min_distance_map.get(analysis_type, 3)
        
        opportunities = find_underserved_areas_concurrent(grid_points, unique_stations, min_distance, fast_mode)
        
        # Calculate coverage statistics
        total_area_km2 = target_boundary.area * 111.32 * 111.32
        current_coverage_ratio = len(unique_stations) / total_area_km2 * 1000 if total_area_km2 > 0 else 0
        
        # Government mandate compliance calculation
        required_stations_urban = total_area_km2 / (3 * 3) if analysis_type == "urban" else 0
        compliance_percentage = min(100, (len(unique_stations) / required_stations_urban * 100)) if required_stations_urban > 0 else 100
        
        analysis_time = time.time() - start_time
        logging.info(f"{analysis_scope.title()}-level analysis completed in {analysis_time:.2f} seconds")
        
        # Enhanced response with district information
        response_data = {
            "state": state_name,
            "district": district_name if district_name else None,
            "analysis_scope": analysis_scope,
            "target_area": target_name,
            "analysis_type": analysis_type,
            "analysis_mode": f"ENHANCED {analysis_scope.upper()} CONCURRENT",
            "analysis_time_seconds": round(analysis_time, 2),
            "existing_stations": unique_stations,
            "opportunities": opportunities,
            "coverage_statistics": {
                "total_area_analyzed_km2": round(total_area_km2, 2),
                "existing_station_count": len(unique_stations),
                "opportunity_locations": len(opportunities),
                "current_station_density_per_1000km2": round(current_coverage_ratio, 2),
                "government_mandate_compliance_percent": round(compliance_percentage, 1),
                "min_distance_requirement_km": min_distance,
                "areas_processed": len(districts_to_process),
                "processing_method": f"Enhanced concurrent multithreading for {analysis_scope} analysis"
            },
            "government_mandates": {
                "urban_areas": "EV charging station every 3 km",
                "highways": "EV charging station every 25 km on both sides",
                "heavy_duty_routes": "Long-range chargers every 100 km"
            },
            "visualization_note": f"Business opportunities displayed for {target_name} with precise boundary analysis",
            "performance_note": f"Enhanced {analysis_scope} analysis completed in {analysis_time:.2f} seconds"
        }
        
        # Add district-specific information if applicable
        if analysis_scope == "district":
            response_data["district_info"] = {
                "name": district_name,
                "state": state_name,
                "boundary_coordinates": list(target_boundary.exterior.coords) if hasattr(target_boundary, 'exterior') else None
            }
        
        return response_data
        
    except Exception as e:
        logging.error(f"Error in production charging gap analysis: {e}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "state": request.state_name,
            "opportunities": []
        }

@app.get("/get-available-states")
def get_available_states():
    """Get list of available states for analysis"""
    try:
        states_data = load_geojson_data("india_state.geojson")
        if not states_data:
            return {"error": "Failed to load states data"}
        
        states = []
        for feature in states_data['features']:
            state_name = feature['properties']['NAME_1']
            states.append({
                "name": state_name,
                "code": feature['properties'].get('ID_1', ''),
                "type": feature['properties'].get('TYPE_1', 'State')
            })
        
        # Sort alphabetically
        states.sort(key=lambda x: x['name'])
        
        return {
            "states": states,
            "total_states": len(states)
        }
        
    except Exception as e:
        logging.error(f"Error getting available states: {e}")
        return {
            "error": f"Failed to get states: {str(e)}",
            "states": []
        }

@app.get("/get-districts-by-state/{state_name}")
def get_districts_by_state(state_name: str):
    """Get list of districts for a specific state"""
    try:
        districts_data = load_geojson_data("india_district.geojson")
        if not districts_data:
            return {"error": "Failed to load districts data"}
        
        districts = []
        for feature in districts_data['features']:
            # Match state name (case insensitive)
            if feature['properties']['NAME_1'].lower() == state_name.lower():
                district_name = feature['properties']['NAME_2']
                districts.append({
                    "name": district_name,
                    "state": feature['properties']['NAME_1'],
                    "code": feature['properties'].get('ID_2', ''),
                    "state_code": feature['properties'].get('ID_1', '')
                })
        
        # Sort alphabetically
        districts.sort(key=lambda x: x['name'])
        
        if not districts:
            return {
                "error": f"No districts found for state '{state_name}'",
                "districts": [],
                "state": state_name
            }
        
        return {
            "districts": districts,
            "total_districts": len(districts),
            "state": state_name
        }
        
    except Exception as e:
        logging.error(f"Error getting districts for state {state_name}: {e}")
        return {
            "error": f"Failed to get districts: {str(e)}",
            "districts": [],
            "state": state_name
        }

@app.post("/battery-health-analysis")
def analyze_battery_health(request: BatteryHealthRequest):
    """Enhanced battery health analysis with EV database integration"""
    try:
        result = predict_battery_health(request)
        return result
    except Exception as e:
        logging.error(f"Battery health analysis failed: {e}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "current_state_of_health_percent": 85.0,
            "effective_range_km": MAX_VEHICLE_RANGE_KM * 0.85
        }

@app.get("/available-manufacturers")
def get_available_manufacturers():
    """Get list of all EV manufacturers in the database"""
    try:
        manufacturers = ev_data_loader.get_all_manufacturers()
        return {
            "manufacturers": manufacturers,
            "total_count": len(manufacturers)
        }
    except Exception as e:
        logging.error(f"Error getting manufacturers: {e}")
        return {
            "error": f"Failed to get manufacturers: {str(e)}",
            "manufacturers": []
        }

@app.get("/available-models/{manufacturer}")
def get_available_models(manufacturer: str):
    """Get all available models for a specific manufacturer (case-insensitive)"""
    try:
        models = ev_data_loader.get_manufacturer_models(manufacturer)
        
        # Get additional info for each model
        enhanced_models = []
        if models and ev_data_loader.ev_data is not None:
            df = ev_data_loader.ev_data
            manufacturer_df = df[df['Brand'].str.lower() == manufacturer.lower()]
            
            for model in models:
                model_row = manufacturer_df[manufacturer_df['Model'] == model]
                if not model_row.empty:
                    row = model_row.iloc[0]
                    enhanced_models.append({
                        "name": model,
                        "battery_capacity_kwh": float(row['Battery Capacity Max (kWh)']) if pd.notna(row['Battery Capacity Max (kWh)']) else None,
                        "range_km": float(row['Real-World Range (km)']) if pd.notna(row['Real-World Range (km)']) else float(row['ARAI-Claimed Range Max (km)']) if pd.notna(row['ARAI-Claimed Range Max (km)']) else None,
                        "price_min_lakh": float(row['Ex-Showroom Price Min (₹ Lakh)']) if pd.notna(row['Ex-Showroom Price Min (₹ Lakh)']) else None
                    })
                else:
                    enhanced_models.append({"name": model})
        
        return {
            "manufacturer": manufacturer,
            "models": models if not enhanced_models else enhanced_models,
            "total_count": len(models)
        }
    except Exception as e:
        logging.error(f"Error getting models for {manufacturer}: {e}")
        return {
            "error": f"Failed to get models: {str(e)}",
            "manufacturer": manufacturer,
            "models": []
        }

@app.get("/vehicle-specs/{manufacturer}/{model}")
def get_vehicle_specifications(manufacturer: str, model: str):
    """Get detailed specifications for a specific vehicle"""
    try:
        specs = ev_data_loader.get_vehicle_specs(manufacturer, model)
        return specs
    except Exception as e:
        logging.error(f"Error getting specs for {manufacturer} {model}: {e}")
        return {
            "error": f"Failed to get specifications: {str(e)}",
            "found": False
        }

@app.post("/battery-health-with-autodetect")
def analyze_battery_health_with_autodetect(request: BatteryHealthRequest):
    """
    Enhanced battery health analysis that automatically detects and validates 
    vehicle specifications against the database
    """
    try:
        # Force auto-detection
        request.auto_detect_specs = True
        
        # Get vehicle specs first
        vehicle_specs = ev_data_loader.get_vehicle_specs(request.manufacturer, request.model)
        
        # If vehicle found, validate and potentially adjust input parameters
        warnings = []
        if vehicle_specs and vehicle_specs.get('found'):
            # Check battery capacity against database
            actual_min = vehicle_specs.get('battery_capacity_min_kwh')
            actual_max = vehicle_specs.get('battery_capacity_max_kwh')
            
            if actual_min and actual_max:
                avg_capacity = (actual_min + actual_max) / 2
                capacity_diff = abs(request.battery_capacity_kwh - avg_capacity) / avg_capacity
                
                if capacity_diff > 0.15:  # 15% difference threshold
                    warnings.append(f"Input capacity ({request.battery_capacity_kwh}kWh) differs from database average ({avg_capacity:.1f}kWh)")
        
        # Perform analysis
        result = predict_battery_health(request)
        
        # Add validation warnings
        if warnings:
            result['validation_warnings'] = warnings
        
        return result
        
    except Exception as e:
        logging.error(f"Enhanced battery health analysis failed: {e}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "current_state_of_health_percent": 85.0,
            "effective_range_km": MAX_VEHICLE_RANGE_KM * 0.85
        }

@app.get("/battery-form/manufacturers")
def get_manufacturers_for_form():
    """
    Get manufacturers list formatted for frontend dropdown form
    Returns manufacturers with their model count for better UX
    """
    try:
        if ev_data_loader.ev_data is None:
            return {
                "error": "EV database not loaded",
                "manufacturers": []
            }
        
        df = ev_data_loader.ev_data
        
        # Group by manufacturer and count models
        manufacturer_stats = df.groupby('Brand').agg({
            'Model': 'count',
            'Ex-Showroom Price Min (₹ Lakh)': 'min',
            'Battery Capacity Max (kWh)': 'max'
        }).reset_index()
        
        manufacturers = []
        for _, row in manufacturer_stats.iterrows():
            # Handle NaN values properly
            price_min = row['Ex-Showroom Price Min (₹ Lakh)']
            battery_max = row['Battery Capacity Max (kWh)']
            
            manufacturers.append({
                "value": row['Brand'],  # Value to send to backend
                "label": f"{row['Brand']} ({int(row['Model'])} models)",  # Display text
                "model_count": int(row['Model']),
                "price_starts_from": float(price_min) if pd.notna(price_min) else None,
                "max_battery_capacity": float(battery_max) if pd.notna(battery_max) else None
            })
        
        # Sort by popularity (model count) and then alphabetically
        manufacturers.sort(key=lambda x: (-x['model_count'], x['value']))
        
        return {
            "manufacturers": manufacturers,
            "total_manufacturers": len(manufacturers),
            "total_models": int(df.shape[0])
        }
        
    except Exception as e:
        logging.error(f"Error getting manufacturers for form: {e}")
        return {
            "error": f"Failed to get manufacturers: {str(e)}",
            "manufacturers": []
        }

@app.get("/battery-form/models/{manufacturer}")
def get_models_for_form(manufacturer: str):
    """
    Get models for a specific manufacturer formatted for frontend dropdown
    Returns models with key specifications for better UX
    """
    try:
        if ev_data_loader.ev_data is None:
            return {
                "error": "EV database not loaded",
                "models": []
            }
        
        df = ev_data_loader.ev_data
        
        # Filter by manufacturer (case-insensitive)
        manufacturer_models = df[df['Brand'].str.lower() == manufacturer.lower()]
        
        if manufacturer_models.empty:
            return {
                "error": f"No models found for manufacturer: {manufacturer}",
                "models": [],
                "available_manufacturers": sorted(df['Brand'].unique().tolist())
            }
        
        models = []
        for _, row in manufacturer_models.iterrows():
            # Create display label with key specs
            battery_info = ""
            if pd.notna(row['Battery Capacity Max (kWh)']):
                battery_info = f" | {row['Battery Capacity Max (kWh)']}kWh"
            
            range_info = ""
            if pd.notna(row['Real-World Range (km)']):
                range_info = f" | {int(row['Real-World Range (km)'])}km"
            elif pd.notna(row['ARAI-Claimed Range Max (km)']):
                range_info = f" | {int(row['ARAI-Claimed Range Max (km)'])}km*"
            
            price_info = ""
            if pd.notna(row['Ex-Showroom Price Min (₹ Lakh)']):
                price_info = f" | ₹{row['Ex-Showroom Price Min (₹ Lakh)']}L+"
            
            models.append({
                "value": row['Model'],  # Value to send to backend
                "label": f"{row['Model']}{battery_info}{range_info}{price_info}",  # Display text
                "battery_capacity_kwh": float(row['Battery Capacity Max (kWh)']) if pd.notna(row['Battery Capacity Max (kWh)']) else None,
                "real_world_range_km": float(row['Real-World Range (km)']) if pd.notna(row['Real-World Range (km)']) else None,
                "arai_range_km": float(row['ARAI-Claimed Range Max (km)']) if pd.notna(row['ARAI-Claimed Range Max (km)']) else None,
                "price_min_lakh": float(row['Ex-Showroom Price Min (₹ Lakh)']) if pd.notna(row['Ex-Showroom Price Min (₹ Lakh)']) else None,
                "price_max_lakh": float(row['Ex-Showroom Price Max (₹ Lakh)']) if pd.notna(row['Ex-Showroom Price Max (₹ Lakh)']) else None,
                "body_type": row['Body Type'] if pd.notna(row['Body Type']) else "Unknown",
                "dc_charge_time": float(row['DC Charge Time (min)']) if pd.notna(row['DC Charge Time (min)']) else None
            })
        
        # Sort by price (ascending) to show affordable options first
        models.sort(key=lambda x: x['price_min_lakh'] if x['price_min_lakh'] else 999)
        
        return {
            "manufacturer": manufacturer,
            "models": models,
            "total_models": len(models)
        }
        
    except Exception as e:
        logging.error(f"Error getting models for {manufacturer}: {e}")
        return {
            "error": f"Failed to get models for {manufacturer}: {str(e)}",
            "models": []
        }

@app.get("/battery-form/model-specs/{manufacturer}/{model}")
def get_model_specs_for_form(manufacturer: str, model: str):
    """
    Get detailed specifications for a specific model to pre-populate form fields
    This helps auto-fill battery capacity and other fields when user selects a model
    """
    try:
        specs = ev_data_loader.get_vehicle_specs(manufacturer, model)
        
        if not specs or not specs.get('found'):
            return {
                "error": f"Specifications not found for {manufacturer} {model}",
                "found": False
            }
        
        # Return form-friendly data
        return {
            "found": True,
            "manufacturer": specs.get('brand'),
            "model": specs.get('model'),
            "suggested_battery_capacity_kwh": specs.get('battery_capacity_max_kwh'),  # Use max as default
            "battery_capacity_range": {
                "min_kwh": specs.get('battery_capacity_min_kwh'),
                "max_kwh": specs.get('battery_capacity_max_kwh')
            },
            "range_specifications": {
                "real_world_km": specs.get('real_world_range_km'),
                "arai_max_km": specs.get('arai_range_max_km'),
                "arai_min_km": specs.get('arai_range_min_km')
            },
            "charging_info": {
                "dc_charge_time_min": specs.get('dc_charge_time_min')
            },
            "vehicle_info": {
                "body_type": specs.get('body_type'),
                "drivetrain": specs.get('drivetrain'),
                "price_range_lakh": f"₹{specs.get('price_min_lakh')}-{specs.get('price_max_lakh')} Lakh",
                "seating_capacity": specs.get('seating_capacity')
            },
            "form_prefill_suggestions": {
                "battery_capacity_kwh": specs.get('battery_capacity_max_kwh'),
                "manufacturer": specs.get('brand'),
                "model": specs.get('model')
            }
        }
        
    except Exception as e:
        logging.error(f"Error getting model specs for form: {e}")
        return {
            "error": f"Failed to get specifications: {str(e)}",
            "found": False
        }

@app.get("/ev-database-stats")
def get_ev_database_stats():
    """Get statistics about the EV database"""
    try:
        if ev_data_loader.ev_data is None:
            return {"error": "EV database not loaded"}
        
        df = ev_data_loader.ev_data
        
        stats = {
            "total_vehicles": len(df),
            "manufacturers": len(df['Brand'].unique()),
            "body_types": df['Body Type'].unique().tolist(),
            "price_range": {
                "min_lakh": float(df['Ex-Showroom Price Min (₹ Lakh)'].min()),
                "max_lakh": float(df['Ex-Showroom Price Max (₹ Lakh)'].max())
            },
            "battery_capacity_range": {
                "min_kwh": float(df['Battery Capacity Min (kWh)'].min()),
                "max_kwh": float(df['Battery Capacity Max (kWh)'].max())
            },
            "range_specifications": {
                "min_arai_km": float(df['ARAI-Claimed Range Min (km)'].min()),
                "max_arai_km": float(df['ARAI-Claimed Range Max (km)'].max()),
                "avg_real_world_km": float(df['Real-World Range (km)'].mean())
            },
            "manufacturers_list": sorted(df['Brand'].unique().tolist())
        }
        
        return stats
        
    except Exception as e:
        logging.error(f"Error getting database stats: {e}")
        return {
            "error": f"Failed to get database statistics: {str(e)}"
        }

@app.post("/calculate-real-time-range")
def calculate_real_time_range(request: dict):
    """
    Calculate real-time range based on current conditions and vehicle specifications
    """
    try:
        # Extract parameters
        manufacturer = request.get('manufacturer', 'Tata')
        model = request.get('model', 'Nexon EV')
        battery_health_percent = request.get('battery_health_percent', 85.0)
        current_temperature = request.get('temperature', 25.0)
        driving_style = request.get('driving_style', 'moderate')
        
        # Calculate range adjustment
        range_data = calculate_range_adjustment(
            battery_health_percent=battery_health_percent,
            temperature=current_temperature,
            driving_style=driving_style,
            manufacturer=manufacturer,
            model=model
        )
        
        # Add timestamp
        range_data['calculation_timestamp'] = datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()
        
        return range_data
        
    except Exception as e:
        logging.error(f"Real-time range calculation failed: {e}")
        return {
            "error": f"Calculation failed: {str(e)}",
            "adjusted_range_km": MAX_VEHICLE_RANGE_KM * 0.8
        }

# NEW: Custom Station Models and Storage
class OwnerInfo(BaseModel):
    name: str
    email: str
    phone: str
    business_name: Optional[str] = None
    business_type: Optional[str] = "Individual"  # Individual, Business, Hotel, Mall, etc.

class StationSubmission(BaseModel):
    name: str
    latitude: float
    longitude: float
    address: str
    station_type: str  # Public, Private, Semi-Public, Hotel, Mall, Restaurant, Office
    charger_types: List[str]  # AC, DC Fast, Tesla Supercharger, CCS, CHAdeMO
    power_output_kw: Optional[float] = None
    number_of_charging_points: int = 1
    operating_hours: Optional[str] = "24/7"
    amenities: Optional[List[str]] = []  # Restroom, Food, WiFi, Parking, Shopping
    pricing_info: Optional[str] = None
    owner_info: OwnerInfo
    additional_notes: Optional[str] = None
    
    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v
    
    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v
    
    @validator('station_type')
    def validate_station_type(cls, v):
        valid_types = ["Public", "Private", "Semi-Public", "Hotel", "Mall", "Restaurant", "Office", "Gas Station", "Parking Lot", "Highway Rest Stop"]
        if v not in valid_types:
            raise ValueError(f"Station type must be one of: {', '.join(valid_types)}")
        return v
    
    @validator('charger_types')
    def validate_charger_types(cls, v):
        valid_chargers = ["AC Slow (3kW)", "AC Fast (7-22kW)", "DC Fast (25-50kW)", "DC Ultra Fast (50kW+)", "Tesla Supercharger", "CCS", "CHAdeMO", "Type 2"]
        for charger in v:
            if charger not in valid_chargers:
                raise ValueError(f"Invalid charger type: {charger}. Valid types: {', '.join(valid_chargers)}")
        return v

class StationUpdateRequest(BaseModel):
    station_id: str
    name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    station_type: Optional[str] = None
    charger_types: Optional[List[str]] = None
    power_output_kw: Optional[float] = None
    number_of_charging_points: Optional[int] = None
    operating_hours: Optional[str] = None
    amenities: Optional[List[str]] = None
    pricing_info: Optional[str] = None
    additional_notes: Optional[str] = None

# Custom stations storage file
CUSTOM_STATIONS_FILE = "station_owner_listed_stations.json"

def load_custom_stations():
    """Load custom stations from JSON file"""
    try:
        if os.path.exists(CUSTOM_STATIONS_FILE):
            with open(CUSTOM_STATIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        logging.error(f"Error loading custom stations: {e}")
        return []

def save_custom_stations(stations):
    """Save custom stations to JSON file"""
    try:
        with open(CUSTOM_STATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stations, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Error saving custom stations: {e}")
        return False

def convert_custom_station_to_standard_format(custom_station):
    """Convert custom station format to standard station format used throughout the app"""
    return {
        'name': custom_station['name'],
        'location': {
            'lat': custom_station['location']['lat'],
            'lng': custom_station['location']['lng']
        },
        'place_id': f"custom_{custom_station['station_id']}",
        'rating': custom_station.get('rating', 4.0),  # Default rating for custom stations
        'user_ratings_total': custom_station.get('user_ratings_total', 10),  # Default rating count
        'address': custom_station['address'],
        'operational_status': 'OPERATIONAL',
        'station_id': custom_station['station_id'],
        'station_type': custom_station['station_type'],
        'charger_types': custom_station['charger_types'],
        'power_output_kw': custom_station.get('power_output_kw'),
        'number_of_charging_points': custom_station['number_of_charging_points'],
        'operating_hours': custom_station.get('operating_hours', '24/7'),
        'amenities': custom_station.get('amenities', []),
        'pricing_info': custom_station.get('pricing_info'),
        'is_custom_station': True,  # Flag to identify custom stations
        'open_now': True,  # Assume custom stations are open (can be enhanced later)
        'maps_link': f"https://www.google.com/maps/search/?api=1&query={custom_station['location']['lat']},{custom_station['location']['lng']}"
    }

def get_combined_stations_nearby(location, radius=5000, keyword="EV charging"):
    """Get combined results from Google Maps and custom stations"""
    try:
        # Get Google Maps stations
        gmaps_stations = []
        try:
            response = gmaps.places_nearby(
                location=location,
                radius=radius,
                keyword=keyword,
                type="charging_station"
            )
            
            for place in response.get('results', []):
                station = {
                    'name': place['name'],
                    'location': place['geometry']['location'],
                    'place_id': place['place_id'],
                    'rating': place.get('rating', 0),
                    'user_ratings_total': place.get('user_ratings_total', 0),
                    'address': place.get('vicinity', ''),
                    'operational_status': place.get('business_status', 'OPERATIONAL'),
                    'is_custom_station': False
                }
                gmaps_stations.append(station)
        except Exception as e:
            logging.error(f"Error fetching Google Maps stations: {e}")
        
        # Get custom stations
        custom_stations = []
        try:
            all_custom_stations = load_custom_stations()
            for custom_station in all_custom_stations:
                # Calculate distance to filter nearby custom stations
                station_location = (custom_station['location']['lat'], custom_station['location']['lng'])
                distance_km = haversine(location, station_location)
                
                if distance_km <= radius / 1000:  # radius is in meters, convert to km
                    standard_station = convert_custom_station_to_standard_format(custom_station)
                    standard_station['distance_km'] = round(distance_km, 2)
                    custom_stations.append(standard_station)
        except Exception as e:
            logging.error(f"Error fetching custom stations: {e}")
        
        # Combine and deduplicate (in case a custom station matches a Google Maps station)
        all_stations = gmaps_stations + custom_stations
        
        logging.info(f"Found {len(gmaps_stations)} Google Maps stations and {len(custom_stations)} custom stations")
        return all_stations
        
    except Exception as e:
        logging.error(f"Error in combined station search: {e}")
        return []

# NEW: Custom Station CRUD Endpoints

@app.post("/submit-charging-station")
def submit_charging_station(request: StationSubmission):
    """Submit a new charging station by station owner"""
    try:
        # Load existing stations
        stations = load_custom_stations()
        
        # Generate unique station ID
        station_id = str(uuid.uuid4())
        
        # Create new station entry
        new_station = {
            "station_id": station_id,
            "name": request.name,
            "location": {
                "lat": request.latitude,
                "lng": request.longitude
            },
            "address": request.address,
            "station_type": request.station_type,
            "charger_types": request.charger_types,
            "power_output_kw": request.power_output_kw,
            "number_of_charging_points": request.number_of_charging_points,
            "operating_hours": request.operating_hours,
            "amenities": request.amenities,
            "pricing_info": request.pricing_info,
            "owner_info": {
                "name": request.owner_info.name,
                "email": request.owner_info.email,
                "phone": request.owner_info.phone,
                "business_name": request.owner_info.business_name,
                "business_type": request.owner_info.business_type
            },
            "additional_notes": request.additional_notes,
            "submission_date": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat(),
            "status": "active",  # Direct verification as requested
            "rating": 4.0,  # Default rating for new stations
            "user_ratings_total": 1  # Default rating count
        }
        
        # Add to stations list
        stations.append(new_station)
        
        # Save to file
        if save_custom_stations(stations):
            logging.info(f"New station submitted: {request.name} by {request.owner_info.name}")
            return {
                "success": True,
                "message": "Your charging station has been successfully added to our network!",
                "station_id": station_id,
                "station_name": request.name,
                "location": {
                    "latitude": request.latitude,
                    "longitude": request.longitude
                },
                "maps_link": f"https://www.google.com/maps/search/?api=1&query={request.latitude},{request.longitude}",
                "next_steps": [
                    "Your station is now visible to EV users in our route planning",
                    "Users can find your station in nearby searches",
                    "Station will appear in charging gap analysis results",
                    "You can update your station details anytime using your station ID"
                ]
            }
        else:
            return {
                "success": False,
                "error": "Failed to save station data. Please try again."
            }
            
    except Exception as e:
        logging.error(f"Error submitting charging station: {e}")
        return {
            "success": False,
            "error": f"Failed to submit station: {str(e)}"
        }

@app.get("/custom-stations")
def get_all_custom_stations():
    """Get all custom charging stations"""
    try:
        stations = load_custom_stations()
        
        # Convert to standard format for frontend
        formatted_stations = []
        for station in stations:
            formatted_station = convert_custom_station_to_standard_format(station)
            formatted_station.update({
                "submission_date": station.get("submission_date"),
                "owner_business_name": station.get("owner_info", {}).get("business_name"),
                "owner_business_type": station.get("owner_info", {}).get("business_type")
            })
            formatted_stations.append(formatted_station)
        
        return {
            "stations": formatted_stations,
            "total_count": len(formatted_stations),
            "last_updated": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error getting custom stations: {e}")
        return {
            "error": f"Failed to retrieve stations: {str(e)}",
            "stations": []
        }

@app.get("/custom-stations/{station_id}")
def get_custom_station_details(station_id: str):
    """Get details of a specific custom station"""
    try:
        stations = load_custom_stations()
        
        for station in stations:
            if station["station_id"] == station_id:
                # Convert to standard format
                formatted_station = convert_custom_station_to_standard_format(station)
                formatted_station.update({
                    "submission_date": station.get("submission_date"),
                    "owner_info": station.get("owner_info"),
                    "additional_notes": station.get("additional_notes"),
                    "status": station.get("status", "active")
                })
                return formatted_station
        
        return {
            "error": f"Station with ID {station_id} not found",
            "station_id": station_id
        }
        
    except Exception as e:
        logging.error(f"Error getting station details: {e}")
        return {
            "error": f"Failed to retrieve station details: {str(e)}"
        }

@app.put("/custom-stations/{station_id}")
def update_custom_station(station_id: str, request: StationUpdateRequest):
    """Update a custom charging station"""
    try:
        stations = load_custom_stations()
        
        station_found = False
        for i, station in enumerate(stations):
            if station["station_id"] == station_id:
                station_found = True
                
                # Update only provided fields
                if request.name is not None:
                    station["name"] = request.name
                if request.latitude is not None:
                    station["location"]["lat"] = request.latitude
                if request.longitude is not None:
                    station["location"]["lng"] = request.longitude
                if request.address is not None:
                    station["address"] = request.address
                if request.station_type is not None:
                    station["station_type"] = request.station_type
                if request.charger_types is not None:
                    station["charger_types"] = request.charger_types
                if request.power_output_kw is not None:
                    station["power_output_kw"] = request.power_output_kw
                if request.number_of_charging_points is not None:
                    station["number_of_charging_points"] = request.number_of_charging_points
                if request.operating_hours is not None:
                    station["operating_hours"] = request.operating_hours
                if request.amenities is not None:
                    station["amenities"] = request.amenities
                if request.pricing_info is not None:
                    station["pricing_info"] = request.pricing_info
                if request.additional_notes is not None:
                    station["additional_notes"] = request.additional_notes
                
                # Update last modified date
                station["last_updated"] = datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()
                
                stations[i] = station
                break
        
        if not station_found:
            return {
                "success": False,
                "error": f"Station with ID {station_id} not found"
            }
        
        # Save updated stations
        if save_custom_stations(stations):
            logging.info(f"Station {station_id} updated successfully")
            return {
                "success": True,
                "message": "Station updated successfully",
                "station_id": station_id,
                "last_updated": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()
            }
        else:
            return {
                "success": False,
                "error": "Failed to save updated station data"
            }
        
    except Exception as e:
        logging.error(f"Error updating station {station_id}: {e}")
        return {
            "success": False,
            "error": f"Failed to update station: {str(e)}"
        }

@app.delete("/custom-stations/{station_id}")
def delete_custom_station(station_id: str):
    """Delete a custom charging station"""
    try:
        stations = load_custom_stations()
        
        original_count = len(stations)
        stations = [station for station in stations if station["station_id"] != station_id]
        
        if len(stations) == original_count:
            return {
                "success": False,
                "error": f"Station with ID {station_id} not found"
            }
        
        # Save updated stations list
        if save_custom_stations(stations):
            logging.info(f"Station {station_id} deleted successfully")
            return {
                "success": True,
                "message": "Station deleted successfully",
                "station_id": station_id,
                "deleted_at": datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()
            }
        else:
            return {
                "success": False,
                "error": "Failed to save updated station data"
            }
        
    except Exception as e:
        logging.error(f"Error deleting station {station_id}: {e}")
        return {
            "success": False,
            "error": f"Failed to delete station: {str(e)}"
        }

@app.get("/station-form-options")
def get_station_form_options():
    """Get dropdown options for the station submission form"""
    return {
        "station_types": [
            "Public", "Private", "Semi-Public", "Hotel", "Mall", 
            "Restaurant", "Office", "Gas Station", "Parking Lot", "Highway Rest Stop"
        ],
        "charger_types": [
            "AC Slow (3kW)", "AC Fast (7-22kW)", "DC Fast (25-50kW)", 
            "DC Ultra Fast (50kW+)", "Tesla Supercharger", "CCS", "CHAdeMO", "Type 2"
        ],
        "business_types": [
            "Individual", "Business", "Hotel", "Mall", "Restaurant", 
            "Gas Station", "Parking Operator", "Real Estate", "Government", "NGO"
        ],
        "amenities": [
            "Restroom", "Food & Beverages", "WiFi", "Covered Parking", 
            "Shopping", "ATM", "Valet Parking", "Waiting Area", "CCTV Security", "24x7 Staff"
        ],
        "power_ranges": [
            {"label": "3 kW (AC Slow)", "value": 3},
            {"label": "7 kW (AC Fast)", "value": 7},
            {"label": "22 kW (AC Fast)", "value": 22},
            {"label": "25 kW (DC Fast)", "value": 25},
            {"label": "50 kW (DC Fast)", "value": 50},
            {"label": "100 kW (DC Ultra Fast)", "value": 100},
            {"label": "150 kW+ (DC Ultra Fast)", "value": 150}
        ]
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
