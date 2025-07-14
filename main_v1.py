import os
from dotenv import load_dotenv
import googlemaps
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
from math import radians, cos, sin, sqrt, atan2
import logging
import joblib
from datetime import datetime
import pytz

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# CORS (for frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Maps API Key
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# Constants
MAX_VEHICLE_RANGE_KM = 200  # Maximum range at 100% charge
GEOFENCING_RADIUS_KM = 3  # 3km radius for better station detection

# Request Models
class EVRouteRequest(BaseModel):
    origin: str
    destination: str
    current_range_km: float  # Current remaining range shown on dashboard

class BreakdownRequest(BaseModel):
    latitude: float
    longitude: float

class GeofenceRerouteRequest(BaseModel):
    new_station_place_id: str
    original_route_data: dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Load the trained occupancy model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ev_occupancy_model.pkl")
try:
    occupancy_model = joblib.load(MODEL_PATH)
    logging.info("Occupancy prediction model loaded successfully.")
except Exception as e:
    occupancy_model = None
    logging.error(f"Failed to load occupancy model: {e}")

# Helper: Haversine distance calculator
def haversine(coord1, coord2):
    R = 6371  # Earth radius in km
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# Helper: Predict occupancy for a charging station
def predict_station_occupancy(station, dt=None):
    """
    Predict occupancy for a given station dict (from Google Places API).
    dt: datetime object (defaults to now, IST)
    """
    if occupancy_model is None:
        return None

    try:
        # Use IST timezone for India
        if dt is None:
            dt = datetime.now(pytz.timezone("Asia/Kolkata"))
        hour = dt.hour
        day = dt.weekday()
        is_weekend = int(day >= 5)
        # For demo, use dummy values for weather/traffic
        temperature = 30.0
        rainfall = 1
        traffic_level = 1
        historical_avg_occupancy = 0.5

        # If you have real data, replace above with actual values

        features = [
            station.get("station_id", 0),
            station["location"]["lat"],
            station["location"]["lng"],
            hour,
            day,
            is_weekend,
            temperature,
            rainfall,
            traffic_level,
            historical_avg_occupancy
        ]
        pred = occupancy_model.predict([features])[0]
        logging.info(
            f"Predicted occupancy for station {station.get('name', '')} ({station.get('place_id', '')}): {pred:.2f}"
        )
        return round(pred, 2)
    except Exception as e:
        logging.error(f"Occupancy prediction failed: {e}")
        return None

def find_better_stations_nearby(original_station_location, original_rating=None, original_occupancy=None):
    """Find better charging stations within 3km radius"""
    try:
        response = gmaps.places_nearby(
            location=original_station_location,
            radius=3000,  # 3km radius
            keyword="EV charging",
            type="charging_station"
        )

        better_stations = []
        for idx, place in enumerate(response.get('results', [])):
            station_rating = place.get('rating', 0)
            min_rating_threshold = original_rating if original_rating else 4.0

            distance = haversine(
                (original_station_location['lat'], original_station_location['lng']),
                (place['geometry']['location']['lat'], place['geometry']['location']['lng'])
            )
            if distance <= GEOFENCING_RADIUS_KM and distance > 0.1:
                station_info = {
                    "name": place['name'],
                    "address": place.get('vicinity'),
                    "rating": station_rating,
                    "location": place['geometry']['location'],
                    "place_id": place['place_id'],
                    "station_id": idx + 2,  # Dummy id for demo
                    "distance_from_original": round(distance, 2),
                    "rating_improvement": round(station_rating - (original_rating or 0), 1),
                    "maps_link": f"https://www.google.com/maps/place/?q=place_id:{place['place_id']}"
                }
                # Predict occupancy
                occupancy = predict_station_occupancy(station_info)
                station_info["predicted_occupancy"] = occupancy

                # --- NEW LOGIC: Compare both rating and occupancy ---
                if (
                    station_rating > min_rating_threshold and
                    (original_occupancy is None or occupancy <= original_occupancy)
                ):
                    better_stations.append(station_info)

        better_stations.sort(key=lambda x: (-x['rating'], x['distance_from_original']))
        return better_stations

    except Exception as e:
        print(f"Error finding better stations: {e}")
        return []

def find_nearby_services(location, keyword, service_type):
    try:
        response = gmaps.places_nearby(
            location=location,
            radius=5000,  # 5 km radius for breakdown services
            keyword=keyword,
            type=service_type
        )
        
        services = []
        for place in response.get('results', []):  # Limit to 10 results
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
            
            services.append(service)
        
        # Sort by distance
        services.sort(key=lambda x: x["distance_km"])
        return services
        
    except Exception as e:
        print(f"Error finding {service_type}: {e}")
        return []

# --- Modify find_nearest_charger to include occupancy prediction ---
def find_nearest_charger(location):
    try:
        response = gmaps.places_nearby(
            location=location,
            radius=5000,  # 5 km radius
            keyword="EV charging",
            type="charging_station"
        )

        stations = response.get('results', [])
        if not stations:
            return None

        best = stations[0]
        # Add dummy station_id for demo (since Google API doesn't provide)
        station_info = {
            "name": best['name'],
            "address": best.get('vicinity'),
            "rating": best.get('rating'),
            "location": best['geometry']['location'],
            "place_id": best['place_id'],
            "station_id": 1,  # You may want to map this properly
            "maps_link": f"https://www.google.com/maps/place/?q=place_id:{best['place_id']}"
        }
        # Predict occupancy
        occupancy = predict_station_occupancy(station_info)
        station_info["predicted_occupancy"] = occupancy
        return station_info
    except Exception as e:
        return None

@app.post("/ev-route")
def generate_ev_route(request: EVRouteRequest):
    # Step 1: Get driving directions
    directions = gmaps.directions(
        request.origin,
        request.destination,
        mode="driving"
    )

    if not directions:
        return {"error": "No route found between locations."}

    route_legs = directions[0]['legs'][0]
    steps = route_legs['steps']

    total_route_distance_km = route_legs['distance']['value'] / 1000  # in km

    # Check if current battery can complete the trip
    if total_route_distance_km <= request.current_range_km:
        return {
            "message": "Route is within current EV range, no charging needed.",
            "total_distance_km": total_route_distance_km,
            "current_battery_sufficient": True,
            "charging_stops": [],
            "geofencing_opportunities": [],
            "waypoints": []  # Add waypoints for route
        }

    # Step 2: Walk through the route step-by-step
    charging_stops = []
    geofencing_opportunities = []
    waypoints = []  # Add waypoints list
    distance_tracker = 0.0
    current_battery_range = request.current_range_km
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
                
                charging_stop = {
                    **charger_location,
                    "distance_from_start": distance_tracker - segment_distance,
                    "battery_level_on_arrival": f"{max(0, current_battery_range - (distance_tracker - segment_distance)):.1f} km remaining",
                    "stop_index": len(charging_stops)
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
                    geofencing_opportunities.append({
                        "original_station": charging_stop,
                        "better_alternative": best_alternative,
                        "improvement_reason": (
                            f"Better rating ({best_alternative['rating']}⭐ vs {charger_location.get('rating', 'N/A')}⭐) "
                            f"and occupancy ({best_alternative.get('predicted_occupancy', 'N/A')} vs {original_occupancy})"
                        ),
                        "distance_detour": best_alternative['distance_from_original'],
                        "better_predicted_occupancy": best_alternative.get('predicted_occupancy')
                    })
                
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

    return {
        "origin": request.origin,
        "destination": request.destination,
        "total_distance_km": total_route_distance_km,
        "initial_range_km": request.current_range_km,
        "max_vehicle_range_km": MAX_VEHICLE_RANGE_KM,
        "charging_stops": charging_stops,
        "geofencing_opportunities": geofencing_opportunities,
        "total_charging_stops": len(charging_stops),
        "waypoints": waypoints  # Include waypoints in response
    }

@app.post("/reroute-to-better-station")
def reroute_to_better_station(request: GeofenceRerouteRequest):
    """Handle rerouting when user accepts better station suggestion"""
    try:
        # Get details of the new station
        place_details = gmaps.place(place_id=request.new_station_place_id)
        new_station = place_details['result']
        
        # Get the original route data
        original_data = request.original_route_data
        
        # Find the index of the station being replaced
        station_index = None
        for i, stop in enumerate(original_data.get('charging_stops', [])):
            if stop.get('place_id') == request.new_station_place_id:
                station_index = i
                break
        
        # Update waypoints with new station location
        waypoints = original_data.get('waypoints', [])
        if station_index is not None and station_index < len(waypoints):
            waypoints[station_index] = {
                "lat": new_station['geometry']['location']['lat'],
                "lng": new_station['geometry']['location']['lng']
            }
        
        return {
            "success": True,
            "message": "Route updated successfully",
            "waypoints": waypoints,
            "new_station": {
                "name": new_station['name'],
                "address": new_station.get('formatted_address'),
                "rating": new_station.get('rating'),
                "location": new_station['geometry']['location'],
                "place_id": new_station['place_id'],
                "maps_link": f"https://www.google.com/maps/place/?q=place_id:{new_station['place_id']}"
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to reroute: {str(e)}"
        }

@app.post("/car-breakdown")
def handle_car_breakdown(request: BreakdownRequest):
    try:
        location = (request.latitude, request.longitude)
        
        # Find nearby garages
        garages = find_nearby_services(location, "car repair", "car_repair")
        
        # Find nearby tow services
        tow_services = find_nearby_services(location, "towing service", "towing")
        
        return {
            "user_location": {
                "lat": request.latitude,
                "lng": request.longitude
            },
            "garages": garages,
            "tow_services": tow_services,
            "total_services": len(garages) + len(tow_services)
        }
    except Exception as e:
        return {
            "error": f"Failed to find breakdown services: {str(e)}",
            "garages": [],
            "tow_services": []
        }