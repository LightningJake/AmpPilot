import pandas as pd
import os
import logging
from typing import Dict, Any, Optional

class EVDataLoader:
    def __init__(self):
        """Initialize the EV data loader with the CSV file."""
        self.csv_path = os.path.join(os.path.dirname(__file__), "cars_data.csv")
        self.ev_data = None
        self.load_data()
    
    def load_data(self):
        """Load the EV data from CSV file."""
        try:
            self.ev_data = pd.read_csv(self.csv_path)
            logging.info(f"Successfully loaded {len(self.ev_data)} EV models from database")
        except Exception as e:
            logging.error(f"Failed to load EV data: {e}")
            self.ev_data = None
    
    def find_vehicle_data(self, manufacturer: str, model: str) -> Optional[Dict[str, Any]]:
        """Find vehicle data based on manufacturer and model."""
        if self.ev_data is None:
            return None
        
        try:
            # Convert to lowercase for case-insensitive matching
            manufacturer_lower = manufacturer.lower().strip()
            model_lower = model.lower().strip()
            
            # Try exact match first
            exact_match = self.ev_data[
                (self.ev_data['Brand'].str.lower().str.strip() == manufacturer_lower) &
                (self.ev_data['Model'].str.lower().str.strip() == model_lower)
            ]
            
            if not exact_match.empty:
                return exact_match.iloc[0].to_dict()
            
            # Try partial match for model
            partial_model_match = self.ev_data[
                (self.ev_data['Brand'].str.lower().str.strip() == manufacturer_lower) &
                (self.ev_data['Model'].str.lower().str.contains(model_lower, na=False))
            ]
            
            if not partial_model_match.empty:
                return partial_model_match.iloc[0].to_dict()
            
            # Try manufacturer only and return most popular model
            manufacturer_match = self.ev_data[
                self.ev_data['Brand'].str.lower().str.strip() == manufacturer_lower
            ]
            
            if not manufacturer_match.empty:
                # Return the model with lowest starting price (usually most popular)
                popular_model = manufacturer_match.loc[
                    manufacturer_match['Ex-Showroom Price Min (₹ Lakh)'].idxmin()
                ]
                return popular_model.to_dict()
            
            return None
            
        except Exception as e:
            logging.error(f"Error finding vehicle data: {e}")
            return None
    
    def get_manufacturer_models(self, manufacturer: str) -> list:
        """Get all models for a specific manufacturer."""
        if self.ev_data is None:
            return []
        
        try:
            manufacturer_lower = manufacturer.lower().strip()
            models = self.ev_data[
                self.ev_data['Brand'].str.lower().str.strip() == manufacturer_lower
            ]['Model'].tolist()
            return models
        except Exception as e:
            logging.error(f"Error getting manufacturer models: {e}")
            return []
    
    def get_all_manufacturers(self) -> list:
        """Get list of all manufacturers."""
        if self.ev_data is None:
            return []
        
        try:
            return sorted(self.ev_data['Brand'].unique().tolist())
        except Exception as e:
            logging.error(f"Error getting manufacturers: {e}")
            return []
    
    def get_vehicle_specs(self, manufacturer: str, model: str) -> Dict[str, Any]:
        """Get comprehensive vehicle specifications."""
        vehicle_data = self.find_vehicle_data(manufacturer, model)
        
        if not vehicle_data:
            return {
                'found': False,
                'message': f'Vehicle {manufacturer} {model} not found in database'
            }
        
        def safe_get_numeric(value, default=None):
            """Safely convert value to float, return default if conversion fails."""
            if pd.isna(value) or value == 'N/A':
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        def safe_get_string(value, default='Unknown'):
            """Safely get string value."""
            if pd.isna(value) or value == 'N/A':
                return default
            return str(value)
        
        # Extract and clean data
        specs = {
            'found': True,
            'brand': safe_get_string(vehicle_data.get('Brand')),
            'model': safe_get_string(vehicle_data.get('Model')),
            'body_type': safe_get_string(vehicle_data.get('Body Type')),
            'seating_capacity': safe_get_numeric(vehicle_data.get('Seating Capacity'), 5),
            
            # Battery specifications
            'battery_capacity_min_kwh': safe_get_numeric(vehicle_data.get('Battery Capacity Min (kWh)')),
            'battery_capacity_max_kwh': safe_get_numeric(vehicle_data.get('Battery Capacity Max (kWh)')),
            
            # Range specifications
            'arai_range_min_km': safe_get_numeric(vehicle_data.get('ARAI-Claimed Range Min (km)')),
            'arai_range_max_km': safe_get_numeric(vehicle_data.get('ARAI-Claimed Range Max (km)')),
            'real_world_range_km': safe_get_numeric(vehicle_data.get('Real-World Range (km)')),
            
            # Performance specifications
            'power_min_bhp': safe_get_numeric(vehicle_data.get('Power Min (bhp)')),
            'power_max_bhp': safe_get_numeric(vehicle_data.get('Power Max (bhp)')),
            'drivetrain': safe_get_string(vehicle_data.get('Drivetrain')),
            'top_speed_kmh': safe_get_numeric(vehicle_data.get('Top Speed (km/h)')),
            
            # Charging specifications
            'dc_charge_time_min': safe_get_numeric(vehicle_data.get('DC Charge Time (min)')),
            
            # Safety and features
            'ncap_rating': safe_get_string(vehicle_data.get('NCAP Rating')),
            'standard_airbags': safe_get_numeric(vehicle_data.get('Standard Airbags')),
            'adas': safe_get_string(vehicle_data.get('ADAS')),
            
            # Pricing
            'price_min_lakh': safe_get_numeric(vehicle_data.get('Ex-Showroom Price Min (₹ Lakh)')),
            'price_max_lakh': safe_get_numeric(vehicle_data.get('Ex-Showroom Price Max (₹ Lakh)')),
        }
        
        return specs

# Global instance for easy access
ev_data_loader = EVDataLoader()
