import pandas as pd
import numpy as np
from typing import List, Dict

class Vehicle:
    def __init__(self, name: str, capacity: int, mileage: float, charging_curve: List[List[float]], 
                 min_charging_power: float, v2g: bool, v2g_power_factor: float, discharge_limit: float, 
                 statistical_values: Dict, no_drive_days: List[int], cp_charge_power: float):
        self.name = name
        self.capacity = capacity
        self.mileage = mileage
        self.charging_curve = charging_curve
        self.min_charging_power = min_charging_power
        self.v2g = v2g
        self.v2g_power_factor = v2g_power_factor
        self.discharge_limit = discharge_limit
        self.statistical_values = statistical_values
        self.no_drive_days = no_drive_days
        self.charging_curve = [[0, cp_charge_power], [0.8, cp_charge_power], [1, cp_charge_power]]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vehicle):
            return False

        differences = self.diff(other)
        if differences:
            print("Differences found:")
            for diff in differences:
                print(diff)
            return False
        return True

    def diff(self, other: 'Vehicle') -> List[str]:
        differences = []
        
        if self.name != other.name:
            differences.append(f"Different name: {self.name} != {other.name}")
        if self.capacity != other.capacity:
            differences.append(f"Different capacity: {self.capacity} != {other.capacity}")
        if abs(self.mileage - other.mileage) >= 0.001:
            differences.append(f"Different mileage: {self.mileage} != {other.mileage}")
        if self.charging_curve != other.charging_curve:
            differences.append(f"Different charging_curve: {self.charging_curve} != {other.charging_curve}")
        if self.min_charging_power != other.min_charging_power:
            differences.append(f"Different min_charging_power: {self.min_charging_power} != {other.min_charging_power}")
        if self.v2g != other.v2g:
            differences.append(f"Different v2g: {self.v2g} != {other.v2g}")
        if self.v2g_power_factor != other.v2g_power_factor:
            differences.append(f"Different v2g_power_factor: {self.v2g_power_factor} != {other.v2g_power_factor}")
        if self.discharge_limit != other.discharge_limit:
            differences.append(f"Different discharge_limit: {self.discharge_limit} != {other.discharge_limit}")
        if self.statistical_values != other.statistical_values:
            differences.append(f"Different statistical_values: {self.statistical_values} != {other.statistical_values}")
        if self.no_drive_days != other.no_drive_days:
            differences.append(f"Different no_drive_days: {self.no_drive_days} != {other.no_drive_days}")

        return differences
        

def convert_to_vehicle_objects(df: pd.DataFrame) -> List[Vehicle]:
    default_values = {
        "name": "No Name",
        "capacity": -1,
        "mileage": -1,
        "charging_curve": [[0, 11], [0.8, 11], [1, 11]],
        "min_charging_power": 0.2,
        "v2g": False,
        "v2g_power_factor": 0.5,
        "discharge_limit": 0.5,
        "statistical_values": {
            "distance_in_km": {
                "avg_distance": 47.13,
                "std_distance": 0,
                "min_distance": 0,
                "max_distance": 1000
            },
            "departure": {
                "avg_start": None,
                "std_start_in_hours": 0,
                "min_start": "06:00",
                "max_start": "10:00"
            },
            "duration_in_hours": {
                "avg_driving": None,
                "std_driving": 0,
                "min_driving": 0,
                "max_driving": 20
            }
        },
        "no_drive_days": []
    }
    
    vehicles = []
    
    for _, row in df.iterrows():
        name = row.get("name")
        capacity = row.get("capacity_kWh")
        range = row.get("range_km")
        milage = capacity * 100 / range
        charging_curve = default_values["charging_curve"]
        min_charging_power = default_values["min_charging_power"]
        v2g = default_values["v2g"]
        v2g_power_factor = default_values["v2g_power_factor"]
        discharge_limit = default_values["discharge_limit"]
        
        distance_in_km = {
            "avg_distance": row.get("drive_km"),
            "std_distance": default_values["statistical_values"]["distance_in_km"]["std_distance"],
            "min_distance": default_values["statistical_values"]["distance_in_km"]["min_distance"],
            "max_distance": default_values["statistical_values"]["distance_in_km"]["max_distance"]
        }
        
        departure = {
            "avg_start": row.get("start_time"),
            "std_start_in_hours": default_values["statistical_values"]["departure"]["std_start_in_hours"],
            "min_start": default_values["statistical_values"]["departure"]["min_start"],
            "max_start": default_values["statistical_values"]["departure"]["max_start"]
        }
        
        duration_in_hours = {
            "avg_driving": (pd.to_datetime(row.get("end_time")) -
                            pd.to_datetime(row.get("start_time"))).seconds / 3600,
            "std_driving": default_values["statistical_values"]["duration_in_hours"]["std_driving"],
            "min_driving": default_values["statistical_values"]["duration_in_hours"]["min_driving"],
            "max_driving": default_values["statistical_values"]["duration_in_hours"]["max_driving"]
        }
        
        statistical_values = {
            "distance_in_km": distance_in_km,
            "departure": departure,
            "duration_in_hours": duration_in_hours
        }
        
        no_drive_days = row.get("days_of_week", default_values["no_drive_days"])
        
        vehicle = Vehicle(
            name=name,
            capacity=capacity,
            mileage=milage,
            charging_curve=charging_curve,
            min_charging_power=min_charging_power,
            v2g=v2g,
            v2g_power_factor=v2g_power_factor,
            discharge_limit=discharge_limit,
            statistical_values=statistical_values,
            no_drive_days=no_drive_days,
            cp_charge_power=row.get("cp_power_kW"),
        )
        
        vehicles.append(vehicle)
    
    return vehicles
