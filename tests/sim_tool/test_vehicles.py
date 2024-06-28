from vehicles import convert_to_vehicle_objects
import pandas as pd


def test_vehicle_from_dataframe():
    data = {
        "name": ["Tesla Model 3", "Tesla Model Y"],
        "count": [1, 1],
        "capacity_kWh": [75, 75],
        "start_time": ["08:00", "07:30"],
        "end_time": ["19:45", "20:00"],
        "drive_km": [100, 120],
        "range_km": [500, 525],
        "min_soc": [20, 20],
        "days_of_week": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]],
    }

    df = pd.DataFrame(data)

    expected = [
        {
            "name": "Tesla Model 3",
            "capacity": 75,
            "mileage": 6.666666666666667,
            "charging_curve": [[0, 11], [0.8, 11], [1, 11]],
            "min_charging_power": 0.2,
            "v2g": False,
            "v2g_power_factor": 0.5,
            "discharge_limit": 0.5,
            "statistical_values": {
                "distance_in_km": {
                    "avg_distance": 100,
                    "std_distance": 0,
                    "min_distance": 0,
                    "max_distance": 0,
                },
                "departure": {
                    "avg_start": "08:00",
                    "std_start_in_hours": 0,
                    "min_start": None,
                    "max_start": None,
                },
                "duration_in_hours": {
                    "avg_driving": 11.75,
                    "std_driving": 0,
                    "min_driving": 0,
                    "max_driving": 0,
                },
            },
            "no_drive_days": [1, 2, 3, 4, 5],
        },
        {
            "name": "Tesla Model Y",
            "capacity": 75,
            "mileage": 7.0,
            "charging_curve": [[0, 11], [0.8, 11], [1, 11]],
            "min_charging_power": 0.2,
            "v2g": False,
            "v2g_power_factor": 0.5,
            "discharge_limit": 0.5,
            "statistical_values": {
                "distance_in_km": {
                    "avg_distance": 120,
                    "std_distance": 0,
                    "min_distance": 0,
                    "max_distance": 0,
                },
                "departure": {
                    "avg_start": "07:30",
                    "std_start_in_hours": 0,
                    "min_start": None,
                    "max_start": None,
                },
                "duration_in_hours": {
                    "avg_driving": 12.5,
                    "std_driving": 0,
                    "min_driving": 0,
                    "max_driving": 0,
                },
            },
            "no_drive_days": [1, 2, 3, 4, 5, 6],
        },
    ]

    result = convert_to_vehicle_objects(df)

    for i in range(len(result)):
        assert(expected[i] == result[i].__dict__)
