import requests

def get_current_weather(city, api_key):
    """Fetch current weather from OpenWeatherMap."""
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": api_key, "units": "metric"},
        timeout=15
    )
    response.raise_for_status()
    data = response.json()
    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "description": data["weather"][0]["description"],
        "rainfall_mm": data.get("rain", {}).get("1h", 0)
    }
