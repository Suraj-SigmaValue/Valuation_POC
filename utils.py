"""
Utility functions for Real Estate Valuation POC:
- Distance calculation (Haversine)
- Geocoding and Reverse Geocoding (Google Places / Geocoding + Nominatim fallback)
- Google Places API Client
- Demo / Mock Fallback Generator (for testing without an active API key)
"""

import os
import math
import requests
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def get_google_api_key(override_key: Optional[str] = None) -> str:
    """Retrieve Google Places API key from argument, environment, or .env."""
    if override_key and override_key.strip():
        return override_key.strip()
    for env_var in ["GOOGLE_PLACES_API_KEY", "GOOGLE_MAPS_API_KEY", "GOOGLE_API_KEY"]:
        key = os.getenv(env_var)
        if key and key.strip():
            return key.strip()
    return ""


def get_tavily_api_key(override_key: Optional[str] = None) -> str:
    """Retrieve Tavily API key from argument, environment, or .env."""
    if override_key and override_key.strip():
        return override_key.strip()
    key = os.getenv("TAVILY_API_KEY")
    return key.strip() if key else ""


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    R = 6371.0  # Earth's radius in kilometers

    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)


def geocode_project(
    project_name: str, api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Geocode a project name or address to get (latitude, longitude, formatted_address, city).
    Uses Google Geocoding / TextSearch API first if key available,
    otherwise falls back to Nominatim (OpenStreetMap).
    """
    api_key = get_google_api_key(api_key)

    # 1. Try Google Places Text Search if key available
    if api_key:
        try:
            url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            params = {"query": project_name, "key": api_key}
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            if data.get("status") == "OK" and data.get("results"):
                top_result = data["results"][0]
                loc = top_result["geometry"]["location"]
                addr = top_result.get("formatted_address", project_name)
                # Try to extract city from address components or address string
                city = _extract_city_from_address(addr)
                return {
                    "name": top_result.get("name", project_name),
                    "lat": loc["lat"],
                    "lng": loc["lng"],
                    "formatted_address": addr,
                    "city": city,
                    "source": "Google Places API",
                }
        except Exception as e:
            pass

    # 2. Fallback to geopy Nominatim
    try:
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent="valuation_poc_agent")
        location = geolocator.geocode(project_name, timeout=10)
        if location:
            addr = location.address
            city = _extract_city_from_address(addr)
            return {
                "name": project_name,
                "lat": location.latitude,
                "lng": location.longitude,
                "formatted_address": addr,
                "city": city,
                "source": "OpenStreetMap Nominatim",
            }
    except Exception:
        pass

    # Default fallback if nothing works (e.g. offline / not found)
    return {
        "name": project_name,
        "lat": 19.0760,
        "lng": 72.8777,
        "formatted_address": f"{project_name} (Location approximated)",
        "city": "Mumbai",
        "source": "Default Fallback",
    }


def reverse_geocode(
    lat: float, lng: float, api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Reverse geocode coordinates to find city, neighborhood, and formatted address.
    """
    api_key = get_google_api_key(api_key)

    # 1. Try Google Geocoding API if key available
    if api_key:
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {"latlng": f"{lat},{lng}", "key": api_key}
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            if data.get("status") == "OK" and data.get("results"):
                first = data["results"][0]
                addr = first.get("formatted_address", "")
                city = _extract_city_from_google_components(first.get("address_components", []))
                if not city:
                    city = _extract_city_from_address(addr)
                return {
                    "formatted_address": addr,
                    "city": city,
                    "source": "Google Geocoding API",
                }
        except Exception:
            pass

    # 2. Fallback to Nominatim
    try:
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent="valuation_poc_agent")
        location = geolocator.reverse((lat, lng), timeout=10)
        if location:
            addr = location.address
            raw = location.raw.get("address", {})
            city = (
                raw.get("city")
                or raw.get("town")
                or raw.get("municipality")
                or raw.get("county")
                or _extract_city_from_address(addr)
            )
            return {
                "formatted_address": addr,
                "city": city,
                "source": "OpenStreetMap Nominatim",
            }
    except Exception:
        pass

    return {
        "formatted_address": f"Lat: {lat:.5f}, Lng: {lng:.5f}",
        "city": "Unknown Region",
        "source": "Coordinates",
    }


def _extract_city_from_google_components(components: List[Dict[str, Any]]) -> str:
    """Extract locality or administrative area from Google address components."""
    locality = ""
    admin_area = ""
    for comp in components:
        types = comp.get("types", [])
        if "locality" in types:
            locality = comp.get("long_name", "")
        elif "administrative_area_level_2" in types:
            admin_area = comp.get("long_name", "")
        elif "administrative_area_level_1" in types and not admin_area:
            admin_area = comp.get("long_name", "")
    return locality or admin_area


def _extract_city_from_address(address: str) -> str:
    """Extract city name from address string, avoiding state and country names where possible."""
    KNOWN_MAJOR_CITIES = [
        "Mumbai", "Delhi", "Bengaluru", "Bangalore", "Hyderabad", "Chennai", "Kolkata",
        "Pune", "Ahmedabad", "Gurgaon", "Gurugram", "Noida", "Navi Mumbai", "Thane",
        "Jaipur", "Chandigarh", "Kochi", "Lucknow", "New York", "London", "Dubai", "Singapore",
        "Chicago", "San Francisco", "Los Angeles", "Toronto", "Sydney"
    ]
    addr_lower = address.lower()
    for c in KNOWN_MAJOR_CITIES:
        if c.lower() in addr_lower:
            return c

    tokens = [t.strip() for t in address.split(",") if t.strip()]
    if len(tokens) >= 3:
        # Avoid common country / state words at the very end
        non_city_words = {"india", "usa", "united states", "uk", "united kingdom", "uae", "maharashtra", "karnataka", "telangana", "haryana", "uttar pradesh", "tamil nadu"}
        for token in reversed(tokens[:-1]):
            clean_tok = token.strip()
            if not clean_tok.isdigit() and len(clean_tok) > 2 and clean_tok.lower() not in non_city_words:
                return clean_tok
    return tokens[0] if tokens else "Unknown City"


def search_google_places_nearby(
    lat: float,
    lng: float,
    radius_meters: int,
    keyword: Optional[str] = None,
    place_type: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Calls Google Places API Nearby Search.
    Returns: (list_of_places, error_message_or_none)
    """
    key = get_google_api_key(api_key)
    if not key:
        return [], "No Google Places API Key configured. Please add GOOGLE_PLACES_API_KEY in .env or enter it in the sidebar."

    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius_meters,
        "key": key,
    }
    if keyword:
        params["keyword"] = keyword
    if place_type:
        params["type"] = place_type

    try:
        res = requests.get(url, params=params, timeout=12)
        data = res.json()
        status = data.get("status")

        if status == "OK":
            results = data.get("results", [])
            places = []
            for item in results:
                loc = item.get("geometry", {}).get("location", {})
                p_lat = loc.get("lat")
                p_lng = loc.get("lng")
                dist = haversine_distance(lat, lng, p_lat, p_lng) if p_lat and p_lng else None
                places.append({
                    "name": item.get("name"),
                    "place_id": item.get("place_id"),
                    "lat": p_lat,
                    "lng": p_lng,
                    "distance_km": dist,
                    "address": item.get("vicinity") or item.get("formatted_address", ""),
                    "rating": item.get("rating", 0.0),
                    "user_ratings_total": item.get("user_ratings_total", 0),
                    "types": item.get("types", []),
                    "business_status": item.get("business_status", "OPERATIONAL"),
                    "google_maps_url": f"https://www.google.com/maps/place/?q=place_id:{item.get('place_id')}",
                })
            return places, None

        elif status == "ZERO_RESULTS":
            return [], None
        elif status == "REQUEST_DENIED":
            err = data.get("error_message", "Request Denied. Check API key permissions and billing.")
            return [], f"Google Places API Error: {err}"
        elif status == "OVER_QUERY_LIMIT":
            return [], "Google Places API Error: Over query limit / quota exceeded."
        else:
            err = data.get("error_message", f"Status: {status}")
            return [], f"Google Places API Error: {err}"

    except Exception as e:
        return [], f"Network error connecting to Google Places API: {str(e)}"


def search_google_places_text(
    query: str,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    radius_meters: Optional[int] = None,
    api_key: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Calls Google Places API Text Search.
    Returns: (list_of_places, error_message_or_none)
    """
    key = get_google_api_key(api_key)
    if not key:
        return [], "No Google Places API Key configured."

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": key}
    if lat is not None and lng is not None and radius_meters:
        params["location"] = f"{lat},{lng}"
        params["radius"] = radius_meters

    try:
        res = requests.get(url, params=params, timeout=12)
        data = res.json()
        status = data.get("status")

        if status == "OK":
            results = data.get("results", [])
            places = []
            for item in results:
                loc = item.get("geometry", {}).get("location", {})
                p_lat = loc.get("lat")
                p_lng = loc.get("lng")
                dist = haversine_distance(lat, lng, p_lat, p_lng) if (lat and lng and p_lat and p_lng) else None
                places.append({
                    "name": item.get("name"),
                    "place_id": item.get("place_id"),
                    "lat": p_lat,
                    "lng": p_lng,
                    "distance_km": dist,
                    "address": item.get("formatted_address") or item.get("vicinity", ""),
                    "rating": item.get("rating", 0.0),
                    "user_ratings_total": item.get("user_ratings_total", 0),
                    "types": item.get("types", []),
                    "business_status": item.get("business_status", "OPERATIONAL"),
                    "google_maps_url": f"https://www.google.com/maps/place/?q=place_id:{item.get('place_id')}",
                })
            return places, None
        elif status == "ZERO_RESULTS":
            return [], None
        else:
            err = data.get("error_message", f"Status: {status}")
            return [], f"Google Places API Error: {err}"
    except Exception as e:
        return [], f"Network error: {str(e)}"
