"""
Utility functions for Real Estate Valuation POC:
- Distance calculation (Haversine)
- Geocoding and Reverse Geocoding (Google Places / Geocoding + Nominatim fallback)
- Google Places API Client
- Demo / Mock Fallback Generator (for testing without an active API key)
"""

import os
import re
import math
import requests
import threading
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Concurrency & Persistent Quota Management
_ddg_semaphore = threading.Semaphore(8)
_QUOTA_FLAG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".quota_status.json")

def _is_quota_exhausted(service: str) -> bool:
    """Check if service quota has been marked exhausted recently (persisted across reloads)."""
    try:
        if os.path.exists(_QUOTA_FLAG_FILE):
            with open(_QUOTA_FLAG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                ts = data.get(f"{service}_exhausted_at", 0)
                # Keep circuit breaker active for 2 hours
                if time.time() - ts < 7200:
                    return True
    except Exception:
        pass
    return False

def _mark_quota_exhausted(service: str) -> None:
    """Record service quota exhaustion so reloads and subsequent calls skip instantly."""
    try:
        data = {}
        if os.path.exists(_QUOTA_FLAG_FILE):
            try:
                with open(_QUOTA_FLAG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data[f"{service}_exhausted_at"] = time.time()
        with open(_QUOTA_FLAG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass



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


def get_openai_api_key(override_key: Optional[str] = None) -> str:
    """Retrieve OpenAI API key from argument, environment, or .env."""
    if override_key and override_key.strip():
        return override_key.strip()
    for env_var in ["OPENAI_API_KEY", "OPENAI_KEY", "Open_AI_API", "OPEN_AI_API", "OPENAI_API"]:
        key = os.getenv(env_var)
        if key and key.strip():
            return key.strip()
    return ""


def parse_coordinates(text: Optional[str]) -> Optional[Tuple[float, float]]:
    """
    Robustly extract (latitude, longitude) from various coordinate strings:
    - '18.539574, 73.9463397'
    - '18.539574,73.9463397'
    - '18.539574 73.9463397'
    - 'Lat: 18.539574, Lng: 73.9463397'
    - 'latitude: 18.539574, longitude: 73.9463397'
    - '18.539574° N, 73.9463397° E'
    - '-33.8688, 151.2093'
    Returns (lat, lng) as floats or None.
    """
    if not text:
        return None
    s = str(text).strip()

    # 1. Check for labeled coordinates (e.g. Lat: 18.539574, Lng: 73.9463397)
    labeled = re.search(
        r'(?:lat|latitude)?\s*[:=]?\s*([-+]?[0-9]{1,2}(?:\.[0-9]+)?)[°\s]*[NS]?\s*[,;/|\s]\s*(?:lon|lng|long|longitude)?\s*[:=]?\s*([-+]?[0-9]{1,3}(?:\.[0-9]+)?)[°\s]*[EW]?',
        s,
        re.I
    )
    if labeled:
        try:
            lat = float(labeled.group(1))
            lng = float(labeled.group(2))
            if -90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0:
                return lat, lng
        except (ValueError, TypeError):
            pass

    # 2. Check for standard comma/space delimited numbers
    m = re.search(r'([-+]?[0-9]{1,2}(?:\.[0-9]+)?)\s*[,;/|\s]\s*([-+]?[0-9]{1,3}(?:\.[0-9]+)?)', s)
    if m:
        try:
            lat = float(m.group(1))
            lng = float(m.group(2))
            if -90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0:
                return lat, lng
        except (ValueError, TypeError):
            pass

    return None


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
    Geocode a project name, address, or coordinate string to get (latitude, longitude, formatted_address, city).
    If the input is coordinates (e.g. '18.539574, 73.9463397'), it automatically parses and reverse-geocodes them.
    """
    api_key = get_google_api_key(api_key)

    # 0. Automatically detect if input itself is coordinates!
    coords = parse_coordinates(project_name)
    if coords:
        c_lat, c_lng = coords
        rev = reverse_geocode(c_lat, c_lng, api_key=api_key)
        first_part = rev.get("formatted_address", "").split(",")[0].strip()
        clean_name = first_part if (first_part and not first_part.startswith("Lat:")) else f"Location ({c_lat:.5f}, {c_lng:.5f})"
        return {
            "name": clean_name,
            "lat": c_lat,
            "lng": c_lng,
            "formatted_address": rev.get("formatted_address", f"{c_lat:.5f}, {c_lng:.5f}"),
            "city": rev.get("city", "Unknown City"),
            "source": "Coordinates Auto-Detected",
        }

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
                first_name = addr.split(",")[0].strip() if addr else ""
                return {
                    "name": first_name,
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
        "Faridabad", "Ghaziabad", "Greater Noida", "Jaipur", "Chandigarh", "Kochi", "Lucknow",
        "Indore", "Bhopal", "Nagpur", "Surat", "Vadodara", "Coimbatore", "Visakhapatnam",
        "Patna", "Dehradun", "Mysore", "New York", "London", "Dubai", "Singapore",
        "Chicago", "San Francisco", "Los Angeles", "Toronto", "Sydney"
    ]
    addr_lower = address.lower()
    for c in KNOWN_MAJOR_CITIES:
        if c.lower() in addr_lower:
            return c

    tokens = [t.strip() for t in address.split(",") if t.strip()]
    if len(tokens) >= 3:
        # Avoid common country / state words at the very end
        non_city_words = {
            "india", "usa", "united states", "uk", "united kingdom", "uae",
            "maharashtra", "karnataka", "telangana", "haryana", "uttar pradesh",
            "tamil nadu", "delhi", "punjab", "rajasthan", "gujarat"
        }
        for token in reversed(tokens[:-1]):
            clean_tok = re.sub(r"\d+", "", token).strip()
            if len(clean_tok) > 2 and clean_tok.lower() not in non_city_words:
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
                    "source": "Google Places API",
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
                    "source": "Google Places API",
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


def search_duckduckgo(query: str, max_results: int = 4) -> List[Dict[str, str]]:
    """Execute DuckDuckGo search with rate-limiting protection and fallback backends."""
    with _ddg_semaphore:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    from ddgs import DDGS
                except ImportError:
                    from duckduckgo_search import DDGS

                results = []
                raw_res = []
                # 1. Primary standard query (fast 3s timeout)
                try:
                    with DDGS(timeout=3) as ddgs:
                        raw_res = list(ddgs.text(query, max_results=max_results))
                except Exception:
                    pass

                # 2. Resilient lite backend query if standard was empty
                if not raw_res:
                    try:
                        with DDGS(timeout=3) as ddgs:
                            raw_res = list(ddgs.text(query, max_results=max_results, backend="lite"))
                    except Exception:
                        pass

                for item in raw_res:
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("body", ""),
                        "url": item.get("href", ""),
                    })
                return results
        except Exception:
            return []


def search_tavily(query: str, api_key: str, max_results: int = 4) -> List[Dict[str, str]]:
    """Execute Tavily AI search with quota-exhaustion protection."""
    if _is_quota_exhausted("tavily"):
        return []

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        res = client.search(query=query, search_depth="basic", max_results=max_results)
        results = []
        for item in res.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("content", ""),
                "url": item.get("url", ""),
            })
        return results
    except Exception as e:
        err_msg = str(e).lower()
        if any(term in err_msg for term in ["exceeds", "limit", "forbidden", "403", "quota", "pay-as-you-go"]):
            _mark_quota_exhausted("tavily")
        return []


def search_openai_web(query: str, api_key: Optional[str] = None, max_results: int = 4) -> List[Dict[str, str]]:
    """Execute web search using OpenAI's web search tool or model intelligence with quota protection."""
    if _is_quota_exhausted("openai"):
        return []

    key = get_openai_api_key(api_key)
    if not key:
        return []

    try:
        from openai import OpenAI
        # Strict 3.0s timeout and zero retries to prevent blocking worker threads when quota is low or connection hangs
        client = OpenAI(api_key=key, timeout=3.0, max_retries=0)

        # 1. Try OpenAI Responses API with official web_search_preview tool
        try:
            resp = client.responses.create(
                model="gpt-4o-mini",
                tools=[{"type": "web_search_preview"}],
                input=(
                    f"Perform a web search to find exact real estate project details for: '{query}'. "
                    f"Extract per sq ft rates, typical ticket size, carpet areas, unit configurations (1/2/3 BHK), "
                    f"developer / builder name, possession status, and project scale. Include source links."
                ),
            )
            results = []
            extracted_text = ""
            if hasattr(resp, "output") and resp.output:
                for item in resp.output:
                    if getattr(item, "type", "") == "message" and hasattr(item, "content"):
                        for c in item.content:
                            c_text = getattr(c, "text", "")
                            if c_text:
                                extracted_text += "\n" + c_text
                            annotations = getattr(c, "annotations", []) or []
                            for ann in annotations:
                                if getattr(ann, "type", "") == "url_citation":
                                    url = getattr(ann, "url", "")
                                    title = getattr(ann, "title", "") or f"OpenAI Web Source: {query[:35]}"
                                    if url and not any(r["url"] == url for r in results):
                                        results.append({
                                            "title": title,
                                            "snippet": c_text[:3500] if c_text else f"Source for {query}",
                                            "url": url,
                                        })

            # Ensure the primary structured extraction is always available for parsing
            if extracted_text and not any(r.get("title", "").startswith("OpenAI Structured Intelligence") for r in results):
                results.insert(0, {
                    "title": f"OpenAI Structured Intelligence: {query[:45]}",
                    "snippet": extracted_text.strip()[:3500],
                    "url": results[0]["url"] if results else "https://openai.com",
                })

            if results:
                return results[:max_results]
        except Exception as e_resp:
            err_msg = str(e_resp).lower()
            if any(term in err_msg for term in ["insufficient_quota", "credit_balance_exhausted", "429"]):
                _mark_quota_exhausted("openai")
                return []

        # 2. Fallback to Chat Completions if responses API web tool is unavailable
        try:
            chat_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a real estate research intelligence assistant. Provide verified facts on "
                            "typical carpet areas (sq ft), per sq ft rates (INR), unit typologies (BHK), "
                            "developer name, construction/possession status, and project scale for the queried project."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Find information for real estate query: {query}",
                    },
                ],
                max_tokens=600,
                temperature=0.2,
            )
            content = (chat_resp.choices[0].message.content or "").strip()
            if content:
                return [{
                    "title": f"OpenAI Intelligence: {query[:45]}",
                    "snippet": content[:3500],
                    "url": "https://openai.com",
                }]
        except Exception as e_chat:
            err_msg = str(e_chat).lower()
            if any(term in err_msg for term in ["insufficient_quota", "credit_balance_exhausted", "429"]):
                _mark_quota_exhausted("openai")
                return []

        return []
    except Exception as e_all:
        err_msg = str(e_all).lower()
        if any(term in err_msg for term in ["insufficient_quota", "credit_balance_exhausted", "429"]):
            _mark_quota_exhausted("openai")
        return []


def is_relevant_real_estate_result(results: List[Dict[str, str]], query: str = "") -> bool:
    """
    Checks if search results actually contain meaningful real estate facts.
    Avoids accepting irrelevant SaaS tools (e.g. floorplanmaker.ai), YouTube home tours from other cities,
    or generic Wikipedia articles that contain no real estate pricing or property metrics.
    """
    if not results:
        return False

    banned_domain_patterns = [
        "floorplanmaker", "floorplanner", "sweethome3d", "roomsketcher",
        "homestyler", "wikipedia.org/wiki/Royal_Highness", "youtube.com/@hrh",
        "myhome.ie", "dictionary", "merriam-webster"
    ]

    re_tokens = [
        "sqft", "sq ft", "sq.ft", "bhk", "carpet area", "price", "rate", "rera",
        "flat", "apartment", "residence", "housing", "possession", "lacs", "lakh", "cr", "crore", "/sq", "₹"
    ]

    valid_count = 0
    for r in results:
        url = (r.get("url") or "").lower()
        title = (r.get("title") or "").lower()
        snippet = (r.get("snippet") or "").lower()
        combined = f"{title} {snippet} {url}"

        # If it matches any banned domain pattern, ignore this result
        if any(bd in url for bd in banned_domain_patterns):
            continue

        # If real estate metrics exist in snippet/title, consider it valid
        if any(k in combined for k in re_tokens):
            valid_count += 1

    return valid_count >= 1


def search_web_with_fallback(
    query: str,
    max_results: int = 4,
    tavily_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    preferred_engine: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], str]:
    """
    Multi-tier web search with resilient fallback chain:
    1. Primary (1st Priority): OpenAI Web Search Tool (High-Precision Live Browsing via gpt-4o-mini)
    2. First Fallback (2nd Priority): DuckDuckGo (Free / No Key Required)
    3. Second Fallback (3rd Priority): Tavily AI Search (REST Client)

    Returns: (results_list, engine_used_name)
    """
    pref = (preferred_engine or "auto").lower()

    # User explicitly chose DuckDuckGo
    if "duckduckgo" in pref:
        try:
            res = search_duckduckgo(query, max_results=max_results)
            if res:
                return res, "DuckDuckGo"
        except Exception:
            pass
        o_key = get_openai_api_key(openai_key)
        if o_key:
            try:
                res = search_openai_web(query, api_key=o_key, max_results=max_results)
                if res:
                    return res, "OpenAI Web Search"
            except Exception:
                pass
        t_key = get_tavily_api_key(tavily_key)
        if t_key:
            try:
                res = search_tavily(query, api_key=t_key, max_results=max_results)
                if res:
                    return res, "Tavily AI Search"
            except Exception:
                pass
        return [], "None"

    # User explicitly chose Tavily
    if "tavily" in pref:
        t_key = get_tavily_api_key(tavily_key)
        if t_key:
            try:
                res = search_tavily(query, api_key=t_key, max_results=max_results)
                if res:
                    return res, "Tavily AI Search"
            except Exception:
                pass
        o_key = get_openai_api_key(openai_key)
        if o_key:
            try:
                res = search_openai_web(query, api_key=o_key, max_results=max_results)
                if res:
                    return res, "OpenAI Web Search"
            except Exception:
                pass
        try:
            res = search_duckduckgo(query, max_results=max_results)
            if res:
                return res, "DuckDuckGo"
        except Exception:
            pass
        return [], "None"

    # User explicitly chose OpenAI
    if "openai" in pref:
        o_key = get_openai_api_key(openai_key)
        if o_key:
            try:
                res = search_openai_web(query, api_key=o_key, max_results=max_results)
                if res:
                    return res, "OpenAI Web Search"
            except Exception:
                pass
        try:
            res = search_duckduckgo(query, max_results=max_results)
            if res:
                return res, "DuckDuckGo"
        except Exception:
            pass
        t_key = get_tavily_api_key(tavily_key)
        if t_key:
            try:
                res = search_tavily(query, api_key=t_key, max_results=max_results)
                if res:
                    return res, "Tavily AI Search"
            except Exception:
                pass
        return [], "None"

    # =========================================================================
    # DEFAULT STRATEGY: Priority 1: OpenAI ➔ Priority 2: DuckDuckGo ➔ Priority 3: Tavily
    # =========================================================================
    last_resort_results = []
    last_resort_engine = "None"

    # 1. Primary (1st Priority): OpenAI Web Search Tool (Fast check of persistent circuit breaker)
    if not _is_quota_exhausted("openai"):
        o_key = get_openai_api_key(openai_key)
        if o_key:
            try:
                oai_res = search_openai_web(query, api_key=o_key, max_results=max_results)
                if oai_res:
                    last_resort_results = oai_res
                    last_resort_engine = "OpenAI Web Search"
                    if is_relevant_real_estate_result(oai_res, query):
                        return oai_res, "OpenAI Web Search"
            except Exception:
                pass

    # 2. First Fallback (2nd Priority): DuckDuckGo (Free, High Concurrency)
    try:
        ddg_res = search_duckduckgo(query, max_results=max_results)
        if ddg_res:
            if not last_resort_results:
                last_resort_results = ddg_res
                last_resort_engine = "DuckDuckGo"
            if is_relevant_real_estate_result(ddg_res, query):
                return ddg_res, "DuckDuckGo"
    except Exception:
        pass

    # 3. Second Fallback (3rd Priority): Tavily AI Search
    if not _is_quota_exhausted("tavily"):
        t_key = get_tavily_api_key(tavily_key)
        if t_key:
            try:
                tav_res = search_tavily(query, api_key=t_key, max_results=max_results)
                if tav_res:
                    if not last_resort_results:
                        last_resort_results = tav_res
                        last_resort_engine = "Tavily AI Search"
                    if is_relevant_real_estate_result(tav_res, query):
                        return tav_res, "Tavily AI Search"
            except Exception:
                pass

    if last_resort_results:
        return last_resort_results, last_resort_engine

    return [], "None"
