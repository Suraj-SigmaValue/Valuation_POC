"""
neighborehood_amenity.py - Fetch and Analyze Neighborhood Amenities

Inputs:
- Latitude & Longitude OR Project Name (with geocoding)
- Radius (e.g. 2 km, 5 km, 10 km or custom)
- Amenity Categories (Schools, Hospitals, Transit/Metro, Supermarkets, Parks, Banks, Dining)

Uses Google Places API Nearby Search to fetch amenities within the chosen radius.
Calculates distances, ratings, and amenity density metrics.
Includes mock fallback data when API key is not yet set in .env.
"""

import os
import sys
import re
import argparse
from typing import Dict, List, Optional, Any
import pandas as pd

from utils import (
    get_google_api_key,
    haversine_distance,
    geocode_project,
    reverse_geocode,
    search_google_places_nearby,
)


AMENITY_CATEGORIES = {
    "education": {
        "label": "Education & Schools",
        "icon": "🏫",
        "types": ["school", "secondary_school", "university"],
        "keywords": ["school", "international school", "college", "university", "academy"],
    },
    "healthcare": {
        "label": "Hospitals & Healthcare",
        "icon": "🏥",
        "types": ["hospital", "pharmacy", "doctor"],
        "keywords": ["hospital", "multispeciality hospital", "clinic", "medical center", "pharmacy"],
    },
    "metro": {
        "label": "Metro Stations",
        "icon": "🚇",
        "types": ["subway_station", "light_rail_station"],
        "keywords": ["metro station", "metro train station", "metro stop"],
    },
    "bus": {
        "label": "Bus Stops & Stands",
        "icon": "🚌",
        "types": ["bus_station", "bus_stop"],
        "keywords": ["bus stop", "bus stand", "bus station", "BRT bus stop"],
    },
    "railway": {
        "label": "Railway Stations",
        "icon": "🚆",
        "types": ["train_station"],
        "keywords": ["railway station", "train station", "railway junction"],
    },
    "shopping": {
        "label": "Shopping & Supermarkets",
        "icon": "🛒",
        "types": ["shopping_mall", "supermarket", "department_store"],
        "keywords": ["shopping mall", "supermarket", "hypermarket", "grocery store"],
    },
    "parks": {
        "label": "Parks & Recreation",
        "icon": "🌳",
        "types": ["park", "gym"],
        "keywords": ["park", "public garden", "gym", "sports club", "recreation ground"],
    },
    "banking": {
        "label": "Banks & Financial Services",
        "icon": "🏦",
        "types": ["bank", "atm"],
        "keywords": ["bank", "atm", "financial center"],
    },
    "dining": {
        "label": "Dining & Cafes",
        "icon": "🍽️",
        "types": ["restaurant", "cafe"],
        "keywords": ["restaurant", "cafe", "fine dining", "bistro"],
    },
}

# Unwanted non-transit types returned inadvertently by Google Places
TRANSIT_EXCLUDE_TYPES = {
    "car_repair", "car_dealer", "car_wash", "auto_repair",
    "restaurant", "cafe", "bar", "food", "meal_takeaway", "meal_delivery",
    "store", "clothing_store", "lodging", "electronics_store", "gas_station"
}

TRANSIT_EXCLUDE_WORDS = re.compile(
    r"\b(garage|workshop|repairs?|motors?|service center|automobile|sandwich|pizza|bakery|sweets|hotel|resort|lodge)\b",
    re.IGNORECASE,
)


def is_valid_transit_amenity(place: Dict[str, Any], cat_key: str) -> bool:
    """Validate transit places to eliminate false positives like car garages and fast-food chains."""
    types = set(place.get("types") or [])
    name = place.get("name") or ""

    # Immediate exclusion of car repair / food chains
    if types & TRANSIT_EXCLUDE_TYPES or TRANSIT_EXCLUDE_WORDS.search(name):
        return False

    name_lower = name.lower()

    if cat_key == "metro":
        # Must be an actual metro station / subway station
        if "bus" in name_lower and "metro" not in name_lower:
            return False
        return bool(
            "subway_station" in types
            or "light_rail_station" in types
            or "metro" in name_lower
            or "subway station" in name_lower
        )

    elif cat_key == "bus":
        # Must be an actual bus stop / bus station / BRT
        if "metro" in name_lower or "subway_station" in types:
            return False
        return bool(
            "bus_station" in types
            or "bus_stop" in types
            or "bus" in name_lower
            or "brt" in name_lower
            or "stand" in name_lower
            or "depot" in name_lower
        )

    elif cat_key == "railway":
        # Must be a train / railway station
        if "metro" in name_lower or "bus" in name_lower:
            return False
        return bool(
            "train_station" in types
            or "railway" in name_lower
            or "train" in name_lower
            or "junction" in name_lower
            or "terminus" in name_lower
        )

    return True


def _generate_mock_amenities(
    lat: float,
    lng: float,
    radius_km: float,
    selected_categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generates realistic mock amenities for testing/demo when Google API key
    is not yet entered or has hit quota limit.
    """
    cats = selected_categories or list(AMENITY_CATEGORIES.keys())
    mock_data_catalog = {
        "education": [
            ("St. Xavier's International Academy", "school", 0.7, 4.7, 320, "CBSE & IB Curriculum (K-12)"),
            ("Greenwood World School", "school", 1.4, 4.5, 210, "International Baccalaureate World School"),
            ("City Institute of Technology & Management", "university", 2.8, 4.3, 540, "Autonomous Engineering & Management Campus"),
            ("Oakridge High School", "school", 3.9, 4.6, 180, "Pre-primary to Grade 12 Campus"),
        ],
        "healthcare": [
            ("Apex Super-Speciality Hospital", "hospital", 0.9, 4.6, 850, "24x7 Emergency, Trauma, & ICU"),
            ("LifeCare Multi-speciality Clinic", "hospital", 1.5, 4.3, 190, "OPD, Diagnostics & Specialty Clinics"),
            ("Apollo Pharmacy & Wellness", "pharmacy", 0.4, 4.4, 120, "24-Hour Chemist & Diagnostics"),
            ("Fortis Health Institute", "hospital", 3.2, 4.7, 1200, "JCI & NABH Accredited Quaternary Care"),
        ],
        "metro": [
            ("Central Metro Station (Line 3)", "subway_station", 0.6, 4.5, 620, "Underground Rapid Transit Hub"),
            ("Tech Park Elevated Metro Station", "subway_station", 1.8, 4.4, 480, "Rapid Transit Elevated Station"),
        ],
        "bus": [
            ("Express City Bus Terminal & BRTS", "bus_station", 0.5, 4.1, 140, "Inter-city and feeder bus lines"),
            ("Main Chowk Public Bus Stop", "bus_stop", 0.3, 4.0, 85, "Local municipal bus transit point"),
        ],
        "railway": [
            ("Junction Railway Station", "train_station", 3.5, 4.2, 2300, "Suburban and regional rail connection"),
            ("City Suburban Rail Terminus", "train_station", 5.2, 4.1, 1150, "Express and suburban trains"),
        ],
        "shopping": [
            ("Phoenix Grand Galleria Mall", "shopping_mall", 1.3, 4.6, 4200, "Multiplex, Anchor Retail & Food Court"),
            ("Nature's Basket Gourmet Supermarket", "supermarket", 0.5, 4.4, 310, "Daily Essentials & Organic Groceries"),
            ("Metro Walk Retail Plaza", "shopping_mall", 2.6, 4.3, 780, "Apparel, Electronics & High-street Brands"),
        ],
        "parks": [
            ("Botanical Garden & Joggers Park", "park", 0.8, 4.6, 450, "Lush greenery, walking tracks & children play area"),
            ("Gold's Fitness & Crossfit Center", "gym", 0.5, 4.5, 290, "Cardio, Strength & Aerobics Gym"),
            ("City Waterfront Promenade Park", "park", 2.2, 4.7, 980, "Lakefront walkway and sports zone"),
        ],
        "banking": [
            ("HDFC Bank & 24hr ATM", "bank", 0.3, 4.2, 95, "Retail Banking, Foreign Exchange & Lockers"),
            ("State Bank of India Main Branch", "bank", 0.9, 4.0, 160, "Commercial & Retail Branch with ATM"),
            ("ICICI Bank Wealth Management Branch", "bank", 1.2, 4.3, 110, "Priority Banking Center"),
        ],
        "dining": [
            ("The Olive Bistro & Cafe", "cafe", 0.4, 4.6, 480, "Artisan Coffee, Bakery & Continental Dining"),
            ("Copper Chimney Fine Dining", "restaurant", 1.0, 4.4, 620, "North Indian & Mughlai Specialties"),
            ("Starbucks Reserve", "cafe", 1.2, 4.5, 750, "Specialty Coffee & Workspace"),
        ],
    }

    results = []
    for cat_key in cats:
        items = mock_data_catalog.get(cat_key, [])
        cat_meta = AMENITY_CATEGORIES.get(cat_key, {"label": cat_key.title(), "icon": "📍"})
        for name, p_type, dist, rating, reviews, desc in items:
            if dist <= radius_km:
                d_lat = (dist / 111.0) * 0.65
                d_lng = (dist / (111.0 * 0.85)) * 0.65
                results.append({
                    "name": name,
                    "category": cat_key,
                    "category_label": cat_meta["label"],
                    "icon": cat_meta["icon"],
                    "place_id": f"mock_amenity_{abs(hash(name)) % 100000}",
                    "lat": round(lat + d_lat, 5),
                    "lng": round(lng + d_lng, 5),
                    "distance_km": round(dist, 2),
                    "address": f"Near Subject Property (Demo Data)",
                    "rating": rating,
                    "user_ratings_total": reviews,
                    "primary_type": p_type,
                    "description": desc,
                    "source": "Demo Mock (Add GOOGLE_PLACES_API_KEY in .env for live Google Places)",
                    "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={lat + d_lat},{lng + d_lng}",
                })

    return results


def fetch_neighborhood_amenities(
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    project_name: Optional[str] = None,
    radius_km: float = 3.0,
    categories: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch amenities around a subject site for the given radius and categories.

    Parameters:
    - lat, lng: Subject coordinates
    - project_name: Project name / location string if coordinates are not provided
    - radius_km: Search radius in kilometers (2, 5, 10 or custom)
    - categories: List of category keys to fetch (defaults to all)
    - api_key: Google Places API key (optional)

    Returns:
    - subject_site: {name, lat, lng, address, city}
    - radius_km: float
    - total_found: int
    - amenities: List[Dict]
    - category_summary: Dict with counts and nearest distances per category
    - is_demo_data: bool
    - status_message: str
    """
    key = get_google_api_key(api_key)

    # 1. Resolve coordinates & subject site
    subject_site = {}
    if project_name and (lat is None or lng is None):
        geocoded = geocode_project(project_name, api_key=key)
        lat = geocoded["lat"]
        lng = geocoded["lng"]
        subject_site = {
            "name": geocoded.get("name", project_name),
            "lat": lat,
            "lng": lng,
            "address": geocoded.get("formatted_address", ""),
            "city": geocoded.get("city", "Unknown City"),
        }
    elif lat is not None and lng is not None:
        rev = reverse_geocode(lat, lng, api_key=key)
        subject_site = {
            "name": project_name or f"Subject Site ({lat:.4f}, {lng:.4f})",
            "lat": lat,
            "lng": lng,
            "address": rev.get("formatted_address", ""),
            "city": rev.get("city", "Unknown City"),
        }
    else:
        raise ValueError("Must provide either (lat, lng) or project_name.")

    target_cats = categories or list(AMENITY_CATEGORIES.keys())
    radius_meters = int(radius_km * 1000)

    amenities = []
    is_demo = False
    status_msg = ""
    seen_place_ids = set()

    # 2. Query Google Places API if key available
    if key:
        collected = []
        for cat_key in target_cats:
            cat_cfg = AMENITY_CATEGORIES.get(cat_key)
            if not cat_cfg:
                continue

            # Query primary types/keywords
            for kw in cat_cfg["keywords"][:2]:
                places, err = search_google_places_nearby(
                    lat=lat,
                    lng=lng,
                    radius_meters=radius_meters,
                    keyword=kw,
                    api_key=key,
                )
                if err:
                    status_msg = err
                for p in places:
                    pid = p.get("place_id") or p.get("name")
                    if pid not in seen_place_ids and p.get("distance_km") is not None and p["distance_km"] <= radius_km:
                        # Validate transit amenities to filter out false positives (e.g. car garages, food chains)
                        if cat_key in ("metro", "bus", "railway") and not is_valid_transit_amenity(p, cat_key):
                            continue
                        seen_place_ids.add(pid)
                        p["category"] = cat_key
                        p["category_label"] = cat_cfg["label"]
                        p["icon"] = cat_cfg["icon"]
                        collected.append(p)

        if collected:
            collected.sort(key=lambda x: x.get("distance_km", 999))
            amenities = collected
            status_msg = f"Successfully fetched {len(amenities)} live amenities from Google Places API."
        else:
            # Fallback to mock
            is_demo = True
            amenities = _generate_mock_amenities(lat, lng, radius_km, target_cats)
            status_msg = "No live amenities found or API limit reached. Displaying representative mock amenities."

    else:
        # No API key provided -> Return mock demonstration data
        is_demo = True
        amenities = _generate_mock_amenities(lat, lng, radius_km, target_cats)
        status_msg = "Google Places API key not found in .env. Showing simulated neighborhood amenities. Add GOOGLE_PLACES_API_KEY in .env to query live Google Places."

    # 3. Calculate category summary statistics
    cat_summary = {}
    for cat_key in target_cats:
        cat_meta = AMENITY_CATEGORIES.get(cat_key, {"label": cat_key.title(), "icon": "📍"})
        cat_items = [a for a in amenities if a.get("category") == cat_key]
        if cat_items:
            nearest = min(cat_items, key=lambda x: x.get("distance_km", 999))
            avg_rating = round(sum(a.get("rating", 0) for a in cat_items if a.get("rating", 0) > 0) / max(1, len([a for a in cat_items if a.get("rating", 0) > 0])), 1)
            cat_summary[cat_key] = {
                "label": cat_meta["label"],
                "icon": cat_meta["icon"],
                "count": len(cat_items),
                "nearest_name": nearest["name"],
                "nearest_distance_km": nearest["distance_km"],
                "avg_rating": avg_rating if avg_rating > 0 else "N/A",
            }
        else:
            cat_summary[cat_key] = {
                "label": cat_meta["label"],
                "icon": cat_meta["icon"],
                "count": 0,
                "nearest_name": "None found within radius",
                "nearest_distance_km": None,
                "avg_rating": "N/A",
            }

    return {
        "subject_site": subject_site,
        "radius_km": radius_km,
        "total_found": len(amenities),
        "amenities": amenities,
        "category_summary": cat_summary,
        "is_demo_data": is_demo,
        "status_message": status_msg,
    }


def to_dataframe(amenities: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert amenities list to pandas DataFrame."""
    if not amenities:
        return pd.DataFrame(columns=["Category", "Name", "Distance (km)", "Rating", "Reviews", "Address", "Google Maps"])

    rows = []
    for a in amenities:
        rows.append({
            "Category": f"{a.get('icon', '')} {a.get('category_label', '')}",
            "Name": a.get("name", ""),
            "Distance (km)": a.get("distance_km", 0.0),
            "Rating": a.get("rating", 0.0),
            "Reviews": a.get("user_ratings_total", 0),
            "Address": a.get("address", ""),
            "Latitude": a.get("lat"),
            "Longitude": a.get("lng"),
            "Google Maps": a.get("google_maps_url", ""),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(by="Distance (km)").reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Neighborhood Amenities")
    parser.add_argument("--project", type=str, help="Project name or address")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lng", type=float, help="Longitude")
    parser.add_argument("--radius", type=float, default=3.0, help="Radius in km (e.g. 2, 5, 10)")
    args = parser.parse_args()

    if not args.project and (args.lat is None or args.lng is None):
        print("Usage: python neighborehood_amenity.py --project 'Hiranandani Gardens Powai' --radius 3")
        print("   or: python neighborehood_amenity.py --lat 19.1197 --lng 72.9056 --radius 3")
        sys.exit(1)

    result = fetch_neighborhood_amenities(
        lat=args.lat,
        lng=args.lng,
        project_name=args.project,
        radius_km=args.radius,
    )

    print("\n" + "=" * 60)
    print(f"Subject Site: {result['subject_site']['name']}")
    print(f"Coordinates: {result['subject_site']['lat']}, {result['subject_site']['lng']}")
    print(f"Radius: {result['radius_km']} km | Total Amenities Found: {result['total_found']}")
    print(f"Status: {result['status_message']}")
    print("=" * 60)

    print("\nCategory Summary Breakdown:")
    for cat, info in result["category_summary"].items():
        dist_str = f"{info['nearest_distance_km']} km" if info['nearest_distance_km'] is not None else "N/A"
        icon_display = info.get('icon', '')
        try:
            print(f"  [{icon_display}] {info['label']:<28} | Count: {info['count']:<2} | Nearest: {info['nearest_name']} ({dist_str})")
        except UnicodeEncodeError:
            print(f"  * {info['label']:<28} | Count: {info['count']:<2} | Nearest: {info['nearest_name']} ({dist_str})")

    df = to_dataframe(result["amenities"])
    if not df.empty:
        print("\nTop Amenities:")
        try:
            print(df[["Category", "Name", "Distance (km)", "Rating", "Address"]].head(10).to_string(index=False))
        except UnicodeEncodeError:
            print(df[["Name", "Distance (km)", "Rating", "Address"]].head(10).to_string(index=False))
