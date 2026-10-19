"""
comparable.py - Find Comparable Real Estate Projects

Inputs:
- Latitude & Longitude OR Project Name (with geocoding)
- Radius (e.g. 2 km, 5 km, 10 km or custom)
- Property Type:
    * flat (residential flats, apartments, housing societies, condominiums)
    * shop (retail shops, commercial showrooms, shopping arcades, retail plazas)
    * office (corporate towers, business parks, commercial office spaces, IT tech parks)
    * others (mixed-use developments, commercial complexes, warehouses, industrial)

Filtering Rule:
- When "office" is selected, ONLY projects with office/commercial presence are shown. Pure residential projects are strictly excluded. If a project is mixed-use with office, it is included.
- When "shop" is selected, ONLY projects with retail/shop presence are shown. Pure residential projects are strictly excluded.
- When "flat" is selected, ONLY projects with residential flat/apartment presence are shown. Pure office/commercial projects are strictly excluded.
- When "others" is selected, commercial/industrial/mixed-use projects are shown.
"""

import os
import sys
import re
import argparse
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from utils import (
    get_google_api_key,
    haversine_distance,
    geocode_project,
    reverse_geocode,
    search_google_places_nearby,
    search_google_places_text,
)


# Google Place Types mapping
GOOGLE_RESIDENTIAL_TYPES = {
    "apartment_complex", "housing_complex", "condominium_complex",
    "residential", "housing_estate", "gated_community", "lodging"
}

GOOGLE_OFFICE_TYPES = {
    "corporate_office", "business_center", "coworking_space",
    "office", "commercial_services", "software_company", "corporate_campus"
}

GOOGLE_SHOP_TYPES = {
    "shopping_mall", "store", "department_store", "supermarket",
    "convenience_store", "clothing_store", "shoe_store", "electronics_store",
    "retail", "shopping_center"
}

# Semantic keyword indicators for dynamic detection
OFFICE_KEYWORDS = [
    r"\boffice\b", r"\bcorporate\b", r"\bbusiness park\b", r"\btech park\b",
    r"\bcyber\b", r"\binfotech\b", r"\bworkplace\b", r"\bcowork\b", r"\bco-work\b",
    r"\bbusiness center\b", r"\bbusiness centre\b", r"\btrade center\b", r"\btrade centre\b",
    r"\bcommercial office\b", r"\bexecutive suites\b", r"\boffices\b", r"\bsoftware park\b",
    r"\bit park\b", r"\bworkspace\b", r"\bcommercial tower\b"
]

SHOP_KEYWORDS = [
    r"\bshop\b", r"\bshops\b", r"\bretail\b", r"\bshowroom\b", r"\bshowrooms\b",
    r"\bmall\b", r"\barcade\b", r"\bbazaar\b", r"\bgalleria\b", r"\bplaza\b",
    r"\bmarket\b", r"\bstore\b", r"\boutlet\b", r"\bshopping center\b",
    r"\bshopping centre\b", r"\bhigh street\b", r"\bcommercial plaza\b"
]

RESIDENTIAL_KEYWORDS = [
    r"\bflat\b", r"\bflats\b", r"\bapartment\b", r"\bapartments\b", r"\bsociety\b",
    r"\bhousing society\b", r"\bchs\b", r"\bco-operative housing\b", r"\bcooperative housing\b",
    r"\bresidency\b", r"\bresidences\b", r"\bresidential\b", r"\bcondo\b", r"\bcondominium\b",
    r"\bvillas\b", r"\benclave\b", r"\btowers?\b", r"\bheights?\b", r"\bgreens?\b",
    r"\bpalms?\b", r"\bmeadows?\b", r"\bwoods?\b", r"\bhomes?\b"
]


def normalize_property_type(prop_type: str) -> str:
    """Normalize input string to one of: 'flat', 'shop', 'office', 'others'."""
    p = (prop_type or "").lower().strip()
    if "flat" in p or "res" in p or "apartment" in p or "housing" in p:
        return "flat"
    elif "shop" in p or "retail" in p or "showroom" in p or "mall" in p:
        return "shop"
    elif "office" in p or "corporate" in p or "tech" in p or "business" in p:
        return "office"
    elif "other" in p or "mixed" in p or "all" in p:
        return "others"
    return "flat"


def detect_property_attributes(place: Dict[str, Any]) -> Dict[str, bool]:
    """
    Dynamically analyze a property's name, address, and Google types
    to detect presence of Office, Shop, and Residential components.
    """
    name = (place.get("name") or "").lower()
    address = (place.get("address") or "").lower()
    full_text = f"{name} {address}"
    types_set = set(place.get("types") or [])

    # 1. Office presence detection
    has_office = bool(types_set & GOOGLE_OFFICE_TYPES) or any(
        re.search(pat, full_text, re.IGNORECASE) for pat in OFFICE_KEYWORDS
    )

    # 2. Shop presence detection
    has_shop = bool(types_set & GOOGLE_SHOP_TYPES) or any(
        re.search(pat, full_text, re.IGNORECASE) for pat in SHOP_KEYWORDS
    )

    # 3. Residential presence detection
    has_residential = bool(types_set & GOOGLE_RESIDENTIAL_TYPES) or any(
        re.search(pat, full_text, re.IGNORECASE) for pat in RESIDENTIAL_KEYWORDS
    )

    return {
        "has_office": has_office,
        "has_shop": has_shop,
        "has_residential": has_residential,
        "is_pure_residential": has_residential and not has_office and not has_shop,
        "is_pure_office": has_office and not has_residential,
        "is_pure_shop": has_shop and not has_residential,
        "is_mixed_use": (has_office or has_shop) and has_residential,
    }


def should_include_place(attrs: Dict[str, bool], target_type: str) -> Tuple[bool, str]:
    """
    Apply the user's golden rule:
    - If user selects 'office', project MUST have office. If project carrying only residential,
      it MUST NOT be shown. If project is mixed-use having office, it CAN be shown.
    - If user selects 'shop', project MUST have shop/retail. If carrying only residential,
      it MUST NOT be shown.
    - If user selects 'flat', project MUST have residential flats. Pure office/commercial cannot be shown.
    - If user selects 'others', mixed-use or commercial projects can be shown.

    Returns: (is_included: bool, display_tag: str)
    """
    t = normalize_property_type(target_type)

    if t == "office":
        # Pure residential projects MUST NOT be shown
        if attrs["is_pure_residential"]:
            return False, ""
        # Must have office presence
        if not attrs["has_office"]:
            return False, ""
        if attrs["is_mixed_use"]:
            return True, "Mixed-Use (Office & Residential)"
        return True, "Office / Corporate"

    elif t == "shop":
        # Pure residential projects MUST NOT be shown
        if attrs["is_pure_residential"]:
            return False, ""
        # Must have shop/retail presence
        if not attrs["has_shop"]:
            return False, ""
        if attrs["is_mixed_use"]:
            return True, "Mixed-Use (Retail & Residential)"
        return True, "Shop / Retail"

    elif t == "flat":
        # Pure office or pure shop projects MUST NOT be shown
        if attrs["is_pure_office"] or attrs["is_pure_shop"]:
            return False, ""
        # Must have residential presence
        if not attrs["has_residential"]:
            return False, ""
        if attrs["is_mixed_use"]:
            return True, "Mixed-Use (Residential & Commercial)"
        return True, "Flat / Residential"

    else:  # "others"
        if attrs["is_mixed_use"]:
            return True, "Mixed-Use Development"
        elif attrs["has_office"] or attrs["has_shop"]:
            return True, "Commercial Complex"
        return True, "Real Estate Development"


def clean_project_name(name: str) -> str:
    """Clean promotional taglines, marketing suffix and parentheses from building names."""
    if not name:
        return ""
    # Strip marketing suffixes like ' - Best Showroom/Offices for Sale' or '| Office Space in...'
    name = re.sub(
        r"\s*[-|–]\s*(?:Best|Premier|Luxurious|Top|Affordable|Offices? for Sale|Showroom|Office Space|Co-working|Coworking|Shared Space|Rental).*$",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    # Strip trailing parentheses with phone numbers or city
    name = re.sub(
        r"\s*\((?:Pune|Mumbai|Delhi|PCMC|Bangalore|Hyderabad|Chennai|India|Branch|\d+)\)\s*$",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    return name


def extract_building_project(name: str, address: str) -> Tuple[str, str, str]:
    """
    Identifies if a place is an individual unit/tenant or the parent building/project.
    Extracts the parent building / commercial project name and cleans out unit numbers.

    Returns:
    (project_name: str, clean_address: str, unit_name: str)
    """
    bldg_keywords = [
        "mall", "plaza", "complex", "business park", "tech park", "it park",
        "corporate park", "corporate tower", "corporate center", "corporate centre",
        "towers?", "heights?", "society", "enclave", "residency", "regency",
        "terraces?", "arcade", "square", "chambers?", "house", "campus",
        "world towers?", "center", "centre", "hub", "building", "apartments?",
        "condo", "greens?", "palms?", "woods?", "meadows?", "villas?", "court",
        "b-zone", "zone", "promenade"
    ]
    bldg_re = re.compile(r"\b(" + "|".join(bldg_keywords) + r")\b", re.IGNORECASE)

    unit_prefix_re = re.compile(
        r"^\s*(?:office|shop|unit|flat|suite|cabin|room|gf|ff|sf|plot|sr|sy|survey|ward|cts|g-flr|ground flr|first flr|second flr|1st floor|2nd floor|3rd floor|4th floor|5th floor|6th floor|gf and ff)\b[\s\d\w\-\.,#/]*",
        re.IGNORECASE,
    )

    tenant_indicators = [
        r"\bpvt\b", r"\bltd\b", r"\bllp\b", r"\bservices\b", r"\btechnologies\b",
        r"\bsolutions\b", r"\bconsulting\b", r"\brealty\b", r"\bdevelopers\b",
        r"\btraders\b", r"\bagencies\b", r"\bstore\b", r"\bsupermarket\b",
        r"\bcafe\b", r"\brestaurant\b", r"\bpizza\b", r"\bfashion\b", r"\bdoctor\b",
        r"\bclinic\b", r"\blab\b", r"\bclasses\b", r"\bacademy\b", r"\bdesk\b",
        r"\bco-working\b", r"\bcoworking\b", r"\bworkspaces?\b", r"\bdistribution house\b",
    ]
    is_tenant = any(re.search(p, name, re.IGNORECASE) for p in tenant_indicators)

    parts = [p.strip() for p in address.split(",") if p.strip()]

    # 1. If name itself is clearly a building/project and not a tenant
    if bldg_re.search(name) and not is_tenant:
        clean_name = clean_project_name(name)
        clean_addr = address
        if parts and unit_prefix_re.search(parts[0]):
            clean_addr = ", ".join(parts[1:])
        return clean_name, clean_addr, ""

    # 2. Check address parts for parent building / project name
    for i, p in enumerate(parts[:4]):
        stripped = unit_prefix_re.sub("", p).strip()
        if stripped and bldg_re.search(stripped):
            clean_addr = ", ".join(parts[i:])
            return clean_project_name(stripped), clean_addr, name
        if bldg_re.search(p) and not unit_prefix_re.match(p):
            clean_addr = ", ".join(parts[i:])
            return clean_project_name(p), clean_addr, name

    # 3. If first part had unit prefix, second part is often the building
    if len(parts) >= 2 and unit_prefix_re.search(parts[0]):
        candidate = unit_prefix_re.sub("", parts[0]).strip()
        if candidate and len(candidate) > 3 and not re.search(r"\b(road|rd|street|lane|nagar|wadi|chowk)\b", candidate, re.IGNORECASE):
            return clean_project_name(candidate), ", ".join(parts[1:]), name
        if not re.search(r"\b(road|rd|hwy|highway|lane|street|marg|chowk|bypass|nagar|sector|wadi)\b", parts[1], re.IGNORECASE):
            return clean_project_name(parts[1]), ", ".join(parts[1:]), name

    # Fallback: clean the place name
    clean_name = clean_project_name(name)
    clean_addr = address
    if parts and unit_prefix_re.search(parts[0]):
        clean_addr = ", ".join(parts[1:])
    return clean_name, clean_addr, ""


def _generate_mock_comparables(
    lat: float,
    lng: float,
    radius_km: float,
    prop_type: str = "flat",
    city: str = "Metropolitan Area",
) -> List[Dict[str, Any]]:
    """
    Generates realistic synthetic comparable projects for testing/demo
    when Google API key is not yet provided or quota is exceeded.
    """
    prop_type_key = normalize_property_type(prop_type)
    items = []

    mock_catalogs = {
        "flat": [
            ("Greenwood Heights Luxury Flats & Residences", 0.6, 4.5, 340, "2, 3 & 4 BHK Luxury Flats", "Flat / Residential"),
            ("Prestige Lakeview Apartments & Towers", 1.2, 4.4, 520, "High-rise Gated Residential Society", "Flat / Residential"),
            ("Suncrest Enclave Residential Housing Society", 2.1, 4.2, 185, "Residential Society & Club House", "Flat / Residential"),
            ("Emerald Palms Gated Condominiums", 2.8, 4.6, 610, "Premium Gated Flats & Condos", "Flat / Residential"),
            ("Oakwood Serenity 2 & 3 BHK Apartments", 3.7, 4.3, 290, "High-rise Residential Apartments", "Flat / Residential"),
            ("Grandview Royal Meadows Residential Township", 4.5, 4.1, 145, "Integrated Residential Township", "Flat / Residential"),
            ("Silver Springs Residential Complex", 6.2, 4.4, 380, "Flats, Penthouses & Sports Amenities", "Flat / Residential"),
            ("Brigade Horizon Greens Luxury Flats", 7.9, 4.6, 750, "Eco-luxury Residential Flats", "Flat / Residential"),
        ],
        "shop": [
            ("Metro Galleria High-Street Retail Shops", 0.5, 4.5, 380, "High-Footfall Commercial Retail Shops", "Shop / Retail"),
            ("Central Commercial Showrooms & Retail Plaza", 1.1, 4.4, 210, "Ground-Floor Retail Showrooms", "Shop / Retail"),
            ("Royal Arcade Commercial Shopping Complex", 1.9, 4.3, 165, "Boutiques, Brand Outlets & Shops", "Shop / Retail"),
            ("Grand Bazaar Retail Showrooms & Plazas", 2.7, 4.6, 490, "Premium Road-Facing Commercial Shops", "Shop / Retail"),
            ("Boulevard Corner High-Street Retail Shops", 3.6, 4.2, 110, "Retail Outlets & Food/Beverage Shops", "Shop / Retail"),
            ("Nexus Commercial Retail Arcade", 4.8, 4.4, 330, "Two-Level Commercial Shopping Plaza", "Shop / Retail"),
            ("City Promenade Commercial Showroom Center", 6.1, 4.5, 270, "Automobile & Fashion Retail Showrooms", "Shop / Retail"),
            ("Sovereign Market Retail Commercial Complex", 7.8, 4.3, 140, "Daily Needs & Electronics Retail Hub", "Shop / Retail"),
        ],
        "office": [
            ("Apex Corporate Tower & Office Suites", 0.8, 4.6, 420, "Grade-A Corporate IT & Office Spaces", "Office / Corporate"),
            ("Infinity Tech Park & Business Center", 1.6, 4.5, 510, "Multi-Tenant Corporate Office Campus", "Office / Corporate"),
            ("Horizon Cyber Hub & Executive Office Spaces", 2.4, 4.7, 680, "Fortune 500 Tech & Corporate Offices", "Office / Corporate"),
            ("Metropolis Corporate Center & Tech Hub", 3.2, 4.4, 340, "Commercial Office Suites & Co-Working", "Office / Corporate"),
            ("Prestige Business Point Office Campus", 4.5, 4.6, 590, "Large Floor-Plate Corporate Offices", "Office / Corporate"),
            ("Vanguard Commercial Office Park", 6.2, 4.3, 280, "IT/ITES SEZ & Corporate Towers", "Office / Corporate"),
            ("Pinnacle Financial Towers & Office Complex", 7.5, 4.5, 410, "BFSI & Professional Office Spaces", "Office / Corporate"),
            ("Elysium Corporate Boulevard", 8.8, 4.4, 190, "Grade-A Commercial Office Suites", "Office / Corporate"),
        ],
        "others": [
            ("Gateway Integrated Mixed-Use Complex", 1.0, 4.3, 210, "Mixed Retail, Commercial & Studio Spaces", "Mixed-Use Development"),
            ("Transworld Logistics & Commercial Park", 2.2, 4.2, 130, "Commercial Warehousing & Flex Space", "Commercial Complex"),
            ("Pioneer Industrial & Business Hub", 3.5, 4.4, 180, "Light Industrial & Business Center", "Commercial Complex"),
            ("Synergy Commercial Plaza & Multi-use Center", 4.6, 4.1, 95, "Mixed-Use Commercial Development", "Mixed-Use Development"),
        ],
    }

    templates = mock_catalogs.get(prop_type_key, mock_catalogs["flat"])

    for name, dist, rating, reviews, desc, tag in templates:
        if dist <= radius_km:
            d_lat = (dist / 111.0) * 0.7
            d_lng = (dist / (111.0 * 0.9)) * 0.7
            items.append({
                "name": name,
                "place_id": f"mock_prop_{abs(hash(name)) % 100000}",
                "lat": round(lat + d_lat, 5),
                "lng": round(lng + d_lng, 5),
                "distance_km": round(dist, 2),
                "address": f"Near Subject Property, {city}",
                "rating": rating,
                "user_ratings_total": reviews,
                "property_type": tag,
                "subtype_description": desc,
                "source": "Demo Mock",
                "google_maps_url": f"https://www.google.com/maps/search/?api=1&query={lat + d_lat},{lng + d_lng}",
            })

    return items


def find_comparable_projects(
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    project_name: Optional[str] = None,
    radius_km: float = 5.0,
    property_type: str = "flat",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main function to find comparable projects for a given project name or lat/long.

    Parameters:
    - lat, lng: Latitude & Longitude coordinates of subject site
    - project_name: Name of project/location if lat/long is not provided
    - radius_km: Radius in kilometers (2, 5, 10 or custom)
    - property_type: 'flat', 'shop', 'office', or 'others'
    - api_key: Google Places API key (optional; defaults to .env)
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

    radius_meters = int(radius_km * 1000)
    prop_type_key = normalize_property_type(property_type)

    comparables = []
    is_demo = False
    status_msg = ""

    # Search queries tailored per property type for Google Places API
    type_query_map = {
        "office": [
            "business park commercial complex",
            "corporate park commercial office building",
            "IT tech park office tower",
            "commercial office towers business center",
        ],
        "shop": [
            "shopping mall commercial complex",
            "commercial shopping complex retail center",
            "shopping arcade retail plaza",
            "commercial retail shopping mall",
        ],
        "flat": [
            "residential apartment society",
            "housing society residential complex",
            "gated community residential township",
            "condominium residential towers",
        ],
        "others": [
            "commercial complex business center",
            "integrated mixed use development complex",
            "logistics industrial business park",
        ],
    }

    target_queries = type_query_map.get(prop_type_key, type_query_map["flat"])

    # 2. Live Google Places API Search
    if key:
        collected = []
        seen_ids = set()
        city_name = subject_site.get("city", "")

        # A. Targeted Text Search (best for commercial / office / retail complexes)
        for q in target_queries:
            full_query = f"{q} in {city_name}" if city_name else q
            t_places, err = search_google_places_text(
                query=full_query,
                lat=lat,
                lng=lng,
                radius_meters=radius_meters,
                api_key=key,
            )
            if err:
                status_msg = err
            for p in t_places:
                pid = p.get("place_id") or p.get("name")
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    collected.append(p)

        # B. Targeted Nearby Search for additional density
        for kw in target_queries[:2]:
            n_places, _ = search_google_places_nearby(
                lat=lat,
                lng=lng,
                radius_meters=radius_meters,
                keyword=kw,
                api_key=key,
            )
            for p in n_places:
                pid = p.get("place_id") or p.get("name")
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    collected.append(p)

        # C. Filter strictly using the user's rule
        valid_comparables = []
        for p in collected:
            # Check radius boundary first
            p_dist = p.get("distance_km")
            if p_dist is None or p_dist > radius_km:
                continue

            attrs = detect_property_attributes(p)
            include, tag = should_include_place(attrs, prop_type_key)

            if include:
                p["property_type"] = tag
                valid_comparables.append(p)

        # D. Consolidate multiple units/tenants belonging to the same project/building
        consolidated_map: Dict[str, Dict[str, Any]] = {}
        for p in valid_comparables:
            proj_name, clean_addr, unit_name = extract_building_project(
                p.get("name", ""), p.get("address", "")
            )
            # Normalization key for deduplicating identical projects
            norm_key = re.sub(r"[^a-z0-9]", "", proj_name.lower())
            if not norm_key:
                norm_key = re.sub(r"[^a-z0-9]", "", (p.get("name") or "").lower())

            if norm_key not in consolidated_map:
                entry = dict(p)
                entry["name"] = proj_name
                if clean_addr:
                    entry["address"] = clean_addr
                entry["units_in_project"] = [unit_name] if unit_name else [p.get("name", "")]
                entry["unit_count"] = 1
                consolidated_map[norm_key] = entry
            else:
                existing = consolidated_map[norm_key]
                u_name = unit_name or p.get("name", "")
                if u_name and u_name not in existing["units_in_project"]:
                    existing["units_in_project"].append(u_name)
                existing["unit_count"] = len(existing["units_in_project"])

                # Keep closest distance
                if p.get("distance_km", 999) < existing.get("distance_km", 999):
                    existing["distance_km"] = p.get("distance_km")
                    existing["lat"] = p.get("lat")
                    existing["lng"] = p.get("lng")
                    existing["google_maps_url"] = p.get("google_maps_url")

                # Keep best rating & max reviews
                if (p.get("rating") or 0) > (existing.get("rating") or 0):
                    existing["rating"] = p.get("rating")
                existing["user_ratings_total"] = max(
                    existing.get("user_ratings_total") or 0, p.get("user_ratings_total") or 0
                )

        comparables = list(consolidated_map.values())
        comparables.sort(key=lambda x: x.get("distance_km", 999))

        if not comparables:
            if status_msg:
                is_demo = True
                comparables = _generate_mock_comparables(lat, lng, radius_km, prop_type_key, subject_site.get("city", "City Area"))
                status_msg += " (Showing demo mock comparables for illustration)."
            else:
                status_msg = f"Zero live Google Places results found within {radius_km} km for property type '{prop_type_key}'."
        else:
            status_msg = f"Successfully fetched {len(comparables)} verified {prop_type_key} projects from Google Places API."

    else:
        # No API key provided -> Return mock demonstration data
        is_demo = True
        comparables = _generate_mock_comparables(lat, lng, radius_km, prop_type_key, subject_site.get("city", "City Area"))
        status_msg = "Google Places API key not found in .env. Showing simulated comparable properties. Add GOOGLE_PLACES_API_KEY in .env for live Google data."

    label_map = {
        "flat": "Flat (Residential Apartments, Societies, Condos)",
        "shop": "Shop (Retail Shops, Showrooms, Plazas)",
        "office": "Office (Corporate Towers, IT Parks, Business Centers)",
        "others": "Others (Mixed-Use, Commercial & Industrial)",
    }

    return {
        "subject_site": subject_site,
        "radius_km": radius_km,
        "property_type": label_map.get(prop_type_key, prop_type_key.title()),
        "property_type_key": prop_type_key,
        "total_found": len(comparables),
        "comparables": comparables,
        "is_demo_data": is_demo,
        "status_message": status_msg,
    }


def to_dataframe(comparables: List[Dict[str, Any]]) -> pd.DataFrame:
    """Helper to convert comparable projects list into a pandas DataFrame."""
    if not comparables:
        return pd.DataFrame(columns=["Name", "Distance (km)", "Type", "Units Consolidated", "Rating", "Reviews", "Address", "Google Maps"])

    rows = []
    for p in comparables:
        u_count = p.get("unit_count", 1)
        units_str = f"Consolidated ({u_count} units)" if u_count > 1 else "Building / Complex"
        rows.append({
            "Name": p.get("name", ""),
            "Distance (km)": p.get("distance_km", 0.0),
            "Type": p.get("property_type", ""),
            "Units Consolidated": units_str,
            "Rating": p.get("rating", 0.0),
            "Reviews": p.get("user_ratings_total", 0),
            "Address": p.get("address", ""),
            "Latitude": p.get("lat"),
            "Longitude": p.get("lng"),
            "Google Maps": p.get("google_maps_url", ""),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(by="Distance (km)").reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find Comparable Projects using Google Places API")
    parser.add_argument("--project", type=str, help="Project name or address")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lng", type=float, help="Longitude")
    parser.add_argument("--radius", type=float, default=5.0, help="Radius in km (e.g. 2, 5, 10)")
    parser.add_argument("--type", type=str, default="office", choices=["flat", "shop", "office", "others"], help="Property type")
    args = parser.parse_args()

    if not args.project and (args.lat is None or args.lng is None):
        print("Usage: python comparable.py --project 'Pimple Saudagar, Pune' --radius 5 --type office")
        sys.exit(1)

    result = find_comparable_projects(
        lat=args.lat,
        lng=args.lng,
        project_name=args.project,
        radius_km=args.radius,
        property_type=args.type,
    )

    print("\n" + "=" * 65)
    print(f"Subject Site: {result['subject_site']['name']}")
    print(f"Coordinates: {result['subject_site']['lat']}, {result['subject_site']['lng']}")
    print(f"Address: {result['subject_site']['address']}")
    print(f"Search Radius: {result['radius_km']} km | Property Type: {result['property_type']}")
    print(f"Status: {result['status_message']}")
    print("=" * 65)

    df = to_dataframe(result["comparables"])
    if not df.empty:
        try:
            print(df[["Name", "Distance (km)", "Type", "Rating", "Address"]].to_string(index=False))
        except UnicodeEncodeError:
            print(df[["Name", "Distance (km)", "Type", "Rating", "Address"]].to_string(index=False).encode("ascii", errors="replace").decode())
    else:
        print("No comparable projects found matching criteria.")
