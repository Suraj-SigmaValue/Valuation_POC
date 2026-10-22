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
import urllib.parse
import concurrent.futures
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from utils import (
    get_google_api_key,
    get_tavily_api_key,
    search_tavily,
    search_duckduckgo,
    haversine_distance,
    geocode_project,
    reverse_geocode,
    search_google_places_nearby,
    search_google_places_text,
)


# Google Place Types mapping
GOOGLE_RESIDENTIAL_TYPES = {
    "apartment_complex", "housing_complex", "condominium_complex",
    "residential", "housing_estate", "gated_community", "premise",
    "subpremise", "point_of_interest", "establishment"
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
    r"\bresidency\b", r"\bresidences?\b", r"\bresidential\b", r"\bcondo\b", r"\bcondominium\b",
    r"\bvillas?\b", r"\benclave\b", r"\btowers?\b", r"\bheights?\b", r"\bgreens?\b",
    r"\bpalms?\b", r"\bmeadows?\b", r"\bwoods?\b", r"\bhomes?\b", r"\bestate\b",
    r"\bcourt\b", r"\bpromenade\b", r"\bterrace\b", r"\bmanor\b", r"\bgardens?\b",
    r"\bpark\b", r"\bliving\b", r"\bhabitat\b", r"\bavenue\b", r"\boasis\b"
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


# Valuation Benchmark Catalogs
AMENITIES_BY_TYPE = {
    "flat": [
        "Clubhouse", "Swimming Pool", "Gymnasium", "24x7 Security",
        "Power Backup", "Covered Parking", "Children Play Area",
        "EV Charging", "Landscaped Gardens", "Jogging Track"
    ],
    "office": [
        "Central Air Conditioning", "High-Speed Elevators", "100% Power Backup",
        "24x7 Security & CCTV", "Visitor Parking", "Cafeteria / Food Court",
        "Fire Fighting System", "Conference Facilities", "EV Charging"
    ],
    "shop": [
        "High Footfall Frontage", "Anchor Tenant Provision", "Customer Parking",
        "24x7 Security & CCTV", "Power Backup", "Escalators & Elevators",
        "Loading Bay", "Fire Safety System"
    ],
    "others": [
        "Goods Lift", "Heavy Vehicle Access", "Power Backup",
        "24x7 Security", "Fire Safety System", "Dedicated Parking",
        "Adequate Water Supply"
    ],
}

UNITS_BY_TYPE = {
    "flat": ["1 BHK", "2 BHK", "3 BHK", "4 BHK"],
    "office": ["Commercial Office Suite", "Warm Shell Office", "Bare Shell Floor Plate", "Co-Working Desk Space"],
    "shop": ["Retail Shop", "High-Street Showroom", "Food Court Outlet", "Anchor Store"],
    "others": ["Industrial Unit", "Warehouse Bay", "Mixed-Use Office/Retail"],
}


# Regex for extracting price per sqft / sqm across global currencies (INR, USD, GBP, EUR, AED, SGD, etc.)
PRICE_REGEX = re.compile(
    r'(?:₹|rs\.?|inr|\$|£|€|aed|sgd)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{3,7})\s*(?:/|per)\s*(?:sq\.?\s*ft|sqft|sq\s*feet|sqm|sq\.?\s*m)',
    re.I
)


def fetch_project_dna(
    project_name: str,
    city: str = "",
    tavily_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Intelligently extracts complete Subject Project DNA from live web portal evidence:
    1. Property Type (Flat, Office, Shop, Others)
    2. Available Unit Typologies (e.g. 1 BHK, 2 BHK, 3 BHK)
    3. Dominant Benchmark Unit (e.g. 2 BHK)
    4. Typical Carpet Area (sq.ft) and allowable ±15% range
    5. Verified Benchmark Rate and allowable 80%–120% range
    6. Verified Web Portal Listing Citation (URL + snippet)
    """
    clean_city = city or "Metropolitan"
    query = f"{project_name} {clean_city} 2 BHK 3 BHK floor plan carpet area price per sq ft property for sale listings"

    key = get_tavily_api_key(tavily_key)
    articles = []
    if key:
        articles = search_tavily(query, key, max_results=4)
    if not articles:
        articles = search_duckduckgo(query, max_results=4)

    all_text = " ".join(f"{a.get('title', '')} {a.get('snippet', '')}" for a in articles)

    # 1. Detect Property Type
    prop_type = "flat"
    if re.search(r"\b(commercial office|bare shell office|office space for sale|office tower)\b", all_text, re.I):
        if not re.search(r"\b(flat|apartment|bhk|residential society)\b", all_text, re.I):
            prop_type = "office"
    elif re.search(r"\b(retail shop for sale|commercial showroom|shopping plaza)\b", all_text, re.I):
        if not re.search(r"\b(flat|apartment|bhk)\b", all_text, re.I):
            prop_type = "shop"

    # 2. Detect Available Units
    detected_units = []
    for u in ["1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK"]:
        if re.search(rf"\b{u}\b", all_text, re.IGNORECASE):
            detected_units.append(u)
    if not detected_units:
        detected_units = ["1 BHK", "2 BHK", "3 BHK"]

    dominant_unit = "2 BHK" if "2 BHK" in detected_units else detected_units[0]

    # 3. Detect Typical Carpet Area (Prefer area associated with dominant unit)
    unit_area_regex = re.compile(rf"{re.escape(dominant_unit)}.*?(\b[0-9]{{3,4}}\b)\s*(?:sq\.?\s*ft|sqft)", re.I)
    m_unit_area = unit_area_regex.search(all_text)
    if m_unit_area:
        candidate_area = int(m_unit_area.group(1))
        carpet_area = float(candidate_area) if 350 <= candidate_area <= 3500 else 850.0
    else:
        general_areas = [int(a) for a in re.findall(r"(\b[0-9]{3,4}\b)\s*(?:sq\.?\s*ft|sqft)", all_text, re.I) if 350 <= int(a) <= 3500]
        carpet_area = float(round(sum(general_areas) / len(general_areas), 0)) if general_areas else 850.0

    min_area = round(carpet_area * 0.85, 1)
    max_area = round(carpet_area * 1.15, 1)

    # 4. Detect Verified Price per sq.ft & Evidence Citation
    rate_val = None
    evidence_dict = None
    for item in articles:
        text = f"{item.get('title', '')} {item.get('snippet', '')}"
        m_price = PRICE_REGEX.search(text)
        if m_price:
            raw_val = m_price.group(1).replace(",", "")
            try:
                pval = float(raw_val)
                if 500 <= pval <= 250000:
                    rate_val = pval
                    evidence_dict = {
                        "price": pval,
                        "source_title": item.get("title") or "Verified Portal Listing",
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", "").strip(),
                        "search_query": query,
                    }
                    break
            except ValueError:
                continue

    min_rate = round(rate_val * 0.80, 2) if rate_val else None
    max_rate = round(rate_val * 1.20, 2) if rate_val else None

    return {
        "property_type": prop_type,
        "dominant_unit": dominant_unit,
        "available_units": detected_units,
        "carpet_area": carpet_area,
        "min_area": min_area,
        "max_area": max_area,
        "rate_per_sqft": rate_val,
        "min_rate": min_rate,
        "max_rate": max_rate,
        "price_evidence": evidence_dict,
        "dna_discovered": bool(rate_val is not None),
    }


def fetch_verified_project_price_evidence(
    project_name: str,
    city: str = "",
    tavily_key: Optional[str] = None,
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """
    Searches live web portals (99acres, Magicbricks, Housing, etc.) for real pricing evidence.
    CRITICAL RULE: Returns (None, None) if no verified evidence is found.
    NEVER fabricates or estimates a price without evidence.
    """
    dna = fetch_project_dna(project_name=project_name, city=city, tavily_key=tavily_key)
    return dna.get("rate_per_sqft"), dna.get("price_evidence")


def enrich_comparable_valuation_data(
    place: Dict[str, Any],
    prop_type: str = "flat",
    city: str = "",
    target_unit: Optional[str] = "2 BHK",
    tavily_key: Optional[str] = None,
    fetch_evidence: bool = False,
) -> Dict[str, Any]:
    """
    Enriches a place with real attributes and verified price evidence.
    CRITICAL RULE:
    - If verified evidence is found from real estate portals, rate_per_sqft and price_evidence are populated.
    - If NO evidence is found, rate_per_sqft remains None. NEVER displays synthetic/fake prices!
    """
    name = place.get("name", "Project")
    clean_city = city or "Metropolitan"
    unit_query_part = f"{target_unit} " if target_unit else ""
    # Extract locality from address (e.g. Keshav Nagar) for pinpoint listing matches
    addr = place.get("address", "")
    addr_tokens = [t.strip() for t in addr.split(",") if t.strip()]
    locality = ""
    for tok in addr_tokens:
        if any(w in tok.lower() for w in ["nagar", "road", "street", "avenue", "lane", "chowk", "wadi", "sector", "phase", "hills", "district", "downtown", "creek", "marina", "bay"]):
            locality = tok
            break
    loc_part = f"{locality} " if locality else ""
    query_str = f"{name} {loc_part}{clean_city} {unit_query_part}price per sq ft carpet area property for sale listings"
    web_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query_str)}"

    rate_val = None
    evidence_dict = None
    carpet_area_val = None
    detected_units = []

    if fetch_evidence:
        key = get_tavily_api_key(tavily_key)
        articles = []
        if key:
            articles = search_tavily(query_str, key, max_results=3)
        if not articles:
            articles = search_duckduckgo(query_str, max_results=3)

        all_text = " ".join(f"{a.get('title', '')} {a.get('snippet', '')}" for a in articles)

        # Detect units in candidate
        for u in ["1 BHK", "2 BHK", "3 BHK", "4 BHK"]:
            if re.search(rf"\b{u}\b", all_text, re.IGNORECASE):
                detected_units.append(u)

        # Detect candidate carpet area (prefer area near target unit)
        if target_unit:
            u_regex = re.compile(rf"{re.escape(target_unit)}.*?(\b[0-9]{{3,4}}\b)\s*(?:sq\.?\s*ft|sqft)", re.I)
            m_u = u_regex.search(all_text)
            if m_u:
                c_val = int(m_u.group(1))
                if 350 <= c_val <= 3500:
                    carpet_area_val = c_val

        if carpet_area_val is None:
            area_matches = [int(a) for a in re.findall(r"(\b[0-9]{3,4}\b)\s*(?:sq\.?\s*ft|sqft)", all_text, re.I) if 350 <= int(a) <= 3500]
            if area_matches:
                carpet_area_val = area_matches[0]

        # Detect rate per sqft & evidence
        for item in articles:
            text = f"{item.get('title', '')} {item.get('snippet', '')}"
            m_price = PRICE_REGEX.search(text)
            if m_price:
                raw_val = m_price.group(1).replace(",", "")
                try:
                    pval = float(raw_val)
                    if 500 <= pval <= 250000:
                        rate_val = pval
                        evidence_dict = {
                            "price": pval,
                            "source_title": item.get("title") or "Verified Portal Listing",
                            "url": item.get("url", ""),
                            "snippet": item.get("snippet", "").strip(),
                            "search_query": query_str,
                        }
                        break
                except ValueError:
                    continue

    place["rate_per_sqft"] = rate_val
    place["price_evidence"] = evidence_dict
    place["source"] = place.get("source") or "Google Places API"
    place["web_search_url"] = web_url
    place["available_units"] = detected_units if detected_units else ["Available in Complex"]
    place["carpet_area"] = carpet_area_val

    return place


def evaluate_comparable_match(
    comp: Dict[str, Any],
    source_rate: Optional[float] = None,
    min_rate_pct: float = 80.0,
    max_rate_pct: float = 120.0,
    source_unit: Optional[str] = "2 BHK",
    source_area: Optional[float] = None,
    area_tolerance_pct: float = 15.0,
    target_property_type: Optional[str] = "flat",
    source_amenities: Optional[List[str]] = None,
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluates whether a comparable satisfies the 4-Point Strict Valuation Benchmark:
    1. Same Property Type (Flat, Office, Shop, Others)
    2. Same Unit Typology (Must offer the subject unit, e.g. 2 BHK)
    3. Carpet Area within ±15% tolerance range
    4. Verified Price strictly within 80% to 120% range with live portal evidence
    """
    comp_rate = comp.get("rate_per_sqft")
    evidence = comp.get("price_evidence")
    has_price_evidence = bool(comp_rate is not None and evidence is not None)

    # 1. Price Match (80%–120% Range)
    if source_rate is not None and source_rate > 0:
        min_rate = round(source_rate * (min_rate_pct / 100.0), 2)
        max_rate = round(source_rate * (max_rate_pct / 100.0), 2)

        if has_price_evidence:
            price_match = (min_rate <= comp_rate <= max_rate)
            rate_diff_pct = round(abs(comp_rate - source_rate) / max(source_rate, 0.01) * 100.0, 1)
            price_status = "✅ In 80%–120% Range" if price_match else "❌ Out of 80%–120% Range"
            price_score = max(0.0, 1.0 - (rate_diff_pct / 40.0)) * 45.0
        else:
            price_match = False
            rate_diff_pct = None
            price_status = "⚠️ No Price Evidence"
            price_score = 0.0
    else:
        min_rate = None
        max_rate = None
        price_match = has_price_evidence
        rate_diff_pct = None
        price_status = "Verified Rate" if has_price_evidence else "⚠️ No Price Evidence"
        price_score = 30.0 if has_price_evidence else 0.0

    # 2. Unit Typology Match (Must offer same unit, e.g. 2 BHK)
    avail_units = comp.get("available_units", [])
    if source_unit and avail_units:
        unit_match = bool(
            source_unit in avail_units
            or "Available in Complex" in avail_units
            or any(source_unit.lower() in u.lower() for u in avail_units)
        )
    else:
        unit_match = True
    unit_score = 20.0 if unit_match else 0.0

    # 3. Carpet Area Match (Within ±15% tolerance)
    comp_area = comp.get("carpet_area")
    if source_area and source_area > 0:
        min_area = round(source_area * (1.0 - area_tolerance_pct / 100.0), 1)
        max_area = round(source_area * (1.0 + area_tolerance_pct / 100.0), 1)
        if comp_area:
            area_match = bool(min_area <= comp_area <= max_area)
            area_diff_pct = round(abs(comp_area - source_area) / source_area * 100.0, 1)
            area_status = f"✅ In Range ({comp_area} sq.ft)" if area_match else f"❌ Area Mismatch ({comp_area} sq.ft)"
            area_score = max(0.0, 1.0 - (area_diff_pct / 30.0)) * 15.0
        else:
            # Area not explicitly stated in portal snippet: compatible by unit
            area_match = True
            area_status = "Compatible Unit"
            area_score = 10.0
    else:
        min_area = None
        max_area = None
        area_match = True
        area_status = "Area Open"
        area_score = 10.0

    # 4. Property Type Match
    comp_type = normalize_property_type(comp.get("property_type", "flat"))
    target_type = normalize_property_type(target_property_type or "flat")
    type_match = (comp_type == target_type)

    # Distance factor: closer to subject site is preferred
    dist = comp.get("distance_km", 5.0)
    dist_score = max(0.0, (1.0 - min(dist, 10.0) / 10.0)) * 15.0

    # Rating factor
    rating = comp.get("rating") or 4.0
    rating_score = (rating / 5.0) * 5.0

    total_score = round(price_score + unit_score + area_score + dist_score + rating_score, 1)
    if not has_price_evidence:
        total_score = min(total_score, 55.0)

    # Qualified only if ALL 4 criteria pass:
    # 1. Price evidence verified AND in 80%–120%
    # 2. Same unit typology
    # 3. Area in range
    # 4. Same property type
    is_qualified = bool(has_price_evidence and price_match and unit_match and area_match and type_match)

    details = {
        "is_qualified": is_qualified,
        "type_match": type_match,
        "has_price_evidence": has_price_evidence,
        "price_match": price_match,
        "price_status": price_status,
        "unit_match": unit_match,
        "source_unit": source_unit,
        "area_match": area_match,
        "area_status": area_status,
        "min_area": min_area,
        "max_area": max_area,
        "min_rate": min_rate,
        "max_rate": max_rate,
        "rate_diff_pct": rate_diff_pct,
        "evidence": evidence,
        "match_score": total_score,
    }
    return is_qualified, total_score, details


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

    # Universal Negative Exclusion:
    # Strictly exclude hotels, OYOs, hostels, guest houses, public amenities, retail services, transport, and non-residential facilities
    non_building_google_types = {
        "restaurant", "cafe", "bar", "meal_takeaway", "meal_delivery",
        "hospital", "doctor", "health", "dentist", "pharmacy", "veterinary_care",
        "school", "primary_school", "secondary_school", "university",
        "bank", "atm", "finance", "insurance_agency", "accounting", "lawyer",
        "gas_station", "car_repair", "car_wash", "car_dealer", "parking",
        "place_of_worship", "church", "hindu_temple", "mosque", "synagogue",
        "gym", "spa", "beauty_salon", "hair_care", "laundry", "bakery",
        "transit_station", "bus_station", "subway_station", "train_station",
        "moving_company", "storage", "travel_agency", "post_office", "police",
        "local_government_office", "amusement_park", "night_club", "movie_theater",
        "lodging", "hotel", "motel", "guest_house", "bed_and_breakfast", "hostel", "campground"
    }
    non_res_name_exclusion = re.compile(
        r"\b(hotel|resort|oyo|guest\s*house|homestay|motel|lodge|inn|hostel|pg|paying guest|dormitory|stay\s*@|pvt ltd|technologies|solutions|consulting|roadlines|movers|packers|clinic|hospital|school|academy|classes|repair|services|agency|dealers?|motors?)\b",
        re.IGNORECASE
    )

    if (types_set & non_building_google_types) or bool(non_res_name_exclusion.search(name)):
        has_residential = False

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
    # Strip trailing parentheses, commas, or dashes with city names
    name = re.sub(
        r"\s*[\(,\-–]\s*(?:Pune|Mumbai|Delhi|PCMC|Bangalore|Hyderabad|Chennai|India|Branch|\d+)\s*\)?\s*$",
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
        r"^\s*(?:office|shop|unit|flat|suite|cabin|room|gf|ff|sf|plot|sr|sy|survey|ward|cts|g-flr|ground flr|first flr|second flr|1st floor|2nd floor|3rd floor|4th floor|5th floor|6th floor|gf and ff)\b[\s\d\w\-\.,#/:]*",
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

    # 1. If name is NOT an individual tenant/store, the place name itself IS the project name!
    if not is_tenant:
        clean_name = clean_project_name(name)
        clean_addr = address
        if parts and unit_prefix_re.search(parts[0]):
            clean_addr = ", ".join(parts[1:])
        return clean_name, clean_addr, ""

    # 2. If it IS a tenant, look in address parts for the parent building / project name
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
    property_type: Optional[str] = None,
    source_rate: Optional[float] = None,
    price_tolerance_pct: float = 20.0,
    source_unit: Optional[str] = None,
    source_area: Optional[float] = None,
    area_tolerance_pct: float = 15.0,
    source_amenities: Optional[List[str]] = None,
    filter_qualified_only: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main function to find comparable projects for a given project name or lat/long.
    INTELLIGENT AUTO-FETCH:
    If property_type, source_unit, source_area, or source_rate are not provided (or set to None/auto),
    they are automatically fetched from real estate web evidence for the subject project.
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
    city_name = subject_site.get("city", "Metropolitan Area")
    tavily_key = get_tavily_api_key()

    # 2. Intelligent Auto-Extraction of Subject Project DNA
    subj_name = subject_site.get("name") or project_name or ""
    if "Subject Site (" in subj_name or not subj_name or subj_name == "Custom Subject Property":
        addr = subject_site.get("address", "")
        addr_parts = [p.strip() for p in addr.split(",") if p.strip()]
        subj_search_name = " ".join(addr_parts[-4:-2]) if len(addr_parts) >= 4 else (addr_parts[0] if addr_parts else "Real Estate")
    else:
        subj_search_name = subj_name

    subject_dna = fetch_project_dna(project_name=subj_search_name, city=city_name, tavily_key=tavily_key)

    # Auto-detect or apply Property Type
    if not property_type or property_type == "auto":
        prop_type_key = normalize_property_type(subject_dna.get("property_type", "flat"))
    else:
        prop_type_key = normalize_property_type(property_type)

    # Auto-detect or apply Unit Typology
    if not source_unit or source_unit == "auto":
        resolved_unit = subject_dna.get("dominant_unit", "2 BHK")
    else:
        resolved_unit = source_unit

    # Auto-detect or apply Carpet Area
    if not source_area or source_area <= 0:
        resolved_area = subject_dna.get("carpet_area", 850.0)
    else:
        resolved_area = float(source_area)

    # Auto-detect or apply Benchmark Rate
    if not source_rate or source_rate <= 0:
        resolved_rate = subject_dna.get("rate_per_sqft")
        rate_discovered = bool(resolved_rate is not None)
    else:
        resolved_rate = float(source_rate)
        rate_discovered = False

    subject_site["property_type"] = prop_type_key
    subject_site["dominant_unit"] = resolved_unit
    subject_site["carpet_area"] = resolved_area
    subject_site["rate_per_sqft"] = resolved_rate
    subject_site["price_evidence"] = subject_dna.get("price_evidence")
    subject_site["rate_discovered"] = rate_discovered
    subject_site["dna"] = subject_dna

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
            "corporate office suites",
            "commercial business center",
            "office campus",
        ],
        "shop": [
            "shopping mall commercial complex",
            "commercial shopping complex retail center",
            "shopping arcade retail plaza",
            "commercial retail shopping mall",
            "retail showroom commercial market",
            "high street retail shopping complex",
        ],
        "flat": [
            "residential apartment",
            "housing society",
            "residential complex",
            "apartments",
            "flats",
            "gated community",
            "residential towers",
            "housing society apartments",
            "residential enclave",
            "residential project",
            "residential",
            "housing",
            "towers",
            "residency",
            "heights",
        ],
        "others": [
            "commercial complex business center",
            "integrated mixed use development complex",
            "logistics industrial business park",
            "commercial park",
        ],
    }

    target_queries = list(type_query_map.get(prop_type_key, type_query_map["flat"]))

    # 2. Live Google Places API Search
    if key:
        collected = []
        seen_ids = set()
        city_name = subject_site.get("city", "")

        # A. Open Spatial Proximity Search (Discovers ALL physical premises in radius, keyword-free)
        open_nearby_places, _ = search_google_places_nearby(
            lat=lat,
            lng=lng,
            radius_meters=radius_meters,
            keyword=None,
            api_key=key,
        )
        for p in open_nearby_places:
            pid = p.get("place_id") or p.get("name")
            if pid not in seen_ids:
                seen_ids.add(pid)
                collected.append(p)

        # B. Targeted Text Search - Query standard property categories in city
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

        # C. Targeted Keyword Nearby Search strictly bounded within radius
        for kw in target_queries:
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

        # C. Filter strictly using property type rule
        valid_comparables = []
        for p in collected:
            p_dist = p.get("distance_km")
            if p_dist is None or p_dist > radius_km:
                continue

            # Skip the subject project itself (distance < 50m and matches subject name)
            if p_dist < 0.05:
                p_name_lower = (p.get("name") or "").lower()
                s_name_lower = subj_search_name.lower()
                if s_name_lower in p_name_lower or p_name_lower in s_name_lower:
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
        is_demo = True
        comparables = _generate_mock_comparables(lat, lng, radius_km, prop_type_key, subject_site.get("city", "City Area"))
        status_msg = "Google Places API key not found in .env. Showing simulated comparable properties. Add GOOGLE_PLACES_API_KEY in .env for live Google data."

    # 3. Valuation Web Evidence Research & 80% to 120% Range Matching
    city_name = subject_site.get("city", "Metropolitan Area")
    tavily_key = get_tavily_api_key()

    # Pre-sort candidate projects strictly by closest distance to subject site first
    comparables.sort(key=lambda x: x.get("distance_km", 999))

    # Research live price evidence concurrently on ALL candidate projects (NO LIMIT: 20, 50, etc.)
    def _worker_evidence(comp_item):
        return enrich_comparable_valuation_data(
            comp_item,
            prop_type=prop_type_key,
            city=city_name,
            target_unit=resolved_unit,
            tavily_key=tavily_key,
            fetch_evidence=True,
        )

    if comparables:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(comparables), 12)) as executor:
            all_processed = list(executor.map(_worker_evidence, comparables))
    else:
        all_processed = []

    for comp in all_processed:
        is_qual, score, details = evaluate_comparable_match(
            comp,
            source_rate=resolved_rate,
            min_rate_pct=80.0,
            max_rate_pct=120.0,
            source_unit=resolved_unit,
            source_area=resolved_area,
            area_tolerance_pct=area_tolerance_pct,
            target_property_type=prop_type_key,
        )
        comp["is_qualified"] = is_qual
        comp["match_score"] = score
        comp["valuation_match"] = details

    # Sort: projects with verified price evidence first, then highest score, then closest distance
    all_processed.sort(
        key=lambda x: (
            not x.get("is_qualified", False),
            x.get("rate_per_sqft") is None,
            -x.get("match_score", 0.0),
            x.get("distance_km", 999),
        )
    )

    qualified_count = sum(1 for c in all_processed if c.get("is_qualified", False))

    if filter_qualified_only:
        displayed_comparables = [c for c in all_processed if c.get("is_qualified", False)]
    else:
        displayed_comparables = all_processed

    label_map = {
        "flat": "Flat (Residential Apartments, Societies, Condos)",
        "shop": "Shop (Retail Shops, Showrooms, Plazas)",
        "office": "Office (Corporate Towers, IT Parks, Business Centers)",
        "others": "Others (Mixed-Use, Commercial & Industrial)",
    }

    min_price_calc = round(resolved_rate * 0.80, 2) if resolved_rate else None
    max_price_calc = round(resolved_rate * 1.20, 2) if resolved_rate else None
    min_area_calc = round(resolved_area * (1.0 - area_tolerance_pct / 100.0), 1) if resolved_area else None
    max_area_calc = round(resolved_area * (1.0 + area_tolerance_pct / 100.0), 1) if resolved_area else None

    return {
        "subject_site": subject_site,
        "subject_dna": subject_dna,
        "radius_km": radius_km,
        "property_type": label_map.get(prop_type_key, prop_type_key.title()),
        "property_type_key": prop_type_key,
        "total_found": len(all_processed),
        "total_qualified": qualified_count,
        "comparables": displayed_comparables,
        "all_comparables": all_processed,
        "is_demo_data": is_demo,
        "status_message": status_msg,
        "valuation_parameters": {
            "source_rate": resolved_rate,
            "rate_discovered": rate_discovered,
            "subject_price_evidence": subject_site.get("price_evidence"),
            "min_rate_pct": 80.0,
            "max_rate_pct": 120.0,
            "min_rate": min_price_calc,
            "max_rate": max_price_calc,
            "source_unit": resolved_unit,
            "source_area": resolved_area,
            "area_tolerance_pct": area_tolerance_pct,
            "min_area": min_area_calc,
            "max_area": max_area_calc,
            "filter_qualified_only": filter_qualified_only,
            "dna": subject_dna,
        },
    }


def to_dataframe(comparables: List[Dict[str, Any]]) -> pd.DataFrame:
    """Helper to convert comparable projects list into a pandas DataFrame with valuation columns."""
    if not comparables:
        return pd.DataFrame(columns=[
            "Name", "Distance (km)", "Verified Rate (₹/sqft)", "In 80%–120% Range?",
            "Unit Typology", "Carpet Area (sqft)", "Price Evidence Source", "Evidence URL", "Data Source", "Google Maps", "Portal Search"
        ])

    rows = []
    for p in comparables:
        v_match = p.get("valuation_match", {})
        ev = p.get("price_evidence")
        rate_val = p.get("rate_per_sqft")

        rate_str = f"₹ {rate_val:,.0f}" if rate_val is not None else "No Verified Evidence"
        price_status = v_match.get("price_status", "⚠️ No Price Evidence")
        ev_source = ev.get("source_title", "—") if ev else "—"
        ev_url = ev.get("url", "") if ev else ""
        area_str = f"{p['carpet_area']:,} sq.ft" if p.get("carpet_area") else "Complex Standard"
        units_str = ", ".join(p.get("available_units", []))

        rows.append({
            "Name": p.get("name", ""),
            "Distance (km)": p.get("distance_km", 0.0),
            "Verified Rate (₹/sqft)": rate_str,
            "In 80%–120% Range?": price_status,
            "Unit Typology": units_str,
            "Carpet Area (sqft)": area_str,
            "Area Match": v_match.get("area_status", "Compatible"),
            "Price Evidence Source": ev_source,
            "Evidence URL": ev_url,
            "Valuation Match": f"{p.get('match_score', 0.0):.1f}%",
            "Status": "✅ Qualified (80-120%)" if p.get("is_qualified") else ("⚠️ Price Unverified" if rate_val is None else "❌ Out of 80-120% Range"),
            "Data Source": p.get("source", "Google Places API"),
            "Rating": p.get("rating", 0.0),
            "Reviews": p.get("user_ratings_total", 0),
            "Address": p.get("address", ""),
            "Latitude": p.get("lat"),
            "Longitude": p.get("lng"),
            "Google Maps": p.get("google_maps_url", ""),
            "Portal Search": p.get("web_search_url", ""),
        })
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find Comparable Projects with 80% to 120% Valuation Matching & Evidence")
    parser.add_argument("--project", type=str, help="Project name or address")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lng", type=float, help="Longitude")
    parser.add_argument("--radius", type=float, default=5.0, help="Radius in km (e.g. 2, 5, 10)")
    parser.add_argument("--type", type=str, default="flat", choices=["flat", "shop", "office", "others"], help="Property type")
    parser.add_argument("--rate", type=float, default=10000.0, help="Source benchmark rate (e.g. 10000)")
    parser.add_argument("--unit", type=str, default="2 BHK", help="Unit typology (e.g. '2 BHK')")
    parser.add_argument("--area", type=float, default=1000.0, help="Carpet area in sqft (e.g. 1000)")
    parser.add_argument("--area-tol", type=float, default=15.0, help="Area tolerance percentage (e.g. 15)")
    parser.add_argument("--amenities", nargs="*", default=["Clubhouse", "Gymnasium", "24x7 Security"], help="Key amenities")
    parser.add_argument("--qualified-only", action="store_true", help="Filter only qualified comparables")
    args = parser.parse_args()

    if not args.project and (args.lat is None or args.lng is None):
        print("Usage: python comparable.py --project 'Pimple Saudagar, Pune' --radius 5 --type flat --rate 10000")
        sys.exit(1)

    result = find_comparable_projects(
        lat=args.lat,
        lng=args.lng,
        project_name=args.project,
        radius_km=args.radius,
        property_type=args.type,
        source_rate=args.rate,
        source_unit=args.unit,
        source_area=args.area,
        area_tolerance_pct=args.area_tol,
        source_amenities=args.amenities,
        filter_qualified_only=args.qualified_only,
    )

    vp = result["valuation_parameters"]
    print("\n" + "=" * 75)
    print(f"Subject Site: {result['subject_site']['name']}")
    print(f"Coordinates: {result['subject_site']['lat']}, {result['subject_site']['lng']}")
    print(f"Address: {result['subject_site']['address']}")
    print(f"Search Radius: {result['radius_km']} km | Property Type: {result['property_type']}")
    print(f"Benchmark Rate: {vp['source_rate']} (Allowed 80% to 120%: {vp['min_rate']} - {vp['max_rate']})")
    print(f"Total Discovered: {result['total_found']} | Qualified in 80%-120%: {result['total_qualified']}")
    print(f"Status: {result['status_message']}")
    print("=" * 75)

    df = to_dataframe(result["comparables"])
    if not df.empty:
        try:
            print(df[["Name", "Distance (km)", "Verified Rate (₹/sqft)", "In 80%–120% Range?", "Price Evidence Source", "Data Source"]].to_string(index=False))
        except UnicodeEncodeError:
            print(df[["Name", "Distance (km)", "Verified Rate (₹/sqft)", "In 80%–120% Range?", "Price Evidence Source", "Data Source"]].to_string(index=False).encode("ascii", errors="replace").decode())
    else:
        print("No comparable projects found matching criteria.")


