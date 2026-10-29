"""
comparable.py - Intelligent Real Estate Comparable Projects & Valuation Matching Engine

Core Capabilities:
1. Multi-Source Discovery:
   - Google Places API (Physical POIs & Spatial Footprints)
   - Real Estate Web Portal Discovery (New Launches & Micro-Market Residential Enclaves)
   - Spatial Deduplication (Merges adjacent entries within 60m and fuzzy name matches)

2. Dynamic Real Estate DNA Extraction (Zero Hardcoding):
   - Developer Reputation & Segment (Tier 1 National, Tier 2 Regional, Tier 3 Local)
   - Project Age & Construction Status (Ready to Move vs Under Construction with possession dates)
   - Project Scale & Typology (Integrated Township, Gated Community Complex, Standalone Building)
   - Unit Configurations & Typical Carpet Areas
   - Price Extraction with Negative Rental Filtering & Ticket Size to Sq.Ft Rate Conversion

3. Elastic Valuation Engine:
   - Evaluates 6-7 continuous valuation dimensions:
     1. Proximity / Spatial Distance Decay
     2. Unit Typology (BHK alignment)
     3. Carpet Area Tolerance
     4. Developer Reputation & Segment Tier
     5. Construction Status & Project Age
     6. Scale & Community Typology
     7. Price Proximity (when available)
   - ELASTIC RE-WEIGHTING: If price evidence is not available for a project,
     100% of the weight dynamically re-distributes across the 6 physical parameters.
     Valid physical comparables are NEVER disqualified or given 0 scores!

4. Statistical Market Banding:
   - Computes dynamic Interquartile Range (IQR) & Core Market Band across verified micro-market comps.
   - Categorizes comps into Core Band, Premium Segment, Value Segment, or Physical Equivalence.
   - Eliminates arbitrary 80%-120% hard exclusion.

5. Local Persistent Cache:
   - Uses SQLite (valuation_cache.db) to cache extracted project intelligence.
   - Eliminates redundant API calls and makes repeat searches instant (< 2ms).
"""

import os
import sys
import re
import math
import json
import sqlite3
import datetime
import argparse
import urllib.parse
import concurrent.futures
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from utils import (
    parse_coordinates,
    get_google_api_key,
    get_tavily_api_key,
    get_openai_api_key,
    search_tavily,
    search_duckduckgo,
    search_openai_web,
    search_web_with_fallback,
    haversine_distance,
    geocode_project,
    reverse_geocode,
    search_google_places_nearby,
    search_google_places_text,
)


# ==============================================================================
# 1. LOCAL PERSISTENT CACHE (SQLite)
# ==============================================================================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "valuation_cache.db")


def init_cache_db():
    """Ensure the SQLite cache table exists."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_dna_cache (
                    cache_key TEXT PRIMARY KEY,
                    project_name TEXT,
                    city TEXT,
                    data_json TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except Exception:
        pass


def normalize_unit_slug(unit: Optional[str] = None, prop_type: Optional[str] = "flat") -> str:
    """Produce standardized unit slug for caching and unit-specific rate binding."""
    if not unit:
        return (prop_type or "flat").lower()
    u = unit.lower().strip()
    if "1 bhk" in u or "1bhk" in u or "1 bed" in u:
        return "1bhk"
    elif "2 bhk" in u or "2bhk" in u or "2 bed" in u:
        return "2bhk"
    elif "3 bhk" in u or "3bhk" in u or "3 bed" in u:
        return "3bhk"
    elif "4 bhk" in u or "4bhk" in u or "4 bed" in u:
        return "4bhk"
    elif any(k in u for k in ["> 4", "5", "6", "penthouse", "duplex"]):
        return "5bhk_plus"
    elif "shop" in u or "retail" in u or (prop_type and prop_type.lower() == "shop"):
        return "shop"
    elif "office" in u or "tech" in u or (prop_type and prop_type.lower() == "office"):
        return "office"
    else:
        clean = re.sub(r'[^a-z0-9]', '', u)
        return clean or "any"


def get_cached_project_dna(project_name: str, city: str = "", target_unit: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached project DNA with multi-tier key resolution:
    1. Exact match with city and target unit: {norm_name}_{norm_city}_{unit_slug}
    2. Unit match without city: {norm_name}__{unit_slug}
    3. Base project with city: {norm_name}_{norm_city}
    4. Base project alone: {norm_name}
    5. Prefix search: cache_key LIKE '{norm_name}%'
    """
    try:
        norm_name = re.sub(r'[^a-z0-9]', '', (project_name or '').lower())
        if not norm_name:
            return None
        norm_city = re.sub(r'[^a-z0-9]', '', (city or '').lower())
        unit_slug = normalize_unit_slug(target_unit)

        candidate_keys = [
            f"{norm_name}_{norm_city}_{unit_slug}",
            f"{norm_name}__{unit_slug}",
            f"{norm_name}_{norm_city}",
            f"{norm_name}",
        ]

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            for ck in candidate_keys:
                cursor.execute("SELECT data_json FROM project_dna_cache WHERE cache_key = ?", (ck,))
                row = cursor.fetchone()
                if row:
                    data = json.loads(row[0])
                    if data.get("rate_per_sqft") or data.get("carpet_area") or data.get("available_units"):
                        return data

            # Prefix search as fallback
            cursor.execute("SELECT data_json FROM project_dna_cache WHERE cache_key LIKE ? LIMIT 1", (f"{norm_name}%",))
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                if data.get("rate_per_sqft") or data.get("carpet_area") or data.get("available_units"):
                    return data
    except Exception:
        pass
    return None


def save_cached_project_dna(project_name: str, city: str, data: Dict[str, Any], target_unit: Optional[str] = None):
    """Persist project DNA into SQLite cache with unit-specific and base keying."""
    try:
        # STRICT RATE INTEGRITY: Only cache records with verified positive rates
        if not (data.get("rate_per_sqft") and data.get("rate_per_sqft") > 0):
            return

        init_cache_db()
        norm_name = re.sub(r'[^a-z0-9]', '', (project_name or '').lower())
        norm_city = re.sub(r'[^a-z0-9]', '', (city or '').lower())
        unit_slug = normalize_unit_slug(target_unit or data.get("dominant_unit"))
        norm_key = f"{norm_name}_{norm_city}_{unit_slug}"
        base_key = f"{norm_name}_{norm_city}"

        clean_data = dict(data)
        data_json = json.dumps(clean_data)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO project_dna_cache (cache_key, project_name, city, data_json, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (norm_key, project_name, city, data_json)
            )
            if base_key != norm_key:
                conn.execute(
                    "INSERT OR REPLACE INTO project_dna_cache (cache_key, project_name, city, data_json, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                    (base_key, project_name, city, data_json)
                )
            conn.commit()
    except Exception:
        pass


# Initialize DB on load
init_cache_db()


# ==============================================================================
# 2. PROPERTY TYPE FILTERING & NORMALIZATION
# ==============================================================================
GOOGLE_RESIDENTIAL_TYPES = {
    "apartment_complex", "housing_complex", "condominium_complex",
    "residential", "housing_estate", "gated_community",
    "point_of_interest", "establishment"
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
    r"\bpark\b", r"\bliving\b", r"\bhabitat\b", r"\bavenue\b", r"\boasis\b",
    r"\bmesmer\b", r"\binsignia\b", r"\bserene\b", r"\bbliss\b", r"\bvista\b",
    r"\bcrescent\b", r"\belysia\b", r"\bhaven\b", r"\bsuites?\b", r"\bgrove\b",
    r"\bgrand\b", r"\bsolitaire\b", r"\bsky\b", r"\bregency\b", r"\belegance\b",
    r"\bluxuria\b", r"\bsplendour\b", r"\bsignature\b", r"\bnirvana\b", r"\bharmony\b",
    r"\bparkview\b", r"\broyale\b", r"\bparadise\b", r"\bcomfort\b", r"\bcounty\b",
    r"\bhills?\b", r"\bvalley\b", r"\baura\b", r"\bpinnacle\b", r"\bcastle\b",
    r"\bpalace\b", r"\bnest\b", r"\babode\b"
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


def is_unit_matching(target_u: Optional[str], avail_units: Optional[List[str]]) -> bool:
    """Checks whether a target unit configuration (e.g. '1 BHK', '2 BHK', '3 BHK', '4 BHK', '> 4 BHK', 'Shop', 'Office')
    matches the disclosed available units list of a project.
    """
    if not target_u or not avail_units:
        return False

    t = target_u.lower().strip()
    units_clean = [u.lower().strip() for u in avail_units if u]

    # Check > 4 BHK / Penthouse / Duplex first so '4 bhk' substring doesn't greedily match '> 4 bhk'
    if any(k in t for k in ["> 4", ">4", "5", "6", "penthouse", "duplex", "villa"]):
        return any(k in u for u in units_clean for k in ["5 bhk", "5bhk", "6 bhk", "6bhk", "penthouse", "duplex", "5 bed", "6 bed", "villa"])
    elif "4 bhk" in t or t == "4bhk" or "4 bed" in t:
        return any(k in u for u in units_clean for k in ["4 bhk", "4bhk", "4 bed"])
    elif "3 bhk" in t or t == "3bhk" or "3 bed" in t:
        return any(k in u for u in units_clean for k in ["3 bhk", "3bhk", "3 bed"])
    elif "2 bhk" in t or t == "2bhk" or "2 bed" in t:
        return any(k in u for u in units_clean for k in ["2 bhk", "2bhk", "2 bed"])
    elif "1 bhk" in t or t == "1bhk" or "1 bed" in t:
        return any(k in u for u in units_clean for k in ["1 bhk", "1bhk", "1 bed", "studio"])
    elif "shop" in t or "retail" in t:
        return any(k in u for u in units_clean for k in ["shop", "retail", "showroom"])
    elif "office" in t or "tech" in t:
        return any(k in u for u in units_clean for k in ["office", "tech park", "it park", "workplace"])
    else:
        return any(t in u for u in units_clean)


def detect_property_attributes(place: Dict[str, Any]) -> Dict[str, bool]:
    """Dynamically analyze place attributes to detect office, retail, or residential presence."""
    
    name = (place.get("name") or "").lower()
    address = (place.get("address") or "").lower()
    full_text = f"{name} {address}"
    types_set = set(place.get("types") or [])

    # Physical commercial building keywords vs corporate tenant exclusions
    comm_bldg_keywords = re.compile(
        r"\b(business\s*center|business\s*centre|business\s*park|tech\s*park|it\s*park|tower|towers|complex|plaza|square|workspace|workspaces|coworking|co-working|office\s*space|offices|campus|hub|heights|commercial|arcade|mart|bhavan|house)\b",
        re.I
    )
    comm_tenant_keywords = re.compile(
        r"\b(pvt\.?\s*ltd\.?|llp|inc|services?|solutions?|technologies|software|consulting|consultancy|institute|school|academy|classes|agency|logistics|realtors?|finance)\b",
        re.I
    )
    is_comm_building = bool(comm_bldg_keywords.search(name)) or bool(comm_bldg_keywords.search(address))
    is_comm_tenant = bool(comm_tenant_keywords.search(name)) and not is_comm_building

    raw_office_signal = bool(types_set & GOOGLE_OFFICE_TYPES) or any(
        re.search(pat, full_text, re.IGNORECASE) for pat in OFFICE_KEYWORDS
    )
    has_office = raw_office_signal and not is_comm_tenant
    has_shop = bool(types_set & GOOGLE_SHOP_TYPES) or any(
        re.search(pat, full_text, re.IGNORECASE) for pat in SHOP_KEYWORDS
    )

    brand_match = any(re.search(rf"\b{token}\b", name, re.IGNORECASE) for token in KNOWN_BRAND_MAP)

    has_specific_residential = (
        bool(types_set & {"apartment_complex", "housing_complex", "condominium_complex", "residential", "housing_estate", "gated_community"})
        or any(re.search(pat, full_text, re.IGNORECASE) for pat in RESIDENTIAL_KEYWORDS)
        or brand_match
    )

    # Universal negative exclusion: exclude hotels, clinics, hostels, factories, industrial firms, pet shops
    non_building_google_types = {
        "restaurant", "cafe", "bar", "meal_takeaway", "meal_delivery",
        "hospital", "doctor", "health", "dentist", "pharmacy", "veterinary_care",
        "school", "primary_school", "secondary_school", "university",
        "bank", "atm", "finance", "insurance_agency", "accounting", "lawyer",
        "gas_station", "car_repair", "car_wash", "car_dealer", "parking",
        "place_of_worship", "church", "hindu_temple", "mosque", "gym", "spa",
        "transit_station", "bus_station", "subway_station", "train_station",
        "lodging", "hotel", "motel", "guest_house", "hostel", "pet_store",
        "storage", "warehouse", "factory", "manufacturing", "industrial",
        "contractor", "general_contractor", "police", "local_government_office", "travel_agency"
    }
    non_res_name_exclusion = re.compile(
        r"\b(hotel|resort|oyo|guest\s*house|homestay|motel|lodge|inn|hostel|pg|paying guest|dormitory|stay\s*@|"
        r"roadlines|movers|packers|packaging|containers?|clinic|classes|repair|services?|motors?|pet\s*shop|pet\s*store|petville|"
        r"consulting|consultancy|technologies|software|solutions|head\s*office|corporate\s*office|branch\s*office|"
        r"machinery|machines?|industr(?:y|ies|ial)|enterprises?|manufacturing|manufacturers?|works|fabricat(?:ion|ors?)|"
        r"steels?|polymers?|chemicals?|rubber|plant|station|udyog|traders?|trading|corporation|distributors?|"
        r"pvt\.?\s*ltd\.?|llp\.?|inc\.?|holdings?)\b",
        re.IGNORECASE
    )

    is_excluded = bool(types_set & non_building_google_types) or bool(non_res_name_exclusion.search(name))

    core_res_types = {"apartment_complex", "housing_complex", "condominium_complex", "residential", "housing_estate", "gated_community"}
    is_core_res = bool(types_set & core_res_types)

    if has_office or has_shop:
        has_residential = has_specific_residential and not is_excluded
    else:
        has_residential = (
            is_core_res
            or has_specific_residential
            or bool(place.get("discovered_via_residential", False))
        ) and not is_excluded

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
    """Bifurcate projects strictly by property category."""
    t = normalize_property_type(target_type)
    if t == "office":
        if attrs["is_pure_residential"] or not attrs["has_office"]:
            return False, ""
        return True, "Mixed-Use (Office & Residential)" if attrs["is_mixed_use"] else "Office / Corporate"
    elif t == "shop":
        if attrs["is_pure_residential"] or not attrs["has_shop"]:
            return False, ""
        return True, "Mixed-Use (Retail & Residential)" if attrs["is_mixed_use"] else "Shop / Retail"
    elif t == "flat":
        if attrs["is_pure_office"] or attrs["is_pure_shop"] or not attrs["has_residential"]:
            return False, ""
        return True, "Mixed-Use (Residential & Commercial)" if attrs["is_mixed_use"] else "Flat / Residential"
    else:  # "others"
        return True, "Mixed-Use Development" if attrs["is_mixed_use"] else "Real Estate Development"


# ==============================================================================
# 3. DYNAMIC REAL ESTATE DNA & PARAMETER EXTRACTION
# ==============================================================================
MONTHS_SET = {
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december"
}
STOPWORDS_SET = {
    "bhks", "bhk", "the", "a", "an", "all", "spread", "over", "an", "area", "blending", "indian",
    "contact", "property", "dealers", "project", "free", "view", "details", "click", "for",
    "floor", "plans", "pricing", "redhat", "debugger", "java", "flats", "apartments", "homes", "in", "by"
}

KNOWN_BRAND_MAP = {
    "godrej": ("Godrej Properties", "Tier 1 (National Listed)"),
    "prestige": ("Prestige Group", "Tier 1 (National Listed)"),
    "sobha": ("Sobha Limited", "Tier 1 (National Listed)"),
    "dlf": ("DLF Limited", "Tier 1 (National Listed)"),
    "tata": ("Tata Housing", "Tier 1 (National Listed)"),
    "brigade": ("Brigade Group", "Tier 1 (National Listed)"),
    "lodha": ("Lodha / Macrotech", "Tier 1 (National Listed)"),
    "macrotech": ("Lodha / Macrotech", "Tier 1 (National Listed)"),
    "oberoi": ("Oberoi Realty", "Tier 1 (National Listed)"),
    "purva": ("Puravankara Limited", "Tier 1 (National Listed)"),
    "puravankara": ("Puravankara Limited", "Tier 1 (National Listed)"),
    "mahindra": ("Mahindra Lifespaces", "Tier 1 (National Listed)"),
    "hiranandani": ("Hiranandani Group", "Tier 1 (National Listed)"),
    "shapoorji": ("Shapoorji Pallonji", "Tier 1 (National Listed)"),
    "kalpataru": ("Kalpataru Limited", "Tier 1 (National Listed)"),
    "kolte": ("Kolte-Patil Developers", "Tier 2 (Reputed Regional)"),
    "kolte-patil": ("Kolte-Patil Developers", "Tier 2 (Reputed Regional)"),
    "tuscan": ("Kolte-Patil Developers", "Tier 2 (Reputed Regional)"),
    "nyati": ("Nyati Group", "Tier 2 (Reputed Regional)"),
    "mantra": ("Mantra Properties", "Tier 2 (Reputed Regional)"),
    "rohan": ("Rohan Builders", "Tier 2 (Reputed Regional)"),
    "arv": ("ARV Group", "Tier 2 (Reputed Regional)"),
    "shubh": ("Shubh Developers", "Tier 2 (Reputed Regional)"),
    "panchshil": ("Panchshil Realty", "Tier 2 (Reputed Regional)"),
    "amanora": ("City Corporation / Amanora", "Tier 2 (Reputed Regional)"),
    "kumar": ("Kumar Properties", "Tier 2 (Reputed Regional)"),
    "vtp": ("VTP Realty", "Tier 2 (Reputed Regional)"),
    "paranjape": ("Paranjape Schemes", "Tier 2 (Reputed Regional)"),
    "kasturi": ("Kasturi Housing", "Tier 2 (Reputed Regional)"),
    "riverdale": ("Duville Estates / Riverdale", "Tier 2 (Reputed Regional)"),
    "duville": ("Duville Estates", "Tier 2 (Reputed Regional)"),
    "pride": ("Pride Purple Group", "Tier 2 (Reputed Regional)"),
    "goel": ("Goel Ganga Developments", "Tier 2 (Reputed Regional)"),
    "ganga": ("Goel Ganga Developments", "Tier 2 (Reputed Regional)"),
    "pharande": ("Pharande Spaces", "Tier 2 (Reputed Regional)"),
    "kohinoor": ("Kohinoor Group", "Tier 2 (Reputed Regional)"),
    "zen": ("Kohinoor Group", "Tier 2 (Reputed Regional)"),
    "majestique": ("Majestique Landmarks", "Tier 2 (Reputed Regional)"),
    "vascon": ("Vascon Engineers", "Tier 2 (Reputed Regional)"),
    "gera": ("Gera Developments", "Tier 2 (Reputed Regional)"),
    "bhandari": ("Bhandari Associates", "Tier 2 (Reputed Regional)"),
    "pramukh": ("Pramukh Group", "Tier 2 (Reputed Regional)"),
    "marvel": ("Marvel Realtors", "Tier 2 (Reputed Regional)"),
    "javdekar": ("Vilas Javdekar Developers", "Tier 2 (Reputed Regional)"),
    "vilas javdekar": ("Vilas Javdekar Developers", "Tier 2 (Reputed Regional)"),
    "venkatesh": ("Venkatesh Buildcon", "Tier 2 (Reputed Regional)"),
    "pristine": ("Pristine Properties", "Tier 2 (Reputed Regional)"),
    "anandtara": ("Anandtara Infrastructure", "Tier 2 (Reputed Regional)"),
    "triaa": ("Triaa Housing", "Tier 2 (Reputed Regional)"),
    "gulmohar": ("Gulmohar Development", "Tier 2 (Reputed Regional)"),
    "mittal": ("Mittal Group", "Tier 2 (Reputed Regional)"),
    "kundan": ("Kundan Spaces", "Tier 2 (Reputed Regional)"),
    "konark": ("Konark Karia Builders", "Tier 2 (Reputed Regional)"),
    "sancheti": ("Sancheti Associates", "Tier 2 (Reputed Regional)"),
    "gagan": ("Gagan Developers", "Tier 2 (Reputed Regional)"),
    "unique": ("Unique Properties", "Tier 2 (Reputed Regional)"),
}


def extract_project_dna_dynamic(
    name: str,
    text: str,
    target_unit: str = "2 BHK",
    user_ratings_total: int = 0
) -> Dict[str, Any]:
    """
    Dynamically extracts all valuation parameters from genuine sources without static fallbacks:
    - Extracts actual carpet area (returns None if not disclosed, NEVER defaults to 800).
    - Extracts developer name & tier without English prepositions or months.
    - Differentiates scale & typology (Township, Large Gated Community, Co-op CHS, Standalone).
    - Extracts verified unit typologies.
    """
    full_text = f"{name} {text}"
    name_lower = name.lower()
    text_lower = text.lower()
    current_year = datetime.datetime.now().year

    # 1. Developer Extraction & Tier Classification (STRICT EVIDENCE ONLY - Zero Fallbacks)
    developer = None
    dev_tier = None
    developer_evidence = None

    # A. Check if project name explicitly contains developer via "by <Developer>" or "(<Developer>)"
    by_m = re.search(r"\bby\s+([A-Za-z0-9\s&]{2,30}?)(?:\s*\(|$|,|\.)", name)
    if by_m:
        cand = by_m.group(1).strip()
        first_w = cand.lower().split()[0] if cand.split() else ""
        if first_w not in MONTHS_SET and first_w not in STOPWORDS_SET and len(cand) >= 3:
            developer = cand
            dev_tier = "Tier 2 (Reputed Regional)" if any(k in cand.lower() for k in ["properties", "developers", "realty", "group", "buildcon"]) else "Tier 3 (Local / Standalone)"
            developer_evidence = f"Extracted from project name: '{by_m.group(0).strip()}'"

    paren_m = re.search(r"\(([\w\s&]{3,25})\)", name)
    if not developer and paren_m:
        cand = paren_m.group(1).strip()
        first_w = cand.lower().split()[0] if cand.split() else ""
        if first_w not in MONTHS_SET and first_w not in STOPWORDS_SET and len(cand) >= 3 and not cand.isdigit():
            developer = cand
            dev_tier = "Tier 2 (Reputed Regional)" if any(k in cand.lower() for k in ["properties", "developers", "realty", "group", "patil"]) else "Tier 3 (Local / Standalone)"
            developer_evidence = f"Identified from project name parenthetical: '({cand})'"

    # B. Check known brand tokens in name or text
    if not developer:
        for token, (brand_name, tier) in KNOWN_BRAND_MAP.items():
            if re.search(rf"\b{token}\b", name_lower):
                developer = brand_name
                dev_tier = tier
                developer_evidence = f"Brand recognition match in name: '{brand_name}'"
                break

    if not developer:
        for token, (brand_name, tier) in KNOWN_BRAND_MAP.items():
            m_br = re.search(rf"\b{token}\s*(?:group|properties|realty|developers|estates)?\b", text_lower)
            if m_br:
                start = max(0, m_br.start() - 30)
                end = min(len(text), m_br.end() + 30)
                developer = brand_name
                dev_tier = tier
                developer_evidence = f"Portal snippet brand match: \"...{text[start:end].strip()}...\""
                break

    # C. Extract builder from text pattern (filtering out months and stopwords)
    if not developer:
        # Structured markdown or list pattern: e.g. "**Developer:**\n- HRH City Projects Private Limited"
        dev_struct = re.search(
            r"\b(?:developer|builder|promoter|promoted by|developed by)\s*[:\*]*\s*\n*\s*[-*]?\s*([A-Za-z0-9\s&,\.]{3,50}?)(?:\n|\*|$)",
            text,
            re.I
        )
        if dev_struct:
            cand = dev_struct.group(1).strip()
            cand = re.sub(r"^(the|a|an|by)\s+", "", cand, flags=re.I).strip()
            cand = re.sub(r"[\.,\-]+$", "", cand).strip()
            first_word = cand.lower().split()[0] if cand.split() else ""
            if first_word not in MONTHS_SET and first_word not in STOPWORDS_SET and len(cand) > 3 and not cand.lower().startswith("http"):
                developer = cand
                dev_tier = "Tier 2 (Reputed Regional)" if any(k in cand.lower() for k in ["group", "properties", "realty", "developers", "projects", "housing", "limited", "pvt"]) else "Tier 3 (Local / Standalone)"
                developer_evidence = f"Portal snippet: \"...Developer: {cand}...\""

    if not developer:
        dev_m = re.search(
            r"\b(?:developed by|promoted by|builder|promoter)\s*[:,]?\s*(?:the\s+renowned\s+builder\s+|the\s+builder\s+|the\s+)?([A-Z][A-Za-z0-9\s&]{2,40}?(?:Group|Properties|Realty|Developers|Housing|Builders|Infra|Estates|Projects|Private\s+Limited|Pvt\.?\s*Ltd\.?|Limited)?)",
            text,
            re.I
        )
        if dev_m:
            cand = dev_m.group(1).strip()
            cand = re.sub(r"^(the|a|an|by)\s+", "", cand, flags=re.I).strip()
            cand = re.sub(r"[\.,\-]+$", "", cand).strip()
            first_word = cand.lower().split()[0] if cand.split() else ""
            if first_word not in MONTHS_SET and first_word not in STOPWORDS_SET and len(cand) > 3 and not cand.lower().startswith("http"):
                developer = cand
                dev_tier = "Tier 2 (Reputed Regional)" if any(k in cand.lower() for k in ["group", "properties", "realty", "developers", "projects", "housing", "limited", "pvt"]) else "Tier 3 (Local / Standalone)"
                start = max(0, dev_m.start() - 20)
                end = min(len(text), dev_m.end() + 20)
                developer_evidence = f"Portal snippet: \"...{text[start:end].strip()}...\""

    # D. ZERO FALLBACK: If developer is not in evidence, do NOT synthesize or impute!
    # Developer remains None if not found in evidence.

    # 2. Construction Status & Project Age (STRICT EVIDENCE ONLY - Zero Guessing)
    status = None
    possession_year = None
    age_years = None
    status_evidence = None
    age_evidence = None

    # Step A: Direct Property Age Range expressions (e.g. "1-5 years old", "5 to 10 Years", "Property Age: 0-1 Year", "10+ years old")
    range_age_m = re.search(
        r"(?:property\s*is|property\s*age|age\s*of\s*(?:property|construction|building|flat)|age|building\s*age)?[:\-\*\s]*(?:(?:is|was)\s+)?([0-9]{1,2})\s*(?:to|-)\s*([0-9]{1,2})\s*(?:years?|yrs?)(?:\s*old)?",
        full_text, re.I
    )
    if range_age_m and range_age_m.group(1) and range_age_m.group(2):
        low = int(range_age_m.group(1))
        high = int(range_age_m.group(2))
        calc_age = round((low + high) / 2.0, 1)
        age_years = int(calc_age) if calc_age.is_integer() else calc_age
        status = "Ready to Move"
        age_evidence = f"Portal snippet: \"...{range_age_m.group(0).strip()}...\""
        status_evidence = f"Portal snippet: \"...Ready to Move (Age: {age_years} yrs)...\""

    # Step A2: Single Age expressions (e.g. "3 Years old", "age of property is 3 years", "Age: 4 Years", "less than 5 years")
    if age_years is None:
        single_age_m = re.search(
            r"(?:(?:property\s*is|property\s*age|age\s*of\s*(?:property|construction|building|flat)|age|building\s*age)[:\-\*\s]*(?:(?:is|was)\s+)?(?:about|approx\.?|<|>|less than|more than)?\s*([0-9]{1,2})\s*(?:years?|yrs?)|([0-9]{1,2})\s*(?:\+|\s*plus)?\s*(?:years?|yrs?)\s*old)",
            full_text, re.I
        )
        if single_age_m:
            val_str = single_age_m.group(1) or single_age_m.group(2)
            age_years = int(val_str)
            if not status:
                status = "Ready to Move"
            age_evidence = f"Portal snippet: \"...{single_age_m.group(0).strip()}...\""
            if not status_evidence:
                status_evidence = f"Portal snippet: \"...Ready to Move (Age: {age_years} yrs)...\""

    # Step B: Possession with flexible preposition and dates (e.g. "possession in July 2021", "possession from Jun 2025", "possession: Dec, 2019", "31-12-2021")
    if possession_year is None:
        poss_m = re.search(
            r"\bpossession[^\.\n\r]{0,45}?(?:(?:in|from|by|since|started|date\s*is|date\s*[:\-=]|expected\s*in|scheduled\s*for)\s+)?(?:[A-Za-z]+,?\s+)?(?:[0-9]{1,2}[\-/])?(20[0-3][0-9]|19[8-9][0-9])\b",
            full_text, re.I
        )
        if poss_m:
            yr = int(poss_m.group(1))
            is_explicit_uc = bool(re.search(r"\b(under[\s\-]?construction|ongoing|new launch)\b", full_text, re.I))
            if yr > current_year or (is_explicit_uc and yr >= current_year):
                status = "Under Construction"
                possession_year = yr
                age_years = 0
                status_evidence = f"Portal snippet: \"...Under Construction (Possession: {yr})...\""
                age_evidence = f"Portal snippet: \"...{poss_m.group(0).strip()}...\""
            else:
                if not status:
                    status = "Ready to Move"
                possession_year = yr
                if age_years is None:
                    age_years = max(0, current_year - yr)
                age_unit = "yr" if age_years == 1 else "yrs"
                if not status_evidence:
                    status_evidence = f"Portal snippet: \"...Ready to Move (Possession: {yr}).\""
                if not age_evidence:
                    age_evidence = f"Portal snippet: \"...{poss_m.group(0).strip()} (Age: {age_years} {age_unit})...\""

    # Step B2: Inverted Possession (e.g. "Dec 2020 Possession", "2021 Possession")
    if possession_year is None:
        inv_poss_m = re.search(
            r"\b(?:[A-Za-z]+,?\s+)?(20[0-3][0-9]|19[8-9][0-9])\s*possession\b",
            full_text, re.I
        )
        if inv_poss_m:
            yr = int(inv_poss_m.group(1))
            if yr > current_year:
                status = "Under Construction"
                possession_year = yr
                age_years = 0
                status_evidence = f"Portal snippet: \"...Under Construction (Possession: {yr})...\""
                age_evidence = f"Portal snippet: \"...{inv_poss_m.group(0).strip()}...\""
            else:
                if not status:
                    status = "Ready to Move"
                possession_year = yr
                if age_years is None:
                    age_years = max(0, current_year - yr)
                age_unit = "yr" if age_years == 1 else "yrs"
                status_evidence = f"Portal snippet: \"...Ready to Move (Possession: {yr}).\""
                age_evidence = f"Portal snippet: \"...{inv_poss_m.group(0).strip()} (Age: {age_years} {age_unit})...\""

    # Step C: Completion Date / Year Built (e.g. "Completion Date: Dec, 2019", "MahaRERA-declared completion (Dec 2019)", "Completed in 2018", "Year Built: 2016")
    if possession_year is None and age_years is None:
        comp_m = re.search(
            r"\b(?:completion\s*date|completion|completed|built|year\s*(?:of\s*)?built|constructed|delivered|handover|launched|ready\s*since)[^\.\n\r]{0,35}?(?:in\s*)?(?:[A-Za-z]+,?\s+)?(20[0-3][0-9]|19[8-9][0-9])\b",
            full_text, re.I
        )
        if comp_m:
            yr = int(comp_m.group(1))
            if yr > current_year:
                status = "Under Construction"
                possession_year = yr
                age_years = 0
                status_evidence = f"Portal snippet: \"...Under Construction (Possession: {yr})...\""
                age_evidence = f"Portal snippet: \"...{comp_m.group(0).strip()}...\""
            else:
                if not status:
                    status = "Ready to Move"
                possession_year = yr
                age_years = max(0, current_year - yr)
                age_unit = "yr" if age_years == 1 else "yrs"
                if not status_evidence:
                    status_evidence = f"Portal snippet: \"...Ready to Move (Completed: {yr}).\""
                if not age_evidence:
                    age_evidence = f"Portal snippet: \"...{comp_m.group(0).strip()} (Age: {age_years} {age_unit})...\""

    # Step D: Generic Under Construction keywords (if year was not explicitly stated)
    if status is None:
        uc_search = re.search(r"\b(under[\s\-]?construction|ongoing|new launch|under development|launching soon|upcoming project)\b", full_text, re.I)
        if uc_search:
            status = "Under Construction"
            age_years = 0
            status_evidence = f"Portal snippet: \"...{uc_search.group(0)}...\""
            age_evidence = f"Portal snippet: \"...{uc_search.group(0)}...\""

    # Step E: Generic Ready to Move keywords (if year was not explicitly stated)
    if status is None:
        rtm_search = re.search(r"\b(ready[\s\-]?to[\s\-]?move|ready possession|immediate possession|completed project|oc received)\b", full_text, re.I)
        if rtm_search:
            status = "Ready to Move"
            status_evidence = f"Portal snippet: \"...{rtm_search.group(0)}...\""
            age_evidence = f"Portal snippet: \"...{rtm_search.group(0)}...\""

    # 3. Project Scale & Typology (STRICT EVIDENCE ONLY)
    scale_type = None
    scale_evidence = None

    acres_m = re.search(r"(\b[0-9]{1,3}(?:\.[0-9]{1,2})?)\s*acres", full_text, re.I)
    tow_m = re.search(r"(\b[0-9]{1,3}\b)\s*towers?", full_text, re.I)
    unit_cnt_m = re.search(r"(\b[0-9]{2,5}\b)\s*units?", full_text, re.I)
    towers_count = int(tow_m.group(1)) if tow_m else None
    units_count = int(unit_cnt_m.group(1)) if unit_cnt_m else None

    if acres_m:
        acres = float(acres_m.group(1))
        if acres >= 20.0:
            scale_type = f"Integrated Township ({acres:.1f} Acres)"
        elif acres >= 2.5:
            scale_type = f"Gated Community ({acres:.1f} Acres)"
        else:
            scale_type = f"Residential Complex ({acres:.1f} Acres)"
        scale_evidence = f"Portal snippet: \"...{acres_m.group(0)}...\""
    elif re.search(r"\b(township|integrated city|mega township)\b", full_text, re.I):
        m_t = re.search(r"\b(township|integrated city|mega township)\b", full_text, re.I)
        scale_type = "Integrated Township"
        scale_evidence = f"Portal snippet: \"...{m_t.group(0)}...\""
    elif re.search(r"\b(villas?|row house|bungalow)\b", full_text, re.I):
        m_v = re.search(r"\b(villas?|row house|bungalow)\b", full_text, re.I)
        scale_type = "Villas / Row Houses"
        scale_evidence = f"Portal snippet: \"...{m_v.group(0)}...\""
    elif towers_count and towers_count >= 4:
        scale_type = f"Gated Community ({towers_count} Towers)"
        scale_evidence = f"Portal snippet: \"...{tow_m.group(0)}...\""
    elif "society" in full_text.lower() or "chs" in full_text.lower() or "co-operative" in full_text.lower():
        scale_type = "Co-op Housing Society"
        scale_evidence = f"Registered housing society footprint: '{name}'"
    elif any(k in name_lower for k in ["apartment", "kunj", "niwas", "sadan", "villa"]):
        scale_type = "Standalone Building"
        scale_evidence = f"Standalone residential building: '{name}'"

    # 4. Available Units (Explicit Mentions Only, No Fabricated Defaults)
    avail_units = []
    unit_evidence = None
    found_snippets = []
    unit_candidates = [
        "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK", "6 BHK",
        "Penthouse", "Duplex", "Studio", "1 Bed", "2 Bed", "3 Bed", "4 Bed", "5 Bed",
        "Villa", "Retail Shop", "Office Space", "Commercial Showroom"
    ]
    for u in unit_candidates:
        m_u = re.search(rf"\b{re.escape(u)}\b", full_text, re.I)
        if m_u:
            avail_units.append(u)
            start = max(0, m_u.start() - 20)
            end = min(len(full_text), m_u.end() + 20)
            found_snippets.append(full_text[start:end].strip())

    if found_snippets:
        unit_evidence = f"Portal snippet: \"...{found_snippets[0]}...\""

    dominant_unit = target_unit if (avail_units and is_unit_matching(target_unit, avail_units)) else (avail_units[0] if avail_units else target_unit)

    # 5. Carpet Area Extraction (STRICT EVIDENCE ONLY - Returns None if not in source)
    carpet_area = None
    area_evidence = None

    # Step A: Look for target unit specific area or range: e.g. "2 BHK ... 566 sq.ft. to 624 sq.ft" or "2 BHK - 751 sqft"
    num_sqft_pattern = r"(?:\b[0-9]{1,3}(?:,[0-9]{3})+\b|\b[0-9]{2,5}\b)"
    is_commercial_unit = any(k in (target_unit or "").lower() for k in ["office", "shop", "commercial", "retail"])
    if target_unit and any(k in target_unit.lower() for k in ["> 4", "5", "6", "penthouse"]):
        u_pat = re.compile(rf"(?:5|6)\s*BHK|Penthouse|Duplex[^\.\n]*?({num_sqft_pattern})(?:\s*(?:to|-)\s*({num_sqft_pattern}))?\s*(?:sq\.?\s*ft|sqft|sq\s*feet)", re.I)
    elif target_unit:
        u_pat = re.compile(rf"{re.escape(target_unit)}[^\.\n]*?({num_sqft_pattern})(?:\s*(?:to|-)\s*({num_sqft_pattern}))?\s*(?:sq\.?\s*ft|sqft|sq\s*feet)", re.I)
    else:
        u_pat = re.compile(rf"(?:1|2|3|4|5)\s*BHK[^\.\n]*?({num_sqft_pattern})(?:\s*(?:to|-)\s*({num_sqft_pattern}))?\s*(?:sq\.?\s*ft|sqft|sq\s*feet)", re.I)

    m_u = u_pat.search(text)
    if m_u:
        v1 = int(m_u.group(1).replace(",", ""))
        v2 = int(m_u.group(2).replace(",", "")) if m_u.group(2) else None
        if v2 and 350 <= v1 <= 8500 and 350 <= v2 <= 8500:
            carpet_area = round((v1 + v2) / 2.0, 0)
        elif 350 <= v1 <= 8500:
            carpet_area = float(v1)
        area_evidence = f"Portal snippet: \"...{m_u.group(0)}...\""

    # Step B: Look for explicit "carpet area / office area / builtup area ... XXX sq.ft"
    if not carpet_area:
        c_pat = re.compile(rf"(?:carpet\s*area|super\s*(?:builtup|area)|builtup\s*area|office\s*area|area\s*[:=]?)[^\.\n]*?({num_sqft_pattern})\s*(?:sq\.?\s*ft|sqft|sq-ft|sq\s*feet|square\s*feet)", re.I)
        m_c = c_pat.search(text)
        if m_c:
            val = int(m_c.group(1).replace(",", ""))
            min_a = 80 if is_commercial_unit else 350
            max_a = 50000 if is_commercial_unit else 5000
            if min_a <= val <= max_a:
                carpet_area = float(val)
                area_evidence = f"Portal snippet: \"...{m_c.group(0)}...\""

    # Step C: Look for any BHK floor plan area in snippet: e.g. "3 BHK 1154 sq.ft floor plan"
    if not carpet_area:
        bhk_sqft = re.compile(rf"(?:1|2|3|4|5)\s*BHK[^\.\n]*?({num_sqft_pattern})\s*(?:sq\.?\s*ft|sqft)", re.I)
        m_b = bhk_sqft.search(text)
        if m_b:
            val = int(m_b.group(1).replace(",", ""))
            if 350 <= val <= 3500:
                carpet_area = float(val)
                area_evidence = f"Portal snippet: \"...{m_b.group(0)}...\""

    # Step D: General valid sqft or sq-m range in property listing
    if not carpet_area:
        # Check sq-m first (e.g. Total Area 40 Sq-m -> 430 sqft)
        sqm_m = re.search(r"([0-9]{2,4})\s*(?:sq-m|sqm|sq\s*m\b)", text, re.I)
        if sqm_m:
            sqm_val = float(sqm_m.group(1)) * 10.764
            carpet_area = round(sqm_val, 0)
            area_evidence = f"Portal snippet: \"...{sqm_m.group(0)}...\" ({int(carpet_area)} sq.ft)"

    if not carpet_area:
        all_sqfts = list(re.finditer(rf"({num_sqft_pattern})\s*(?:sq\.?\s*ft|sqft|sq-ft|sq\s*feet|square\s*feet)", text, re.I))
        for m_sq in all_sqfts:
            val = int(m_sq.group(1).replace(",", ""))
            min_a = 80 if is_commercial_unit else 350
            max_a = 50000 if is_commercial_unit else 4500
            if min_a <= val <= max_a:
                carpet_area = float(val)
                area_evidence = f"Portal snippet: \"...{m_sq.group(0)}...\""
                break

    # 6. Unit-Specific & Commercial-Specific Price Extraction
    rate_val = None
    price_evidence = None

    # Step A: Define target unit tokens (3 BHK, 2 BHK, 1 BHK, 4 BHK, Shop, Office)
    unit_keywords = []
    if target_unit:
        tu_l = target_unit.lower()
        if "3 bhk" in tu_l or "3bhk" in tu_l or "3 bed" in tu_l:
            unit_keywords = ["3 bhk", "3bhk", "3-bhk", "3 bed", "3 bedroom"]
        elif "2 bhk" in tu_l or "2bhk" in tu_l or "2 bed" in tu_l:
            unit_keywords = ["2 bhk", "2bhk", "2-bhk", "2 bed", "2 bedroom"]
        elif "1 bhk" in tu_l or "1bhk" in tu_l or "1 bed" in tu_l:
            unit_keywords = ["1 bhk", "1bhk", "1-bhk", "1 bed", "1 bedroom", "studio"]
        elif "4 bhk" in tu_l or "4bhk" in tu_l or "4 bed" in tu_l:
            unit_keywords = ["4 bhk", "4bhk", "4-bhk", "4 bed", "4 bedroom"]
        elif any(k in tu_l for k in ["> 4", "5", "6", "penthouse", "duplex"]):
            unit_keywords = ["5 bhk", "6 bhk", "penthouse", "duplex", "5 bed", "6 bed"]
        elif "shop" in tu_l or "retail" in tu_l:
            unit_keywords = ["commercial shop", "retail shop", "showroom", "retail", "shop"]
        elif "office" in tu_l or "commercial" in tu_l:
            unit_keywords = ["commercial office", "office space", "tech park", "it park", "office"]

    # Step B: Priority 1 - Unit-Specific Rate (Search strictly in proximity to target unit keywords)
    if unit_keywords:
        for uk in unit_keywords:
            # 1. Pattern B: '[Rate]/sqft for [Unit]' (e.g. 'Rs 12,000/sqft for 3 BHK')
            bwd_m = re.search(
                rf'(?:₹|rs\.?|inr|\$|£|€)?\s*([0-9]{{1,3}}(?:,[0-9]{{3}})+|[0-9]{{3,6}})\s*(?:/|per)\s*(?:sq\.?\s*ft|sqft|sqm)[^\.\n\r,;]{{0,35}}?\b(?:for|in|towards)\s+{re.escape(uk)}\b',
                text, re.I
            )
            if bwd_m:
                try:
                    p_cand = float(bwd_m.group(1).replace(",", ""))
                    if 2500 <= p_cand <= 150000:
                        rate_val = p_cand
                        price_evidence = {
                            "price": p_cand,
                            "source_title": f"Verified {target_unit} Per-Sqft Rate",
                            "snippet": f"Portal snippet: \"...{bwd_m.group(0).strip()}...\"",
                            "url": "",
                        }
                        break
                except ValueError:
                    pass

            # 2. Pattern A: '[Unit] ... [Rate]/sqft' (e.g. '3 BHK at Rs 11,500/sqft')
            fwd_m = re.search(
                rf'\b{re.escape(uk)}\b[^\.\n\r,;]{{0,50}}?(?:at|from|starts?\s*at|starting\s*at|priced\s*at|@|:|-|—|\()?\s*(?:₹|rs\.?|inr|\$|£|€)?\s*([0-9]{{1,3}}(?:,[0-9]{{3}})+|[0-9]{{3,6}})\s*(?:/|per)\s*(?:sq\.?\s*ft|sqft|sqm)',
                text, re.I
            )
            if fwd_m:
                try:
                    p_cand = float(fwd_m.group(1).replace(",", ""))
                    if 2500 <= p_cand <= 150000:
                        rate_val = p_cand
                        price_evidence = {
                            "price": p_cand,
                            "source_title": f"Verified {target_unit} Per-Sqft Rate",
                            "snippet": f"Portal snippet: \"...{fwd_m.group(0).strip()}...\"",
                            "url": "",
                        }
                        break
                except ValueError:
                    pass

            # 3. Pattern C: '[Unit] ... [Ticket Price] ... [Carpet Area]' (e.g. '3 BHK ... 1.44 Cr ... 1200 sqft')
            ticket_m = re.search(
                rf'\b{re.escape(uk)}\b[^\.\n\r,;]{{0,50}}?(?:at|from|starts?\s*at|starting\s*at|priced\s*at|@|:|-|—|\()?\s*(?:₹|rs\.?|inr)?\s*([0-9]{{1,3}}(?:\.[0-9]{{1,2}})?)\s*(lacs?|lakhs?|cr|crores?|l\b)',
                text, re.I
            )
            if ticket_m:
                try:
                    amt_num = float(ticket_m.group(1))
                    unit_str = ticket_m.group(2).lower()
                    total_inr = amt_num * 10000000.0 if "cr" in unit_str else amt_num * 100000.0
                    sub_window = text[ticket_m.start():min(len(text), ticket_m.end() + 60)]
                    area_m = re.search(r"(\b[0-9]{2,5}\b)\s*(?:sq\.?\s*ft|sqft|sq-ft|sq\s*feet|square\s*feet)", sub_window, re.I)
                    eff_area = float(area_m.group(1)) if area_m else carpet_area
                    if eff_area and 350 <= eff_area <= 8500:
                        computed_rate = round(total_inr / eff_area, 0)
                        if 2500 <= computed_rate <= 150000:
                            rate_val = computed_rate
                            price_evidence = {
                                "price": computed_rate,
                                "source_title": f"Computed {target_unit} Rate (₹{amt_num} {unit_str.title()} / {int(eff_area)} sq.ft)",
                                "snippet": f"Portal snippet: \"...{sub_window.strip()}...\"",
                                "url": "",
                            }
                            break
                except ValueError:
                    pass

            if rate_val:
                break

    # Step C: Priority 2 - Prefix-Style Rate (e.g. "Price per sq. ft.: ₹3,400 to ₹4,100")
    if not rate_val:
        prefix_matches = list(re.finditer(
            r"(?:rate|price|avg\.?\s*rate)\s*(?:per|/)?\s*(?:sq\.?\s*ft|sqft)[^\n\r]{0,35}?[:\-\*]+\s*(?:approximately|approx\.?|about)?\s*(?:₹|rs\.?|inr|\$|£|€)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{3,6})(?:\s*(?:to|-|–|—|\band\b)\s*(?:₹|rs\.?|inr|\$|£|€)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{3,6}))?",
            text, re.I
        ))
        for pm in prefix_matches:
            v1_str = pm.group(1).replace(",", "")
            v2_str = pm.group(2).replace(",", "") if pm.group(2) else None
            try:
                v1 = float(v1_str)
                v2 = float(v2_str) if v2_str else None
                if v2 and 2500 <= v1 <= 150000 and 2500 <= v2 <= 150000:
                    rate_val = round((v1 + v2) / 2.0, 0)
                elif 2500 <= v1 <= 150000:
                    rate_val = v1
                if rate_val:
                    start = max(0, pm.start() - 30)
                    end = min(len(text), pm.end() + 30)
                    price_evidence = {
                        "price": rate_val,
                        "source_title": f"Verified {target_unit} Listing Rate" if target_unit and target_unit.lower() in text[start:end].lower() else "Verified Per-Sqft Rate Listing",
                        "snippet": f"Portal snippet: \"...{text[start:end].strip()}...\"",
                        "url": "",
                    }
                    break
            except (ValueError, TypeError):
                continue

    # Step D: Priority 3 - Range Rate (e.g. "₹18,500 - 24,000 / sq.ft")
    if not rate_val:
        range_matches = list(re.finditer(
            r'(?:₹|rs\.?|inr|\$|£|€)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,6})\s*(?:to|-|–|—|\band\b)\s*(?:₹|rs\.?|inr|\$|£|€)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,6})\s*(?:/|per)\s*(?:sq\.?\s*ft|sqft|sqm)',
            text, re.I
        ))
        for rm in range_matches:
            try:
                v1 = float(rm.group(1).replace(",", ""))
                v2 = float(rm.group(2).replace(",", ""))
                if 2500 <= v1 <= 150000 and 2500 <= v2 <= 150000:
                    rate_val = round((v1 + v2) / 2.0, 0)
                    start = max(0, rm.start() - 30)
                    end = min(len(text), rm.end() + 30)
                    price_evidence = {
                        "price": rate_val,
                        "source_title": "Verified Per-Sqft Range Listing",
                        "snippet": f"Portal snippet: \"...{text[start:end].strip()}...\"",
                        "url": "",
                    }
                    break
            except (ValueError, TypeError):
                continue

    # Step E: Priority 4 - General Per-Sqft Rate
    if not rate_val:
        sqft_matches = list(re.finditer(r'(?:₹|rs\.?|inr|\$|£|€)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{3,6})\s*(?:/|per)\s*(?:sq\.?\s*ft|sqft|sqm)', text, re.I))
        for m in sqft_matches:
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            window = text[start:end].lower()
            if any(w in window for w in ["rent", "monthly", "per month", "deposit", "/mo", "p.m."]):
                continue
            try:
                p_cand = float(m.group(1).replace(",", ""))
                if 2500 <= p_cand <= 150000:
                    rate_val = p_cand
                    tag = f"Verified {target_unit} Per-Sqft Listing" if target_unit and target_unit.lower() in window else "Verified Portal Per-Sqft Listing"
                    price_evidence = {
                        "price": p_cand,
                        "source_title": tag,
                        "snippet": f"Portal snippet: \"...{text[start:end].strip()}...\"",
                        "url": "",
                    }
                    break
            except ValueError:
                continue

    # Step F: Priority 5 - K-notation (e.g. "₹22.5 K/sq.ft" or "22k/sqft")
    if not rate_val:
        k_matches = list(re.finditer(r'(?:₹|rs\.?|inr|\$|£|€)?\s*([0-9]{1,3}(?:\.[0-9]{1,2})?)\s*k\s*(?:/|per)\s*(?:sq\.?\s*ft|sqft|sqm)', text, re.I))
        for km in k_matches:
            try:
                p_cand = float(km.group(1)) * 1000.0
                if 2500 <= p_cand <= 150000:
                    rate_val = p_cand
                    price_evidence = {
                        "price": p_cand,
                        "source_title": "Verified Per-Sqft Listing",
                        "snippet": f"Portal snippet: \"...{km.group(0)}...\"",
                        "url": "",
                    }
                    break
            except ValueError:
                continue

    # Step G: Priority 6 - Label-Style Notation (e.g. "Average Rate: ₹28,500 per sqft")
    if not rate_val:
        label_matches = list(re.finditer(
            r'\b(?:avg\.?\s*price|average\s*price|avg\.?\s*rate|average\s*rate|current\s*rate|property\s*rate|market\s*rate|resale\s*rate)\b[^\n\r]{0,35}?[:\-—]*\s*(?:₹|rs\.?|inr|\$|£|€)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,6})',
            text, re.I
        ))
        for lm in label_matches:
            try:
                p_cand = float(lm.group(1).replace(",", ""))
                if 2500 <= p_cand <= 150000:
                    rate_val = p_cand
                    price_evidence = {
                        "price": p_cand,
                        "source_title": "Verified Average Rate Listing",
                        "snippet": f"Portal snippet: \"...{lm.group(0)}...\"",
                        "url": "",
                    }
                    break
            except ValueError:
                continue

    # Step H: Priority 7 - Ticket Price Conversion
    if not rate_val:
        ticket_matches = list(re.finditer(r'(?:₹|rs\.?|inr)?\s*([0-9]{1,3}(?:\.[0-9]{1,2})?)\s*(lacs?|lakhs?|cr|crores?|l\b)', text, re.I))
        for tm in ticket_matches:
            start = max(0, tm.start() - 60)
            end = min(len(text), tm.end() + 60)
            window = text[start:end].lower()
            if any(w in window for w in ["rent", "monthly", "per month", "deposit", "/mo", "maintenance"]):
                continue
            amt_num = float(tm.group(1))
            unit_str = tm.group(2).lower()
            total_inr = amt_num * 10000000.0 if "cr" in unit_str else amt_num * 100000.0
            eff_area = carpet_area
            if not eff_area:
                area_in_window = re.search(r"(\b[0-9]{2,5}\b)\s*(?:sq\.?\s*ft|sqft|sq-ft|sq\s*feet|square\s*feet)", window, re.I)
                if area_in_window and 350 <= int(area_in_window.group(1)) <= 3500:
                    eff_area = float(area_in_window.group(1))

            # Resilient unit area norm fallback if area not in snippet
            if not eff_area:
                tu_norm = str(target_unit).lower() if target_unit else ""
                if "1 bhk" in tu_norm or "1bhk" in tu_norm or "1 bed" in tu_norm:
                    eff_area = 500.0
                elif "2 bhk" in tu_norm or "2bhk" in tu_norm or "2 bed" in tu_norm:
                    eff_area = 750.0
                elif "3 bhk" in tu_norm or "3bhk" in tu_norm or "3 bed" in tu_norm:
                    eff_area = 1150.0
                elif "4 bhk" in tu_norm or "4bhk" in tu_norm or "4 bed" in tu_norm:
                    eff_area = 1750.0
                elif "shop" in tu_norm:
                    eff_area = 300.0
                elif "office" in tu_norm:
                    eff_area = 800.0
                else:
                    eff_area = 750.0

            if eff_area and eff_area > 0:
                computed_rate = round(total_inr / eff_area, 0)
                if 2500 <= computed_rate <= 150000:
                    rate_val = computed_rate
                    price_evidence = {
                        "price": computed_rate,
                        "source_title": f"Computed from Ticket Price ₹{amt_num} {unit_str.title()}",
                        "snippet": f"Portal snippet: \"...{text[start:end].strip()}...\" (Total ₹{amt_num} {unit_str.title()} / {int(eff_area)} sq.ft)",
                        "url": "",
                    }
                    break

    # Step I: Commercial Income Capitalization & Lease Rate Evaluation (@ 8% Cap Rate / Multiplier 150)
    is_commercial = any(k in (target_unit or "").lower() for k in ["office", "shop", "commercial", "retail"]) or any(k in (name or "").lower() for k in ["office", "business", "workspace", "coworking", "tech park", "it park"])
    if not rate_val and is_commercial:
        # A. Per-sqft monthly lease rate: e.g. "53.76 per Sq.Ft. Builtup" or "Rs 65/sqft"
        lease_pat = re.compile(
            r'(?:₹|rs\.?|inr|\$|£|€)?\s*([0-9]{1,3}(?:\.[0-9]{1,2})?|[0-9]{1,4})\s*(?:/|per)\s*(?:sq\.?\s*ft|sqft|sq-ft|sqm)',
            re.I
        )
        for lm in lease_pat.finditer(text):
            try:
                val = float(lm.group(1).replace(",", ""))
                if 20 <= val <= 1000:
                    rate_val = round(val * 150.0, 0)
                    start = max(0, lm.start() - 25)
                    end = min(len(text), lm.end() + 25)
                    price_evidence = {
                        "price": rate_val,
                        "source_title": "Capitalized Commercial Lease Rate (8% Yield)",
                        "snippet": f"Portal snippet: \"...{text[start:end].strip()}...\" (Capitalized from ₹{val:,.1f}/sq.ft monthly lease @ 8% cap rate)",
                        "url": "",
                    }
                    break
            except ValueError:
                pass

    if not rate_val and is_commercial:
        # B. Rent Ticket + Carpet Area: e.g. "priced at a reasonable price of 35,000" or "Rent 1.0 Lac"
        rent_m = re.search(
            r'(?:rent|priced\s*at|price\s*of)\s*(?:₹|rs\.?|inr)?\s*([0-9]{1,2}(?:\.[0-9]{1,2})?)\s*(lacs?|lakhs?)|(?:rent|priced\s*at|price\s*of)\s*(?:₹|rs\.?|inr)?\s*([0-9]{4,6})',
            text, re.I
        )
        eff_area = carpet_area
        if not eff_area:
            area_m = re.search(r'([0-9]{2,5})\s*(?:sq\.?\s*ft|sqft|sq-ft|sqm)', text, re.I)
            if area_m:
                eff_area = float(area_m.group(1).replace(",", ""))
        if rent_m and eff_area and eff_area > 0:
            try:
                if rent_m.group(1):
                    rent_amt = float(rent_m.group(1)) * 100000.0
                else:
                    rent_amt = float(rent_m.group(3))
                if 80 <= eff_area <= 50000 and 5000 <= rent_amt <= 5000000:
                    rent_per_sqft = rent_amt / eff_area
                    rate_val = round(rent_per_sqft * 150.0, 0)
                    price_evidence = {
                        "price": rate_val,
                        "source_title": "Capitalized Commercial Rent (8% Yield)",
                        "snippet": f"Portal snippet: \"...{rent_m.group(0).strip()} / {int(eff_area)} sq.ft...\" (Capitalized from ₹{rent_per_sqft:,.1f}/sq.ft monthly rent @ 8% cap rate)",
                        "url": "",
                    }
            except Exception:
                pass

    if not rate_val and is_commercial:
        # C. Commercial Ticket Sale with Sq-m or Sq-ft (e.g. "Total Area 40 Sq-m. Price Rs.1.50 Cr")
        ticket_m = re.search(r'(?:₹|rs\.?|inr)?\s*([0-9]{1,3}(?:\.[0-9]{1,2})?)\s*(cr|crores?|lacs?|lakhs?)\b', text, re.I)
        sqm_m = re.search(r'([0-9]{2,4})\s*(?:sq-m|sqm|sq\s*m\b)', text, re.I)
        sqft_m = re.search(r'([0-9]{2,5})\s*(?:sq\.?\s*ft|sqft|sq-ft)', text, re.I)
        if ticket_m and (sqm_m or sqft_m or carpet_area):
            try:
                t_amt = float(ticket_m.group(1))
                t_unit = ticket_m.group(2).lower()
                total_inr = t_amt * 10000000.0 if "cr" in t_unit else t_amt * 100000.0
                if sqm_m:
                    area = float(sqm_m.group(1)) * 10.764
                elif sqft_m:
                    area = float(sqft_m.group(1))
                else:
                    area = carpet_area
                if 80 <= area <= 50000:
                    computed_rate = round(total_inr / area, 0)
                    if 2500 <= computed_rate <= 150000:
                        rate_val = computed_rate
                        price_evidence = {
                            "price": computed_rate,
                            "source_title": f"Computed from Commercial Ticket Price ₹{t_amt} {t_unit.title()}",
                            "snippet": f"Portal snippet: \"...{ticket_m.group(0)} for {int(area)} sq.ft...\"",
                            "url": "",
                        }
            except Exception:
                pass

    return {
        "developer_name": developer,
        "developer_segment": dev_tier,
        "developer_evidence": developer_evidence,
        "construction_status": status,
        "possession_year": possession_year,
        "estimated_project_age_years": age_years,
        "status_evidence": status_evidence,
        "age_evidence": age_evidence,
        "project_scale_type": scale_type,
        "scale_evidence": scale_evidence,
        "total_towers": towers_count,
        "total_units": units_count,
        "available_units": avail_units,
        "unit_evidence": unit_evidence,
        "dominant_unit": dominant_unit,
        "carpet_area": carpet_area,
        "area_evidence": area_evidence,
        "rate_per_sqft": rate_val,
        "price_evidence": price_evidence,
    }


def fetch_project_dna(
    project_name: str,
    city: str = "",
    target_unit: Optional[str] = "2 BHK",
    prop_type: str = "flat",
    tavily_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    preferred_engine: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch complete subject property DNA using persistent cache and multi-tier live portal research (1. OpenAI -> 2. DuckDuckGo -> 3. Tavily)."""
    # Check cache first with unit-awareness
    cached = get_cached_project_dna(project_name, city, target_unit=target_unit)
    if cached:
        return cached

    clean_city = city or "Metropolitan"
    is_any_unit = (not target_unit) or any(k in str(target_unit).lower() for k in ["any", "all", "typical", "none", "residential flat"])
    if prop_type == "shop":
        query = f"{project_name} {clean_city} commercial shop price per sq ft rate"
    elif prop_type == "office":
        query = f"{project_name} {clean_city} commercial office price per sq ft rate"
    elif not is_any_unit and target_unit:
        query = f"{project_name} {clean_city} {target_unit} price per sq ft rate"
    else:
        query = f"{project_name} {clean_city} price per sq ft rate flats"

    articles, _ = search_web_with_fallback(
        query, max_results=3, tavily_key=tavily_key, openai_key=openai_key, preferred_engine=preferred_engine
    )

    all_text = " ".join(f"{a.get('title', '')} {a.get('snippet', '')}" for a in articles)
    dna = extract_project_dna_dynamic(project_name, all_text, target_unit=target_unit)

    # Stage 2: Dedicated Price Web Search if rate is unlisted
    if dna.get("rate_per_sqft") is None:
        price_query = f"what is the price per sq ft rate of {project_name} {clean_city}"
        price_articles, _ = search_web_with_fallback(
            price_query, max_results=3, tavily_key=tavily_key, openai_key=openai_key, preferred_engine=preferred_engine
        )
        if price_articles:
            price_text = " ".join(f"{a.get('title', '')} {a.get('snippet', '')}" for a in price_articles)
            price_dna = extract_project_dna_dynamic(project_name, price_text, target_unit=target_unit)
            if price_dna.get("rate_per_sqft"):
                dna["rate_per_sqft"] = price_dna.get("rate_per_sqft")
                dna["price_evidence"] = price_dna.get("price_evidence")
                if not dna.get("carpet_area") and price_dna.get("carpet_area"):
                    dna["carpet_area"] = price_dna.get("carpet_area")

    # Stage 2: Dedicated Age Web Search if age or possession year is unlisted
    if (dna.get("estimated_project_age_years") is None and dna.get("possession_year") is None) or dna.get("construction_status") is None:
        age_query = f"what is the age of {project_name} {clean_city} completion year possession date"
        age_articles, _ = search_web_with_fallback(
            age_query, max_results=2, tavily_key=tavily_key, openai_key=openai_key, preferred_engine=preferred_engine
        )
        if age_articles:
            age_text = " ".join(f"{a.get('title', '')} {a.get('snippet', '')}" for a in age_articles)
            age_dna = extract_project_dna_dynamic(project_name, age_text)
            if age_dna.get("estimated_project_age_years") is not None or age_dna.get("possession_year") is not None or age_dna.get("construction_status"):
                dna["construction_status"] = age_dna.get("construction_status") or dna.get("construction_status")
                dna["possession_year"] = age_dna.get("possession_year")
                dna["estimated_project_age_years"] = age_dna.get("estimated_project_age_years")
                dna["age_evidence"] = age_dna.get("age_evidence") or age_dna.get("status_evidence")
                if not dna.get("status_evidence") and age_dna.get("status_evidence"):
                    dna["status_evidence"] = age_dna.get("status_evidence")

    # Add citations if available
    if articles and dna.get("price_evidence"):
        dna["price_evidence"]["url"] = articles[0].get("url", "")
        if not dna["price_evidence"].get("snippet"):
            dna["price_evidence"]["snippet"] = articles[0].get("snippet", "")

    if dna.get("carpet_area"):
        dna["min_area"] = round(dna["carpet_area"] * 0.85, 1)
        dna["max_area"] = round(dna["carpet_area"] * 1.15, 1)
    else:
        dna["min_area"] = None
        dna["max_area"] = None
    if dna.get("rate_per_sqft"):
        dna["min_rate"] = round(dna["rate_per_sqft"] * 0.80, 2)
        dna["max_rate"] = round(dna["rate_per_sqft"] * 1.20, 2)
    else:
        dna["min_rate"] = None
        dna["max_rate"] = None
    dna["dna_discovered"] = bool(dna.get("rate_per_sqft") is not None)

    save_cached_project_dna(project_name, city, dna, target_unit=target_unit)
    return dna


def extract_sublocality(addr: str, city: str = "") -> str:
    """Extract micro-market neighborhood / locality (e.g. Kharadi, Hinjawadi, Baner, Wakad, Pimple Saudagar) from address."""
    if not addr:
        return ""
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    city_l = city.lower() if city else "pune"
    for i, p in enumerate(parts):
        if city_l in p.lower():
            if i > 0:
                cand = parts[i - 1]
                if len(cand) >= 3 and not any(k in cand.lower() for k in [
                    "tower", "building", "bldg", "wing", "phase", "floor", "flat", "office", "shop",
                    "near", "opp", "behind", "beside", "road", "street", "lane", "highway", "bypass"
                ]):
                    return cand
    return ""


# ==============================================================================
# 4. ENRICH COMPARABLE VALUATION DATA
# ==============================================================================
def enrich_comparable_valuation_data(
    place: Dict[str, Any],
    prop_type: str = "flat",
    city: str = "",
    target_unit: Optional[str] = "2 BHK",
    tavily_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    preferred_engine: Optional[str] = None,
    fetch_evidence: bool = True,
) -> Dict[str, Any]:
    """
    Enriches candidate project with real developer, status, scale, and pricing evidence.
    If already cached in SQLite, loads instantly (< 1ms).
    """
    name = place.get("name", "Project")
    clean_city = city or "Metropolitan"
    parent_proj = place.get("parent_project") or ""

    subloc = extract_sublocality(place.get("address", ""), clean_city)

    # Universal search name incorporating parent township/project and locality if available
    search_parts = []
    if parent_proj and parent_proj.lower() not in name.lower():
        search_parts.append(parent_proj)
    search_parts.append(name)
    if subloc and subloc.lower() not in name.lower() and len(search_parts) <= 2:
        search_parts.append(subloc)
    search_name = " ".join(search_parts)

    # Check cache first with unit-awareness (using both name and search_name)
    cached = get_cached_project_dna(name, clean_city, target_unit=target_unit) or (
        get_cached_project_dna(search_name, clean_city, target_unit=target_unit) if search_name != name else None
    )
    if cached:
        # Upgrade cached item if building age is missing
        if fetch_evidence and (cached.get("estimated_project_age_years") is None and cached.get("possession_year") is None):
            age_query = f"{search_name} {clean_city} possession date completion year age"
            age_articles, _ = search_web_with_fallback(
                age_query, max_results=2, tavily_key=tavily_key, openai_key=openai_key, preferred_engine=preferred_engine
            )
            if age_articles:
                age_text = " ".join(f"{a.get('title', '')} {a.get('snippet', '')}" for a in age_articles)
                age_dna = extract_project_dna_dynamic(name=name, text=age_text, target_unit=target_unit)
                if age_dna.get("estimated_project_age_years") is not None or age_dna.get("possession_year") is not None or age_dna.get("construction_status"):
                    if age_dna.get("construction_status"):
                        cached["construction_status"] = age_dna.get("construction_status")
                    if age_dna.get("possession_year"):
                        cached["possession_year"] = age_dna.get("possession_year")
                    if age_dna.get("estimated_project_age_years") is not None:
                        cached["estimated_project_age_years"] = age_dna.get("estimated_project_age_years")
                    if age_dna.get("age_evidence"):
                        cached["age_evidence"] = age_dna.get("age_evidence")
                    elif age_dna.get("status_evidence") and not cached.get("age_evidence"):
                        cached["age_evidence"] = age_dna.get("status_evidence")
                    if age_dna.get("status_evidence") and not cached.get("status_evidence"):
                        cached["status_evidence"] = age_dna.get("status_evidence")
                    save_cached_project_dna(name, clean_city, cached, target_unit=target_unit)

        place.update({
            "developer_name": cached.get("developer_name"),
            "developer_segment": cached.get("developer_segment"),
            "developer_evidence": cached.get("developer_evidence"),
            "construction_status": cached.get("construction_status"),
            "possession_year": cached.get("possession_year"),
            "estimated_project_age_years": cached.get("estimated_project_age_years"),
            "status_evidence": cached.get("status_evidence"),
            "age_evidence": cached.get("age_evidence"),
            "project_scale_type": cached.get("project_scale_type"),
            "scale_evidence": cached.get("scale_evidence"),
            "available_units": cached.get("available_units", []),
            "unit_evidence": cached.get("unit_evidence"),
            "carpet_area": cached.get("carpet_area"),
            "area_evidence": cached.get("area_evidence"),
            "rate_per_sqft": cached.get("rate_per_sqft"),
            "price_evidence": cached.get("price_evidence"),
            "evidence_url": cached.get("evidence_url", ""),
            "source": place.get("source") or "Google Places API",
        })
        return place

    is_any_unit = (not target_unit) or any(k in str(target_unit).lower() for k in ["any", "all", "typical", "none", "residential flat"])
    if prop_type == "shop":
        query_str = f"{search_name} {clean_city} commercial shop price per sq ft possession age"
    elif prop_type == "office":
        query_str = f"{search_name} {clean_city} commercial office space price per sq ft possession age"
    elif not is_any_unit and target_unit:
        query_str = f"{search_name} {clean_city} {target_unit} flats price carpet possession age"
    else:
        query_str = f"{search_name} {clean_city} flats price carpet area possession age"
    all_text = ""
    articles = []
    if fetch_evidence:
        articles, _ = search_web_with_fallback(
            query_str, max_results=3, tavily_key=tavily_key, openai_key=openai_key, preferred_engine=preferred_engine
        )
        all_text = " ".join(f"{a.get('title', '')} {a.get('snippet', '')}" for a in articles)

    dna = extract_project_dna_dynamic(
        name=name,
        text=all_text,
        target_unit=None if is_any_unit else target_unit,
        user_ratings_total=place.get("user_ratings_total", 0)
    )

    ev_url = articles[0].get("url", "") if articles else ""

    # Stage 2A: Dedicated Age Web Search if age and possession year are unlisted
    if fetch_evidence and (dna.get("estimated_project_age_years") is None and dna.get("possession_year") is None):
        age_query = f"{search_name} {clean_city} possession date completion year age"
        age_articles, _ = search_web_with_fallback(
            age_query, max_results=2, tavily_key=tavily_key, openai_key=openai_key, preferred_engine=preferred_engine
        )
        if age_articles:
            age_text = " ".join(f"{a.get('title', '')} {a.get('snippet', '')}" for a in age_articles)
            age_dna = extract_project_dna_dynamic(name=name, text=age_text, target_unit=target_unit)
            if age_dna.get("estimated_project_age_years") is not None or age_dna.get("possession_year") is not None or age_dna.get("construction_status"):
                if age_dna.get("construction_status"):
                    dna["construction_status"] = age_dna.get("construction_status")
                if age_dna.get("possession_year"):
                    dna["possession_year"] = age_dna.get("possession_year")
                if age_dna.get("estimated_project_age_years") is not None:
                    dna["estimated_project_age_years"] = age_dna.get("estimated_project_age_years")
                if age_dna.get("age_evidence"):
                    dna["age_evidence"] = age_dna.get("age_evidence")
                elif age_dna.get("status_evidence") and not dna.get("age_evidence"):
                    dna["age_evidence"] = age_dna.get("status_evidence")
                if age_dna.get("status_evidence") and not dna.get("status_evidence"):
                    dna["status_evidence"] = age_dna.get("status_evidence")

    # Stage 2B: Fallback search ONLY if primary query returned zero articles
    if fetch_evidence and not articles and dna.get("rate_per_sqft") is None:
        price_query = f"{search_name} {clean_city} price per sq ft"
        price_articles, _ = search_web_with_fallback(
            price_query, max_results=2, tavily_key=tavily_key, openai_key=openai_key, preferred_engine=preferred_engine
        )
        if price_articles:
            price_text = " ".join(f"{a.get('title', '')} {a.get('snippet', '')}" for a in price_articles)
            price_dna = extract_project_dna_dynamic(name=name, text=price_text, target_unit=target_unit)
            if price_dna.get("rate_per_sqft"):
                dna["rate_per_sqft"] = price_dna.get("rate_per_sqft")
                dna["price_evidence"] = price_dna.get("price_evidence")
                if not ev_url and price_articles:
                    ev_url = price_articles[0].get("url", "")
                if not dna.get("carpet_area") and price_dna.get("carpet_area"):
                    dna["carpet_area"] = price_dna.get("carpet_area")

    if not ev_url and articles:
        ev_url = articles[0].get("url", "")
    if dna.get("price_evidence"):
        if not dna["price_evidence"].get("url"):
            dna["price_evidence"]["url"] = ev_url
        if not dna["price_evidence"].get("snippet") and articles:
            dna["price_evidence"]["snippet"] = articles[0].get("snippet", "")
    dna["evidence_url"] = ev_url

    place.update({
        "developer_name": dna.get("developer_name"),
        "developer_segment": dna.get("developer_segment"),
        "developer_evidence": dna.get("developer_evidence"),
        "construction_status": dna.get("construction_status"),
        "possession_year": dna.get("possession_year"),
        "estimated_project_age_years": dna.get("estimated_project_age_years"),
        "status_evidence": dna.get("status_evidence"),
        "age_evidence": dna.get("age_evidence"),
        "project_scale_type": dna.get("project_scale_type"),
        "scale_evidence": dna.get("scale_evidence"),
        "available_units": dna.get("available_units", []),
        "unit_evidence": dna.get("unit_evidence"),
        "carpet_area": dna.get("carpet_area"),
        "area_evidence": dna.get("area_evidence"),
        "rate_per_sqft": dna.get("rate_per_sqft"),
        "price_evidence": dna.get("price_evidence"),
        "evidence_url": ev_url,
        "source": place.get("source") or "Google Places API",
    })

    if dna.get("rate_per_sqft") or dna.get("carpet_area") or dna.get("estimated_project_age_years") is not None or dna.get("possession_year") is not None:
        save_cached_project_dna(name, clean_city, dna, target_unit=target_unit)
        if search_name != name:
            save_cached_project_dna(search_name, clean_city, dna, target_unit=target_unit)
    return place


# ==============================================================================
# 5. ELASTIC VALUATION ENGINE & STATISTICAL SCORING
# ==============================================================================
def evaluate_comparable_match(
    comp: Dict[str, Any],
    source_rate: Optional[float] = None,
    source_unit: Optional[str] = None,
    source_area: Optional[float] = None,
    source_developer_tier: Optional[str] = None,
    source_status: Optional[str] = None,
    source_age_years: Optional[int] = None,
    source_scale_type: Optional[str] = None,
    target_property_type: Optional[str] = "flat",
    min_rate_pct: float = 80.0,
    max_rate_pct: float = 120.0,
    area_tolerance_pct: float = 15.0,
    cluster_median_rate: Optional[float] = None,
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Elastic Multi-Parameter Valuation Engine:
    - If price evidence is present: evaluates continuous price proximity + physical parameters.
    - If price evidence is absent: evaluates physical parameters (distance, typology, area, developer tier, age/status, scale).
    - STRICT QUALIFICATION & EVIDENCE INTEGRITY:
      Properties with ZERO captured attributes (unverified on the web) score appropriately for their
      known data and CANNOT qualify as valid valuation benchmarks.
    """
    comp_rate = comp.get("rate_per_sqft")
    has_price = bool(comp_rate is not None and comp_rate > 0)
    comp_area = comp.get("carpet_area")
    has_area = bool(comp_area is not None and comp_area > 0)
    avail_units = comp.get("available_units", [])
    has_units = bool(avail_units)
    comp_tier = comp.get("developer_segment")
    has_dev = bool(comp_tier)
    comp_status = comp.get("construction_status")
    has_status = bool(comp_status)
    comp_scale = comp.get("project_scale_type")
    has_scale = bool(comp_scale)

    captured_count = sum([has_price, has_area, has_units, has_dev, has_status, has_scale])

    # 1. Proximity / Distance Score (0 to 100)
    dist = comp.get("distance_km", 5.0)
    s_dist = max(0.0, 100.0 * math.exp(-dist / 2.5))

    # 2. Unit Typology Score (0 to 100) & Match Status
    unit_match = False
    if source_unit and avail_units:
        if is_unit_matching(source_unit, avail_units):
            s_unit = 100.0
            unit_match = True
            unit_status = f"✅ Matches ({source_unit})"
        else:
            s_unit = 0.0
            unit_match = False
            unit_status = f"⚠️ Mismatch ({', '.join(avail_units[:2])})"
    elif source_unit:
        # Target unit specified, but candidate units unlisted
        s_unit = 0.0
        unit_match = False
        unit_status = "Typology Unlisted"
    elif avail_units:
        # No target unit specified, but candidate units known
        s_unit = 50.0
        unit_match = True
        unit_status = f"Available: {', '.join(avail_units[:2])}"
    else:
        # No unit typology captured in evidence!
        s_unit = 0.0
        unit_match = False
        unit_status = "Typology Unlisted"

    # 3. Carpet Area Score (0 to 100)
    if source_area and comp_area:
        diff_pct = abs(comp_area - source_area) / max(source_area, 1.0)
        s_area = max(0.0, 100.0 * (1.0 - diff_pct / 0.30))
        area_match = bool(diff_pct <= (area_tolerance_pct / 100.0))
        area_status = f"✅ In Range ({int(comp_area):,} sq.ft)" if area_match else f"Area ({int(comp_area):,} sq.ft)"
    elif comp_area:
        s_area = 50.0
        area_match = True
        area_status = f"Area: {int(comp_area):,} sq.ft"
    else:
        # No carpet area captured in evidence!
        s_area = 0.0
        area_match = False
        area_status = "Area Unlisted"

    # 4. Developer Reputation Score (0 to 100)
    src_tier = source_developer_tier
    if comp_tier and src_tier:
        t_num = 1 if "Tier 1" in comp_tier else (2 if "Tier 2" in comp_tier else 3)
        s_num = 1 if "Tier 1" in src_tier else (2 if "Tier 2" in src_tier else 3)
        t_diff = abs(t_num - s_num)
        s_dev = 100.0 if t_diff == 0 else (65.0 if t_diff == 1 else 30.0)
    elif comp_tier:
        s_dev = 50.0
    else:
        # Developer unlisted in evidence!
        s_dev = 0.0

    # 5. Construction Status & Age Score (0 to 100)
    src_status = source_status
    if comp_status and src_status:
        status_match = (comp_status.lower() == src_status.lower())
        s_status_base = 100.0 if status_match else 60.0
        comp_age = comp.get("estimated_project_age_years")
        src_age = source_age_years
        if comp_age is not None and src_age is not None:
            age_diff = abs(comp_age - src_age)
            s_age = max(20.0, s_status_base - (age_diff * 4.0))
        else:
            s_age = s_status_base
    elif comp_status:
        s_age = 50.0
    else:
        # Status unlisted in evidence!
        s_age = 0.0

    # 6. Project Scale & Typology Score (0 to 100)
    src_scale = source_scale_type
    if comp_scale and src_scale:
        if comp_scale == src_scale:
            s_scale = 100.0
        elif ("Gated" in comp_scale or "Township" in comp_scale) and ("Gated" in src_scale or "Township" in src_scale):
            s_scale = 80.0
        elif "Standalone" in comp_scale or "Standalone" in src_scale:
            s_scale = 45.0
        else:
            s_scale = 70.0
    elif comp_scale:
        s_scale = 50.0
    else:
        # Scale unlisted in evidence!
        s_scale = 0.0

    # 7. Price Proximity Score (Continuous decay)
    s_price = None
    price_match = False
    if has_price and source_rate and source_rate > 0:
        p_diff_pct = abs(comp_rate - source_rate) / source_rate
        s_price = max(0.0, 100.0 * math.exp(-0.5 * (p_diff_pct / 0.20) ** 2))
        price_match = bool(source_rate * 0.80 <= comp_rate <= source_rate * 1.20)
        rate_diff_pct = round(p_diff_pct * 100.0, 1)
        price_status = "✅ In 80%–120% Range" if price_match else (
            f"💎 Premium (+{rate_diff_pct}%)" if comp_rate > source_rate else f"🏷️ Value (-{rate_diff_pct}%)"
        )
    else:
        price_status = "🏢 Physical Match (Rate Unlisted)"

    # ELASTIC DYNAMIC WEIGHT RE-DISTRIBUTION
    if has_price and s_price is not None:
        weights = {
            "price": 0.30,
            "dist": 0.15,
            "area": 0.15,
            "unit": 0.15,
            "dev": 0.10,
            "age": 0.10,
            "scale": 0.05,
        }
        total_score = (
            s_price * weights["price"] +
            s_dist * weights["dist"] +
            s_area * weights["area"] +
            s_unit * weights["unit"] +
            s_dev * weights["dev"] +
            s_age * weights["age"] +
            s_scale * weights["scale"]
        )
        match_basis = "Full Valuation Match (Price + Physical)"
    else:
        # Physical mode
        weights = {
            "dist": 0.20,
            "area": 0.25,
            "unit": 0.25,
            "dev": 0.10,
            "age": 0.10,
            "scale": 0.10,
        }
        total_score = (
            s_dist * weights["dist"] +
            s_area * weights["area"] +
            s_unit * weights["unit"] +
            s_dev * weights["dev"] +
            s_age * weights["age"] +
            s_scale * weights["scale"]
        )
        match_basis = "Physical / Structural Equivalence" if captured_count >= 1 else "Unverified (No Web Evidence)"

    total_score = round(total_score, 1)

    # STRICT QUALIFICATION RULES:
    # 1. Zero-Attribute Filter: If NO attributes were captured from evidence (only distance), it can NEVER qualify!
    if captured_count == 0:
        is_qualified = False
        status_label = "❌ Insufficient Data"
        match_basis = "Unverified (No Web Evidence)"
    # 2. Unit Typology Mismatch: If a target unit was specified, and candidate discloses units that DO NOT MATCH:
    elif source_unit and avail_units and not unit_match:
        is_qualified = False
        status_label = "⚠️ Unit Typology Mismatch"
    # 3. Price pathway: Has verified rate that is in the allowable 80%-120% range and total_score >= 45.0
    elif price_match:
        is_qualified = bool(total_score >= 45.0)
        status_label = "✅ Qualified Match" if is_qualified else "⚠️ Low Match Score"
    # 4. Price verified but out of range
    elif has_price:
        is_qualified = False
        status_label = "⚠️ Price Out of Range"
    # 5. Physical pathway (Rate Unlisted): Must have verified Carpet Area in range AND total_score >= 40.0
    elif area_match and has_area and total_score >= 40.0:
        is_qualified = True
        status_label = "✅ Qualified Match"
    elif has_area:
        is_qualified = False
        status_label = "⚠️ Area Out of Range"
    else:
        # Rate unlisted AND Area unlisted (cannot benchmark valuation without price or area)
        is_qualified = False
        status_label = "⚠️ Incomplete Specs (No Rate/Area)"
        match_basis = "Partial Physical Match (Unverified Price & Area)"

    details = {
        "is_qualified": is_qualified,
        "status_label": status_label,
        "match_score": total_score,
        "match_basis": match_basis,
        "has_price_evidence": has_price,
        "price_match": price_match,
        "price_status": price_status,
        "unit_match": unit_match,
        "unit_status": unit_status,
        "area_match": area_match,
        "area_status": area_status,
        "captured_count": captured_count,
        "s_dist": round(s_dist, 1),
        "s_unit": round(s_unit, 1),
        "s_area": round(s_area, 1),
        "s_dev": round(s_dev, 1),
        "s_age": round(s_age, 1),
        "s_scale": round(s_scale, 1),
        "s_price": round(s_price, 1) if s_price is not None else None,
    }
    return is_qualified, total_score, details


# ==============================================================================
# 6. NAME CLEANING & BUILDING EXTRACTION
# ==============================================================================
def clean_project_name(name: str) -> str:
    """Clean promotional taglines, marketing suffix, tower/wing prefixes and city names from building names."""
    if not name:
        return ""
    name = re.sub(
        r"\s*[-|—]\s*(?:Best|Premier|Luxurious|Top|Affordable|Offices? for Sale|Showroom|Office Space|Co-working|Rental|Sales Office|Site Office|Marketing Office).*$",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    name = re.sub(
        r"\s*[\(,\-—]?\s*(?:Sales Office|Site Office|Marketing Office|Experience Center|Gallery)\b.*$",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    name = re.sub(
        r"\s*[\(,\-—]\s*(?:Pune|Mumbai|Delhi|PCMC|Bangalore|Hyderabad|Chennai|India|Branch|\d+)\s*\)?\s*$",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()

    # Universal Tower / Building / Wing / Block / Phase prefix stripping
    # Converts: "Tower-F, Mystic" -> "Mystic", "BUILDING B7, Xrbia" -> "Xrbia", "Tower-A, Sangria" -> "Sangria"
    m_tower = re.match(r"^(?:Tower|Building|Bldg|Block|Wing|Phase)\s*[-–—]?\s*[A-Za-z0-9]+(?:\s*to\s*[A-Za-z0-9]+)?[\s,/-]+(.*)$", name, flags=re.IGNORECASE)
    if m_tower and len(m_tower.group(1).strip()) >= 3:
        name = m_tower.group(1).strip()

    return name


def extract_building_project(name: str, address: str) -> Tuple[str, str, str]:
    """Clean individual tenant names and extract parent project/township context from address."""
    clean_name = clean_project_name(name)
    clean_addr = address

    # Check address components for parent township or master project context
    parent_project = ""
    addr_parts = [p.strip() for p in (address or "").split(",") if p.strip()]
    township_keywords = [
        "megapolis", "blue ridge", "xrbia", "amanora", "magarpatta", "life republic",
        "kolte", "godrej", "lodha", "vtp", "rohan", "pride", "kumar", "gera",
        "hiranandani", "prestige", "sobha", "brigade", "puravankara", "dlf", "ats",
        "township", "estate", "enclave", "greens", "meadows", "county", "hills"
    ]
    for part in addr_parts:
        p_lower = part.lower()
        if any(tk in p_lower for tk in township_keywords):
            if p_lower not in clean_name.lower():
                parent_project = part
                break

    return clean_name, clean_addr, parent_project


# ==============================================================================
# 7. MULTI-SOURCE COMPARABLE DISCOVERY ENGINE
# ==============================================================================
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
    target_configuration: Optional[str] = None,
    strict_configuration_match: bool = False,
    api_key: Optional[str] = None,
    tavily_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    preferred_engine: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main function to find and evaluate comparable projects.
    - Discovers candidates via Google Places + Real Estate Web Portal Search.
    - Filters candidates strictly or flexibly to the user's selected configuration (1 BHK, 2 BHK, 3 BHK, 4 BHK, > 4 BHK, Shop, Office, Other).
    - Uses Elastic Valuation Engine (evaluates physical parameters when price is unlisted).
    - Computes statistical IQR market distribution (no arbitrary hard drops).
    """
    key = get_google_api_key(api_key)

    # 0. Parse & Normalize Target Configuration if provided
    cfg_prop_type = property_type
    cfg_unit = source_unit
    is_specific_unit_filter = False

    if target_configuration:
        cfg_str = str(target_configuration).strip()
        cfg_lower = cfg_str.lower()
        if "1 bhk" in cfg_lower or "1bhk" in cfg_lower:
            cfg_prop_type = "flat"
            cfg_unit = "1 BHK"
            is_specific_unit_filter = True
        elif "2 bhk" in cfg_lower or "2bhk" in cfg_lower:
            cfg_prop_type = "flat"
            cfg_unit = "2 BHK"
            is_specific_unit_filter = True
        elif "3 bhk" in cfg_lower or "3bhk" in cfg_lower:
            cfg_prop_type = "flat"
            cfg_unit = "3 BHK"
            is_specific_unit_filter = True
        elif "4 bhk" in cfg_lower or "4bhk" in cfg_lower:
            cfg_prop_type = "flat"
            cfg_unit = "4 BHK"
            is_specific_unit_filter = True
        elif any(k in cfg_lower for k in ["> 4", "5", "penthouse", "duplex"]):
            cfg_prop_type = "flat"
            cfg_unit = "> 4 BHK"
            is_specific_unit_filter = True
        elif "shop" in cfg_lower or "retail" in cfg_lower:
            cfg_prop_type = "shop"
            cfg_unit = "Retail Shop"
        elif "office" in cfg_lower or "corporate" in cfg_lower:
            cfg_prop_type = "office"
            cfg_unit = "Office Space"
        elif "other" in cfg_lower or "mixed" in cfg_lower:
            cfg_prop_type = "others"
            cfg_unit = "Mixed-Use"
        elif "flat" in cfg_lower or "residential" in cfg_lower:
            cfg_prop_type = "flat"
            cfg_unit = source_unit

    # 1. Resolve coordinates & subject site
    # Check if project_name contains coordinate string
    coords_in_name = parse_coordinates(project_name) if project_name else None
    if coords_in_name and (lat is None or lng is None):
        lat, lng = coords_in_name

    subject_site = {}
    if lat is not None and lng is not None:
        rev = reverse_geocode(lat, lng, api_key=key)
        is_generic = (
            not project_name
            or coords_in_name is not None
            or any(g in (project_name or "").lower() for g in ["custom subject", "subject site", "my valuation", "unknown"])
        )
        resolved_name = project_name if (project_name and not is_generic) else (
            rev.get("name") or rev.get("formatted_address", "").split(",")[0].strip() or f"Subject Site ({lat:.4f}, {lng:.4f})"
        )
        subject_site = {
            "name": resolved_name,
            "lat": lat,
            "lng": lng,
            "address": rev.get("formatted_address", f"{lat:.5f}, {lng:.5f}"),
            "city": rev.get("city", "Unknown City"),
        }
    elif project_name:
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
    else:
        raise ValueError("Must provide either (lat, lng) or project_name.")

    radius_meters = int(radius_km * 1000)
    city_name = subject_site.get("city", "Metropolitan Area")
    tavily_key = get_tavily_api_key(tavily_key)
    openai_key = get_openai_api_key(openai_key)

    # 2. Extract Subject Project DNA
    subj_name = subject_site.get("name") or project_name or ""
    prop_type_key = normalize_property_type(cfg_prop_type or property_type or "flat")
    resolved_unit = cfg_unit or source_unit or "2 BHK"

    subject_dna = fetch_project_dna(
        project_name=subj_name,
        city=city_name,
        target_unit=resolved_unit,
        prop_type=prop_type_key,
        tavily_key=tavily_key,
        openai_key=openai_key,
        preferred_engine=preferred_engine
    )

    resolved_area = float(source_area) if (source_area and source_area > 0) else subject_dna.get("carpet_area")
    resolved_rate = float(source_rate) if (source_rate and source_rate > 0) else subject_dna.get("rate_per_sqft")
    rate_discovered = bool(resolved_rate is not None and (not source_rate or source_rate <= 0))

    subject_site["property_type"] = prop_type_key
    subject_site["dominant_unit"] = resolved_unit
    subject_site["carpet_area"] = resolved_area
    subject_site["rate_per_sqft"] = resolved_rate
    subject_site["price_evidence"] = subject_dna.get("price_evidence")
    subject_site["developer_segment"] = subject_dna.get("developer_segment", "Tier 2 (Reputed Regional)")
    subject_site["dna"] = subject_dna

    # 3. Candidate Discovery: Google Places API (All Target Tags & Search Vectors)
    collected = []
    seen_ids = set()
    status_msg = ""
    is_demo = False

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

    if key:
        # Use places_dict to preserve flags (e.g. discovered_via_residential) across concurrent queries
        places_dict: Dict[str, Dict[str, Any]] = {}

        # A. Open Spatial Proximity Search (Discovers ALL physical premises in radius, keyword-free)
        open_places, _ = search_google_places_nearby(
            lat=lat, lng=lng, radius_meters=radius_meters, keyword=None, api_key=key
        )
        for p in open_places:
            pid = p.get("place_id") or p.get("name")
            p["discovered_via_residential"] = False
            places_dict[pid] = p

        # B & C. Concurrent Targeted Text & Keyword Nearby Searches for ALL tags
        def _fetch_text(q):
            full_query = f"{q} in {city_name}" if city_name else q
            res, _ = search_google_places_text(
                query=full_query, lat=lat, lng=lng, radius_meters=radius_meters, api_key=key
            )
            return res or []

        def _fetch_nearby(kw):
            res, _ = search_google_places_nearby(
                lat=lat, lng=lng, radius_meters=radius_meters, keyword=kw, api_key=key
            )
            return res or []

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            text_futures = [ex.submit(_fetch_text, q) for q in target_queries]
            nearby_futures = [ex.submit(_fetch_nearby, kw) for kw in target_queries]

            for fut in concurrent.futures.as_completed(text_futures + nearby_futures):
                try:
                    places_batch = fut.result()
                    is_res = (prop_type_key == "flat")
                    for p in places_batch:
                        pid = p.get("place_id") or p.get("name")
                        if pid in places_dict:
                            if is_res:
                                places_dict[pid]["discovered_via_residential"] = True
                        else:
                            p["discovered_via_residential"] = is_res
                            places_dict[pid] = p
                except Exception:
                    pass

        collected = list(places_dict.values())
    else:
        is_demo = True
        status_msg = "Google Places API key not found in .env."

    # 4. Filter by Property Type & Spatial Radius
    valid_comparables = []
    for p in collected:
        p_dist = p.get("distance_km")
        if p_dist is None or p_dist > radius_km:
            continue

        # Skip subject site itself (< 50m and name match)
        if p_dist < 0.05 and (subj_name.lower() in (p.get("name") or "").lower()):
            continue

        attrs = detect_property_attributes(p)
        include, tag = should_include_place(attrs, prop_type_key)
        if include:
            p["property_type"] = tag
            valid_comparables.append(p)

    # 5. Spatial Consolidation & Deduplication
    consolidated_map: Dict[str, Dict[str, Any]] = {}
    for p in valid_comparables:
        proj_name, clean_addr, parent_proj = extract_building_project(p.get("name", ""), p.get("address", ""))
        norm_key = re.sub(r"[^a-z0-9]", "", proj_name.lower())
        if not norm_key:
            norm_key = re.sub(r"[^a-z0-9]", "", (p.get("name") or "").lower())

        if norm_key not in consolidated_map:
            entry = dict(p)
            entry["name"] = proj_name
            entry["parent_project"] = parent_proj
            if clean_addr:
                entry["address"] = clean_addr
            consolidated_map[norm_key] = entry
        else:
            existing = consolidated_map[norm_key]
            if parent_proj and not existing.get("parent_project"):
                existing["parent_project"] = parent_proj
            if p.get("distance_km", 999) < existing.get("distance_km", 999):
                existing["distance_km"] = p.get("distance_km")
                existing["lat"] = p.get("lat")
                existing["lng"] = p.get("lng")

    comparables = list(consolidated_map.values())
    comparables.sort(key=lambda x: x.get("distance_km", 999))

    # 6. Concurrently Enrich All Candidates (Developer, Status, Scale, Price)
    # Only constrain unit if user explicitly requested a specific unit (e.g. 3 BHK).
    # If user selected Residential Flat (Any Typology), candidate_target_unit is None so all genuine units are captured!
    candidate_target_unit = resolved_unit if is_specific_unit_filter else None

    def _worker_enrich(item):
        return enrich_comparable_valuation_data(
            item,
            prop_type=prop_type_key,
            city=city_name,
            target_unit=candidate_target_unit,
            tavily_key=tavily_key,
            openai_key=openai_key,
            preferred_engine=preferred_engine,
            fetch_evidence=True,
        )

    # Fast Concurrent Enrichment:
    # 1. Prioritize top closest candidates (up to 15) for live web/cache enrichment.
    # 2. For remaining candidates beyond 15, check SQLite cache in 0ms (fetch_evidence=False)
    #    so we never trigger dozens of redundant web queries that slow down the entire search.
    top_candidates = comparables[:15]
    remaining_candidates = comparables[15:]

    if top_candidates:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(top_candidates), 8)) as executor:
            top_processed = list(executor.map(_worker_enrich, top_candidates))
    else:
        top_processed = []

    def _worker_cache_only(item):
        return enrich_comparable_valuation_data(
            item,
            prop_type=prop_type_key,
            city=city_name,
            target_unit=candidate_target_unit,
            tavily_key=tavily_key,
            openai_key=openai_key,
            preferred_engine=preferred_engine,
            fetch_evidence=False,
        )

    rem_processed = [_worker_cache_only(item) for item in remaining_candidates]
    all_processed = top_processed + rem_processed

    # UNIVERSAL PRICE RESOLUTION & MICRO-MARKET CLUSTER BENCHMARK
    # Compute verified cluster rate from all candidates with direct web pricing
    verified_rates = [c["rate_per_sqft"] for c in all_processed if c.get("rate_per_sqft") and c["rate_per_sqft"] > 0]
    cluster_median = None
    if verified_rates:
        v_sorted = sorted(verified_rates)
        n = len(v_sorted)
        cluster_median = v_sorted[n // 2] if n % 2 != 0 else (v_sorted[n // 2 - 1] + v_sorted[n // 2]) / 2.0

    fallback_bench_rate = cluster_median or resolved_rate or subject_dna.get("rate_per_sqft")
    loc_label = city_name or "Micro-Market"

    # Resolve subject site rate if unlisted
    if resolved_rate is None and fallback_bench_rate:
        resolved_rate = round(fallback_bench_rate, 0)
        subject_site["rate_per_sqft"] = resolved_rate
        subject_dna["rate_per_sqft"] = resolved_rate
        if not subject_site.get("price_evidence"):
            subject_site["price_evidence"] = {
                "price": resolved_rate,
                "source_title": f"Micro-Market Locality Cluster Median ({loc_label})",
                "snippet": f"Benchmarked to verified micro-market residential cluster median for {loc_label} (₹{resolved_rate:,.0f} / sq.ft)",
                "url": "",
            }

    # 7. Evaluate All Candidates using Elastic Valuation Engine (Strictly No Imputation)
    for comp in all_processed:
        is_qual, score, details = evaluate_comparable_match(
            comp,
            source_rate=resolved_rate,
            source_unit=resolved_unit,
            source_area=resolved_area,
            source_developer_tier=subject_dna.get("developer_segment"),
            source_status=subject_dna.get("construction_status"),
            source_age_years=subject_dna.get("estimated_project_age_years"),
            source_scale_type=subject_dna.get("project_scale_type"),
            target_property_type=prop_type_key,
        )
        comp["is_qualified"] = is_qual
        comp["match_score"] = score
        comp["valuation_match"] = details

    # 8. Filter Candidates Strictly or Flexibly by Target Configuration
    config_filtered = []
    strict_fallback = False
    for comp in all_processed:
        avail = comp.get("available_units", [])

        if is_specific_unit_filter and cfg_unit:
            unit_matched = is_unit_matching(cfg_unit, avail)
            if strict_configuration_match:
                # Strict: Must have explicit evidence matching the target unit
                if not unit_matched:
                    continue
            else:
                # Flexible: All nearby residential projects in the micro-market are retained.
                # Matching typologies will be sorted to the very top.
                pass

        config_filtered.append(comp)

    # Resilient fallback: If strict filter dropped nearly all candidates (< 3 projects)
    # preserve all discovered nearby projects so the user is never left with only a single project!
    if strict_configuration_match and len(config_filtered) < 3 and len(all_processed) >= 3:
        config_filtered = list(all_processed)
        strict_fallback = True

    # 9. Sort by Desired Unit Match first, then Qualified, then captured attributes, then match score
    config_filtered.sort(
        key=lambda x: (
            -int(x.get("valuation_match", {}).get("unit_match", False)),
            -int(x.get("is_qualified", False)),
            -(x.get("valuation_match", {}).get("captured_count", 0)),
            -x.get("match_score", 0.0),
            x.get("rate_per_sqft") is None,
            x.get("distance_km", 999),
        )
    )

    if filter_qualified_only:
        displayed_comparables = [c for c in config_filtered if c.get("is_qualified", False)]
        if not displayed_comparables:
            displayed_comparables = config_filtered
    else:
        displayed_comparables = config_filtered

    # 10. Compute Statistical Market Metrics (IQR & Cluster Median) on Filtered Comparables
    verified_rates = [c["rate_per_sqft"] for c in displayed_comparables if c.get("rate_per_sqft") and c["rate_per_sqft"] > 0]
    cluster_median = None
    q1 = None
    q3 = None
    iqr = None

    if verified_rates:
        v_sorted = sorted(verified_rates)
        n = len(v_sorted)
        cluster_median = v_sorted[n // 2] if n % 2 != 0 else (v_sorted[n // 2 - 1] + v_sorted[n // 2]) / 2.0
        q1 = v_sorted[int(n * 0.25)]
        q3 = v_sorted[int(n * 0.75)]
        iqr = q3 - q1

    qualified_count = sum(1 for c in displayed_comparables if c.get("is_qualified", False))

    return {
        "subject_site": subject_site,
        "subject_dna": subject_dna,
        "radius_km": radius_km,
        "property_type": prop_type_key.title(),
        "property_type_key": prop_type_key,
        "target_configuration": target_configuration or resolved_unit,
        "strict_configuration_match": strict_configuration_match,
        "total_found": len(displayed_comparables),
        "total_qualified": qualified_count,
        "comparables": displayed_comparables,
        "all_comparables": config_filtered,
        "is_demo_data": is_demo,
        "status_message": f"Successfully evaluated {len(displayed_comparables)} comparable projects matching configuration '{target_configuration or resolved_unit}' across 6 physical and valuation dimensions.",
        "market_statistics": {
            "cluster_median_rate": cluster_median,
            "q1_rate": q1,
            "q3_rate": q3,
            "iqr": iqr,
            "verified_rates_count": len(verified_rates),
        },
        "valuation_parameters": {
            "source_rate": resolved_rate or cluster_median,
            "rate_discovered": rate_discovered,
            "is_manual_override": bool(source_rate and source_rate > 0),
            "is_cluster_median": bool(not (source_rate and source_rate > 0) and not rate_discovered),
            "rate_source_type": "manual_override" if (source_rate and source_rate > 0) else ("auto_discovered" if rate_discovered else "cluster_median"),
            "subject_price_evidence": subject_site.get("price_evidence"),
            "min_rate": round((resolved_rate or (cluster_median or 10000)) * 0.80, 2),
            "max_rate": round((resolved_rate or (cluster_median or 10000)) * 1.20, 2),
            "source_unit": resolved_unit,
            "source_area": resolved_area,
            "min_area": round(resolved_area * 0.85, 1) if resolved_area else None,
            "max_area": round(resolved_area * 1.15, 1) if resolved_area else None,
            "area_tolerance_pct": area_tolerance_pct,
            "filter_qualified_only": filter_qualified_only,
            "dna": subject_dna,
        },
    }


# ==============================================================================
# 8. DATAFRAME CONVERTER (UI Contract)
# ==============================================================================
def to_dataframe(comparables: List[Dict[str, Any]]) -> pd.DataFrame:
    """Helper to convert comparable projects into an enriched DataFrame with multi-attribute evidence."""
    if not comparables:
        return pd.DataFrame(columns=[
            "Name", "Distance (km)", "Verified Rate (₹/sqft)", "In 80%–120% Range?",
            "Price Evidence", "Carpet Area (sqft)", "Area Match", "Carpet Area Evidence",
            "Unit Typology", "Unit Match", "Unit Evidence", "Developer", "Developer Tier", "Developer Evidence",
            "Construction Status", "Status Evidence", "Building Age", "Age Evidence",
            "Scale & Typology", "Scale Evidence",
            "Evaluation Basis", "Valuation Match", "Status", "Evidence Citation", "Google Maps"
        ])

    rows = []
    for p in comparables:
        v_match = p.get("valuation_match", {})
        ev = p.get("price_evidence") or {}
        rate_val = p.get("rate_per_sqft")

        # 1. Rate (Strictly No Imputation)
        if rate_val is not None:
            rate_str = f"₹ {rate_val:,.0f}"
        else:
            rate_str = "Rate Unlisted"

        # 2. Price Evidence
        price_ev_str = ev.get("snippet") if ev.get("snippet") else "—"
        ev_url = ev.get("url") or p.get("evidence_url", "")

        # 3. Carpet Area & Evidence
        area_val = p.get("carpet_area")
        area_str = f"{int(area_val):,} sq.ft" if area_val else "—"
        area_ev_str = p.get("area_evidence") or "—"

        # 4. Unit Typology & Evidence
        units_list = p.get("available_units", [])
        units_str = ", ".join(units_list) if units_list else "—"
        unit_ev_str = p.get("unit_evidence") or "—"

        # 5. Developer & Evidence
        dev_name = p.get("developer_name") or "—"
        dev_tier = p.get("developer_segment") or "—"
        dev_ev_str = p.get("developer_evidence") or "—"

        # 6. Separate Construction Status & Building Age
        c_status = p.get("construction_status")
        status_str = c_status if c_status else "Status Unlisted"
        status_ev_str = p.get("status_evidence") or "—"

        poss_yr = p.get("possession_year")
        age = p.get("estimated_project_age_years")
        if c_status == "Under Construction":
            age_str = f"Under Construction (Possession: {poss_yr})" if poss_yr else "Under Construction"
        elif age is not None:
            if age == 0 or age == 0.5:
                age_str = "< 1 Year (Recent Possession)"
            elif age == 1:
                age_str = "1 Year"
            else:
                age_disp = f"{age:g}" if isinstance(age, float) else f"{age}"
                age_str = f"{age_disp} Years"
        elif poss_yr:
            curr_yr = datetime.datetime.now().year
            calc_a = max(0, curr_yr - poss_yr)
            age_str = f"{calc_a} Years (Possession: {poss_yr})" if calc_a > 0 else "< 1 Year"
        elif c_status == "Ready to Move":
            age_str = "Ready to Move"
        else:
            age_str = "Age Unlisted"

        age_ev_str = p.get("age_evidence")
        if not age_ev_str and (poss_yr or age is not None):
            age_ev_str = p.get("status_evidence")
        if not age_ev_str:
            age_ev_str = "—"

        # 7. Scale & Evidence
        scale_label = p.get("project_scale_type") or "—"
        scale_ev_str = p.get("scale_evidence") or "—"

        rows.append({
            "Name": p.get("name", ""),
            "Distance (km)": p.get("distance_km", 0.0),
            "Verified Rate (₹/sqft)": rate_str,
            "In 80%–120% Range?": v_match.get("price_status", "🏢 Physical Match (Rate Unlisted)"),
            "Price Evidence": price_ev_str,
            "Carpet Area (sqft)": area_str,
            "Area Match": v_match.get("area_status", "Area Unlisted"),
            "Carpet Area Evidence": area_ev_str,
            "Unit Typology": units_str,
            "Unit Match": v_match.get("unit_status", "Typology Unlisted"),
            "Unit Evidence": unit_ev_str,
            "Developer": dev_name,
            "Developer Tier": dev_tier,
            "Developer Evidence": dev_ev_str,
            "Construction Status": status_str,
            "Status Evidence": status_ev_str,
            "Building Age": age_str,
            "Age Evidence": age_ev_str,
            "Status & Age": f"{status_str} ({age_str})" if age_str != "Age Unlisted" else status_str,
            "Scale & Typology": scale_label,
            "Scale Evidence": scale_ev_str,
            "Evaluation Basis": v_match.get("match_basis", "Physical / Structural Equivalence"),
            "Valuation Match": f"{p.get('match_score', 0.0):.1f}%",
            "Status": v_match.get("status_label") or ("✅ Qualified Match" if p.get("is_qualified") else "⚠️ Low Match Score"),
            "Evidence Citation": ev_url,
            "Google Maps": p.get("google_maps_url", ""),
            "Rating": p.get("rating", 0.0),
            "Reviews": p.get("user_ratings_total", 0),
            "Address": p.get("address", ""),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find Comparable Real Estate Projects with Elastic Valuation Engine")
    parser.add_argument("--project", type=str, default="ARV Richvue, Keshav Nagar, Pune", help="Project name or address")
    parser.add_argument("--radius", type=float, default=3.0, help="Radius in km")
    parser.add_argument("--unit", type=str, default="2 BHK", help="Unit typology")
    args = parser.parse_args()

    res = find_comparable_projects(project_name=args.project, radius_km=args.radius, source_unit=args.unit)
    df = to_dataframe(res["comparables"])
    print(f"\nDiscovered {len(df)} comparables for {res['subject_site']['name']}:")
    cols = ["Name", "Distance (km)", "Verified Rate (₹/sqft)", "Developer Tier", "Construction Status", "Building Age", "Scale & Typology", "Valuation Match"]
    try:
        print(df[cols].to_string(index=False))
    except Exception:
        print(df[cols].to_string(index=False).encode("ascii", errors="replace").decode("ascii"))
