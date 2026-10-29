# Real Estate Valuation POC

An AI and API-powered Real Estate Site Intelligence & Valuation Engine that takes **Latitude & Longitude** or a **Project / Location Name** and generates comprehensive institutional underwriting inputs across three core modules:

1. **`comparable.py`**: Nearby comparable real estate projects within **2 km, 5 km, 10 km** (or custom radius), bifurcated strictly across **Flat**, **Shop**, **Office**, and **Others**. Features automatic **parent building/project extraction**, **multi-unit consolidation**, **commercial income capitalization**, and **dual-pass Building Age discovery**.
2. **`cbd.py`**: Central Business District (CBD) and commercial node identification powered by a **LangGraph StateGraph Deep Research Agent** with fast-tracked Metropolitan Economic Market Resolution and dual-tier business hub extraction (< 1.5s).
3. **`neighborehood_amenity.py`**: Neighborhood civic infrastructure (Schools, Healthcare, Metro Stations, Bus Stands, Railway Stations, Supermarkets, Parks, Banks, Dining) with transit bifurcation, false-positive filtering, and high-concurrency category search (< 2.5s).
4. **`app.py`**: A unified, high-performance **Streamlit** dashboard integrating all modules with 3-way concurrent execution, `@st.cache_data` caching, live status tracking, interactive Folium maps, metric cards, 25-column directory table, and CSV exports.

---

## ⚡ Performance & Architectural Highlights

* 🚀 **Concurrent Multi-Threaded Pipeline**: Employs a 3-way `ThreadPoolExecutor` running Comparables Discovery, CBD Deep Agent, and Amenities concurrently. End-to-end execution dropped from > 5 minutes down to **6–20 seconds** (a ~17x speedup).
* 🛡️ **Persistent Quota Circuit Breaker**: File-backed quota management (`.quota_status.json`) instantly identifies exhausted API keys (`OpenAI`, `Tavily`) and auto-routes searches to DuckDuckGo with **zero retry delay** and zero thread blocking.
* 🏢 **Commercial Office Income Capitalization**: Converts monthly commercial lease rates (₹/sqft/mo) to capital rates at an **8% institutional cap rate (multiplier: 150x)**, with support for large commercial floor plates up to 50,000 sq.ft and commercial tenant filtering.
* 📅 **Dual-Pass Building Age & Possession Timeline Engine**: Features dedicated, separated columns for **`Construction Status`**, **`Status Evidence`**, **`Building Age`**, and **`Age Evidence`**. Robust regex captures direct age ranges (`1-5 years old`), completion years (`Completion Date: Dec, 2019`), and forward/inverted possession milestones (`Dec 2020 Possession`).
* ⚡ **Streamlit Multi-Tier Caching**: Uses `@st.cache_data(ttl=3600)` to ensure repeated views, tab switches, and filter adjustments render in **< 0.001 seconds**.

---

## 🚀 Quick Start

### 1. Configure `.env`
Open or create `.env` in the project root directory and add your Google Places API Key:

```env
GOOGLE_PLACES_API_KEY=your_actual_google_places_api_key_here

# Optional: OpenAI / Tavily Keys (DuckDuckGo fallback works automatically without any key!)
OPENAI_API_KEY=your_optional_openai_key_here
TAVILY_API_KEY=your_optional_tavily_key_here
```

*(Note: If no Google Places key is provided, the application automatically runs in demo mode with representative synthetic projects for immediate testing!)*

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
The app will open at `http://localhost:8501`.

---

## 🧩 Core Architecture & Modules

### 1. `comparable.py` — Comparable Real Estate Projects
* **Property Type Bifurcation**:
  - **Flat**: Residential apartments, gated housing societies, condominiums. (Excludes purely commercial towers).
  - **Shop**: Retail commercial complexes, shopping malls, arcades, high-street plazas. (Excludes purely residential projects).
  - **Office**: Grade-A commercial towers, IT tech parks, corporate campuses, business centers. Incorporates commercial lease-to-capital capitalization and tenant company filtering.
  - **Others**: Integrated mixed-use developments, commercial complexes, and industrial parks.
* **Target Configuration & Typology Matching**:
  - Allows selecting specific unit typologies (`1 BHK`, `2 BHK`, `3 BHK`, `4 BHK`, `> 4 BHK`, `Shop`, `Office Space`).
  - Strict Unit Filter option displays only comparables with verified evidence of that exact configuration.
* **Building Age & Possession Timelines**:
  - Discloses exact age (e.g., `3 Years`, `5 Years`, `11 Years`, `Ready to Move`) or scheduled delivery date for under-construction assets (`Under Construction (Possession: 2027)`).
* **Parent Building / Project Extraction**:
  - Extracts the true **parent building, mall, or commercial complex** (e.g., `Vision Flora Mall`, `Eon IT Park`, `World Trade Center`) from fragmented unit listings.
* **Multi-Unit Consolidation**:
  - Automatically merges duplicate units from the same development into a **single consolidated project entry**.

### 2. `cbd.py` — LangGraph StateGraph Deep Research Agent
* **Autonomous 6-Step Agent Workflow**:
  - `init_location`: Geocodes the site and resolves satellite towns (e.g., Wakad, Hinjawadi -> **Pune**; Gurgaon, Noida -> **Delhi NCR**; Thane, Navi Mumbai -> **Mumbai**) using `METRO_MARKET_MAP`.
  - `plan_deep_research`: Formulates multi-vector targeted search queries for primary CBDs, tech corridors, and secondary business districts.
  - `execute_deep_web_search`: Fast-tracks recognized institutional metropolitan markets in < 0.01s; executes fallback search for emerging markets.
  - `extract_classify_hubs`: Combines macro institutional CBD benchmarks with micro-market corporate business centers near the subject property (within 12 km).
  - `geocode_and_measure_distances`: Computes straight-line Haversine distances to all detected commercial centers.
  - `synthesize_valuation_report`: Compiles an institutional real estate report detailing proximity to major employment hubs.

### 3. `neighborehood_amenity.py` — Social Infrastructure & Amenities
* **Bifurcated Transit Categories**:
  - 🚇 **Metro Stations**: Rapid transit underground/elevated metro stations.
  - 🚌 **Bus Stops & Stands**: Municipal bus stops, MSRTC bus terminals, and BRT stations.
  - 🚆 **Railway Stations**: Suburban and mainline railway stations.
* **High-Concurrency Category Execution**:
  - Uses `ThreadPoolExecutor(max_workers=10)` across all 9 civic categories, fetching over 120+ amenities in **< 2.5 seconds**.
* **False-Positive Filtering (`is_valid_transit_amenity`)**:
  - Eliminates automobile repair shops (e.g., "Metro Garage"), car washes, and fast-food chains (e.g., "Subway") from appearing under transit.
* **Complete Amenity Directory**:
  - 🏫 Education & Schools | 🏥 Hospitals & Healthcare | 🚇 Metro Stations | 🚌 Bus Stops | 🚆 Railway Stations | 🛒 Shopping & Supermarkets | 🌳 Parks & Recreation | 🏦 Banks & Financial Services | 🍽️ Dining & Cafes

### 4. `app.py` — Unified Streamlit Dashboard
* **Dynamic Parallel Reactivity**: Launches 3-way concurrent valuation analysis on click or parameter changes.
* **Privacy by Design**: API keys are securely read from `.env` and never displayed in the UI.
* **Interactive Folium Map**: Color-coded visualization:
  - 🔴 **Subject Site** (with customizable buffer radius circle)
  - 🟣 **Comparable Projects**
  - 🟠 **Central Business Districts (CBDs)**
  - 🟢 **Neighborhood Amenities**
* **Instant Export**: Download filtered Comparables and Amenities directly as `.csv`.

---

## 📋 Comprehensive 25-Column Comparable Directory

| # | Column Name | Data Source | Underwriting Description |
| :---: | :--- | :--- | :--- |
| **1** | `Name` | Google Places / Web Portal | Project, society, or commercial park title |
| **2** | `Distance (km)` | Haversine Formula | Radial distance from subject site (km) |
| **3** | `Verified Rate (₹/sqft)` | Web Portals / Income Cap | Authentic capital rate or Income Capitalization |
| **4** | `In 80%–120% Range?` | Valuation Corridor | Indicates if price falls within subject benchmark tolerance |
| **5** | `Price Evidence` | Portal Web Snippet | Verbatim quotation of price per sq ft or lease rent |
| **6** | `Carpet Area (sqft)` | Floor Plans / Portals | Disclosed carpet area in sq.ft |
| **7** | `Area Match` | Tolerance Calculation | Match status against subject property carpet area (±15%) |
| **8** | `Carpet Area Evidence` | Portal Web Snippet | Verbatim quotation of unit floor area |
| **9** | `Unit Typology` | Portal Text Parser | Disclosed unit configurations (e.g. `2 BHK, 3 BHK`) |
| **10** | `Unit Match` | Rule Engine | Verification against desired unit configuration |
| **11** | `Unit Evidence` | Portal Web Snippet | Snippet proving available unit configurations |
| **12** | `Developer` | Title / Brand Ontology | Builder identity cleaned of prefixes and months |
| **13** | `Developer Tier` | Institutional Ontology | Classification: Tier 1 (National), Tier 2 (Regional), Tier 3 (Local) |
| **14** | `Developer Evidence` | Snippet / Title Match | Audit trail confirming builder entity |
| **15** | `Construction Status` | RERA / Portal Status | `Ready to Move` vs `Under Construction` |
| **16** | `Status Evidence` | Portal Web Snippet | Verbatim quotation of delivery milestone |
| **17** | `Building Age` | Completion Date / Year Built | Chronological age (e.g. `3 Years`) or possession timeline |
| **18** | `Age Evidence` | Portal Web Snippet | Verbatim snippet quoting completion year or age span |
| **19** | `Scale & Typology` | Land Acreage / Towers | Integrated Township, Gated Community, CHS, or Standalone |
| **20** | `Scale Evidence` | Portal Web Snippet | Verbatim snippet declaring acres, towers, or total units |
| **21** | `Evaluation Basis` | Mode Selector | `Full Valuation Match` vs `Physical / Structural Equivalence` |
| **22** | `Valuation Match` | Elastic Engine Score | Continuous multi-dimensional composite similarity score (%) |
| **23** | `Status` | Qualification Rules | Institutional benchmark qualification status |
| **24** | `Evidence Citation` | Portal URL | Direct link to source listing on 99acres, Magicbricks, etc. |
| **25** | `Google Maps` | Place ID Deep-Link | Direct URL to satellite, street view, and spatial footprint |

---

## 🛠️ Standalone CLI Usage

Each module can also be run independently from the command line:

```bash
# 1. Comparable Projects
python comparable.py --project "Pimple Saudagar, Pune" --radius 5 --type flat
python comparable.py --project "Eon IT Park, Kharadi, Pune" --radius 3 --type office
python comparable.py --lat 18.53957 --lng 73.94634 --radius 2 --type flat

# 2. CBD Deep Research Agent
python cbd.py --lat 18.53957 --lng 73.94634
python cbd.py --project "Bandra Kurla Complex, Mumbai"

# 3. Neighborhood Amenities
python neighborehood_amenity.py --lat 18.53957 --lng 73.94634 --radius 2
python neighborehood_amenity.py --project "Whitefield, Bengaluru" --radius 3
```
