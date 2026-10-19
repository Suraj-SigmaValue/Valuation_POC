# Real Estate Valuation POC

An AI and API-powered Real Estate Site Intelligence system that takes **Latitude & Longitude** or a **Project / Location Name** and generates comprehensive valuation inputs across three core modules:

1. **`comparable.py`**: Nearby comparable real estate projects within **2 km, 5 km, 10 km** (or custom radius), bifurcated strictly across **Flat**, **Shop**, **Office**, and **Others**. Features automatic **parent building/project extraction** and **multi-unit consolidation**.
2. **`cbd.py`**: Central Business District (CBD) and commercial node identification powered by a **LangGraph StateGraph Deep Research Agent** using **DuckDuckGo (`ddgs`)** or **Tavily AI Search** with Metropolitan Economic Market Resolution and dual-tier business hub extraction.
3. **`neighborehood_amenity.py`**: Neighborhood civic infrastructure (Schools, Healthcare, Metro Stations, Bus Stands, Railway Stations, Supermarkets, Parks, Banks, Dining) with transit bifurcation and false-positive filtering.
4. **`app.py`**: A unified, premium **Streamlit** dashboard integrating all modules with interactive Folium maps, live parameter reactivity, metric cards, data tables, and CSV exports.

---

## 🚀 Quick Start

### 1. Configure `.env`
Open or create `.env` in the project root directory and add your Google Places API Key:

```env
GOOGLE_PLACES_API_KEY=your_actual_google_places_api_key_here

# Optional: Tavily API Key (DuckDuckGo works automatically without any key)
TAVILY_API_KEY=your_optional_tavily_key_here
```

*(Note: If no API key is provided, the application automatically runs in demo mode with representative synthetic projects for immediate testing!)*

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
  - **Office**: Grade-A commercial towers, IT tech parks, corporate campuses, business centers. (Excludes purely residential projects).
  - **Others**: Integrated mixed-use developments, commercial complexes, and industrial parks.
* **Parent Building / Project Extraction**:
  - Real estate Google Places queries frequently return individual shop and office tenants (e.g., *"Pizza Arcade"*, *"Space Realty"*, *"Cubeage Technology Services"*).
  - `extract_building_project` dynamically parses the place name and address lines to extract the true **parent building, mall, or commercial complex** (e.g., `Vision Flora Mall`, `Sagar Complex`, `Vision 9 Mall`, `Rose Icon Society`).
* **Multi-Unit Consolidation**:
  - When multiple units/tenants originate from the same project, they are automatically merged into a **single consolidated project entry**.
  - Displays the project name, closest distance, best rating, and a `Units Consolidated` indicator.

### 2. `cbd.py` — LangGraph StateGraph Deep Research Agent
* **Autonomous 6-Step Agent Workflow**:
  - `init_location`: Geocodes the site and resolves satellite towns (e.g., Wakad, Pimpri-Chinchwad, Hinjawadi -> **Pune**; Gurgaon, Noida -> **Delhi NCR**; Thane, Navi Mumbai -> **Mumbai**) using `METRO_MARKET_MAP`.
  - `plan_deep_research`: Formulates multi-vector targeted search queries for primary CBDs, tech corridors, and secondary business districts.
  - `execute_deep_web_search`: Runs live web search via DuckDuckGo (`ddgs`) or Tavily AI.
  - `extract_classify_hubs`: Extracts verified commercial hubs with boundary enforcement (distance <= 35 km) to prevent hallucinations.
  - `geocode_and_measure_distances`: Computes straight-line Haversine distances to all detected commercial centers.
  - `synthesize_valuation_report`: Compiles an institutional real estate report detailing proximity to major employment hubs.

### 3. `neighborehood_amenity.py` — Social Infrastructure & Amenities
* **Bifurcated Transit Categories**:
  - 🚇 **Metro Stations**: Rapid transit underground/elevated metro stations.
  - 🚌 **Bus Stops & Stands**: Municipal bus stops, MSRTC bus terminals, and BRT stations.
  - 🚆 **Railway Stations**: Suburban and mainline railway stations.
* **False-Positive Filtering (`is_valid_transit_amenity`)**:
  - Eliminates automobile repair shops (e.g., "Metro Garage"), car washes, and fast-food chains (e.g., "Subway") from appearing under transit.
* **Complete Amenity Directory**:
  - 🏫 Education & Schools
  - 🏥 Hospitals & Healthcare
  - 🚇 Metro Stations
  - 🚌 Bus Stops & Stands
  - 🚆 Railway Stations
  - 🛒 Shopping & Supermarkets
  - 🌳 Parks & Recreation
  - 🏦 Banks & Financial Services
  - 🍽️ Dining & Cafes

### 4. `app.py` — Unified Streamlit Dashboard
* **Dynamic Reactivity**: Auto-triggers live analysis whenever property type (Flat/Shop/Office/Others) or search radius changes.
* **Privacy by Design**: API keys are securely read from `.env` and never displayed in the UI.
* **Interactive Folium Map**: Color-coded visualization:
  - 🔴 **Subject Site** (with customizable buffer radius circle)
  - 🟣 **Comparable Projects**
  - 🟠 **Central Business Districts (CBDs)**
  - 🟢 **Neighborhood Amenities**
* **Instant Export**: Download filtered Comparables and Amenities directly as `.csv`.

---

## 🛠️ Standalone CLI Usage

Each module can also be run independently from the command line:

```bash
# 1. Comparable Projects
python comparable.py --project "Pimple Saudagar, Pune" --radius 5 --type office
python comparable.py --lat 18.60047 --lng 73.75644 --radius 5 --type shop
python comparable.py --lat 19.0660 --lng 72.8670 --radius 5 --type flat

# 2. CBD Deep Research Agent
python cbd.py --lat 18.60047 --lng 73.75644
python cbd.py --project "Bandra Kurla Complex, Mumbai" --engine duckduckgo

# 3. Neighborhood Amenities
python neighborehood_amenity.py --lat 18.60047 --lng 73.75644 --radius 5
python neighborehood_amenity.py --project "Whitefield, Bengaluru" --radius 3
```

---

## 📁 Repository Structure

```
Valuation_POC/
├── .env                        # Environment configuration (GOOGLE_PLACES_API_KEY, TAVILY_API_KEY)
├── .env.example                # Template environment file
├── requirements.txt            # Python dependencies (streamlit, ddgs, langgraph, folium, etc.)
├── README.md                   # Documentation and usage guide
│
├── utils.py                    # Shared geocoding, reverse geocoding, and Haversine distance utilities
├── comparable.py               # Module 1: Nearby comparable projects with unit consolidation
├── cbd.py                      # Module 2: LangGraph StateGraph deep research agent for CBDs
├── neighborehood_amenity.py    # Module 3: Civic amenities with bifurcated transit infrastructure
└── app.py                      # Unified Streamlit Dashboard with interactive mapping
```
