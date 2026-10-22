"""
app.py - Real Estate Valuation & Site Intelligence POC
Unified Streamlit Dashboard integrating:
1. comparable.py (Nearby Comparable Projects with 2/5/10 km radius and Flat / Shop / Office / Others bifurcation)
2. cbd.py (Central Business District Intelligence via LangGraph StateGraph Deep Research Agent)
3. neighborehood_amenity.py (Schools, Hospitals, Transit, Retail, Parks & Civic Amenities)
"""

import os
import html
import warnings
import importlib

# Suppress library deprecation and package rename warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*renamed to.*")
warnings.filterwarnings("ignore", message=".*use_container_width.*")

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Import and reload project modules so running server never retains stale in-memory code
import utils
import comparable
import cbd
import neighborehood_amenity

importlib.reload(utils)
importlib.reload(comparable)
importlib.reload(cbd)
importlib.reload(neighborehood_amenity)

# Load .env
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Real Estate Valuation POC",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #cbd5e1;
        margin-top: 4px;
    }

    /* Badge Pills for bifurcated property types */
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-flat {
        background-color: #065f46;
        color: #6ee7b7;
        border: 1px solid #059669;
    }
    .badge-shop {
        background-color: #7c2d12;
        color: #fdba74;
        border: 1px solid #ea580c;
    }
    .badge-office {
        background-color: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #2563eb;
    }
    .badge-others {
        background-color: #581c87;
        color: #d8b4fe;
        border: 1px solid #9333ea;
    }
    .badge-live {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #10b981;
    }
    .badge-demo {
        background-color: #78350f;
        color: #fcd34d;
        border: 1px solid #f59e0b;
    }
    .badge-source {
        background-color: #0f172a;
        color: #94a3b8;
        border: 1px solid #475569;
    }
    .badge-qualified {
        background-color: #064e3b;
        color: #6ee7b7;
        border: 1px solid #10b981;
    }
    .badge-partial {
        background-color: #451a03;
        color: #fcd34d;
        border: 1px solid #d97706;
    }
    .badge-inrange {
        background-color: #042f2e;
        color: #5eead4;
        border: 1px solid #0d9488;
    }
    .badge-outrange {
        background-color: #450a0a;
        color: #fca5a5;
        border: 1px solid #dc2626;
    }

    /* Result Card */
    .prop-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: transform 0.15s ease-in-out;
    }
    .prop-card:hover {
        border-color: #38bdf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================= SIDEBAR =================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/apartment.png", width=64)
    st.title("Valuation POC")
    st.caption("Site & Micro-Market Valuation Intelligence")

    st.markdown("---")
    st.subheader("📍 1. Custom Subject Site Input")

    input_mode = st.radio(
        "Input Mode:",
        options=["Project / Location Name", "Coordinates (Lat / Long)"],
        horizontal=True,
    )

    if input_mode == "Project / Location Name":
        project_name = st.text_input(
            "Project / Locality Name:",
            value="Bandra Kurla Complex, Mumbai",
            placeholder="e.g. Bandra Kurla Complex, Mumbai",
            help="Enter any custom project or locality name. Coordinates will be geocoded automatically.",
        )
        input_lat = None
        input_lng = None
    else:
        col_lat, col_lng = st.columns(2)
        with col_lat:
            input_lat = st.number_input("Latitude:", value=19.0660, format="%.6f")
        with col_lng:
            input_lng = st.number_input("Longitude:", value=72.8670, format="%.6f")
        project_name = st.text_input(
            "Project / Site Label (Optional):",
            value="Custom Subject Property",
            placeholder="e.g. My Valuation Site",
        )

    st.markdown("---")
    st.subheader("🎯 2. Radius Selection")
    radius_mode = st.radio(
        "Select Search Radius:",
        options=["2 km", "5 km", "10 km", "Custom Slider"],
        index=1,  # 5 km default
        horizontal=True,
    )

    if radius_mode == "2 km":
        chosen_radius_km = 2.0
    elif radius_mode == "5 km":
        chosen_radius_km = 5.0
    elif radius_mode == "10 km":
        chosen_radius_km = 10.0
    else:
        chosen_radius_km = st.slider("Custom Radius (km):", min_value=0.5, max_value=25.0, value=5.0, step=0.5)

    st.markdown("---")
    st.subheader("🎯 3. Valuation Benchmark Mode")
    st.caption(
        "🤖 **100% Automated**: Property type, unit typology, carpet area, and benchmark rate are all automatically discovered from real estate web evidence for your subject project."
    )

    with st.expander("⚙️ Optional Manual Overrides", expanded=False):
        override_active = st.checkbox("Enable manual parameter overrides", value=False)
        if override_active:
            prop_choice = st.selectbox(
                "Property Type Override:",
                options=["Auto-Detect from Site", "Flat (Residential)", "Shop (Retail)", "Office (Corporate)", "Others (Mixed-Use)"],
                index=0
            )
            unit_override = st.text_input("Unit Typology Override:", value="", placeholder="Leave blank for auto-detected unit (e.g. 2 BHK)")
            area_override = st.number_input("Carpet Area Override (sq.ft):", min_value=0.0, value=0.0, step=50.0, help="Set 0 for auto-detection")
            rate_override = st.number_input("Benchmark Rate Override (₹ / sq.ft):", min_value=0.0, value=0.0, step=500.0, help="Set 0 for auto-detection")
        else:
            prop_choice = "Auto-Detect from Site"
            unit_override = None
            area_override = None
            rate_override = None

        area_tolerance = st.slider(
            "Area Tolerance (± %):",
            min_value=5,
            max_value=30,
            value=15,
        )

        filter_qualified_toggle = st.checkbox(
            "Filter only Qualified Comparables (All 4 Criteria Matched)",
            value=False,
            help="Show only properties matching: Same Property Type + Same Unit + Carpet Area in Range + Price in 80%–120%."
        )

    if "flat" in prop_choice.lower():
        selected_prop_type = "flat"
    elif "shop" in prop_choice.lower():
        selected_prop_type = "shop"
    elif "office" in prop_choice.lower():
        selected_prop_type = "office"
    elif "other" in prop_choice.lower():
        selected_prop_type = "others"
    else:
        selected_prop_type = None

    source_bench_rate = float(rate_override) if (rate_override and rate_override > 0) else None
    selected_unit = unit_override.strip() if (unit_override and unit_override.strip()) else None
    source_bench_area = float(area_override) if (area_override and area_override > 0) else None

    st.markdown("---")
    st.subheader("⚙️ 4. System Status & Search")

    # API Keys are loaded purely from .env without exposing in UI
    has_google_key = bool(utils.get_google_api_key())
    if has_google_key:
        st.success("Google Places API: Active (.env)", icon="✅")
    else:
        st.info("Google Places API: Demo Mode\n(Add GOOGLE_PLACES_API_KEY in .env for live API)", icon="💡")

    search_engine_choice = st.selectbox(
        "CBD Web Search Engine:",
        options=["DuckDuckGo (Free / No Key)", "Tavily AI Search (from .env)"],
        index=0,
    )

    st.markdown("---")
    run_btn = st.button("🚀 Run Valuation Analysis", type="primary", width="stretch")

# Main Title & Header
st.title("🏢 Real Estate Valuation & Site Analysis POC")
st.markdown(
    "Automated intelligence report extracting **Comparable Projects**, **CBD Connectivity**, and **Neighborhood Amenities**."
)

# Current search parameters signature
current_params = {
    "input_mode": input_mode,
    "project_name": (project_name or "").strip().lower(),
    "lat": round(float(input_lat), 5) if input_lat is not None else None,
    "lng": round(float(input_lng), 5) if input_lng is not None else None,
    "radius_km": float(chosen_radius_km),
    "prop_type": selected_prop_type,
    "source_rate": source_bench_rate,
    "source_unit": selected_unit,
    "source_area": source_bench_area,
    "area_tol": float(area_tolerance),
    "qualified_only": bool(filter_qualified_toggle),
    "engine": search_engine_choice,
}

# Automatically re-run when parameters change or clicks Run
should_run = (
    run_btn
    or "analysis_data" not in st.session_state
    or st.session_state.get("last_params") != current_params
)

if should_run:
    with st.spinner("Analyzing site micro-market & auto-extracting property DNA..."):
        try:
            # Keys loaded automatically from .env
            active_google_key = utils.get_google_api_key()
            tavily_key = utils.get_tavily_api_key() if "Tavily" in search_engine_choice else None

            # 1. Fetch Comparable Projects with 4-Point Auto-Discovered Baseline
            comp_result = comparable.find_comparable_projects(
                lat=input_lat,
                lng=input_lng,
                project_name=project_name if (input_mode == "Project / Location Name" or project_name) else None,
                radius_km=chosen_radius_km,
                property_type=selected_prop_type,
                source_rate=source_bench_rate,
                source_unit=selected_unit,
                source_area=source_bench_area,
                area_tolerance_pct=area_tolerance,
                filter_qualified_only=filter_qualified_toggle,
                api_key=active_google_key,
            )

            resolved_lat = comp_result["subject_site"]["lat"]
            resolved_lng = comp_result["subject_site"]["lng"]
            resolved_name = comp_result["subject_site"]["name"]
            resolved_addr = comp_result["subject_site"]["address"]
            resolved_city = comp_result["subject_site"]["city"]

            # 2. Fetch CBD Analysis via LangGraph Deep Agent
            cbd_result = cbd.analyze_cbd(
                lat=resolved_lat,
                lng=resolved_lng,
                project_name=resolved_name,
                search_engine="tavily" if "Tavily" in search_engine_choice else "duckduckgo",
                tavily_key=tavily_key,
                google_key=active_google_key,
            )

            # 3. Fetch Amenities
            amenity_result = neighborehood_amenity.fetch_neighborhood_amenities(
                lat=resolved_lat,
                lng=resolved_lng,
                project_name=resolved_name,
                radius_km=chosen_radius_km,
                api_key=active_google_key,
            )

            st.session_state["analysis_data"] = {
                "comparables": comp_result,
                "cbd": cbd_result,
                "amenities": amenity_result,
                "subject_site": {
                    "name": resolved_name,
                    "lat": resolved_lat,
                    "lng": resolved_lng,
                    "address": resolved_addr,
                    "city": resolved_city,
                },
                "radius_km": chosen_radius_km,
                "property_type": selected_prop_type,
            }
            st.session_state["last_params"] = current_params
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

# Display Results
if "analysis_data" in st.session_state:
    data = st.session_state["analysis_data"]
    site = data["subject_site"]
    comp_data = data["comparables"]
    cbd_data = data["cbd"]
    amenity_data = data["amenities"]

    # Subject Site Overview Card
    st.markdown("### 📌 Subject Site Overview")
    info_col1, info_col2, info_col3, info_col4 = st.columns([3, 2, 2, 2])

    with info_col1:
        st.markdown(f"**Site:** `{site['name']}`")
        st.markdown(f"**Address:** {site['address'] or 'N/A'}")

    with info_col2:
        st.markdown(f"**Coordinates:** `{site['lat']:.5f}, {site['lng']:.5f}`")
        st.markdown(f"**City / Market:** `{site['city']}`")

    with info_col3:
        st.markdown(f"**Buffer Radius:** `{data['radius_km']} km`")
        prop_tag = data.get("property_type", "flat")
        if prop_tag == "flat":
            st.markdown('<span class="badge-pill badge-flat">Flats / Residential</span>', unsafe_allow_html=True)
        elif prop_tag == "shop":
            st.markdown('<span class="badge-pill badge-shop">Shops / Retail</span>', unsafe_allow_html=True)
        elif prop_tag == "office":
            st.markdown('<span class="badge-pill badge-office">Offices / Corporate</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-pill badge-others">Others / Mixed-Use</span>', unsafe_allow_html=True)

    with info_col4:
        if comp_data.get("is_demo_data"):
            st.markdown('<span class="badge-pill badge-demo">Demo Mode (Set .env for Live)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-pill badge-live">Live Google Places API</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Multi-tab view for the 3 files
    tab_comp, tab_cbd, tab_amenities, tab_map = st.tabs([
        "🏢 Comparable Projects (comparable.py)",
        "🌆 Central Business District (cbd.py)",
        "🎒 Neighborhood Amenities (neighborehood_amenity.py)",
        "🗺️ Interactive Location Map",
    ])

    # ================= TAB 1: COMPARABLE PROJECTS =================
    with tab_comp:
        st.subheader("🏢 Comparable Real Estate Projects & Valuation Matching")
        st.caption(f"Showing nearby projects within **{data['radius_km']} km** matching property filter: **{comp_data['property_type']}**")

        vp = comp_data.get("valuation_parameters", {})
        comps = comp_data.get("comparables", [])

        # Valuation Benchmark Target Banner (80% to 120%)
        rate_val = vp.get("source_rate")
        unit_val = vp.get("source_unit", "2 BHK")
        area_val = vp.get("source_area", 850.0)
        prop_label = comp_data.get("property_type", "Residential Flat")
        subj_ev = vp.get("subject_price_evidence") or site.get("price_evidence") or {}

        if rate_val:
            clean_subj_title = html.escape((subj_ev.get("source_title") or "Verified Portal Listing")[:45])
            subj_url = html.escape(subj_ev.get("url", "#"))
            ev_link = f' <a href="{subj_url}" target="_blank" style="color: #38bdf8; text-decoration: underline;">({clean_subj_title} ↗)</a>' if subj_ev.get("url") else ""
            discovered_tag = " (Auto-Discovered)" if vp.get("rate_discovered") else " (Custom Override)"
            rate_info = f'Source Rate: <b style="color: #38bdf8;">₹{rate_val:,.0f} / sq.ft</b>{discovered_tag}{ev_link} <span style="color: #94a3b8;">(80%–120%: <b>₹{vp.get("min_rate", 0):,.0f} — ₹{vp.get("max_rate", 0):,.0f}</b>)</span>'
        else:
            rate_info = 'Source Rate: <i style="color: #fbbf24;">Rate not disclosed on web for subject property</i>'

        area_info = f'Carpet Area: <b style="color: #fcd34d;">{area_val:,.0f} sq.ft</b> <span style="color: #94a3b8;">(±15%: <b>{vp.get("min_area", 0):,.0f} — {vp.get("max_area", 0):,.0f} sq.ft</b>)</span>' if area_val else 'Carpet Area: <b style="color: #fcd34d;">Standard Area</b>'

        banner_html = (
            f'<div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%); border: 1px solid #3b82f6; border-radius: 10px; padding: 12px 18px; margin-bottom: 16px;">'
            f'<div style="display: flex; flex-wrap: wrap; gap: 14px; align-items: center; font-size: 0.88rem;">'
            f'<div><span style="color: #93c5fd; font-weight: 700;">🎯 Auto-Extracted Baseline:</span></div>'
            f'<div>Type: <b style="color: #34d399;">{prop_label}</b></div>'
            f'<div><span style="color: #475569;">|</span></div>'
            f'<div>Unit: <b style="color: #a78bfa;">{unit_val}</b></div>'
            f'<div><span style="color: #475569;">|</span></div>'
            f'<div>{area_info}</div>'
            f'<div><span style="color: #475569;">|</span></div>'
            f'<div>{rate_info}</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(banner_html, unsafe_allow_html=True)

        # High-level Metrics
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Total Comparables Found", f"{comp_data.get('total_found', len(comps))}")
        with mcol2:
            st.metric("Qualified (All 4 Criteria)", f"{comp_data.get('total_qualified', 0)} projects")
        with mcol3:
            if vp.get("min_rate"):
                range_str = f"₹{vp['min_rate']:,.0f} — ₹{vp['max_rate']:,.0f}"
            else:
                range_str = "Awaiting Subject Rate"
            st.metric("Allowable Price Range (80%–120%)", range_str)
        with mcol4:
            top_score = f"{comps[0].get('match_score', 0):.1f}%" if comps else "N/A"
            st.metric("Highest Match Score", top_score)

        if comp_data.get("status_message"):
            if comp_data.get("is_demo_data"):
                st.warning(comp_data["status_message"], icon="⚠️")
            else:
                st.success(comp_data["status_message"], icon="✅")

        # Top Highlight Cards (Up to 6 top matching projects in rows of 3)
        if comps:
            st.markdown("#### 🌟 Top Matching Comparable Properties")
            highlight_count = min(len(comps), 6)
            for chunk_start in range(0, highlight_count, 3):
                chunk = comps[chunk_start:chunk_start + 3]
                card_cols = st.columns(len(chunk))
                for i, p in enumerate(chunk):
                    v_match = p.get("valuation_match", {})
                    has_ev = bool(p.get("rate_per_sqft") is not None and p.get("price_evidence"))
                    ev = p.get("price_evidence") or {}

                    if has_ev:
                        price_badge = (
                            '<span class="badge-pill badge-inrange">✅ In 80%–120% Range</span>'
                            if v_match.get("price_match")
                            else '<span class="badge-pill badge-outrange">❌ Out of 80%–120% Range</span>'
                        )
                        status_badge = (
                            '<span class="badge-pill badge-qualified">✅ Qualified Match</span>'
                            if p.get("is_qualified")
                            else '<span class="badge-pill badge-partial">⚠️ Partial Match</span>'
                        )
                        unit_badge = (
                            f'<span class="badge-pill badge-qualified">🛏️ {p["available_units"][0] if p.get("available_units") else vp.get("source_unit", "2 BHK")}</span>'
                            if v_match.get("unit_match")
                            else '<span class="badge-pill badge-outrange">🛏️ Unit Mismatch</span>'
                        )
                        area_badge = (
                            f'<span class="badge-pill badge-inrange">📐 Area in Range</span>'
                            if v_match.get("area_match")
                            else '<span class="badge-pill badge-outrange">📐 Area Mismatch</span>'
                        )
                        price_html = f'<p style="margin: 3px 0; color: #f8fafc; font-size: 0.95rem;">💰 Verified Rate: <b>₹{p["rate_per_sqft"]:,.0f} / sq.ft</b></p>'
                        clean_title = html.escape((ev.get("source_title") or "Listing Source")[:45])
                        raw_snip = (ev.get("snippet") or "")[:140].replace("\r", " ").replace("\n", " ")
                        clean_snippet = html.escape(raw_snip)
                        clean_ev_url = html.escape(ev.get("url", "#"))
                        evidence_box = (
                            f'<div style="background: rgba(15, 23, 42, 0.7); border-left: 3px solid #10b981; border-radius: 4px; padding: 6px 10px; margin: 8px 0; font-size: 0.78rem;">'
                            f'<span style="color: #6ee7b7; font-weight: 600;">📑 Price Evidence:</span> '
                            f'<a href="{clean_ev_url}" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: 500;">{clean_title} ↗</a>'
                            f'<p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.74rem; line-height: 1.3;">"{clean_snippet}..."</p>'
                            f'</div>'
                        )
                    else:
                        price_badge = '<span class="badge-pill badge-source">⚠️ No Price Evidence</span>'
                        status_badge = '<span class="badge-pill badge-partial">⚠️ Price Unverified</span>'
                        unit_badge = '<span class="badge-pill badge-source">🛏️ Unit Unverified</span>'
                        area_badge = '<span class="badge-pill badge-source">📐 Area Unverified</span>'
                        price_html = '<p style="margin: 3px 0; color: #94a3b8; font-size: 0.85rem;">💰 Price: <i>Not Disclosed / No Verified Web Evidence</i></p>'
                        evidence_box = (
                            f'<div style="background: rgba(15, 23, 42, 0.4); border-left: 3px solid #475569; border-radius: 4px; padding: 6px 10px; margin: 8px 0; font-size: 0.75rem; color: #94a3b8;">'
                            f'Price not disclosed on web. Search real estate portals manually below.'
                            f'</div>'
                        )

                    source_name = html.escape(str(p.get("source", "Google Places API")))
                    source_badge = f'<span class="badge-pill badge-source">🌐 {source_name}</span>'
                    carpet_str = f"{p['carpet_area']:,} sq.ft" if p.get("carpet_area") else "Complex Standard"
                    units_str = html.escape(", ".join(p.get("available_units", [])))
                    prop_name = html.escape(str(p.get("name", "Project")))
                    maps_url = html.escape(str(p.get("google_maps_url", "#")))
                    search_url = html.escape(str(p.get("web_search_url", "#")))

                    card_html = (
                        f'<div class="prop-card">'
                        f'<div style="display: flex; justify-content: space-between; align-items: start;">'
                        f'<h4 style="margin: 0; color: #38bdf8; font-size: 1.05rem;">{prop_name}</h4>'
                        f'<span style="font-weight: 700; color: #38bdf8; font-size: 1.05rem;">{p.get("match_score", 0):.1f}%</span>'
                        f'</div>'
                        f'<div style="margin: 6px 0 8px 0; display: flex; flex-wrap: wrap; gap: 4px;">{status_badge} {price_badge} {unit_badge} {area_badge}</div>'
                        f'{price_html}'
                        f'<p style="margin: 3px 0; color: #e2e8f0; font-size: 0.85rem;">📐 Carpet Area: <b>{carpet_str}</b> | Units: <b>{units_str}</b></p>'
                        f'<p style="margin: 3px 0; color: #e2e8f0; font-size: 0.85rem;">📍 <b>{p["distance_km"]} km</b> from subject site</p>'
                        f'<p style="margin: 3px 0; color: #94a3b8; font-size: 0.8rem;">⭐ Rating: <b>{p.get("rating", "N/A")}</b> ({p.get("user_ratings_total", 0)} reviews)</p>'
                        f'{evidence_box}'
                        f'<div style="margin-top: 8px;">{source_badge}</div>'
                        f'<div style="margin-top: 10px; display: flex; gap: 12px;">'
                        f'<a href="{maps_url}" target="_blank" style="color: #60a5fa; font-size: 0.8rem; text-decoration: none;">Google Maps ↗</a>'
                        f'<a href="{search_url}" target="_blank" style="color: #34d399; font-size: 0.8rem; text-decoration: none;">Search 99acres ↗</a>'
                        f'</div>'
                        f'</div>'
                    )

                    with card_cols[i]:
                        st.markdown(card_html, unsafe_allow_html=True)

        # Full Dataframe Table
        st.markdown("#### 📋 All Comparable Projects Directory")
        df_comps = comparable.to_dataframe(comps)
        if not df_comps.empty:
            cols_to_display = [
                c for c in [
                    "Name", "Distance (km)", "Verified Rate (₹/sqft)", "In 80%–120% Range?",
                    "Unit Typology", "Carpet Area (sqft)", "Area Match",
                    "Price Evidence Source", "Evidence URL", "Valuation Match",
                    "Status", "Google Maps", "Portal Search"
                ] if c in df_comps.columns
            ]
            st.dataframe(
                df_comps[cols_to_display],
                width="stretch",
                column_config={
                    "Evidence URL": st.column_config.LinkColumn("Evidence Citation", display_text="View Evidence ↗"),
                    "Google Maps": st.column_config.LinkColumn("Map Link", display_text="Open Map ↗"),
                    "Portal Search": st.column_config.LinkColumn("Portal Search", display_text="Search 99acres ↗"),
                    "Distance (km)": st.column_config.NumberColumn(format="%.2f km"),
                },
            )
            # Download CSV
            csv_comp = df_comps.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Comparables Valuation CSV",
                data=csv_comp,
                file_name="comparable_projects_valuation.csv",
                mime="text/csv",
            )
        else:
            st.info("No comparable projects found matching the criteria.")

    # ================= TAB 2: CBD ANALYSIS =================
    with tab_cbd:
        st.subheader("🌆 Central Business District (CBD) Intelligence")
        col_cbd_badge, col_cbd_engine = st.columns([3, 1])
        with col_cbd_badge:
            st.markdown(
                '<span class="badge-pill badge-live">🤖 Powered by LangGraph StateGraph Deep Agent</span>'
                f'<span class="badge-pill badge-office">{cbd_data.get("search_engine_used", "Web Search")}</span>',
                unsafe_allow_html=True,
            )
        with col_cbd_engine:
            st.caption(f"Market: **{cbd_data.get('city', 'Metropolitan')}**")

        st.markdown("")

        # Agent Reasoning Trace Expander
        agent_trace = cbd_data.get("agent_thought_trace", [])
        if agent_trace:
            with st.expander("🧠 LangGraph Deep Agent Reasoning & Graph Traversal Trace", expanded=True):
                for idx, t in enumerate(agent_trace, 1):
                    st.markdown(f"**Step {idx}: `{t.get('step', 'Step')}`**")
                    st.caption(f"↳ {t.get('thought', '')}")

        # Executive Summary Box
        st.markdown("#### 📑 Executive Valuation & Market Connectivity Report")
        st.info(cbd_data.get("executive_summary", ""), icon="💡")

        cbds = cbd_data.get("cbds", [])
        if cbds:
            st.markdown("#### 🏢 Identified Business Districts & Commercial Nodes")
            cbd_cols = st.columns(min(len(cbds), 3))
            for i, c in enumerate(cbds[:3]):
                with cbd_cols[i]:
                    c_name = html.escape(str(c.get("name", "CBD")))
                    c_type = html.escape(str(c.get("type", "Commercial")))
                    c_desc = html.escape(str(c.get("description", "")))
                    c_map = html.escape(str(c.get("google_maps_url", "#")))
                    cbd_card_html = (
                        f'<div class="prop-card">'
                        f'<h4 style="margin: 0; color: #fbbf24;">{c_name}</h4>'
                        f'<p style="margin: 4px 0; color: #f8fafc; font-size: 0.85rem; font-weight: 600;">{c_type}</p>'
                        f'<p style="margin: 4px 0; color: #38bdf8; font-size: 1.1rem; font-weight: 700;">🚗 {c["distance_km"]} km</p>'
                        f'<p style="margin: 4px 0; color: #cbd5e1; font-size: 0.8rem;">{c_desc}</p>'
                        f'<a href="{c_map}" target="_blank" style="color: #60a5fa; font-size: 0.8rem; text-decoration: none;">Get Driving Directions ↗</a>'
                        f'</div>'
                    )
                    st.markdown(cbd_card_html, unsafe_allow_html=True)

            st.markdown("#### 📊 CBD Distance & Specification Matrix")
            df_cbd = cbd.to_dataframe(cbds)
            st.dataframe(
                df_cbd[["CBD Hub", "Classification", "Distance (km)", "Description", "Directions"]],
                width="stretch",
                column_config={
                    "Directions": st.column_config.LinkColumn("Navigation", display_text="Get Route ↗"),
                    "Distance (km)": st.column_config.NumberColumn(format="%.2f km"),
                },
            )

        # Web Citations
        sources = cbd_data.get("web_sources", [])
        if sources:
            with st.expander(f"🌐 Web Citations & Evidence ({len(sources)} sources from {cbd_data.get('search_engine_used')}):", expanded=False):
                for idx, src in enumerate(sources, 1):
                    st.markdown(f"**[{idx}] [{src['title']}]({src['url']})**")
                    st.caption(src.get("snippet", ""))

    # ================= TAB 3: NEIGHBORHOOD AMENITIES =================
    with tab_amenities:
        st.subheader("🎒 Neighborhood Amenities & Social Infrastructure")
        st.caption(f"Amenities identified within **{amenity_data['radius_km']} km** of the subject site")

        cat_summary = amenity_data.get("category_summary", {})
        amenities = amenity_data.get("amenities", [])

        # Category Metric Grid
        if cat_summary:
            st.markdown("#### 📊 Amenity Density by Category")
            cols = st.columns(4)
            for idx, (cat_key, cat_info) in enumerate(cat_summary.items()):
                with cols[idx % 4]:
                    nearest_text = f"{cat_info['nearest_distance_km']} km" if cat_info['nearest_distance_km'] is not None else "N/A"
                    c_icon = cat_info.get("icon", "📍")
                    c_label = html.escape(str(cat_info.get("label", "")))
                    c_nearest = html.escape(str(cat_info.get("nearest_name", "")))
                    amenity_card_html = (
                        f'<div class="metric-card">'
                        f'<div class="metric-title">{c_icon} {c_label}</div>'
                        f'<div class="metric-value">{cat_info["count"]} <span style="font-size: 0.8rem; color: #94a3b8; font-weight: normal;">places</span></div>'
                        f'<div class="metric-sub">Nearest: <b>{nearest_text}</b></div>'
                        f'<div class="metric-sub" style="color: #64748b; font-size: 0.72rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{c_nearest}</div>'
                        f'</div>'
                    )
                    st.markdown(amenity_card_html, unsafe_allow_html=True)

        # Category filter for the table
        filter_cat = st.multiselect(
            "Filter Amenities by Category:",
            options=list(neighborehood_amenity.AMENITY_CATEGORIES.keys()),
            format_func=lambda x: f"{neighborehood_amenity.AMENITY_CATEGORIES[x]['icon']} {neighborehood_amenity.AMENITY_CATEGORIES[x]['label']}",
            default=list(neighborehood_amenity.AMENITY_CATEGORIES.keys()),
        )

        filtered_amenities = [a for a in amenities if a.get("category") in filter_cat]

        st.markdown(f"#### 📋 Amenities Directory ({len(filtered_amenities)} places)")
        df_amenities = neighborehood_amenity.to_dataframe(filtered_amenities)
        if not df_amenities.empty:
            st.dataframe(
                df_amenities[["Category", "Name", "Distance (km)", "Rating", "Reviews", "Address", "Google Maps"]],
                width="stretch",
                column_config={
                    "Google Maps": st.column_config.LinkColumn("Map", display_text="View ↗"),
                    "Distance (km)": st.column_config.NumberColumn(format="%.2f km"),
                    "Rating": st.column_config.NumberColumn(format="⭐ %.1f"),
                },
            )
            # Download CSV
            csv_amenity = df_amenities.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Amenities CSV",
                data=csv_amenity,
                file_name="neighborhood_amenities.csv",
                mime="text/csv",
            )
        else:
            st.info("No amenities found for selected categories.")

    # ================= TAB 4: INTERACTIVE MAP =================
    with tab_map:
        st.subheader("🗺️ Unified Real Estate Map")
        st.caption("Subject Site (Red Pin), Comparable Projects (Purple), CBDs (Orange), and Amenities (Green)")

        # Prepare map data
        map_records = []

        # 1. Subject Site
        map_records.append({
            "lat": site["lat"],
            "lon": site["lng"],
            "name": f"★ SUBJECT SITE: {site['name']}",
            "type": "Subject Site",
            "color": "#ef4444",
        })

        # 2. Comparables
        for c in comps:
            if c.get("lat") and c.get("lng"):
                map_records.append({
                    "lat": c["lat"],
                    "lon": c["lng"],
                    "name": f"🏢 {c['name']} ({c['distance_km']} km)",
                    "type": "Comparable Project",
                    "color": "#8b5cf6",
                })

        # 3. CBDs
        for b in cbds:
            if b.get("lat") and b.get("lng"):
                map_records.append({
                    "lat": b["lat"],
                    "lon": b["lng"],
                    "name": f"🌆 {b['name']} ({b['distance_km']} km)",
                    "type": "CBD / Hub",
                    "color": "#f59e0b",
                })

        # 4. Amenities
        for a in amenities:
            if a.get("lat") and a.get("lng"):
                map_records.append({
                    "lat": a["lat"],
                    "lon": a["lng"],
                    "name": f"{a.get('icon', '📍')} {a['name']} ({a['distance_km']} km)",
                    "type": "Amenity",
                    "color": "#10b981",
                })

        df_map = pd.DataFrame(map_records)

        # Render Folium map for rich interactive markers and radius buffer circle
        folium_rendered = False
        try:
            import folium
            from streamlit_folium import st_folium

            m = folium.Map(
                location=[site["lat"], site["lng"]],
                zoom_start=13 if data["radius_km"] <= 5 else 11,
                tiles="OpenStreetMap",
            )

            # Radius circle around subject site
            folium.Circle(
                radius=data["radius_km"] * 1000,
                location=[site["lat"], site["lng"]],
                color="#38bdf8",
                fill=True,
                fill_opacity=0.08,
                weight=2,
                tooltip=f"Search Buffer: {data['radius_km']} km",
            ).add_to(m)

            # Subject site marker
            folium.Marker(
                [site["lat"], site["lng"]],
                popup=f"<b>SUBJECT SITE: {site['name']}</b><br>{site['address']}",
                tooltip="Subject Site",
                icon=folium.Icon(color="red", icon="star"),
            ).add_to(m)

            # Comparable markers
            for c in comps:
                if c.get("lat") and c.get("lng"):
                    folium.CircleMarker(
                        location=[c["lat"], c["lng"]],
                        radius=7,
                        color="#a855f7",
                        fill=True,
                        fill_color="#a855f7",
                        fill_opacity=0.85,
                        tooltip=f"Comparable: {c['name']} ({c['distance_km']} km)",
                    ).add_to(m)

            # CBD markers
            for b in cbds:
                if b.get("lat") and b.get("lng"):
                    folium.Marker(
                        location=[b["lat"], b["lng"]],
                        tooltip=f"CBD: {b['name']} ({b['distance_km']} km)",
                        icon=folium.Icon(color="orange", icon="briefcase"),
                    ).add_to(m)

            # Amenities markers (sample top 25 to keep map clean)
            for a in amenities[:25]:
                if a.get("lat") and a.get("lng"):
                    folium.CircleMarker(
                        location=[a["lat"], a["lng"]],
                        radius=5,
                        color="#10b981",
                        fill=True,
                        fill_color="#10b981",
                        fill_opacity=0.7,
                        tooltip=f"{a.get('icon', '📍')} {a['name']} ({a['distance_km']} km)",
                    ).add_to(m)

            st_folium(m, width="100%", height=550)
            folium_rendered = True
        except Exception:
            folium_rendered = False

        if not folium_rendered:
            st.map(df_map, latitude="lat", longitude="lon", size=18, color="color", width="stretch")

        st.caption(
            "🔴 **Red Pin**: Subject Site | 🔵 **Blue Outline**: Search Radius | 🟣 **Purple**: Comparables | 🟠 **Orange**: CBDs | 🟢 **Green**: Amenities"
        )
