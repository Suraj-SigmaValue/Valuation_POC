"""
Builds an executive, publication-grade PDF documentation for the
Comparable Projects Valuation Engine using ReportLab.
"""
import os
import re
import html
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for professional header/footer with total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        # Suppress headers on cover page
        if self._pageNumber > 1:
            # Header
            self.setFont("Helvetica-Bold", 7.5)
            self.setFillColor(colors.HexColor("#1E3A8A"))
            self.drawString(42, 804, "COMPARABLE PROJECTS VALUATION ENGINE")
            self.setFont("Helvetica", 7.5)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawRightString(553, 804, "Technical Specification & Column Reference Guide")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.75)
            self.line(42, 796, 553, 796)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.75)
        self.line(42, 44, 553, 44)
        
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(42, 32, "Confidential — Real Estate Valuation POC Underwriting Engine")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(553, 32, page_str)
        self.restoreState()


def clean_xml(text: str) -> str:
    """Escapes XML entities while preserving simple formatting tags."""
    if not text:
        return ""
    # temporarily protect our allowed tags: <b>, </b>, <i>, </i>, <font ...>, </font>
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    # restore allowed tags
    allowed = ["b", "/b", "i", "/i", "code", "/code"]
    for tag in allowed:
        text = text.replace(f"&lt;{tag}&gt;", f"<{tag}>")
    # restore font color tags
    text = re.sub(r'&lt;font\s+color=([\"\'\#A-Za-z0-9]+)&gt;', r'<font color=\1>', text)
    text = text.replace("&lt;/font&gt;", "</font>")
    return text


def build_pdf(filename: str):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=54,
        bottomMargin=54,
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0F172A")    # Slate 900
    c_brand = colors.HexColor("#1E3A8A")      # Blue 900
    c_accent = colors.HexColor("#2563EB")     # Blue 600
    c_body = colors.HexColor("#334155")       # Slate 700
    c_muted = colors.HexColor("#64748B")      # Slate 500
    c_card_bg = colors.HexColor("#F8FAFC")    # Slate 50
    c_border = colors.HexColor("#E2E8F0")     # Slate 200
    c_green = colors.HexColor("#059669")      # Emerald 600

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=c_brand,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=c_body,
        spaceAfter=14,
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=12,
        textColor=c_muted,
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=c_brand,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
    )
    h3_style = ParagraphStyle(
        'SectionH3',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=c_accent,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_body,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_body,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3,
    )
    formula_style = ParagraphStyle(
        'FormulaBox',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=8,
        leading=11,
        textColor=c_brand,
        backColor=c_card_bg,
        borderColor=c_border,
        borderWidth=0.5,
        borderPadding=5,
        spaceBefore=4,
        spaceAfter=6,
    )
    card_header_style = ParagraphStyle(
        'CardHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=c_brand,
    )
    card_cell_style = ParagraphStyle(
        'CardCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=c_body,
    )
    card_cell_bold = ParagraphStyle(
        'CardCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=c_primary,
    )

    story = []

    # ---------------------------------------------------------
    # COVER / HEADER BLOCK
    # ---------------------------------------------------------
    story.append(Paragraph("COMPARABLE PROJECTS VALUATION ENGINE", title_style))
    story.append(Paragraph("<b>Technical Specification, Column Logic & Mathematical Reference Guide</b>", subtitle_style))
    
    meta_text = (
        "<b>Platform:</b> Real Estate Valuation POC &nbsp;|&nbsp; "
        "<b>Architecture:</b> Elastic Valuation Engine v2.4 &nbsp;|&nbsp; "
        "<b>Date:</b> October 2026<br/>"
        "<b>Scope:</b> Complete Logic, Verification Rules, Formulas & Edge-Cases for all 25 Table and CSV Columns"
    )
    meta_table = Table(
        [[Paragraph(meta_text, meta_style)]],
        colWidths=[511],
    )
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_card_bg),
        ('BOX', (0, 0), (-1, -1), 0.75, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Executive Note
    exec_note = (
        "<b>Executive Summary & Underwriting Standard:</b> The Comparable Projects Valuation Engine "
        "evaluates candidate residential projects through evidence-grounded spatial discovery, live portal web intelligence, "
        "and multi-dimensional elastic similarity scoring. It enforces a <b>Strict Zero-Imputation Policy</b> (missing public rates "
        "or areas are never fabricated or guessed) and institutional qualification rules that eliminate unverified entities."
    )
    story.append(Paragraph(clean_xml(exec_note), body_style))
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------
    # SECTION 1: ARCHITECTURAL PIPELINE
    # ---------------------------------------------------------
    story.append(Paragraph("1. Architectural Pipeline & Discovery Stages", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_brand, spaceBefore=1, spaceAfter=6))
    
    pipeline_steps = [
        ("Stage 1: Input & Concurrent Geocoding", "Resolves user input (Project Name or Lat/Lng coordinates) via Google Places API in < 0.1s. Dispatches 3-way concurrent execution across Comparables, CBD Deep Agent, and Amenities."),
        ("Stage 2: Automated Baseline Calibration", "Dispatches targeted intelligence search for the subject property to extract benchmark rate, carpet area, unit configuration, developer tier, and building age/possession date."),
        ("Stage 3: Spatial Discovery & Strict Asset Filtering", "Scans 2.0 to 10.0 km radius. Employs asset bifurcation (Flat / Shop / Office / Others) and commercial tenant filtering to isolate genuine assets from corporate service companies."),
        ("Stage 4: Multi-Tier Search & Persistent Circuit Breaker", "Executes high-concurrency search (8 threads) with persistent quota circuit breakers (.quota_status.json), automatically routing around exhausted keys with zero retry latency to DuckDuckGo."),
        ("Stage 5: Dual-Pass Project DNA & Age Extraction", "Parses rates (including Commercial Income Capitalization @ 8% cap rate / 150x rent multiplier), carpet areas, unit typologies, and dedicated Building Age & Possession Year timelines."),
        ("Stage 6: Elastic Valuation & Market Banding", "Calculates continuous Gaussian and exponential scores across 7 physical and financial dimensions with dynamic elastic re-weighting when price is unlisted.")
    ]
    
    pipe_table_data = []
    for step_title, step_desc in pipeline_steps:
        pipe_table_data.append([
            Paragraph(f"<b>{step_title}</b>", card_cell_bold),
            Paragraph(step_desc, card_cell_style)
        ])
    t_pipe = Table(pipe_table_data, colWidths=[160, 351])
    t_pipe.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_card_bg),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_pipe)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SECTION 2: BASELINE CALIBRATION & SOURCE PRICE DISCOVERY
    # ---------------------------------------------------------
    story.append(Paragraph("2. Subject Property Baseline Calibration", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_brand, spaceBefore=1, spaceAfter=6))
    
    story.append(Paragraph("2.1 How the Source Benchmark Rate is Derived (4-Tier Hierarchy)", h2_style))
    story.append(Paragraph(
        "The subject property rate (<b>Source Rate</b>, <i>R_src</i>) anchors the entire valuation comparison. "
        "The engine establishes this benchmark through an automated 4-tier decision cascade:",
        body_style
    ))

    tiers = [
        ("Priority 1: User Manual Override", "If the appraiser supplies an explicit benchmark rate in the sidebar under 'Optional Manual Overrides', this rate takes absolute authority, overriding all automated web scraping. Indicated by '(Custom Override)'."),
        ("Priority 2: Direct Disclosed Web Rate", "Extracted directly from top real estate portals (MagicBricks, 99acres, Housing.com, SquareYards). If a range is quoted (e.g. ₹7,500 to ₹9,200/sq.ft), the engine computes the exact midpoint: R_src = (Min + Max) / 2."),
        ("Priority 3: Ticket Size ÷ Carpet Area", "When portals report total apartment ticket prices (e.g., ₹75.61 L) and carpet areas (e.g., 540 sq.ft) but no per-sqft rate, the engine computes: R_src = round(Total INR Ticket / Carpet Area, 0)."),
        ("Priority 4: Locality Cluster Median Fallback", "If the subject site is unlisted or newly launched with no public web DNA, the engine queries all verified nearby residential projects discovered in the radius and applies their statistical cluster median: R_src = Median({R_comp}). Indicated by '(Locality Cluster Median)'.")
    ]
    t_tier_data = []
    for t_name, t_desc in tiers:
        t_tier_data.append([
            Paragraph(f"<b>{t_name}</b>", card_cell_bold),
            Paragraph(t_desc, card_cell_style)
        ])
    t_tier = Table(t_tier_data, colWidths=[150, 361])
    t_tier.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_tier)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2.2 How Source Price is Resolved When Coordinates (Lat / Long) are Input", h2_style))
    coord_workflow = (
        "When the user inputs GPS coordinates directly (e.g., Latitude 19.066000, Longitude 72.860000):<br/>"
        "1. <b>Reverse-Geocoding:</b> The coordinates are passed to Google Maps Geocoding API to resolve the formatted street address, sublocality, and building name.<br/>"
        "2. <b>Building Extraction:</b> If the coordinate pin falls on an established apartment society (e.g., <i>Kalpataru Sparkle</i>), that name is extracted as the subject entity.<br/>"
        "3. <b>Targeted Search:</b> Portal queries are dispatched for that specific building to retrieve live rates.<br/>"
        "4. <b>Unlisted / Vacant Site Handling:</b> If the coordinate represents an open plot or unlisted land parcel, the engine discovers verified residential societies within 2.0 km and computes the <b>Locality Cluster Median</b> as the baseline rate."
    )
    story.append(Paragraph(clean_xml(coord_workflow), body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("2.3 Allowable Price and Carpet Area Tolerance Corridors", h2_style))
    corridor_text = (
        "Once R_src and A_src are established, the engine sets strict institutional boundaries:<br/>"
        "• <b>Allowable Price Range (80%–120%):</b> Min Rate = R_src × 0.80 &nbsp;|&nbsp; Max Rate = R_src × 1.20<br/>"
        "&nbsp;&nbsp;<i>Example: For R_src = ₹8,333/sq.ft ➔ Range = [₹6,666, ₹10,000 / sq.ft]</i><br/>"
        "• <b>Allowable Carpet Area Range (±15%):</b> Min Area = A_src × 0.85 &nbsp;|&nbsp; Max Area = A_src × 1.15<br/>"
        "&nbsp;&nbsp;<i>Example: For A_src = 1,000 sq.ft ➔ Range = [850, 1,150 sq.ft]</i>"
    )
    story.append(Paragraph(clean_xml(corridor_text), formula_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2.4 Target Property Configuration & Typology Filtering", h2_style))
    story.append(Paragraph(
        "Users can explicitly choose the target configuration: <b>Flat (1 BHK, 2 BHK, 3 BHK, 4 BHK, > 4 BHK / Penthouse)</b>, "
        "<b>Commercial Shop</b>, <b>Commercial Office</b>, or <b>Other / Mixed-Use</b>. When selected, the discovery engine "
        "enforces configuration-specific targeting:",
        body_style
    ))
    config_table_data = [
        [Paragraph("<b>Target Configuration</b>", card_cell_bold),
         Paragraph("<b>Property Type & Search Target</b>", card_cell_bold),
         Paragraph("<b>Strict Matching Evidence Rule</b>", card_cell_bold)],
        [Paragraph("Flat - 1 BHK", card_cell_style), Paragraph("Residential Flat", card_cell_style), Paragraph("Matches 1 BHK, 1 Bed, Studio", card_cell_style)],
        [Paragraph("Flat - 2 BHK", card_cell_style), Paragraph("Residential Flat", card_cell_style), Paragraph("Matches 2 BHK, 2 Bed", card_cell_style)],
        [Paragraph("Flat - 3 BHK", card_cell_style), Paragraph("Residential Flat", card_cell_style), Paragraph("Matches 3 BHK, 3 Bed", card_cell_style)],
        [Paragraph("Flat - 4 BHK", card_cell_style), Paragraph("Residential Flat", card_cell_style), Paragraph("Matches 4 BHK, 4 Bed", card_cell_style)],
        [Paragraph("Flat - > 4 BHK", card_cell_style), Paragraph("Residential Luxury", card_cell_style), Paragraph("Matches 5 BHK, 6 BHK, Penthouse, Duplex, Villa", card_cell_style)],
        [Paragraph("Shop (Commercial Retail)", card_cell_style), Paragraph("Commercial Retail", card_cell_style), Paragraph("Shopping malls, retail complexes, showrooms", card_cell_style)],
        [Paragraph("Office (Commercial Corporate)", card_cell_style), Paragraph("Commercial Office", card_cell_style), Paragraph("IT tech parks, business centers, corporate offices", card_cell_style)],
        [Paragraph("Auto-Detect", card_cell_style), Paragraph("Dynamic from Web DNA", card_cell_style), Paragraph("Discovered automatically from subject project", card_cell_style)],
    ]
    t_cfg = Table(config_table_data, colWidths=[150, 150, 211])
    t_cfg.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_card_bg),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_cfg)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SECTION 3: COMPLETE 25 COLUMN-BY-COLUMN REFERENCE
    # ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("3. Comprehensive Column-by-Column Specification (All 25 Fields)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_brand, spaceBefore=1, spaceAfter=8))
    story.append(Paragraph(
        "The directory table presents 22 core underwriting columns, and the downloadable CSV contains 25 comprehensive audit fields. "
        "Below is the exact data source, logic, formula, and display format for every single column:",
        body_style
    ))
    story.append(Spacer(1, 6))

    columns_def = [
        {
            "num": 1,
            "name": "Name",
            "source": "Google Places API (Nearby Search & Text Search) via clean_project_name()",
            "logic": "Extracts entity name, strips commercial noise, marketing taglines ('Luxurious Apartments', 'Showroom for Sale'), city suffixes (', Pune', '- Mumbai'), and consolidates sub-units to parent society.",
            "formula": "Regex sanitization: re.sub(r'\\s*[-|–]\\s*(?:Best|Premier|Luxurious|Top|Affordable|Offices?).*$', '', name)",
            "display": "Clean string (e.g., 'Gokulam by Sarvome', 'Vasant Valley')"
        },
        {
            "num": 2,
            "name": "Distance (km)",
            "source": "Spherical coordinates calculation (Haversine Formula)",
            "logic": "Computes great-circle distance between Subject Site coordinates (lat1, lng1) and Comparable Site coordinates (lat2, lng2).",
            "formula": "d = 2R · arcsin(sqrt(sin²(Δφ/2) + cos(φ1)·cos(φ2)·sin²(Δλ/2))), with R = 6,371.0 km",
            "display": "Two-decimal float with unit (e.g., '0.79 km')"
        },
        {
            "num": 3,
            "name": "Verified Rate (₹/sqft)",
            "source": "Multi-tier live web search parsed via extract_project_dna_dynamic()",
            "logic": "Strict Zero-Imputation Policy: Never fabricated or guessed. Evaluates prefix rates ('Price per sq. ft.: ₹3,400 to ₹4,100'), standard per-sqft strings, or ticket price ÷ carpet area. Bounds: [₹2,500, ₹150,000].",
            "formula": "Direct rate, or (Min + Max)/2 if range, or round(Total INR Ticket / Carpet Area, 0)",
            "display": "'₹ 4,039' or 'Rate Unlisted'"
        },
        {
            "num": 4,
            "name": "In 80%–120% Range?",
            "source": "Comparison between Comparable Verified Rate (R_comp) and Subject Benchmark (R_src)",
            "logic": "Computes percentage difference ΔP = (R_comp - R_src) / R_src. Categorizes into In-Range, Premium, Value, or Physical Match.",
            "formula": "If 0.80·R_src <= R_comp <= 1.20·R_src ➔ In Range; If > 1.20 ➔ Premium (+ΔP%); If < 0.80 ➔ Value (-ΔP%)",
            "display": "'✅ In 80%–120% Range', '💎 Premium (+28.4%)', '🏷️ Value (-22.1%)', or '🏢 Physical Match (Rate Unlisted)'"
        },
        {
            "num": 5,
            "name": "Price Evidence",
            "source": "Unaltered verbatim excerpt from the source real estate portal snippet",
            "logic": "Extracts 120-character contextual snippet surrounding the discovered price token for full auditability.",
            "formula": "Regex contextual window extraction from scraped portal text",
            "display": "Verbatim text: '...Price starts from ₹75.61 L... (Total ₹75.61 L / 1872 sq.ft)' or '—'"
        },
        {
            "num": 6,
            "name": "Carpet Area (sqft)",
            "source": "Portal floor plans, brochure specifications, and RERA registration filings",
            "logic": "Matches target unit configuration area (e.g. '2 BHK 534 to 544 sq.ft'), keyword 'carpet area', or floor plan area tags. Validated within [350, 3,500 sq.ft].",
            "formula": "Area value in sq.ft, or midpoint (A1 + A2)/2 if range",
            "display": "Formatted integer (e.g., '1,872 sq.ft') or '—' if unlisted"
        },
        {
            "num": 7,
            "name": "Area Match",
            "source": "Mathematical divergence comparison against Subject Carpet Area (A_src)",
            "logic": "Computes area divergence ratio δA = |A_comp - A_src| / A_src against standard 15% tolerance threshold.",
            "formula": "If δA <= 0.15 ➔ '✅ In Range ({A_comp} sq.ft)'; Else ➔ 'Area ({A_comp} sq.ft)'; If unlisted ➔ 'Area Unlisted'",
            "display": "'✅ In Range (950 sq.ft)', 'Area (1,450 sq.ft)', or 'Area Unlisted'"
        },
        {
            "num": 8,
            "name": "Carpet Area Evidence",
            "source": "Verbatim source portal snippet declaring carpet area specifications",
            "logic": "Captures the exact text context where unit dimensions or carpet area were published.",
            "formula": "Contextual regex window from portal metadata",
            "display": "Verbatim text: '...Carpet Area: 534 to 544 sq. ft...' or '—'"
        },
        {
            "num": 9,
            "name": "Unit Typology",
            "source": "Web snippet parser scanning layout configurations",
            "logic": "Scans for 1 BHK, 2 BHK, 3 BHK, 4 BHK, 5 BHK, 6 BHK, Studio, Penthouse, Duplex, Villa, Shop, Office configurations.",
            "formula": "Multi-typology token set extraction",
            "display": "Comma-separated list (e.g., '2 BHK, 3 BHK') or '—'"
        },
        {
            "num": 10,
            "name": "Unit Match",
            "source": "is_unit_matching() comparing candidate available units against target configuration",
            "logic": "Tests whether candidate explicitly discloses the target unit configuration. Mismatches disqualify the candidate and assign zero unit score.",
            "formula": "If target unit matched ➔ '✅ Matches ({unit})'; If candidate has other units ➔ '⚠️ Mismatch ({units})'; If unlisted ➔ 'Typology Unlisted'",
            "display": "'✅ Matches (2 BHK)', '⚠️ Mismatch (3 BHK, 4 BHK)', or 'Typology Unlisted'"
        },
        {
            "num": 11,
            "name": "Unit Evidence",
            "source": "Contextual portal text declaring unit configurations",
            "logic": "Provides snippet proof for available room configurations in the project.",
            "formula": "Contextual regex capture",
            "display": "Verbatim snippet: '...Configurations: 2 BHK, 3 BHK Apartments...' or '—'"
        },
        {
            "num": 12,
            "name": "Developer",
            "source": "Entity title parsing and national/regional developer ontology",
            "logic": "Scans parenthetical/prefix titles ('Project by Developer'), known brand dictionary, and structured markdown tags ('Developer: XYZ').",
            "formula": "Ontology lookup + regex matching: r'\\b(?:developer|builder|promoter)\\s*[:\\*]*\\s*([A-Za-z0-9\\s&,\\.]{3,50}?)'",
            "display": "Clean builder title (e.g., 'HRH City Projects Private Limited', 'Godrej Properties') or '—'"
        },
        {
            "num": 13,
            "name": "Developer Tier",
            "source": "Institutional builder classification ontology",
            "logic": "Classifies developer scale into Tier 1 (National Listed/Institutional), Tier 2 (Reputed Regional Multi-Project), or Tier 3 (Local/Standalone Society).",
            "formula": "Lookup across KNOWN_BRAND_MAP registry",
            "display": "'Tier 1 (National Listed)', 'Tier 2 (Reputed Regional)', 'Tier 3 (Local / Standalone)', or '—'"
        },
        {
            "num": 14,
            "name": "Developer Evidence",
            "source": "Source declaration confirming builder identity",
            "logic": "Indicates whether developer was extracted from title, ontology registry, or portal snippet.",
            "formula": "Citation path string",
            "display": "'Extracted from project name: by Sarvome' or verbatim snippet"
        },
        {
            "num": 15,
            "name": "Construction Status",
            "source": "Portal possession timelines and RERA status declarations",
            "logic": "Distinguishes Ready to Move (RTM) from Under Construction (UC) with direct evidence extraction.",
            "formula": "If UC / Ongoing / Launching ➔ 'Under Construction'; If RTM / Completed / Handover ➔ 'Ready to Move'",
            "display": "'Ready to Move', 'Under Construction', or 'Status Unlisted'"
        },
        {
            "num": 16,
            "name": "Status Evidence",
            "source": "Source snippet quoting possession, delivery, or construction milestone",
            "logic": "Verbatim proof of project construction progress and operational delivery status.",
            "formula": "Contextual regex extraction",
            "display": "Verbatim snippet: '...Ready to Move (Completed: 2023)...' or '—'"
        },
        {
            "num": 17,
            "name": "Building Age",
            "source": "Completion date, year built, and property age declarations",
            "logic": "Calculates building age from completion/possession year (Age = Current Year - Year Built). For under-construction assets, displays scheduled delivery timeline.",
            "formula": "If UC ➔ 'Under Construction (Possession: YYYY)'; If RTM ➔ '{Current Year - Poss Year} Years'; If recent ➔ '< 1 Year (Recent Possession)'",
            "display": "'3 Years', '5 Years', 'Under Construction (Possession: 2027)', '< 1 Year', or 'Ready to Move'"
        },
        {
            "num": 18,
            "name": "Age Evidence",
            "source": "Contextual web snippet quoting property age, year built, or completion year",
            "logic": "Captures exact snippet proof for property age and delivery year from real estate portals.",
            "formula": "Contextual regex window",
            "display": "Verbatim snippet: '...Completed in 2023 (Age: 3 yrs)...' or '—'"
        },
        {
            "num": 19,
            "name": "Status & Age (Composite)",
            "source": "Composite status and chronological age synthesis",
            "logic": "Combines physical status with age duration into a single institutional underwriting tag.",
            "formula": "f'{status_str} ({age_str})'",
            "display": "'Ready to Move (3 Years)', 'Under Construction (Possession: 2027)', or 'Ready to Move'"
        },
        {
            "num": 20,
            "name": "Scale & Typology",
            "source": "Project land area (acreage), tower count, and unit count declarations",
            "logic": "Classifies development into Township (>=20 acres or >=1000 units), Gated Community (3-20 acres or 4+ towers), Co-op CHS (40-200 units), or Standalone Building.",
            "formula": "Hierarchical boundary check on acreage, tower, and unit counts",
            "display": "'Gated Community (6.2 Acres)', 'Township (150 Acres)', 'Standalone Building', or '—'"
        },
        {
            "num": 21,
            "name": "Scale Evidence",
            "source": "Source portal snippet referencing land acreage, towers, or total units",
            "logic": "Captures exact text declaration for community physical scale.",
            "formula": "Contextual regex window",
            "display": "Verbatim snippet: '...Project spans 6.25 acres and comprises 8 towers...' or '—'"
        },
        {
            "num": 22,
            "name": "Evaluation Basis",
            "source": "Elastic Valuation Engine mode selector",
            "logic": "Indicates whether comparison was based on Full Valuation (Price + Physical specs), Physical Equivalence (Rate Unlisted), or Unverified.",
            "formula": "If rate listed ➔ Full Valuation; Else if physical specs captured ➔ Physical Equivalence; Else ➔ Unverified",
            "display": "'Full Valuation Match (Price + Physical)', 'Physical / Structural Equivalence', or 'Unverified'"
        },
        {
            "num": 23,
            "name": "Valuation Match",
            "source": "Continuous composite similarity score from Elastic Valuation Engine",
            "logic": "Combines Gaussian price proximity, exponential distance decay, area tolerance, unit match, developer tier, age, and scale. Missing dimensions score 0.0.",
            "formula": "S_total = Σ (w_i · S_i). Normal mode: 25% price + 15% dist + 15% area + 15% unit + 10% dev + 10% age + 10% scale.",
            "display": "Percentage string (e.g., '83.2%', '68.5%', '18.7%')"
        },
        {
            "num": 24,
            "name": "Status",
            "source": "Strict Institutional Qualification Rules",
            "logic": "Governs whether candidate is admitted into the valuation basket. Requires verified in-range rate OR verified in-range carpet area, and matching unit configuration.",
            "formula": "If verified rate in 80%–120% and score >= 45% ➔ Qualified; If rate unlisted but area in ±15% and score >= 40% ➔ Qualified; If unit mismatch ➔ ⚠️ Unit Typology Mismatch.",
            "display": "'✅ Qualified Match', '⚠️ Price Out of Range', '⚠️ Area Out of Range', '⚠️ Unit Typology Mismatch', or '❌ Insufficient Data'"
        },
        {
            "num": 25,
            "name": "Evidence Citation",
            "source": "Direct HTTP URL of the verified listing on MagicBricks, 99acres, Housing.com, or official developer portal",
            "logic": "Provides direct clickable link to source evidence for human underwriter auditability.",
            "formula": "Target URL string",
            "display": "Hyperlink: 'View Evidence ↗'"
        },
        {
            "num": 26,
            "name": "Google Maps",
            "source": "Google Place ID link generator",
            "logic": "Deep-link to inspect location, satellite view, street view, and neighbourhood surroundings.",
            "formula": "https://www.google.com/maps/place/?q=place_id:{place_id}",
            "display": "Hyperlink: 'Open Map ↗'"
        },
        {
            "num": 27,
            "name": "Rating (Exported CSV)",
            "source": "Google Places API (rating attribute)",
            "logic": "The aggregate public star rating (scale 1.0 to 5.0) submitted by verified Google Maps contributors for the society.",
            "formula": "Direct API float",
            "display": "Float (e.g., '4.2') or '0.0' if unrated"
        },
        {
            "num": 28,
            "name": "Reviews (Exported CSV)",
            "source": "Google Places API (user_ratings_total attribute)",
            "logic": "Total count of public reviews submitted on Google Maps. Used in underwriter audit trails to gauge society maturity and occupancy.",
            "formula": "Direct API integer count",
            "display": "Integer count (e.g., '148') or '0'"
        },
        {
            "num": 29,
            "name": "Address (Exported CSV)",
            "source": "Google Places API (vicinity or formatted_address)",
            "logic": "Physical street address or localized municipal address string for the comparable candidate.",
            "formula": "Cleaned address string",
            "display": "Address string (e.g., 'Sector 56A, Faridabad, Haryana 121004')"
        }
    ]

    for col in columns_def:
        col_box = [
            [
                Paragraph(f"<b>Column {col['num']}: <code>{col['name']}</code></b>", card_header_style),
                Paragraph(f"<b>Data Source:</b> {col['source']}", card_cell_style)
            ],
            [
                Paragraph("<b>Underwriting Logic & Rules:</b>", card_cell_bold),
                Paragraph(col['logic'], card_cell_style)
            ],
            [
                Paragraph("<b>Mathematical Formula / Rule:</b>", card_cell_bold),
                Paragraph(f"<code>{col['formula']}</code>", card_cell_style)
            ],
            [
                Paragraph("<b>Display Format / Output:</b>", card_cell_bold),
                Paragraph(f"<b>{col['display']}</b>", card_cell_style)
            ]
        ]
        t_col = Table(col_box, colWidths=[130, 381])
        t_col.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('BACKGROUND', (0, 0), (-1, 0), c_card_bg),
            ('BOX', (0, 0), (-1, -1), 0.75, c_border),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, c_border),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(KeepTogether([t_col, Spacer(1, 6)]))

    # ---------------------------------------------------------
    # SECTION 4: MATHEMATICAL SCORING FORMULAS
    # ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("4. The Elastic Valuation Engine & Mathematical Formulas", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_brand, spaceBefore=1, spaceAfter=8))
    story.append(Paragraph(
        "The engine calculates comparability using smooth continuous functions rather than arbitrary cutoffs. "
        "Missing attributes receive zero score (0.0) — never defaulted or artificial.",
        body_style
    ))
    story.append(Spacer(1, 4))

    formulas_list = [
        ("1. Price Proximity Score (S_price)",
         "Evaluated using a Gaussian exponential decay model centered at Subject Rate R_src:<br/>"
         "<b>p_diff = |R_comp - R_src| / R_src</b><br/>"
         "<b>S_price = 100.0 · exp( -0.5 · (p_diff / 0.20)² )</b><br/>"
         "• At ±0% diff ➔ 100.0 &nbsp;|&nbsp; At ±20% diff ➔ 60.65 &nbsp;|&nbsp; At ±40% diff ➔ 13.53"),

        ("2. Spatial Proximity Score (S_dist)",
         "Evaluated using exponential distance decay with characteristic scale length = 2.5 km:<br/>"
         "<b>S_dist = 100.0 · exp( -d / 2.5 )</b><br/>"
         "• At 0.0 km ➔ 100.0 &nbsp;|&nbsp; At 1.0 km ➔ 67.0 &nbsp;|&nbsp; At 2.5 km ➔ 36.8 &nbsp;|&nbsp; At 5.0 km ➔ 13.5"),

        ("3. Carpet Area Score (S_area)",
         "Evaluated using linear penalty over a 30% maximum divergence corridor:<br/>"
         "<b>a_diff = |A_comp - A_src| / max(A_src, 1.0)</b><br/>"
         "<b>S_area = max(0.0, 100.0 · (1.0 - a_diff / 0.30))</b> if both known;<br/>"
         "• If candidate area unlisted ➔ 0.0 (Strict Zero Imputation)"),

        ("4. Unit Typology Score (S_unit)",
         "• Exact match with target unit (e.g., '2 BHK') ➔ 100.0<br/>"
         "• Target unit missing, but other residential units present ➔ 30.0<br/>"
         "• Undisclosed / unlisted unit typology ➔ 0.0"),

        ("5. Developer Reputation Score (S_dev)",
         "Based on Tier divergence ΔT = |Tier_comp - Tier_src|:<br/>"
         "• Identical Tier (ΔT = 0) ➔ 100.0<br/>"
         "• 1-Tier Gap (ΔT = 1, e.g. Tier 1 vs Tier 2) ➔ 65.0<br/>"
         "• 2-Tier Gap (ΔT = 2, e.g. Tier 1 vs Tier 3) ➔ 30.0<br/>"
         "• Undisclosed candidate developer ➔ 0.0"),

        ("6. Status & Project Age Score (S_age)",
         "Base score: Status_comp == Status_src ➔ 100.0; Status differs (RTM vs UC) ➔ 60.0<br/>"
         "Age depreciation: <b>ΔAge = |Age_comp - Age_src|</b><br/>"
         "<b>S_age = max(20.0, S_status_base - (ΔAge · 4.0))</b>"),

        ("7. Scale & Typology Score (S_scale)",
         "• Identical scale classification ➔ 100.0<br/>"
         "• Multi-acre interchange (Gated Community ↔ Township) ➔ 80.0<br/>"
         "• Standalone Building ↔ Large Gated Complex ➔ 45.0<br/>"
         "• Undisclosed candidate scale ➔ 0.0")
    ]

    for f_title, f_body in formulas_list:
        f_box = [
            [Paragraph(f"<b>{f_title}</b>", card_cell_bold)],
            [Paragraph(clean_xml(f_body), card_cell_style)]
        ]
        t_f = Table(f_box, colWidths=[511])
        t_f.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), c_card_bg),
            ('BOX', (0, 0), (-1, -1), 0.5, c_border),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(KeepTogether([t_f, Spacer(1, 5)]))

    story.append(Spacer(1, 6))
    story.append(Paragraph("4.2 Dynamic Elastic Weight Redistribution", h2_style))
    story.append(Paragraph(
        "When a candidate's rate is unlisted on public portals, the engine dynamically suppresses the price weight and "
        "elastically redistributes 100% of the scoring mass across physical equivalence dimensions:",
        body_style
    ))

    weight_data = [
        [Paragraph("<b>Evaluation Dimension</b>", card_cell_bold),
         Paragraph("<b>Full Valuation Mode (Price Listed)</b>", card_cell_bold),
         Paragraph("<b>Physical Mode (Rate Unlisted)</b>", card_cell_bold)],
        [Paragraph("Price Proximity (S_price)", card_cell_style), Paragraph("25%", card_cell_style), Paragraph("0% (Suppressed)", card_cell_style)],
        [Paragraph("Spatial Proximity (S_dist)", card_cell_style), Paragraph("15%", card_cell_style), Paragraph("20%", card_cell_style)],
        [Paragraph("Carpet Area Proximity (S_area)", card_cell_style), Paragraph("15%", card_cell_style), Paragraph("20%", card_cell_style)],
        [Paragraph("Unit Typology Match (S_unit)", card_cell_style), Paragraph("15%", card_cell_style), Paragraph("20%", card_cell_style)],
        [Paragraph("Developer Tier Match (S_dev)", card_cell_style), Paragraph("10%", card_cell_style), Paragraph("15%", card_cell_style)],
        [Paragraph("Status & Age Match (S_age)", card_cell_style), Paragraph("10%", card_cell_style), Paragraph("15%", card_cell_style)],
        [Paragraph("Scale & Typology Match (S_scale)", card_cell_style), Paragraph("10%", card_cell_style), Paragraph("10%", card_cell_style)],
        [Paragraph("<b>Total Weight Mass</b>", card_cell_bold), Paragraph("<b>100%</b>", card_cell_bold), Paragraph("<b>100%</b>", card_cell_bold)],
    ]
    t_weights = Table(weight_data, colWidths=[211, 150, 150])
    t_weights.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_card_bg),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_weights)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # SECTION 5: INSTITUTIONAL QUALIFICATION & KPI METRICS
    # ---------------------------------------------------------
    story.append(Paragraph("5. Institutional Qualification & Summary KPI Metrics", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_brand, spaceBefore=1, spaceAfter=8))
    
    story.append(Paragraph("5.1 Institutional Qualification Thresholds (Eliminating Unverified Noise)", h2_style))
    qual_rules = (
        "Underwriting integrity requires that no project enters the qualified valuation basket without genuine evidence:<br/>"
        "• <b>✅ Qualified Match:</b> (Verified rate in [0.80·R_src, 1.20·R_src] AND S_total >= 45.0%) OR (Rate unlisted, verified area in ±15%, AND S_total >= 40.0%).<br/>"
        "• <b>⚠️ Price Out of Range:</b> Verified rate exists but exceeds or falls below the 80%–120% boundary.<br/>"
        "• <b>⚠️ Area Out of Range:</b> Verified carpet area exists but deviates beyond ±15% from subject baseline.<br/>"
        "• <b>⚠️ Incomplete Specs (No Rate/Area):</b> Both rate and area unlisted; only minor metadata captured.<br/>"
        "• <b>❌ Insufficient Data:</b> Zero verified attributes captured from public portals (raw proximity score 15%–20%)."
    )
    story.append(Paragraph(clean_xml(qual_rules), body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.2 Summary KPI Metrics & Outlier Statistics", h2_style))
    kpis = [
        ("Total Comparables Found", "N_total = Count of all spatial entities within radius passing non-residential exclusion."),
        ("Qualified (All 4 Criteria)", "N_qual = Count of projects meeting institutional qualification thresholds above."),
        ("Allowable Price Range", "Corridor = [0.80 · R_src, 1.20 · R_src] anchoring the appraisal boundary."),
        ("Locality Cluster Median", "Median = med({ R_comp | R_comp > 0 }), resilient to speculative portal spikes."),
        ("Interquartile Range (IQR)", "IQR = Q3 - Q1 (75th percentile minus 25th percentile rate), measuring market dispersion.")
    ]
    t_kpi_data = []
    for k_name, k_desc in kpis:
        t_kpi_data.append([
            Paragraph(f"<b>{k_name}</b>", card_cell_bold),
            Paragraph(k_desc, card_cell_style)
        ])
    t_kpi = Table(t_kpi_data, colWidths=[160, 351])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_card_bg),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_kpi)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF at: {filename}")


if __name__ == "__main__":
    out_pdf = os.path.abspath("Comparable_Projects_Valuation_Engine_Documentation.pdf")
    build_pdf(out_pdf)

