import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
import streamlit.components.v1 as components
import searoute as sr
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Sailing & Marine Weather Historian Pro", layout="wide")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 350px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Taal & Vertalingen ---
with st.sidebar:
    lang = st.selectbox("Taal / Language", ["🇳🇱 Nederlands", "🇬🇧 English"], index=0)

is_nl = "Nederlands" in lang

t = {
    "title": "⛵ Sailing Weather & Marine History Planner",
    "subtitle": "Geavanceerde tochtplanning met automatische maritieme water-routing, getijdenstroom (+/- Oost/West as), getijhoogte in decimeters (dm) en GPX-export." if is_nl else "Advanced passage planning with automated maritime water routing, tidal currents (+/- East/West axis), tide height in decimeters (dm), and GPX export.",
    "nav_header": "Navigatie" if is_nl else "Navigation",
    "m1": "Enkele Locatie & 12M Historie" if is_nl else "Single Location & 12M History",
    "m2": "Multi-Jaar Vergelijking (10 Jaar)" if is_nl else "Multi-Year Comparison (10 Years)",
    "m3": "Optimale Route & Stroomrouting" if is_nl else "Optimal Route & Current Routing",
    "m4": "14-Daagse Verwachting & Getijden" if is_nl else "14-Day Forecast & Tides",
    "m5": "Technische Bronnen & Disclaimer" if is_nl else "Technical Sources & Disclaimer",
    "m6": "☕ Donaties & Support" if is_nl else "☕ Donations & Support",
    "wind_unit": "Wind eenheid" if is_nl else "Wind Unit",
    "filters_sub": "Zeilcondities Filter" if is_nl else "Sailing Conditions Filter",
    "w_speed": "Windsnelheid (Knopen)" if is_nl else "Wind Speed (Knots)",
    "w_gust": "Max. Windstoot (Knopen)" if is_nl else "Max. Wind Gust (Knots)",
    "w_wave": "Max. Golfhoogte (Meters)" if is_nl else "Max. Wave Height (Meters)",
    "apply": "Toepassen" if is_nl else "Apply",
    "why_title": "💡 Waarom deze app is ontwikkeld" if is_nl else "💡 Why this app was developed",
    "why_text": (
        "Deze app is ontstaan vanuit de persoonlijke behoefte om te kunnen bepalen wat de beste periode is "
        "om een bepaalde bestemming te bezoeken qua wind, golven, stormrisico en andere weersomstandigheden."
        if is_nl else
        "This app was created out of a personal need to determine the best period to visit a specific destination "
        "regarding wind, waves, storm risk, and other weather conditions."
    ),
    "loc_input": "Locatie / Haven" if is_nl else "Location / Harbor",
    "country_filter": "Landfilter" if is_nl else "Country Filter",
    "all_countries": "Alle Landen (Geen filter)" if is_nl else "All Countries (No filter)",
}

# FIX 5: Taalafhankelijke maandnamen als lookup (was hardcoded NL in m2)
MONTH_NAMES_NL = {1:"01 - Jan", 2:"02 - Feb", 3:"03 - Mar", 4:"04 - Apr",
                  5:"05 - Mei", 6:"06 - Jun", 7:"07 - Jul", 8:"08 - Aug",
                  9:"09 - Sep", 10:"10 - Okt", 11:"11 - Nov", 12:"12 - Dec"}
MONTH_NAMES_EN = {1:"01 - Jan", 2:"02 - Feb", 3:"03 - Mar", 4:"04 - Apr",
                  5:"05 - May", 6:"06 - Jun", 7:"07 - Jul", 8:"08 - Aug",
                  9:"09 - Sep", 10:"10 - Oct", 11:"11 - Nov", 12:"12 - Dec"}

st.title(t["title"])
st.write(t["subtitle"])

def degrees_to_cardinal(d):
    if pd.isna(d):
        return "-"
    try:
        dirs = ['N', 'NNO', 'NO', 'ONO', 'O', 'OZO', 'ZO', 'ZZO',
                'Z', 'ZZW', 'ZW', 'WZW', 'W', 'WNW', 'NW', 'NNW']
        ix = int((float(d) + 11.25) / 22.5) % 16
        return dirs[ix]
    except (ValueError, TypeError):
        return "-"

# --- Sidebar Navigatie (FIX 7: st.radio vervangt losse buttons) ---
with st.sidebar:
    st.markdown(f"### {t['nav_header']}")
    nav_options = [t["m1"], t["m2"], t["m3"], t["m4"], t["m5"], t["m6"]]
    app_mode = st.radio(
        label="nav",
        options=nav_options,
        label_visibility="collapsed",
        key="nav_radio"
    )

    st.markdown("---")
    unit_wind = st.selectbox(t["wind_unit"], ["Knopen (kt)", "Beaufort (Bft)", "m/s", "km/h"])

    with st.form(key="zeil_filters_form"):
        st.subheader(t["filters_sub"])
        min_wind_kt, max_wind_kt = st.slider(t["w_speed"], 0, 40, (10, 22), 1)
        max_gust_kt = st.slider(t["w_gust"], 10, 50, 28, 1)
        max_wave_m = st.slider(t["w_wave"], 0.5, 5.0, 1.8, 0.1)
        applied = st.form_submit_button(t["apply"], use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<small>⚠️ **Disclaimer:** Uitsluitend ter ondersteuning van passageplanning. "
        "Geen vervanging voor officiële kaarten of getijdentabellen.</small>",
        unsafe_allow_html=True
    )

# FIX 1: is_nl als parameter meegeven aan gecachte functies zodat
# de cache-key de taalinstelling bevat (was: globale variabele buiten cache-scope)
@st.cache_data
def search_locations(query, country_code=None, language="nl"):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=20&language={language}&format=json"
    try:
        res = requests.get(url, timeout=10).json()
        if "results" in res:
            results = res["results"]
            if country_code:
                results = [loc for loc in results if loc.get("country_code", "").upper() == country_code.upper()]
            return results
    except Exception:
        pass
    return []

# FIX 2: try/except toegevoegd op beide API-calls in fetch_combined_history
@st.cache_data
def fetch_combined_history(lat, lon, start_str, end_str):
    df_w = pd.DataFrame()
    df_m = pd.DataFrame()

    try:
        weather_url = "https://archive-api.open-meteo.com/v1/archive"
        weather_params = {
            "latitude": lat, "longitude": lon,
            "start_date": start_str, "end_date": end_str,
            "hourly": ["wind_speed_10m", "wind_gusts_10m", "wind_direction_10m", "temperature_2m", "precipitation"],
            "wind_speed_unit": "kn", "timezone": "auto"
        }
        res_w = requests.get(weather_url, params=weather_params, timeout=15).json()
        df_w = pd.DataFrame(res_w.get("hourly", {}))
    except Exception as e:
        st.warning(f"Weerdata ophalen mislukt: {e}")

    try:
        marine_url = "https://marine-api.open-meteo.com/v1/marine"
        marine_params = {
            "latitude": lat, "longitude": lon,
            "start_date": start_str, "end_date": end_str,
            "hourly": ["wave_height", "wave_period"], "timezone": "auto"
        }
        res_m = requests.get(marine_url, params=marine_params, timeout=15).json()
        df_m = pd.DataFrame(res_m.get("hourly", {}))
    except Exception:
        pass  # Mariene data is optioneel; app werkt ook zonder

    if df_w.empty:
        return pd.DataFrame()

    df = df_w.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["wave_height"] = df_m["wave_height"].fillna(0) if not df_m.empty and "wave_height" in df_m else 0.0
    df["wave_period"] = df_m["wave_period"].fillna(0) if not df_m.empty and "wave_period" in df_m else 0.0
    df["precipitation"] = df_w["precipitation"].fillna(0) if "precipitation" in df_w else 0.0
    df["cardinal"] = df["wind_direction_10m"].apply(degrees_to_cardinal)
    return df

# FIX 2: try/except ook toegevoegd in fetch_forecast_and_tides
@st.cache_data
def fetch_forecast_and_tides(lat, lon):
    df_w = pd.DataFrame()
    df_m = pd.DataFrame()

    try:
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat, "longitude": lon,
            "forecast_days": 14,
            "hourly": ["wind_speed_10m", "wind_gusts_10m", "wind_direction_10m", "temperature_2m", "precipitation"],
            "wind_speed_unit": "kn", "timezone": "auto"
        }
        res_w = requests.get(weather_url, params=weather_params, timeout=15).json()
        df_w = pd.DataFrame(res_w.get("hourly", {}))
    except Exception as e:
        st.warning(f"Verwachtingsdata ophalen mislukt: {e}")

    try:
        marine_url = "https://marine-api.open-meteo.com/v1/marine"
        marine_params = {
            "latitude": lat, "longitude": lon,
            "forecast_days": 14,
            "hourly": ["wave_height", "ocean_current_velocity", "ocean_current_direction", "sea_level_height_msl"],
            "timezone": "auto"
        }
        res_m = requests.get(marine_url, params=marine_params, timeout=15).json()
        df_m = pd.DataFrame(res_m.get("hourly", {}))
    except Exception:
        pass

    if df_w.empty:
        return pd.DataFrame()

    df = df_w.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["wave_height"] = df_m.get("wave_height", pd.Series(0)).fillna(0) if not df_m.empty and "wave_height" in df_m else 0.0
    df["ocean_current_velocity"] = df_m.get("ocean_current_velocity", pd.Series(0, index=df.index)) if not df_m.empty and "ocean_current_velocity" in df_m else 0.0
    df["ocean_current_direction"] = df_m.get("ocean_current_direction", pd.Series(0, index=df.index)) if not df_m.empty and "ocean_current_direction" in df_m else 0.0

    df["current_cardinal"] = df["ocean_current_direction"].apply(degrees_to_cardinal)

    dir_rad = np.radians(df["ocean_current_direction"])
    df["current_east_ms"] = (df["ocean_current_velocity"] * np.sin(dir_rad)).round(2)
    df["current_north_ms"] = (df["ocean_current_velocity"] * np.cos(dir_rad)).round(2)

    if not df_m.empty and "sea_level_height_msl" in df_m:
        df["tide_dm"] = df_m["sea_level_height_msl"] * 10.0
    else:
        df["tide_dm"] = 0.0

    df["cardinal"] = df["wind_direction_10m"].apply(degrees_to_cardinal)
    return df

def haversine(lat1, lon1, lat2, lon2):
    R = 3440.065
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def apply_sailing_filters(df, min_wind, max_wind, max_gust, max_wave):
    if df.empty:
        return df
    df["is_ideal"] = (
        (df["wind_speed_10m"] >= min_wind) &
        (df["wind_speed_10m"] <= max_wind) &
        (df["wind_gusts_10m"] <= max_gust) &
        (df["wave_height"] <= max_wave)
    )
    return df

def convert_units(df, unit_wind):
    if unit_wind == "Beaufort (Bft)":
        def knots_to_bft(k):
            if pd.isna(k) or k < 1: return 0
            elif k <= 3: return 1
            elif k <= 6: return 2
            elif k <= 10: return 3
            elif k <= 16: return 4
            elif k <= 21: return 5
            elif k <= 27: return 6
            elif k <= 33: return 7
            elif k <= 40: return 8
            elif k <= 47: return 9
            elif k <= 55: return 10
            elif k <= 63: return 11
            else: return 12
        df["wind_display"] = df["wind_speed_10m"].apply(knots_to_bft)
        df["gust_display"] = df["wind_gusts_10m"].apply(knots_to_bft)
        return df, "Bft"
    elif unit_wind == "m/s":
        df["wind_display"] = df["wind_speed_10m"] * 0.514444
        df["gust_display"] = df["wind_gusts_10m"] * 0.514444
        return df, "m/s"
    elif unit_wind == "km/h":
        df["wind_display"] = df["wind_speed_10m"] * 1.852
        df["gust_display"] = df["wind_gusts_10m"] * 1.852
        return df, "km/h"
    else:
        df["wind_display"] = df["wind_speed_10m"]
        df["gust_display"] = df["wind_gusts_10m"]
        return df, "kt"

def get_dominant_wind_dir(series):
    valid_series = series[series != "-"]
    if valid_series.empty:
        return "-"
    mode_val = valid_series.mode()
    return mode_val[0] if not mode_val.empty else valid_series.iloc[0]

country_mapping = {
    t["all_countries"]: None,
    "Verenigd Koninkrijk (GB)" if is_nl else "United Kingdom (GB)": "GB",
    "Nederland (NL)" if is_nl else "Netherlands (NL)": "NL",
    "Duitsland (DE)" if is_nl else "Germany (DE)": "DE",
    "Denemarken (DK)" if is_nl else "Denmark (DK)": "DK",
    "Frankrijk (FR)" if is_nl else "France (FR)": "FR",
    "Noorwegen (NO)" if is_nl else "Norway (NO)": "NO",
    "Zweden (SE)" if is_nl else "Sweden (SE)": "SE"
}

# FIX 1: taal als parameter meegeven aan search_locations (language="nl"/"en")
_lang_code = "nl" if is_nl else "en"

# ==============================================================================
# MODUS: ENKELE LOCATIE & 12M HISTORIE
# ==============================================================================
if app_mode == t["m1"]:
    st.subheader(t["m1"])

    with st.expander(t["why_title"], expanded=True):
        st.write(t["why_text"])

    c_col1, c_col2 = st.columns([2, 1])
    with c_col1:
        location_query = st.text_input(t["loc_input"], value="Lauwersoog")
    with c_col2:
        selected_country_label = st.selectbox(t["country_filter"], list(country_mapping.keys()))

    country_code = country_mapping[selected_country_label]
    # FIX 1: language param toegevoegd
    results = search_locations(location_query, country_code, language=_lang_code)

    if not results:
        st.error(f"Geen locatie gevonden voor '{location_query}'.")
        st.stop()

    loc_options = {f"{loc['name']} ({loc.get('country', '')}, {loc.get('admin1', '')})": loc for loc in results}
    selected_label = st.selectbox("Selecteer de juiste locatie uit de resultaten", list(loc_options.keys()))
    selected_loc = loc_options[selected_label]

    lat, lon, loc_name = selected_loc['latitude'], selected_loc['longitude'], selected_loc['name']

    end_date = datetime.now().date() - timedelta(days=3)
    start_date = end_date - timedelta(days=365)

    with st.spinner(f"Historie ophalen voor {loc_name}..."):
        df = fetch_combined_history(lat, lon, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if not df.empty:
        df = apply_sailing_filters(df, min_wind_kt, max_wind_kt, max_gust_kt, max_wave_m)
        df, unit_label = convert_units(df, unit_wind)
        df['month'] = df['time'].dt.strftime('%Y-%m (%b)')

        monthly_df = df.groupby('month').agg(pct_ideal=('is_ideal', lambda x: round(x.mean()*100, 2))).reset_index()

        fig_m = px.bar(
            monthly_df, x='month', y='pct_ideal',
            title="Geschikt Zeilweer per Maand (%)",
            labels={'pct_ideal': 'Geschikt Zeilweer (%)', 'month': 'Maand'},
            color='pct_ideal', color_continuous_scale='Greens',
            text='pct_ideal'
        )
        fig_m.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig_m, use_container_width=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Geschikt Zeilweer", f"{df['is_ideal'].mean()*100:.2f}%")
        col2.metric("Gem. Wind", f"{df['wind_display'].mean():.2f} {unit_label}")
        col3.metric("Max. Stoot", f"{df['gust_display'].max():.2f} {unit_label}")
        col4.metric("Overheersende Wind", get_dominant_wind_dir(df['cardinal']))
        col5.metric("Totale Neerslag", f"{df['precipitation'].sum():.2f} mm")

        st.markdown("#### 📋 Maandelijkse Details")
        monthly_table = df.groupby('month').agg(
            Geschikt_Zeilweer_Pct=('is_ideal', lambda x: round(x.mean()*100, 2)),
            Gem_Wind=('wind_display', lambda x: round(x.mean(), 2)),
            Max_Stoot=('gust_display', lambda x: round(x.max(), 2)),
            Overheersende_Wind=('cardinal', get_dominant_wind_dir),
            Totale_Neerslag_mm=('precipitation', lambda x: round(x.sum(), 2))
        ).reset_index().rename(columns={
            'month': 'Maand',
            'Geschikt_Zeilweer_Pct': 'Geschikt Zeilweer (%)',
            'Gem_Wind': f'Gem. Wind ({unit_label})',
            'Max_Stoot': f'Max. Stoot ({unit_label})',
            'Overheersende_Wind': 'Overheersende Wind',
            'Totale_Neerslag_mm': 'Neerslag (mm)'
        })
        st.dataframe(monthly_table, use_container_width=True)

# ==============================================================================
# MODUS: MULTI-JAAR VERGELIJKING (10 JAAR)
# ==============================================================================
elif app_mode == t["m2"]:
    st.subheader(t["m2"])

    col_l, col_c, col_r = st.columns([2, 1, 1])
    with col_l:
        location_query = st.text_input(t["loc_input"], value="Lauwersoog", key="m2_loc")
    with col_c:
        selected_country_label = st.selectbox(t["country_filter"], list(country_mapping.keys()), key="multi_country")
    with col_r:
        time_groupby = st.selectbox("Groepering op X-as", ["Per Maand", "Per Week"])

    country_code = country_mapping[selected_country_label]
    # FIX 1: language param toegevoegd
    results = search_locations(location_query, country_code, language=_lang_code)
    if not results:
        st.error("Geen locatie gevonden met dit landfilter.")
        st.stop()
    loc_options = {f"{loc['name']} ({loc.get('country', '')}, {loc.get('admin1', '')})": loc for loc in results}
    selected_label = st.selectbox("Selecteer de juiste locatie uit de zoekresultaten", list(loc_options.keys()), key="m2_sel")
    selected_loc = loc_options[selected_label]
    lat, lon, loc_name = selected_loc['latitude'], selected_loc['longitude'], selected_loc['name']

    current_year = datetime.now().year
    years = [current_year - i for i in range(1, 11)]
    latest_year = str(years[0])
    older_years = [str(y) for y in years[1:]]

    # FIX 3: parallelle fetches via ThreadPoolExecutor (was: sequentiële loop)
    # FIX 5: taalafhankelijke maandnamen
    month_names = MONTH_NAMES_NL if is_nl else MONTH_NAMES_EN

    def fetch_year(yr):
        start_yr = datetime(yr, 1, 1)
        end_yr = datetime(yr, 12, 31)
        df_yr = fetch_combined_history(lat, lon, start_yr.strftime("%Y-%m-%d"), end_yr.strftime("%Y-%m-%d"))
        if not df_yr.empty:
            df_yr['year'] = str(yr)
            df_yr['week'] = df_yr['time'].dt.isocalendar().week
            df_yr['month_num'] = df_yr['time'].dt.month
            df_yr['month_name'] = df_yr['month_num'].map(month_names)
        return df_yr

    multi_df_list = []
    with st.spinner(f"Data verzamelen voor {loc_name} over de afgelopen 10 jaar..."):
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_year, yr): yr for yr in years}
            for future in as_completed(futures):
                result = future.result()
                if not result.empty:
                    multi_df_list.append(result)

    if multi_df_list:
        df_multi = pd.concat(multi_df_list, ignore_index=True)
        df_multi = apply_sailing_filters(df_multi, min_wind_kt, max_wind_kt, max_gust_kt, max_wave_m)
        df_multi, unit_label = convert_units(df_multi, unit_wind)

        is_month_group = time_groupby == "Per Maand"
        x_col = 'month_name' if is_month_group else 'week'
        x_label = "Maand" if is_month_group else "Week (1-52)"

        grouped_df = df_multi.groupby([x_col, 'year']).agg(
            pct_ideal=('is_ideal', lambda x: round(x.mean() * 100, 2)),
            gem_wind=('wind_display', lambda x: round(x.mean(), 2)),
            max_stoot=('gust_display', lambda x: round(x.max(), 2)),
            dominant_dir=('cardinal', get_dominant_wind_dir),
            gem_temp=('temperature_2m', lambda x: round(x.mean(), 2)),
            tot_rain=('precipitation', lambda x: round(x.sum(), 2)),
            gem_golf=('wave_height', lambda x: round(x.mean(), 2))
        ).reset_index().rename(columns={
            x_col: x_label,
            'year': 'Jaar',
            'pct_ideal': 'Geschikt Zeilweer (%)',
            'gem_wind': f'Gem. Wind ({unit_label})',
            'max_stoot': f'Max. Stoot ({unit_label})',
            'dominant_dir': 'Overheersende Wind',
            'gem_temp': 'Gem. Temp (°C)',
            'tot_rain': 'Neerslag (mm)',
            'gem_golf': 'Gem. Golf (m)'
        })

        fig_combo = go.Figure()
        df_latest = grouped_df[grouped_df['Jaar'] == latest_year]
        if not df_latest.empty:
            fig_combo.add_trace(go.Bar(
                x=df_latest[x_label], y=df_latest['Geschikt Zeilweer (%)'],
                name=f"{latest_year} (Meest Recent)",
                marker_color='rgba(44, 160, 44, 0.45)', marker_line_color='#2ca02c', marker_line_width=1.5
            ))

        for yr in older_years:
            df_yr = grouped_df[grouped_df['Jaar'] == yr]
            if not df_yr.empty:
                fig_combo.add_trace(go.Scatter(
                    x=df_yr[x_label], y=df_yr['Geschikt Zeilweer (%)'],
                    name=f"Jaar {yr}", mode='lines+markers',
                    line=dict(width=1.8, shape='spline')
                ))

        fig_combo.update_layout(
            title=f"Geschikt Zeilweer (%): Staven ({latest_year}) vs. Lijnen (10 Jaar Historie)",
            xaxis_title=x_label, yaxis_title="Geschikt Zeilweer (%)", hovermode="x unified"
        )
        st.plotly_chart(fig_combo, use_container_width=True)
        st.dataframe(grouped_df, use_container_width=True)

# ==============================================================================
# MODUS: OPTIMALE ROUTE & STROOMROUTING
# ==============================================================================
elif app_mode == t["m3"]:
    st.subheader(t["m3"])
    st.write("Voer havennamen in en filter eventueel per land.")

    with st.form(key="route_form"):
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            start_query = st.text_input("Startpunt", value="Lauwersoog")
            start_country_label = st.selectbox("Start Landfilter", list(country_mapping.keys()), key="start_c")
        with col_r2:
            via_query = st.text_input("Optionele Tussenhaven", value="")
            via_country_label = st.selectbox("Tussen Landfilter", list(country_mapping.keys()), key="via_c")
        with col_r3:
            end_query = st.text_input("Bestemming", value="Dover")
            end_country_label = st.selectbox("Bestemming Landfilter", list(country_mapping.keys()), key="end_c")
        with col_r4:
            boat_speed_kt = st.number_input("Gem. Bootsnelheid (knopen)", min_value=2.0, max_value=15.0, value=5.0, step=0.5)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            dep_date = st.date_input("Vertrekdatum", value=datetime.now().date())
        with col_t2:
            dep_time = st.time_input("Vertrektijd", value=datetime.now().time())

        submit_search = st.form_submit_button("🔍 Zoek & Bevestig Havens", use_container_width=True)

    if submit_search:
        st.session_state["search_done"] = True
        st.session_state["start_query"] = start_query
        st.session_state["start_cc"] = country_mapping[start_country_label]
        st.session_state["via_query"] = via_query
        st.session_state["via_cc"] = country_mapping[via_country_label]
        st.session_state["end_query"] = end_query
        st.session_state["end_cc"] = country_mapping[end_country_label]
        st.session_state["boat_speed_kt"] = boat_speed_kt
        st.session_state["departure_dt"] = datetime.combine(dep_date, dep_time)
        st.session_state["calc_route"] = False

    if st.session_state.get("search_done", False):
        # FIX 1: language param toegevoegd
        start_results = search_locations(st.session_state["start_query"], st.session_state["start_cc"], language=_lang_code)
        end_results = search_locations(st.session_state["end_query"], st.session_state["end_cc"], language=_lang_code)

        if not start_results or not end_results:
            st.error("Eén van de havens kon niet worden gevonden.")
            st.stop()

        st.markdown("---")
        st.markdown("#### ⚓ Bevestig de juiste havens:")
        col_s_sel, col_e_sel = st.columns(2)

        start_options = {f"{loc['name']} ({loc.get('country', '')}, {loc.get('admin1', '')})": loc for loc in start_results}
        end_options = {f"{loc['name']} ({loc.get('country', '')}, {loc.get('admin1', '')})": loc for loc in end_results}

        with col_s_sel:
            selected_start_label = st.selectbox("Startpunt bevestigen", list(start_options.keys()))
            chosen_start = start_options[selected_start_label]

        with col_e_sel:
            selected_end_label = st.selectbox("Bestemming bevestigen", list(end_options.keys()))
            chosen_end = end_options[selected_end_label]

        chosen_via = None
        if st.session_state["via_query"].strip() != "":
            # FIX 1: language param toegevoegd
            via_results = search_locations(st.session_state["via_query"], st.session_state["via_cc"], language=_lang_code)
            if via_results:
                via_options = {f"{loc['name']} ({loc.get('country', '')}, {loc.get('admin1', '')})": loc for loc in via_results}
                selected_via_label = st.selectbox("Tussenhaven bevestigen", list(via_options.keys()))
                chosen_via = via_options[selected_via_label]

        if st.button("🚀 Start Maritieme Route Berekening", use_container_width=True):
            st.session_state["calc_route"] = True
            st.session_state["chosen_start"] = chosen_start
            st.session_state["chosen_end"] = chosen_end
            st.session_state["chosen_via"] = chosen_via

    if st.session_state.get("calc_route", False):
        c_start = st.session_state["chosen_start"]
        c_end = st.session_state["chosen_end"]
        c_via = st.session_state.get("chosen_via", None)
        b_spd = st.session_state["boat_speed_kt"]
        dep_dt = st.session_state["departure_dt"]

        lat_a, lon_a, name_a = c_start['latitude'], c_start['longitude'], c_start['name']
        lat_b, lon_b, name_b = c_end['latitude'], c_end['longitude'], c_end['name']

        lat_via, lon_via = None, None
        if c_via:
            lat_via, lon_via = c_via['latitude'], c_via['longitude']

        with st.spinner("Maritieme route berekenen..."):
            try:
                if lat_via and lon_via:
                    route_leg1 = sr.searoute([lon_a, lat_a], [lon_via, lat_via], units="naut")
                    route_leg2 = sr.searoute([lon_via, lat_via], [lon_b, lat_b], units="naut")
                    coords_raw = route_leg1.geometry['coordinates'] + route_leg2.geometry['coordinates'][1:]
                    total_dist_nm = route_leg1.properties['length'] + route_leg2.properties['length']
                else:
                    route_res = sr.searoute([lon_a, lat_a], [lon_b, lat_b], units="naut")
                    coords_raw = route_res.geometry['coordinates']
                    total_dist_nm = route_res.properties['length']
            except Exception as e:
                st.error(f"Fout bij route berekening: {e}")
                st.stop()

            all_points_raw = [[c[1], c[0]] for c in coords_raw]
            harbor_target = [lat_b, lon_b]

            target_waypoints = 45
            all_points = []

            if len(all_points_raw) > 1:
                distances = [0.0]
                for j in range(1, len(all_points_raw)):
                    d = haversine(all_points_raw[j-1][0], all_points_raw[j-1][1], all_points_raw[j][0], all_points_raw[j][1])
                    distances.append(distances[-1] + d)

                total_path_dist = distances[-1]
                if total_path_dist > 0:
                    target_distances = np.linspace(0, total_path_dist, target_waypoints)
                    for target_d in target_distances:
                        idx = 0
                        while idx < len(distances) - 1 and distances[idx+1] < target_d:
                            idx += 1
                        if idx >= len(distances) - 1:
                            all_points.append(all_points_raw[-1])
                        else:
                            d1, d2 = distances[idx], distances[idx+1]
                            p1, p2 = np.array(all_points_raw[idx]), np.array(all_points_raw[idx+1])
                            factor = (target_d - d1) / (d2 - d1) if d2 > d1 else 0
                            interpolated_point = p1 + factor * (p2 - p1)
                            all_points.append(interpolated_point.tolist())
                else:
                    all_points = all_points_raw
            else:
                all_points = all_points_raw

            approach_distances = np.linspace(0, haversine(all_points[-1][0], all_points[-1][1], harbor_target[0], harbor_target[1]), 6)
            for step_d in approach_distances[1:]:
                p_prev = np.array(all_points[-1])
                p_dest = np.array(harbor_target)
                total_d_app = haversine(p_prev[0], p_prev[1], p_dest[0], p_dest[1])
                factor = (step_d / total_d_app) if total_d_app > 0 else 1.0
                intermediate_pt = p_prev + factor * (p_dest - p_prev)
                all_points.append(intermediate_pt.tolist())

            hours_per_segment = (total_dist_nm / b_spd) / (len(all_points) - 1) if len(all_points) > 1 else 0

            route_data = []
            for i, (lt, ln) in enumerate(all_points):
                wp_name = f"WP {i+1}" if (0 < i < len(all_points)-1) else (name_a if i == 0 else name_b)
                arrival_time = dep_dt + timedelta(hours=i * hours_per_segment)

                df_wp = fetch_forecast_and_tides(lt, ln)

                # FIX 4: None/NaN als fallback i.p.v. misleidende dummy-waarden
                current_v = None
                current_d = None
                current_card = "-"
                current_e = None
                wind_s = None
                wave_h = None
                tide_val = None

                if not df_wp.empty and 'time' in df_wp.columns:
                    df_wp['time_diff'] = (df_wp['time'] - arrival_time).abs()
                    closest_row = df_wp.loc[df_wp['time_diff'].idxmin()]

                    current_v = closest_row.get('ocean_current_velocity', None)
                    if pd.isna(current_v): current_v = None
                    current_d = closest_row.get('ocean_current_direction', None)
                    if pd.isna(current_d): current_d = None
                    current_card = closest_row.get('current_cardinal', "-")
                    current_e = closest_row.get('current_east_ms', None)
                    wind_s = closest_row.get('wind_speed_10m', None)
                    wave_h = closest_row.get('wave_height', None)
                    raw_tide = closest_row.get('tide_dm', None)
                    tide_val = None if pd.isna(raw_tide) else raw_tide

                route_data.append({
                    "Waypoint": wp_name,
                    "Latitude": round(float(lt), 4),
                    "Longitude": round(float(ln), 4),
                    "Tijd": arrival_time,
                    "Getij (dm)": round(float(tide_val), 1) if tide_val is not None else "n/a",
                    "Stroom (m/s)": round(float(current_v), 2) if current_v is not None else "n/a",
                    "Oost/West (+/-)": round(float(current_e), 2) if current_e is not None else "n/a",
                    "Stroomrichting": current_card,
                    "Windsnelheid (kt)": round(float(wind_s), 1) if wind_s is not None else "n/a",
                    "Golfhoogte (m)": round(float(wave_h), 2) if wave_h is not None else "n/a"
                })

            df_route_res = pd.DataFrame(route_data)

        st.success(f"🧭 **Route Berekend:** Open water afstand is **{total_dist_nm:.1f} NM**.")

        center_lat = df_route_res['Latitude'].mean()
        center_lon = df_route_res['Longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap")

        split_idx = len(all_points) - 6
        open_water_line = all_points[:split_idx+1]
        approach_line = all_points[split_idx:]

        folium.PolyLine(open_water_line, color="darkorange", weight=4, tooltip="Open Water Route").add_to(m)
        folium.PolyLine(approach_line, color="deepskyblue", weight=4, dash_array="5, 10", tooltip="Approximation").add_to(m)
        folium.Marker([lat_a, lon_a], popup=f"Start: {name_a}", icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker([lat_b, lon_b], popup=f"End: {name_b}", icon=folium.Icon(color="red", icon="flag")).add_to(m)

        components.html(m._repr_html_(), height=550)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            gpx_content = '<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="Sailing Weather Historian Pro">\n  <trk>\n    <n>' + name_a + ' to ' + name_b + '</n>\n    <trkseg>\n'
            for lt, ln in all_points:
                gpx_content += f'      <trkpt lat="{lt}" lon="{ln}"></trkpt>\n'
            gpx_content += '    </trkseg>\n  </trk>\n</gpx>'

            st.download_button(
                label="📥 Download GPX",
                data=gpx_content,
                file_name=f"route_{name_a}_naar_{name_b}.gpx",
                mime="application/gpx+xml",
                use_container_width=True
            )
        with col_b2:
            # FIX 8: st.link_button i.p.v. inline HTML button
            gmaps_url = f"https://www.google.com/maps/dir/?api=1&origin={lat_a},{lon_a}&destination={lat_b},{lon_b}"
            st.link_button("🗺️ Open in Google Maps", gmaps_url, use_container_width=True)

        st.markdown("#### 📊 Waypoint Details, Oost/West Stroom (+/-) & Getij (dm)")
        st.dataframe(df_route_res, use_container_width=True)

# ==============================================================================
# MODUS: 14-DAAGSE VERWACHTING & GETIJDEN
# ==============================================================================
elif app_mode == t["m4"]:
    st.subheader(t["m4"])

    col_fc1, col_fc_c, col_fc2 = st.columns([2, 1, 1])
    with col_fc1:
        location_query = st.text_input(t["loc_input"], value="Lauwersoog", key="fc_loc")
    with col_fc_c:
        selected_country_label = st.selectbox(t["country_filter"], list(country_mapping.keys()), key="fc_country")

    country_code = country_mapping[selected_country_label]
    # FIX 1: language param toegevoegd
    results = search_locations(location_query, country_code, language=_lang_code)
    if not results:
        st.error("Locatie niet gevonden.")
        st.stop()
    loc_options = {f"{loc['name']} ({loc.get('country', '')}, {loc.get('admin1', '')})": loc for loc in results}
    selected_label = st.selectbox("Selecteer locatie", list(loc_options.keys()), key="fc_sel")
    selected_loc = loc_options[selected_label]
    lat, lon, loc_name = selected_loc['latitude'], selected_loc['longitude'], selected_loc['name']

    with st.spinner("Verwachting & getijden ophalen..."):
        df_fc = fetch_forecast_and_tides(lat, lon)

    if not df_fc.empty:
        df_fc = apply_sailing_filters(df_fc, min_wind_kt, max_wind_kt, max_gust_kt, max_wave_m)
        df_fc, unit_label = convert_units(df_fc, unit_wind)
        df_fc['date_str'] = df_fc['time'].dt.strftime('%Y-%m-%d (%a)')

        available_dates = sorted(df_fc['date_str'].unique().tolist())
        with col_fc2:
            selected_date = st.selectbox("Kies dag", available_dates)

        daily_summary = df_fc.groupby('date_str').agg(
            pct_ideal=('is_ideal', lambda x: round(x.mean() * 100, 2)),
            gem_wind=('wind_display', lambda x: round(x.mean(), 2)),
            max_stoot=('gust_display', lambda x: round(x.max(), 2)),
            dominant_dir=('cardinal', get_dominant_wind_dir),
            tot_rain=('precipitation', lambda x: round(x.sum(), 2)),
            max_golf=('wave_height', lambda x: round(x.max(), 2))
        ).reset_index()

        st.markdown(f"### 14-Daags Overzicht voor **{loc_name}**")
        fig_fc_bar = px.bar(
            daily_summary, x='date_str', y='pct_ideal',
            title="Percentage Geschikt Zeilweer per Dag (%)",
            labels={'date_str': 'Datum', 'pct_ideal': 'Geschikt Zeilweer (%)'},
            color='pct_ideal', color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_fc_bar, use_container_width=True)

        df_day = df_fc[df_fc['date_str'] == selected_date].copy()
        st.markdown(f"### Uur-tot-Uur Verwachting, Oost/West Stroom (+/-) & Getij voor **{selected_date}**")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Zeilbaarheid Dag", f"{df_day['is_ideal'].mean()*100:.2f}%")
        c2.metric("Gem. Wind", f"{df_day['wind_display'].mean():.2f} {unit_label}")
        c3.metric("Max Stoot", f"{df_day['gust_display'].max():.2f} {unit_label}")
        c4.metric("Overheersende Wind", get_dominant_wind_dir(df_day['cardinal']))
        c5.metric("Totale Regen", f"{df_day['precipitation'].sum():.2f} mm")

        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Scatter(x=df_day['time'], y=df_day['wind_display'], name=f"Wind ({unit_label})", line=dict(color='blue', width=2)))
        fig_hourly.add_trace(go.Scatter(x=df_day['time'], y=df_day['wave_height'], name="Golfhoogte (m)", yaxis="y2", line=dict(color='teal', width=2)))
        fig_hourly.update_layout(
            title=f"Wind & Golven op {selected_date}",
            yaxis_title=f"Wind ({unit_label})", yaxis2=dict(title="Golf (m)", overlaying="y", side="right"),
            hovermode="x unified"
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

        fig_tide = go.Figure()
        if 'current_east_ms' in df_day.columns:
            fig_tide.add_trace(go.Scatter(
                x=df_day['time'], y=df_day['current_east_ms'],
                name="Oost/West Stroom (+ = Oost, - = West) [m/s]",
                line=dict(color='orange', width=2)
            ))
        if 'tide_dm' in df_day.columns:
            fig_tide.add_trace(go.Scatter(
                x=df_day['time'], y=df_day['tide_dm'],
                name="Getijhoogte (dm)", yaxis="y2",
                line=dict(color='purple', width=2, dash='dot')
            ))

        fig_tide.update_layout(
            title=f"Getijdenstroom (+/- Oost/West As) & Getijhoogte (dm) op {selected_date}",
            yaxis=dict(title="Oost/West Stroom (m/s) [+ = Oost, - = West]", zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
            yaxis2=dict(title="Getijhoogte (dm)", overlaying="y", side="right"),
            hovermode="x unified"
        )
        st.plotly_chart(fig_tide, use_container_width=True)

        display_cols = ['time', 'wind_display', 'wind_gusts_10m', 'cardinal', 'wave_height', 'ocean_current_velocity', 'current_east_ms', 'current_north_ms', 'current_cardinal', 'tide_dm', 'precipitation']
        rename_map = {
            'time': 'Tijd',
            'wind_display': f'Wind ({unit_label})',
            'wind_gusts_10m': 'Stoten (kt)',
            'cardinal': 'Windrichting',
            'wave_height': 'Golfhoogte (m)',
            'ocean_current_velocity': 'Tot. Stroom (m/s)',
            'current_east_ms': 'Oost/West Stroom (+/- m/s)',
            'current_north_ms': 'Noord/Zuid Stroom (+/- m/s)',
            'current_cardinal': 'Stroomrichting',
            'tide_dm': 'Getijhoogte (dm)',
            'precipitation': 'Neerslag (mm)'
        }
        df_table = df_day[[c for c in display_cols if c in df_day.columns]].rename(columns=rename_map)
        st.dataframe(df_table, use_container_width=True)

# ==============================================================================
# MODUS: TECHNISCHE BRONNEN & DISCLAIMER
# ==============================================================================
elif app_mode == t["m5"]:
    st.subheader(t["m5"])
    st.markdown("""
    ### Overzicht van Open-Source Databronnen en API's
    Deze applicatie maakt gebruik van geavanceerde open-source weer-, mariene en routing-modellen:

    1. **Open-Meteo Weather Archive & Forecast API:**
       - **Parameters:** Windsnelheid op 10m hoogte (`wind_speed_10m`), windstoten (`wind_gusts_10m`), windrichting (`wind_direction_10m`), luchttemperatuur (`temperature_2m`) en neerslag (`precipitation`).
       - **Gebruik:** Berekent historische weerpatronen (tot 10 jaar terug) en 14-daagse verwachtingen per uur.

    2. **Open-Meteo Marine API:**
       - **Parameters:** Golfhoogte (`wave_height`), golfperiode (`wave_period`), oceaangolfrichting, oceaankanaal-stroomsnelheid (`ocean_current_velocity`) en stroomrichting (`ocean_current_direction`).
       - **Getijgegevens:** Waterstand ten opzichte van gemiddeld zeeniveau (`sea_level_height_msl`), omgerekend naar **decimeters (dm)** voor nauwkeurige ondiepte- en drempelbeoordelingen.
       - **Stroomanalyse (+/- as):** Vector-ontbinding van de stroomsnelheid in een **Oost/West-component** (waarbij Oost positief en West negatief is) en een **Noord/Zuid-component**.

    3. **SeaRoute Library (`searoute`):**
       - Berekent automatisch de meest optimale maritieme water-routing (om landmassa's en ondieptes heen) tussen havens en waypoints in zeemijlen (NM).

    4. **OpenStreetMap & Folium:**
       - Interactieve kaartvisualisatie van de berekende routes, inclusief GPX-exportfunctionaliteit en directe integratie met navigatie-apps.

    ---
    ⚠️ **Disclaimer:** Deze software is uitsluitend bedoeld als hulpmiddel en ter ondersteuning van de passageplanning. De gebruiker blijft te allen tijde zelf verantwoordelijk voor de veiligheid aan boord, actuele officiële waterkaarten, lokale getijdentabellen en weersvoorspellingen.
    """)

# ==============================================================================
# MODUS: DONATIES & SUPPORT
# ==============================================================================
elif app_mode == t["m6"]:
    st.subheader(t["m6"])
    st.write("Steun de verdere ontwikkeling van deze app en geef feedback en suggesties via Ko-fi!")
    # FIX 8: st.link_button i.p.v. inline HTML button
    st.link_button("☕ Steun via Ko-fi", "https://ko-fi.com/sailingplanner", use_container_width=True)