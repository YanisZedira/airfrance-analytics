"""
AIR FRANCE ANALYTICS PLATFORM
Dashboard Data Analytics — Compagnie Aérienne
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==========================================================================
# CONFIG
# ==========================================================================
st.set_page_config(
    page_title="Air France • Analytics Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# COULEURS
# ==========================================================================
AF_NAVY = "#002157"
AF_BLUE = "#003580"
AF_LIGHT_BLUE = "#0055A0"
AF_RED = "#E4002B"
AF_GOLD = "#C4A35A"
AF_GREY = "#6B7B8D"
AF_BG = "#F5F7FA"
AF_WHITE = "#FFFFFF"

AF_PALETTE = [AF_NAVY, AF_RED, AF_LIGHT_BLUE, AF_GOLD, "#00A86B", "#FF6B35",
              "#8B5CF6", "#EC4899", "#14B8A6", "#F59E0B"]

# ==========================================================================
# CSS
# ==========================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #F5F7FA;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #002157 0%, #003580 100%);
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2);
    }
    section[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] {
        background: rgba(255,255,255,0.1);
        border-color: rgba(255,255,255,0.3);
    }

    /* BANNER */
    .af-banner {
        background: linear-gradient(135deg, #002157 0%, #003580 60%, #0055A0 100%);
        border-radius: 18px;
        padding: 30px 40px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    .af-banner::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 4px;
        background: #E4002B;
    }
    .af-banner h1 {
        font-size: 2.2rem;
        font-weight: 900;
        margin: 0;
        color: #FFFFFF !important;
    }
    .af-banner p {
        font-size: 1.05rem;
        margin: 5px 0 0 0;
        color: rgba(255,255,255,0.85) !important;
    }

    /* KPI CARD — chaque carte dans sa propre colonne Streamlit */
    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 20px 16px;
        box-shadow: 0 2px 12px rgba(0,33,87,0.08);
        border-left: 5px solid #002157;
        text-align: center;
        min-height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,33,87,0.14);
    }
    .kpi-card.red { border-left-color: #E4002B; }
    .kpi-card.green { border-left-color: #00A86B; }
    .kpi-card.gold { border-left-color: #C4A35A; }
    .kpi-card.blue { border-left-color: #0055A0; }

    .kpi-icon { font-size: 1.5rem; margin-bottom: 4px; }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6B7B8D;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        line-height: 1.3;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #002157;
        line-height: 1.1;
    }

    /* SECTION HEADER */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #002157;
        margin: 35px 0 18px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #E4002B;
        display: inline-block;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# CHARGEMENT DONNÉES
# ==========================================================================
@st.cache_data(ttl=3600)
def load_data():
    import os
    if not os.path.exists("data/vols.csv"):
        import subprocess
        subprocess.run(["python", "generate_dataset.py"])
    vols = pd.read_csv("data/vols.csv", sep=";", parse_dates=["date_vol"])
    aeroports = pd.read_csv("data/aeroports.csv", sep=";")
    flotte = pd.read_csv("data/flotte.csv", sep=";")
    satisfaction = pd.read_csv("data/satisfaction.csv", sep=";", parse_dates=["date_vol"])
    return vols, aeroports, flotte, satisfaction

try:
    vols_df, aeroports_df, flotte_df, satisfaction_df = load_data()
except Exception as e:
    st.error(f"⚠️ Erreur : {e}")
    st.info("Exécutez `python generate_dataset.py`")
    st.stop()

apt_dict = aeroports_df.set_index("code_iata").to_dict("index")


# ==========================================================================
# FONCTIONS
# ==========================================================================
def format_number(n, decimals=0):
    if abs(n) >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f} Mrd"
    elif abs(n) >= 1_000_000:
        return f"{n/1_000_000:.1f} M"
    elif abs(n) >= 1_000:
        return f"{n/1_000:.{decimals}f} k"
    return f"{n:,.{decimals}f}"


def kpi(icon, label, value, accent=""):
    """Affiche UN KPI dans la colonne courante."""
    cls = f" {accent}" if accent else ""
    st.markdown(f"""
    <div class="kpi-card{cls}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def section_title(text, icon=""):
    st.markdown(f'<div class="section-header">{icon} {text}</div>', unsafe_allow_html=True)


def great_circle_points(lat1, lon1, lat2, lon2, n=50):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    d = np.arccos(np.clip(
        np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1), -1, 1))
    if d < 1e-10:
        return [np.degrees(lat1)], [np.degrees(lon1)]
    fracs = np.linspace(0, 1, n)
    a = np.sin((1-fracs)*d) / np.sin(d)
    b = np.sin(fracs*d) / np.sin(d)
    x = a*np.cos(lat1)*np.cos(lon1) + b*np.cos(lat2)*np.cos(lon2)
    y = a*np.cos(lat1)*np.sin(lon1) + b*np.cos(lat2)*np.sin(lon2)
    z = a*np.sin(lat1) + b*np.sin(lat2)
    lats = np.degrees(np.arctan2(z, np.sqrt(x**2+y**2)))
    lons = np.degrees(np.arctan2(y, x))
    return lats.tolist(), lons.tolist()


def build_3d_aircraft():
    fig = go.Figure()
    n_x, n_theta = 80, 40
    x_fus = np.linspace(0, 40, n_x)
    theta = np.linspace(0, 2*np.pi, n_theta)
    X_fus, Theta = np.meshgrid(x_fus, theta)
    R_profile = np.where(
        x_fus < 5, 1.8 * np.sqrt(x_fus / 5),
        np.where(x_fus < 30, 1.8, 1.8 * (1 - ((x_fus - 30) / 10)**1.5)))
    R_profile = np.clip(R_profile, 0.05, 1.8)
    R_grid = np.outer(np.ones(n_theta), R_profile)
    Y_fus = R_grid * np.cos(Theta)
    Z_fus = R_grid * np.sin(Theta)
    color_num = np.full(X_fus.shape, 0.5)
    color_num[X_fus > 32] = 0.0
    color_num[X_fus < 2.5] = 0.15
    stripe_mask = (np.abs(Theta - np.pi/2) < 0.15) & (X_fus > 3) & (X_fus < 33)
    color_num[stripe_mask] = 1.0
    colorscale = [[0.0, AF_NAVY], [0.15, "#2C3E50"], [0.5, "#F0F2F5"], [1.0, AF_RED]]
    fig.add_trace(go.Surface(
        x=X_fus, y=Y_fus, z=Z_fus, surfacecolor=color_num,
        colorscale=colorscale, showscale=False, opacity=0.95,
        lighting=dict(ambient=0.5, diffuse=0.6, specular=0.3, roughness=0.4)))

    for y_sign in [1, -1]:
        fig.add_trace(go.Mesh3d(
            x=[12, 10, 22, 20], y=[v*y_sign for v in [1.8, 12, 12, 1.8]],
            z=[0, -0.09, -0.09, 0], i=[0,0], j=[1,2], k=[2,3],
            color="#D0D5DD", opacity=0.9, flatshading=True, showlegend=False))

    fig.add_trace(go.Mesh3d(
        x=[33, 35, 40, 40], y=[0,0,0,0], z=[1.8, 5.5, 4.5, 1.2],
        i=[0,0], j=[1,2], k=[2,3], color=AF_NAVY, opacity=0.95,
        flatshading=True, showlegend=False))

    for sign in [1, -1]:
        fig.add_trace(go.Mesh3d(
            x=[35, 34, 40, 39], y=[v*sign for v in [0, 4.5, 4.5, 0]],
            z=[1.5, 1.2, 1.2, 1.3], i=[0,0], j=[1,2], k=[2,3],
            color="#B0B8C4", opacity=0.9, flatshading=True, showlegend=False))

    for y_pos in [5.5, -5.5]:
        theta_e = np.linspace(0, 2*np.pi, 20)
        x_eng = np.linspace(13, 17, 10)
        Xe, Te = np.meshgrid(x_eng, theta_e)
        Ye = y_pos + 0.7 * np.cos(Te)
        Ze = -0.8 + 0.7 * np.sin(Te)
        fig.add_trace(go.Surface(
            x=Xe, y=Ye, z=Ze, colorscale=[[0,"#4A5568"],[1,"#718096"]],
            showscale=False, opacity=0.9, showlegend=False))

    fig.add_trace(go.Scatter3d(
        x=[38], y=[0.3], z=[3.5], mode="text", text=["AF"],
        textfont=dict(size=18, color="white", family="Arial Black"),
        showlegend=False))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-5,45]),
            yaxis=dict(visible=False, range=[-15,15]),
            zaxis=dict(visible=False, range=[-5,10]),
            aspectratio=dict(x=2.2, y=1.2, z=0.6),
            camera=dict(eye=dict(x=1.5, y=1.0, z=0.6), up=dict(x=0,y=0,z=1)),
            bgcolor="#F5F7FA"),
        margin=dict(l=0, r=0, t=30, b=0), height=550, paper_bgcolor="#F5F7FA",
        title=dict(text="Modèle 3D — Airbus A350-900 | Livrée Air France",
                   font=dict(size=14, color=AF_NAVY), x=0.5))
    return fig


# ==========================================================================
# SIDEBAR
# ==========================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0;">
        <div style="font-size:2.5rem;">✈️</div>
        <h2 style="font-size:1.3rem; margin:5px 0; font-weight:800;">AIR FRANCE</h2>
        <p style="font-size:0.8rem; opacity:0.7;">Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    pages = [
        "🏠 Vue d'ensemble",
        "🗺️ Réseau & Routes",
        "🛩️ Flotte & 3D",
        "📊 Opérations",
        "💰 Performance Financière",
        "⭐ Satisfaction Client",
        "🌿 Impact Environnemental",
    ]
    page = st.radio("Navigation", pages, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 🎛️ Filtres")

    annees_dispo = sorted(vols_df["annee"].unique())
    sel_annees = st.multiselect("Année(s)", annees_dispo, default=annees_dispo)

    courriers = sorted(vols_df["courrier"].unique())
    sel_courriers = st.multiselect("Type de courrier", courriers, default=courriers)

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; font-size:0.7rem; opacity:0.5;">'
        'Portfolio Data Analytics<br>Données simulées</p>',
        unsafe_allow_html=True)

# Filtres
mask = (vols_df["annee"].isin(sel_annees)) & (vols_df["courrier"].isin(sel_courriers))
vf = vols_df[mask].copy()
sf = satisfaction_df[satisfaction_df["date_vol"].dt.year.isin(sel_annees) &
                     satisfaction_df["courrier"].isin(sel_courriers)].copy()


# ==========================================================================
# PAGE 1 : VUE D'ENSEMBLE
# ==========================================================================
if page == "🏠 Vue d'ensemble":
    st.markdown("""
    <div class="af-banner">
        <h1>✈️ Air France — Tableau de Bord Analytique</h1>
        <p>Performance opérationnelle, financière et environnementale</p>
    </div>
    """, unsafe_allow_html=True)

    total_vols = len(vf)
    total_pax = vf["passagers"].sum()
    total_rev = vf["revenu_vol_eur"].sum()
    tx_ponctualite = (vf["statut_vol"] == "À l'heure").mean() * 100
    tx_remplissage = vf["taux_remplissage"].mean() * 100
    total_co2 = vf["co2_tonnes"].sum()

    # Ligne 1 : 3 KPIs
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        kpi("🛫", "Vols totaux", format_number(total_vols))
    with r1c2:
        kpi("👥", "Passagers transportés", format_number(total_pax))
    with r1c3:
        kpi("💰", "Revenus totaux", f"{format_number(total_rev)} €", "gold")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Ligne 2 : 3 KPIs
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        kpi("⏱️", "Taux de ponctualité", f"{tx_ponctualite:.1f}%", "green")
    with r2c2:
        kpi("💺", "Taux de remplissage", f"{tx_remplissage:.1f}%", "blue")
    with r2c3:
        kpi("🌿", "Émissions CO₂", f"{format_number(total_co2)} t", "red")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # --- Graphiques ---
    col1, col2 = st.columns(2)

    with col1:
        section_title("Évolution mensuelle des vols", "📈")
        monthly = vf.groupby(vf["date_vol"].dt.to_period("M")).agg(
            nb_vols=("vol_id", "count"),
            passagers=("passagers", "sum"),
        ).reset_index()
        monthly["date_vol"] = monthly["date_vol"].dt.to_timestamp()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=monthly["date_vol"], y=monthly["nb_vols"],
            name="Vols", marker_color=AF_NAVY, opacity=0.7,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=monthly["date_vol"], y=monthly["passagers"],
            name="Passagers", line=dict(color=AF_RED, width=3),
            mode="lines+markers", marker=dict(size=5),
        ), secondary_y=True)
        fig.update_layout(
            height=400, template="plotly_white",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            margin=dict(l=50, r=50, t=40, b=40),
            paper_bgcolor="white", plot_bgcolor="white")
        fig.update_yaxes(title_text="Nombre de vols", secondary_y=False)
        fig.update_yaxes(title_text="Passagers", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_title("Répartition par continent", "🌍")
        continent_stats = vf.groupby("continent_destination").agg(
            revenu=("revenu_vol_eur", "sum"),
        ).reset_index().sort_values("revenu", ascending=True)

        fig = go.Figure(go.Bar(
            y=continent_stats["continent_destination"],
            x=continent_stats["revenu"],
            orientation="h",
            marker_color=[AF_PALETTE[i % len(AF_PALETTE)] for i in range(len(continent_stats))],
            text=[f"{v/1e6:.1f}M €" for v in continent_stats["revenu"]],
            textposition="inside",
            textfont=dict(color="white", size=12),
        ))
        fig.update_layout(
            height=400, template="plotly_white",
            margin=dict(l=130, r=30, t=20, b=40),
            paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        section_title("Ponctualité par courrier", "⏱️")
        ponct = vf.groupby("courrier").apply(
            lambda x: pd.Series({
                "À l'heure": (x["statut_vol"] == "À l'heure").mean() * 100,
                "Retardé": (x["statut_vol"] == "Retardé").mean() * 100,
                "Annulé": (x["statut_vol"] == "Annulé").mean() * 100,
            })
        ).reset_index()

        fig = go.Figure()
        for i, (col_name, color) in enumerate([
            ("À l'heure", "#00A86B"), ("Retardé", "#F59E0B"), ("Annulé", AF_RED)
        ]):
            fig.add_trace(go.Bar(
                x=ponct["courrier"], y=ponct[col_name],
                name=col_name, marker_color=color,
                text=[f"{v:.1f}%" for v in ponct[col_name]],
                textposition="inside",
                textfont=dict(color="white", size=11),
            ))
        fig.update_layout(
            barmode="stack", height=400, template="plotly_white",
            yaxis_title="%",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            margin=dict(l=50, r=30, t=40, b=40),
            paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        section_title("Top 10 routes par revenu", "🏆")
        vf2 = vf.copy()
        vf2["route"] = vf2["aeroport_depart"] + " → " + vf2["aeroport_arrivee"]
        top = vf2.groupby("route")["revenu_vol_eur"].sum().nlargest(10).reset_index()
        top = top.sort_values("revenu_vol_eur")

        fig = go.Figure(go.Bar(
            y=top["route"], x=top["revenu_vol_eur"], orientation="h",
            marker=dict(color=top["revenu_vol_eur"],
                        colorscale=[[0, AF_LIGHT_BLUE], [1, AF_NAVY]]),
            text=[f"{v/1e6:.1f}M €" for v in top["revenu_vol_eur"]],
            textposition="inside",
            textfont=dict(color="white", size=12),
        ))
        fig.update_layout(
            height=400, template="plotly_white",
            margin=dict(l=130, r=30, t=20, b=40),
            paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # Tableau
    section_title("Résumé par type d'avion", "📋")
    summary = vf.groupby("type_avion").agg(
        nb_vols=("vol_id", "count"),
        passagers=("passagers", "sum"),
        tx_remplissage=("taux_remplissage", "mean"),
        revenu_total=("revenu_vol_eur", "sum"),
        retard_moyen=("retard_depart_min", "mean"),
        co2_total=("co2_tonnes", "sum"),
    ).reset_index()
    summary["tx_remplissage"] = (summary["tx_remplissage"] * 100).round(1)
    summary["retard_moyen"] = summary["retard_moyen"].round(1)
    summary.columns = ["Type avion", "Vols", "Passagers", "Remplissage (%)",
                        "Revenu (€)", "Retard moy (min)", "CO₂ (t)"]
    st.dataframe(summary.sort_values("Vols", ascending=False),
                 use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE 2 : RÉSEAU & ROUTES
# ==========================================================================
elif page == "🗺️ Réseau & Routes":
    st.markdown("""
    <div class="af-banner">
        <h1>🗺️ Réseau de Routes Air France</h1>
        <p>Cartographie interactive du réseau mondial</p>
    </div>
    """, unsafe_allow_html=True)

    n_dest = vf["aeroport_arrivee"].nunique()
    n_pays = aeroports_df[aeroports_df["code_iata"].isin(vf["aeroport_arrivee"].unique())]["pays"].nunique()
    n_cont = vf["continent_destination"].nunique()
    dist_tot = vf["distance_km"].sum()

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        kpi("📍", "Destinations", str(n_dest))
    with r2:
        kpi("🏳️", "Pays desservis", str(n_pays))
    with r3:
        kpi("🌍", "Continents", str(n_cont), "blue")
    with r4:
        kpi("📏", "Distance totale", f"{format_number(dist_tot)} km", "gold")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    section_title("Carte du réseau mondial", "🌐")

    routes_agg = vf.groupby(["aeroport_depart", "aeroport_arrivee"]).agg(
        nb_vols=("vol_id", "count"),
        passagers=("passagers", "sum"),
        revenu=("revenu_vol_eur", "sum"),
    ).reset_index()
    routes_agg = routes_agg.merge(
        aeroports_df[["code_iata","latitude","longitude","ville"]].rename(
            columns={"code_iata":"aeroport_depart","latitude":"lat_dep",
                     "longitude":"lon_dep","ville":"ville_dep"}), on="aeroport_depart"
    ).merge(
        aeroports_df[["code_iata","latitude","longitude","ville"]].rename(
            columns={"code_iata":"aeroport_arrivee","latitude":"lat_arr",
                     "longitude":"lon_arr","ville":"ville_arr"}), on="aeroport_arrivee")

    continent_color = {
        "Europe": AF_LIGHT_BLUE, "Amérique du Nord": AF_RED,
        "Amérique du Sud": AF_GOLD, "Afrique": "#00A86B",
        "Asie": "#8B5CF6", "Moyen-Orient": "#FF6B35",
        "Caraïbes": "#EC4899", "Océan Indien": "#14B8A6",
    }

    fig_map = go.Figure()
    for _, row in routes_agg.iterrows():
        lats, lons = great_circle_points(
            row["lat_dep"], row["lon_dep"], row["lat_arr"], row["lon_arr"], 40)
        cont = apt_dict.get(row["aeroport_arrivee"], {}).get("continent", "Europe")
        width = max(1, min(4, row["nb_vols"] / routes_agg["nb_vols"].max() * 4))
        fig_map.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines",
            line=dict(width=width, color=continent_color.get(cont, AF_GREY)),
            opacity=0.45, hoverinfo="text",
            text=f"{row['ville_dep']} → {row['ville_arr']}<br>"
                 f"Vols: {row['nb_vols']:,} | Pax: {row['passagers']:,}<br>"
                 f"Revenu: {row['revenu']/1e6:.1f}M €",
            showlegend=False))

    apt_traffic = pd.concat([
        vf.groupby("aeroport_depart")["passagers"].sum(),
        vf.groupby("aeroport_arrivee")["passagers"].sum(),
    ]).groupby(level=0).sum().reset_index()
    apt_traffic.columns = ["code_iata", "total_pax"]
    apt_traffic = apt_traffic.merge(aeroports_df, on="code_iata")
    apt_traffic["size"] = np.clip(apt_traffic["total_pax"]/apt_traffic["total_pax"].max()*25, 4, 25)
    apt_traffic["color"] = apt_traffic["type_aeroport"].map({
        "Hub": AF_RED, "Domestique": AF_NAVY, "Européen": AF_LIGHT_BLUE,
        "Long-courrier": AF_GOLD, "Moyen-courrier": "#00A86B"}).fillna(AF_GREY)

    fig_map.add_trace(go.Scattergeo(
        lat=apt_traffic["latitude"], lon=apt_traffic["longitude"],
        mode="markers+text",
        marker=dict(size=apt_traffic["size"], color=apt_traffic["color"],
                    line=dict(width=1, color="white"), opacity=0.9),
        text=apt_traffic["code_iata"], textposition="top center",
        textfont=dict(size=8, color=AF_NAVY),
        hoverinfo="text",
        hovertext=apt_traffic.apply(
            lambda r: f"<b>{r['nom']}</b><br>{r['ville']}, {r['pays']}<br>"
                      f"Trafic: {r['total_pax']:,} pax", axis=1),
        showlegend=False))

    fig_map.update_geos(
        projection_type="natural earth",
        showcoastlines=True, coastlinecolor="#B0BEC5",
        showland=True, landcolor="#F5F5F0",
        showocean=True, oceancolor="#E8F0FE",
        showlakes=True, lakecolor="#E8F0FE",
        showcountries=True, countrycolor="#D5DBDB",
        showframe=False, lonaxis_range=[-130, 160], lataxis_range=[-40, 65])
    fig_map.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=AF_BG)
    st.plotly_chart(fig_map, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        section_title("Top 15 destinations", "🏆")
        top_dest = vf.groupby("aeroport_arrivee")["passagers"].sum().nlargest(15).reset_index()
        top_dest = top_dest.merge(aeroports_df[["code_iata","ville"]],
                                   left_on="aeroport_arrivee", right_on="code_iata")
        top_dest["label"] = top_dest["ville"] + " (" + top_dest["aeroport_arrivee"] + ")"
        top_dest = top_dest.sort_values("passagers")

        fig = go.Figure(go.Bar(
            y=top_dest["label"], x=top_dest["passagers"], orientation="h",
            marker=dict(color=top_dest["passagers"],
                        colorscale=[[0, AF_LIGHT_BLUE], [0.5, AF_NAVY], [1, AF_RED]]),
            text=[format_number(v) for v in top_dest["passagers"]],
            textposition="inside", textfont=dict(color="white", size=11),
        ))
        fig.update_layout(height=520, template="plotly_white",
                          margin=dict(l=170, r=30, t=10, b=40),
                          paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_title("Répartition des distances", "📏")
        fig = px.histogram(
            vf, x="distance_km", nbins=50, color="courrier",
            color_discrete_map={"Court-courrier": AF_LIGHT_BLUE,
                                "Moyen-courrier": AF_GOLD, "Long-courrier": AF_NAVY},
            labels={"distance_km": "Distance (km)"})
        fig.update_layout(height=520, template="plotly_white",
                          legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                          margin=dict(l=50, r=30, t=30, b=40),
                          paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# PAGE 3 : FLOTTE & 3D
# ==========================================================================
elif page == "🛩️ Flotte & 3D":
    st.markdown("""
    <div class="af-banner">
        <h1>🛩️ Flotte Air France & Modèle 3D</h1>
        <p>Composition de la flotte et visualisation interactive</p>
    </div>
    """, unsafe_allow_html=True)

    section_title("Modèle 3D interactif", "✨")
    st.markdown('<p style="color:#6B7B8D; font-size:0.9rem;">'
                '🖱️ Cliquez et glissez pour pivoter • Molette pour zoomer</p>',
                unsafe_allow_html=True)
    st.plotly_chart(build_3d_aircraft(), use_container_width=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        kpi("🛩️", "Avions", str(len(flotte_df)))
    with r2:
        kpi("📋", "Types", str(flotte_df["type_avion"].nunique()))
    with r3:
        kpi("📅", "Âge moyen", f"{flotte_df['age_avion_ans'].mean():.1f} ans", "blue")
    with r4:
        kpi("✅", "En service", str((flotte_df["statut"]=="En service").sum()), "green")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_title("Composition de la flotte", "📊")
        fc = flotte_df.groupby("type_avion").size().reset_index(name="count").sort_values("count", ascending=False)
        fig = go.Figure(go.Bar(
            x=fc["type_avion"], y=fc["count"],
            marker_color=[AF_PALETTE[i%len(AF_PALETTE)] for i in range(len(fc))],
            text=fc["count"], textposition="outside",
            textfont=dict(size=13, color=AF_NAVY)))
        fig.update_layout(height=420, template="plotly_white", xaxis_tickangle=-35,
                          margin=dict(l=50, r=30, t=20, b=110),
                          paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_title("Narrow vs Wide body", "✈️")
        fs = flotte_df.groupby("famille").size().reset_index(name="count")
        fig = go.Figure(go.Pie(
            labels=fs["famille"], values=fs["count"], hole=0.55,
            marker_colors=[AF_NAVY, AF_RED],
            textinfo="label+percent+value",
            textfont=dict(size=13, color="white")))
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20),
                          paper_bgcolor="white",
                          annotations=[dict(text="Flotte", x=0.5, y=0.5,
                                            font_size=18, font_color=AF_NAVY, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    section_title("Distribution de l'âge", "📅")
    fig = px.histogram(flotte_df, x="age_avion_ans", nbins=25, color="famille",
                       color_discrete_map={"Narrow-body": AF_LIGHT_BLUE, "Wide-body": AF_NAVY},
                       labels={"age_avion_ans": "Âge (années)"})
    fig.update_layout(height=350, template="plotly_white",
                      legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                      margin=dict(l=50, r=30, t=30, b=40),
                      paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    section_title("Détail par appareil", "📋")
    fs2 = flotte_df.groupby("type_avion").agg(
        nb=("immatriculation","count"), sieges=("nb_sieges_total","mean"),
        age=("age_avion_ans","mean"), biz=("nb_sieges_business","first"),
        prem=("nb_sieges_premium_eco","first"), eco=("nb_sieges_eco","first"),
        auto=("autonomie_km","first")).reset_index()
    fs2.columns = ["Type","Avions","Sièges","Âge moy","Business","Premium","Éco","Autonomie km"]
    fs2["Âge moy"] = fs2["Âge moy"].round(1)
    fs2["Sièges"] = fs2["Sièges"].round(0).astype(int)
    st.dataframe(fs2.sort_values("Avions", ascending=False),
                 use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE 4 : OPÉRATIONS
# ==========================================================================
elif page == "📊 Opérations":
    st.markdown("""
    <div class="af-banner">
        <h1>📊 Performance Opérationnelle</h1>
        <p>Ponctualité, retards et taux de remplissage</p>
    </div>
    """, unsafe_allow_html=True)

    retard_moy = vf[vf["retard_depart_min"] > 0]["retard_depart_min"].mean()
    tx_annul = (vf["statut_vol"] == "Annulé").mean() * 100
    otp15 = (vf["retard_depart_min"] <= 15).mean() * 100
    rpk = vf["rpk"].sum()
    ask = vf["ask"].sum()
    lf = rpk / ask * 100 if ask > 0 else 0

    r1, r2, r3 = st.columns(3)
    with r1:
        kpi("⏱️", "OTP (15 min)", f"{otp15:.1f}%", "green")
    with r2:
        kpi("⏳", "Retard moyen", f"{retard_moy:.0f} min", "red")
    with r3:
        kpi("❌", "Taux annulation", f"{tx_annul:.2f}%", "red")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    r4, r5, r6 = st.columns(3)
    with r4:
        kpi("📏", "RPK total", format_number(rpk), "blue")
    with r5:
        kpi("💺", "ASK total", format_number(ask))
    with r6:
        kpi("📊", "Load Factor", f"{lf:.1f}%", "green")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_title("Causes de retard", "🔍")
        causes = vf[vf["cause_retard"]!="Aucun"]["cause_retard"].value_counts().reset_index()
        causes.columns = ["Cause", "Nombre"]
        fig = go.Figure(go.Pie(
            labels=causes["Cause"], values=causes["Nombre"],
            hole=0.5, marker_colors=AF_PALETTE[:len(causes)],
            textinfo="label+percent",
            textfont=dict(color="white", size=12)))
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20),
                          paper_bgcolor="white",
                          annotations=[dict(text="Retards", x=0.5, y=0.5,
                                            font_size=16, font_color=AF_NAVY, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_title("Distribution des retards", "📊")
        fig = px.histogram(
            vf[vf["retard_depart_min"]>0], x="retard_depart_min", nbins=60,
            color="courrier",
            color_discrete_map={"Court-courrier": AF_LIGHT_BLUE,
                                "Moyen-courrier": AF_GOLD, "Long-courrier": AF_NAVY},
            labels={"retard_depart_min": "Retard (min)"})
        fig.update_layout(height=420, template="plotly_white", xaxis_range=[0,200],
                          legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                          margin=dict(l=50, r=30, t=30, b=40),
                          paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    section_title("Ponctualité mensuelle (OTP 15 min)", "📈")
    m_otp = vf.groupby(vf["date_vol"].dt.to_period("M")).apply(
        lambda x: (x["retard_depart_min"] <= 15).mean() * 100).reset_index()
    m_otp.columns = ["mois", "otp"]
    m_otp["mois"] = m_otp["mois"].dt.to_timestamp()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=m_otp["mois"], y=m_otp["otp"], mode="lines+markers",
        line=dict(color=AF_NAVY, width=3), marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(0,33,87,0.08)"))
    fig.add_hline(y=80, line_dash="dash", line_color=AF_RED,
                  annotation_text="Objectif 80%")
    fig.update_layout(height=350, template="plotly_white",
                      yaxis_title="OTP (%)", yaxis_range=[55,100],
                      margin=dict(l=50, r=30, t=20, b=40),
                      paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    section_title("Heatmap retards : jour × heure", "🗓️")
    vf_h = vf.copy()
    vf_h["jour"] = vf_h["date_vol"].dt.day_name()
    vf_h["heure"] = vf_h["heure_depart"].str[:2].astype(int)
    hp = vf_h.groupby(["jour","heure"])["retard_depart_min"].mean().reset_index()
    hp = hp.pivot(index="jour", columns="heure", values="retard_depart_min")
    days_en = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    days_fr = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
    hp = hp.reindex(days_en)

    fig = go.Figure(go.Heatmap(
        z=hp.values, x=[f"{h}h" for h in hp.columns], y=days_fr,
        colorscale=[[0,"#E8F0FE"],[0.5,AF_LIGHT_BLUE],[1,AF_RED]],
        colorbar=dict(title="Min")))
    fig.update_layout(height=350, template="plotly_white",
                      margin=dict(l=100, r=30, t=20, b=40),
                      paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        section_title("Remplissage mensuel", "💺")
        mlr = vf.groupby([vf["date_vol"].dt.to_period("M"), "courrier"])[
            "taux_remplissage"].mean().reset_index()
        mlr["date_vol"] = mlr["date_vol"].dt.to_timestamp()
        mlr["taux_remplissage"] *= 100
        fig = px.line(mlr, x="date_vol", y="taux_remplissage", color="courrier",
                      color_discrete_map={"Court-courrier": AF_LIGHT_BLUE,
                                          "Moyen-courrier": AF_GOLD, "Long-courrier": AF_NAVY},
                      labels={"taux_remplissage": "%", "date_vol": ""})
        fig.update_layout(height=380, template="plotly_white",
                          legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                          margin=dict(l=50, r=30, t=30, b=40),
                          paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_title("Remplissage par avion", "📊")
        la = vf.groupby("type_avion")["taux_remplissage"].mean().reset_index()
        la["taux_remplissage"] *= 100
        la = la.sort_values("taux_remplissage")
        fig = go.Figure(go.Bar(
            y=la["type_avion"], x=la["taux_remplissage"], orientation="h",
            marker_color=AF_NAVY,
            text=[f"{v:.1f}%" for v in la["taux_remplissage"]],
            textposition="inside", textfont=dict(color="white", size=12)))
        fig.add_vline(x=80, line_dash="dash", line_color=AF_RED)
        fig.update_layout(height=380, template="plotly_white",
                          xaxis_range=[60,100],
                          margin=dict(l=160, r=30, t=20, b=40),
                          paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# PAGE 5 : PERFORMANCE FINANCIÈRE
# ==========================================================================
elif page == "💰 Performance Financière":
    st.markdown("""
    <div class="af-banner">
        <h1>💰 Performance Financière</h1>
        <p>Revenus, rendement, RASK et indicateurs financiers</p>
    </div>
    """, unsafe_allow_html=True)

    rev = vf["revenu_vol_eur"].sum()
    rpv = vf["revenu_vol_eur"].mean()
    rpp = vf["revenu_par_pax_eur"].mean()
    yld = rev / vf["rpk"].sum() * 100 if vf["rpk"].sum() > 0 else 0
    rask = rev / vf["ask"].sum() * 100 if vf["ask"].sum() > 0 else 0

    r1, r2, r3 = st.columns(3)
    with r1:
        kpi("💰", "Revenu total", f"{format_number(rev)} €", "gold")
    with r2:
        kpi("🛫", "Revenu / vol", f"{rpv:,.0f} €")
    with r3:
        kpi("👤", "Revenu / pax", f"{rpp:,.0f} €")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    r4, r5 = st.columns(2)
    with r4:
        kpi("📈", "Yield", f"{yld:.2f} c€/RPK", "green")
    with r5:
        kpi("📊", "RASK", f"{rask:.2f} c€/ASK", "blue")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_title("Revenus mensuels", "📈")
        mr = vf.groupby(vf["date_vol"].dt.to_period("M")).agg(
            revenu=("revenu_vol_eur","sum"), vols=("vol_id","count")).reset_index()
        mr["date_vol"] = mr["date_vol"].dt.to_timestamp()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=mr["date_vol"], y=mr["revenu"],
                             name="Revenu (€)", marker_color=AF_NAVY, opacity=0.7),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=mr["date_vol"], y=mr["revenu"]/mr["vols"],
                                 name="Rev/vol (€)", line=dict(color=AF_RED, width=3),
                                 mode="lines+markers", marker=dict(size=5)),
                      secondary_y=True)
        fig.update_layout(height=420, template="plotly_white",
                          legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                          margin=dict(l=60, r=60, t=40, b=40),
                          paper_bgcolor="white", plot_bgcolor="white")
        fig.update_yaxes(title_text="Revenu total (€)", secondary_y=False)
        fig.update_yaxes(title_text="Rev/vol (€)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_title("Revenus par continent", "🌍")
        rc = vf.groupby("continent_destination")["revenu_vol_eur"].sum().reset_index()
        rc = rc.sort_values("revenu_vol_eur", ascending=False)
        fig = go.Figure(go.Treemap(
            labels=rc["continent_destination"], parents=[""]*len(rc),
            values=rc["revenu_vol_eur"],
            textinfo="label+value+percent root",
            texttemplate="<b>%{label}</b><br>%{value:,.0f} €<br>%{percentRoot:.1%}",
            textfont=dict(color="white", size=13),
            marker_colors=AF_PALETTE[:len(rc)]))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    section_title("Yield et RASK par courrier", "📊")
    yc = vf.groupby("courrier").apply(lambda x: pd.Series({
        "Yield (c€/RPK)": x["revenu_vol_eur"].sum()/x["rpk"].sum()*100 if x["rpk"].sum()>0 else 0,
        "RASK (c€/ASK)": x["revenu_vol_eur"].sum()/x["ask"].sum()*100 if x["ask"].sum()>0 else 0,
        "Rev/pax (€)": x["revenu_par_pax_eur"].mean(),
    })).reset_index()

    c1, c2, c3 = st.columns(3)
    for i, metric in enumerate(["Yield (c€/RPK)", "RASK (c€/ASK)", "Rev/pax (€)"]):
        with [c1, c2, c3][i]:
            fig = go.Figure(go.Bar(
                x=yc["courrier"], y=yc[metric],
                marker_color=[AF_LIGHT_BLUE, AF_GOLD, AF_NAVY][:len(yc)],
                text=[f"{v:.2f}" for v in yc[metric]],
                textposition="outside", textfont=dict(color=AF_NAVY, size=13)))
            fig.update_layout(height=360, template="plotly_white",
                              title=dict(text=metric, font=dict(size=14, color=AF_NAVY)),
                              margin=dict(l=50, r=30, t=50, b=40),
                              paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    section_title("Comparatif annuel", "📅")
    yr = vf.groupby("annee").agg(
        vols=("vol_id","count"), passagers=("passagers","sum"),
        revenu=("revenu_vol_eur","sum"), remplissage=("taux_remplissage","mean"),
        co2=("co2_tonnes","sum")).reset_index()
    yr["remplissage"] = (yr["remplissage"]*100).round(1)
    yr.columns = ["Année","Vols","Passagers","Revenu (€)","Remplissage (%)","CO₂ (t)"]
    st.dataframe(yr, use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE 6 : SATISFACTION CLIENT
# ==========================================================================
elif page == "⭐ Satisfaction Client":
    st.markdown("""
    <div class="af-banner">
        <h1>⭐ Satisfaction Client</h1>
        <p>Enquêtes passagers, NPS et axes d'amélioration</p>
    </div>
    """, unsafe_allow_html=True)

    note_moy = sf["note_globale"].mean()
    nps_p = (sf["nps_categorie"]=="Promoteur").mean()*100
    nps_d = (sf["nps_categorie"]=="Détracteur").mean()*100
    nps = nps_p - nps_d
    rec = (sf["recommandation"]=="Oui").mean()*100

    r1, r2, r3 = st.columns(3)
    with r1:
        kpi("⭐", "Note globale", f"{note_moy:.1f} / 10", "gold")
    with r2:
        kpi("📊", "Score NPS", f"{nps:.0f}", "blue")
    with r3:
        kpi("😊", "Promoteurs", f"{nps_p:.1f}%", "green")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    r4, r5, r6 = st.columns(3)
    with r4:
        kpi("😞", "Détracteurs", f"{nps_d:.1f}%", "red")
    with r5:
        kpi("👍", "Recommandation", f"{rec:.1f}%", "green")
    with r6:
        kpi("📝", "Enquêtes", format_number(len(sf)))

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    dims = ["note_confort","note_repas","note_divertissement",
            "note_equipage","note_ponctualite","note_enregistrement"]
    labels_dim = ["Confort","Repas","Divertissement","Équipage","Ponctualité","Enregistrement"]

    col1, col2 = st.columns(2)

    with col1:
        section_title("Notes par dimension", "📊")
        vals = [sf[d].mean() for d in dims]
        fig = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=labels_dim+[labels_dim[0]],
            fill="toself", fillcolor="rgba(0,33,87,0.15)",
            line=dict(color=AF_NAVY, width=3), marker=dict(size=8, color=AF_NAVY)))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,10], ticksuffix="/10")),
            height=450, margin=dict(l=60, r=60, t=40, b=40),
            paper_bgcolor="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_title("Répartition NPS", "📈")
        nd = sf["nps_categorie"].value_counts().reset_index()
        nd.columns = ["Cat","Nb"]
        cm = {"Promoteur":"#00A86B","Passif":AF_GOLD,"Détracteur":AF_RED}
        fig = go.Figure(go.Pie(
            labels=nd["Cat"], values=nd["Nb"], hole=0.55,
            marker_colors=[cm.get(c, AF_GREY) for c in nd["Cat"]],
            textinfo="label+percent",
            textfont=dict(size=14, color="white")))
        fig.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20),
                          paper_bgcolor="white",
                          annotations=[dict(text=f"NPS<br><b>{nps:.0f}</b>", x=0.5, y=0.5,
                                            font_size=22, font_color=AF_NAVY, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    section_title("Satisfaction par classe de voyage", "💎")
    sc = sf.groupby("classe")[dims].mean().reset_index()
    ccls = {"Business": AF_NAVY, "Premium Éco": AF_GOLD, "Économique": AF_LIGHT_BLUE}
    fig = go.Figure()
    for _, row in sc.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[d] for d in dims]+[row[dims[0]]],
            theta=labels_dim+[labels_dim[0]],
            name=row["classe"],
            line=dict(color=ccls.get(row["classe"], AF_GREY), width=2),
            fill="toself", opacity=0.4))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,10])),
        height=420, legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        margin=dict(l=60, r=60, t=20, b=60), paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        section_title("Par type de voyageur", "🧳")
        st2 = sf.groupby("type_voyageur")["note_globale"].mean().reset_index().sort_values("note_globale")
        fig = go.Figure(go.Bar(
            y=st2["type_voyageur"], x=st2["note_globale"], orientation="h",
            marker_color=AF_NAVY,
            text=[f"{v:.1f}/10" for v in st2["note_globale"]],
            textposition="inside", textfont=dict(color="white", size=12)))
        fig.update_layout(height=360, template="plotly_white",
                          xaxis_range=[5,10],
                          margin=dict(l=100, r=30, t=20, b=40),
                          paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_title("Par programme fidélité", "🏅")
        fo = ["Non inscrit","Explorer","Silver","Gold","Platinum","Ultimate"]
        sf2 = sf.groupby("programme_fidelite")["note_globale"].mean().reset_index()
        sf2["programme_fidelite"] = pd.Categorical(sf2["programme_fidelite"], categories=fo, ordered=True)
        sf2 = sf2.sort_values("programme_fidelite")
        fig = go.Figure(go.Bar(
            x=sf2["programme_fidelite"], y=sf2["note_globale"],
            marker=dict(color=sf2["note_globale"],
                        colorscale=[[0,AF_LIGHT_BLUE],[1,AF_GOLD]]),
            text=[f"{v:.1f}" for v in sf2["note_globale"]],
            textposition="outside", textfont=dict(color=AF_NAVY, size=13)))
        fig.update_layout(height=360, template="plotly_white",
                          yaxis_range=[5,10],
                          margin=dict(l=50, r=30, t=20, b=40),
                          paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    section_title("Évolution mensuelle", "📈")
    ms = sf.groupby(sf["date_vol"].dt.to_period("M")).agg(
        glob=("note_globale","mean"), equip=("note_equipage","mean"),
        ponct=("note_ponctualite","mean")).reset_index()
    ms["date_vol"] = ms["date_vol"].dt.to_timestamp()
    fig = go.Figure()
    for col_name, label, color in [("glob","Globale",AF_NAVY),
                                    ("equip","Équipage","#00A86B"),
                                    ("ponct","Ponctualité",AF_RED)]:
        fig.add_trace(go.Scatter(
            x=ms["date_vol"], y=ms[col_name], mode="lines+markers",
            name=label, line=dict(color=color, width=2), marker=dict(size=4)))
    fig.update_layout(height=380, template="plotly_white",
                      yaxis_title="Note (/10)", yaxis_range=[5,10],
                      legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                      margin=dict(l=50, r=30, t=30, b=40),
                      paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# PAGE 7 : IMPACT ENVIRONNEMENTAL
# ==========================================================================
elif page == "🌿 Impact Environnemental":
    st.markdown("""
    <div class="af-banner">
        <h1>🌿 Impact Environnemental</h1>
        <p>Émissions CO₂, consommation carburant et efficacité</p>
    </div>
    """, unsafe_allow_html=True)

    co2t = vf["co2_tonnes"].sum()
    fuelt = vf["carburant_litres"].sum()
    co2pk = co2t * 1e6 / vf["rpk"].sum() if vf["rpk"].sum() > 0 else 0
    co2v = vf["co2_tonnes"].mean()
    fuelv = vf["carburant_litres"].mean()

    r1, r2, r3 = st.columns(3)
    with r1:
        kpi("🏭", "CO₂ total", f"{format_number(co2t)} t", "red")
    with r2:
        kpi("⛽", "Carburant total", f"{format_number(fuelt)} L")
    with r3:
        kpi("📏", "CO₂ / pax·km", f"{co2pk:.1f} g", "green")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    r4, r5 = st.columns(2)
    with r4:
        kpi("🛫", "CO₂ moyen / vol", f"{co2v:.1f} t")
    with r5:
        kpi("⛽", "Carburant / vol", f"{format_number(fuelv)} L", "blue")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_title("CO₂ mensuelles", "📈")
        mc = vf.groupby(vf["date_vol"].dt.to_period("M")).agg(
            co2=("co2_tonnes","sum"), vols=("vol_id","count")).reset_index()
        mc["date_vol"] = mc["date_vol"].dt.to_timestamp()
        mc["cpv"] = mc["co2"]/mc["vols"]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=mc["date_vol"], y=mc["co2"],
                             name="CO₂ total (t)", marker_color="#00A86B", opacity=0.7),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=mc["date_vol"], y=mc["cpv"],
                                 name="CO₂/vol (t)", line=dict(color=AF_NAVY, width=3),
                                 mode="lines+markers", marker=dict(size=5)),
                      secondary_y=True)
        fig.update_layout(height=420, template="plotly_white",
                          legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                          margin=dict(l=60, r=60, t=40, b=40),
                          paper_bgcolor="white", plot_bgcolor="white")
        fig.update_yaxes(title_text="CO₂ total (t)", secondary_y=False)
        fig.update_yaxes(title_text="CO₂/vol (t)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_title("CO₂ par courrier", "🌍")
        cc = vf.groupby("courrier").agg(co2=("co2_tonnes","sum"), rpk=("rpk","sum")).reset_index()
        cc["gpr"] = cc["co2"]*1e6/cc["rpk"]

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("CO₂ total","CO₂/pax·km (g)"),
                            specs=[[{"type":"pie"},{"type":"bar"}]])
        fig.add_trace(go.Pie(labels=cc["courrier"], values=cc["co2"],
                             marker_colors=[AF_LIGHT_BLUE, AF_GOLD, AF_NAVY],
                             hole=0.45, textinfo="label+percent",
                             textfont=dict(color="white", size=12)), row=1, col=1)
        fig.add_trace(go.Bar(x=cc["courrier"], y=cc["gpr"],
                             marker_color=[AF_LIGHT_BLUE, AF_GOLD, AF_NAVY],
                             text=[f"{v:.1f}g" for v in cc["gpr"]],
                             textposition="outside",
                             textfont=dict(color=AF_NAVY, size=13)), row=1, col=2)
        fig.update_layout(height=420, template="plotly_white", showlegend=False,
                          margin=dict(l=40, r=40, t=50, b=40), paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    section_title("Efficacité carbone par avion", "✈️")
    eff = vf.groupby("type_avion").apply(lambda x: pd.Series({
        "gpr": x["co2_tonnes"].sum()*1e6/x["rpk"].sum() if x["rpk"].sum()>0 else 0
    })).reset_index().sort_values("gpr")

    fig = go.Figure(go.Bar(
        y=eff["type_avion"], x=eff["gpr"], orientation="h",
        marker=dict(color=eff["gpr"],
                    colorscale=[[0,"#00A86B"],[0.5,AF_GOLD],[1,AF_RED]],
                    colorbar=dict(title="g/pax·km")),
        text=[f"{v:.1f}g" for v in eff["gpr"]],
        textposition="inside", textfont=dict(color="white", size=12)))
    fig.update_layout(height=420, template="plotly_white",
                      xaxis_title="CO₂ par passager-km (g)",
                      margin=dict(l=160, r=40, t=20, b=40),
                      paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    section_title("Distance vs CO₂", "📊")
    samp = vf.sample(min(5000, len(vf)), random_state=42)
    fig = px.scatter(samp, x="distance_km", y="co2_tonnes",
                     color="courrier", size="passagers",
                     color_discrete_map={"Court-courrier": AF_LIGHT_BLUE,
                                         "Moyen-courrier": AF_GOLD, "Long-courrier": AF_NAVY},
                     labels={"distance_km":"Distance (km)","co2_tonnes":"CO₂ (t)"},
                     opacity=0.5, size_max=15)
    fig.update_layout(height=450, template="plotly_white",
                      legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                      margin=dict(l=60, r=30, t=30, b=50),
                      paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    section_title("Empreinte carbone annuelle", "📅")
    ye = vf.groupby("annee").apply(lambda x: pd.Series({
        "kt": x["co2_tonnes"].sum()/1000,
        "gpr": x["co2_tonnes"].sum()*1e6/x["rpk"].sum() if x["rpk"].sum()>0 else 0
    })).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=ye["annee"].astype(str), y=ye["kt"], marker_color="#00A86B",
            text=[f"{v:.1f} kt" for v in ye["kt"]],
            textposition="outside", textfont=dict(color=AF_NAVY, size=13)))
        fig.update_layout(height=360, template="plotly_white",
                          title=dict(text="CO₂ total (kt)", font=dict(size=14, color=AF_NAVY)),
                          margin=dict(l=50, r=30, t=50, b=40),
                          paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure(go.Bar(
            x=ye["annee"].astype(str), y=ye["gpr"], marker_color=AF_NAVY,
            text=[f"{v:.1f} g" for v in ye["gpr"]],
            textposition="outside", textfont=dict(color=AF_NAVY, size=13)))
        fig.update_layout(height=360, template="plotly_white",
                          title=dict(text="Intensité (g CO₂/pax·km)",
                                     font=dict(size=14, color=AF_NAVY)),
                          margin=dict(l=50, r=30, t=50, b=40),
                          paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)


# ==========================================================================
# FOOTER
# ==========================================================================
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center; padding:20px; color:{AF_GREY}; font-size:0.8rem;">
        <strong style="color:{AF_NAVY};">Air France Analytics Platform</strong>
        — Projet Portfolio Data Analytics<br>
        📊 {len(vols_df):,} vols • {vols_df['passagers'].sum():,} passagers •
        {aeroports_df['code_iata'].nunique()} aéroports • {len(flotte_df)} avions<br>
        <em>Données simulées à des fins de démonstration</em>
    </div>
    """,
    unsafe_allow_html=True)
