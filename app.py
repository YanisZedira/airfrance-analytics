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
# COULEURS AIR FRANCE
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
# CSS CUSTOM
# ==========================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #F5F7FA;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #002157 0%, #003580 100%);
        color: white;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: white !important;
    }

    /* KPI Cards */
    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0,33,87,0.08);
        border-left: 5px solid #002157;
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,33,87,0.14);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #002157;
        line-height: 1.1;
        margin: 4px 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #6B7B8D;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-delta {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .kpi-delta.positive { color: #00A86B; }
    .kpi-delta.negative { color: #E4002B; }

    /* Section header */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #002157;
        margin: 30px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #E4002B;
        display: inline-block;
    }

    /* Header banner */
    .af-banner {
        background: linear-gradient(135deg, #002157 0%, #003580 60%, #0055A0 100%);
        border-radius: 18px;
        padding: 30px 40px;
        color: white;
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
    }
    .af-banner::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: #E4002B;
    }
    .af-banner h1 {
        font-size: 2.2rem;
        font-weight: 900;
        margin: 0;
        color: white !important;
    }
    .af-banner p {
        font-size: 1.05rem;
        opacity: 0.85;
        margin: 5px 0 0 0;
        color: white !important;
    }

    /* Hide default streamlit header */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Plotly chart container */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,33,87,0.06);
        margin-bottom: 20px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,33,87,0.06);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# CHARGEMENT DES DONNÉES
# ==========================================================================
@st.cache_data(ttl=3600)
def load_data():
    vols = pd.read_csv("data/vols.csv", sep=";", parse_dates=["date_vol"])
    aeroports = pd.read_csv("data/aeroports.csv", sep=";")
    flotte = pd.read_csv("data/flotte.csv", sep=";")
    satisfaction = pd.read_csv("data/satisfaction.csv", sep=";", parse_dates=["date_vol"])
    return vols, aeroports, flotte, satisfaction


try:
    vols_df, aeroports_df, flotte_df, satisfaction_df = load_data()
except FileNotFoundError:
    st.error("⚠️ Fichiers de données introuvables. Exécutez d'abord `python generate_dataset.py`.")
    st.stop()

apt_dict = aeroports_df.set_index("code_iata").to_dict("index")


# ==========================================================================
# FONCTIONS UTILITAIRES
# ==========================================================================
def format_number(n, decimals=0):
    if abs(n) >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f} Mrd"
    elif abs(n) >= 1_000_000:
        return f"{n/1_000_000:.1f} M"
    elif abs(n) >= 1_000:
        return f"{n/1_000:.{decimals}f} k"
    return f"{n:,.{decimals}f}"


def kpi_card(label, value, delta=None, delta_dir="positive", icon=""):
    delta_html = ""
    if delta:
        cls = "positive" if delta_dir == "positive" else "negative"
        arrow = "▲" if delta_dir == "positive" else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{icon} {label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def section_title(text, icon=""):
    st.markdown(f'<div class="section-header">{icon} {text}</div>', unsafe_allow_html=True)


def great_circle_points(lat1, lon1, lat2, lon2, n=50):
    """Calcule n points sur un arc de grand cercle."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    d = np.arccos(np.clip(
        np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1), -1, 1
    ))
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
    """Construit un modèle 3D stylisé d'un avion Air France."""
    fig = go.Figure()

    # --- Fuselage (surface de révolution) ---
    n_x, n_theta = 80, 40
    x_fus = np.linspace(0, 40, n_x)
    theta = np.linspace(0, 2*np.pi, n_theta)
    X_fus, Theta = np.meshgrid(x_fus, theta)

    # Profil de rayon : nez effilé, corps cylindrique, queue effilée
    R_profile = np.where(
        x_fus < 5,
        1.8 * np.sqrt(x_fus / 5),
        np.where(x_fus < 30, 1.8, 1.8 * (1 - ((x_fus - 30) / 10)**1.5))
    )
    R_profile = np.clip(R_profile, 0.05, 1.8)
    R_grid = np.outer(np.ones(n_theta), R_profile)

    Y_fus = R_grid * np.cos(Theta)
    Z_fus = R_grid * np.sin(Theta)

    # Couleurs du fuselage
    colors = np.full(X_fus.shape, AF_WHITE, dtype=object)
    # Queue en bleu marine
    colors[X_fus > 32] = AF_NAVY
    # Nez en gris foncé (cockpit)
    colors[X_fus < 2.5] = "#2C3E50"
    # Bande rouge (stripe latérale)
    stripe_mask = (np.abs(Theta - np.pi/2) < 0.15) & (X_fus > 3) & (X_fus < 33)
    colors[stripe_mask] = AF_RED

    # Convertir en valeurs numériques pour colorscale
    color_num = np.zeros(X_fus.shape)
    color_num[colors == AF_WHITE] = 0.5
    color_num[colors == AF_NAVY] = 0.0
    color_num[colors == "#2C3E50"] = 0.15
    color_num[colors == AF_RED] = 1.0

    colorscale = [
        [0.0, AF_NAVY],
        [0.15, "#2C3E50"],
        [0.5, "#F0F2F5"],
        [1.0, AF_RED],
    ]

    fig.add_trace(go.Surface(
        x=X_fus, y=Y_fus, z=Z_fus,
        surfacecolor=color_num,
        colorscale=colorscale,
        showscale=False,
        opacity=0.95,
        name="Fuselage",
        lighting=dict(ambient=0.5, diffuse=0.6, specular=0.3, roughness=0.4),
    ))

    # --- Ailes ---
    def add_wing(y_sign=1):
        # Aile principale
        x_w = np.array([12, 10, 22, 20])
        y_w = np.array([1.8, 12, 12, 1.8]) * y_sign
        z_w = np.array([0, -0.3, -0.3, 0]) * y_sign * 0.3

        fig.add_trace(go.Mesh3d(
            x=x_w, y=y_w, z=z_w,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color="#D0D5DD",
            opacity=0.9,
            flatshading=True,
            name=f"Aile {'droite' if y_sign > 0 else 'gauche'}",
            showlegend=False,
        ))

    add_wing(1)
    add_wing(-1)

    # --- Dérive verticale (empennage) ---
    x_tail = np.array([33, 35, 40, 40])
    y_tail = np.array([0, 0, 0, 0])
    z_tail = np.array([1.8, 5.5, 4.5, 1.2])

    fig.add_trace(go.Mesh3d(
        x=x_tail, y=y_tail, z=z_tail,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=AF_NAVY,
        opacity=0.95,
        flatshading=True,
        name="Dérive",
        showlegend=False,
    ))

    # --- Stabilisateurs horizontaux ---
    for sign in [1, -1]:
        x_h = np.array([35, 34, 40, 39])
        y_h = np.array([0, 4.5, 4.5, 0]) * sign
        z_h = np.array([1.5, 1.2, 1.2, 1.3])

        fig.add_trace(go.Mesh3d(
            x=x_h, y=y_h, z=z_h,
            i=[0, 0], j=[1, 2], k=[2, 3],
            color="#B0B8C4",
            opacity=0.9,
            flatshading=True,
            showlegend=False,
        ))

    # --- Moteurs (cylindres simplifiés) ---
    for y_pos in [5.5, -5.5]:
        n_eng = 20
        theta_e = np.linspace(0, 2*np.pi, n_eng)
        x_eng_arr = np.linspace(13, 17, 10)
        X_eng, T_eng = np.meshgrid(x_eng_arr, theta_e)
        r_eng = 0.7
        Y_eng = y_pos + r_eng * np.cos(T_eng)
        Z_eng = -0.8 + r_eng * np.sin(T_eng)

        fig.add_trace(go.Surface(
            x=X_eng, y=Y_eng, z=Z_eng,
            colorscale=[[0, "#4A5568"], [1, "#718096"]],
            showscale=False,
            opacity=0.9,
            showlegend=False,
        ))

    # --- Annotation texte AF sur la dérive ---
    fig.add_trace(go.Scatter3d(
        x=[38], y=[0.3], z=[3.5],
        mode="text",
        text=["AF"],
        textfont=dict(size=18, color="white", family="Arial Black"),
        showlegend=False,
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-5, 45]),
            yaxis=dict(visible=False, range=[-15, 15]),
            zaxis=dict(visible=False, range=[-5, 10]),
            aspectratio=dict(x=2.2, y=1.2, z=0.6),
            camera=dict(
                eye=dict(x=1.5, y=1.0, z=0.6),
                up=dict(x=0, y=0, z=1),
            ),
            bgcolor="#F5F7FA",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=550,
        paper_bgcolor="#F5F7FA",
        title=dict(
            text="Modèle 3D — Airbus A350-900 | Livrée Air France",
            font=dict(size=14, color=AF_NAVY, family="Inter"),
            x=0.5,
        ),
    )
    return fig


# ==========================================================================
# SIDEBAR
# ==========================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0;">
        <h1 style="font-size:2.5rem; margin:0; color:white;">✈️</h1>
        <h2 style="font-size:1.3rem; margin:5px 0; color:white; font-weight:800;">
            AIR FRANCE
        </h2>
        <p style="font-size:0.8rem; opacity:0.7; color:white;">
            Analytics Platform
        </p>
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

    # Filtres globaux
    st.markdown("### 🎛️ Filtres")

    annees_dispo = sorted(vols_df["annee"].unique())
    sel_annees = st.multiselect(
        "Année(s)",
        annees_dispo,
        default=annees_dispo,
    )

    courriers = sorted(vols_df["courrier"].unique())
    sel_courriers = st.multiselect(
        "Type de courrier",
        courriers,
        default=courriers,
    )

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; font-size:0.7rem; opacity:0.5; color:white;">'
        'Portfolio Data Analytics<br>Données simulées</p>',
        unsafe_allow_html=True,
    )

# Appliquer les filtres
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

    # --- KPIs principaux ---
    total_vols = len(vf)
    total_pax = vf["passagers"].sum()
    total_rev = vf["revenu_vol_eur"].sum()
    tx_ponctualite = (vf["statut_vol"] == "À l'heure").mean() * 100
    tx_remplissage = vf["taux_remplissage"].mean() * 100
    total_co2 = vf["co2_tonnes"].sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(kpi_card("Vols totaux", format_number(total_vols), icon="🛫"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Passagers", format_number(total_pax), icon="👥"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Revenus", f"{format_number(total_rev)} €", icon="💰"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Ponctualité", f"{tx_ponctualite:.1f}%", icon="⏱️"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Taux remplissage", f"{tx_remplissage:.1f}%", icon="💺"), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi_card("CO₂ émis", f"{format_number(total_co2)} t", icon="🌿"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Graphiques ligne 1 ---
    col1, col2 = st.columns(2)

    with col1:
        section_title("Évolution mensuelle des vols", "📈")
        monthly = vf.groupby(vf["date_vol"].dt.to_period("M")).agg(
            nb_vols=("vol_id", "count"),
            passagers=("passagers", "sum"),
            revenu=("revenu_vol_eur", "sum"),
        ).reset_index()
        monthly["date_vol"] = monthly["date_vol"].dt.to_timestamp()

        fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
        fig_monthly.add_trace(go.Bar(
            x=monthly["date_vol"], y=monthly["nb_vols"],
            name="Vols", marker_color=AF_NAVY, opacity=0.7,
        ), secondary_y=False)
        fig_monthly.add_trace(go.Scatter(
            x=monthly["date_vol"], y=monthly["passagers"],
            name="Passagers", line=dict(color=AF_RED, width=3),
            mode="lines+markers", marker=dict(size=5),
        ), secondary_y=True)
        fig_monthly.update_layout(
            height=380, template="plotly_white",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            margin=dict(l=50, r=50, t=40, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        fig_monthly.update_yaxes(title_text="Nombre de vols", secondary_y=False)
        fig_monthly.update_yaxes(title_text="Passagers", secondary_y=True)
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col2:
        section_title("Répartition par continent", "🌍")
        continent_stats = vf.groupby("continent_destination").agg(
            vols=("vol_id", "count"),
            passagers=("passagers", "sum"),
            revenu=("revenu_vol_eur", "sum"),
        ).reset_index().sort_values("revenu", ascending=True)

        fig_cont = go.Figure()
        fig_cont.add_trace(go.Bar(
            y=continent_stats["continent_destination"],
            x=continent_stats["revenu"],
            orientation="h",
            marker_color=[AF_NAVY, AF_RED, AF_LIGHT_BLUE, AF_GOLD, "#00A86B", "#FF6B35",
                          "#8B5CF6", "#EC4899"][:len(continent_stats)],
            text=[f"{v/1e6:.1f}M €" for v in continent_stats["revenu"]],
            textposition="auto",
        ))
        fig_cont.update_layout(
            height=380, template="plotly_white",
            xaxis_title="Revenu (€)", yaxis_title="",
            margin=dict(l=120, r=30, t=20, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_cont, use_container_width=True)

    # --- Graphiques ligne 2 ---
    col3, col4 = st.columns(2)

    with col3:
        section_title("Ponctualité par type de courrier", "⏱️")
        ponct_courrier = vf.groupby("courrier").apply(
            lambda x: pd.Series({
                "À l'heure": (x["statut_vol"] == "À l'heure").mean() * 100,
                "Retardé": (x["statut_vol"] == "Retardé").mean() * 100,
                "Annulé": (x["statut_vol"] == "Annulé").mean() * 100,
            })
        ).reset_index()

        fig_ponct = go.Figure()
        for i, col_name in enumerate(["À l'heure", "Retardé", "Annulé"]):
            colors_ponct = ["#00A86B", "#F59E0B", AF_RED]
            fig_ponct.add_trace(go.Bar(
                x=ponct_courrier["courrier"],
                y=ponct_courrier[col_name],
                name=col_name,
                marker_color=colors_ponct[i],
            ))
        fig_ponct.update_layout(
            barmode="stack", height=380, template="plotly_white",
            yaxis_title="Pourcentage (%)",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            margin=dict(l=50, r=30, t=40, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_ponct, use_container_width=True)

    with col4:
        section_title("Top 10 routes par revenu", "🏆")
        vf_routes = vf.copy()
        vf_routes["route"] = vf_routes["aeroport_depart"] + " → " + vf_routes["aeroport_arrivee"]
        top_routes = vf_routes.groupby("route")["revenu_vol_eur"].sum().nlargest(10).reset_index()
        top_routes = top_routes.sort_values("revenu_vol_eur")

        fig_top = go.Figure(go.Bar(
            y=top_routes["route"],
            x=top_routes["revenu_vol_eur"],
            orientation="h",
            marker=dict(
                color=top_routes["revenu_vol_eur"],
                colorscale=[[0, AF_LIGHT_BLUE], [1, AF_NAVY]],
            ),
            text=[f"{v/1e6:.1f}M €" for v in top_routes["revenu_vol_eur"]],
            textposition="auto",
        ))
        fig_top.update_layout(
            height=380, template="plotly_white",
            margin=dict(l=120, r=30, t=20, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_top, use_container_width=True)

    # --- Tableau résumé ---
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
    summary.columns = [
        "Type avion", "Vols", "Passagers", "Remplissage (%)",
        "Revenu total (€)", "Retard moyen (min)", "CO₂ (tonnes)"
    ]
    st.dataframe(summary.sort_values("Vols", ascending=False), use_container_width=True, hide_index=True)


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

    # --- Statistiques réseau ---
    n_destinations = vf["aeroport_arrivee"].nunique()
    n_pays = aeroports_df[aeroports_df["code_iata"].isin(vf["aeroport_arrivee"].unique())]["pays"].nunique()
    n_continents = vf["continent_destination"].nunique()
    dist_totale = vf["distance_km"].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Destinations", str(n_destinations), icon="📍"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Pays desservis", str(n_pays), icon="🏳️"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Continents", str(n_continents), icon="🌍"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Distance totale", f"{format_number(dist_totale)} km", icon="📏"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    section_title("Carte du réseau mondial", "🌐")

    # Agréger routes
    routes_agg = vf.groupby(["aeroport_depart", "aeroport_arrivee"]).agg(
        nb_vols=("vol_id", "count"),
        passagers=("passagers", "sum"),
        revenu=("revenu_vol_eur", "sum"),
    ).reset_index()

    # Ajouter coordonnées
    routes_agg = routes_agg.merge(
        aeroports_df[["code_iata", "latitude", "longitude", "ville"]].rename(
            columns={"code_iata": "aeroport_depart", "latitude": "lat_dep",
                     "longitude": "lon_dep", "ville": "ville_dep"}),
        on="aeroport_depart"
    ).merge(
        aeroports_df[["code_iata", "latitude", "longitude", "ville"]].rename(
            columns={"code_iata": "aeroport_arrivee", "latitude": "lat_arr",
                     "longitude": "lon_arr", "ville": "ville_arr"}),
        on="aeroport_arrivee"
    )

    # Couleur par continent
    continent_color = {
        "Europe": AF_LIGHT_BLUE, "Amérique du Nord": AF_RED,
        "Amérique du Sud": AF_GOLD, "Afrique": "#00A86B",
        "Asie": "#8B5CF6", "Moyen-Orient": "#FF6B35",
        "Caraïbes": "#EC4899", "Océan Indien": "#14B8A6",
    }

    fig_map = go.Figure()

    # Routes (arcs de grand cercle)
    for _, row in routes_agg.iterrows():
        lats, lons = great_circle_points(
            row["lat_dep"], row["lon_dep"],
            row["lat_arr"], row["lon_arr"], n=40
        )
        # Chercher le continent de destination
        arr_info = apt_dict.get(row["aeroport_arrivee"], {})
        cont = arr_info.get("continent", "Europe")
        route_color = continent_color.get(cont, AF_GREY)

        width = max(1, min(4, row["nb_vols"] / routes_agg["nb_vols"].max() * 4))

        fig_map.add_trace(go.Scattergeo(
            lat=lats, lon=lons,
            mode="lines",
            line=dict(width=width, color=route_color),
            opacity=0.45,
            hoverinfo="text",
            text=f"{row['ville_dep']} → {row['ville_arr']}<br>"
                 f"Vols: {row['nb_vols']:,}<br>"
                 f"Passagers: {row['passagers']:,}<br>"
                 f"Revenu: {row['revenu']/1e6:.1f}M €",
            showlegend=False,
        ))

    # Aéroports (taille par trafic)
    apt_traffic = pd.concat([
        vf.groupby("aeroport_depart")["passagers"].sum(),
        vf.groupby("aeroport_arrivee")["passagers"].sum(),
    ]).groupby(level=0).sum().reset_index()
    apt_traffic.columns = ["code_iata", "total_pax"]
    apt_traffic = apt_traffic.merge(aeroports_df, on="code_iata")
    apt_traffic["size"] = np.clip(apt_traffic["total_pax"] / apt_traffic["total_pax"].max() * 25, 4, 25)

    # Colorer les hubs différemment
    apt_traffic["color"] = apt_traffic["type_aeroport"].map({
        "Hub": AF_RED, "Domestique": AF_NAVY,
        "Européen": AF_LIGHT_BLUE, "Long-courrier": AF_GOLD,
        "Moyen-courrier": "#00A86B",
    }).fillna(AF_GREY)

    fig_map.add_trace(go.Scattergeo(
        lat=apt_traffic["latitude"],
        lon=apt_traffic["longitude"],
        mode="markers+text",
        marker=dict(
            size=apt_traffic["size"],
            color=apt_traffic["color"],
            line=dict(width=1, color="white"),
            opacity=0.9,
        ),
        text=apt_traffic["code_iata"],
        textposition="top center",
        textfont=dict(size=8, color=AF_NAVY, family="Inter"),
        hoverinfo="text",
        hovertext=apt_traffic.apply(
            lambda r: f"<b>{r['nom']}</b><br>{r['ville']}, {r['pays']}<br>"
                      f"Trafic: {r['total_pax']:,} passagers", axis=1
        ),
        showlegend=False,
    ))

    fig_map.update_geos(
        projection_type="natural earth",
        showcoastlines=True, coastlinecolor="#B0BEC5",
        showland=True, landcolor="#F5F5F0",
        showocean=True, oceancolor="#E8F0FE",
        showlakes=True, lakecolor="#E8F0FE",
        showcountries=True, countrycolor="#D5DBDB",
        showframe=False,
        lonaxis_range=[-130, 160],
        lataxis_range=[-40, 65],
    )
    fig_map.update_layout(
        height=600, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=AF_BG,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # --- Top destinations ---
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section_title("Top 15 destinations par passagers", "🏆")
        top_dest = vf.groupby("aeroport_arrivee").agg(
            passagers=("passagers", "sum"),
            vols=("vol_id", "count"),
        ).nlargest(15, "passagers").reset_index()
        top_dest = top_dest.merge(
            aeroports_df[["code_iata", "ville", "pays"]],
            left_on="aeroport_arrivee", right_on="code_iata"
        )
        top_dest["label"] = top_dest["ville"] + " (" + top_dest["aeroport_arrivee"] + ")"
        top_dest = top_dest.sort_values("passagers")

        fig_dest = go.Figure(go.Bar(
            y=top_dest["label"],
            x=top_dest["passagers"],
            orientation="h",
            marker=dict(
                color=top_dest["passagers"],
                colorscale=[[0, AF_LIGHT_BLUE], [0.5, AF_NAVY], [1, AF_RED]],
            ),
            text=[format_number(v) for v in top_dest["passagers"]],
            textposition="auto",
        ))
        fig_dest.update_layout(
            height=500, template="plotly_white",
            margin=dict(l=160, r=30, t=10, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_dest, use_container_width=True)

    with col2:
        section_title("Répartition des distances", "📏")
        fig_dist = px.histogram(
            vf, x="distance_km", nbins=50,
            color="courrier",
            color_discrete_map={
                "Court-courrier": AF_LIGHT_BLUE,
                "Moyen-courrier": AF_GOLD,
                "Long-courrier": AF_NAVY,
            },
            labels={"distance_km": "Distance (km)", "count": "Nombre de vols"},
        )
        fig_dist.update_layout(
            height=500, template="plotly_white",
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            margin=dict(l=50, r=30, t=30, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_dist, use_container_width=True)


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

    # --- Modèle 3D ---
    section_title("Modèle 3D interactif — Faites-le tourner !", "✨")
    st.markdown(
        '<p style="color:#6B7B8D; font-size:0.9rem;">'
        '🖱️ Cliquez et faites glisser pour faire pivoter l\'avion • Molette pour zoomer</p>',
        unsafe_allow_html=True,
    )
    fig_3d = build_3d_aircraft()
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- KPIs flotte ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Avions", str(len(flotte_df)), icon="🛩️"), unsafe_allow_html=True)
    with c2:
        nb_types = flotte_df["type_avion"].nunique()
        st.markdown(kpi_card("Types d'avions", str(nb_types), icon="📋"), unsafe_allow_html=True)
    with c3:
        age_moy = flotte_df["age_avion_ans"].mean()
        st.markdown(kpi_card("Âge moyen", f"{age_moy:.1f} ans", icon="📅"), unsafe_allow_html=True)
    with c4:
        en_service = (flotte_df["statut"] == "En service").sum()
        st.markdown(kpi_card("En service", str(en_service), icon="✅"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_title("Composition de la flotte", "📊")
        fleet_comp = flotte_df.groupby("type_avion").size().reset_index(name="count")
        fleet_comp = fleet_comp.sort_values("count", ascending=False)

        fig_fleet = go.Figure(go.Bar(
            x=fleet_comp["type_avion"],
            y=fleet_comp["count"],
            marker_color=[AF_PALETTE[i % len(AF_PALETTE)] for i in range(len(fleet_comp))],
            text=fleet_comp["count"],
            textposition="outside",
        ))
        fig_fleet.update_layout(
            height=420, template="plotly_white",
            xaxis_tickangle=-35,
            yaxis_title="Nombre d'appareils",
            margin=dict(l=50, r=30, t=20, b=100),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_fleet, use_container_width=True)

    with col2:
        section_title("Répartition Narrow-body / Wide-body", "✈️")
        famille_stats = flotte_df.groupby("famille").agg(
            count=("immatriculation", "count"),
            age_moy=("age_avion_ans", "mean"),
        ).reset_index()

        fig_fam = go.Figure(go.Pie(
            labels=famille_stats["famille"],
            values=famille_stats["count"],
            hole=0.55,
            marker_colors=[AF_NAVY, AF_RED],
            textinfo="label+percent+value",
            textfont=dict(size=13),
        ))
        fig_fam.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="white",
            annotations=[dict(text="Flotte", x=0.5, y=0.5, font_size=18,
                              font_color=AF_NAVY, showarrow=False)],
        )
        st.plotly_chart(fig_fam, use_container_width=True)

    # --- Diagramme d'âge ---
    section_title("Distribution de l'âge de la flotte", "📅")
    fig_age = px.histogram(
        flotte_df, x="age_avion_ans", nbins=25,
        color="famille",
        color_discrete_map={"Narrow-body": AF_LIGHT_BLUE, "Wide-body": AF_NAVY},
        labels={"age_avion_ans": "Âge (années)", "count": "Nombre d'avions"},
    )
    fig_age.update_layout(
        height=350, template="plotly_white",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        margin=dict(l=50, r=30, t=30, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    st.plotly_chart(fig_age, use_container_width=True)

    # --- Tableau détaillé de la flotte ---
    section_title("Détail de la flotte", "📋")
    fleet_summary = flotte_df.groupby("type_avion").agg(
        nb_avions=("immatriculation", "count"),
        sieges_moy=("nb_sieges_total", "mean"),
        age_moy=("age_avion_ans", "mean"),
        business=("nb_sieges_business", "first"),
        premium=("nb_sieges_premium_eco", "first"),
        eco=("nb_sieges_eco", "first"),
        autonomie=("autonomie_km", "first"),
    ).reset_index()
    fleet_summary.columns = [
        "Type", "Nb avions", "Sièges", "Âge moyen",
        "Business", "Premium Éco", "Économique", "Autonomie (km)"
    ]
    fleet_summary["Âge moyen"] = fleet_summary["Âge moyen"].round(1)
    fleet_summary["Sièges"] = fleet_summary["Sièges"].round(0).astype(int)
    st.dataframe(fleet_summary.sort_values("Nb avions", ascending=False),
                 use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE 4 : OPÉRATIONS
# ==========================================================================
elif page == "📊 Opérations":
    st.markdown("""
    <div class="af-banner">
        <h1>📊 Opérations — Performance Opérationnelle</h1>
        <p>Ponctualité, retards, taux de remplissage</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    retard_moyen = vf[vf["retard_depart_min"] > 0]["retard_depart_min"].mean()
    tx_annulation = (vf["statut_vol"] == "Annulé").mean() * 100
    otp15 = (vf["retard_depart_min"] <= 15).mean() * 100
    rpk_total = vf["rpk"].sum()
    ask_total = vf["ask"].sum()
    load_factor_global = rpk_total / ask_total * 100 if ask_total > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("OTP (15 min)", f"{otp15:.1f}%", icon="⏱️"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Retard moyen", f"{retard_moyen:.0f} min", icon="⏳"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Taux annulation", f"{tx_annulation:.2f}%", icon="❌"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("RPK", format_number(rpk_total), icon="📏"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Load Factor", f"{load_factor_global:.1f}%", icon="💺"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_title("Causes de retard", "🔍")
        causes = vf[vf["cause_retard"] != "Aucun"]["cause_retard"].value_counts().reset_index()
        causes.columns = ["Cause", "Nombre"]

        fig_causes = go.Figure(go.Pie(
            labels=causes["Cause"],
            values=causes["Nombre"],
            hole=0.5,
            marker_colors=AF_PALETTE[:len(causes)],
            textinfo="label+percent",
        ))
        fig_causes.update_layout(
            height=400, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="white",
            annotations=[dict(text="Retards", x=0.5, y=0.5,
                              font_size=16, font_color=AF_NAVY, showarrow=False)],
        )
        st.plotly_chart(fig_causes, use_container_width=True)

    with col2:
        section_title("Distribution des retards", "📊")
        vf_delayed = vf[vf["retard_depart_min"] > 0].copy()
        fig_delay_dist = px.histogram(
            vf_delayed, x="retard_depart_min", nbins=60,
            color="courrier",
            color_discrete_map={
                "Court-courrier": AF_LIGHT_BLUE,
                "Moyen-courrier": AF_GOLD,
                "Long-courrier": AF_NAVY,
            },
            labels={"retard_depart_min": "Retard (minutes)"},
        )
        fig_delay_dist.update_layout(
            height=400, template="plotly_white",
            xaxis_range=[0, 200],
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            margin=dict(l=50, r=30, t=30, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_delay_dist, use_container_width=True)

    # --- Ponctualité mensuelle ---
    section_title("Ponctualité mensuelle (OTP 15 min)", "📈")
    monthly_otp = vf.groupby(vf["date_vol"].dt.to_period("M")).apply(
        lambda x: (x["retard_depart_min"] <= 15).mean() * 100
    ).reset_index()
    monthly_otp.columns = ["mois", "otp"]
    monthly_otp["mois"] = monthly_otp["mois"].dt.to_timestamp()

    fig_otp = go.Figure()
    fig_otp.add_trace(go.Scatter(
        x=monthly_otp["mois"], y=monthly_otp["otp"],
        mode="lines+markers",
        line=dict(color=AF_NAVY, width=3),
        marker=dict(size=6, color=AF_NAVY),
        fill="tozeroy",
        fillcolor="rgba(0,33,87,0.08)",
        name="OTP",
    ))
    fig_otp.add_hline(y=80, line_dash="dash", line_color=AF_RED,
                      annotation_text="Objectif 80%", annotation_position="top right")
    fig_otp.update_layout(
        height=350, template="plotly_white",
        yaxis_title="OTP (%)", yaxis_range=[60, 100],
        margin=dict(l=50, r=30, t=20, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    st.plotly_chart(fig_otp, use_container_width=True)

    # --- Heatmap jour/heure ---
    section_title("Heatmap des retards par jour et heure", "🗓️")
    vf_heat = vf.copy()
    vf_heat["jour_semaine"] = vf_heat["date_vol"].dt.day_name()
    vf_heat["heure"] = vf_heat["heure_depart"].str[:2].astype(int)
    heat_data = vf_heat.groupby(["jour_semaine", "heure"])["retard_depart_min"].mean().reset_index()
    heat_pivot = heat_data.pivot(index="jour_semaine", columns="heure", values="retard_depart_min")

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    days_fr = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    heat_pivot = heat_pivot.reindex(days_order)

    fig_heat = go.Figure(go.Heatmap(
        z=heat_pivot.values,
        x=[f"{h}h" for h in heat_pivot.columns],
        y=days_fr,
        colorscale=[[0, "#E8F0FE"], [0.5, AF_LIGHT_BLUE], [1, AF_RED]],
        colorbar=dict(title="Retard moyen (min)"),
    ))
    fig_heat.update_layout(
        height=350, template="plotly_white",
        margin=dict(l=100, r=30, t=20, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- Taux de remplissage ---
    col1, col2 = st.columns(2)
    with col1:
        section_title("Taux de remplissage mensuel", "💺")
        monthly_lr = vf.groupby([vf["date_vol"].dt.to_period("M"), "courrier"])[
            "taux_remplissage"
        ].mean().reset_index()
        monthly_lr["date_vol"] = monthly_lr["date_vol"].dt.to_timestamp()
        monthly_lr["taux_remplissage"] *= 100

        fig_lr = px.line(
            monthly_lr, x="date_vol", y="taux_remplissage", color="courrier",
            color_discrete_map={
                "Court-courrier": AF_LIGHT_BLUE,
                "Moyen-courrier": AF_GOLD,
                "Long-courrier": AF_NAVY,
            },
            labels={"taux_remplissage": "Taux de remplissage (%)", "date_vol": ""},
        )
        fig_lr.update_layout(
            height=380, template="plotly_white",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            margin=dict(l=50, r=30, t=30, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_lr, use_container_width=True)

    with col2:
        section_title("Remplissage par type d'avion", "📊")
        lr_avion = vf.groupby("type_avion")["taux_remplissage"].mean().reset_index()
        lr_avion["taux_remplissage"] *= 100
        lr_avion = lr_avion.sort_values("taux_remplissage")

        fig_lr_av = go.Figure(go.Bar(
            y=lr_avion["type_avion"],
            x=lr_avion["taux_remplissage"],
            orientation="h",
            marker_color=AF_NAVY,
            text=[f"{v:.1f}%" for v in lr_avion["taux_remplissage"]],
            textposition="auto",
        ))
        fig_lr_av.add_vline(x=80, line_dash="dash", line_color=AF_RED)
        fig_lr_av.update_layout(
            height=380, template="plotly_white",
            xaxis_title="Taux de remplissage (%)",
            xaxis_range=[60, 100],
            margin=dict(l=150, r=30, t=20, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_lr_av, use_container_width=True)


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

    rev_total = vf["revenu_vol_eur"].sum()
    rev_par_vol = vf["revenu_vol_eur"].mean()
    rev_par_pax = vf["revenu_par_pax_eur"].mean()
    yield_val = rev_total / vf["rpk"].sum() * 100 if vf["rpk"].sum() > 0 else 0
    rask = rev_total / vf["ask"].sum() * 100 if vf["ask"].sum() > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Revenu total", f"{format_number(rev_total)} €", icon="💰"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Revenu / vol", f"{rev_par_vol:,.0f} €", icon="🛫"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Revenu / pax", f"{rev_par_pax:,.0f} €", icon="👤"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Yield", f"{yield_val:.2f} c€/RPK", icon="📈"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("RASK", f"{rask:.2f} c€/ASK", icon="📊"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_title("Revenus mensuels", "📈")
        monthly_rev = vf.groupby(vf["date_vol"].dt.to_period("M")).agg(
            revenu=("revenu_vol_eur", "sum"),
            vols=("vol_id", "count"),
        ).reset_index()
        monthly_rev["date_vol"] = monthly_rev["date_vol"].dt.to_timestamp()

        fig_rev = make_subplots(specs=[[{"secondary_y": True}]])
        fig_rev.add_trace(go.Bar(
            x=monthly_rev["date_vol"],
            y=monthly_rev["revenu"],
            name="Revenu (€)",
            marker_color=AF_NAVY, opacity=0.7,
        ), secondary_y=False)
        fig_rev.add_trace(go.Scatter(
            x=monthly_rev["date_vol"],
            y=monthly_rev["revenu"] / monthly_rev["vols"],
            name="Rev/vol (€)",
            line=dict(color=AF_RED, width=3),
            mode="lines+markers", marker=dict(size=5),
        ), secondary_y=True)
        fig_rev.update_layout(
            height=400, template="plotly_white",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            margin=dict(l=60, r=60, t=40, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        fig_rev.update_yaxes(title_text="Revenu total (€)", secondary_y=False)
        fig_rev.update_yaxes(title_text="Revenu par vol (€)", secondary_y=True)
        st.plotly_chart(fig_rev, use_container_width=True)

    with col2:
        section_title("Revenus par continent", "🌍")
        rev_cont = vf.groupby("continent_destination")["revenu_vol_eur"].sum().reset_index()
        rev_cont = rev_cont.sort_values("revenu_vol_eur", ascending=False)

        fig_rev_cont = go.Figure(go.Treemap(
            labels=rev_cont["continent_destination"],
            parents=[""] * len(rev_cont),
            values=rev_cont["revenu_vol_eur"],
            textinfo="label+value+percent root",
            texttemplate="%{label}<br>%{value:,.0f} €<br>%{percentRoot:.1%}",
            marker_colors=[AF_NAVY, AF_RED, AF_LIGHT_BLUE, AF_GOLD, "#00A86B",
                           "#FF6B35", "#8B5CF6", "#EC4899"][:len(rev_cont)],
        ))
        fig_rev_cont.update_layout(
            height=400, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_rev_cont, use_container_width=True)

    # --- Yield et RASK par courrier ---
    section_title("Yield et RASK par type de courrier", "📊")
    yield_courrier = vf.groupby("courrier").apply(
        lambda x: pd.Series({
            "Yield (c€/RPK)": x["revenu_vol_eur"].sum() / x["rpk"].sum() * 100 if x["rpk"].sum() > 0 else 0,
            "RASK (c€/ASK)": x["revenu_vol_eur"].sum() / x["ask"].sum() * 100 if x["ask"].sum() > 0 else 0,
            "Rev/pax moyen (€)": x["revenu_par_pax_eur"].mean(),
        })
    ).reset_index()

    col1, col2, col3 = st.columns(3)
    for i, metric in enumerate(["Yield (c€/RPK)", "RASK (c€/ASK)", "Rev/pax moyen (€)"]):
        with [col1, col2, col3][i]:
            fig_m = go.Figure(go.Bar(
                x=yield_courrier["courrier"],
                y=yield_courrier[metric],
                marker_color=[AF_LIGHT_BLUE, AF_GOLD, AF_NAVY][:len(yield_courrier)],
                text=[f"{v:.2f}" for v in yield_courrier[metric]],
                textposition="outside",
            ))
            fig_m.update_layout(
                height=350, template="plotly_white",
                title=dict(text=metric, font=dict(size=14, color=AF_NAVY)),
                margin=dict(l=50, r=30, t=50, b=40),
                paper_bgcolor="white", plot_bgcolor="white",
            )
            st.plotly_chart(fig_m, use_container_width=True)

    # --- Évolution annuelle comparative ---
    section_title("Évolution annuelle comparative", "📅")
    yearly = vf.groupby("annee").agg(
        vols=("vol_id", "count"),
        passagers=("passagers", "sum"),
        revenu=("revenu_vol_eur", "sum"),
        remplissage=("taux_remplissage", "mean"),
        co2=("co2_tonnes", "sum"),
    ).reset_index()
    yearly["remplissage"] = (yearly["remplissage"] * 100).round(1)

    yearly_display = yearly.copy()
    yearly_display.columns = ["Année", "Vols", "Passagers", "Revenu (€)",
                              "Remplissage (%)", "CO₂ (t)"]
    st.dataframe(yearly_display, use_container_width=True, hide_index=True)


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
    nps_promo = (sf["nps_categorie"] == "Promoteur").mean() * 100
    nps_detrac = (sf["nps_categorie"] == "Détracteur").mean() * 100
    nps_score = nps_promo - nps_detrac
    recomm = (sf["recommandation"] == "Oui").mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Note globale", f"{note_moy:.1f}/10", icon="⭐"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("NPS", f"{nps_score:.0f}", icon="📊"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Promoteurs", f"{nps_promo:.1f}%", icon="😊"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Détracteurs", f"{nps_detrac:.1f}%", icon="😞"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Recommandation", f"{recomm:.1f}%", icon="👍"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_title("Notes moyennes par dimension", "📊")
        dims = ["note_confort", "note_repas", "note_divertissement",
                "note_equipage", "note_ponctualite", "note_enregistrement"]
        labels = ["Confort", "Repas", "Divertissement",
                  "Équipage", "Ponctualité", "Enregistrement"]
        values = [sf[d].mean() for d in dims]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(0,33,87,0.15)",
            line=dict(color=AF_NAVY, width=3),
            marker=dict(size=8, color=AF_NAVY),
            name="Note moyenne",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10], ticksuffix="/10"),
                bgcolor="white",
            ),
            height=420,
            margin=dict(l=60, r=60, t=40, b=40),
            paper_bgcolor="white",
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        section_title("Répartition NPS", "📈")
        nps_data = sf["nps_categorie"].value_counts().reset_index()
        nps_data.columns = ["Catégorie", "Nombre"]
        color_map = {"Promoteur": "#00A86B", "Passif": AF_GOLD, "Détracteur": AF_RED}

        fig_nps = go.Figure(go.Pie(
            labels=nps_data["Catégorie"],
            values=nps_data["Nombre"],
            hole=0.55,
            marker_colors=[color_map.get(c, AF_GREY) for c in nps_data["Catégorie"]],
            textinfo="label+percent",
            textfont=dict(size=14),
        ))
        fig_nps.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="white",
            annotations=[dict(text=f"NPS<br>{nps_score:.0f}", x=0.5, y=0.5,
                              font_size=22, font_color=AF_NAVY, showarrow=False)],
        )
        st.plotly_chart(fig_nps, use_container_width=True)

    # --- Satisfaction par classe ---
    section_title("Satisfaction par classe de voyage", "💎")
    sat_classe = sf.groupby("classe")[dims].mean().reset_index()

    fig_classe = go.Figure()
    colors_cls = {"Business": AF_NAVY, "Premium Éco": AF_GOLD, "Économique": AF_LIGHT_BLUE}
    for _, row in sat_classe.iterrows():
        fig_classe.add_trace(go.Scatterpolar(
            r=[row[d] for d in dims] + [row[dims[0]]],
            theta=labels + [labels[0]],
            name=row["classe"],
            line=dict(color=colors_cls.get(row["classe"], AF_GREY), width=2),
            fill="toself",
            opacity=0.4,
        ))
    fig_classe.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        height=420,
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        margin=dict(l=60, r=60, t=20, b=60),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_classe, use_container_width=True)

    # --- Satisfaction par type de voyageur et par nationalité ---
    col1, col2 = st.columns(2)

    with col1:
        section_title("Par type de voyageur", "🧳")
        sat_type = sf.groupby("type_voyageur")["note_globale"].mean().reset_index()
        sat_type = sat_type.sort_values("note_globale")

        fig_type = go.Figure(go.Bar(
            y=sat_type["type_voyageur"],
            x=sat_type["note_globale"],
            orientation="h",
            marker_color=AF_NAVY,
            text=[f"{v:.1f}/10" for v in sat_type["note_globale"]],
            textposition="auto",
        ))
        fig_type.update_layout(
            height=350, template="plotly_white",
            xaxis_range=[5, 10], xaxis_title="Note globale (/10)",
            margin=dict(l=100, r=30, t=20, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with col2:
        section_title("Par programme fidélité", "🏅")
        fid_order = ["Non inscrit", "Explorer", "Silver", "Gold", "Platinum", "Ultimate"]
        sat_fid = sf.groupby("programme_fidelite")["note_globale"].mean().reset_index()
        sat_fid["programme_fidelite"] = pd.Categorical(
            sat_fid["programme_fidelite"], categories=fid_order, ordered=True
        )
        sat_fid = sat_fid.sort_values("programme_fidelite")

        fig_fid = go.Figure(go.Bar(
            x=sat_fid["programme_fidelite"],
            y=sat_fid["note_globale"],
            marker=dict(
                color=sat_fid["note_globale"],
                colorscale=[[0, AF_LIGHT_BLUE], [1, AF_GOLD]],
            ),
            text=[f"{v:.1f}" for v in sat_fid["note_globale"]],
            textposition="outside",
        ))
        fig_fid.update_layout(
            height=350, template="plotly_white",
            yaxis_range=[5, 10], yaxis_title="Note globale (/10)",
            margin=dict(l=50, r=30, t=20, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_fid, use_container_width=True)

    # --- Évolution de la satisfaction ---
    section_title("Évolution mensuelle de la satisfaction", "📈")
    monthly_sat = sf.groupby(sf["date_vol"].dt.to_period("M")).agg(
        note_globale=("note_globale", "mean"),
        note_equipage=("note_equipage", "mean"),
        note_ponctualite=("note_ponctualite", "mean"),
    ).reset_index()
    monthly_sat["date_vol"] = monthly_sat["date_vol"].dt.to_timestamp()

    fig_sat_evo = go.Figure()
    for col_name, label, color in [
        ("note_globale", "Globale", AF_NAVY),
        ("note_equipage", "Équipage", "#00A86B"),
        ("note_ponctualite", "Ponctualité", AF_RED),
    ]:
        fig_sat_evo.add_trace(go.Scatter(
            x=monthly_sat["date_vol"], y=monthly_sat[col_name],
            mode="lines+markers", name=label,
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ))
    fig_sat_evo.update_layout(
        height=350, template="plotly_white",
        yaxis_title="Note (/10)", yaxis_range=[5, 10],
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        margin=dict(l=50, r=30, t=30, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    st.plotly_chart(fig_sat_evo, use_container_width=True)


# ==========================================================================
# PAGE 7 : IMPACT ENVIRONNEMENTAL
# ==========================================================================
elif page == "🌿 Impact Environnemental":
    st.markdown("""
    <div class="af-banner">
        <h1>🌿 Impact Environnemental</h1>
        <p>Émissions CO₂, consommation carburant et efficacité énergétique</p>
    </div>
    """, unsafe_allow_html=True)

    co2_total = vf["co2_tonnes"].sum()
    fuel_total = vf["carburant_litres"].sum()
    co2_par_pax_km = co2_total * 1e6 / vf["rpk"].sum() if vf["rpk"].sum() > 0 else 0
    fuel_par_vol = vf["carburant_litres"].mean()
    co2_par_vol = vf["co2_tonnes"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("CO₂ total", f"{format_number(co2_total)} t", icon="🏭"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Carburant total", f"{format_number(fuel_total)} L", icon="⛽"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("CO₂/pax·km", f"{co2_par_pax_km:.1f} g", icon="📏"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("CO₂/vol moyen", f"{co2_par_vol:.1f} t", icon="🛫"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("Carburant/vol", f"{format_number(fuel_par_vol)} L", icon="⛽"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        section_title("Émissions CO₂ mensuelles", "📈")
        monthly_co2 = vf.groupby(vf["date_vol"].dt.to_period("M")).agg(
            co2=("co2_tonnes", "sum"),
            vols=("vol_id", "count"),
        ).reset_index()
        monthly_co2["date_vol"] = monthly_co2["date_vol"].dt.to_timestamp()
        monthly_co2["co2_par_vol"] = monthly_co2["co2"] / monthly_co2["vols"]

        fig_co2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_co2.add_trace(go.Bar(
            x=monthly_co2["date_vol"], y=monthly_co2["co2"],
            name="CO₂ total (t)", marker_color="#00A86B", opacity=0.7,
        ), secondary_y=False)
        fig_co2.add_trace(go.Scatter(
            x=monthly_co2["date_vol"], y=monthly_co2["co2_par_vol"],
            name="CO₂/vol (t)", line=dict(color=AF_NAVY, width=3),
            mode="lines+markers", marker=dict(size=5),
        ), secondary_y=True)
        fig_co2.update_layout(
            height=400, template="plotly_white",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            margin=dict(l=60, r=60, t=40, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        fig_co2.update_yaxes(title_text="CO₂ total (t)", secondary_y=False)
        fig_co2.update_yaxes(title_text="CO₂ par vol (t)", secondary_y=True)
        st.plotly_chart(fig_co2, use_container_width=True)

    with col2:
        section_title("CO₂ par type de courrier", "🌍")
        co2_courrier = vf.groupby("courrier").agg(
            co2_total=("co2_tonnes", "sum"),
            co2_moyen=("co2_tonnes", "mean"),
            rpk=("rpk", "sum"),
        ).reset_index()
        co2_courrier["co2_per_rpk_g"] = co2_courrier["co2_total"] * 1e6 / co2_courrier["rpk"]

        fig_co2_c = make_subplots(
            rows=1, cols=2,
            subplot_titles=("CO₂ total par courrier", "CO₂ par pax·km (g)"),
            specs=[[{"type": "pie"}, {"type": "bar"}]],
        )
        fig_co2_c.add_trace(go.Pie(
            labels=co2_courrier["courrier"],
            values=co2_courrier["co2_total"],
            marker_colors=[AF_LIGHT_BLUE, AF_GOLD, AF_NAVY],
            hole=0.45,
            textinfo="label+percent",
        ), row=1, col=1)
        fig_co2_c.add_trace(go.Bar(
            x=co2_courrier["courrier"],
            y=co2_courrier["co2_per_rpk_g"],
            marker_color=[AF_LIGHT_BLUE, AF_GOLD, AF_NAVY],
            text=[f"{v:.1f}g" for v in co2_courrier["co2_per_rpk_g"]],
            textposition="outside",
        ), row=1, col=2)
        fig_co2_c.update_layout(
            height=400, template="plotly_white",
            showlegend=False,
            margin=dict(l=40, r=40, t=50, b=40),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_co2_c, use_container_width=True)

    # --- Efficacité par type d'avion ---
    section_title("Efficacité carbone par type d'avion", "✈️")
    eff_avion = vf.groupby("type_avion").apply(
        lambda x: pd.Series({
            "CO₂/pax·km (g)": x["co2_tonnes"].sum() * 1e6 / x["rpk"].sum() if x["rpk"].sum() > 0 else 0,
            "Carburant/vol (L)": x["carburant_litres"].mean(),
            "Nb vols": len(x),
        })
    ).reset_index().sort_values("CO₂/pax·km (g)")

    fig_eff = go.Figure()
    fig_eff.add_trace(go.Bar(
        y=eff_avion["type_avion"],
        x=eff_avion["CO₂/pax·km (g)"],
        orientation="h",
        marker=dict(
            color=eff_avion["CO₂/pax·km (g)"],
            colorscale=[[0, "#00A86B"], [0.5, AF_GOLD], [1, AF_RED]],
            colorbar=dict(title="g CO₂/pax·km"),
        ),
        text=[f"{v:.1f}g" for v in eff_avion["CO₂/pax·km (g)"]],
        textposition="auto",
    ))
    fig_eff.update_layout(
        height=400, template="plotly_white",
        xaxis_title="CO₂ par passager-kilomètre (g)",
        margin=dict(l=160, r=40, t=20, b=40),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    st.plotly_chart(fig_eff, use_container_width=True)

    # --- Scatter : Distance vs CO2 ---
    section_title("Distance vs Émissions CO₂", "📊")
    sample_vf = vf.sample(min(5000, len(vf)), random_state=42)

    fig_scatter = px.scatter(
        sample_vf, x="distance_km", y="co2_tonnes",
        color="courrier",
        size="passagers",
        color_discrete_map={
            "Court-courrier": AF_LIGHT_BLUE,
            "Moyen-courrier": AF_GOLD,
            "Long-courrier": AF_NAVY,
        },
        labels={"distance_km": "Distance (km)", "co2_tonnes": "CO₂ (tonnes)"},
        opacity=0.5,
        size_max=15,
    )
    fig_scatter.update_layout(
        height=450, template="plotly_white",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        margin=dict(l=60, r=30, t=30, b=50),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Comparatif annuel ---
    section_title("Évolution annuelle de l'empreinte carbone", "📅")
    yearly_env = vf.groupby("annee").apply(
        lambda x: pd.Series({
            "CO₂ total (kt)": x["co2_tonnes"].sum() / 1000,
            "CO₂/pax·km (g)": x["co2_tonnes"].sum() * 1e6 / x["rpk"].sum() if x["rpk"].sum() > 0 else 0,
            "Carburant (ML)": x["carburant_litres"].sum() / 1e6,
            "Vols": len(x),
        })
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig_y1 = go.Figure()
        fig_y1.add_trace(go.Bar(
            x=yearly_env["annee"].astype(str),
            y=yearly_env["CO₂ total (kt)"],
            marker_color="#00A86B",
            text=[f"{v:.1f} kt" for v in yearly_env["CO₂ total (kt)"]],
            textposition="outside",
        ))
        fig_y1.update_layout(
            height=350, template="plotly_white",
            title=dict(text="CO₂ total par année (kilotonnes)", font=dict(size=14, color=AF_NAVY)),
            margin=dict(l=50, r=30, t=50, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_y1, use_container_width=True)

    with col2:
        fig_y2 = go.Figure()
        fig_y2.add_trace(go.Bar(
            x=yearly_env["annee"].astype(str),
            y=yearly_env["CO₂/pax·km (g)"],
            marker_color=AF_NAVY,
            text=[f"{v:.1f} g" for v in yearly_env["CO₂/pax·km (g)"]],
            textposition="outside",
        ))
        fig_y2.update_layout(
            height=350, template="plotly_white",
            title=dict(text="Intensité carbone (g CO₂/pax·km)", font=dict(size=14, color=AF_NAVY)),
            margin=dict(l=50, r=30, t=50, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_y2, use_container_width=True)


# ==========================================================================
# FOOTER
# ==========================================================================
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center; padding:20px; color:{AF_GREY}; font-size:0.8rem;">
        <strong>Air France Analytics Platform</strong> — Projet Portfolio Data Analytics<br>
        📊 {len(vols_df):,} vols • {vols_df['passagers'].sum():,} passagers •
        {aeroports_df['code_iata'].nunique()} aéroports • {len(flotte_df)} avions<br>
        <em>Données simulées à des fins de démonstration</em>
    </div>
    """,
    unsafe_allow_html=True,
)