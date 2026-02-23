"""
============================================================================
 AIR FRANCE ANALYTICS - Dataset Generator
 Données réalistes : Flotte, Vols, Passagers, Satisfaction
============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE_DEBUT = datetime(2023, 1, 1)
DATE_FIN = datetime(2025, 5, 31)
N_DAYS = (DATE_FIN - DATE_DEBUT).days

print("=" * 70)
print("  ✈️  AIR FRANCE — Générateur de Dataset Analytics")
print("=" * 70)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ============================================================
# 1. AÉROPORTS
# ============================================================
print("\n[1/4] Génération des AÉROPORTS...")

AEROPORTS_RAW = [
    ("CDG","Paris Charles de Gaulle","Paris","France","Europe",49.01,2.55,"Hub"),
    ("ORY","Paris Orly","Paris","France","Europe",48.72,2.38,"Hub"),
    ("MRS","Marseille Provence","Marseille","France","Europe",43.44,5.22,"Domestique"),
    ("LYS","Lyon Saint-Exupéry","Lyon","France","Europe",45.73,5.08,"Domestique"),
    ("TLS","Toulouse-Blagnac","Toulouse","France","Europe",43.63,1.36,"Domestique"),
    ("NCE","Nice Côte d'Azur","Nice","France","Europe",43.66,7.22,"Domestique"),
    ("BOD","Bordeaux-Mérignac","Bordeaux","France","Europe",44.83,-0.72,"Domestique"),
    ("NTE","Nantes Atlantique","Nantes","France","Europe",47.15,-1.61,"Domestique"),
    ("LHR","London Heathrow","Londres","Royaume-Uni","Europe",51.47,-0.45,"Européen"),
    ("AMS","Amsterdam Schiphol","Amsterdam","Pays-Bas","Europe",52.31,4.76,"Européen"),
    ("FRA","Frankfurt","Francfort","Allemagne","Europe",50.03,8.57,"Européen"),
    ("FCO","Roma Fiumicino","Rome","Italie","Europe",41.80,12.24,"Européen"),
    ("MAD","Madrid Barajas","Madrid","Espagne","Europe",40.50,-3.57,"Européen"),
    ("BCN","Barcelona El Prat","Barcelone","Espagne","Europe",41.30,2.08,"Européen"),
    ("LIS","Lisbon","Lisbonne","Portugal","Europe",38.78,-9.14,"Européen"),
    ("MUC","Munich","Munich","Allemagne","Europe",48.35,11.79,"Européen"),
    ("VIE","Vienna","Vienne","Autriche","Europe",48.11,16.57,"Européen"),
    ("ZRH","Zurich","Zurich","Suisse","Europe",47.46,8.55,"Européen"),
    ("CPH","Copenhagen","Copenhague","Danemark","Europe",55.62,12.65,"Européen"),
    ("ARN","Stockholm Arlanda","Stockholm","Suède","Europe",59.65,17.92,"Européen"),
    ("ATH","Athens","Athènes","Grèce","Europe",37.94,23.94,"Européen"),
    ("IST","Istanbul","Istanbul","Turquie","Europe",41.28,28.75,"Européen"),
    ("JFK","New York JFK","New York","États-Unis","Amérique du Nord",40.64,-73.78,"Long-courrier"),
    ("LAX","Los Angeles","Los Angeles","États-Unis","Amérique du Nord",33.94,-118.41,"Long-courrier"),
    ("SFO","San Francisco","San Francisco","États-Unis","Amérique du Nord",37.62,-122.38,"Long-courrier"),
    ("YUL","Montréal Trudeau","Montréal","Canada","Amérique du Nord",45.47,-73.74,"Long-courrier"),
    ("MIA","Miami","Miami","États-Unis","Amérique du Nord",25.80,-80.29,"Long-courrier"),
    ("ATL","Atlanta","Atlanta","États-Unis","Amérique du Nord",33.64,-84.43,"Long-courrier"),
    ("ORD","Chicago O'Hare","Chicago","États-Unis","Amérique du Nord",41.97,-87.91,"Long-courrier"),
    ("GRU","São Paulo","São Paulo","Brésil","Amérique du Sud",-23.44,-46.47,"Long-courrier"),
    ("GIG","Rio de Janeiro","Rio de Janeiro","Brésil","Amérique du Sud",-22.81,-43.25,"Long-courrier"),
    ("BOG","Bogotá","Bogotá","Colombie","Amérique du Sud",4.70,-74.15,"Long-courrier"),
    ("EZE","Buenos Aires","Buenos Aires","Argentine","Amérique du Sud",-34.82,-58.54,"Long-courrier"),
    ("CMN","Casablanca","Casablanca","Maroc","Afrique",33.37,-7.59,"Moyen-courrier"),
    ("ALG","Alger","Alger","Algérie","Afrique",36.69,3.22,"Moyen-courrier"),
    ("DKR","Dakar Blaise Diagne","Dakar","Sénégal","Afrique",14.67,-17.07,"Long-courrier"),
    ("ABJ","Abidjan","Abidjan","Côte d'Ivoire","Afrique",5.26,-3.93,"Long-courrier"),
    ("JNB","Johannesburg","Johannesburg","Afrique du Sud","Afrique",-26.14,28.25,"Long-courrier"),
    ("NRT","Tokyo Narita","Tokyo","Japon","Asie",35.77,140.39,"Long-courrier"),
    ("PVG","Shanghai Pudong","Shanghai","Chine","Asie",31.14,121.81,"Long-courrier"),
    ("HKG","Hong Kong","Hong Kong","Chine","Asie",22.31,113.92,"Long-courrier"),
    ("ICN","Seoul Incheon","Séoul","Corée du Sud","Asie",37.46,126.44,"Long-courrier"),
    ("BKK","Bangkok","Bangkok","Thaïlande","Asie",13.69,100.75,"Long-courrier"),
    ("SGN","Ho Chi Minh","Ho Chi Minh","Vietnam","Asie",10.82,106.65,"Long-courrier"),
    ("DEL","Delhi","New Delhi","Inde","Asie",28.56,77.10,"Long-courrier"),
    ("SIN","Singapore Changi","Singapour","Singapour","Asie",1.36,103.99,"Long-courrier"),
    ("DXB","Dubai","Dubaï","Émirats Arabes Unis","Moyen-Orient",25.25,55.36,"Long-courrier"),
    ("DOH","Doha Hamad","Doha","Qatar","Moyen-Orient",25.27,51.61,"Long-courrier"),
    ("TLV","Tel Aviv","Tel Aviv","Israël","Moyen-Orient",32.01,34.89,"Moyen-courrier"),
    ("PTP","Pointe-à-Pitre","Pointe-à-Pitre","Guadeloupe","Caraïbes",16.27,-61.53,"Long-courrier"),
    ("FDF","Fort-de-France","Fort-de-France","Martinique","Caraïbes",14.59,-61.00,"Long-courrier"),
    ("RUN","La Réunion Roland Garros","Saint-Denis","La Réunion","Océan Indien",-20.89,55.51,"Long-courrier"),
]

aeroports_df = pd.DataFrame(AEROPORTS_RAW, columns=[
    "code_iata","nom","ville","pays","continent","latitude","longitude","type_aeroport"
])
aeroports_df.to_csv(f"{OUTPUT_DIR}/aeroports.csv", index=False, sep=";")
print(f"   ✅ {len(aeroports_df)} aéroports générés")

apt_dict = aeroports_df.set_index("code_iata").to_dict("index")

# ============================================================
# 2. FLOTTE
# ============================================================
print("\n[2/4] Génération de la FLOTTE...")

TYPES_AVIONS = [
    {"type":"Airbus A220-300","famille":"Narrow-body","moteur":"PW1500G","sieges":148,"business":0,"premium":0,"eco":148,"range_km":6300,"cout_h":4500,"nb":18},
    {"type":"Airbus A318","famille":"Narrow-body","moteur":"CFM56-5B","sieges":131,"business":0,"premium":0,"eco":131,"range_km":6000,"cout_h":5200,"nb":6},
    {"type":"Airbus A319","famille":"Narrow-body","moteur":"CFM56-5A","sieges":142,"business":0,"premium":0,"eco":142,"range_km":6850,"cout_h":5500,"nb":12},
    {"type":"Airbus A320","famille":"Narrow-body","moteur":"CFM56-5B","sieges":178,"business":0,"premium":0,"eco":178,"range_km":6150,"cout_h":5800,"nb":20},
    {"type":"Airbus A321","famille":"Narrow-body","moteur":"V2500","sieges":212,"business":0,"premium":0,"eco":212,"range_km":5950,"cout_h":6200,"nb":10},
    {"type":"Airbus A330-200","famille":"Wide-body","moteur":"CF6-80E1","sieges":224,"business":36,"premium":21,"eco":167,"range_km":13450,"cout_h":9500,"nb":10},
    {"type":"Airbus A350-900","famille":"Wide-body","moteur":"Trent XWB","sieges":324,"business":34,"premium":24,"eco":266,"range_km":15000,"cout_h":11000,"nb":18},
    {"type":"Boeing 777-200ER","famille":"Wide-body","moteur":"GE90-94B","sieges":280,"business":40,"premium":24,"eco":216,"range_km":14300,"cout_h":12000,"nb":12},
    {"type":"Boeing 777-300ER","famille":"Wide-body","moteur":"GE90-115B","sieges":468,"business":58,"premium":28,"eco":382,"range_km":13650,"cout_h":14000,"nb":15},
    {"type":"Boeing 787-9","famille":"Wide-body","moteur":"GEnx-1B","sieges":276,"business":30,"premium":21,"eco":225,"range_km":14140,"cout_h":10500,"nb":10},
]

PREFIXES_REG = {
    "Airbus A220-300":"HZ","Airbus A318":"GU","Airbus A319":"GH",
    "Airbus A320":"GK","Airbus A321":"GT","Airbus A330-200":"GP",
    "Airbus A350-900":"HT","Boeing 777-200ER":"GS","Boeing 777-300ER":"GQ","Boeing 787-9":"HR",
}

flotte = []
for spec in TYPES_AVIONS:
    prefix = PREFIXES_REG[spec["type"]]
    for i in range(spec["nb"]):
        age = round(np.random.uniform(1, 22), 1)
        flotte.append({
            "immatriculation": f"F-{prefix}{chr(65 + i // 26)}{chr(65 + i % 26)}",
            "type_avion": spec["type"],
            "famille": spec["famille"],
            "motorisation": spec["moteur"],
            "nb_sieges_total": spec["sieges"],
            "nb_sieges_business": spec["business"],
            "nb_sieges_premium_eco": spec["premium"],
            "nb_sieges_eco": spec["eco"],
            "age_avion_ans": age,
            "date_livraison": (datetime.now() - timedelta(days=int(age * 365))).strftime("%Y-%m-%d"),
            "statut": np.random.choice(["En service","En service","En service","Maintenance"], p=[0.35,0.35,0.25,0.05]),
            "base": np.random.choice(["CDG","CDG","CDG","ORY"], p=[0.4,0.3,0.15,0.15]),
            "autonomie_km": spec["range_km"],
            "cout_heure_vol_eur": spec["cout_h"],
        })

flotte_df = pd.DataFrame(flotte)
flotte_df.to_csv(f"{OUTPUT_DIR}/flotte.csv", index=False, sep=";")
print(f"   ✅ {len(flotte_df)} avions générés")

# Lookup capacité par type
capacite_map = {s["type"]: s["sieges"] for s in TYPES_AVIONS}
short_haul_types = ["Airbus A220-300","Airbus A318","Airbus A319","Airbus A320","Airbus A321"]
medium_haul_types = ["Airbus A321","Airbus A330-200"]
long_haul_types = ["Airbus A330-200","Airbus A350-900","Boeing 777-200ER","Boeing 777-300ER","Boeing 787-9"]

# ============================================================
# 3. VOLS
# ============================================================
print("\n[3/4] Génération des VOLS...")

ROUTES = [
    ("CDG","MRS",14),("CDG","LYS",12),("CDG","TLS",12),("CDG","NCE",16),("CDG","BOD",10),("CDG","NTE",8),
    ("ORY","MRS",10),("ORY","TLS",8),("ORY","NCE",12),("ORY","BOD",7),("ORY","LYS",7),("ORY","NTE",5),
    ("CDG","LHR",18),("CDG","AMS",14),("CDG","FRA",12),("CDG","FCO",10),("CDG","MAD",10),("CDG","BCN",12),
    ("CDG","LIS",7),("CDG","MUC",8),("CDG","VIE",5),("CDG","ZRH",7),("CDG","CPH",7),("CDG","ARN",5),
    ("CDG","ATH",5),("CDG","IST",7),
    ("CDG","JFK",14),("CDG","LAX",7),("CDG","SFO",5),("CDG","YUL",7),("CDG","MIA",5),
    ("CDG","ATL",4),("CDG","ORD",4),
    ("CDG","GRU",5),("CDG","GIG",3),("CDG","BOG",3),("CDG","EZE",3),
    ("CDG","CMN",10),("CDG","ALG",7),("CDG","DKR",5),("CDG","ABJ",5),("CDG","JNB",4),
    ("CDG","NRT",7),("CDG","PVG",5),("CDG","HKG",5),("CDG","ICN",4),
    ("CDG","BKK",5),("CDG","SGN",4),("CDG","DEL",4),("CDG","SIN",5),
    ("CDG","DXB",7),("CDG","DOH",4),("CDG","TLV",7),
    ("CDG","PTP",5),("CDG","FDF",5),("CDG","RUN",4),
]

# Générer les numéros de vol
flight_nums = {}
fn_counter = 100
for dep, arr, _ in ROUTES:
    if dep == "ORY":
        flight_nums[(dep, arr)] = f"AF{7000 + fn_counter}"
    elif arr in ["MRS","LYS","TLS","NCE","BOD","NTE"]:
        flight_nums[(dep, arr)] = f"AF{7500 + fn_counter}"
    elif haversine(apt_dict[dep]["latitude"], apt_dict[dep]["longitude"],
                   apt_dict[arr]["latitude"], apt_dict[arr]["longitude"]) > 5000:
        flight_nums[(dep, arr)] = f"AF{fn_counter:04d}"
    else:
        flight_nums[(dep, arr)] = f"AF{1000 + fn_counter}"
    fn_counter += np.random.randint(2, 8)

CAUSE_RETARD = ["Météo","Technique","Régulation ATC","Correspondances","Équipage","Aéroport"]
CAUSE_RETARD_P = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
CAUSE_ANNUL = ["Technique","Météo","Grève","Opérationnel"]

all_chunks = []
for dep, arr, freq_week in ROUTES:
    dist = haversine(
        apt_dict[dep]["latitude"], apt_dict[dep]["longitude"],
        apt_dict[arr]["latitude"], apt_dict[arr]["longitude"]
    )
    duration_min = round(dist / 850 * 60 + 30)

    # Type avion selon distance
    if dist < 2000:
        pool = short_haul_types
    elif dist < 5000:
        pool = medium_haul_types
    else:
        pool = long_haul_types

    # Catégorie courrier
    if dist < 2000:
        courrier = "Court-courrier"
    elif dist < 5000:
        courrier = "Moyen-courrier"
    else:
        courrier = "Long-courrier"

    daily_prob = freq_week / 7
    flights_per_day = np.random.poisson(daily_prob, N_DAYS)
    total = flights_per_day.sum()
    if total == 0:
        continue

    dates = np.repeat(
        [DATE_DEBUT + timedelta(days=d) for d in range(N_DAYS)],
        flights_per_day
    )

    months = np.array([d.month for d in dates])

    # Heures de départ
    if dist < 2000:
        dep_h = np.random.uniform(6, 22, total)
    else:
        dep_h = np.where(
            np.random.random(total) < 0.5,
            np.random.uniform(7, 12, total),
            np.random.uniform(19, 23.5, total)
        )

    # Retards (saisonnalité)
    weather_boost = np.where(np.isin(months, [1, 2, 7, 8, 12]), 1.4, 1.0)
    delay_cat = np.random.choice([0, 1, 2, 3], total, p=[0.68, 0.19, 0.09, 0.04])
    base_delays = np.where(delay_cat == 0, np.random.uniform(0, 15, total),
                  np.where(delay_cat == 1, np.random.uniform(15, 60, total),
                  np.where(delay_cat == 2, np.random.uniform(60, 180, total),
                  np.random.uniform(180, 420, total))))
    delays = np.round(base_delays * weather_boost).astype(int)

    cancelled = np.random.random(total) < 0.012
    statut = np.where(cancelled, "Annulé", np.where(delays > 15, "Retardé", "À l'heure"))

    # Taux remplissage saisonnier
    seasonal_lr = np.where(np.isin(months, [6, 7, 8]), 0.90,
                  np.where(np.isin(months, [12]), 0.88,
                  np.where(np.isin(months, [1, 2, 11]), 0.76, 0.82)))
    load_factors = np.clip(np.random.normal(seasonal_lr, 0.07), 0.40, 0.99)

    aircraft = np.random.choice(pool, total)
    capacities = np.array([capacite_map[a] for a in aircraft])
    passengers = np.round(capacities * load_factors).astype(int)

    # Revenue
    if dist < 2000:
        rev_pax = np.random.uniform(80, 280, total)
    elif dist < 5000:
        rev_pax = np.random.uniform(200, 650, total)
    else:
        rev_pax = np.random.uniform(400, 1900, total)
    revenue = np.round(passengers * rev_pax, 2)

    # Carburant & CO2
    fuel_factor = 3.0 if dist > 6000 else 3.5 if dist > 2000 else 4.2
    fuel = dist * fuel_factor * capacities / 100 * np.random.uniform(0.88, 1.12, total)
    co2 = fuel * 0.8 * 3.16 / 1000

    # Causes retard / annulation
    causes = np.where(
        cancelled,
        np.random.choice(CAUSE_ANNUL, total),
        np.where(delays > 15, np.random.choice(CAUSE_RETARD, total, p=CAUSE_RETARD_P), "Aucun")
    )

    continent_arr = apt_dict[arr]["continent"]

    chunk = pd.DataFrame({
        "numero_vol": flight_nums[(dep, arr)],
        "date_vol": dates,
        "aeroport_depart": dep,
        "aeroport_arrivee": arr,
        "continent_destination": continent_arr,
        "courrier": courrier,
        "distance_km": round(dist),
        "duree_prevue_min": duration_min,
        "heure_depart": [f"{int(h):02d}:{int((h % 1) * 60):02d}" for h in dep_h],
        "retard_depart_min": delays,
        "statut_vol": statut,
        "type_avion": aircraft,
        "capacite": capacities,
        "passagers": passengers,
        "taux_remplissage": np.round(load_factors, 3),
        "revenu_vol_eur": np.round(revenue, 2),
        "revenu_par_pax_eur": np.round(rev_pax, 2),
        "carburant_litres": np.round(fuel, 0).astype(int),
        "co2_tonnes": np.round(co2, 2),
        "cause_retard": causes,
    })
    all_chunks.append(chunk)

vols_df = pd.concat(all_chunks, ignore_index=True)
vols_df = vols_df.sort_values("date_vol").reset_index(drop=True)
vols_df.insert(0, "vol_id", [f"VOL-{i:06d}" for i in range(len(vols_df))])
vols_df["annee"] = pd.to_datetime(vols_df["date_vol"]).dt.year
vols_df["mois"] = pd.to_datetime(vols_df["date_vol"]).dt.month
vols_df["rpk"] = (vols_df["passagers"] * vols_df["distance_km"]).astype(np.int64)
vols_df["ask"] = (vols_df["capacite"] * vols_df["distance_km"]).astype(np.int64)

vols_df.to_csv(f"{OUTPUT_DIR}/vols.csv", index=False, sep=";")
print(f"   ✅ {len(vols_df):,} vols générés")

# ============================================================
# 4. SATISFACTION PASSAGERS
# ============================================================
print("\n[4/4] Génération de la SATISFACTION...")

N_SURVEYS = 22000
survey_sample = vols_df.sample(n=N_SURVEYS, random_state=42)

classes = np.random.choice(["Business","Premium Éco","Économique"], N_SURVEYS, p=[0.15, 0.10, 0.75])
class_bonus = np.where(classes == "Business", 0.9, np.where(classes == "Premium Éco", 0.4, 0.0))

delays_survey = survey_sample["retard_depart_min"].values

note_confort = np.clip(np.random.normal(7.2, 1.6, N_SURVEYS) + class_bonus, 1, 10).round(1)
note_repas = np.clip(np.random.normal(6.8, 1.9, N_SURVEYS) + class_bonus, 1, 10).round(1)
note_divertissement = np.clip(np.random.normal(7.0, 1.7, N_SURVEYS) + class_bonus * 0.5, 1, 10).round(1)
note_equipage = np.clip(np.random.normal(8.1, 1.2, N_SURVEYS), 1, 10).round(1)
note_ponctualite = np.clip(10 - delays_survey / 25 + np.random.normal(0, 0.8, N_SURVEYS), 1, 10).round(1)
note_enregistrement = np.clip(np.random.normal(7.5, 1.5, N_SURVEYS), 1, 10).round(1)
note_globale = np.clip(
    (note_confort + note_repas + note_divertissement + note_equipage + note_ponctualite + note_enregistrement) / 6
    + np.random.normal(0, 0.3, N_SURVEYS), 1, 10
).round(1)

nps_score = np.where(note_globale >= 9, "Promoteur",
            np.where(note_globale >= 7, "Passif", "Détracteur"))

satisfaction_df = pd.DataFrame({
    "survey_id": [f"SAT-{i:06d}" for i in range(N_SURVEYS)],
    "vol_id": survey_sample["vol_id"].values,
    "numero_vol": survey_sample["numero_vol"].values,
    "date_vol": survey_sample["date_vol"].values,
    "route": survey_sample["aeroport_depart"].values + " → " + survey_sample["aeroport_arrivee"].values,
    "courrier": survey_sample["courrier"].values,
    "classe": classes,
    "note_globale": note_globale,
    "note_confort": note_confort,
    "note_repas": note_repas,
    "note_divertissement": note_divertissement,
    "note_equipage": note_equipage,
    "note_ponctualite": note_ponctualite,
    "note_enregistrement": note_enregistrement,
    "nps_categorie": nps_score,
    "recommandation": np.random.choice(["Oui","Non","Peut-être"], N_SURVEYS, p=[0.62, 0.12, 0.26]),
    "type_voyageur": np.random.choice(["Affaires","Loisirs","Famille","Solo"], N_SURVEYS, p=[0.30, 0.35, 0.20, 0.15]),
    "programme_fidelite": np.random.choice(
        ["Explorer","Silver","Gold","Platinum","Ultimate","Non inscrit"],
        N_SURVEYS, p=[0.22, 0.13, 0.08, 0.04, 0.01, 0.52]
    ),
    "nationalite": np.random.choice(
        ["Française","Américaine","Britannique","Allemande","Japonaise","Brésilienne","Chinoise","Autre"],
        N_SURVEYS, p=[0.35, 0.14, 0.08, 0.07, 0.05, 0.04, 0.04, 0.23]
    ),
})

satisfaction_df.to_csv(f"{OUTPUT_DIR}/satisfaction.csv", index=False, sep=";")
print(f"   ✅ {len(satisfaction_df):,} enquêtes de satisfaction générées")

# ============================================================
print("\n" + "=" * 70)
print("  GÉNÉRATION TERMINÉE")
print("=" * 70)
print(f"  ✈️  Aéroports     : {len(aeroports_df):>10,}")
print(f"  🛩️  Flotte        : {len(flotte_df):>10,}")
print(f"  🛫 Vols           : {len(vols_df):>10,}")
print(f"  ⭐ Satisfaction   : {len(satisfaction_df):>10,}")
print("=" * 70)