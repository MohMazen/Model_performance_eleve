
import io
import os
import sys
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

logger = logging.getLogger(__name__)

# Permettre l'import depuis la racine du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score
)
from sklearn.model_selection import train_test_split

from src.config import COLS_TO_DROP, TARGET_CLF, TARGET_REG, MODEL_FILE, SEUIL_REUSSITE
from src.data_utils import charger_donnees, generer_donnees_synthetiques, nettoyer_donnees, valider_schema
from src.explainability import generate_shap_analysis, generate_shap_failure_analysis
from src.features import add_advanced_features, nettoyer_horaires, get_column_mapping
from src.models import ModelManager
from src.reporting import generer_rapport_markdown
from app.utils_st import _get, _set, cached_generer_donnees

st.sidebar.title("🎓 EduStats")
st.sidebar.caption("Analyse Prédictive des Performances Scolaires v3.0")

st.title("📂 Données")

_QUESTIONNAIRE_FILE = os.path.join('data', 'questionnaire_responses.csv')

_FREQ_5 = ['jamais', 'rarement', 'parfois', 'souvent', 'toujours']
_FREQ_MEDIATHEQUE = ['jamais', 'rarement', 'parfois', 'souvent', 'plusieurs_fois_semaine']
_ETATS_PSY = ['serein', 'amoureux', 'anxiete', 'deprime', 'depression', 'autre']
_SOUTIEN_POOL = ['francais', 'maths', 'sciences', 'anglais', 'histoire_geo']
_STRESS_LABELS = {0: "Jamais", 1: "Presque jamais", 2: "Parfois", 3: "Assez souvent", 4: "Très souvent"}

tab1, tab2, tab3 = st.tabs(["🔧 Données synthétiques", "📁 Charger un fichier", "📝 Questionnaire individuel"])

# ── Tab 1 : Données synthétiques ──────────────────────────────────────────────
with tab1:
    st.subheader("Générer des données synthétiques")
    n_eleves = st.number_input("Nombre d'élèves", min_value=50, max_value=2000, value=300, step=50)

    classes_dispo = [
        "Sixième", "Cinquième", "Quatrième", "Troisième",
        "Seconde", "Première", "Terminale"
    ]
    classes_sel = st.multiselect(
        "Classes à inclure",
        options=classes_dispo,
        default=["Quatrième", "Troisième"]
    )

    if st.button("🔄 Générer", key="btn_gen"):
        if not classes_sel:
            st.error("⚠️ Veuillez sélectionner au moins une classe.")
        else:
            with st.spinner("Génération en cours…"):
                df = cached_generer_donnees(int(n_eleves), classes_selectionnees=classes_sel)
            _set("df_raw", df)
            _set("data_source", "synthetic")
            _set("df_clean", None)
            _set("df_feat", None)
            _set("models", None)
            st.success(f"✅ {len(df)} élèves générés ({', '.join(classes_sel)}).")

# ── Tab 2 : Chargement CSV ────────────────────────────────────────────────────
with tab2:
    st.subheader("Charger un fichier CSV")
    uploaded = st.file_uploader("Fichier CSV (séparateur ',' ou ';')", type=["csv"])

    last_uploaded_name = _get("last_uploaded_name")

    if uploaded is not None and uploaded.name != last_uploaded_name:
        try:
            df_up = pd.read_csv(uploaded, sep=None, engine='python', encoding='utf-8-sig')
            df_up.columns = [c.lower() for c in df_up.columns]
            _set("df_raw", df_up)
            _set("data_source", "uploaded")
            _set("df_clean", None)
            _set("df_feat", None)
            _set("models", None)
            _set("last_uploaded_name", uploaded.name)
            st.success(f"✅ Fichier chargé : {df_up.shape[0]} lignes, {df_up.shape[1]} colonnes.")
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")

# ── Tab 3 : Questionnaire individuel ─────────────────────────────────────────
with tab3:
    st.subheader("Saisir le questionnaire d'un élève")
    st.info(
        "Remplissez le formulaire ci-dessous. Chaque validation ajoute une ligne au fichier "
        f"`{_QUESTIONNAIRE_FILE}`. Les données cumulées sont automatiquement chargées pour l'analyse."
    )

    # Afficher le nombre de réponses déjà enregistrées
    if os.path.exists(_QUESTIONNAIRE_FILE):
        try:
            _existing = pd.read_csv(_QUESTIONNAIRE_FILE, sep=';', encoding='utf-8-sig')
            st.caption(f"📋 {len(_existing)} réponse(s) déjà enregistrée(s) dans le fichier.")
        except Exception:
            pass

    with st.form("questionnaire_form", border=True):

        # ── Section 1 : État civil ────────────────────────────────────────
        st.markdown("#### 1. État civil")
        c1, c2, c3 = st.columns(3)
        nom = c1.text_input("Nom", placeholder="NOM")
        prenom = c2.text_input("Prénom", placeholder="Prénom")
        age = c3.number_input("Âge", min_value=10, max_value=20, value=15)

        c4, c5, c6 = st.columns(3)
        classe = c4.selectbox("Classe", ['6eme', '5eme', '4eme', '3eme', '2nde', '1ere', 'terminale'])
        etablissement = c5.text_input("Établissement", placeholder="Nom de l'établissement")
        duree_trajet = c6.number_input("Trajet A/R (min)", min_value=0, max_value=300, value=30)
        adresse = st.text_input("Adresse (optionnel)", placeholder="Rue, ville…")

        st.divider()

        # ── Section 2 : Organisation ──────────────────────────────────────
        st.markdown("#### 2. Capacités organisationnelles")
        c1, c2 = st.columns(2)
        organisation = c1.slider("Organisation générale (1 = faible, 10 = excellente)", 1, 10, 5)
        gestion_temps = c2.radio("Gestion du temps", ['faible', 'moyen', 'fort'], horizontal=True)

        st.divider()

        # ── Section 3 : Motivation ────────────────────────────────────────
        st.markdown("#### 3. Motivation et engagement")
        st.markdown("**Motivation par matière (0 = aucun intérêt, 10 = très intéressé)**")
        _matieres = ['francais', 'maths', 'hgemc', 'anglais', 'arabe', 'sciences', 'eps']
        _labels_mat = ['Français', 'Maths', 'HG-EMC', 'Anglais', 'Arabe', 'Sciences', 'EPS']
        mot_cols = st.columns(4)
        mot_vals = {}
        for i, (m, lbl) in enumerate(zip(_matieres, _labels_mat)):
            mot_vals[m] = mot_cols[i % 4].slider(lbl, 0, 10, 5, key=f"mot_{m}")

        mot_ens_sci = st.slider(
            "Motivation enseignement scientifique (lycée uniquement, 0 si non concerné)", 0, 10, 0
        )

        st.markdown("**Spécialités Première** (laisser vide si non concerné)")
        spe_cols = st.columns(3)
        spe1_noms, spe1_mots = [], []
        for i in range(1, 4):
            spe1_noms.append(spe_cols[i - 1].text_input(f"Spé {i} — Nom", key=f"spe1ere_{i}_nom"))
            spe1_mots.append(spe_cols[i - 1].slider(f"Spé {i} — Motivation", 0, 10, 0, key=f"spe1ere_{i}_mot"))

        st.markdown("**Spécialités Terminale** (laisser vide si non concerné)")
        speterm_cols = st.columns(2)
        speterm_noms, speterm_mots = [], []
        for i in range(1, 3):
            speterm_noms.append(speterm_cols[i - 1].text_input(f"Spé Term {i} — Nom", key=f"speterm_{i}_nom"))
            speterm_mots.append(speterm_cols[i - 1].slider(f"Spé Term {i} — Motivation", 0, 10, 0, key=f"speterm_{i}_mot"))

        c1, c2 = st.columns(2)
        mot_famille = c1.slider("Motivation par les encouragements de la famille (0-10)", 0, 10, 5)
        mot_recompenses = c2.slider("Motivation par les récompenses/notes (0-10)", 0, 10, 5)

        st.markdown("**Persévérance — Grit** (1 = pas du tout d'accord, 5 = tout à fait d'accord)")
        gc1, gc2, gc3 = st.columns(3)
        grit1 = gc1.slider("Je termine ce que je commence", 1, 5, 3, key="grit1")
        grit2 = gc2.slider("Je travaille dur même si c'est difficile", 1, 5, 3, key="grit2")
        grit3 = gc3.slider("Je persévère malgré les échecs", 1, 5, 3, key="grit3")

        st.divider()

        # ── Section 4 : Temps d'étude et écrans ──────────────────────────
        st.markdown("#### 4. Temps d'étude et écrans")
        ec1, ec2, ec3, ec4, ec5 = st.columns(5)
        h_etude = ec1.number_input("Étude (h/soir)", 0.0, 8.0, 2.0, 0.5)
        h_jeux = ec2.number_input("Jeux vidéo (h/j)", 0.0, 12.0, 1.0, 0.5)
        h_reseaux = ec3.number_input("Réseaux sociaux (h/j)", 0.0, 12.0, 1.0, 0.5)
        h_streaming = ec4.number_input("Streaming (h/j)", 0.0, 12.0, 1.0, 0.5)
        h_sites_educ = ec5.number_input("Sites éducatifs (h/j)", 0.0, 8.0, 0.5, 0.5)

        st.divider()

        # ── Section 5 : Sommeil et bien-être ─────────────────────────────
        st.markdown("#### 5. Sommeil et bien-être")
        sc1, sc2 = st.columns(2)
        h_sommeil = sc1.number_input("Heures de sommeil/nuit", 4.0, 12.0, 8.0, 0.5)
        qualite_sommeil = sc2.slider("Qualité du sommeil (1 = très mauvaise, 10 = excellente)", 1, 10, 7)

        _coucher_opts = (
            [f"{h:02d}:{m:02d}" for h in range(19, 24) for m in [0, 30]]
            + [f"{h:02d}:{m:02d}" for h in range(0, 4) for m in [0, 30]]
        )
        _lever_opts = [f"{h:02d}:{m:02d}" for h in range(4, 11) for m in [0, 30]]
        hc1, hc2 = st.columns(2)
        heure_coucher = hc1.selectbox("Heure de coucher habituelle", _coucher_opts,
                                       index=_coucher_opts.index("22:00"))
        heure_lever = hc2.selectbox("Heure de lever habituelle", _lever_opts,
                                     index=_lever_opts.index("07:00"))

        st.markdown("**Gestion du stress** (PSS-2)")
        stc1, stc2 = st.columns(2)
        stress1 = stc1.select_slider(
            "Contrarié par des événements inattendus",
            options=[0, 1, 2, 3, 4], value=1,
            format_func=lambda x: _STRESS_LABELS[x]
        )
        stress2 = stc2.select_slider(
            "Sentiment d'être incapable de contrôler les choses importantes",
            options=[0, 1, 2, 3, 4], value=1,
            format_func=lambda x: _STRESS_LABELS[x]
        )

        st.divider()

        # ── Section 6 : Nutrition et activité physique ────────────────────
        st.markdown("#### 6. Nutrition et activité physique")
        nc1, nc2 = st.columns(2)
        nb_repas = nc1.radio("Nombre de repas par jour", ['1', '2', '3', 'plus'], horizontal=True)
        repas_eq = nc2.radio("Fréquence des repas équilibrés",
                              ['jamais', 'rarement', 'parfois', 'souvent', 'toujours'], horizontal=True)

        nc3, nc4, nc5 = st.columns(3)
        activite_sport = nc3.radio("Activité sportive extra-scolaire", ['oui', 'non'], horizontal=True)
        h_activite = nc4.number_input("Heures d'activité physique/semaine", 0.0, 20.0, 2.0, 0.5)
        niveau_sport = nc5.radio("Niveau sportif", ['debutant', 'intermediaire', 'confirme'], horizontal=True)

        st.divider()

        # ── Section 7 : Relations et soutien ─────────────────────────────
        st.markdown("#### 7. Relations et soutien")
        rc1, rc2 = st.columns(2)
        pref_travail = rc1.radio("Préférence de travail", ['seul', 'groupe', 'mixte'], horizontal=True)
        soutien_mutuel = rc2.radio("Participation à des groupes de soutien mutuel",
                                    ['jamais', 'rarement', 'parfois', 'souvent'], horizontal=True)

        rc3, rc4, rc5, rc6 = st.columns(4)
        tuteur = rc3.radio("Bénéficiez-vous d'un tuteur ?", ['oui', 'non'], horizontal=True)
        qualite_tuteur = rc4.slider("Qualité du tuteur (0 si aucun)", 0, 10, 0)
        mentor = rc5.radio("Bénéficiez-vous d'un mentor ?", ['oui', 'non'], horizontal=True)
        qualite_mentor = rc6.slider("Qualité du mentor (0 si aucun)", 0, 10, 0)

        st.divider()

        # ── Section 8 : Confiance et psychologie ─────────────────────────
        st.markdown("#### 8. Confiance et psychologie")
        pc1, pc2 = st.columns(2)
        confiance_soi = pc1.slider("Confiance en soi (1-10)", 1, 10, 7)
        estime_soi = pc2.slider("Estime de soi (1-10)", 1, 10, 7)

        pc3, pc4 = st.columns(2)
        evitement = pc3.radio("J'évite les tâches difficiles",
                               ['jamais', 'rarement', 'parfois', 'souvent', 'toujours'], horizontal=True)
        abandon = pc4.radio("J'abandonne face aux obstacles",
                             ['jamais', 'rarement', 'parfois', 'souvent', 'toujours'], horizontal=True)

        etat_psy = st.selectbox("État psychologique dominant", _ETATS_PSY)
        autre_etat = st.text_input("Précisez si 'autre'", placeholder="Décrivez l'état…")

        pc5, pc6, pc7 = st.columns(3)
        suivi_psy = pc5.radio("Suivi psychologique en cours ?", ['oui', 'non'], horizontal=True)
        stress_examens = pc6.slider("Niveau de stress aux examens (1-10)", 1, 10, 5)
        pression_familiale = pc7.slider("Pression familiale perçue (1-10)", 1, 10, 5)

        st.divider()

        # ── Section 9 : Environnement de travail ──────────────────────────
        st.markdown("#### 9. Environnement de travail")
        en1, en2, en3 = st.columns(3)
        bureau_perso = en1.radio("Bureau personnel disponible ?", ['oui', 'non'], horizontal=True)
        calme_maison = en2.slider("Calme à la maison (1-10)", 1, 10, 7)
        lumiere = en3.radio("Lumière adaptée pour étudier ?", ['oui', 'non'], horizontal=True)

        en4, en5, en6 = st.columns(3)
        perturbations = en4.radio("Fréquence des perturbations", _FREQ_5, horizontal=True)
        temps_libre = en5.slider("Temps libre disponible (h/j)", 0, 10, 3)
        taches_menageres = en6.radio("Fréquence des tâches ménagères", _FREQ_5, horizontal=True)

        heures_garde = st.number_input("Heures de garde de frères/sœurs par semaine", 0, 40, 0)

        soutien_mat = st.multiselect(
            "Matières bénéficiant d'un soutien scolaire",
            _SOUTIEN_POOL,
            help="Sélectionner les matières concernées."
        )

        st.markdown("**Heures de soutien par matière/semaine** (0 si non concerné)")
        sh_cols = st.columns(5)
        sh_labels = ['Français', 'Maths', 'Sciences', 'Anglais', 'Histoire-Géo']
        soutien_h = {}
        for i, m in enumerate(_SOUTIEN_POOL):
            soutien_h[m] = sh_cols[i].number_input(sh_labels[i], 0, 8, 0, key=f"hs_{m}")

        en7, en8, en9 = st.columns(3)
        abonnement = en7.radio("Abonnement à une plateforme éducative ?", ['oui', 'non'], horizontal=True)
        nom_plateforme = en8.text_input("Nom de la plateforme (si oui)", placeholder="Kartable, Khan Academy…")
        freq_mediath = en9.radio("Fréquence de visite à la médiathèque", _FREQ_MEDIATHEQUE, horizontal=True)

        st.divider()

        # ── Notes ──────────────────────────────────────────────────────────
        st.markdown("#### Notes obtenues (/20)")
        no1, no2, no3, no4 = st.columns(4)
        note_fr = no1.number_input("Français", 0.0, 20.0, 10.0, 0.5)
        note_ma = no2.number_input("Maths", 0.0, 20.0, 10.0, 0.5)
        note_hg = no3.number_input("Histoire-Géo", 0.0, 20.0, 10.0, 0.5)
        note_sc = no4.number_input("Sciences", 0.0, 20.0, 10.0, 0.5)

        st.markdown("**Notes de spécialités** (laisser à 0 si non concerné — sera ignoré si le nom de spécialité est vide)")
        nos1, nos2, nos3 = st.columns(3)
        note_spe1 = nos1.number_input("Spé 1ère 1", 0.0, 20.0, 0.0, 0.5, key="n_spe1")
        note_spe2 = nos2.number_input("Spé 1ère 2", 0.0, 20.0, 0.0, 0.5, key="n_spe2")
        note_spe3 = nos3.number_input("Spé 1ère 3", 0.0, 20.0, 0.0, 0.5, key="n_spe3")

        nost1, nost2 = st.columns(2)
        note_spet1 = nost1.number_input("Spé Term 1", 0.0, 20.0, 0.0, 0.5, key="n_spet1")
        note_spet2 = nost2.number_input("Spé Term 2", 0.0, 20.0, 0.0, 0.5, key="n_spet2")

        submitted = st.form_submit_button("✅ Valider et sauvegarder", type="primary", use_container_width=True)

    # ── Traitement après soumission ───────────────────────────────────────────
    if submitted:
        # Agrégats calculés
        perseverance = round((grit1 + grit2 + grit3) / 3)
        stress_personnel = round((stress1 + stress2) / 2)

        # Notes spécialités : np.nan si aucun nom renseigné
        ns1 = note_spe1 if spe1_noms[0].strip() else np.nan
        ns2 = note_spe2 if spe1_noms[1].strip() else np.nan
        ns3 = note_spe3 if spe1_noms[2].strip() else np.nan
        nst1 = note_spet1 if speterm_noms[0].strip() else np.nan
        nst2 = note_spet2 if speterm_noms[1].strip() else np.nan

        # Moyenne des notes (NaN ignorés)
        toutes_notes = [note_fr, note_ma, note_hg, note_sc, ns1, ns2, ns3, nst1, nst2]
        note_moyenne = float(np.nanmean(toutes_notes))

        row = {
            # Section 1
            'nom': nom, 'prenom': prenom, 'adresse': adresse,
            'age': age, 'classe': classe, 'etablissement': etablissement,
            'duree_trajet_ar_min': duree_trajet,
            # Section 2
            'organisation': organisation, 'gestion_temps': gestion_temps,
            # Section 3
            'motivation_francais': mot_vals['francais'],
            'motivation_maths': mot_vals['maths'],
            'motivation_hgemc': mot_vals['hgemc'],
            'motivation_anglais': mot_vals['anglais'],
            'motivation_arabe': mot_vals['arabe'],
            'motivation_sciences': mot_vals['sciences'],
            'motivation_eps': mot_vals['eps'],
            'motivation_enseignement_scientifique': mot_ens_sci,
            'specialite1ere_1_nom': spe1_noms[0], 'specialite1ere_1_motivation': spe1_mots[0],
            'specialite1ere_2_nom': spe1_noms[1], 'specialite1ere_2_motivation': spe1_mots[1],
            'specialite1ere_3_nom': spe1_noms[2], 'specialite1ere_3_motivation': spe1_mots[2],
            'specialiteterm_1_nom': speterm_noms[0], 'specialiteterm_1_motivation': speterm_mots[0],
            'specialiteterm_2_nom': speterm_noms[1], 'specialiteterm_2_motivation': speterm_mots[1],
            'motivation_famille': mot_famille,
            'motivation_recompenses': mot_recompenses,
            'grit1': grit1, 'grit2': grit2, 'grit3': grit3,
            'perseverance': perseverance,
            # Section 4
            'heures_etude_soir': h_etude,
            'heures_jeux_video': h_jeux,
            'heures_reseaux_sociaux': h_reseaux,
            'heures_streaming': h_streaming,
            'heures_sites_educatifs': h_sites_educ,
            # Section 5
            'heures_sommeil': h_sommeil,
            'qualite_sommeil': qualite_sommeil,
            'heure_coucher': heure_coucher,
            'heure_lever': heure_lever,
            'stress1': stress1, 'stress2': stress2,
            'stress_personnel': stress_personnel,
            # Section 6
            'nb_repas': nb_repas,
            'repas_equilibres': repas_eq,
            'activite_sportive': activite_sport,
            'heures_activite_physique': h_activite,
            'niveau_sportif': niveau_sport,
            # Section 7
            'pref_travail': pref_travail,
            'soutien_mutuel': soutien_mutuel,
            'tuteur': tuteur,
            'qualite_tuteur': qualite_tuteur if tuteur == 'oui' else 0,
            'mentor': mentor,
            'qualite_mentor': qualite_mentor if mentor == 'oui' else 0,
            # Section 8
            'confiance_soi': confiance_soi,
            'estime_soi': estime_soi,
            'evitement': evitement,
            'abandon': abandon,
            'etat_psychologique': etat_psy,
            'autre_etat_psy': autre_etat if etat_psy == 'autre' else '',
            'suivi_psy': suivi_psy,
            'stress_examens': stress_examens,
            'pression_familiale': pression_familiale,
            # Section 9
            'bureau_personnel': bureau_perso,
            'calme_maison': calme_maison,
            'lumiere_adaptee': lumiere,
            'perturbations': perturbations,
            'temps_libre': temps_libre,
            'taches_menageres': taches_menageres,
            'heures_garde_freres_soeurs': heures_garde,
            'soutien_matieres': ','.join(soutien_mat),
            'heures_soutien_francais': soutien_h['francais'] if 'francais' in soutien_mat else 0,
            'heures_soutien_maths': soutien_h['maths'] if 'maths' in soutien_mat else 0,
            'heures_soutien_sciences': soutien_h['sciences'] if 'sciences' in soutien_mat else 0,
            'heures_soutien_anglais': soutien_h['anglais'] if 'anglais' in soutien_mat else 0,
            'heures_soutien_histoire_geo': soutien_h['histoire_geo'] if 'histoire_geo' in soutien_mat else 0,
            'abonnement_plateforme': abonnement,
            'nom_plateforme': nom_plateforme if abonnement == 'oui' else '',
            'frequence_mediatheque': freq_mediath,
            # Notes
            'note_francais': note_fr,
            'note_maths': note_ma,
            'note_histoire_geo': note_hg,
            'note_sciences': note_sc,
            'note_specialite1ere_1': ns1,
            'note_specialite1ere_2': ns2,
            'note_specialite1ere_3': ns3,
            'note_specialiteterm_1': nst1,
            'note_specialiteterm_2': nst2,
            'note_moyenne': round(note_moyenne, 2),
        }

        df_row = pd.DataFrame([row])

        os.makedirs('data', exist_ok=True)
        if os.path.exists(_QUESTIONNAIRE_FILE):
            try:
                df_existing = pd.read_csv(_QUESTIONNAIRE_FILE, sep=';', encoding='utf-8-sig')
                df_combined = pd.concat([df_existing, df_row], ignore_index=True)
            except Exception:
                df_combined = df_row
        else:
            df_combined = df_row

        df_combined.to_csv(_QUESTIONNAIRE_FILE, sep=';', index=False, encoding='utf-8-sig')

        # Charger dans la session pour l'analyse
        _set("df_raw", df_combined)
        _set("data_source", "questionnaire")
        _set("df_clean", None)
        _set("df_feat", None)
        _set("models", None)

        label = f"{prenom} {nom}".strip() or "l'élève"
        st.success(
            f"✅ Réponse de **{label}** enregistrée. "
            f"Fichier `{_QUESTIONNAIRE_FILE}` : **{len(df_combined)} ligne(s)**. "
            "Données disponibles pour l'analyse."
        )

# ── Aperçu des données (commun aux 3 modes) ───────────────────────────────────
df = _get("df_raw")
if df is not None:
    st.markdown("---")
    st.subheader("Aperçu des données")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe(), use_container_width=True)
    with col2:
        st.subheader("Valeurs manquantes")
        na_df = df.isnull().sum().rename("NaN").reset_index()
        na_df.columns = ["Colonne", "Valeurs manquantes"]
        na_df = na_df[na_df["Valeurs manquantes"] > 0]
        if na_df.empty:
            st.info("Aucune valeur manquante.")
        else:
            st.dataframe(na_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Visualisations des données")

    cols_a_exclure = ['nom', 'prenom', 'prénom', 'prenoms', 'prénoms', 'Nom', 'Prenom', 'Adresse', 'id', 'mail']
    all_cols = [c for c in df.columns if str(c).lower() not in [x.lower() for x in cols_a_exclure]]

    if all_cols:
        palette = px.colors.qualitative.Prism
        for i in range(0, len(all_cols), 2):
            cols = st.columns(2)
            for j, col_name in enumerate(all_cols[i:i + 2]):
                idx = i + j
                couleur = palette[idx % len(palette)]
                with cols[j]:
                    unique_vals = df[col_name].nunique()
                    if pd.api.types.is_datetime64_any_dtype(df[col_name]) or 'date' in str(col_name).lower():
                        vc = df[col_name].value_counts().sort_index().reset_index(name="count")
                        fig = px.line(
                            vc, x=col_name, y="count",
                            title=f"Évolution de {col_name}",
                            labels={col_name: col_name, "count": "Nombre"},
                            color_discrete_sequence=[couleur]
                        )
                    elif pd.api.types.is_numeric_dtype(df[col_name]) and unique_vals > 10:
                        fig = px.histogram(
                            df, x=col_name,
                            color_discrete_sequence=[couleur],
                            title=f"Distribution de {col_name}",
                            labels={col_name: col_name, "count": "Nombre"}
                        )
                        fig.update_traces(marker_line_width=1, marker_line_color="white")
                    elif unique_vals <= 10:
                        vc = df[col_name].value_counts().reset_index(name="count")
                        fig = px.pie(
                            vc, names=col_name, values="count",
                            title=f"Répartition de {col_name}",
                            hole=0.3
                        )
                    else:
                        vc = df[col_name].value_counts().reset_index(name="count")
                        if col_name in ['heure_lever', 'heure_coucher']:
                            vc = vc.sort_values(by=col_name)
                        else:
                            vc = vc.sort_values(by="count", ascending=False).head(20)
                        fig = px.bar(
                            vc, x=col_name, y="count",
                            color="count",
                            color_continuous_scale='Plasma',
                            title=f"Distribution de {col_name}",
                            labels={col_name: col_name, "count": "Nombre"}
                        )
                    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Générez des données synthétiques, chargez un fichier CSV, ou remplissez le questionnaire pour commencer.")


# ---------------------------------------------------------------------------
# PAGE 2 : Preprocessing
# ---------------------------------------------------------------------------
