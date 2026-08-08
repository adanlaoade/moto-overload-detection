import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

st.set_page_config(
    page_title="Détection de surcharge sur motos",
    page_icon="🏍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE (charte graphique cohérente avec la présentation)
# ============================================================
st.markdown("""
<style>
    :root {
        --navy: #1A1A2E;
        --terracotta: #D9782D;
        --teal: #2A6F6F;
        --gray: #6B6B76;
        --lightgray: #F2F2F5;
    }

    .stApp {
        background-color: #FFFFFF;
    }

    /* Bandeau titre */
    .hero-banner {
        background: linear-gradient(135deg, #1A1A2E 0%, #2A2A45 100%);
        padding: 2.2rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
    }
    .hero-kicker {
        color: #D9782D;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.15;
    }
    .hero-subtitle {
        color: #C9C9D6;
        font-size: 0.95rem;
        margin-top: 0.6rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(217, 120, 45, 0.15);
        color: #D9782D;
        border: 1px solid rgba(217, 120, 45, 0.4);
        border-radius: 20px;
        padding: 0.25rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.9rem;
    }

    /* Cartes de résultat */
    .result-card {
        border-radius: 12px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.7rem;
        border-left: 5px solid;
    }
    .result-card.surcharge {
        background: #FBEAE8;
        border-left-color: #C0392B;
    }
    .result-card.conforme {
        background: #E9F3EE;
        border-left-color: #2E7D5B;
    }
    .result-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.2rem;
    }
    .result-title.surcharge { color: #C0392B; }
    .result-title.conforme { color: #2E7D5B; }
    .result-detail {
        color: #4A4A55;
        font-size: 0.9rem;
    }

    /* Métriques du bandeau */
    .metric-row {
        display: flex;
        gap: 1.5rem;
        margin-top: 1rem;
    }
    .metric-item {
        color: #FFFFFF;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #D9782D;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8B8B9A;
    }

    section[data-testid="stSidebar"] {
        background-color: #F8F8FA;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CHARGEMENT DES MODÈLES
# ============================================================
@st.cache_resource
def charger_modeles():
    return YOLO("yolov8n-pose.pt"), YOLO("yolov8n.pt")

model_pose, model_detect = charger_modeles()

L_HIP, R_HIP = 11, 12
MAX_PASSAGERS_PAR_MOTO = 6

# ============================================================
# PIPELINE VALIDÉ : pose + normalisation d'échelle + hongrois + déduplication
# ============================================================

def get_person_anchor(kpts, conf_thresh=0.3):
    hips = [kpts[i] for i in (L_HIP, R_HIP) if kpts[i][2] > conf_thresh]
    if not hips:
        return None
    x = np.mean([p[0] for p in hips])
    y = np.mean([p[1] for p in hips])
    return x, y

def association_cost(person_anchor, moto_box):
    x1, y1, x2, y2 = moto_box
    moto_w, moto_h = x2 - x1, y2 - y1
    moto_cx, moto_cy = (x1 + x2) / 2, (y1 + y2) / 2
    px, py = person_anchor
    dx_norm = abs(px - moto_cx) / moto_w
    dy_norm = (py - moto_cy) / moto_h
    if dx_norm > 1.2 or dy_norm < -0.5 or dy_norm > 1.5:
        return None
    return dx_norm + max(0, dy_norm)

def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aire1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    aire2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = aire1 + aire2 - inter
    return inter / union if union > 0 else 0

def dedupliquer_personnes(boxes, confs, seuil_iou=0.5):
    indices_tries = sorted(range(len(boxes)), key=lambda i: confs[i], reverse=True)
    gardees = []
    for i in indices_tries:
        est_doublon = any(iou(boxes[i], boxes[j]) > seuil_iou for j in gardees)
        if not est_doublon:
            gardees.append(i)
    return gardees

def match_people_to_motos(people_kpts, moto_boxes):
    anchors = [get_person_anchor(k) for k in people_kpts]
    valid_idx = [i for i, a in enumerate(anchors) if a is not None]
    n_p, n_m = len(valid_idx), len(moto_boxes)
    if n_p == 0 or n_m == 0:
        return {j: [] for j in range(n_m)}

    cost_reel = np.full((n_p, n_m), 1e6)
    for i_row, i in enumerate(valid_idx):
        for j, moto_box in enumerate(moto_boxes):
            c = association_cost(anchors[i], moto_box)
            if c is not None:
                cost_reel[i_row, j] = c

    cost_dupl = np.repeat(cost_reel, MAX_PASSAGERS_PAR_MOTO, axis=1)
    row_ind, col_ind = linear_sum_assignment(cost_dupl)

    assignments = {j: [] for j in range(n_m)}
    for r, c in zip(row_ind, col_ind):
        moto_idx = c // MAX_PASSAGERS_PAR_MOTO
        if cost_reel[r, moto_idx] < 1e6:
            assignments[moto_idx].append(valid_idx[r])
    return assignments

# ============================================================
# FONCTION PRINCIPALE DU PIPELINE
# ============================================================

def analyser_image(image_pil, seuil_surcharge, conf_min_moto):
    img = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    results_pose = model_pose(img, verbose=False)
    people_kpts, boxes_p, confs_p = [], [], []
    if results_pose[0].keypoints is not None and len(results_pose[0].keypoints) > 0:
        people_kpts = [kp.data[0].cpu().numpy() for kp in results_pose[0].keypoints]
        boxes_p = [b.xyxy[0].tolist() for b in results_pose[0].boxes]
        confs_p = [float(b.conf[0]) for b in results_pose[0].boxes]

    if boxes_p:
        indices_valides = dedupliquer_personnes(boxes_p, confs_p, seuil_iou=0.5)
        people_kpts = [people_kpts[i] for i in indices_valides]
        boxes_p = [boxes_p[i] for i in indices_valides]

    results_detect = model_detect(img, verbose=False)
    moto_boxes = [b.xyxy[0].tolist() for b in results_detect[0].boxes
                  if model_detect.names[int(b.cls[0])] == "motorcycle" and float(b.conf[0]) >= conf_min_moto]

    img_annotated = img.copy()
    motos_info = []

    if not moto_boxes:
        img_annotated_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
        return img_annotated_rgb, []

    assignments = match_people_to_motos(people_kpts, moto_boxes)

    for i, m_box in enumerate(moto_boxes):
        x1, y1, x2, y2 = [int(v) for v in m_box]
        nb_personnes = len(assignments.get(i, []))
        statut = "surcharge" if nb_personnes >= seuil_surcharge else "conforme"
        couleur = (0, 0, 255) if statut == "surcharge" else (0, 200, 0)

        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), couleur, 3)
        label = f"Moto {i+1}: {nb_personnes}p"
        cv2.putText(img_annotated, label, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)

        for idx in assignments.get(i, []):
            px1, py1, px2, py2 = [int(v) for v in boxes_p[idx]]
            cv2.rectangle(img_annotated, (px1, py1), (px2, py2), (255, 150, 0), 2)

        motos_info.append({"num": i + 1, "nb": nb_personnes, "statut": statut})

    img_annotated_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
    return img_annotated_rgb, motos_info

# ============================================================
# INTERFACE
# ============================================================

st.markdown("""
<div class="hero-banner">
    <div class="hero-kicker">Vision artificielle — Bénin</div>
    <p class="hero-title">🏍️ Détection de surcharge sur motos</p>
    <p class="hero-subtitle">YOLOv8-pose + assignment optimal — zemidjan et motocyclistes au Bénin</p>
    <span class="hero-badge">Sans entraînement spécifique (training-free)</span>
    <div class="metric-row">
        <div class="metric-item"><div class="metric-value">66,4%</div><div class="metric-label">Exactitude</div></div>
        <div class="metric-item"><div class="metric-value">92,7%</div><div class="metric-label">Précision</div></div>
        <div class="metric-item"><div class="metric-value">57,3%</div><div class="metric-label">Rappel</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    seuil_surcharge = st.slider("Seuil de surcharge (personnes)", 2, 5, 3,
                                 help="Nombre total de personnes, conducteur inclus, à partir duquel une moto est classée en surcharge.")
    conf_min_moto = st.slider("Confiance minimale (moto)", 0.10, 0.60, 0.20, 0.05,
                               help="Seuil calibré empiriquement à 0,20 sur 250 images annotées.")
    st.markdown("---")
    st.markdown("**Projet AMA — Groupe 1**")
    st.caption("LUBUNGU KISAMBULA Pacifique\n\nADANLAO Adéyinka Laurinda")

uploaded_file = st.file_uploader("📤 Upload une image de moto-taxi", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    with st.spinner("Analyse en cours..."):
        img_result, motos_info = analyser_image(image_pil, seuil_surcharge, conf_min_moto)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(img_result, caption="Résultat annoté", use_container_width=True)

    with col2:
        st.markdown("#### Résultats")
        if not motos_info:
            st.warning("Aucune moto exploitable détectée dans cette image.")
        for m in motos_info:
            label = "🔴 SURCHARGE" if m["statut"] == "surcharge" else "🟢 CONFORME"
            st.markdown(f"""
            <div class="result-card {m['statut']}">
                <div class="result-title {m['statut']}">Moto {m['num']} — {label}</div>
                <div class="result-detail">{m['nb']} personne(s) détectée(s)</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("👆 Uploade une image pour lancer l'analyse.")

st.divider()
st.caption("Système évalué sur un corpus externe de 250 images annotées (voir article scientifique). Positionné comme outil d'aide à la décision, non comme dispositif de verbalisation automatique.")
