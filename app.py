import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Détection de surcharge sur motos", layout="centered")

@st.cache_resource
def charger_modele():
    return YOLO("yolov8n.pt")

model = charger_modele()

def centre(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)/2, (y1+y2)/2)

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def aire(box):
    x1, y1, x2, y2 = box
    return (x2-x1) * (y2-y1)

def analyser_image(image_pil, seuil_surcharge, conf_min_detection):
    img = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_h, img_w = img.shape[:2]
    aire_image = img_h * img_w

    results = model(img)
    boxes_filtered = [b for b in results[0].boxes if float(b.conf[0]) >= conf_min_detection]

    motos_avec_conf = []
    personnes = []
    for box in boxes_filtered:
        cls_name = model.names[int(box.cls[0])]
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        if cls_name == "motorcycle":
            motos_avec_conf.append((xyxy, conf))
        elif cls_name == "person":
            personnes.append(xyxy)

    SEUIL_TAILLE_MIN = 0.015
    SEUIL_CONFIANCE_BASSE = 0.55

    motos_valides = []
    for m_box, conf in motos_avec_conf:
        ratio = aire(m_box) / aire_image
        if ratio >= SEUIL_TAILLE_MIN:
            fiabilite = "fiable" if conf >= SEUIL_CONFIANCE_BASSE else "à vérifier"
            motos_valides.append((m_box, fiabilite))

    associations = {i: [] for i in range(len(motos_valides))}
    for p_box in personnes:
        centre_p = centre(p_box)
        meilleure_moto, meilleure_distance = None, float("inf")
        for i, (m_box, _) in enumerate(motos_valides):
            d = distance(centre_p, centre(m_box))
            if d < meilleure_distance:
                meilleure_distance = d
                meilleure_moto = i
        if meilleure_moto is not None:
            associations[meilleure_moto].append(p_box)

    img_annotated = img.copy()
    resume_lignes = []

    for i, (m_box, fiabilite) in enumerate(motos_valides):
        x1, y1, x2, y2 = [int(v) for v in m_box]
        nb_personnes = len(associations[i])
        statut = "SURCHARGE" if nb_personnes >= seuil_surcharge else "CONFORME"
        couleur = (0, 0, 255) if statut == "SURCHARGE" else (0, 200, 0)

        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), couleur, 3)
        label = f"Moto {i+1}: {nb_personnes}p - {statut}"
        cv2.putText(img_annotated, label, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)

        for p_box in associations[i]:
            px1, py1, px2, py2 = [int(v) for v in p_box]
            cv2.rectangle(img_annotated, (px1, py1), (px2, py2), (255, 150, 0), 2)

        resume_lignes.append(f"**Moto {i+1}** : {nb_personnes} personne(s) → **{statut}** _(fiabilité : {fiabilite})_")

    if not motos_valides:
        resume_lignes.append("Aucune moto exploitable détectée dans cette image.")

    img_annotated_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
    return img_annotated_rgb, "\n\n".join(resume_lignes)

# --- INTERFACE ---
st.title("🏍️ Détection de surcharge de passagers sur motos")
st.write("Système de vision artificielle (YOLOv8) pour détecter automatiquement une surcharge de passagers.")

seuil_surcharge = st.slider("Seuil de surcharge (nb total de personnes, conducteur inclus)", 2, 5, 3)
conf_min = st.slider("Seuil de confiance de détection", 0.1, 0.9, 0.4, 0.05)

uploaded_file = st.file_uploader("Upload une image de moto-taxi", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    with st.spinner("Analyse en cours..."):
        img_result, resume = analyser_image(image_pil, seuil_surcharge, conf_min)
    st.image(img_result, caption="Résultat annoté", use_container_width=True)
    st.markdown(resume)