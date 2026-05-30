import io
import os
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION INTERNE (non visible dans l'UI)
# ─────────────────────────────────────────────────────────────────

_DETECTOR_PATH   = "models/best_dtc.pt"
_CLASSIFIER_PATH = "models/best_cls.pt"
_CLF_THRESH      = 0.50
_POSITIVE_IDX    = 1   # 0 = no_fire, 1 = fire/smoke

APP_TITLE = "Fire & Smoke Detection"
APP_ICON  = "🔥"

COLORS = {
    "fire":  (0,  50, 255),
    "smoke": (160, 160, 160),
}
CLASS_EMOJIS = {
    "fire":  "🔥",
    "smoke": "💨",
}


# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

    /* ── fond global ── */
    .stApp { background: #0a0a0f; }
    section[data-testid="stSidebar"] { background: #0d0d18 !important; border-right: 1px solid #1e1e35; }

    /* ── titre principal ── */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff4444 0%, #ff8800 60%, #ffcc00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 0;
        line-height: 1.1;
    }
    .subtitle {
        color: #666;
        font-size: 0.95rem;
        margin-top: 0.3rem;
        margin-bottom: 0;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── badge statut ── */
    .status-ready {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(0,200,100,0.12);
        border: 1px solid #00c864;
        border-radius: 999px;
        padding: 0.35rem 1rem;
        font-size: 0.85rem; font-weight: 700; color: #00c864;
    }
    .status-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #00c864;
        animation: blink 1.8s infinite;
        display: inline-block;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

    /* ── upload zone ── */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #2a2a45 !important;
        border-radius: 16px !important;
        background: #0e0e1c !important;
        transition: border-color 0.3s;
    }
    div[data-testid="stFileUploader"]:hover { border-color: #ff4444 !important; }

    /* ── cards détections ── */
    .detection-card {
        border-radius: 14px;
        padding: 1rem 1.3rem;
        margin: 0.5rem 0;
        font-family: 'JetBrains Mono', monospace;
        position: relative;
        overflow: hidden;
    }
    .detection-card::before {
        content: '';
        position: absolute; left: 0; top: 0; bottom: 0;
        width: 4px;
        border-radius: 4px 0 0 4px;
    }
    .det-fire  { background: rgba(255,68,68,0.07); border: 1px solid rgba(255,68,68,0.3); }
    .det-fire::before { background: #ff4444; }
    .det-smoke { background: rgba(180,180,180,0.06); border: 1px solid rgba(180,180,180,0.25); }
    .det-smoke::before { background: #aaaaaa; }
    .det-class { font-size: 1rem; font-weight: 700; color: #fff; margin-bottom: 4px; }
    .det-conf  { font-size: 0.82rem; color: #ff8800; margin-bottom: 2px; }
    .det-bbox  { font-size: 0.73rem; color: #555; }

    /* ── banners ── */
    .alarm-banner {
        background: linear-gradient(135deg, rgba(255,68,68,0.18), rgba(255,100,0,0.12));
        border: 1.5px solid #ff4444;
        border-radius: 14px;
        padding: 1.2rem 1.8rem;
        text-align: center;
        font-size: 1.25rem;
        font-weight: 800;
        color: #ff5555;
        letter-spacing: 0.03em;
        animation: pulse 1.5s infinite;
        margin: 1rem 0;
    }
    .safe-banner {
        background: linear-gradient(135deg, rgba(0,200,100,0.1), rgba(0,180,120,0.07));
        border: 1px solid #00c864;
        border-radius: 14px;
        padding: 1.2rem 1.8rem;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        color: #00c864;
        margin: 1rem 0;
    }
    @keyframes pulse { 0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(255,68,68,0.3)} 50%{opacity:0.85;box-shadow:0 0 20px 4px rgba(255,68,68,0.15)} }

    /* ── boutons ── */
    .stButton > button {
        background: linear-gradient(135deg, #ff4444, #ff7700);
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 2rem !important;
        width: 100%;
        transition: all 0.2s ease;
        letter-spacing: 0.03em;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(255,68,68,0.45) !important;
        background: linear-gradient(135deg, #ff5555, #ff8800) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ── métriques ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0f0f1e, #13132a);
        border-radius: 14px;
        padding: 1.1rem 1rem;
        border: 1px solid #1e1e38;
        text-align: center;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 800 !important; }
    div[data-testid="stMetricLabel"] { color: #666 !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.08em; }

    /* ── tabs ── */
    button[data-baseweb="tab"] {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em;
    }

    /* ── sliders sidebar ── */
    .stSlider > div > div > div { background: linear-gradient(90deg, #ff4444, #ff8800) !important; }

    /* ── image frames ── */
    img { border-radius: 12px; }

    /* ── divider ── */
    hr { border-color: #1a1a2e !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  CHARGEMENT DES MODÈLES
# ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(path: str):
    if not Path(path).exists():
        return None
    return YOLO(path)


# ─────────────────────────────────────────────────────────────────
#  CLASSIFICATION
# ─────────────────────────────────────────────────────────────────

def classify_frame(clf, frame: np.ndarray):
    r = clf.predict(frame, verbose=False)[0]
    probs = r.probs
    if probs is None:
        return True
    top1_idx  = int(probs.top1)
    top1_conf = float(probs.top1conf)
    label     = r.names.get(top1_idx, "").lower()
    fire_kw   = {"fire", "smoke", "feu", "fumee", "flamme"}
    is_pos    = any(kw in label for kw in fire_kw) or top1_idx == _POSITIVE_IDX
    if top1_conf < _CLF_THRESH:
        return True  # incertain → laisser le détecteur décider
    return is_pos


# ─────────────────────────────────────────────────────────────────
#  BOUNDING BOXES
# ─────────────────────────────────────────────────────────────────

def draw_boxes(frame: np.ndarray, result, conf_thresh: float):
    names, detections = result.names, []
    if result.boxes is None or len(result.boxes) == 0:
        return frame, detections
    for box in result.boxes:
        cls_id     = int(box.cls[0])
        conf       = float(box.conf[0])
        class_name = names.get(cls_id, f"class_{cls_id}").lower()
        if conf < conf_thresh:
            continue
        color = COLORS.get(class_name, (0, 255, 0))
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tl = max(4, min((x2-x1)//6, (y2-y1)//6, 24))
        for (px, py), (dx, dy) in [
            ((x1,y1),(tl,tl)), ((x2,y1),(-tl,tl)),
            ((x1,y2),(tl,-tl)), ((x2,y2),(-tl,-tl))
        ]:
            cv2.line(frame, (px,py), (px+dx,py), color, 3)
            cv2.line(frame, (px,py), (px,py+dy), color, 3)
        lbl = f"{class_name.upper()}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        lx, ly = max(x1,0), max(y1-th-12, 0)
        cv2.rectangle(frame, (lx,ly), (lx+tw+10, ly+th+10), color, -1)
        cv2.putText(frame, lbl, (lx+5, ly+th+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        detections.append({
            "class": class_name, "conf": conf,
            "bbox": (x1, y1, x2, y2),
            "width": x2-x1, "height": y2-y1, "area": (x2-x1)*(y2-y1),
        })
    return frame, detections


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR — paramètres utilisateur uniquement
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 0.5rem'>
        <div style='font-size:2.8rem'>🔥</div>
        <div style='font-size:1.1rem;font-weight:800;color:#fff;letter-spacing:0.04em'>FIRE DETECT</div>
        <div style='font-size:0.7rem;color:#444;text-transform:uppercase;letter-spacing:0.1em'>AI Vision System</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("<p style='color:#888;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem'>⚙ Paramètres de détection</p>", unsafe_allow_html=True)

    conf_thresh = st.slider(
        "Sensibilité",
        min_value=0.10, max_value=0.90,
        value=0.35, step=0.05,
        format="%.2f",
        help="Seuil de confiance minimum pour valider une détection"
    )
    iou_thresh = st.slider(
        "Précision (IoU)",
        min_value=0.05, max_value=0.70,
        value=0.10, step=0.05,
        format="%.2f",
        help="Suppression des boîtes redondantes"
    )

    st.divider()
    st.markdown("<p style='color:#888;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem'>🖥 Affichage</p>", unsafe_allow_html=True)
    show_stats   = st.toggle("Statistiques",       value=True)
    show_details = st.toggle("Détails des zones",  value=True)

    st.divider()
    st.markdown("""
    <div style='text-align:center;color:#2a2a45;font-size:0.7rem;padding:0.5rem 0'>
        Fire & Smoke Detection v2.0
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  CHARGEMENT
# ─────────────────────────────────────────────────────────────────

detector   = load_model(_DETECTOR_PATH)
classifier = load_model(_CLASSIFIER_PATH)


# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────

col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown('<p class="main-title">🔥 Fire & Smoke Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Système de détection intelligente · Temps réel · IA embarquée</p>',
                unsafe_allow_html=True)
with col_status:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if detector:
        st.markdown('<span class="status-ready"><span class="status-dot"></span> Système actif</span>',
                    unsafe_allow_html=True)
    else:
        st.error("✘ Indisponible")

if not detector:
    st.error("Le système de détection est indisponible. Contactez l'administrateur.")
    st.stop()

st.divider()


# ─────────────────────────────────────────────────────────────────
#  NIVEAU DE DÉGÂTS & REBOISEMENT
# ─────────────────────────────────────────────────────────────────

def compute_damage(detections: list, frame_w: int, frame_h: int) -> float:
    """Retourne le niveau de dégâts en % (surface totale des zones feu/fumée / surface image)."""
    if not detections or frame_w == 0 or frame_h == 0:
        return 0.0
    img_area = frame_w * frame_h
    # union des pixels couverts (évite double comptage)
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        mask[max(0,y1):min(frame_h,y2), max(0,x1):min(frame_w,x2)] = 1
    return float(mask.sum()) / img_area * 100


def show_reforestation(damage_pct: float):
    """Affiche le bloc de recommandation reboisement selon le niveau de dégâts."""
    if damage_pct < 30:
        level, color, icon = "Faible", "#00c864", "🌱"
        title = "Régénération naturelle"
        reco  = [
            "Le niveau de dégâts est faible (<30 %). La forêt peut se régénérer naturellement.",
            "Aucune intervention majeure n'est requise.",
            "Surveiller la zone pendant 1 à 2 saisons pour confirmer la repousse.",
            "Limiter l'accès humain pour favoriser la récupération spontanée.",
        ]
    elif damage_pct < 70:
        level, color, icon = "Modéré", "#ff8800", "🌿"
        title = "Reboisement assisté recommandé"
        reco  = [
            "Le niveau de dégâts est modéré (30–70 %). Une aide au reboisement est conseillée.",
            "**Espèces recommandées selon le climat :**",
            "— Zone méditerranéenne : Chêne-liège, Pin d'Alep, Cèdre de l'Atlas",
            "— Zone tempérée : Chêne pédonculé, Frêne, Érable champêtre",
            "— Zone semi-aride : Acacia, Casuarina, Argania spinosa",
            "Préparer le sol avant plantation (décompactage, amendement organique).",
            "Planter en automne ou au printemps pour optimiser la reprise.",
        ]
    else:
        level, color, icon = "Sévère", "#ff4444", "🚨"
        title = "Intervention d'experts requise"
        reco  = [
            "Le niveau de dégâts est sévère (>70 %). Une intervention professionnelle est indispensable.",
            "**Étapes recommandées :**",
            "1. Évaluation pédologique — analyse de la structure et fertilité du sol.",
            "2. Traitement anti-érosion — mise en place de fascines ou géotextiles si nécessaire.",
            "3. Restauration du sol — apport de matière organique et micro-organismes bénéfiques.",
            "4. Plan de reboisement — définir les espèces, densités et calendrier avec un expert forestier.",
            "5. Suivi pluriannuel — monitoring de la reprise végétale sur 3 à 5 ans.",
        ]

    items_html = "".join(
        f"<li style='margin:0.35rem 0;color:#ccc;font-size:0.88rem'>{r.replace('**','<strong>').replace('**','</strong>')}</li>"
        for r in reco
    )
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(0,0,0,0.4),rgba(20,20,40,0.6));
                border:1.5px solid {color};border-radius:16px;padding:1.4rem 1.8rem;margin:1.2rem 0'>
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:0.8rem'>
            <span style='font-size:1.8rem'>{icon}</span>
            <div>
                <div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:{color};font-weight:700'>
                    Niveau de dégâts — {level}
                </div>
                <div style='font-size:1.15rem;font-weight:800;color:#fff'>{title}</div>
            </div>
            <div style='margin-left:auto;background:{color}22;border:1px solid {color};
                        border-radius:999px;padding:0.3rem 1rem;font-size:1rem;font-weight:800;color:{color}'>
                {damage_pct:.1f}%
            </div>
        </div>
        <ul style='margin:0;padding-left:1.2rem'>
            {items_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_pipeline(frame: np.ndarray):
    """Classifie d'abord, puis détecte si positif. Retourne (annotated, detections, discarded)."""
    if classifier is not None:
        if not classify_frame(classifier, frame):
            return frame.copy(), [], True   # pas de feu → on ignore
    result = detector.predict(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
    annotated, detections = draw_boxes(frame.copy(), result, conf_thresh)
    return annotated, detections, False


# ─────────────────────────────────────────────────────────────────
#  ONGLETS
# ─────────────────────────────────────────────────────────────────

tab_image, tab_video, tab_webcam = st.tabs(["📷 Image", "🎬 Vidéo", "📹 Webcam"])


# ══════ IMAGE ════════════════════════════════════════════════════

with tab_image:
    st.markdown("### Analyser une image")
    uploaded = st.file_uploader(
        "Glisse une image ici ou clique pour choisir",
        type=["jpg", "jpeg", "png", "bmp", "webp"], key="img_upload"
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        col_orig, col_result = st.columns(2)
        with col_orig:
            st.markdown("**Image originale**")
            st.image(img_rgb, use_column_width=True)
            h, w = img_bgr.shape[:2]
            st.caption(f"{w} × {h} px  ·  {uploaded.size/1024:.0f} Ko")

        if st.button("🔍 Analyser", key="btn_detect"):
            with st.spinner("Analyse en cours..."):
                t0 = time.perf_counter()
                annotated, detections, discarded = run_pipeline(img_bgr)
                ms = (time.perf_counter() - t0) * 1000

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            with col_result:
                st.markdown("**Résultat**")
                st.image(annotated_rgb, use_column_width=True)
                st.caption(f"Analyse : {ms:.0f} ms")

            st.divider()

            if not discarded and detections:
                classes = set(d["class"] for d in detections)
                labels  = " + ".join(f"{CLASS_EMOJIS.get(c,'⚠')} {c.upper()}" for c in classes)
                st.markdown(
                    f'<div class="alarm-banner">⚠ ALARME — {labels} DÉTECTÉ(S)</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="safe-banner">✓ Aucun feu ni fumée détecté</div>',
                    unsafe_allow_html=True)

            if show_stats and detections:
                n_fire   = sum(1 for d in detections if d["class"] == "fire")
                n_smoke  = sum(1 for d in detections if d["class"] == "smoke")
                avg_conf = sum(d["conf"] for d in detections) / len(detections)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total détections", len(detections))
                c2.metric("🔥 Feu",   n_fire)
                c3.metric("💨 Fumée", n_smoke)
                c4.metric("Confiance moy.", f"{avg_conf:.0%}")

            if show_details and detections:
                st.markdown("### Détails des zones détectées")
                for i, d in enumerate(detections, 1):
                    cls = d["class"]
                    x1, y1, x2, y2 = d["bbox"]
                    cc = "det-fire" if cls == "fire" else "det-smoke"
                    st.markdown(f"""
                    <div class="detection-card {cc}">
                        <div class="det-class">{CLASS_EMOJIS.get(cls,'⚠')} Zone #{i} — {cls.upper()}</div>
                        <div class="det-conf">Confiance : {d['conf']:.2%}</div>
                        <div class="det-bbox">
                            Position : x1={x1} y1={y1} x2={x2} y2={y2}<br>
                            Taille : {d['width']} × {d['height']} px | Surface : {d['area']:,} px²
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            if not discarded and detections:
                st.divider()
                st.markdown("### 🌳 Recommandations de reboisement")
                h_img, w_img = img_bgr.shape[:2]
                dmg = compute_damage(detections, w_img, h_img)
                show_reforestation(dmg)

            if not discarded:
                st.markdown("### Télécharger le résultat")
                pil_img = Image.fromarray(annotated_rgb)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=95)
                st.download_button(
                    label="⬇️ Télécharger l'image annotée",
                    data=buf.getvalue(),
                    file_name=f"detection_{uploaded.name}",
                    mime="image/jpeg",
                )


# ══════ VIDÉO ════════════════════════════════════════════════════

with tab_video:
    st.markdown("### Analyser une vidéo")
    video_file = st.file_uploader(
        "Glisse une vidéo ici ou clique pour choisir",
        type=["mp4", "avi", "mov", "mkv"], key="vid_upload"
    )

    if video_file:
        st.video(video_file)
        max_frames = st.number_input(
            "Nombre max de frames à analyser",
            min_value=10, max_value=500, value=100, step=10,
        )

        if st.button("🔍 Lancer l'analyse", key="btn_video"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.close()

            cap          = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_vid      = cap.get(cv2.CAP_PROP_FPS) or 25
            w_vid        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_vid        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            st.info(f"Vidéo : {w_vid}×{h_vid} · {fps_vid:.0f} FPS · {total_frames} frames")

            progress          = st.progress(0, text="Analyse en cours...")
            frame_placeholder = st.empty()
            results_summary   = []
            n_alarm_frames = frame_idx = 0
            limit = min(int(max_frames), total_frames)

            while frame_idx < limit:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, detections, discarded = run_pipeline(frame)
                if not discarded and detections:
                    n_alarm_frames += 1
                    results_summary.extend(detections)

                if frame_idx % 10 == 0 and not discarded:
                    ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    status  = f"{len(detections)} détection(s)" if detections else "aucune détection"
                    frame_placeholder.image(ann_rgb, use_column_width=True,
                                            caption=f"Frame {frame_idx}/{limit} — {status}")

                frame_idx += 1
                progress.progress(frame_idx / limit, text=f"Analyse : {frame_idx}/{limit} frames")

            cap.release()
            os.unlink(tfile.name)
            progress.empty()

            st.divider()
            st.markdown("### Résumé")
            c1, c2, c3 = st.columns(3)
            c1.metric("Frames analysées", frame_idx)
            c2.metric("⚠ Frames alarme", n_alarm_frames,
                      f"{n_alarm_frames/max(frame_idx,1):.0%}")
            c3.metric("Total détections", len(results_summary))

            if results_summary:
                n_fire  = sum(1 for d in results_summary if d["class"] == "fire")
                n_smoke = sum(1 for d in results_summary if d["class"] == "smoke")
                avg_c   = sum(d["conf"] for d in results_summary) / len(results_summary)
                c4, c5, c6 = st.columns(3)
                c4.metric("🔥 Feu",   n_fire)
                c5.metric("💨 Fumée", n_smoke)
                c6.metric("Confiance moy.", f"{avg_c:.0%}")
                st.markdown(
                    '<div class="alarm-banner">⚠ FEU ET/OU FUMÉE DÉTECTÉ(S) DANS LA VIDÉO</div>',
                    unsafe_allow_html=True)
                st.divider()
                st.markdown("### 🌳 Recommandations de reboisement")
                total_area   = sum(d["area"] for d in results_summary)
                img_area_vid = w_vid * h_vid if w_vid > 0 and h_vid > 0 else 1
                dmg_vid = min(total_area / (img_area_vid * max(frame_idx, 1)) * 100, 100)
                show_reforestation(dmg_vid)
            else:
                st.markdown(
                    '<div class="safe-banner">✓ Aucun feu ni fumée détecté dans la vidéo</div>',
                    unsafe_allow_html=True)


# ══════ WEBCAM ═══════════════════════════════════════════════════

with tab_webcam:
    st.markdown("### 📹 Détection en temps réel")

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶ Démarrer la webcam", key="btn_webcam_start",
                     disabled=st.session_state.webcam_running):
            st.session_state.webcam_running = True
            st.rerun()
    with col_btn2:
        if st.button("⏹ Arrêter", key="btn_webcam_stop",
                     disabled=not st.session_state.webcam_running):
            st.session_state.webcam_running = False
            st.rerun()

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("✘ Webcam introuvable — vérifier qu'elle est connectée.")
            st.session_state.webcam_running = False
        else:
            frame_win = st.empty()
            info_win  = st.empty()
            stop_flag = st.session_state.get("webcam_running", True)

            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.session_state.webcam_running = False
                    break
                t0 = time.perf_counter()
                annotated, detections, discarded = run_pipeline(frame)
                ms = (time.perf_counter() - t0) * 1000

                display = frame if discarded else annotated
                frame_win.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                                channels="RGB", use_column_width=True)

                if not discarded and detections:
                    classes = ", ".join(set(d["class"].upper() for d in detections))
                    info_win.markdown(
                        f'<div class="alarm-banner">⚠ ALARME : {classes} — {ms:.0f} ms</div>',
                        unsafe_allow_html=True)
                else:
                    info_win.markdown(
                        f'<div class="safe-banner">✓ Aucune menace — {ms:.0f} ms</div>',
                        unsafe_allow_html=True)

            cap.release()


# ─────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────

st.divider()
st.markdown("""
<div style='text-align:center;padding:1.5rem 0 0.5rem'>
    <div style='font-size:1.5rem;margin-bottom:0.4rem'>🔥</div>
    <div style='color:#333;font-size:0.78rem;letter-spacing:0.1em;text-transform:uppercase'>
        Fire &amp; Smoke Detection &nbsp;·&nbsp; AI Vision System &nbsp;·&nbsp; v2.0
    </div>
</div>
""", unsafe_allow_html=True)
