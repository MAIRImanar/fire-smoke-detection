"""
╔══════════════════════════════════════════════════════════════════╗
║   Fire & Smoke Detection — Application Web avec Interface       ║
║   Framework : Streamlit                                          ║
║   Modèle    : best_nano_111.pt (sayedgamal99)                   ║
╚══════════════════════════════════════════════════════════════════╝

Installation :
    pip install streamlit ultralytics opencv-python pillow

Lancement :
    streamlit run app.py

Sur Google Colab :
    !pip install streamlit ultralytics opencv-python pillow pyngrok -q
    !streamlit run app.py &
    from pyngrok import ngrok
    public_url = ngrok.connect(8501)
    print(public_url)
"""

import io
import os
import sys
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────

MODEL_PATH   = "models/best_nano_111.pt"
APP_TITLE    = "Fire & Smoke Detection"
APP_ICON     = "🔥"

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

# CSS personnalisé
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff4444, #ff8800);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    .detection-card {
        background: #1a1a2e;
        border: 1px solid #ff4444;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.4rem 0;
        font-family: 'JetBrains Mono', monospace;
    }

    .det-fire {
        border-color: #ff4444;
        background: rgba(255, 68, 68, 0.08);
    }

    .det-smoke {
        border-color: #aaaaaa;
        background: rgba(170, 170, 170, 0.08);
    }

    .det-class {
        font-size: 1rem;
        font-weight: 600;
        color: #fff;
    }

    .det-conf {
        font-size: 0.85rem;
        color: #ff8800;
    }

    .det-bbox {
        font-size: 0.75rem;
        color: #666;
        margin-top: 4px;
    }

    .stat-box {
        background: #0f0f1a;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #222;
    }

    .stat-num {
        font-size: 2rem;
        font-weight: 800;
        color: #ff4444;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .alarm-banner {
        background: rgba(255, 68, 68, 0.15);
        border: 2px solid #ff4444;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #ff4444;
        animation: pulse 1.5s infinite;
    }

    .safe-banner {
        background: rgba(0, 200, 100, 0.1);
        border: 1px solid #00c864;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        color: #00c864;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .stButton > button {
        background: linear-gradient(135deg, #ff4444, #ff8800);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        padding: 0.6rem 2rem;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255, 68, 68, 0.4);
    }

    div[data-testid="stMetric"] {
        background: #0f0f1a;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #222;
    }

    .upload-zone {
        border: 2px dashed #333;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  CHARGEMENT DU MODÈLE (mis en cache)
# ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_path: str):
    if not Path(model_path).exists():
        return None
    return YOLO(model_path)


# ─────────────────────────────────────────────────────────────────
#  DESSIN DES BOUNDING BOXES
# ─────────────────────────────────────────────────────────────────

def draw_boxes(frame: np.ndarray, result, conf_thresh: float):
    names      = result.names
    detections = []

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

        # Rectangle principal
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Coins épais style pro
        tl = max(4, min((x2-x1)//6, (y2-y1)//6, 24))
        for (px, py), (dx, dy) in [
            ((x1,y1),(tl,tl)), ((x2,y1),(-tl,tl)),
            ((x1,y2),(tl,-tl)), ((x2,y2),(-tl,-tl))
        ]:
            cv2.line(frame, (px,py), (px+dx,py), color, 3)
            cv2.line(frame, (px,py), (px,py+dy), color, 3)

        # Label avec fond
        label = f"{class_name.upper()}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        lx, ly = max(x1,0), max(y1-th-12, 0)
        cv2.rectangle(frame, (lx,ly), (lx+tw+10, ly+th+10), color, -1)
        cv2.putText(frame, label, (lx+5, ly+th+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        detections.append({
            "class":  class_name,
            "conf":   conf,
            "bbox":   (x1, y1, x2, y2),
            "width":  x2 - x1,
            "height": y2 - y1,
            "area":   (x2-x1) * (y2-y1),
        })

    return frame, detections


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    st.divider()

    # Chemin modèle
    model_path = st.text_input(
        "Chemin du modèle (.pt)",
        value=MODEL_PATH,
        help="Chemin vers best_nano_111.pt"
    )

    st.divider()
    st.markdown("### Seuils de détection")

    conf_thresh = st.slider(
        "Confiance minimale",
        min_value=0.10, max_value=0.90,
        value=0.35, step=0.05,
        format="%.2f",
        help="Détections en dessous de ce seuil sont ignorées"
    )

    iou_thresh = st.slider(
        "Seuil IoU (NMS)",
        min_value=0.05, max_value=0.70,
        value=0.10, step=0.05,
        format="%.2f",
        help="Élimine les boîtes qui se chevauchent"
    )

    st.divider()
    st.markdown("### Affichage")

    show_stats   = st.toggle("Afficher les statistiques", value=True)
    show_details = st.toggle("Afficher les détails bbox", value=True)

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem;color:#555;'>
    Modèle : <b>best_nano_111.pt</b><br>
    Classes : fire · smoke<br>
    mAP@50 : 77% · Précision : 80.6%
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────

col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<p class="main-title">🔥 Fire & Smoke Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">YOLOv11 · Détection temps réel · best_nano_111.pt</p>', unsafe_allow_html=True)

# Chargement du modèle
model = load_model(model_path)

with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    if model:
        st.success("✔ Modèle chargé")
    else:
        st.error("✘ Modèle introuvable")
        st.info(f"Place `best_nano_111.pt` dans `{model_path}`")

if not model:
    st.stop()

st.divider()


# ─────────────────────────────────────────────────────────────────
#  ONGLETS : Image | Vidéo | Webcam
# ─────────────────────────────────────────────────────────────────

tab_image, tab_video, tab_webcam = st.tabs(["📷 Image", "🎬 Vidéo", "📹 Webcam"])


# ══════════════════════════════════════════════════════════════════
#  ONGLET IMAGE
# ══════════════════════════════════════════════════════════════════

with tab_image:
    st.markdown("### Uploader une image")

    uploaded = st.file_uploader(
        "Glisse une image ici ou clique pour choisir",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload"
    )

    if uploaded:
        # Lire l'image
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        col_orig, col_result = st.columns(2)

        with col_orig:
            st.markdown("**Image originale**")
            st.image(img_rgb, use_container_width=True)
            h, w = img_bgr.shape[:2]
            st.caption(f"{w} × {h} px  ·  {uploaded.size/1024:.0f} Ko")

        # Bouton détecter
        if st.button("🔍 Détecter feu & fumée", key="btn_detect"):

            with st.spinner("Analyse en cours..."):
                t0      = time.perf_counter()
                results = model.predict(img_bgr, conf=conf_thresh,
                                        iou=iou_thresh, verbose=False)
                ms      = (time.perf_counter() - t0) * 1000
                result  = results[0]

            annotated, detections = draw_boxes(img_bgr.copy(), result, conf_thresh)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            with col_result:
                st.markdown("**Résultat annoté**")
                st.image(annotated_rgb, use_container_width=True)
                st.caption(f"Inférence : {ms:.0f} ms")

            st.divider()

            # ── Bannière alarme / safe ────────────────────────────
            if detections:
                classes = set(d["class"] for d in detections)
                labels  = " + ".join(
                    f"{CLASS_EMOJIS.get(c,'⚠')} {c.upper()}"
                    for c in classes
                )
                st.markdown(
                    f'<div class="alarm-banner">⚠ ALARME — {labels} DÉTECTÉ(S)</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="safe-banner">✓ Aucun feu ni fumée détecté</div>',
                    unsafe_allow_html=True
                )

            st.markdown("")

            # ── Statistiques ─────────────────────────────────────
            if show_stats and detections:
                n_fire  = sum(1 for d in detections if d["class"] == "fire")
                n_smoke = sum(1 for d in detections if d["class"] == "smoke")
                avg_conf = sum(d["conf"] for d in detections) / len(detections)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total détections", len(detections))
                c2.metric("🔥 Feu",   n_fire)
                c3.metric("💨 Fumée", n_smoke)
                c4.metric("Confiance moy.", f"{avg_conf:.0%}")

            # ── Détails par détection ─────────────────────────────
            if show_details and detections:
                st.markdown("### Détails des détections")
                for i, d in enumerate(detections, 1):
                    cls   = d["class"]
                    emoji = CLASS_EMOJIS.get(cls, "⚠")
                    color_class = "det-fire" if cls == "fire" else "det-smoke"
                    x1,y1,x2,y2 = d["bbox"]
                    st.markdown(f"""
                    <div class="detection-card {color_class}">
                        <div class="det-class">{emoji} Détection #{i} — {cls.upper()}</div>
                        <div class="det-conf">Confiance : {d['conf']:.2%}</div>
                        <div class="det-bbox">
                            Position : x1={x1} y1={y1} x2={x2} y2={y2}<br>
                            Taille   : {d['width']} × {d['height']} px &nbsp;|&nbsp;
                            Surface  : {d['area']:,} px²
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Télécharger l'image annotée ───────────────────────
            st.markdown("### Télécharger le résultat")
            pil_img    = Image.fromarray(annotated_rgb)
            buf        = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=95)
            st.download_button(
                label="⬇️ Télécharger l'image annotée",
                data=buf.getvalue(),
                file_name=f"detection_{uploaded.name}",
                mime="image/jpeg",
            )


# ══════════════════════════════════════════════════════════════════
#  ONGLET VIDÉO
# ══════════════════════════════════════════════════════════════════

with tab_video:
    st.markdown("### Uploader une vidéo")

    video_file = st.file_uploader(
        "Glisse une vidéo ici ou clique pour choisir",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload"
    )

    if video_file:
        st.video(video_file)

        max_frames = st.number_input(
            "Nombre max de frames à traiter",
            min_value=10, max_value=500, value=100, step=10,
            help="Réduis pour accélérer l'analyse"
        )

        if st.button("🔍 Analyser la vidéo", key="btn_video"):

            # Sauvegarder la vidéo temporairement
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_vid      = cap.get(cv2.CAP_PROP_FPS) or 25
            w_vid        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_vid        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            st.info(f"Vidéo : {w_vid}×{h_vid} · {fps_vid:.0f} FPS · {total_frames} frames")

            progress  = st.progress(0, text="Analyse en cours...")
            frame_placeholder = st.empty()
            results_summary   = []

            n_fire_frames = 0
            frame_idx     = 0
            limit         = min(int(max_frames), total_frames)

            while frame_idx < limit:
                ret, frame = cap.read()
                if not ret:
                    break

                result    = model.predict(frame, conf=conf_thresh,
                                          iou=iou_thresh, verbose=False)[0]
                annotated, detections = draw_boxes(frame.copy(), result, conf_thresh)

                if detections:
                    n_fire_frames += 1
                    results_summary.extend(detections)

                # Afficher 1 frame sur 10
                if frame_idx % 10 == 0:
                    ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(ann_rgb, use_container_width=True,
                                            caption=f"Frame {frame_idx}/{limit}")

                frame_idx += 1
                progress.progress(frame_idx / limit,
                                  text=f"Frame {frame_idx}/{limit} — {len(detections)} détections")

            cap.release()
            os.unlink(tfile.name)
            progress.empty()

            st.divider()
            st.markdown("### Résumé de l'analyse")

            c1, c2, c3 = st.columns(3)
            c1.metric("Frames analysées", frame_idx)
            c2.metric("Frames avec alarme",
                      n_fire_frames, f"{n_fire_frames/max(frame_idx,1):.0%}")
            c3.metric("Total détections", len(results_summary))

            if results_summary:
                n_fire  = sum(1 for d in results_summary if d["class"] == "fire")
                n_smoke = sum(1 for d in results_summary if d["class"] == "smoke")
                avg_c   = sum(d["conf"] for d in results_summary) / len(results_summary)

                c4, c5, c6 = st.columns(3)
                c4.metric("🔥 Détections feu",   n_fire)
                c5.metric("💨 Détections fumée", n_smoke)
                c6.metric("Confiance moyenne",  f"{avg_c:.0%}")

                st.markdown(
                    '<div class="alarm-banner">⚠ FEU ET/OU FUMÉE DÉTECTÉ(S) DANS LA VIDÉO</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="safe-banner">✓ Aucun feu ni fumée détecté dans la vidéo</div>',
                    unsafe_allow_html=True
                )


# ══════════════════════════════════════════════════════════════════
#  ONGLET WEBCAM
# ══════════════════════════════════════════════════════════════════

with tab_webcam:
    st.markdown("### Détection par webcam")
    st.info("La webcam fonctionne en local seulement (pas sur Colab). Lance `streamlit run app.py` sur ton PC.")

    st.markdown("""
    **Pour utiliser la webcam :**
    1. Lance l'app localement : `streamlit run app.py`
    2. Va sur l'onglet Webcam
    3. Clique sur **Démarrer**
    """)

    if st.button("📹 Démarrer la webcam (local)", key="btn_webcam"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("✘ Webcam introuvable")
        else:
            frame_win  = st.empty()
            info_win   = st.empty()
            stop_btn   = st.button("⏹ Arrêter", key="stop_webcam")

            while not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break

                t0        = time.perf_counter()
                result    = model.predict(frame, conf=conf_thresh,
                                          iou=iou_thresh, verbose=False)[0]
                ms        = (time.perf_counter() - t0) * 1000
                annotated, detections = draw_boxes(frame.copy(), result, conf_thresh)
                ann_rgb   = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                frame_win.image(ann_rgb, channels="RGB", use_container_width=True)

                if detections:
                    classes = ", ".join(set(d["class"].upper() for d in detections))
                    info_win.error(f"⚠ ALARME : {classes} — {ms:.0f} ms")
                else:
                    info_win.success(f"✓ Aucune détection — {ms:.0f} ms")

            cap.release()


# ─────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────

st.divider()
st.markdown("""
<div style='text-align:center;color:#444;font-size:0.8rem;padding:1rem 0'>
    Fire & Smoke Detection · YOLOv11 · best_nano_111.pt · sayedgamal99
</div>
""", unsafe_allow_html=True)
