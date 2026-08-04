"""
PCB Defect Detection Studio
============================
A modular Streamlit application for the YOLOv8-based PCB Defect Detection
research project.

Repo layout expected alongside this file:
    pcb-defect-yolov8/
    ├── app.py            <- this file
    ├── best.pt           <- trained YOLOv8 weights
    ├── class.yaml        <- class id -> name mapping
    └── README.md

Run with:
    streamlit run app.py

Author note: the app is organized into clearly separated "modules" (one
function per concern) so that each screen, chart, or utility can be
edited, tested, or reused independently.
"""

import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Optional heavy deps are imported lazily inside the functions that need
# them (YOLO, plotly, PIL, yaml) so the landing page still renders even if
# a dependency or the model weights are missing.


# ============================================================================
# 1. APP CONFIG & CONSTANTS
# ============================================================================

APP_TITLE = "PCB Defect Detection Studio"
APP_ICON = "🔎"
REPO_ROOT = Path(__file__).resolve().parent
MODEL_PATH = REPO_ROOT / "best.pt"
CLASS_YAML_PATH = REPO_ROOT / "class.yaml"

DEFAULT_CLASS_NAMES = {
    0: "Open",
    1: "Short",
    2: "Mousebite",
    3: "Spur",
    4: "Copper",
    5: "Pin-hole",
}

CLASS_DESCRIPTIONS = {
    "Open": "A broken copper trace that interrupts an intended electrical connection.",
    "Short": "An unwanted connection bridging two traces that should remain isolated.",
    "Mousebite": "Small notches eaten out of a trace edge, thinning the conductor.",
    "Spur": "A stray copper protrusion branching off a trace where none should exist.",
    "Copper": "Excess or spurious copper residue left on the board surface.",
    "Pin-hole": "A tiny void or pit in the copper/pad, often sub-millimeter in size.",
}

CLASS_COLORS = {
    "Open": "#EF4444",
    "Short": "#F59E0B",
    "Mousebite": "#8B5CF6",
    "Spur": "#10B981",
    "Copper": "#3B82F6",
    "Pin-hole": "#EC4899",
}

PRIMARY = "#6366F1"
BG_CARD = "#111827"


# ---- Research-report-derived data (used across the Insights page) --------

@dataclass
class ModelVariant:
    name: str
    architecture: str
    params_m: float
    size_mb: float
    epochs: int
    key_feature: str
    map50: float
    map5095: float
    precision: float
    recall: float
    latency_ms: float
    latency_std_ms: float
    train_time_min: float


MODEL_VARIANTS = [
    ModelVariant("Baseline", "YOLOv8 Nano", 3.01, 6.0, 50, "Speed-optimized",
                 0.982, 0.684, 0.972, 0.944, 78.3, 650.6, 14),
    ModelVariant("V1 (Recommended)", "YOLOv8 Small", 11.14, 21.5, 80, "Aggressive augmentation",
                 0.962, 0.569, 0.959, 0.898, 19.6, 7.7, 20),
    ModelVariant("V2", "YOLOv8 Medium", 25.86, 49.6, 100, "Extended LR warmup",
                 0.987, 0.628, 0.982, 0.960, 29.7, 12.7, 57),
    ModelVariant("V3", "YOLOv8 Medium", 25.84, 49.6, 120, "Balanced robustness",
                 0.982, 0.696, 0.963, 0.937, 31.1, 10.3, 54),
]

PER_CLASS_PRECISION = pd.DataFrame(
    {
        "Class": ["Open", "Short", "Mousebite", "Spur", "Copper", "Pin-hole"],
        "Baseline": [0.980, 0.959, 0.967, 0.962, 0.974, 0.990],
        "V1": [0.960, 0.926, 0.961, 0.952, 0.989, 0.963],
        "V2": [0.992, 0.989, 0.955, 0.969, 0.995, 0.990],
        "V3": [0.962, 0.956, 0.948, 0.957, 0.967, 0.989],
    }
)

CROSS_DATASET = pd.DataFrame(
    {
        "Dataset": ["DeepPCB (V1)", "HRIPCB (V1)"],
        "mAP@0.5": [0.962, 0.954],
        "mAP@0.5:0.95": [0.569, 0.548],
        "Precision": [0.959, 0.951],
        "Recall": [0.898, 0.889],
    }
)

PROJECT_TIMELINE = [
    ("Problem framing", "Scoped bare-board PCB defect detection as an automated-optical-inspection "
                          "replacement/aid, targeting 6 defect classes."),
    ("Dataset sourcing", "Assembled two benchmarks — DeepPCB (1,000 imgs, random split) and "
                          "HRIPCB (693 imgs, 10 physical board templates)."),
    ("Evaluation design", "Built a board-template-level split for HRIPCB to eliminate layout "
                           "memorization, contrasting it against DeepPCB's random image-wise split."),
    ("Model sweep", "Trained 4 YOLOv8 variants (Nano → Medium) across differing augmentation, "
                     "LR schedules, and epoch budgets."),
    ("Cross-dataset validation", "Re-evaluated the strongest configuration (V1) on the held-out "
                                  "HRIPCB template board to quantify true generalization gap."),
    ("Deployment packaging", "Selected YOLOv8-Small + aggressive augmentation as the production "
                               "config; documented latency, memory, and threshold guidance."),
]

KEY_METRICS = {
    "Recommended mAP@0.5": "0.962",
    "Inference Latency": "19.6 ms",
    "Throughput": "51 img/s",
    "Model Size": "21.5 MB",
    "GPU Memory": "~2.0 GB",
    "Defect Classes": "6",
}

KEY_INSIGHTS = [
    "Evaluation protocol matters: random image-wise splitting overestimated generalization "
    "by 1–3 percentage points versus board-template-level stratification.",
    "Aggressive augmentation beat raw model scaling: YOLOv8-Small with heavy augmentation "
    "matched YOLOv8-Medium's practical performance at ~2× the speed.",
    "Dataset size plateaus beyond ~700–800 images; template diversity becomes the binding "
    "constraint on further accuracy gains.",
    "Inference consistency matters as much as mean latency — the Baseline model's 650.6 ms "
    "std-dev made it unusable in production despite a lower mean.",
    "The ~1% mAP gap between DeepPCB and HRIPCB evaluation represents the upper bound of "
    "layout-memorization effects for this domain.",
]

TEAM = ["Bhabanti Paul", "Ankit Mittal", "Kasheena Mulla", "Nandini Bag"]


# ============================================================================
# 2. GLOBAL STYLE MODULE
# ============================================================================

def inject_global_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: radial-gradient(circle at top left, #0f172a 0%, #020617 55%);
        }}
        section[data-testid="stSidebar"] {{
            background: #0b1120;
            border-right: 1px solid #1e293b;
        }}
        .hero-title {{
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #818cf8, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }}
        .hero-subtitle {{
            color: #94a3b8;
            font-size: 1.05rem;
            margin-bottom: 1.4rem;
        }}
        .metric-card {{
            background: {BG_CARD};
            border: 1px solid #1f2937;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            text-align: center;
        }}
        .metric-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #f8fafc;
        }}
        .metric-label {{
            color: #94a3b8;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .section-card {{
            background: {BG_CARD};
            border: 1px solid #1f2937;
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 1rem;
        }}
        .pill {{
            display: inline-block;
            padding: 0.15rem 0.7rem;
            border-radius: 999px;
            background: rgba(99,102,241,0.15);
            color: #a5b4fc;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }}
        .timeline-step {{
            border-left: 2px solid {PRIMARY};
            padding-left: 1rem;
            margin-bottom: 1.1rem;
        }}
        .timeline-step h4 {{
            margin-bottom: 0.15rem;
            color: #e0e7ff;
        }}
        .timeline-step p {{
            color: #94a3b8;
            font-size: 0.9rem;
            margin: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> str:
    return f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
    """


# ============================================================================
# 3. DATA / MODEL UTILITIES
# ============================================================================

@st.cache_data(show_spinner=False)
def load_class_names() -> dict:
    """Read class.yaml if present, else fall back to the known mapping."""
    if CLASS_YAML_PATH.exists():
        try:
            import yaml  # lazy import
            with open(CLASS_YAML_PATH, "r") as f:
                data = yaml.safe_load(f)
            names = data.get("names", DEFAULT_CLASS_NAMES)
            return {int(k): v for k, v in names.items()}
        except Exception:
            return DEFAULT_CLASS_NAMES
    return DEFAULT_CLASS_NAMES


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model(weights_path: str):
    """Load a YOLO model from disk. Cached so it only loads once per session."""
    from ultralytics import YOLO
    return YOLO(weights_path)


def run_inference(model, image, conf_threshold: float, iou_threshold: float):
    """Run YOLO inference on a PIL image and return the first Results object."""
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )
    return results[0]


def results_to_dataframe(result, class_names: dict) -> pd.DataFrame:
    """Convert a YOLO Results object into a tidy detections dataframe."""
    rows = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(columns=["Class", "Confidence", "x1", "y1", "x2", "y2"])
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
        rows.append(
            {
                "Class": class_names.get(cls_id, f"Class {cls_id}"),
                "Confidence": round(conf, 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            }
        )
    return pd.DataFrame(rows)


def model_status() -> tuple[bool, str]:
    if MODEL_PATH.exists():
        return True, f"Model found at `{MODEL_PATH.name}` ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)"
    return False, f"`{MODEL_PATH.name}` not found next to app.py. Upload weights on the Detection page."


# ============================================================================
# 4. PAGE: LANDING / PROJECT JOURNEY
# ============================================================================

def page_landing() -> None:
    st.markdown(f'<div class="hero-title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">YOLOv8-based automated PCB defect detection — '
        'a comparative study across DeepPCB and HRIPCB benchmarks.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="pill">🧪 Research-backed</span>'
        f'<span class="pill">⚡ 51 img/s</span>'
        f'<span class="pill">🎯 0.962 mAP@0.5</span>'
        f'<span class="pill">👥 4-person team</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    # --- headline metrics -------------------------------------------------
    cols = st.columns(len(KEY_METRICS))
    for col, (label, value) in zip(cols, KEY_METRICS.items()):
        with col:
            st.markdown(metric_card(label, value), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📌 What this project does")
        st.write(
            "This studio packages a YOLOv8 object-detection pipeline trained to spot six "
            "common bare-board PCB defects — **Open, Short, Mousebite, Spur, Copper, and "
            "Pin-hole** — directly from board images, aiming to replace or assist manual "
            "and template-matching AOI inspection."
        )
        st.write(
            "The research behind it went further than a single training run: it compared "
            "**random image-wise splitting vs. board-template-level stratification** to "
            "expose how evaluation protocol choice can quietly inflate reported accuracy, "
            "and it weighed **aggressive data augmentation against simply scaling up the "
            "model**."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🛤️ Project journey")
        for i, (title, desc) in enumerate(PROJECT_TIMELINE, start=1):
            st.markdown(
                f"""
                <div class="timeline-step">
                    <h4>{i}. {title}</h4>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🧠 Key takeaways")
        for insight in KEY_INSIGHTS:
            st.markdown(f"- {insight}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📁 Repository")
        st.code(
            "pcb-defect-yolov8/\n"
            "├── best.pt        # trained YOLOv8 weights\n"
            "├── class.yaml     # 6-class label map\n"
            "└── README.md      # project documentation",
            language="text",
        )
        found, msg = model_status()
        (st.success if found else st.warning)(msg)
        st.markdown("</div>", unsafe_allow_html=True)

    st.info("👉 Head to **Live Detection** in the sidebar to try the model on your own PCB images.")


# ============================================================================
# 5. PAGE: LIVE DETECTION
# ============================================================================

def page_detection() -> None:
    st.header("🔍 Live Defect Detection")
    st.caption("Upload a PCB image and run it through the trained YOLOv8 model.")

    class_names = load_class_names()
    found, msg = model_status()

    with st.sidebar:
        st.markdown("### ⚙️ Inference settings")
        conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
        iou_threshold = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.45, 0.05)
        custom_weights = st.file_uploader("Optional: override weights (.pt)", type=["pt"])

    weights_path = str(MODEL_PATH)
    if custom_weights is not None:
        tmp_path = REPO_ROOT / "uploaded_weights.pt"
        with open(tmp_path, "wb") as f:
            f.write(custom_weights.getbuffer())
        weights_path = str(tmp_path)
        st.success("Using uploaded custom weights for this session.")
    elif not found:
        st.warning(msg + " — upload a `.pt` file in the sidebar to proceed.")
        return

    uploaded_image = st.file_uploader("Upload a PCB image", type=["png", "jpg", "jpeg", "bmp"])

    demo_col1, demo_col2 = st.columns([1, 3])
    with demo_col1:
        run_button = st.button("🚀 Run Detection", type="primary", use_container_width=True)

    if uploaded_image is None:
        st.info("Upload a board image above, then click **Run Detection**.")
        return

    from PIL import Image
    image = Image.open(uploaded_image).convert("RGB")

    col_input, col_output = st.columns(2)
    with col_input:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    if run_button:
        try:
            with st.spinner("Running YOLOv8 inference..."):
                model = load_model(weights_path)
                start = time.time()
                result = run_inference(model, image, conf_threshold, iou_threshold)
                elapsed_ms = (time.time() - start) * 1000

            annotated = result.plot()[:, :, ::-1]  # BGR -> RGB

            with col_output:
                st.subheader("Detections")
                st.image(annotated, use_container_width=True)

            df = results_to_dataframe(result, class_names)
            st.markdown("#### 📋 Detection summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Defects found", len(df))
            m2.metric("Inference time", f"{elapsed_ms:.1f} ms")
            m3.metric("Unique classes", df["Class"].nunique() if not df.empty else 0)

            if df.empty:
                st.success("No defects detected above the confidence threshold. ✅")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
                counts = df["Class"].value_counts().reset_index()
                counts.columns = ["Class", "Count"]
                st.bar_chart(counts.set_index("Class"))
        except Exception as e:
            st.error(f"Inference failed: {e}")
    else:
        with col_output:
            st.subheader("Detections")
            st.caption("Click **Run Detection** to see results here.")


# ============================================================================
# 6. PAGE: MODEL INSIGHTS (research results)
# ============================================================================

def page_insights() -> None:
    st.header("📊 Model Insights & Research Results")
    st.caption("Findings from the comparative study across DeepPCB and HRIPCB benchmarks.")

    tab_overview, tab_perclass, tab_generalization, tab_recommend = st.tabs(
        ["Model Comparison", "Per-Class Precision", "Cross-Dataset Generalization", "Recommended Config"]
    )

    with tab_overview:
        st.subheader("Four YOLOv8 variants, one training regime")
        df = pd.DataFrame([vars(m) for m in MODEL_VARIANTS])
        df_display = df.rename(columns={
            "name": "Model", "architecture": "Architecture", "params_m": "Params (M)",
            "size_mb": "Size (MB)", "epochs": "Epochs", "key_feature": "Key Feature",
            "map50": "mAP@0.5", "map5095": "mAP@0.5:0.95", "precision": "Precision",
            "recall": "Recall", "latency_ms": "Latency (ms)", "latency_std_ms": "Latency Std (ms)",
            "train_time_min": "Train Time (min)",
        })
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.caption("mAP@0.5 by model")
            st.bar_chart(df.set_index("name")["map50"])
        with c2:
            st.caption("Inference latency (ms) — note Baseline's volatility isn't shown by the mean alone")
            st.bar_chart(df.set_index("name")["latency_ms"])

        st.warning(
            "⚠️ Baseline looks fast on average (78.3 ms) but its latency **std-dev is 650.6 ms** "
            "(max 9086.6 ms) — highly inconsistent and unsuitable for a production line."
        )

    with tab_perclass:
        st.subheader("Per-class precision (DeepPCB validation, 1,368 instances)")
        chart_df = PER_CLASS_PRECISION.set_index("Class")
        st.bar_chart(chart_df)
        st.dataframe(PER_CLASS_PRECISION, use_container_width=True, hide_index=True)
        st.caption(
            "Copper and Spur stay above 0.95 precision across every variant. Pin-hole — the "
            "smallest defect type — shows the widest spread (0.96–0.99), the class worth "
            "monitoring most closely in production."
        )

    with tab_generalization:
        st.subheader("Does the model generalize to unseen board layouts?")
        st.write(
            "V1 was re-evaluated on HRIPCB's held-out third template board — a PCB design the "
            "model had never seen during training — to isolate genuine generalization from "
            "layout memorization."
        )
        st.dataframe(CROSS_DATASET, use_container_width=True, hide_index=True)
        st.bar_chart(CROSS_DATASET.set_index("Dataset")[["mAP@0.5", "mAP@0.5:0.95"]])
        st.caption(
            "mAP@0.5 dropped only 0.8 points (0.962 → 0.954) on a completely novel board — "
            "the ~1% gap represents the upper bound of layout-memorization benefit that random "
            "splitting was quietly granting other models."
        )

    with tab_recommend:
        st.subheader("🏆 Production configuration: YOLOv8-Small + aggressive augmentation (V1)")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("mAP@0.5", "0.962")
        r2.metric("Latency", "19.6 ms", "±7.7 ms")
        r3.metric("Throughput", "51 img/s")
        r4.metric("GPU memory", "~2.0 GB")
        st.markdown(
            "- **Model file:** `pcb_yolov8_tuned_v1_aggressive_aug_best.pt` (packaged here as `best.pt`)\n"
            "- **Confidence operating point:** 0.65 → ~77% recall at ~90% precision, suited to "
            "triage-assist rather than fully autonomous accept/reject\n"
            "- **Hardware target:** NVIDIA Jetson Xavier, RTX 4000-series, or equivalent industrial GPU\n"
            "- **Monitoring:** track per-class precision in production, with extra attention on Pin-hole"
        )


# ============================================================================
# 7. PAGE: ABOUT / TEAM
# ============================================================================

def page_about() -> None:
    st.header("ℹ️ About this Project")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Research team")
    st.write(", ".join(TEAM) + " — July 2026")
    st.write(
        "*A Comparative Deep Learning Study Across Heterogeneous Benchmark Datasets: "
        "YOLOv8-based PCB Defect Classification.*"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Defect classes (`class.yaml`)")
    class_names = load_class_names()
    rows = [
        {"ID": cid, "Class": name, "Description": CLASS_DESCRIPTIONS.get(name, "—")}
        for cid, name in sorted(class_names.items())
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Datasets used")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.markdown("**DeepPCB**")
        st.markdown(
            "- 1,000 images (640×640)\n"
            "- 6,873 annotated instances\n"
            "- Random image-wise split (800/200)\n"
            "- Class imbalance ratio: 1.37×"
        )
    with dcol2:
        st.markdown("**HRIPCB**")
        st.markdown(
            "- 693 images, 10 physical board templates\n"
            "- 2,953 annotated instances\n"
            "- Board-template-level split (7/2/1 boards)\n"
            "- Prevents layout-memorization leakage"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Limitations & future work")
    st.markdown(
        "- Both datasets use synthetically-injected defects on controlled imaging setups\n"
        "- Only YOLOv8 variants were explored (YOLOv9, ViTs remain future work)\n"
        "- No active/continual learning loop yet for production drift\n"
        "- Planned: ensembling V1's speed with V2's accuracy, knowledge distillation, "
        "domain adaptation to customer-specific PCB designs"
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# 8. APP ENTRYPOINT / ROUTER
# ============================================================================

PAGES = {
    "🏠 Landing": page_landing,
    "🔍 Live Detection": page_detection,
    "📊 Model Insights": page_insights,
    "ℹ️ About": page_about,
}


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    inject_global_css()

    with st.sidebar:
        st.markdown(f"## {APP_ICON} {APP_TITLE}")
        st.caption("YOLOv8 · PCB Defect Detection")
        selection = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        st.caption("pcb-defect-yolov8 repo")
        found, _ = model_status()
        st.caption(("🟢 weights loaded" if found else "🔴 weights missing"))

    PAGES[selection]()


if __name__ == "__main__":
    main()
