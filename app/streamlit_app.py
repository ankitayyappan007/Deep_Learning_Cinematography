
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import cv2
import numpy as np
import streamlit as st

import torch
import torch.nn as nn
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT / "app"))

from data_loader import load_all, format_percent


st.set_page_config(
    page_title="CineScope AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CINEMA NOIR THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

    .stApp {
        background: #101012;
        color: #F3EEE5;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(16, 16, 18, 0.95);
    }

    h1, h2, h3 {
        color: #F3EEE5;
        font-family: 'Playfair Display', Georgia, serif;
    }

    p, label, li {
        color: #D1CBC2;
    }

    .eyebrow {
        color: #C9A96E;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.20rem;
        text-transform: uppercase;
    }

    .hero {
        padding: 2.1rem 0 1.6rem;
        animation: heroReveal 850ms ease-out both;
    }

    .hero-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: clamp(2.8rem, 6vw, 5.5rem);
        font-weight: 600;
        line-height: 1.05;
        color: #F3EEE5;
        margin: 0.7rem 0 1.2rem;
    }

    .hero-copy {
        max-width: 720px;
        color: #B9B1A7;
        font-size: 1.08rem;
        line-height: 1.8;
    }

    [data-testid="stMetric"] {
        background: #19191D;
        border: 1px solid #37343A;
        border-radius: 10px;
        padding: 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #B9B1A7;
    }

    [data-testid="stMetricValue"] {
        color: #C9A96E;
    }

    div[data-testid="stFileUploader"] {
        background: #19191D;
        border: 1px dashed #62594A;
        border-radius: 10px;
        padding: 0.6rem;
    }

    .stButton > button {
        border-radius: 7px;
        border: 1px solid #C9A96E;
        background: #C9A96E;
        color: #171719;
        font-weight: 700;
        min-height: 2.8rem;
    }

    .stButton > button:hover {
        background: #E0C38D;
        border-color: #E0C38D;
        color: #101012;
    }

    div[role="radiogroup"] label {
        color: #E5DED3;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #37343A;
        border-radius: 8px;
    }

    hr {
        border-color: #37343A;
    }

    .footer {
        color: #77716A;
        font-size: 0.82rem;
        padding: 1.5rem 0;
    }

    @keyframes heroReveal {
        from {
            opacity: 0;
            transform: translateY(18px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .hero {
            animation: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================

@st.cache_data
def get_data():
    return load_all()


data = get_data()


# ============================================================
# MODEL ARCHITECTURES
# Compatible with the existing saved checkpoints.
# ============================================================

class ShotScaleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CameraMovementCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(12, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# LOAD SAVED MODELS
# ============================================================

@st.cache_resource
def load_shot_scale_model():
    checkpoint_path = (
        ROOT / "data" / "processed" / "models"
        / "shot_scale_cnn" / "best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    classes = checkpoint["classes"]
    model = ShotScaleCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes, checkpoint


@st.cache_resource
def load_camera_movement_model():
    checkpoint_path = (
        ROOT / "data" / "processed" / "models"
        / "camera_movement_cnn" / "best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    classes = checkpoint["classes"]
    model = CameraMovementCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_shot_scale(image):
    model, classes, checkpoint = load_shot_scale_model()

    image = image.convert("RGB").resize((128, 128))
    array = np.asarray(image, dtype=np.float32) / 255.0

    mean = np.asarray(
        checkpoint["normalization_mean"],
        dtype=np.float32,
    )
    std = np.asarray(
        checkpoint["normalization_std"],
        dtype=np.float32,
    )

    if mean.size == 1:
        mean = float(mean.item())

    if std.size == 1:
        std = float(std.item())

    array = (array - mean) / std

    tensor = torch.from_numpy(
        array.transpose(2, 0, 1)
    ).unsqueeze(0)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    return classes[predicted_index], probabilities.numpy(), classes


def predict_camera_movement(video_bytes):
    model, classes = load_camera_movement_model()
    frames = []

    with tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    ) as temp:
        temp.write(video_bytes)
        video_path = temp.name

    capture = None

    try:
        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            raise ValueError("Could not open the uploaded video.")

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames < 5:
            raise ValueError("The video must contain at least five frames.")

        indices = np.linspace(
            0, total_frames - 1, 5
        ).round().astype(int)

        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = capture.read()

            if not success:
                raise ValueError("Could not read all five sampled frames.")

            frame = cv2.resize(
                frame,
                (112, 112),
                interpolation=cv2.INTER_AREA,
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    finally:
        if capture is not None:
            capture.release()

        Path(video_path).unlink(missing_ok=True)

    frames = np.asarray(frames, dtype=np.float32)
    differences = np.diff(frames, axis=0) / 255.0
    differences = differences.transpose(
        3, 0, 1, 2
    ).reshape(12, 112, 112)

    tensor = torch.from_numpy(differences).unsqueeze(0)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    return classes[predicted_index], probabilities.numpy(), classes


# ============================================================
# SHARED PREDICTION DISPLAY
# ============================================================

def show_probabilities(classes, probabilities):
    results = sorted(
        zip(classes, probabilities.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    st.markdown("### Class probabilities")

    for label, probability in results:
        st.progress(
            float(probability),
            text=f"{label}: {probability:.1%}",
        )


# ============================================================
# TOP NAVIGATION
# ============================================================

PAGES = [
    "Home",
    "Shot-Scale Explorer",
    "Camera-Movement Explorer",
]

if "page_navigation" not in st.session_state:
    st.session_state["page_navigation"] = "Home"


def navigate_to(page_name):
    st.session_state["page_navigation"] = page_name


st.markdown(
    '<p class="eyebrow">CINESCOPE AI / THE ART OF SEEING FILM</p>',
    unsafe_allow_html=True,
)

page = st.radio(
    "MAIN NAVIGATION",
    PAGES,
    horizontal=True,
    key="page_navigation",
    label_visibility="collapsed",
)

st.divider()


# ============================================================
# 1. HOME
# ============================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <p class="eyebrow">DEEP LEARNING · CINEMATOGRAPHY</p>
            <div class="hero-title">
                Every frame<br>tells a story.
            </div>
            <p class="hero-copy">
                CineScope AI explores the visual language of cinema using
                deep learning. Examine shot scale from individual frames
                and explore camera movement across video clips.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.iframe(
    Path(__file__).parent / "cinema_gallery.html",
    height=370,
)


    st.markdown("---")

    st.markdown(
        """
        <div class="hero">
            <p class="eyebrow">SEE BEYOND THE FRAME</p>
            <div class="hero-title" style="font-size: 2.6rem;">
                Every shot has a language.
            </div>
            <p class="hero-copy">
                From intimate close-ups to expansive long shots,
                from stillness to movement, explore the visual choices
                that shape the cinematic experience.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feature1, feature2 = st.columns(2, gap="large")

    with feature1:
        st.markdown("### Shot scale")
        st.caption(
            "Explore how framing changes what the audience sees and feels."
        )

    with feature2:
        st.markdown("### Camera movement")
        st.caption(
            "Explore motion patterns and follow shot-scale changes "
            "through the Shot DNA timeline."
        )


# ============================================================
# 2. SHOT-SCALE EXPLORER
# ============================================================

elif page == "Shot-Scale Explorer":

    st.markdown(
        '<p class="eyebrow">01 / FRAME ANALYSIS</p>',
        unsafe_allow_html=True,
    )

    st.title("Shot-Scale Explorer")

    st.write(
        "Upload a movie frame to estimate its shot-scale category "
        "using the existing trained model."
    )

    st.caption("Model classes: CS, ECS, FS, LS, MS")

    uploaded_image = st.file_uploader(
        "Choose a movie frame",
        type=["jpg", "jpeg", "png"],
        key="shot_scale_upload",
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")

        st.image(
            image,
            caption="Uploaded movie frame",
            width="stretch",
        )

        if st.button("Analyze frame", key="predict_shot_scale"):
            try:
                with st.spinner("Analyzing the frame..."):
                    predicted_class, probabilities, classes = (
                        predict_shot_scale(image)
                    )

                st.success(
                    f"Predicted shot scale: {predicted_class}"
                )

                show_probabilities(classes, probabilities)

                st.caption(
                    "Predictions are estimates, not guaranteed ground truth."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# ============================================================
# 3. CAMERA-MOVEMENT EXPLORER
# Includes camera movement and Shot DNA as selectable tools.
# ============================================================

elif page == "Camera-Movement Explorer":

    st.markdown(
        '<p class="eyebrow">02 / MOTION & TEMPORAL ANALYSIS</p>',
        unsafe_allow_html=True,
    )

    st.title("Camera-Movement Studio")

    mode = st.radio(
        "Choose an analysis",
        [
            "Camera-Movement Prediction",
            "Shot DNA Timeline",
        ],
        horizontal=True,
        key="movement_studio_mode",
    )

    st.divider()

    # --------------------------------------------------------
    # CAMERA-MOVEMENT PREDICTION
    # --------------------------------------------------------

    if mode == "Camera-Movement Prediction":

        st.subheader("Camera-Movement Explorer")

        st.write(
            "Upload a short video clip to estimate its movement category "
            "using the existing trained model."
        )

        st.caption(
            "Model classes: Static, Motion, Push, Pull, Multi_movement"
        )

        uploaded_video = st.file_uploader(
            "Choose a video clip",
            type=["mp4", "avi", "mov", "mkv"],
            key="movement_video_upload",
        )

        if uploaded_video is not None:
            st.video(uploaded_video)

            if st.button(
                "Analyze camera movement",
                key="predict_camera_movement",
            ):
                try:
                    with st.spinner("Analyzing video frames..."):
                        predicted_class, probabilities, classes = (
                            predict_camera_movement(
                                uploaded_video.getvalue()
                            )
                        )

                    st.success(
                        f"Predicted camera movement: {predicted_class}"
                    )

                    show_probabilities(classes, probabilities)

                    st.caption(
                        "Predictions are estimates from the existing model."
                    )

                except Exception as error:
                    st.error(f"Video prediction failed: {error}")

    # --------------------------------------------------------
    # SHOT DNA TIMELINE
    # --------------------------------------------------------

    else:

        st.subheader("Shot DNA Timeline")

        st.write(
            "Explore how predicted shot scales change across a video clip. "
            "Frames are sampled at regular intervals, not detected cuts."
        )

        timeline_video = st.file_uploader(
            "Upload a clip for Shot DNA analysis",
            type=["mp4", "avi", "mov", "mkv"],
            key="timeline_video_upload",
        )

        sample_count = st.slider(
            "Number of sampled frames",
            min_value=5,
            max_value=20,
            value=8,
            step=1,
            key="timeline_sample_count",
        )

        if timeline_video is not None:
            st.video(timeline_video)

            if st.button(
                "Generate Shot DNA",
                key="generate_shot_dna",
            ):
                frames_for_timeline = []
                timeline_path = None
                capture = None

                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".mp4",
                        delete=False,
                    ) as temp:
                        temp.write(timeline_video.getvalue())
                        timeline_path = temp.name

                    capture = cv2.VideoCapture(timeline_path)

                    if not capture.isOpened():
                        raise ValueError(
                            "Could not open the uploaded video."
                        )

                    total_frames = int(
                        capture.get(cv2.CAP_PROP_FRAME_COUNT)
                    )

                    fps = capture.get(cv2.CAP_PROP_FPS)

                    if total_frames < 1:
                        raise ValueError(
                            "No frames were found in this video."
                        )

                    indices = np.linspace(
                        0,
                        total_frames - 1,
                        min(sample_count, total_frames),
                    ).round().astype(int)

                    for index in indices:
                        capture.set(
                            cv2.CAP_PROP_POS_FRAMES,
                            int(index),
                        )

                        success, frame = capture.read()

                        if not success:
                            continue

                        rgb = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB,
                        )

                        pil_image = Image.fromarray(rgb)

                        label, probabilities, classes = (
                            predict_shot_scale(pil_image)
                        )

                        timestamp = (
                            float(index / fps)
                            if fps and fps > 0
                            else 0.0
                        )

                        frames_for_timeline.append({
                            "Frame": int(index),
                            "Time (s)": round(timestamp, 2),
                            "Prediction": label,
                            "Probability": float(
                                np.max(probabilities)
                            ),
                            "Image": rgb,
                        })

                except Exception as error:
                    st.error(f"Shot DNA analysis failed: {error}")

                finally:
                    if capture is not None:
                        capture.release()

                    if timeline_path:
                        Path(timeline_path).unlink(missing_ok=True)

                if frames_for_timeline:
                    st.markdown("### Sampled frames")

                    columns = st.columns(
                        min(4, len(frames_for_timeline))
                    )

                    for i, item in enumerate(frames_for_timeline):
                        with columns[i % len(columns)]:
                            st.image(
                                item["Image"],
                                caption=(
                                    f"{item['Time (s)']}s · "
                                    f"{item['Prediction']}"
                                ),
                                width="stretch",
                            )

                            st.caption(
                                f"Frame {item['Frame']} · "
                                f"Confidence {item['Probability']:.1%}"
                            )

                    st.markdown("### Shot-scale sequence")

                    sequence = " → ".join(
                        item["Prediction"]
                        for item in frames_for_timeline
                    )

                    st.markdown(f"**{sequence}**")

                    st.dataframe(
                        [
                            {
                                "Frame": item["Frame"],
                                "Time (s)": item["Time (s)"],
                                "Predicted scale": item["Prediction"],
                                "Top probability": round(
                                    item["Probability"], 4
                                ),
                            }
                            for item in frames_for_timeline
                        ],
                        width="stretch",
                        hide_index=True,
                    )

                    st.caption(
                        "Shot DNA uses the existing shot-scale model. "
                        "A sampled frame is not necessarily a separate shot."
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">CINESCOPE AI · DEEP LEARNING CINEMATOGRAPHY'
    ' · THE ART OF SEEING FILM</div>',
    unsafe_allow_html=True,
)
