import os
import streamlit as st
import cv2
import numpy as np
import yaml
import tomli
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
import base64
# Add to imports
import hashlib
import matplotlib.pyplot as plt
import tomli_w  # For TOML writing
import json     # For JSON support
from itertools import combinations
import tempfile
import platform
import subprocess
from pathlib import Path
import math
import glob
from nems_lbhb import db
import deeplabcut
import shutil
import pandas as pd
import re
import datetime


# --- Persistence for user preferences ---
PREFS_FILE = Path.home() / ".dlc_labeler_prefs.json"


class _PlotlyFigureWithConfig:
    def __init__(self, fig, config):
        self._fig = fig
        self._config = config

    def to_json(self):
        payload = json.loads(self._fig.to_json())
        payload["config"] = self._config
        return json.dumps(payload)


def load_user_prefs():
    """Load user preferences from persistent storage."""
    if PREFS_FILE.exists():
        try:
            with open(PREFS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_user_prefs(prefs):
    """Save user preferences to persistent storage."""
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save preferences: {e}")


st.set_page_config(layout="wide")

params = st.query_params

initial_params = {
    # "cap": None,
    "video_path": params.get('video', ''),
    "current_frame": None,
    "points": {},
    "selected_point": None,
    "fixation_point": None,
    "distances": {},
    "speakers": {"R": {"distance": 1.0, "angle": 30.0}, "L": {"distance": 1.0, "angle": -30.0}},
    "speaker_orientation": "Top",
    "ppm_avg": 1.0,
    "cdict": None,
    "plotly_key_version": 0,
    "last_imported_config": None,
    }


for k, v in initial_params.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("""
    <style>
    .plotly-plot {
        background-color: transparent !important;
    }
    .modebar,
    .modebar-container,
    .plotly .modebar,
    .plotly .modebar-container,
    .plotly .modebar-btn {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .block-container {
        padding-left: 0rem;
        padding-right: 0rem;
    }
    .element-container {
        width: 100% !important;
    }
    iframe {
        width: 100% !important;
    }
    </style>
""", unsafe_allow_html=True)

def load_dlc_config_fields(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    st.session_state.dlc_project_name = config.get("Task", "")
    st.session_state.dlc_experimenter = config.get("scorer", "")
    st.session_state.dlc_working_dir = config.get("project_path", "")
    st.session_state.dlc_bodyparts = config.get("bodyparts", [])
    # video_sets is a dict with paths as keys
    st.session_state.dlc_videos = list(config.get("video_sets", {}).keys())
    st.session_state.dlc_notes = config.get("notes", "")
    project_path = Path(config.get("project_path", ""))
    dlc_model_path = project_path / "dlc-models"

    # --- Handle iteration ---
    iterations = sorted([p.name for p in dlc_model_path.glob("iteration-*") if p.is_dir()])
    current_iter = config.get("iteration", iterations[0] if iterations else "iteration-0")
    st.session_state.selected_iter = f'iteration-{current_iter}'


def get_cmap_color(cmap_name='gist_rainbow'):
    names = list(st.session_state.points.keys())
    if not names:
        st.session_state.cdict = {}
        return
    crange = np.linspace(0, 1, len(names))
    try:
        cmap = plt.get_cmap('viridis')
    except Exception:
        cmap = plt.cm.viridis
    cdict = {name: cmap(crange[ni]) for ni, name in enumerate(names)}
    cdict = {n: f'rgb({int(v[0] * 255)}, {int(v[1] * 255)}, {int(v[2] * 255)})' for n, v in cdict.items()}
    st.session_state.cdict = cdict

def compute_fixation_point_from_state():
    """Compute fixation point from current state without UI side-effects."""
    if not st.session_state.points:
        return None
    fix_key = st.session_state.get("fixation_point_select", None)
    if fix_key not in st.session_state.points:
        return None
    try:
        shift_x_m = float(st.session_state.get("fixation_shift_x", "0.0"))
        shift_y_m = float(st.session_state.get("fixation_shift_y", "0.0"))
    except ValueError:
        return None

    base_x, base_y = st.session_state.points[fix_key]
    ppm = st.session_state.ppm_avg
    fx, fy = base_x, base_y

    xaxis_points = st.session_state.get("xaxis_points", None)
    if xaxis_points and all(p in st.session_state.points for p in xaxis_points):
        x1 = np.array(st.session_state.points[xaxis_points[0]])
        x2 = np.array(st.session_state.points[xaxis_points[1]])
        delta = x2 - x1
        if np.hypot(delta[0], delta[1]) > 0:
            theta = np.arctan2(delta[1], delta[0])
            cos_theta = np.cos(-theta)
            sin_theta = np.sin(-theta)
            R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            base_rot = (np.array([base_x, base_y]) - x1) @ R.T
            base_rot = base_rot + np.array([shift_x_m, shift_y_m]) * ppm
            R_inv = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
            fx, fy = (base_rot @ R_inv.T) + x1
    else:
        fx = base_x + shift_x_m * ppm
        fy = base_y + shift_y_m * ppm

    return float(fx), float(fy)

def handle_image_click(x, y):
    """Validate coordinates against actual image dimensions"""
    img = st.session_state.get('current_frame', None)
    if img is None:
        st.error("No image loaded")
        return
    # PIL Image provides width/height
    iw, ih = getattr(img, 'width', None), getattr(img, 'height', None)
    if iw is None or ih is None:
        st.error("Invalid image")
        return
    if x < 0 or x >= iw or y < 0 or y >= ih:
        st.error("Coordinates out of image bounds")
        return
    if st.session_state.selected_point:
        # Check minimum distance from other markers
        min_dist = 20  # pixels
        for name, (ox, oy) in st.session_state.points.items():
            if name != st.session_state.selected_point:
                dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                if dist < min_dist:
                    st.error(f"Too close to {name}")
                    return
        st.session_state.points[st.session_state.selected_point] = (x, y)

def create_plotly_figure(image):
    """Create Plotly figure with guaranteed 3-channel RGB"""
    # Ensure PIL Image -> RGB numpy array
    try:
        pil_img = image
        if hasattr(pil_img, 'convert'):
            pil_img = pil_img.convert('RGB')
        img_array = np.asarray(pil_img)
    except Exception:
        img_array = np.array(image)

    # If grayscale (2D), expand to 3 channels
    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    # If single channel last-dimension, repeat
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array]*3, axis=-1)

    # Create figure with RGB image
    fig = go.Figure(go.Image(
        z=img_array,
        colormodel='rgb',
        hoverinfo='none'
    ))
    cdict = st.session_state.get('cdict', None)
    if cdict is None and st.session_state.points:
        get_cmap_color()
        cdict = st.session_state.get('cdict', {})

    # In your create_plotly_figure function:
    for name, (x, y) in st.session_state.points.items():
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(
                size=12,
                color=cdict.get(name, 'rgb(255,255,255)'),  # Use generated color
                line=dict(width=2, color='white')
            ),
            text=[name],
            textposition='top center',
            textfont=dict(color='white'),
            showlegend=False
        ))

    fig.update_layout(
        width=image.width,
        height=image.height,
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode='closest',
        hoverdistance=5,
        xaxis=dict(
            visible=False,
            range=[0, image.width],
            fixedrange=True,
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            visible=False,
            range=[image.height, 0],
            fixedrange=True
        )
    )

    # Optional debug output (commented out by default)
    # st.write("Image shape:", img_array.shape)
    # st.write("Image dtype:", img_array.dtype)
    # st.write("Image min/max:", img_array.min(), img_array.max())

    return fig

def save_config(config_name, video_path):
    """Save current configuration and figure snapshot"""
    if not config_name:
        st.error("Please enter a configuration name")
        return

    # Define output paths
    config_path = Path(config_name)
    plot_path = config_path.with_name(config_path.stem.replace("dlc_arena_config", "dlc_arena_plot") + ".png")

    def _extract_year_from_string(s):
        if not s:
            return None
        for m in re.finditer(r"(19|20)\d{2}", s):
            year = int(m.group(0))
            if 1990 <= year <= datetime.datetime.now().year + 1:
                return str(year)
        return None

    def _query_year_from_celldb(penname):
        if not penname:
            st.error("Celldb query skipped: missing penname.")
            return None

        sql = f""" select gDataRaw.*,gPenetration.pendate from gDataRaw INNER JOIN gCellMaster on gDataRaw.masterid = gCellMaster.id INNER JOIN gPenetration on gPenetration.id = gCellMaster.penid where parmfile like '{penname}' order by pendate desc"""
        try:
            # [AGENT EDIT START | agent: claude-sonnet-5 | user: svd | reason: con=engine.connect() was never closed (leaked connection on every call); the params dict was also dead -- sql has no `:parm` placeholder, penname is already inlined above. db.pd_query() already opens/closes its connection correctly and is the standard path used elsewhere in this codebase. | date: 2026-08-07]
            pend = db.pd_query(sql)
            # [AGENT EDIT END]
        except Exception as e:
            st.error(f"Celldb query failed for penname {penname}: {e}")
            return None
        if pend is None or pend.empty:
            st.error(f"Celldb query returned no records for penname {penname}.")
            return None
        st.write(f'Queried celldb for penname {penname}, found {len(pend)} records.')
        pendate = pend.pendate.iloc[0]
        try:
            year_val = int(pendate.year)
        except Exception:
            try:
                year_val = int(pd.to_datetime(pendate).year)
            except Exception:
                st.error(f"Failed to parse pendate for penname {penname}: {pendate}")
                return None
        return str(year_val)


    def _build_web_arena_jpg_path(video_path_value, config_path_value=None):
        st.write('Building web arena JPG path...')
        vp_local = Path(video_path_value)
        runid = vp_local.parent.name if vp_local.parent and vp_local.parent.name else vp_local.stem
        st.write(f'Inferred run ID: {runid}')
        animal_id = None
        if config_path_value:
            cp_local = Path(config_path_value)
            # Expected: /auto/data/daq/<Animal>/<AnimalID>/sorted/<file>
            if len(cp_local.parents) > 2 and cp_local.parents[0].name == "sorted":
                animal_id = cp_local.parents[2].name
        if not animal_id and len(vp_local.parents) > 1 and vp_local.parents[1].name:
            animal_id = vp_local.parents[1].name
        if not animal_id and runid:
            m = re.match(r"([A-Za-z]+\d+)", runid)
            if m:
                animal_id = m.group(1)
        year = _query_year_from_celldb(runid)
        if not (animal_id and year and runid):
            return None
        out_dir = Path("/auto/data/web/analysis") / animal_id.lower() / year
        return out_dir / f"{runid}.arena_config.jpg"

    def _infer_dlc_model_from_sorted(video_path_value):
        vp_local = Path(video_path_value)
        runid = vp_local.parent.name if vp_local.parent and vp_local.parent.name else vp_local.stem
        sorted_path = vp_local.parents[1] / "sorted"
        h5_path = sorted_path / f"{runid}.dlc.h5"
        if not h5_path.exists():
            return None, None
        try:
            target = h5_path.resolve()
        except Exception:
            target = h5_path
        target_stem = target.stem
        if "_DLC_" in target_stem:
            model_part = target_stem.split("_DLC_", 1)[1]
        else:
            model_part = target_stem
        # Strip trailing shuffle suffix like shuffle1_50000
        model_part = re.sub(r"(?:_)?shuffle\\d+_\\d+$", "", model_part)
        config_guess = Path("/auto/data/dlc") / model_part / "config.yaml"
        config_path = str(config_guess) if config_guess.exists() else None
        return model_part, config_path

    # Try saving the Plotly figure
    if st.session_state.get('current_frame', None) is not None:
        fig = create_plotly_figure(st.session_state.get('current_frame', None))

        # Add fixation if defined
        if st.session_state.fixation_point is not None:
            fx, fy = st.session_state.fixation_point
            fig.add_trace(go.Scatter(
                x=[fx], y=[fy],
                mode="markers",
                marker=dict(color="red", size=15, symbol="x"),
                name="Fixation Point",
                showlegend=False
            ))

        # Export to PNG
        try:
            fig.write_image(str(plot_path), format="png", width=st.session_state.current_frame.width, height=st.session_state.current_frame.height)
            st.success(f"Saved plot image to: {plot_path}")
        except Exception as e:
            st.error(f"Could not save plot image: {e}")
        # Also export a JPG copy for web, when saving to /sorted
        try:
            st.write('Attempting to save web JPG copy if applicable...')
            if "sorted" in config_path.parts:
                web_jpg_path = _build_web_arena_jpg_path(video_path, config_path)
                if web_jpg_path is None:
                    st.error("Could not determine web JPG path from celldb. Skipping web JPG save.")
                else:
                    web_jpg_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.write_image(str(web_jpg_path), format="jpg", width=st.session_state.current_frame.width, height=st.session_state.current_frame.height)
                    st.success(f"Saved web JPG to: {web_jpg_path}")
        except Exception as e:
            st.warning(f"Could not save web JPG: {e}")
    else:
        st.warning("No frame loaded — cannot save plot image.")

    # Build config dictionary
    dlc_model_name = st.session_state.get("dlc_model_select", "")
    dlc_model_config = st.session_state.get("selected_dlc_model_config", "")
    inferred_model_name, inferred_config_path = _infer_dlc_model_from_sorted(video_path)
    if inferred_model_name:
        dlc_model_name = inferred_model_name
        if inferred_config_path:
            dlc_model_config = inferred_config_path

    config = {
        'metadata': {
            'video_path': st.session_state.get('video_path', None),
            'frame_number': st.session_state.get('frame_number_input', None),
            'frame_width': getattr(st.session_state.get('current_frame', None), 'width', None),
            'frame_height': getattr(st.session_state.get('current_frame', None), 'height', None)
        },
        'points': st.session_state.points,
        'distances': st.session_state.distances,
        'ppm_pairs': st.session_state.get('pixels_per_meter', {}),
        'average_ppm': {'average_ppm': st.session_state.ppm_avg},
        'fixation_base_marker': st.session_state.get("fixation_point_select", None),
        'fixation_shift': {
            'x_shift_m': st.session_state.get("fixation_shift_x", "0.0"),
            'y_shift_m': st.session_state.get("fixation_shift_y", "0.0")
        },
        'fixation_point': {'fixation_point': st.session_state.fixation_point},
        'xaxis_pair': {'xaxis_pair': st.session_state.xaxis_points},
        'speaker_orientation': st.session_state.speaker_orientation,
        'speakers': st.session_state.speakers,
        'arena_plot_path': str(plot_path),
        'dlc_model': {
            'name': dlc_model_name,
            'config_path': dlc_model_config
        }
    }

    # Save to file
    if config_name.endswith('.toml'):
        with open(config_name, 'wb') as f:
            tomli_w.dump(config, f)
    elif config_name.endswith('.yaml'):
        with open(config_name, 'w') as f:
            yaml.dump(config, f)
    elif config_name.endswith('.json'):
        with open(config_name, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        st.error("Unsupported format. Use .toml, .yaml, or .json")
        return

    st.success(f"Configuration saved as {config_name}")


def import_config(config_file):
    """Import previously saved configuration from TOML and apply it"""
    try:
        config = tomli.load(config_file)

        if 'metadata' in config:
            config_video = config['metadata'].get('video_path', None)
            frame_number = config['metadata'].get('frame_number', 0)
            st.session_state.imported_config_video = config_video
            if config_video and os.path.exists(config_video):
                current_video = st.session_state.get("video_path", None)
                if current_video and str(Path(current_video)) == str(Path(config_video)):
                    try:
                        st.session_state.frame_number_input = int(frame_number)
                    except (TypeError, ValueError):
                        st.session_state.frame_number_input = 0

                    cap = cv2.VideoCapture(config_video)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_number_input)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        if frame.ndim == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        elif frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                        else:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.session_state.current_frame = Image.fromarray(frame)
                else:
                    st.warning(f"Config video path differs from current: {config_video}")
            elif config_video:
                st.warning(f"Config video path not found: {config_video}")


        # Points (marker positions)
        if 'points' in config:
            st.session_state.points = {k: tuple(v) for k, v in config['points'].items()}
            get_cmap_color()
            if st.session_state.points:
                first_key = next(iter(st.session_state.points))
                if st.session_state.get("selected_point") not in st.session_state.points:
                    st.session_state.selected_point = first_key
                if st.session_state.get("point_remove_select") not in st.session_state.points:
                    st.session_state.point_remove_select = first_key

        # Distances between marker pairs
        if 'distances' in config:
            st.session_state.distances = config['distances']

        # Pixels per meter values
        if 'ppm_pairs' in config:
            st.session_state.pixels_per_meter = config['ppm_pairs']

        # Average ppm
        if 'average_ppm' in config and 'average_ppm' in config['average_ppm']:
            st.session_state.ppm_avg = config['average_ppm']['average_ppm']

        if 'fixation_base_marker' in config:
            fix_key = config['fixation_base_marker']
            if not st.session_state.points or fix_key in st.session_state.points:
                st.session_state.fixation_point_select = fix_key

        if 'fixation_shift' in config:
            st.session_state.fixation_shift_x = str(config['fixation_shift'].get('x_shift_m', "0.0"))
            st.session_state.fixation_shift_y = str(config['fixation_shift'].get('y_shift_m', "0.0"))

        # Fixation point
        if 'fixation_point' in config and 'fixation_point' in config['fixation_point']:
            st.session_state.fixation_point = tuple(config['fixation_point']['fixation_point'])

        # X-axis points
        if 'xaxis_pair' in config and 'xaxis_pair' in config['xaxis_pair']:
            xaxis_points = tuple(config['xaxis_pair']['xaxis_pair'])
            st.session_state.xaxis_points = xaxis_points
            if len(xaxis_points) == 2 and all(p in st.session_state.points for p in xaxis_points):
                st.session_state.xaxis1_point_select = xaxis_points[0]
                st.session_state.xaxis2_point_select = xaxis_points[1]

        if 'speakers' in config:
            st.session_state.speakers = config['speakers']

        if 'speaker_orientation' in config:
            st.session_state.speaker_orientation = config['speaker_orientation']

        if 'dlc_model' in config:
            st.session_state.selected_dlc_model_config = config['dlc_model'].get('config_path', "")

        current_frame = st.session_state.get('current_frame', None)
        if current_frame is not None and 'metadata' in config:
            saved_w = config['metadata'].get('frame_width', None)
            saved_h = config['metadata'].get('frame_height', None)
            if saved_w and saved_h:
                scale_x = current_frame.width / float(saved_w)
                scale_y = current_frame.height / float(saved_h)
                if abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6:
                    st.session_state.points = {
                        k: (float(v[0]) * scale_x, float(v[1]) * scale_y)
                        for k, v in st.session_state.points.items()
                    }
                    if st.session_state.fixation_point is not None:
                        fx, fy = st.session_state.fixation_point
                        st.session_state.fixation_point = (float(fx) * scale_x, float(fy) * scale_y)
                    if abs(scale_x - scale_y) <= 1e-3:
                        if 'ppm_pairs' in config:
                            st.session_state.pixels_per_meter = {
                                k: float(v) * scale_x for k, v in config['ppm_pairs'].items()
                            }
                        if 'average_ppm' in config and 'average_ppm' in config['average_ppm']:
                            st.session_state.ppm_avg = float(config['average_ppm']['average_ppm']) * scale_x
                        st.warning(
                            f"Scaled imported points/fixation and ppm by x={scale_x:.4f}, y={scale_y:.4f} "
                            f"to match current frame size."
                        )
                    else:
                        st.warning(
                            f"Scaled imported points/fixation by x={scale_x:.4f}, y={scale_y:.4f} to match "
                            f"current frame size. PPM values were not scaled due to aspect ratio change."
                        )
            else:
                st.warning("Imported config is missing saved frame size; points may not align to current frame.")

        current_video = st.session_state.get("video_path", None)
        if current_video:
            st.session_state.auto_config_loaded_for = str(Path(current_video))
        st.session_state.plotly_key_version = st.session_state.get("plotly_key_version", 0) + 1
        st.success("Configuration imported successfully!")

    except Exception as e:
        st.error(f"Failed to load configuration: {e}")

def import_config_path(config_path):
    """Import configuration from a file path."""
    try:
        with open(config_path, "rb") as f:
            import_config(f)
        return True
    except Exception as e:
        st.error(f"Failed to load configuration from {config_path}: {e}")
        return False

def load_sorted_config_for_session(video_path):
    vp = Path(video_path)
    session_name = vp.parents[0].name
    sorted_path = vp.parents[1] / "sorted"
    if not sorted_path.exists():
        return None
    expected_name = f"{vp.stem}-{vp.parent.stem}-dlc_arena_config.toml"
    exact_match = sorted_path / expected_name
    if exact_match.exists():
        latest_config = exact_match
    else:
        candidates = list(sorted_path.glob(f"{vp.stem}*dlc_arena_config.toml"))
        if not candidates:
            candidates = list(sorted_path.glob(f"{session_name}*dlc_arena_config.toml"))
        if not candidates:
            return None
        def score(p):
            common = os.path.commonprefix([p.name, expected_name])
            return (len(common), p.stat().st_mtime)
        latest_config = max(candidates, key=score)
    if import_config_path(latest_config):
        st.session_state.auto_config_loaded_for = str(vp)
        st.info(f"Auto-loaded config: {latest_config.name}")
        return latest_config
    return None

# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- training runs are
# attributed to a person and no longer force_rerun over an identical queued job
# | date: 2026-08-06]
def current_dlc_user():
    """Who is driving this Streamlit session, for job attribution.

    The app and its queued jobs both run as `lbhb`, so the Linux account says
    nothing about who asked for a training run.  This is the name that ends up in
    the tQueue row, in `<project>/.dlc_train.lock` and in `TRAINRUN.toml`; when a
    lock refuses a second run, it is the name the refusal prints.
    """
    name = st.session_state.get("dlc_run_user") or load_user_prefs().get("dlc_run_user")
    if name:
        return str(name).strip()
    import getpass
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"
# [AGENT EDIT END]


def queue_dlc_job(videopath, configpath, destpath=None, action='fit', extra_note="", init_weights=None,
                  user=None):
    """
    Launch a DLC analysis job using a wrapper script, based on selected video and DLC config.

    Args:
        init_weights: Optional path to snapshot for transfer learning (without .meta extension)
        user: who requested the job (defaults to current_dlc_user()).  Recorded on the
            tQueue row and passed to dlc_fit_script as `user=`, which puts it in the
            training lock and TRAINRUN.toml.
    """
    try:
        # Extract raw ID or video ID from video path
        video_name = str(Path(videopath).parents[0].name) + '/' + str(Path(videopath).name)

        # Command setup (edit these paths as needed)
        wrapper_path = "/auto/users/lbhb/Code/nems_db/nems_lbhb/motor/dlc_fit_wrapper"
        script_path = "/auto/users/lbhb/Code/nems_db/nems_lbhb/motor/dlc_fit_script.py"

        # pass model specifics out to make notes more meaningful
        with open(configpath, "r") as f:
            cfg = yaml.safe_load(f)
        iter = cfg['iteration']
        snapshot = cfg['snapshotindex']
        if destpath is None:
            args = [videopath, action, configpath]
        else:
            args = [videopath, action, configpath, destpath]

        # Add init_weights if provided (for transfer learning)
        if init_weights:
            args.append(f"init_weights={init_weights}")

        # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- attribute
        # the run to a person, and stop force_rerun from stacking duplicate TRAINING jobs
        # on one project | date: 2026-08-06]
        if user is None:
            user = current_dlc_user()
        # review fix 6 -- "Your name" is free text from a Streamlit text_input; it was
        # being appended verbatim as a `key=value` argv token. Anything containing '='
        # would itself be mis-parsed as another key=value arg by
        # dlc_fit_script.parse_key_value_args, and anything with spaces/shell metacharacters
        # is asking for trouble once it crosses into a queued subprocess's argv. Slug it
        # with the same helper dlc_train_guard.run_modelprefix already uses for user/run_id.
        from nems_lbhb.motor import dlc_train_guard
        args.append(f"user={dlc_train_guard._slug(user)}")

        is_training = ('train' in action)
        # force_rerun=True re-queues a job whose note already exists.  For an
        # analysis that is just a redo; for training it is how two runs end up
        # writing into one project's dlc-models at once, which destroys both.
        # dlc_train_guard refuses that at the last moment; not queueing it in the
        # first place is better.
        force_rerun = not is_training

        if is_training:
            add_on = ''
        else:
            add_on = '_'+extra_note

        GPU_job = 1
        # Run the job via subprocess
        job_note = f'DLC Job:config_{str(Path(configpath).name)}-iter_{iter}-snapshotindex_{snapshot}-video_{video_name}-action_{action+add_on}'
        result = db.add_job_to_queue(
                args,
                job_note,
                force_rerun=force_rerun, GPU_job=GPU_job, linux_user='lbhb', user=user,
                executable_path=wrapper_path, script_path=script_path,
                priority=1)
        # [AGENT EDIT END]

        # Unpack the zip result to get queue IDs and messages
        result_list = list(result)
        if result_list:
            queueids, messages = zip(*result_list)
            return f"Job queued (ID: {queueids[0]}): {messages[0].strip()}"
        else:
            return f"Job submitted: {job_note}"

    except Exception as e:
        return f"Error launching job: {e}"

def is_dlc_model_trained(config_path, shuffle=1):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    project_path = Path(cfg["project_path"])
    training_folder = project_path / "dlc-models"
    if not training_folder.exists():
        return False

    # Search for snapshot files
    snapshot_files = list(training_folder.rglob(f"snapshot-*.meta"))

    return len(snapshot_files) > 0

def outlier_frames_exist(config_path, video_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    project_path = Path(cfg["project_path"])
    video_name = Path(video_path).parents[0].name

    labeled_data_dir = project_path / "labeled-data" / video_name
    if not labeled_data_dir.exists():
        return False

    frame_files = list(labeled_data_dir.glob("*.png")) + list(labeled_data_dir.glob("*.bmp"))
    st.write(f'Labeled frames ({len(frame_files)}): {labeled_data_dir}')
    return len(frame_files) > 0

def analyze_to_temp():
    video_path = Path(st.session_state.get("video_path", ""))
    config_path = st.session_state.get("selected_dlc_model_config", "")
    temp_path = str(video_path.parents[1].joinpath(f'tmp/dlc/{video_path.parents[0].name}'))
    st.write(f'Adding job to analyze to: {temp_path}')
    if not video_path or not os.path.exists(video_path):
        st.error("Invalid or missing video path.")
    elif not config_path or not os.path.exists(config_path):
        st.error("Invalid or missing DLC config path.")
    else:
        message = queue_dlc_job(video_path, config_path, temp_path, 'fit',extra_note='temp_analysis')
        st.success(f"DLC job queued: {message}")
    # deeplabcut.analyze_videos(config_path, videos=[str(video_path)], videotype='.avi', destfolder=str(temp_path))
    # st.success('Results saved. ')

def list_dlc_h5_files(folder_path, match_substr=None):
    if not folder_path.exists():
        return []
    files = sorted(folder_path.glob("*.h5"))
    if match_substr:
        files = [p for p in files if match_substr in p.name]
    return files

def get_dlc_highlight_tokens(config_path):
    if not config_path or not os.path.exists(config_path):
        return None, None
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return None, None

    model_name = Path(config_path).parent.name
    highlight_base = model_name.split('-', 1)[0] if model_name else None
    shuffle = cfg.get("shuffle", None)

    iter_num = cfg.get("iteration", None)
    snapshot_idx = cfg.get("snapshotindex", None)
    snapshot_val = None

    if iter_num is not None:
        try:
            iter_num = int(iter_num)
        except Exception:
            iter_num = None

    if iter_num is not None:
        project_path = Path(cfg.get("project_path", ""))
        dlc_model_path = project_path / "dlc-models"
        iter_dir = dlc_model_path / f"iteration-{iter_num}"
        snapshot_dir = next(iter(iter_dir.glob("*/train")), None)
        if snapshot_dir and snapshot_dir.exists():
            snapshot_indices = sorted(
                int(p.stem.split("-")[1]) for p in snapshot_dir.glob("snapshot-*.meta")
            )
            if snapshot_indices:
                if not isinstance(snapshot_idx, int):
                    snapshot_idx = len(snapshot_indices) - 1
                snapshot_idx = max(0, min(snapshot_idx, len(snapshot_indices) - 1))
                snapshot_val = snapshot_indices[snapshot_idx]

    tokens = []
    if shuffle is not None:
        tokens.append(f"shuffle{shuffle}")
    if snapshot_val is not None:
        tokens.append(f"_{snapshot_val}")

    return highlight_base, tokens if tokens else None

def render_h5_list(label, folder_path, match_substr=None, highlight_substr=None, highlight_tokens=None):
    st.caption(f"{label}: {folder_path}")
    h5_files = list_dlc_h5_files(folder_path, match_substr=match_substr)
    if not h5_files:
        st.write("No .h5 files found.")
        return
    lines = []
    highlight_key = highlight_substr.lower() if highlight_substr else None
    highlight_tokens_l = [t.lower() for t in highlight_tokens] if highlight_tokens else []
    for p in h5_files:
        name_lower = p.name.lower()
        token_match = all(t in name_lower for t in highlight_tokens_l) if highlight_tokens_l else True
        if highlight_key and highlight_key in name_lower and token_match:
            lines.append(f"- <span style='color:#d35400; font-weight:600;'>{p.name}</span>")
        else:
            lines.append(f"- {p.name}")
    h5_text = "\n".join(lines)
    st.markdown(
        "<div style='white-space: pre-wrap; word-break: break-word;'>"
        f"{h5_text}"
        "</div>",
        unsafe_allow_html=True,
    )

def video_to_temp():
    video_path = Path(st.session_state.get("video_path", ""))
    config_path = st.session_state.get("selected_dlc_model_config", "")
    temp_path = video_path.parents[1].joinpath(f'tmp/dlc/{video_path.parents[0].name}')
    st.write(f'Adding job to label video to: {temp_path}')
    if not video_path or not os.path.exists(video_path):
        st.error("Invalid or missing video path.")
    elif not config_path or not os.path.exists(config_path):
        st.error("Invalid or missing DLC config path.")
    else:
        # Check if .h5 file exists (required for labeled video creation)
        h5_files = list(temp_path.glob(f"{video_path.stem}*.h5"))
        if not h5_files:
            st.warning(f"No .h5 analysis file found in {temp_path}. Run 'Analyze DLC to temp' first.")
            return

        message = queue_dlc_job(video_path, config_path, str(temp_path), 'video', extra_note='temp_labeled_video')
        st.success(f"DLC job queued: {message}")

        # Show expected labeled video path
        # DLC labeled videos follow pattern: <video_stem>DLC_<model>_labeled.mp4
        model_name = Path(config_path).parent.name
        expected_video = temp_path / f"{video_path.stem}DLC_{model_name}_labeled.mp4"
        st.info(f"Expected labeled video path: {expected_video}")


def get_video_duration(video_path):
    """Get video duration in seconds using OpenCV, ffprobe, or ffmpeg."""
    # Try OpenCV first (most likely to be available in DLC environment)
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and frame_count > 0:
                return frame_count / fps
    except Exception:
        pass

    # Try ffprobe
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass

    # Fallback: use ffmpeg to get duration from stream info
    cmd2 = [
        "ffmpeg", "-i", str(video_path),
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
        # Parse duration from stderr (ffmpeg outputs info to stderr)
        import re
        match = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", result.stderr)
        if match:
            h, m, s = match.groups()
            return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        pass

    return None


def extract_video_montage(video_path, n_clips=5, clip_duration=1.0, output_suffix="_montage"):
    """Extract multiple evenly-spaced clips and concatenate into a montage.

    Args:
        video_path: Path to the source video
        n_clips: Number of clips to extract (default: 5)
        clip_duration: Duration of each clip in seconds (default: 1.0)
        output_suffix: Suffix for output filename (default: "_montage")

    Returns:
        Tuple of (Path to montage video, list of start times, error_msg), or (None, None, error_msg) if failed
    """
    try:
        import cv2
    except ImportError:
        return None, None, "OpenCV (cv2) not available"

    video_path = Path(video_path)
    if not video_path.exists():
        return None, None, f"Video path does not exist: {video_path}"

    # Open video and get properties
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, f"Could not open video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or frame_count <= 0:
        cap.release()
        return None, None, f"Invalid video properties: fps={fps}, frames={frame_count}"

    total_duration = frame_count / fps
    if total_duration < n_clips * clip_duration:
        cap.release()
        return None, None, f"Video too short: {total_duration:.1f}s < {n_clips * clip_duration:.1f}s needed"

    # Calculate evenly spaced start times
    usable_duration = total_duration - clip_duration
    start_times = [usable_duration * i / (n_clips - 1) if n_clips > 1 else 0 for i in range(n_clips)]

    frames_per_clip = int(clip_duration * fps)

    # Collect all frames first
    all_frames = []
    try:
        for start_time in start_times:
            start_frame = int(start_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for _ in range(frames_per_clip):
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB for imageio
                all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not all_frames:
        return None, None, "No frames were extracted"

    # Try imageio first (has bundled ffmpeg for H.264)
    output_path = video_path.parent / f"{video_path.stem}{output_suffix}.mp4"
    try:
        import imageio
        writer = imageio.get_writer(str(output_path), fps=fps, codec='libx264',
                                    pixelformat='yuv420p', quality=7)
        for frame in all_frames:
            writer.append_data(frame)
        writer.close()

        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path, start_times, None
    except Exception as e:
        # imageio failed, try fallback
        pass

    # Fallback: use OpenCV with XVID codec in AVI container
    output_path = video_path.parent / f"{video_path.stem}{output_suffix}.avi"
    try:
        height, width = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if out.isOpened():
            for frame in all_frames:
                # Convert back to BGR for OpenCV
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()

            if output_path.exists() and output_path.stat().st_size > 0:
                return output_path, start_times, None
    except Exception as e:
        pass

    return None, None, "Could not create video with any available codec"


def extract_video_sample(video_path, start_time=0, duration=2, output_suffix="_sample"):
    """Extract a short sample from a video using ffmpeg.

    Args:
        video_path: Path to the source video
        start_time: Start time in seconds (default: 0)
        duration: Duration in seconds (default: 2)
        output_suffix: Suffix for output filename (default: "_sample")

    Returns:
        Path to the sample video, or None if failed
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return None

    # Create output path with suffix
    output_path = video_path.parent / f"{video_path.stem}{output_suffix}.mp4"

    # Use ffmpeg to extract sample
    cmd = [
        "ffmpeg", "-y",  # Overwrite output
        "-ss", str(start_time),  # Start time
        "-i", str(video_path),  # Input
        "-t", str(duration),  # Duration
        "-c:v", "libx264",  # Video codec
        "-preset", "fast",  # Encoding speed
        "-crf", "23",  # Quality (lower = better)
        "-an",  # No audio
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and output_path.exists():
            return output_path
        else:
            return None
    except Exception:
        return None


def get_available_labeled_videos():
    """Scan temp directory for available labeled videos."""
    video_path = Path(st.session_state.get("video_path", ""))
    if not video_path or not video_path.exists():
        return []

    temp_path = video_path.parents[1] / f'tmp/dlc/{video_path.parents[0].name}'
    if not temp_path.exists():
        return []

    # Find all labeled videos (any model)
    labeled_videos = list(temp_path.glob("*_labeled.mp4"))
    # Sort by modification time, most recent first
    labeled_videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return labeled_videos


@st.fragment(run_every=60)
def render_video_preview_section():
    """Render a dropdown-based video preview section with auto-refresh every 60 seconds."""
    st.markdown("**Labeled Video Preview**")

    # Always get fresh video list (fragment handles periodic refresh)
    labeled_videos = get_available_labeled_videos()

    if not labeled_videos:
        st.info("No labeled videos in temp. Run 'Labeled video from temp' first.")
        return

    # Create display names (just filename for cleaner display)
    video_names = [v.name for v in labeled_videos]

    selected_idx = st.selectbox(
        "Select video",
        range(len(video_names)),
        format_func=lambda i: video_names[i],
        key="labeled_video_select"
    )

    if selected_idx is not None and labeled_videos:
        selected_video = labeled_videos[selected_idx]

        # Option to play full video or create montage sample
        play_mode = st.radio("Mode", ["Montage", "Full"], horizontal=True, key="play_mode")

        if play_mode == "Montage":
            # Check for existing montage (could be .mp4 or .avi)
            montage_path_mp4 = selected_video.parent / f"{selected_video.stem}_montage.mp4"
            montage_path_avi = selected_video.parent / f"{selected_video.stem}_montage.avi"
            montage_path = montage_path_mp4 if montage_path_mp4.exists() else montage_path_avi

            # Settings for montage - use compact layout
            n_clips = st.number_input("Clips", value=5, min_value=3, max_value=10, step=1, key="n_clips")
            clip_duration = st.number_input("Duration (sec)", value=1.0, min_value=0.5, max_value=2.0, step=0.5, key="clip_duration")

            if st.button("Generate montage", key="gen_montage"):
                with st.spinner(f"Creating {n_clips}-clip montage..."):
                    montage_result, start_times, err_msg = extract_video_montage(
                        selected_video,
                        n_clips=int(n_clips),
                        clip_duration=float(clip_duration)
                    )
                    if montage_result is not None:
                        montage_path = montage_result
                        st.session_state["montage_start_times"] = start_times
                        st.session_state["montage_path"] = str(montage_path)
                    else:
                        st.error(f"Failed: {err_msg}")

            # Use stored path if available
            if "montage_path" in st.session_state:
                montage_path = Path(st.session_state["montage_path"])

            if montage_path and montage_path.exists():
                st.video(str(montage_path))
                start_times = st.session_state.get("montage_start_times", None)
                if start_times:
                    times_str = ", ".join([f"{t:.0f}s" for t in start_times])
                    st.caption(f"From: {times_str}")
            else:
                st.info("Click 'Generate montage'")
        else:
            # Full video - warn about size
            file_size_mb = selected_video.stat().st_size / (1024 * 1024)
            if file_size_mb > 100:
                st.warning(f"Video is {file_size_mb:.0f} MB")
            st.video(str(selected_video))


def add_video_to_dataset():
    video_path = Path(st.session_state.get("video_path", ""))
    config_path = st.session_state.get("selected_dlc_model_config", "")

    if not video_path or not config_path:
        return "Missing video or config path."

    # get target dlc directory
    model_video_base = Path(st.session_state.dlc_videos[0]).parent
    new_name = str(video_path.parents[0].name)+'_'+str(video_path.name)
    new_video = str(model_video_base / new_name)

    # Check if video already exists in model directory
    if os.path.exists(new_video):
        st.info(f'Video already exists at {new_video}, skipping copy')
    else:
        st.write(f'Copying video to {new_video}...')
        shutil.copy(video_path, new_video)
        st.write('Copy complete')

    try:
        # add video to model path
        deeplabcut.add_new_videos(
            config=config_path,
            videos=[new_video],
            copy_videos=False
        )
        # analyze added video if model exists
        if is_dlc_model_trained(config_path):
            st.write('existing trained model found')
            # deeplabcut.analyze_videos(config_path, videos=[new_video], videotype='.avi')
            st.write(str(Path(new_video).parents[0]))
            message = queue_dlc_job(new_video, config_path, str(Path(new_video).parents[0]), 'fit,refine')
            st.success(f"DLC job queued: {message}")
            return f"Video added to project, copied, and analysis/outlier extraction job started"
        else:
            st.write('Extracting frames')
            message = queue_dlc_job(new_video, config_path, str(Path(new_video).parents[0]), 'extract')
            st.success(f"DLC job queued: {message}")
            return f"Video added to project, but no trained model was found. Train the model first."

    except Exception as e:
        return f"Failed to add video: {e}"

def run_dlc_labeling():
    video_path = Path(st.session_state.get("video_path", ""))
    config_path = st.session_state.get("selected_dlc_model_config", "")

    # get target dlc directory and video name of current video
    model_video_base = Path(st.session_state.dlc_videos[0]).parent
    new_name = str(video_path.parents[0].name)+'.avi'
    new_video = str(model_video_base / new_name)

    if not video_path or not config_path:
        return "Missing video or config path."

    try:
        # Extract frames if not already done
        # check to see if frames already exist in training path
        if is_dlc_model_trained(config_path):
            st.info("To label frames, activate env with DLC (2.2.3 used in this app), open python shell, run:")
            st.code(f"import deeplabcut\ndeeplabcut.refine_labels('{config_path}')", language='python')
        else:
            st.info("To label frames, activate env with DLC(2.2.3 used in this app), open interactive python shell, paste:")
            st.code(f"import deeplabcut\ndeeplabcut.label_frames('{config_path}')", language='python')
        return 'Happy labeling!'
    except Exception as e:
        return f"Labeling failed: {e}"

def train_dlc_model():
    video_path = Path(st.session_state.get("video_path", ""))
    config_path = st.session_state.get("selected_dlc_model_config", "")
    if not config_path:
        return "Missing DLC config path."

    try:
        # Check if transfer learning is enabled and get init_weights
        init_weights = None
        training_mode = st.session_state.get("training_mode", "Train from scratch")
        if training_mode == "Transfer learning from checkpoint":
            selected_snapshot = st.session_state.get("transfer_learning_snapshot", None)
            if selected_snapshot:
                # Remove .meta extension if present
                init_weights = selected_snapshot.replace(".meta", "")

        # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- attribute
        # the training run to a person | date: 2026-08-06]
        message = queue_dlc_job(video_path, config_path, 'training', 'training',
                                init_weights=init_weights, user=current_dlc_user())
        # [AGENT EDIT END]

        if init_weights:
            return f"Model training started with transfer learning from: {Path(init_weights).name}"
        return "Model training started from scratch."
    except Exception as e:
        return f"Training failed: {e}"

def create_dlc_model():
    # Placeholder: retrain the selected DLC model
    return "Training launched for selected model (stub)."

def label_dlc_video():
    video_path = Path(st.session_state.get("video_path", ""))
    config_path = st.session_state.get("selected_dlc_model_config", "")
    temp_path = video_path.parents[1].joinpath(f'tmp/dlc/{video_path.parents[0].name}')

    if not config_path or not video_path:
        st.error("Missing video or DLC config path.")
        return
    # Use original video path (not the copied version in the model folder)
    try:
        st.write(f"Using original video for labeled output: {video_path}")
        deeplabcut.create_labeled_video(config_path, [str(video_path)], draw_skeleton=False, destfolder=temp_path)

        # Look for labeled video in the same folder as original video
        labeled_video_candidates = list(temp_path.glob(f"{video_path.stem}*_labeled.mp4"))
        if labeled_video_candidates:
            labeled_video_path = labeled_video_candidates[0]
            st.success("Labeled video created successfully.")
            st.write(f"Labeled video found: {labeled_video_path}")
            # st.video(str(labeled_video_path))
        else:
            st.warning("Labeled video not found in expected location.")
    except Exception as e:
        st.error(f"Failed to create labeled video: {e}")

def save_notes_callback():
    config_path = st.session_state.get("selected_dlc_model_config", "")
    if not config_path:
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["notes"] = st.session_state.get("dlc_notes", "")

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    st.success("Notes saved!")

def set_snapshotindex(config_path, value):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    old_index = config.get("snapshotindex", "all")
    config["snapshotindex"] = value
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return old_index


def evaluate_all_snapshots(config_path):
    original_index = set_snapshotindex(config_path, "all")
    st.write(original_index)
    deeplabcut.evaluate_network(config_path, plotting=False)
    set_snapshotindex(config_path, original_index)  # restore

def get_eval_file(config_path):
    cfg = yaml.safe_load(open(config_path))
    proj_path = Path(cfg["project_path"])
    iteration = f"iteration-{cfg['iteration']}"

    # Find latest eval CSV
    results_file = str(proj_path.joinpath('evaluation-results').joinpath(iteration).joinpath('CombinedEvaluation-results.csv'))
    if os.path.exists(results_file):
        return results_file
    else:
        return None


def plot_snapshot_errors(results_file):

    df = pd.read_csv(results_file)
    # Group by snapshot and plot test error
    snapshot_errors = df.groupby('Training iterations:')["Test error with p-cutoff"].mean()

    fig, ax = plt.subplots(figsize=(6, 2))
    snapshot_errors.plot(marker="o", ax=ax)
    ax.set_ylabel("Mean Test Error (px)")
    ax.set_xlabel("Snapshot")
    ax.set_title("Snapshot Error Overview")
    ax.grid(True)

    return fig

# [AGENT EDIT START | agent: claude | user: wingertj | reason: head-aligned DLC marker spread QC panel | date: 2026-08-04]
from nems_lbhb.motor import dlc_headfix_qc as hfqc

_HFQC_FLAG_COLOR = {"ok": "#2e7d32", "warn": "#ef6c00", "fail": "#c62828", "": "#9e9e9e"}


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 6 review finding 1 -
# config_mtime alone does not change when training labels are added/edited (those live
# under labeled-data/*/CollectedData_*.h5, not config.yaml), so the cached reference went
# stale; add label_files_fingerprint as an explicit cache-busting argument | date: 2026-08-04]
@st.cache_data(show_spinner=False)
def _hfqc_reference(config_path, config_mtime, labels_fingerprint):
    """Cached training reference.

    config_mtime and labels_fingerprint both bust the cache; labels_fingerprint is the
    one that actually tracks the labeled-data/*/CollectedData_*.h5 files this reference
    is built from.
    """
    return hfqc.build_training_reference(config_path)


@st.cache_data(show_spinner=False)
def _hfqc_score(h5_path, h5_mtime, config_path, config_mtime, labels_fingerprint, p_cut):
    ref = _hfqc_reference(config_path, config_mtime, labels_fingerprint)
    return hfqc.score_video(h5_path, ref, p_cut=p_cut)
# [AGENT EDIT END]


def _hfqc_candidate_h5(video_path, config_path):
    """.h5 files for this session produced by the currently selected model."""
    video_path = Path(video_path)
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: final review finding J -
    # a short video_path (fewer than 2 parent levels) made parents[1] raise inside the
    # fragment, showing a traceback instead of the intended st.info | date: 2026-08-04]
    if len(video_path.parents) < 2:
        return []
    # [AGENT EDIT END]
    session = video_path.parents[0].name
    temp_dir = video_path.parents[1] / f"tmp/dlc/{session}"
    sorted_dir = video_path.parents[1] / "sorted"

    candidates = list(temp_dir.glob("*.h5"))
    candidates += [p for p in sorted_dir.glob("*.h5") if session in p.name]

    highlight_substr, highlight_tokens = get_dlc_highlight_tokens(config_path)
    matched = hfqc.match_h5_files(candidates, highlight_substr, highlight_tokens)
    return sorted(matched, key=lambda p: p.stat().st_mtime, reverse=True)


def _hfqc_figure(result):
    """Head-fixed scatter with training 2-sigma ellipses overlaid."""
    fig = go.Figure()
    parts = list(result.table.index)
    palette = plt.get_cmap("tab10")

    for i, part in enumerate(parts):
        rgb = palette(i % 10)
        color = f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})"
        pts = result.scatter_hf[part]
        fig.add_trace(go.Scattergl(
            x=pts[:, 0], y=pts[:, 1], mode="markers", name=part,
            marker=dict(size=3, color=color, opacity=0.35)))

        ref = result.reference
        if ref is not None and part in ref.hf_cov:
            ell = hfqc.ellipse_path(ref.hf_mean[part], ref.hf_cov[part], n_sd=2.0)
            if np.isfinite(ell).all():
                fig.add_trace(go.Scatter(
                    x=np.append(ell[:, 0], ell[0, 0]),
                    y=np.append(ell[:, 1], ell[0, 1]),
                    mode="lines", name=f"{part} (train 2sd)",
                    line=dict(color=color, width=2), showlegend=False))

    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="head-fixed x (head-lengths)",
        yaxis_title="head-fixed y (head-lengths)",
        legend=dict(font=dict(size=9)))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# [AGENT EDIT START | agent: claude | user: wingertj | reason: tracking-rate bars in the spread QC panel | date: 2026-08-04]
def _hfqc_bar_figure(result):
    """Spread ratios (top) and tracking rate (bottom), sharing marker rows.

    The bottom panel needs no training reference, so it renders either way; the
    top panel is omitted when there is nothing to compare against.
    """
    table = result.table
    parts = list(table.index)[::-1]          # plotly draws the first row at the bottom
    show_spread = bool(result.has_reference)
    n_rows = 2 if show_spread else 1

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False,
                        vertical_spacing=0.13,
                        subplot_titles=(["Spread vs training", "Frames tracked"]
                                        if show_spread else ["Frames tracked"]))
    row = 1
    if show_spread:
        spread = [float(table.loc[p, "ratio_spread"]) for p in parts]
        offset = [float(table.loc[p, "ratio_offset"]) for p in parts]
        fig.add_trace(go.Bar(
            y=parts, x=spread, orientation="h", name="ratio_spread (flagged)",
            marker_color=[_HFQC_FLAG_COLOR.get(table.loc[p, "flag"], "#9e9e9e")
                          for p in parts]), row=1, col=1)
        fig.add_trace(go.Bar(
            y=parts, x=offset, orientation="h", name="ratio_offset",
            marker_color="#bdbdbd"), row=1, col=1)
        for xval, color in ((hfqc.RATIO_WARN, "#ef6c00"), (hfqc.RATIO_FAIL, "#c62828")):
            fig.add_vline(x=xval, line=dict(color=color, dash="dash", width=2),
                          row=1, col=1)
        fig.update_xaxes(title_text="ratio vs training spread", row=1, col=1)
        row = 2

    tracked = [float(table.loc[p, "frac_tracked"]) for p in parts]
    fig.add_trace(go.Bar(
        y=parts, x=tracked, orientation="h", name="frames tracked",
        marker_color=[_HFQC_FLAG_COLOR.get(table.loc[p, "flag_track"], "#9e9e9e")
                      for p in parts]), row=row, col=1)
    fig.add_trace(go.Scatter(
        y=parts, x=[float(table.loc[p, "p_median"]) for p in parts],
        mode="markers", name="median likelihood",
        marker=dict(symbol="diamond", size=9, color="#212121")), row=row, col=1)
    for xval, color in ((hfqc.TRACK_WARN, "#ef6c00"), (hfqc.TRACK_FAIL, "#c62828")):
        fig.add_vline(x=xval, line=dict(color=color, dash="dash", width=2),
                      row=row, col=1)
    fig.update_xaxes(title_text="fraction of frames", range=[0, 1.02], row=row, col=1)

    fig.update_layout(height=270 * n_rows, barmode="group",
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(font=dict(size=9), orientation="h",
                                  yanchor="bottom", y=-0.22))
    return fig
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 6 review finding 3 -
# ratio_spread, ratio_offset and flag are meaningless without a training reference to
# compare against, and bias_x/bias_y (the offset components) are likewise guaranteed
# all-NaN in that path (score_video sets offset = np.full(2, np.nan) when reference is
# None); previously they rendered as all-NaN/all-empty columns that read as "computed
# but empty" rather than "not applicable", so drop all four when has_reference is False
# | date: 2026-08-04]
def _hfqc_style_table(table, has_reference=True):
    def _row_style(row):
        color = _HFQC_FLAG_COLOR.get(row.get("flag", ""), "#9e9e9e")
        if row["role"] == "nonrigid":
            return ["color: #9e9e9e"] * len(row)
        return [f"color: {color}"] * len(row)

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: final review finding A -
    # ratio split into ratio_spread (tracking noise, drives flag) and ratio_offset
    # (systematic shape difference); rms_resid renamed rms_spread. Extended in the
    # final review closeout to also drop bias_x/bias_y, which are the offset
    # components and are all-NaN in lockstep with ratio_offset when there is no
    # reference | date: 2026-08-04]
    if not has_reference:
        table = table.drop(columns=[c for c in
                                    ("ratio_spread", "ratio_offset", "bias_x", "bias_y", "flag")
                                    if c in table.columns])

    numeric = ["frac_valid", "rms_spread", "rms_spread_px", "ratio_spread",
               "ratio_offset", "bias_x", "bias_y", "hf_cx", "hf_cy", "hf_sd",
               "radius", "frac_tracked", "p_median"]
    # [AGENT EDIT END]
    numeric = [c for c in numeric if c in table.columns]
    return table.style.apply(_row_style, axis=1).format(
        {c: "{:.3f}" for c in numeric})
# [AGENT EDIT END]


@st.fragment(run_every=60)
def render_headfix_spread_section():
    """Head-aligned marker spread QC, gated on a model-matched analysis existing."""
    video_path = st.session_state.get("video_path", "")
    config_path = st.session_state.get("selected_dlc_model_config", "")
    if not video_path or not config_path or not os.path.exists(config_path):
        return

    st.markdown("**Head-Aligned Marker Spread**")

    bodyparts = st.session_state.get("dlc_bodyparts", [])
    missing = hfqc.missing_rigid_roles(bodyparts) if bodyparts else list(hfqc.RIGID_ROLES)
    if missing:
        st.info(f"Model has no {', '.join(missing)} marker(s); spread QC needs "
                f"{', '.join(hfqc.RIGID_ROLES)}.")
        return

    candidates = _hfqc_candidate_h5(video_path, config_path)
    if not candidates:
        st.info("Analyze DLC to temp with this model first.")
        return

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: final review finding H -
    # the selectbox stored a positional index into a list this 60s fragment can reorder
    # (sorted by mtime) or shrink, so the selection silently pointed at a different file;
    # key on the path string instead | date: 2026-08-04]
    options = [str(p) for p in candidates]
    if st.session_state.get("hfqc_h5_select") not in options:
        st.session_state.pop("hfqc_h5_select", None)
    choice = st.selectbox(
        "Analysis file", options,
        format_func=lambda s: Path(s).name, key="hfqc_h5_select")
    # [AGENT EDIT END]
    p_cut = st.slider("Likelihood cutoff", 0.0, 1.0,
                      float(hfqc.DEFAULT_P_CUT), 0.05, key="hfqc_p_cut")

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 6 review finding 2 -
    # this fragment reruns every 60s (run_every=60, kept so the panel can appear on its own
    # once a queued analysis lands) and st.button only returns True on the click that
    # handles it, so a plain "if not button: return" made the scatter/table/caption vanish
    # on every auto-rerun. Persist the last computed result in session_state, keyed by
    # exactly what it was computed from (h5 path + p_cut), so a stale selection never
    # renders a result computed for a different file/cutoff. | date: 2026-08-04]
    h5_path = Path(choice)
    result_key = "hfqc_result"
    request_key = (str(h5_path), float(p_cut))

    if st.button("Compute spread", key="hfqc_compute"):
        with st.spinner("Head-fixing frames..."):
            try:
                labels_fp = hfqc.label_files_fingerprint(config_path)
                result = _hfqc_score(
                    str(h5_path), h5_path.stat().st_mtime,
                    str(config_path), os.path.getmtime(config_path), labels_fp,
                    float(p_cut))
            except Exception as exc:
                st.error(f"Spread QC failed: {exc}")
                st.session_state.pop(result_key, None)
                return
            st.session_state[result_key] = (request_key, result)

    stored = st.session_state.get(result_key)
    if stored is None or stored[0] != request_key:
        return
    result = stored[1]
    # [AGENT EDIT END]

    if not result.has_reference:
        st.info("No CollectedData_*.h5 training labels in this project — showing "
                "within-video spread only (no ratio, no reference ellipses).")
    for msg in result.warnings:
        st.warning(msg)

    st.plotly_chart(_hfqc_figure(result), use_container_width=True)
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: tracking-rate bars in the spread QC panel | date: 2026-08-04]
    st.plotly_chart(_hfqc_bar_figure(result), use_container_width=True)
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: final review finding I -
    # in the head-fix frame headpost is (0,0) and front is pinned to +y by construction,
    # so their training ellipses collapse to a point/line and read as flawless tracking
    # rather than as a property of the coordinate system | date: 2026-08-04]
    st.caption("Plot is the head-fix frame: headpost is (0, 0) and front is pinned to "
               "+y **by construction**, so their ellipses collapse to a point and a "
               "line — that is the coordinate system, not tracking quality. Only "
               "leftear is an unconstrained readout here. The table's numbers come "
               "from the Procrustes fit, not from this plot.")
    # [AGENT EDIT END]
    st.dataframe(_hfqc_style_table(result.table, has_reference=result.has_reference),
                 use_container_width=True)

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: worst-of caption
    # reporting both the spread flag and the new tracking flag; a session that
    # discards most of its frames previously read green on spread alone | date: 2026-08-04]
    part, ratio = hfqc.worst_rigid(result)
    track_part, track_frac = hfqc.worst_rigid_tracking(result)
    scale_note = f"1 head-length = {result.scale_px:.1f} px"
    frames_note = (f"{result.n_valid}/{result.n_total} frames "
                   f"({100 * result.frac_valid:.0f}%) passed p>={p_cut:.2f}.")
    ref = result.reference
    if ref is not None:
        ref_note = (f" Reference: {ref.n_frames} labeled frames from "
                    f"{len(ref.sources)} labeled-data folder(s).")
    else:
        ref_note = ""

    worst = []
    if part is not None:
        worst.append(f"spread: **{part}** {ratio:.2f}x "
                     f"({hfqc.flag_for_ratio(ratio)})")
    if track_part is not None:
        worst.append(f"tracking: **{track_part}** {100 * track_frac:.0f}% "
                     f"({hfqc.flag_for_fraction(track_frac)})")
    worst_note = ("Worst rigid — " + "; ".join(worst) + ". ") if worst else ""

    offset_note = ("ratio_offset is reported separately and never flags — it measures "
                   "a systematic shape difference from the training animal, not "
                   "tracking noise. ") if part is not None else ""

    st.caption(f"{worst_note}{offset_note}{frames_note}{ref_note} {scale_note}")
    # [AGENT EDIT END]
# [AGENT EDIT END]


# Main app layout
st.title("DLC Arena Labeler")

st.subheader("DLC Model Selection")

dlc_configs = glob.glob("/auto/data/dlc/**/config.yaml", recursive=True)
dlc_model_names = [Path(p).parent.name for p in dlc_configs]
dlc_model_lookup = dict(zip(dlc_model_names, dlc_configs))

if dlc_model_names:
    # Load user preferences to get last selected model
    user_prefs = load_user_prefs()
    saved_model = user_prefs.get("dlc_model_select", None)

    # Determine default: use saved if valid, else session state, else first model
    if saved_model and saved_model in dlc_model_names:
        default_model = saved_model
    else:
        default_model = st.session_state.get("dlc_model_select", dlc_model_names[0])

    # Callback to save selection when changed
    def on_model_change():
        prefs = load_user_prefs()
        prefs["dlc_model_select"] = st.session_state.dlc_model_select
        save_user_prefs(prefs)

    selected_model = st.selectbox(
        "DLC model selection",
        options=dlc_model_names,
        index=dlc_model_names.index(default_model) if default_model in dlc_model_names else 0,
        key="dlc_model_select",
        on_change=on_model_change
    )
    with st.expander("DLC Configs", expanded=False):
        # Load config and extract fields
        selected_config_path = dlc_model_lookup[selected_model]
        st.session_state.selected_dlc_model_config = selected_config_path
        load_dlc_config_fields(selected_config_path)

        st.session_state.selected_dlc_model_config = dlc_model_lookup[selected_model]

        name = st.text_input("Model name", value=st.session_state.get("dlc_project_name", ""), key="dlc_project_name")
        experimenter = st.text_input("Experimenter", value=st.session_state.get("dlc_experimenter", ""),
                                     key="dlc_experimenter")
        working_dir = st.text_input("Working directory", value=st.session_state.get("dlc_working_dir", ""),
                                    key="dlc_working_dir")
        # Load current config
        try:
            config_path = Path(st.session_state.get('selected_dlc_model_config'))
            # Load config
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            project_path = Path(config.get("project_path", ""))
            dlc_model_path = project_path / "dlc-models"

            # --- Handle iteration ---
            iterations = sorted([p.name for p in dlc_model_path.glob("iteration-*") if p.is_dir()])
            current_iter = config.get("iteration", iterations[0] if iterations else "iteration-0")
            # Use session_state to detect changes
            if "selected_iter" not in st.session_state:
                st.session_state.selected_iter = current_iter

            selected_iter = st.selectbox("Select DLC Iteration", iterations,
                                         index=iterations.index(st.session_state.selected_iter))

            # If user selected a different iteration, update snapshots
            if selected_iter != st.session_state.selected_iter:
                st.session_state.selected_iter = selected_iter

                # Find new snapshot list
                snapshot_dir = next(iter((dlc_model_path / selected_iter).glob("*/train")), None)
                if snapshot_dir and snapshot_dir.exists():
                    snapshot_paths = list(snapshot_dir.glob("snapshot-*.meta"))
                    snapshot_indices = sorted(int(p.stem.split("-")[1]) for p in snapshot_paths)

                    # DLC stores the index into this list
                    current_snap_index = config.get("snapshotindex", len(snapshot_indices) - 1)
                    current_snap_index = min(current_snap_index, len(snapshot_indices) - 1)  # clip if out of range
                    current_snap_val = snapshot_indices[current_snap_index] if snapshot_indices else 0

                    # Show the snapshot values, but map to index on selection
                    new_snap_val = st.selectbox("Select Snapshot (training epoch)", snapshot_indices,
                                                index=current_snap_index)

                    if snapshot_indices:
                        config["iteration"] = int(selected_iter.split('-')[-1])
                        new_index = snapshot_indices.index(new_snap_val)
                        config["snapshotindex"] = new_index
                        with open(config_path, "w") as f:
                            yaml.dump(config, f)
                        st.success(f"Updated snapshot index: {new_index} (value: {new_snap_val})")
                        if hasattr(st, "rerun"):
                            st.rerun()
                else:
                    st.warning(f"No snapshots found for {selected_iter}")
            else:
                # --- Handle snapshot index dropdown ---
                snapshot_dir = next(iter((dlc_model_path / selected_iter).glob("*/train")), None)
                snapshot_indices = sorted(
                    int(p.stem.split("-")[1]) for p in snapshot_dir.glob("snapshot-*.meta")) if snapshot_dir else []
                # DLC stores the index into this list
                current_snap_index = config.get("snapshotindex", len(snapshot_indices) - 1)
                if current_snap_index == -1:
                    current_snap_index = len(snapshot_indices) - 1
                current_snap_index = min(current_snap_index, len(snapshot_indices) - 1)  # clip if out of range
                current_snap_val = snapshot_indices[current_snap_index] if snapshot_indices else 0

                # Show the snapshot values, but map to index on selection
                new_snap_val = st.selectbox("Select Snapshot (training epoch)", snapshot_indices,
                                            index=current_snap_index)

                # Save the *index* back to config
                if new_snap_val != current_snap_val:
                    new_index = snapshot_indices.index(new_snap_val)
                    config["snapshotindex"] = new_index
                    with open(config_path, "w") as f:
                        yaml.dump(config, f)
                    st.success(f"Updated snapshot index: {new_index} (value: {new_snap_val})")
                    if hasattr(st, "rerun"):
                        st.rerun()
        except Exception as e:
            st.error(f"Failed to load or update DLC config: {e}")

        notes  = st.text_area(
            "Notes",
            value=st.session_state.get("dlc_notes", ""),
            key="dlc_notes",
            height=200,
            on_change=save_notes_callback
        )
        bodypart_text = st.text_area(
            "Bodyparts",
            value="\n".join(st.session_state.dlc_bodyparts),
            key="bodyparts_edit"
        )

        # Convert back to list
        bodyparts_list = [bp.strip() for bp in bodypart_text.splitlines() if bp.strip()]

        video_list_str = st.text_area(
            "Videos",
            value="\n".join(st.session_state.get("dlc_videos", [])),
            key="dlc_video_list"
        )
else:
    st.warning("No DLC models found under /auto/data/dlc/")

st.subheader("DLC Actions")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    if st.button("Analyze DLC to temp"):
        analyze_to_temp()

    if st.button("Labeled video from temp"):
        # msg = label_dlc_video()
        msg = video_to_temp()

    video_path = st.session_state.get("video_path", "")
    if video_path:
        session_name = Path(video_path).parents[0].name
        temp_path = Path(video_path).parents[1].joinpath(f'tmp/dlc/{session_name}')
        config_path = st.session_state.get("selected_dlc_model_config", "")
        highlight_substr, highlight_tokens = get_dlc_highlight_tokens(config_path)
        render_h5_list(
            "Temp .h5 files",
            temp_path,
            highlight_substr=highlight_substr,
            highlight_tokens=highlight_tokens,
        )
    else:
        st.caption("Temp .h5 files: (set a video path to list files)")

    # Labeled Video Preview Section
    render_video_preview_section()

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: head-aligned DLC marker spread QC panel | date: 2026-08-04]
    render_headfix_spread_section()
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: camera-slot assertion in
    # the GUI. The old pipeline routed on camera NAME, so a psi slot swap (USB re-enumeration)
    # was undetectable; the correction is a human assertion, and this is where a human is
    # already looking at the frames | date: 2026-08-12]
    try:
        from nems_lbhb.motor.streamlit_vid_labeler.panels import camera as camera_panel
        camera_panel.render(st.session_state.get("video_path", ""), user=current_dlc_user())
    except Exception as _e:
        st.caption(f"camera panel unavailable: {_e}")
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: nothing in the GUI could
    # answer "which model produced this session's tracking, and is it still current?" --
    # the question the whole lifecycle project exists to make answerable | date: 2026-08-12]
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: routing suggestion from the
    # recording's own geometry, with the legacy site-based router still selectable for
    # reproducing a historical analysis | date: 2026-08-13]
    try:
        from nems_lbhb.motor.streamlit_vid_labeler.panels import routing as routing_panel
        routing_panel.render(st.session_state.get("video_path", ""))
    except Exception as _e:
        st.caption(f"routing panel unavailable: {_e}")
    # [AGENT EDIT END]

    try:
        from nems_lbhb.motor.streamlit_vid_labeler.panels import provenance as prov_panel
        prov_panel.render_session(st.session_state.get("video_path", ""))
        prov_panel.render_deployed()
    except Exception as _e:
        st.caption(f"provenance panels unavailable: {_e}")
    # [AGENT EDIT END]

with col2:
    if st.button("Analyze DLC to sorted"):
        video_path = st.session_state.get("video_path", "")
        config_path = st.session_state.get("selected_dlc_model_config", "")

        if not video_path or not os.path.exists(video_path):
            st.error("Invalid or missing video path.")
        elif not config_path or not os.path.exists(config_path):
            st.error("Invalid or missing DLC config path.")
        else:
            message = queue_dlc_job(video_path, config_path, extra_note='analyze_to_sorted')
            st.success(f"DLC job queued: {message}")

    video_path = st.session_state.get("video_path", "")
    if video_path:
        session_name = Path(video_path).parents[0].name
        sorted_path = Path(video_path).parents[1] / "sorted"
        config_path = st.session_state.get("selected_dlc_model_config", "")
        highlight_substr, highlight_tokens = get_dlc_highlight_tokens(config_path)
        render_h5_list(
            "Sorted .h5 files",
            sorted_path,
            match_substr=session_name,
            highlight_substr=highlight_substr,
            highlight_tokens=highlight_tokens,
        )
    else:
        st.caption("Sorted .h5 files: (set a video path to list files)")

with col3:
    if st.button("Add Current Video Frames to Model Dataset"):
        msg = add_video_to_dataset()
        st.info(msg)

with col4:
    if st.button("DLC Label Frames"):
        msg = run_dlc_labeling()

with col5:
    config_path = st.session_state.selected_dlc_model_config
    if st.button("Evaluate all snapshots"):
        results_file = get_eval_file(config_path)
        if results_file is not None:
            st.write('found eval file')
            fig = plot_snapshot_errors(results_file)
            st.pyplot(fig)
        else:
            st.write('evaluating model')
            evaluate_all_snapshots(config_path)
            results_file = get_eval_file(config_path)
            fig = plot_snapshot_errors(results_file)
            st.pyplot(fig)

    training_type = st.radio(
        "Training mode:",
        options=["Train from scratch", "Transfer learning from checkpoint"],
        key="training_mode"
    )

    config_path = st.session_state.selected_dlc_model_config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    project_path = Path(cfg["project_path"])
    snapshot_paths = list(project_path.rglob("snapshot-*.meta"))

    if training_type == "Transfer learning from checkpoint":
        snapshot_options = [str(s) for s in snapshot_paths]
        selected_snapshot = st.selectbox(
            "Select snapshot to initialize weights",
            options=snapshot_options,
            key="transfer_learning_snapshot"
        )

        if selected_snapshot:
            st.info(f"Will use: {Path(selected_snapshot).name} for transfer learning")

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- who is running
    # this training job; goes on the tQueue row, the project lock and TRAINRUN.toml
    # | date: 2026-08-06]
    run_user = st.text_input("Your name (for training job attribution)",
                             value=current_dlc_user(), key="dlc_run_user")
    if run_user and run_user != load_user_prefs().get("dlc_run_user"):
        prefs = load_user_prefs()
        prefs["dlc_run_user"] = run_user
        save_user_prefs(prefs)
    # [AGENT EDIT END]

    if st.button("Train DLC Model with New Data"):
        msg = train_dlc_model()
        st.info(msg)

with col6:
    if st.button("Create New DLC Project"):
        videos = [v.strip() for v in st.session_state.dlc_video_list.split(",")]
        try:
            config_path = deeplabcut.create_new_project(
                st.session_state.dlc_project_name,
                st.session_state.dlc_experimenter,
                videos,
                working_directory=st.session_state.dlc_working_dir,
                copy_videos=True
            )
            st.success(f"New DLC project created at: {config_path}")
        except Exception as e:
            st.error(f"Failed to create DLC project: {e}")


# # Only proceed if the user has entered a path
# if st.session_state.get('video_path', None) is not None:
#     if not os.path.exists(st.session_state.get('video_path', None) ):
#         st.error("File not found")
#         st.session_state.cap = None
#     else:
#         # Open a new capture if needed
#         if st.session_state.get("cap") is None:
#             st.session_state.cap = cv2.VideoCapture(st.session_state.get('video_path', None) )
#
#         cap = st.session_state.get("cap", None)
#         if cap.isOpened():
#             st.write("Video opened successfully!")
#             # → Your frame‐reading / marker‐drawing code here
#         else:
#             st.error("Could not open video file")
#             st.session_state.cap = None

with st.sidebar:
    st.markdown("### DLC Marker Overlay")
    show_dlc_markers = st.radio("Show DLC markers?", ["No", "Yes"], index=0)

    selected_dlc_file = None
    video_path = Path(st.session_state.get("video_path", ""))
    config_path = st.session_state.get("selected_dlc_model_config", "")
    dlc_h5_dir = video_path.parents[1].joinpath(f'tmp/dlc/{video_path.parents[0].name}')
    st.session_state.dlc_h5_dir = dlc_h5_dir
    available_h5s = list(dlc_h5_dir.glob("*.h5"))

    if show_dlc_markers == "Yes":
        if not available_h5s:
            st.warning("No DLC .h5 files found.")
        else:
            selected_dlc_file = st.selectbox(
                "Select a DLC .h5 file:",
                options=[str(p.name) for p in available_h5s]
            )

    st.header("Annotation Points")

    with st.form("add_point"):
        new_point_name = st.text_input("New point name")
        if st.form_submit_button("Add Point") and new_point_name:
            if st.session_state.get('current_frame', None) is not None:
                # Initialize at center of image
                img = st.session_state.get('current_frame', None)
                st.session_state.points[new_point_name] = (img.width // 2, img.height // 2)
                get_cmap_color()
            else:
                st.error("Load a frame first!")

    # Point removal
    if st.session_state.points:
        remove_point = st.selectbox(
            "Select point to remove",
            list(st.session_state.points.keys()),
            key="point_remove_select"  # Unique key
        )
        if st.button("Remove Point"):
            del st.session_state.points[remove_point]
            get_cmap_color()

    st.header("Speaker Setup")

    # Add a new speaker by name
    with st.form("add_speaker_form"):
        new_speaker_name = st.text_input("New speaker name")
        if st.form_submit_button("Add Speaker") and new_speaker_name:
            if new_speaker_name in st.session_state.speakers:
                st.warning("Speaker already exists.")
            else:
                st.session_state.speakers[new_speaker_name] = {"distance": 1.0, "angle": 0.0}


    st.header("Import Configuration")

    uploaded_config = st.file_uploader("Upload config (.toml)", type=["toml"])

    if uploaded_config is not None:
        file_id = f"{uploaded_config.name}:{uploaded_config.size}"
        if st.session_state.get("last_imported_config") != file_id:
            if st.button("Import selected config"):
                import_config(uploaded_config)
                st.session_state.last_imported_config = file_id
        else:
            st.info("Config already imported. Upload a different file to re-import.")


# if st.session_state.get("cap") is not None:
#     total_frames = int(st.session_state.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_number = st.number_input(
#         "Frame Number",
#         min_value=0,
#         max_value=total_frames - 1,
#         value=0,
#         key="frame_number_input"  # Unique key
#     )
#
#     if st.button("Load Frame"):
#         st.session_state.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#         ret, frame = st.session_state.cap.read()
#         if ret:
#             # Convert to proper 3-channel RGB
#             if frame.ndim == 2:  # Grayscale
#                 frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#             elif frame.shape[2] == 4:  # BGRA
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
#             else:  # BGR
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             # Convert to PIL Image
#             st.session_state.current_frame = Image.fromarray(frame)
st.header("Video Loader")

# Text input (NOT connected to session state)
video_path_input = st.text_input(
    "Enter video file path:",
    value=st.session_state.get("video_path", ""),
    key="video_path_input_field"
)

# Load button
if st.button("Update video path"):
    if video_path_input.strip() and os.path.exists(video_path_input.strip()):
        st.session_state.video_path = video_path_input.strip()
        st.success("Video path updated")
        video_key = str(Path(st.session_state.video_path))
        if st.session_state.get("auto_config_loaded_for") != video_key:
            load_sorted_config_for_session(st.session_state.video_path)
    else:
        st.error("Invalid or missing video path.")

video_path = st.session_state.get("video_path", None)

if video_path and os.path.exists(video_path):
    video_key = str(Path(video_path))
    if st.session_state.get("auto_config_loaded_for") != video_key:
        load_sorted_config_for_session(video_path)
    cap_temp = cv2.VideoCapture(video_path)
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()

    frame_number = st.number_input(
        "Frame Number",
        min_value=0,
        max_value=total_frames - 1,
        value=0,
        key="frame_number_input"
    )

    if st.button("Load Frame"):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if ret:
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            st.session_state.current_frame = Image.fromarray(frame)
            st.success(f"Loaded frame {frame_number}")
        else:
            st.error("Failed to read frame.")
else:
    st.error("Invalid or missing video path.")

# Helper function to convert image to base64
def img_to_base64(image):
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# Modify the plot display section
if st.session_state.get('current_frame', None) is not None:
    img = st.session_state.get('current_frame', None)
    if st.session_state.points:
        if st.session_state.selected_point not in st.session_state.points:
            st.session_state.selected_point = next(iter(st.session_state.points))
    selected_name = st.session_state.selected_point
    plotly_key_version = st.session_state.get("plotly_key_version", 0)
    click_key = f"plotly_click_{selected_name or 'none'}_{plotly_key_version}"
    click_data = st.session_state.get(click_key)
    if isinstance(click_data, str):
        try:
            click_data = json.loads(click_data)
        except Exception:
            click_data = None
    if click_data and selected_name in st.session_state.points:
        event = click_data[0]
        curve_num = event.get("curveNumber", None)
        # Only accept clicks on the image trace (trace 0).
        if curve_num is None or curve_num == 0:
            x = int(event['x'])
            y = int(event['y'])
        else:
            x = y = None
        old_x, old_y = st.session_state.points[selected_name]
        if x is not None and y is not None and (x, y) != (old_x, old_y):
            st.session_state.points[selected_name] = (x, y)

    fixation_point = compute_fixation_point_from_state()
    if fixation_point is not None:
        st.session_state.fixation_point = fixation_point
    def build_plotly_figure():
        fig = create_plotly_figure(img)
        if st.session_state.fixation_point is not None:
            fx, fy = st.session_state.fixation_point
            fig.add_trace(go.Scatter(
                   x=[fx], y=[fy],
                   mode="markers",
                   marker=dict(color="red", size=15, symbol="x"),
                   name="Fixation Point",
                   showlegend=False))

        if show_dlc_markers == "Yes" and selected_dlc_file:
            try:
                h5_path = dlc_h5_dir / selected_dlc_file
                df = pd.read_hdf(h5_path)

                # Assume single frame for now (first row of MultiIndex)
                frame = st.session_state.get('frame_number_input', 0)
                scorer = df.columns.get_level_values(0)[0]
                bodyparts = df[scorer].columns.get_level_values(0)
                names_bodyparts = list(bodyparts.unique(level=0))
                keypoints = df.iloc[frame]  # shape: (bodypart, x/y/p) or flat if single animal
                for i, bp in enumerate(names_bodyparts):
                    x = keypoints[scorer][bp]['x']
                    y = keypoints[scorer][bp]['y']
                    fig.add_trace(go.Scatter(
                        x=[x],
                        y=[y],
                        mode='markers+text',
                        marker=dict(color='cyan', size=10),
                        text=[bp],
                        name=f"DLC: {bp}",
                        showlegend=False
                    ))

            except Exception as e:
                st.error(f"Failed to load DLC predictions: {e}")

        return fig

    fig = build_plotly_figure()
    fig_with_config = _PlotlyFigureWithConfig(
        fig,
        {"displayModeBar": False, "displaylogo": False}
    )
    _ = plotly_events(
        fig_with_config,
        click_event=True,
        key=click_key,
        override_height=img.height + 60  # Add room for controls/axes
    )

# Point selection radio buttons
st.header("Select Point to Move")
if st.session_state.points:
    selected_point = st.radio(
        "Active point:",
        options=list(st.session_state.points.keys()),
        index=0,
        key="selected_point"
    )
else:
    st.info("No points added yet. Add points using the sidebar.")

# Add this section after your marker management code
st.header("Marker Pair Distances")

if st.session_state.points:
    # Generate sorted unique pairs
    marker_names = sorted(st.session_state.points.keys())
    current_pairs = [f"{a}_{b}" for a, b in combinations(marker_names, 2)]

    # Clean up old pairs
    existing_pairs = list(st.session_state.distances.keys())
    for pair in existing_pairs:
        if pair not in current_pairs:
            del st.session_state.distances[pair]

    # Add new pairs with default value 0.0
    for pair in current_pairs:
        if pair not in st.session_state.distances:
            st.session_state.distances[pair] = 0.0

    # Create input fields
    for pair in sorted(st.session_state.distances.keys()):
        st.session_state.distances[pair] = st.number_input(
            f"Distance for {pair} (meters)",
            min_value=0.0,
            value=st.session_state.distances[pair],
            step=0.01,
            key=f"dist_{pair}"
        )
else:
    st.info("Add at least 2 markers to enable distance inputs")


st.subheader("Pixels per Meter for Marker Pairs")

if st.session_state.distances:
    # dict to accumulate our results
    ppm_dict = {}

    for pair_name, dist_m in st.session_state.distances.items():
        m1, m2 = pair_name.split("_")
        # dist_m = info["distance_m"]

        # get pixel coords
        px1, py1 = st.session_state.points[m1]
        px2, py2 = st.session_state.points[m2]

        # compute pixel distance
        pixel_dist = math.hypot(px2 - px1, py2 - py1)

        # avoid zero‑division
        if dist_m and dist_m > 0:
            ppm = pixel_dist / dist_m
            ppm_dict[pair_name] = ppm
            st.write(f"**{pair_name}**: {pixel_dist:.1f} px / {dist_m:.2f} m → {ppm:.1f} px/m")
        else:
            st.error(f"Invalid distance for {pair_name}")

    avg = np.average(list(ppm_dict.values()))
    st.session_state.ppm_avg = avg
    st.success(f"**average: {avg:.2f} px/m**")
    # store back in session_state so you can use it elsewhere
    st.session_state.pixels_per_meter = ppm_dict

else:
    st.info('Add at least 2 markers to enable distance inputs')

st.header("Fixation Point")

# 1) Make sure we actually have some markers defined:
if st.session_state.points:
    marker_keys = list(st.session_state.points.keys())

    # 2) Let the user pick one (default to the first key)
    default_fixation_key = st.session_state.get("fixation_point_select", marker_keys[0])
    fix_pt_key = st.selectbox(
        "Choose fixation point",
        options=marker_keys,
        key="fixation_point_select",
        index=marker_keys.index(default_fixation_key) if default_fixation_key in marker_keys else 0
    )

    # 3) Ask for shift amounts
    col1, col2 = st.columns(2)
    with col1:
        shift_x = st.text_input(
            "Shift X (m)",
            value=st.session_state.get("fixation_shift_x", "0.0"),
            key="fixation_shift_x"
        )
    with col2:
        shift_y = st.text_input(
            "Shift Y (m)",
            value=st.session_state.get("fixation_shift_y", "0.0"),
            key="fixation_shift_y"
        )

    # 4) Safely compute and store (apply shift in rotated basis if x-axis is defined)
    try:
        fixation_point = compute_fixation_point_from_state()
        if fixation_point is None:
            raise ValueError
        fx, fy = fixation_point
        st.session_state.fixation_point = fixation_point
        st.success(f"Shifted fixation at: x={fx:.2f}px, y={fy:.2f}px")
    except ValueError:
        st.error("Please enter valid numbers for Shift X and Shift Y")
else:
    st.warning("No markers defined yet.  Add at least one point above before setting a fixation.")

st.header("New X-axis: 2 points")

# 1) Make sure we actually have some markers defined:
if st.session_state.points:
    marker_keys = list(st.session_state.points.keys())

    # 2) Let the user pick one (default to the first key)
    x1_pt_key = st.selectbox(
        "Choose x1 point (left end of axis)",
        options=marker_keys,
        index=0,
        key="xaxis1_point_select"
    )
    # 2) Let the user pick one (default to the first key)
    x2_pt_key = st.selectbox(
        "Choose x2 point (right end of axis)",
        options=marker_keys,
        index=0,
        key="xaxis2_point_select"
    )

    st.session_state.xaxis_points=(x1_pt_key, x2_pt_key)

st.subheader("Speaker Orientation")

orientation = st.selectbox(
    "Speakers are positioned relative to the new X-axis at the:",
    options=["Top", "Right", "Bottom", "Left"],
    index=["Top", "Right", "Bottom", "Left"].index(st.session_state.get("speaker_orientation", "Top")),
    key="speaker_orientation"
)


st.header("Speaker Configuration")

if st.session_state.speakers:
    remove_speaker = st.selectbox("Remove speaker", list(st.session_state.speakers.keys()))
    if st.button("Delete Selected Speaker"):
        del st.session_state.speakers[remove_speaker]

    for name in st.session_state.speakers:
        st.markdown(f"**{name}**")
        col1, col2 = st.columns(2)
        with col1:
            dist = st.number_input(
                f"{name} distance (m)",
                min_value=0.0,
                value=st.session_state.speakers[name]["distance"],
                step=0.1,
                key=f"{name}_dist"
            )
        with col2:
            ang = st.number_input(
                f"{name} angle (°)",
                min_value=-180.0,
                max_value=180.0,
                value=st.session_state.speakers[name]["angle"],
                step=1.0,
                key=f"{name}_angle"
            )
        # Store updated values back
        st.session_state.speakers[name]["distance"] = dist
        st.session_state.speakers[name]["angle"] = ang
else:
    st.info("No speakers defined yet.")

if st.session_state.get('video_path', None) is not None:
    # Save configuration (modified to use video_path)
    st.header("Save Configuration")
    video_path = st.session_state.get('video_path', None)
    vp = Path(video_path)
    out_path = vp.parents[1].joinpath('sorted')
    os.makedirs(out_path, exist_ok=True)
    dlc_arena_config=out_path.joinpath(f"{vp.stem}-{vp.parent.stem}-dlc_arena_config.toml")
    config_name = st.text_input(
        "Configuration name", value=dlc_arena_config,
        key="config_name_input"  # Unique key
    )
    if st.button("Save Configuration") and video_path:
        save_config(config_name, video_path)
else:
    st.info('Need to input a video path first')
