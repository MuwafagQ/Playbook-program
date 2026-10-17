from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ROBOFLOW_API_KEY: str = Field(..., description="Roboflow API key for hosted inference")
    HF_TOKEN: str | None = Field(default=None, description="Hugging Face token (optional)")

    PLAYER_MODEL_ID: str = "football-players-detection-3zvbc/11"
    FIELD_MODEL_ID: str = "football-field-detection-f07vi-it2xv/10"
    # Our open RF-DETR people model (checkpoint path). When set it replaces PLAYER_MODEL_ID.
    PLAYER_MODEL_PATH: str = ""
    # Our open RF-DETR pitch keypoint model (checkpoint path). When set it replaces FIELD_MODEL_ID.
    FIELD_MODEL_PATH: str = ""
    # Give FIELD_MODEL_PATH plain frames, not PREPROCESS_ENABLED-enhanced ones (it was trained on plain
    # frames; enhanced: median error 35 px vs 15 px on the test frames).
    FIELD_MODEL_PLAIN_FRAMES: bool = True
    # "ours" (trained by notebooks/train_field_keypoints.ipynb) or "roboflow_v10" (weights.pt downloaded from
    # Roboflow, Apache-2.0: its own point order and contrast stretching, see vision/field_model.py).
    FIELD_MODEL_KIND: str = "ours"
    # Robust homography (RANSAC, pitch cm): ignores keypoints that disagree with the rest. 0 = plain fit.
    H_RANSAC_CM: float = 0.0
    # Speed-up for our RF-DETR models (people + ball): none | fp16 | tensorrt (vision/fast_rfdetr.py).
    MODEL_ACCEL: str = "none"
    # The ball model's own setting ("" = MODEL_ACCEL). TensorRT lost ~5% of ball detections on the
    # half-1 window (data/speed/settings.json), so the pilot runs the ball model in fp16.
    BALL_MODEL_ACCEL: str = ""

    # Runtime
    DEVICE: str = "cpu"
    FAST_MODE: bool = True
    MAX_FRAMES: int = 0  # 0 = all frames
    DETECT_EVERY_N: int = 2
    HOMOGRAPHY_EVERY_N: int = 2
    TEAM_UPDATE_EVERY_N: int = 3
    PREPROCESS_ENABLED: bool = False
    TEAM_MODE: str = "color"  # color | embedding
    SIDE_BY_SIDE_VIEW: bool = True
    LEFT_VIEW_RATIO: float = 0.62

    # 2D radar (top-down pitch) rendering.
    # Uses the roboflow/sports pitch annotators. The radar is only refreshed
    # every RADAR_EVERY_N frames (and held in between) to remove the per-frame
    # jitter caused by detection/homography noise.
    RADAR_EVERY_N: int = 5
    RADAR_SCALE: float = 0.1
    RADAR_PADDING: int = 50

    # Tracker tuning (applied when supported by installed supervision version)
    TRACKER_TYPE: str = "bytetrack"  # bytetrack | botsort
    TRACK_ACTIVATION_THRESHOLD: float = 0.25
    TRACK_LOW_THRESHOLD: float = 0.10
    TRACK_MIN_MATCHING_THRESHOLD: float = 0.80
    TRACK_LOST_BUFFER: int = 90
    TRACK_MIN_CONSEC_FRAMES: int = 2
    TRACK_STALE_FRAMES: int = 2
    # Track all people with one tracker and give each track the role it was labelled with most
    # often (vision/roles.py), instead of separate player / official trackers.
    ROLE_BY_TRACK: bool = False
    BOTSORT_GMC_METHOD: str = "sparseOptFlow"
    BOTSORT_WITH_REID: bool = False
    BOTSORT_REID_MODEL: str = "yolo11n-cls.pt"
    BOTSORT_PROXIMITY_THRESH: float = 0.5
    BOTSORT_APPEARANCE_THRESH: float = 0.55
    ID_RELINK_FRAMES: int = 120
    ID_MEMORY_FRAMES: int = 90
    ID_RELINK_PX: float = 95.0
    ID_APP_WEIGHT: float = 0.80
    ID_AMBIGUITY_MARGIN: float = 0.08
    ID_MAX_RELINK_COST: float = 1.45
    ID_UNKNOWN_APP_PENALTY: float = 0.60
    ID_UNKNOWN_APP_MAX_GAP: int = 12
    ID_UNKNOWN_APP_MAX_SPACE_FRAC: float = 0.85
    ID_GALLERY_SIZE: int = 36
    ID_GALLERY_MIN_ADD_DIST: float = 0.10
    ID_FRAME_CONTINUITY_IOU: float = 0.45
    ID_FRAME_CONTINUITY_MARGIN: float = 0.03
    ID_SPATIAL_RESCUE_MAX_GAP: int = 220
    ID_SPATIAL_RESCUE_MARGIN: float = 0.15
    ID_SPATIAL_RESCUE_RATIO: float = 0.70
    ID_MAX_PLAYER_IDS: int = 22
    ID_MAX_GOALKEEPER_IDS: int = 2
    ID_MAX_REFEREE_IDS: int = 2
    ID_HARD_REUSE_WHEN_CAPPED: bool = True
    ID_LOCK_FRAMES: int = 12
    ID_CAPPED_REUSE_MIN_GAP: int = 8

    # Color-team classifier
    TEAM_COLOR_INIT_SAMPLES: int = 30
    TEAM_COLOR_LR: float = 0.05
    TEAM_COLOR_MIN_MARGIN: float = 0.08
    # Re-cluster the two kit colours every N player samples from the last TEAM_COLOR_BUFFER (0 = off:
    # the first centroids drift with TEAM_COLOR_LR instead).
    TEAM_COLOR_REFIT_EVERY: int = 200
    TEAM_COLOR_BUFFER: int = 3000
    # Judge each track by the mean of all its colour features, re-assigned with the current centroids
    # (pair with TEAM_VOTE_HISTORY=1, TEAM_VOTE_MIN=1, TEAM_LOCK=false: the assignment itself is stable).
    TEAM_COLOR_TRACK_MEAN: bool = False
    # Per-track team = majority of the last TEAM_VOTE_HISTORY votes (needs TEAM_VOTE_MIN votes);
    # TEAM_LOCK freezes a track's team once 8 votes agree at 70%. A ~1.5 s window follows an ID that
    # passes to another player; a lock or a whole-life vote keeps the wrong team (data/team/).
    TEAM_VOTE_HISTORY: int = 45
    TEAM_VOTE_MIN: int = 8
    TEAM_LOCK: bool = False
    # Use each box's single-frame team guess to veto ID re-links across teams. A noisy guess splits
    # IDs (night clip: 9 -> 175 player ID switches with team on), so off unless proven useful.
    TEAM_IN_ID_RELINK: bool = False
    # Goalkeepers (kit matches neither team): team whose players' centre is nearer, majority of the
    # last TEAM_GK_VOTE_HISTORY frames (~5 s), needing TEAM_GK_VOTE_MIN votes.
    TEAM_GOALKEEPER: bool = True
    TEAM_GK_VOTE_HISTORY: int = 150
    TEAM_GK_VOTE_MIN: int = 45

    DET_CONF: float = 0.30
    DET_CONF_PLAYER: float = 0.24
    DET_CONF_REFEREE: float = 0.24
    DET_CONF_GOALKEEPER: float = 0.22
    DET_CONF_BALL: float = 0.10
    DETECT_UPSCALE: float = 1.35
    # FIELD_CONF = object (pitch) detection threshold passed to .infer(); must stay
    # well below the pitch detection's confidence floor (~0.55) or whole frames drop.
    # KP_CONF = per-keypoint confidence filter applied before findHomography.
    # 0.30 / 0.50 match the original notebook recipe.
    FIELD_CONF: float = 0.30
    KP_CONF: float = 0.50
    BALL_PAD_PX: int = 10
    BALL_MAX_MISSING: int = 10
    BALL_MAX_INTERP_FRAMES: int = 8
    # Hard ceiling on accepted ball displacement from the predicted position (px).
    # Raised from 140: the velocity-aware gate (below) keeps slow-ball matching
    # tight, while this only blocks genuinely wild jumps. A struck ball can move
    # several hundred px/frame in a broadcast clip.
    BALL_MAX_JUMP_PX: float = 600.0
    BALL_MIN_CONF: float = 0.12
    # Velocity-aware acceptance gate: radius = BALL_GATE_BASE_PX + BALL_GATE_VEL_K*speed,
    # capped at BALL_MAX_JUMP_PX. Base covers detection jitter on a slow ball.
    BALL_GATE_BASE_PX: float = 90.0
    BALL_GATE_VEL_K: float = 3.0
    # Wider cold-start gate, used until a real velocity has been measured.
    BALL_ACQUIRE_GATE_PX: float = 300.0
    # Reject a ball candidate whose box area exceeds this multiple of the running
    # ball-size estimate (filters boots/limbs misread as the ball). 0 = off.
    BALL_SIZE_MAX_RATIO: float = 5.0
    # EMA weight of the newest velocity measurement; per-frame decay while bridging.
    BALL_VEL_ALPHA: float = 0.5
    BALL_HOLD_DECAY: float = 0.85
    # On-pitch boundary gate: drop ball detections whose projected position falls
    # outside the pitch rectangle (defined by the corner keypoints) expanded by
    # these margins (cm). Catches balls detected behind the goal / in the stands.
    # 0 = off. Pitch is ~12000x7000 cm.
    BALL_ON_PITCH_MARGIN_X: float = 500.0
    BALL_ON_PITCH_MARGIN_Y: float = 500.0
    # ROI re-detection: when the full-frame pass finds no ball but a track is
    # active, re-run detection on an upscaled crop around the predicted position
    # (the tiny ball becomes several times larger in the zoomed crop). Costs one
    # extra inference call on miss frames only. The recovered detection still
    # passes the smoother's distance/size/confidence gates.
    BALL_ROI_RECOVERY: bool = True
    BALL_ROI_PX: int = 320
    BALL_ROI_UPSCALE: float = 2.0
    BALL_ROI_CONF: float = 0.10

    # Dedicated ball model (vision/ball_model.py, trained by notebooks/train_ball_tiles.ipynb).
    # Replaces the detector's ball candidates. It searches BALL_MODEL_ROI_TILES^2 tiles
    # around the predicted ball position, and the whole frame when the ball is lost (for
    # BALL_MODEL_LOST_FRAMES frames), unknown, or every BALL_MODEL_FULL_EVERY_N frames.
    # In replay the candidates come from the cache (ballm.csv.gz); no model is loaded.
    BALL_MODEL_ENABLED: bool = False
    BALL_MODEL_PATH: str = ""
    BALL_MODEL_TILE: int = 320
    BALL_MODEL_CONF: float = 0.10        # model threshold (candidates recorded to the cache)
    BALL_MODEL_MIN_CONF: float = 0.30    # candidates the pipeline uses (0.30 beat 0.15 on HILAL-HAZM)
    BALL_MODEL_ROI_TILES: int = 2
    BALL_MODEL_LOST_FRAMES: int = 3
    BALL_MODEL_FULL_EVERY_N: int = 30
    # With the ball model: a clearly more confident ball outside the tracking gate for
    # this many frames takes over the ball track (vision/ball.py, challenger switch). 0 = off.
    BALL_SWITCH_FRAMES: int = 3

    # Tiny box filtering (ratio relative to frame area)
    MIN_AREA_RATIO_PEOPLE: float = 0.00008
    MIN_AREA_RATIO_BALL: float = 0.00001

    # Homography estimation
    H_EMA_ALPHA: float = 0.50
    RANSAC_REPROJ_THRESH: float = 25.0
    MIN_KP: int = 6
    MIN_INLIER_RATIO: float = 0.40
    MAX_REPROJ_ERR: float = 80.0
    # Absolute inlier-count acceptance (alongside the ratio bar). 0 = ratio only.
    H_MIN_INLIERS_ABS: int = 5
    H_REINIT_FRAMES: int = 30
    # Inlier-ratio hysteresis: once locked, maintain at (MIN_INLIER_RATIO - this).
    H_INLIER_HYSTERESIS: float = 0.08
    # Frame-to-frame discontinuity gate (cm of median projected jump). 0 = off.
    H_MAX_JUMP_M: float = 2000.0
    # Minimum keypoint spread (px, weaker PCA axis) to attempt a fit. 0 = off.
    H_MIN_KP_SPREAD_PX: float = 12.0
    # Bounded short hold: frames to keep the last good H when a degeneracy guard fires.
    H_MAX_HOLD_FRAMES: int = 15
    # Optical-flow homography propagation: when a fresh keypoint H can't be solved
    # (degenerate midfield geometry), propagate the last good H using camera motion
    # from background features. Propagation accumulates drift, so the cap bounds how
    # long it is allowed to compound; the state machine then HOLDS the frozen H for up
    # to H_REINIT_FRAMES more frames before blanking.
    H_OPTFLOW_BRIDGE: bool = True
    H_MAX_PROPAGATION_FRAMES: int = 60

    # ---- Firebase Auth ----
    # Backend: verifies ID tokens via firebase-admin. Point this at a service
    # account JSON downloaded from Firebase console -> Project settings ->
    # Service accounts -> Generate new private key.
    FIREBASE_SERVICE_ACCOUNT_JSON: str | None = Field(
        default=None, description="Path to Firebase service-account JSON (backend token verification)"
    )
    FIREBASE_PROJECT_ID: str | None = None

    # Frontend (Streamlit): the Firebase Web App config object, from Firebase
    # console -> Project settings -> General -> Your apps -> Web app. These
    # are not secret (they identify the project to the client SDK; access is
    # still governed by Firebase Auth + your security rules).
    FIREBASE_API_KEY: str | None = None
    FIREBASE_AUTH_DOMAIN: str | None = None
    FIREBASE_APP_ID: str | None = None


def load_settings() -> Settings:
    s = Settings()
    if s.HF_TOKEN:
        import os
        os.environ["HF_TOKEN"] = s.HF_TOKEN
        os.environ["HUGGINGFACE_HUB_TOKEN"] = s.HF_TOKEN
    return s
