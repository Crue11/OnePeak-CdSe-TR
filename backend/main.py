"""CdSe OnePeak prediction API.

Position-only. Simulated peak intensity was never calibrated to a real
diffractometer's count scale (simulated ~6,375 vs. real fitted peaks
~150-230 counts), so intensity is dropped from both directions --
predictions rely solely on peak_pos_002, a physical angle unaffected by
instrument calibration.

Forward: the champion ensemble (RandomForest + GradientBoosting + MLP,
mean of the three) from "A Single-Reflection Machine Learning Approach
for Non-Contact Temperature Prediction of CdSe Thin Films", retrained on
the single feature, gated by a 1D z-score out-of-distribution check.

Reverse: two independent position estimates from temperature -- a
single-output RandomForest, and a closed-form physics model (thermal
expansion of the c lattice parameter + Bragg's law for the (002)
reflection) -- shown side by side.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

DATA_PATH = Path(__file__).parent / "data" / "cdse_simulated_dataset.csv"
CACHE_PATH = Path(__file__).parent / "model_cache.joblib"

# Recalibrated 99th percentile of training z-scores for the 1D case.
OOD_THRESHOLD = 2.29

# Resolution of the shared position -> temperature curve.
CURVE_RESOLUTION = 60

# Closed-form physics inverse: thermal expansion of the c lattice parameter,
# Bragg's law for the (002) reflection.
C0 = 7.010  # Angstrom, room-temperature c lattice parameter
ALPHA_C = 2.8e-6  # /K, thermal expansion coefficient
T0 = 25  # C, reference temperature
WAVELENGTH = 1.5418  # Angstrom, Cu-K-alpha (weighted Ka1/Ka2 average)


def physics_inverse(temperature_c: float) -> float:
    c = C0 * (1 + ALPHA_C * (temperature_c - T0))
    d = c / 2  # (002) reflection: d-spacing simplifies to c/2
    two_theta_rad = 2 * np.arcsin(WAVELENGTH / (2 * d))
    return float(np.degrees(two_theta_rad))


def build_ensemble(X: np.ndarray, y: np.ndarray) -> tuple:
    rf = RandomForestRegressor(
        n_estimators=400, max_features=0.5, criterion="absolute_error", random_state=42, n_jobs=1
    )
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, loss="absolute_error", random_state=42)
    mlp = make_pipeline(RobustScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=2000, random_state=42))
    rf.fit(X, y)
    gb.fit(X, y)
    mlp.fit(X, y)
    return (rf, gb, mlp)


def ensemble_predict(models: tuple, X: np.ndarray) -> np.ndarray:
    return np.column_stack([m.predict(X) for m in models]).mean(axis=1)


class ModelState:
    forward_models: tuple
    ml_inverse: RandomForestRegressor
    mu: float
    std: float
    t_min: float
    t_max: float
    test_mae: float
    test_r2: float
    curve: dict


state = ModelState()


def train_all() -> None:
    print("[startup] training started", flush=True)
    df = pd.read_csv(DATA_PATH)
    train = df[df["split"].isin(["train_clean", "train_aug"])]
    test = df[df["split"] == "test_heldout"]

    X_train = train[["peak_pos_002"]].to_numpy()
    y_train = train["T"].to_numpy()

    state.forward_models = build_ensemble(X_train, y_train)

    state.mu = float(X_train.mean())
    state.std = float(X_train.std())

    X_test = test[["peak_pos_002"]].to_numpy()
    y_test = test["T"].to_numpy()
    y_pred = ensemble_predict(state.forward_models, X_test)
    state.test_mae = float(mean_absolute_error(y_test, y_pred))
    state.test_r2 = float(r2_score(y_test, y_pred))

    state.ml_inverse = RandomForestRegressor(n_estimators=400, max_features=0.5, random_state=42, n_jobs=1)
    state.ml_inverse.fit(train[["T"]], train["peak_pos_002"])
    # Padded to the intended 25-400C experimental design range: the simulated
    # data's actual extremes (~25.13-399.85) are noise-shifted off the nominal
    # bounds, which would otherwise reject exactly 25 or 400 as OOD.
    state.t_min = min(25.0, float(y_train.min()))
    state.t_max = max(400.0, float(y_train.max()))

    pos_min, pos_max = float(X_train.min()), float(X_train.max())
    positions = np.linspace(pos_min, pos_max, CURVE_RESOLUTION)
    temperatures = ensemble_predict(state.forward_models, positions.reshape(-1, 1))
    state.curve = {
        "pos_min": pos_min,
        "pos_max": pos_max,
        "positions": positions.tolist(),
        "temperatures": temperatures.tolist(),
    }

    print(f"[startup] trained on {len(train)} rows. held-out MAE={state.test_mae:.2f} R2={state.test_r2:.3f}", flush=True)


def ood_distance(x: float) -> float:
    return abs((x - state.mu) / state.std)


CACHE_FIELDS = [
    "forward_models",
    "ml_inverse",
    "mu",
    "std",
    "t_min",
    "t_max",
    "test_mae",
    "test_r2",
    "curve",
]


def save_cache() -> None:
    joblib.dump({field: getattr(state, field) for field in CACHE_FIELDS}, CACHE_PATH, compress=3)


def load_cache() -> bool:
    if not CACHE_PATH.exists():
        return False
    data = joblib.load(CACHE_PATH)
    for field in CACHE_FIELDS:
        setattr(state, field, data[field])
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # criterion="absolute_error" makes the forward RandomForest fit slow
    # (median-based splits have no incremental update, unlike mean-based
    # squared_error); load a pre-trained cache instead of refitting on
    # every cold start, which was blowing past Render's port-bind timeout.
    if not load_cache():
        train_all()
        save_cache()
    else:
        print(f"[startup] loaded cached model. held-out MAE={state.test_mae:.2f} R2={state.test_r2:.3f}", flush=True)
    yield


app = FastAPI(title="CdSe OnePeak API", lifespan=lifespan)

allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGIN", "http://localhost:8443").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TemperatureRequest(BaseModel):
    peak_position: float


class TemperatureResponse(BaseModel):
    ood: bool
    temperature: Optional[float] = None
    distance: Optional[float] = None


class SignatureRequest(BaseModel):
    temperature: float


class SignatureResponse(BaseModel):
    ood: bool
    physics_position: Optional[float] = None
    model_position: Optional[float] = None


@app.get("/api/health")
def health():
    return {"status": "ok", "test_mae": state.test_mae, "test_r2": state.test_r2}


@app.get("/api/curve")
def curve():
    return state.curve


@app.post("/api/predict/temperature", response_model=TemperatureResponse)
def predict_temperature(req: TemperatureRequest) -> TemperatureResponse:
    distance = ood_distance(req.peak_position)
    if distance > OOD_THRESHOLD:
        return TemperatureResponse(ood=True, distance=distance)
    prediction = ensemble_predict(state.forward_models, np.array([[req.peak_position]]))[0]
    return TemperatureResponse(ood=False, temperature=float(prediction), distance=distance)


@app.post("/api/predict/signature", response_model=SignatureResponse)
def predict_signature(req: SignatureRequest) -> SignatureResponse:
    t = req.temperature
    if t < state.t_min or t > state.t_max:
        return SignatureResponse(ood=True)
    model_position = float(state.ml_inverse.predict([[t]])[0])
    return SignatureResponse(ood=False, physics_position=physics_inverse(t), model_position=model_position)
