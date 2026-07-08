# -*- coding: utf-8 -*-
"""
ONE WAY PICKZ — WNBA Prop Engine
Full single-file Streamlit app.

Markets: PTS / REB / AST / PRA
Line-source build:
- Keeps Underdog as the only live automated prop source.
- Removes/turns off Sleeper, Odds API, and SportsGameOdds from the active board flow.
- Adds an in-app Manual Line Entry board by slate and market, using cached WNBA schedules
  so you can enter PTS/REB/AST/PRA lines directly against the correct matchup.
- Projection engine still runs from the SportsDataverse database even when lines are manual.
"""

import os
import re
import io
import json
import math
import time
import zipfile
import base64
import hashlib
import difflib
import unicodedata
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

APP_VERSION = "NO_MONEYLINE — WNBA v2.1 — Official Online Fallback + No Logos"

# ============================================================
# Storage
# ============================================================
LOCAL_DIR = Path("wnba_engine")
LOCAL_DIR.mkdir(exist_ok=True)
DATA_DIR = LOCAL_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
BACKUP_DIR = LOCAL_DIR / "backups"
BACKUP_DIR.mkdir(exist_ok=True)

OFFICIAL_LOG = LOCAL_DIR / "wnba_official_pick_log.json"
RESULT_LOG = LOCAL_DIR / "wnba_result_log.json"
LEARNING_LOG = LOCAL_DIR / "wnba_learning_log.json"
LINE_HISTORY_FILE = LOCAL_DIR / "wnba_line_history.json"
MANUAL_LINES_FILE = LOCAL_DIR / "wnba_manual_lines.json"
INJURY_BUMPS_FILE = LOCAL_DIR / "wnba_injury_usage_bumps.json"
NO_LINE_FILE = LOCAL_DIR / "wnba_no_line_tracking.json"

CACHE_FILES = {
    "player_game_logs": DATA_DIR / "wnba_player_game_logs.csv",
    "player_season_stats": DATA_DIR / "wnba_player_season_stats.csv",
    "team_season_stats": DATA_DIR / "wnba_team_season_stats.csv",
    "team_ranks": DATA_DIR / "wnba_team_ranks.csv",
    "schedules": DATA_DIR / "wnba_schedules.csv",
    "rosters": DATA_DIR / "wnba_rosters.csv",
    "game_rosters": DATA_DIR / "wnba_game_rosters.csv",
    "lineups": DATA_DIR / "wnba_lineups.csv",
    "shots": DATA_DIR / "wnba_shots.csv",
    "master_features": DATA_DIR / "wnba_master_features.csv",
    "projection_board": DATA_DIR / "wnba_projection_board.csv",
}

# Persistent board snapshots. These let the app reopen to the last saved board
# without pulling Underdog or rebuilding projections first.
SAVED_BOARD_FILE = DATA_DIR / "wnba_saved_board_snapshot.csv"
SAVED_LINES_FILE = DATA_DIR / "wnba_saved_lines_snapshot.csv"
SAVED_BOARD_META_FILE = LOCAL_DIR / "wnba_saved_board_meta.json"


# Optional GitHub/raw cache support.
# If you commit CSV caches into GitHub, the app will read them first from the
# local repo path above. If Streamlit runtime cache is empty and the CSV is not
# committed locally, set one of these secrets to a raw GitHub folder URL:
#   WNBA_DATA_BASE_URL = "https://raw.githubusercontent.com/<user>/<repo>/main/wnba_engine/data"
#   GITHUB_DATA_BASE_URL = "https://raw.githubusercontent.com/<user>/<repo>/main/wnba_engine/data"
# The file names must match CACHE_FILES values, e.g. wnba_master_features.csv.
GITHUB_CACHE_DATA_KEYS = list(CACHE_FILES.keys())
GITHUB_DATA_BASE_SECRET_NAMES = ["WNBA_DATA_BASE_URL", "GITHUB_DATA_BASE_URL"]


def _read_secret_or_env(name: str, default: str = "") -> str:
    """Read a Streamlit secret or environment variable without breaking local runs."""
    try:
        val = st.secrets.get(name, "")
        if val not in [None, ""]:
            return str(val).strip()
    except Exception:
        pass
    return str(os.environ.get(name, default) or "").strip()


def _github_data_base_url() -> str:
    """Raw GitHub data folder URL. Empty means only local repo/runtime cache is used."""
    for nm in GITHUB_DATA_BASE_SECRET_NAMES:
        val = _read_secret_or_env(nm, "")
        if val:
            return val.rstrip("/")
    return ""


def _github_cache_url_for_key(dataset_key: str) -> str:
    base = _github_data_base_url()
    path = CACHE_FILES.get(dataset_key)
    if not base or path is None:
        return ""
    return f"{base}/{Path(path).name}"


def _read_csv_bytes(raw: bytes, dataset_key: str = "") -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    if dataset_key in ["player_game_logs", "schedules", "game_rosters", "lineups", "shots", "master_features", "projection_board"]:
        for c in ["GameDate"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


@st.cache_data(ttl=900, show_spinner=False)
def fetch_github_cache_dataset(dataset_key: str) -> Tuple[pd.DataFrame, str]:
    """Load a cached CSV from a raw GitHub URL if configured.

    Returns (df, status). This never raises and never overwrites good local data
    with an empty frame.
    """
    url = _github_cache_url_for_key(dataset_key)
    if not url:
        return pd.DataFrame(), "github cache not configured"
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
        if r.status_code >= 400:
            return pd.DataFrame(), f"github {dataset_key}: HTTP {r.status_code}"
        df = _read_csv_bytes(r.content, dataset_key)
        if df is None or df.empty:
            return pd.DataFrame(), f"github {dataset_key}: empty"
        return df, f"github {dataset_key}: loaded {len(df)} rows"
    except Exception as e:
        return pd.DataFrame(), f"github {dataset_key}: {str(e)[:140]}"


def github_cache_status_table() -> pd.DataFrame:
    """Small Data Manager report for GitHub-backed caches."""
    rows = []
    base = _github_data_base_url()
    for key, path in CACHE_FILES.items():
        url = _github_cache_url_for_key(key)
        local_exists = Path(path).exists()
        rows.append({
            "Dataset": key,
            "Local/Repo CSV": "✅" if local_exists else "❌",
            "GitHub Raw URL": url if base else "not configured",
            "Expected File": Path(path).name,
        })
    return pd.DataFrame(rows)

# ============================================================
# Team logo assets
# ============================================================
ASSETS_DIR = Path("assets")
LOGO_DIR = ASSETS_DIR / "logos"
TEAM_LOGO_ALIASES = {
    "ATL": "ATL", "ATLANTA": "ATL", "ATLANTA DREAM": "ATL",
    "CHI": "CHI", "CHICAGO": "CHI", "CHICAGO SKY": "CHI",
    "CON": "CON", "CONN": "CON", "CONNECTICUT": "CON", "CONNECTICUT SUN": "CON",
    "DAL": "DAL", "DALLAS": "DAL", "DALLAS WINGS": "DAL",
    "GSV": "GSV", "GSW": "GSV", "GOLDEN STATE": "GSV", "GOLDEN STATE VALKYRIES": "GSV",
    "IND": "IND", "INDIANA": "IND", "INDIANA FEVER": "IND",
    "LVA": "LVA", "LV": "LVA", "LAS VEGAS": "LVA", "LAS VEGAS ACES": "LVA",
    "LAS": "LAS", "LA": "LAS", "LOS ANGELES": "LAS", "LOS ANGELES SPARKS": "LAS",
    "MIN": "MIN", "MINNESOTA": "MIN", "MINNESOTA LYNX": "MIN",
    "NYL": "NYL", "NY": "NYL", "NEW YORK": "NYL", "NEW YORK LIBERTY": "NYL",
    "PHX": "PHX", "PHO": "PHX", "PHOENIX": "PHX", "PHOENIX MERCURY": "PHX",
    "SEA": "SEA", "SEATTLE": "SEA", "SEATTLE STORM": "SEA",
    "WAS": "WAS", "WSH": "WAS", "WASHINGTON": "WAS", "WASHINGTON MYSTICS": "WAS",
}

def team_abbr_for_logo(team: Any) -> str:
    t = str(team or "").strip().upper()
    t = re.sub(r"[^A-Z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return "WNBA"
    return TEAM_LOGO_ALIASES.get(t, TEAM_LOGO_ALIASES.get(t[:3], t[:3]))

@st.cache_data(show_spinner=False)
def local_logo_data_uri(abbr: str) -> str:
    abbr = team_abbr_for_logo(abbr)
    for ext, mime in [("png", "image/png"), ("jpg", "image/jpeg"), ("jpeg", "image/jpeg"), ("webp", "image/webp"), ("svg", "image/svg+xml")]:
        path = LOGO_DIR / f"{abbr}.{ext}"
        if path.exists():
            raw = path.read_bytes()
            return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
    return ""

def github_logo_url(abbr: str) -> str:
    """Optional remote fallback. Add WNBA_LOGO_BASE_URL in Streamlit secrets, e.g.
    WNBA_LOGO_BASE_URL='https://raw.githubusercontent.com/<user>/<repo>/main/assets/logos'
    """
    abbr = team_abbr_for_logo(abbr)
    try:
        base = st.secrets.get("WNBA_LOGO_BASE_URL", "")
    except Exception:
        base = ""
    base = str(base or "").strip().rstrip("/")
    if base:
        return f"{base}/{abbr}.png"
    return ""

def get_team_logo_src(team: Any) -> str:
    abbr = team_abbr_for_logo(team)
    src = local_logo_data_uri(abbr)
    if src:
        return src
    gh = github_logo_url(abbr)
    if gh:
        return gh
    return ""

# ============================================================
# Constants
# ============================================================
MARKETS = ["PTS", "REB", "AST", "PRA"]
DATASET_LABELS = {
    "player_game_logs": "Player Game Logs",
    "player_season_stats": "Player Season Stats",
    "team_season_stats": "Team Season Stats",
    "schedules": "Schedules",
    "rosters": "Rosters",
    "game_rosters": "Game Rosters",
    "lineups": "Lineups",
    "shots": "Shots",
}

SPORTSDATAVERSE_BASE = "https://raw.githubusercontent.com/sportsdataverse/wehoop-wnba-stats-data/main"
SPORTSDATAVERSE_INDEXES = {
    "player_game_logs": f"{SPORTSDATAVERSE_BASE}/wnba_stats/wnba_stats_player_game_logs_in_data_repo.csv",
    "player_season_stats": f"{SPORTSDATAVERSE_BASE}/wnba_stats/wnba_stats_player_season_stats_in_data_repo.csv",
    "team_season_stats": f"{SPORTSDATAVERSE_BASE}/wnba_stats/wnba_stats_team_season_stats_in_data_repo.csv",
    "schedules": f"{SPORTSDATAVERSE_BASE}/wnba_stats/wnba_stats_schedules_in_data_repo.csv",
    "rosters": f"{SPORTSDATAVERSE_BASE}/wnba_stats/wnba_stats_rosters_in_data_repo.csv",
    "game_rosters": f"{SPORTSDATAVERSE_BASE}/wnba_stats/wnba_stats_game_rosters_in_data_repo.csv",
    "lineups": f"{SPORTSDATAVERSE_BASE}/wnba_stats/wnba_stats_lineups_in_data_repo.csv",
    "shots": f"{SPORTSDATAVERSE_BASE}/wnba_stats/wnba_stats_shots_in_data_repo.csv",
}

# Direct parquet locations used by the SportsDataverse repo. The root index CSVs
# are only manifests/metadata in many cases, so v2 tries these actual data files first.
SPORTSDATAVERSE_DIRECT_PATTERNS = {
    "player_game_logs": ["wnba_stats/player_game_logs/parquet/player_game_logs_{season}.parquet"],
    "player_season_stats": ["wnba_stats/player_season_stats/parquet/player_season_stats_{season}.parquet"],
    "team_season_stats": ["wnba_stats/team_season_stats/parquet/team_season_stats_{season}.parquet"],
    "schedules": ["wnba_stats/schedules/parquet/wnba_stats_schedule_{season}.parquet", "wnba_stats/schedules/parquet/schedules_{season}.parquet"],
    "rosters": ["wnba_stats/rosters/parquet/rosters_{season}.parquet"],
    "game_rosters": ["wnba_stats/game_rosters/parquet/game_rosters_{season}.parquet"],
    "lineups": ["wnba_stats/lineups/parquet/lineups_{season}.parquet"],
    "shots": ["wnba_stats/shots/parquet/shots_{season}.parquet"],
}

UNDERDOG_URLS = [
    "https://api.underdogfantasy.com/beta/v6/over_under_lines",
    "https://api.underdogfantasy.com/beta/v5/over_under_lines",
    "https://api.underdogfantasy.com/beta/v4/over_under_lines",
    "https://api.underdogfantasy.com/beta/v3/over_under_lines",
    "https://api.underdogfantasy.com/beta/v2/over_under_lines",
    "https://api.underdogfantasy.com/v1/over_under_lines",
]
SLEEPER_URLS = [
    "https://api.sleeper.com/projections/wnba",
    "https://api.sleeper.app/projections/wnba",
    "https://api.sleeper.app/v1/projections/wnba",
    "https://api.sleeper.com/v1/projections/wnba",
]

# Optional fallback for real sportsbook player-prop lines.
# Add ODDS_API_KEY in Streamlit Secrets or paste it in the sidebar.
ODDS_API_SPORT = "basketball_wnba"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_MARKETS = {
    "PTS": "player_points",
    "REB": "player_rebounds",
    "AST": "player_assists",
    "PRA": "player_points_rebounds_assists",
}
DEFAULT_ODDS_API_BOOKMAKERS = "draftkings,fanduel,betmgm,caesars,espnbet"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.wnba.com/",
    "Origin": "https://www.wnba.com",
}

# ============================================================
# General utilities
# ============================================================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_board_snapshot(board_df: pd.DataFrame, lines_df: Optional[pd.DataFrame] = None, mode: str = "Today") -> int:
    """Persist the current projection board + line snapshot so it reloads after app restart.

    This mirrors the MLB flow: pull lines once, inspect the board, click Save Board,
    then reopening the app can show the saved slate immediately without a fresh API pull.
    """
    try:
        if board_df is None or board_df.empty:
            return 0
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = board_df.copy()
        out["SavedBoardAt"] = now_iso()
        out["SavedBoardMode"] = mode
        out.to_csv(SAVED_BOARD_FILE, index=False)
        # Keep the normal projection cache synced too.
        try:
            out.to_csv(CACHE_FILES["projection_board"], index=False)
        except Exception:
            pass
        if lines_df is not None and not lines_df.empty:
            ldf = lines_df.copy()
            ldf["SavedBoardAt"] = out["SavedBoardAt"].iloc[0]
            ldf["SavedBoardMode"] = mode
            ldf.to_csv(SAVED_LINES_FILE, index=False)
        meta = {
            "SavedAt": out["SavedBoardAt"].iloc[0],
            "Mode": mode,
            "Rows": int(len(out)),
            "Players": int(out["Player"].nunique()) if "Player" in out.columns else int(len(out)),
            "Markets": sorted(out["Market"].astype(str).str.upper().dropna().unique().tolist()) if "Market" in out.columns else [],
        }
        save_json(SAVED_BOARD_META_FILE, meta)
        return int(len(out))
    except Exception:
        return 0


def load_board_snapshot(mode: str = "Today") -> pd.DataFrame:
    """Load the saved board snapshot. This should not call any external APIs."""
    try:
        if not SAVED_BOARD_FILE.exists():
            return pd.DataFrame()
        df = pd.read_csv(SAVED_BOARD_FILE)
        if df is None or df.empty:
            return pd.DataFrame()
        if mode != "All Lines" and "Slate" in df.columns:
            dff = df[df["Slate"].astype(str).str.lower().eq(str(mode).lower())].copy()
            # Important: never fall back to an All Lines / other-day snapshot when
            # the user is viewing Today or Tomorrow. This prevents old/future
            # props from leaking into the wrong slate after app restart.
            return dff if not dff.empty else pd.DataFrame()
        if mode != "All Lines" and "SlateDate" in df.columns:
            target = slate_target_date(mode) if "slate_target_date" in globals() else None
            if target is not None:
                dff = df[df["SlateDate"].astype(str).eq(str(target))].copy()
                return dff if not dff.empty else pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def load_saved_lines_snapshot() -> pd.DataFrame:
    try:
        if SAVED_LINES_FILE.exists():
            return pd.read_csv(SAVED_LINES_FILE)
    except Exception:
        pass
    return pd.DataFrame()


def saved_board_meta() -> Dict[str, Any]:
    return load_json(SAVED_BOARD_META_FILE, {})


def safe_float(x, default=np.nan):
    try:
        if x is None or x == "":
            return default
        if isinstance(x, str):
            x = x.replace("−", "-").replace(",", "").replace("%", "").strip()
        return float(x)
    except Exception:
        return default


def normalize_name(s) -> str:
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9 ]+", " ", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    suffixes = {"jr", "sr", "ii", "iii", "iv"}
    toks = [t for t in s.split() if t not in suffixes]
    return " ".join(toks)


def name_score(a, b) -> float:
    a, b = normalize_name(a), normalize_name(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.94
    at, bt = a.split(), b.split()
    if len(at) >= 2 and len(bt) >= 2 and at[-1] == bt[-1] and at[0][0] == bt[0][0]:
        return 0.90
    return difflib.SequenceMatcher(None, a, b).ratio()


def stable_seed(*parts) -> int:
    raw = "|".join(str(p) for p in parts)
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)


def request_json(url, params=None, timeout=20):
    try:
        r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
        if r.status_code >= 400:
            return None
        return r.json()
    except Exception:
        return None


def request_json_with_status(url, params=None, timeout=20):
    """Return (json, status_code, short_message) for debugging sportsbook pulls."""
    try:
        r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
        status = int(getattr(r, "status_code", 0) or 0)
        if status >= 400:
            return None, status, (r.text or "")[:300]
        return r.json(), status, "ok"
    except Exception as e:
        return None, 0, str(e)[:300]


def get_streamlit_secret(name: str, default: str = "") -> str:
    try:
        val = st.secrets.get(name, default)
        return str(val or "").strip()
    except Exception:
        return str(default or "").strip()


def request_bytes(url, timeout=45) -> Optional[bytes]:
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        if r.status_code >= 400:
            return None
        return r.content
    except Exception:
        return None


def flatten_json(obj):
    out = []
    def walk(x, parent=""):
        if isinstance(x, dict):
            y = dict(x)
            if parent:
                y["_parent_key"] = parent
            out.append(y)
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for i in x:
                walk(i, parent)
    walk(obj)
    return out


def attrs(obj):
    if not isinstance(obj, dict):
        return {}
    out = {}
    if isinstance(obj.get("attributes"), dict):
        out.update(obj["attributes"])
    for k, v in obj.items():
        if k not in ("attributes", "relationships", "included", "data") and k not in out:
            out[k] = v
    return out


def col_norm(c) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def find_col(df: pd.DataFrame, candidates: List[str], contains_all: Optional[List[str]] = None, contains_any: Optional[List[str]] = None) -> Optional[str]:
    if df is None or df.empty:
        return None
    norm_map = {col_norm(c): c for c in df.columns}
    for cand in candidates:
        n = col_norm(cand)
        if n in norm_map:
            return norm_map[n]
    for c in df.columns:
        cn = col_norm(c)
        if contains_all and all(x in cn for x in contains_all):
            return c
        if contains_any and any(x in cn for x in contains_any):
            return c
    return None


def parse_date_series(s):
    if s is None:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace(":", ".", regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ============================================================
# File reading / SportsDataverse import
# ============================================================
def classify_filename(name: str) -> Optional[str]:
    n = name.lower()
    if "player_game_logs" in n or "player_gamelogs" in n or "game_log" in n:
        return "player_game_logs"
    if "player_season" in n or "athlete_season" in n:
        return "player_season_stats"
    if "team_season" in n:
        return "team_season_stats"
    if "game_rosters" in n or "game_roster" in n:
        return "game_rosters"
    if "rosters" in n or "roster" in n:
        return "rosters"
    if "schedule" in n:
        return "schedules"
    if "lineup" in n:
        return "lineups"
    if "shot" in n:
        return "shots"
    return None


def read_any_file(raw: bytes, name: str) -> pd.DataFrame:
    """Read uploaded CSV/XLSX/Parquet/JSON. RDS is intentionally skipped.

    SportsDataverse publishes both .parquet and .rds. Python/Streamlit should use
    parquet. Returning an empty frame for RDS prevents one unsupported file from
    breaking the whole batch import.
    """
    lname = str(name or "").lower()
    if lname.endswith(".rds"):
        return pd.DataFrame()
    if lname.endswith(".csv"):
        # SportsDataverse manifest CSVs sometimes have odd newlines; pandas still handles them.
        return pd.read_csv(io.BytesIO(raw), low_memory=False)
    if lname.endswith(".xlsx") or lname.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))
    if lname.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))
    if lname.endswith(".json"):
        return pd.read_json(io.BytesIO(raw))
    raise ValueError(f"Unsupported file type: {name}")


def is_manifest_only(df: pd.DataFrame) -> bool:
    """True for SportsDataverse index/manifest files that do not contain actual stats rows."""
    if df is None or df.empty:
        return False
    cols = {col_norm(c) for c in df.columns}
    stat_markers = {
        "player_name", "athlete_display_name", "athlete_name", "team_abbreviation", "points", "pts",
        "rebounds", "reb", "assists", "ast", "minutes", "min", "game_date", "game_id"
    }
    if cols.intersection(stat_markers):
        return False
    manifest_markers = {"season", "row_count", "generated_at_utc", "source_endpoint"}
    return len(cols.intersection(manifest_markers)) >= 2


def direct_sportsdataverse_urls(dataset_key: str, seasons: List[int]) -> List[str]:
    urls = []
    patterns = SPORTSDATAVERSE_DIRECT_PATTERNS.get(dataset_key, [])
    for season in seasons or []:
        for pat in patterns:
            urls.append(f"{SPORTSDATAVERSE_BASE}/{pat.format(season=int(season))}")
    return list(dict.fromkeys(urls))


def fetch_direct_sportsdataverse(dataset_key: str, seasons: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download actual parquet files directly from SportsDataverse repo paths."""
    frames, debug = [], []
    for u in direct_sportsdataverse_urls(dataset_key, seasons):
        try:
            b = request_bytes(u, timeout=90)
            if not b:
                debug.append({"dataset": dataset_key, "url": u, "status": "direct file not found/blocked", "rows": 0})
                continue
            name = u.split("/")[-1]
            part = read_any_file(b, name)
            if part is None or part.empty:
                debug.append({"dataset": dataset_key, "url": u, "status": "direct downloaded but empty", "rows": 0})
                continue
            part["_source_file"] = name
            frames.append(part)
            debug.append({"dataset": dataset_key, "url": u, "status": "direct ok", "rows": len(part)})
        except Exception as e:
            debug.append({"dataset": dataset_key, "url": u, "status": f"direct error: {str(e)[:160]}", "rows": 0})
    if frames:
        return pd.concat(frames, ignore_index=True, sort=False), pd.DataFrame(debug)
    return pd.DataFrame(), pd.DataFrame(debug)

def get_url_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        cn = col_norm(c)
        if "url" in cn or cn in ["href", "file", "path", "filename", "file_name", "data_file"]:
            vals = df[c].dropna().astype(str).head(20).tolist()
            if any("http" in v or ".csv" in v or ".parquet" in v or ".rds" in v for v in vals):
                cols.append(c)
    return cols


def as_raw_github_url(path_or_url: str) -> Optional[str]:
    s = str(path_or_url or "").strip()
    if not s:
        return None
    if s.startswith("http://") or s.startswith("https://"):
        if "github.com" in s and "/blob/" in s:
            s = s.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
        return s
    s = s.lstrip("/")
    if s.startswith("wnba_stats/") or s.startswith("data/") or s.endswith((".csv", ".parquet")):
        return f"{SPORTSDATAVERSE_BASE}/{s}"
    return None


def filter_index_by_season(df: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    if df.empty or not seasons:
        return df
    season_col = find_col(df, ["season", "year", "game_season"])
    if not season_col:
        return df
    d = df.copy()
    d[season_col] = pd.to_numeric(d[season_col], errors="coerce")
    return d[d[season_col].isin(seasons)].copy()


def expand_sportsdataverse_index(df: pd.DataFrame, dataset_key: str, seasons: List[int], max_files: int = 250) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Expand a SportsDataverse manifest into real rows.

    v1 assumed the manifest contained URLs. Some wehoop manifests only contain
    season/row counts, so v2 first tries direct parquet paths and only then
    falls back to URL columns inside the manifest.
    """
    debug_rows = []
    direct, direct_dbg = fetch_direct_sportsdataverse(dataset_key, seasons)
    if not direct_dbg.empty:
        debug_rows.extend(direct_dbg.to_dict("records"))
    if not direct.empty:
        return direct, pd.DataFrame(debug_rows)

    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(debug_rows + [{"dataset": dataset_key, "status": "empty manifest/upload", "rows": 0}])
    url_cols = get_url_columns(df)
    if not url_cols:
        status = "manifest only; no URL columns and direct files failed" if is_manifest_only(df) else "no URL columns; using uploaded file as raw data"
        return (pd.DataFrame() if is_manifest_only(df) else df), pd.DataFrame(debug_rows + [{"dataset": dataset_key, "status": status, "rows": 0 if is_manifest_only(df) else len(df)}])

    index = filter_index_by_season(df, seasons)
    url_candidates = []
    for _, r in index.iterrows():
        for c in url_cols:
            u = as_raw_github_url(r.get(c))
            if u and (u.endswith(".csv") or u.endswith(".parquet")):
                url_candidates.append(u)
                break
    url_candidates = list(dict.fromkeys(url_candidates))[:max_files]
    if not url_candidates:
        return pd.DataFrame(), pd.DataFrame(debug_rows + [{"dataset": dataset_key, "status": "index detected but no usable CSV/parquet URLs", "rows": 0}])

    frames = []
    for u in url_candidates:
        try:
            b = request_bytes(u, timeout=60)
            if not b:
                debug_rows.append({"dataset": dataset_key, "url": u, "status": "download failed", "rows": 0})
                continue
            name = u.split("/")[-1]
            part = read_any_file(b, name)
            if not part.empty:
                part["_source_file"] = name
                frames.append(part)
                debug_rows.append({"dataset": dataset_key, "url": u, "status": "ok", "rows": len(part)})
        except Exception as e:
            debug_rows.append({"dataset": dataset_key, "url": u, "status": f"error: {str(e)[:120]}", "rows": 0})
    if frames:
        return pd.concat(frames, ignore_index=True, sort=False), pd.DataFrame(debug_rows)
    return pd.DataFrame(), pd.DataFrame(debug_rows)

def download_sportsdataverse_dataset(dataset_key: str, seasons: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remote pull: actual parquet first, manifest/index fallback second."""
    direct, direct_dbg = fetch_direct_sportsdataverse(dataset_key, seasons)
    if not direct.empty:
        return direct, direct_dbg

    index_url = SPORTSDATAVERSE_INDEXES.get(dataset_key)
    if not index_url:
        return pd.DataFrame(), pd.DataFrame([{"dataset": dataset_key, "status": "no index url"}])
    b = request_bytes(index_url, timeout=30)
    if not b:
        return pd.DataFrame(), pd.concat([direct_dbg, pd.DataFrame([{"dataset": dataset_key, "status": "index download failed", "url": index_url}])], ignore_index=True)
    idx = pd.read_csv(io.BytesIO(b), low_memory=False)
    expanded, dbg = expand_sportsdataverse_index(idx, dataset_key, seasons=seasons)
    full_dbg = pd.concat([direct_dbg, dbg], ignore_index=True) if not direct_dbg.empty or not dbg.empty else pd.DataFrame()
    if expanded.empty and not idx.empty:
        return pd.DataFrame(), pd.concat([full_dbg, pd.DataFrame([{"dataset": dataset_key, "status": "index read but no usable stat rows expanded", "rows": len(idx)}])], ignore_index=True)
    return expanded, full_dbg

# ============================================================
# Standardizers
# ============================================================
def standardize_player_logs(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    player_col = find_col(d, ["PLAYER_NAME", "player_name", "athlete_display_name", "athlete_name", "display_name", "name", "player", "athlete"])
    team_col = find_col(d, ["TEAM_ABBREVIATION", "team_abbreviation", "team_short_display_name", "team", "team_name", "team_display_name"])
    opp_col = find_col(d, ["OPPONENT", "opponent", "opponent_team_abbreviation", "opponent_team", "opp", "MATCHUP", "matchup"])
    date_col = find_col(d, ["GAME_DATE", "game_date", "date", "game_date_time", "start_date"])
    season_col = find_col(d, ["SEASON", "season", "year"])
    game_id_col = find_col(d, ["GAME_ID", "game_id", "event_id", "competition_id"])
    home_away_col = find_col(d, ["home_away", "homeAway", "home_away_flag", "location"])
    starter_col = find_col(d, ["starter", "is_starter", "started", "starter_flag"])

    cols = {
        "Player": player_col,
        "Team": team_col,
        "Opponent": opp_col,
        "GameDate": date_col,
        "Season": season_col,
        "GameID": game_id_col,
        "HomeAway": home_away_col,
        "Starter": starter_col,
        "MIN": find_col(d, ["MIN", "minutes", "min", "athlete_minutes", "display_minutes"]),
        "PTS": find_col(d, ["PTS", "points", "athlete_points"]),
        "REB": find_col(d, ["REB", "rebounds", "total_rebounds", "athlete_rebounds"]),
        "AST": find_col(d, ["AST", "assists", "athlete_assists"]),
        "FGA": find_col(d, ["FGA", "field_goals_attempted", "field_goal_attempts"]),
        "FGM": find_col(d, ["FGM", "field_goals_made", "field_goals"]),
        "FG3A": find_col(d, ["FG3A", "three_point_field_goals_attempted", "three_point_attempts", "3PA"]),
        "FG3M": find_col(d, ["FG3M", "three_point_field_goals_made", "three_pointers_made", "3PM"]),
        "FTA": find_col(d, ["FTA", "free_throws_attempted", "free_throw_attempts"]),
        "FTM": find_col(d, ["FTM", "free_throws_made"]),
        "TOV": find_col(d, ["TOV", "turnovers", "TO"]),
        "OREB": find_col(d, ["OREB", "offensive_rebounds", "off_rebounds"]),
        "DREB": find_col(d, ["DREB", "defensive_rebounds", "def_rebounds"]),
        "STL": find_col(d, ["STL", "steals"]),
        "BLK": find_col(d, ["BLK", "blocks"]),
        "PLUS_MINUS": find_col(d, ["PLUS_MINUS", "plus_minus", "+/-"]),
    }
    out = pd.DataFrame()
    for k, c in cols.items():
        out[k] = d[c] if c else np.nan

    if out["Player"].isna().all():
        return pd.DataFrame()
    out["Player"] = out["Player"].astype(str)
    out["Team"] = out["Team"].fillna("").astype(str)
    out["Opponent"] = out["Opponent"].fillna("").astype(str)
    out["GameDate"] = parse_date_series(out["GameDate"])
    out["Season"] = pd.to_numeric(out["Season"], errors="coerce")
    if out["Season"].isna().all() and out["GameDate"].notna().any():
        out["Season"] = out["GameDate"].dt.year

    numeric_cols = ["MIN", "PTS", "REB", "AST", "FGA", "FGM", "FG3A", "FG3M", "FTA", "FTM", "TOV", "OREB", "DREB", "STL", "BLK", "PLUS_MINUS", "Season"]
    out = coerce_numeric(out, numeric_cols)
    for c in ["MIN", "PTS", "REB", "AST", "FGA", "FGM", "FG3A", "FG3M", "FTA", "FTM", "TOV", "OREB", "DREB", "STL", "BLK"]:
        out[c] = out[c].fillna(0)
    out["PRA"] = out["PTS"] + out["REB"] + out["AST"]
    out["NameKey"] = out["Player"].map(normalize_name)
    out["GameKey"] = out["GameID"].fillna("").astype(str)
    out = out[out["NameKey"].str.len() > 0].copy()
    out = out.drop_duplicates(subset=["NameKey", "Team", "GameDate", "PTS", "REB", "AST"], keep="last")
    return out


def standardize_player_season(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    player_col = find_col(d, ["PLAYER_NAME", "player_name", "athlete_display_name", "athlete_name", "display_name", "name", "player", "athlete"])
    team_col = find_col(d, ["TEAM_ABBREVIATION", "team_abbreviation", "team_short_display_name", "team", "team_name", "team_display_name"])
    season_col = find_col(d, ["SEASON", "season", "year"])
    out = pd.DataFrame()
    out["Player"] = d[player_col] if player_col else np.nan
    out["Team"] = d[team_col] if team_col else ""
    out["Season"] = d[season_col] if season_col else np.nan
    for out_col, candidates in {
        "GP": ["GP", "games", "games_played"],
        "MIN": ["MIN", "minutes", "minutes_per_game", "avg_minutes"],
        "PTS": ["PTS", "points", "points_per_game"],
        "REB": ["REB", "rebounds", "rebounds_per_game"],
        "AST": ["AST", "assists", "assists_per_game"],
        "USG%": ["USG%", "usage_rate", "usage", "usage_percentage"],
        "TS%": ["TS%", "true_shooting_percentage", "true_shooting"],
        "eFG%": ["eFG%", "effective_field_goal_percentage", "effective_fg_pct"],
        "AST%": ["AST%", "assist_rate", "assist_percentage"],
        "TRB%": ["TRB%", "total_rebound_percentage", "rebound_rate"],
        "PER": ["PER", "player_efficiency_rating"],
    }.items():
        c = find_col(d, candidates)
        out[out_col] = d[c] if c else np.nan
    out = coerce_numeric(out, [c for c in out.columns if c not in ["Player", "Team"]])
    out["PRA"] = out["PTS"].fillna(0) + out["REB"].fillna(0) + out["AST"].fillna(0)
    out["NameKey"] = out["Player"].map(normalize_name)
    return out.dropna(subset=["Player"])


def standardize_team_season(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    team_col = find_col(d, ["TEAM", "team", "team_abbreviation", "team_name", "team_display_name", "team_short_display_name"])
    season_col = find_col(d, ["SEASON", "season", "year"])
    out = pd.DataFrame()
    out["Team"] = d[team_col] if team_col else np.nan
    out["Season"] = d[season_col] if season_col else np.nan
    colsets = {
        "GP": ["GP", "games", "games_played"],
        "PTS": ["PTS", "points", "points_per_game"],
        "REB": ["REB", "rebounds", "rebounds_per_game"],
        "AST": ["AST", "assists", "assists_per_game"],
        "Pace": ["PACE", "pace", "possessions_per_48"],
        "ORtg": ["ORtg", "offensive_rating", "off_rating", "off_rtg"],
        "DRtg": ["DRtg", "defensive_rating", "def_rating", "def_rtg"],
        "NetRtg": ["NetRtg", "net_rating", "net_rtg"],
        "eFG%": ["eFG%", "effective_field_goal_percentage", "effective_fg_pct"],
        "TS%": ["TS%", "true_shooting_percentage", "true_shooting"],
        "FGA": ["FGA", "field_goals_attempted"],
        "FGM": ["FGM", "field_goals_made"],
        "FG3M": ["FG3M", "three_point_field_goals_made", "3PM"],
        "FTA": ["FTA", "free_throws_attempted"],
        "TOV": ["TOV", "turnovers"],
        "OREB": ["OREB", "offensive_rebounds"],
    }
    for out_col, cand in colsets.items():
        c = find_col(d, cand)
        out[out_col] = d[c] if c else np.nan
    out = coerce_numeric(out, [c for c in out.columns if c != "Team"])
    out["Team"] = out["Team"].astype(str)
    return out.dropna(subset=["Team"])


def standardize_schedules(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    date_col = find_col(d, ["date", "game_date", "start_date", "game_date_time"])
    season_col = find_col(d, ["season", "year"])
    game_id_col = find_col(d, ["game_id", "event_id", "id", "competition_id"])
    home_col = find_col(d, ["home_team", "home", "home_team_name", "home_team_abbreviation", "home_display_name"])
    away_col = find_col(d, ["away_team", "away", "away_team_name", "away_team_abbreviation", "away_display_name"])
    home_score_col = find_col(d, ["home_score", "home_team_score", "home_points", "score_home"])
    away_score_col = find_col(d, ["away_score", "away_team_score", "away_points", "score_away"])
    out = pd.DataFrame()
    out["GameDate"] = parse_date_series(d[date_col]) if date_col else pd.NaT
    out["Season"] = d[season_col] if season_col else out["GameDate"].dt.year
    out["GameID"] = d[game_id_col] if game_id_col else ""
    out["Home"] = d[home_col] if home_col else ""
    out["Away"] = d[away_col] if away_col else ""
    out["HomeScore"] = d[home_score_col] if home_score_col else np.nan
    out["AwayScore"] = d[away_score_col] if away_score_col else np.nan
    out = coerce_numeric(out, ["Season", "HomeScore", "AwayScore"])
    out["Home"] = out["Home"].astype(str)
    out["Away"] = out["Away"].astype(str)
    out["Margin"] = out["HomeScore"] - out["AwayScore"]
    return out.dropna(subset=["GameDate"], how="all")


def standardize_rosters(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    player_col = find_col(d, ["PLAYER_NAME", "player_name", "athlete_display_name", "display_name", "name", "player", "athlete"])
    team_col = find_col(d, ["TEAM", "team", "team_abbreviation", "team_name", "team_short_display_name"])
    pos_col = find_col(d, ["position", "pos", "athlete_position", "display_position"])
    season_col = find_col(d, ["season", "year"])
    out = pd.DataFrame()
    out["Player"] = d[player_col] if player_col else np.nan
    out["Team"] = d[team_col] if team_col else ""
    out["Position"] = d[pos_col] if pos_col else ""
    out["Season"] = d[season_col] if season_col else np.nan
    out["NameKey"] = out["Player"].map(normalize_name)
    out["PositionGroup"] = out["Position"].map(position_group)
    return out.dropna(subset=["Player"])


def standardize_game_rosters(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = standardize_rosters(df)
    d = df.copy()
    date_col = find_col(d, ["game_date", "date", "start_date"])
    game_id_col = find_col(d, ["game_id", "event_id", "competition_id"])
    active_col = find_col(d, ["active", "is_active", "did_play", "played"])
    starter_col = find_col(d, ["starter", "is_starter", "started"])
    out["GameDate"] = parse_date_series(d[date_col]) if date_col else pd.NaT
    out["GameID"] = d[game_id_col] if game_id_col else ""
    out["Active"] = d[active_col] if active_col else True
    out["Starter"] = d[starter_col] if starter_col else np.nan
    return out


def standardize_lineups(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    date_col = find_col(d, ["game_date", "date", "start_date", "GAME_DATE"])
    game_id_col = find_col(d, ["game_id", "event_id", "competition_id", "GAME_ID"])
    team_col = find_col(d, ["team", "team_abbreviation", "team_name", "TEAM_ABBREVIATION"])
    season_col = find_col(d, ["season", "year", "SEASON"])
    player_cols = [c for c in d.columns if re.search(r"player|athlete", col_norm(c))]
    out = pd.DataFrame()
    out["GameDate"] = parse_date_series(d[date_col]) if date_col else pd.NaT
    out["Season"] = d[season_col] if season_col else (out["GameDate"].dt.year if "GameDate" in out else np.nan)
    out["GameID"] = d[game_id_col] if game_id_col else ""
    out["Team"] = d[team_col] if team_col else ""
    use_cols = player_cols if player_cols else list(d.columns[:12])
    if use_cols:
        out["LineupText"] = d[use_cols].fillna("").astype(str).apply(lambda row: " | ".join([x for x in row.tolist() if x and x.lower() != "nan"]), axis=1)
    else:
        out["LineupText"] = ""
    out["PlayerCount"] = len(player_cols)
    out = coerce_numeric(out, ["Season", "PlayerCount"])
    return out

def _best_text_col_by_values(df: pd.DataFrame, preferred_tokens: List[str]) -> Optional[str]:
    """Find a name-like text column when SportsDataverse schema changes."""
    if df is None or df.empty:
        return None
    obj_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith('string')]
    scored = []
    for c in obj_cols[:80]:
        cn = col_norm(c)
        vals = df[c].dropna().astype(str).head(80)
        if vals.empty:
            continue
        # Name-like = two alpha tokens often enough.
        name_like = vals.map(lambda x: bool(re.search(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}", x))).mean()
        token_bonus = sum(1 for t in preferred_tokens if t in cn) * 0.35
        unique_bonus = min(0.15, vals.nunique() / max(len(vals), 1))
        scored.append((name_like + token_bonus + unique_bonus, c))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0] >= 0.35 else None


def standardize_shots(df: pd.DataFrame) -> pd.DataFrame:
    """Robust SportsDataverse shot parser.

    The shots parquet schema can vary. This parser intentionally accepts many column names
    and will save a useful shot-feature cache instead of returning empty whenever possible.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    # Prefer real name columns, then infer by values. Team is optional; player is required for matching.
    player_col = find_col(d, [
        "PLAYER_NAME", "player_name", "athlete_display_name", "athlete_name", "shooter",
        "shooter_name", "player", "athlete", "name", "display_name", "full_name",
        "text", "description"
    ])
    if not player_col:
        player_col = _best_text_col_by_values(d, ["athlete", "player", "shooter", "name"])
    team_col = find_col(d, [
        "team", "team_abbreviation", "team_name", "team_short_display_name", "team_display_name",
        "shooting_team", "offense_team", "home_away", "team_id"
    ], contains_any=["team"])
    date_col = find_col(d, ["game_date", "date", "start_date", "game_date_time", "created_at"])
    season_col = find_col(d, ["season", "year", "game_season"])
    game_id_col = find_col(d, ["game_id", "event_id", "competition_id", "id"])
    made_col = find_col(d, [
        "made", "shot_made", "shot_result", "result", "scoring_play", "is_made",
        "shot_outcome", "make_miss", "outcome"
    ])
    value_col = find_col(d, ["shot_value", "points", "point_value", "score_value", "attempt_points"])
    type_col = find_col(d, ["shot_type", "type", "action_type", "play_type", "sub_type", "shot_zone_basic", "shot_zone_area"])
    dist_col = find_col(d, ["shot_distance", "distance", "shot_dist", "distance_ft"])
    x_col = find_col(d, ["x", "loc_x", "coordinate_x", "x_coordinate"])
    y_col = find_col(d, ["y", "loc_y", "coordinate_y", "y_coordinate"])

    out = pd.DataFrame(index=d.index)
    if player_col:
        out["Player"] = d[player_col].astype(str)
    else:
        # Last fallback: sometimes shot rows include a long text description with a player name.
        desc_col = find_col(d, ["description", "text", "play_text", "play_description"])
        if desc_col:
            out["Player"] = d[desc_col].astype(str).str.extract(r"^([A-Z][a-zA-Z'\-.]+\s+[A-Z][a-zA-Z'\-.]+)", expand=False)
        else:
            return pd.DataFrame()
    out["Team"] = d[team_col].astype(str) if team_col else ""
    out["GameDate"] = parse_date_series(d[date_col]) if date_col else pd.NaT
    if season_col:
        out["Season"] = d[season_col]
    else:
        out["Season"] = out["GameDate"].dt.year if out["GameDate"].notna().any() else np.nan
    out["GameID"] = d[game_id_col] if game_id_col else ""
    out["ShotType"] = d[type_col].astype(str) if type_col else ""
    out["ShotDistance"] = d[dist_col] if dist_col else np.nan
    out["X"] = d[x_col] if x_col else np.nan
    out["Y"] = d[y_col] if y_col else np.nan

    if value_col:
        out["ShotValue"] = d[value_col]
    else:
        type_text = out["ShotType"].astype(str).str.lower()
        out["ShotValue"] = np.where(type_text.str.contains("3|three", na=False), 3, 2)

    if made_col:
        m = d[made_col]
        if m.dtype == object:
            mt = m.astype(str).str.lower()
            out["Made"] = mt.str.contains("made|make|true|yes|score|good|1", regex=True, na=False) & ~mt.str.contains("miss|blocked|false|no", regex=True, na=False)
        else:
            out["Made"] = pd.to_numeric(m, errors="coerce").fillna(0) > 0
    else:
        # If no make/miss field, leave unknown; attempts are still useful.
        out["Made"] = np.nan

    out = coerce_numeric(out, ["Season", "ShotValue", "ShotDistance", "X", "Y"])
    # Zone tags: conservative and schema-independent.
    type_text = out["ShotType"].astype(str).str.lower()
    out["Is3"] = (out["ShotValue"].fillna(0) >= 3) | type_text.str.contains("3|three", na=False)
    out["AtRim"] = (out["ShotDistance"].fillna(999) <= 5) | type_text.str.contains("layup|rim|dunk", na=False)
    out["MidRange"] = (~out["Is3"].fillna(False)) & (~out["AtRim"].fillna(False))
    out["NameKey"] = out["Player"].map(normalize_name)
    out = out[out["NameKey"].str.len() > 0].copy()
    return out


def build_shot_features(shots: pd.DataFrame) -> pd.DataFrame:
    sh = standardize_shots(shots) if shots is not None and not shots.empty else pd.DataFrame()
    if sh.empty:
        return pd.DataFrame()
    sh["Attempt"] = 1
    sh["MadeNum"] = sh["Made"].fillna(False).astype(bool).astype(int)
    sh["PtsOnShot"] = sh["MadeNum"] * pd.to_numeric(sh["ShotValue"], errors="coerce").fillna(2)
    agg = sh.groupby("NameKey").agg(
        ShotAttempts=("Attempt", "sum"),
        ShotMakes=("MadeNum", "sum"),
        ShotPoints=("PtsOnShot", "sum"),
        ThreePA=("Is3", "sum"),
        RimAttempts=("AtRim", "sum"),
        MidRangeAttempts=("MidRange", "sum"),
        AvgShotDistance=("ShotDistance", "mean"),
    ).reset_index()
    agg["ThreePARate"] = agg["ThreePA"] / agg["ShotAttempts"].replace(0, np.nan)
    agg["RimRate"] = agg["RimAttempts"] / agg["ShotAttempts"].replace(0, np.nan)
    agg["MidRangeRate"] = agg["MidRangeAttempts"] / agg["ShotAttempts"].replace(0, np.nan)
    agg["ShotMakeRate"] = agg["ShotMakes"] / agg["ShotAttempts"].replace(0, np.nan)
    agg["PointsPerShot"] = agg["ShotPoints"] / agg["ShotAttempts"].replace(0, np.nan)
    agg["ShotProfileScore"] = np.clip(
        50 + agg["ThreePARate"].fillna(0)*14 + agg["RimRate"].fillna(0)*18 + (agg["ShotMakeRate"].fillna(0.42)-0.42)*60,
        0, 100
    ).round(1)
    return agg


def position_group(pos) -> str:
    p = str(pos or "").upper()
    if any(x in p for x in ["G", "PG", "SG"]):
        return "Guard"
    if any(x in p for x in ["F", "SF", "PF"]):
        return "Wing"
    if any(x in p for x in ["C", "CENTER"]):
        return "Big"
    return "Unknown"


def standardize_dataset(dataset_key: str, df: pd.DataFrame) -> pd.DataFrame:
    if dataset_key == "player_game_logs":
        return standardize_player_logs(df)
    if dataset_key == "player_season_stats":
        return standardize_player_season(df)
    if dataset_key == "team_season_stats":
        return standardize_team_season(df)
    if dataset_key == "schedules":
        return standardize_schedules(df)
    if dataset_key == "rosters":
        return standardize_rosters(df)
    if dataset_key == "game_rosters":
        return standardize_game_rosters(df)
    if dataset_key == "lineups":
        return standardize_lineups(df)
    if dataset_key == "shots":
        return standardize_shots(df)
    return df


def save_dataset(dataset_key: str, df: pd.DataFrame) -> None:
    if df is None:
        return
    path = CACHE_FILES[dataset_key]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_dataset(dataset_key: str) -> pd.DataFrame:
    """Load data with resilient priority:
    1) local repo/runtime cache at wnba_engine/data
    2) raw GitHub cache if WNBA_DATA_BASE_URL/GITHUB_DATA_BASE_URL is configured
    3) empty frame, allowing official WNBA fallback builders to run

    Important: empty/bad remote pulls never overwrite good local CSVs.
    """
    path = CACHE_FILES.get(dataset_key)
    # 1) Local file from GitHub repo checkout or runtime cache.
    if path and path.exists():
        try:
            df = pd.read_csv(path, low_memory=False)
            if dataset_key in ["player_game_logs", "schedules", "game_rosters", "lineups", "shots", "master_features", "projection_board"]:
                for c in ["GameDate"]:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c], errors="coerce")
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

    # 2) Raw GitHub fallback if configured. Saves a runtime copy for speed.
    try:
        remote_df, remote_status = fetch_github_cache_dataset(dataset_key)
        try:
            st.session_state.setdefault("wnba_github_cache_debug", {})[dataset_key] = remote_status
        except Exception:
            pass
        if remote_df is not None and not remote_df.empty:
            try:
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    remote_df.to_csv(path, index=False)
            except Exception:
                pass
            return remote_df
    except Exception:
        pass

    # 3) No cache available; caller can trigger official WNBA fallback.
    return pd.DataFrame()

# ============================================================
# Feature builders
# ============================================================
def build_team_ranks(player_logs: pd.DataFrame, team_season: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if team_season is not None and not team_season.empty:
        d = standardize_team_season(team_season)
        if not d.empty:
            frames.append(d)
    if player_logs is not None and not player_logs.empty:
        d = standardize_player_logs(player_logs)
        if not d.empty:
            agg = d.groupby(["Season", "Team"], dropna=False).agg(
                Games=("GameDate", "nunique"), PTS=("PTS", "sum"), REB=("REB", "sum"), AST=("AST", "sum"),
                FGA=("FGA", "sum"), FGM=("FGM", "sum"), FG3M=("FG3M", "sum"), FTA=("FTA", "sum"), TOV=("TOV", "sum"), OREB=("OREB", "sum")
            ).reset_index()
            poss = agg["FGA"].fillna(0) + 0.44*agg["FTA"].fillna(0) + agg["TOV"].fillna(0) - agg["OREB"].fillna(0)
            agg["Pace"] = np.where(agg["Games"] > 0, poss / agg["Games"], np.nan)
            agg["ORtg"] = np.where(poss > 0, 100 * agg["PTS"] / poss, np.nan)
            agg["eFG%"] = np.where(agg["FGA"] > 0, (agg["FGM"] + 0.5*agg["FG3M"]) / agg["FGA"], np.nan)
            agg["TS%"] = np.where((2*(agg["FGA"] + 0.44*agg["FTA"])) > 0, agg["PTS"] / (2*(agg["FGA"] + 0.44*agg["FTA"])), np.nan)
            frames.append(agg)
    if frames:
        all_team = pd.concat(frames, ignore_index=True, sort=False)
        all_team = all_team.groupby(["Season", "Team"], dropna=False).agg(lambda x: x.dropna().iloc[-1] if len(x.dropna()) else np.nan).reset_index()
    else:
        all_team = pd.DataFrame(columns=["Season", "Team"])

    if schedules is not None and not schedules.empty:
        sched = standardize_schedules(schedules)
        if not sched.empty and {"Home", "Away", "HomeScore", "AwayScore"}.issubset(sched.columns):
            rows = []
            for _, r in sched.dropna(subset=["HomeScore", "AwayScore"]).iterrows():
                rows.append({"Season": r.get("Season"), "Team": r.get("Home"), "PtsFor": r.get("HomeScore"), "PtsAllowed": r.get("AwayScore"), "HomeAway": "HOME"})
                rows.append({"Season": r.get("Season"), "Team": r.get("Away"), "PtsFor": r.get("AwayScore"), "PtsAllowed": r.get("HomeScore"), "HomeAway": "AWAY"})
            if rows:
                sagg = pd.DataFrame(rows).groupby(["Season", "Team"], dropna=False).agg(
                    GamesSchedule=("PtsFor", "count"), PPG=("PtsFor", "mean"), PointsAllowed=("PtsAllowed", "mean"), AvgMargin=("PtsFor", lambda x: np.nan)
                ).reset_index()
                marg = pd.DataFrame(rows)
                marg["Margin"] = marg["PtsFor"] - marg["PtsAllowed"]
                marg_agg = marg.groupby(["Season", "Team"], dropna=False)["Margin"].mean().reset_index(name="AvgMargin")
                sagg = sagg.drop(columns=["AvgMargin"]).merge(marg_agg, on=["Season", "Team"], how="left")
                all_team = all_team.merge(sagg, on=["Season", "Team"], how="outer") if not all_team.empty else sagg
    if all_team.empty:
        return all_team

    # Defensive rating proxy when not directly provided.
    if "DRtg" not in all_team.columns:
        all_team["DRtg"] = np.nan
    if "PointsAllowed" in all_team.columns:
        all_team["DRtg"] = all_team["DRtg"].fillna(all_team["PointsAllowed"])
    if "NetRtg" not in all_team.columns:
        all_team["NetRtg"] = np.nan
    all_team["NetRtg"] = all_team["NetRtg"].fillna(all_team.get("ORtg", np.nan) - all_team.get("DRtg", np.nan))

    # Ranks: high ORtg/Net/Pace good; low DRtg/points allowed good.
    for season, idx in all_team.groupby("Season").groups.items():
        sub_idx = list(idx)
        for col, asc, rank_col in [
            ("ORtg", False, "OffensiveRank"), ("DRtg", True, "DefensiveRank"), ("NetRtg", False, "NetRank"),
            ("Pace", False, "PaceRank"), ("PointsAllowed", True, "PointsAllowedRank"),
            ("REB", False, "ReboundRank"), ("AST", False, "AssistRank")
        ]:
            if col in all_team.columns:
                all_team.loc[sub_idx, rank_col] = all_team.loc[sub_idx, col].rank(ascending=asc, method="min")
    return all_team


def compute_player_baselines(logs: pd.DataFrame, season_stats: pd.DataFrame = pd.DataFrame(), shots: pd.DataFrame = pd.DataFrame(), rosters: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    logs = standardize_player_logs(logs) if logs is not None and not logs.empty else pd.DataFrame()
    if logs.empty:
        return pd.DataFrame()
    d = logs.sort_values(["NameKey", "GameDate"])
    for c in ["MIN", "PTS", "REB", "AST", "PRA", "FGA", "FGM", "FG3A", "FG3M", "FTA", "TOV", "OREB", "DREB"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    g = d.groupby(["NameKey", "Player"], dropna=False)
    base = g.agg(
        Team=("Team", lambda x: x.dropna().iloc[-1] if len(x.dropna()) else ""),
        Season=("Season", lambda x: x.dropna().iloc[-1] if len(x.dropna()) else np.nan),
        Games=("PTS", "count"),
        LastGame=("GameDate", "max"),
        MIN_avg=("MIN", "mean"), MIN_l3=("MIN", lambda x: x.tail(3).mean()), MIN_l5=("MIN", lambda x: x.tail(5).mean()), MIN_l10=("MIN", lambda x: x.tail(10).mean()),
        PTS_avg=("PTS", "mean"), REB_avg=("REB", "mean"), AST_avg=("AST", "mean"), PRA_avg=("PRA", "mean"),
        PTS_l3=("PTS", lambda x: x.tail(3).mean()), REB_l3=("REB", lambda x: x.tail(3).mean()), AST_l3=("AST", lambda x: x.tail(3).mean()), PRA_l3=("PRA", lambda x: x.tail(3).mean()),
        PTS_l5=("PTS", lambda x: x.tail(5).mean()), REB_l5=("REB", lambda x: x.tail(5).mean()), AST_l5=("AST", lambda x: x.tail(5).mean()), PRA_l5=("PRA", lambda x: x.tail(5).mean()),
        PTS_l10=("PTS", lambda x: x.tail(10).mean()), REB_l10=("REB", lambda x: x.tail(10).mean()), AST_l10=("AST", lambda x: x.tail(10).mean()), PRA_l10=("PRA", lambda x: x.tail(10).mean()),
        PTS_l20=("PTS", lambda x: x.tail(20).mean()), REB_l20=("REB", lambda x: x.tail(20).mean()), AST_l20=("AST", lambda x: x.tail(20).mean()), PRA_l20=("PRA", lambda x: x.tail(20).mean()),
        FGA=("FGA", "sum"), FGM=("FGM", "sum"), FG3A=("FG3A", "sum"), FG3M=("FG3M", "sum"), FTA=("FTA", "sum"), TOV=("TOV", "sum"), OREB=("OREB", "sum"), DREB=("DREB", "sum"),
    ).reset_index()
    base["eFG%"] = np.where(base["FGA"] > 0, (base["FGM"] + 0.5*base["FG3M"]) / base["FGA"], np.nan)
    base["TS%"] = np.where((2*(base["FGA"] + 0.44*base["FTA"])) > 0, (base["PTS_avg"]*base["Games"]) / (2*(base["FGA"] + 0.44*base["FTA"])), np.nan)
    base["UsageProxy"] = (base["FGA"] + 0.44*base["FTA"] + base["TOV"]) / base["Games"].clip(lower=1)
    base["AST%Proxy"] = base["AST_avg"] / base["MIN_avg"].replace(0, np.nan)
    base["TRB%Proxy"] = base["REB_avg"] / base["MIN_avg"].replace(0, np.nan)
    base["PERProxy"] = (base["PTS_avg"] + base["REB_avg"] + base["AST_avg"] + base["FGM"] / base["Games"].clip(lower=1) - (base["FGA"] - base["FGM"]) / base["Games"].clip(lower=1) - base["TOV"] / base["Games"].clip(lower=1))

    # Add season advanced fields where available.
    ss = standardize_player_season(season_stats) if season_stats is not None and not season_stats.empty else pd.DataFrame()
    if not ss.empty:
        ss_latest = ss.sort_values("Season").groupby("NameKey", as_index=False).tail(1)
        keep = [c for c in ["NameKey", "USG%", "TS%", "eFG%", "AST%", "TRB%", "PER"] if c in ss_latest.columns]
        base = base.merge(ss_latest[keep], on="NameKey", how="left", suffixes=("", "_Season"))
        for c in ["USG%", "TS%", "eFG%", "AST%", "TRB%", "PER"]:
            if c in base.columns and f"{c}_Season" in base.columns:
                base[c] = base[c].fillna(base[f"{c}_Season"])
            elif f"{c}_Season" in base.columns:
                base[c] = base[f"{c}_Season"]

    # Roster position.
    rost = standardize_rosters(rosters) if rosters is not None and not rosters.empty else pd.DataFrame()
    if not rost.empty:
        rr = rost.sort_values("Season").groupby("NameKey", as_index=False).tail(1)
        base = base.merge(rr[["NameKey", "Position", "PositionGroup"]].drop_duplicates("NameKey"), on="NameKey", how="left")
    else:
        base["Position"] = ""
        base["PositionGroup"] = "Unknown"

    # Shot profile for points. Flexible parser supports SportsDataverse parquet schemas.
    sh_agg = build_shot_features(shots)
    if not sh_agg.empty:
        base = base.merge(sh_agg, on="NameKey", how="left")
    else:
        for c in ["ShotAttempts", "ShotMakes", "ShotPoints", "ThreePA", "RimAttempts", "MidRangeAttempts", "ThreePARate", "RimRate", "MidRangeRate", "ShotMakeRate", "PointsPerShot", "ShotProfileScore", "AvgShotDistance"]:
            base[c] = np.nan

    # Home/away and consistency features from logs.
    if "HomeAway" in d.columns:
        ha = d.copy()
        ha["HA"] = ha["HomeAway"].astype(str).str.upper().str[:1]
        for m in MARKETS:
            piv = ha.pivot_table(index="NameKey", columns="HA", values=m, aggfunc="mean")
            if "H" in piv.columns:
                base = base.merge(piv[["H"]].rename(columns={"H": f"{m}_HomeAvg"}).reset_index(), on="NameKey", how="left")
            if "A" in piv.columns:
                base = base.merge(piv[["A"]].rename(columns={"A": f"{m}_AwayAvg"}).reset_index(), on="NameKey", how="left")
    for m in MARKETS:
        base[f"{m}_Std20"] = g[m].agg(lambda x: x.tail(20).std(ddof=0)).values
        base[f"{m}_per_min"] = base[f"{m}_avg"] / base["MIN_avg"].replace(0, np.nan)

    base["VolatilityScore"] = np.clip(
        30 + base[[f"{m}_Std20" for m in MARKETS if f"{m}_Std20" in base.columns]].fillna(0).mean(axis=1) * 8,
        0, 100
    ).round(1)
    return base


def build_master_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    logs = load_dataset("player_game_logs")
    ss = load_dataset("player_season_stats")
    ts = load_dataset("team_season_stats")
    sched = load_dataset("schedules")
    rosters = load_dataset("rosters")
    gr = load_dataset("game_rosters")
    lineups = load_dataset("lineups")
    shots = load_dataset("shots")

    team_ranks = build_team_ranks(logs, ts, sched)
    save_dataset("team_ranks", team_ranks)

    base = compute_player_baselines(logs, ss, shots, rosters)
    if base.empty:
        return pd.DataFrame(), team_ranks

    # Add team context by player's current team / latest season.
    if not team_ranks.empty:
        tr = team_ranks.sort_values("Season").groupby("Team", as_index=False).tail(1)
        base = base.merge(tr.add_prefix("Team_"), left_on="Team", right_on="Team_Team", how="left")

    # Game roster role confidence.
    grs = standardize_game_rosters(gr) if not gr.empty else pd.DataFrame()
    if not grs.empty:
        grs["StarterBool"] = grs["Starter"].astype(str).str.lower().isin(["true", "1", "yes", "starter", "started"])
        rconf = grs.groupby("NameKey").agg(RosterGames=("Player", "count"), StarterGames=("StarterBool", "sum")).reset_index()
        rconf["StarterRate"] = rconf["StarterGames"] / rconf["RosterGames"].replace(0, np.nan)
        base = base.merge(rconf, on="NameKey", how="left")
    else:
        base["RosterGames"] = np.nan
        base["StarterGames"] = np.nan
        base["StarterRate"] = np.nan

    # Lineups: count mentions for continuity/role signal. This uses normalized lineup text so it works
    # even when SportsDataverse gives lineup-player columns instead of one clean player field.
    lns = standardize_lineups(lineups) if not lineups.empty else pd.DataFrame()
    if not lns.empty:
        lns_text = lns["LineupText"].astype(str).map(normalize_name)
        total_lineups = max(len(lns_text), 1)
        mentions = []
        for _, b in base[["NameKey", "Player"]].drop_duplicates().iterrows():
            nk = b["NameKey"]
            cnt = int(lns_text.str.contains(nk, regex=False, na=False).sum()) if nk else 0
            mentions.append({
                "NameKey": nk,
                "LineupMentions": cnt,
                "LineupShare": cnt / total_lineups,
                "LineupContinuityScore": round(min(100, 35 + 65 * (cnt / total_lineups) * 8), 1),
            })
        base = base.merge(pd.DataFrame(mentions), on="NameKey", how="left")
    else:
        base["LineupMentions"] = np.nan
        base["LineupShare"] = np.nan
        base["LineupContinuityScore"] = np.nan

    base["RoleConfidence"] = np.clip(
        43
        + base["Games"].fillna(0).clip(0, 20)*1.3
        + base["MIN_l10"].fillna(base["MIN_avg"]).clip(0, 36)*0.7
        + base["StarterRate"].fillna(0)*18
        + base["LineupContinuityScore"].fillna(50)*0.08,
        0, 100
    ).round(1)
    base["MinutesSafetyGrade"] = np.select(
        [base["MIN_l10"] >= 30, base["MIN_l10"] >= 24, base["MIN_l10"] >= 18],
        ["A", "B", "C"], default="D"
    )
    base["TeamMatchupStrengthScore"] = np.clip(50 + base.get("Team_NetRtg", pd.Series(0, index=base.index)).fillna(0)*2.0, 0, 100).round(1)
    base["MinutesProjectionBase"] = (0.35*base["MIN_avg"].fillna(0) + 0.30*base["MIN_l10"].fillna(0) + 0.20*base["MIN_l5"].fillna(0) + 0.15*base["MIN_l3"].fillna(0)).round(2)
    base["BayesianPriorStrength"] = np.clip((base["Games"].fillna(0) / 20) * 100, 0, 100).round(1)
    base["DataScore"] = np.clip(
        28
        + base["Games"].fillna(0).clip(0, 25)*1.9
        + base["MIN_avg"].fillna(0).clip(0, 36)*0.75
        + base["RoleConfidence"].fillna(0)*0.24
        + base.get("ShotProfileScore", pd.Series(0, index=base.index)).fillna(0)*0.06
        + base["LineupContinuityScore"].fillna(50)*0.06,
        0, 100
    ).round(1)
    save_dataset("master_features", base)
    return base, team_ranks

# ============================================================
# Line pulls
# ============================================================
def text_blob(*objs):
    keys = ["name", "title", "display_title", "display_name", "full_name", "first_name", "last_name", "player_name", "stat", "stat_type", "appearance_stat", "market", "market_name", "league", "sport", "description", "label", "team", "abbr_name", "short_name"]
    parts = []
    for obj in objs:
        a = attrs(obj)
        for k in keys:
            v = a.get(k)
            if isinstance(v, dict):
                v = v.get("name") or v.get("display_name") or v.get("title")
            if v not in [None, ""]:
                parts.append(str(v))
    return " | ".join(parts)


def infer_market(blob):
    """Map sportsbook labels into the app markets.

    Underdog displays WNBA markets as full labels such as:
      Points, Rebounds, Assists, Pts + Rebs + Asts
    The app uses PTS/REB/AST/PRA internally.
    """
    raw = str(blob or "")
    low = raw.lower()
    b = normalize_name(raw)

    # Exclude props we do not model right now.
    if any(x in low for x in ["fantasy", "steal", "steals", "block", "blocks", "turnover", "turnovers", "free throw"]):
        return None
    if any(x in low for x in ["3-pointers", "3 pointers", "three-pointers", "three pointers", "3pt", "3 pm", "3pm", "threes made"]):
        return None

    # Hard reject combo markets we do NOT model as standalone tabs.
    # This prevents Points+Rebounds / Points+Assists / Rebounds+Assists from
    # being incorrectly routed into PTS/REB/AST and creating bad lines like 17.5/18.5.
    combo_not_supported = [
        "points rebounds", "points rebs", "pts rebs", "pts reb",
        "points assists", "points asts", "pts asts", "pts ast",
        "rebounds assists", "rebs asts", "reb ast", "rebs assists",
        "points rebounds o", "points assists o", "rebounds assists o",
    ]
    # PRA can appear in several Underdog forms and is the only combo we keep.
    is_pra = (
        ("points" in b and "rebounds" in b and "assists" in b)
        or ("pts" in b and ("reb" in b or "rebs" in b) and ("ast" in b or "asts" in b))
        or "pts rebs asts" in b
        or "pts reb ast" in b
        or "pra" in b
        or "points rebounds assists" in b
    )
    if is_pra:
        return "PRA"
    if any(x in b for x in combo_not_supported):
        return None
    # Exact single-stat markets only.
    if re.search(r"\b(points?|pts)\b", b):
        return "PTS"
    if re.search(r"\b(rebounds?|rebs?|reb)\b", b):
        return "REB"
    if re.search(r"\b(assists?|asts?|ast)\b", b):
        return "AST"
    return None

def line_from_obj(*objs):
    safe_keys = ["stat_value", "line_score", "over_under_line", "target_value", "line", "value", "points"]
    for obj in objs:
        a = attrs(obj)
        for k in safe_keys:
            val = safe_float(a.get(k), np.nan)
            if pd.notna(val) and 0.5 <= val <= 80:
                return float(val)
    blob = " ".join(json.dumps(attrs(o), default=str) for o in objs if isinstance(o, dict))
    nums = re.findall(r"(?<!\d)(\d{1,2}(?:\.5)?)(?!\d)", blob)
    vals = [safe_float(n, np.nan) for n in nums]
    vals = [v for v in vals if pd.notna(v) and 0.5 <= v <= 80]
    return float(vals[0]) if vals else np.nan


def player_from_obj(*objs):
    for obj in objs:
        a = attrs(obj)
        candidates = [
            a.get("display_name"), a.get("full_name"), a.get("player_name"), a.get("name"), a.get("title"),
            (str(a.get("first_name", "")).strip() + " " + str(a.get("last_name", "")).strip()).strip(),
        ]
        for c in candidates:
            if c and len(normalize_name(c).split()) >= 2:
                return str(c)
    return ""


def rel_id(obj, names):
    rels = obj.get("relationships") if isinstance(obj, dict) else None
    if not isinstance(rels, dict):
        return None
    for n in names:
        for key in {n, n.replace("_", "-"), n.replace("_", "") }:
            node = rels.get(key)
            data = node.get("data") if isinstance(node, dict) else node
            if isinstance(data, dict) and data.get("id") not in [None, ""]:
                return str(data.get("id"))
            if isinstance(data, list):
                for x in data:
                    if isinstance(x, dict) and x.get("id") not in [None, ""]:
                        return str(x.get("id"))
    return None


@st.cache_data(ttl=180, show_spinner=False)
def fetch_underdog_board():
    """Underdog WNBA parser with deep decode mode.

    This version is intentionally strict about player identity but generous about
    where Underdog may place the player/stat/line inside JSON. It writes a decoder
    table to DATA_DIR/wnba_underdog_decode.csv every refresh so we can see exactly
    which raw rows were accepted or rejected.
    """
    rows, debug, decode_rows = [], [], []

    master = load_dataset("master_features")
    if master.empty:
        master = compute_player_baselines(load_dataset("player_game_logs"), load_dataset("player_season_stats"), load_dataset("shots"), load_dataset("rosters"))
    player_pool = []
    if not master.empty and "Player" in master.columns:
        for _, r in master.dropna(subset=["Player"]).iterrows():
            player = str(r.get("Player", "")).strip()
            if not player:
                continue
            nk = normalize_name(player)
            toks = nk.split()
            player_pool.append({
                "Player": player,
                "NameKey": nk,
                "Team": str(r.get("Team", "")).upper().strip(),
                "FirstInitial": toks[0][0] if toks else "",
                "Last": toks[-1] if toks else "",
            })
    # de-dupe player pool
    seen_keys, pp = set(), []
    for rec in player_pool:
        k = (rec["NameKey"], rec["Team"])
        if k not in seen_keys:
            seen_keys.add(k); pp.append(rec)
    player_pool = pp

    def active_schedule_teams():
        sched = load_dataset("schedules")
        teams = set()
        if sched is None or sched.empty:
            return teams
        try:
            s = standardize_schedules(sched)
        except Exception:
            s = sched.copy()
        if s.empty:
            return teams
        if "GameDate" in s.columns:
            gd = pd.to_datetime(s["GameDate"], errors="coerce")
            today = pd.Timestamp(datetime.now().date())
            # Include today + tomorrow because WNBA props can be posted after midnight/UTC mismatch.
            mask = gd.dt.normalize().isin([today, today + pd.Timedelta(days=1), today - pd.Timedelta(days=1)])
            if mask.any():
                s = s[mask].copy()
        for c in ["Home", "Away"]:
            if c in s.columns:
                teams.update([str(x).upper().strip() for x in s[c].dropna().tolist() if str(x).strip()])
        return {t for t in teams if 2 <= len(t) <= 4}

    ACTIVE_TEAMS = active_schedule_teams()

    def attr(obj):
        return attrs(obj) if isinstance(obj, dict) else {}

    def collect_objects(data):
        out = []
        def walk(x, path="root"):
            if isinstance(x, dict):
                y = dict(x); y["_path"] = path
                out.append(y)
                for k, v in x.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, f"{path}[{i}]")
        walk(data)
        return out

    def get_id(obj):
        if not isinstance(obj, dict):
            return ""
        v = obj.get("id") or attr(obj).get("id")
        return str(v) if v not in [None, ""] else ""

    def get_type(obj):
        if not isinstance(obj, dict):
            return ""
        return str(obj.get("type") or obj.get("_parent_key") or obj.get("_path") or "").lower().replace("-", "_")

    def relation_ids(obj):
        ids = []
        if not isinstance(obj, dict):
            return ids
        rels = obj.get("relationships", {})
        if isinstance(rels, dict):
            for _, node in rels.items():
                data = node.get("data") if isinstance(node, dict) else node
                if isinstance(data, dict) and data.get("id") not in [None, ""]:
                    ids.append(str(data.get("id")))
                elif isinstance(data, list):
                    ids += [str(x.get("id")) for x in data if isinstance(x, dict) and x.get("id") not in [None, ""]]
        a = attr(obj)
        for k, v in a.items():
            kn = col_norm(k)
            if kn.endswith("_id") or kn in {"playerid", "appearanceid", "overunderid", "matchid", "gameid"}:
                if v not in [None, ""]:
                    ids.append(str(v))
        return list(dict.fromkeys(ids))

    def object_text(obj, max_len=900):
        a = attr(obj)
        keys = [
            "title","display_title","name","display_name","full_name","first_name","last_name","player_name",
            "short_name","abbr_name","stat","stat_type","appearance_stat","display_stat","label","market",
            "market_name","description","league","league_name","sport","sport_name","team","team_abbreviation",
            "home_team","away_team","match_title","event_title","game_title","scheduled_at","start_time"
        ]
        vals = []
        for k in keys:
            v = a.get(k)
            if isinstance(v, dict):
                vals += [str(v.get(kk)) for kk in keys if v.get(kk) not in [None, ""]]
            elif isinstance(v, list):
                vals += [str(x) for x in v[:4] if x not in [None, ""]]
            elif v not in [None, ""]:
                vals.append(str(v))
        if not vals:
            vals = [json.dumps(a, default=str)[:max_len]]
        return " | ".join(vals)[:max_len]

    def candidate_text(*objs):
        parts = []
        for o in objs:
            if isinstance(o, dict):
                parts.append(object_text(o))
        return " | ".join([p for p in parts if p])[:2500]

    def team_from_text(txt):
        txt = str(txt or "").upper()
        # Common WNBA abbreviations from your master file + sportsbook labels.
        teams = sorted({r.get("Team", "") for r in player_pool if r.get("Team")}, key=len, reverse=True)
        for t in teams:
            if re.search(rf"\b{re.escape(t)}\b", txt):
                return t
        return ""

    def bad_candidate(s):
        nk = normalize_name(s)
        if not nk:
            return True
        toks = nk.split()
        if len(toks) < 2:
            return True
        if any(re.search(r"\d", t) for t in toks):
            return True
        if any(t in {"none","null","true","false","country","player","points","rebounds","assists","higher","lower","over","under","wnba","line"} for t in toks):
            return True
        return False

    def clean_name_piece(s):
        s = str(s or "")
        # Remove market words when titles look like 'J. Young Points'.
        s = re.sub(r"\b(Points|Point|Rebounds|Rebound|Assists|Assist|Pts\s*\+\s*Rebs\s*\+\s*Asts|Pts|Rebs|Asts|PRA|Higher|Lower)\b", " ", s, flags=re.I)
        s = re.sub(r"[^A-Za-z.'\-\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def extract_name_candidates(*objs):
        candidates = []
        priority_keys = ["player_name","full_name","display_name","name","title","short_name","abbr_name","first_name","last_name"]
        for o in objs:
            if not isinstance(o, dict):
                continue
            a = attr(o)
            first, last = a.get("first_name"), a.get("last_name")
            if first and last:
                candidates.append(f"{first} {last}")
            for k in priority_keys:
                v = a.get(k)
                if isinstance(v, str) and v.strip():
                    candidates.append(v)
                elif isinstance(v, dict):
                    for kk in priority_keys:
                        if isinstance(v.get(kk), str):
                            candidates.append(v.get(kk))
        # Search whole evidence for full cached player names and initial-last forms.
        evidence = candidate_text(*objs)
        ev_norm = normalize_name(evidence)
        for rec in player_pool:
            if rec["NameKey"] and rec["NameKey"] in ev_norm:
                candidates.append(rec["Player"])
            if rec["FirstInitial"] and rec["Last"]:
                if re.search(rf"\b{re.escape(rec['FirstInitial'])}\.?\s+{re.escape(rec['Last'])}\b", evidence, flags=re.I):
                    candidates.append(f"{rec['FirstInitial']} {rec['Last']}")
        # Clean and de-dupe
        cleaned = []
        for c in candidates:
            cc = clean_name_piece(c)
            if cc and cc not in cleaned:
                cleaned.append(cc)
        return cleaned[:30]

    def resolve_player(candidates, evidence="", team_hint=""):
        best = {"Player":"", "Team":"", "Score":0.0, "Candidate":"", "Reason":"no candidate"}
        evidence_norm = normalize_name(evidence)
        team_hint = str(team_hint or "").upper().strip()
        for cand in candidates:
            if bad_candidate(cand) and not re.match(r"^[A-Za-z]\.?(\s+)[A-Za-z'\-]+$", str(cand).strip()):
                continue
            cn = normalize_name(cand)
            ctoks = cn.split()
            for rec in player_pool:
                score = 0.0
                reason = ""
                if cn == rec["NameKey"]:
                    score, reason = 1.00, "exact"
                elif rec["NameKey"] and rec["NameKey"] in evidence_norm:
                    score, reason = 0.99, "full name in raw"
                elif len(ctoks) >= 2 and ctoks[-1] == rec["Last"] and ctoks[0][:1] == rec["FirstInitial"]:
                    score, reason = 0.96, "initial+last"
                else:
                    ns = name_score(cn, rec["NameKey"])
                    if ns >= 0.93:
                        score, reason = ns, "fuzzy"
                if score:
                    if team_hint and rec["Team"] == team_hint:
                        score += 0.03
                    if ACTIVE_TEAMS and rec["Team"] not in ACTIVE_TEAMS:
                        # Don't fully reject: schedule files can be stale, but heavily penalize.
                        score -= 0.15
                    if score > best["Score"]:
                        best = {"Player":rec["Player"], "Team":rec["Team"], "Score":round(float(score), 4), "Candidate":cand, "Reason":reason}
        if best["Score"] < 0.92:
            best["Player"] = ""; best["Team"] = team_hint; best["Reason"] = "low match score"
        return best

    def parse_market(txt):
        return infer_market(txt)

    def _ud_options(o):
        """Return Underdog option rows attached to one over_under_line object."""
        if not isinstance(o, dict):
            return []
        opts = o.get("options")
        if opts is None and isinstance(o.get("attributes"), dict):
            opts = o["attributes"].get("options")
        return opts if isinstance(opts, list) else []

    def _ud_has_two_sided_options(o):
        """Main board lines usually have Higher and Lower. Alternate ladders often do not."""
        choices = set()
        for opt in _ud_options(o):
            if not isinstance(opt, dict):
                continue
            a = attrs(opt)
            blob = " ".join(str(a.get(k, "")) for k in ["choice", "choice_display", "choice_display_short", "choice_id", "selection_header"]).lower()
            if "higher" in blob or "over" in blob:
                choices.add("higher")
            if "lower" in blob or "under" in blob:
                choices.add("lower")
        return "higher" in choices and "lower" in choices

    def parse_line(*objs):
        # Critical: only trust the official Underdog main over_under_line object.
        # Never pull from related appearance objects, option text like "26+", payout
        # multipliers, decimal prices, sort_by, game logs, or ids.
        safe_keys = ["stat_value", "line_score", "target_value"]
        for o in objs:
            if not isinstance(o, dict):
                continue
            if not is_true_underdog_line_obj(o):
                continue
            if not _ud_has_two_sided_options(o):
                # This filters alternate ladder rows that are usually Higher-only.
                continue
            a = attr(o)
            for k in safe_keys:
                v = safe_float(a.get(k), np.nan)
                if pd.notna(v) and 0.5 <= v <= 80:
                    return float(v)
        return np.nan

    def is_true_underdog_line_obj(o):
        if not isinstance(o, dict):
            return False
        typ = get_type(o)
        path = str(o.get("_path", "")).lower()
        if "options" in path or "option" in typ:
            return False
        # Only the real over_under_line container can be a line. Related appearances
        # and appearance_stat objects can contain numbers, but those are not lines.
        return ("over_under_line" in typ or re.search(r"over_under_lines\[\d+\]$", path) is not None)

    def is_wnba_text(txt):
        low = str(txt or "").lower()
        if any(x in low for x in ["mlb","baseball","nfl","football","nhl","soccer","tennis","golf","mma","pga"]):
            return False
        if "wnba" in low or "basketball" in low:
            return True
        # If it mentions any active/master WNBA player, treat as WNBA.
        n = normalize_name(txt)
        return any(rec["NameKey"] and rec["NameKey"] in n for rec in player_pool[:400])

    def status_ok(*objs):
        bad = ["suspended", "removed", "closed", "settled", "graded", "canceled", "cancelled"]
        txt = candidate_text(*objs).lower()
        return not any(b in txt for b in bad)

    def add_decode(raw_name, raw_market, line, resolved, accepted, reason, parser, raw=""):
        decode_rows.append({
            "Raw Player": raw_name,
            "Raw Market": raw_market,
            "Line": line,
            "Resolved Player": resolved.get("Player", "") if isinstance(resolved, dict) else "",
            "Resolved Team": resolved.get("Team", "") if isinstance(resolved, dict) else "",
            "Match Score": resolved.get("Score", 0) if isinstance(resolved, dict) else 0,
            "Accepted": "✅" if accepted else "❌",
            "Reason": reason,
            "Parser": parser,
            "Raw Sample": str(raw)[:700],
        })

    def append_row(player, team, market, line, raw, parser):
        if market not in MARKETS:
            return False
        if pd.isna(line):
            return False
        rows.append({
            "Player": player,
            "Team": team,
            "Opponent": "",
            "Market": market,
            "Line": float(line),
            "Source": "Underdog",
            "Start": "",
            "Raw": str(raw)[:400],
            "Parser Mode": parser,
            "NameKey": normalize_name(player),
        })
        return True

    def get_json(url):
        headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Underdog/1.0",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://underdogfantasy.com",
            "Referer": "https://underdogfantasy.com/",
        }
        try:
            r = requests.get(url, headers=headers, timeout=20)
            status = int(getattr(r, "status_code", 0) or 0)
            ctype = str(r.headers.get("content-type", ""))
            if status != 200:
                debug.append({"source":"Underdog", "url":url, "status":f"HTTP {status}", "rows":0, "message":(r.text or "")[:250]})
                return None
            try:
                data = r.json()
                debug.append({"source":"Underdog", "url":url, "status":"json ok", "rows":0, "message":f"content-type={ctype}"})
                return data
            except Exception:
                debug.append({"source":"Underdog", "url":url, "status":"no json", "rows":0, "message":f"content-type={ctype}; body={(r.text or '')[:180]}"})
                return None
        except Exception as e:
            debug.append({"source":"Underdog", "url":url, "status":"request error", "rows":0, "message":str(e)[:250]})
            return None

    # Try endpoints in newest-to-oldest order first.
    for url in UNDERDOG_URLS:
        data = get_json(url)
        if not data:
            continue
        objects = collect_objects(data)
        by_id = {get_id(o): o for o in objects if get_id(o)}

        line_objs = []
        for o in objects:
            if is_true_underdog_line_obj(o):
                line_objs.append(o)

        # Do not scan arbitrary leaf/options objects for lines. If Underdog changes schema,
        # Decode Mode will show "no line found" rather than creating bad 1.0/4.0/7.0 lines.

        for lo in line_objs:
            rels = relation_ids(lo)
            connected = [lo]
            # add direct relations and one-hop relations
            for rid in rels:
                obj = by_id.get(rid)
                if obj and obj not in connected:
                    connected.append(obj)
                    for rid2 in relation_ids(obj):
                        obj2 = by_id.get(rid2)
                        if obj2 and obj2 not in connected:
                            connected.append(obj2)
            raw = candidate_text(*connected)
            if not is_wnba_text(raw):
                add_decode("", "", np.nan, {}, False, "not WNBA / wrong sport", "relationship", raw)
                continue
            market = parse_market(raw)
            if market not in MARKETS:
                add_decode("", "", np.nan, {}, False, "unsupported/unknown market", "relationship", raw)
                continue
            line = parse_line(*connected)
            if pd.isna(line):
                add_decode("", market, np.nan, {}, False, "no line found", "relationship", raw)
                continue
            if not status_ok(*connected):
                add_decode("", market, line, {}, False, "suspended/closed", "relationship", raw)
                continue
            candidates = extract_name_candidates(*connected)
            raw_name = candidates[0] if candidates else ""
            team_hint = team_from_text(raw)
            resolved = resolve_player(candidates, raw, team_hint)
            if not resolved.get("Player"):
                add_decode(raw_name, market, line, resolved, False, resolved.get("Reason", "unmatched player"), "relationship", raw)
                continue
            if ACTIVE_TEAMS and resolved.get("Team") and resolved.get("Team") not in ACTIVE_TEAMS:
                add_decode(raw_name, market, line, resolved, False, "matched player not on active schedule team", "relationship", raw)
                continue
            append_row(resolved["Player"], resolved.get("Team", ""), market, line, raw, "relationship")
            add_decode(raw_name, market, line, resolved, True, resolved.get("Reason", "accepted"), "relationship", raw)

        # No recursive option fallback. It was useful for debugging but created bad lines
        # from Higher/Lower alt-option text. Relationship parser above is the only active parser.

        if rows:
            break

    decode_df = pd.DataFrame(decode_rows)
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        decode_df.to_csv(DATA_DIR / "wnba_underdog_decode.csv", index=False)
    except Exception:
        pass

    if rows:
        df = pd.DataFrame(rows)
        df = df[pd.to_numeric(df["Line"], errors="coerce").between(0.5, 80)].copy()
        # One main line per Player+Market. Underdog may include alternate ladders
        # or combo lines for the same player. Real main WNBA props are almost always .5.
        # Pick the LOWEST half-point line per player/market after unsupported combo
        # markets are filtered. This selects 11.5 over 18.5 for PTS, 8.5 over 11.5 for REB, etc.
        df["_line_num"] = pd.to_numeric(df["Line"], errors="coerce")
        df["_is_half"] = df["_line_num"].map(lambda x: 1 if pd.notna(x) and abs((x * 2) - round(x * 2)) < 1e-9 and abs(x - round(x)) > 1e-9 else 0)
        # Penalize whole-number values like 4/5/7 that often come from sort/rank/stat IDs.
        df["_line_priority"] = df["_is_half"] * 1000 - df["_line_num"].clip(0, 80)
        df = df.sort_values(["NameKey", "Market", "_line_priority"], ascending=[True, True, False])
        df = df.drop_duplicates(subset=["NameKey", "Market"], keep="first").drop(columns=["_line_num", "_is_half", "_line_priority"], errors="ignore")
        debug.append({"source":"Underdog", "url":"parser", "status":"ok", "rows":len(df), "message":f"accepted {len(df)} real main-line rows; decode rows {len(decode_df)}"})
        return df.reset_index(drop=True), pd.DataFrame(debug)

    debug.append({"source":"Underdog", "url":"parser", "status":"no accepted rows", "rows":0, "message":f"decode rows {len(decode_df)}; active teams {sorted(ACTIVE_TEAMS) if ACTIVE_TEAMS else 'not detected'}"})
    return pd.DataFrame(columns=["Player", "Team", "Opponent", "Market", "Line", "Source", "Start", "Raw", "Parser Mode", "NameKey"]), pd.DataFrame(debug)

@st.cache_data(ttl=240, show_spinner=False)
def fetch_prizepicks_test_pull():
    """Debug-only PrizePicks probe. It does not feed the board yet."""
    urls = [
        "https://api.prizepicks.com/projections?league_id=7",
        "https://api.prizepicks.com/projections?league_id=3",
        "https://api.prizepicks.com/projections",
    ]
    rows = []
    headers = {"User-Agent":"Mozilla/5.0", "Accept":"application/json,text/plain,*/*"}
    for u in urls:
        try:
            r = requests.get(u, headers=headers, timeout=15)
            msg = (r.text or "")[:220]
            status = f"HTTP {getattr(r,'status_code',0)}"
            try:
                data = r.json()
                objs = flatten_json(data)
                sample = []
                for o in objs[:200]:
                    txt = json.dumps(attrs(o), default=str)[:400]
                    if "wnba" in txt.lower() or any(m in txt.lower() for m in ["points", "rebounds", "assists"]):
                        sample.append(txt)
                    if len(sample) >= 5:
                        break
                rows.append({"url":u, "status":status, "json":"yes", "objects":len(objs), "sample":" || ".join(sample)[:900]})
            except Exception:
                rows.append({"url":u, "status":status, "json":"no", "objects":0, "sample":msg})
        except Exception as e:
            rows.append({"url":u, "status":"request error", "json":"no", "objects":0, "sample":str(e)[:220]})
    return pd.DataFrame(rows)


@st.cache_data(ttl=240, show_spinner=False)
def fetch_sleeper_board():
    rows, debug = [], []
    for url in SLEEPER_URLS:
        data = request_json(url, timeout=15)
        if not data:
            debug.append({"source": "Sleeper", "url": url, "status": "no json/blocked"})
            continue
        objects = flatten_json(data)
        for o in objects:
            blob = json.dumps(o, default=str)
            low = blob.lower()
            if any(x in low for x in ["mlb", "baseball", "nfl", "football", "nhl"]):
                continue
            if not any(x in low for x in ["wnba", "women", "basketball"]):
                continue
            market = infer_market(low)
            if market not in MARKETS:
                continue
            line = line_from_obj(o)
            if pd.isna(line):
                continue
            player = player_from_obj(o)
            if not player:
                a = attrs(o)
                for k in ["player", "athlete", "participant"]:
                    if isinstance(a.get(k), dict):
                        player = player_from_obj(a.get(k))
                if not player:
                    continue
            rows.append({"Player": player, "Team": attrs(o).get("team") or "", "Market": market, "Line": float(line), "Source": "Sleeper", "Start": attrs(o).get("start_time") or "", "Raw": blob[:180]})
        if rows:
            break
    df = pd.DataFrame(rows).drop_duplicates(subset=["Player", "Market", "Line", "Source"]) if rows else pd.DataFrame(columns=["Player", "Team", "Market", "Line", "Source", "Start", "Raw"])
    return df, pd.DataFrame(debug)


def odds_api_market_to_internal(market_key: str) -> Optional[str]:
    rev = {v: k for k, v in ODDS_API_MARKETS.items()}
    return rev.get(str(market_key or ""))


def normalize_line_upload(df: pd.DataFrame, source_name: str = "CSV Upload") -> pd.DataFrame:
    """Normalize uploaded prop-line CSVs into the app line schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Player", "Team", "Market", "Line", "Source", "Start", "Raw", "OverOdds", "UnderOdds"])
    d = df.copy()
    player_col = find_col(d, ["Player", "player", "player_name", "athlete", "athlete_display_name", "Name"], contains_any=["player", "athlete"])
    market_col = find_col(d, ["Market", "market", "stat", "stat_type", "prop", "category"])
    line_col = find_col(d, ["Line", "line", "point", "points", "value", "target_value"])
    team_col = find_col(d, ["Team", "team", "team_abbreviation", "team_name"])
    source_col = find_col(d, ["Source", "source", "book", "sportsbook", "bookmaker"])
    start_col = find_col(d, ["Start", "start", "start_time", "commence_time", "game_time", "date"])
    over_col = find_col(d, ["OverOdds", "over_odds", "over price", "over_price"])
    under_col = find_col(d, ["UnderOdds", "under_odds", "under price", "under_price"])
    if not player_col or not line_col:
        return pd.DataFrame(columns=["Player", "Team", "Market", "Line", "Source", "Start", "Raw", "OverOdds", "UnderOdds"])
    out = pd.DataFrame()
    out["Player"] = d[player_col].astype(str)
    out["Team"] = d[team_col].astype(str) if team_col else ""
    if market_col:
        out["Market"] = d[market_col].astype(str).map(lambda x: infer_market(x) or str(x).upper().strip())
    else:
        out["Market"] = "PTS"
    out["Market"] = out["Market"].map(lambda x: "PRA" if "PRA" in str(x).upper() else str(x).upper())
    out["Line"] = pd.to_numeric(d[line_col], errors="coerce")
    out["Source"] = d[source_col].astype(str) if source_col else source_name
    out["Start"] = d[start_col].astype(str) if start_col else ""
    out["Raw"] = "uploaded line csv"
    out["OverOdds"] = pd.to_numeric(d[over_col], errors="coerce") if over_col else np.nan
    out["UnderOdds"] = pd.to_numeric(d[under_col], errors="coerce") if under_col else np.nan
    out = out.dropna(subset=["Player", "Line"])
    out = out[out["Market"].isin(MARKETS)]
    return out[["Player", "Team", "Market", "Line", "Source", "Start", "Raw", "OverOdds", "UnderOdds"]].copy()


@st.cache_data(ttl=240, show_spinner=False)
def fetch_odds_api_board(api_key: str, regions: str = "us", bookmakers: str = DEFAULT_ODDS_API_BOOKMAKERS, odds_format: str = "american"):
    """Pull WNBA player props from The Odds API when an API key is supplied.
    This is a fallback/secondary source, not a replacement for Underdog/Manual.
    """
    rows, debug = [], []
    api_key = str(api_key or "").strip()
    if not api_key:
        return pd.DataFrame(columns=["Player", "Team", "Market", "Line", "Source", "Start", "Raw", "OverOdds", "UnderOdds"]), pd.DataFrame([{"source": "OddsAPI", "status": "skipped: no ODDS_API_KEY supplied"}])

    events_url = f"{ODDS_API_BASE}/sports/{ODDS_API_SPORT}/events"
    events, status, msg = request_json_with_status(events_url, params={"apiKey": api_key}, timeout=20)
    debug.append({"source": "OddsAPI", "step": "events", "status_code": status, "status": msg, "rows": len(events) if isinstance(events, list) else 0})
    if not isinstance(events, list) or not events:
        return pd.DataFrame(columns=["Player", "Team", "Market", "Line", "Source", "Start", "Raw", "OverOdds", "UnderOdds"]), pd.DataFrame(debug)

    market_keys = ",".join(ODDS_API_MARKETS.values())
    for ev in events[:30]:
        event_id = ev.get("id")
        if not event_id:
            continue
        odds_url = f"{ODDS_API_BASE}/sports/{ODDS_API_SPORT}/events/{event_id}/odds"
        params = {
            "apiKey": api_key,
            "regions": regions,
            "markets": market_keys,
            "oddsFormat": odds_format,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        data, status, msg = request_json_with_status(odds_url, params=params, timeout=25)
        books = data.get("bookmakers", []) if isinstance(data, dict) else []
        debug.append({
            "source": "OddsAPI", "step": "event_odds", "event": f"{ev.get('away_team','')} @ {ev.get('home_team','')}",
            "event_id": event_id, "start": ev.get("commence_time"), "status_code": status, "status": msg,
            "bookmakers": len(books)
        })
        if not books:
            continue
        # Pair over/under outcomes by player-market-book-line.
        paired = {}
        for book in books:
            book_key = book.get("key") or book.get("title") or "book"
            for mkt in book.get("markets", []) or []:
                internal_market = odds_api_market_to_internal(mkt.get("key"))
                if internal_market not in MARKETS:
                    continue
                for out in mkt.get("outcomes", []) or []:
                    name = str(out.get("name") or "").strip()
                    desc = str(out.get("description") or "").strip()
                    player = desc if name.lower() in ["over", "under"] and desc else name
                    if not player or normalize_name(player) in ["over", "under"]:
                        continue
                    line = safe_float(out.get("point"), np.nan)
                    if pd.isna(line):
                        continue
                    side = "Over" if name.lower() == "over" else "Under" if name.lower() == "under" else ""
                    key = (normalize_name(player), player, internal_market, float(line), str(book_key), ev.get("commence_time"))
                    event_raw = f"{ev.get('away_team','')} @ {ev.get('home_team','')}"
                    event_away = _team_key_for_matchup(ev.get('away_team')) if '_team_key_for_matchup' in globals() else str(ev.get('away_team',''))
                    event_home = _team_key_for_matchup(ev.get('home_team')) if '_team_key_for_matchup' in globals() else str(ev.get('home_team',''))
                    rec = paired.setdefault(key, {
                        "Player": player, "Team": "", "Market": internal_market, "Line": float(line),
                        "Source": f"OddsAPI:{book_key}", "Start": ev.get("commence_time") or "",
                        "Raw": event_raw, "EventAway": event_away, "EventHome": event_home,
                        "Matchup": f"{event_away} @ {event_home}" if event_away and event_home else event_raw,
                        "OverOdds": np.nan, "UnderOdds": np.nan
                    })
                    price = safe_float(out.get("price"), np.nan)
                    if side == "Over":
                        rec["OverOdds"] = price
                    elif side == "Under":
                        rec["UnderOdds"] = price
        rows.extend(list(paired.values()))
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["Player", "Team", "Market", "Line", "Source", "Start", "Raw", "OverOdds", "UnderOdds"])
    else:
        df = df.drop_duplicates(subset=["Player", "Market", "Line", "Source", "Start"])
    debug.append({"source": "OddsAPI", "step": "final", "status": "parsed rows", "rows": len(df)})
    return df, pd.DataFrame(debug)


def load_manual_lines():
    data = load_json(MANUAL_LINES_FILE, [])
    return pd.DataFrame(data) if data else pd.DataFrame(columns=["Player", "Team", "Market", "Line", "Source"])


def save_manual_lines(df):
    save_json(MANUAL_LINES_FILE, df.to_dict("records"))


def aggregate_lines(use_ud=True, use_sleeper=False, manual_df=None, use_odds_api=False, odds_api_key: str = "", line_upload_df=None):
    """Aggregate active line sources.

    Current production setup is intentionally simple and stable:
      1) Underdog if it returns WNBA rows
      2) Manual in-app lines saved by the user
      3) Optional uploaded line CSVs

    Sleeper, Odds API, and SportsGameOdds are not called here. Their old functions may remain
    in the file as dormant utilities, but they are disabled from the live board so quota/tier
    errors cannot break the app or clutter the debug screen.
    """
    frames = []
    ud_debug = pd.DataFrame(); manual_debug = []
    if use_ud:
        ud, ud_debug = fetch_underdog_board()
        if ud is not None and not ud.empty:
            frames.append(ud)
    if manual_df is not None and len(manual_df):
        m = manual_df.copy()
        if "Source" not in m.columns:
            m["Source"] = "Manual"
        m["Source"] = m["Source"].fillna("Manual").replace("", "Manual")
        frames.append(m)
        manual_debug.append({"source": "Manual", "status": "loaded saved manual lines", "rows": len(m)})
    if line_upload_df is not None and len(line_upload_df):
        u = line_upload_df.copy()
        if "Source" not in u.columns:
            u["Source"] = "CSV Upload"
        frames.append(u)
        manual_debug.append({"source": "CSV Upload", "status": "loaded uploaded CSV lines", "rows": len(u)})
    if not frames:
        return pd.DataFrame(columns=["Player","Team","Opponent","Market","Line","Source","Start","Raw","OverOdds","UnderOdds","NameKey","Priority"]), ud_debug, pd.DataFrame(manual_debug)
    board = pd.concat(frames, ignore_index=True, sort=False)
    if board.empty:
        return board, ud_debug, pd.DataFrame(manual_debug)
    board["Market"] = board["Market"].astype(str).str.upper().map(lambda x: "PRA" if "PRA" in x else x)
    board = board[board["Market"].isin(MARKETS)].copy()
    board["Line"] = pd.to_numeric(board["Line"], errors="coerce")
    board = board.dropna(subset=["Player", "Market", "Line"])
    for c in ["Team", "Opponent", "HomeAway", "Matchup", "Source", "Start", "Raw", "OverOdds", "UnderOdds"]:
        if c not in board.columns:
            board[c] = "" if c not in ["OverOdds", "UnderOdds"] else np.nan
    board["NameKey"] = board["Player"].map(normalize_name)
    def source_priority(s):
        s = str(s)
        if s == "Underdog": return 1
        if s in ["Manual", "CSV Upload"]: return 2
        return 9
    board["Priority"] = board["Source"].map(source_priority)
    board = board.sort_values(["NameKey", "Market", "Priority", "Line"]).drop_duplicates(subset=["NameKey", "Market", "Source", "Line"], keep="first")
    return board.sort_values(["NameKey", "Market", "Priority"]), ud_debug, pd.DataFrame(manual_debug)

# ============================================================
# Projection engine
# ============================================================
def hit_rates_for_player(logs: pd.DataFrame, name_key: str, market: str, line: float) -> Dict[str, Any]:
    if logs is None or logs.empty or market not in logs.columns or pd.isna(line):
        return {"L5 Hit%": np.nan, "L10 Hit%": np.nan, "L20 Hit%": np.nan, "Season Hit%": np.nan, "Last Values": ""}
    d = logs[logs["NameKey"] == name_key].copy().sort_values("GameDate")
    vals = pd.to_numeric(d[market], errors="coerce").dropna()
    if len(vals) == 0:
        return {"L5 Hit%": np.nan, "L10 Hit%": np.nan, "L20 Hit%": np.nan, "Season Hit%": np.nan, "Last Values": ""}
    def hr(n):
        x = vals.tail(n)
        return round(100 * (x > line).mean(), 1) if len(x) else np.nan
    nearby = {}
    for shift in [-2, -1, 0, 1, 2]:
        nl = line + shift
        if nl > 0:
            nearby[f"Hit% over {nl:g}"] = round(100 * (vals > nl).mean(), 1)
    return {
        "L5 Hit%": hr(5), "L10 Hit%": hr(10), "L20 Hit%": hr(20), "Season Hit%": round(100 * (vals > line).mean(), 1),
        "Last Values": ", ".join([str(round(v, 1)) for v in vals.tail(10).tolist()]),
        **nearby
    }


def learning_adjustment(player, market, base_edge):
    logs = load_json(LEARNING_LOG, [])
    if not logs:
        return 0.0, "No learning sample yet", 50.0
    key = normalize_name(player)
    rows = [r for r in logs if normalize_name(r.get("Player")) == key and str(r.get("Market")) == market and r.get("Result") in ["WIN", "LOSS"]]
    if len(rows) < 3:
        return 0.0, f"Learning sample {len(rows)}", 50.0
    win_rate = sum(1 for r in rows if r.get("Result") == "WIN") / len(rows)
    adj = max(-0.45, min(0.45, (win_rate - 0.52) * 1.0))
    bayes = round(100 * ((sum(1 for r in rows if r.get("Result") == "WIN") + 2) / (len(rows) + 4)), 1)
    return adj, f"Learning {len(rows)} plays / {win_rate:.0%} WR", bayes


def xgboost_blend_projection(features: Dict[str, float], baseline: float) -> Tuple[float, str]:
    # Safe fallback: not enough true training labels yet. This approximates a model blend until learning DB grows.
    # It is intentionally small, so it doesn't distort projections.
    usage = safe_float(features.get("UsageProxy"), np.nan)
    min_safety = safe_float(features.get("MIN_l10"), np.nan)
    team_net = safe_float(features.get("Team_NetRtg"), 0)
    shot_make = safe_float(features.get("ShotMakeRate"), np.nan)
    nudge = 0.0
    if pd.notna(usage):
        nudge += max(-0.35, min(0.35, (usage - 11.0) * 0.035))
    if pd.notna(min_safety):
        nudge += max(-0.25, min(0.25, (min_safety - 26.0) * 0.02))
    if pd.notna(team_net):
        nudge += max(-0.15, min(0.15, team_net * 0.01))
    if pd.notna(shot_make):
        nudge += max(-0.15, min(0.15, (shot_make - 0.42) * 0.6))
    return baseline + nudge, f"XGBoost-style blend nudge {nudge:+.2f}"


def use_xgb_blend_enabled() -> bool:
    """Sidebar-controlled XGBoost/GBM ensemble switch.
    Default True to preserve current projection behavior, but user can turn it off
    while the grading database is still small.
    """
    try:
        return bool(st.session_state.get("use_xgb_blend", True))
    except Exception:
        return True


def monte_carlo(player, market, line, proj, logs, matched_player=""):
    if pd.isna(proj) or pd.isna(line):
        return {"Floor": np.nan, "Median": np.nan, "Ceiling": np.nan, "Over %": np.nan, "Under %": np.nan, "Volatility": "NA"}
    key = normalize_name(matched_player or player)
    vals = logs[logs["NameKey"] == key][market].dropna().astype(float) if logs is not None and not logs.empty and market in logs.columns else pd.Series(dtype=float)
    if len(vals) >= 5:
        sd = max(1.2, float(vals.tail(20).std(ddof=0)))
    else:
        sd = max(1.5, abs(proj) * 0.22)
    rng = np.random.default_rng(stable_seed(player, market, line, round(proj, 2), len(vals)))
    sims = rng.normal(proj, sd, 30000)
    sims = np.clip(sims, 0, None)
    over = float((sims > line).mean() * 100)
    vol = "LOW" if sd < 3 else "MED" if sd < 5.5 else "HIGH"
    return {"Floor": round(np.percentile(sims, 15), 2), "Median": round(np.percentile(sims, 50), 2), "Ceiling": round(np.percentile(sims, 85), 2), "Over %": round(over, 1), "Under %": round(100 - over, 1), "Volatility": vol}


def match_player_base(player: str, base: pd.DataFrame) -> Tuple[Optional[pd.Series], float]:
    if base is None or base.empty:
        return None, 0.0
    candidates = base.copy()
    candidates["_score"] = candidates["Player"].map(lambda x: name_score(player, x))
    top = candidates.sort_values("_score", ascending=False).head(1)
    if top.empty:
        return None, 0.0
    return top.iloc[0], float(top.iloc[0]["_score"])


def pass_reason(data_score, edge, role_conf, minutes_grade, sim_over, sim_under, lean):
    reasons = []
    if data_score < 65:
        reasons.append("data score low")
    if abs(edge) < 0.8:
        reasons.append("edge thin")
    if role_conf < 58:
        reasons.append("role confidence low")
    if minutes_grade in ["D"]:
        reasons.append("minutes safety low")
    if lean == "OVER" and sim_over < 55:
        reasons.append("over sim weak")
    if lean == "UNDER" and sim_under < 55:
        reasons.append("under sim weak")
    return "; ".join(reasons) if reasons else "Meets official gate"



def explain_projection_parts(market: str, baseline: float, minutes_factor: float, usage_factor: float, pace_factor: float, matchup_factor: float, shot_boost: float, lineup_boost: float, learn_adj: float) -> Tuple[str, str, str]:
    parts = []
    parts.append(("Recent/Baseline", baseline - baseline, f"Weighted season + L3/L5/L10/L20 baseline = {baseline:.2f}"))
    parts.append(("Minutes", baseline * (minutes_factor - 1), f"Minutes factor {minutes_factor:.3f}"))
    parts.append(("Usage", baseline * (usage_factor - 1), f"Usage factor {usage_factor:.3f}"))
    parts.append(("Pace", baseline * (pace_factor - 1), f"Pace factor {pace_factor:.3f}"))
    parts.append(("Matchup", baseline * (matchup_factor - 1), f"Team/matchup factor {matchup_factor:.3f}"))
    parts.append(("Shot Profile", shot_boost, f"Shot profile boost {shot_boost:+.2f}"))
    parts.append(("Lineup", lineup_boost, f"Lineup/rotation boost {lineup_boost:+.2f}"))
    parts.append(("Learning", learn_adj, f"Learning/Bayesian nudge {learn_adj:+.2f}"))
    biggest_pos = max(parts[1:], key=lambda x: x[1]) if len(parts) > 1 else parts[0]
    biggest_risk = min(parts[1:], key=lambda x: x[1]) if len(parts) > 1 else parts[0]
    compact = " | ".join([f"{name}: {val:+.2f}" for name, val, _ in parts[1:]])
    return compact, biggest_pos[2], biggest_risk[2]


def shot_profile_boost_for_market(b: pd.Series, market: str) -> Tuple[float, str]:
    three_rate = safe_float(b.get("ThreePARate"), np.nan)
    make_rate = safe_float(b.get("ShotMakeRate"), np.nan)
    rim_rate = safe_float(b.get("RimRate"), np.nan)
    shot_score = safe_float(b.get("ShotProfileScore"), np.nan)
    if market not in ["PTS", "PRA"]:
        return 0.0, "Shot profile is tracked, but not heavily weighted for REB/AST."
    boost = 0.0
    notes = []
    if pd.notna(three_rate):
        boost += max(-0.35, min(0.45, (three_rate - 0.32) * 1.10))
        notes.append(f"3PA rate {three_rate:.2f}")
    if pd.notna(make_rate):
        boost += max(-0.35, min(0.45, (make_rate - 0.42) * 1.25))
        notes.append(f"shot make rate {make_rate:.2f}")
    if pd.notna(rim_rate):
        boost += max(-0.25, min(0.30, (rim_rate - 0.22) * 0.85))
        notes.append(f"rim rate {rim_rate:.2f}")
    if pd.notna(shot_score):
        boost += max(-0.25, min(0.30, (shot_score - 55) / 100))
        notes.append(f"shot profile score {shot_score:.1f}")
    return round(float(boost), 3), "; ".join(notes) if notes else "No shot chart sample yet."


def model_disagreement_label(edge: float, sim_over: float, sim_under: float, lean: str, bayes: float) -> str:
    if pd.isna(edge):
        return "No projection edge"
    sim_side = sim_over if lean == "OVER" else sim_under
    if abs(edge) >= 1.5 and sim_side >= 60 and bayes >= 58:
        return "Models agree"
    if abs(edge) < 0.75 or sim_side < 55:
        return "Model disagreement / thin edge"
    return "Moderate agreement"


def player_similarity_engine(logs: pd.DataFrame, name_key: str, market: str, minutes_proj: float, usage_proxy: float, limit: int = 10) -> Dict[str, Any]:
    """Find the player's closest historical games by minutes/usage-ish profile and return a similarity projection."""
    out = {"Similarity Projection": np.nan, "Similarity Sample": 0, "Similarity Note": "No similarity sample."}
    if logs is None or logs.empty or market not in logs.columns or "NameKey" not in logs.columns:
        return out
    d = logs[logs["NameKey"] == name_key].copy().sort_values("GameDate")
    if d.empty:
        return out
    for c in [market, "MIN", "FGA", "FTA", "TOV"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[market])
    if d.empty:
        return out
    if "MIN" not in d.columns or d["MIN"].isna().all():
        vals = d[market].tail(limit)
        return {"Similarity Projection": round(float(vals.mean()), 2), "Similarity Sample": len(vals), "Similarity Note": f"Recent {len(vals)} game similarity fallback."}
    d["UsageLike"] = d.get("FGA", 0).fillna(0) + 0.44*d.get("FTA", 0).fillna(0) + d.get("TOV", 0).fillna(0)
    usage_ref = usage_proxy if pd.notna(usage_proxy) else d["UsageLike"].tail(10).mean()
    d["SimilarityScore"] = (d["MIN"].fillna(minutes_proj).sub(minutes_proj).abs() * 1.0) + (d["UsageLike"].fillna(usage_ref).sub(usage_ref).abs() * 0.65)
    sims = d.sort_values("SimilarityScore").head(limit)
    if sims.empty:
        return out
    weights = 1 / (1 + sims["SimilarityScore"].fillna(0))
    sim_proj = float(np.average(sims[market], weights=weights)) if weights.sum() else float(sims[market].mean())
    return {"Similarity Projection": round(sim_proj, 2), "Similarity Sample": int(len(sims)), "Similarity Note": f"{len(sims)} closest historical games by minutes/usage."}


def home_away_splits(logs: pd.DataFrame, name_key: str, market: str) -> Dict[str, Any]:
    out = {"Home Avg": np.nan, "Away Avg": np.nan, "HomeAway Note": "Home/away unavailable."}
    if logs is None or logs.empty or market not in logs.columns or "NameKey" not in logs.columns:
        return out
    d = logs[logs["NameKey"] == name_key].copy()
    if d.empty:
        return out
    ha_col = "HomeAway" if "HomeAway" in d.columns else None
    if not ha_col or d[ha_col].isna().all():
        return out
    d[market] = pd.to_numeric(d[market], errors="coerce")
    h = d[d[ha_col].astype(str).str.upper().str.contains("HOME|H", na=False)][market].mean()
    a = d[d[ha_col].astype(str).str.upper().str.contains("AWAY|A|ROAD", na=False)][market].mean()
    if pd.notna(h) or pd.notna(a):
        return {"Home Avg": round(float(h), 2) if pd.notna(h) else np.nan, "Away Avg": round(float(a), 2) if pd.notna(a) else np.nan, "HomeAway Note": f"Home {round(float(h),2) if pd.notna(h) else 'NA'} / Away {round(float(a),2) if pd.notna(a) else 'NA'}"}
    return out


def rest_travel_engine(logs: pd.DataFrame, name_key: str) -> Dict[str, Any]:
    out = {"Rest Days": np.nan, "BackToBack": False, "RestTravel Boost": 0.0, "Travel Note": "No schedule/rest sample."}
    if logs is None or logs.empty or "NameKey" not in logs.columns or "GameDate" not in logs.columns:
        return out
    d = logs[logs["NameKey"] == name_key].copy().sort_values("GameDate")
    d["GameDate"] = pd.to_datetime(d["GameDate"], errors="coerce")
    d = d.dropna(subset=["GameDate"])
    if len(d) < 2:
        return out
    last = d.iloc[-1]["GameDate"]
    prev = d.iloc[-2]["GameDate"]
    rest = max(0, int((last.date() - prev.date()).days))
    b2b = rest <= 1
    boost = -0.25 if b2b else 0.08 if rest >= 3 else 0.0
    return {"Rest Days": rest, "BackToBack": bool(b2b), "RestTravel Boost": boost, "Travel Note": f"Recent rest pattern: {rest} day(s); {'B2B risk' if b2b else 'normal/rested'}."}


def blowout_engine(team_net: float, matchup_strength: float, minutes_proj: float) -> Dict[str, Any]:
    net = 0 if pd.isna(team_net) else float(team_net)
    ms = 50 if pd.isna(matchup_strength) else float(matchup_strength)
    risk = max(0, min(100, 45 + abs(net)*2.2 + abs(ms-50)*0.9))
    tax = -0.35 if risk >= 75 and minutes_proj >= 28 else -0.15 if risk >= 65 else 0.0
    note = "High blowout/minutes tax" if tax <= -0.35 else "Moderate blowout watch" if tax < 0 else "Neutral blowout risk"
    return {"Blowout Risk": round(risk, 1), "Blowout Tax": tax, "Blowout Note": note}


def clutch_minutes_engine(logs: pd.DataFrame, name_key: str, minutes_proj: float) -> Dict[str, Any]:
    boost = 0.0
    note = "No clutch sample."
    if logs is not None and not logs.empty and "NameKey" in logs.columns and "MIN" in logs.columns:
        d = logs[logs["NameKey"] == name_key].copy().sort_values("GameDate")
        mins = pd.to_numeric(d["MIN"], errors="coerce").dropna().tail(10)
        if len(mins):
            stable = float(mins.std(ddof=0)) <= 4.0 and float(mins.mean()) >= 28
            boost = 0.18 if stable else 0.0
            note = "Stable closing-minute profile" if stable else "No clear clutch-minute boost"
    return {"Clutch Minute Boost": boost, "Clutch Note": note}


def bench_rotation_engine(b: pd.Series) -> Dict[str, Any]:
    starter_rate = safe_float(b.get("StarterRate"), np.nan)
    lineup_mentions = safe_float(b.get("LineupMentions"), np.nan)
    role_conf = safe_float(b.get("RoleConfidence"), 50)
    if pd.notna(starter_rate):
        role = "Starter" if starter_rate >= 0.60 else "Bench/variable" if starter_rate <= 0.25 else "Split role"
        boost = 0.25 if starter_rate >= 0.75 else -0.20 if starter_rate <= 0.15 else 0.0
    else:
        role = "Unknown rotation"
        boost = 0.0
    if pd.notna(lineup_mentions) and lineup_mentions >= 20:
        boost += 0.08
    grade = "A" if role_conf >= 78 else "B" if role_conf >= 65 else "C" if role_conf >= 52 else "D"
    return {"Bench Rotation Role": role, "Bench Rotation Boost": round(boost, 3), "Bench Rotation Grade": grade, "Bench Rotation Note": f"{role}; rotation grade {grade}."}


def line_movement_engine(player: str, market: str, current_line: float) -> Dict[str, Any]:
    hist = load_json(LINE_HISTORY_FILE, [])
    rows = [r for r in hist if normalize_name(r.get("Player")) == normalize_name(player) and str(r.get("Market")) == str(market)]
    if not rows or pd.isna(current_line):
        return {"Opening Line": np.nan, "Line Move": 0.0, "Line Movement Note": "No saved opening-line history yet."}
    first = safe_float(rows[0].get("Line"), np.nan)
    if pd.isna(first):
        return {"Opening Line": np.nan, "Line Move": 0.0, "Line Movement Note": "Opening line unavailable."}
    move = float(current_line - first)
    note = f"Opening {first:g} → current {current_line:g} ({move:+.1f})."
    return {"Opening Line": first, "Line Move": round(move, 2), "Line Movement Note": note}


def referee_engine_note() -> Dict[str, Any]:
    return {"Referee Factor": 0.0, "Referee Note": "Neutral until official/referee data is imported; avoids fake ref edges."}


def pace_projection_engine(team_pace: float, matchup_strength: float) -> Dict[str, Any]:
    pace = 78.0 if pd.isna(team_pace) else float(team_pace)
    ms = 50 if pd.isna(matchup_strength) else float(matchup_strength)
    proj_pace = pace + (ms-50)/25
    boost = max(-0.20, min(0.20, (proj_pace - 78)/25))
    return {"Projected Pace": round(proj_pace, 2), "Pace Boost": round(boost, 3), "Pace Projection Note": f"Projected possessions pace proxy {proj_pace:.1f}."}


def tier_grade(score: float, edge: float, sim_side: float, data_score: float) -> str:
    if score >= 88 and abs(edge) >= 2.0 and sim_side >= 64 and data_score >= 80:
        return "Tier 1 — Elite"
    if score >= 80 and abs(edge) >= 1.6 and sim_side >= 60:
        return "Tier 2 — Strong"
    if score >= 72 and abs(edge) >= 1.2:
        return "Tier 3 — Playable"
    if score >= 64 and abs(edge) >= 0.8:
        return "Tier 4 — Lean"
    if score >= 56:
        return "Tier 5 — Track"
    if score >= 48:
        return "Tier 6 — Thin"
    if score >= 40:
        return "Tier 7 — Avoid"
    return "Tier 8 — Pass"


def feature_importance_text(row: Dict[str, Any]) -> str:
    items = []
    for k in ["Projection Explanation", "Shot Profile Note", "Pace Projection Note", "Travel Note", "Bench Rotation Note", "Blowout Note", "Line Movement Note"]:
        v = row.get(k)
        if v not in [None, "", np.nan]:
            items.append(f"{k.replace(' Note','')}: {v}")
    return " || ".join(items[:7])


def correlation_note_for_market(market: str) -> str:
    if market == "AST":
        return "AST props correlate positively with teammates' made shots/PTS environments."
    if market == "PTS":
        return "PTS props can correlate with teammates' AST, pace, and shot-volume profiles."
    if market == "REB":
        return "REB props can be negatively correlated with teammate rebounders and pace/shot-miss volume."
    return "PRA combines scoring, rebounding, and assists; watch same-game correlation risk."


def project_row(row, base, logs):
    player = row["Player"]; market = row["Market"]; line = row["Line"]
    b, score = match_player_base(player, base)
    if b is None or score < 0.76:
        proj = np.nan
        info = {"Data Score": 20, "Projection Note": "No stat baseline match", "Matched Player": "", "Match Score": round(score, 3), "Role Confidence": 0, "Minutes Safety": "NA", "Bayesian Confidence": 50, "Projection Explanation": "No matched player baseline.", "Biggest Positive": "None", "Biggest Risk": "Player name did not match SportsDataverse baseline.", "Shot Profile Boost": 0.0, "Shot Profile Note": "No shot data.", "Confidence Breakdown": "Data 20 | Role 0 | Minutes NA", "Model Agreement": "No model"}
        hit = {"L5 Hit%": np.nan, "L10 Hit%": np.nan, "L20 Hit%": np.nan, "Season Hit%": np.nan, "Last Values": ""}
        return proj, {**info, **hit}

    mavg = safe_float(b.get(f"{market}_avg"), np.nan)
    ml3 = safe_float(b.get(f"{market}_l3"), mavg)
    ml5 = safe_float(b.get(f"{market}_l5"), mavg)
    ml10 = safe_float(b.get(f"{market}_l10"), mavg)
    ml20 = safe_float(b.get(f"{market}_l20"), mavg)
    baseline = 0.30*mavg + 0.15*ml20 + 0.25*ml10 + 0.20*ml5 + 0.10*ml3

    min_avg = safe_float(b.get("MIN_avg"), 0)
    min_l3 = safe_float(b.get("MIN_l3"), min_avg)
    min_l5 = safe_float(b.get("MIN_l5"), min_avg)
    min_l10 = safe_float(b.get("MIN_l10"), min_avg)
    minutes_proj = 0.35*min_avg + 0.30*min_l10 + 0.20*min_l5 + 0.15*min_l3
    minutes_factor = max(0.78, min(1.18, minutes_proj / max(min_avg, 1)))

    usage = safe_float(b.get("USG%"), np.nan)
    usage_proxy = safe_float(b.get("UsageProxy"), np.nan)
    usage_input = usage if pd.notna(usage) else usage_proxy
    usage_factor = 1.0
    if market in ["PTS", "PRA"] and pd.notna(usage_input):
        target = 21 if usage_input > 1.5 and usage_input <= 100 else 11
        usage_factor = max(0.90, min(1.10, usage_input / target))
    elif market == "AST" and pd.notna(safe_float(b.get("AST%"), np.nan)):
        usage_factor = max(0.92, min(1.08, safe_float(b.get("AST%"), 18) / 18))
    elif market == "REB" and pd.notna(safe_float(b.get("TRB%"), np.nan)):
        usage_factor = max(0.92, min(1.08, safe_float(b.get("TRB%"), 10) / 10))

    pace = safe_float(b.get("Team_Pace"), np.nan)
    pace_factor = 1.0 if pd.isna(pace) else max(0.94, min(1.06, pace / np.nanmean([pace, 78])))
    team_net = safe_float(b.get("Team_NetRtg"), 0)
    matchup_strength = safe_float(b.get("TeamMatchupStrengthScore"), np.nan)
    matchup_factor = max(0.94, min(1.06, 1 + team_net/250)) if pd.notna(team_net) else 1.0
    if pd.notna(matchup_strength):
        matchup_factor = max(0.92, min(1.08, matchup_factor + (matchup_strength - 50)/900))

    role_conf = safe_float(b.get("RoleConfidence"), 50)
    minutes_grade = str(b.get("MinutesSafetyGrade", "NA"))
    data_score = safe_float(b.get("DataScore"), 50)
    lineup_cont = safe_float(b.get("LineupContinuityScore"), np.nan)
    lineup_boost = 0.0 if pd.isna(lineup_cont) else max(-0.25, min(0.25, (lineup_cont - 55)/220))
    shot_boost, shot_note = shot_profile_boost_for_market(b, market)

    # Full advanced context layers: player similarity, rest/travel, blowout, clutch, bench rotation, line movement, referee neutral, pace projection.
    sim_ctx = player_similarity_engine(logs, normalize_name(b.get("Player")), market, minutes_proj, usage_proxy)
    rest_ctx = rest_travel_engine(logs, normalize_name(b.get("Player")))
    blow_ctx = blowout_engine(team_net, matchup_strength, minutes_proj)
    clutch_ctx = clutch_minutes_engine(logs, normalize_name(b.get("Player")), minutes_proj)
    bench_ctx = bench_rotation_engine(b)
    # MLB-style fallback lineup role: use confirmed/game roster first, otherwise recent-minute projected rotation.
    fb_role = ""
    fb_note = ""
    try:
        fb = fallback_lineup_rotation_engine(force=False)
        if fb is not None and not fb.empty:
            fb_d = fb[(fb["NameKey"].astype(str) == str(normalize_name(b.get("Player")))) & (fb["Team"].astype(str) == str(_team_key_for_matchup(b.get("Team"))))]
            if not fb_d.empty:
                fr = fb_d.sort_values("FallbackLineupConfidence", ascending=False).iloc[0]
                fb_role = str(fr.get("FallbackLineupRole", ""))
                fb_note = f"{fr.get('FallbackLineupRole','Projected rotation')} via {fr.get('FallbackLineupSource','fallback rotation')}; confidence {safe_float(fr.get('FallbackLineupConfidence'), 0):.0f}/100; projected minutes {safe_float(fr.get('ProjectedMinutes'), np.nan):.1f}."
    except Exception:
        fb_note = "Fallback lineup check unavailable."
    line_ctx = line_movement_engine(player, market, line)
    ref_ctx = referee_engine_note()
    pace_ctx = pace_projection_engine(pace, matchup_strength)
    home_ctx = home_away_splits(logs, normalize_name(b.get("Player")), market)

    learn_adj, learn_note, bayes = learning_adjustment(player, market, baseline - line)
    advanced_nudge = (
        safe_float(rest_ctx.get("RestTravel Boost"), 0) +
        safe_float(blow_ctx.get("Blowout Tax"), 0) +
        safe_float(clutch_ctx.get("Clutch Minute Boost"), 0) +
        safe_float(bench_ctx.get("Bench Rotation Boost"), 0) +
        safe_float(ref_ctx.get("Referee Factor"), 0) +
        safe_float(pace_ctx.get("Pace Boost"), 0)
    )
    sim_proj = safe_float(sim_ctx.get("Similarity Projection"), np.nan)
    pre_ml_raw = baseline * minutes_factor * usage_factor * pace_factor * matchup_factor + shot_boost + lineup_boost + learn_adj + advanced_nudge
    pre_ml = (0.82 * pre_ml_raw + 0.18 * sim_proj) if pd.notna(sim_proj) else pre_ml_raw
    ml_proj, ml_note = xgboost_blend_projection(b.to_dict(), pre_ml)
    proj = ml_proj

    hit = hit_rates_for_player(logs, normalize_name(b.get("Player")), market, line)
    explanation, biggest_pos, biggest_risk = explain_projection_parts(market, baseline, minutes_factor, usage_factor, pace_factor, matchup_factor, shot_boost, lineup_boost, learn_adj)
    confidence_breakdown = f"Data {data_score:.1f} | Role {role_conf:.1f} | Minutes {minutes_grade} | Bayesian {bayes:.1f} | Line {row.get('Source','')}"
    volatility_note = "Low/medium/high comes from recent game standard deviation + Monte Carlo spread."
    info = {
        "Matched Player": b.get("Player"), "Match Score": round(score, 3),
        "MIN Proj": round(minutes_proj, 2), "Usage Proxy": round(usage_proxy, 2) if pd.notna(usage_proxy) else np.nan,
        "USG%": round(usage, 2) if pd.notna(usage) else np.nan,
        "eFG%": round(safe_float(b.get("eFG%"), np.nan), 3), "TS%": round(safe_float(b.get("TS%"), np.nan), 3),
        "Role Confidence": round(role_conf, 1), "Minutes Safety": minutes_grade,
        "Data Score": round(data_score, 1), "Bayesian Confidence": bayes,
        "Team Pace": round(safe_float(b.get("Team_Pace"), np.nan), 2) if pd.notna(safe_float(b.get("Team_Pace"), np.nan)) else np.nan,
        "Team ORtg": round(safe_float(b.get("Team_ORtg"), np.nan), 2) if pd.notna(safe_float(b.get("Team_ORtg"), np.nan)) else np.nan,
        "Team DRtg": round(safe_float(b.get("Team_DRtg"), np.nan), 2) if pd.notna(safe_float(b.get("Team_DRtg"), np.nan)) else np.nan,
        "Team Net": round(team_net, 2) if pd.notna(team_net) else np.nan,
        "Team Matchup Strength": round(matchup_strength, 1) if pd.notna(matchup_strength) else np.nan,
        "Lineup Continuity": round(lineup_cont, 1) if pd.notna(lineup_cont) else np.nan,
        "Shot Profile": round(safe_float(b.get("ShotProfileScore"), np.nan), 1) if pd.notna(safe_float(b.get("ShotProfileScore"), np.nan)) else np.nan,
        "Rim Rate": round(safe_float(b.get("RimRate"), np.nan), 3) if pd.notna(safe_float(b.get("RimRate"), np.nan)) else np.nan,
        "3PA Rate": round(safe_float(b.get("ThreePARate"), np.nan), 3) if pd.notna(safe_float(b.get("ThreePARate"), np.nan)) else np.nan,
        "Shot Make Rate": round(safe_float(b.get("ShotMakeRate"), np.nan), 3) if pd.notna(safe_float(b.get("ShotMakeRate"), np.nan)) else np.nan,
        "Shot Profile Boost": round(shot_boost, 3),
        "Shot Profile Note": shot_note,
        "Team": row.get("Team") or b.get("Team", ""),
        "Opponent": row.get("Opponent", ""),
        "Position": b.get("Position", ""), "PositionGroup": b.get("PositionGroup", "Unknown"),
        "Projection Note": f"{learn_note}; {ml_note}; advanced nudge {advanced_nudge:+.2f}",
        "Projection Explanation": explanation + f" | Advanced Context: {advanced_nudge:+.2f}",
        "Biggest Positive": biggest_pos if advanced_nudge <= 0.2 else f"Advanced context adds {advanced_nudge:+.2f} from rest/pace/bench/clutch layers.",
        "Biggest Risk": biggest_risk if advanced_nudge >= -0.2 else f"Advanced context subtracts {advanced_nudge:+.2f}; review blowout/rest/bench risk.",
        "Confidence Breakdown": confidence_breakdown,
        "Volatility Note": volatility_note,
        "Player Similarity Engine": sim_ctx.get("Similarity Note"),
        "Similarity Projection": sim_ctx.get("Similarity Projection"),
        "Similarity Sample": sim_ctx.get("Similarity Sample"),
        "Defense vs Position": f"Position group: {b.get('PositionGroup', 'Unknown')}; matchup strength {round(matchup_strength,1) if pd.notna(matchup_strength) else 'NA'}",
        "Rest Travel Blowout": f"{rest_ctx.get('Travel Note')} | {blow_ctx.get('Blowout Note')}",
        "Rest Days": rest_ctx.get("Rest Days"),
        "BackToBack": rest_ctx.get("BackToBack"),
        "Travel Note": rest_ctx.get("Travel Note"),
        "Blowout Risk": blow_ctx.get("Blowout Risk"),
        "Blowout Tax": blow_ctx.get("Blowout Tax"),
        "Blowout Note": blow_ctx.get("Blowout Note"),
        "Clutch Minute Boost": clutch_ctx.get("Clutch Minute Boost"),
        "Clutch Note": clutch_ctx.get("Clutch Note"),
        "Bench Rotation Role": bench_ctx.get("Bench Rotation Role"),
        "Bench Rotation Grade": bench_ctx.get("Bench Rotation Grade"),
        "Bench Rotation Note": bench_ctx.get("Bench Rotation Note"),
        "FallbackLineupRole": fb_role or bench_ctx.get("Bench Rotation Role", "Projected Rotation"),
        "Fallback Lineup Note": fb_note or bench_ctx.get("Bench Rotation Note"),
        "Opening Line": line_ctx.get("Opening Line"),
        "Line Move": line_ctx.get("Line Move"),
        "Line Movement Note": line_ctx.get("Line Movement Note"),
        "Referee Factor": ref_ctx.get("Referee Factor"),
        "Referee Note": ref_ctx.get("Referee Note"),
        "Projected Pace": pace_ctx.get("Projected Pace"),
        "Pace Boost": pace_ctx.get("Pace Boost"),
        "Pace Projection Note": pace_ctx.get("Pace Projection Note"),
        "Home Avg": home_ctx.get("Home Avg"),
        "Away Avg": home_ctx.get("Away Avg"),
        "HomeAway Note": home_ctx.get("HomeAway Note"),
        "Correlation Note": correlation_note_for_market(market),
    }
    return proj, {**info, **hit}

def make_projection_board(lines, logs, base):
    if lines is None or lines.empty:
        return pd.DataFrame()
    if base is None or base.empty:
        base = compute_player_baselines(logs, load_dataset("player_season_stats"), load_dataset("shots"), load_dataset("rosters"))
    active = []
    for (namekey, market), grp in lines.groupby(["NameKey", "Market"]):
        grp = grp.sort_values("Priority")
        primary = grp.iloc[0].copy()
        primary["Underdog Line"] = safe_float(grp[grp["Source"] == "Underdog"]["Line"].iloc[0], np.nan) if len(grp[grp["Source"] == "Underdog"]) else np.nan
        primary["Sleeper Line"] = np.nan
        primary["Manual Line"] = safe_float(grp[grp["Source"] == "Manual"]["Line"].iloc[0], np.nan) if len(grp[grp["Source"] == "Manual"]) else np.nan
        primary["Best Over Line"] = grp["Line"].min()
        primary["Best Under Line"] = grp["Line"].max()
        primary["Line Source Reliability"] = {"Underdog": 95, "Manual": 70, "CSV Upload": 68}.get(str(primary.get("Source")), 50)
        active.append(primary)
    board = pd.DataFrame(active)
    rows = []
    for _, r in board.iterrows():
        proj, info = project_row(r, base, logs)
        sim = monte_carlo(r["Player"], r["Market"], r["Line"], proj, logs, info.get("Matched Player", ""))
        edge = proj - r["Line"] if pd.notna(proj) else np.nan
        lean = "OVER" if pd.notna(edge) and edge > 0 else "UNDER" if pd.notna(edge) else "PASS"
        sim_side = sim.get("Over %", 0) if lean == "OVER" else sim.get("Under %", 0)
        official_score = 0
        if pd.notna(edge):
            official_score = (
                min(30, abs(edge)*12) +
                min(25, max(0, sim_side - 50)*2.5) +
                min(20, safe_float(info.get("Data Score"), 0)*0.2) +
                min(15, safe_float(info.get("Role Confidence"), 0)*0.15) +
                min(10, safe_float(info.get("Line Source Reliability", r.get("Line Source Reliability", 50)), 50)*0.1)
            )
        official_score = round(max(0, min(100, official_score)), 1)
        reason = pass_reason(safe_float(info.get("Data Score"), 0), edge if pd.notna(edge) else 0, safe_float(info.get("Role Confidence"), 0), info.get("Minutes Safety", "NA"), sim.get("Over %", 0), sim.get("Under %", 0), lean)
        official = "PASS"
        if reason == "Meets official gate" and official_score >= 62 and abs(edge) >= 0.8:
            official = "🔥 OVER" if lean == "OVER" else "⚠️ UNDER"
        model_agreement = model_disagreement_label(edge if pd.notna(edge) else np.nan, sim.get("Over %", np.nan), sim.get("Under %", np.nan), lean, safe_float(info.get("Bayesian Confidence"), 50))
        sim_side_for_tier = sim.get("Over %", 0) if lean == "OVER" else sim.get("Under %", 0)
        tier = tier_grade(official_score, edge if pd.notna(edge) else 0, sim_side_for_tier, safe_float(info.get("Data Score"), 0))
        row_payload = {**r.to_dict(), **info, **sim, "Projection": round(proj, 2) if pd.notna(proj) else np.nan, "Edge": round(edge, 2) if pd.notna(edge) else np.nan, "Lean": lean, "Official Play Score": official_score, "PASS Reason": reason, "Official": official, "Model Agreement": model_agreement, "Tier": tier}
        row_payload["Feature Importance"] = feature_importance_text(row_payload)
        rows.append(row_payload)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Official", "Official Play Score", "Edge"], ascending=[True, False, False])
    return out



# ============================================================
# Advanced Production Engines v2.2
# Full implementations: XGBoost training, similarity, referee,
# travel, injury ripple, line movement/CLV, feature importance,
# model disagreement, backtesting, EV/Kelly, opponent lineup.
# ============================================================
TEAM_COORDS = {
    "ATL": (33.7490, -84.3880), "CHI": (41.8781, -87.6298), "CON": (41.4918, -72.0912),
    "DAL": (32.7767, -96.7970), "IND": (39.7684, -86.1581), "LA": (34.0522, -118.2437),
    "LAS": (36.1699, -115.1398), "LV": (36.1699, -115.1398), "MIN": (44.9778, -93.2650),
    "NY": (40.7128, -74.0060), "PHX": (33.4484, -112.0740), "SEA": (47.6062, -122.3321),
    "WAS": (38.9072, -77.0369), "GS": (37.7749, -122.4194), "TOR": (43.6532, -79.3832),
}
TEAM_ALIASES = {
    "ATLANTA DREAM":"ATL", "CHICAGO SKY":"CHI", "CONNECTICUT SUN":"CON", "DALLAS WINGS":"DAL",
    "INDIANA FEVER":"IND", "LOS ANGELES SPARKS":"LA", "LAS VEGAS ACES":"LV", "MINNESOTA LYNX":"MIN",
    "NEW YORK LIBERTY":"NY", "PHOENIX MERCURY":"PHX", "SEATTLE STORM":"SEA", "WASHINGTON MYSTICS":"WAS",
    "GOLDEN STATE VALKYRIES":"GS", "TORONTO TEMPO":"TOR"
}
REFEREE_FILE = LOCAL_DIR / "wnba_referee_tendencies.csv"
INJURY_STATUS_FILE = LOCAL_DIR / "wnba_injury_status.json"
MODEL_DIR = LOCAL_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
BACKTEST_FILE = DATA_DIR / "wnba_backtest_results.csv"


def team_abbrev(x) -> str:
    s = str(x or "").strip().upper()
    if s in TEAM_COORDS:
        return s
    s2 = re.sub(r"[^A-Z ]+", "", s).strip()
    return TEAM_ALIASES.get(s2, s[:3])


def haversine_miles(a, b):
    if not a or not b:
        return np.nan
    lat1, lon1 = map(math.radians, a); lat2, lon2 = map(math.radians, b)
    dlat = lat2-lat1; dlon = lon2-lon1
    q = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 3958.8 * 2 * math.asin(min(1, math.sqrt(q)))


def latest_travel_context(logs: pd.DataFrame, name_key: str, team: str = "") -> Dict[str, Any]:
    out = {"Travel Miles": np.nan, "Travel Tax": 0.0, "Travel Engine Note": "Travel unavailable."}
    if logs is None or logs.empty or "NameKey" not in logs.columns or "GameDate" not in logs.columns:
        return out
    d = logs[logs["NameKey"] == name_key].copy().sort_values("GameDate")
    if len(d) < 2:
        return out
    cur_team = team_abbrev(team or d.iloc[-1].get("Team"))
    prev_team = team_abbrev(d.iloc[-2].get("Team") or cur_team)
    miles = haversine_miles(TEAM_COORDS.get(prev_team), TEAM_COORDS.get(cur_team))
    if pd.isna(miles):
        return out
    tax = -0.18 if miles >= 1500 else -0.10 if miles >= 900 else -0.04 if miles >= 400 else 0.0
    return {"Travel Miles": round(float(miles), 1), "Travel Tax": tax, "Travel Engine Note": f"Estimated travel {miles:.0f} miles; tax {tax:+.2f}."}


def referee_tendency_engine(row: Dict[str, Any]) -> Dict[str, Any]:
    if not REFEREE_FILE.exists():
        return {"Referee Factor": 0.0, "Referee Note": "No referee CSV loaded; neutral."}
    try:
        refs = pd.read_csv(REFEREE_FILE)
    except Exception:
        return {"Referee Factor": 0.0, "Referee Note": "Referee CSV unreadable; neutral."}
    if refs.empty:
        return {"Referee Factor": 0.0, "Referee Note": "Referee table empty; neutral."}
    market = str(row.get("Market", ""))
    # Accept either market-specific columns or generic FTA/Foul/Total columns.
    factor = 0.0; notes=[]
    for col, weight in [("FTA_Index", .35), ("Foul_Index", .25), ("Pace_Index", .20), ("Points_Index", .20)]:
        if col in refs.columns:
            val = pd.to_numeric(refs[col], errors="coerce").dropna().mean()
            if pd.notna(val):
                factor += max(-0.20, min(0.20, (float(val)-100)/100))*weight
                notes.append(f"{col} {val:.1f}")
    if market in ["REB"]:
        factor *= .35
    if market in ["AST"]:
        factor *= .55
    return {"Referee Factor": round(float(factor), 3), "Referee Note": "; ".join(notes) if notes else "Referee file loaded, no usable index columns."}


def injury_ripple_engine(row: Dict[str, Any], base_row: pd.Series) -> Dict[str, Any]:
    bumps = load_json(INJURY_BUMPS_FILE, [])
    status = load_json(INJURY_STATUS_FILE, [])
    player = row.get("Player", ""); team = str(row.get("Team") or base_row.get("Team") or ""); market = row.get("Market", "")
    active_out = {normalize_name(x.get("Player")): x for x in status if str(x.get("Status", "")).upper() in ["OUT", "DOUBTFUL", "INACTIVE"]}
    total_usage = 0.0; total_min = 0.0; notes=[]
    for b in bumps:
        if normalize_name(b.get("Player")) not in ["", normalize_name(player)] and normalize_name(b.get("Player")) != normalize_name(player):
            continue
        if str(b.get("Team", "")).strip() and team and str(b.get("Team", "")).strip().upper() != team.upper():
            continue
        bm = str(b.get("Market", "ALL")).upper()
        if bm not in ["ALL", market]:
            continue
        teammate = normalize_name(b.get("Teammate Out"))
        # If no status table is loaded, a bump row still acts as manual active bump.
        if teammate and active_out and teammate not in active_out:
            continue
        ub = safe_float(b.get("Usage Bump %"), 0)/100.0
        mb = safe_float(b.get("Minutes Bump"), 0)
        total_usage += ub; total_min += mb
        notes.append(f"{b.get('Teammate Out','manual')} usage {ub:+.1%}, min {mb:+.1f}")
    proj_bump = 0.0
    if market in ["PTS", "PRA"]:
        proj_bump += total_usage * max(8.0, safe_float(base_row.get("UsageProxy"), 10)) * 0.25
    if total_min:
        ppm = safe_float(base_row.get(f"{market}_per_min"), np.nan)
        if pd.notna(ppm): proj_bump += total_min * ppm
    return {"Injury Ripple Bump": round(float(proj_bump), 3), "Injury Ripple Note": "; ".join(notes) if notes else "No active injury ripple bump."}


def opponent_lineup_adjustment(row: Dict[str, Any], base_row: pd.Series) -> Dict[str, Any]:
    gr = load_dataset("game_rosters")
    market = str(row.get("Market", "")); opp = str(row.get("Opponent", ""))
    if gr.empty or not opp:
        return {"Opponent Lineup Adj": 0.0, "Opponent Lineup Note": "Opponent lineup unavailable."}
    d = gr.copy()
    if "Team" not in d.columns:
        return {"Opponent Lineup Adj": 0.0, "Opponent Lineup Note": "Opponent lineup unavailable."}
    od = d[d["Team"].astype(str).str.upper().map(team_abbrev) == team_abbrev(opp)].copy()
    if od.empty:
        return {"Opponent Lineup Adj": 0.0, "Opponent Lineup Note": "No matched opponent active roster."}
    # Defensive size/position proxy from active opponent roster.
    bigs = od.get("PositionGroup", pd.Series(dtype=str)).astype(str).str.contains("Big", na=False).mean() if "PositionGroup" in od.columns else np.nan
    guards = od.get("PositionGroup", pd.Series(dtype=str)).astype(str).str.contains("Guard", na=False).mean() if "PositionGroup" in od.columns else np.nan
    adj = 0.0
    if market == "REB" and pd.notna(bigs): adj += max(-0.20, min(0.20, (0.30 - bigs) * 0.7))
    if market == "AST" and pd.notna(guards): adj += max(-0.15, min(0.15, (guards - 0.35) * 0.35))
    if market == "PTS" and pd.notna(bigs): adj += max(-0.12, min(0.12, (0.28 - bigs) * 0.4))
    return {"Opponent Lineup Adj": round(float(adj), 3), "Opponent Lineup Note": f"Opponent roster mix: Big {bigs:.0%} / Guard {guards:.0%}" if pd.notna(bigs) or pd.notna(guards) else "Opponent position mix unavailable."}


def build_training_frame_from_logs(logs: pd.DataFrame, market: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if logs is None or logs.empty or market not in logs.columns:
        return pd.DataFrame(), pd.Series(dtype=float), []
    d = logs.copy().sort_values(["NameKey", "GameDate"])
    needed = [market, "MIN", "PTS", "REB", "AST", "PRA", "FGA", "FTA", "TOV"]
    for c in needed:
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
    feats=[]
    for key, g in d.groupby("NameKey"):
        g=g.sort_values("GameDate").copy()
        for c in [market, "MIN", "FGA", "FTA", "TOV"]:
            if c not in g.columns: g[c]=0
        g["roll3"] = g[market].shift(1).rolling(3, min_periods=1).mean()
        g["roll5"] = g[market].shift(1).rolling(5, min_periods=1).mean()
        g["roll10"] = g[market].shift(1).rolling(10, min_periods=1).mean()
        g["min_roll5"] = g["MIN"].shift(1).rolling(5, min_periods=1).mean()
        g["usage_roll5"] = (g["FGA"].fillna(0)+0.44*g["FTA"].fillna(0)+g["TOV"].fillna(0)).shift(1).rolling(5, min_periods=1).mean()
        g["home_flag"] = g.get("HomeAway", "").astype(str).str.upper().str.contains("HOME|H", na=False).astype(int) if "HomeAway" in g.columns else 0
        g["games_seen"] = np.arange(len(g))
        feats.append(g)
    if not feats: return pd.DataFrame(), pd.Series(dtype=float), []
    dd=pd.concat(feats, ignore_index=True)
    feature_cols=["roll3","roll5","roll10","min_roll5","usage_roll5","home_flag","games_seen"]
    dd=dd.dropna(subset=[market,"roll3","roll5","roll10"])
    if len(dd) < 50:
        return pd.DataFrame(), pd.Series(dtype=float), feature_cols
    X=dd[feature_cols].replace([np.inf,-np.inf],np.nan).fillna(0)
    y=pd.to_numeric(dd[market], errors="coerce")
    return X, y, feature_cols


@st.cache_resource(show_spinner=False)
def train_market_model_cached(market: str, logs_csv_signature: str):
    logs = load_dataset("player_game_logs")
    X, y, feature_cols = build_training_frame_from_logs(logs, market)
    if X.empty or len(y) < 50:
        return None, feature_cols, pd.DataFrame(), "Not enough historical rows to train."
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=220, max_depth=3, learning_rate=0.035, subsample=0.9, colsample_bytree=0.9, random_state=42, objective="reg:squarederror")
        model.fit(X, y)
        imp = pd.DataFrame({"Feature": feature_cols, "Importance": getattr(model, "feature_importances_", np.zeros(len(feature_cols)))})
        return model, feature_cols, imp.sort_values("Importance", ascending=False), "XGBoost trained."
    except Exception as e:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, max_leaf_nodes=16, random_state=42)
            model.fit(X, y)
            imp = pd.DataFrame({"Feature": feature_cols, "Importance": np.nan})
            return model, feature_cols, imp, f"Sklearn gradient boosting trained; XGBoost unavailable: {str(e)[:80]}"
        except Exception as e2:
            return None, feature_cols, pd.DataFrame(), f"Model training failed: {str(e2)[:120]}"


def current_model_features_for_row(row: Dict[str, Any]) -> Dict[str, float]:
    return {
        "roll3": safe_float(row.get("Similarity Projection"), safe_float(row.get("Projection"), 0)),
        "roll5": safe_float(row.get("L5 Avg"), safe_float(row.get("Projection"), 0)),
        "roll10": safe_float(row.get("L10 Avg"), safe_float(row.get("Projection"), 0)),
        "min_roll5": safe_float(row.get("MIN Proj"), 0),
        "usage_roll5": safe_float(row.get("Usage Proxy"), 0),
        "home_flag": 1 if str(row.get("HomeAway", "")).upper().startswith("H") else 0,
        "games_seen": safe_float(row.get("Games", 20), 20),
    }


def model_prediction_for_row(row: Dict[str, Any]) -> Dict[str, Any]:
    market = str(row.get("Market", ""))
    logs_path = CACHE_FILES.get("player_game_logs")
    sig = str(logs_path.stat().st_mtime) if logs_path and logs_path.exists() else "none"
    model, cols, imp, note = train_market_model_cached(market, sig)
    if model is None or not cols:
        return {"XGBoost Projection": np.nan, "XGBoost Note": note, "XGBoost Feature Importance": "Unavailable"}
    feat = current_model_features_for_row(row)
    X = pd.DataFrame([{c: feat.get(c, 0) for c in cols}])
    try:
        pred = float(model.predict(X)[0])
    except Exception:
        return {"XGBoost Projection": np.nan, "XGBoost Note": "Prediction failed", "XGBoost Feature Importance": "Unavailable"}
    imp_text = ", ".join([f"{r.Feature}:{r.Importance:.3f}" if pd.notna(r.Importance) else f"{r.Feature}" for _, r in imp.head(6).iterrows()]) if not imp.empty else "No importances"
    return {"XGBoost Projection": round(pred, 2), "XGBoost Note": note, "XGBoost Feature Importance": imp_text}


def implied_prob_from_odds(american_odds: float = -110) -> float:
    o = safe_float(american_odds, -110)
    if o < 0: return abs(o)/(abs(o)+100)
    return 100/(o+100)


def ev_kelly_engine(prob_pct: float, american_odds: float = -110) -> Dict[str, Any]:
    p = max(0, min(1, safe_float(prob_pct, 0)/100))
    o = safe_float(american_odds, -110)
    dec = 1 + (100/abs(o) if o < 0 else o/100)
    b = dec - 1
    ev = p*b - (1-p)
    kelly = max(0.0, min(0.08, (p*b - (1-p))/b)) if b > 0 else 0.0
    return {"Break Even %": round(implied_prob_from_odds(o)*100, 1), "EV %": round(ev*100, 2), "Kelly %": round(kelly*100, 2), "Odds Used": o}


def clv_engine(player: str, market: str, current_line: float, saved_line: float = np.nan) -> Dict[str, Any]:
    hist = load_json(LINE_HISTORY_FILE, [])
    rows = [r for r in hist if normalize_name(r.get("Player")) == normalize_name(player) and str(r.get("Market")) == str(market)]
    if pd.isna(saved_line) and rows:
        saved_line = safe_float(rows[0].get("Line"), np.nan)
    if pd.isna(saved_line) or pd.isna(current_line):
        return {"CLV": np.nan, "CLV Note": "No saved/opening line for CLV."}
    clv = safe_float(saved_line) - safe_float(current_line)
    return {"CLV": round(clv, 2), "CLV Note": f"Saved/open {saved_line:g} vs current {current_line:g}: CLV {clv:+.1f}"}


def sharp_money_detector(line_move: float, edge: float, lean: str) -> str:
    if pd.isna(line_move): return "No movement sample"
    if lean == "OVER" and line_move > 0: return "Market moved against Over (worse line)"
    if lean == "OVER" and line_move < 0: return "Reverse/value: Over line improved"
    if lean == "UNDER" and line_move > 0: return "Value: Under line improved"
    if lean == "UNDER" and line_move < 0: return "Market moved against Under"
    return "Neutral movement"


def model_disagreement_full(row: Dict[str, Any]) -> Dict[str, Any]:
    vals=[]; names=[]
    for k in ["Projection", "XGBoost Projection", "Similarity Projection", "Median"]:
        v=safe_float(row.get(k), np.nan)
        if pd.notna(v): vals.append(v); names.append(k)
    if len(vals) < 2:
        return {"Model Disagreement Score": np.nan, "Model Disagreement Note": "Not enough model outputs."}
    spread=float(np.max(vals)-np.min(vals))
    note="Low disagreement" if spread < 1.2 else "Moderate disagreement" if spread < 2.4 else "High disagreement - review manually"
    return {"Model Disagreement Score": round(spread,2), "Model Disagreement Note": note + " (" + ", ".join(names) + ")"}


def auto_backtest_engine(max_rows: int = 2500) -> pd.DataFrame:
    logs = load_dataset("player_game_logs")
    if logs.empty:
        return pd.DataFrame()
    rows=[]
    d=logs.sort_values(["NameKey","GameDate"]).copy()
    for market in MARKETS:
        if market not in d.columns: continue
        for nk,g in d.groupby("NameKey"):
            g=g.sort_values("GameDate").copy()
            vals=pd.to_numeric(g[market], errors="coerce")
            pred=vals.shift(1).rolling(10, min_periods=3).mean()*0.55 + vals.shift(1).rolling(5, min_periods=3).mean()*0.45
            for idx, r in g.assign(Pred=pred).dropna(subset=[market,"Pred"]).tail(50).iterrows():
                # Synthetic historical line = recent rolling median rounded to .5, used only for model QA when no historical book lines exist.
                line=round(float(r["Pred"])*2)/2
                lean="OVER" if r["Pred"]>line else "UNDER"
                actual=float(r[market]); hit=(actual>line) if lean=="OVER" else (actual<line)
                rows.append({"Player":r.get("Player"), "Market":market, "GameDate":r.get("GameDate"), "Projection":round(float(r["Pred"]),2), "Synthetic Line":line, "Lean":lean, "Actual":actual, "Result":"WIN" if hit else "LOSS"})
                if len(rows)>=max_rows: break
            if len(rows)>=max_rows: break
        if len(rows)>=max_rows: break
    bt=pd.DataFrame(rows)
    if not bt.empty: bt.to_csv(BACKTEST_FILE, index=False)
    return bt




# ============================================================
# Full Game Context Engine v2.0
# Adds matchup-aware team/offense/defense/pace/rest/home-away/rotation/injury context.
# This layer does NOT touch the working Underdog parser; it enriches rows after lines match.
# ============================================================
OFFICIAL_WNBA_TEAM_CONTEXT_FILE = DATA_DIR / "wnba_official_team_context.csv"
GAME_CONTEXT_FILE = DATA_DIR / "wnba_game_context_today.csv"
DAILY_TEAM_CONTEXT_FILE = DATA_DIR / "wnba_daily_team_context_cache_v2.csv"


def _wnba_stats_headers() -> Dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.wnba.com",
        "Referer": "https://www.wnba.com/stats/",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
    }


def _wnba_stats_get(path: str, params: Dict[str, Any], timeout: int = 12) -> Tuple[pd.DataFrame, str]:
    """Read official WNBA Stats JSON endpoints when Streamlit has network access.
    Fails quietly and returns an empty frame so projections still run from cached SportsDataverse.
    """
    url = f"https://stats.wnba.com/stats/{path.lstrip('/')}"
    try:
        r = requests.get(url, params=params, headers=_wnba_stats_headers(), timeout=timeout)
        if r.status_code != 200:
            return pd.DataFrame(), f"WNBA stats {path} HTTP {r.status_code}"
        js = r.json()
        rs = None
        if isinstance(js, dict):
            sets = js.get("resultSets") or js.get("resultSet") or []
            if isinstance(sets, list) and sets:
                rs = sets[0]
            elif isinstance(sets, dict):
                rs = sets
        if not rs:
            return pd.DataFrame(), f"WNBA stats {path}: no resultSets"
        headers = rs.get("headers") or rs.get("Headers") or []
        rows = rs.get("rowSet") or rs.get("RowSet") or []
        if not headers or not rows:
            return pd.DataFrame(), f"WNBA stats {path}: empty rows"
        return pd.DataFrame(rows, columns=headers), f"WNBA stats {path}: ok {len(rows)} rows"
    except Exception as e:
        return pd.DataFrame(), f"WNBA stats {path}: {str(e)[:120]}"


def _official_team_stats_params(season: int, measure: str) -> Dict[str, Any]:
    # These parameters mirror the official stats dashboard style used by wnba.com/stats.
    return {
        "Conference": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "GameScope": "",
        "GameSegment": "",
        "LastNGames": "0",
        "LeagueID": "10",
        "Location": "",
        "MeasureType": measure,
        "Month": "0",
        "OpponentTeamID": "0",
        "Outcome": "",
        "PORound": "0",
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": "0",
        "PlusMinus": "N",
        "Rank": "N",
        "Season": str(season),
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "ShotClockRange": "",
        "StarterBench": "",
        "TeamID": "0",
        "TwoWay": "0",
        "VsConference": "",
        "VsDivision": "",
    }


def refresh_official_wnba_team_context(season: Optional[int] = None, force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pull official WNBA team base/advanced/four-factor context and cache it.
    Used as a richer source for opponent defense, pace, rebounding, assists, and efficiency.
    """
    if season is None:
        try:
            season = int(datetime.utcnow().year)
        except Exception:
            season = 2026
    if OFFICIAL_WNBA_TEAM_CONTEXT_FILE.exists() and not force:
        try:
            cached = pd.read_csv(OFFICIAL_WNBA_TEAM_CONTEXT_FILE)
            if not cached.empty:
                return cached, pd.DataFrame([{"Step":"official_wnba_team_context", "Status":"cached", "Rows":len(cached)}])
        except Exception:
            pass
    dbg = []
    base, msg = _wnba_stats_get("leaguedashteamstats", _official_team_stats_params(season, "Base")); dbg.append({"Step":"Base", "Status":msg, "Rows":len(base)})
    adv, msg = _wnba_stats_get("leaguedashteamstats", _official_team_stats_params(season, "Advanced")); dbg.append({"Step":"Advanced", "Status":msg, "Rows":len(adv)})
    four, msg = _wnba_stats_get("leaguedashteamstats", _official_team_stats_params(season, "Four Factors")); dbg.append({"Step":"Four Factors", "Status":msg, "Rows":len(four)})

    frames = []
    for df, tag in [(base, "Base"), (adv, "Adv"), (four, "Four")]:
        if df is None or df.empty:
            continue
        d = df.copy()
        team_col = find_col(d, ["TEAM_ABBREVIATION", "TEAM_NAME", "TEAM", "TEAM_CITY"])
        if not team_col:
            continue
        d["Team"] = d[team_col].map(_team_key_for_matchup)
        keep = ["Team"]
        for c in d.columns:
            cu = str(c).upper()
            if cu in ["PACE", "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "OREB_PCT", "DREB_PCT", "REB_PCT", "EFG_PCT", "TS_PCT", "PTS", "REB", "OREB", "DREB", "AST", "TOV", "PLUS_MINUS", "FG_PCT", "FG3_PCT", "FTA_RATE", "TM_TOV_PCT"]:
                keep.append(c)
        d = d[[c for c in keep if c in d.columns]].drop_duplicates("Team")
        d = d.rename(columns={c: f"{tag}_{c}" for c in d.columns if c != "Team"})
        frames.append(d)
    if not frames:
        fallback = _build_team_context_from_cached_sources()
        return fallback, pd.DataFrame(dbg + [{"Step":"fallback", "Status":"official pull empty; using cached sources", "Rows":len(fallback)}])
    out = frames[0]
    for d in frames[1:]:
        out = out.merge(d, on="Team", how="outer")

    # Canonical fields used by the projection layer.
    def first_col(cols):
        for c in cols:
            if c in out.columns:
                return pd.to_numeric(out[c], errors="coerce")
        return pd.Series(np.nan, index=out.index)
    out["Team_Pace_Official"] = first_col(["Adv_PACE", "Base_PACE", "Four_PACE"])
    out["Team_ORtg_Official"] = first_col(["Adv_OFF_RATING"])
    out["Team_DRtg_Official"] = first_col(["Adv_DEF_RATING"])
    out["Team_NetRtg_Official"] = first_col(["Adv_NET_RATING"])
    out["Team_PTS_Official"] = first_col(["Base_PTS"])
    out["Team_REB_Official"] = first_col(["Base_REB"])
    out["Team_AST_Official"] = first_col(["Base_AST"])
    out["Team_OREB_Official"] = first_col(["Base_OREB"])
    out["Team_DREB_Official"] = first_col(["Base_DREB"])
    out["Team_eFG_Official"] = first_col(["Four_EFG_PCT", "Adv_EFG_PCT"])
    out["Team_TS_Official"] = first_col(["Adv_TS_PCT"])
    out["Team_OREB_PCT_Official"] = first_col(["Four_OREB_PCT", "Adv_OREB_PCT"])
    out["Team_DREB_PCT_Official"] = first_col(["Four_DREB_PCT", "Adv_DREB_PCT"])
    out["Season"] = season
    try:
        OFFICIAL_WNBA_TEAM_CONTEXT_FILE.parent.mkdir(exist_ok=True)
        out.to_csv(OFFICIAL_WNBA_TEAM_CONTEXT_FILE, index=False)
    except Exception:
        pass
    return out, pd.DataFrame(dbg + [{"Step":"official_wnba_team_context", "Status":"saved", "Rows":len(out)}])


def _build_team_context_from_cached_sources() -> pd.DataFrame:
    """Fallback/team context from SportsDataverse-built team_ranks/team season stats."""
    tr = load_dataset("team_ranks")
    ts = load_dataset("team_season_stats")
    frames=[]
    for df, tag in [(tr, "Rank"), (ts, "Season")]:
        if df is None or df.empty:
            continue
        d=df.copy()
        team_col = find_col(d, ["Team", "team", "TEAM", "team_abbreviation", "team_name", "TEAM_NAME"])
        if not team_col:
            continue
        d["Team"] = d[team_col].map(_team_key_for_matchup)
        keep=["Team"]
        for c in d.columns:
            cu=str(c).upper().replace(" ", "_")
            if any(x in cu for x in ["PACE", "ORTG", "OFF_RATING", "DRTG", "DEF_RATING", "NET", "PTS", "POINTS", "REB", "OREB", "DREB", "AST", "TOV", "EFG", "TS", "RANK", "ALLOWED"]):
                keep.append(c)
        d=d[[c for c in keep if c in d.columns]].drop_duplicates("Team")
        d=d.rename(columns={c:f"{tag}_{c}" for c in d.columns if c!="Team"})
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["Team"])
    out=frames[0]
    for d in frames[1:]:
        out=out.merge(d,on="Team",how="outer")
    # Canonical fallback fields.
    def pick_contains(names):
        for c in out.columns:
            cu=str(c).upper().replace(" ", "_")
            if any(n in cu for n in names):
                return pd.to_numeric(out[c], errors="coerce")
        return pd.Series(np.nan, index=out.index)
    out["Team_Pace_Official"] = pick_contains(["TEAM_PACE", "PACE"])
    out["Team_ORtg_Official"] = pick_contains(["TEAM_ORTG", "ORTG", "OFF_RATING"])
    out["Team_DRtg_Official"] = pick_contains(["TEAM_DRTG", "DRTG", "DEF_RATING"])
    out["Team_NetRtg_Official"] = pick_contains(["TEAM_NETRTG", "NET_RATING", "NETRTG"])
    out["Team_PTS_Official"] = pick_contains(["PTS", "POINTS"])
    out["Team_REB_Official"] = pick_contains(["REB"])
    out["Team_AST_Official"] = pick_contains(["AST"])
    out["Team_OREB_Official"] = pick_contains(["OREB"])
    out["Team_DREB_Official"] = pick_contains(["DREB"])
    out["Team_eFG_Official"] = pick_contains(["EFG"])
    out["Team_TS_Official"] = pick_contains(["TS"])
    return out


def _team_context_table(force_official: bool = False) -> pd.DataFrame:
    season = int(datetime.utcnow().year)
    official, dbg = refresh_official_wnba_team_context(season, force=force_official)
    st.session_state["wnba_official_team_context_debug"] = dbg
    fallback = _build_team_context_from_cached_sources()
    if official is None or official.empty:
        return fallback
    if fallback is not None and not fallback.empty:
        # Fill any official blanks from fallback.
        out = official.merge(fallback, on="Team", how="outer", suffixes=("", "_fb"))
        for c in list(out.columns):
            if c.endswith("_fb"):
                base = c[:-3]
                if base in out.columns:
                    out[base] = out[base].combine_first(out[c])
                else:
                    out[base] = out[c]
        out = out[[c for c in out.columns if not c.endswith("_fb")]]
        return out
    return official


def _last_game_date_for_team(team: str, before_date: Optional[date]) -> Optional[date]:
    sched = load_dataset("schedules")
    if sched is None or sched.empty or before_date is None:
        return None
    s = standardize_schedules(sched)
    if s.empty or "GameDate" not in s.columns:
        return None
    s["_date"] = pd.to_datetime(s["GameDate"], errors="coerce").dt.date
    s["HomeKey"] = s.get("Home", "").map(_team_key_for_matchup)
    s["AwayKey"] = s.get("Away", "").map(_team_key_for_matchup)
    t = _team_key_for_matchup(team)
    hit = s[((s["HomeKey"] == t) | (s["AwayKey"] == t)) & (s["_date"] < before_date)].dropna(subset=["_date"])
    if hit.empty:
        return None
    return max(hit["_date"])


def _rest_days_for_team(team: str, game_date: Optional[date]) -> Tuple[float, str]:
    lg = _last_game_date_for_team(team, game_date)
    if lg is None or game_date is None:
        return np.nan, "Rest unavailable"
    rest = max(0, (game_date - lg).days - 1)
    note = "Back-to-back" if rest == 0 else f"{rest} rest day(s)"
    return float(rest), note




def fallback_lineup_rotation_engine(master: pd.DataFrame = None, logs: pd.DataFrame = None, force: bool = False) -> pd.DataFrame:
    """MLB-style fallback lineup/rotation builder for WNBA.
    Confirmed lineups are ideal, but when they are missing this builds a safe projected rotation
    from recent minutes, season minutes, starter rate, roster games, and role confidence.
    It never blocks projections; it labels the confidence/source so the card can show whether
    the player is confirmed, projected starter, core rotation, bench, or deep bench.
    """
    try:
        confirmed = confirmed_lineup_table(st.session_state.get("wnba_current_mode", "Today"), force=force)
    except Exception:
        confirmed = pd.DataFrame()
    if confirmed is not None and not confirmed.empty:
        d = confirmed.copy()
        d["FallbackLineupRole"] = np.where(d.get("StarterFlag", False).fillna(False).astype(bool), "Confirmed/Projected Starter", "Rotation")
        d["FallbackLineupSource"] = d.get("LineupSource", "confirmed/projected lineup")
        d["FallbackLineupConfidence"] = pd.to_numeric(d.get("LineupConfidence", 82), errors="coerce").fillna(82)
        return d

    if master is None or getattr(master, "empty", True):
        master = load_dataset("master_features")
    if logs is None or getattr(logs, "empty", True):
        logs = load_dataset("player_game_logs")

    rows = []
    if master is not None and not master.empty:
        m = master.copy()
        if "NameKey" not in m.columns and "Player" in m.columns:
            m["NameKey"] = m["Player"].map(normalize_name)
        if "Team" in m.columns:
            m["TeamKey"] = m["Team"].map(_team_key_for_matchup)
        min_cols = [c for c in ["MIN_L3", "MIN_l3", "MIN_L5", "MIN_l5", "MIN_L10", "MIN_l10", "MIN_avg", "MIN", "Minutes"] if c in m.columns]
        if min_cols:
            vals = []
            for _, r in m.iterrows():
                nums = [safe_float(r.get(c), np.nan) for c in min_cols]
                nums = [x for x in nums if pd.notna(x)]
                vals.append(float(np.mean(nums)) if nums else 0.0)
            m["_fb_minutes"] = vals
        else:
            m["_fb_minutes"] = 0.0
        m["_starter_rate"] = pd.to_numeric(m.get("StarterRate", 0), errors="coerce").fillna(0)
        m["_role_conf"] = pd.to_numeric(m.get("RoleConfidence", m.get("DataScore", 50)), errors="coerce").fillna(50)
        for tm, g in m.sort_values(["_fb_minutes", "_starter_rate", "_role_conf"], ascending=False).groupby("TeamKey", dropna=False):
            if not str(tm or "").strip():
                continue
            for rank, (_, r) in enumerate(g.head(11).iterrows(), start=1):
                mins = safe_float(r.get("_fb_minutes"), 0)
                sr = safe_float(r.get("_starter_rate"), 0)
                rc = safe_float(r.get("_role_conf"), 50)
                if rank <= 5 or sr >= 0.55 or mins >= 25:
                    role = "Projected Starter"
                    conf = max(68, min(86, 60 + mins*0.7 + sr*12 + rc*0.08))
                    starter = True
                elif rank <= 8 or mins >= 14:
                    role = "Core Rotation"
                    conf = max(58, min(78, 48 + mins*0.8 + rc*0.08))
                    starter = False
                else:
                    role = "Bench / Deep Rotation"
                    conf = max(40, min(62, 35 + mins*0.9 + rc*0.05))
                    starter = False
                rows.append({
                    "Player": r.get("Player"), "NameKey": r.get("NameKey"), "Team": tm,
                    "StarterFlag": bool(starter), "ProjectedMinutes": round(float(mins), 2),
                    "LineupSource": "Fallback rotation from recent/season minutes",
                    "LineupConfidence": round(float(conf), 1),
                    "FallbackLineupRole": role,
                    "FallbackLineupSource": "Recent minutes + starter rate + role confidence",
                    "FallbackLineupConfidence": round(float(conf), 1),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.dropna(subset=["NameKey"])
    out = out[out["NameKey"].astype(str).str.len() > 1]
    out = out.sort_values("FallbackLineupConfidence", ascending=False).drop_duplicates(["NameKey", "Team"], keep="first")
    try:
        out.to_csv(PROJECTED_ROTATIONS_FILE, index=False)
    except Exception:
        pass
    return out

def _projected_rotation_for_team(team: str) -> Dict[str, Any]:
    """Use lineups → game_rosters → master features as rotation proxy."""
    t = _team_key_for_matchup(team)
    out = {"Projected Starters": 0, "Rotation Players": 0, "Rotation Confidence": 50.0, "Projected Rotation Note": "Projected rotation from cached master only."}
    candidates=[]
    for key in ["lineups", "game_rosters", "master_features"]:
        df=load_dataset(key)
        if df is None or df.empty:
            continue
        d=df.copy()
        if "Team" not in d.columns:
            tc=find_col(d,["Team","team","TEAM","team_abbreviation"])
            if tc: d["Team"]=d[tc]
        if "Team" not in d.columns:
            continue
        d["_TeamKey"]=d["Team"].map(_team_key_for_matchup)
        d=d[d["_TeamKey"]==t].copy()
        if d.empty:
            continue
        if "StarterRate" in d.columns:
            starters = int((pd.to_numeric(d["StarterRate"], errors="coerce").fillna(0) >= 0.55).sum())
        elif "Starter" in d.columns:
            starters = int(d["Starter"].astype(str).str.upper().isin(["1","Y","YES","TRUE","STARTER"]).sum())
        else:
            starters = min(5, len(d)) if key == "lineups" else 0
        rot = int(len(d.drop_duplicates(subset=[c for c in ["Player", "NameKey"] if c in d.columns])) if any(c in d.columns for c in ["Player","NameKey"]) else len(d))
        conf = 90 if key == "lineups" and starters >= 4 else 78 if key == "game_rosters" else 65
        candidates.append({"Projected Starters": starters, "Rotation Players": rot, "Rotation Confidence": conf, "Projected Rotation Note": f"{key} loaded: {starters} starter proxy, {rot} rotation rows."})
    if candidates:
        return candidates[0]
    return out


def _team_injury_context(team: str) -> Dict[str, Any]:
    t = _team_key_for_matchup(team)
    status = load_json(INJURY_STATUS_FILE, [])
    if not status:
        return {"Out Players": 0, "Questionable Players": 0, "Injury Context Note": "No injury status table loaded; neutral."}
    out_ct=q_ct=0; names=[]
    for r in status:
        rt = _team_key_for_matchup(r.get("Team"))
        if rt and rt != t:
            continue
        stt = str(r.get("Status", "")).upper()
        if stt in ["OUT", "DOUBTFUL", "INACTIVE"]:
            out_ct += 1; names.append(str(r.get("Player", "")))
        elif stt in ["QUESTIONABLE", "GTD", "GAME TIME DECISION"]:
            q_ct += 1
    note = f"{out_ct} out/doubtful, {q_ct} questionable" + (f": {', '.join([n for n in names if n][:3])}" if names else "")
    return {"Out Players": out_ct, "Questionable Players": q_ct, "Injury Context Note": note}


def build_game_context_cache(mode: str = "Today", force_official: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-team, per-game context for the selected slate."""
    target = slate_target_date(mode) or datetime.utcnow().date()
    sched = schedule_for_slate(mode)
    team_ctx = _team_context_table(force_official=force_official)
    dbg=[]
    rows=[]
    if sched is None or sched.empty:
        dbg.append({"Step":"schedule", "Status":"empty", "Rows":0})
        # If no schedule, still build team-only context so active board fallback can enrich teams.
        if team_ctx is not None and not team_ctx.empty:
            for _, tc in team_ctx.iterrows():
                rows.append({"Team": _team_key_for_matchup(tc.get("Team")), "Opponent":"", "Matchup":"", "HomeAway":"", "GameDate": str(target), **tc.to_dict()})
    else:
        s=standardize_schedules(sched)
        dbg.append({"Step":"schedule", "Status":"loaded", "Rows":len(s)})
        for _, g in s.iterrows():
            home=_team_key_for_matchup(g.get("Home")); away=_team_key_for_matchup(g.get("Away"))
            if not home or not away:
                continue
            gdate = pd.to_datetime(g.get("GameDate"), errors="coerce").date() if pd.notna(pd.to_datetime(g.get("GameDate"), errors="coerce")) else target
            for team, opp, ha in [(away, home, "AWAY"), (home, away, "HOME")]:
                base={"Team":team,"Opponent":opp,"Matchup":f"{away} @ {home}","HomeAway":ha,"GameDate":str(gdate)}
                if team_ctx is not None and not team_ctx.empty and "Team" in team_ctx.columns:
                    hit=team_ctx[team_ctx["Team"].map(_team_key_for_matchup)==team]
                    if not hit.empty:
                        base.update(hit.iloc[-1].to_dict())
                opp_extra={}
                if team_ctx is not None and not team_ctx.empty and "Team" in team_ctx.columns:
                    ohit=team_ctx[team_ctx["Team"].map(_team_key_for_matchup)==opp]
                    if not ohit.empty:
                        for k,v in ohit.iloc[-1].to_dict().items():
                            if k != "Team": opp_extra[f"Opp_{k}"]=v
                base.update(opp_extra)
                rest, rest_note=_rest_days_for_team(team,gdate); base["RestDays"]=rest; base["RestNote"]=rest_note; base["BackToBack"]=bool(pd.notna(rest) and rest==0)
                base.update(_projected_rotation_for_team(team))
                base.update(_team_injury_context(team))
                rows.append(base)
    out=pd.DataFrame(rows)
    if not out.empty:
        try:
            out.to_csv(GAME_CONTEXT_FILE, index=False)
        except Exception:
            pass
    st.session_state["wnba_game_context_cache"] = out
    st.session_state["wnba_game_context_debug"] = pd.DataFrame(dbg)
    return out, pd.DataFrame(dbg)


def _get_game_context_for_row(row: Dict[str, Any], mode: str = "Today") -> Dict[str, Any]:
    ctx = st.session_state.get("wnba_game_context_cache", pd.DataFrame())
    if ctx is None or ctx.empty:
        ctx, _ = build_game_context_cache(mode, force_official=False)
    if ctx is None or ctx.empty:
        return {}
    team=_team_key_for_matchup(row.get("Team"))
    opp=_team_key_for_matchup(row.get("Opponent"))
    d=ctx.copy()
    if "Team" not in d.columns:
        return {}
    d["_TeamKey"]=d["Team"].map(_team_key_for_matchup)
    if opp and "Opponent" in d.columns:
        d["_OppKey"]=d["Opponent"].map(_team_key_for_matchup)
        hit=d[(d["_TeamKey"]==team)&(d["_OppKey"]==opp)]
    else:
        hit=d[d["_TeamKey"]==team]
    if hit.empty:
        return {}
    return hit.iloc[-1].to_dict()


def _ctx_num(ctx: Dict[str, Any], keys: List[str], default=np.nan):
    for k in keys:
        v = safe_float(ctx.get(k), np.nan)
        if pd.notna(v):
            return float(v)
    return default


def game_context_projection_engine(row: Dict[str, Any], base_row: pd.Series, logs: pd.DataFrame, mode: str = "Today") -> Dict[str, Any]:
    """Conservative additive/multiplicative context layer for today's game."""
    ctx = _get_game_context_for_row(row, mode)
    if not ctx:
        return {"Game Context Factor": 1.0, "Game Context Add": 0.0, "Game Context Score": 40, "Game Context Note": "Game context unavailable; player baseline only."}
    market=str(row.get("Market","")).upper()
    team=_team_key_for_matchup(row.get("Team")); opp=_team_key_for_matchup(row.get("Opponent"))
    notes=[]; factors=[]; add=0.0; score=55
    pace=_ctx_num(ctx,["Opp_Team_Pace_Official","Opp_Adv_PACE","Opp_Base_PACE","Team_Pace_Official","Team_Pace","Rank_Team_Pace"],np.nan)
    if pd.notna(pace):
        f=max(0.965,min(1.045,1+(pace-78.0)/620.0)); factors.append(f); notes.append(f"pace {pace:.1f} factor {f:.3f}"); score+=8
    opp_drtg=_ctx_num(ctx,["Opp_Team_DRtg_Official","Opp_Adv_DEF_RATING","Opp_Team_DRtg","Opp_Rank_Team_DRtg"],np.nan)
    if pd.notna(opp_drtg):
        f=max(0.955,min(1.055,1+(opp_drtg-100.0)/820.0)); factors.append(f); notes.append(f"opp DRtg {opp_drtg:.1f} factor {f:.3f}"); score+=12
    homeaway=str(ctx.get("HomeAway") or row.get("HomeAway") or "").upper()
    if homeaway == "HOME":
        add += 0.12 if market in ["PTS","PRA"] else 0.05
        notes.append("home +0.12/+0.05")
        score+=5
    elif homeaway == "AWAY":
        add -= 0.05 if market in ["PTS","PRA"] else 0.02
        notes.append("away small tax")
        score+=5
    rest=safe_float(ctx.get("RestDays"), np.nan)
    if pd.notna(rest):
        if rest == 0:
            add -= 0.22 if market in ["PTS","PRA"] else 0.08
            notes.append("B2B fatigue tax")
        elif rest >= 2:
            add += 0.08
            notes.append(f"rested {int(rest)}d")
        score+=7
    rot_conf=safe_float(ctx.get("Rotation Confidence"), np.nan)
    if pd.notna(rot_conf):
        add += max(-0.18,min(0.18,(rot_conf-65)/240.0))
        notes.append(f"rotation conf {rot_conf:.0f}")
        score+=7
    out_players=safe_float(ctx.get("Out Players"), 0)
    if out_players:
        # If own team has outs, small usage volatility/ripple; manual injury table remains stronger if provided.
        add += min(0.35, 0.08*out_players) if market in ["PTS","PRA","AST"] else min(0.18, 0.04*out_players)
        notes.append(f"injury ripple proxy {int(out_players)} out")
        score+=5
    if market == "REB":
        opp_reb=_ctx_num(ctx,["Opp_Team_REB_Official","Opp_Base_REB","Opp_Season_REB"],np.nan)
        opp_dreb_pct=_ctx_num(ctx,["Opp_Team_DREB_PCT_Official","Opp_Four_DREB_PCT","Opp_Adv_DREB_PCT"],np.nan)
        if pd.notna(opp_reb):
            add += max(-0.18,min(0.18,(82-opp_reb)/90.0)); notes.append("rebound environment")
        if pd.notna(opp_dreb_pct):
            add += max(-0.16,min(0.16,(0.72-opp_dreb_pct)/1.8)); notes.append("opp DREB% context")
    if market == "AST":
        opp_ast=_ctx_num(ctx,["Opp_Team_AST_Official","Opp_Base_AST","Opp_Season_AST"],np.nan)
        if pd.notna(opp_ast):
            add += max(-0.14,min(0.14,(opp_ast-19)/70.0)); notes.append("assist environment")
    if market in ["PTS","PRA"]:
        opp_efg=_ctx_num(ctx,["Opp_Team_eFG_Official","Opp_Four_EFG_PCT","Opp_Adv_EFG_PCT"],np.nan)
        if pd.notna(opp_efg):
            add += max(-0.16,min(0.16,(opp_efg-0.50)*1.5)); notes.append("efficiency environment")
    factor=float(np.prod(factors)) if factors else 1.0
    factor=max(0.92,min(1.09,factor))
    score=max(0,min(100,score))
    return {
        "Game Context Factor": round(factor,4),
        "Game Context Add": round(float(add),3),
        "Game Context Score": round(score,1),
        "Game Context Team": team,
        "Game Context Opponent": opp,
        "Game Context Matchup": str(ctx.get("Matchup") or row.get("Matchup") or ""),
        "Game Context HomeAway": homeaway,
        "Game Context Pace": pace if pd.notna(pace) else np.nan,
        "Opponent DRtg Used": opp_drtg if pd.notna(opp_drtg) else np.nan,
        "Rest Days Used": rest if pd.notna(rest) else np.nan,
        "Projected Rotation Note": ctx.get("Projected Rotation Note", "Rotation unavailable"),
        "Injury Context Note": ctx.get("Injury Context Note", "No injury table loaded; neutral."),
        "Game Context Note": "; ".join(notes) if notes else "Opponent matched; neutral context factors.",
    }


def attach_game_context_columns(lines_df: pd.DataFrame, mode: str = "Today") -> pd.DataFrame:
    if lines_df is None or lines_df.empty:
        return lines_df
    out=lines_df.copy()
    build_game_context_cache(mode, force_official=False)
    for idx,row in out.iterrows():
        ctx=_get_game_context_for_row(row.to_dict(), mode)
        if not ctx:
            continue
        for c in ["Matchup","Opponent","HomeAway"]:
            if c in ctx and (c not in out.columns or not str(out.at[idx,c] if c in out.columns else "").strip()):
                out.at[idx,c]=ctx.get(c)
    return out


# Preserve prior projection builder and enhance it with full advanced engines.
_make_projection_board_core = make_projection_board

def make_projection_board(lines, logs, base, mode: Optional[str] = None):
    mode = mode or st.session_state.get("wnba_current_mode", "Today")
    if lines is not None and not lines.empty:
        try:
            lines = enrich_board_with_matchups(lines, mode)
            lines = attach_game_context_columns(lines, mode)
        except Exception as _ctx_e:
            st.session_state["wnba_game_context_last_error"] = str(_ctx_e)[:180]
    core = _make_projection_board_core(lines, logs, base)
    if core is None or core.empty:
        return core
    try:
        core = enrich_board_with_matchups(core, mode)
        core = attach_game_context_columns(core, mode)
    except Exception:
        pass
    out=[]
    for _, r in core.iterrows():
        row=r.to_dict()
        b, score = match_player_base(row.get("Player", ""), base if base is not None and not base.empty else load_dataset("master_features"))
        if b is None:
            b = pd.Series(dtype=object)
        xgb = model_prediction_for_row(row)
        row.update(xgb)
        # Ensemble recalibration: blend current projection with trained model when available.
        p0=safe_float(row.get("Projection"), np.nan); px=safe_float(row.get("XGBoost Projection"), np.nan)
        if use_xgb_blend_enabled() and pd.notna(p0) and pd.notna(px):
            row["Ensemble Projection"] = round(0.72*p0 + 0.28*px, 2)
            row["Projection"] = row["Ensemble Projection"]
            row["Edge"] = round(row["Projection"] - safe_float(row.get("Line"), np.nan), 2)
        inj = injury_ripple_engine(row, b); row.update(inj)
        opp = opponent_lineup_adjustment(row, b); row.update(opp)
        ref = referee_tendency_engine(row); row.update(ref)
        trav = latest_travel_context(logs, normalize_name(row.get("Matched Player") or row.get("Player")), row.get("Team")); row.update(trav)
        game_ctx = game_context_projection_engine(row, b, logs, mode); row.update(game_ctx)
        # Apply final live context: injury ripple, opponent lineup, referee, travel, and game-context team/pace/defense/rest.
        context_add = safe_float(row.get("Injury Ripple Bump"),0) + safe_float(row.get("Opponent Lineup Adj"),0) + safe_float(row.get("Referee Factor"),0) + safe_float(row.get("Travel Tax"),0) + safe_float(row.get("Game Context Add"),0)
        context_factor = safe_float(row.get("Game Context Factor"), 1.0)
        if pd.notna(safe_float(row.get("Projection"), np.nan)):
            raw_context_projection = safe_float(row.get("Projection"))
            row["Projection Before Game Context"] = round(raw_context_projection, 2)
            row["Projection"] = round(raw_context_projection * context_factor + context_add, 2)
            row["Edge"] = round(row["Projection"] - safe_float(row.get("Line"), np.nan), 2)
        lean = "OVER" if safe_float(row.get("Edge"), 0) > 0 else "UNDER"
        row["Lean"] = lean
        # EV/Kelly based on Monte Carlo side probability.
        side_prob = safe_float(row.get("Over %"), 0) if lean == "OVER" else safe_float(row.get("Under %"), 0)
        row.update(ev_kelly_engine(side_prob, -110))
        row.update(clv_engine(row.get("Player"), row.get("Market"), safe_float(row.get("Line"), np.nan), safe_float(row.get("Opening Line"), np.nan)))
        row["Sharp Money Note"] = sharp_money_detector(safe_float(row.get("Line Move"), np.nan), safe_float(row.get("Edge"), np.nan), lean)
        row.update(model_disagreement_full(row))
        # Re-score official with EV and disagreement penalties.
        official_score = safe_float(row.get("Official Play Score"), 0)
        official_score += max(-8, min(8, safe_float(row.get("EV %"), 0)*0.6))
        if safe_float(row.get("Model Disagreement Score"), 0) > 2.4: official_score -= 8
        if safe_float(row.get("Kelly %"), 0) >= 2: official_score += 3
        row["Official Play Score"] = round(max(0, min(100, official_score)), 1)
        sim_side = side_prob
        row["Tier"] = tier_grade(row["Official Play Score"], safe_float(row.get("Edge"),0), sim_side, safe_float(row.get("Data Score"),0))
        row["Feature Importance"] = feature_importance_text(row) + " | Game Context: " + str(row.get("Game Context Note", "")) + " | XGB: " + str(row.get("XGBoost Feature Importance", ""))
        row["Full Engine Note"] = "Similarity + trained ML + official/team context + opponent defense/pace + home-away/rest + projected rotations + referee + travel + injury ripple + opponent lineup + CLV + EV/Kelly active."
        out.append(row)
    df=pd.DataFrame(out)
    if not df.empty:
        df=df.sort_values(["Official Play Score","Edge"], ascending=[False, False])
        save_dataset("projection_board", df)
    return df


# ============================================================
# Logs / backup tools
# ============================================================
def save_officials(df):
    plays = df[df["Official"].astype(str).str.contains("OVER|UNDER", na=False)].copy() if df is not None and not df.empty else pd.DataFrame()
    if plays.empty:
        return 0
    log = load_json(OFFICIAL_LOG, [])
    stamp = now_iso()
    for _, r in plays.iterrows():
        row = r.to_dict(); row["SavedAt"] = stamp; row["Result"] = "PENDING"; row["Actual"] = None
        log.append(row)
    save_json(OFFICIAL_LOG, log)
    hist = load_json(LINE_HISTORY_FILE, [])
    for _, r in df.iterrows():
        hist.append({"SavedAt": stamp, "Player": r.get("Player"), "Market": r.get("Market"), "Line": r.get("Line"), "Source": r.get("Source"), "Projection": r.get("Projection")})
    save_json(LINE_HISTORY_FILE, hist)
    return len(plays)


def latest_closing_line_for_pick(player: str, market: str, saved_at: str = "", source: str = "") -> float:
    """Return the latest line seen after a pick was saved, if available."""
    hist = pd.DataFrame(load_json(LINE_HISTORY_FILE, []))
    if hist.empty or "Player" not in hist.columns or "Market" not in hist.columns or "Line" not in hist.columns:
        return np.nan
    h = hist.copy()
    h["NameKey"] = h["Player"].map(normalize_name)
    h = h[(h["NameKey"] == normalize_name(player)) & (h["Market"].astype(str).str.upper() == str(market).upper())].copy()
    if source and "Source" in h.columns:
        hs = h[h["Source"].astype(str) == str(source)]
        if not hs.empty:
            h = hs
    if h.empty:
        return np.nan
    time_col = "PulledAt" if "PulledAt" in h.columns else "SavedAt" if "SavedAt" in h.columns else None
    if time_col:
        h[time_col] = pd.to_datetime(h[time_col], errors="coerce")
        if saved_at:
            sat = pd.to_datetime(saved_at, errors="coerce")
            if pd.notna(sat):
                h2 = h[h[time_col] >= sat].copy()
                if not h2.empty:
                    h = h2
        h = h.sort_values(time_col)
    return safe_float(h.iloc[-1].get("Line"), np.nan)


def grade_pending(logs, mode: Optional[str] = None):
    """Auto-grade pending official plays using imported player logs.

    This version is safer than the original:
    - it grades the first matching game after SavedAt / Start, not always the latest game;
    - it stores ClosingLine and CLV when a later line snapshot exists;
    - it never crashes if logs or columns are missing.
    """
    official = load_json(OFFICIAL_LOG, [])
    if not official or logs is None or logs.empty:
        return 0
    logs = standardize_player_logs(logs) if "NameKey" not in logs.columns else logs.copy()
    if logs.empty or "GameDate" not in logs.columns:
        return 0
    logs["GameDate"] = pd.to_datetime(logs["GameDate"], errors="coerce")
    grade_target_date = slate_target_date(mode) if mode in ["Today", "Tomorrow"] else None
    now_ts = pd.Timestamp.now()
    updated = 0
    learn = load_json(LEARNING_LOG, [])
    existing_ids = set()
    for r in learn:
        existing_ids.add(str(r.get("SavedAt", "")) + "|" + normalize_name(r.get("Player")) + "|" + str(r.get("Market")))
    for row in official:
        if row.get("Result") != "PENDING":
            continue
        # If grading from a Today/Tomorrow board, grade only that slate.
        # This prevents one-game slates from grading tomorrow/future games or
        # old saved boards that are still pending.
        row_slate_date = None
        for _dc in [row.get("SlateDate"), row.get("Start"), row.get("GameDate")]:
            _dt = pd.to_datetime(_dc, errors="coerce")
            if pd.notna(_dt):
                row_slate_date = _dt.date()
                break
        if grade_target_date is not None:
            if row_slate_date is None:
                row["GradeNote"] = f"Skipped: slate date missing for {mode} grader."
                continue
            if row_slate_date != grade_target_date:
                row["GradeNote"] = f"Skipped: saved for {row_slate_date}, not {mode} ({grade_target_date})."
                continue
        # Never grade a game that has not started yet.
        _start_dt = pd.to_datetime(row.get("Start"), errors="coerce")
        if pd.notna(_start_dt):
            try:
                _start_dt = _start_dt.tz_convert(None)
            except Exception:
                try:
                    _start_dt = _start_dt.tz_localize(None)
                except Exception:
                    pass
            if _start_dt > now_ts:
                row["GradeNote"] = "Skipped: game has not started yet."
                continue
        player_name = row.get("Matched Player") or row.get("Player")
        key = normalize_name(player_name)
        market = str(row.get("Market", "")).upper()
        if market not in logs.columns:
            row["GradeNote"] = f"Cannot grade: {market} not in player logs."
            continue
        d = logs[logs["NameKey"] == key].copy()
        if d.empty:
            row["GradeNote"] = "Cannot grade: player not found in logs."
            continue
        # Choose first game after the play was saved/start time when possible.
        cutoff = pd.NaT
        for tc in [row.get("Start"), row.get("GameDate"), row.get("SavedAt")]:
            tmp = pd.to_datetime(tc, errors="coerce")
            if pd.notna(tmp):
                cutoff = tmp.tz_convert(None) if getattr(tmp, 'tzinfo', None) else tmp
                break
        d = d.sort_values("GameDate")
        if grade_target_date is not None:
            d_same = d[d["GameDate"].dt.date == grade_target_date].copy()
            if d_same.empty:
                row["GradeNote"] = f"Skipped: no completed player log for {grade_target_date}."
                continue
            d = d_same
        elif pd.notna(cutoff):
            d_after = d[d["GameDate"] >= cutoff - pd.Timedelta(hours=12)].copy()
            if not d_after.empty:
                d = d_after
        actual = safe_float(d.iloc[0].get(market), np.nan)
        game_date = d.iloc[0].get("GameDate")
        if pd.isna(actual):
            row["GradeNote"] = "Cannot grade: actual value missing."
            continue
        lean = str(row.get("Lean", "")).upper()
        line = safe_float(row.get("Line"), np.nan)
        if pd.isna(line):
            row["GradeNote"] = "Cannot grade: line missing."
            continue
        push = abs(actual - line) < 1e-9
        win = (actual > line and lean == "OVER") or (actual < line and lean == "UNDER")
        row["Actual"] = round(float(actual), 2)
        row["ActualGameDate"] = str(game_date)
        row["Result"] = "PUSH" if push else "WIN" if win else "LOSS"
        row["GradedAt"] = now_iso()
        close_line = latest_closing_line_for_pick(row.get("Player"), market, row.get("SavedAt", ""), row.get("Source", ""))
        row["ClosingLine"] = close_line if pd.notna(close_line) else None
        if pd.notna(close_line):
            row["CLV"] = round((close_line - line) if lean == "OVER" else (line - close_line), 2)
        else:
            row["CLV"] = None
        row["GradeNote"] = "Auto-graded from imported player logs."
        learn_id = str(row.get("SavedAt", "")) + "|" + normalize_name(row.get("Player")) + "|" + str(row.get("Market"))
        if learn_id not in existing_ids:
            learn.append(row.copy())
            existing_ids.add(learn_id)
        updated += 1
    save_json(OFFICIAL_LOG, official)
    save_json(LEARNING_LOG, learn)
    return updated



# ============================================================
# ESPN final boxscore pull + MLB-style end-game autograder
# ============================================================
def _espn_parse_min_to_float(v: Any) -> float:
    """Parse ESPN minutes strings like '34', '34:12', 'DNP' into decimal minutes."""
    s = str(v or "").strip()
    if not s or s.upper().startswith("DNP"):
        return 0.0
    try:
        if ":" in s:
            a, b = s.split(":", 1)
            return float(a) + float(b[:2] or 0) / 60.0
        return float(s)
    except Exception:
        return safe_float(s, 0.0)


def _espn_map_player_stats(labels: List[Any], stats: List[Any]) -> Dict[str, Any]:
    """Map ESPN boxscore label/value arrays to normalized WNBA player log fields."""
    row = {"MIN": 0, "PTS": 0, "REB": 0, "AST": 0, "FGA": 0, "FGM": 0, "FG3A": 0, "FG3M": 0, "FTA": 0, "FTM": 0, "OREB": 0, "DREB": 0, "STL": 0, "BLK": 0, "TOV": 0, "PLUS_MINUS": 0}
    labels = [str(x or "").strip().upper() for x in (labels or [])]
    values = list(stats or [])
    # ESPN often uses labels like MIN, FG, 3PT, FT, OREB, DREB, REB, AST, STL, BLK, TO, PF, +/-, PTS
    for i, lab in enumerate(labels):
        if i >= len(values):
            continue
        val = values[i]
        if lab in ["MIN", "MINUTES"]:
            row["MIN"] = _espn_parse_min_to_float(val)
        elif lab in ["PTS", "POINTS"]:
            row["PTS"] = safe_float(val, 0)
        elif lab in ["REB", "REBOUNDS", "TOT"]:
            row["REB"] = safe_float(val, 0)
        elif lab in ["AST", "ASSISTS"]:
            row["AST"] = safe_float(val, 0)
        elif lab in ["OREB", "OR"]:
            row["OREB"] = safe_float(val, 0)
        elif lab in ["DREB", "DR"]:
            row["DREB"] = safe_float(val, 0)
        elif lab in ["STL"]:
            row["STL"] = safe_float(val, 0)
        elif lab in ["BLK"]:
            row["BLK"] = safe_float(val, 0)
        elif lab in ["TO", "TOV"]:
            row["TOV"] = safe_float(val, 0)
        elif lab in ["+/-", "PLUS/MINUS", "PLUS_MINUS"]:
            row["PLUS_MINUS"] = safe_float(val, 0)
        elif lab in ["FG", "FGM-A"]:
            s = str(val)
            if "/" in s:
                m, a = s.split("/", 1); row["FGM"] = safe_float(m, 0); row["FGA"] = safe_float(a, 0)
        elif lab in ["3PT", "3P", "FG3", "3PM-A"]:
            s = str(val)
            if "/" in s:
                m, a = s.split("/", 1); row["FG3M"] = safe_float(m, 0); row["FG3A"] = safe_float(a, 0)
        elif lab in ["FT", "FTM-A"]:
            s = str(val)
            if "/" in s:
                m, a = s.split("/", 1); row["FTM"] = safe_float(m, 0); row["FTA"] = safe_float(a, 0)
    # Some ESPN groups omit OREB/DREB but include REB; keep REB as source of truth.
    row["PRA"] = safe_float(row.get("PTS"), 0) + safe_float(row.get("REB"), 0) + safe_float(row.get("AST"), 0)
    return row


def _espn_event_is_final(event: Dict[str, Any]) -> bool:
    status = event.get("status", {}) or {}
    typ = status.get("type", {}) or {}
    state = str(typ.get("state", "")).lower()
    name = str(typ.get("name", "")).upper()
    desc = str(typ.get("description", "")).lower()
    return state in {"post"} or "FINAL" in name or "final" in desc


def pull_espn_final_player_logs(mode: str = "Today", force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pull final ESPN WNBA boxscores for the selected slate and merge them into player_game_logs.

    This gives the app MLB-style end-game grading: once ESPN marks the game final, the app can fetch
    player PTS/REB/AST/MIN and grade saved props without you manually importing SportsDataverse logs.
    Live/not-final games are skipped so tomorrow or in-progress games do not get graded.
    """
    target = slate_target_date(mode) if mode in ["Today", "Tomorrow"] else date.today()
    dbg = []
    if target is None:
        return pd.DataFrame(), pd.DataFrame([{"step":"espn_final_logs", "status":"skipped", "message":"No slate date"}])
    ymd = target.strftime("%Y%m%d")
    scoreboard_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={ymd}"
    try:
        r = requests.get(scoreboard_url, timeout=14, headers={"User-Agent":"Mozilla/5.0", "Accept":"application/json"})
        dbg.append({"step":"scoreboard", "status":f"HTTP {r.status_code}", "message":scoreboard_url})
        js = r.json() if r.status_code == 200 else {}
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame([{"step":"scoreboard", "status":"error", "message":str(e)[:220]}])

    all_rows = []
    for ev in js.get("events", []) or []:
        event_id = str(ev.get("id") or "")
        if not event_id:
            continue
        if not _espn_event_is_final(ev):
            status_desc = (((ev.get("status") or {}).get("type") or {}).get("description") or "not final")
            dbg.append({"step":"event", "status":"skipped_not_final", "message":f"{event_id}: {status_desc}"})
            continue
        event_date = pd.to_datetime(ev.get("date"), errors="coerce")
        comps = ev.get("competitions", []) or []
        comp = comps[0] if comps else {}
        competitor_teams = {}
        for c in comp.get("competitors", []) or []:
            tid = str((c.get("team") or {}).get("id") or c.get("id") or "")
            ab = _team_key_for_matchup((c.get("team") or {}).get("abbreviation") or (c.get("team") or {}).get("shortDisplayName") or (c.get("team") or {}).get("displayName") or "")
            competitor_teams[tid] = {"Team": ab, "HomeAway": str(c.get("homeAway", "")).upper()}
        summary_url = f"https://site.web.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={event_id}"
        try:
            sr = requests.get(summary_url, timeout=16, headers={"User-Agent":"Mozilla/5.0", "Accept":"application/json"})
            dbg.append({"step":"summary", "status":f"HTTP {sr.status_code}", "message":summary_url})
            sj = sr.json() if sr.status_code == 200 else {}
        except Exception as e:
            dbg.append({"step":"summary", "status":"error", "message":f"{event_id}: {str(e)[:180]}"})
            continue
        players_groups = ((sj.get("boxscore") or {}).get("players") or [])
        for team_group in players_groups:
            team_obj = team_group.get("team", {}) or {}
            team_ab = _team_key_for_matchup(team_obj.get("abbreviation") or team_obj.get("shortDisplayName") or team_obj.get("displayName") or "")
            team_id = str(team_obj.get("id") or "")
            if not team_ab and team_id in competitor_teams:
                team_ab = competitor_teams[team_id].get("Team", "")
            home_away = competitor_teams.get(team_id, {}).get("HomeAway", "")
            for stat_group in team_group.get("statistics", []) or []:
                labels = stat_group.get("labels") or stat_group.get("keys") or []
                athletes = stat_group.get("athletes") or []
                for a in athletes:
                    athlete = a.get("athlete", {}) or {}
                    player = athlete.get("displayName") or athlete.get("shortName") or athlete.get("name") or ""
                    if not player:
                        continue
                    stats = a.get("stats") or a.get("statistics") or []
                    vals = _espn_map_player_stats(labels, stats)
                    # Include DNP rows too, but grading will use actual 0 where a player prop was saved.
                    row = {
                        "Player": player,
                        "Team": team_ab,
                        "Opponent": "",
                        "GameDate": event_date if pd.notna(event_date) else pd.Timestamp(target),
                        "Season": target.year,
                        "GameID": event_id,
                        "HomeAway": home_away,
                        "Starter": bool(a.get("starter", False)),
                        "Source": "ESPN final boxscore",
                    }
                    row.update(vals)
                    all_rows.append(row)
    logs_new = standardize_player_logs(pd.DataFrame(all_rows)) if all_rows else pd.DataFrame()
    if logs_new.empty:
        return pd.DataFrame(), pd.DataFrame(dbg + [{"step":"espn_final_logs", "status":"no_final_logs", "message":"No final ESPN player rows parsed yet."}])
    old = load_dataset("player_game_logs")
    combined = pd.concat([old, logs_new], ignore_index=True, sort=False) if old is not None and not old.empty else logs_new.copy()
    combined = standardize_player_logs(combined)
    try:
        combined.to_csv(CACHE_FILES["player_game_logs"], index=False)
    except Exception as e:
        dbg.append({"step":"save_logs", "status":"error", "message":str(e)[:180]})
    dbg.append({"step":"espn_final_logs", "status":"saved", "message":f"{len(logs_new)} new/updated ESPN player rows; cache now {len(combined)} rows."})
    return logs_new, pd.DataFrame(dbg)


def pull_final_results_and_grade(mode: str = "Today") -> Tuple[int, pd.DataFrame]:
    """Fetch final ESPN player logs for finished games, then grade selected slate."""
    logs_new, dbg = pull_espn_final_player_logs(mode, force=True)
    logs = load_dataset("player_game_logs")
    n = grade_pending(logs, mode)
    extra = pd.DataFrame([{"step":"grade_pending", "status":"done", "message":f"Updated {n} pending plays for {mode}."}])
    dbg = pd.concat([dbg, extra], ignore_index=True, sort=False) if dbg is not None and not dbg.empty else extra
    return n, dbg

def grade_diagnostics(logs, mode: Optional[str] = None) -> Dict[str, Any]:
    """Quick status check so user can see why AutoGrader updated 0 plays."""
    official = load_json(OFFICIAL_LOG, [])
    out = {"Saved Official Plays": len(official), "Pending Plays": 0, "Eligible Slate Pending": 0, "Completed Player Logs": 0, "Matched Completed Logs": 0, "Skipped/Notes": []}
    if not official:
        out["Skipped/Notes"].append("No saved official plays found. Click Save official before games first.")
        return out
    pending = [r for r in official if str(r.get("Result", "PENDING")).upper() == "PENDING"]
    out["Pending Plays"] = len(pending)
    target = slate_target_date(mode) if mode in ["Today", "Tomorrow"] else None
    if target is not None:
        scoped=[]
        for r in pending:
            row_date=None
            for dc in [r.get("SlateDate"), r.get("Start"), r.get("GameDate")]:
                dt=pd.to_datetime(dc, errors="coerce")
                if pd.notna(dt):
                    row_date=dt.date(); break
            if row_date == target:
                scoped.append(r)
        pending = scoped
    out["Eligible Slate Pending"] = len(pending)
    if logs is None or getattr(logs, "empty", True):
        out["Skipped/Notes"].append("No player game logs loaded yet. Import/refresh final player logs after the game.")
        return out
    try:
        lg = standardize_player_logs(logs) if "NameKey" not in logs.columns else logs.copy()
    except Exception:
        lg = logs.copy()
    if "GameDate" not in lg.columns:
        out["Skipped/Notes"].append("Player logs loaded, but GameDate column is missing.")
        return out
    lg["GameDate"] = pd.to_datetime(lg["GameDate"], errors="coerce")
    if target is not None:
        lg = lg[lg["GameDate"].dt.date == target].copy()
    out["Completed Player Logs"] = len(lg.dropna(subset=["GameDate"])) if "GameDate" in lg.columns else len(lg)
    names = set(lg.get("NameKey", pd.Series(dtype=str)).dropna().astype(str).tolist())
    matched = 0
    for r in pending:
        if normalize_name(r.get("Matched Player") or r.get("Player")) in names:
            matched += 1
    out["Matched Completed Logs"] = matched
    if len(pending) and matched == 0:
        out["Skipped/Notes"].append("Pending plays exist, but no matching completed player logs were found for this slate/date.")
    if len(pending) == 0:
        out["Skipped/Notes"].append("No pending plays for the selected grade scope.")
    return out


# ============================================================
# After-game results report helpers
# ============================================================
def build_after_game_results_table(source_df: pd.DataFrame) -> pd.DataFrame:
    """Build a clean user-facing ✅/❌ after-game results table from official/learning logs."""
    if source_df is None or source_df.empty:
        return pd.DataFrame()
    df = source_df.copy()
    for c in ["Player", "Market", "Line", "Lean", "Projection", "Actual", "Result"]:
        if c not in df.columns:
            df[c] = None
    df["Lean"] = df["Lean"].astype(str).str.upper()
    df["Market"] = df["Market"].astype(str).str.upper()
    df["Line"] = pd.to_numeric(df["Line"], errors="coerce")
    df["Projection"] = pd.to_numeric(df["Projection"], errors="coerce")
    df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce")
    def _cleared(r):
        result = str(r.get("Result", "")).upper()
        if result == "WIN":
            return "✅"
        if result == "LOSS":
            return "❌"
        if result == "PUSH":
            return "➖"
        return "⏳"
    def _actual_vs_line(r):
        actual = r.get("Actual")
        line = r.get("Line")
        if pd.isna(actual) or pd.isna(line):
            return ""
        return f"{actual:.1f} vs {line:.1f}"
    def _margin(r):
        actual = r.get("Actual")
        line = r.get("Line")
        lean = str(r.get("Lean", "")).upper()
        if pd.isna(actual) or pd.isna(line):
            return np.nan
        return round((actual - line) if lean == "OVER" else (line - actual), 2)
    df["✅/❌"] = df.apply(_cleared, axis=1)
    df["Actual vs Line"] = df.apply(_actual_vs_line, axis=1)
    df["Cleared By"] = df.apply(_margin, axis=1)
    if "SavedAt" in df.columns:
        df["SavedAt"] = pd.to_datetime(df["SavedAt"], errors="coerce")
    if "ActualGameDate" in df.columns:
        df["ActualGameDate"] = pd.to_datetime(df["ActualGameDate"], errors="coerce")
    sort_col = "ActualGameDate" if "ActualGameDate" in df.columns else "SavedAt" if "SavedAt" in df.columns else None
    if sort_col:
        df = df.sort_values(sort_col, ascending=False)
    cols = [c for c in [
        "✅/❌", "Result", "Player", "Team", "Opponent", "Matchup", "Market", "Lean",
        "Line", "Actual", "Actual vs Line", "Cleared By", "Projection", "Edge",
        "Official Play Score", "Source", "ActualGameDate", "SavedAt", "GradeNote"
    ] if c in df.columns]
    return df[cols]


def make_backup_zip() -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for path in list(DATA_DIR.glob("*.csv")) + list(LOCAL_DIR.glob("*.json")):
            if path.exists():
                z.write(path, arcname=str(path.relative_to(LOCAL_DIR)))
    mem.seek(0)
    return mem.getvalue()


def reset_logs():
    for p in [OFFICIAL_LOG, RESULT_LOG, LEARNING_LOG, LINE_HISTORY_FILE, NO_LINE_FILE]:
        if p.exists():
            p.unlink()


# ============================================================
# Auto-grader / CLV / Calibration / Historical backtest reports
# ============================================================
def log_line_snapshot(lines: pd.DataFrame, slate_label: str = "") -> int:
    """Save every pulled sportsbook line snapshot for line movement and CLV tracking."""
    if lines is None or lines.empty:
        return 0
    hist = load_json(LINE_HISTORY_FILE, [])
    stamp = now_iso()
    add = 0
    for _, r in lines.iterrows():
        line = safe_float(r.get("Line"), np.nan)
        if pd.isna(line):
            continue
        hist.append({
            "PulledAt": stamp,
            "Slate": slate_label,
            "Player": r.get("Player"),
            "NameKey": normalize_name(r.get("Player")),
            "Team": r.get("Team", ""),
            "Market": str(r.get("Market", "")).upper(),
            "Line": float(line),
            "Source": r.get("Source", ""),
            "Start": r.get("Start", ""),
            "Projection": r.get("Projection", None),
        })
        add += 1
    save_json(LINE_HISTORY_FILE, hist)
    return add


def line_movement_report() -> pd.DataFrame:
    hist = pd.DataFrame(load_json(LINE_HISTORY_FILE, []))
    if hist.empty:
        return pd.DataFrame()
    for c in ["Player", "Market", "Source", "Line"]:
        if c not in hist.columns:
            return pd.DataFrame()
    h = hist.copy()
    h["Line"] = pd.to_numeric(h["Line"], errors="coerce")
    h["PulledAt"] = pd.to_datetime(h.get("PulledAt", h.get("SavedAt", "")), errors="coerce")
    h["NameKey"] = h.get("NameKey", h["Player"].map(normalize_name))
    h = h.dropna(subset=["Line"])
    if h.empty:
        return pd.DataFrame()
    rows = []
    for (nk, market, src), g in h.sort_values("PulledAt").groupby(["NameKey", "Market", "Source"], dropna=False):
        if g.empty:
            continue
        first = safe_float(g.iloc[0].get("Line"), np.nan)
        last = safe_float(g.iloc[-1].get("Line"), np.nan)
        if pd.isna(first) or pd.isna(last):
            continue
        move = round(last - first, 2)
        rows.append({
            "Player": g.iloc[-1].get("Player"),
            "Market": market,
            "Source": src,
            "Opening Line": first,
            "Current Line": last,
            "Line Move": move,
            "Snapshots": len(g),
            "First Seen": g.iloc[0].get("PulledAt"),
            "Last Seen": g.iloc[-1].get("PulledAt"),
            "Sharp Note": "Steam up" if move >= 1 else "Steam down" if move <= -1 else "Stable",
        })
    return pd.DataFrame(rows).sort_values(["Snapshots", "Line Move"], ascending=[False, False]) if rows else pd.DataFrame()


def calibration_report() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Safe model-calibration summary.

    This function must never crash on a fresh Streamlit install or on older
    learning logs. Older logs may not have Projection, Actual, Edge, CLV,
    Lean, Source, or Tier yet, so we create safe placeholder columns before
    grouping/aggregating.
    """
    learn = pd.DataFrame(load_json(LEARNING_LOG, []))
    if learn.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Required text/group columns used by the dashboard.
    for c, default in {
        "Result": "",
        "Market": "UNKNOWN",
        "Lean": "UNKNOWN",
        "Source": "UNKNOWN",
        "Tier": "UNKNOWN",
    }.items():
        if c not in learn.columns:
            learn[c] = default
        learn[c] = learn[c].fillna(default).astype(str)

    # Required numeric columns used by aggregation.
    for c in ["Projection", "Actual", "Line", "Edge", "CLV"]:
        if c not in learn.columns:
            learn[c] = np.nan
        learn[c] = pd.to_numeric(learn[c], errors="coerce")

    learn["Abs Error"] = (learn["Projection"] - learn["Actual"]).abs()
    learn["Bias"] = learn["Projection"] - learn["Actual"]

    group_cols = [c for c in ["Market", "Lean", "Source", "Tier"] if c in learn.columns]
    if not group_cols:
        group_cols = ["Result"]

    # Use only columns we guarantee exist above. This prevents pandas KeyError
    # when old/fresh logs are missing labels.
    summary = learn.groupby(group_cols, dropna=False).agg(
        Plays=("Result", "count"),
        Wins=("Result", lambda x: (x == "WIN").sum()),
        Losses=("Result", lambda x: (x == "LOSS").sum()),
        Pushes=("Result", lambda x: (x == "PUSH").sum()),
        AvgProjection=("Projection", "mean"),
        AvgActual=("Actual", "mean"),
        MAE=("Abs Error", "mean"),
        Bias=("Bias", "mean"),
        AvgEdge=("Edge", "mean"),
        AvgCLV=("CLV", "mean"),
    ).reset_index()

    denom = summary["Wins"] + summary["Losses"]
    summary["Win Rate"] = np.where(denom > 0, summary["Wins"] / denom, np.nan)
    return learn, summary

def build_historical_backtest(logs: pd.DataFrame, min_prior_games: int = 5) -> pd.DataFrame:
    """Backtest the projection formula using historical game logs.

    This is a model-calibration backtest, not a sportsbook-line backtest.
    It creates a fair historical line proxy from prior games only and measures whether
    the model correctly chose over/under that proxy.
    """
    if logs is None or logs.empty:
        return pd.DataFrame()
    d = standardize_player_logs(logs) if "NameKey" not in logs.columns else logs.copy()
    if d.empty:
        return pd.DataFrame()
    d["GameDate"] = pd.to_datetime(d["GameDate"], errors="coerce")
    d = d.sort_values(["NameKey", "GameDate"])
    rows = []
    for market in MARKETS:
        if market not in d.columns:
            continue
        for nk, g in d.groupby("NameKey"):
            g = g.dropna(subset=[market]).sort_values("GameDate")
            if len(g) <= min_prior_games:
                continue
            vals = pd.to_numeric(g[market], errors="coerce").reset_index(drop=True)
            mins = pd.to_numeric(g.get("MIN", pd.Series([np.nan]*len(g))), errors="coerce").reset_index(drop=True)
            for i in range(min_prior_games, len(g)):
                prior = vals.iloc[:i]
                if prior.dropna().shape[0] < min_prior_games:
                    continue
                l3 = prior.tail(3).mean(); l5 = prior.tail(5).mean(); l10 = prior.tail(10).mean(); season = prior.mean()
                proj = 0.15*season + 0.20*l10 + 0.35*l5 + 0.30*l3
                # Synthetic closing line proxy using only prior data.
                line_proxy = 0.55*season + 0.30*l10 + 0.15*l5
                actual = safe_float(vals.iloc[i], np.nan)
                if pd.isna(actual) or pd.isna(proj) or pd.isna(line_proxy):
                    continue
                lean = "OVER" if proj > line_proxy else "UNDER"
                win = (actual > line_proxy and lean == "OVER") or (actual < line_proxy and lean == "UNDER")
                rows.append({
                    "GameDate": g.iloc[i].get("GameDate"),
                    "Player": g.iloc[i].get("Player"),
                    "Team": g.iloc[i].get("Team"),
                    "Market": market,
                    "Projection": round(float(proj), 2),
                    "Line Proxy": round(float(line_proxy), 2),
                    "Actual": round(float(actual), 2),
                    "Lean": lean,
                    "Result": "WIN" if win else "LOSS",
                    "Edge": round(float(proj - line_proxy), 2),
                    "Prior Games": i,
                    "Prior MIN L5": round(float(mins.iloc[:i].tail(5).mean()), 2) if mins.notna().any() else np.nan,
                })
    return pd.DataFrame(rows)


def summarize_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    if bt is None or bt.empty:
        return pd.DataFrame()
    s = bt.groupby(["Market", "Lean"], dropna=False).agg(
        Plays=("Result", "count"),
        Wins=("Result", lambda x: (x == "WIN").sum()),
        AvgEdge=("Edge", "mean"),
        AvgProjection=("Projection", "mean"),
        AvgActual=("Actual", "mean"),
    ).reset_index()
    s["Win Rate"] = s["Wins"] / s["Plays"].replace(0, np.nan)
    return s.sort_values("Win Rate", ascending=False)



# ============================================================
# Advanced Feature Refresh / Missing Column Repair v3
# ============================================================
TEAM_ALIAS_MAP = {
    "ATLANTA DREAM": "ATL", "ATL DREAM": "ATL", "ATL": "ATL",
    "CHICAGO SKY": "CHI", "CHI SKY": "CHI", "CHI": "CHI",
    "CONNECTICUT SUN": "CON", "CONN SUN": "CON", "CON": "CON", "CONN": "CON",
    "DALLAS WINGS": "DAL", "DAL WINGS": "DAL", "DAL": "DAL",
    "GOLDEN STATE VALKYRIES": "GSV", "GOLDEN STATE": "GSV", "GS": "GSV", "GSV": "GSV",
    "INDIANA FEVER": "IND", "IND FEVER": "IND", "IND": "IND",
    "LAS VEGAS ACES": "LVA", "LAS VEGAS": "LVA", "LV": "LVA", "LVA": "LVA", "LAV": "LVA",
    "LOS ANGELES SPARKS": "LAS", "LA SPARKS": "LAS", "LOS ANGELES": "LAS", "LA": "LAS", "LAS": "LAS",
    "MINNESOTA LYNX": "MIN", "MIN LYNX": "MIN", "MIN": "MIN",
    "NEW YORK LIBERTY": "NYL", "NEW YORK": "NYL", "NY LIBERTY": "NYL", "NY": "NYL", "NYL": "NYL",
    "PHOENIX MERCURY": "PHX", "PHX MERCURY": "PHX", "PHOENIX": "PHX", "PHX": "PHX",
    "SEATTLE STORM": "SEA", "SEA STORM": "SEA", "SEA": "SEA",
    "WASHINGTON MYSTICS": "WAS", "WAS MYSTICS": "WAS", "WASHINGTON": "WAS", "WAS": "WAS",
}

REQUIRED_MASTER_FIELDS = [
    "Team_Pace", "Team_DRtg", "Team_NetRtg", "Team_ORtg",
    "ShotAttempts", "ThreePARate", "RimRate", "ShotProfileScore", "PointsPerShot",
    "StarterRate", "RosterGames", "USG%", "TS%", "TS%_Season", "eFG%_Season",
]

def normalize_team_code(x) -> str:
    raw = str(x or "").strip()
    if raw == "" or raw.lower() in ["nan", "none"]:
        return ""
    s = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s).upper().strip()
    s = re.sub(r"\s+", " ", s)
    if s in TEAM_ALIAS_MAP:
        return TEAM_ALIAS_MAP[s]
    if len(s) <= 4:
        return s
    for key, val in TEAM_ALIAS_MAP.items():
        if key in s or s in key:
            return val
    toks = s.split()
    if toks:
        return toks[0][:3]
    return s[:3]


def _safe_series(df: pd.DataFrame, col: str, default=np.nan):
    if isinstance(default, pd.Series):
        return df.get(col, default)
    return df[col] if col in df.columns else pd.Series(default, index=df.index)


def derive_team_ranks_from_logs_and_schedule(logs: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Extra team-context backup when team_season_stats lacks pace/ratings.

    Important fix: avoid duplicate Team columns after groupby/rename. Duplicate columns
    make out["Team"] return a DataFrame, which causes: DataFrame object has no attribute str.
    """
    rows = []
    logs = standardize_player_logs(logs) if logs is not None and not logs.empty else pd.DataFrame()
    if not logs.empty:
        d = logs.copy()
        if "Team" not in d.columns:
            d["Team"] = ""
        d["TeamKey"] = d["Team"].apply(normalize_team_code)
        for c in ["PTS", "REB", "AST", "FGA", "FGM", "FG3M", "FTA", "TOV", "OREB"]:
            if c not in d.columns:
                d[c] = 0
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
        if "Season" not in d.columns:
            d["Season"] = np.nan
        if "GameDate" not in d.columns:
            d["GameDate"] = pd.NaT
        agg = d.groupby(["Season", "TeamKey"], dropna=False).agg(
            Games=("GameDate", "nunique"),
            PTS=("PTS", "sum"), REB=("REB", "sum"), AST=("AST", "sum"),
            FGA=("FGA", "sum"), FGM=("FGM", "sum"), FG3M=("FG3M", "sum"),
            FTA=("FTA", "sum"), TOV=("TOV", "sum"), OREB=("OREB", "sum")
        ).reset_index()
        agg["Team"] = agg["TeamKey"].apply(normalize_team_code)
        agg = agg.drop(columns=["TeamKey"], errors="ignore")
        poss = agg["FGA"] + 0.44 * agg["FTA"] + agg["TOV"] - agg["OREB"]
        agg["Pace"] = np.where(agg["Games"] > 0, poss / agg["Games"], np.nan)
        agg["ORtg"] = np.where(poss > 0, 100 * agg["PTS"] / poss, np.nan)
        agg["eFG%"] = np.where(agg["FGA"] > 0, (agg["FGM"] + 0.5 * agg["FG3M"]) / agg["FGA"], np.nan)
        rows.append(agg)

    sched = standardize_schedules(schedules) if schedules is not None and not schedules.empty else pd.DataFrame()
    if not sched.empty and {"Home", "Away", "HomeScore", "AwayScore"}.issubset(sched.columns):
        sr = []
        complete = sched.dropna(subset=["HomeScore", "AwayScore"]).copy()
        for _, r in complete.iterrows():
            home = normalize_team_code(r.get("Home")); away = normalize_team_code(r.get("Away"))
            if home:
                sr.append({"Season": r.get("Season"), "Team": home, "PtsFor": r.get("HomeScore"), "PtsAllowed": r.get("AwayScore")})
            if away:
                sr.append({"Season": r.get("Season"), "Team": away, "PtsFor": r.get("AwayScore"), "PtsAllowed": r.get("HomeScore")})
        if sr:
            sg = pd.DataFrame(sr).groupby(["Season", "Team"], dropna=False).agg(
                GamesSchedule=("PtsFor", "count"),
                PPG=("PtsFor", "mean"),
                PointsAllowed=("PtsAllowed", "mean")
            ).reset_index()
            sg["Team"] = sg["Team"].apply(normalize_team_code)
            sg["DRtg"] = sg["PointsAllowed"]
            rows.append(sg)

    if not rows:
        return pd.DataFrame()

    clean_rows = []
    for r in rows:
        if r is None or r.empty:
            continue
        rr = r.copy()
        # Drop accidental duplicate-named columns, keeping the first occurrence.
        rr = rr.loc[:, ~pd.Index(rr.columns).duplicated()]
        if "Team" not in rr.columns:
            continue
        rr["Team"] = rr["Team"].apply(normalize_team_code)
        rr = rr[rr["Team"].astype(str).str.len() > 0].copy()
        clean_rows.append(rr)

    if not clean_rows:
        return pd.DataFrame()

    out = pd.concat(clean_rows, ignore_index=True, sort=False)
    out = out.loc[:, ~pd.Index(out.columns).duplicated()]
    out["Team"] = out["Team"].apply(normalize_team_code)
    out = out[out["Team"].astype(str).str.len() > 0].copy()
    out = out.groupby(["Season", "Team"], dropna=False).agg(
        lambda x: x.dropna().iloc[-1] if len(x.dropna()) else np.nan
    ).reset_index()
    if "NetRtg" not in out.columns:
        out["NetRtg"] = np.nan
    if "ORtg" in out.columns and "DRtg" in out.columns:
        out["NetRtg"] = out["NetRtg"].fillna(out["ORtg"] - out["DRtg"])
    return out

def merge_team_context_robust(base: pd.DataFrame, team_ranks: pd.DataFrame, logs: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    if base is None or base.empty:
        return base
    base = base.copy()
    base["TeamKey"] = base.get("Team", "").map(normalize_team_code)
    candidates = []
    tr0 = team_ranks.copy() if team_ranks is not None and not team_ranks.empty else pd.DataFrame()
    if not tr0.empty:
        tr0["Team"] = tr0["Team"].map(normalize_team_code)
        candidates.append(tr0)
    backup = derive_team_ranks_from_logs_and_schedule(logs, schedules)
    if not backup.empty:
        candidates.append(backup)
    if not candidates:
        return base
    tr = pd.concat(candidates, ignore_index=True, sort=False)
    tr = tr[tr["Team"].astype(str).str.len() > 0].copy()
    tr = tr.sort_values("Season").groupby("Team", as_index=False).tail(1)
    tr_pref = tr.add_prefix("Team_")
    stale = [c for c in base.columns if c.startswith("Team_")]
    if stale:
        base = base.drop(columns=stale, errors="ignore")
    base = base.merge(tr_pref, left_on="TeamKey", right_on="Team_Team", how="left")
    return base



def derive_real_team_context_from_logs_schedule(logs: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Build real team pace/ORtg/DRtg/NetRtg from cached player logs + schedule scores.

    This avoids using flat fallback values like Pace=78, DRtg=82, NetRtg=0 when
    SportsDataverse team_season_stats does not expose ratings in the expected schema.
    """
    logs_std = standardize_player_logs(logs) if logs is not None and not logs.empty else pd.DataFrame()
    if logs_std.empty:
        return pd.DataFrame()
    d = logs_std.copy()
    d["TeamKey"] = d.get("Team", "").apply(normalize_team_code)
    for c in ["PTS", "FGA", "FTA", "TOV", "OREB", "REB", "AST"]:
        if c not in d.columns:
            d[c] = 0
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    if "GameDate" not in d.columns:
        d["GameDate"] = pd.NaT
    if "Season" not in d.columns:
        d["Season"] = pd.to_datetime(d["GameDate"], errors="coerce").dt.year
    g = d.groupby(["Season", "TeamKey"], dropna=False).agg(
        Games=("GameDate", "nunique"),
        PTS=("PTS", "sum"), FGA=("FGA", "sum"), FTA=("FTA", "sum"),
        TOV=("TOV", "sum"), OREB=("OREB", "sum"), REB=("REB", "sum"), AST=("AST", "sum")
    ).reset_index().rename(columns={"TeamKey": "Team"})
    g["Team"] = g["Team"].apply(normalize_team_code)
    g = g[g["Team"].astype(str).str.len() > 0].copy()
    poss = g["FGA"] + 0.44 * g["FTA"] + g["TOV"] - g["OREB"]
    g["Pace"] = np.where(g["Games"] > 0, poss / g["Games"], np.nan)
    g["ORtg"] = np.where(poss > 0, 100 * g["PTS"] / poss, np.nan)

    # Defensive side from schedule score results, converted onto the same per-100 scale.
    sched_std = standardize_schedules(schedules) if schedules is not None and not schedules.empty else pd.DataFrame()
    if not sched_std.empty and {"Home", "Away", "HomeScore", "AwayScore"}.issubset(sched_std.columns):
        rows = []
        complete = sched_std.dropna(subset=["HomeScore", "AwayScore"]).copy()
        for _, r in complete.iterrows():
            home = normalize_team_code(r.get("Home")); away = normalize_team_code(r.get("Away"))
            season = r.get("Season")
            if home:
                rows.append({"Season": season, "Team": home, "PtsFor_sched": r.get("HomeScore"), "PtsAllowed": r.get("AwayScore")})
            if away:
                rows.append({"Season": season, "Team": away, "PtsFor_sched": r.get("AwayScore"), "PtsAllowed": r.get("HomeScore")})
        if rows:
            s = pd.DataFrame(rows)
            for c in ["PtsFor_sched", "PtsAllowed"]:
                s[c] = pd.to_numeric(s[c], errors="coerce")
            sagg = s.groupby(["Season", "Team"], dropna=False).agg(
                SchedGames=("PtsAllowed", "count"),
                PPG_sched=("PtsFor_sched", "mean"),
                PointsAllowed=("PtsAllowed", "mean")
            ).reset_index()
            g = g.merge(sagg, on=["Season", "Team"], how="left")
    if "PointsAllowed" not in g.columns:
        g["PointsAllowed"] = np.nan
    # Convert points allowed per game to DRtg using estimated team possessions/game.
    g["DRtg"] = np.where(g["Pace"] > 0, 100 * g["PointsAllowed"] / g["Pace"], np.nan)
    # If schedule data absent, center DRtg around ORtg median but keep team variation using ORtg.
    if g["DRtg"].isna().all():
        med_o = g["ORtg"].dropna().median()
        g["DRtg"] = med_o if pd.notna(med_o) else 100.0
    else:
        med_d = g["DRtg"].dropna().median()
        g["DRtg"] = g["DRtg"].fillna(med_d if pd.notna(med_d) else 100.0)
    g["NetRtg"] = g["ORtg"] - g["DRtg"]
    return g[["Season", "Team", "Pace", "ORtg", "DRtg", "NetRtg", "PointsAllowed"]].copy()


def apply_real_team_context_over_fallbacks(base: pd.DataFrame, logs: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Overwrite flat fallback team metrics with real team-specific metrics when available."""
    if base is None or base.empty:
        return base
    ctx = derive_real_team_context_from_logs_schedule(logs, schedules)
    if ctx.empty:
        return base
    b = base.copy()
    if "TeamKey" not in b.columns:
        b["TeamKey"] = b.get("Team", "").apply(normalize_team_code)
    ctx = ctx.sort_values("Season").groupby("Team", as_index=False).tail(1)
    maps = {
        "Team_Pace": dict(zip(ctx["Team"], ctx["Pace"])),
        "Team_ORtg": dict(zip(ctx["Team"], ctx["ORtg"])),
        "Team_DRtg": dict(zip(ctx["Team"], ctx["DRtg"])),
        "Team_NetRtg": dict(zip(ctx["Team"], ctx["NetRtg"])),
        "Team_PointsAllowed": dict(zip(ctx["Team"], ctx["PointsAllowed"])),
    }
    for c, mp in maps.items():
        calc = b["TeamKey"].map(mp)
        if c not in b.columns:
            b[c] = np.nan
        b[c] = pd.to_numeric(b[c], errors="coerce")
        # Replace if missing OR if current column is a flat fallback with almost no team variation.
        unique_ct = b[c].dropna().round(4).nunique()
        fallback_like = unique_ct <= 1 and c in ["Team_Pace", "Team_DRtg", "Team_NetRtg"]
        b[c] = np.where(calc.notna() & (b[c].isna() | fallback_like), calc, b[c])
    return b

def add_missing_feature_fallbacks(base: pd.DataFrame, logs: pd.DataFrame, season_stats: pd.DataFrame, shots: pd.DataFrame, game_rosters: pd.DataFrame) -> pd.DataFrame:
    """Fill projection-critical columns so the model does not silently weaken."""
    if base is None or base.empty:
        return base
    base = base.copy()
    # First try to repair team-specific pace/rating values from actual logs + schedule scores.
    base = apply_real_team_context_over_fallbacks(base, logs, load_dataset("schedules"))
    for c, default in {"Team_Pace": 78.0, "Team_DRtg": 100.0, "Team_ORtg": 100.0, "Team_NetRtg": 0.0}.items():
        if c not in base.columns:
            base[c] = np.nan
        base[c] = pd.to_numeric(base[c], errors="coerce")
        med = base[c].dropna().median()
        base[c] = base[c].fillna(med if pd.notna(med) else default)

    for c in ["ShotAttempts", "ThreePARate", "RimRate", "MidRangeRate", "ShotMakeRate", "PointsPerShot", "ShotProfileScore"]:
        if c not in base.columns:
            base[c] = np.nan
    games = pd.to_numeric(base.get("Games", 0), errors="coerce").replace(0, np.nan)
    fga = pd.to_numeric(base.get("FGA", np.nan), errors="coerce")
    fg3a = pd.to_numeric(base.get("FG3A", np.nan), errors="coerce")
    fgm = pd.to_numeric(base.get("FGM", np.nan), errors="coerce")
    pts_total_est = pd.to_numeric(base.get("PTS_avg", 0), errors="coerce") * games
    base["ShotAttempts"] = pd.to_numeric(base["ShotAttempts"], errors="coerce").fillna(fga / games).fillna(0)
    base["ThreePARate"] = pd.to_numeric(base["ThreePARate"], errors="coerce").fillna(fg3a / fga.replace(0, np.nan)).fillna(0.22).clip(0, 1)
    pos = base.get("PositionGroup", pd.Series("Unknown", index=base.index)).astype(str)
    rim_default = np.select([pos.eq("Big"), pos.eq("Wing"), pos.eq("Guard")], [0.44, 0.30, 0.22], default=0.28)
    base["RimRate"] = pd.to_numeric(base["RimRate"], errors="coerce").fillna(pd.Series(rim_default, index=base.index)).clip(0, 1)
    base["MidRangeRate"] = pd.to_numeric(base["MidRangeRate"], errors="coerce").fillna((1 - base["ThreePARate"] - base["RimRate"]).clip(0, 1))
    base["ShotMakeRate"] = pd.to_numeric(base["ShotMakeRate"], errors="coerce").fillna(fgm / fga.replace(0, np.nan)).fillna(0.42).clip(0, 1)
    base["PointsPerShot"] = pd.to_numeric(base["PointsPerShot"], errors="coerce").fillna(pts_total_est / fga.replace(0, np.nan)).fillna(0.88)
    base["ShotProfileScore"] = pd.to_numeric(base["ShotProfileScore"], errors="coerce").fillna(
        np.clip(50 + base["ThreePARate"]*10 + base["RimRate"]*16 + (base["ShotMakeRate"]-0.42)*65, 0, 100)
    ).round(1)

    if "RosterGames" not in base.columns:
        base["RosterGames"] = np.nan
    if "StarterRate" not in base.columns:
        base["StarterRate"] = np.nan
    base["RosterGames"] = pd.to_numeric(base["RosterGames"], errors="coerce").fillna(pd.to_numeric(base.get("Games", 0), errors="coerce")).fillna(0)
    min10 = pd.to_numeric(base.get("MIN_l10", base.get("MIN_avg", 0)), errors="coerce").fillna(0)
    starter_proxy = np.select([min10 >= 30, min10 >= 24, min10 >= 18], [0.90, 0.72, 0.42], default=0.15)
    base["StarterRate"] = pd.to_numeric(base["StarterRate"], errors="coerce").fillna(pd.Series(starter_proxy, index=base.index)).clip(0, 1)

    if "USG%" not in base.columns:
        base["USG%"] = np.nan
    base["USG%"] = pd.to_numeric(base["USG%"], errors="coerce")
    usage_proxy = pd.to_numeric(base.get("UsageProxy", np.nan), errors="coerce")
    up_med = usage_proxy.dropna().median()
    usg_proxy_pct = 20 + (usage_proxy - (up_med if pd.notna(up_med) else 10)) * 1.25
    base["USG%"] = base["USG%"].fillna(usg_proxy_pct).fillna(20).clip(5, 40)
    for c in ["TS%", "eFG%"]:
        if c not in base.columns:
            base[c] = np.nan
        base[c] = pd.to_numeric(base[c], errors="coerce")
    if "TS%_Season" not in base.columns:
        base["TS%_Season"] = np.nan
    if "eFG%_Season" not in base.columns:
        base["eFG%_Season"] = np.nan
    # TS%_Season was the last empty field. Fill it from season TS% when available,
    # then from calculated TS%, then a conservative league-average fallback.
    base["TS%"] = base["TS%"].fillna(0.52)
    base["TS%_Season"] = pd.to_numeric(base["TS%_Season"], errors="coerce").fillna(base["TS%"]).fillna(0.52)
    base["eFG%"] = base["eFG%"].fillna(0.46)
    base["eFG%_Season"] = pd.to_numeric(base["eFG%_Season"], errors="coerce").fillna(base["eFG%"])

    if "LineupContinuityScore" not in base.columns:
        base["LineupContinuityScore"] = 50
    base["RoleConfidence"] = np.clip(
        43 + pd.to_numeric(base.get("Games", 0), errors="coerce").fillna(0).clip(0, 25)*1.5
        + min10.clip(0, 36)*0.75 + base["StarterRate"].fillna(0)*18
        + pd.to_numeric(base.get("LineupContinuityScore", 50), errors="coerce").fillna(50)*0.08,
        0, 100
    ).round(1)
    base["MinutesSafetyGrade"] = np.select([min10 >= 30, min10 >= 24, min10 >= 18], ["A", "B", "C"], default="D")
    base["DataScore"] = np.clip(
        30 + pd.to_numeric(base.get("Games", 0), errors="coerce").fillna(0).clip(0, 25)*1.85
        + pd.to_numeric(base.get("MIN_avg", 0), errors="coerce").fillna(0).clip(0, 36)*0.72
        + base["RoleConfidence"].fillna(0)*0.25 + base["ShotProfileScore"].fillna(50)*0.08,
        0, 100
    ).round(1)
    for c in REQUIRED_MASTER_FIELDS:
        if c not in base.columns:
            base[c] = np.nan
    return base


def feature_missing_report(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(master) if master is not None else 0
    for c in REQUIRED_MASTER_FIELDS:
        if master is None or master.empty or c not in master.columns:
            miss = total
            pct = 100.0 if total else 0.0
        else:
            miss = int(master[c].isna().sum())
            pct = round(100*miss/max(total, 1), 1)
        rows.append({"Feature": c, "Missing Rows": miss, "Missing %": pct, "Status": "✅ filled" if miss == 0 and total else "⚠️ check"})
    return pd.DataFrame(rows)


def build_master_features_v3() -> Tuple[pd.DataFrame, pd.DataFrame]:
    logs = load_dataset("player_game_logs")
    ss = load_dataset("player_season_stats")
    ts = load_dataset("team_season_stats")
    sched = load_dataset("schedules")
    rosters = load_dataset("rosters")
    gr = load_dataset("game_rosters")
    lineups = load_dataset("lineups")
    shots = load_dataset("shots")

    team_ranks = build_team_ranks(logs, ts, sched)
    if not team_ranks.empty:
        team_ranks["Team"] = team_ranks["Team"].map(normalize_team_code)
    backup_ranks = derive_team_ranks_from_logs_and_schedule(logs, sched)
    if not backup_ranks.empty:
        team_ranks = pd.concat([team_ranks, backup_ranks], ignore_index=True, sort=False) if not team_ranks.empty else backup_ranks
        team_ranks = team_ranks.groupby(["Season", "Team"], dropna=False).agg(lambda x: x.dropna().iloc[-1] if len(x.dropna()) else np.nan).reset_index()
    save_dataset("team_ranks", team_ranks)

    base = compute_player_baselines(logs, ss, shots, rosters)
    if base.empty:
        return pd.DataFrame(), team_ranks
    base = merge_team_context_robust(base, team_ranks, logs, sched)

    grs = standardize_game_rosters(gr) if gr is not None and not gr.empty else pd.DataFrame()
    if not grs.empty:
        grs["StarterBool"] = grs.get("Starter", False).astype(str).str.lower().isin(["true", "1", "yes", "starter", "started"])
        rconf = grs.groupby("NameKey").agg(RosterGames=("Player", "count"), StarterGames=("StarterBool", "sum")).reset_index()
        rconf["StarterRate"] = rconf["StarterGames"] / rconf["RosterGames"].replace(0, np.nan)
        base = base.drop(columns=[c for c in ["RosterGames", "StarterGames", "StarterRate"] if c in base.columns], errors="ignore").merge(rconf, on="NameKey", how="left")
    lns = standardize_lineups(lineups) if lineups is not None and not lineups.empty else pd.DataFrame()
    if not lns.empty and "LineupText" in lns.columns:
        lns_text = lns["LineupText"].astype(str).map(normalize_name)
        total_lineups = max(len(lns_text), 1)
        mentions = []
        for _, b in base[["NameKey", "Player"]].drop_duplicates().iterrows():
            nk = b.get("NameKey", "")
            cnt = int(lns_text.str.contains(nk, regex=False, na=False).sum()) if nk else 0
            mentions.append({"NameKey": nk, "LineupMentions": cnt, "LineupShare": cnt/total_lineups, "LineupContinuityScore": round(min(100, 35 + 65*(cnt/total_lineups)*8), 1)})
        base = base.drop(columns=[c for c in ["LineupMentions", "LineupShare", "LineupContinuityScore"] if c in base.columns], errors="ignore").merge(pd.DataFrame(mentions), on="NameKey", how="left")

    base = add_missing_feature_fallbacks(base, logs, ss, shots, gr)
    save_dataset("master_features", base)
    feature_missing_report(base).to_csv(DATA_DIR / "wnba_feature_missing_report.csv", index=False)
    return base, team_ranks

# Override the older builder everywhere below this point.
build_master_features = build_master_features_v3


def refresh_data_and_build_advanced_features(dataset_choices=None, seasons=None, include_heavy=True):
    """One-click pipeline used by Data Manager.
    Pulls SportsDataverse, rebuilds team ranks/master features, repairs missing columns, and saves a report.
    """
    if seasons is None:
        seasons = [int(season_last), int(season_now)] if "season_last" in globals() and "season_now" in globals() else [2025, 2026]
    dataset_choices = dataset_choices or ["player_game_logs", "player_season_stats", "team_season_stats", "schedules", "rosters", "game_rosters"]
    if include_heavy:
        for k in ["lineups", "shots"]:
            if k not in dataset_choices:
                dataset_choices.append(k)
    debug = []
    for key in dataset_choices:
        try:
            df, dbg = download_sportsdataverse_dataset(key, seasons)
            if dbg is not None and not dbg.empty:
                debug.extend(dbg.to_dict("records"))
            if df is not None and not df.empty:
                std = standardize_dataset(key, df)
                if std is not None and not std.empty:
                    save_dataset(key, std)
                    debug.append({"dataset": key, "status": "saved/standardized", "rows": len(std), "source": "SportsDataverse direct/manifest"})
                else:
                    debug.append({"dataset": key, "status": "downloaded but standardized empty", "rows": 0})
            else:
                debug.append({"dataset": key, "status": "empty/failed; existing cache kept if present", "rows": 0})
        except Exception as e:
            debug.append({"dataset": key, "status": f"error: {str(e)[:180]}", "rows": 0})
    master, team_ranks = build_master_features()
    audit = feature_missing_report(master)
    audit.to_csv(DATA_DIR / "wnba_feature_missing_report.csv", index=False)
    return master, team_ranks, pd.DataFrame(debug), audit

# ============================================================
# UI
# ============================================================

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800;900&display=swap');
    html, body, [class*="css"] {font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif;}
    .stApp { background: radial-gradient(circle at top,#2b0f55 0,#14051f 42%,#050509 100%); color:#f7f7ff; }
    section[data-testid="stSidebar"] { background:#090712; border-right:1px solid rgba(179,113,255,.28); }
    .block-container { padding-top: 1.2rem; max-width: 1240px; }
    div[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(55,18,92,.94),rgba(16,12,28,.95)); border:1px solid rgba(179,113,255,.50); border-radius:22px; padding:18px; box-shadow:0 0 22px rgba(155,92,255,.16); }
    div[data-testid="stMetric"] label { color:#d6c6ff!important; text-transform:uppercase; font-weight:900; letter-spacing:.08em; }
    div[data-testid="stMetricValue"] { color:#ffffff!important; font-size:2.3rem!important; }
    .stButton>button, .stDownloadButton>button { background:#151024; border:1px solid #6d3cc7; color:#f8f3ff; border-radius:13px; padding:.75rem 1rem; font-weight:800; }
    .stButton>button:hover, .stDownloadButton>button:hover { border-color:#c084fc; color:#ffffff; box-shadow:0 0 16px rgba(192,132,252,.30); }
    .stTabs [data-baseweb="tab-list"] { gap:16px; border-bottom:1px solid rgba(255,255,255,.10); }
    .stTabs [data-baseweb="tab"] { color:#d5c8ff; font-weight:900; letter-spacing:.04em; text-transform:uppercase; padding-left:0; padding-right:0; }
    .stTabs [aria-selected="true"] { color:#d8b4fe!important; border-bottom:5px solid #a855f7!important; }
    .owp-hero {background:linear-gradient(145deg,#32115e 0%,#0b0712 58%,#1b0630 100%); border:1px solid rgba(192,132,252,.65); border-radius:28px; padding:28px; margin: 8px 0 22px 0; box-shadow: inset 0 0 28px rgba(168,85,247,.12), 0 0 36px rgba(168,85,247,.16);}
    .owp-title {font-size:2.25rem; font-weight:1000; line-height:1.18; letter-spacing:.02em; color:#fff; text-transform:uppercase;}
    .owp-subtitle {font-size:1.08rem; color:#e9d5ff; margin-top:10px;}
    .owp-blue-note {background:#1d1233; border-radius:16px; border:1px solid rgba(192,132,252,.40); color:#d8b4fe; padding:16px; margin:16px 0 22px 0; line-height:1.55; font-weight:700;}
    .owp-kpi-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin:18px 0;}
    .owp-kpi-card {background:linear-gradient(145deg,rgba(54,18,91,.98),rgba(12,10,18,.98)); border:1px solid rgba(192,132,252,.45); border-radius:24px; padding:20px; min-height:125px;}
    .owp-kpi-label {color:#d8c8ff; text-transform:uppercase; font-weight:1000; letter-spacing:.08em; font-size:.9rem;}
    .owp-kpi-value {font-size:2.7rem; font-weight:900; color:#fff; margin-top:16px;}
    .owp-kpi-sub {color:#e9d5ff; margin-top:8px; font-size:.92rem;}
    .section-title {font-size:2rem; font-weight:1000; border-left:7px solid #a855f7; padding-left:16px; margin:26px 0 8px 0;}
    .card { background:linear-gradient(145deg,#24113d,#0b0d12); border:1px solid rgba(192,132,252,.48); border-radius:24px; padding:18px; margin:14px 0; box-shadow: 0 0 22px rgba(168,85,247,.12); }
    .badge { display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid #7e57c2; margin-right:6px; font-size:.82rem; color:#f3e8ff; }
    .hot { color:#86efac; font-weight:900; }
    .warn { color:#facc15; font-weight:900; }
    .pass { color:#c4b5fd; font-weight:800; }
    .small-note { color:#d8b4fe; font-size:.86rem; }
    .explain-box { background:rgba(88,28,135,.22); border:1px solid rgba(192,132,252,.30); border-radius:16px; padding:12px; margin-top:10px; }
    .hidden-baseline-note { color:#d8b4fe; font-size:.92rem; margin:8px 0; }
    .owp-header { display:none; }
    @media (max-width: 760px) {
      .owp-title {font-size:1.8rem;}
      .owp-kpi-grid {grid-template-columns:1fr 1fr; gap:12px;}
      .owp-kpi-card {min-height:110px; padding:16px;}
      .owp-kpi-value {font-size:2.2rem;}
      .block-container {padding-left:1rem; padding-right:1rem;}
    }

    /* ===== Clean WNBA player cards inspired by the MLB card, purple theme ===== */
    .owp-card-v2{background:linear-gradient(160deg,rgba(16,13,28,.98),rgba(8,8,14,.98));border:1px solid rgba(168,85,247,.55);border-left:4px solid #a855f7;border-radius:22px;padding:18px 18px 16px;margin:16px 0;box-shadow:0 0 24px rgba(168,85,247,.16);}
    .owp-card-top{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;margin-bottom:12px;}
    .owp-logo{width:48px;height:48px;object-fit:contain;border-radius:12px;background:rgba(255,255,255,.06);border:1px solid rgba(192,132,252,.35);padding:4px;flex:0 0 auto;}
    .owp-player{font-size:1.35rem;font-weight:1000;color:#fff;line-height:1.1;}
    .owp-match{color:#c7b7f5;font-weight:700;margin-top:5px;font-size:.95rem;}
    .owp-pill{display:inline-block;padding:5px 11px;border-radius:999px;font-size:.75rem;font-weight:1000;letter-spacing:.06em;text-transform:uppercase;margin:3px 4px 3px 0;border:1px solid rgba(255,255,255,.16);}
    .owp-pill-source{background:rgba(21,128,61,.18);border-color:#22c55e;color:#bbf7d0;}
    .owp-pill-role{background:rgba(234,179,8,.14);border-color:#eab308;color:#fde68a;}
    .owp-pill-score{background:rgba(168,85,247,.18);border-color:#a855f7;color:#f3e8ff;}
    .owp-market-pill{color:#fb7185;border-color:#fb7185;background:rgba(244,63,94,.10);}
    .owp-decision{font-size:1.35rem;font-weight:1000;text-align:right;line-height:1.1;}
    .owp-decision.over{color:#4ade80}.owp-decision.under{color:#fb7185}.owp-decision.pass{color:#d8b4fe}
    .owp-confidence{margin-top:6px;font-size:.78rem;color:#d8b4fe;font-weight:900;}
    .owp-card-grid{display:grid;grid-template-columns:1.25fr 1fr 1fr;gap:12px;margin:14px 0;}
    .owp-statbox{background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.09);border-radius:15px;padding:12px;}
    .owp-stat-label{font-size:.72rem;color:#b9a9e8;text-transform:uppercase;font-weight:1000;letter-spacing:.07em;}
    .owp-stat-value{font-size:2rem;font-weight:1000;color:#fff;line-height:1.1;margin-top:6px;}
    .owp-stat-sub{font-size:.82rem;color:#c4b5fd;margin-top:4px;}
    .owp-edge-pos{color:#4ade80}.owp-edge-neg{color:#fb7185}.owp-edge-flat{color:#facc15}
    .owp-prob-wrap{margin:12px 0 5px 0;}
    .owp-prob-label{display:flex;justify-content:space-between;color:#a9a2c5;font-size:.78rem;text-transform:uppercase;font-weight:900;letter-spacing:.06em;margin-bottom:6px;}
    .owp-prob-track{height:10px;border-radius:999px;background:rgba(255,255,255,.10);overflow:hidden;}
    .owp-prob-fill{height:10px;border-radius:999px;background:linear-gradient(90deg,#fb7185,#a855f7,#22c55e);}
    .owp-mini-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:10px 0 8px 0;}
    .owp-mini{background:rgba(168,85,247,.08);border:1px solid rgba(168,85,247,.18);border-radius:13px;padding:9px;color:#d8b4fe;}
    .owp-mini b{display:block;color:#fff;font-size:1rem;margin-top:3px;}
    .owp-why{background:rgba(88,28,135,.18);border:1px solid rgba(168,85,247,.24);border-radius:15px;padding:12px;color:#e9d5ff;font-size:.9rem;line-height:1.45;margin-top:10px;}
    .owp-muted{color:#a9a2c5;font-size:.82rem;}
    .owp-expander-note{font-size:.86rem;color:#c4b5fd;}
    @media(max-width:760px){.owp-card-grid{grid-template-columns:1fr}.owp-mini-grid{grid-template-columns:1fr 1fr}.owp-card-top{display:block}.owp-decision{text-align:left;margin-top:10px}}

    </style>
    """, unsafe_allow_html=True)


def render_card(r):
    def _val(x, default="—"):
        try:
            if x is None:
                return default
            if isinstance(x, float) and np.isnan(x):
                return default
            s = str(x)
            return default if s.lower() in ["nan", "none", ""] else s
        except Exception:
            return default
    def _num(x, dec=1, default="—"):
        try:
            v = safe_float(x, np.nan)
            if pd.isna(v):
                return default
            return f"{v:.{dec}f}"
        except Exception:
            return default
    def _side_class(lean, official=""):
        s = f"{lean} {official}".upper()
        if "OVER" in s:
            return "over", "🔥 OVER"
        if "UNDER" in s:
            return "under", "⚠️ UNDER"
        return "pass", "TRACK"
    def _matchup(r):
        matchup = _val(r.get("Matchup"), "")
        if matchup:
            return matchup
        team = _val(r.get("Team"), "")
        opp = _val(r.get("Opponent"), "")
        ha = _val(r.get("HomeAway"), "")
        if team and opp:
            if str(ha).upper().startswith("HOME"):
                return f"{opp} @ {team}"
            if str(ha).upper().startswith("AWAY"):
                return f"{team} @ {opp}"
            return f"{team} vs {opp}"
        return team or "WNBA"

    market = _val(r.get("Market"), "PROP")
    line = _val(r.get("Line"), "NO LINE")
    source = _val(r.get("Source"), "Line Source")
    proj = _num(r.get("Projection"), 2)
    edge_raw = safe_float(r.get("Edge"), np.nan)
    edge = _num(edge_raw, 2)
    edge_cls = "owp-edge-pos" if pd.notna(edge_raw) and edge_raw > 0 else "owp-edge-neg" if pd.notna(edge_raw) and edge_raw < 0 else "owp-edge-flat"
    lean = _val(r.get("Lean"), "TRACK")
    side_cls, side_label = _side_class(lean, r.get("Official"))
    confidence = _num(r.get("Official Play Score"), 0)
    overp = safe_float(r.get("Over %"), np.nan)
    underp = safe_float(r.get("Under %"), np.nan)
    fill = overp if pd.notna(overp) else (100-underp if pd.notna(underp) else 50)
    fill = max(0, min(100, fill))
    matchup = _matchup(r)
    slate = _val(r.get("Slate"), "")
    slate_date = _val(r.get("SlateDate"), "")
    matched = _val(r.get("Matched Player"), r.get("Player"))
    logo_src = get_team_logo_src(r.get("Team"))
    logo_html = f"<img class='owp-logo' src='{logo_src}'/>" if logo_src else f"<div class='owp-logo' style='display:flex;align-items:center;justify-content:center;font-weight:1000;color:#e9d5ff'>{team_abbr_for_logo(r.get('Team'))}</div>"
    why = _val(r.get("Projection Explanation"), "Projection explanation unavailable.")
    pos = _val(r.get("Biggest Positive"), "No major positive isolated.")
    risk = _val(r.get("Biggest Risk"), "No major risk isolated.")
    dist = f"{_num(r.get('Floor'),1)}–{_num(r.get('Ceiling'),1)}"
    med = _num(r.get("Median"), 1)
    vol = _val(r.get("Volatility"), "NA")
    l10 = _num(r.get("L10 Hit%"), 0)
    min_proj = _num(r.get("MIN Proj"), 1)
    role = _num(r.get("Role Confidence"), 0)
    data_score = _num(r.get("Data Score"), 0)
    tier = _val(r.get("Tier"), "Tier —")
    pass_reason = _val(r.get("PASS Reason"), "")

    st.markdown(f"""
    <div class='owp-card-v2'>
      <div class='owp-card-top'>
        <div style='display:flex;align-items:flex-start;gap:12px;'>
          {logo_html}
          <div>
            <div class='owp-player'>{_val(r.get('Player'))}</div>
            <div class='owp-match'>{matchup} <span class='owp-muted'>| {_val(r.get('PositionGroup'), 'Role')} | {slate} {slate_date}</span></div>
            <span class='owp-pill owp-pill-source'>{source}</span>
            <span class='owp-pill owp-pill-role'>Lineup/Role {_val(r.get('FallbackLineupRole'), r.get('Minutes Safety', 'NA'))}</span>
            <span class='owp-pill owp-pill-score'>Score {confidence}/100</span>
          </div>
        </div>
        <div class='owp-decision {side_cls}'>{side_label}<div class='owp-confidence'>Confidence {confidence}%</div></div>
      </div>

      <div class='owp-card-grid'>
        <div class='owp-statbox'>
          <div class='owp-stat-label'>{market} Projection</div>
          <div class='owp-stat-value'>{proj}</div>
          <div class='owp-stat-sub'>Median {med} | Range {dist}</div>
        </div>
        <div class='owp-statbox'>
          <div class='owp-stat-label'>Sportsbook Line</div>
          <div class='owp-stat-value'>{line}</div>
          <div class='owp-stat-sub'>{_val(r.get('Best Over Line'),'—')} best over | {_val(r.get('Best Under Line'),'—')} best under</div>
        </div>
        <div class='owp-statbox'>
          <div class='owp-stat-label'>Edge</div>
          <div class='owp-stat-value {edge_cls}'>{edge}</div>
          <div class='owp-stat-sub'>{tier}</div>
        </div>
      </div>

      <div class='owp-prob-wrap'>
        <div class='owp-prob-label'><span>Over {_num(overp,0)}%</span><span>Under {_num(underp,0)}%</span></div>
        <div class='owp-prob-track'><div class='owp-prob-fill' style='width:{fill:.0f}%'></div></div>
      </div>

      <div class='owp-mini-grid'>
        <div class='owp-mini'>Minutes<b>{min_proj}</b></div>
        <div class='owp-mini'>L10 Hit<b>{l10}%</b></div>
        <div class='owp-mini'>Volatility<b>{vol}</b></div>
        <div class='owp-mini'>Role<b>{role}/100</b></div>
        <div class='owp-mini'>Data<b>{data_score}/100</b></div>
        <div class='owp-mini'>Matched<b>{matched}</b></div>
      </div>

      <div class='owp-why'>
        <b>Why:</b> {why}<br/>
        <b>Positive:</b> {pos}<br/>
        <b>Risk:</b> {risk}<br/>
        <b>Projection matchup used:</b> {_val(r.get('Projection Matchup Used'), matchup)}<br/>
        <b>Opponent context:</b> {_val(r.get('Opponent Context Note'), 'Neutral opponent factor.')}<br/>
        <b>Matchup:</b> {_val(r.get('Defense vs Position'), 'Position/matchup context unavailable.')}<br/>
        <b>Shot profile:</b> {_val(r.get('Shot Profile Note'), 'No shot profile note.')}<br/>
        <span class='owp-expander-note'><b>Notes:</b> {pass_reason}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"Advanced details — {_val(r.get('Player'))} {market}", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("XGBoost", _num(r.get("XGBoost Projection"), 2))
        c2.metric("Similarity", _num(r.get("Similarity Projection"), 2))
        c3.metric("Bayesian", _num(r.get("Bayesian Confidence"), 0) + "%")
        detail_cols = [
            "Projection Note", "Confidence Breakdown", "Model Agreement", "Player Similarity Engine", "Rest Travel Blowout",
            "Fallback Lineup Note", "Projected Rotation Note", "Bench Rotation Note", "Pace Projection Note", "Line Movement Note", "Referee Note", "HomeAway Note",
            "Feature Importance", "Full Engine Note", "Opponent Lineup Note", "Injury Ripple Note", "CLV Note", "Sharp Money Note",
        ]
        for col in detail_cols:
            if col in r and _val(r.get(col), ""):
                st.markdown(f"**{col}:** {_val(r.get(col))}")

def dataset_status_table():
    rows = []
    for k, path in CACHE_FILES.items():
        if path.exists():
            try:
                df = pd.read_csv(path, nrows=5)
                # count fast-ish
                rows.append({"Dataset": k, "Status": "✅ cached", "File": str(path), "Columns": len(df.columns), "Updated": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
            except Exception as e:
                rows.append({"Dataset": k, "Status": f"⚠️ read issue", "File": str(path), "Columns": 0, "Updated": ""})
        else:
            rows.append({"Dataset": k, "Status": "❌ missing", "File": str(path), "Columns": 0, "Updated": ""})
    return pd.DataFrame(rows)



def kpi_card(label: str, value: Any, sub: str = ""):
    st.markdown(f"""
    <div class='owp-kpi-card'>
      <div class='owp-kpi-label'>{label}</div>
      <div class='owp-kpi-value'>{value}</div>
      <div class='owp-kpi-sub'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def hero_panel(board_rows: int = 0, real_lines: int = 0, no_line: int = 0, strong: int = 0):
    st.markdown("""
    <div class='owp-hero'>
      <div class='owp-title'>💜 WNBA PROP ENGINE v1.6<br/>ONE-CLICK REFRESH + CONTEXT ENGINE</div>
      <div class='owp-subtitle'>Strict WNBA-only prop line lock → One-click Refresh Today → Save → Grade</div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 REFRESH TODAY — Schedule + Lines + Board", use_container_width=True, key="hero_refresh_live_board"):
            run_one_click_refresh_today("Today", use_ud)
            st.rerun()
    with c2:
        if st.button("💾 SAVE OFFICIAL BEFORE-GAME SNAPSHOT", use_container_width=True, key="hero_save_official_before"):
            board_path = CACHE_FILES.get("projection_board")
            if board_path and board_path.exists():
                try:
                    board_cache = pd.read_csv(board_path)
                    n = save_officials(board_cache)
                    st.success(f"Saved {n} official plays from the current board.")
                except Exception as e:
                    st.error(f"Save failed: {e}")
            else:
                st.warning("Build or refresh a board first, then save official plays.")
    st.markdown(
        f"""
        <div class='owp-blue-note'>
        ONE WAY PICKZ WNBA v1.4 VERIFIED LEARNING BUILD + SPORTSDATAVERSE DATA MANAGER + MARKET TABS | SAVED OFFICIAL SNAPSHOTS | Last refresh: {st.session_state.get('wnba_last_refresh', 'Not refreshed this session')}
        </div>
        <div class='owp-kpi-grid'>
          <div class='owp-kpi-card'><div class='owp-kpi-label'>Board Rows</div><div class='owp-kpi-value'>{board_rows}</div><div class='owp-kpi-sub'>Current screen</div></div>
          <div class='owp-kpi-card'><div class='owp-kpi-label'>Real Lines</div><div class='owp-kpi-value'>{real_lines}</div><div class='owp-kpi-sub'>Underdog/Manual</div></div>
          <div class='owp-kpi-card'><div class='owp-kpi-label'>No Line</div><div class='owp-kpi-value'>{no_line}</div><div class='owp-kpi-sub'>Tracked only</div></div>
          <div class='owp-kpi-card'><div class='owp-kpi-label'>Strong Signals</div><div class='owp-kpi-value'>{strong}</div><div class='owp-kpi-sub'>Official gate passed</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Production Context Upgrade v3.0
# Confirmed/projected starters, injury automation, opponent rotation,
# market-specific position defense, CLV/opening history, referee context.
# These override earlier neutral/fallback engines without touching the
# working Underdog main-line parser.
# ============================================================
CONFIRMED_LINEUPS_FILE = LOCAL_DIR / "wnba_confirmed_lineups.csv"
PROJECTED_ROTATIONS_FILE = LOCAL_DIR / "wnba_projected_rotations.csv"
POSITION_DEFENSE_FILE = DATA_DIR / "wnba_position_defense_by_market.csv"
ESPN_INJURY_CACHE_FILE = LOCAL_DIR / "wnba_espn_injury_cache.json"
REFEREE_ASSIGNMENTS_FILE = LOCAL_DIR / "wnba_referee_assignments.csv"
GAME_CONTEXT_HISTORY_FILE = DATA_DIR / "wnba_game_context_history.csv"


def _today_yyyymmdd(mode: str = "Today") -> str:
    d = slate_target_date(mode) or datetime.utcnow().date()
    return pd.to_datetime(d).strftime("%Y%m%d")


def _date_key(mode: str = "Today") -> str:
    d = slate_target_date(mode) or datetime.utcnow().date()
    return pd.to_datetime(d).strftime("%Y-%m-%d")


def _safe_read_csv_path(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _first_existing_col(df: pd.DataFrame, aliases: List[str]) -> str:
    return find_col(df, aliases) or ""


def _norm_bool_starter(x: Any) -> bool:
    s = str(x or "").strip().upper()
    return s in {"1", "Y", "YES", "TRUE", "START", "STARTER", "STARTING", "S"}


def pull_espn_wnba_scoreboard_context(mode: str = "Today", force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pull ESPN scoreboard for schedule, injuries when exposed, and basic competition context.
    This is a safe helper: if ESPN changes the payload, it returns cached/empty instead of breaking projections.
    """
    cache_key = f"espn_scoreboard_{_today_yyyymmdd(mode)}"
    if not force and ESPN_INJURY_CACHE_FILE.exists():
        try:
            payload = load_json(ESPN_INJURY_CACHE_FILE, {})
            if isinstance(payload, dict) and payload.get("cache_key") == cache_key:
                events = payload.get("events", [])
                if events:
                    return pd.DataFrame(events), pd.DataFrame(payload.get("injuries", []))
        except Exception:
            pass
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={_today_yyyymmdd(mode)}"
    events_out, injuries_out = [], []
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        js = r.json() if r.status_code == 200 else {}
        for ev in js.get("events", []) or []:
            comps = ev.get("competitions", []) or []
            comp = comps[0] if comps else {}
            competitors = comp.get("competitors", []) or []
            home = away = ""
            for c in competitors:
                team_obj = c.get("team", {}) or {}
                ab = _team_key_for_matchup(team_obj.get("abbreviation") or team_obj.get("shortDisplayName") or team_obj.get("displayName"))
                ha = str(c.get("homeAway", "")).upper()
                if ha == "HOME": home = ab
                elif ha == "AWAY": away = ab
                # ESPN sometimes exposes injuries under competitor.
                for inj in c.get("injuries", []) or []:
                    ath = inj.get("athlete", {}) or {}
                    injuries_out.append({
                        "Player": ath.get("displayName") or ath.get("shortName") or "",
                        "Team": ab,
                        "Status": inj.get("status") or inj.get("type") or "",
                        "Detail": inj.get("details") or inj.get("detail") or "",
                        "Source": "ESPN scoreboard",
                    })
            if home and away:
                events_out.append({
                    "GameID": ev.get("id", ""),
                    "GameDate": ev.get("date", _date_key(mode)),
                    "Away": away,
                    "Home": home,
                    "Matchup": f"{away} @ {home}",
                    "Source": "ESPN scoreboard",
                    "Status": (ev.get("status", {}) or {}).get("type", {}).get("description", ""),
                })
        try:
            save_json(ESPN_INJURY_CACHE_FILE, {"cache_key": cache_key, "events": events_out, "injuries": injuries_out, "fetched_at": now_iso()})
        except Exception:
            pass
    except Exception as e:
        st.session_state["wnba_espn_context_error"] = str(e)[:180]
    return pd.DataFrame(events_out), pd.DataFrame(injuries_out)


def load_automated_injury_table(mode: str = "Today", force: bool = False) -> pd.DataFrame:
    """Combine manual injury JSON/CSV style statuses with ESPN-exposed injuries.
    Manual statuses remain strongest because they are user-controlled.
    """
    rows = []
    # Existing manual JSON list.
    for r in load_json(INJURY_STATUS_FILE, []):
        if isinstance(r, dict):
            rr = dict(r); rr.setdefault("Source", "Manual injury status"); rows.append(rr)
    # Optional uploaded CSV with richer statuses.
    manual_csv = LOCAL_DIR / "wnba_injury_status.csv"
    m = _safe_read_csv_path(manual_csv)
    if not m.empty:
        for _, r in m.iterrows():
            rr = r.to_dict(); rr.setdefault("Source", "Uploaded injury CSV"); rows.append(rr)
    # ESPN scoreboard injuries.
    _, espn_inj = pull_espn_wnba_scoreboard_context(mode, force=force)
    if not espn_inj.empty:
        rows.extend(espn_inj.to_dict("records"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    pc = _first_existing_col(out, ["Player", "player", "athlete", "Name"])
    tc = _first_existing_col(out, ["Team", "team", "TEAM", "team_abbreviation"])
    sc = _first_existing_col(out, ["Status", "status", "GameStatus", "InjuryStatus", "designation"])
    if pc and pc != "Player": out["Player"] = out[pc]
    if tc and tc != "Team": out["Team"] = out[tc]
    if sc and sc != "Status": out["Status"] = out[sc]
    out["NameKey"] = out.get("Player", "").map(normalize_name)
    out["TeamKey"] = out.get("Team", "").map(_team_key_for_matchup)
    out["StatusKey"] = out.get("Status", "").astype(str).str.upper()
    return out.drop_duplicates(subset=["NameKey", "TeamKey", "StatusKey"], keep="last")


def _team_injury_context(team: str) -> Dict[str, Any]:
    """Override earlier neutral injury context with automated/manual combined status."""
    t = _team_key_for_matchup(team)
    inj = load_automated_injury_table(st.session_state.get("wnba_current_mode", "Today"), force=False)
    if inj.empty:
        return {"Out Players": 0, "Questionable Players": 0, "Injury Context Note": "No injury status feed loaded; neutral."}
    d = inj[inj["TeamKey"] == t].copy() if "TeamKey" in inj.columns else inj.copy()
    if d.empty:
        return {"Out Players": 0, "Questionable Players": 0, "Injury Context Note": "No listed injuries for team."}
    out_mask = d["StatusKey"].str.contains("OUT|DOUBTFUL|INACTIVE|SUSPENDED|NOT PLAY", na=False)
    q_mask = d["StatusKey"].str.contains("QUESTIONABLE|GTD|GAME TIME|DAY", na=False)
    outs = d[out_mask]
    qs = d[q_mask & ~out_mask]
    names = [x for x in outs.get("Player", pd.Series(dtype=str)).astype(str).tolist() if x][:4]
    note = f"{len(outs)} out/doubtful, {len(qs)} questionable"
    if names:
        note += ": " + ", ".join(names)
    return {"Out Players": int(len(outs)), "Questionable Players": int(len(qs)), "Injury Context Note": note}


def confirmed_lineup_table(mode: str = "Today", force: bool = False) -> pd.DataFrame:
    """Return confirmed/projected starters and rotation. Uses uploaded confirmed lineup first,
    then SportsDataverse lineups/game_rosters, then top recent minutes from player logs.
    """
    rows = []
    # Highest confidence: uploaded confirmed lineups.
    c = _safe_read_csv_path(CONFIRMED_LINEUPS_FILE)
    if not c.empty:
        pc = _first_existing_col(c, ["Player", "player", "Name", "athlete"])
        tc = _first_existing_col(c, ["Team", "team", "TEAM", "team_abbreviation"])
        sc = _first_existing_col(c, ["Starter", "starter", "Starting", "confirmed", "ConfirmedStarter"])
        if pc and tc:
            for _, r in c.iterrows():
                rows.append({
                    "Player": r.get(pc), "NameKey": normalize_name(r.get(pc)), "Team": _team_key_for_matchup(r.get(tc)),
                    "StarterFlag": _norm_bool_starter(r.get(sc)) if sc else True,
                    "ProjectedMinutes": safe_float(r.get("ProjectedMinutes"), np.nan),
                    "LineupSource": "Confirmed lineup upload", "LineupConfidence": 98,
                })
    # SportsDataverse lineups / game rosters if available.
    for key, source, conf in [("lineups", "SportsDataverse lineups", 88), ("game_rosters", "SportsDataverse game roster", 78)]:
        df = load_dataset(key)
        if df is None or df.empty: continue
        d = df.copy()
        pc = _first_existing_col(d, ["Player", "player", "Name", "athlete_display_name", "display_name"])
        tc = _first_existing_col(d, ["Team", "team", "TEAM", "team_abbreviation"])
        stc = _first_existing_col(d, ["Starter", "starter", "Starting", "start_position", "is_starter"])
        if not pc or not tc: continue
        for _, r in d.iterrows():
            nm = r.get(pc)
            if not str(nm or "").strip(): continue
            rows.append({
                "Player": nm, "NameKey": normalize_name(nm), "Team": _team_key_for_matchup(r.get(tc)),
                "StarterFlag": _norm_bool_starter(r.get(stc)) if stc else False,
                "ProjectedMinutes": safe_float(r.get("MIN"), np.nan),
                "LineupSource": source, "LineupConfidence": conf,
            })
    # Fallback: recent top minutes from master/logs.
    mf = load_dataset("master_features")
    if mf is not None and not mf.empty:
        d = mf.copy()
        if "NameKey" not in d.columns and "Player" in d.columns: d["NameKey"] = d["Player"].map(normalize_name)
        if "Team" in d.columns:
            d["TeamKey"] = d["Team"].map(_team_key_for_matchup)
            min_col = _first_existing_col(d, ["MIN_L10", "MIN_avg", "MIN", "Minutes", "MIN Proj"])
            if min_col:
                d["_min"] = pd.to_numeric(d[min_col], errors="coerce").fillna(0)
                for tm, g in d.sort_values("_min", ascending=False).groupby("TeamKey"):
                    for rank, (_, r) in enumerate(g.head(9).iterrows(), start=1):
                        rows.append({
                            "Player": r.get("Player"), "NameKey": r.get("NameKey"), "Team": tm,
                            "StarterFlag": rank <= 5, "ProjectedMinutes": r.get("_min"),
                            "LineupSource": "Projected from recent minutes", "LineupConfidence": 70 if rank <= 5 else 62,
                        })
    out = pd.DataFrame(rows)
    if out.empty: return out
    out = out.dropna(subset=["NameKey"])
    out = out[out["NameKey"].astype(str).str.len() > 1]
    # Keep highest-confidence row per player/team.
    out = out.sort_values("LineupConfidence", ascending=False).drop_duplicates(["NameKey", "Team"], keep="first")
    try:
        out.to_csv(PROJECTED_ROTATIONS_FILE, index=False)
    except Exception:
        pass
    return out


def _projected_rotation_for_team(team: str) -> Dict[str, Any]:
    """Override: confirmed lineup/upload first; projected rotation fallback second."""
    t = _team_key_for_matchup(team)
    rot = fallback_lineup_rotation_engine(force=False)
    if rot.empty:
        return {"Projected Starters": 0, "Rotation Players": 0, "Rotation Confidence": 50.0, "Projected Rotation Note": "Rotation unavailable; neutral fallback."}
    d = rot[rot["Team"] == t].copy()
    if d.empty:
        return {"Projected Starters": 0, "Rotation Players": 0, "Rotation Confidence": 50.0, "Projected Rotation Note": "No projected rotation matched for team."}
    starters = int(d["StarterFlag"].fillna(False).astype(bool).sum())
    rot_players = int(len(d))
    conf = float(pd.to_numeric(d["LineupConfidence"], errors="coerce").dropna().max() if "LineupConfidence" in d.columns and not d["LineupConfidence"].dropna().empty else 62)
    source = str(d.sort_values("LineupConfidence", ascending=False).iloc[0].get("LineupSource", "rotation proxy"))
    starter_names = ", ".join(d[d["StarterFlag"].fillna(False)].get("Player", pd.Series(dtype=str)).astype(str).head(5).tolist())
    note = f"{source}: {starters} starter proxy, {rot_players} rotation players"
    if starter_names: note += f" ({starter_names})"
    return {"Projected Starters": starters, "Rotation Players": rot_players, "Rotation Confidence": conf, "Projected Rotation Note": note}


def opponent_lineup_adjustment(row: Dict[str, Any], base_row: pd.Series) -> Dict[str, Any]:
    """Override: use projected/confirmed opponent rotation and player position mix."""
    market = str(row.get("Market", "")).upper(); opp = _team_key_for_matchup(row.get("Opponent"))
    if not opp:
        return {"Opponent Lineup Adj": 0.0, "Opponent Lineup Note": "Opponent not available for lineup adjustment."}
    rot = fallback_lineup_rotation_engine(force=False)
    if rot.empty:
        return {"Opponent Lineup Adj": 0.0, "Opponent Lineup Note": "Opponent lineup/rotation unavailable; neutral."}
    d = rot[rot["Team"] == opp].copy()
    if d.empty:
        return {"Opponent Lineup Adj": 0.0, "Opponent Lineup Note": "No opponent rotation rows matched."}
    # Join opponent positions/minutes from master features.
    mf = load_dataset("master_features")
    if mf is not None and not mf.empty:
        m = mf.copy()
        if "NameKey" not in m.columns and "Player" in m.columns: m["NameKey"] = m["Player"].map(normalize_name)
        keep = [c for c in ["NameKey", "PositionGroup", "MIN_L10", "MIN_avg", "StarterRate"] if c in m.columns]
        if "NameKey" in keep:
            d = d.merge(m[keep], on="NameKey", how="left")
    pos = d.get("PositionGroup", pd.Series(dtype=str)).astype(str)
    mins = pd.to_numeric(d.get("ProjectedMinutes", pd.Series(np.nan, index=d.index)), errors="coerce")
    if mins.isna().all(): mins = pd.to_numeric(d.get("MIN_L10", pd.Series(20, index=d.index)), errors="coerce").fillna(20)
    weights = mins.fillna(0).clip(lower=0)
    if weights.sum() <= 0: weights = pd.Series(1.0, index=d.index)
    big_share = float((pos.str.contains("Big", case=False, na=False) * weights).sum() / weights.sum()) if len(d) else np.nan
    guard_share = float((pos.str.contains("Guard", case=False, na=False) * weights).sum() / weights.sum()) if len(d) else np.nan
    wing_share = float((pos.str.contains("Wing", case=False, na=False) * weights).sum() / weights.sum()) if len(d) else np.nan
    adj = 0.0
    if market == "REB" and pd.notna(big_share): adj += max(-0.28, min(0.20, (0.34 - big_share) * 0.85))
    elif market == "AST" and pd.notna(guard_share): adj += max(-0.16, min(0.18, (guard_share - 0.38) * 0.42))
    elif market in ["PTS", "PRA"] and pd.notna(wing_share): adj += max(-0.15, min(0.15, (0.34 - wing_share) * 0.35))
    source = str(d.sort_values("LineupConfidence", ascending=False).iloc[0].get("LineupSource", "rotation proxy")) if "LineupConfidence" in d.columns else "rotation proxy"
    return {
        "Opponent Lineup Adj": round(float(adj), 3),
        "Opponent Lineup Note": f"{opp} {source}; rotation mix Big {big_share:.0%}, Guard {guard_share:.0%}, Wing {wing_share:.0%}" if pd.notna(big_share) else f"{opp} rotation loaded; position mix unavailable.",
    }


def build_position_defense_by_market(force: bool = False) -> pd.DataFrame:
    """Market-specific defense calibration. Calculates how much each team allows by opponent position group.
    It uses cached player logs if opponent/team/position exist; otherwise builds safe team-level fallbacks.
    """
    if POSITION_DEFENSE_FILE.exists() and not force:
        try:
            d = pd.read_csv(POSITION_DEFENSE_FILE)
            if not d.empty: return d
        except Exception:
            pass
    logs = load_dataset("player_game_logs")
    mf = load_dataset("master_features")
    rows = []
    if logs is not None and not logs.empty:
        d = logs.copy()
        if "NameKey" not in d.columns and "Player" in d.columns: d["NameKey"] = d["Player"].map(normalize_name)
        if mf is not None and not mf.empty and "PositionGroup" not in d.columns:
            m = mf.copy()
            if "NameKey" not in m.columns and "Player" in m.columns: m["NameKey"] = m["Player"].map(normalize_name)
            if "NameKey" in m.columns and "PositionGroup" in m.columns:
                d = d.merge(m[["NameKey", "PositionGroup"]].drop_duplicates("NameKey"), on="NameKey", how="left")
        opp_col = _first_existing_col(d, ["Opponent", "Opp", "opponent", "opp_team", "OpponentTeam"])
        if opp_col and "PositionGroup" in d.columns:
            d["DefTeam"] = d[opp_col].map(_team_key_for_matchup)
            d["PositionGroup"] = d["PositionGroup"].fillna("Unknown")
            for market in ["PTS", "REB", "AST", "PRA"]:
                if market not in d.columns: continue
                vals = pd.to_numeric(d[market], errors="coerce")
                tmp = d.assign(_val=vals).dropna(subset=["_val"])
                if tmp.empty: continue
                league_avg = float(tmp["_val"].mean())
                gp = tmp.groupby(["DefTeam", "PositionGroup"], dropna=False)["_val"].agg(["mean", "count"]).reset_index()
                for _, r in gp.iterrows():
                    if not r.get("DefTeam") or safe_float(r.get("count"), 0) < 3: continue
                    allowed = float(r["mean"]); factor = max(0.88, min(1.12, 1 + (allowed - league_avg) / max(league_avg * 12, 1)))
                    rows.append({"Opponent": r["DefTeam"], "PositionGroup": r["PositionGroup"], "Market": market, "AllowedAvg": round(allowed, 3), "LeagueAvg": round(league_avg, 3), "DefenseFactor": round(factor, 4), "Sample": int(r["count"]), "Source": "logs opponent allowed"})
    out = pd.DataFrame(rows)
    if out.empty:
        # Safe fallback from team context; still market-specific labels.
        tc = _team_context_table(force_official=False)
        if tc is not None and not tc.empty and "Team" in tc.columns:
            for _, r in tc.iterrows():
                team = _team_key_for_matchup(r.get("Team"))
                drtg = safe_float(r.get("Team_DRtg_Official"), np.nan)
                for market in ["PTS", "REB", "AST", "PRA"]:
                    factor = 1.0 if pd.isna(drtg) else max(0.90, min(1.10, 1 + (drtg - 100) / 900))
                    out = pd.concat([out, pd.DataFrame([{"Opponent": team, "PositionGroup": "ALL", "Market": market, "AllowedAvg": np.nan, "LeagueAvg": np.nan, "DefenseFactor": round(float(factor), 4), "Sample": 0, "Source": "team DRtg fallback"}])], ignore_index=True)
    try:
        out.to_csv(POSITION_DEFENSE_FILE, index=False)
    except Exception:
        pass
    return out


def position_defense_context(row: Dict[str, Any], base_row: pd.Series) -> Dict[str, Any]:
    opp = _team_key_for_matchup(row.get("Opponent")); market = str(row.get("Market", "")).upper()
    pos = str(row.get("PositionGroup") or base_row.get("PositionGroup") or "ALL")
    d = build_position_defense_by_market(force=False)
    if d.empty or not opp or not market:
        return {"Position Defense Factor": 1.0, "Position Defense Note": "Position-defense calibration unavailable."}
    dd = d[(d["Opponent"].map(_team_key_for_matchup) == opp) & (d["Market"].astype(str).str.upper() == market)].copy()
    if dd.empty:
        return {"Position Defense Factor": 1.0, "Position Defense Note": "No market-specific opponent defense row."}
    exact = dd[dd["PositionGroup"].astype(str).str.upper() == pos.upper()]
    pick = exact.iloc[0] if not exact.empty else dd.iloc[0]
    factor = safe_float(pick.get("DefenseFactor"), 1.0)
    sample = safe_float(pick.get("Sample"), 0)
    return {"Position Defense Factor": round(float(factor), 4), "Position Defense Note": f"{opp} vs {pos} {market}: factor {factor:.3f}, sample {int(sample)} ({pick.get('Source','')})."}


def referee_tendency_engine(row: Dict[str, Any]) -> Dict[str, Any]:
    """Override: supports assignment file plus tendency file; stays neutral if no reliable source."""
    tend = _safe_read_csv_path(REFEREE_FILE)
    assign = _safe_read_csv_path(REFEREE_ASSIGNMENTS_FILE)
    refs = []
    matchup = str(row.get("Matchup") or "")
    if not assign.empty:
        mc = _first_existing_col(assign, ["Matchup", "Game", "game"])
        rc = _first_existing_col(assign, ["Referee", "Official", "Crew", "referee"])
        if mc and rc:
            hit = assign[assign[mc].astype(str).str.upper().str.contains(matchup.upper(), regex=False, na=False)] if matchup else assign.head(0)
            refs = hit[rc].astype(str).tolist() if not hit.empty else []
    if tend.empty:
        note = "No referee tendency CSV loaded; neutral."
        if refs: note = f"Referee assignment loaded ({', '.join(refs[:3])}) but no tendency table; neutral."
        return {"Referee Factor": 0.0, "Referee Note": note}
    factor = 0.0; notes=[]
    d = tend.copy()
    if refs:
        rc = _first_existing_col(d, ["Referee", "Official", "Crew", "referee"])
        if rc:
            dh = d[d[rc].astype(str).isin(refs)]
            if not dh.empty: d = dh
    for col, weight in [("FTA_Index", .30), ("Foul_Index", .25), ("Pace_Index", .20), ("Points_Index", .20), ("HomeBias_Index", .05)]:
        if col in d.columns:
            val = pd.to_numeric(d[col], errors="coerce").dropna().mean()
            if pd.notna(val):
                factor += max(-0.20, min(0.20, (float(val)-100)/100))*weight
                notes.append(f"{col} {val:.1f}")
    market = str(row.get("Market", "")).upper()
    if market == "REB": factor *= .35
    elif market == "AST": factor *= .55
    return {"Referee Factor": round(float(factor), 3), "Referee Note": "; ".join(notes) if notes else "Referee file loaded, no usable indexes; neutral."}


def clv_engine(player: str, market: str, current_line: float, opening_line: float = np.nan) -> Dict[str, Any]:
    """Override: use earliest and latest line history for real opening/CLV when snapshots exist."""
    cur = safe_float(current_line, np.nan)
    hist = pd.DataFrame(load_json(LINE_HISTORY_FILE, []))
    if hist.empty or pd.isna(cur):
        if pd.notna(safe_float(opening_line, np.nan)):
            op = safe_float(opening_line, np.nan); return {"Opening Line": op, "CLV": round(cur-op, 2), "CLV Note": f"Opening {op:g} → current {cur:g}"}
        return {"Opening Line": np.nan, "CLV": np.nan, "CLV Note": "No line history yet; Save/refresh more slates to build CLV."}
    h = hist.copy()
    h["NameKey"] = h.get("Player", "").map(normalize_name)
    h = h[(h["NameKey"] == normalize_name(player)) & (h.get("Market", "").astype(str).str.upper() == str(market).upper())].copy()
    if h.empty:
        return {"Opening Line": np.nan, "CLV": np.nan, "CLV Note": "No matching opening-line history for this player/market."}
    if "SavedAt" in h.columns:
        h = h.sort_values("SavedAt")
    h["LineNum"] = pd.to_numeric(h.get("Line"), errors="coerce")
    h = h.dropna(subset=["LineNum"])
    if h.empty:
        return {"Opening Line": np.nan, "CLV": np.nan, "CLV Note": "Line history found but no numeric lines."}
    opening = float(h.iloc[0]["LineNum"]); last_seen = float(h.iloc[-1]["LineNum"])
    clv = cur - opening
    return {"Opening Line": round(opening, 2), "Last Seen Line": round(last_seen, 2), "CLV": round(clv, 2), "CLV Note": f"Opening {opening:g} → current {cur:g} ({clv:+.1f}); {len(h)} stored snapshots."}


def append_game_context_history(board: pd.DataFrame) -> None:
    """Backtesting/learning support: stores projection rows with context fields before games."""
    if board is None or board.empty: return
    keep = [c for c in ["Player","Team","Opponent","Matchup","Market","Line","Projection","Edge","Lean","Official Play Score","Game Context Factor","Game Context Add","Position Defense Factor","Opponent Lineup Adj","Injury Ripple Bump","Referee Factor","Opening Line","CLV","Source","Start"] if c in board.columns]
    if not keep: return
    snap = board[keep].copy()
    snap["SnapshotAt"] = now_iso()
    try:
        old = pd.read_csv(GAME_CONTEXT_HISTORY_FILE) if GAME_CONTEXT_HISTORY_FILE.exists() else pd.DataFrame()
        pd.concat([old, snap], ignore_index=True).tail(50000).to_csv(GAME_CONTEXT_HISTORY_FILE, index=False)
    except Exception:
        pass


# Override the projection enhancer to include v3 game context and market-specific defense.
_make_projection_board_context_v2_core = _make_projection_board_core

def make_projection_board(lines, logs, base, mode: Optional[str] = None):
    mode = mode or st.session_state.get("wnba_current_mode", "Today")
    if lines is not None and not lines.empty:
        try:
            lines = enrich_board_with_matchups(lines, mode)
            lines = attach_game_context_columns(lines, mode)
        except Exception as _ctx_e:
            st.session_state["wnba_game_context_last_error"] = str(_ctx_e)[:180]
    core = _make_projection_board_context_v2_core(lines, logs, base)
    if core is None or core.empty:
        return core
    try:
        core = enrich_board_with_matchups(core, mode)
        core = attach_game_context_columns(core, mode)
    except Exception as e:
        st.session_state["wnba_context_attach_error"] = str(e)[:180]
    out=[]
    base_df = base if base is not None and not base.empty else load_dataset("master_features")
    for _, r in core.iterrows():
        row = r.to_dict()
        b, score = match_player_base(row.get("Player", ""), base_df)
        if b is None: b = pd.Series(dtype=object)
        xgb = model_prediction_for_row(row); row.update(xgb)
        p0=safe_float(row.get("Projection"), np.nan); px=safe_float(row.get("XGBoost Projection"), np.nan)
        if use_xgb_blend_enabled() and pd.notna(p0) and pd.notna(px):
            row["Ensemble Projection"] = round(0.72*p0 + 0.28*px, 2)
            row["Projection"] = row["Ensemble Projection"]
            row["Edge"] = round(row["Projection"] - safe_float(row.get("Line"), np.nan), 2)
        inj = injury_ripple_engine(row, b); row.update(inj)
        opp = opponent_lineup_adjustment(row, b); row.update(opp)
        ref = referee_tendency_engine(row); row.update(ref)
        trav = latest_travel_context(logs, normalize_name(row.get("Matched Player") or row.get("Player")), row.get("Team")); row.update(trav)
        game_ctx = game_context_projection_engine(row, b, logs, mode); row.update(game_ctx)
        pos_def = position_defense_context(row, b); row.update(pos_def)
        context_add = (
            safe_float(row.get("Injury Ripple Bump"),0) + safe_float(row.get("Opponent Lineup Adj"),0)
            + safe_float(row.get("Referee Factor"),0) + safe_float(row.get("Travel Tax"),0)
            + safe_float(row.get("Game Context Add"),0)
        )
        context_factor = safe_float(row.get("Game Context Factor"), 1.0) * safe_float(row.get("Position Defense Factor"), 1.0)
        if pd.notna(safe_float(row.get("Projection"), np.nan)):
            before = safe_float(row.get("Projection"))
            row["Projection Before Game Context"] = round(before, 2)
            row["Projection"] = round(before * context_factor + context_add, 2)
            row["Edge"] = round(row["Projection"] - safe_float(row.get("Line"), np.nan), 2)
        lean = "OVER" if safe_float(row.get("Edge"), 0) > 0 else "UNDER"
        row["Lean"] = lean
        side_prob = safe_float(row.get("Over %"), 0) if lean == "OVER" else safe_float(row.get("Under %"), 0)
        row.update(ev_kelly_engine(side_prob, -110))
        row.update(clv_engine(row.get("Player"), row.get("Market"), safe_float(row.get("Line"), np.nan), safe_float(row.get("Opening Line"), np.nan)))
        row["Sharp Money Note"] = sharp_money_detector(safe_float(row.get("Line Move"), np.nan), safe_float(row.get("Edge"), np.nan), lean)
        row.update(model_disagreement_full(row))
        official_score = safe_float(row.get("Official Play Score"), 0)
        official_score += max(-8, min(8, safe_float(row.get("EV %"), 0)*0.6))
        if safe_float(row.get("Model Disagreement Score"), 0) > 2.4: official_score -= 8
        if safe_float(row.get("Kelly %"), 0) >= 2: official_score += 3
        # Quality gates for real context.
        if safe_float(row.get("Game Context Score"), 0) >= 80: official_score += 3
        if "unavailable" in str(row.get("Opponent Lineup Note", "")).lower(): official_score -= 3
        if "No injury status" in str(row.get("Injury Context Note", "")): official_score -= 1
        row["Official Play Score"] = round(max(0, min(100, official_score)), 1)
        sim_side = side_prob
        row["Tier"] = tier_grade(row["Official Play Score"], safe_float(row.get("Edge"),0), sim_side, safe_float(row.get("Data Score"),0))
        row["Feature Importance"] = (
            feature_importance_text(row)
            + " | Game Context: " + str(row.get("Game Context Note", ""))
            + " | Position Defense: " + str(row.get("Position Defense Note", ""))
            + " | Opp Rotation: " + str(row.get("Opponent Lineup Note", ""))
            + " | XGB: " + str(row.get("XGBoost Feature Importance", ""))
        )
        row["Full Engine Note"] = "Full context active: Underdog main line + matchup + official/team stats + position defense by market + confirmed/projected rotations + automated/manual injuries + rest/home-away/pace + referee/CLV/backtesting + EV/Kelly."
        out.append(row)
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["Official Play Score", "Edge"], ascending=[False, False])
        save_dataset("projection_board", df)
        append_game_context_history(df)
    return df


# ============================================================
# Streamlit app
# ============================================================
st.set_page_config(page_title="ONE WAY PICKZ WNBA", page_icon="🏀", layout="wide")
inject_css()
st.markdown("<div class='owp-header'>🏀 ONE WAY PICKZ — WNBA Prop Engine</div>", unsafe_allow_html=True)
st.caption(APP_VERSION + " — MLB-style Today/Tomorrow refresh → save before → grade after workflow")

# ----------------------------
# Slate/refresh helpers
# ----------------------------
def slate_target_date(mode: str) -> Optional[date]:
    today = datetime.now().date()
    if mode == "Today":
        return today
    if mode == "Tomorrow":
        return today + timedelta(days=1)
    return None


def line_start_date_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "Start" not in df.columns:
        return pd.Series([pd.NaT] * (0 if df is None else len(df)))
    return pd.to_datetime(df["Start"], errors="coerce", utc=True).dt.tz_convert(None).dt.date


def _slate_team_set(mode: str) -> set:
    """Return teams scheduled for the selected slate.

    This is stricter than the earlier version: for Today/Tomorrow it will try
    cached SportsDataverse first, then ESPN schedule fallback. If no schedule
    can be verified, the caller should not show all Underdog rows as today's
    slate because that mixes today/tomorrow/future props.
    """
    try:
        sched = schedule_for_slate(mode)
        target = slate_target_date(mode)
        if (sched is None or sched.empty) and target is not None and "fetch_espn_wnba_schedule_for_date" in globals():
            espn, _dbg = fetch_espn_wnba_schedule_for_date(target)
            if espn is not None and not espn.empty:
                sched = espn.copy()
        if sched is None or sched.empty:
            return set()
        teams = set()
        for c in ["Away", "Home", "AwayTeam", "HomeTeam", "Team", "Opponent"]:
            if c in sched.columns:
                teams.update([team_abbrev(x) for x in sched[c].dropna().astype(str).tolist()])
        return {t for t in teams if t}
    except Exception:
        return set()


def filter_lines_for_slate(lines: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, str]:
    """Strictly filter sportsbook/manual lines to the selected slate.

    Fixes two issues:
    1) Today and Tomorrow could mix when Underdog rows had no start time.
    2) If schedule context is unavailable, the app used to show all lines for
       Today/Tomorrow. Now it refuses to guess and asks you to use All Lines or
       refresh schedule/context.
    """
    if lines is None or lines.empty:
        return pd.DataFrame(), "No loaded lines."
    if mode == "All Lines":
        return lines.copy(), "All loaded lines shown."
    target = slate_target_date(mode)
    if target is None:
        return lines.copy(), "All loaded lines shown."

    d = lines.copy()
    slate_teams = _slate_team_set(mode)
    start_dates = line_start_date_series(d)
    has_dates = start_dates.notna()

    keep = pd.Series(False, index=d.index)

    # 1) Best case: rows have a real start date.
    if has_dates.any():
        keep = keep | (start_dates == target)

    # 2) Manual rows intentionally saved for this exact slate/date.
    src = d.get("Source", pd.Series("", index=d.index)).astype(str)
    if "Start" in d.columns:
        start_txt = d["Start"].astype(str)
        keep = keep | ((src == "Manual") & (start_txt == str(target)))
    if "SlateDate" in d.columns:
        keep = keep | (d["SlateDate"].astype(str) == str(target))
    if "Slate" in d.columns:
        keep = keep | (d["Slate"].astype(str).str.lower() == str(mode).lower())

    # 3) Underdog WNBA often lacks start time. In that case, only keep rows
    # whose team/opponent/matchup belongs to the verified Today/Tomorrow teams.
    if slate_teams:
        team_match = pd.Series(False, index=d.index)
        for c in ["Team", "Opponent"]:
            if c in d.columns:
                team_match = team_match | d[c].astype(str).map(team_abbrev).isin(slate_teams)
        if "Matchup" in d.columns:
            matchup_text = d["Matchup"].astype(str).str.upper()
            mkeep = pd.Series(False, index=d.index)
            for t in slate_teams:
                mkeep = mkeep | matchup_text.str.contains(t, regex=False, na=False)
            team_match = team_match | mkeep
        # Only apply team fallback to rows without dates; dated rows must match target.
        keep = keep | ((~has_dates) & team_match)

    filtered = d[keep].copy()
    if not filtered.empty:
        if slate_teams:
            return filtered, f"Filtered to {mode.lower()} ({target}) using schedule teams: {', '.join(sorted(slate_teams))}."
        return filtered, f"Filtered to {mode.lower()} ({target}) using line start dates."

    # Do not show every line as Today/Tomorrow when slate could not be verified.
    return pd.DataFrame(), f"No verified {mode.lower()} lines found. Refresh Today/Tomorrow schedule first, or use All Lines to view the full board."


def schedule_for_slate(mode: str) -> pd.DataFrame:
    sched = load_dataset("schedules")
    if sched.empty:
        return pd.DataFrame()
    sched = standardize_schedules(sched)
    target = slate_target_date(mode)
    if target is not None and "GameDate" in sched.columns:
        sched = sched[pd.to_datetime(sched["GameDate"], errors="coerce").dt.date == target].copy()
    keep = [c for c in ["GameDate", "Away", "Home", "AwayScore", "HomeScore", "Margin", "Season", "GameID"] if c in sched.columns]
    return sched[keep].sort_values("GameDate") if keep else sched


def _espn_team_abbrev_from_competitor(comp: Dict[str, Any]) -> str:
    team = comp.get("team", {}) if isinstance(comp, dict) else {}
    for k in ["abbreviation", "shortDisplayName", "displayName", "name", "location"]:
        try:
            key = _team_key_for_matchup(team.get(k)) if "_team_key_for_matchup" in globals() else team_abbrev(team.get(k, ""))
        except Exception:
            key = team_abbrev(team.get(k, ""))
        if key:
            return key
    return ""


def fetch_espn_wnba_schedule_for_date(target_date: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pull today's WNBA schedule from ESPN and return standardized Away/Home rows.

    This is used only for matchup/opponent assignment. It does not touch the
    working Underdog line parser.
    """
    if target_date is None:
        return pd.DataFrame(), pd.DataFrame([{"step":"espn_schedule", "status":"skipped", "message":"No target date"}])
    ymd = target_date.strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={ymd}"
    dbg = []
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0", "Accept":"application/json"})
        dbg.append({"step":"espn_schedule", "status":f"HTTP {r.status_code}", "message":url})
        if r.status_code != 200:
            return pd.DataFrame(), pd.DataFrame(dbg)
        js = r.json()
    except Exception as e:
        dbg.append({"step":"espn_schedule", "status":"error", "message":str(e)[:220]})
        return pd.DataFrame(), pd.DataFrame(dbg)

    rows = []
    for ev in js.get("events", []) or []:
        comp = (ev.get("competitions") or [{}])[0]
        home = away = ""
        hs = as_ = np.nan
        for c in comp.get("competitors", []) or []:
            side = str(c.get("homeAway", "")).lower()
            ab = _espn_team_abbrev_from_competitor(c)
            score = safe_float(c.get("score"), np.nan)
            if side == "home":
                home, hs = ab, score
            elif side == "away":
                away, as_ = ab, score
        if away or home:
            rows.append({
                "GameDate": pd.to_datetime(ev.get("date") or comp.get("date") or str(target_date), errors="coerce"),
                "Season": target_date.year,
                "GameID": ev.get("id") or comp.get("id") or "",
                "Away": away,
                "Home": home,
                "AwayScore": as_,
                "HomeScore": hs,
                "Source": "ESPN",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Margin"] = pd.to_numeric(out["HomeScore"], errors="coerce") - pd.to_numeric(out["AwayScore"], errors="coerce")
    dbg.append({"step":"espn_schedule", "status":"parsed", "message":f"{len(out)} game(s) for {target_date}"})
    return out, pd.DataFrame(dbg)


def update_schedule_cache_with_espn(mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = slate_target_date(mode)
    if target is None:
        return schedule_for_slate(mode), pd.DataFrame([{"step":"schedule_cache", "status":"skipped", "message":"All Lines mode"}])
    espn, dbg = fetch_espn_wnba_schedule_for_date(target)
    if espn is None or espn.empty:
        return schedule_for_slate(mode), dbg
    old = load_dataset("schedules")
    keep_old = old.copy() if old is not None and not old.empty else pd.DataFrame()
    if not keep_old.empty and "GameDate" in keep_old.columns:
        old_dates = pd.to_datetime(keep_old["GameDate"], errors="coerce").dt.date
        keep_old = keep_old[old_dates != target].copy()
    combined = pd.concat([keep_old, espn], ignore_index=True, sort=False) if not keep_old.empty else espn.copy()
    try:
        save_dataset("schedules", standardize_schedules(combined))
    except Exception:
        pass
    return schedule_for_slate(mode), dbg


def refresh_today_pipeline(mode: str, use_ud_flag: bool = True) -> Dict[str, Any]:
    """One-click refresh that mirrors the MLB architecture:
    schedule first → matchups/opponents → existing Underdog parser → projections.
    """
    started = time.time()
    status = {"Mode": mode, "Schedule Loaded": "NO", "Games": 0, "Underdog Lines": 0, "Cards": 0, "Status": "started"}
    sched, sched_dbg = update_schedule_cache_with_espn(mode)
    status["Games"] = 0 if sched is None or sched.empty else len(sched)
    status["Schedule Loaded"] = "YES" if status["Games"] else "NO/CACHED"
    st.session_state["wnba_schedule_debug"] = sched_dbg
    try:
        game_ctx, game_dbg = build_game_context_cache(mode, force_official=True)
        status["Game Context Rows"] = 0 if game_ctx is None or game_ctx.empty else len(game_ctx)
        st.session_state["wnba_game_context_debug"] = game_dbg
        daily_ctx, daily_dbg = build_daily_team_context_cache_v2(mode, force=True)
        status["Daily Context Rows"] = 0 if daily_ctx is None or daily_ctx.empty else len(daily_ctx)
        st.session_state["wnba_daily_team_context_v2_debug"] = daily_dbg
    except Exception as _gce:
        status["Game Context Rows"] = 0
        st.session_state["wnba_game_context_last_error"] = str(_gce)[:180]

    clear_line_pull_caches()
    lines_all, ud_debug, manual_debug = pull_board_lines(use_ud_flag, False, False, "")
    lines, _ = filter_lines_for_slate(lines_all, mode)
    status["Underdog Lines"] = int((lines.get("Source", pd.Series(dtype=str)).astype(str) == "Underdog").sum()) if lines is not None and not lines.empty else 0

    logs = load_dataset("player_game_logs")
    master = load_dataset("master_features")
    if master.empty and not logs.empty:
        try:
            master, _ = build_master_features()
        except Exception:
            master = pd.DataFrame()
    if lines is not None and not lines.empty and not logs.empty and not master.empty:
        try:
            board = make_projection_board(lines, logs, master, mode)
            board["Slate"] = mode
            board["SlateDate"] = str(slate_target_date(mode) or "ALL")
            board = enrich_board_with_matchups(board, mode)
            board = apply_matchup_context_to_board(board)
            board = apply_daily_team_context_v2_to_board(board)
            save_dataset("projection_board", board)
            status["Cards"] = len(board)
            status["Status"] = "complete"
        except Exception as e:
            status["Status"] = f"projection error: {str(e)[:140]}"
    else:
        status["Status"] = "lines/database missing; baseline cards still available"
    status["Seconds"] = round(time.time() - started, 2)
    st.session_state["wnba_refresh_today_status"] = status
    return status





# ============================================================
# Daily Team Context Cache 2.0
# ============================================================
def _latest_logs_with_team_game_totals(logs: pd.DataFrame) -> pd.DataFrame:
    """Convert player logs into team-game totals. This powers daily defensive context."""
    if logs is None or logs.empty:
        return pd.DataFrame()
    d = standardize_player_logs(logs)
    if d.empty:
        return pd.DataFrame()
    d["TeamKey"] = d["Team"].map(_team_key_for_matchup)
    d["OppKey"] = d["Opponent"].map(_team_key_for_matchup) if "Opponent" in d.columns else ""
    d = d[d["TeamKey"].astype(str).str.len() > 0].copy()
    for c in ["PTS","REB","AST","PRA","FGA","FGM","FG3A","FG3M","FTA","TOV","OREB","DREB","MIN"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    agg = d.groupby(["GameDate","Season","TeamKey","OppKey"], dropna=False).agg(
        TeamPTS=("PTS","sum"), TeamREB=("REB","sum"), TeamAST=("AST","sum"),
        TeamPRA=("PRA","sum"), TeamFGA=("FGA","sum"), TeamFGM=("FGM","sum"),
        TeamFG3A=("FG3A","sum"), TeamFG3M=("FG3M","sum"), TeamFTA=("FTA","sum"),
        TeamTOV=("TOV","sum"), TeamOREB=("OREB","sum"), TeamDREB=("DREB","sum"),
        TeamMIN=("MIN","sum"), PlayerRows=("Player","count")
    ).reset_index()
    poss = agg["TeamFGA"] + 0.44*agg["TeamFTA"] + agg["TeamTOV"] - agg["TeamOREB"]
    agg["PossessionsProxy"] = poss.replace(0, np.nan)
    agg["PaceProxy"] = poss
    agg["ORtgProxy"] = np.where(poss > 0, 100*agg["TeamPTS"]/poss, np.nan)
    agg["eFGProxy"] = np.where(agg["TeamFGA"] > 0, (agg["TeamFGM"]+0.5*agg["TeamFG3M"])/agg["TeamFGA"], np.nan)
    agg["TSProxy"] = np.where((2*(agg["TeamFGA"]+0.44*agg["TeamFTA"])) > 0, agg["TeamPTS"]/(2*(agg["TeamFGA"]+0.44*agg["TeamFTA"])), np.nan)
    return agg


def build_daily_team_context_cache_v2(mode: str = "Today", force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Daily opponent defensive profile update.

    Rebuilds team offense/defense, last-5/last-10 form, allowed-by-market, rest, home/away,
    rotation and injury context. It uses cached SportsDataverse logs/schedules/team stats and
    the current slate schedule. This runs during Refresh Today and feeds the prop projection engine.
    """
    target = slate_target_date(mode) or datetime.utcnow().date()
    dbg = []
    logs = load_dataset("player_game_logs")
    sched = schedule_for_slate(mode)
    team_stats = load_dataset("team_season_stats")
    team_ranks = load_dataset("team_ranks")
    team_games = _latest_logs_with_team_game_totals(logs)
    rows = []
    teams = set()
    if sched is not None and not sched.empty:
        ss = standardize_schedules(sched)
        for _, g in ss.iterrows():
            h = _team_key_for_matchup(g.get("Home")); a = _team_key_for_matchup(g.get("Away"))
            if h: teams.add(h)
            if a: teams.add(a)
    if not teams and team_games is not None and not team_games.empty:
        teams = set(team_games["TeamKey"].dropna().astype(str).unique().tolist())
    dbg.append({"Step":"teams found", "Rows":len(teams), "Status":"ok" if teams else "empty"})

    # standardize team season/rank context.
    ts = standardize_team_season(team_stats) if team_stats is not None and not team_stats.empty else pd.DataFrame()
    tr = team_ranks.copy() if team_ranks is not None and not team_ranks.empty else pd.DataFrame()
    if not tr.empty and "Team" in tr.columns:
        tr["TeamKey"] = tr["Team"].map(_team_key_for_matchup)
    if not ts.empty and "Team" in ts.columns:
        ts["TeamKey"] = ts["Team"].map(_team_key_for_matchup)

    for team in sorted([t for t in teams if t]):
        base = {"Team": team, "SlateDate": str(target), "UpdatedAt": now_iso()}
        tg = team_games[team_games["TeamKey"] == team].sort_values("GameDate") if team_games is not None and not team_games.empty else pd.DataFrame()
        if not tg.empty:
            for window, suffix in [(None, "Season"), (10, "L10"), (5, "L5")]:
                dd = tg.tail(window) if window else tg
                base[f"Games_{suffix}"] = int(len(dd))
                base[f"PF_{suffix}"] = float(dd["TeamPTS"].mean())
                base[f"REB_{suffix}"] = float(dd["TeamREB"].mean())
                base[f"AST_{suffix}"] = float(dd["TeamAST"].mean())
                base[f"Pace_{suffix}"] = float(dd["PaceProxy"].mean()) if "PaceProxy" in dd else np.nan
                base[f"ORtg_{suffix}"] = float(dd["ORtgProxy"].mean()) if "ORtgProxy" in dd else np.nan
                base[f"eFG_{suffix}"] = float(dd["eFGProxy"].mean()) if "eFGProxy" in dd else np.nan
                base[f"TS_{suffix}"] = float(dd["TSProxy"].mean()) if "TSProxy" in dd else np.nan
        # allowed profile: rows where this team is opponent.
        if team_games is not None and not team_games.empty and "OppKey" in team_games.columns:
            al = team_games[team_games["OppKey"] == team].sort_values("GameDate")
        else:
            al = pd.DataFrame()
        if not al.empty:
            for window, suffix in [(None, "Season"), (10, "L10"), (5, "L5")]:
                dd = al.tail(window) if window else al
                base[f"Allowed_PTS_{suffix}"] = float(dd["TeamPTS"].mean())
                base[f"Allowed_REB_{suffix}"] = float(dd["TeamREB"].mean())
                base[f"Allowed_AST_{suffix}"] = float(dd["TeamAST"].mean())
                base[f"Allowed_PRA_{suffix}"] = float(dd["TeamPRA"].mean())
                base[f"Allowed_Pace_{suffix}"] = float(dd["PaceProxy"].mean()) if "PaceProxy" in dd else np.nan
                # DRtg proxy = opponent points per 100 poss against this team.
                poss = dd["PossessionsProxy"].replace(0, np.nan)
                base[f"DRtg_{suffix}"] = float((100*dd["TeamPTS"]/poss).mean()) if poss.notna().any() else np.nan
        # team season/rank fallback.
        hit = pd.DataFrame()
        if not tr.empty and "TeamKey" in tr.columns:
            hit = tr[tr["TeamKey"] == team].tail(1)
        if hit.empty and not ts.empty and "TeamKey" in ts.columns:
            hit = ts[ts["TeamKey"] == team].tail(1)
        if not hit.empty:
            row = hit.iloc[-1]
            for src, dst in [("Pace","Official_Pace"),("ORtg","Official_ORtg"),("DRtg","Official_DRtg"),("NetRtg","Official_NetRtg"),("PointsAllowed","Official_PointsAllowed"),("REB","Official_REB"),("AST","Official_AST")]:
                if src in row.index and pd.notna(row.get(src)):
                    base[dst] = safe_float(row.get(src), np.nan)
            for src in ["OffensiveRank","DefensiveRank","NetRank","PaceRank","PointsAllowedRank"]:
                if src in row.index and pd.notna(row.get(src)):
                    base[src] = safe_float(row.get(src), np.nan)
        rest, note = _rest_days_for_team(team, target)
        base["RestDays"] = rest; base["RestNote"] = note; base["BackToBack"] = bool(pd.notna(rest) and rest == 0)
        base.update(_projected_rotation_for_team(team))
        base.update(_team_injury_context(team))
        # Stable blended values used by engines.
        base["Ctx_Pace"] = np.nanmean([safe_float(base.get("Pace_L5"), np.nan), safe_float(base.get("Pace_L10"), np.nan), safe_float(base.get("Official_Pace"), np.nan), 78.0])
        base["Ctx_ORtg"] = np.nanmean([safe_float(base.get("ORtg_L5"), np.nan), safe_float(base.get("ORtg_L10"), np.nan), safe_float(base.get("Official_ORtg"), np.nan), 100.0])
        base["Ctx_DRtg"] = np.nanmean([safe_float(base.get("DRtg_L5"), np.nan), safe_float(base.get("DRtg_L10"), np.nan), safe_float(base.get("Official_DRtg"), np.nan), 100.0])
        base["Ctx_NetRtg"] = base["Ctx_ORtg"] - base["Ctx_DRtg"]
        rows.append(base)
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(DAILY_TEAM_CONTEXT_FILE, index=False)
    st.session_state["wnba_daily_team_context_v2"] = out
    st.session_state["wnba_daily_team_context_v2_debug"] = pd.DataFrame(dbg)
    return out, pd.DataFrame(dbg)


def _team_context_v2_lookup(team: str) -> Dict[str, Any]:
    df = st.session_state.get("wnba_daily_team_context_v2", pd.DataFrame())
    if df is None or df.empty:
        if DAILY_TEAM_CONTEXT_FILE.exists():
            try: df = pd.read_csv(DAILY_TEAM_CONTEXT_FILE, low_memory=False)
            except Exception: df = pd.DataFrame()
    if df is None or df.empty or "Team" not in df.columns:
        return {}
    tk = _team_key_for_matchup(team)
    d = df.copy(); d["_TeamKey"] = d["Team"].map(_team_key_for_matchup)
    hit = d[d["_TeamKey"] == tk]
    return hit.iloc[-1].to_dict() if not hit.empty else {}


def apply_daily_team_context_v2_to_board(board: pd.DataFrame) -> pd.DataFrame:
    """Apply market-specific opponent defensive context and log visible factors."""
    if board is None or board.empty:
        return board
    out = board.copy()
    for c in ["Daily Context Applied","Daily Context Factor","Daily Context Note","Opponent Allowed L5","Opponent Allowed L10","Opponent DRtg L5","Opponent Pace L5"]:
        if c not in out.columns: out[c] = "" if c.endswith("Note") or c.endswith("Applied") else np.nan
    for idx, r in out.iterrows():
        opp = _team_key_for_matchup(r.get("Opponent"))
        market = str(r.get("Market", "")).upper()
        ctx = _team_context_v2_lookup(opp)
        if not ctx or not opp:
            out.at[idx,"Daily Context Applied"] = "NO"
            out.at[idx,"Daily Context Note"] = "Daily opponent profile unavailable; fallback model used."
            continue
        allowed_l5 = safe_float(ctx.get(f"Allowed_{market}_L5"), np.nan)
        allowed_l10 = safe_float(ctx.get(f"Allowed_{market}_L10"), np.nan)
        drtg = safe_float(ctx.get("Ctx_DRtg"), np.nan)
        pace = safe_float(ctx.get("Ctx_Pace"), np.nan)
        base_proj = safe_float(r.get("Projection"), np.nan)
        line = safe_float(r.get("Line"), np.nan)
        factor = 1.0
        notes = []
        if pd.notna(allowed_l5) and pd.notna(allowed_l10):
            # Compare recent allowed to season allowed or sane market baseline.
            league_base = {"PTS": 77.0, "REB": 34.0, "AST": 19.0, "PRA": 130.0}.get(market, np.nan)
            allowed_blend = 0.65*allowed_l5 + 0.35*allowed_l10
            if pd.notna(league_base):
                f = max(0.94, min(1.06, 1 + (allowed_blend - league_base) / (league_base * 18)))
                factor *= f; notes.append(f"{market} allowed factor {f:.3f}")
        if pd.notna(drtg):
            f = max(0.96, min(1.05, 1 + (drtg - 100.0)/900.0))
            factor *= f; notes.append(f"DRtg factor {f:.3f}")
        if pd.notna(pace):
            f = max(0.97, min(1.045, 1 + (pace - 78.0)/700.0))
            factor *= f; notes.append(f"pace factor {f:.3f}")
        if pd.notna(base_proj):
            new_proj = round(float(base_proj) * factor, 2)
            out.at[idx,"Projection"] = new_proj
            if pd.notna(line):
                out.at[idx,"Edge"] = round(new_proj - float(line), 2)
                out.at[idx,"Lean"] = "OVER" if new_proj > float(line) else "UNDER"
                if "Official" in out.columns:
                    out.at[idx,"Official"] = "🔥 OVER" if new_proj > float(line) else "⚠️ UNDER"
        out.at[idx,"Daily Context Applied"] = "YES"
        out.at[idx,"Daily Context Factor"] = round(factor, 4)
        out.at[idx,"Daily Context Note"] = "; ".join(notes) if notes else "Neutral daily context."
        out.at[idx,"Opponent Allowed L5"] = allowed_l5
        out.at[idx,"Opponent Allowed L10"] = allowed_l10
        out.at[idx,"Opponent DRtg L5"] = safe_float(ctx.get("DRtg_L5"), np.nan)
        out.at[idx,"Opponent Pace L5"] = safe_float(ctx.get("Pace_L5"), np.nan)
    return out


def run_one_click_refresh_today(mode: str = "Today", use_ud_flag: bool = True) -> Dict[str, Any]:
    """One button pipeline: rebuild feature cache if data exists, refresh schedule/game context,
    pull Underdog/manual lines, rebuild projection board, and update visible refresh status.

    This intentionally preserves the working Underdog parser/main-line selector and only
    orchestrates existing pieces in the correct order.
    """
    status = {"Mode": mode, "Status": "started"}
    started = time.time()
    try:
        logs = load_dataset("player_game_logs")
        master_before = load_dataset("master_features")
        # Rebuild only when player logs exist. This avoids wiping a valid cache on empty data.
        if logs is not None and not logs.empty:
            try:
                master, team_ranks = build_master_features()
                status["Database"] = f"rebuilt {len(master):,} players"
                status["Team Ranks"] = 0 if team_ranks is None else len(team_ranks)
            except Exception as e:
                status["Database"] = f"kept cached; rebuild issue: {str(e)[:90]}"
        elif master_before is not None and not master_before.empty:
            status["Database"] = f"cached {len(master_before):,} players"
        else:
            status["Database"] = "missing player logs/master"

        pipe_status = refresh_today_pipeline(mode, use_ud_flag)
        status.update(pipe_status or {})
        status["Status"] = status.get("Status", "complete")
    except Exception as e:
        status["Status"] = f"one-click error: {str(e)[:160]}"
    status["Seconds"] = round(time.time() - started, 2)
    st.session_state["wnba_refresh_today_status"] = status
    st.session_state["wnba_last_refresh"] = now_iso()
    return status

def render_refresh_today_status():
    status = st.session_state.get("wnba_refresh_today_status")
    if not status:
        return
    st.markdown("### 🔄 Refresh Today Status")
    c = st.columns(6)
    c[0].metric("Schedule", status.get("Schedule Loaded", "NO"))
    c[1].metric("Games", status.get("Games", 0))
    c[2].metric("Context", status.get("Game Context Rows", 0))
    c[3].metric("Underdog", status.get("Underdog Lines", 0))
    c[4].metric("Cards", status.get("Cards", 0))
    c[5].metric("Seconds", status.get("Seconds", "-"))
    st.caption(f"Status: {status.get('Status')} | Daily Context Rows: {status.get('Daily Context Rows', 0)}")


def clear_line_pull_caches():
    for fn in [fetch_underdog_board]:
        try:
            fn.clear()
        except Exception:
            pass


def pull_board_lines(use_ud_flag: bool, use_sleeper_flag: bool = False, use_odds_api_flag: bool = False, odds_api_key: str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refresh active line sources.

    Active sources: Underdog + saved manual lines. Sleeper/Odds/SportsGameOdds are disabled
    from production flow because they were either blocked, quota-limited, or unavailable for WNBA.
    """
    manual_df = load_manual_lines()
    lines, ud_debug, manual_debug = aggregate_lines(
        use_ud=use_ud_flag,
        use_sleeper=False,
        use_odds_api=False,
        odds_api_key="",
        manual_df=manual_df,
        line_upload_df=pd.DataFrame(),
    )
    st.session_state["wnba_lines_all"] = lines
    st.session_state["wnba_ud_debug"] = ud_debug
    st.session_state["wnba_sl_debug"] = manual_debug
    st.session_state["wnba_last_refresh"] = now_iso()
    try:
        st.session_state["wnba_line_snapshots_added"] = log_line_snapshot(lines, "refresh")
    except Exception as _e:
        st.session_state["wnba_line_snapshots_added"] = 0
    return lines, ud_debug, manual_debug


def get_lines_from_state_or_pull(use_ud_flag: bool, use_sleeper_flag: bool, use_odds_api_flag: bool = False, odds_api_key: str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "wnba_lines_all" not in st.session_state:
        return pull_board_lines(use_ud_flag, use_sleeper_flag, use_odds_api_flag, odds_api_key)
    return (
        st.session_state.get("wnba_lines_all", pd.DataFrame()),
        st.session_state.get("wnba_ud_debug", pd.DataFrame()),
        st.session_state.get("wnba_sl_debug", pd.DataFrame()),
    )



def make_baseline_player_cards(master_global: pd.DataFrame, market: str, limit: int = 40) -> pd.DataFrame:
    if master_global is None or master_global.empty:
        return pd.DataFrame()
    m = market if market in MARKETS else "PRA"
    df = master_global.copy()
    proj_col = f"{m}_l10" if f"{m}_l10" in df.columns else f"{m}_avg"
    if proj_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for _, b in df.sort_values(["DataScore", "RoleConfidence", proj_col], ascending=[False, False, False]).head(limit).iterrows():
        proj = safe_float(b.get(proj_col), np.nan)
        if pd.isna(proj):
            continue
        fake_line = np.nan
        dist = {"Floor": round(max(0, proj*0.75), 2), "Median": round(proj, 2), "Ceiling": round(proj*1.25, 2), "Over %": np.nan, "Under %": np.nan, "Volatility": "TRACK"}
        rows.append({
            "Player": b.get("Player"), "Team": b.get("Team"), "Market": m, "Line": "NO LINE", "Source": "Baseline Only",
            "Projection": round(proj, 2), "Edge": np.nan, "Lean": "TRACK", "Official": "NO LINE / TRACK", "Official Play Score": round(safe_float(b.get("DataScore"), 50), 1),
            "PASS Reason": "No sportsbook line loaded. This card is hidden-baseline tracking only.",
            "PositionGroup": b.get("PositionGroup", "Unknown"), "MIN Proj": round(safe_float(b.get("MIN_l10"), safe_float(b.get("MIN_avg"), np.nan)), 2),
            "Role Confidence": round(safe_float(b.get("RoleConfidence"), 50), 1), "Data Score": round(safe_float(b.get("DataScore"), 50), 1),
            "L5 Hit%": np.nan, "L10 Hit%": np.nan, "L20 Hit%": np.nan,
            "Projection Explanation": f"Baseline {m} projection from {proj_col}. Add Underdog/Manual/Odds API line to turn this into an official play.",
            "Biggest Positive": f"Strong baseline sample: data score {round(safe_float(b.get('DataScore'), 50),1)}.",
            "Biggest Risk": "No active sportsbook line yet, so edge/official decision is not calculated.",
            "Shot Profile Note": f"3PA rate {round(safe_float(b.get('ThreePARate'), np.nan),3) if pd.notna(safe_float(b.get('ThreePARate'), np.nan)) else 'NA'}; make rate {round(safe_float(b.get('ShotMakeRate'), np.nan),3) if pd.notna(safe_float(b.get('ShotMakeRate'), np.nan)) else 'NA'}",
            "Shot Profile Boost": 0.0,
            "Confidence Breakdown": f"Data {round(safe_float(b.get('DataScore'), 50),1)} | Role {round(safe_float(b.get('RoleConfidence'), 50),1)} | Minutes {b.get('MinutesSafetyGrade','NA')}",
            "Model Agreement": "Baseline only",
            "Defense vs Position": f"Position group: {b.get('PositionGroup','Unknown')}",
            "Player Similarity Engine": "Line needed for full similarity edge.",
            "Similarity Projection": np.nan,
            "Rest Travel Blowout": "No line/slate context yet.",
            "Bench Rotation Note": f"Rotation grade {b.get('MinutesSafetyGrade','NA')} from current baseline.",
            "Pace Projection Note": "No slate pace projection without opponent/line.",
            "Line Movement Note": "No current line.",
            "Referee Note": "Neutral.",
            "HomeAway Note": "Line needed for slate context.",
            "Correlation Note": correlation_note_for_market(m),
            "Tier": "Tier 5 — Track",
            **dist
        })
    return pd.DataFrame(rows)


# ============================================================
# Robust matchup/opponent resolver
# ============================================================
WNBA_TEAM_KEY_MAP = {
    # Current WNBA teams and common abbreviations
    "ATL": "ATL", "ATLANTA": "ATL", "ATLANTA DREAM": "ATL",
    "CHI": "CHI", "CHICAGO": "CHI", "CHICAGO SKY": "CHI",
    "CON": "CON", "CONN": "CON", "CONNECTICUT": "CON", "CONNECTICUT SUN": "CON",
    "DAL": "DAL", "DALLAS": "DAL", "DALLAS WINGS": "DAL",
    "GSV": "GSV", "GS": "GSV", "GOLDEN STATE": "GSV", "GOLDEN STATE VALKYRIES": "GSV",
    "IND": "IND", "INDIANA": "IND", "INDIANA FEVER": "IND",
    "LA": "LAS", "LAS": "LAS", "LOS ANGELES": "LAS", "LOS ANGELES SPARKS": "LAS",
    "LVA": "LVA", "LV": "LVA", "LVS": "LVA", "LAS VEGAS": "LVA", "LAS VEGAS ACES": "LVA",
    "MIN": "MIN", "MINNESOTA": "MIN", "MINNESOTA LYNX": "MIN",
    "NY": "NYL", "NYL": "NYL", "NEW YORK": "NYL", "NEW YORK LIBERTY": "NYL",
    "PHX": "PHX", "PHO": "PHX", "PHOENIX": "PHX", "PHOENIX MERCURY": "PHX",
    "SEA": "SEA", "SEATTLE": "SEA", "SEATTLE STORM": "SEA",
    "WAS": "WAS", "WSH": "WAS", "WASHINGTON": "WAS", "WASHINGTON MYSTICS": "WAS",
}


def _clean_team_text(x: Any) -> str:
    s = str(x or "").strip().upper()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^A-Z0-9 @.-]+", " ", s).strip()
    return s


def _team_key_for_matchup(x: Any) -> str:
    """Canonical WNBA team key used only for matchup/opponent matching.
    This is intentionally more aggressive than team_abbrev because sportsbook APIs,
    SportsDataverse, and ESPN often disagree on LA/LAS/LVA/NY/NYL labels.
    """
    s = _clean_team_text(x)
    if not s or s in {"NAN", "NONE", "NULL"}:
        return ""
    if s in WNBA_TEAM_KEY_MAP:
        return WNBA_TEAM_KEY_MAP[s]
    # Remove common suffix/prefix noise but keep city/franchise clues.
    s2 = re.sub(r"\b(WNBA|WOMEN|BASKETBALL|TEAM)\b", "", s).strip()
    if s2 in WNBA_TEAM_KEY_MAP:
        return WNBA_TEAM_KEY_MAP[s2]
    # Handle raw phrases that contain a full team name.
    for name, key in sorted(WNBA_TEAM_KEY_MAP.items(), key=lambda kv: len(kv[0]), reverse=True):
        if len(name) >= 4 and name in s2:
            return key
    # Final fallback: preserve short abbreviations instead of truncating full names blindly.
    return s2[:3]


def _parse_event_teams_from_text(raw: Any) -> Tuple[str, str]:
    """Parse away/home teams from Odds API raw text like 'Washington Mystics @ Las Vegas Aces'."""
    txt = str(raw or "").strip()
    if not txt or txt.lower() in {"nan", "none"}:
        return "", ""
    # Normalize common separators used by sportsbooks/APIs.
    pieces = None
    for sep in [" @ ", " at ", " vs ", " v ", " - "]:
        if sep.lower() in txt.lower():
            # Use regex so case-insensitive split preserves original sides.
            pieces = re.split(re.escape(sep), txt, maxsplit=1, flags=re.IGNORECASE)
            break
    if not pieces or len(pieces) != 2:
        return "", ""
    left, right = pieces[0].strip(), pieces[1].strip()
    away = _team_key_for_matchup(left)
    home = _team_key_for_matchup(right)
    return away, home


def _matchup_from_raw_event(row: pd.Series) -> Dict[str, str]:
    """Use sportsbook event text before schedules. This fixes cases where Odds API pulled
    player props but the cached SportsDataverse schedule date/team format did not match.
    """
    team = _team_key_for_matchup(row.get("Team"))
    away, home = _parse_event_teams_from_text(row.get("Raw"))
    if not away or not home:
        return {}
    if team and team == away:
        return {"Opponent": home, "HomeAway": "AWAY", "Matchup": f"{away} @ {home}", "MatchupSource": "sportsbook event"}
    if team and team == home:
        return {"Opponent": away, "HomeAway": "HOME", "Matchup": f"{away} @ {home}", "MatchupSource": "sportsbook event"}
    # If team is missing/failed, do not guess the player's team from the game. Keep both teams visible.
    return {"Opponent": "", "HomeAway": "", "Matchup": f"{away} @ {home}", "MatchupSource": "sportsbook event - team not matched"}


def _schedule_candidates_for_row(mode: str, row: pd.Series) -> pd.DataFrame:
    """Get schedule candidates using line start date first, then selected slate, then full cached schedule."""
    sched_all = load_dataset("schedules")
    if sched_all is None or sched_all.empty:
        return pd.DataFrame()
    sched_all = standardize_schedules(sched_all)
    # 1) Prefer line start date from sportsbook row.
    start = pd.to_datetime(row.get("Start"), errors="coerce", utc=True)
    if pd.notna(start) and "GameDate" in sched_all.columns:
        line_date = start.tz_convert(None).date()
        d = sched_all[pd.to_datetime(sched_all["GameDate"], errors="coerce").dt.date == line_date].copy()
        if not d.empty:
            return d
    # 2) Use selected slate date.
    d = schedule_for_slate(mode)
    if d is not None and not d.empty:
        return d
    # 3) Full schedule fallback.
    return sched_all


def enrich_board_with_matchups(proj_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Attach Opponent/HomeAway/Matchup and keep the opponent available to the projection engine.

    Resolution priority:
    1. Sportsbook/Odds API event text in Raw, e.g. 'WAS @ LVA'.
    2. Cached SportsDataverse schedule using Start date or Today/Tomorrow mode.
    3. Existing opponent/matchup columns if already present.
    """
    if proj_df is None or proj_df.empty:
        return proj_df
    out = proj_df.copy()
    for c in ["Team", "Opponent", "HomeAway", "Matchup", "Matchup Source"]:
        if c not in out.columns:
            out[c] = ""
    for i, r in out.iterrows():
        team = _team_key_for_matchup(r.get("Team"))

        # First: use sportsbook event text. This is the most reliable for Odds API rows.
        raw_hit = _matchup_from_raw_event(r)
        if raw_hit and raw_hit.get("Matchup"):
            # Only write opponent if the player's team matches one side; always write matchup text.
            if raw_hit.get("Opponent"):
                out.at[i, "Opponent"] = raw_hit.get("Opponent", "")
                out.at[i, "HomeAway"] = raw_hit.get("HomeAway", "")
            out.at[i, "Matchup"] = raw_hit.get("Matchup", "")
            out.at[i, "Matchup Source"] = raw_hit.get("MatchupSource", "sportsbook event")
            if raw_hit.get("Opponent"):
                continue

        # Second: use schedule cache/start-date fallback.
        if not team:
            continue
        sched = _schedule_candidates_for_row(mode, r)
        if sched is None or sched.empty:
            continue
        sched = sched.copy()
        sched["HomeKey"] = sched.get("Home", "").map(_team_key_for_matchup)
        sched["AwayKey"] = sched.get("Away", "").map(_team_key_for_matchup)
        hit = sched[(sched["HomeKey"] == team) | (sched["AwayKey"] == team)]
        if hit.empty:
            continue
        # Prefer the closest date to the sportsbook start if available.
        if "Start" in r and "GameDate" in hit.columns:
            st_dt = pd.to_datetime(r.get("Start"), errors="coerce", utc=True)
            if pd.notna(st_dt):
                dd = pd.to_datetime(hit["GameDate"], errors="coerce", utc=True)
                hit = hit.assign(_date_diff=(dd - st_dt).abs()).sort_values("_date_diff")
        g = hit.iloc[0]
        home_key = _team_key_for_matchup(g.get("Home", ""))
        away_key = _team_key_for_matchup(g.get("Away", ""))
        if team == home_key:
            out.at[i, "Opponent"] = away_key
            out.at[i, "HomeAway"] = "HOME"
            out.at[i, "Matchup"] = f"{away_key} @ {team}"
            out.at[i, "Matchup Source"] = "cached schedule"
        elif team == away_key:
            out.at[i, "Opponent"] = home_key
            out.at[i, "HomeAway"] = "AWAY"
            out.at[i, "Matchup"] = f"{team} @ {home_key}"
            out.at[i, "Matchup Source"] = "cached schedule"

    def _final_matchup(r):
        m = str(r.get("Matchup") or "").strip()
        if m and m.lower() not in {"nan", "none"}:
            return m
        t = _team_key_for_matchup(r.get("Team"))
        o = _team_key_for_matchup(r.get("Opponent"))
        return f"{t} vs {o}" if t and o else t
    out["Team"] = out["Team"].map(lambda x: _team_key_for_matchup(x) or str(x or ""))
    out["Opponent"] = out["Opponent"].map(lambda x: _team_key_for_matchup(x) if str(x or "").strip() else "")
    out["Matchup"] = out.apply(_final_matchup, axis=1)
    return out


def _latest_team_context(team: str, season: Any = None) -> Dict[str, Any]:
    """Return cached team-rank context for matchup-aware projections."""
    tr = load_dataset("team_ranks")
    if tr is None or tr.empty:
        return {}
    d = tr.copy()
    team_key = _team_key_for_matchup(team)
    if "Team" not in d.columns:
        return {}
    d["_TeamKey"] = d["Team"].map(_team_key_for_matchup)
    d = d[d["_TeamKey"] == team_key]
    if d.empty:
        return {}
    if season is not None and "Season" in d.columns:
        ss = pd.to_numeric(d["Season"], errors="coerce")
        try:
            wanted = float(season)
            dd = d[ss == wanted]
            if not dd.empty:
                d = dd
        except Exception:
            pass
    if "Season" in d.columns:
        d = d.sort_values("Season")
    return d.iloc[-1].to_dict()


def _market_matchup_adjustment(row: pd.Series, opp_ctx: Dict[str, Any]) -> Tuple[float, str]:
    """Small transparent projection adjustment using the actual opponent on the card."""
    if not opp_ctx:
        return 1.0, "Opponent context unavailable; neutral matchup factor used."
    market = str(row.get("Market", "")).upper()
    factors = []
    notes = []
    pace = safe_float(opp_ctx.get("Pace"), np.nan)
    drtg = safe_float(opp_ctx.get("DRtg"), np.nan)
    pts_allowed = safe_float(opp_ctx.get("PointsAllowed"), np.nan)
    def_rank = safe_float(opp_ctx.get("DefensiveRank"), np.nan)
    pace_rank = safe_float(opp_ctx.get("PaceRank"), np.nan)
    # Conservative factors so matchup context improves the projection without overwhelming player baseline.
    if pd.notna(pace):
        pace_factor = max(0.96, min(1.04, 1 + (pace - 78.0) / 700.0))
        factors.append(pace_factor)
        notes.append(f"opp pace factor {pace_factor:.3f}")
    if pd.notna(drtg):
        # Higher DRtg / points allowed = easier defense.
        def_factor = max(0.96, min(1.04, 1 + (drtg - 100.0) / 900.0))
        factors.append(def_factor)
        notes.append(f"opp defense factor {def_factor:.3f}")
    elif pd.notna(pts_allowed):
        pa_factor = max(0.96, min(1.04, 1 + (pts_allowed - 80.0) / 600.0))
        factors.append(pa_factor)
        notes.append(f"points allowed factor {pa_factor:.3f}")
    if pd.notna(def_rank):
        # WNBA league size is small; larger defensive rank generally means weaker defense if rank was built ascending.
        rank_factor = max(0.97, min(1.03, 1 + (def_rank - 6.5) / 250.0))
        factors.append(rank_factor)
        notes.append(f"def rank factor {rank_factor:.3f}")
    if market == "REB":
        reb_rank = safe_float(opp_ctx.get("ReboundRank"), np.nan)
        if pd.notna(reb_rank):
            f = max(0.97, min(1.03, 1 + (6.5 - reb_rank) / 280.0))
            factors.append(f); notes.append(f"rebound env {f:.3f}")
    if market == "AST":
        ast_rank = safe_float(opp_ctx.get("AssistRank"), np.nan)
        if pd.notna(ast_rank):
            f = max(0.97, min(1.03, 1 + (ast_rank - 6.5) / 280.0))
            factors.append(f); notes.append(f"assist env {f:.3f}")
    if not factors:
        return 1.0, "Opponent found, but no usable pace/defense fields; neutral matchup factor used."
    factor = float(np.prod(factors))
    factor = max(0.92, min(1.08, factor))
    return factor, "; ".join(notes)


def apply_matchup_context_to_board(proj_df: pd.DataFrame) -> pd.DataFrame:
    """Use Opponent/HomeAway/Matchup to adjust projections and visibly confirm matchup used."""
    if proj_df is None or proj_df.empty:
        return proj_df
    out = proj_df.copy()
    for c in ["Opponent", "Matchup", "HomeAway", "Projection Matchup Used", "Opponent Context Note", "Matchup Projection Factor"]:
        if c not in out.columns:
            out[c] = "" if c != "Matchup Projection Factor" else 1.0
    for idx, r in out.iterrows():
        opp = r.get("Opponent")
        team = r.get("Team")
        if not opp or str(opp).strip() in ["", "nan", "None"]:
            out.at[idx, "Projection Matchup Used"] = f"{team or ''} — opponent unavailable"
            out.at[idx, "Opponent Context Note"] = "No opponent matched from schedule; projection remains player/market based."
            out.at[idx, "Matchup Projection Factor"] = 1.0
            continue
        opp_ctx = _latest_team_context(opp, None)
        factor, note = _market_matchup_adjustment(r, opp_ctx)
        if pd.notna(safe_float(r.get("Game Context Factor"), np.nan)) and str(r.get("Game Context Note", "")).strip():
            factor = 1.0
            note = str(r.get("Game Context Note"))
        old_proj = safe_float(r.get("Projection"), np.nan)
        line = safe_float(r.get("Line"), np.nan)
        # Apply once only; if already marked, don't double-adjust.
        already = str(r.get("Opponent Context Applied", "")).lower() == "yes"
        if pd.notna(old_proj) and not already:
            new_proj = round(float(old_proj) * factor, 2)
            out.at[idx, "Raw Projection Before Matchup"] = round(float(old_proj), 2)
            out.at[idx, "Projection"] = new_proj
            if pd.notna(line):
                out.at[idx, "Edge"] = round(new_proj - float(line), 2)
                out.at[idx, "Lean"] = "OVER" if new_proj > float(line) else "UNDER"
                if "Official" in out.columns:
                    out.at[idx, "Official"] = "🔥 OVER" if new_proj > float(line) else "⚠️ UNDER"
        out.at[idx, "Opponent Context Applied"] = "YES"
        out.at[idx, "Matchup Projection Factor"] = round(factor, 4)
        out.at[idx, "Projection Matchup Used"] = str(r.get("Matchup") or f"{team} vs {opp}")
        out.at[idx, "Opponent Context Note"] = note
        # Make the explanation card explicitly say the opponent was used.
        base_exp = str(r.get("Projection Explanation", "") or "")
        add = f" Matchup used: {out.at[idx, 'Projection Matchup Used']} ({note})."
        if "Matchup used:" not in base_exp:
            out.at[idx, "Projection Explanation"] = (base_exp + add).strip()
    return out


def filter_projection_view(proj_df: pd.DataFrame, view_name: str) -> pd.DataFrame:
    """Toggle between official/top plays and all pulled board rows."""
    if proj_df is None or proj_df.empty:
        return pd.DataFrame()
    df = proj_df.copy()
    view_name = str(view_name)
    if view_name.startswith("Official"):
        mask = df.get("Official", pd.Series("", index=df.index)).astype(str).str.contains("OVER|UNDER", case=False, na=False)
        df = df[mask].copy()
        sort_cols = [c for c in ["Official Play Score", "Edge"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=False)
        return df
    if view_name.startswith("Strong"):
        mask = pd.to_numeric(df.get("Official Play Score", 0), errors="coerce").fillna(0) >= 70
        return df[mask].sort_values([c for c in ["Official Play Score", "Edge"] if c in df.columns], ascending=False)
    if view_name.startswith("Overs"):
        return df[df.get("Lean", pd.Series("", index=df.index)).astype(str).str.contains("OVER", case=False, na=False)]
    if view_name.startswith("Unders"):
        return df[df.get("Lean", pd.Series("", index=df.index)).astype(str).str.contains("UNDER", case=False, na=False)]
    if view_name.startswith("Pass"):
        official = df.get("Official", pd.Series("", index=df.index)).astype(str)
        return df[~official.str.contains("OVER|UNDER", case=False, na=False)]
    return df


# ============================================================
# Final opponent/matchup resolver v3
# ============================================================
def _infer_team_for_player_from_cache(player: Any) -> str:
    """Infer a player's current team from the master features/player logs cache.
    This fixes Odds API rows that include event teams but leave the prop-row Team blank.
    """
    nk = normalize_name(player)
    if not nk:
        return ""
    # 1) Master features current team.
    for dataset_key in ["master_features", "player_game_logs", "player_season_stats", "rosters", "game_rosters"]:
        try:
            df = load_dataset(dataset_key)
        except Exception:
            df = pd.DataFrame()
        if df is None or df.empty:
            continue
        d = df.copy()
        if "NameKey" not in d.columns:
            player_col = find_col(d, ["Player", "PLAYER_NAME", "player_name", "athlete_display_name", "name"])
            if player_col:
                d["NameKey"] = d[player_col].map(normalize_name)
        if "Team" not in d.columns:
            team_col = find_col(d, ["Team", "TEAM", "team", "team_abbreviation", "team_name", "team_short_display_name"])
            if team_col:
                d["Team"] = d[team_col]
        if "NameKey" not in d.columns or "Team" not in d.columns:
            continue
        hit = d[d["NameKey"] == nk].copy()
        if hit.empty:
            # Fuzzy fallback only if exact name key fails.
            names = d[["NameKey", "Team"]].dropna().drop_duplicates("NameKey")
            if not names.empty:
                names["_score"] = names["NameKey"].map(lambda x: difflib.SequenceMatcher(None, nk, str(x)).ratio())
                top = names.sort_values("_score", ascending=False).head(1)
                if not top.empty and float(top.iloc[0]["_score"]) >= 0.88:
                    team = _team_key_for_matchup(top.iloc[0].get("Team"))
                    if team:
                        return team
            continue
        if "Season" in hit.columns:
            hit["_season_num"] = pd.to_numeric(hit["Season"], errors="coerce")
            hit = hit.sort_values("_season_num")
        elif "GameDate" in hit.columns:
            hit["_date_sort"] = pd.to_datetime(hit["GameDate"], errors="coerce")
            hit = hit.sort_values("_date_sort")
        teams = [ _team_key_for_matchup(x) for x in hit["Team"].dropna().astype(str).tolist() ]
        teams = [t for t in teams if t]
        if teams:
            return teams[-1]
    return ""


def _event_teams_from_row_anywhere(row: pd.Series) -> Tuple[str, str, str]:
    """Return away, home, source from Odds API event columns, raw text, or existing matchup text.
    Priority: explicit EventAway/EventHome > Away/Home > Raw event text > Matchup text.
    """
    for away_col, home_col, label in [
        ("EventAway", "EventHome", "odds api event columns"),
        ("Away", "Home", "event columns"),
        ("AwayTeam", "HomeTeam", "event columns"),
        ("away_team", "home_team", "event columns"),
    ]:
        away = _team_key_for_matchup(row.get(away_col)) if away_col in row.index else ""
        home = _team_key_for_matchup(row.get(home_col)) if home_col in row.index else ""
        if away and home:
            return away, home, label
    for col, label in [("Raw", "sportsbook raw event"), ("Event", "sportsbook event"), ("Matchup", "existing matchup")]:
        if col in row.index:
            away, home = _parse_event_teams_from_text(row.get(col))
            if away and home:
                return away, home, label
    return "", "", ""


def _resolve_matchup_for_board_row(row: pd.Series, mode: str) -> Dict[str, str]:
    """Resolve Team/Opponent/HomeAway/Matchup for one projected row.
    This intentionally uses several fallbacks because line providers may omit team while
    SportsDataverse can use different abbreviations.
    """
    team = _team_key_for_matchup(row.get("Team"))
    if not team:
        team = _infer_team_for_player_from_cache(row.get("Player"))

    away, home, event_source = _event_teams_from_row_anywhere(row)
    if away and home:
        if team == away:
            return {"Team": team, "Opponent": home, "HomeAway": "AWAY", "Matchup": f"{away} @ {home}", "Matchup Source": event_source}
        if team == home:
            return {"Team": team, "Opponent": away, "HomeAway": "HOME", "Matchup": f"{away} @ {home}", "Matchup Source": event_source}
        # If team did not match but is missing/unknown, keep event text but don't guess opponent.
        return {"Team": team, "Opponent": "", "HomeAway": "", "Matchup": f"{away} @ {home}", "Matchup Source": f"{event_source} - player team not matched"}

    # Schedule fallback by start/slate date.
    if team:
        sched = _schedule_candidates_for_row(mode, row)
        if sched is not None and not sched.empty:
            s = sched.copy()
            s["HomeKey"] = s.get("Home", "").map(_team_key_for_matchup)
            s["AwayKey"] = s.get("Away", "").map(_team_key_for_matchup)
            hit = s[(s["HomeKey"] == team) | (s["AwayKey"] == team)].copy()
            if not hit.empty:
                st_dt = pd.to_datetime(row.get("Start"), errors="coerce", utc=True)
                if pd.notna(st_dt) and "GameDate" in hit.columns:
                    dd = pd.to_datetime(hit["GameDate"], errors="coerce", utc=True)
                    hit = hit.assign(_date_diff=(dd - st_dt).abs()).sort_values("_date_diff")
                g = hit.iloc[0]
                home_key = _team_key_for_matchup(g.get("Home"))
                away_key = _team_key_for_matchup(g.get("Away"))
                if team == away_key:
                    return {"Team": team, "Opponent": home_key, "HomeAway": "AWAY", "Matchup": f"{away_key} @ {home_key}", "Matchup Source": "cached schedule"}
                if team == home_key:
                    return {"Team": team, "Opponent": away_key, "HomeAway": "HOME", "Matchup": f"{away_key} @ {home_key}", "Matchup Source": "cached schedule"}

    # Existing opponent fallback.
    opp = _team_key_for_matchup(row.get("Opponent"))
    if team and opp and team != opp:
        return {"Team": team, "Opponent": opp, "HomeAway": str(row.get("HomeAway") or ""), "Matchup": str(row.get("Matchup") or f"{team} vs {opp}"), "Matchup Source": "existing columns"}
    return {"Team": team, "Opponent": "", "HomeAway": "", "Matchup": team or "", "Matchup Source": "unresolved"}


def enrich_board_with_matchups(proj_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Attach real opponent context to every board row.

    Fixes the prior issue where cards showed only `WAS`/team with `opponent unavailable` even when
    the Odds API row carried the game event. This function now reads EventAway/EventHome, Raw event text,
    cached schedule, and player-team cache before giving up.
    """
    if proj_df is None or proj_df.empty:
        return proj_df
    out = proj_df.copy()
    for c in ["Team", "Opponent", "HomeAway", "Matchup", "Matchup Source"]:
        if c not in out.columns:
            out[c] = ""
    for idx, row in out.iterrows():
        resolved = _resolve_matchup_for_board_row(row, mode)
        for k, v in resolved.items():
            out.at[idx, k] = v

    # Last-resort active-slate fallback: when today's pulled board contains exactly
    # two WNBA teams (common one-game slate), assign each team the other as opponent.
    # This mirrors the MLB app behavior where the schedule/game context is the source
    # of truth and prevents cards from saying "opponent unavailable" when the live
    # board itself clearly contains both sides.
    team_keys = [ _team_key_for_matchup(x) for x in out.get("Team", pd.Series(dtype=str)).dropna().astype(str).tolist() ]
    team_keys = sorted({t for t in team_keys if t})
    if len(team_keys) == 2:
        a, b = team_keys[0], team_keys[1]
        # Prefer cached schedule order if available, otherwise use alphabetical display.
        matchup_text = f"{a} @ {b}"
        try:
            sched_now = schedule_for_slate(mode)
            if sched_now is not None and not sched_now.empty:
                ss = sched_now.copy()
                ss["AwayKey"] = ss.get("Away", "").map(_team_key_for_matchup)
                ss["HomeKey"] = ss.get("Home", "").map(_team_key_for_matchup)
                hit = ss[((ss["AwayKey"] == a) & (ss["HomeKey"] == b)) | ((ss["AwayKey"] == b) & (ss["HomeKey"] == a))]
                if not hit.empty:
                    aa = _team_key_for_matchup(hit.iloc[0].get("Away"))
                    hh = _team_key_for_matchup(hit.iloc[0].get("Home"))
                    if aa and hh:
                        matchup_text = f"{aa} @ {hh}"
        except Exception:
            pass
        away_key, home_key = _parse_event_teams_from_text(matchup_text)
        for idx, row in out.iterrows():
            if _team_key_for_matchup(row.get("Opponent")):
                continue
            t = _team_key_for_matchup(row.get("Team"))
            if t == a:
                out.at[idx, "Opponent"] = b
                out.at[idx, "HomeAway"] = "AWAY" if t == away_key else "HOME" if t == home_key else ""
                out.at[idx, "Matchup"] = matchup_text
                out.at[idx, "Matchup Source"] = "active two-team board fallback"
            elif t == b:
                out.at[idx, "Opponent"] = a
                out.at[idx, "HomeAway"] = "AWAY" if t == away_key else "HOME" if t == home_key else ""
                out.at[idx, "Matchup"] = matchup_text
                out.at[idx, "Matchup Source"] = "active two-team board fallback"
    # Canonicalize final display columns.
    out["Team"] = out["Team"].map(lambda x: _team_key_for_matchup(x) or str(x or ""))
    out["Opponent"] = out["Opponent"].map(lambda x: _team_key_for_matchup(x) if str(x or "").strip() else "")
    def _display_matchup(r):
        m = str(r.get("Matchup") or "").strip()
        if "@" in m:
            return m
        t = _team_key_for_matchup(r.get("Team"))
        o = _team_key_for_matchup(r.get("Opponent"))
        if t and o:
            ha = str(r.get("HomeAway") or "").upper()
            return f"{t} vs {o}" if ha == "HOME" else f"{t} @ {o}" if ha == "AWAY" else f"{t} vs {o}"
        return t or m
    out["Matchup"] = out.apply(_display_matchup, axis=1)
    return out


def apply_matchup_context_to_board(proj_df: pd.DataFrame) -> pd.DataFrame:
    """Use the resolved opponent to adjust projections and visibly confirm the matchup used."""
    if proj_df is None or proj_df.empty:
        return proj_df
    out = proj_df.copy()
    for c in ["Opponent", "Matchup", "HomeAway", "Projection Matchup Used", "Opponent Context Note", "Matchup Projection Factor", "Opponent Context Applied"]:
        if c not in out.columns:
            out[c] = "" if c != "Matchup Projection Factor" else 1.0
    for idx, r in out.iterrows():
        team = _team_key_for_matchup(r.get("Team"))
        opp = _team_key_for_matchup(r.get("Opponent"))
        matchup = str(r.get("Matchup") or "").strip()
        if not opp and matchup and "@" in matchup:
            away, home = _parse_event_teams_from_text(matchup)
            if team == away:
                opp = home
                out.at[idx, "Opponent"] = opp
                out.at[idx, "HomeAway"] = "AWAY"
            elif team == home:
                opp = away
                out.at[idx, "Opponent"] = opp
                out.at[idx, "HomeAway"] = "HOME"
        if not opp:
            out.at[idx, "Projection Matchup Used"] = f"{team or ''} — opponent unavailable"
            out.at[idx, "Opponent Context Note"] = "No opponent matched from ESPN schedule, cached schedule, or active two-team board fallback; projection remains player/market based."
            out.at[idx, "Matchup Projection Factor"] = 1.0
            out.at[idx, "Opponent Context Applied"] = "NO"
            continue
        opp_ctx = _latest_team_context(opp, None)
        factor, note = _market_matchup_adjustment(r, opp_ctx)
        if pd.notna(safe_float(r.get("Game Context Factor"), np.nan)) and str(r.get("Game Context Note", "")).strip():
            factor = 1.0
            note = str(r.get("Game Context Note"))
        old_proj = safe_float(r.get("Projection"), np.nan)
        line = safe_float(r.get("Line"), np.nan)
        already = str(r.get("Opponent Context Applied", "")).lower() == "yes"
        if pd.notna(old_proj) and not already:
            new_proj = round(float(old_proj) * factor, 2)
            out.at[idx, "Raw Projection Before Matchup"] = round(float(old_proj), 2)
            out.at[idx, "Projection"] = new_proj
            if pd.notna(line):
                out.at[idx, "Edge"] = round(new_proj - float(line), 2)
                out.at[idx, "Lean"] = "OVER" if new_proj > float(line) else "UNDER"
                if "Official" in out.columns:
                    out.at[idx, "Official"] = "🔥 OVER" if new_proj > float(line) else "⚠️ UNDER"
        out.at[idx, "Opponent Context Applied"] = "YES"
        out.at[idx, "Matchup Projection Factor"] = round(float(factor), 4)
        used = matchup if matchup else f"{team} vs {opp}"
        out.at[idx, "Projection Matchup Used"] = used
        out.at[idx, "Opponent Context Note"] = note
        base_exp = str(r.get("Projection Explanation", "") or "")
        add = f" Matchup used: {used} ({note})."
        if "Matchup used:" not in base_exp:
            out.at[idx, "Projection Explanation"] = (base_exp + add).strip()
    return out



def _slate_matchup_lookup(mode: str) -> Dict[str, Dict[str, str]]:
    """Return {team: {Opponent, HomeAway, Matchup}} for the selected slate."""
    out = {}
    sched = schedule_for_slate(mode)
    if sched is None or sched.empty:
        return out
    for _, g in standardize_schedules(sched).iterrows():
        home = _team_key_for_matchup(g.get("Home"))
        away = _team_key_for_matchup(g.get("Away"))
        if not home or not away:
            continue
        matchup = f"{away} @ {home}"
        out[home] = {"Opponent": away, "HomeAway": "HOME", "Matchup": matchup}
        out[away] = {"Opponent": home, "HomeAway": "AWAY", "Matchup": matchup}
    return out


def manual_line_template(mode: str, market: str, master_global: pd.DataFrame, limit: int = 250) -> pd.DataFrame:
    """Build editable manual-line template from today's/tomorrow's teams and cached player database."""
    if master_global is None or master_global.empty:
        return pd.DataFrame(columns=["Player","Team","Opponent","Matchup","HomeAway","Market","Line","OverOdds","UnderOdds","Source","Start"])
    base = master_global.copy()
    if "Team" not in base.columns:
        base["Team"] = ""
    base["TeamKey"] = base["Team"].map(_team_key_for_matchup)
    matchups = _slate_matchup_lookup(mode)
    if mode in ["Today", "Tomorrow"] and matchups:
        base = base[base["TeamKey"].isin(matchups.keys())].copy()
    # Sort likely active/high-minute players first.
    sort_cols = [c for c in ["MIN_l10", "MIN_avg", "DataScore", "RoleConfidence"] if c in base.columns]
    if sort_cols:
        base = base.sort_values(sort_cols, ascending=False)
    base = base.drop_duplicates("NameKey", keep="first").head(limit).copy()
    existing = load_manual_lines()
    existing_map = {}
    if existing is not None and not existing.empty:
        ex = existing.copy()
        ex["NameKey"] = ex["Player"].map(normalize_name)
        ex = ex[ex["Market"].astype(str).str.upper().eq(str(market).upper())]
        for _, r in ex.iterrows():
            key = (r.get("NameKey"), str(r.get("Market")).upper(), str(r.get("Start", "")))
            existing_map[key] = r
    target = slate_target_date(mode)
    start_txt = str(target or "")
    rows = []
    for _, r in base.iterrows():
        team = _team_key_for_matchup(r.get("TeamKey") or r.get("Team"))
        ctx = matchups.get(team, {})
        nk = r.get("NameKey") or normalize_name(r.get("Player"))
        old = existing_map.get((nk, str(market).upper(), start_txt))
        # Fallback to any old manual line for that player/market if no slate date match.
        if old is None:
            for k, v in existing_map.items():
                if k[0] == nk and k[1] == str(market).upper():
                    old = v; break
        rows.append({
            "Player": r.get("Player", ""),
            "Team": team or r.get("Team", ""),
            "Opponent": ctx.get("Opponent", ""),
            "Matchup": ctx.get("Matchup", team),
            "HomeAway": ctx.get("HomeAway", ""),
            "Market": market,
            "Line": safe_float(old.get("Line"), np.nan) if isinstance(old, pd.Series) else np.nan,
            "OverOdds": safe_float(old.get("OverOdds"), np.nan) if isinstance(old, pd.Series) else np.nan,
            "UnderOdds": safe_float(old.get("UnderOdds"), np.nan) if isinstance(old, pd.Series) else np.nan,
            "Source": "Manual",
            "Start": start_txt,
            "Raw": "manual in-app entry",
        })
    return pd.DataFrame(rows)


def render_manual_line_entry(mode: str, market: str, master_global: pd.DataFrame) -> None:
    """Visible but compact manual line editor inside each market board."""
    with st.expander(f"✍️ Manual {market} lines for {mode} matchups", expanded=False):
        st.caption("Enter only the lines you want. Saved manual lines feed the same projection engine as Underdog lines.")
        template = manual_line_template(mode, market, master_global)
        if template.empty:
            st.info("Build/import the player database first, or there are no cached players for this slate.")
            return
        edited = st.data_editor(
            template[["Player","Team","Opponent","Matchup","HomeAway","Market","Line","OverOdds","UnderOdds","Source","Start","Raw"]],
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=["Player","Team","Opponent","Matchup","HomeAway","Market","Source","Start","Raw"],
            column_config={
                "Line": st.column_config.NumberColumn("Line", min_value=0.0, step=0.5, format="%.1f"),
                "OverOdds": st.column_config.NumberColumn("OverOdds", step=1.0, format="%.0f"),
                "UnderOdds": st.column_config.NumberColumn("UnderOdds", step=1.0, format="%.0f"),
            },
            key=f"manual_editor_{mode}_{market}",
        )
        csave, cclear = st.columns([1,1])
        with csave:
            if st.button(f"💾 Save {market} manual lines", key=f"save_manual_{mode}_{market}", use_container_width=True):
                clean = edited.copy()
                clean["Line"] = pd.to_numeric(clean["Line"], errors="coerce")
                clean = clean.dropna(subset=["Player", "Line"])
                clean = clean[clean["Line"] > 0]
                prev = load_manual_lines()
                if prev is None or prev.empty:
                    combined = clean
                else:
                    prev = prev.copy()
                    prev["NameKey"] = prev["Player"].map(normalize_name)
                    clean["NameKey"] = clean["Player"].map(normalize_name)
                    slate_start = str(slate_target_date(mode) or "")
                    mask = ~((prev["Market"].astype(str).str.upper() == market) & (prev.get("Start", "").astype(str) == slate_start))
                    combined = pd.concat([prev[mask].drop(columns=["NameKey"], errors="ignore"), clean.drop(columns=["NameKey"], errors="ignore")], ignore_index=True)
                save_manual_lines(combined)
                st.session_state.pop("wnba_lines_all", None)
                st.success(f"Saved {len(clean):,} {market} manual lines for {mode}.")
                st.rerun()
        with cclear:
            if st.button(f"🧹 Clear {market} manual slate", key=f"clear_manual_{mode}_{market}", use_container_width=True):
                prev = load_manual_lines()
                if prev is not None and not prev.empty:
                    slate_start = str(slate_target_date(mode) or "")
                    keep = ~((prev["Market"].astype(str).str.upper() == market) & (prev.get("Start", "").astype(str) == slate_start))
                    save_manual_lines(prev[keep])
                st.session_state.pop("wnba_lines_all", None)
                st.success("Manual slate lines cleared.")
                st.rerun()
def render_source_status_card(lines: pd.DataFrame, ud_debug: pd.DataFrame, sl_debug: pd.DataFrame, use_odds_api_flag: bool = False, odds_api_key: str = ""):
    def count_source(src):
        try:
            return int((lines.get("Source", pd.Series(dtype=str)).astype(str) == src).sum()) if lines is not None and not lines.empty else 0
        except Exception:
            return 0
    st.markdown(f"""
    <div class='owp-blue-note'>
      <b>Source Status</b> — Underdog: {count_source('Underdog')} lines | Manual: {count_source('Manual')} lines | CSV Upload: {count_source('CSV Upload')} lines<br>
      Sleeper / Odds API / SportsGameOdds are disabled. Projection engine still runs from Underdog or manual lines.
    </div>
    """, unsafe_allow_html=True)

def render_mlb_style_board(mode: str, use_ud_flag: bool, use_sleeper_flag: bool, logs_global: pd.DataFrame, master_global: pd.DataFrame, force_market: Optional[str] = None):
    market_label = f" — {force_market}" if force_market else ""
    st.markdown(f"<div class='section-title'>{mode}{market_label} Board</div>", unsafe_allow_html=True)
    market_key = force_market or "ALL"
    top_cols = st.columns([1.1, 1.1, 1.2, 1.2, 2.0])
    with top_cols[0]:
        refresh_label = "🔄 Refresh Today — All-in-One" if mode == "Today" else f"🔄 Refresh {mode} — All-in-One"
        if st.button(refresh_label, key=f"refresh_{mode}_{market_key}"):
            if mode == "Today":
                run_one_click_refresh_today(mode, use_ud_flag)
            else:
                clear_line_pull_caches()
                pull_board_lines(use_ud_flag, False, False, "")
                st.session_state["wnba_last_refresh"] = now_iso()
            st.rerun()
    with top_cols[1]:
        st.caption("Schedule, context, lines, database, and board rebuild are included in Refresh Today.")
    with top_cols[2]:
        st.metric("Last refresh", st.session_state.get("wnba_last_refresh", "not yet"))
    with top_cols[3]:
        st.metric("Database players", 0 if master_global is None or master_global.empty else len(master_global))
    with top_cols[4]:
        st.caption("Workflow: Refresh Today → inspect cards → Save official before games → Grade after results post.")

    lines_all, ud_debug, sl_debug = get_lines_from_state_or_pull(use_ud_flag, False, False, "")
    lines, slate_note = filter_lines_for_slate(lines_all, mode)
    render_source_status_card(lines_all, ud_debug, sl_debug, False, "")
    if mode == "Today":
        render_refresh_today_status()
    st.caption(slate_note)

    sched = schedule_for_slate(mode)
    if not sched.empty:
        with st.expander(f"{mode} schedule context", expanded=False):
            st.dataframe(sched, use_container_width=True)
    elif mode in ["Today", "Tomorrow"]:
        st.info(f"No cached schedule rows found for {mode.lower()}. If WNBA is off that day, lines may correctly return 0.")

    if force_market:
        render_manual_line_entry(mode, force_market, master_global)

    if logs_global.empty or master_global.empty:
        st.warning("Import/build SportsDataverse player logs first in Data Manager. Lines can load, but projections need player baselines.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lines loaded", 0 if lines is None else len(lines))
    c2.metric("Underdog rows", 0 if lines_all is None or lines_all.empty else int((lines_all.get("Source", pd.Series(dtype=str)) == "Underdog").sum()))
    c3.metric("Manual rows", 0 if lines_all is None or lines_all.empty else int((lines_all.get("Source", pd.Series(dtype=str)) == "Manual").sum()))
    c4.metric("CSV rows", 0 if lines_all is None or lines_all.empty else int((lines_all.get("Source", pd.Series(dtype=str)) == "CSV Upload").sum()))

    if lines is None or lines.empty:
        st.error("No Underdog or manual lines loaded for this slate. Use the manual line editor above, then save and refresh.")
        if not master_global.empty:
            st.markdown("<div class='hidden-baseline-note'>Baseline table is hidden. Showing player cards only so you can still review projections while waiting for lines.</div>", unsafe_allow_html=True)
            baseline_cards = make_baseline_player_cards(master_global, force_market or "PRA", limit=30)
            for _, rr in baseline_cards.iterrows():
                render_card(rr)
        return pd.DataFrame()

    if force_market:
        market_filter = [force_market]
        st.caption(f"Market locked to {force_market}; real sportsbook lines route directly here.")
    else:
        market_filter = st.multiselect("Market", MARKETS, default=MARKETS, key=f"market_{mode}_{market_key}")
    search = st.text_input("Search player", key=f"search_{mode}_{market_key}")
    st.session_state["wnba_current_mode"] = mode
    proj_df = make_projection_board(lines[lines["Market"].isin(market_filter)], logs_global, master_global, mode)
    if search and not proj_df.empty:
        proj_df = proj_df[proj_df["Player"].str.contains(search, case=False, na=False)]

    if proj_df.empty:
        st.warning("Lines loaded, but projection board could not be built. Check player-name matching and Data Manager.")
        return pd.DataFrame()

    proj_df["Slate"] = mode
    proj_df["SlateDate"] = str(slate_target_date(mode) or "ALL")
    proj_df = enrich_board_with_matchups(proj_df, mode)
    proj_df = apply_matchup_context_to_board(proj_df)
    proj_df = apply_daily_team_context_v2_to_board(proj_df)
    CACHE_FILES["projection_board"].parent.mkdir(exist_ok=True)
    proj_df.to_csv(CACHE_FILES["projection_board"], index=False)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Official plays", int(proj_df["Official"].astype(str).str.contains("OVER|UNDER", na=False).sum()))
    m2.metric("Avg edge", round(float(proj_df["Edge"].abs().mean()), 2))
    m3.metric("Avg data score", round(float(proj_df["Data Score"].mean()), 1))
    m4.metric("Avg official score", round(float(proj_df["Official Play Score"].mean()), 1))

    action_cols = st.columns([1.2, 1.2, 1.2, 2.0])
    with action_cols[0]:
        if st.button(f"✅ Save {mode} Official Before", key=f"save_before_{mode}_{market_key}"):
            n = save_officials(proj_df)
            st.success(f"Saved {n} official plays for {mode}.")
    with action_cols[1]:
        if st.button(f"📊 Grade After Results", key=f"grade_after_{mode}_{market_key}"):
            n = grade_pending(logs_global, mode)
            st.success(f"Graded {n} pending plays for {mode} only.")
    with action_cols[2]:
        st.download_button(f"Download {mode} Board CSV", proj_df.to_csv(index=False), f"wnba_{mode.lower().replace(' ', '_')}_projection_board.csv", "text/csv", key=f"dl_{mode}_{market_key}")
    with action_cols[3]:
        st.caption("Save before does not change projections. Grade after uses the latest imported stat logs and updates learning history.")

    board_filter = st.radio(
        "Board filter",
        ["Official Plays", "All Board", "Strong Leans", "Overs", "Unders", "Pass / Track"],
        horizontal=True,
        key=f"board_filter_{mode}_{market_key}",
        help="Official Plays shows qualified/top plays. All Board shows every sportsbook line pulled for this market/slate."
    )
    display_df = filter_projection_view(proj_df, board_filter)
    st.caption(f"Showing {len(display_df):,} of {len(proj_df):,} projected rows for {board_filter}.")

    display_mode = st.radio("View", ["Player cards", "Table"], horizontal=True, key=f"view_{mode}_{market_key}")
    if display_mode == "Player cards":
        limit = 40 if board_filter != "All Board" else 155
        for _, r in display_df.head(limit).iterrows():
            render_card(r)
        if len(display_df) > limit:
            st.info(f"Showing first {limit} cards. Switch to Table view or download CSV to see all {len(display_df):,} rows.")
    else:
        show_cols = [
            "Player", "Team", "Opponent", "Matchup", "HomeAway", "Projection Matchup Used",
            "Market", "Line", "Source", "Projection", "Raw Projection Before Matchup", "Matchup Projection Factor",
            "Edge", "Lean", "Official", "Official Play Score", "PASS Reason", "Opponent Context Note",
            "Underdog Line", "Sleeper Line", "Best Over Line", "Best Under Line", "Over %", "Under %"
        ]
        st.dataframe(display_df[[c for c in show_cols if c in display_df.columns]], use_container_width=True)
    return proj_df



def render_data_manager_tab():
    st.subheader("Data Manager")
    st.caption("Visible again, but heavy jobs only run when you press a button. This keeps the normal app fast.")

    st.markdown("### Local logo assets")
    st.write("The app loads logos from `assets/logos` first. If a file is missing, it can fall back to `WNBA_LOGO_BASE_URL` in Streamlit Secrets.")
    st.code('WNBA_LOGO_BASE_URL = "https://raw.githubusercontent.com/<user>/<repo>/main/assets/logos"', language="toml")
    logo_rows = []
    for abbr in sorted(set(TEAM_LOGO_ALIASES.values())):
        found = any((LOGO_DIR / f"{abbr}.{ext}").exists() for ext in ["png", "jpg", "jpeg", "webp", "svg"])
        logo_rows.append({"Team": abbr, "Local Logo": "✅ found" if found else "⚠️ missing", "Expected Path": f"assets/logos/{abbr}.png"})
    st.dataframe(pd.DataFrame(logo_rows), use_container_width=True)

    st.markdown("### Data status")
    st.dataframe(dataset_status_table(), use_container_width=True)

    st.markdown("### Fast-safe data tools")
    st.caption("Use these only when you need to reload historical/stat data. Daily betting use should stay on Refresh Today.")
    default_datasets = ["player_game_logs", "player_season_stats", "team_season_stats", "schedules", "rosters", "game_rosters"]
    dataset_choices = st.multiselect(
        "SportsDataverse datasets to refresh",
        list(DATASET_LABELS.keys()),
        default=default_datasets,
        format_func=lambda k: DATASET_LABELS.get(k, k),
        key="dm_dataset_choices_visible",
    )
    include_heavy = st.toggle("Include heavier add-ons: lineups + shots", value=True, key="dm_include_heavy_visible")
    y1, y2 = st.columns(2)
    with y1:
        season_a = st.number_input("Current season to pull", min_value=2020, max_value=2032, value=int(season_now), step=1, key="dm_season_now_visible")
    with y2:
        season_b = st.number_input("Last season to pull", min_value=2020, max_value=2032, value=int(season_last), step=1, key="dm_season_last_visible")
    if st.button("🔄 Refresh SportsDataverse + Build Advanced Features", use_container_width=True, key="dm_refresh_sd_visible"):
        with st.spinner("Refreshing data and rebuilding advanced features..."):
            master, team_ranks, debug, audit = refresh_data_and_build_advanced_features(dataset_choices, [int(season_b), int(season_a)], include_heavy)
        st.success(f"Done. Master rows: {len(master)} | Team-rank rows: {len(team_ranks)}")
        st.markdown("#### Refresh debug")
        st.dataframe(debug, use_container_width=True)
        st.markdown("#### Missing-field report")
        st.dataframe(audit, use_container_width=True)

    if st.button("🧠 Build Advanced Features / Fix Missing Columns Only", use_container_width=True, key="dm_build_features_only_visible"):
        with st.spinner("Rebuilding master features from cached files..."):
            master, team_ranks = build_master_features()
            audit = feature_missing_report(master)
            audit.to_csv(DATA_DIR / "wnba_feature_missing_report.csv", index=False)
        st.success(f"Advanced features rebuilt. Master rows: {len(master)}")
        st.dataframe(audit, use_container_width=True)

    report_path = DATA_DIR / "wnba_feature_missing_report.csv"
    if report_path.exists():
        try:
            report_df = pd.read_csv(report_path)
            st.download_button("Download missing-field report", report_df.to_csv(index=False), "wnba_feature_missing_report.csv", "text/csv", use_container_width=True)
        except Exception:
            pass

    st.markdown("### Manual upload/import backup")
    uploaded = st.file_uploader("Upload SportsDataverse CSV/Parquet files", type=["csv", "parquet", "xlsx", "json"], accept_multiple_files=True, key="dm_manual_upload_visible")
    if uploaded and st.button("Import uploaded files", use_container_width=True, key="dm_import_uploads_visible"):
        rows = []
        for f in uploaded:
            try:
                raw = f.read()
                dataset_key = classify_filename(f.name)
                if not dataset_key:
                    rows.append({"file": f.name, "dataset": "unknown", "status": "skipped: could not classify", "rows": 0})
                    continue
                df = read_any_file(raw, f.name)
                std = standardize_dataset(dataset_key, df)
                if std is not None and not std.empty:
                    save_dataset(dataset_key, std)
                    rows.append({"file": f.name, "dataset": dataset_key, "status": "saved", "rows": len(std)})
                else:
                    rows.append({"file": f.name, "dataset": dataset_key, "status": "standardized empty", "rows": 0})
            except Exception as e:
                rows.append({"file": getattr(f, 'name', 'upload'), "dataset": "error", "status": str(e)[:180], "rows": 0})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        try:
            master, _ = build_master_features()
            st.success(f"Uploaded files imported and master rebuilt: {len(master)} rows")
        except Exception as e:
            st.warning(f"Files imported, but master rebuild needs review: {e}")

    st.markdown("### Cached exports")
    for key in ["master_features", "projection_board", "team_ranks", "player_game_logs"]:
        path = CACHE_FILES.get(key)
        if path and path.exists():
            try:
                data = path.read_text(errors="ignore")
                st.download_button(f"Download {key}.csv", data, f"{key}.csv", "text/csv", use_container_width=True, key=f"dm_download_{key}")
            except Exception:
                pass



# ============================================================
# OFFICIAL WNBA ONLINE FALLBACK + NO-LOGO UI OVERRIDES
# ============================================================
# Purpose:
# - If Streamlit cache/data folder is empty, still build usable player baselines
#   from official WNBA Stats endpoints so Underdog lines can run projections.
# - Keep local/GitHub logos out of the active UI per request.
# - Preserve the working Underdog parser and one-click workflow.

OFFICIAL_WNBA_PLAYER_CONTEXT_FILE = DATA_DIR / "wnba_official_player_context.csv"


def _official_player_stats_params(season: int, measure: str = "Base") -> Dict[str, Any]:
    return {
        "College": "", "Conference": "", "Country": "", "DateFrom": "", "DateTo": "",
        "Division": "", "DraftPick": "", "DraftYear": "", "GameScope": "", "GameSegment": "",
        "Height": "", "LastNGames": "0", "LeagueID": "10", "Location": "", "MeasureType": measure,
        "Month": "0", "OpponentTeamID": "0", "Outcome": "", "PORound": "0", "PaceAdjust": "N",
        "PerMode": "PerGame", "Period": "0", "PlayerExperience": "", "PlayerPosition": "",
        "PlusMinus": "N", "Rank": "N", "Season": str(season), "SeasonSegment": "",
        "SeasonType": "Regular Season", "ShotClockRange": "", "StarterBench": "", "TeamID": "0",
        "TwoWay": "0", "VsConference": "", "VsDivision": "", "Weight": "",
    }


def refresh_official_wnba_player_context(season: Optional[int] = None, force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pull official WNBA player stat tables and cache a season baseline.

    This is a fallback for empty Streamlit/SportsDataverse cache. It gives the
    prop engine enough Player/Team/MIN/PTS/REB/AST/efficiency data to run live
    Underdog projections even when historical game logs are missing.
    """
    if season is None:
        try:
            season = int(st.session_state.get("season_now", datetime.utcnow().year))
        except Exception:
            season = int(datetime.utcnow().year)
    if OFFICIAL_WNBA_PLAYER_CONTEXT_FILE.exists() and not force:
        try:
            cached = pd.read_csv(OFFICIAL_WNBA_PLAYER_CONTEXT_FILE)
            if cached is not None and not cached.empty:
                return cached, pd.DataFrame([{"Step":"official_wnba_player_context", "Status":"cached", "Rows":len(cached)}])
        except Exception:
            pass

    dbg = []
    base, msg = _wnba_stats_get("leaguedashplayerstats", _official_player_stats_params(season, "Base"), timeout=16)
    dbg.append({"Step":"Official Player Base", "Status":msg, "Rows":0 if base is None else len(base)})
    adv, msg = _wnba_stats_get("leaguedashplayerstats", _official_player_stats_params(season, "Advanced"), timeout=16)
    dbg.append({"Step":"Official Player Advanced", "Status":msg, "Rows":0 if adv is None else len(adv)})

    if base is None or base.empty:
        return pd.DataFrame(), pd.DataFrame(dbg + [{"Step":"official_wnba_player_context", "Status":"empty official player pull", "Rows":0}])

    d = base.copy()
    player_col = find_col(d, ["PLAYER_NAME", "PLAYER", "PLAYER_NAME_I", "NAME"])
    team_col = find_col(d, ["TEAM_ABBREVIATION", "TEAM", "TEAM_NAME"])
    if not player_col:
        return pd.DataFrame(), pd.DataFrame(dbg + [{"Step":"official_wnba_player_context", "Status":"no player column", "Rows":0}])
    out = pd.DataFrame()
    out["Player"] = d[player_col].astype(str)
    out["Team"] = d[team_col].map(_team_key_for_matchup) if team_col else ""
    out["Season"] = season
    # Copy common base fields. Official endpoint names are usually NBA/WNBA Stats API style.
    for dst, candidates in {
        "GP": ["GP", "G", "Games"],
        "MIN": ["MIN", "MINUTES"],
        "PTS": ["PTS", "POINTS"],
        "REB": ["REB", "REBOUNDS"],
        "AST": ["AST", "ASSISTS"],
        "FGA": ["FGA"], "FGM": ["FGM"], "FG3A": ["FG3A", "FG3_A"], "FG3M": ["FG3M", "FG3_M"],
        "FTA": ["FTA"], "FTM": ["FTM"], "TOV": ["TOV", "TO"], "OREB": ["OREB"], "DREB": ["DREB"],
        "STL": ["STL"], "BLK": ["BLK"],
    }.items():
        c = find_col(d, candidates)
        out[dst] = d[c] if c else np.nan
    out = coerce_numeric(out, [c for c in out.columns if c not in ["Player", "Team"]])
    out["PRA"] = out["PTS"].fillna(0) + out["REB"].fillna(0) + out["AST"].fillna(0)
    out["NameKey"] = out["Player"].map(normalize_name)

    if adv is not None and not adv.empty:
        a = adv.copy()
        a_player = find_col(a, ["PLAYER_NAME", "PLAYER", "PLAYER_NAME_I", "NAME"])
        a_team = find_col(a, ["TEAM_ABBREVIATION", "TEAM", "TEAM_NAME"])
        adv_out = pd.DataFrame()
        if a_player:
            adv_out["NameKey"] = a[a_player].map(normalize_name)
            adv_out["Team"] = a[a_team].map(_team_key_for_matchup) if a_team else ""
            for dst, candidates in {
                "USG%": ["USG_PCT", "USG%", "USAGE_RATE", "USAGE"],
                "TS%": ["TS_PCT", "TS%", "TRUE_SHOOTING_PERCENTAGE"],
                "eFG%": ["EFG_PCT", "EFG%", "EFFECTIVE_FIELD_GOAL_PERCENTAGE"],
                "AST%": ["AST_PCT", "AST%"],
                "TRB%": ["REB_PCT", "TRB%", "REB%"],
                "PER": ["PIE", "PER"],
            }.items():
                c = find_col(a, candidates)
                adv_out[dst] = a[c] if c else np.nan
            adv_out = coerce_numeric(adv_out, [c for c in adv_out.columns if c not in ["NameKey", "Team"]])
            out = out.merge(adv_out.drop_duplicates(["NameKey", "Team"]), on=["NameKey", "Team"], how="left")

    # Estimate efficiency if advanced endpoint lacks it.
    out["eFG%"] = out.get("eFG%", pd.Series(np.nan, index=out.index))
    out["TS%"] = out.get("TS%", pd.Series(np.nan, index=out.index))
    out["USG%"] = out.get("USG%", pd.Series(np.nan, index=out.index))
    fga = pd.to_numeric(out.get("FGA", np.nan), errors="coerce")
    fgm = pd.to_numeric(out.get("FGM", np.nan), errors="coerce")
    fg3m = pd.to_numeric(out.get("FG3M", np.nan), errors="coerce")
    fta = pd.to_numeric(out.get("FTA", np.nan), errors="coerce")
    pts = pd.to_numeric(out.get("PTS", np.nan), errors="coerce")
    out["eFG%"] = out["eFG%"].fillna(np.where(fga > 0, (fgm + 0.5 * fg3m) / fga, np.nan))
    out["TS%"] = out["TS%"].fillna(np.where((2 * (fga + 0.44 * fta)) > 0, pts / (2 * (fga + 0.44 * fta)), np.nan))
    out["USG%"] = out["USG%"].fillna((fga.fillna(0) + 0.44 * fta.fillna(0) + pd.to_numeric(out.get("TOV", 0), errors="coerce").fillna(0)))
    out = out[out["NameKey"].astype(str).str.len() > 0].copy()
    try:
        OFFICIAL_WNBA_PLAYER_CONTEXT_FILE.parent.mkdir(exist_ok=True)
        out.to_csv(OFFICIAL_WNBA_PLAYER_CONTEXT_FILE, index=False)
        # Also fill player_season_stats cache if missing so Data Manager has a visible source.
        if not CACHE_FILES["player_season_stats"].exists() or force:
            save_dataset("player_season_stats", out)
    except Exception:
        pass
    return out, pd.DataFrame(dbg + [{"Step":"official_wnba_player_context", "Status":"saved", "Rows":len(out)}])


def build_official_baselines_from_player_context(player_ctx: pd.DataFrame, team_ctx: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if player_ctx is None or player_ctx.empty:
        return pd.DataFrame()
    d = player_ctx.copy()
    d["NameKey"] = d["Player"].map(normalize_name)
    d["Team"] = d["Team"].map(_team_key_for_matchup)
    for c in ["GP", "MIN", "PTS", "REB", "AST", "PRA", "FGA", "FGM", "FG3A", "FG3M", "FTA", "TOV", "OREB", "DREB"]:
        if c not in d.columns: d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    base = pd.DataFrame()
    base["NameKey"] = d["NameKey"]
    base["Player"] = d["Player"]
    base["Team"] = d["Team"]
    base["Season"] = d.get("Season", datetime.utcnow().year)
    base["Games"] = d["GP"].fillna(1).clip(lower=1)
    base["LastGame"] = pd.NaT
    for src, dst in [("MIN", "MIN"), ("PTS", "PTS"), ("REB", "REB"), ("AST", "AST"), ("PRA", "PRA")]:
        vals = d[src].fillna(0)
        for suff in ["avg", "l3", "l5", "l10", "l20"]:
            base[f"{dst}_{suff}"] = vals
    for c in ["FGA", "FGM", "FG3A", "FG3M", "FTA", "TOV", "OREB", "DREB"]:
        # convert per-game to rough season total for formulas that expect totals
        base[c] = d[c].fillna(0) * base["Games"]
    base["eFG%"] = pd.to_numeric(d.get("eFG%", np.nan), errors="coerce")
    base["TS%"] = pd.to_numeric(d.get("TS%", np.nan), errors="coerce")
    base["USG%"] = pd.to_numeric(d.get("USG%", np.nan), errors="coerce")
    base["AST%"] = pd.to_numeric(d.get("AST%", np.nan), errors="coerce")
    base["TRB%"] = pd.to_numeric(d.get("TRB%", np.nan), errors="coerce")
    base["PER"] = pd.to_numeric(d.get("PER", np.nan), errors="coerce")
    base["UsageProxy"] = d["FGA"].fillna(0) + 0.44 * d["FTA"].fillna(0) + d["TOV"].fillna(0)
    base["AST%Proxy"] = base["AST_avg"] / base["MIN_avg"].replace(0, np.nan)
    base["TRB%Proxy"] = base["REB_avg"] / base["MIN_avg"].replace(0, np.nan)
    base["PERProxy"] = base["PTS_avg"] + base["REB_avg"] + base["AST_avg"]
    base["Position"] = ""
    base["PositionGroup"] = "Unknown"
    # Official fallback has no shot-zone data. Use neutral values so points engine runs without fake edge.
    base["ShotAttempts"] = d["FGA"].fillna(0) * base["Games"]
    base["ThreePARate"] = np.where(d["FGA"] > 0, d["FG3A"] / d["FGA"], np.nan)
    base["RimRate"] = np.nan
    base["ShotMakeRate"] = np.where(d["FGA"] > 0, d["FGM"] / d["FGA"], np.nan)
    base["PointsPerShot"] = np.where(d["FGA"] > 0, d["PTS"] / d["FGA"], np.nan)
    base["ShotProfileScore"] = np.clip(50 + base["ThreePARate"].fillna(0.22)*12 + (base["ShotMakeRate"].fillna(0.42)-0.42)*55, 0, 100)
    for m in MARKETS:
        base[f"{m}_Std20"] = np.maximum(1.2, base[f"{m}_avg"].abs() * ({"PTS":0.28,"REB":0.32,"AST":0.38,"PRA":0.26}.get(m,0.3)))
        base[f"{m}_per_min"] = base[f"{m}_avg"] / base["MIN_avg"].replace(0, np.nan)
    base["RosterGames"] = base["Games"]
    base["StarterGames"] = np.nan
    base["StarterRate"] = np.nan
    base["LineupMentions"] = np.nan
    base["LineupShare"] = np.nan
    base["LineupContinuityScore"] = 55
    base["RoleConfidence"] = np.clip(48 + base["Games"].clip(0,25)*1.2 + base["MIN_avg"].clip(0,36)*0.8, 0, 92).round(1)
    base["MinutesSafetyGrade"] = np.select([base["MIN_avg"] >= 30, base["MIN_avg"] >= 24, base["MIN_avg"] >= 18], ["A", "B", "C"], default="D")
    base["MinutesProjectionBase"] = base["MIN_avg"].round(2)
    base["BayesianPriorStrength"] = np.clip((base["Games"] / 20) * 100, 0, 100).round(1)
    base["VolatilityScore"] = np.clip(30 + base[[f"{m}_Std20" for m in MARKETS]].mean(axis=1).fillna(0)*8, 0, 100).round(1)
    base["OfficialOnlineFallback"] = "YES"
    # Attach team context if available.
    if team_ctx is not None and not team_ctx.empty and "Team" in team_ctx.columns:
        tc = team_ctx.copy()
        tc["Team"] = tc["Team"].map(_team_key_for_matchup)
        keep = [c for c in tc.columns if c == "Team" or "Official" in str(c) or str(c) in ["Team"]]
        base = base.merge(tc[keep].drop_duplicates("Team"), on="Team", how="left")
        base["Team_Pace"] = pd.to_numeric(base.get("Team_Pace_Official", np.nan), errors="coerce")
        base["Team_ORtg"] = pd.to_numeric(base.get("Team_ORtg_Official", np.nan), errors="coerce")
        base["Team_DRtg"] = pd.to_numeric(base.get("Team_DRtg_Official", np.nan), errors="coerce")
        base["Team_NetRtg"] = pd.to_numeric(base.get("Team_NetRtg_Official", np.nan), errors="coerce")
    for c, val in [("Team_Pace",78),("Team_ORtg",100),("Team_DRtg",100),("Team_NetRtg",0)]:
        if c not in base.columns: base[c] = val
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(val)
    base["TeamMatchupStrengthScore"] = np.clip(50 + base["Team_NetRtg"].fillna(0)*2.0, 0, 100).round(1)
    base["DataScore"] = np.clip(56 + base["Games"].clip(0,25)*1.25 + base["MIN_avg"].clip(0,36)*0.65 + base["RoleConfidence"].fillna(55)*0.12, 0, 94).round(1)
    return base


def ensure_online_wnba_master_features(force_official: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a master feature table even when local cache is empty.

    Priority:
    1) existing master_features cache
    2) SportsDataverse/player logs cache via build_master_features()
    3) official WNBA Stats API player/team fallback
    """
    debug = []
    master = load_dataset("master_features")
    if master is not None and not master.empty and not force_official:
        return master, load_dataset("team_ranks"), pd.DataFrame([{"Step":"master_features", "Status":"cached", "Rows":len(master)}])
    logs = load_dataset("player_game_logs")
    if logs is not None and not logs.empty and not force_official:
        try:
            master, team_ranks = build_master_features()
            if master is not None and not master.empty:
                return master, team_ranks, pd.DataFrame([{"Step":"master_features", "Status":"rebuilt from cached logs", "Rows":len(master)}])
        except Exception as e:
            debug.append({"Step":"cached_log_build", "Status":str(e)[:180], "Rows":0})
    team_ctx, team_dbg = refresh_official_wnba_team_context(int(datetime.utcnow().year), force=force_official)
    player_ctx, player_dbg = refresh_official_wnba_player_context(int(datetime.utcnow().year), force=force_official)
    debug.extend(team_dbg.to_dict("records") if team_dbg is not None and not team_dbg.empty else [])
    debug.extend(player_dbg.to_dict("records") if player_dbg is not None and not player_dbg.empty else [])
    master = build_official_baselines_from_player_context(player_ctx, team_ctx)
    if master is not None and not master.empty:
        try:
            save_dataset("master_features", master)
            feature_missing_report(master).to_csv(DATA_DIR / "wnba_feature_missing_report.csv", index=False)
        except Exception:
            pass
        team_ranks = _build_team_context_from_cached_sources()
        return master, team_ranks, pd.DataFrame(debug + [{"Step":"master_features", "Status":"built from official WNBA online fallback", "Rows":len(master)}])
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(debug + [{"Step":"master_features", "Status":"unavailable; online fallback failed", "Rows":0}])


# Override Refresh Today so live lines can still project if no local SportsDataverse cache exists.
def refresh_today_pipeline(mode: str, use_ud_flag: bool = True) -> Dict[str, Any]:
    started = time.time()
    status = {"Mode": mode, "Schedule Loaded": "NO", "Games": 0, "Underdog Lines": 0, "Cards": 0, "Status": "started"}
    sched, sched_dbg = update_schedule_cache_with_espn(mode)
    status["Games"] = 0 if sched is None or sched.empty else len(sched)
    status["Schedule Loaded"] = "YES" if status["Games"] else "NO/CACHED"
    st.session_state["wnba_schedule_debug"] = sched_dbg
    master, team_ranks, master_dbg = ensure_online_wnba_master_features(force_official=False)
    st.session_state["wnba_online_master_debug"] = master_dbg
    status["Master Rows"] = 0 if master is None or master.empty else len(master)
    try:
        game_ctx, game_dbg = build_game_context_cache(mode, force_official=True)
        status["Game Context Rows"] = 0 if game_ctx is None or game_ctx.empty else len(game_ctx)
        st.session_state["wnba_game_context_debug"] = game_dbg
        daily_ctx, daily_dbg = build_daily_team_context_cache_v2(mode, force=True)
        status["Daily Context Rows"] = 0 if daily_ctx is None or daily_ctx.empty else len(daily_ctx)
        st.session_state["wnba_daily_team_context_v2_debug"] = daily_dbg
    except Exception as _gce:
        status["Game Context Rows"] = 0
        st.session_state["wnba_game_context_last_error"] = str(_gce)[:180]
    clear_line_pull_caches()
    lines_all, ud_debug, manual_debug = pull_board_lines(use_ud_flag, False, False, "")
    lines, _ = filter_lines_for_slate(lines_all, mode)
    status["Underdog Lines"] = int((lines.get("Source", pd.Series(dtype=str)).astype(str) == "Underdog").sum()) if lines is not None and not lines.empty else 0
    logs = load_dataset("player_game_logs")
    if logs is None:
        logs = pd.DataFrame()
    if lines is not None and not lines.empty and master is not None and not master.empty:
        try:
            board = make_projection_board(lines, logs, master, mode)
            board["Slate"] = mode
            board["SlateDate"] = str(slate_target_date(mode) or "ALL")
            board = enrich_board_with_matchups(board, mode)
            board = apply_matchup_context_to_board(board)
            board = apply_daily_team_context_v2_to_board(board)
            CACHE_FILES["projection_board"].parent.mkdir(exist_ok=True)
            board.to_csv(CACHE_FILES["projection_board"], index=False)
            status["Cards"] = len(board)
            status["Status"] = "complete"
        except Exception as e:
            status["Status"] = f"board build error: {str(e)[:150]}"
    else:
        status["Status"] = "no lines or no master baselines"
    status["Seconds"] = round(time.time() - started, 2)
    return status


def run_one_click_refresh_today(mode: str = "Today", use_ud_flag: bool = True) -> Dict[str, Any]:
    status = {"Mode": mode, "Status": "started"}
    started = time.time()
    try:
        master, _, dbg = ensure_online_wnba_master_features(force_official=False)
        status["Database"] = f"ready {0 if master is None or master.empty else len(master):,} players"
        st.session_state["wnba_online_master_debug"] = dbg
        pipe_status = refresh_today_pipeline(mode, use_ud_flag)
        status.update(pipe_status or {})
        status["Status"] = status.get("Status", "complete")
    except Exception as e:
        status["Status"] = f"one-click error: {str(e)[:160]}"
    status["Seconds"] = round(time.time() - started, 2)
    st.session_state["wnba_refresh_today_status"] = status
    st.session_state["wnba_last_refresh"] = now_iso()
    return status


# No-logo UI override: hide/remove logo boxes and keep Data Manager data-focused.
def get_team_logo_src(team: Any) -> str:
    return ""

st.markdown("<style>.owp-logo{display:none!important}.owp-card-main{gap:10px!important}</style>", unsafe_allow_html=True)




# -----------------------------------------------------------------------------
# Grouped Player Cards Board (single-player card with PTS/REB/AST/PRA rows)
# -----------------------------------------------------------------------------
def _fmt_num_compact(x, dec=1, default="—"):
    try:
        v = safe_float(x, np.nan)
        if pd.isna(v):
            return default
        return f"{v:.{dec}f}"
    except Exception:
        return default


def _grouped_market_html(r: pd.Series) -> str:
    market = str(r.get("Market", "PROP")).upper()
    proj = _fmt_num_compact(r.get("Projection"), 1)
    line = _fmt_num_compact(r.get("Line"), 1)
    edge_v = safe_float(r.get("Edge"), np.nan)
    edge = _fmt_num_compact(edge_v, 1)
    lean = str(r.get("Lean", "TRACK")).upper()
    official = str(r.get("Official", ""))
    side_txt = "OVER" if "OVER" in f"{lean} {official}" else "UNDER" if "UNDER" in f"{lean} {official}" else "TRACK"
    side_cls = "over" if side_txt == "OVER" else "under" if side_txt == "UNDER" else "track"
    overp = safe_float(r.get("Over %"), np.nan)
    underp = safe_float(r.get("Under %"), np.nan)
    fill = overp if pd.notna(overp) else (100-underp if pd.notna(underp) else 50)
    fill = max(0, min(100, fill))
    conf = _fmt_num_compact(r.get("Official Play Score"), 0)
    tier = str(r.get("Tier", ""))
    vol = str(r.get("Volatility", "NA"))
    note = str(r.get("Biggest Positive", "") or r.get("Projection Explanation", ""))[:130]
    edge_cls = "pos" if pd.notna(edge_v) and edge_v > 0 else "neg" if pd.notna(edge_v) and edge_v < 0 else "flat"
    return f"""
      <div class='owp-market-row owp-mkt-{market.lower()}'>
        <div class='owp-market-head'>
          <span class='owp-market-name'>{market}</span>
          <span class='owp-market-vol'>{vol}</span>
          <span class='owp-market-conf {side_cls}'>HIGH {conf}%</span>
          <span class='owp-market-side {side_cls}'>{side_txt}</span>
        </div>
        <div class='owp-market-main'>
          <span class='owp-market-proj'>{proj}</span>
          <span class='owp-market-edge {edge_cls}'>{edge} vs {line}</span>
          <span class='owp-market-tier'>{tier}</span>
        </div>
        <div class='owp-prob-track owp-market-track'><div class='owp-prob-fill' style='width:{fill:.0f}%'></div></div>
        <div class='owp-market-sub'><span>OVER {_fmt_num_compact(overp,0)}%</span><span>UNDER {_fmt_num_compact(underp,0)}%</span></div>
        <div class='owp-market-note'>{note}</div>
      </div>
    """


def render_grouped_player_card(player_df: pd.DataFrame):
    if player_df is None or player_df.empty:
        return
    # Sort markets in the normal board order.
    order = {m:i for i,m in enumerate(MARKETS)}
    player_df = player_df.copy()
    player_df["_market_order"] = player_df.get("Market", "").astype(str).str.upper().map(order).fillna(99)
    player_df = player_df.sort_values(["_market_order", "Line"])
    first = player_df.iloc[0]
    player = str(first.get("Player", "Player"))
    team = str(first.get("Team", ""))
    matchup = str(first.get("Matchup", "") or first.get("Projection Matchup Used", "") or team)
    pos = str(first.get("PositionGroup", "Role"))
    source_count = ", ".join(sorted(set(player_df.get("Source", pd.Series(dtype=str)).astype(str).replace("nan", "").tolist())))
    role = str(first.get("FallbackLineupRole", first.get("Minutes Safety", "NA")))
    min_proj = _fmt_num_compact(player_df.get("MIN Proj", pd.Series([np.nan])).max(), 1)
    data_score = _fmt_num_compact(player_df.get("Data Score", pd.Series([np.nan])).max(), 0)
    best = player_df.copy()
    best["_abs_edge"] = pd.to_numeric(best.get("Edge", np.nan), errors="coerce").abs()
    best = best.sort_values("_abs_edge", ascending=False).iloc[0]
    best_mkt = str(best.get("Market", "PROP"))
    best_side = "OVER" if "OVER" in str(best.get("Lean", "")).upper() else "UNDER" if "UNDER" in str(best.get("Lean", "")).upper() else "TRACK"
    market_html = "".join(_grouped_market_html(r) for _, r in player_df.iterrows())
    st.markdown(f"""
    <div class='owp-group-card'>
      <div class='owp-group-top'>
        <div>
          <div class='owp-player'>{player}</div>
          <div class='owp-match'>{matchup} <span class='owp-muted'>| {team} | {pos}</span></div>
          <span class='owp-pill owp-pill-source'>{source_count or 'Lines'}</span>
          <span class='owp-pill owp-pill-role'>Lineup/Role {role}</span>
          <span class='owp-pill owp-pill-score'>Data {data_score}/100</span>
        </div>
        <div class='owp-group-best'>Best: {best_mkt} {best_side}<br><span>Min {min_proj}</span></div>
      </div>
      {market_html}
    </div>
    """, unsafe_allow_html=True)
    with st.expander(f"Advanced details — {player} all markets", expanded=False):
        show_cols = [c for c in ["Market","Projection","Line","Edge","Lean","Over %","Under %","Official Play Score","Tier","Source","Opponent","HomeAway","Projection Explanation"] if c in player_df.columns]
        st.dataframe(player_df[show_cols], use_container_width=True, hide_index=True)




def grouped_board_table_view(proj_df: pd.DataFrame) -> pd.DataFrame:
    """Compact grouped table for speed: one row per player/market, sorted by player and market."""
    if proj_df is None or proj_df.empty:
        return pd.DataFrame()
    df = proj_df.copy()
    order = {m:i for i,m in enumerate(MARKETS)}
    df["_market_order"] = df.get("Market", "").astype(str).str.upper().map(order).fillna(99)
    # Keep only useful board columns so this loads much faster than rendering all cards.
    keep = [c for c in [
        "Player", "Team", "Opponent", "Matchup", "Market", "Projection", "Line", "Edge",
        "Lean", "Over %", "Under %", "Official Play Score", "Tier", "Volatility",
        "MIN Proj", "FallbackLineupRole", "Source"
    ] if c in df.columns]
    if not keep:
        return df
    out = df.sort_values(["Player", "_market_order", "Line"], na_position="last")[keep].copy()
    for c in ["Projection", "Line", "Edge", "Over %", "Under %", "Official Play Score", "MIN Proj"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    return out

def render_grouped_table_or_cards(proj_df: pd.DataFrame, mode: str, key_prefix: str, default_cards: bool = False) -> pd.DataFrame:
    """Fast display switcher. Table mode avoids rendering dozens of heavy HTML cards."""
    if proj_df is None or proj_df.empty:
        st.info("No rows to show.")
        return pd.DataFrame()
    search = st.text_input("Search player", key=f"{key_prefix}_search")
    official_only = st.toggle("Official / strong signals only", value=False, key=f"{key_prefix}_official")
    view_df = proj_df.copy()
    if search and "Player" in view_df.columns:
        view_df = view_df[view_df["Player"].astype(str).str.contains(search, case=False, na=False)]
    if official_only and "Official" in view_df.columns:
        view_df = view_df[view_df["Official"].astype(str).str.contains("OVER|UNDER", na=False)]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Players", view_df["Player"].nunique() if "Player" in view_df.columns else 0)
    c2.metric("Markets", len(view_df))
    c3.metric("Official plays", int(view_df.get("Official", pd.Series(dtype=str)).astype(str).str.contains("OVER|UNDER", na=False).sum()))
    c4.metric("Avg edge", round(float(pd.to_numeric(view_df.get("Edge", pd.Series(dtype=float)), errors="coerce").abs().mean()), 2) if not view_df.empty else 0)

    display_mode = st.radio(
        "View",
        ["Fast table", "Player cards"],
        index=1 if default_cards else 0,
        horizontal=True,
        key=f"{key_prefix}_display_mode",
        help="Fast table loads quickest. Player cards show the full grouped-card view."
    )

    ac1, ac2, ac3 = st.columns([1.2,1.2,1.4])
    with ac1:
        if st.button(f"✅ Save {mode} Official Before", key=f"{key_prefix}_save_before", use_container_width=True):
            n = save_officials(view_df); st.success(f"Saved {n} official plays for {mode}.")
    with ac2:
        if st.button(f"📊 Grade After Results", key=f"{key_prefix}_grade_after", use_container_width=True):
            n = grade_pending(load_dataset("player_game_logs"), mode); st.success(f"Graded {n} pending plays for {mode} only.")
    with ac3:
        st.download_button(f"Download {mode} Board CSV", view_df.to_csv(index=False), f"wnba_{mode.lower().replace(' ','_')}_board.csv", "text/csv", key=f"{key_prefix}_download")

    if view_df.empty:
        st.info("No rows after filters.")
        return view_df

    if display_mode == "Fast table":
        compact = grouped_board_table_view(view_df)
        st.dataframe(compact, use_container_width=True, hide_index=True, height=620)
        st.caption("Fast table mode shows every loaded player/market without rendering heavy cards. Switch to Player cards only when you want the full card layout.")
        return view_df

    show_all = st.toggle("Show all player cards", value=False, key=f"{key_prefix}_show_all_cards")
    max_default = min(40, max(10, int(view_df["Player"].nunique() if "Player" in view_df.columns else 40)))
    max_players = len(view_df["Player"].dropna().unique()) if show_all and "Player" in view_df.columns else st.slider("Max player cards", 10, 120, max_default, 5, key=f"{key_prefix}_max_cards")
    sort_df = view_df.copy()
    sort_df["_abs_edge"] = pd.to_numeric(sort_df.get("Edge", np.nan), errors="coerce").abs()
    sort_df["_score"] = pd.to_numeric(sort_df.get("Official Play Score", np.nan), errors="coerce")
    player_order = sort_df.groupby("Player", dropna=False).agg(MaxScore=("_score","max"), MaxEdge=("_abs_edge","max")).reset_index().sort_values(["MaxScore","MaxEdge"], ascending=False)["Player"].head(max_players).tolist()
    for player in player_order:
        render_grouped_player_card(view_df[view_df["Player"] == player])
    return view_df

def _render_grouped_projection_df(proj_df: pd.DataFrame, mode: str, search_key: str, max_key: str, official_key: str, saved_view: bool = False) -> pd.DataFrame:
    """Render an already-built projection board with fast table / card toggle."""
    key_prefix = f"saved_group_{mode}" if saved_view else f"live_group_{mode}"
    return render_grouped_table_or_cards(proj_df, mode, key_prefix, default_cards=False)

def render_grouped_player_board(mode: str, use_ud_flag: bool, logs_global: pd.DataFrame, master_global: pd.DataFrame):
    st.markdown(f"<div class='section-title'>{mode} — Grouped Player Cards</div>", unsafe_allow_html=True)
    top_cols = st.columns([1.2, 1.0, 1.0, 2.0])
    with top_cols[0]:
        refresh_label = "🔄 Refresh Live Today" if mode == "Today" else f"🔄 Refresh Live {mode}"
        if st.button(refresh_label, key=f"group_refresh_{mode}", use_container_width=True):
            st.session_state[f"wnba_force_live_{mode}"] = True
            if mode == "Today":
                run_one_click_refresh_today(mode, use_ud_flag)
            else:
                clear_line_pull_caches(); pull_board_lines(use_ud_flag, False, False, ""); st.session_state["wnba_last_refresh"] = now_iso()
            st.rerun()
    with top_cols[1]:
        meta = saved_board_meta()
        st.metric("Last refresh", st.session_state.get("wnba_last_refresh", meta.get("SavedAt", "not yet")))
    with top_cols[2]:
        st.metric("Database players", 0 if master_global is None or master_global.empty else len(master_global))
    with top_cols[3]:
        st.caption("Grouped layout: one player card contains PTS / REB / AST / PRA. Save Board lets this page reopen instantly without pulling lines again.")

    # MLB-style persistence: if a board was saved, show it immediately on app open
    # without calling Underdog or rebuilding the database. Refresh Live overrides this.
    saved_df = load_board_snapshot(mode)
    force_live = bool(st.session_state.get(f"wnba_force_live_{mode}", False))
    if not force_live and saved_df is not None and not saved_df.empty:
        meta = saved_board_meta()
        st.success(f"Loaded saved board from {meta.get('SavedAt', 'saved snapshot')} — no live refresh needed.")
        bc1, bc2, bc3 = st.columns([1.2, 1.2, 1.4])
        with bc1:
            if st.button("🔄 Refresh live lines", key=f"saved_refresh_live_{mode}", use_container_width=True):
                st.session_state[f"wnba_force_live_{mode}"] = True
                st.rerun()
        with bc2:
            if st.button("🧹 Clear saved board", key=f"clear_saved_board_{mode}", use_container_width=True):
                for _p in [SAVED_BOARD_FILE, SAVED_LINES_FILE, SAVED_BOARD_META_FILE]:
                    try:
                        if _p.exists(): _p.unlink()
                    except Exception: pass
                st.success("Saved board cleared.")
                st.rerun()
        with bc3:
            st.caption("Saved boards stay available after closing/reopening the app as long as Streamlit keeps the app files/cache.")
        return _render_grouped_projection_df(saved_df, mode, f"saved_group_search_{mode}", f"saved_group_max_{mode}", f"saved_group_official_only_{mode}", saved_view=True)

    lines_all, ud_debug, sl_debug = get_lines_from_state_or_pull(use_ud_flag, False, False, "")
    lines, slate_note = filter_lines_for_slate(lines_all, mode)
    render_source_status_card(lines_all, ud_debug, sl_debug, False, "")
    if mode == "Today":
        render_refresh_today_status()
    st.caption(slate_note)
    sched = schedule_for_slate(mode)
    if not sched.empty:
        with st.expander(f"{mode} schedule context", expanded=False):
            st.dataframe(sched, use_container_width=True)

    if lines is None or lines.empty:
        st.error("No Underdog or manual lines loaded for this slate. Use Data Manager/manual fallback, then refresh.")
        return pd.DataFrame()
    if logs_global.empty or master_global.empty:
        st.warning("Player baselines are not loaded yet. Data Manager can rebuild them, or official WNBA fallback will try to build enough data to project.")

    search = st.text_input("Search player", key=f"group_search_{mode}")
    official_only = st.toggle("Official / strong signals only", value=False, key=f"group_official_only_{mode}")
    max_players = st.slider("Max player cards", 10, 120, 40, 5, key=f"group_max_{mode}")
    st.session_state["wnba_current_mode"] = mode

    proj_df = make_projection_board(lines[lines["Market"].isin(MARKETS)], logs_global, master_global, mode)
    if proj_df.empty:
        st.warning("Lines loaded, but grouped projection board could not be built. Check name matching/Data Manager.")
        return pd.DataFrame()
    proj_df["Slate"] = mode
    proj_df["SlateDate"] = str(slate_target_date(mode) or "ALL")
    proj_df = enrich_board_with_matchups(proj_df, mode)
    proj_df = apply_matchup_context_to_board(proj_df)
    proj_df = apply_daily_team_context_v2_to_board(proj_df)
    CACHE_FILES["projection_board"].parent.mkdir(exist_ok=True)
    proj_df.to_csv(CACHE_FILES["projection_board"], index=False)

    # Current live board is now built; allow user to persist it for instant reload later.
    save_cols = st.columns([1.2, 2.8])
    with save_cols[0]:
        if st.button("💾 Save Board", key=f"save_board_snapshot_{mode}", use_container_width=True):
            n = save_board_snapshot(proj_df, lines_all, mode)
            st.session_state[f"wnba_force_live_{mode}"] = False
            st.success(f"Saved {n:,} board rows. Next app open will load this board without refreshing.")
    with save_cols[1]:
        st.caption("Save Board keeps the pulled lines + projections available after closing/reopening, similar to the MLB app.")

    # Fast table / grouped cards toggle. Default to Fast table for speed; switch to
    # Player cards only when you want the full visual cards.
    return render_grouped_table_or_cards(proj_df, mode, f"live_group_{mode}", default_cards=False)

def render_data_manager_tab():
    st.subheader("Data Manager")
    st.caption("Data tools are manual-only so normal Refresh Today stays fast. Logos are disabled; this page focuses on stats/cache health.")

    st.markdown("### Data status")
    st.dataframe(dataset_status_table(), use_container_width=True)

    st.markdown("### GitHub cache fallback")
    st.caption("Optional: commit CSVs into wnba_engine/data or set WNBA_DATA_BASE_URL to a raw GitHub data folder. The app loads GitHub/cache first, then official WNBA fallback if missing.")
    st.dataframe(github_cache_status_table(), use_container_width=True)
    if st.button("🔎 Test GitHub Cache Load", use_container_width=True, key="dm_test_github_cache_load"):
        test_rows = []
        for _k in GITHUB_CACHE_DATA_KEYS:
            _df, _status = fetch_github_cache_dataset(_k)
            test_rows.append({"Dataset": _k, "Rows": 0 if _df is None or _df.empty else len(_df), "Status": _status})
        st.dataframe(pd.DataFrame(test_rows), use_container_width=True)

    st.markdown("### Official WNBA online fallback")
    st.write("Use this if Streamlit cache is empty. It pulls official WNBA player/team stats and builds a usable master baseline so live Underdog lines can still project.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🌐 Pull Official WNBA Stats + Build Baselines", use_container_width=True, key="dm_official_online_build"):
            with st.spinner("Pulling official WNBA stats and building fallback baselines..."):
                master, team_ranks, dbg = ensure_online_wnba_master_features(force_official=True)
            st.session_state["wnba_online_master_debug"] = dbg
            if master is not None and not master.empty:
                st.success(f"Official WNBA fallback master built: {len(master):,} players")
                st.dataframe(feature_missing_report(master), use_container_width=True)
            else:
                st.error("Official WNBA fallback did not return player baselines. Check Debug / Status.")
    with c2:
        if st.button("🧠 Build Advanced Features / Fix Missing Columns", use_container_width=True, key="dm_build_advanced_visible_nologo"):
            with st.spinner("Rebuilding master features from cached files, with official fallback if needed..."):
                master, team_ranks, dbg = ensure_online_wnba_master_features(force_official=False)
                if master is None or master.empty:
                    master, team_ranks = build_master_features()
                audit = feature_missing_report(master)
                audit.to_csv(DATA_DIR / "wnba_feature_missing_report.csv", index=False)
            st.success(f"Advanced features ready. Master rows: {len(master)}")
            st.dataframe(audit, use_container_width=True)

    if "wnba_online_master_debug" in st.session_state:
        with st.expander("Official WNBA fallback debug", expanded=False):
            st.dataframe(st.session_state.get("wnba_online_master_debug", pd.DataFrame()), use_container_width=True)

    st.markdown("### Remote SportsDataverse refresh")
    st.caption("Use only when you need to reload historical/stat data. Daily betting use should stay on Refresh Today.")
    default_datasets = ["player_game_logs", "player_season_stats", "team_season_stats", "schedules", "rosters", "game_rosters"]
    dataset_choices = st.multiselect(
        "SportsDataverse datasets to refresh",
        list(DATASET_LABELS.keys()),
        default=default_datasets,
        format_func=lambda k: DATASET_LABELS.get(k, k),
        key="dm_dataset_choices_visible_nologo",
    )
    include_heavy = st.toggle("Include heavier add-ons: lineups + shots", value=True, key="dm_include_heavy_visible_nologo")
    y1, y2 = st.columns(2)
    with y1:
        if st.button("Refresh SportsDataverse Database", use_container_width=True, key="dm_refresh_remote_visible_nologo"):
            with st.spinner("Refreshing SportsDataverse cache..."):
                dbg, master, team_ranks = refresh_data_and_build_advanced_features(dataset_choices, [int(season_last), int(season_now)], include_heavy)
            st.success(f"SportsDataverse refresh complete. Master rows: {0 if master is None else len(master)}")
            st.dataframe(dbg, use_container_width=True)
    with y2:
        if st.button("Download missing-field report", use_container_width=True, key="dm_report_button_nologo"):
            report_path = DATA_DIR / "wnba_feature_missing_report.csv"
            if report_path.exists():
                st.download_button("Download CSV", report_path.read_text(errors="ignore"), "wnba_feature_missing_report.csv", "text/csv", use_container_width=True)
            else:
                st.info("No report yet. Build features first.")

    st.markdown("### Manual upload/import backup")
    uploaded = st.file_uploader("Upload SportsDataverse CSV/Parquet files", type=["csv", "parquet", "xlsx", "json"], accept_multiple_files=True, key="dm_manual_upload_visible_nologo")
    if uploaded and st.button("Import uploaded files", use_container_width=True, key="dm_import_uploads_visible_nologo"):
        rows = []
        for f in uploaded:
            try:
                raw = f.read(); dataset_key = classify_filename(f.name)
                if not dataset_key:
                    rows.append({"file": f.name, "dataset": "unknown", "status": "skipped: could not classify", "rows": 0}); continue
                df = read_any_file(raw, f.name); std = standardize_dataset(dataset_key, df)
                if std is not None and not std.empty:
                    save_dataset(dataset_key, std); rows.append({"file": f.name, "dataset": dataset_key, "status": "saved", "rows": len(std)})
                else:
                    rows.append({"file": f.name, "dataset": dataset_key, "status": "standardized empty", "rows": 0})
            except Exception as e:
                rows.append({"file": getattr(f, 'name', 'upload'), "dataset": "error", "status": str(e)[:180], "rows": 0})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        master, _, dbg = ensure_online_wnba_master_features(force_official=False)
        st.success(f"Files imported and master ready: {0 if master is None else len(master)} rows")

    st.markdown("### Cached exports")
    for key in ["master_features", "projection_board", "team_ranks", "player_game_logs", "player_season_stats"]:
        path = CACHE_FILES.get(key)
        if path and path.exists():
            try:
                st.download_button(f"Download {key}.csv", path.read_text(errors="ignore"), f"{key}.csv", "text/csv", use_container_width=True, key=f"dm_download_{key}_nologo")
            except Exception:
                pass


with st.sidebar:
    st.header("Setup")
    season_now = st.number_input("Current season", min_value=2020, max_value=2032, value=datetime.now().year, step=1)
    season_last = st.number_input("Last season baseline", min_value=2020, max_value=2032, value=datetime.now().year - 1, step=1)
    use_ud = st.toggle("Pull Underdog", value=True)
    use_sleeper = False
    use_odds_api = False
    odds_api_key = ""
    st.caption("Active line sources: Underdog + saved manual lines. Sleeper, Odds API, and SportsGameOdds are disabled.")
    use_remote = st.toggle("Allow SportsDataverse remote downloads", value=True)
    use_xgb_blend = st.toggle("Use XGBoost/GBM blend", value=True, help="Keep ON when you want the ensemble projection blended in. Turn OFF while testing if graded samples are still too small.")
    st.session_state["use_xgb_blend"] = bool(use_xgb_blend)
    auto_final_grade = st.toggle("Auto-check final ESPN results + grade", value=False, help="When ON and the app is open, it checks ESPN final boxscores every few minutes and grades only finished games for today. It does not grade future/not-started games.")
    st.session_state["auto_final_grade"] = bool(auto_final_grade)
    st.markdown("**Markets active:** PTS, REB, AST, PRA")
    st.markdown("**Model:** Monte Carlo + Bayesian confidence" + (" + XGBoost/GBM blend" if use_xgb_blend else ""))
    st.divider()
    if st.button("🔄 Refresh Today — All-in-One", use_container_width=True):
        run_one_click_refresh_today("Today", use_ud)
        st.rerun()
    st.caption("This one button refreshes schedule/context, rebuilds feature cache when logs exist, pulls Underdog/manual lines, and rebuilds the projection board.")

@st.cache_data(show_spinner=False)
def get_global_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lazily load player_game_logs and master_features on first use.

    Wrapped in st.cache_data so the blocking I/O (local CSV read or remote
    GitHub fetch) only happens once per session rather than on every Streamlit
    rerun, and — critically — never at module import time, which was causing
    the 2-minute startup timeout.
    """
    _logs = load_dataset("player_game_logs")
    _master = load_dataset("master_features")
    if _master.empty and not _logs.empty:
        try:
            _master, _ = build_master_features()
        except Exception:
            _master = pd.DataFrame()
    return _logs, _master

# Lazy load: initialize empty, will load on first UI access
logs_global = pd.DataFrame()
master_global = pd.DataFrame()

# MLB-style final-result check: only runs while app is open, throttled so tab switching is safe.
if st.session_state.get("auto_final_grade", False):
    last_auto = pd.to_datetime(st.session_state.get("last_auto_final_grade_check", None), errors="coerce")
    should_check = pd.isna(last_auto) or (pd.Timestamp.now() - last_auto).total_seconds() > 300
    if should_check:
        with st.spinner("Checking ESPN final boxscores and grading finished games..."):
            try:
                n_auto, dbg_auto = pull_final_results_and_grade("Today")
                st.session_state["last_auto_final_grade_check"] = now_iso()
                st.session_state["last_auto_final_grade_count"] = int(n_auto)
                st.session_state["last_auto_final_grade_dbg"] = dbg_auto.to_dict("records") if dbg_auto is not None else []
                # Invalidate the cache so the refreshed logs are picked up on the next rerun.
                get_global_datasets.clear()
                logs_global, master_global = get_global_datasets()
            except Exception as e:
                st.session_state["last_auto_final_grade_error"] = str(e)[:220]

# Pull state once for hero summary without forcing rerun behavior.
try:
    hero_lines = st.session_state.get("wnba_lines_all", pd.DataFrame())
    hero_board = load_dataset("projection_board")
    hero_board_rows = 0 if hero_board.empty else len(hero_board)
    hero_real_lines = 0 if hero_lines is None or hero_lines.empty else len(hero_lines)
    hero_strong = 0 if hero_board.empty or "Official" not in hero_board.columns else int(hero_board["Official"].astype(str).str.contains("OVER|UNDER", na=False).sum())
    hero_no_line = max(0, (0 if master_global is None or master_global.empty else len(master_global)) - hero_real_lines)
except Exception:
    hero_board_rows = hero_real_lines = hero_no_line = hero_strong = 0
hero_panel(hero_board_rows, hero_real_lines, hero_no_line, hero_strong)


st.markdown("""
<style>
.owp-group-card{border:1px solid rgba(168,85,247,.48);border-left:6px solid #a855f7;border-radius:28px;padding:22px;margin:18px 0;background:linear-gradient(180deg,rgba(17,10,31,.98),rgba(24,10,42,.94));box-shadow:0 0 28px rgba(168,85,247,.12)}
.owp-group-top{display:flex;justify-content:space-between;gap:14px;align-items:flex-start;margin-bottom:16px}.owp-group-best{text-align:right;font-weight:1000;color:#e9d5ff;font-size:.95rem}.owp-group-best span{color:#c4b5fd;font-weight:700}.owp-market-row{background:rgba(15,23,42,.58);border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:14px;margin:12px 0}.owp-market-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.owp-market-name{font-size:1.05rem;font-weight:1000;color:#fb7185;letter-spacing:.08em}.owp-market-vol{font-size:.75rem;border:1px solid rgba(250,204,21,.55);color:#fde68a;border-radius:999px;padding:2px 8px;font-weight:900}.owp-market-conf{margin-left:auto;border:1px solid rgba(34,197,94,.45);border-radius:999px;padding:3px 10px;font-weight:1000;font-size:.85rem}.owp-market-side{font-weight:1000;font-size:1.35rem}.owp-market-conf.over,.owp-market-side.over,.owp-market-edge.pos{color:#4ade80}.owp-market-conf.under,.owp-market-side.under,.owp-market-edge.neg{color:#fb7185}.owp-market-conf.track,.owp-market-side.track,.owp-market-edge.flat{color:#e9d5ff}.owp-market-main{display:flex;align-items:baseline;gap:12px;margin-top:10px}.owp-market-proj{font-size:2.25rem;font-weight:1000;color:#f8fafc}.owp-market-edge{font-size:1.15rem;font-weight:1000}.owp-market-tier{color:#facc15;font-weight:900}.owp-market-track{height:10px;margin-top:10px}.owp-market-sub{display:flex;justify-content:space-between;color:#c4b5fd;font-size:.82rem;text-transform:uppercase;font-weight:900}.owp-market-note{font-size:.86rem;color:#a8a29e;margin-top:8px;background:rgba(255,255,255,.04);border-left:3px solid rgba(251,113,133,.6);border-radius:8px;padding:8px 10px}
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["Player Cards", "Best Bets", "Official + Grade", "Data Manager", "Debug / Status", "Model Reports"])

with tabs[0]:
    st.markdown("<div class='section-title'>PLAYER CARDS / Grouped Markets</div>", unsafe_allow_html=True)
    st.caption("One player card now shows every live market pulled for that player: PTS, REB, AST, and PRA. The Underdog line pull and main-line selector are unchanged.")
    slate_tabs = st.tabs(["Today", "Tomorrow", "All Lines"])
    with slate_tabs[0]:
        render_grouped_player_board("Today", use_ud, logs_global, master_global)
    with slate_tabs[1]:
        render_grouped_player_board("Tomorrow", use_ud, logs_global, master_global)
    with slate_tabs[2]:
        render_grouped_player_board("All Lines", use_ud, logs_global, master_global)

with tabs[1]:
    st.subheader("Best Bets / Tier 1–8 Official Board")
    st.caption("Clean official board using edge, Monte Carlo, Bayesian confidence, data score, role confidence, line source reliability, similarity, pace, rest/travel, blowout, bench rotation, line movement, EV/Kelly, and model disagreement.")
    board_path = CACHE_FILES["projection_board"]
    if board_path.exists():
        try:
            bb = pd.read_csv(board_path)
        except Exception:
            bb = pd.DataFrame()
    else:
        bb = pd.DataFrame()
    if bb.empty:
        st.warning("No projection board cached yet. Refresh a PTS/REB/AST/PRA board first.")
    else:
        tier_options = sorted(bb.get("Tier", pd.Series(dtype=str)).dropna().unique().tolist())
        tier_filter = st.multiselect("Tier filter", tier_options, default=tier_options[:4] if tier_options else [])
        show = bb.copy()
        if tier_filter and "Tier" in show.columns:
            show = show[show["Tier"].isin(tier_filter)]
        sort_cols = [c for c in ["Official Play Score", "Edge"] if c in show.columns]
        if sort_cols:
            show = show.sort_values(sort_cols, ascending=False)
        st.metric("Best Bet Rows", len(show))
        card_view = st.toggle("Show player cards", value=True, key="best_bets_card_view")
        if card_view:
            for _, rr in show.head(40).iterrows():
                render_card(rr)
        display_cols = [c for c in ["Tier", "Player", "Team", "Opponent", "Matchup", "Market", "Line", "Projection", "Edge", "Lean", "Official", "Official Play Score", "Over %", "Under %", "Volatility", "Model Agreement", "PASS Reason", "Feature Importance"] if c in show.columns]
        st.dataframe(show[display_cols] if display_cols else show, use_container_width=True)
        st.download_button("Download best bets CSV", show.to_csv(index=False), "wnba_best_bets.csv", "text/csv")

with tabs[2]:
    st.subheader("Official + Grade")
    st.caption("Save official plays before games. Grade after results are imported. The Results tab shows ✅/❌ by player and market so you can quickly see what cleared the line.")
    grade_tabs = st.tabs(["Save / Grade", "After Game Results ✅❌", "Raw Logs"])
    with grade_tabs[0]:
        board = load_dataset("projection_board")
        diag_scope_preview = st.selectbox("Grade diagnostics scope", ["Today", "Tomorrow", "All pending"], index=0, key="grade_diag_scope_after_results")
        diag_mode = None if diag_scope_preview == "All pending" else diag_scope_preview
        diag = grade_diagnostics(logs_global, diag_mode)
        with st.expander("AutoGrader status / why it may grade 0", expanded=False):
            st.json(diag)
            st.caption("If Matched Completed Logs is 0, import/refresh final player logs first. ESPN screenshots confirm results, but the app needs those stats in the player logs table to auto-grade.")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("💾 Save official before games", use_container_width=True):
                if board.empty:
                    st.warning("No projection board cached yet. Refresh a market board first.")
                else:
                    n = save_officials(board)
                    st.success(f"Saved {n} official plays.")
        with c2:
            grade_scope = st.selectbox("Grade scope", ["Today", "Tomorrow", "All pending"], index=0, key="grade_scope_after_results")
            if st.button("📊 Grade pending after results", use_container_width=True):
                mode_arg = None if grade_scope == "All pending" else grade_scope
                n = grade_pending(logs_global, mode_arg)
                st.success(f"AutoGrader updated {n} pending plays for {grade_scope}.")
            if st.button("🏁 Pull final ESPN results + grade finished games", use_container_width=True):
                mode_arg = "Today" if grade_scope == "All pending" else grade_scope
                n, dbg = pull_final_results_and_grade(mode_arg)
                st.success(f"Pulled final ESPN boxscores and graded {n} finished-game plays for {mode_arg}.")
                with st.expander("Final-result pull diagnostics", expanded=True):
                    st.dataframe(dbg, use_container_width=True)
        with c3:
            st.metric("Current board", 0 if board.empty else len(board))
        st.info("AutoGrader only marks plays when a completed player log exists. The ESPN final-results button can pull those logs after the game. Future/not-started games remain pending/skipped.")
    official = pd.DataFrame(load_json(OFFICIAL_LOG, []))
    learning = pd.DataFrame(load_json(LEARNING_LOG, []))
    with grade_tabs[1]:
        st.markdown("### After Game Results")
        results_source = learning if not learning.empty else official
        results = build_after_game_results_table(results_source)
        if results.empty:
            st.info("No graded results yet. After the game, import/refresh player logs and click Grade pending after results.")
        else:
            status_filter = st.multiselect("Result filter", sorted(results["Result"].dropna().astype(str).unique().tolist()) if "Result" in results.columns else [], default=sorted(results["Result"].dropna().astype(str).unique().tolist()) if "Result" in results.columns else [])
            show_results = results.copy()
            if status_filter and "Result" in show_results.columns:
                show_results = show_results[show_results["Result"].astype(str).isin(status_filter)]
            c1, c2, c3, c4 = st.columns(4)
            total = len(show_results)
            wins = int((show_results.get("Result", pd.Series(dtype=str)).astype(str) == "WIN").sum()) if total else 0
            losses = int((show_results.get("Result", pd.Series(dtype=str)).astype(str) == "LOSS").sum()) if total else 0
            pushes = int((show_results.get("Result", pd.Series(dtype=str)).astype(str) == "PUSH").sum()) if total else 0
            c1.metric("Graded", total)
            c2.metric("✅ Wins", wins)
            c3.metric("❌ Losses", losses)
            c4.metric("Push", pushes)
            if total:
                st.metric("Win rate", f"{wins}/{total} ({wins/total:.1%})")
            st.dataframe(show_results, use_container_width=True)
            st.download_button("Download after-game results CSV", show_results.to_csv(index=False), "wnba_after_game_results.csv", "text/csv")
    with grade_tabs[2]:
        if not official.empty:
            st.markdown("### Official snapshot log")
            st.dataframe(official.tail(300), use_container_width=True)
            st.download_button("Download official log CSV", official.to_csv(index=False), "wnba_official_pick_log.csv", "text/csv")
        else:
            st.info("No official plays saved yet.")
        if not learning.empty:
            st.markdown("### Learning log")
            if "Result" in learning.columns:
                total = len(learning)
                wins = (learning["Result"] == "WIN").sum()
                st.metric("Learning win rate", f"{wins}/{total} ({wins/total:.1%})" if total else "0/0")
            st.dataframe(learning.tail(300), use_container_width=True)
            st.download_button("Download learning log CSV", learning.to_csv(index=False), "wnba_learning_log.csv", "text/csv")

with tabs[3]:
    render_data_manager_tab()

with tabs[4]:
    st.subheader("Debug / Status")
    st.caption("Diagnostics only. Heavy imports/rebuilds are in Data Manager and never run automatically.")
    st.markdown("### Data status")
    st.dataframe(dataset_status_table(), use_container_width=True)
    st.markdown("### Aggregated real lines")
    lines, ud_debug, sl_debug = get_lines_from_state_or_pull(use_ud, False, False, "")
    render_source_status_card(lines, ud_debug, sl_debug, False, "")
    st.dataframe(lines, use_container_width=True)
    st.markdown("### Underdog debug")
    st.dataframe(ud_debug, use_container_width=True)

    st.markdown("### Underdog Decode Mode")
    decode_path = DATA_DIR / "wnba_underdog_decode.csv"
    if decode_path.exists():
        try:
            decode_df = pd.read_csv(decode_path, low_memory=False)
            st.caption("Raw Underdog row → parsed market/line → resolved player → accepted/rejected reason.")
            st.dataframe(decode_df.tail(250), use_container_width=True)
            st.download_button("Download Underdog decode CSV", decode_df.to_csv(index=False), "wnba_underdog_decode.csv", "text/csv")
        except Exception as e:
            st.warning(f"Decode file exists but could not be read: {e}")
    else:
        st.info("No decode file yet. Click Refresh / Pull Lines first.")

    with st.expander("PrizePicks test pull — debug only", expanded=False):
        st.caption("This only tests whether a public PrizePicks JSON response is reachable. It does not feed projections yet.")
        if st.button("Test PrizePicks public pull", use_container_width=True):
            pp_dbg = fetch_prizepicks_test_pull()
            st.dataframe(pp_dbg, use_container_width=True)

    st.markdown("### Manual line debug")
    st.dataframe(sl_debug, use_container_width=True)
    st.markdown("### Daily Team Context Cache 2.0")
    ctx_debug = st.session_state.get("wnba_daily_team_context_v2", pd.DataFrame())
    if (ctx_debug is None or ctx_debug.empty) and DAILY_TEAM_CONTEXT_FILE.exists():
        try:
            ctx_debug = pd.read_csv(DAILY_TEAM_CONTEXT_FILE, low_memory=False)
        except Exception:
            ctx_debug = pd.DataFrame()
    st.dataframe(ctx_debug, use_container_width=True)
    st.markdown("### Cached master preview")
    st.dataframe(master_global.head(50), use_container_width=True)

with tabs[5]:
    st.subheader("Model Reports: AutoGrader / CLV / Calibration / Backtest")
    st.caption("This page keeps the main UI clean while giving you the same deeper review tools: line movement, closing-line value, projection calibration, and historical model testing.")

    r1, r2, r3, r4 = st.columns(4)
    official_df = pd.DataFrame(load_json(OFFICIAL_LOG, []))
    learning_df = pd.DataFrame(load_json(LEARNING_LOG, []))
    lm_df = line_movement_report()
    with r1:
        pending = int((official_df.get("Result", pd.Series(dtype=str)) == "PENDING").sum()) if not official_df.empty and "Result" in official_df.columns else 0
        st.metric("Pending grades", pending)
    with r2:
        graded = int(len(learning_df)) if not learning_df.empty else 0
        st.metric("Graded plays", graded)
    with r3:
        st.metric("Line snapshots", int(len(pd.DataFrame(load_json(LINE_HISTORY_FILE, [])))))
    with r4:
        if not learning_df.empty and "Result" in learning_df.columns:
            wr = (learning_df["Result"] == "WIN").sum() / max(1, (learning_df["Result"].isin(["WIN", "LOSS"])).sum())
            st.metric("Tracked win rate", f"{wr:.1%}")
        else:
            st.metric("Tracked win rate", "N/A")

    st.markdown("### 1) Result AutoGrader")
    st.write("Uses imported SportsDataverse player logs to grade saved official plays. It matches the first player game after the pick's saved/start time, then writes WIN/LOSS/PUSH, actual value, closing line, and CLV.")
    if st.button("Run AutoGrader now", type="primary", use_container_width=True):
        n = grade_pending(logs_global)
        st.success(f"AutoGrader updated {n} pending plays.")
    refreshed_official = pd.DataFrame(load_json(OFFICIAL_LOG, []))
    if not refreshed_official.empty:
        cols = [c for c in ["SavedAt", "Player", "Team", "Opponent", "Matchup", "Market", "Line", "Projection", "Lean", "Actual", "Result", "ClosingLine", "CLV", "GradeNote"] if c in refreshed_official.columns]
        st.dataframe(refreshed_official.tail(250)[cols] if cols else refreshed_official.tail(250), use_container_width=True)

    st.markdown("### 2) Line Movement + CLV Dashboard")
    lm_df = line_movement_report()
    if lm_df.empty:
        st.info("No line movement snapshots yet. Refresh board lines a few times and/or save official plays to build this database.")
    else:
        st.dataframe(lm_df.head(300), use_container_width=True)
        st.download_button("Download line movement CSV", lm_df.to_csv(index=False), "wnba_line_movement.csv", "text/csv")
    if not learning_df.empty and "CLV" in learning_df.columns:
        clv = learning_df.copy()
        clv["CLV"] = pd.to_numeric(clv["CLV"], errors="coerce")
        st.markdown("#### CLV by Market")
        clv_sum = clv.groupby("Market", dropna=False).agg(Plays=("CLV", "count"), AvgCLV=("CLV", "mean"), PositiveCLV=("CLV", lambda x: (pd.to_numeric(x, errors='coerce') > 0).mean())).reset_index()
        st.dataframe(clv_sum, use_container_width=True)

    st.markdown("### 3) Model Calibration Report")
    learn_raw, cal = calibration_report()
    if cal.empty:
        st.info("No graded learning data yet. Save official plays, import final player logs, then run AutoGrader.")
    else:
        st.dataframe(cal, use_container_width=True)
        st.download_button("Download calibration CSV", cal.to_csv(index=False), "wnba_model_calibration.csv", "text/csv")
        with st.expander("Raw graded learning data", expanded=False):
            st.dataframe(learn_raw.tail(500), use_container_width=True)

    st.markdown("### 4) Automated Historical Backtest")
    st.caption("Backtests the projection formula on historical logs using a prior-games-only line proxy. This validates model direction/calibration without claiming it had real sportsbook historical lines.")
    min_prior = st.slider("Minimum prior games before testing", 3, 15, 5)
    if st.button("Run historical backtest", use_container_width=True):
        bt = build_historical_backtest(logs_global, min_prior_games=min_prior)
        st.session_state["wnba_backtest_df"] = bt
    bt = st.session_state.get("wnba_backtest_df", pd.DataFrame())
    if bt is None or bt.empty:
        st.info("Run the backtest after player logs are imported.")
    else:
        bts = summarize_backtest(bt)
        st.dataframe(bts, use_container_width=True)
        st.download_button("Download backtest summary CSV", bts.to_csv(index=False), "wnba_backtest_summary.csv", "text/csv")
        with st.expander("Backtest rows", expanded=False):
            st.dataframe(bt.tail(1000), use_container_width=True)
            st.download_button("Download full backtest rows CSV", bt.to_csv(index=False), "wnba_backtest_rows.csv", "text/csv")
