# -*- coding: utf-8 -*-
"""
ONE WAY PICKZ — WNBA Prop Engine
Full single-file Streamlit app.

Markets: PTS / REB / AST / PRA
Main fix in this version:
- Replaces the broken WNBA Stats API baseline button with a SportsDataverse Import Wizard.
- Reads CSV + Parquet uploads.
- Handles SportsDataverse index CSVs by downloading/combining the referenced files when possible.
- Builds clean local caches and one master feature table.
- Keeps MLB-style dark UI, player cards, official save/grade, manual lines, learning logs.
"""

import os
import re
import io
import json
import math
import time
import zipfile
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

APP_VERSION = "WNBA v1.4 — SportsDataverse Import Engine v2 + Robust CSV/Parquet Mapping"

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
    path = CACHE_FILES.get(dataset_key)
    if not path or not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
        if dataset_key in ["player_game_logs", "schedules", "game_rosters", "lineups", "shots", "master_features"]:
            for c in ["GameDate"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
        return df
    except Exception:
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
    b = normalize_name(blob)
    if any(x in b for x in ["fantasy", "steal", "block", "turnover", "three", "3 pointer", "free throw"]):
        return None
    if ("points" in b and "rebounds" in b and "assists" in b) or "pts reb ast" in b or "pra" in b:
        return "PRA"
    if "point" in b or "pts" in b:
        return "PTS"
    if "rebound" in b or "reb" in b:
        return "REB"
    if "assist" in b or "ast" in b:
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


@st.cache_data(ttl=240, show_spinner=False)
def fetch_underdog_board():
    rows, debug = [], []
    for url in UNDERDOG_URLS:
        data = request_json(url, timeout=20)
        if not data:
            debug.append({"source": "Underdog", "url": url, "status": "no json"})
            continue
        objects = flatten_json(data)
        by_id = {str(o.get("id")): o for o in objects if isinstance(o, dict) and o.get("id") not in [None, ""]}
        line_objs = []
        for o in objects:
            typ = str(o.get("type") or o.get("_parent_key") or "").lower().replace("-", "_")
            a = attrs(o)
            if "over_under_line" in typ or any(a.get(k) not in [None, ""] for k in ["stat_value", "line_score", "over_under_line", "target_value"]):
                line_objs.append(o)
        for lo in line_objs:
            ou = by_id.get(str(rel_id(lo, ["over_under", "over_unders"])))
            app = by_id.get(str(rel_id(lo, ["appearance", "appearances"])))
            player = by_id.get(str(rel_id(lo, ["player", "players"])))
            if not player and app:
                player = by_id.get(str(rel_id(app, ["player", "players"])))
            blob = text_blob(lo, ou, app, player)
            low = blob.lower() + " " + json.dumps(attrs(lo), default=str).lower()
            if any(x in low for x in ["mlb", "baseball", "nfl", "football", "nhl", "tennis", "golf", "mma", "soccer"]):
                continue
            market = infer_market(blob + " " + low)
            if market not in MARKETS:
                continue
            line = line_from_obj(lo, ou)
            if pd.isna(line):
                continue
            player_name = player_from_obj(player, app, ou, lo)
            if not player_name:
                continue
            status_blob = " ".join(str(attrs(o).get(k, "")) for o in [lo, ou, app] if isinstance(o, dict) for k in ["status", "state", "hidden", "active"]).lower()
            if any(x in status_blob for x in ["suspended", "closed", "hidden", "inactive", "disabled"]):
                continue
            rows.append({
                "Player": player_name, "Team": attrs(player).get("team") or attrs(app).get("team") or "",
                "Market": market, "Line": float(line), "Source": "Underdog", "Start": attrs(app).get("start_time") or attrs(lo).get("start_time") or "",
                "Raw": blob[:180]
            })
        if rows:
            break
    df = pd.DataFrame(rows).drop_duplicates(subset=["Player", "Market", "Line", "Source"]) if rows else pd.DataFrame(columns=["Player", "Team", "Market", "Line", "Source", "Start", "Raw"])
    return df, pd.DataFrame(debug)


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
    This is a fallback/secondary source, not a replacement for Underdog/Sleeper.
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
                    rec = paired.setdefault(key, {
                        "Player": player, "Team": "", "Market": internal_market, "Line": float(line),
                        "Source": f"OddsAPI:{book_key}", "Start": ev.get("commence_time") or "",
                        "Raw": f"{ev.get('away_team','')} @ {ev.get('home_team','')}", "OverOdds": np.nan, "UnderOdds": np.nan
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


def aggregate_lines(use_ud=True, use_sleeper=True, manual_df=None, use_odds_api=False, odds_api_key: str = "", line_upload_df=None):
    frames = []
    ud_debug = pd.DataFrame(); sl_debug = pd.DataFrame(); extra_debug = []
    if use_ud:
        ud, ud_debug = fetch_underdog_board(); frames.append(ud)
    if use_sleeper:
        sl, sl_debug = fetch_sleeper_board(); frames.append(sl)
    if use_odds_api:
        oa, oa_debug = fetch_odds_api_board(odds_api_key)
        frames.append(oa)
        if oa_debug is not None and not oa_debug.empty:
            extra_debug.extend(oa_debug.to_dict("records"))
    if manual_df is not None and len(manual_df):
        m = manual_df.copy(); m["Source"] = m.get("Source", "Manual").fillna("Manual"); frames.append(m)
    if line_upload_df is not None and len(line_upload_df):
        frames.append(line_upload_df)
    if not frames:
        combined_debug = pd.concat([sl_debug, pd.DataFrame(extra_debug)], ignore_index=True) if extra_debug else sl_debug
        return pd.DataFrame(), ud_debug, combined_debug
    board = pd.concat(frames, ignore_index=True)
    if board.empty:
        combined_debug = pd.concat([sl_debug, pd.DataFrame(extra_debug)], ignore_index=True) if extra_debug else sl_debug
        return board, ud_debug, combined_debug
    board["Market"] = board["Market"].astype(str).str.upper().map(lambda x: "PRA" if "PRA" in x else x)
    board = board[board["Market"].isin(MARKETS)].copy()
    board["Line"] = pd.to_numeric(board["Line"], errors="coerce")
    board = board.dropna(subset=["Player", "Market", "Line"])
    for c in ["Team", "Source", "Start", "Raw", "OverOdds", "UnderOdds"]:
        if c not in board.columns:
            board[c] = "" if c not in ["OverOdds", "UnderOdds"] else np.nan
    board["NameKey"] = board["Player"].map(normalize_name)
    def source_priority(s):
        s = str(s)
        if s == "Underdog": return 1
        if s == "Sleeper": return 2
        if s.startswith("OddsAPI"): return 3
        if s in ["Manual", "CSV Upload"]: return 4
        return 9
    board["Priority"] = board["Source"].map(source_priority)
    # Store all source alternatives but keep the top source first. Projection board can still show line shopping columns.
    board = board.sort_values(["NameKey", "Market", "Priority", "Line"]).drop_duplicates(subset=["NameKey", "Market", "Source", "Line"], keep="first")
    combined_debug = pd.concat([sl_debug, pd.DataFrame(extra_debug)], ignore_index=True) if extra_debug else sl_debug
    return board.sort_values(["NameKey", "Market", "Priority"]), ud_debug, combined_debug

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
        primary["Sleeper Line"] = safe_float(grp[grp["Source"] == "Sleeper"]["Line"].iloc[0], np.nan) if len(grp[grp["Source"] == "Sleeper"]) else np.nan
        primary["Manual Line"] = safe_float(grp[grp["Source"] == "Manual"]["Line"].iloc[0], np.nan) if len(grp[grp["Source"] == "Manual"]) else np.nan
        primary["Best Over Line"] = grp["Line"].min()
        primary["Best Under Line"] = grp["Line"].max()
        primary["Line Source Reliability"] = {"Underdog": 95, "Sleeper": 85, "Manual": 60}.get(str(primary.get("Source")), 50)
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


# Preserve prior projection builder and enhance it with full advanced engines.
_make_projection_board_core = make_projection_board

def make_projection_board(lines, logs, base):
    core = _make_projection_board_core(lines, logs, base)
    if core is None or core.empty:
        return core
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
        if pd.notna(p0) and pd.notna(px):
            row["Ensemble Projection"] = round(0.72*p0 + 0.28*px, 2)
            row["Projection"] = row["Ensemble Projection"]
            row["Edge"] = round(row["Projection"] - safe_float(row.get("Line"), np.nan), 2)
        inj = injury_ripple_engine(row, b); row.update(inj)
        opp = opponent_lineup_adjustment(row, b); row.update(opp)
        ref = referee_tendency_engine(row); row.update(ref)
        trav = latest_travel_context(logs, normalize_name(row.get("Matched Player") or row.get("Player")), row.get("Team")); row.update(trav)
        # Apply small final additive context to edge/projection.
        context_add = safe_float(row.get("Injury Ripple Bump"),0) + safe_float(row.get("Opponent Lineup Adj"),0) + safe_float(row.get("Referee Factor"),0) + safe_float(row.get("Travel Tax"),0)
        if pd.notna(safe_float(row.get("Projection"), np.nan)):
            row["Projection"] = round(safe_float(row.get("Projection")) + context_add, 2)
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
        row["Feature Importance"] = feature_importance_text(row) + " | XGB: " + str(row.get("XGBoost Feature Importance", ""))
        row["Full Engine Note"] = "Similarity + trained ML + referee + travel + injury ripple + opponent lineup + CLV + EV/Kelly active."
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


def grade_pending(logs):
    official = load_json(OFFICIAL_LOG, [])
    if not official or logs is None or logs.empty:
        return 0
    updated = 0
    learn = load_json(LEARNING_LOG, [])
    for row in official:
        if row.get("Result") != "PENDING":
            continue
        key = normalize_name(row.get("Matched Player") or row.get("Player"))
        market = row.get("Market")
        if market not in logs.columns:
            continue
        d = logs[logs["NameKey"] == key].copy()
        if d.empty:
            continue
        actual = safe_float(d.sort_values("GameDate").iloc[-1].get(market), np.nan)
        if pd.isna(actual):
            continue
        lean = str(row.get("Lean", ""))
        line = safe_float(row.get("Line"), np.nan)
        if pd.isna(line):
            continue
        win = (actual > line and lean == "OVER") or (actual < line and lean == "UNDER")
        row["Actual"] = actual; row["Result"] = "WIN" if win else "LOSS"; row["GradedAt"] = now_iso()
        learn.append(row.copy())
        updated += 1
    save_json(OFFICIAL_LOG, official)
    save_json(LEARNING_LOG, learn)
    return updated


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
        <div>
          <div class='owp-player'>{_val(r.get('Player'))}</div>
          <div class='owp-match'>{matchup} <span class='owp-muted'>| {_val(r.get('PositionGroup'), 'Role')} | {slate} {slate_date}</span></div>
          <span class='owp-pill owp-pill-source'>{source}</span>
          <span class='owp-pill owp-pill-role'>Lineup/Role {_val(r.get('Minutes Safety'), 'NA')}</span>
          <span class='owp-pill owp-pill-score'>Score {confidence}/100</span>
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
            "Bench Rotation Note", "Pace Projection Note", "Line Movement Note", "Referee Note", "HomeAway Note",
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
      <div class='owp-title'>💜 WNBA PROP ENGINE v1.5<br/>PLAYER CARDS + EXPLANATION ENGINE</div>
      <div class='owp-subtitle'>Strict WNBA-only prop line lock → Refresh → Save → Grade</div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 REFRESH LIVE BOARD — Do Not Save Yet", use_container_width=True, key="hero_refresh_live_board"):
            clear_line_pull_caches()
            pull_board_lines(use_ud, use_sleeper, use_odds_api, odds_api_key)
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
          <div class='owp-kpi-card'><div class='owp-kpi-label'>Real Lines</div><div class='owp-kpi-value'>{real_lines}</div><div class='owp-kpi-sub'>Underdog/Sleeper/Manual</div></div>
          <div class='owp-kpi-card'><div class='owp-kpi-label'>No Line</div><div class='owp-kpi-value'>{no_line}</div><div class='owp-kpi-sub'>Tracked only</div></div>
          <div class='owp-kpi-card'><div class='owp-kpi-label'>Strong Signals</div><div class='owp-kpi-value'>{strong}</div><div class='owp-kpi-sub'>Official gate passed</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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


def filter_lines_for_slate(lines: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, str]:
    if lines is None or lines.empty or mode == "All Lines":
        return lines if lines is not None else pd.DataFrame(), "All loaded lines shown."
    target = slate_target_date(mode)
    if target is None:
        return lines, "All loaded lines shown."
    d = lines.copy()
    start_dates = line_start_date_series(d)
    has_dates = start_dates.notna()
    if has_dates.any():
        filtered = d[(start_dates == target) | (~has_dates & d.get("Source", "").astype(str).eq("Manual"))].copy()
        return filtered, f"Filtered to {mode.lower()} ({target}). Manual lines without a start date are included."
    return d, f"Sportsbook rows did not include reliable start times, so all loaded lines are shown for {mode.lower()}."


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


def clear_line_pull_caches():
    for fn in [fetch_underdog_board, fetch_sleeper_board, fetch_odds_api_board]:
        try:
            fn.clear()
        except Exception:
            pass


def pull_board_lines(use_ud_flag: bool, use_sleeper_flag: bool, use_odds_api_flag: bool = False, odds_api_key: str = "") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Production mode: manual lines and CSV line fallbacks are intentionally disabled/hidden.
    # Only real sources feed the board: Underdog, Sleeper, and optional Odds API.
    lines, ud_debug, sl_debug = aggregate_lines(
        use_ud=use_ud_flag,
        use_sleeper=use_sleeper_flag,
        use_odds_api=use_odds_api_flag,
        odds_api_key=odds_api_key,
        manual_df=pd.DataFrame(),
        line_upload_df=pd.DataFrame(),
    )
    st.session_state["wnba_lines_all"] = lines
    st.session_state["wnba_ud_debug"] = ud_debug
    st.session_state["wnba_sl_debug"] = sl_debug
    st.session_state["wnba_last_refresh"] = now_iso()
    return lines, ud_debug, sl_debug


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
            "Projection Explanation": f"Baseline {m} projection from {proj_col}. Add Underdog/Sleeper/Odds API line to turn this into an official play.",
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

def _team_key_for_matchup(x: Any) -> str:
    try:
        return team_abbrev(x)
    except Exception:
        return str(x or "").strip().upper()[:3]


def enrich_board_with_matchups(proj_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Attach Opponent/HomeAway/Matchup from cached schedules so every player card can show who they face."""
    if proj_df is None or proj_df.empty:
        return proj_df
    out = proj_df.copy()
    for c in ["Team", "Opponent", "HomeAway", "Matchup"]:
        if c not in out.columns:
            out[c] = ""
    sched = schedule_for_slate(mode)
    if sched is None or sched.empty:
        # Fallback: preserve existing matchup if line source supplied one.
        out["Matchup"] = out.apply(lambda r: r.get("Matchup") or (f"{r.get('Team','')} vs {r.get('Opponent','')}" if r.get('Team') and r.get('Opponent') else r.get('Team','')), axis=1)
        return out
    sched = sched.copy()
    sched["HomeKey"] = sched.get("Home", "").map(_team_key_for_matchup)
    sched["AwayKey"] = sched.get("Away", "").map(_team_key_for_matchup)
    for i, r in out.iterrows():
        team = _team_key_for_matchup(r.get("Team"))
        if not team:
            continue
        hit = sched[(sched["HomeKey"] == team) | (sched["AwayKey"] == team)]
        if hit.empty:
            continue
        g = hit.iloc[0]
        home = str(g.get("Home", "")); away = str(g.get("Away", ""))
        home_key = _team_key_for_matchup(home); away_key = _team_key_for_matchup(away)
        if team == home_key:
            out.at[i, "Opponent"] = away_key or away
            out.at[i, "HomeAway"] = "HOME"
            out.at[i, "Matchup"] = f"{away_key or away} @ {team}"
        elif team == away_key:
            out.at[i, "Opponent"] = home_key or home
            out.at[i, "HomeAway"] = "AWAY"
            out.at[i, "Matchup"] = f"{team} @ {home_key or home}"
    out["Matchup"] = out.apply(lambda r: r.get("Matchup") or (f"{r.get('Team','')} vs {r.get('Opponent','')}" if r.get('Team') and r.get('Opponent') else r.get('Team','')), axis=1)
    return out


def render_source_status_card(lines: pd.DataFrame, ud_debug: pd.DataFrame, sl_debug: pd.DataFrame, use_odds_api_flag: bool, odds_api_key: str):
    def count_source(src):
        try:
            return int((lines.get("Source", pd.Series(dtype=str)).astype(str) == src).sum()) if lines is not None and not lines.empty else 0
        except Exception:
            return 0
    odds_status = "✅ Connected" if use_odds_api_flag and odds_api_key else "⚪ Off / no key"
    if use_odds_api_flag and not odds_api_key:
        odds_status = "❌ Missing key"
    st.markdown(f"""
    <div class='owp-blue-note'>
      <b>Source Status</b> — Underdog: {count_source('Underdog')} lines | Sleeper: {count_source('Sleeper')} lines | Odds API: {count_source('Odds API')} lines ({odds_status}) | CSV/Manual: disabled for clean production mode
    </div>
    """, unsafe_allow_html=True)

def render_mlb_style_board(mode: str, use_ud_flag: bool, use_sleeper_flag: bool, logs_global: pd.DataFrame, master_global: pd.DataFrame, force_market: Optional[str] = None):
    market_label = f" — {force_market}" if force_market else ""
    st.markdown(f"<div class='section-title'>{mode}{market_label} Board</div>", unsafe_allow_html=True)
    market_key = force_market or "ALL"
    top_cols = st.columns([1.1, 1.1, 1.2, 1.2, 2.0])
    with top_cols[0]:
        if st.button(f"🔄 Refresh {mode} Lines", key=f"refresh_{mode}_{market_key}"):
            clear_line_pull_caches()
            pull_board_lines(use_ud_flag, use_sleeper_flag, use_odds_api, odds_api_key)
            st.rerun()
    with top_cols[1]:
        if st.button(f"🧱 Rebuild {mode} Board", key=f"rebuild_{mode}_{market_key}"):
            try:
                master, team_ranks = build_master_features()
                st.session_state["wnba_rebuilt_at"] = now_iso()
                st.success(f"Rebuilt master features: {len(master):,} rows. Team ranks: {len(team_ranks):,} rows.")
            except Exception as e:
                st.error(f"Rebuild failed: {e}")
    with top_cols[2]:
        st.metric("Last refresh", st.session_state.get("wnba_last_refresh", "not yet"))
    with top_cols[3]:
        st.metric("Database players", 0 if master_global is None or master_global.empty else len(master_global))
    with top_cols[4]:
        st.caption("Workflow: Refresh board lines → inspect cards → Save official before games → Grade after results post.")

    lines_all, ud_debug, sl_debug = get_lines_from_state_or_pull(use_ud_flag, use_sleeper_flag, use_odds_api, odds_api_key)
    lines, slate_note = filter_lines_for_slate(lines_all, mode)
    render_source_status_card(lines_all, ud_debug, sl_debug, use_odds_api, odds_api_key)
    st.caption(slate_note)

    sched = schedule_for_slate(mode)
    if not sched.empty:
        with st.expander(f"{mode} schedule context", expanded=False):
            st.dataframe(sched, use_container_width=True)
    elif mode in ["Today", "Tomorrow"]:
        st.info(f"No cached schedule rows found for {mode.lower()}. If WNBA is off that day, lines may correctly return 0.")

    if logs_global.empty or master_global.empty:
        st.warning("Import/build SportsDataverse player logs first in Data Manager. Lines can load, but projections need player baselines.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lines loaded", 0 if lines is None else len(lines))
    c2.metric("Underdog rows", 0 if lines_all is None or lines_all.empty else int((lines_all.get("Source", pd.Series(dtype=str)) == "Underdog").sum()))
    c3.metric("Sleeper rows", 0 if lines_all is None or lines_all.empty else int((lines_all.get("Source", pd.Series(dtype=str)) == "Sleeper").sum()))
    c4.metric("Odds API rows", 0 if lines_all is None or lines_all.empty else int((lines_all.get("Source", pd.Series(dtype=str)) == "Odds API").sum()))

    if lines is None or lines.empty:
        st.error("No real sportsbook lines loaded for this slate. Check Debug/Status. If there are no WNBA games, this is expected.")
        if not master_global.empty:
            st.markdown("<div class='hidden-baseline-note'>Baseline table is hidden. Showing player cards only so you can still review projections while waiting for lines.</div>", unsafe_allow_html=True)
            baseline_cards = make_baseline_player_cards(master_global, force_market or "PRA", limit=30)
            for _, rr in baseline_cards.iterrows():
                render_card(rr)
        return pd.DataFrame()

    if force_market:
        market_filter = [force_market]
        st.caption(f"Market locked to {force_market}; lines from Underdog/Sleeper/Manual route directly here.")
    else:
        market_filter = st.multiselect("Market", MARKETS, default=MARKETS, key=f"market_{mode}_{market_key}")
    search = st.text_input("Search player", key=f"search_{mode}_{market_key}")
    proj_df = make_projection_board(lines[lines["Market"].isin(market_filter)], logs_global, master_global)
    if search and not proj_df.empty:
        proj_df = proj_df[proj_df["Player"].str.contains(search, case=False, na=False)]

    if proj_df.empty:
        st.warning("Lines loaded, but projection board could not be built. Check player-name matching and Data Manager.")
        return pd.DataFrame()

    proj_df["Slate"] = mode
    proj_df["SlateDate"] = str(slate_target_date(mode) or "ALL")
    proj_df = enrich_board_with_matchups(proj_df, mode)
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
            n = grade_pending(logs_global)
            st.success(f"Graded {n} pending plays.")
    with action_cols[2]:
        st.download_button(f"Download {mode} Board CSV", proj_df.to_csv(index=False), f"wnba_{mode.lower().replace(' ', '_')}_projection_board.csv", "text/csv", key=f"dl_{mode}_{market_key}")
    with action_cols[3]:
        st.caption("Save before does not change projections. Grade after uses the latest imported stat logs and updates learning history.")

    display_mode = st.radio("View", ["Player cards", "Table"], horizontal=True, key=f"view_{mode}_{market_key}")
    if display_mode == "Player cards":
        for _, r in proj_df.head(100).iterrows():
            render_card(r)
    else:
        show_cols = ["Player", "Team", "Opponent", "Matchup", "HomeAway", "Market", "Line", "Source", "Projection", "Edge", "Lean", "Official", "Official Play Score", "PASS Reason", "Underdog Line", "Sleeper Line", "Best Over Line", "Best Under Line", "Over %", "Under %", "L5 Hit%", "L10 Hit%", "L20 Hit%", "MIN Proj", "Role Confidence", "Minutes Safety", "Data Score", "Bayesian Confidence", "Team Pace", "Team ORtg", "Team DRtg", "Team Net", "Team Matchup Strength", "Lineup Continuity", "Shot Profile", "Rim Rate", "3PA Rate", "Shot Make Rate", "Slate", "SlateDate"]
        st.dataframe(proj_df[[c for c in show_cols if c in proj_df.columns]], use_container_width=True)
    return proj_df


with st.sidebar:
    st.header("Setup")
    season_now = st.number_input("Current season", min_value=2020, max_value=2032, value=datetime.now().year, step=1)
    season_last = st.number_input("Last season baseline", min_value=2020, max_value=2032, value=datetime.now().year - 1, step=1)
    use_ud = st.toggle("Pull Underdog", value=True)
    use_sleeper = st.toggle("Pull Sleeper", value=True)
    use_odds_api = st.toggle("Pull Odds API fallback", value=False, help="Optional: requires ODDS_API_KEY. Useful when Underdog/Sleeper are empty.")
    odds_api_key = st.text_input("ODDS_API_KEY", value=get_streamlit_secret("ODDS_API_KEY", ""), type="password", help="Optional fallback for sportsbook WNBA player props.")
    use_remote = st.toggle("Allow SportsDataverse remote downloads", value=True)
    st.markdown("**Markets active:** PTS, REB, AST, PRA")
    st.markdown("**Model:** Monte Carlo + Bayesian confidence + XGBoost-style blend")
    st.divider()
    if st.button("🔄 Refresh board lines", use_container_width=True):
        clear_line_pull_caches()
        pull_board_lines(use_ud, use_sleeper, use_odds_api, odds_api_key)
        st.rerun()
    if st.button("🧱 Rebuild database", use_container_width=True):
        try:
            build_master_features()
            st.success("Database rebuilt.")
            st.rerun()
        except Exception as e:
            st.error(f"Rebuild failed: {e}")

logs_global = load_dataset("player_game_logs")
master_global = load_dataset("master_features")
if master_global.empty and not logs_global.empty:
    try:
        master_global, _ = build_master_features()
    except Exception:
        master_global = pd.DataFrame()

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

tabs = st.tabs(["PTS", "REB", "AST", "PRA", "Best Bets", "Official + Grade", "Data Manager", "Debug / Status"])

MARKET_TAB_META = {
    "PTS": ("POINTS", "Points board: scoring projection, shot profile, pace, usage, matchup, line edge."),
    "REB": ("REBOUNDS", "Rebounds board: minutes, role, team rebounding, opponent context, recent form."),
    "AST": ("ASSISTS", "Assists board: minutes, usage proxy, lineup continuity, team shot environment."),
    "PRA": ("PRA", "Combo board: points + rebounds + assists with full Monte Carlo distribution."),
}

for idx, market in enumerate(MARKETS):
    with tabs[idx]:
        title, caption = MARKET_TAB_META[market]
        st.markdown(f"<div class='section-title'>{title} / Player Prop Model</div>", unsafe_allow_html=True)
        st.caption(caption + " Main flow: Refresh → inspect cards → save before games → grade after results.")
        slate_tabs = st.tabs(["Today", "Tomorrow", "All Lines"])
        with slate_tabs[0]:
            render_mlb_style_board("Today", use_ud, use_sleeper, logs_global, master_global, force_market=market)
        with slate_tabs[1]:
            render_mlb_style_board("Tomorrow", use_ud, use_sleeper, logs_global, master_global, force_market=market)
        with slate_tabs[2]:
            render_mlb_style_board("All Lines", use_ud, use_sleeper, logs_global, master_global, force_market=market)

with tabs[4]:
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

with tabs[5]:
    st.subheader("Official + Grade")
    st.caption("Save official plays before games. Grade after results are imported to update the learning log. Manual line tools are hidden; real sportsbook lines drive this board.")
    board = load_dataset("projection_board")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("💾 Save official before games", use_container_width=True):
            if board.empty:
                st.warning("No projection board cached yet. Refresh a market board first.")
            else:
                n = save_officials(board)
                st.success(f"Saved {n} official plays.")
    with c2:
        if st.button("📊 Grade pending after results", use_container_width=True):
            n = grade_pending(logs_global)
            st.success(f"Graded {n} pending plays.")
    with c3:
        if board.empty:
            st.metric("Current board", 0)
        else:
            st.metric("Current board", len(board))
    official = pd.DataFrame(load_json(OFFICIAL_LOG, []))
    learning = pd.DataFrame(load_json(LEARNING_LOG, []))
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

with tabs[6]:
    st.subheader("SportsDataverse Data Manager")
    st.caption("Upload SportsDataverse CSV index files and/or actual CSV/Parquet files. Parquet is preferred. RDS is ignored. Team ranks, research, injury/referee, ML, and backtest tools are hidden from the main navigation but their backend logic remains available.")
    st.info("Importing files saves the stat database automatically under wnba_engine/data. Save Before is only for your betting slate; Grade After updates learning.")

    cA, cB, cC = st.columns(3)
    with cA:
        st.metric("Player logs", "✅" if CACHE_FILES["player_game_logs"].exists() else "Missing")
    with cB:
        st.metric("Master features", "✅" if CACHE_FILES["master_features"].exists() else "Missing")
    with cC:
        st.metric("Team ranks", "✅" if CACHE_FILES["team_ranks"].exists() else "Missing")

    uploaded_files = st.file_uploader(
        "Upload WNBA data files together",
        type=["csv", "xlsx", "parquet", "json", "rds"],
        accept_multiple_files=True,
        help="Use player_game_logs, player_season_stats, team_season_stats, schedules, rosters, game_rosters, lineups, shots. Parquet/CSV only."
    )

    if uploaded_files:
        st.info(f"Ready to import {len(uploaded_files)} uploaded file(s). The app will auto-classify by filename.")
        preview_rows = [{"File": f.name, "Detected dataset": classify_filename(f.name) or "unknown/manual select"} for f in uploaded_files]
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

    if st.button("Import uploaded files + build database", type="primary"):
        all_debug = []
        grouped: Dict[str, List[pd.DataFrame]] = {k: [] for k in DATASET_LABELS}
        for f in uploaded_files or []:
            key = classify_filename(f.name)
            if not key:
                all_debug.append({"file": f.name, "dataset": "unknown", "status": "skipped: could not classify by filename"})
                continue
            try:
                if f.name.lower().endswith(".rds"):
                    all_debug.append({"file": f.name, "dataset": key, "status": "skipped: RDS unsupported; use parquet", "rows": 0})
                    continue
                raw = f.read()
                df0 = read_any_file(raw, f.name)
                expanded, dbg = expand_sportsdataverse_index(df0, key, [int(season_last), int(season_now)])
                if not dbg.empty:
                    for _, rr in dbg.iterrows():
                        row = rr.to_dict(); row["file"] = f.name; all_debug.append(row)
                if not expanded.empty:
                    df_source = expanded
                elif is_manifest_only(df0):
                    all_debug.append({"file": f.name, "dataset": key, "status": "manifest/index only; actual data not inside this CSV", "rows": 0})
                    continue
                else:
                    df_source = df0
                std = standardize_dataset(key, df_source)
                if std.empty:
                    all_debug.append({"file": f.name, "dataset": key, "status": "standardized empty: missing required player/team/stat columns", "rows": 0})
                else:
                    grouped[key].append(std)
                    all_debug.append({"file": f.name, "dataset": key, "status": "ok", "rows": len(std)})
            except Exception as e:
                all_debug.append({"file": f.name, "dataset": key or "unknown", "status": f"error: {str(e)[:180]}", "rows": 0})

        for key, frames in grouped.items():
            if frames:
                combined = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates()
                save_dataset(key, combined)
        try:
            master, team_ranks = build_master_features()
            st.success(f"Import complete. Master rows: {len(master):,}. Team-rank rows: {len(team_ranks):,}.")
        except Exception as e:
            st.error(f"Import saved files, but master build failed: {e}")
        st.dataframe(pd.DataFrame(all_debug), use_container_width=True)

    with st.expander("Optional: one-click remote pull from SportsDataverse", expanded=False):
        st.caption("Use only if Streamlit has internet access. This pulls needed index files and expands referenced CSV/Parquet data.")
        dataset_choices = st.multiselect(
            "Datasets to pull",
            list(DATASET_LABELS.keys()),
            default=["player_game_logs", "player_season_stats", "team_season_stats", "schedules", "rosters", "game_rosters"]
        )
        include_heavy = st.checkbox("Include heavier add-ons: lineups + shots", value=False)
        if include_heavy:
            for k in ["lineups", "shots"]:
                if k not in dataset_choices:
                    dataset_choices.append(k)
        if st.button("Refresh SportsDataverse Database"):
            if not use_remote:
                st.error("Turn on 'Allow SportsDataverse remote downloads' in the sidebar first.")
            else:
                debug = []
                progress = st.progress(0)
                for i, key in enumerate(dataset_choices):
                    with st.spinner(f"Pulling {key}..."):
                        df, dbg = download_sportsdataverse_dataset(key, [int(season_last), int(season_now)])
                        if not df.empty:
                            std = standardize_dataset(key, df)
                            if not std.empty:
                                save_dataset(key, std)
                                debug.append({"dataset": key, "status": "saved", "rows": len(std)})
                            else:
                                debug.append({"dataset": key, "status": "downloaded but standardized empty", "rows": 0})
                        else:
                            debug.append({"dataset": key, "status": "empty/failed", "rows": 0})
                        if not dbg.empty:
                            debug.extend(dbg.to_dict("records"))
                    progress.progress((i+1)/max(1, len(dataset_choices)))
                try:
                    master, team_ranks = build_master_features()
                    st.success(f"Remote refresh complete. Master rows: {len(master):,}. Team-rank rows: {len(team_ranks):,}.")
                except Exception as e:
                    st.warning(f"Remote files pulled, but master build needs review: {e}")
                st.dataframe(pd.DataFrame(debug), use_container_width=True)

    st.divider()
    st.subheader("Data Status")
    status = dataset_status_table()
    st.dataframe(status, use_container_width=True)
    for k, path in CACHE_FILES.items():
        if path.exists():
            st.download_button(f"Download {k}.csv", path.read_bytes(), file_name=path.name, mime="text/csv")

with tabs[7]:
    st.subheader("Debug / Status")
    st.caption("Hidden tools are not removed from the backend. This page is only for diagnostics when a pull or projection looks wrong.")
    st.markdown("### Data status")
    st.dataframe(dataset_status_table(), use_container_width=True)
    st.markdown("### Aggregated real lines")
    lines, ud_debug, sl_debug = get_lines_from_state_or_pull(use_ud, use_sleeper, use_odds_api, odds_api_key)
    render_source_status_card(lines, ud_debug, sl_debug, use_odds_api, odds_api_key)
    st.dataframe(lines, use_container_width=True)
    st.markdown("### Underdog debug")
    st.dataframe(ud_debug, use_container_width=True)
    st.markdown("### Sleeper / Odds API debug")
    st.dataframe(sl_debug, use_container_width=True)
    st.markdown("### Cached master preview")
    st.dataframe(master_global.head(50), use_container_width=True)

