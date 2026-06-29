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

APP_VERSION = "WNBA v1.6 — MLB Underdog Parser Port + Manual Line Fallback"

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

    # PRA can appear in several Underdog forms.
    if (
        ("points" in b and "rebounds" in b and "assists" in b)
        or ("pts" in b and ("reb" in b or "rebs" in b) and ("ast" in b or "asts" in b))
        or "pts rebs asts" in b
        or "pts reb ast" in b
        or "pra" in b
        or "points rebounds assists" in b
    ):
        return "PRA"
    if "point" in b or "points" in b or "pts" in b:
        return "PTS"
    if "rebound" in b or "rebounds" in b or "reb" in b or "rebs" in b:
        return "REB"
    if "assist" in b or "assists" in b or "ast" in b or "asts" in b:
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
    """Robust WNBA Underdog board parser ported from the working MLB parser style.

    The old WNBA parser depended on one specific nested path. This version:
      - uses the same light Underdog request headers as the MLB app,
      - tries every Underdog over_under_lines endpoint,
      - parses relationship objects first,
      - falls back to recursive object parsing,
      - maps Underdog full market labels into PTS/REB/AST/PRA,
      - accepts abbreviated display names like J. Young / C. Gray,
      - keeps manual line fallback untouched when Underdog returns nothing.
    """
    rows, debug = [], []

    def ud_get_json(url, timeout=18):
        try:
            h = {
                "User-Agent": "Mozilla/5.0 MLBKPropEngine/refresh-save-build",
                "Accept": "application/json,text/plain,*/*",
            }
            r = requests.get(url, headers=h, timeout=timeout)
            if r.status_code != 200:
                debug.append({"source": "Underdog", "url": url, "status": f"HTTP {r.status_code}", "rows": 0, "message": (r.text or "")[:180]})
                return None
            try:
                return r.json()
            except Exception as e:
                debug.append({"source": "Underdog", "url": url, "status": "bad json", "rows": 0, "message": str(e)[:180]})
                return None
        except Exception as e:
            debug.append({"source": "Underdog", "url": url, "status": "request error", "rows": 0, "message": str(e)[:180]})
            return None

    def obj_type(obj, fallback=""):
        return str(obj.get("type") or fallback or "").lower().replace("-", "_") if isinstance(obj, dict) else ""

    def obj_id(obj):
        if not isinstance(obj, dict):
            return None
        val = obj.get("id") or attrs(obj).get("id")
        return str(val) if val not in [None, ""] else None

    def collect_objects(data):
        objects = []
        def walk(x, parent_key=""):
            if isinstance(x, dict):
                y = dict(x)
                if parent_key and "_parent_key" not in y:
                    y["_parent_key"] = parent_key
                objects.append(y)
                for k, v in x.items():
                    walk(v, k)
            elif isinstance(x, list):
                for item in x:
                    walk(item, parent_key)
        walk(data)
        return objects

    def text_from(*objs):
        parts = []
        wanted = [
            "title", "display_title", "name", "player_name", "full_name", "first_name", "last_name",
            "display_name", "stat", "stat_type", "appearance_stat", "display_stat", "label", "market",
            "market_name", "sport", "league", "sport_name", "league_name", "position", "description",
            "over_under", "over_under_title", "scoring_type", "projection_type", "team", "abbr_name",
            "short_name", "event_title", "game_title", "home_team", "away_team", "scheduled_at", "start_time",
        ]
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            a = attrs(obj)
            for k in wanted:
                v = a.get(k)
                if isinstance(v, dict):
                    for kk in wanted:
                        if v.get(kk) not in [None, ""]:
                            parts.append(str(v.get(kk)))
                elif isinstance(v, list):
                    parts.extend([str(x) for x in v if x not in [None, ""]][:4])
                elif v not in [None, ""]:
                    parts.append(str(v))
        return " | ".join(parts)

    def player_name_from(player_obj=None, appearance_obj=None, line_obj=None, ou_obj=None):
        candidates = []
        for obj in [player_obj, appearance_obj, line_obj, ou_obj]:
            a = attrs(obj) if isinstance(obj, dict) else {}
            candidates.extend([
                a.get("display_name"), a.get("full_name"), a.get("name"), a.get("player_name"),
                a.get("short_name"), a.get("abbreviation"), a.get("abbr_name"),
                (str(a.get("first_name", "")).strip() + " " + str(a.get("last_name", "")).strip()).strip(),
            ])
        for c in candidates:
            if c and normalize_name(c):
                return str(c)
        return ""

    def line_from_underdog(*objs):
        # Do not use generic team/game totals. Only read true prop line fields.
        safe_keys = ["stat_value", "line_score", "over_under_line", "target_value"]
        for obj in objs:
            a = attrs(obj)
            for k in safe_keys:
                val = safe_float(a.get(k), np.nan)
                if pd.notna(val) and 0.5 <= val <= 80:
                    return float(val)
        blob = " | ".join(text_from(o) for o in objs if isinstance(o, dict))
        nums = re.findall(r"(?<!\d)(\d{1,2}(?:\.5)?)(?!\d)", blob)
        vals = [safe_float(n, np.nan) for n in nums]
        vals = [v for v in vals if pd.notna(v) and 0.5 <= v <= 80]
        return float(vals[0]) if vals else np.nan

    def status_ok(*objs):
        status_blob = " ".join(
            str(attrs(o).get(k, ""))
            for o in objs if isinstance(o, dict)
            for k in ["status", "state", "display_status", "over_status", "under_status", "hidden", "active"]
        ).lower()
        if any(x in status_blob for x in ["suspended", "removed", "hidden", "inactive", "closed", "disabled"]):
            return False
        return True

    def bad_sport(blob):
        low = str(blob or "").lower()
        return any(x in low for x in ["mlb", "baseball", "nfl", "football", "nhl", "hockey", "tennis", "golf", "mma", "soccer", "cs2", "lol", "dota"])

    def is_possible_wnba(blob):
        low = str(blob or "").lower()
        wnba_tokens = ["wnba", "women", "basketball"]
        team_tokens = ["lva", "nyl", "atl", "was", "dal", "sea", "min", "phx", "chi", "ind", "con", "las", "gs", "golden state", "valkyries", "aces", "liberty", "dream", "mystics", "wings", "storm", "lynx", "mercury", "sky", "fever", "sun", "sparks"]
        return any(t in low for t in wnba_tokens + team_tokens)

    def infer_team(*objs):
        for obj in objs:
            a = attrs(obj) if isinstance(obj, dict) else {}
            for k in ["team", "team_abbreviation", "abbr", "abbr_name", "short_name", "team_name"]:
                v = a.get(k)
                if isinstance(v, dict):
                    v = v.get("abbr") or v.get("abbr_name") or v.get("name") or v.get("display_name")
                if v not in [None, ""]:
                    s = str(v).upper().strip()
                    if 2 <= len(s) <= 4:
                        return s
        return ""

    def start_from(*objs):
        for obj in objs:
            a = attrs(obj) if isinstance(obj, dict) else {}
            for k in ["start_time", "scheduled_at", "start_date", "game_time", "match_time"]:
                if a.get(k) not in [None, ""]:
                    return str(a.get(k))
        return ""

    def known_player_records():
        """Known WNBA players from the local feature/stat caches.

        Underdog sometimes returns abbreviated display text or internal ids in the
        fields we first inspect. Matching the full raw prop JSON against our known
        player list prevents fake names like `ju4nn` or `none sh1n` from entering
        the projection board.
        """
        frames = []
        for key in ["master_features", "player_game_logs", "player_season_stats", "rosters"]:
            try:
                d = load_dataset(key)
                if d is not None and not d.empty and "Player" in d.columns:
                    cols = [c for c in ["Player", "Team", "Position", "PositionGroup"] if c in d.columns]
                    frames.append(d[cols].copy())
            except Exception:
                pass
        if not frames:
            return []
        kp = pd.concat(frames, ignore_index=True, sort=False)
        kp["Player"] = kp["Player"].fillna("").astype(str)
        kp["NameKey"] = kp["Player"].map(normalize_name)
        kp = kp[kp["NameKey"].str.len() > 0].copy()
        kp["Team"] = kp.get("Team", "").fillna("").astype(str).str.upper()
        kp = kp.drop_duplicates("NameKey", keep="last")
        last_counts = kp["NameKey"].map(lambda x: x.split()[-1] if x.split() else "").value_counts().to_dict()
        records = []
        for _, r in kp.iterrows():
            toks = str(r.get("NameKey", "")).split()
            if len(toks) < 2:
                continue
            records.append({
                "Player": str(r.get("Player", "")),
                "NameKey": str(r.get("NameKey", "")),
                "Team": str(r.get("Team", "")),
                "FirstInitial": toks[0][0] if toks[0] else "",
                "Last": toks[-1],
                "LastUnique": last_counts.get(toks[-1], 0) == 1,
            })
        return records

    KNOWN_PLAYERS = known_player_records()

    def bad_player_candidate(player):
        nk = normalize_name(player)
        if not nk or len(nk.split()) < 2:
            return True
        if re.search(r"\d", str(player or "")):
            return True
        bad_tokens = {"none", "null", "country", "player", "players", "over", "under", "higher", "lower", "wnba", "basketball"}
        toks = set(nk.split())
        if toks.intersection(bad_tokens):
            return True
        # Internal ids/slugs often have no normal full-name shape.
        if any(len(t) <= 1 for t in toks):
            return True
        return False

    def resolve_known_player(raw_text, candidate="", team_hint=""):
        raw_text = str(raw_text or "")
        raw_norm = normalize_name(raw_text)
        team_hint = str(team_hint or "").upper().strip()

        # If a candidate looks valid, map it back to the official cached name.
        if candidate and not bad_player_candidate(candidate):
            cand_key = normalize_name(candidate)
            best, best_score = None, 0.0
            for rec in KNOWN_PLAYERS:
                sc = name_score(cand_key, rec["NameKey"])
                if team_hint and rec.get("Team") == team_hint:
                    sc += 0.03
                if sc > best_score:
                    best, best_score = rec, sc
            if best and best_score >= 0.86:
                return best["Player"], best.get("Team", team_hint)
            return candidate, team_hint

        # Full-name containment.
        for rec in KNOWN_PLAYERS:
            nk = rec["NameKey"]
            if nk and nk in raw_norm:
                return rec["Player"], rec.get("Team", team_hint)

        # Underdog cards often show initials like J. Young -> normalized `j young`.
        best, best_score = None, 0.0
        for rec in KNOWN_PLAYERS:
            abbrev = f"{rec['FirstInitial']} {rec['Last']}".strip()
            score = 0.0
            if abbrev and abbrev in raw_norm:
                score = 0.94
            elif rec["LastUnique"] and re.search(rf"\b{re.escape(rec['Last'])}\b", raw_norm):
                score = 0.88
            if team_hint and rec.get("Team") == team_hint:
                score += 0.04
            if score > best_score:
                best, best_score = rec, score
        if best and best_score >= 0.88:
            return best["Player"], best.get("Team", team_hint)

        return "", team_hint

    def add_row(player, team, market, line, source_mode, raw, start=""):
        if market not in MARKETS or pd.isna(line):
            return
        resolved_player, resolved_team = resolve_known_player(raw, candidate=player, team_hint=team)
        if not resolved_player or bad_player_candidate(resolved_player):
            # Keep fake Underdog ids out of the board. They can still be inspected in Raw/debug.
            return
        rows.append({
            "Player": str(resolved_player),
            "Team": str(resolved_team or team or ""),
            "Opponent": "",
            "Market": market,
            "Line": float(line),
            "Source": "Underdog",
            "Start": start,
            "Raw": str(raw)[:280],
            "Parser Mode": source_mode,
        })

    LINE_TYPES = {"over_under_line", "over_under_lines"}
    OU_TYPES = {"over_under", "over_unders"}
    APP_TYPES = {"appearance", "appearances"}
    PLAYER_TYPES = {"player", "players"}

    for url in UNDERDOG_URLS:
        data = ud_get_json(url)
        if not data:
            continue

        objects = collect_objects(data)
        by_id_any = {}
        over_unders, appearances, players, line_candidates = {}, {}, {}, []
        for obj in objects:
            typ = obj_type(obj, obj.get("_parent_key", ""))
            oid = obj_id(obj)
            if oid:
                by_id_any[oid] = obj
            if typ in LINE_TYPES or "over_under_line" in typ:
                line_candidates.append(obj)
            elif typ in OU_TYPES or typ == "over_under":
                if oid:
                    over_unders[oid] = obj
            elif typ in APP_TYPES or "appearance" in typ:
                if oid:
                    appearances[oid] = obj
            elif typ in PLAYER_TYPES or typ == "player":
                if oid:
                    players[oid] = obj

        def get_by_id(oid):
            return by_id_any.get(str(oid)) if oid not in [None, ""] else None

        if not line_candidates:
            for obj in objects:
                a = attrs(obj)
                if any(a.get(k) not in [None, ""] for k in ["stat_value", "line_score", "over_under_line", "target_value"]):
                    line_candidates.append(obj)

        # Relationship parser first.
        for line_obj in line_candidates:
            ou_id = rel_id(line_obj, ["over_under", "over_unders", "overUnder", "over_under_id", "over"])
            ou_obj = over_unders.get(str(ou_id)) or get_by_id(ou_id)

            app_id = rel_id(line_obj, ["appearance", "appearances", "appearance_id"])
            if not app_id and isinstance(ou_obj, dict):
                app_id = rel_id(ou_obj, ["appearance", "appearances", "appearance_id"])
            app_obj = appearances.get(str(app_id)) or get_by_id(app_id)

            player_id = rel_id(line_obj, ["player", "players", "player_id"])
            if not player_id and isinstance(ou_obj, dict):
                player_id = rel_id(ou_obj, ["player", "players", "player_id"])
            if not player_id and isinstance(app_obj, dict):
                player_id = rel_id(app_obj, ["player", "players", "player_id"])
            if not player_id and isinstance(app_obj, dict):
                player_id = attrs(app_obj).get("player_id") or attrs(app_obj).get("playerId")
            player_obj = players.get(str(player_id)) or get_by_id(player_id)

            evidence = text_from(line_obj, ou_obj, app_obj, player_obj)
            blob = evidence + " | " + json.dumps(attrs(line_obj), default=str)
            if bad_sport(blob):
                continue
            market = infer_market(blob)
            if market not in MARKETS:
                continue
            if not is_possible_wnba(blob):
                # Do not require this too strictly if player/team relationships are present.
                pass
            line = line_from_underdog(line_obj, ou_obj)
            if pd.isna(line):
                continue
            player = player_name_from(player_obj, app_obj, line_obj, ou_obj)
            if not player:
                continue
            if not status_ok(line_obj, ou_obj, app_obj):
                continue
            add_row(player, infer_team(player_obj, app_obj, line_obj, ou_obj), market, line, "relationship", evidence, start_from(app_obj, line_obj, ou_obj))

        # Recursive fallback parser for changed Underdog JSON.
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            blob = json.dumps(obj, default=str)
            low = blob.lower()
            if bad_sport(low):
                continue
            market = infer_market(low)
            if market not in MARKETS:
                continue
            if not is_possible_wnba(low):
                continue
            line = line_from_underdog(obj)
            if pd.isna(line):
                continue
            player = player_name_from(obj, None, obj, None)
            if not player:
                continue
            if not status_ok(obj):
                continue
            add_row(player, infer_team(obj), market, line, "recursive fallback", blob, start_from(obj))

        debug.append({"source": "Underdog", "url": url, "status": "ok", "rows": len(rows), "message": f"parsed {len(rows)} WNBA rows"})
        if rows:
            break

    if rows:
        df = pd.DataFrame(rows)
        df["NameKey"] = df["Player"].map(normalize_name)
        # Final safety: remove any internal ids/slugs that slipped through.
        df = df[~df["Player"].map(bad_player_candidate)].copy()
        if df.empty:
            debug.append({"source": "Underdog", "url": "parser", "status": "name-match failed", "rows": 0, "message": "Lines parsed, but no Underdog player names matched cached WNBA players"})
            return pd.DataFrame(columns=["Player", "Team", "Opponent", "Market", "Line", "Source", "Start", "Raw", "Parser Mode", "NameKey"]), pd.DataFrame(debug)
        df = df.sort_values(["NameKey", "Market", "Line"]).drop_duplicates(subset=["NameKey", "Market", "Line", "Source"], keep="first")
        return df.reset_index(drop=True), pd.DataFrame(debug)

    if not debug:
        debug.append({"source": "Underdog", "url": "all", "status": "empty", "rows": 0, "message": "No endpoints checked"})
    return pd.DataFrame(columns=["Player", "Team", "Opponent", "Market", "Line", "Source", "Start", "Raw", "Parser Mode", "NameKey"]), pd.DataFrame(debug)

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


def grade_pending(logs):
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
    updated = 0
    learn = load_json(LEARNING_LOG, [])
    existing_ids = set()
    for r in learn:
        existing_ids.add(str(r.get("SavedAt", "")) + "|" + normalize_name(r.get("Player")) + "|" + str(r.get("Market")))
    for row in official:
        if row.get("Result") != "PENDING":
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
        if pd.notna(cutoff):
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
            pull_board_lines(use_ud, False, False, "")
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
            out.at[idx, "Opponent Context Note"] = "No opponent matched from Odds API event text or cached schedule; projection remains player/market based."
            out.at[idx, "Matchup Projection Factor"] = 1.0
            out.at[idx, "Opponent Context Applied"] = "NO"
            continue
        opp_ctx = _latest_team_context(opp, None)
        factor, note = _market_matchup_adjustment(r, opp_ctx)
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
        if st.button(f"🔄 Refresh {mode} Lines", key=f"refresh_{mode}_{market_key}"):
            clear_line_pull_caches()
            pull_board_lines(use_ud_flag, False, False, "")
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

    lines_all, ud_debug, sl_debug = get_lines_from_state_or_pull(use_ud_flag, False, False, "")
    lines, slate_note = filter_lines_for_slate(lines_all, mode)
    render_source_status_card(lines_all, ud_debug, sl_debug, False, "")
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
    proj_df = make_projection_board(lines[lines["Market"].isin(market_filter)], logs_global, master_global)
    if search and not proj_df.empty:
        proj_df = proj_df[proj_df["Player"].str.contains(search, case=False, na=False)]

    if proj_df.empty:
        st.warning("Lines loaded, but projection board could not be built. Check player-name matching and Data Manager.")
        return pd.DataFrame()

    proj_df["Slate"] = mode
    proj_df["SlateDate"] = str(slate_target_date(mode) or "ALL")
    proj_df = enrich_board_with_matchups(proj_df, mode)
    proj_df = apply_matchup_context_to_board(proj_df)
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
    st.markdown("**Markets active:** PTS, REB, AST, PRA")
    st.markdown("**Model:** Monte Carlo + Bayesian confidence + XGBoost-style blend")
    st.divider()
    if st.button("🔄 Refresh board lines", use_container_width=True):
        clear_line_pull_caches()
        pull_board_lines(use_ud, False, False, "")
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

tabs = st.tabs(["PTS", "REB", "AST", "PRA", "Best Bets", "Official + Grade", "Data Manager", "Debug / Status", "Model Reports"])

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
    st.caption("Save official plays before games. Grade after results are imported to update the learning log. Manual line tools are built into each market board. Underdog lines are used when available.")
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
            st.success(f"AutoGrader updated {n} pending plays from imported player logs.")
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

    st.markdown("### One-click pipeline")
    st.caption("This fixes missing Team_Pace/DRtg/NetRtg, shot profile, starter/roster, and usage/efficiency fields by pulling data, rebuilding features, and applying safe fallbacks.")
    pipe_cols = st.columns([1.15, 1.15, 1])
    with pipe_cols[0]:
        if st.button("🔄 Refresh WNBA Stats + ESPN Backup", use_container_width=True):
            if not use_remote:
                st.error("Turn on 'Allow SportsDataverse remote downloads' in the sidebar first.")
            else:
                with st.spinner("Refreshing SportsDataverse + backup feature pipeline..."):
                    master, team_ranks, dbg, audit = refresh_data_and_build_advanced_features(include_heavy=True)
                st.success(f"Refresh complete. Master rows: {len(master):,}. Team-rank rows: {len(team_ranks):,}.")
                st.markdown("#### Missing-field audit")
                st.dataframe(audit, use_container_width=True)
                with st.expander("Refresh debug", expanded=False):
                    st.dataframe(dbg, use_container_width=True)
    with pipe_cols[1]:
        if st.button("🧠 Build Advanced Features / Fix Missing Columns", use_container_width=True):
            with st.spinner("Rebuilding team ranks, shot profile, role, usage, and efficiency features..."):
                master, team_ranks = build_master_features()
                audit = feature_missing_report(master)
            st.success(f"Advanced features rebuilt. Master rows: {len(master):,}.")
            st.dataframe(audit, use_container_width=True)
    with pipe_cols[2]:
        report_path = DATA_DIR / "wnba_feature_missing_report.csv"
        if report_path.exists():
            st.download_button("Download missing-field report", report_path.read_bytes(), file_name="wnba_feature_missing_report.csv", mime="text/csv", use_container_width=True)
        else:
            st.info("No missing-field report yet.")

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
    lines, ud_debug, sl_debug = get_lines_from_state_or_pull(use_ud, False, False, "")
    render_source_status_card(lines, ud_debug, sl_debug, False, "")
    st.dataframe(lines, use_container_width=True)
    st.markdown("### Underdog debug")
    st.dataframe(ud_debug, use_container_width=True)
    st.markdown("### Manual line debug")
    st.dataframe(sl_debug, use_container_width=True)
    st.markdown("### Cached master preview")
    st.dataframe(master_global.head(50), use_container_width=True)

with tabs[8]:
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
