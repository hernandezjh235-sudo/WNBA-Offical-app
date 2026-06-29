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

APP_VERSION = "WNBA v1.3 — SportsDataverse Import Wizard + PTS/REB/AST/PRA Engine"

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
    lname = name.lower()
    if lname.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw), low_memory=False)
    if lname.endswith(".xlsx") or lname.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))
    if lname.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))
    if lname.endswith(".json"):
        return pd.read_json(io.BytesIO(raw))
    if lname.endswith(".rds"):
        raise ValueError("RDS files are R-only. Use the .parquet version for Streamlit/Python.")
    raise ValueError(f"Unsupported file type: {name}")


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
    """If an uploaded file is an index CSV, download the referenced files and combine them."""
    debug_rows = []
    if df.empty:
        return df, pd.DataFrame(debug_rows)
    url_cols = get_url_columns(df)
    if not url_cols:
        return df, pd.DataFrame(debug_rows)

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
        return df, pd.DataFrame([{"dataset": dataset_key, "status": "index detected but no usable CSV/parquet URLs", "rows": 0}])

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
        return pd.concat(frames, ignore_index=True), pd.DataFrame(debug_rows)
    return pd.DataFrame(), pd.DataFrame(debug_rows)


def download_sportsdataverse_dataset(dataset_key: str, seasons: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    index_url = SPORTSDATAVERSE_INDEXES.get(dataset_key)
    if not index_url:
        return pd.DataFrame(), pd.DataFrame([{"dataset": dataset_key, "status": "no index url"}])
    b = request_bytes(index_url, timeout=30)
    if not b:
        return pd.DataFrame(), pd.DataFrame([{"dataset": dataset_key, "status": "index download failed", "url": index_url}])
    idx = pd.read_csv(io.BytesIO(b))
    expanded, dbg = expand_sportsdataverse_index(idx, dataset_key, seasons=seasons)
    if expanded.empty and not idx.empty:
        # Some index files are already metadata only; keep metadata as debug but do not mark as model-ready.
        return pd.DataFrame(), pd.concat([pd.DataFrame([{"dataset": dataset_key, "status": "index read but no data files expanded", "rows": len(idx)}]), dbg], ignore_index=True)
    return expanded, dbg

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
    date_col = find_col(d, ["game_date", "date", "start_date"])
    game_id_col = find_col(d, ["game_id", "event_id", "competition_id"])
    team_col = find_col(d, ["team", "team_abbreviation", "team_name"])
    player_cols = [c for c in d.columns if re.search(r"player|athlete", col_norm(c))]
    out = pd.DataFrame()
    out["GameDate"] = parse_date_series(d[date_col]) if date_col else pd.NaT
    out["GameID"] = d[game_id_col] if game_id_col else ""
    out["Team"] = d[team_col] if team_col else ""
    out["LineupText"] = d[player_cols].astype(str).agg(" | ".join, axis=1) if player_cols else d.astype(str).agg(" | ".join, axis=1)
    out["PlayerCount"] = len(player_cols)
    return out


def standardize_shots(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    player_col = find_col(d, ["PLAYER_NAME", "player_name", "athlete_display_name", "shooter", "player", "athlete"])
    team_col = find_col(d, ["team", "team_abbreviation", "team_name"])
    date_col = find_col(d, ["game_date", "date", "start_date"])
    season_col = find_col(d, ["season", "year"])
    made_col = find_col(d, ["made", "shot_made", "shot_result", "result"])
    value_col = find_col(d, ["shot_value", "points", "point_value"])
    type_col = find_col(d, ["shot_type", "type", "action_type", "play_type"])
    dist_col = find_col(d, ["shot_distance", "distance"])
    out = pd.DataFrame()
    out["Player"] = d[player_col] if player_col else np.nan
    out["Team"] = d[team_col] if team_col else ""
    out["GameDate"] = parse_date_series(d[date_col]) if date_col else pd.NaT
    out["Season"] = d[season_col] if season_col else out["GameDate"].dt.year
    out["ShotValue"] = d[value_col] if value_col else np.nan
    out["ShotType"] = d[type_col] if type_col else ""
    out["ShotDistance"] = d[dist_col] if dist_col else np.nan
    if made_col:
        m = d[made_col]
        if m.dtype == object:
            out["Made"] = m.astype(str).str.lower().isin(["made", "true", "1", "yes", "make"])
        else:
            out["Made"] = pd.to_numeric(m, errors="coerce").fillna(0) > 0
    else:
        out["Made"] = np.nan
    out = coerce_numeric(out, ["Season", "ShotValue", "ShotDistance"])
    if out["ShotValue"].isna().all():
        out["ShotValue"] = np.where(out["ShotType"].astype(str).str.contains("3|three", case=False, na=False), 3, 2)
    out["NameKey"] = out["Player"].map(normalize_name)
    return out.dropna(subset=["Player"])


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

    # Shot profile for points.
    sh = standardize_shots(shots) if shots is not None and not shots.empty else pd.DataFrame()
    if not sh.empty:
        sh["Is3"] = sh["ShotValue"].fillna(2) >= 3
        sh_agg = sh.groupby("NameKey").agg(
            ShotAttempts=("ShotValue", "count"), ShotMakes=("Made", lambda x: pd.Series(x).fillna(False).astype(bool).sum()),
            ThreePA=("Is3", "sum"), AvgShotDistance=("ShotDistance", "mean")
        ).reset_index()
        sh_agg["ThreePARate"] = sh_agg["ThreePA"] / sh_agg["ShotAttempts"].replace(0, np.nan)
        sh_agg["ShotMakeRate"] = sh_agg["ShotMakes"] / sh_agg["ShotAttempts"].replace(0, np.nan)
        base = base.merge(sh_agg, on="NameKey", how="left")
    else:
        for c in ["ShotAttempts", "ThreePA", "ThreePARate", "ShotMakeRate", "AvgShotDistance"]:
            base[c] = np.nan

    for m in MARKETS:
        base[f"{m}_per_min"] = base[f"{m}_avg"] / base["MIN_avg"].replace(0, np.nan)
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

    # Lineups: count mentions for continuity signal.
    lns = standardize_lineups(lineups) if not lineups.empty else pd.DataFrame()
    if not lns.empty:
        mentions = []
        for _, b in base[["NameKey", "Player"]].drop_duplicates().iterrows():
            nk = b["NameKey"]
            # Cheap but workable for uploaded parquet: lineup text contains names.
            cnt = lns["LineupText"].astype(str).map(normalize_name).str.contains(nk, regex=False, na=False).sum()
            mentions.append({"NameKey": nk, "LineupMentions": cnt})
        base = base.merge(pd.DataFrame(mentions), on="NameKey", how="left")
    else:
        base["LineupMentions"] = np.nan

    base["RoleConfidence"] = np.clip(
        45 + base["Games"].fillna(0).clip(0, 20)*1.4 + base["MIN_l10"].fillna(base["MIN_avg"]).clip(0, 36)*0.7 + base["StarterRate"].fillna(0)*18,
        0, 100
    ).round(1)
    base["MinutesSafetyGrade"] = np.select(
        [base["MIN_l10"] >= 30, base["MIN_l10"] >= 24, base["MIN_l10"] >= 18],
        ["A", "B", "C"], default="D"
    )
    base["DataScore"] = np.clip(30 + base["Games"].fillna(0).clip(0, 25)*2 + base["MIN_avg"].fillna(0).clip(0, 36)*0.8 + base["RoleConfidence"].fillna(0)*0.25, 0, 100).round(1)
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


def load_manual_lines():
    data = load_json(MANUAL_LINES_FILE, [])
    return pd.DataFrame(data) if data else pd.DataFrame(columns=["Player", "Team", "Market", "Line", "Source"])


def save_manual_lines(df):
    save_json(MANUAL_LINES_FILE, df.to_dict("records"))


def aggregate_lines(use_ud=True, use_sleeper=True, manual_df=None):
    frames = []
    ud_debug = pd.DataFrame(); sl_debug = pd.DataFrame()
    if use_ud:
        ud, ud_debug = fetch_underdog_board(); frames.append(ud)
    if use_sleeper:
        sl, sl_debug = fetch_sleeper_board(); frames.append(sl)
    if manual_df is not None and len(manual_df):
        m = manual_df.copy(); m["Source"] = m.get("Source", "Manual").fillna("Manual"); frames.append(m)
    if not frames:
        return pd.DataFrame(), ud_debug, sl_debug
    board = pd.concat(frames, ignore_index=True)
    board["Market"] = board["Market"].astype(str).str.upper().map(lambda x: "PRA" if "PRA" in x else x)
    board = board[board["Market"].isin(MARKETS)].copy()
    board["Line"] = pd.to_numeric(board["Line"], errors="coerce")
    board = board.dropna(subset=["Player", "Market", "Line"])
    board["NameKey"] = board["Player"].map(normalize_name)
    priority = {"Underdog": 1, "Sleeper": 2, "Manual": 3}
    board["Priority"] = board["Source"].map(priority).fillna(9)
    return board.sort_values(["NameKey", "Market", "Priority"]), ud_debug, sl_debug

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


def project_row(row, base, logs):
    player = row["Player"]; market = row["Market"]; line = row["Line"]
    b, score = match_player_base(player, base)
    if b is None or score < 0.76:
        proj = np.nan
        info = {"Data Score": 20, "Projection Note": "No stat baseline match", "Matched Player": "", "Match Score": round(score, 3), "Role Confidence": 0, "Minutes Safety": "NA", "Bayesian Confidence": 50}
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
        # Works for either percentage-like or raw attempts proxy.
        target = 21 if usage_input > 1.5 and usage_input <= 100 else 11
        usage_factor = max(0.90, min(1.10, usage_input / target))

    pace = safe_float(b.get("Team_Pace"), np.nan)
    pace_factor = 1.0 if pd.isna(pace) else max(0.94, min(1.06, pace / np.nanmean([pace, 78])))
    team_net = safe_float(b.get("Team_NetRtg"), 0)
    matchup_factor = max(0.94, min(1.06, 1 + team_net/250)) if pd.notna(team_net) else 1.0

    role_conf = safe_float(b.get("RoleConfidence"), 50)
    minutes_grade = str(b.get("MinutesSafetyGrade", "NA"))
    data_score = safe_float(b.get("DataScore"), 50)

    learn_adj, learn_note, bayes = learning_adjustment(player, market, baseline - line)
    pre_ml = baseline * minutes_factor * usage_factor * pace_factor * matchup_factor + learn_adj
    ml_proj, ml_note = xgboost_blend_projection(b.to_dict(), pre_ml)
    proj = ml_proj

    hit = hit_rates_for_player(logs, normalize_name(b.get("Player")), market, line)
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
        "Position": b.get("Position", ""), "PositionGroup": b.get("PositionGroup", "Unknown"),
        "Projection Note": f"{learn_note}; {ml_note}",
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
        rows.append({**r.to_dict(), **info, **sim, "Projection": round(proj, 2) if pd.notna(proj) else np.nan, "Edge": round(edge, 2) if pd.notna(edge) else np.nan, "Lean": lean, "Official Play Score": official_score, "PASS Reason": reason, "Official": official})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Official", "Official Play Score", "Edge"], ascending=[True, False, False])
    return out

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
    .stApp { background: #090d12; color: #eef3f7; }
    section[data-testid="stSidebar"] { background:#0b1118; }
    div[data-testid="stMetric"] { background:#111822; border:1px solid #202c3b; border-radius:16px; padding:12px; }
    .card { background:linear-gradient(145deg,#101722,#111b29); border:1px solid #263649; border-radius:18px; padding:16px; margin:10px 0; box-shadow: 0 0 16px rgba(0,0,0,.25); }
    .badge { display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid #33465c; margin-right:6px; font-size:.82rem; }
    .hot { color:#70ffbd; font-weight:800; }
    .warn { color:#ffd166; font-weight:800; }
    .pass { color:#9aa7b2; font-weight:700; }
    .small-note { color:#9fb2c3; font-size:.86rem; }
    .owp-header { font-size:2.25rem; font-weight:900; letter-spacing:.2px; }
    </style>
    """, unsafe_allow_html=True)


def render_card(r):
    cls = "hot" if "OVER" in str(r.get("Official")) else "warn" if "UNDER" in str(r.get("Official")) else "pass"
    st.markdown(f"""
    <div class='card'>
      <h3>{r.get('Player','')} <span class='badge'>{r.get('Market','')}</span> <span class='badge'>{r.get('PositionGroup','')}</span></h3>
      <div class='{cls}'>{r.get('Official','PASS')} — {r.get('Lean','')} <span class='badge'>Official Score {r.get('Official Play Score','')}</span></div>
      <p><b>Line:</b> {r.get('Line','')} ({r.get('Source','')}) &nbsp; | &nbsp; <b>Projection:</b> {r.get('Projection','')} &nbsp; | &nbsp; <b>Edge:</b> {r.get('Edge','')}</p>
      <p><b>UD:</b> {r.get('Underdog Line','')} &nbsp; <b>Sleeper:</b> {r.get('Sleeper Line','')} &nbsp; <b>Best Over:</b> {r.get('Best Over Line','')} &nbsp; <b>Best Under:</b> {r.get('Best Under Line','')}</p>
      <p><b>MC:</b> Over {r.get('Over %','')}% / Under {r.get('Under %','')}% &nbsp; | &nbsp; <b>Floor/Median/Ceiling:</b> {r.get('Floor','')} / {r.get('Median','')} / {r.get('Ceiling','')} &nbsp; | &nbsp; <b>Vol:</b> {r.get('Volatility','')}</p>
      <p><b>L5/L10/L20 Hit:</b> {r.get('L5 Hit%','')}% / {r.get('L10 Hit%','')}% / {r.get('L20 Hit%','')}% &nbsp; | &nbsp; <b>MIN:</b> {r.get('MIN Proj','')} &nbsp; | &nbsp; <b>Role:</b> {r.get('Role Confidence','')}/100 &nbsp; | &nbsp; <b>Data:</b> {r.get('Data Score','')}/100</p>
      <p><b>PASS:</b> {r.get('PASS Reason','')}</p>
      <small>{r.get('Projection Note','')}</small>
    </div>
    """, unsafe_allow_html=True)


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

# ============================================================
# Streamlit app
# ============================================================
st.set_page_config(page_title="ONE WAY PICKZ WNBA", page_icon="🏀", layout="wide")
inject_css()
st.markdown("<div class='owp-header'>🏀 ONE WAY PICKZ — WNBA Prop Engine</div>", unsafe_allow_html=True)
st.caption(APP_VERSION)

with st.sidebar:
    st.header("Setup")
    season_now = st.number_input("Current season", min_value=2020, max_value=2032, value=datetime.now().year, step=1)
    season_last = st.number_input("Last season baseline", min_value=2020, max_value=2032, value=datetime.now().year - 1, step=1)
    use_ud = st.toggle("Pull Underdog", value=True)
    use_sleeper = st.toggle("Pull Sleeper", value=True)
    use_remote = st.toggle("Allow SportsDataverse remote downloads", value=True)
    st.markdown("**Markets active:** PTS, REB, AST, PRA")
    st.markdown("**Model:** Monte Carlo + Bayesian confidence + XGBoost-style blend")

logs_global = load_dataset("player_game_logs")
master_global = load_dataset("master_features")
if master_global.empty and not logs_global.empty:
    try:
        master_global, _ = build_master_features()
    except Exception:
        master_global = pd.DataFrame()

tabs = st.tabs(["Board", "Manual Lines", "Data Manager", "Research Hub", "Team Ranks", "Official + Grade", "Log Tools", "Debug"])

with tabs[2]:
    st.subheader("SportsDataverse Import Wizard")
    st.caption("Upload the SportsDataverse CSV index files and/or actual CSV/Parquet files. Parquet is preferred. RDS is not supported in Python.")

    cA, cB, cC = st.columns(3)
    with cA:
        st.metric("Player logs", "✅" if CACHE_FILES["player_game_logs"].exists() else "Missing")
    with cB:
        st.metric("Master features", "✅" if CACHE_FILES["master_features"].exists() else "Missing")
    with cC:
        st.metric("Team ranks", "✅" if CACHE_FILES["team_ranks"].exists() else "Missing")

    uploaded_files = st.file_uploader(
        "Upload WNBA files together",
        type=["csv", "xlsx", "parquet", "json", "rds"],
        accept_multiple_files=True,
        help="Use player_game_logs, player_season_stats, team_season_stats, schedules, rosters, game_rosters, lineups, shots. Parquet/CSV only."
    )

    if uploaded_files:
        st.info(f"Ready to import {len(uploaded_files)} uploaded file(s). The app will auto-classify by filename.")
        preview_rows = []
        for f in uploaded_files:
            preview_rows.append({"File": f.name, "Detected dataset": classify_filename(f.name) or "unknown/manual select"})
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
                raw = f.read()
                df0 = read_any_file(raw, f.name)
                expanded, dbg = expand_sportsdataverse_index(df0, key, [int(season_last), int(season_now)])
                df_source = expanded if not expanded.empty else df0
                std = standardize_dataset(key, df_source)
                if std.empty:
                    all_debug.append({"file": f.name, "dataset": key, "status": "standardized empty", "rows": 0})
                else:
                    grouped[key].append(std)
                    all_debug.append({"file": f.name, "dataset": key, "status": "ok", "rows": len(std)})
                if not dbg.empty:
                    for _, rr in dbg.iterrows():
                        row = rr.to_dict(); row["file"] = f.name; all_debug.append(row)
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

    st.divider()
    st.subheader("One-click remote pull from SportsDataverse")
    st.caption("Use this if Streamlit has internet access. It pulls only the needed index files and expands referenced CSV/Parquet data.")
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
            data = path.read_bytes()
            st.download_button(f"Download {k}.csv", data, file_name=path.name, mime="text/csv")

with tabs[1]:
    st.subheader("Manual fallback lines")
    st.caption("Use this when Underdog/Sleeper miss players. These lines are included with Source=Manual.")
    existing = load_manual_lines()
    edited = st.data_editor(existing, num_rows="dynamic", use_container_width=True, column_config={"Market": st.column_config.SelectboxColumn(options=MARKETS)})
    if st.button("Save manual lines"):
        save_manual_lines(edited)
        st.success("Manual lines saved.")

with tabs[0]:
    st.subheader("Projection Board")
    manual_df = load_manual_lines()
    lines, ud_debug, sl_debug = aggregate_lines(use_ud=use_ud, use_sleeper=use_sleeper, manual_df=manual_df)
    if not lines.empty:
        CACHE_FILES["projection_board"].parent.mkdir(exist_ok=True)
        lines.to_csv(CACHE_FILES["projection_board"], index=False)
    if logs_global.empty or master_global.empty:
        st.warning("Import SportsDataverse player logs first in Data Manager. Lines can load, but projections need player baselines.")
    st.metric("Lines loaded", len(lines))
    if lines.empty:
        st.error("No lines loaded. Try manual lines, or check Debug.")
        # Track no-line players from master.
        if not master_global.empty:
            nl = master_global[["Player", "Team", "PositionGroup", "MIN_l10", "PTS_l10", "REB_l10", "AST_l10", "PRA_l10"]].head(200)
            st.caption("Top baseline players available with no matched line yet:")
            st.dataframe(nl, use_container_width=True)
    else:
        market_filter = st.multiselect("Market", MARKETS, default=MARKETS)
        search = st.text_input("Search player")
        proj_df = make_projection_board(lines[lines["Market"].isin(market_filter)], logs_global, master_global)
        if search and not proj_df.empty:
            proj_df = proj_df[proj_df["Player"].str.contains(search, case=False, na=False)]
        if not proj_df.empty:
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Official plays", int(proj_df["Official"].astype(str).str.contains("OVER|UNDER", na=False).sum()))
            with c2: st.metric("Avg edge", round(float(proj_df["Edge"].abs().mean()), 2))
            with c3: st.metric("Avg data score", round(float(proj_df["Data Score"].mean()), 1))
            with c4: st.metric("Avg official score", round(float(proj_df["Official Play Score"].mean()), 1))
            for _, r in proj_df.head(80).iterrows():
                render_card(r)
            with st.expander("Table view"):
                show_cols = ["Player", "Market", "Line", "Source", "Projection", "Edge", "Lean", "Official", "Official Play Score", "PASS Reason", "Underdog Line", "Sleeper Line", "Best Over Line", "Best Under Line", "Over %", "Under %", "L5 Hit%", "L10 Hit%", "L20 Hit%", "MIN Proj", "Role Confidence", "Minutes Safety", "Data Score", "Bayesian Confidence", "Team Pace", "Team ORtg", "Team DRtg", "Team Net"]
                st.dataframe(proj_df[[c for c in show_cols if c in proj_df.columns]], use_container_width=True)
                st.download_button("Download projection board CSV", proj_df.to_csv(index=False), "wnba_projection_board.csv", "text/csv")
        else:
            st.warning("Lines loaded, but projection board could not be built. Check player-name matching and Data Manager.")

with tabs[3]:
    st.subheader("Research Hub")
    if master_global.empty:
        st.warning("Build master features in Data Manager first.")
    else:
        player = st.selectbox("Player", sorted(master_global["Player"].dropna().unique().tolist()))
        p = master_global[master_global["Player"] == player].tail(1)
        if not p.empty:
            p = p.iloc[0]
            cols = st.columns(4)
            cols[0].metric("MIN L10", round(safe_float(p.get("MIN_l10"), 0), 2))
            cols[1].metric("PTS L10", round(safe_float(p.get("PTS_l10"), 0), 2))
            cols[2].metric("REB L10", round(safe_float(p.get("REB_l10"), 0), 2))
            cols[3].metric("AST L10", round(safe_float(p.get("AST_l10"), 0), 2))
            cols2 = st.columns(4)
            cols2[0].metric("Role Conf", round(safe_float(p.get("RoleConfidence"), 0), 1))
            cols2[1].metric("Minutes Safety", str(p.get("MinutesSafetyGrade", "NA")))
            cols2[2].metric("Usage", round(safe_float(p.get("UsageProxy"), 0), 2))
            cols2[3].metric("Data Score", round(safe_float(p.get("DataScore"), 0), 1))
            st.markdown("### Recent game log")
            d = logs_global[logs_global["NameKey"] == normalize_name(player)].sort_values("GameDate", ascending=False).head(20)
            st.dataframe(d[[c for c in ["GameDate", "Team", "Opponent", "MIN", "PTS", "REB", "AST", "PRA", "FGA", "FG3A", "FTA", "PLUS_MINUS"] if c in d.columns]], use_container_width=True)
            st.markdown("### Feature row")
            st.dataframe(pd.DataFrame([p.to_dict()]), use_container_width=True)

with tabs[4]:
    st.subheader("Team Ranks")
    team_ranks = load_dataset("team_ranks")
    if team_ranks.empty:
        st.warning("No team ranks yet. Build in Data Manager.")
    else:
        st.dataframe(team_ranks, use_container_width=True)
        st.download_button("Download team ranks CSV", team_ranks.to_csv(index=False), "wnba_team_ranks.csv", "text/csv")

with tabs[5]:
    st.subheader("Official picks + grading")
    board_path = CACHE_FILES["projection_board"]
    if board_path.exists() and not logs_global.empty and not master_global.empty:
        board_cache = pd.read_csv(board_path)
        proj_df = make_projection_board(board_cache, logs_global, master_global)
        if st.button("Save official plays before games"):
            n = save_officials(proj_df)
            st.success(f"Saved {n} official plays.")
    else:
        st.warning("Need a board and stats loaded first.")
    if st.button("Grade pending with latest stat log"):
        n = grade_pending(logs_global)
        st.success(f"Graded {n} pending plays.")
    official = pd.DataFrame(load_json(OFFICIAL_LOG, []))
    if not official.empty:
        st.dataframe(official, use_container_width=True)
        st.download_button("Download official log", official.to_csv(index=False), "wnba_official_log.csv", "text/csv")
    learning = pd.DataFrame(load_json(LEARNING_LOG, []))
    if learning.empty:
        st.info("No graded learning data yet.")
    else:
        if "Result" not in learning.columns:
            learning["Result"] = "PENDING"
        graded_learning = learning[learning["Result"].isin(["WIN", "LOSS"])].copy()
        if graded_learning.empty:
            st.info("No graded learning data yet. Saved picks will show here after grading.")
        else:
            wins = int((graded_learning["Result"] == "WIN").sum())
            total = int(len(graded_learning))
            st.metric("Learning win rate", f"{wins}/{total} ({wins/total:.1%})")
        st.dataframe(learning, use_container_width=True)
        st.download_button("Download learning log", learning.to_csv(index=False), "wnba_learning_log.csv", "text/csv")

with tabs[6]:
    st.subheader("Log tools + injury bump table")
    st.caption("Import prior learning logs, backup all CSVs/logs, or reset logs. Injury bumps are manual for now and do not change code.")
    up = st.file_uploader("Import previous learning/offical log CSV", type=["csv"], key="learning_import")
    if up is not None and st.button("Import learning CSV"):
        df = pd.read_csv(up)
        existing = load_json(LEARNING_LOG, [])
        save_json(LEARNING_LOG, existing + df.to_dict("records"))
        st.success(f"Imported {len(df)} learning rows.")
    st.download_button("Backup all CSVs + logs now", make_backup_zip(), file_name=f"wnba_engine_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.zip", mime="application/zip")
    if st.checkbox("I understand reset will clear official/learning/line-history logs"):
        if st.button("Reset/backup logs button — clear logs"):
            reset_logs()
            st.success("Logs reset.")
    st.markdown("### Injury usage bump table")
    bumps = pd.DataFrame(load_json(INJURY_BUMPS_FILE, []))
    if bumps.empty:
        bumps = pd.DataFrame(columns=["Player", "Team", "Market", "Teammate Out", "Usage Bump %", "Minutes Bump", "Note"])
    edited_bumps = st.data_editor(bumps, num_rows="dynamic", use_container_width=True, column_config={"Market": st.column_config.SelectboxColumn(options=["ALL"] + MARKETS)})
    if st.button("Save injury bump table"):
        save_json(INJURY_BUMPS_FILE, edited_bumps.to_dict("records"))
        st.success("Injury bump table saved.")

with tabs[7]:
    st.subheader("Debug")
    st.markdown("### Data status")
    st.dataframe(dataset_status_table(), use_container_width=True)
    st.markdown("### Aggregated lines")
    lines, ud_debug, sl_debug = aggregate_lines(use_ud=use_ud, use_sleeper=use_sleeper, manual_df=load_manual_lines())
    st.dataframe(lines, use_container_width=True)
    st.markdown("### Underdog debug")
    st.dataframe(ud_debug, use_container_width=True)
    st.markdown("### Sleeper debug")
    st.dataframe(sl_debug, use_container_width=True)
    st.markdown("### Cached master preview")
    st.dataframe(master_global.head(50), use_container_width=True)
