# -*- coding: utf-8 -*-
# ============================================================
# ONE WAY PICKZ — WNBA PROP ENGINE v1.0
# Markets: PTS / REB / AST / PRA only
# Built from MLB workflow concepts: line pull, player cards, official saves,
# grading, learning logs, Monte Carlo, line history, and Outlier-style hit rates.
# ============================================================

import os
import re
import io
import json
import math
import time
import hashlib
import difflib
import unicodedata
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None

APP_VERSION = "WNBA v1.2 — Role Safety + Bayesian Confidence + Log Tools"

# -----------------------------
# Storage
# -----------------------------
DRIVE_DIR = "/content/drive/MyDrive/wnba_engine"
LOCAL_DIR = "wnba_engine"
try:
    from google.colab import drive  # type: ignore
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive", force_remount=False)
    os.makedirs(DRIVE_DIR, exist_ok=True)
    STORAGE_DIR = DRIVE_DIR
except Exception:
    os.makedirs(LOCAL_DIR, exist_ok=True)
    STORAGE_DIR = LOCAL_DIR

OFFICIAL_LOG = os.path.join(STORAGE_DIR, "wnba_official_pick_log.json")
RESULT_LOG = os.path.join(STORAGE_DIR, "wnba_result_log.json")
LEARNING_LOG = os.path.join(STORAGE_DIR, "wnba_learning_log.json")
LINE_HISTORY_FILE = os.path.join(STORAGE_DIR, "wnba_line_history.json")
MANUAL_LINES_FILE = os.path.join(STORAGE_DIR, "wnba_manual_lines.json")
BASELINE_CACHE_FILE = os.path.join(STORAGE_DIR, "wnba_baseline_cache.csv")
BOARD_CACHE_FILE = os.path.join(STORAGE_DIR, "wnba_board_cache.csv")
LAST_YEAR_PLAYER_STATS_FILE = os.path.join(STORAGE_DIR, "wnba_last_year_player_stats.csv")
CURRENT_OPENING_STATS_FILE = os.path.join(STORAGE_DIR, "wnba_current_opening_stats.csv")
TEAM_RANKS_LAST_YEAR_FILE = os.path.join(STORAGE_DIR, "wnba_team_ranks_last_year.csv")
TEAM_RANKS_CURRENT_FILE = os.path.join(STORAGE_DIR, "wnba_team_ranks_current.csv")
RESULTS_LEARNING_CSV = os.path.join(STORAGE_DIR, "wnba_results_learning_log.csv")
OFFICIAL_PICKS_CSV = os.path.join(STORAGE_DIR, "wnba_official_picks_log.csv")
LINES_HISTORY_CSV = os.path.join(STORAGE_DIR, "wnba_lines_history.csv")
NO_LINE_TRACKING_FILE = os.path.join(STORAGE_DIR, "wnba_no_line_tracking.csv")
INJURY_USAGE_BUMP_FILE = os.path.join(STORAGE_DIR, "wnba_injury_usage_bump_table.csv")
LOG_BACKUP_DIR = os.path.join(STORAGE_DIR, "backups")

# -----------------------------
# Constants
# -----------------------------
MARKETS = ["PTS", "REB", "AST", "PRA"]
MARKET_ALIASES = {
    "points": "PTS", "point": "PTS", "pts": "PTS", "player points": "PTS",
    "rebounds": "REB", "rebound": "REB", "reb": "REB", "boards": "REB", "player rebounds": "REB",
    "assists": "AST", "assist": "AST", "ast": "AST", "player assists": "AST",
    "pts+reb+ast": "PRA", "pra": "PRA", "points rebounds assists": "PRA",
    "points + rebounds + assists": "PRA", "points+rebounds+assists": "PRA",
    "pts reb ast": "PRA", "fantasy score": "IGNORE", "steals": "IGNORE", "blocks": "IGNORE",
}
UNDERDOG_URLS = [
    "https://api.underdogfantasy.com/beta/v6/over_under_lines",
    "https://api.underdogfantasy.com/beta/v5/over_under_lines",
    "https://api.underdogfantasy.com/beta/v4/over_under_lines",
    "https://api.underdogfantasy.com/beta/v3/over_under_lines",
    "https://api.underdogfantasy.com/beta/v2/over_under_lines",
    "https://api.underdogfantasy.com/v1/over_under_lines",
]
# Sleeper Pick'em endpoints change often. We try multiple public-style routes and fail cleanly.
SLEEPER_URLS = [
    "https://api.sleeper.com/projections/wnba",
    "https://api.sleeper.app/projections/wnba",
    "https://api.sleeper.app/v1/projections/wnba",
    "https://api.sleeper.com/v1/projections/wnba",
]
WNBA_STATS_BASE = "https://stats.wnba.com/stats"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.wnba.com/",
    "Origin": "https://www.wnba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

# -----------------------------
# Utility
# -----------------------------
def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def safe_float(x, default=np.nan):
    try:
        if x is None or x == "":
            return default
        if isinstance(x, str):
            x = x.replace("−", "-").replace(",", "").strip()
        return float(x)
    except Exception:
        return default

def normalize_name(s):
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9 ]+", " ", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    suffixes = {"jr", "sr", "ii", "iii", "iv"}
    toks = [t for t in s.split() if t not in suffixes]
    return " ".join(toks)

def extract_opp_team(matchup, team=""):
    txt = str(matchup or "").upper().replace("@", " ").replace("VS.", "VS").replace("V.", "VS")
    toks = re.findall(r"[A-Z]{2,4}", txt)
    team = str(team or "").upper()
    for t in toks:
        if t not in {team, "WNBA", "USA"}:
            return t
    return ""

def bucket_position(pos):
    p = str(pos or "").upper()
    if any(x in p for x in ["PG", "SG", "G", "GUARD"]):
        return "GUARD"
    if any(x in p for x in ["SF", "W", "WING"]):
        return "WING"
    if any(x in p for x in ["PF", "C", "F-C", "CENTER", "FORWARD"]):
        return "BIG"
    return "UNKNOWN"

def name_score(a, b):
    a, b = normalize_name(a), normalize_name(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.94
    # initial + last name match: A Wilson vs A'ja Wilson
    at, bt = a.split(), b.split()
    if len(at) >= 2 and len(bt) >= 2 and at[-1] == bt[-1] and at[0][0] == bt[0][0]:
        return 0.90
    return difflib.SequenceMatcher(None, a, b).ratio()

def stable_seed(*parts):
    raw = "|".join(str(p) for p in parts)
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)

def request_json(url, params=None, timeout=18):
    try:
        r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
        if r.status_code >= 400:
            return None
        return r.json()
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

def text_blob(*objs):
    keys = ["name","title","display_title","display_name","full_name","first_name","last_name","player_name","stat","stat_type","appearance_stat","market","market_name","league","sport","description","label","team","abbr_name","short_name"]
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
    if "wnba" not in b and "women" not in b and "basketball" not in b:
        # keep open because some rows omit league text
        pass
    if any(x in b for x in ["fantasy", "steal", "block", "turnover", "three", "3 pointer", "free throw"]):
        return None
    # PRA first so individual words do not steal it
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

# -----------------------------
# Line pulls
# -----------------------------
@st.cache_data(ttl=240, show_spinner=False)
def fetch_underdog_board():
    rows, debug = [], []
    for url in UNDERDOG_URLS:
        data = request_json(url, timeout=20)
        if not data:
            debug.append({"source":"Underdog", "url":url, "status":"no json"})
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
            if not any(x in low for x in ["wnba", "women", "basketball"]):
                # Underdog sometimes omits league in child object; do not hard reject yet.
                pass
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
    df = pd.DataFrame(rows).drop_duplicates(subset=["Player","Market","Line","Source"]) if rows else pd.DataFrame(columns=["Player","Team","Market","Line","Source","Start","Raw"])
    return df, pd.DataFrame(debug)

@st.cache_data(ttl=240, show_spinner=False)
def fetch_sleeper_board():
    rows, debug = [], []
    for url in SLEEPER_URLS:
        data = request_json(url, timeout=15)
        if not data:
            debug.append({"source":"Sleeper", "url":url, "status":"no json/blocked"})
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
                # Try common nested keys
                a = attrs(o)
                for k in ["player", "athlete", "participant"]:
                    if isinstance(a.get(k), dict):
                        player = player_from_obj(a.get(k))
                if not player:
                    continue
            rows.append({"Player": player, "Team": attrs(o).get("team") or "", "Market": market, "Line": float(line), "Source": "Sleeper", "Start": attrs(o).get("start_time") or "", "Raw": blob[:180]})
        if rows:
            break
    df = pd.DataFrame(rows).drop_duplicates(subset=["Player","Market","Line","Source"]) if rows else pd.DataFrame(columns=["Player","Team","Market","Line","Source","Start","Raw"])
    return df, pd.DataFrame(debug)

def load_manual_lines():
    data = load_json(MANUAL_LINES_FILE, [])
    return pd.DataFrame(data) if data else pd.DataFrame(columns=["Player","Team","Market","Line","Source"])

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
    board = board.dropna(subset=["Player","Market","Line"])
    board["NameKey"] = board["Player"].map(normalize_name)
    # active line priority: Underdog, then Sleeper, then Manual
    priority = {"Underdog": 1, "Sleeper": 2, "Manual": 3}
    board["Priority"] = board["Source"].map(priority).fillna(9)
    return board.sort_values(["NameKey","Market","Priority"]), ud_debug, sl_debug

# -----------------------------
# WNBA stats pulls
# -----------------------------
def stats_table(endpoint, params):
    data = request_json(f"{WNBA_STATS_BASE}/{endpoint}", params=params, timeout=25)
    if not data:
        return pd.DataFrame()
    rs = data.get("resultSets") or data.get("resultSet") or []
    if isinstance(rs, dict):
        rs = [rs]
    for s in rs:
        headers = s.get("headers") or s.get("Headers")
        rows = s.get("rowSet") or s.get("RowSet")
        if headers and rows:
            return pd.DataFrame(rows, columns=headers)
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_player_game_logs(seasons):
    all_frames = []
    for season in seasons:
        # LeagueDashPlayerStats gives season rates; playergamelogs gives game-level for hit rates if available.
        params = {
            "LeagueID": "10", "Season": str(season), "SeasonType": "Regular Season", "PlayerOrTeam": "P",
            "PerMode": "PerGame", "MeasureType": "Base", "DateFrom": "", "DateTo": ""
        }
        df = stats_table("leaguegamelog", {"LeagueID":"10", "Season":str(season), "SeasonType":"Regular Season", "PlayerOrTeam":"P", "Sorter":"DATE", "Direction":"DESC"})
        if df.empty:
            df = stats_table("playergamelogs", {"LeagueID":"10", "Season":str(season), "SeasonType":"Regular Season"})
        if not df.empty:
            df["SEASON"] = int(season)
            all_frames.append(df)
    if all_frames:
        out = pd.concat(all_frames, ignore_index=True)
        return standardize_logs(out)
    return pd.DataFrame()

def standardize_logs(df):
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    colmap = {c.upper(): c for c in d.columns}
    def find(*names):
        for n in names:
            if n.upper() in colmap:
                return colmap[n.upper()]
        return None
    mapping = {
        "Player": find("PLAYER_NAME", "PLAYER", "ATHLETE_NAME", "NAME"),
        "Team": find("TEAM_ABBREVIATION", "TEAM", "TEAM_NAME"),
        "OpponentRaw": find("MATCHUP", "OPPONENT", "OPPONENT_TEAM_ABBREVIATION"),
        "GameDate": find("GAME_DATE", "DATE"),
        "Position": find("POSITION", "POS"),
        "Starter": find("STARTER", "START_POSITION", "GS"),
        "HomeAway": find("HOME_AWAY", "HOMEAWAY", "LOCATION"),
        "MIN": find("MIN", "MINUTES"),
        "PTS": find("PTS", "POINTS"),
        "REB": find("REB", "REBOUNDS"),
        "AST": find("AST", "ASSISTS"),
        "STL": find("STL", "STEALS"), "BLK": find("BLK", "BLOCKS"),
        "FGA": find("FGA"), "FGM": find("FGM"), "FG3M": find("FG3M", "3PM"), "FTA": find("FTA"), "FTM": find("FTM"), "TOV": find("TOV", "TO"),
        "OREB": find("OREB"), "DREB": find("DREB"), "PLUS_MINUS": find("PLUS_MINUS", "PM", "+/-"),
        "SEASON": find("SEASON", "YEAR"),
    }
    out = pd.DataFrame()
    for k, c in mapping.items():
        out[k] = d[c] if c else np.nan
    for c in ["MIN","PTS","REB","AST","STL","BLK","FGA","FGM","FG3M","FTA","FTM","TOV","OREB","DREB","PLUS_MINUS","SEASON"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["PRA"] = out["PTS"].fillna(0) + out["REB"].fillna(0) + out["AST"].fillna(0)
    out["EfficiencyRating"] = (out["PTS"].fillna(0) + out["REB"].fillna(0) + out["AST"].fillna(0) + out["STL"].fillna(0) + out["BLK"].fillna(0)) - ((out["FGA"].fillna(0)-out["FGM"].fillna(0)) + (out["FTA"].fillna(0)-out["FTM"].fillna(0)) + out["TOV"].fillna(0))
    out["NameKey"] = out["Player"].map(normalize_name)
    out["Team"] = out["Team"].astype(str).str.upper().replace("NAN", "")
    out["OpponentTeam"] = [extract_opp_team(o, t) for o, t in zip(out["OpponentRaw"], out["Team"])]
    out["PositionBucket"] = out["Position"].map(bucket_position)
    # Starter role accepts STARTER=true/1, start position, or games started columns.
    out["StarterFlag"] = out["Starter"].astype(str).str.upper().isin(["1", "TRUE", "Y", "YES", "G", "F", "C", "PG", "SG", "SF", "PF"]).astype(int)
    out = out.dropna(subset=["Player"])
    return out

def uploaded_stats_to_logs(file):
    if file is None:
        return pd.DataFrame()
    raw = file.read()
    if file.name.lower().endswith(".csv"):
        return standardize_logs(pd.read_csv(io.BytesIO(raw)))
    return standardize_logs(pd.read_excel(io.BytesIO(raw)))

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_espn_schedule(days_forward=2):
    rows = []
    for offset in range(days_forward + 1):
        ds = (datetime.now() + timedelta(days=offset)).strftime("%Y%m%d")
        data = request_json(ESPN_SCOREBOARD, params={"dates": ds, "limit": 100}, timeout=15)
        if not data:
            continue
        for e in data.get("events", []):
            comp = (e.get("competitions") or [{}])[0]
            teams = comp.get("competitors") or []
            if len(teams) >= 2:
                rows.append({
                    "GameID": e.get("id"), "Date": e.get("date"), "Status": (comp.get("status") or {}).get("type", {}).get("description", ""),
                    "Away": teams[1].get("team", {}).get("abbreviation", "") if teams[1].get("homeAway") == "away" else teams[0].get("team", {}).get("abbreviation", ""),
                    "Home": teams[0].get("team", {}).get("abbreviation", "") if teams[0].get("homeAway") == "home" else teams[1].get("team", {}).get("abbreviation", ""),
                })
    return pd.DataFrame(rows)

# -----------------------------
# Projection engine layers
# -----------------------------
def compute_team_ranks(logs, season=None):
    cols = ["Team","Games","Pace","ORtg","DRtg","NetRtg","PythagWin%","PowerRating","OffRank","DefRank","PaceRank","NetRank","PtsAllowedRank","RebAllowedRank","AstAllowedRank"]
    if logs.empty:
        return pd.DataFrame(columns=cols)
    d = logs.copy()
    if season is not None and "SEASON" in d.columns:
        d = d[pd.to_numeric(d["SEASON"], errors="coerce") == int(season)]
    if d.empty:
        return pd.DataFrame(columns=cols)
    for c in ["PTS","REB","AST","FGA","FTA","TOV","OREB"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    scored = d.groupby("Team", dropna=False).agg(
        PTS=("PTS","sum"), REB=("REB","sum"), AST=("AST","sum"), FGA=("FGA","sum"), FTA=("FTA","sum"), TOV=("TOV","sum"), OREB=("OREB","sum"),
        Games=("GameDate","nunique")
    ).reset_index()
    allowed = d.groupby("OpponentTeam", dropna=False).agg(
        PtsAllowed=("PTS","sum"), RebAllowed=("REB","sum"), AstAllowed=("AST","sum"),
        OppFGA=("FGA","sum"), OppFTA=("FTA","sum"), OppTOV=("TOV","sum"), OppOREB=("OREB","sum")
    ).reset_index().rename(columns={"OpponentTeam":"Team"})
    team = scored.merge(allowed, on="Team", how="left")
    team = team[(team["Team"].astype(str) != "") & (team["Team"].astype(str) != "NAN")].copy()
    team["Poss"] = team["FGA"] + 0.44*team["FTA"] + team["TOV"] - team["OREB"]
    team["OppPoss"] = team["OppFGA"].fillna(0) + 0.44*team["OppFTA"].fillna(0) + team["OppTOV"].fillna(0) - team["OppOREB"].fillna(0)
    team["Pace"] = np.where(team["Games"]>0, team["Poss"] / team["Games"], np.nan)
    team["ORtg"] = np.where(team["Poss"]>0, 100*team["PTS"] / team["Poss"], np.nan)
    team["DRtg"] = np.where(team["OppPoss"]>0, 100*team["PtsAllowed"] / team["OppPoss"], np.nan)
    # If opponent possession data is missing from API logs, fallback to league average.
    if team["DRtg"].isna().all():
        team["DRtg"] = np.nanmean(team["ORtg"])
    team["NetRtg"] = team["ORtg"] - team["DRtg"]
    exp = 11
    team["PythagWin%"] = np.where((team["PTS"]**exp + team["PtsAllowed"].fillna(team["PTS"])**exp)>0, (team["PTS"]**exp)/(team["PTS"]**exp + team["PtsAllowed"].fillna(team["PTS"])**exp), np.nan)
    team["PowerRating"] = team["NetRtg"].fillna(0) + (team["PythagWin%"].fillna(.5)-.5)*10
    team["OffRank"] = team["ORtg"].rank(ascending=False, method="min").astype("Int64")
    team["DefRank"] = team["DRtg"].rank(ascending=True, method="min").astype("Int64")
    team["PaceRank"] = team["Pace"].rank(ascending=False, method="min").astype("Int64")
    team["NetRank"] = team["NetRtg"].rank(ascending=False, method="min").astype("Int64")
    team["PtsAllowedRank"] = (team["PtsAllowed"] / team["Games"].replace(0, np.nan)).rank(ascending=True, method="min").astype("Int64")
    team["RebAllowedRank"] = (team["RebAllowed"] / team["Games"].replace(0, np.nan)).rank(ascending=True, method="min").astype("Int64")
    team["AstAllowedRank"] = (team["AstAllowed"] / team["Games"].replace(0, np.nan)).rank(ascending=True, method="min").astype("Int64")
    return team[[c for c in cols if c in team.columns] + ["PtsAllowed","RebAllowed","AstAllowed"]]

def compute_team_context(logs):
    return compute_team_ranks(logs)

def compute_opponent_allowed(logs):
    if logs.empty:
        return pd.DataFrame(columns=["OpponentTeam","PTS_allowed_pg","REB_allowed_pg","AST_allowed_pg","PRA_allowed_pg"])
    d = logs.copy()
    allowed = d.groupby("OpponentTeam", dropna=False).agg(
        PTS_allowed_pg=("PTS","mean"), REB_allowed_pg=("REB","mean"), AST_allowed_pg=("AST","mean"), PRA_allowed_pg=("PRA","mean"),
        GamesAllowed=("GameDate","count")
    ).reset_index()
    return allowed

def compute_position_allowed(logs):
    if logs.empty or "PositionBucket" not in logs.columns:
        return pd.DataFrame(columns=["OpponentTeam","PositionBucket","PTS_pos_allowed","REB_pos_allowed","AST_pos_allowed","PRA_pos_allowed"])
    d = logs.copy()
    d = d[d["PositionBucket"].isin(["GUARD","WING","BIG"])]
    if d.empty:
        return pd.DataFrame(columns=["OpponentTeam","PositionBucket","PTS_pos_allowed","REB_pos_allowed","AST_pos_allowed","PRA_pos_allowed"])
    return d.groupby(["OpponentTeam","PositionBucket"], dropna=False).agg(
        PTS_pos_allowed=("PTS","mean"), REB_pos_allowed=("REB","mean"), AST_pos_allowed=("AST","mean"), PRA_pos_allowed=("PRA","mean"),
        PosSamples=("GameDate","count")
    ).reset_index()

def compute_baselines(logs):
    if logs.empty:
        return pd.DataFrame()
    d = logs.copy().sort_values(["NameKey","GameDate"])
    for c in ["MIN","PTS","REB","AST","PRA","FGA","FGM","FG3M","FTA","TOV","PLUS_MINUS","EfficiencyRating","StarterFlag"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    g = d.groupby(["NameKey","Player"], dropna=False)
    base = g.agg(
        Team=("Team", lambda x: x.dropna().iloc[-1] if len(x.dropna()) else ""),
        PositionBucket=("PositionBucket", lambda x: x.dropna().iloc[-1] if len(x.dropna()) else "UNKNOWN"),
        Games=("PTS","count"),
        StarterRate=("StarterFlag","mean"),
        MIN_avg=("MIN","mean"), MIN_l3=("MIN", lambda x: x.tail(3).mean()), MIN_l5=("MIN", lambda x: x.tail(5).mean()), MIN_l10=("MIN", lambda x: x.tail(10).mean()),
        PTS_avg=("PTS","mean"), REB_avg=("REB","mean"), AST_avg=("AST","mean"), PRA_avg=("PRA","mean"),
        PTS_l3=("PTS", lambda x: x.tail(3).mean()), REB_l3=("REB", lambda x: x.tail(3).mean()), AST_l3=("AST", lambda x: x.tail(3).mean()), PRA_l3=("PRA", lambda x: x.tail(3).mean()),
        PTS_l5=("PTS", lambda x: x.tail(5).mean()), REB_l5=("REB", lambda x: x.tail(5).mean()), AST_l5=("AST", lambda x: x.tail(5).mean()), PRA_l5=("PRA", lambda x: x.tail(5).mean()),
        PTS_l10=("PTS", lambda x: x.tail(10).mean()), REB_l10=("REB", lambda x: x.tail(10).mean()), AST_l10=("AST", lambda x: x.tail(10).mean()), PRA_l10=("PRA", lambda x: x.tail(10).mean()),
        FGA=("FGA","sum"), FGM=("FGM","sum"), FG3M=("FG3M","sum"), FTA=("FTA","sum"), TOV=("TOV","sum"),
        PlusMinusAvg=("PLUS_MINUS","mean"), EfficiencyAvg=("EfficiencyRating","mean"),
    ).reset_index()
    base["eFG%"] = np.where(base["FGA"]>0, (base["FGM"] + 0.5*base["FG3M"]) / base["FGA"], np.nan)
    base["TS%"] = np.where((2*(base["FGA"] + 0.44*base["FTA"]))>0, (base["PTS_avg"]*base["Games"]) / (2*(base["FGA"] + 0.44*base["FTA"])), np.nan)
    base["UsageProxy"] = (base["FGA"] + 0.44*base["FTA"] + base["TOV"]) / base["Games"].clip(lower=1)
    base["UsageRecent"] = base[["PTS_l3","PTS_l5","PTS_l10"]].mean(axis=1) / base["MIN_l5"].replace(0, np.nan)
    for m in MARKETS:
        base[f"{m}_per_min"] = base[f"{m}_avg"] / base["MIN_avg"].replace(0, np.nan)
    return base

def build_xgb_training_frame(logs):
    if logs.empty:
        return pd.DataFrame()
    d = logs.copy().sort_values(["NameKey","GameDate"])
    frames = []
    for market in MARKETS:
        tmp = d[["NameKey","GameDate","MIN","PTS","REB","AST","PRA","FGA","FTA","TOV","PLUS_MINUS","EfficiencyRating", market]].copy()
        tmp["Market"] = market
        g = tmp.groupby("NameKey", group_keys=False)
        tmp["Target"] = g[market].shift(-1)
        tmp["Prev_MIN_l3"] = g["MIN"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        tmp["Prev_MIN_l5"] = g["MIN"].transform(lambda x: x.rolling(5, min_periods=1).mean())
        tmp["Prev_MKT_l3"] = g[market].transform(lambda x: x.rolling(3, min_periods=1).mean())
        tmp["Prev_MKT_l5"] = g[market].transform(lambda x: x.rolling(5, min_periods=1).mean())
        tmp["Prev_MKT_l10"] = g[market].transform(lambda x: x.rolling(10, min_periods=1).mean())
        tmp["UsageBox"] = tmp["FGA"].fillna(0) + .44*tmp["FTA"].fillna(0) + tmp["TOV"].fillna(0)
        tmp["MarketCode"] = MARKETS.index(market)
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True).dropna(subset=["Target"])
    return out

@st.cache_resource(show_spinner=False)
def train_xgb_model_from_csv(cache_path, mtime):
    if not XGBOOST_AVAILABLE or not os.path.exists(cache_path):
        return None, []
    logs = standardize_logs(pd.read_csv(cache_path))
    train = build_xgb_training_frame(logs)
    features = ["Prev_MIN_l3","Prev_MIN_l5","Prev_MKT_l3","Prev_MKT_l5","Prev_MKT_l10","UsageBox","PLUS_MINUS","EfficiencyRating","MarketCode"]
    if len(train) < 80:
        return None, features
    X = train[features].fillna(0)
    y = train["Target"].astype(float)
    model = XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=42)
    model.fit(X, y)
    return model, features

def xgb_projection(model, features, market, b):
    if model is None:
        return np.nan
    row = {
        "Prev_MIN_l3": safe_float(b.get("MIN_l3"), 0),
        "Prev_MIN_l5": safe_float(b.get("MIN_l5"), 0),
        "Prev_MKT_l3": safe_float(b.get(f"{market}_l3"), 0),
        "Prev_MKT_l5": safe_float(b.get(f"{market}_l5"), 0),
        "Prev_MKT_l10": safe_float(b.get(f"{market}_l10"), 0),
        "UsageBox": safe_float(b.get("UsageProxy"), 0),
        "PLUS_MINUS": safe_float(b.get("PlusMinusAvg"), 0),
        "EfficiencyRating": safe_float(b.get("EfficiencyAvg"), 0),
        "MarketCode": MARKETS.index(market) if market in MARKETS else 0,
    }
    X = pd.DataFrame([[row.get(f, 0) for f in features]], columns=features).fillna(0)
    try:
        return float(model.predict(X)[0])
    except Exception:
        return np.nan

def hit_rates_for_player(logs, name_key, market, line):
    d = logs[logs["NameKey"] == name_key].copy().sort_values("GameDate")
    if d.empty or market not in d.columns or pd.isna(line):
        return {"L5 Hit%": np.nan, "L10 Hit%": np.nan, "L20 Hit%": np.nan, "Season Hit%": np.nan, "Last Values": ""}
    vals = pd.to_numeric(d[market], errors="coerce").dropna()
    def hr(n):
        x = vals.tail(n)
        return round(100 * (x > line).mean(), 1) if len(x) else np.nan
    return {
        "L5 Hit%": hr(5), "L10 Hit%": hr(10), "L20 Hit%": hr(20), "Season Hit%": round(100 * (vals > line).mean(), 1) if len(vals) else np.nan,
        "Last Values": ", ".join([str(round(v,1)) for v in vals.tail(10).tolist()])
    }

def learning_adjustment(player, market, base_edge):
    logs = load_json(LEARNING_LOG, [])
    if not logs:
        return 0.0, "No learning yet"
    key = normalize_name(player)
    rows = [r for r in logs if normalize_name(r.get("Player")) == key and str(r.get("Market")) == market and r.get("Result") in ["WIN","LOSS"]]
    if len(rows) < 3:
        return 0.0, f"Learning sample {len(rows)}"
    win_rate = sum(1 for r in rows if r.get("Result") == "WIN") / len(rows)
    # Small nudge only.
    adj = max(-0.35, min(0.35, (win_rate - 0.52) * 0.9))
    return adj, f"Learning {len(rows)} plays / {win_rate:.0%} WR"


def bayesian_confidence(player, market, line, lean, hit_info, data_score, sim):
    """Lightweight Bayesian prior layer: combines season/recent hit rates, model sim, and learned win rate.
    This is intentionally not heavy MCMC, so the Streamlit app stays fast.
    """
    prior_alpha, prior_beta = 7.0, 7.0  # neutral 50% prior, small sample protection
    evidence = []
    for col, weight in [("Season Hit%", 8), ("L20 Hit%", 6), ("L10 Hit%", 5), ("L5 Hit%", 3)]:
        val = safe_float(hit_info.get(col), np.nan)
        if pd.notna(val):
            p = val / 100.0
            if str(lean).upper() == "UNDER":
                p = 1.0 - p
            evidence.append((p, weight))
    sim_p = safe_float(sim.get("Over %" if str(lean).upper()=="OVER" else "Under %"), np.nan)
    if pd.notna(sim_p):
        evidence.append((sim_p/100.0, 8))
    logs = load_json(LEARNING_LOG, [])
    key = normalize_name(player)
    rows = [r for r in logs if normalize_name(r.get("Player")) == key and r.get("Market") == market and r.get("Result") in ["WIN","LOSS"]]
    if len(rows) >= 3:
        wr = sum(1 for r in rows if r.get("Result") == "WIN") / len(rows)
        evidence.append((wr, min(10, len(rows))))
    alpha, beta = prior_alpha, prior_beta
    for p, w in evidence:
        alpha += max(0.0, min(1.0, p)) * w
        beta += (1 - max(0.0, min(1.0, p))) * w
    posterior = alpha / (alpha + beta) if (alpha+beta) else 0.50
    sample_strength = min(100, (alpha + beta - prior_alpha - prior_beta) * 4)
    note = f"Bayes {posterior:.0%} / evidence {sample_strength:.0f}"
    return round(posterior*100, 1), round(sample_strength, 1), note

def role_confidence_and_minutes_grade(info):
    min_proj = safe_float(info.get("MIN Proj"), 0)
    min_l3 = safe_float(info.get("MIN L3"), min_proj)
    min_l5 = safe_float(info.get("MIN L5"), min_proj)
    starter_rate = safe_float(info.get("Starter Rate"), 0)
    role = str(info.get("Role", ""))
    stability = 100 - min(45, abs(min_l3 - min_l5) * 6)
    role_bonus = 18 if "STARTER" in role else 5
    starter_bonus = min(22, starter_rate * 22)
    minutes_bonus = min(30, max(0, min_proj - 16) * 1.7)
    role_conf = max(0, min(100, stability*.35 + role_bonus + starter_bonus + minutes_bonus))
    if min_proj >= 30 and role_conf >= 75:
        grade = "A"
    elif min_proj >= 26 and role_conf >= 65:
        grade = "B"
    elif min_proj >= 22 and role_conf >= 55:
        grade = "C"
    else:
        grade = "D"
    return round(role_conf,1), grade

def line_source_reliability(source, ud_line=np.nan, sleeper_line=np.nan, manual_line=np.nan):
    source = str(source or "")
    base = {"Underdog": 92, "Sleeper": 86, "Manual": 70}.get(source, 62)
    vals = [safe_float(x, np.nan) for x in [ud_line, sleeper_line, manual_line]]
    vals = [v for v in vals if pd.notna(v)]
    if len(vals) >= 2:
        spread = max(vals)-min(vals)
        if spread <= .5: base += 4
        elif spread >= 2.0: base -= 10
    return int(max(0, min(100, base)))

def ensure_default_injury_bump_table():
    if not os.path.exists(INJURY_USAGE_BUMP_FILE):
        pd.DataFrame([
            {"Player":"", "Team":"", "Market":"PTS", "Teammate Out":"", "Usage Bump %":0.0, "Minutes Bump":0.0, "Note":"manual optional"},
            {"Player":"", "Team":"", "Market":"REB", "Teammate Out":"", "Usage Bump %":0.0, "Minutes Bump":0.0, "Note":"manual optional"},
            {"Player":"", "Team":"", "Market":"AST", "Teammate Out":"", "Usage Bump %":0.0, "Minutes Bump":0.0, "Note":"manual optional"},
            {"Player":"", "Team":"", "Market":"PRA", "Teammate Out":"", "Usage Bump %":0.0, "Minutes Bump":0.0, "Note":"manual optional"},
        ]).to_csv(INJURY_USAGE_BUMP_FILE, index=False)

def load_injury_bumps():
    ensure_default_injury_bump_table()
    try:
        return pd.read_csv(INJURY_USAGE_BUMP_FILE)
    except Exception:
        return pd.DataFrame(columns=["Player","Team","Market","Teammate Out","Usage Bump %","Minutes Bump","Note"])

def injury_bump_for(player, market):
    bumps = load_injury_bumps()
    if bumps.empty:
        return 1.0, 0.0, ""
    key = normalize_name(player)
    d = bumps[(bumps.get("Player", pd.Series(dtype=str)).map(normalize_name) == key) & (bumps.get("Market", pd.Series(dtype=str)).astype(str).str.upper().isin([market, "ALL"]))]
    if d.empty:
        return 1.0, 0.0, ""
    usage_bump = safe_float(d.iloc[-1].get("Usage Bump %"), 0) / 100.0
    min_bump = safe_float(d.iloc[-1].get("Minutes Bump"), 0)
    note = f"Injury bump {usage_bump:+.0%}, min {min_bump:+.1f}"
    return max(.90, min(1.15, 1+usage_bump)), min_bump, note

def backup_logs():
    os.makedirs(LOG_BACKUP_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    copied = []
    for fp in [OFFICIAL_LOG, RESULT_LOG, LEARNING_LOG, LINE_HISTORY_FILE, MANUAL_LINES_FILE, BASELINE_CACHE_FILE, OFFICIAL_PICKS_CSV, RESULTS_LEARNING_CSV, LINES_HISTORY_CSV, NO_LINE_TRACKING_FILE, INJURY_USAGE_BUMP_FILE]:
        if os.path.exists(fp):
            dst = os.path.join(LOG_BACKUP_DIR, f"{stamp}_{os.path.basename(fp)}")
            try:
                import shutil
                shutil.copy2(fp, dst)
                copied.append(dst)
            except Exception:
                pass
    return copied

def reset_logs_only():
    backup_logs()
    for fp in [OFFICIAL_LOG, RESULT_LOG, LEARNING_LOG, LINE_HISTORY_FILE, OFFICIAL_PICKS_CSV, RESULTS_LEARNING_CSV, LINES_HISTORY_CSV, NO_LINE_TRACKING_FILE]:
        try:
            if os.path.exists(fp): os.remove(fp)
        except Exception:
            pass

def import_learning_csv(uploaded):
    try:
        df = pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded)
    except Exception:
        return 0
    current = load_json(LEARNING_LOG, [])
    rows = df.to_dict('records')
    current.extend(rows)
    save_json(LEARNING_LOG, current)
    pd.DataFrame(current).to_csv(RESULTS_LEARNING_CSV, index=False)
    return len(rows)

def project_row(row, base, logs, xgb_model=None, xgb_features=None):
    player = row["Player"]; market = row["Market"]; line = row["Line"]
    name_key = normalize_name(player)
    candidates = base.copy()
    candidates["Score"] = candidates["Player"].map(lambda x: name_score(player, x))
    match = candidates.sort_values("Score", ascending=False).head(1)
    if match.empty or match.iloc[0]["Score"] < 0.78:
        proj = np.nan
        info = {"Data Score": 25, "Projection Note": "No stat baseline match", "Matched Player": "", "XGB Projection": np.nan}
    else:
        b = match.iloc[0]
        mavg = safe_float(b.get(f"{market}_avg"), np.nan)
        ml3 = safe_float(b.get(f"{market}_l3"), mavg)
        ml5 = safe_float(b.get(f"{market}_l5"), mavg)
        ml10 = safe_float(b.get(f"{market}_l10"), mavg)
        min_avg = safe_float(b.get("MIN_avg"), 0)
        min_l3 = safe_float(b.get("MIN_l3"), min_avg)
        min_l5 = safe_float(b.get("MIN_l5"), min_avg)
        min_l10 = safe_float(b.get("MIN_l10"), min_avg)
        # Layer 1 + 2: baseline/recent form
        baseline = 0.35*mavg + 0.25*ml10 + 0.25*ml5 + 0.15*ml3
        # Layer 4: minutes engine
        minutes_proj = 0.35*min_avg + 0.25*min_l10 + 0.25*min_l5 + 0.15*min_l3
        injury_factor, injury_min_bump, injury_note = injury_bump_for(player, market)
        minutes_proj = max(0, minutes_proj + injury_min_bump)
        minutes_factor = max(0.80, min(1.18, minutes_proj / max(min_avg, 1)))
        # Layer 3/5: matchup + team environment from team-rank/allowed tables.
        pace_factor = safe_float(row.get("Pace Factor"), 1.0)
        matchup_factor = safe_float(row.get("Matchup Factor"), 1.0)
        power_factor = safe_float(row.get("Power Factor"), 1.0)
        usage = safe_float(b.get("UsageProxy"), np.nan)
        usage_recent = safe_float(b.get("UsageRecent"), np.nan)
        usage_factor = 1.00
        if market in ["PTS", "PRA"] and pd.notna(usage):
            usage_factor = max(0.92, min(1.08, usage / 12.0))
        if pd.notna(usage_recent) and market in ["PTS", "PRA"]:
            usage_factor *= max(0.96, min(1.04, usage_recent / max((mavg / max(min_avg,1)), .01)))
        # XGBoost blend: only if enough historical rows exist and package is installed.
        xgb_est = xgb_projection(xgb_model, xgb_features or [], market, b) if xgb_model is not None else np.nan
        model_base = baseline
        if pd.notna(xgb_est) and 0 <= xgb_est <= 80:
            model_base = 0.72*baseline + 0.28*xgb_est
        learn_adj, learn_note = learning_adjustment(player, market, baseline-line)
        proj = model_base * minutes_factor * pace_factor * matchup_factor * power_factor * usage_factor * injury_factor + learn_adj
        data_score = 55 + min(25, safe_float(b.get("Games"), 0)*2) + (10 if min_avg >= 24 else 0) + (5 if match.iloc[0]["Score"] >= .92 else 0)
        if xgb_model is not None:
            data_score += 4
        starter_role = "STARTER" if safe_float(b.get("StarterRate"), 0) >= .45 or min_avg >= 26 else "BENCH/ROLE"
        info = {
            "Matched Player": b.get("Player"), "Match Score": round(float(match.iloc[0]["Score"]), 3),
            "MIN Proj": round(minutes_proj, 2), "MIN L3": round(min_l3,2), "MIN L5": round(min_l5,2), "MIN L10": round(min_l10,2),
            "Role": starter_role, "Starter Rate": round(safe_float(b.get("StarterRate"), 0),2),
            "Usage Proxy": round(usage, 2) if pd.notna(usage) else np.nan,
            "Usage Recent": round(usage_recent, 3) if pd.notna(usage_recent) else np.nan,
            "eFG%": round(safe_float(b.get("eFG%"), np.nan), 3), "TS%": round(safe_float(b.get("TS%"), np.nan), 3),
            "PlusMinus Avg": round(safe_float(b.get("PlusMinusAvg"), np.nan),2),
            "Efficiency Avg": round(safe_float(b.get("EfficiencyAvg"), np.nan),2),
            "XGB Projection": round(xgb_est, 2) if pd.notna(xgb_est) else np.nan,
            "Data Score": int(max(0, min(100, data_score))),
            "Projection Note": (learn_note + (" | " + injury_note if injury_note else "")),
        }
    hit = hit_rates_for_player(logs, name_key if info.get("Matched Player","")=="" else normalize_name(info.get("Matched Player")), market, line)
    return proj, {**info, **hit}

def monte_carlo(player, market, line, proj, logs, matched_player=""):
    if pd.isna(proj) or pd.isna(line):
        return {"Floor": np.nan, "Median": np.nan, "Ceiling": np.nan, "Over %": np.nan, "Under %": np.nan, "Volatility": "NA"}
    key = normalize_name(matched_player or player)
    vals = logs[logs["NameKey"] == key][market].dropna().astype(float)
    if len(vals) >= 5:
        sd = max(1.2, float(vals.tail(20).std(ddof=0)))
    else:
        sd = max(1.5, abs(proj)*0.22)
    rng = np.random.default_rng(stable_seed(player, market, line, round(proj,2), len(vals)))
    sims = rng.normal(proj, sd, 30000)
    sims = np.clip(sims, 0, None)
    over = float((sims > line).mean()*100)
    vol = "LOW" if sd < 3 else "MED" if sd < 5.5 else "HIGH"
    return {"Floor": round(np.percentile(sims, 15),2), "Median": round(np.percentile(sims,50),2), "Ceiling": round(np.percentile(sims,85),2), "Over %": round(over,1), "Under %": round(100-over,1), "Volatility": vol}

def make_projection_board(lines, logs):
    base = compute_baselines(logs)
    team_ranks = compute_team_ranks(logs)
    allowed = compute_opponent_allowed(logs)
    pos_allowed = compute_position_allowed(logs)
    xgb_model, xgb_features = (None, [])
    if os.path.exists(BASELINE_CACHE_FILE):
        try:
            xgb_model, xgb_features = train_xgb_model_from_csv(BASELINE_CACHE_FILE, os.path.getmtime(BASELINE_CACHE_FILE))
        except Exception:
            xgb_model, xgb_features = (None, [])
    if lines.empty:
        return pd.DataFrame(), base
    active = []
    for (namekey, market), grp in lines.groupby(["NameKey","Market"]):
        grp = grp.sort_values("Priority")
        primary = grp.iloc[0].copy()
        primary["Underdog Line"] = safe_float(grp[grp["Source"]=="Underdog"]["Line"].iloc[0], np.nan) if len(grp[grp["Source"]=="Underdog"]) else np.nan
        primary["Sleeper Line"] = safe_float(grp[grp["Source"]=="Sleeper"]["Line"].iloc[0], np.nan) if len(grp[grp["Source"]=="Sleeper"]) else np.nan
        primary["Manual Line"] = safe_float(grp[grp["Source"]=="Manual"]["Line"].iloc[0], np.nan) if len(grp[grp["Source"]=="Manual"]) else np.nan
        primary["Best Over Line"] = grp["Line"].min()
        primary["Best Under Line"] = grp["Line"].max()
        # Opening/current movement from line history if available.
        hist = pd.DataFrame(load_json(LINE_HISTORY_FILE, []))
        if not hist.empty:
            h = hist[(hist["Player"].map(normalize_name)==namekey) & (hist["Market"]==market)].copy()
            primary["Opening Line"] = safe_float(h.iloc[0].get("Line"), np.nan) if not h.empty else np.nan
            primary["Line Move"] = primary["Line"] - primary["Opening Line"] if pd.notna(primary.get("Opening Line", np.nan)) else np.nan
        else:
            primary["Opening Line"] = np.nan; primary["Line Move"] = np.nan
        active.append(primary)
    board = pd.DataFrame(active)
    rows = []
    league_pace = safe_float(team_ranks["Pace"].mean(), np.nan) if not team_ranks.empty and "Pace" in team_ranks else np.nan
    league_power = safe_float(team_ranks["PowerRating"].mean(), 0) if not team_ranks.empty and "PowerRating" in team_ranks else 0
    for _, r in board.iterrows():
        # Match player baseline first so team/opponent context can be assigned.
        tmp_base = base.copy()
        tmp_base["Score"] = tmp_base["Player"].map(lambda x: name_score(r["Player"], x)) if not tmp_base.empty else []
        match = tmp_base.sort_values("Score", ascending=False).head(1) if not tmp_base.empty else pd.DataFrame()
        player_team = match.iloc[0].get("Team", r.get("Team", "")) if not match.empty and match.iloc[0].get("Score",0) >= .78 else r.get("Team", "")
        player_pos = match.iloc[0].get("PositionBucket", "UNKNOWN") if not match.empty and match.iloc[0].get("Score",0) >= .78 else "UNKNOWN"
        # Try opponent from Raw/Start not always available; manual lines can include Opponent column.
        opponent = r.get("Opponent", "") or r.get("Opp", "") or ""
        tr = team_ranks[team_ranks["Team"] == str(player_team).upper()] if not team_ranks.empty else pd.DataFrame()
        team_pace = safe_float(tr.iloc[0].get("Pace"), league_pace) if not tr.empty else league_pace
        team_power = safe_float(tr.iloc[0].get("PowerRating"), league_power) if not tr.empty else league_power
        r["Pace Factor"] = 1.0 if pd.isna(team_pace) or pd.isna(league_pace) or league_pace == 0 else max(.96, min(1.04, team_pace/league_pace))
        r["Power Factor"] = max(.97, min(1.03, 1 + ((team_power - league_power)/100)))
        r["Player Team"] = player_team
        r["Opponent"] = opponent
        r["PositionBucket"] = player_pos
        r["Team ORtg"] = safe_float(tr.iloc[0].get("ORtg"), np.nan) if not tr.empty else np.nan
        r["Team DRtg"] = safe_float(tr.iloc[0].get("DRtg"), np.nan) if not tr.empty else np.nan
        r["Team NetRtg"] = safe_float(tr.iloc[0].get("NetRtg"), np.nan) if not tr.empty else np.nan
        r["Team Off Rank"] = tr.iloc[0].get("OffRank", np.nan) if not tr.empty else np.nan
        r["Team Def Rank"] = tr.iloc[0].get("DefRank", np.nan) if not tr.empty else np.nan
        r["Team Pace Rank"] = tr.iloc[0].get("PaceRank", np.nan) if not tr.empty else np.nan
        r["Team Net Rank"] = tr.iloc[0].get("NetRank", np.nan) if not tr.empty else np.nan
        # Opponent allowed factor. If opponent not known yet, neutral.
        mfac = 1.0
        if opponent and not allowed.empty:
            ar = allowed[allowed["OpponentTeam"] == str(opponent).upper()]
            col = f"{r['Market']}_allowed_pg"
            if not ar.empty and col in ar:
                league_allowed = safe_float(allowed[col].mean(), np.nan)
                opp_allowed = safe_float(ar.iloc[0].get(col), np.nan)
                if pd.notna(opp_allowed) and pd.notna(league_allowed) and league_allowed > 0:
                    mfac *= max(.93, min(1.07, opp_allowed / league_allowed))
        if opponent and player_pos != "UNKNOWN" and not pos_allowed.empty:
            pr = pos_allowed[(pos_allowed["OpponentTeam"] == str(opponent).upper()) & (pos_allowed["PositionBucket"] == player_pos)]
            pcol = f"{r['Market']}_pos_allowed"
            if not pr.empty and pcol in pr:
                league_pos = safe_float(pos_allowed[pcol].mean(), np.nan)
                opp_pos = safe_float(pr.iloc[0].get(pcol), np.nan)
                if pd.notna(opp_pos) and pd.notna(league_pos) and league_pos > 0:
                    mfac *= max(.96, min(1.04, opp_pos/league_pos))
        r["Matchup Factor"] = mfac
        proj, info = project_row(r, base, logs, xgb_model, xgb_features)
        sim = monte_carlo(r["Player"], r["Market"], r["Line"], proj, logs, info.get("Matched Player", ""))
        role_conf, min_grade = role_confidence_and_minutes_grade(info)
        source_rel = line_source_reliability(r.get("Source"), r.get("Underdog Line"), r.get("Sleeper Line"), r.get("Manual Line"))
        edge = proj - r["Line"] if pd.notna(proj) else np.nan
        lean = "OVER" if pd.notna(edge) and edge > 0 else "UNDER" if pd.notna(edge) else "PASS"
        gap = abs(edge) if pd.notna(edge) else 0
        confidence = 0
        if pd.notna(edge):
            confidence = min(100, max(0, info.get("Data Score",0)*.45 + max(sim.get("Over %",0), sim.get("Under %",0))*.45 + min(10, gap*4)))
        pass_reason = []
        if info.get("Data Score",0) < 70: pass_reason.append("data")
        if gap < 1.0: pass_reason.append("thin edge")
        if sim.get("Volatility") == "HIGH": pass_reason.append("volatility")
        if info.get("MIN Proj",0) < 22: pass_reason.append("minutes")
        official = "PASS"
        bayes_pct, bayes_strength, bayes_note = bayesian_confidence(r.get("Player"), r.get("Market"), r.get("Line"), lean, info, info.get("Data Score",0), sim)
        matchup_strength = int(max(0, min(100, 50 + (safe_float(r.get("Matchup Factor"),1)-1)*500 + (safe_float(r.get("Pace Factor"),1)-1)*250)))
        official_score = min(100, max(0, confidence*.55 + bayes_pct*.25 + role_conf*.12 + source_rel*.08))
        if source_rel < 75: pass_reason.append("line source")
        if role_conf < 55: pass_reason.append("role")
        if bayes_pct < 50 and pd.notna(edge): pass_reason.append("bayes")
        if info.get("Data Score",0) >= 70 and gap >= 1.0 and info.get("MIN Proj",0) >= 22 and role_conf >= 55 and source_rel >= 70 and bayes_pct >= 50:
            if lean == "OVER" and sim.get("Over %",0) >= 56:
                official = "🔥 OVER"
            elif lean == "UNDER" and sim.get("Under %",0) >= 56:
                official = "⚠️ UNDER"
        rows.append({**r.to_dict(), **info, **sim, "Projection": round(proj,2) if pd.notna(proj) else np.nan, "Edge": round(edge,2) if pd.notna(edge) else np.nan, "Lean": lean, "Official": official, "Confidence %": round(confidence,1), "Bayesian Confidence %": bayes_pct, "Bayesian Evidence": bayes_strength, "Bayesian Note": bayes_note, "Team Matchup Strength": matchup_strength, "Player Role Confidence": role_conf, "Minutes Safety Grade": min_grade, "Line Source Reliability": source_rel, "Official Play Score": round(official_score,1), "PASS Reason": ", ".join(dict.fromkeys(pass_reason)) if official == "PASS" else ""})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Official", "Confidence %", "Edge"], ascending=[True, False, False])
    return out, base

def save_split_stat_csvs(logs, season_last, season_now):
    if logs.empty:
        return
    d = logs.copy()
    if "SEASON" in d.columns:
        last = d[pd.to_numeric(d["SEASON"], errors="coerce") == int(season_last)]
        current = d[pd.to_numeric(d["SEASON"], errors="coerce") == int(season_now)]
    else:
        last = pd.DataFrame(); current = d
    if not last.empty:
        last.to_csv(LAST_YEAR_PLAYER_STATS_FILE, index=False)
        compute_team_ranks(last).to_csv(TEAM_RANKS_LAST_YEAR_FILE, index=False)
    if not current.empty:
        current.to_csv(CURRENT_OPENING_STATS_FILE, index=False)
        compute_team_ranks(current).to_csv(TEAM_RANKS_CURRENT_FILE, index=False)

# -----------------------------
# UI helpers
# -----------------------------
def inject_css():
    st.markdown("""
    <style>
    .stApp { background: #090d12; color: #eef3f7; }
    div[data-testid="stMetric"] { background:#111822; border:1px solid #202c3b; border-radius:16px; padding:12px; }
    .card { background:linear-gradient(145deg,#101722,#111b29); border:1px solid #263649; border-radius:18px; padding:16px; margin:10px 0; box-shadow: 0 0 16px rgba(0,0,0,.25); }
    .badge { display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid #33465c; margin-right:6px; font-size:.82rem; }
    .hot { color:#70ffbd; font-weight:800; }
    .warn { color:#ffd166; font-weight:800; }
    .pass { color:#9aa7b2; font-weight:700; }
    </style>
    """, unsafe_allow_html=True)

def render_card(r):
    cls = "hot" if "OVER" in str(r.get("Official")) else "warn" if "UNDER" in str(r.get("Official")) else "pass"
    st.markdown(f"""
    <div class='card'>
      <h3>{r.get('Player','')} <span class='badge'>{r.get('Market','')}</span></h3>
      <div class='{cls}'>{r.get('Official','PASS')} — {r.get('Lean','')}</div>
      <p><b>Line:</b> {r.get('Line','')} ({r.get('Source','')}) &nbsp; | &nbsp; <b>Projection:</b> {r.get('Projection','')} &nbsp; | &nbsp; <b>Edge:</b> {r.get('Edge','')}</p>
      <p><b>UD:</b> {r.get('Underdog Line','')} &nbsp; <b>Sleeper:</b> {r.get('Sleeper Line','')} &nbsp; <b>Best Over:</b> {r.get('Best Over Line','')} &nbsp; <b>Best Under:</b> {r.get('Best Under Line','')}</p>
      <p><b>MC:</b> Over {r.get('Over %','')}% / Under {r.get('Under %','')}% &nbsp; | &nbsp; <b>Floor/Median/Ceiling:</b> {r.get('Floor','')} / {r.get('Median','')} / {r.get('Ceiling','')} &nbsp; | &nbsp; <b>Vol:</b> {r.get('Volatility','')}</p>
      <p><b>L5/L10/L20 Hit:</b> {r.get('L5 Hit%','')}% / {r.get('L10 Hit%','')}% / {r.get('L20 Hit%','')}% &nbsp; | &nbsp; <b>MIN:</b> {r.get('MIN Proj','')} &nbsp; | &nbsp; <b>Data:</b> {r.get('Data Score','')}/100</p>
      <p><b>Confidence:</b> {r.get('Confidence %','')}% &nbsp; | &nbsp; <b>Role:</b> {r.get('Role','')} &nbsp; | &nbsp; <b>XGB:</b> {r.get('XGB Projection','')}</p>
      <p><b>Team ORtg/DRtg/Net:</b> {r.get('Team ORtg','')} / {r.get('Team DRtg','')} / {r.get('Team NetRtg','')} &nbsp; | &nbsp; <b>Ranks O/D/Pace:</b> {r.get('Team Off Rank','')} / {r.get('Team Def Rank','')} / {r.get('Team Pace Rank','')}</p>
      <small>{r.get('Projection Note','')} {(' | PASS: ' + str(r.get('PASS Reason',''))) if str(r.get('PASS Reason','')) else ''}</small>
    </div>
    """, unsafe_allow_html=True)

def save_officials(df):
    plays = df[df["Official"].astype(str).str.contains("OVER|UNDER", na=False)].copy()
    if plays.empty:
        return 0
    log = load_json(OFFICIAL_LOG, [])
    stamp = now_iso()
    for _, r in plays.iterrows():
        row = r.to_dict(); row["SavedAt"] = stamp; row["Result"] = "PENDING"; row["Actual"] = None
        log.append(row)
    save_json(OFFICIAL_LOG, log)
    # Add line history
    hist = load_json(LINE_HISTORY_FILE, [])
    for _, r in df.iterrows():
        hist.append({"SavedAt": stamp, "Player": r.get("Player"), "Market": r.get("Market"), "Line": r.get("Line"), "Source": r.get("Source"), "Projection": r.get("Projection")})
    save_json(LINE_HISTORY_FILE, hist)
    try:
        pd.DataFrame(log).to_csv(OFFICIAL_PICKS_CSV, index=False)
        pd.DataFrame(hist).to_csv(LINES_HISTORY_CSV, index=False)
    except Exception:
        pass
    return len(plays)

def grade_pending(logs):
    official = load_json(OFFICIAL_LOG, [])
    if not official:
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
        lean = str(row.get("Lean",""))
        line = safe_float(row.get("Line"), np.nan)
        if pd.isna(line):
            continue
        win = (actual > line and lean == "OVER") or (actual < line and lean == "UNDER")
        row["Actual"] = actual; row["Result"] = "WIN" if win else "LOSS"; row["GradedAt"] = now_iso()
        learn.append(row.copy())
        updated += 1
    save_json(OFFICIAL_LOG, official)
    save_json(LEARNING_LOG, learn)
    try:
        pd.DataFrame(learn).to_csv(RESULTS_LEARNING_CSV, index=False)
    except Exception:
        pass
    return updated

# -----------------------------
# Main app
# -----------------------------
st.set_page_config(page_title="ONE WAY PICKZ WNBA", page_icon="🏀", layout="wide")
inject_css()
st.title("🏀 ONE WAY PICKZ — WNBA Prop Engine")
st.caption(APP_VERSION)

with st.sidebar:
    st.header("Setup")
    season_now = st.number_input("Current season", min_value=2020, max_value=2030, value=datetime.now().year, step=1)
    season_last = st.number_input("Last season baseline", min_value=2020, max_value=2030, value=datetime.now().year-1, step=1)
    use_ud = st.toggle("Pull Underdog", value=True)
    use_sleeper = st.toggle("Pull Sleeper", value=True)
    st.markdown("**Markets active:** PTS, REB, AST, PRA")
    st.caption(f"XGBoost: {'ON' if XGBOOST_AVAILABLE else 'Not installed — app still works'}")

tabs = st.tabs(["Board", "Manual Lines", "Stats/Baselines", "Team Ranks", "Official + Grade", "Log Tools", "Debug"])

with tabs[1]:
    st.subheader("Manual fallback lines")
    st.caption("Use this when Underdog/Sleeper miss players. These lines are included with Source=Manual.")
    existing = load_manual_lines()
    edited = st.data_editor(existing, num_rows="dynamic", use_container_width=True, column_config={"Market": st.column_config.SelectboxColumn(options=MARKETS)})
    if st.button("Save manual lines"):
        save_manual_lines(edited)
        st.success("Manual lines saved.")

with tabs[2]:
    st.subheader("WNBA Stats Baseline")
    st.caption("Pulls last season + current season game logs when the WNBA Stats API allows it. You can also upload a CSV/XLSX with PLAYER_NAME, MIN, PTS, REB, AST.")
    uploaded = st.file_uploader("Optional stats upload", type=["csv","xlsx"])
    if uploaded:
        logs = uploaded_stats_to_logs(uploaded)
        if not logs.empty:
            logs.to_csv(BASELINE_CACHE_FILE, index=False)
            save_split_stat_csvs(logs, int(season_last), int(season_now))
            st.success(f"Loaded/uploaded {len(logs)} stat rows and cached them.")
    if st.button("Pull last year + current season stats"):
        logs = fetch_player_game_logs([int(season_last), int(season_now)])
        if logs.empty:
            st.error("Stats pull returned empty. Upload a baseline CSV/XLSX or try again later.")
        else:
            logs.to_csv(BASELINE_CACHE_FILE, index=False)
            save_split_stat_csvs(logs, int(season_last), int(season_now))
            st.success(f"Pulled and cached {len(logs)} WNBA player game rows.")
    if os.path.exists(BASELINE_CACHE_FILE):
        logs = pd.read_csv(BASELINE_CACHE_FILE)
        logs = standardize_logs(logs)
        st.dataframe(compute_baselines(logs), use_container_width=True)
        save_split_stat_csvs(logs, int(season_last), int(season_now))
        st.download_button("Download combined baseline CSV", logs.to_csv(index=False), "wnba_baseline_cache.csv", "text/csv")
        if os.path.exists(LAST_YEAR_PLAYER_STATS_FILE):
            st.download_button("Download last year player stats CSV", open(LAST_YEAR_PLAYER_STATS_FILE, "rb").read(), "wnba_last_year_player_stats.csv", "text/csv")
        if os.path.exists(CURRENT_OPENING_STATS_FILE):
            st.download_button("Download current/opening stats CSV", open(CURRENT_OPENING_STATS_FILE, "rb").read(), "wnba_current_opening_stats.csv", "text/csv")
    else:
        st.warning("No cached stats yet. Pull stats or upload historical/opening-day stats first.")

# load cached stats globally
if os.path.exists(BASELINE_CACHE_FILE):
    logs_global = standardize_logs(pd.read_csv(BASELINE_CACHE_FILE))
else:
    logs_global = pd.DataFrame()

with tabs[0]:
    st.subheader("Projection Board")
    manual_df = load_manual_lines()
    lines, ud_debug, sl_debug = aggregate_lines(use_ud=use_ud, use_sleeper=use_sleeper, manual_df=manual_df)
    if not lines.empty:
        lines.to_csv(BOARD_CACHE_FILE, index=False)
    # Track players in the stat baseline with no active board line, useful for manual checking.
    try:
        if not logs_global.empty:
            base_tmp = compute_baselines(logs_global)
            line_keys = set(lines["NameKey"].tolist()) if not lines.empty and "NameKey" in lines.columns else set()
            no_line = base_tmp[~base_tmp["NameKey"].isin(line_keys)].copy()
            no_line["TrackedAt"] = now_iso()
            no_line[["TrackedAt","Player","Team","Games","MIN_avg","PTS_avg","REB_avg","AST_avg","PRA_avg"]].to_csv(NO_LINE_TRACKING_FILE, index=False)
    except Exception:
        pass
    if logs_global.empty:
        st.warning("Load stats first in Stats/Baselines. The board can pull lines, but projections need baselines.")
    st.metric("Lines loaded", len(lines))
    if lines.empty:
        st.error("No lines loaded. Try manual lines, or check Debug.")
    else:
        market_filter = st.multiselect("Market", MARKETS, default=MARKETS)
        search = st.text_input("Search player")
        proj_df, base_df = make_projection_board(lines[lines["Market"].isin(market_filter)], logs_global)
        if search:
            proj_df = proj_df[proj_df["Player"].str.contains(search, case=False, na=False)]
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Official plays", int(proj_df["Official"].astype(str).str.contains("OVER|UNDER", na=False).sum()) if not proj_df.empty else 0)
        with c2: st.metric("Avg edge", round(float(proj_df["Edge"].abs().mean()),2) if not proj_df.empty and "Edge" in proj_df else 0)
        with c3: st.metric("Avg data score", round(float(proj_df["Data Score"].mean()),1) if not proj_df.empty and "Data Score" in proj_df else 0)
        for _, r in proj_df.head(80).iterrows():
            render_card(r)
        with st.expander("Table view"):
            show_cols = ["Player","Market","Line","Source","Projection","XGB Projection","Edge","Lean","Official","Confidence %","PASS Reason","Underdog Line","Sleeper Line","Opening Line","Line Move","Best Over Line","Best Under Line","Over %","Under %","L5 Hit%","L10 Hit%","L20 Hit%","MIN Proj","MIN L3","MIN L5","MIN L10","Role","Usage Proxy","Usage Recent","eFG%","TS%","PlusMinus Avg","Efficiency Avg","Team ORtg","Team DRtg","Team NetRtg","Team Off Rank","Team Def Rank","Team Pace Rank","Team Matchup Strength","Player Role Confidence","Minutes Safety Grade","Line Source Reliability","Bayesian Confidence %","Official Play Score","Data Score"]
            st.dataframe(proj_df[[c for c in show_cols if c in proj_df.columns]], use_container_width=True)
            st.download_button("Download projection board CSV", proj_df.to_csv(index=False), "wnba_projection_board.csv", "text/csv")

with tabs[3]:
    st.subheader("Team ranks + advanced formulas")
    st.caption("Adds last-year/current offensive rank, defensive rank, pace rank, net rating rank, points/rebounds/assists allowed ranks, Pythagorean expectation, plus/minus, and BPI/Elo-style power proxy.")
    if logs_global.empty:
        st.warning("Load stats first in Stats/Baselines.")
    else:
        team_all = compute_team_ranks(logs_global)
        st.dataframe(team_all, use_container_width=True)
        st.download_button("Download team ranks CSV", team_all.to_csv(index=False), "wnba_team_ranks_combined.csv", "text/csv")
        if "SEASON" in logs_global.columns:
            last_r = compute_team_ranks(logs_global, int(season_last))
            cur_r = compute_team_ranks(logs_global, int(season_now))
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Last season ranks")
                st.dataframe(last_r, use_container_width=True)
                st.download_button("Download last year team ranks", last_r.to_csv(index=False), "wnba_team_ranks_last_year.csv", "text/csv")
            with c2:
                st.markdown("### Current/opening ranks")
                st.dataframe(cur_r, use_container_width=True)
                st.download_button("Download current team ranks", cur_r.to_csv(index=False), "wnba_team_ranks_current.csv", "text/csv")
        st.markdown("### Opponent allowed by market")
        st.dataframe(compute_opponent_allowed(logs_global), use_container_width=True)
        st.markdown("### Position allowed: guard / wing / big")
        st.dataframe(compute_position_allowed(logs_global), use_container_width=True)

with tabs[4]:
    st.subheader("Official picks + grading")
    if os.path.exists(BOARD_CACHE_FILE) and not logs_global.empty:
        board_cache = pd.read_csv(BOARD_CACHE_FILE)
        proj_df, _ = make_projection_board(board_cache, logs_global)
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
        st.download_button("Download official picks CSV", official.to_csv(index=False), "wnba_official_picks_log.csv", "text/csv")
    learning = pd.DataFrame(load_json(LEARNING_LOG, []))
    if not learning.empty:
        wins = (learning["Result"] == "WIN").sum(); total = len(learning)
        st.metric("Learning win rate", f"{wins}/{total} ({wins/total:.1%})")

with tabs[5]:
    st.subheader("Log tools + injury bump table")
    st.caption("Import previous learning logs, manage backups/resets, and manually apply injury/usage bumps without changing the code.")
    c1, c2 = st.columns(2)
    with c1:
        up_learn = st.file_uploader("Import previous learning log CSV/XLSX", type=["csv","xlsx"], key="import_learning")
        if up_learn and st.button("Import learning log"):
            n = import_learning_csv(up_learn)
            st.success(f"Imported {n} learning rows.")
        if st.button("Backup all logs now"):
            copied = backup_logs()
            st.success(f"Backed up {len(copied)} files.")
    with c2:
        st.warning("Reset clears official/results/learning/line-history tracking only. It does not delete baseline stats or manual lines.")
        confirm_reset = st.checkbox("I understand reset will clear logs")
        if st.button("Reset/backup logs"):
            if confirm_reset:
                reset_logs_only()
                st.success("Logs backed up and reset.")
            else:
                st.error("Check the confirmation box first.")
    st.markdown("### Injury usage bump table")
    bump_df = load_injury_bumps()
    edited_bumps = st.data_editor(bump_df, num_rows="dynamic", use_container_width=True, column_config={"Market": st.column_config.SelectboxColumn(options=["ALL"]+MARKETS)})
    if st.button("Save injury bump table"):
        edited_bumps.to_csv(INJURY_USAGE_BUMP_FILE, index=False)
        st.success("Injury bump table saved.")
    if os.path.exists(NO_LINE_TRACKING_FILE):
        st.markdown("### No-line player tracking")
        no_line = pd.read_csv(NO_LINE_TRACKING_FILE)
        st.dataframe(no_line, use_container_width=True)
        st.download_button("Download no-line tracking CSV", no_line.to_csv(index=False), "wnba_no_line_tracking.csv", "text/csv")
    if os.path.exists(RESULTS_LEARNING_CSV):
        st.download_button("Download results learning CSV", open(RESULTS_LEARNING_CSV, "rb").read(), "wnba_results_learning_log.csv", "text/csv")
    if os.path.exists(OFFICIAL_PICKS_CSV):
        st.download_button("Download official picks CSV", open(OFFICIAL_PICKS_CSV, "rb").read(), "wnba_official_picks_log.csv", "text/csv")

with tabs[6]:
    st.subheader("Debug")
    st.caption("Use this to see whether Underdog/Sleeper returned rows. Sleeper public routes may be blocked/changed; manual fallback remains active.")
    lines, ud_debug, sl_debug = aggregate_lines(use_ud=use_ud, use_sleeper=use_sleeper, manual_df=load_manual_lines())
    st.markdown("### Aggregated lines")
    st.dataframe(lines, use_container_width=True)
    st.markdown("### Underdog debug")
    st.dataframe(ud_debug, use_container_width=True)
    st.markdown("### Sleeper debug")
    st.dataframe(sl_debug, use_container_width=True)
    st.markdown("### ESPN schedule")
    st.dataframe(fetch_espn_schedule(2), use_container_width=True)
