# Streamlit mobile-stability runner for Railway.
# UI-only runtime transform: no WNBA projection, probability, ranking,
# grading, App128 shadow, or App130 gate math is changed.

from pathlib import Path
import re
import sys

SOURCE = Path("app.py")
RUNTIME = Path("/tmp/wnba_app_runtime.py")
text = SOURCE.read_text(encoding="utf-8")
notes = []

# Translate old Streamlit sizing kwargs before Streamlit executes the app.
true_pattern = re.compile(r"\buse_container_width\s*=\s*True\b")
false_pattern = re.compile(r"\buse_container_width\s*=\s*False\b")
n_width = len(true_pattern.findall(text)) + len(false_pattern.findall(text))
text = true_pattern.sub('width="stretch"', text)
text = false_pattern.sub('width="content"', text)


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count == 1:
        text = text.replace(old, new, 1)
        notes.append(label)
    elif count == 0:
        # A direct App132 file may already contain the lazy-render fix.
        notes.append(label + ":already")
    else:
        raise RuntimeError(f"{label}: expected <=1 exact match, found {count}")


# IMPORTANT: st.tabs executes every tab body on every rerun. This app has very
# large cards/tables/reports, so iPhone Safari can terminate the page while the
# Railway/Streamlit server remains healthy. Replace only the UI navigation with
# lazy controls so ONE section is built and sent to the browser at a time.
replace_once(
    'tabs = st.tabs(["Player Cards", "Best Bets", "Slate Tracker", "Moneyline", "Official + Grade", "Data Manager", "Debug / Status", "Model Reports", "HHS / Models"])',
    '''_owp_main_sections = ["Player Cards", "Best Bets", "Slate Tracker", "Moneyline", "Official + Grade", "Data Manager", "Debug / Status", "Model Reports", "HHS / Models"]\n_owp_main_view = st.selectbox(\n    "App section",\n    _owp_main_sections,\n    index=_owp_main_sections.index(st.session_state.get("owp_main_section", "Player Cards")) if st.session_state.get("owp_main_section", "Player Cards") in _owp_main_sections else 0,\n    key="owp_main_section",\n    help="Mobile-safe lazy navigation: only the selected section is rendered. Projection/model calculations are unchanged.",\n)\nif bool(st.session_state.get("owp_mobile_stability_mode", True)):\n    st.caption("📱 Lazy-render mode: only this section is loaded into Safari.")''',
    "main-nav",
)
for i, name in enumerate(["Player Cards", "Best Bets", "Slate Tracker", "Moneyline", "Official + Grade", "Data Manager", "Debug / Status", "Model Reports", "HHS / Models"]):
    replace_once(f"with tabs[{i}]:", f"if _owp_main_view == {name!r}:", f"main-block-{i}")

replace_once(
    '    slate_tabs = st.tabs(["Today", "Tomorrow", "All Lines"])',
    '    _owp_player_slate = st.radio("Player-card slate", ["Today", "Tomorrow", "All Lines"], horizontal=True, key="owp_player_slate_view")',
    "player-slate-nav",
)
for i, name in enumerate(["Today", "Tomorrow", "All Lines"]):
    replace_once(f"    with slate_tabs[{i}]:", f"    if _owp_player_slate == {name!r}:", f"player-slate-{i}")

replace_once(
    '    grade_tabs = st.tabs(["Save / Grade", "After Game Results ✅❌", "Raw Logs"])',
    '    _owp_grade_view = st.radio("Grade view", ["Save / Grade", "After Game Results ✅❌", "Raw Logs"], horizontal=True, key="owp_grade_view")',
    "grade-nav",
)
for i, name in enumerate(["Save / Grade", "After Game Results ✅❌", "Raw Logs"]):
    replace_once(f"    with grade_tabs[{i}]:", f"    if _owp_grade_view == {name!r}:", f"grade-{i}")

replace_once(
    '''    data_tab, compare_tab, evaluation_tab, diagnostics_tab = st.tabs(\n        ["HHS Data", "Model Compare", "Evaluation", "Diagnostics"]\n    )''',
    '    _owp_hhs_view = st.selectbox("HHS section", ["HHS Data", "Model Compare", "Evaluation", "Diagnostics"], key="owp_hhs_view")',
    "hhs-nav",
)
for var, name in [("data_tab", "HHS Data"), ("compare_tab", "Model Compare"), ("evaluation_tab", "Evaluation"), ("diagnostics_tab", "Diagnostics")]:
    replace_once(f"    with {var}:", f"    if _owp_hhs_view == {name!r}:", f"hhs-{var}")

# In mobile mode start with the compact fast table instead of rendering dozens
# of HTML cards. Users can still explicitly switch to Player cards.
replace_once(
    '        index=1 if default_cards else 0,',
    '        index=0 if bool(st.session_state.get("owp_mobile_stability_mode", True)) else (1 if default_cards else 0),',
    "mobile-fast-table",
)
replace_once(
    '''    show_all = st.toggle("Show all player cards", value=True, key=f"{key_prefix}_show_all_cards")\n    max_default = min(40, max(10, int(view_df["Player"].nunique() if "Player" in view_df.columns else 40)))\n    max_players = len(view_df["Player"].dropna().unique()) if show_all and "Player" in view_df.columns else st.slider("Max player cards", 10, 120, max_default, 5, key=f"{key_prefix}_max_cards")''',
    '''    _owp_mobile_cards = bool(st.session_state.get("owp_mobile_stability_mode", True))\n    show_all = st.toggle("Show all player cards", value=(not _owp_mobile_cards), key=f"{key_prefix}_show_all_cards")\n    _owp_card_floor = 5 if _owp_mobile_cards else 10\n    _owp_card_cap = 12 if _owp_mobile_cards else 40\n    max_default = min(_owp_card_cap, max(_owp_card_floor, int(view_df["Player"].nunique() if "Player" in view_df.columns else _owp_card_cap)))\n    max_players = len(view_df["Player"].dropna().unique()) if show_all and "Player" in view_df.columns else st.slider("Max player cards", _owp_card_floor, 120, max_default, 5, key=f"{key_prefix}_max_cards")''',
    "mobile-card-cap",
)

RUNTIME.write_text(text, encoding="utf-8")
print(
    f"[streamlit-mobile] runtime ready: width={n_width}; lazy={sum(1 for n in notes if not n.endswith(':already'))}; tabs_left={text.count('st.tabs(')}",
    flush=True,
)

sys.argv = [
    "streamlit", "run", str(RUNTIME),
    "--server.address=0.0.0.0",
    "--server.port=8080",
    "--server.headless=true",
    "--logger.level=error",
]
from streamlit.web.cli import main
raise SystemExit(main())
