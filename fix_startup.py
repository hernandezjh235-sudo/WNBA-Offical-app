#!/usr/bin/env python3
"""
Quick fix: Replace the blocking module-level call with session state initialization.
"""
import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the problematic lines
old_pattern = r'logs_global, master_global = get_global_datasets\(\)\n\n# MLB-style final-result check'
new_pattern = '''# Initialize datasets lazily via session state to avoid blocking startup
if "wnba_logs_global" not in st.session_state:
    st.session_state["wnba_logs_global"] = pd.DataFrame()
if "wnba_master_global" not in st.session_state:
    st.session_state["wnba_master_global"] = pd.DataFrame()

logs_global = st.session_state.get("wnba_logs_global", pd.DataFrame())
master_global = st.session_state.get("wnba_master_global", pd.DataFrame())

# Load datasets on first access (lazy loading)
if logs_global.empty and master_global.empty:
    logs_global, master_global = get_global_datasets()
    st.session_state["wnba_logs_global"] = logs_global
    st.session_state["wnba_master_global"] = master_global

# MLB-style final-result check'''

content = re.sub(old_pattern, new_pattern, content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed!")

