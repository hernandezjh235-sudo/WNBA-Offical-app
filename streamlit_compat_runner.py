# Streamlit UI compatibility runner for Railway/mobile stability.
# This does NOT change WNBA projection/model math. It creates a runtime-only
# copy of app.py with deprecated Streamlit sizing kwargs translated before
# Streamlit parses/runs the script. The repository app.py remains untouched.

from pathlib import Path
import re
import sys

SOURCE = Path("app.py")
RUNTIME = Path("/tmp/wnba_app_runtime.py")

text = SOURCE.read_text(encoding="utf-8")

# Streamlit 1.62 emits deprecation messages for every legacy sizing kwarg.
# This app renders a large board, so those messages can exceed Railway's
# 500 logs/sec cap. Translate the two legacy forms in a runtime-only copy.
true_pattern = re.compile(r"\buse_container_width\s*=\s*True\b")
false_pattern = re.compile(r"\buse_container_width\s*=\s*False\b")

n_true = len(true_pattern.findall(text))
n_false = len(false_pattern.findall(text))
text = true_pattern.sub('width="stretch"', text)
text = false_pattern.sub('width="content"', text)

RUNTIME.write_text(text, encoding="utf-8")

# One concise startup line only; avoid noisy per-widget logging.
print(f"[streamlit-compat] runtime copy ready: translated {n_true + n_false} use_container_width calls", flush=True)

sys.argv = [
    "streamlit",
    "run",
    str(RUNTIME),
    "--server.address=0.0.0.0",
    "--server.port=8080",
    "--server.headless=true",
    "--logger.level=error",
]

from streamlit.web.cli import main

raise SystemExit(main())
