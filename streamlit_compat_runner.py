# Streamlit UI compatibility runner for Railway/mobile stability.
# This does NOT modify WNBA projection/model logic. It only translates the
# deprecated use_container_width kwarg before Streamlit can emit thousands
# of deprecation messages during a large page render.

import functools
import inspect
import sys

import streamlit as st
from streamlit.delta_generator import DeltaGenerator


def _translate_container_width(fn):
    """Translate legacy Streamlit sizing kwargs without changing widget data/logic."""
    if getattr(fn, "_owp_container_width_compat", False):
        return fn

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if "use_container_width" in kwargs:
            legacy = kwargs.pop("use_container_width")
            if "width" not in kwargs:
                kwargs["width"] = "stretch" if bool(legacy) else "content"
        return fn(*args, **kwargs)

    wrapped._owp_container_width_compat = True
    return wrapped


def _accepts_legacy_kwarg(fn):
    try:
        return "use_container_width" in inspect.signature(fn).parameters
    except Exception:
        return False


def _patch_object(obj):
    patched = 0
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            fn = getattr(obj, name)
        except Exception:
            continue
        if not callable(fn) or not _accepts_legacy_kwarg(fn):
            continue
        try:
            setattr(obj, name, _translate_container_width(fn))
            patched += 1
        except Exception:
            pass
    return patched


# Patch both module-level st.* callables and DeltaGenerator methods because
# Streamlit exposes widgets through both paths.
_patch_object(st)
_patch_object(DeltaGenerator)

# Start the existing app unchanged.
sys.argv = [
    "streamlit",
    "run",
    "app.py",
    "--server.address=0.0.0.0",
    "--server.port=8080",
    "--server.headless=true",
    "--logger.level=error",
]

from streamlit.web.cli import main

raise SystemExit(main())
