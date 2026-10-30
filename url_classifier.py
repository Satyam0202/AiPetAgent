import ipaddress
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd


MODEL_PATH=Path(__file__).resolve().with_name("url_model.joblib")
BAD_TERMS=(
    "login", "verify", "account", "secure", "update", "password", "signin",
    "confirm", "banking", "wallet", "webscr", "support", "invoice",
)
SHORT_HOSTS={
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "buff.ly",
    "ow.ly", "rebrand.ly", "cutt.ly", "shorturl.at",
}
KNOWN_SAFE_ROOTS={
    "python.org", "www.python.org",
    "example.com", "www.example.com",
    "wikipedia.org", "www.wikipedia.org",
    "google.com", "www.google.com",
    "github.com", "www.github.com",
    "microsoft.com", "www.microsoft.com",
    "apple.com", "www.apple.com",
    "mozilla.org", "www.mozilla.org",
}


def url_feats(value):
    url=str(value).strip()
    low=url.lower()
    parsed=url if re.match(r"^[a-z][a-z0-9+.-]*://", url, re.I) else f"//{url}"
    try:
        parts=urlsplit(parsed)
        host=(parts.hostname or "").lower()
        port=parts.port
    except ValueError:
        parts=urlsplit("")
        host=""
        port=None

    ip_host=host.strip("[]")
    try:
        ipaddress.ip_address(ip_host)
        is_ip=1
    except ValueError:
        is_ip=0
    labels=[part for part in host.split(".") if part]
    digits=sum(character.isdigit() for character in url)
    term_n=sum(term in low for term in BAD_TERMS)
    counts=Counter(low)
    length=len(low)
    entropy=(
        -sum((count / length) * math.log2(count / length) for count in counts.values())
        if length else 0.0
    )
    return {
        "url_len": len(url),
        "host_len": len(host),
        "path_len": len(parts.path),
        "query_len": len(parts.query),
        "frag_len": len(parts.fragment),
        "dots": url.count("."),
        "hyphens": url.count("-"),
        "underscores": url.count("_"),
        "slashes": url.count("/"),
        "questions": url.count("?"),
        "equals": url.count("="),
        "ats": url.count("@"),
        "amps": url.count("&"),
        "percents": url.count("%"),
        "digits": digits,
        "digit_ratio": digits / max(len(url), 1),
        "subdomains": max(0, len(labels) - 2),
        "path_depth": len([part for part in parts.path.split("/") if part]),
        "is_ip": is_ip,
        "is_https": int(low.startswith("https://")),
        "is_http": int(low.startswith("http://")),
        "is_puny": int("xn--" in host),
        "has_port": int(port is not None),
        "has_frag": int(bool(parts.fragment)),
        "bad_term_n": term_n,
        "is_short": int(host in SHORT_HOSTS),
        "entropy": entropy,
    }


def is_url_input(text):
    if not isinstance(text, str) or not text or any(char.isspace() for char in text):
        return False

    has_scheme="://" in text
    if has_scheme and not text.lower().startswith(("http://", "https://")):
        return False
    if not has_scheme and "@" in text:
        return False

    candidate=text if has_scheme else f"//{text}"
    try:
        parts=urlsplit(candidate)
        host=parts.hostname
        parts.port
    except ValueError:
        return False
    if not host:
        return False
    if has_scheme or "." in host:
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def predict_url(url):
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Enter a non-empty URL string.")

    url=url.strip()
    if url.lower().startswith("https://"):
        try:
            parts=urlsplit(url)
            host=(parts.hostname or "").lower()
            port=parts.port
        except ValueError:
            parts=urlsplit("")
            host=""
            port=None
        if (
            host in KNOWN_SAFE_ROOTS
            and port is None
            and parts.path in ("", "/")
            and not parts.query
            and not parts.fragment
        ):
            return {
                "prediction": "benign",
                "harmful_probability": 0.0,
                "warning": (
                    "Exact HTTPS homepage matches a small local known-domain list; "
                    "this does not verify redirects or any URL with a path."
                ),
            }

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"URL model not found at {MODEL_PATH}. Run all cells in url.ipynb first."
        )

    artifact=joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("The saved URL model artifact is invalid.")
    model=artifact["model"]
    feature_columns=artifact.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("The saved URL model has no feature schema.")
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError("The saved URL model cannot produce class probabilities.")
    features=url_feats(url)
    if set(feature_columns) != set(features):
        raise ValueError("The saved URL model feature schema does not match the extractor.")
    row=pd.DataFrame([features], columns=feature_columns)
    classes=list(model.classes_)
    if "harmful" not in classes:
        raise ValueError("The saved URL model has no harmful class.")
    risk=float(model.predict_proba(row)[0][classes.index("harmful")])
    return {
        "prediction": "harmful" if risk >= 0.5 else "benign",
        "harmful_probability": risk,
        "warning": "Text-only model estimate; it does not visit the URL or guarantee safety.",
    }
