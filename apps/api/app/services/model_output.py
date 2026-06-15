from __future__ import annotations

from typing import Any


CONFIDENCE_LABELS: dict[str, float] = {
    "none": 0.0,
    "very_low": 0.1,
    "very low": 0.1,
    "low": 0.3,
    "低": 0.3,
    "uncertain": 0.3,
    "medium": 0.6,
    "中": 0.6,
    "中等": 0.6,
    "moderate": 0.6,
    "normal": 0.6,
    "high": 0.85,
    "高": 0.85,
    "very_high": 0.95,
    "very high": 0.95,
    "很高": 0.95,
    "certain": 0.95,
}


def coerce_confidence(value: Any, default: float = 0.0) -> tuple[float, dict[str, Any]]:
    raw = value
    normalized_from: str | None = None
    confidence: float
    try:
        if isinstance(value, bool):
            confidence = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            confidence = float(value)
        elif isinstance(value, str):
            text = value.strip()
            lowered = text.lower()
            if lowered in CONFIDENCE_LABELS:
                confidence = CONFIDENCE_LABELS[lowered]
                normalized_from = "label"
            elif lowered.endswith("%"):
                confidence = float(lowered[:-1].strip()) / 100.0
                normalized_from = "percentage"
            else:
                confidence = float(lowered)
                normalized_from = "numeric_string"
        else:
            confidence = float(default)
            normalized_from = "default"
    except (TypeError, ValueError):
        confidence = float(default)
        normalized_from = "default"
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
        normalized_from = normalized_from or "percentage_number"
    confidence = max(0.0, min(1.0, confidence))
    diagnostics: dict[str, Any] = {"confidence_raw": raw, "confidence": round(confidence, 6)}
    if normalized_from:
        diagnostics["confidence_normalized_from"] = normalized_from
    return round(confidence, 6), diagnostics
