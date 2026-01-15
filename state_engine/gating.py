"""Deterministic gating logic for PA State Engine outputs."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Iterable

import pandas as pd

from .labels import StateLabels


@dataclass(frozen=True)
class GatingThresholds:
    trend_margin_min: float = 0.15
    balance_margin_min: float = 0.10
    transition_margin_min: float = 0.10
    transition_breakmag_min: float = 0.25
    transition_reentry_min: float = 1.0
    allowed_sessions: Iterable[str] | None = None
    state_age_min: int | None = None
    state_age_max: int | None = None
    dist_vwap_atr_min: float | None = None
    dist_vwap_atr_max: float | None = None


class GatingPolicy:
    """Apply ALLOW_* rules based on StateEngine state and margin."""

    def __init__(self, thresholds: GatingThresholds | None = None) -> None:
        self.thresholds = thresholds or GatingThresholds()

    def apply(self, outputs: pd.DataFrame, features: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return DataFrame with ALLOW_* columns."""
        required = {"state_hat", "margin"}
        missing = required - set(outputs.columns)
        if missing:
            raise ValueError(f"Missing required output columns: {sorted(missing)}")

        th = self.thresholds
        state_hat = outputs["state_hat"]
        margin = outputs["margin"]
        ctx_filters_pass = pd.Series(True, index=outputs.index)
        ctx_filters_pass = self._apply_context_filters(ctx_filters_pass, outputs, features)
        allow_trend_pullback = (state_hat == StateLabels.TREND) & (margin >= th.trend_margin_min)
        allow_trend_continuation = (state_hat == StateLabels.TREND) & (margin >= th.trend_margin_min)
        allow_balance_fade = (state_hat == StateLabels.BALANCE) & (margin >= th.balance_margin_min)

        allow_transition_failure = (state_hat == StateLabels.TRANSITION) & (margin >= th.transition_margin_min)
        if features is not None:
            required_features = {"BreakMag", "ReentryCount"}
            missing_features = required_features - set(features.columns)
            if missing_features:
                raise ValueError(f"Missing required features for gating: {sorted(missing_features)}")
            allow_transition_failure &= (
                (features["BreakMag"] >= th.transition_breakmag_min)
                & (features["ReentryCount"] >= th.transition_reentry_min)
            )

        return pd.DataFrame(
            {
                "ALLOW_trend_pullback": (allow_trend_pullback & ctx_filters_pass).astype(int),
                "ALLOW_trend_continuation": (allow_trend_continuation & ctx_filters_pass).astype(int),
                "ALLOW_balance_fade": (allow_balance_fade & ctx_filters_pass).astype(int),
                "ALLOW_transition_failure": (allow_transition_failure & ctx_filters_pass).astype(int),
            },
            index=outputs.index,
        )

    def _apply_context_filters(
        self,
        ctx_pass: pd.Series,
        outputs: pd.DataFrame,
        features: pd.DataFrame | None,
    ) -> pd.Series:
        th = self.thresholds
        if th.allowed_sessions is not None:
            if "ctx_session_bucket" in outputs.columns:
                allowed = {str(val) for val in th.allowed_sessions}
                ctx_pass &= outputs["ctx_session_bucket"].astype(str).isin(allowed)
        if th.state_age_min is not None:
            if "ctx_state_age" in outputs.columns:
                ctx_pass &= outputs["ctx_state_age"] >= th.state_age_min
        if th.state_age_max is not None:
            if "ctx_state_age" in outputs.columns:
                ctx_pass &= outputs["ctx_state_age"] <= th.state_age_max
        if th.dist_vwap_atr_min is not None:
            source = outputs if "ctx_dist_vwap_atr" in outputs.columns else features
            if source is not None and "ctx_dist_vwap_atr" in source.columns:
                ctx_pass &= source["ctx_dist_vwap_atr"] >= th.dist_vwap_atr_min
        if th.dist_vwap_atr_max is not None:
            source = outputs if "ctx_dist_vwap_atr" in outputs.columns else features
            if source is not None and "ctx_dist_vwap_atr" in source.columns:
                ctx_pass &= source["ctx_dist_vwap_atr"] <= th.dist_vwap_atr_max
        return ctx_pass


__all__ = ["GatingThresholds", "GatingPolicy"]
