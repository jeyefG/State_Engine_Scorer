"""Deterministic gating logic for PA State Engine outputs."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Iterable, Sequence

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
        def _get_col(column: str) -> pd.Series | None:
            if features is not None and column in features.columns:
                return features[column]
            if column in outputs.columns:
                return outputs[column]
            return None

        th = self.thresholds
        if th.allowed_sessions is not None:
            series = _get_col("ctx_session_bucket")
            if series is not None:
                allowed = {str(val) for val in th.allowed_sessions}
                ctx_pass &= series.astype(str).isin(allowed)
        if th.state_age_min is not None:
            series = _get_col("ctx_state_age")
            if series is not None:
                ctx_pass &= series >= th.state_age_min
        if th.state_age_max is not None:
            series = _get_col("ctx_state_age")
            if series is not None:
                ctx_pass &= series <= th.state_age_max
        if th.dist_vwap_atr_min is not None:
            series = _get_col("ctx_dist_vwap_atr")
            if series is not None:
                ctx_pass &= series >= th.dist_vwap_atr_min
        if th.dist_vwap_atr_max is not None:
            series = _get_col("ctx_dist_vwap_atr")
            if series is not None:
                ctx_pass &= series <= th.dist_vwap_atr_max
        return ctx_pass


def apply_allow_context_filters(
    gating_df: pd.DataFrame,
    symbol_cfg: dict | None,
    logger,
) -> pd.DataFrame:
    """Apply config-driven context filters to ALLOW_* rules."""
    if not symbol_cfg or not isinstance(symbol_cfg, dict):
        return gating_df
    allow_cfg = symbol_cfg.get("allow_context_filters")
    if not allow_cfg or not isinstance(allow_cfg, dict):
        return gating_df

    def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
        for name in candidates:
            if name in df.columns:
                return name
        return None

    filtered = gating_df.copy()
    for allow_rule, rule_cfg in allow_cfg.items():
        if not isinstance(rule_cfg, dict):
            logger.warning("allow_context_filters.%s must be a mapping; skipping.", allow_rule)
            continue
        if not rule_cfg.get("enabled", False):
            continue
        if allow_rule not in filtered.columns:
            logger.warning("allow_context_filters.%s missing in gating_df; skipping.", allow_rule)
            continue

        allow_series = filtered[allow_rule].astype(bool)
        before_rate = float(allow_series.mean()) if len(allow_series) else 0.0
        require_all = rule_cfg.get("require_all", True)

        masks: list[pd.Series] = []
        applied_conditions: list[str] = []

        sessions_in = rule_cfg.get("sessions_in")
        if sessions_in is not None:
            col_name = _resolve_column(filtered, ["session", "session_bucket"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing session column; skipping sessions_in.",
                    allow_rule,
                )
            else:
                allowed = {str(val) for val in sessions_in}
                mask = filtered[col_name].astype(str).isin(allowed)
                masks.append(mask)
                applied_conditions.append(f"sessions_in={sorted(allowed)} via {col_name}")

        state_age_min = rule_cfg.get("state_age_min")
        if state_age_min is not None:
            col_name = _resolve_column(filtered, ["state_age"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing state_age column; skipping state_age_min.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series >= float(state_age_min)
                masks.append(mask)
                applied_conditions.append(f"state_age_min>={state_age_min} via {col_name}")

        state_age_max = rule_cfg.get("state_age_max")
        if state_age_max is not None:
            col_name = _resolve_column(filtered, ["state_age"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing state_age column; skipping state_age_max.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series <= float(state_age_max)
                masks.append(mask)
                applied_conditions.append(f"state_age_max<={state_age_max} via {col_name}")

        dist_vwap_atr_min = rule_cfg.get("dist_vwap_atr_min")
        if dist_vwap_atr_min is not None:
            col_name = _resolve_column(filtered, ["dist_vwap_atr"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing dist_vwap_atr column; skipping dist_vwap_atr_min.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series >= float(dist_vwap_atr_min)
                masks.append(mask)
                applied_conditions.append(f"dist_vwap_atr_min>={dist_vwap_atr_min} via {col_name}")

        dist_vwap_atr_max = rule_cfg.get("dist_vwap_atr_max")
        if dist_vwap_atr_max is not None:
            col_name = _resolve_column(filtered, ["dist_vwap_atr"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing dist_vwap_atr column; skipping dist_vwap_atr_max.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series <= float(dist_vwap_atr_max)
                masks.append(mask)
                applied_conditions.append(f"dist_vwap_atr_max<={dist_vwap_atr_max} via {col_name}")

        breakmag_min = rule_cfg.get("breakmag_min")
        if breakmag_min is not None:
            col_name = _resolve_column(filtered, ["BreakMag"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing BreakMag column; skipping breakmag_min.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series >= float(breakmag_min)
                masks.append(mask)
                applied_conditions.append(f"breakmag_min>={breakmag_min} via {col_name}")

        breakmag_max = rule_cfg.get("breakmag_max")
        if breakmag_max is not None:
            col_name = _resolve_column(filtered, ["BreakMag"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing BreakMag column; skipping breakmag_max.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series <= float(breakmag_max)
                masks.append(mask)
                applied_conditions.append(f"breakmag_max<={breakmag_max} via {col_name}")

        reentry_min = rule_cfg.get("reentry_min")
        if reentry_min is not None:
            col_name = _resolve_column(filtered, ["ReentryCount"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing ReentryCount column; skipping reentry_min.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series >= float(reentry_min)
                masks.append(mask)
                applied_conditions.append(f"reentry_min>={reentry_min} via {col_name}")

        reentry_max = rule_cfg.get("reentry_max")
        if reentry_max is not None:
            col_name = _resolve_column(filtered, ["ReentryCount"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing ReentryCount column; skipping reentry_max.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series <= float(reentry_max)
                masks.append(mask)
                applied_conditions.append(f"reentry_max<={reentry_max} via {col_name}")

        if not masks:
            logger.info("allow_context_filters.%s no effective conditions; skipping.", allow_rule)
            continue

        if require_all:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask & mask
        else:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask | mask

        filtered[allow_rule] = (allow_series & combined_mask.fillna(False)).astype(int)
        after_rate = float(filtered[allow_rule].mean()) if len(filtered) else 0.0
        delta = after_rate - before_rate
        logger.info(
            "%s: before=%.2f%% after=%.2f%% delta=%.2f%%",
            allow_rule,
            before_rate * 100.0,
            after_rate * 100.0,
            delta * 100.0,
        )
        logger.info("%s conditions=%s", allow_rule, "; ".join(applied_conditions))

    return filtered


__all__ = ["GatingThresholds", "GatingPolicy", "apply_allow_context_filters"]
