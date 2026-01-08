"""Label generation for State Engine states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import pandas as pd


class StateLabels(IntEnum):
    BALANCE = 0
    TRANSITION = 1
    TREND = 2


@dataclass(frozen=True)
class StateLabeler:
    """Rule-based state labeling for offline training."""

    bootstrap_option: str = "A"
    confirmation_feature: str = "ReentryCount"

    trend_er: float = 0.38
    trend_netmove: float = 1.3
    trend_breakmag: float = 0.25

    balance_er: float = 0.22
    balance_netmove: float = 0.9
    balance_reentry: float = 2.0
    balance_inside_ratio: float = 0.20

    transition_breakmag: float = 0.25
    transition_reentry: float = 1.0
    transition_netmove: float = 1.0
    transition_er_low: float = 0.22
    transition_er_high: float = 0.38
    transition_range_slope: float = 0.0
    transition_inside_drop: float = 0.15

    def label(self, features: pd.DataFrame) -> pd.Series:
        """Return Series of StateLabels using PA rules."""
        required = {"ER", "NetMove"}
        missing = required - set(features.columns)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")

        er = features["ER"]
        net_move = features["NetMove"]

        if self.bootstrap_option not in {"A", "B"}:
            raise ValueError("bootstrap_option must be 'A' or 'B'.")

        if self.bootstrap_option == "A":
            balance_confirm, trend_confirm = self._confirmation(features)
            trend = (er >= self.trend_er) & (net_move >= self.trend_netmove) & trend_confirm
            balance = (er <= self.balance_er) & (net_move <= self.balance_netmove) & balance_confirm
            transition = self._transition_rules(features, er, net_move)
        else:
            trend = (er >= self.trend_er) & (net_move >= self.trend_netmove)
            balance = (er <= self.balance_er) & (net_move <= self.balance_netmove)
            transition = self._transition_rules(features, er, net_move)

        labels = pd.Series(StateLabels.TRANSITION, index=features.index)
        labels[balance] = StateLabels.BALANCE
        labels[trend] = StateLabels.TREND
        labels[transition] = StateLabels.TRANSITION
        return labels

    def _confirmation(self, features: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        feature = self.confirmation_feature
        if feature == "ReentryCount":
            balance_confirm = features["ReentryCount"] >= self.balance_reentry
            trend_confirm = features["ReentryCount"] <= 0
            return balance_confirm, trend_confirm
        if feature == "InsideBarsRatio":
            balance_confirm = features["InsideBarsRatio"] >= self.balance_inside_ratio
            trend_confirm = features["InsideBarsRatio"] <= self.balance_inside_ratio
            return balance_confirm, trend_confirm
        if feature == "BreakMag":
            balance_confirm = features["BreakMag"] <= self.trend_breakmag
            trend_confirm = features["BreakMag"] >= self.trend_breakmag
            return balance_confirm, trend_confirm
        raise ValueError("confirmation_feature must be ReentryCount, InsideBarsRatio, or BreakMag.")

    def _transition_rules(
        self,
        features: pd.DataFrame,
        er: pd.Series,
        net_move: pd.Series,
    ) -> pd.Series:
        break_mag = features.get("BreakMag")
        reentry = features.get("ReentryCount")
        range_slope = features.get("RangeSlope")
        inside_ratio = features.get("InsideBarsRatio")

        rule_a = pd.Series(False, index=features.index)
        rule_b = (net_move >= self.transition_netmove) & (er.between(self.transition_er_low, self.transition_er_high))
        rule_c = pd.Series(False, index=features.index)

        if break_mag is not None and reentry is not None:
            rule_a = (break_mag >= self.transition_breakmag) & (reentry >= self.transition_reentry)
        if range_slope is not None and inside_ratio is not None:
            prev_inside = inside_ratio.shift(1)
            drop = prev_inside - inside_ratio
            rule_c = (range_slope > self.transition_range_slope) & (drop >= self.transition_inside_drop)

        return rule_a | rule_b | rule_c


__all__ = ["StateLabels", "StateLabeler"]
