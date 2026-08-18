"""Block A -- row-local features (architecture.md 7.1).

This block alone is experiment **E1**, the control arm of the headline ablation. It must
therefore be a genuinely fair opponent: a deliberately weak baseline would make the
reported graph lift meaningless, and a judge who notices that has good reason to discount
the whole result. So it includes the round-number and threshold-proximity heuristics that a
real transaction-monitoring rules engine would use, not just raw amount and currency.

Every column here is ``row_local`` -- computed from the row and nothing else. No account
history, no graph, no other rows. That makes the class trivially true rather than
argued for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aml.features.base import Causality, FeatureBlock, FeatureContext, FeatureSpec, register

ROW_LOCAL = Causality.ROW_LOCAL

# Common cash-reporting threshold. Structuring is the practice of splitting a payment to sit
# just underneath it, so proximity from below is the signal -- not proximity in general.
REPORTING_THRESHOLD = 10_000.0

# Outside 08:00-18:00. Synthetic data, so this is weaker evidence than it would be on real
# bank traffic; it is included because it costs nothing and is a standard monitoring feature.
OFF_HOURS_START, OFF_HOURS_END = 8, 18


@register
class TabularBlock:
    name = "tabular"
    group = "tabular"
    requires_snapshot = False

    def columns(self) -> list[FeatureSpec]:
        d = lambda n, desc: FeatureSpec(n, ROW_LOCAL, desc)  # noqa: E731
        return [
            d("log_amount_paid", "log1p of amount sent, in payment currency"),
            d("log_amount_received", "log1p of amount received, in receiving currency"),
            d("amount_mismatch_ratio", "received / paid; ~1 same-currency, FX rate otherwise"),
            d("currency_paid_code", "payment currency as an integer code (15 levels)"),
            d("currency_received_code", "receiving currency as an integer code (15 levels)"),
            d("payment_format_code", "payment rail as an integer code (7 levels)"),
            d("is_cross_currency", "payment and receiving currency differ"),
            d("is_cross_bank", "sending and receiving bank differ"),
            d("is_self_loop", "sender and receiver are the same account"),
            d("hour_of_day", "hour of the transaction timestamp, 0-23"),
            d("day_of_week", "weekday of the transaction timestamp, 0=Monday"),
            d("is_off_hours", f"outside {OFF_HOURS_START:02d}:00-{OFF_HOURS_END:02d}:00"),
            d("is_round_100", "amount paid is an exact multiple of 100"),
            d("is_round_1000", "amount paid is an exact multiple of 1000"),
            d("trailing_zero_count", "trailing zeros in the integer part of amount paid"),
            d("threshold_distance", f"signed distance to the {REPORTING_THRESHOLD:,.0f} threshold, log scale"),
            d("below_threshold_band", f"within 10% underneath {REPORTING_THRESHOLD:,.0f} -- structuring proxy"),
        ]

    def compute(self, ctx: FeatureContext) -> pd.DataFrame:
        df = ctx.transactions
        paid = df["amount_paid"].to_numpy(dtype=np.float64)
        received = df["amount_received"].to_numpy(dtype=np.float64)
        ts = df["timestamp"]

        out = pd.DataFrame(index=df.index)
        out["log_amount_paid"] = np.log1p(paid)
        out["log_amount_received"] = np.log1p(received)
        # Guard the divide rather than letting a zero-amount row emit inf: LightGBM handles
        # NaN natively and meaningfully, inf silently poisons split thresholds.
        out["amount_mismatch_ratio"] = np.divide(
            received, paid, out=np.full_like(received, np.nan), where=paid > 0
        )

        # Categoricals become stable integer codes. Model-side one-hot / native categorical
        # handling is a modelling decision (Phase 4), not a feature-engineering one; what
        # matters here is that the mapping is deterministic across runs, which sorting gives.
        for col, name in (
            ("currency_paid", "currency_paid_code"),
            ("currency_received", "currency_received_code"),
            ("payment_format", "payment_format_code"),
        ):
            out[name] = _stable_codes(df[col])

        out["is_cross_currency"] = df["is_cross_currency"].to_numpy(dtype=np.float32)
        out["is_cross_bank"] = df["is_cross_bank"].to_numpy(dtype=np.float32)
        out["is_self_loop"] = df["is_self_loop"].to_numpy(dtype=np.float32)

        out["hour_of_day"] = ts.dt.hour.to_numpy()
        out["day_of_week"] = ts.dt.dayofweek.to_numpy()
        hours = out["hour_of_day"].to_numpy()
        out["is_off_hours"] = ((hours < OFF_HOURS_START) | (hours >= OFF_HOURS_END)).astype(np.float32)

        out["is_round_100"] = (np.mod(paid, 100.0) == 0).astype(np.float32)
        out["is_round_1000"] = (np.mod(paid, 1000.0) == 0).astype(np.float32)
        out["trailing_zero_count"] = _trailing_zeros(paid)

        # Signed and log-scaled: negative below the threshold, positive above. The sign is
        # the informative part -- structuring sits just *under* the line, and an unsigned
        # distance would make 9,900 and 10,100 look identical to the model.
        gap = paid - REPORTING_THRESHOLD
        out["threshold_distance"] = np.sign(gap) * np.log1p(np.abs(gap))
        out["below_threshold_band"] = (
            (paid >= REPORTING_THRESHOLD * 0.9) & (paid < REPORTING_THRESHOLD)
        ).astype(np.float32)

        return out.astype(np.float32)


def _stable_codes(series: pd.Series) -> np.ndarray:
    """Integer codes assigned in sorted category order.

    Sorted rather than first-appearance so the mapping is a pure function of the value set
    and does not shift if the row order ever changes -- the same reasoning as the node
    interner (graph/interner.py).
    """
    categories = pd.Index(sorted(series.dropna().unique()))
    return categories.get_indexer(series).astype(np.float32)


def _trailing_zeros(amounts: np.ndarray) -> np.ndarray:
    """Trailing zeros in the integer part, capped at 6.

    A hand-picked round number (50,000) is a weak but real laundering tell, and it is the
    kind of signal a tabular baseline *should* have access to so the ablation stays fair.
    """
    integral = np.floor(np.abs(amounts)).astype(np.int64)
    count = np.zeros(len(integral), dtype=np.float32)
    active = integral > 0
    for power in range(1, 7):
        divisor = 10**power
        hit = active & (np.mod(integral, divisor) == 0)
        if not hit.any():
            break
        count[hit] = power
    return count


assert isinstance(TabularBlock(), FeatureBlock)
