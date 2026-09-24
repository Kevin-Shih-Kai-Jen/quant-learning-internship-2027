"""Confirmed initial regime slots. Not yet wired to the trading engine.

Weights are fractions of capital available for new positions. Existing carried
positions reserve capital first. The existing stock sign-transfer rule applies
after these initial slots, and may change the final side counts and exposure.
"""
from math import isfinite
from jpx_common import FEATURES, allocate, price_t_features
from jpx_models import PriceVolumeSameModel


class MarketPriceModel(PriceVolumeSameModel):
    """Four-coefficient MSE/GD regression, reusing the unblended model pipeline.

    Callers supply T5/T22/T60 and aligned, training-only Target observations.
    No volume, lag features, contract rolling or label/calendar assumptions here.
    """
    name = 'market_price_t'
    feature_names = FEATURES

    def fit(self, train, lag_train=None):
        super().fit(train, lag_train)
        self.u_fit = {'estimated': False, 'reason': 'market price T model, no blending'}
        return self


def allocate_regime(day, market_g):
    """Reuse original wrong-sign transfer logic with regime initial capital."""
    return allocate(day, side_weights=initial_regime_weights(market_g))


def initial_regime_weights(market_g):
    """Return (long weights, short weights), both positive magnitudes.

    Missing or infinite forecasts are errors, not neutral market signals.
    No smoothing, tolerance band, or stock-selection policy is introduced.
    """
    market_g = float(market_g)
    if not isfinite(market_g):
        raise ValueError("Market prediction must be finite")
    if market_g > 0:
        return (0.35, 0.21, 0.14), (0.15, 0.09, 0.06)
    if market_g < 0:
        return (0.15, 0.09, 0.06), (0.35, 0.21, 0.14)
    return (0.25, 0.15, 0.10), (0.25, 0.15, 0.10)
