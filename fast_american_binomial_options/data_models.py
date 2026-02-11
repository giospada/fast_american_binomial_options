"""
Option data models and utilities.
"""

import numpy as np
from dataclasses import dataclass, fields
from typing import Optional, Tuple, Literal



@dataclass
class AmericanOption:
    """Single American option parameters."""

    spot_price: float
    strike_price: float
    time_to_expiry: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float
    steps: int
    option_type: Literal["call", "put"]


@dataclass
class AmericanOptions:
    """Batch of American options for vectorized processing."""

    spot_price: np.ndarray
    strike_price: np.ndarray
    time_to_expiry: np.ndarray
    risk_free_rate: np.ndarray
    volatility: np.ndarray
    dividend_yield: np.ndarray
    steps: np.ndarray
    option_type: np.ndarray

    @classmethod
    def from_options(cls, options: list[AmericanOption]) -> "AmericanOptions":
        """Create AmericanOptions batch from list of AmericanOption objects."""
        instance = cls()
        for opt in options:
            instance.spot_price.append(opt.spot_price)
            instance.strike_price.append(opt.strike_price)
            instance.time_to_expiry.append(opt.time_to_expiry)
            instance.risk_free_rate.append(opt.risk_free_rate)
            instance.volatility.append(opt.volatility)
            instance.dividend_yield.append(opt.dividend_yield)
            instance.steps.append(opt.steps)
            instance.option_type.append(opt.option_type)
        return instance

    def __getitem__(self, idx):
        """Support indexing and slicing."""
        if (
            isinstance(idx, slice)
            or isinstance(idx, np.ndarray)
            or isinstance(idx, int)
            or isinstance(idx, list)
        ):
            return AmericanOptions(
                spot_price=self.spot_price[idx],
                strike_price=self.strike_price[idx],
                time_to_expiry=self.time_to_expiry[idx],
                risk_free_rate=self.risk_free_rate[idx],
                volatility=self.volatility[idx],
                dividend_yield=self.dividend_yield[idx],
                steps=self.steps[idx],
                option_type=self.option_type[idx],
            )
        else:
            raise TypeError(
                "Invalid index type for AmericanOptions: {}".format(type(idx))
            )

    def __repr__(self) -> str:
        parts = [f"{field.name}={getattr(self, field.name)}" for field in fields(self)]
        return "AmericanOptions(" + ", ".join(parts) + ")"

    @classmethod
    def random(
        cls,
        m: int,
        *,
        seed: Optional[int] = None,
        dtype=np.float64,
        spot_bounds: Tuple[float, float] = (50.0, 250.0),
        strike_bounds: Tuple[float, float] = (50.0, 250.0),
        time_to_expiry_bounds: Tuple[float, float] = (1 / 365, 2.0),
        risk_free_rate_bounds: Tuple[float, float] = (-0.02, 0.10),
        volatility_bounds: Tuple[float, float] = (0.05, 1.00),
        dividend_yield_bounds: Tuple[float, float] = (0.0, 0.08),
        steps_bounds: Tuple[int, int] = (1, 10000),
        strike_mode: Literal["independent", "moneyness"] = "independent",
        moneyness_bounds: Tuple[float, float] = (0.7, 1.3),
        option_type_selection: Optional[Literal["call", "put"]] = None,
    ) -> "AmericanOptions":
        """Generate random American options for testing."""
        rng = np.random.default_rng(seed)

        S = rng.uniform(*spot_bounds, size=m).astype(dtype, copy=False)

        if strike_mode == "independent":
            K = rng.uniform(*strike_bounds, size=m).astype(dtype, copy=False)
        elif strike_mode == "moneyness":
            K = S * rng.uniform(*moneyness_bounds, size=m).astype(dtype, copy=False)
            K = np.clip(K, strike_bounds[0], strike_bounds[1]).astype(dtype, copy=False)
        else:
            raise ValueError('strike_mode must be "independent" or "moneyness".')

        T = rng.uniform(*time_to_expiry_bounds, size=m).astype(dtype, copy=False)
        r = rng.uniform(*risk_free_rate_bounds, size=m).astype(dtype, copy=False)
        sigma = rng.uniform(*volatility_bounds, size=m).astype(dtype, copy=False)
        q = rng.uniform(*dividend_yield_bounds, size=m).astype(dtype, copy=False)
        n = rng.integers(steps_bounds[0], steps_bounds[1] + 1, size=m, dtype=np.uint64)

        if option_type_selection is None:
            option_type = rng.choice(["call", "put"], size=m)
        else:
            option_type = np.full(m, option_type_selection)

        return cls(
            spot_price=S,
            strike_price=K,
            time_to_expiry=T,
            risk_free_rate=r,
            volatility=sigma,
            dividend_yield=q,
            steps=n,
            option_type=option_type,
        )

    def to_gpu_parameters(self, dtype=np.float64) -> np.ndarray:
        """Convert to GPU-compatible structured array."""
        # Calculate parameters
        deltaT = self.time_to_expiry / self.steps
        u_val = np.exp(self.volatility * np.sqrt(deltaT))
        d_val = 1 / u_val
        discount_factor = np.exp(-self.risk_free_rate * deltaT)
        p_val = (
            np.exp((self.risk_free_rate - self.dividend_yield) * deltaT) - d_val
        ) / (u_val - d_val)
        # Risk-neutral probabilities * discount factor
        up_factor = p_val * discount_factor
        down_factor = (1 - p_val) * discount_factor

        sign = np.where(self.option_type == "call", 1, -1)

        # Create structured NumPy array
        structured_array = np.empty(
            self.spot_price.shape,
            dtype=np.dtype(
                [
                    ("S", dtype),
                    ("K", dtype),
                    ("n", np.int32),
                    ("u", dtype),
                    ("up", dtype),
                    ("down", dtype),
                    ("sign", np.int32),
                ],
                align=True,
            ),
        )

        structured_array["S"] = self.spot_price.astype(dtype)
        structured_array["K"] = self.strike_price.astype(dtype)
        structured_array["n"] = self.steps.astype(np.int32)
        structured_array["u"] = u_val.astype(dtype)
        structured_array["up"] = up_factor.astype(dtype)
        structured_array["down"] = down_factor.astype(dtype)
        structured_array["sign"] = sign.astype(np.int32)

        return structured_array
