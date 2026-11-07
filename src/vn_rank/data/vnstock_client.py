"""vnstock generation 5 adapter implementing :class:`~vn_rank.data.download.PriceClient`.

The only module that imports vnstock. It requests adjusted daily prices from one source at
full precision (``floating=None``) and translates library errors: retryable ones become
:class:`TransientSourceError`, per-symbol input or response errors become
:class:`SymbolDataError`, and everything else (authentication, entitlement, expired release)
propagates and stops the download.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

import pandas as pd

from vn_rank.data.download import SnapshotSpec, SymbolDataError, TransientSourceError

logger = logging.getLogger(__name__)
DAILY_INTERVAL = "1D"


class VnstockClient:
    """:class:`PriceClient` backed by vnstock generation 5 (imported lazily)."""

    def __init__(self, spec: SnapshotSpec) -> None:
        """Create the vnstock entry points; requires the ``data`` dependency group."""
        import vnstock  # noqa: PLC0415 - optional dependency, only needed to download
        from vnstock.core import exceptions  # noqa: PLC0415

        self._spec, self._vnstock, self._errors = spec, vnstock, exceptions
        self._market: Any = vnstock.Market()
        self._reference: Any = vnstock.Reference()

    def library_version(self) -> str:
        """Return the installed vnstock version."""
        return str(self._vnstock.__version__)

    def list_symbols(self) -> pd.DataFrame:
        """Return every symbol by exchange, including the ``DELISTED`` code."""
        return self._translate(
            lambda: self._reference.equity.list_by_exchange(source=self._spec.source)
        )

    def equity_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Return one stock's adjusted daily OHLCV at full source precision."""
        return self._translate(lambda: self._market.equity(symbol).ohlcv(**self._ohlcv_args()))

    def index_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Return one index's daily OHLCV; ``limit`` lifts the 1,000-row default."""
        return self._translate(lambda: self._market.index(symbol).ohlcv(**self._ohlcv_args()))

    def _ohlcv_args(self) -> dict[str, Any]:
        spec = self._spec
        return {
            "start": spec.start.isoformat(),
            "end": spec.end.isoformat(),
            "interval": DAILY_INTERVAL,
            "source": spec.source,
            "floating": None,
            "limit": spec.max_rows,
        }

    def _translate(self, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        errors = self._errors
        try:
            with warnings.catch_warnings(record=True) as caught:
                frame = fn()
        except errors.VnstockError as exc:
            if exc.retryable or isinstance(exc, errors.RateLimitedError):
                raise TransientSourceError(exc.message_en) from exc
            if isinstance(exc, errors.InputError | errors.ProviderError | errors.SchemaError):
                raise SymbolDataError(exc.message_en) from exc
            raise
        for warning in caught:  # tier-cap notices; the counts are kept from df.attrs
            logger.debug("vnstock: %s", warning.message)
        return frame
