from __future__ import annotations

from typing import Any

import pandas as pd


class PanelStore:
    _panel: pd.DataFrame | None = None

    @classmethod
    def set_panel(cls, panel: pd.DataFrame) -> None:
        cls._panel = panel

    @classmethod
    def get_panel(cls) -> pd.DataFrame:
        if cls._panel is None:
            raise RuntimeError("Panel not loaded")
        return cls._panel

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._panel is not None

    @classmethod
    def rows(cls) -> int:
        if cls._panel is None:
            return 0
        return int(cls._panel.shape[0])
