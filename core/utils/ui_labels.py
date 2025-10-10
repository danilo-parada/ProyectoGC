"""Static labels and tooltips for KPI visualisations."""

from __future__ import annotations

from typing import Dict, Optional


class _SafeLookup(dict):
    """Dictionary that returns the key when missing instead of raising ``KeyError``."""

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return str(key)


LABELS: Dict[str, str] = _SafeLookup(
    {
        "dpp_emision_pago": "Días Promedio a Pago (DPP)",
        "dic_emision_contab": "Días facturas hasta contabilizarse (DFC)",
        "dcp_contab_pago": "Días desde Contabilización al Pago (DCP)",
        "total_facturado": "Monto total facturado",
        "total_pagado_real": "Monto total pagado (real)",
    }
)

TOOLTIPS: Dict[str, str] = _SafeLookup(
    {
        "dpp_emision_pago": "Promedio de días entre emisión de factura y pago efectivo.",
        "dic_emision_contab": (
            "Promedio de días entre emisión y contabilización de facturas, "
            "destacando cuáles siguen contabilizadas sin pago o pendientes."
        ),
        "dcp_contab_pago": "Promedio de días entre contabilización y pago.",
        "total_facturado": "Incluye todo el monto facturado del periodo, incluso facturas aún sin pago.",
        "total_pagado_real": "Suma `monto_pagado_real`: usa `monto_autorizado` de la hoja Facturas cuando hay fecha de pago.",
        "desglose_facturado": (
            "Usa la columna `monto_facturado`. Pagado considera facturas con "
            "`monto_pagado_real` > 0; Sin pagar agrupa el resto."
        ),
    }
)

def get_label(key: str, default: Optional[str] = None) -> str:
    """Return a human friendly label for ``key``.

    ``default`` allows callers to provide a fallback when the key is unknown.
    When ``default`` is ``None`` the original key is returned so the UI can
    still render something meaningful instead of failing with ``KeyError``.
    """

    if default is None:
        default = key
    return LABELS.get(key, default)


def get_tooltip(key: str, default: str = "") -> str:
    """Return the tooltip text associated with ``key``.

    Missing tooltips yield ``default`` (empty string by default) instead of
    raising ``KeyError`` which previously crashed the Streamlit page.
    """

    return TOOLTIPS.get(key, default)


__all__ = ["LABELS", "TOOLTIPS", "get_label", "get_tooltip"]
