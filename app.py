from __future__ import annotations
import pandas as pd
import streamlit as st

from typing import List, Optional

from lib_common import (
    # setup / UI
    init_session_keys, read_any, style_table, header_ui, sanitize_df, safe_markdown,
    mapping_ui, apply_mapping, normalize_types,
    # data state
    get_df_norm, register_documents, reset_docs, reset_masters, reset_honorarios,
    reset_proveedores, reset_cuentas_especiales, load_honorarios,
    # masters
    load_cuentas_especiales, load_proveedores_prioritarios,
    # honorarios
    clean_estado_cuota, merge_honorarios_con_bancos,
    # resumen
    get_match_summary,
)

st.set_page_config(page_title="Carga de Data", layout="wide")
header_ui(
    "Carga de Data",
    current_page="Inicio",
    subtitle="Primero carga la Maestra de Cuentas de Banco y luego tus documentos.",
    nav_active="inicio",
)

# ---------------------------------------------------------------------
# Estado inicial
# ---------------------------------------------------------------------
init_session_keys()
ss = st.session_state

TAB_LABELS = [
    "1. Cuentas especiales",
    "2. Proveedores prioritarios",
    "3. Facturas",
    "4. Honorarios",
]

ss.setdefault("_active_tab", TAB_LABELS[0])


def _set_active_tab(label: str) -> None:
    """Persist the tab that should remain selected after a rerun."""

    if label in TAB_LABELS:
        ss["_active_tab"] = label


def _flash_and_rerun(level: str, message: str, *, tab: Optional[str] = None) -> None:
    """Persist a flash message and trigger UI refresh."""
    ss["_ui_flash"] = (level, message)
    if tab:
        _set_active_tab(tab)
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def _count_rows(obj) -> int:
    if isinstance(obj, pd.DataFrame):
        return len(obj)
    return 0


def _render_preview_table(
    df: Optional[pd.DataFrame],
    *,
    max_rows: int = 100,
    visible_rows: int = 12,
    drop_columns: Optional[List[str]] = None,
    integerize: bool = False,
) -> None:
    """Renderiza un fragmento de ``df`` con estilo profesional."""

    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No hay datos para mostrar en esta vista previa.")
        return

    preview = sanitize_df(df.head(max_rows).copy())

    if drop_columns:
        to_drop = [col for col in drop_columns if col in preview.columns]
        if to_drop:
            preview = preview.drop(columns=to_drop)

    if "ap" in preview.columns:
        ordered_cols = ["ap"] + [c for c in preview.columns if c != "ap"]
        preview = preview[ordered_cols]

    if integerize:
        numeric_cols = preview.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            series = pd.to_numeric(preview[col], errors="coerce")
            non_na = series.dropna()
            if not non_na.empty and (non_na % 1).abs().lt(1e-9).all():
                preview[col] = series.round().astype("Int64")
    table_styles = [
        {
            "selector": "table",
            "props": [
                ("border-collapse", "separate"),
                ("border-spacing", "0"),
            ],
        },
        {
            "selector": "thead",
            "props": [
                ("background", "linear-gradient(90deg, #1f3c88 0%, #2d77ff 100%)"),
                ("color", "#ffffff"),
            ],
        },
        {
            "selector": "thead th",
            "props": [
                ("padding", "16px 20px"),
                ("font-size", "0.95rem"),
                ("font-weight", "600"),
                ("border-bottom", "1px solid rgba(255, 255, 255, 0.3)"),
                ("letter-spacing", "0.01em"),
            ],
        },
        {
            "selector": "tbody td",
            "props": [
                ("padding", "12px 20px"),
                ("border-bottom", "1px solid #e3e9ff"),
                ("font-size", "0.94rem"),
                ("color", "#1b1f3b"),
            ],
        },
        {
            "selector": "tbody tr:nth-child(even)",
            "props": [("background-color", "#f6f8ff")],
        },
        {
            "selector": "tbody tr:hover",
            "props": [
                ("background-color", "#ecf2ff"),
                ("transition", "background-color 0.2s ease"),
            ],
        },
    ]

    styler = (
        preview.style.format(na_rep="—")
        .set_table_styles(table_styles, overwrite=False)
        .set_properties(**{"text-align": "left"})
    )

    if len(df) > max_rows:
        st.caption(f"Mostrando {max_rows:,} de {len(df):,} filas totales.")

    style_table(styler, visible_rows=visible_rows)


def _card_html(
    title: str,
    rows: int,
    ready: bool,
    hint: str,
    extras: Optional[List[str]] = None,
    actions: Optional[List[str]] = None,
) -> str:
    status = "Listo" if ready else "Pendiente"
    trend_class = "app-card__trend app-card__trend--success" if ready else "app-card__trend app-card__trend--warning"
    value = f"{rows:,}" if rows else "0"
    extras_html = "".join(extras) if extras else ""
    actions_html = "".join(actions) if actions else ""
    hint_html = (
        f'<p class="app-card__subtitle app-card__subtitle--muted">{hint}</p>' if hint else ""
    )
    return (
        '<div class="app-card">'
        f'{actions_html}'
        f'<div class="app-card__title">{title}</div>'
        f'<div class="app-card__value">{value}</div>'
        f'<div class="{trend_class}"><span>&bull; {status}</span></div>'
        f"{extras_html}"
        f"{hint_html}"
        '</div>'
    )

cta_rows = _count_rows(ss.get("df_ctaes_raw"))
prio_rows = _count_rows(ss.get("df_prio_raw"))
hon_rows = _count_rows(ss.get("honorarios"))
doc_rows = _count_rows(ss.get("df"))
if doc_rows > 0:
    ss["_facturas_count"] = doc_rows
else:
    doc_rows = ss.get("_facturas_count", 0)

def build_honorarios_summary(
    df_hon_enr: Optional[pd.DataFrame] = None,
    total_override: Optional[int] = None,
) -> dict[str, float]:
    """Calcula totales y porcentaje de match bancario para honorarios."""
    summary: dict[str, float] = {}
    if isinstance(df_hon_enr, pd.DataFrame) and not df_hon_enr.empty:
        columnas_match = [
            c for c in ["codigo_contable", "banco", "cuenta_corriente"] if c in df_hon_enr.columns
        ]
        matched = int(df_hon_enr[columnas_match].notna().all(axis=1).sum()) if columnas_match else 0
        total = len(df_hon_enr)
        pct = (matched / total * 100) if total else 0.0
        summary = {
            "total": total,
            "matched": matched,
            "pct": pct,
            "no_match": max(total - matched, 0),
        }
    if total_override is not None:
        total = int(total_override)
        matched = int(summary.get("matched", 0))
        pct = summary.get("pct")
        if total and pct is None:
            pct = matched / total * 100
        summary = {
            "total": total,
            "matched": matched,
            "pct": float(pct or 0.0),
            "no_match": max(total - matched, 0),
        }
    return summary


def build_facturas_summary(
    match_stats: Optional[dict],
    total_override: Optional[int] = None,
) -> dict[str, object]:
    """Resume porcentajes y cuentas para facturas normalizadas."""
    summary: dict[str, object] = {}
    if isinstance(match_stats, dict) and match_stats:
        cuenta = match_stats.get("cuenta_especial", {}) or {}
        prov = match_stats.get("prov_prioritario", {}) or {}
        summary = {
            "total": int(match_stats.get("total", 0)),
            "cuenta": {
                "si": int(cuenta.get("si", 0)),
                "no": int(cuenta.get("no", 0)),
                "pct_si": float(cuenta.get("pct_si", 0.0)),
            },
            "prov": {
                "si": int(prov.get("si", 0)),
                "no": int(prov.get("no", 0)),
                "pct_si": float(prov.get("pct_si", 0.0)),
            },
        }
    if total_override is not None:
        base = summary or {}
        base_total = int(total_override)
        summary = {
            "total": base_total,
            "cuenta": base.get("cuenta") or {"si": 0, "no": 0, "pct_si": 0.0},
            "prov": base.get("prov") or {"si": 0, "no": 0, "pct_si": 0.0},
        }
    return summary

hon_summary = ss.get("honorarios_summary")
hon_enr = ss.get("honorarios_enriquecido")
if isinstance(hon_enr, pd.DataFrame) and not hon_enr.empty:
    hon_summary = build_honorarios_summary(hon_enr)
    ss["honorarios_summary"] = hon_summary
elif hon_rows:
    if not hon_summary or int(hon_summary.get("total", 0)) != int(hon_rows):
        hon_summary = build_honorarios_summary(total_override=hon_rows)
        ss["honorarios_summary"] = hon_summary

fact_summary = ss.get("facturas_summary")
match_stats = None
if doc_rows > 0:
    try:
        match_stats = get_match_summary()
    except Exception:
        match_stats = None
    total_override = doc_rows or ss.get("_facturas_count", 0)
    new_summary = build_facturas_summary(match_stats, total_override=total_override)
    if new_summary:
        fact_summary = new_summary
        ss["facturas_summary"] = fact_summary
elif fact_summary is None and ss.get("_match_summary"):
    total_override = ss.get("_facturas_count", 0)
    fact_summary = build_facturas_summary(ss["_match_summary"], total_override=total_override)
    if fact_summary:
        ss["facturas_summary"] = fact_summary

hon_extras: list[str] = []
hon_hint = "Registros disponibles"
if hon_summary and hon_summary.get("total"):
    total_hon = int(hon_summary["total"])
    matched = int(hon_summary["matched"])
    no_match = int(hon_summary.get("no_match", max(total_hon - matched, 0)))
    pct = float(hon_summary["pct"])
    hon_extras = [
        '<div class="app-pill-group">'
        f'<div class="app-pill app-pill--active">Cuenta especial: {pct:.1f}%</div>'
        '</div>',
        '<div class="app-inline-stats">'
        f'<div class="app-inline-stats__item">Cuenta especial Si: {matched:,}</div>'
        f'<div class="app-inline-stats__item">No cuenta especial: {no_match:,}</div>'
        '</div>',
    ]
    hon_hint = "Indicadores cuenta especial para honorarios"
elif hon_rows > 0:
    hon_hint = "Registros listos para match bancario"

fact_extras: list[str] = []
fact_hint = "Registros de facturas activos"
if fact_summary and fact_summary.get("total"):
    cuenta = fact_summary.get("cuenta", {}) or {}
    prov = fact_summary.get("prov", {}) or {}
    cuenta_pct = float(cuenta.get("pct_si", 0.0))
    prov_pct = float(prov.get("pct_si", 0.0))
    fact_extras = [
        '<div class="app-pill-group">'
        f'<div class="app-pill app-pill--active">Cuenta especial: {cuenta_pct:.1f}%</div>'
        f'<div class="app-pill">Proveedor prioritario: {prov_pct:.1f}%</div>'
        '</div>',
        '<div class="app-inline-stats">'
        f'<div class="app-inline-stats__item">Cuenta especial Si: {int(cuenta.get("si", 0)):,}</div>'
        f'<div class="app-inline-stats__item">Prioritario Si: {int(prov.get("si", 0)):,}</div>'
        '</div>',
    ]
    fact_hint = "Indicadores de match actualizados"
elif doc_rows > 0:
    fact_hint = "Registros listos para match bancario/proveedores"

card_cta_html = _card_html("Cuentas especiales", cta_rows, cta_rows > 0, "Filas maestras cargadas")
card_prov_html = _card_html("Proveedores prioritarios", prio_rows, prio_rows > 0, "Filas prioritarias cargadas")
card_hon_html = _card_html("Honorarios", hon_rows, bool(hon_summary), hon_hint, extras=hon_extras)
card_fact_html = _card_html("Facturas normalizadas", doc_rows, bool(fact_summary), fact_hint, extras=fact_extras)

col_cta, col_prov, col_fact, col_hon = st.columns(4)

with col_cta:
    if st.button("Reset cuentas", key="card_reset_cta", use_container_width=True):
        reset_cuentas_especiales()
        _flash_and_rerun("warning", "Maestra de cuentas especiales eliminada.")
    safe_markdown(card_cta_html)

with col_prov:
    if st.button("Reset proveedores", key="card_reset_prov", use_container_width=True):
        reset_proveedores()
        _flash_and_rerun("warning", "Proveedores prioritarios eliminados.")
    safe_markdown(card_prov_html)

with col_fact:
    if st.button("Reset facturas", key="card_reset_fact", use_container_width=True):
        reset_docs()
        _flash_and_rerun("warning", "Facturas eliminadas de la sesion actual.")
    safe_markdown(card_fact_html)

with col_hon:
    if st.button("Reset honorarios", key="card_reset_hon", use_container_width=True):
        reset_honorarios()
        _flash_and_rerun("warning", "Honorarios eliminados de la sesion.")
    safe_markdown(card_hon_html)

flash_payload = ss.pop("_ui_flash", None)
if flash_payload:
    level, message = flash_payload
    notifier = getattr(st, level, st.info)
    notifier(message)

tab_cta, tab_prov, tab_fact, tab_hon = st.tabs(TAB_LABELS, default=ss.get("_active_tab"))

with tab_cta:
    st.markdown("#### 1. Maestra de cuentas (obligatorio)")
    safe_markdown(
        """
        <div class="app-note">
            Archivo requerido con columnas: <code>ano_proyecto</code>, <code>codigo_contable</code>,
            <code>proyecto</code>, <code>banco</code>, <code>cuenta_corriente</code>, <code>cuenta_contable</code>,
            <code>cuenta_cont_descripcion</code>, <code>sede_pago</code>.
        </div>
        """,
    )
    maestra_file = st.file_uploader(
        "Sube la maestra de cuentas (xlsx/csv)",
        type=["xlsx", "xls", "csv"],
        key="upload_ctaes",
    )
    if maestra_file is not None:
        _set_active_tab(TAB_LABELS[0])
        try:
            df_ctaes = read_any(maestra_file)
            load_cuentas_especiales(df_ctaes, col_codigo_contable="codigo_contable")
            ss["df_ctaes_raw"] = df_ctaes
            st.success(f"Maestra de cuentas cargada: {len(df_ctaes):,} filas.")
            if ss.get("df") is not None:
                df_new = normalize_types(ss["df"])
                register_documents(df_new, dedup=False)
                st.info("Se actualizo la marca de Cuenta Especial en la base cargada.")
        except Exception as e:
            st.error(f"No se pudo leer la maestra de cuentas: {e}")
    cta_preview_open = False
    if isinstance(ss.get("df_ctaes_raw"), pd.DataFrame):
        df_cta_preview = ss["df_ctaes_raw"]
        cta_preview_open = not df_cta_preview.empty
    with st.expander("Vista previa maestra (100 filas)", expanded=cta_preview_open):
        _render_preview_table(ss.get("df_ctaes_raw"))

with tab_prov:
    st.markdown("#### 2. Proveedores prioritarios (opcional)")
    safe_markdown(
        """
        <div class="app-note">
            Archivo con columna <code>codigo_proveedor</code> (puede incluir otros campos).
        </div>
        """,
    )
    prov_file = st.file_uploader(
        "Sube listado de proveedores prioritarios (xlsx/csv)",
        type=["xlsx", "xls", "csv"],
        key="upload_prio",
    )
    if prov_file is not None:
        _set_active_tab(TAB_LABELS[1])
        try:
            df_prov = read_any(prov_file)
            load_proveedores_prioritarios(df_prov, col_codigo="codigo_proveedor")
            ss["df_prio_raw"] = df_prov
            st.success(f"Prioritarios cargados: {len(df_prov):,} filas.")
            if ss.get("df") is not None:
                df_new = normalize_types(ss["df"])
                register_documents(df_new, dedup=False)
                st.info("Se actualizo la marca de Proveedor Prioritario en la base cargada.")
        except Exception as e:
            st.error(f"No se pudo leer el archivo de proveedores prioritarios: {e}")
    prio_preview_open = False
    if isinstance(ss.get("df_prio_raw"), pd.DataFrame):
        df_prio_preview = ss["df_prio_raw"]
        prio_preview_open = not df_prio_preview.empty
    with st.expander("Vista previa proveedores (100 filas)", expanded=prio_preview_open):
        _render_preview_table(ss.get("df_prio_raw"))

with tab_fact:
    st.markdown("#### 3. Facturas")
    safe_markdown(
        """
        <div class="app-note">
            Carga uno o varios archivos y luego mapea columnas para habilitar todos los tableros.
        </div>
        """,
    )

    files = st.file_uploader(
        "Sube facturas (xlsx/csv)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        key="upload_docs",
    )

    if files:
        _set_active_tab(TAB_LABELS[2])
        try:
            df_list = [read_any(f) for f in files]
            df_raw = pd.concat(df_list, ignore_index=True)
            ss["df_raw"] = df_raw
            st.success(f"Facturas cargadas: {len(df_raw):,} filas en total.")

            col_map, ok = mapping_ui(df_raw.columns)
            ss["col_map"] = col_map

            dedup_flag = st.checkbox(
                "Evitar duplicados exactos (activa si subiste el mismo archivo)",
                value=False,
            )

            if st.button("Aplicar mapeo y normalizacion", type="primary"):
                _set_active_tab(TAB_LABELS[2])
                if not ok:
                    st.error("Falta asignar fac_fecha_factura en el mapeo. Corrige y vuelve a intentar.")
                else:
                    df_norm = apply_mapping(df_raw, col_map)
                    register_documents(df_norm, dedup=dedup_flag)
                    ss["_facturas_count"] = len(df_norm)
                    match_payload = None
                    try:
                        match_payload = get_match_summary()
                    except Exception:
                        match_payload = None
                    ss["facturas_summary"] = build_facturas_summary(
                        match_payload, total_override=len(df_norm)
                    )
                    _flash_and_rerun(
                        "success",
                        f"Base normalizada, deduplicada y con banderas de prioridad/cuenta especial ({len(df_norm):,} facturas).",
                        tab=TAB_LABELS[2],
                    )
        except Exception as e:
            st.error(f"Error durante la carga o normalizacion: {e}")

    raw_preview_open = False
    df_fact_raw = ss.get("df_raw")
    if isinstance(df_fact_raw, pd.DataFrame):
        raw_preview_open = not df_fact_raw.empty
    with st.expander("Facturas cargadas (vista previa 100 filas)", expanded=raw_preview_open):
        _render_preview_table(
            df_fact_raw,
            integerize=True,
        )

    fact_preview_open = False
    if isinstance(ss.get("df"), pd.DataFrame):
        df_fact_preview = ss["df"]
        fact_preview_open = not df_fact_preview.empty
    with st.expander("Base normalizada (vista previa 100 filas)", expanded=fact_preview_open):
        _render_preview_table(
            ss.get("df"),
            integerize=True,
        )

    st.markdown(
        """
        <style>
        .cm-coachmark { position: relative; }
        .cm-coachmark .cm-toggle { position: absolute; opacity: 0; pointer-events: none; }
        .cm-coachmark .cm-fab {
            position: fixed;
            right: 1.5rem;
            bottom: 1.5rem;
            background: linear-gradient(135deg, #1f3c88, #2d77ff);
            color: #ffffff;
            border-radius: 999px;
            padding: 0.9rem 1.6rem;
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            box-shadow: 0 12px 24px rgba(31, 60, 136, 0.25);
            cursor: pointer;
            z-index: 1000;
            border: none;
        }
        .cm-coachmark .cm-fab:hover { box-shadow: 0 16px 28px rgba(31, 60, 136, 0.35); }
        .cm-coachmark .cm-fab:focus-visible {
            outline: 3px solid rgba(45, 119, 255, 0.6);
            outline-offset: 4px;
        }
        .cm-coachmark .cm-fab__icon {
            width: 2rem;
            height: 2rem;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 1.15rem;
            font-weight: 700;
        }
        .cm-coachmark .cm-fab__label { font-size: 0.95rem; }
        .cm-coachmark .cm-backdrop {
            position: fixed;
            inset: 0;
            background: rgba(10, 16, 28, 0.55);
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.25s ease;
            z-index: 999;
        }
        .cm-coachmark .cm-modal {
            position: fixed;
            inset: 50% auto auto 50%;
            transform: translate(-50%, calc(-50% + 30px));
            background: var(--background-color, #ffffff);
            color: inherit;
            width: min(520px, calc(100% - 32px));
            border-radius: 16px;
            box-shadow: 0 28px 60px rgba(12, 19, 38, 0.3);
            padding: 1.8rem 1.6rem 1.6rem;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.25s ease, transform 0.25s ease;
            z-index: 1001;
        }
        .cm-coachmark .cm-modal__header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .cm-coachmark .cm-modal__title {
            font-size: 1.3rem;
            margin: 0;
        }
        .cm-coachmark .cm-close {
            cursor: pointer;
            border: 1px solid rgba(47, 72, 133, 0.2);
            border-radius: 999px;
            padding: 0.35rem 0.9rem;
            font-size: 0.9rem;
            font-weight: 600;
            background: transparent;
            color: inherit;
        }
        .cm-coachmark .cm-close:hover {
            background: rgba(47, 72, 133, 0.08);
        }
        .cm-coachmark .cm-section { margin-bottom: 1.2rem; }
        .cm-coachmark .cm-section h3 {
            font-size: 1rem;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .cm-coachmark .cm-summary {
            margin: 0;
            font-size: 0.98rem;
            line-height: 1.5;
        }
        .cm-coachmark .cm-steps {
            margin: 0;
            padding-left: 1.1rem;
        }
        .cm-coachmark .cm-steps li {
            margin-bottom: 0.55rem;
            line-height: 1.45;
        }
        .cm-coachmark .cm-step-final {
            border: 1px dashed rgba(45, 119, 255, 0.6);
            border-radius: 12px;
            padding: 0.75rem 0.85rem;
            background: rgba(45, 119, 255, 0.08);
            position: relative;
            list-style-position: inside;
        }
        .cm-coachmark .cm-step-badge {
            position: absolute;
            top: -0.85rem;
            right: 0.75rem;
            background: #2d77ff;
            color: #ffffff;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.04em;
        }
        .cm-coachmark .cm-route {
            margin: 0;
            padding: 0.8rem;
            border-radius: 12px;
            background: rgba(47, 72, 133, 0.08);
            font-size: 0.92rem;
            line-height: 1.4;
        }
        .cm-coachmark .cm-note {
            margin: 0;
            font-size: 0.88rem;
            color: rgba(12, 19, 38, 0.78);
        }
        .cm-coachmark .cm-toggle:checked ~ .cm-backdrop {
            opacity: 1;
            pointer-events: auto;
        }
        .cm-coachmark .cm-toggle:checked ~ .cm-modal {
            opacity: 1;
            pointer-events: auto;
            transform: translate(-50%, -50%);
        }
        @media (max-width: 600px) {
            .cm-coachmark .cm-fab {
                right: 1rem;
                bottom: 1rem;
                padding: 0.75rem 1.2rem;
                font-size: 0.9rem;
            }
            .cm-coachmark .cm-modal {
                width: min(420px, calc(100% - 24px));
                padding: 1.5rem 1.2rem 1.2rem;
            }
            .cm-coachmark .cm-fab__icon { width: 1.75rem; height: 1.75rem; }
        }
        @media (prefers-color-scheme: dark) {
            .cm-coachmark .cm-fab {
                background: linear-gradient(135deg, #5a7aff, #91b4ff);
                box-shadow: 0 12px 24px rgba(145, 180, 255, 0.35);
            }
            .cm-coachmark .cm-fab:hover {
                box-shadow: 0 16px 30px rgba(145, 180, 255, 0.45);
            }
            .cm-coachmark .cm-modal {
                background: #0e1629;
                color: #e4ecff;
            }
            .cm-coachmark .cm-close {
                border-color: rgba(145, 180, 255, 0.4);
                color: #e4ecff;
            }
            .cm-coachmark .cm-step-final {
                background: rgba(145, 180, 255, 0.15);
                border-color: rgba(145, 180, 255, 0.75);
            }
            .cm-coachmark .cm-route {
                background: rgba(145, 180, 255, 0.12);
            }
            .cm-coachmark .cm-note { color: rgba(228, 236, 255, 0.76); }
        }
        </style>
        <div class="cm-coachmark">
            <input type="checkbox" id="cm-toggle" class="cm-toggle" />
            <label for="cm-toggle" class="cm-fab" role="button" aria-haspopup="dialog" aria-controls="cm-modal">
                <span class="cm-fab__icon">?</span>
                <span class="cm-fab__label">¿Dónde descargo?</span>
            </label>
            <label for="cm-toggle" class="cm-backdrop" aria-hidden="true"></label>
            <div class="cm-modal" id="cm-modal" role="dialog" aria-modal="true" aria-labelledby="cm-modal-title">
                <div class="cm-modal__header">
                    <h2 class="cm-modal__title" id="cm-modal-title">Guía de descarga</h2>
                    <label for="cm-toggle" class="cm-close" role="button">Cerrar</label>
                </div>
                <div class="cm-modal__body">
                    <div class="cm-section">
                        <h3>Resumen</h3>
                        <p class="cm-summary">Para descargar información de facturación, sigue esta ruta en el sistema de Contabilidad.</p>
                    </div>
                    <div class="cm-section">
                        <h3>Pasos</h3>
                        <ol class="cm-steps">
                            <li>Abre Contabilidad.</li>
                            <li>En la barra superior, entra a Proveedores.</li>
                            <li>Selecciona Facturación Proveedores.</li>
                            <li>Elige Consultas Facturación Proveedor.</li>
                            <li>Dentro de Proveedores, entra en Facturas Proveedores.</li>
                            <li>Ingresa a Módulo Facturación Proveedores.</li>
                            <li class="cm-step-final">Haz clic en Consulta General Facturación (aquí realizas la consulta y descarga).<span class="cm-step-badge">Aquí descargas</span></li>
                        </ol>
                    </div>
                    <div class="cm-section">
                        <h3>Árbol</h3>
                        <p class="cm-route">Contabilidad ▸ Proveedores ▸ Facturación Proveedores ▸ Consultas Facturación Proveedor ▸ Proveedores ▸ Facturas Proveedores ▸ Módulo Facturación Proveedores ▸ Consulta General Facturación</p>
                    </div>
                    <p class="cm-note">Tip: guarda esta ruta. Es el ingreso estándar para consultar y descargar facturación de proveedores.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_hon:
    st.markdown("#### 4. Honorarios (opcional)")
    safe_markdown(
        """
        <div class="app-note">
            Se descartan filas sin <code>estado_cuota</code> y se normalizan los campos clave antes de guardar.
        </div>
        """,
    )
    honorarios_file = st.file_uploader(
        "Sube archivo de honorarios (xlsx/csv)",
        type=["xlsx", "xls", "csv"],
        key="upload_honorarios",
    )

    hon_actual = ss.get("honorarios")
    if isinstance(hon_actual, pd.DataFrame) and not hon_actual.empty:
        st.caption(f"Honorarios cargados actualmente: {len(hon_actual):,} filas normalizadas.")

    if honorarios_file is not None:
        _set_active_tab(TAB_LABELS[3])
        try:
            df_hon_raw = read_any(honorarios_file)
            df_hon_norm = load_honorarios(df_hon_raw)
            if df_hon_norm is None or df_hon_norm.empty:
                st.warning("No se encontraron filas validas en el archivo de honorarios.")
            else:
                df_hon_clean = clean_estado_cuota(df_hon_raw)
                total_hon = len(df_hon_raw)
                valid_hon = len(df_hon_clean) if isinstance(df_hon_clean, pd.DataFrame) else len(df_hon_norm)
                dropped_hon = max(total_hon - valid_hon, 0)
                filename = getattr(honorarios_file, "name", "archivo")
                message = f"Honorarios cargados: {len(df_hon_norm):,} filas normalizadas desde {filename}."
                if dropped_hon:
                    message += f" Se descartaron {dropped_hon:,} filas sin estado_cuota valido."
                st.success(message)
                hon_actual = df_hon_norm
                ss["honorarios"] = df_hon_norm
                ss["honorarios_enriquecido"] = None
                ss["honorarios_summary"] = build_honorarios_summary(total_override=len(df_hon_norm))
        except Exception as e:
            st.error(f"No se pudo leer el archivo de honorarios: {e}")

    hon_actual = ss.get("honorarios")
    honorarios_cargados = isinstance(hon_actual, pd.DataFrame) and not getattr(hon_actual, "empty", True)
    ss["_honorarios_cargados"] = honorarios_cargados
    cuentas_cargadas = isinstance(ss.get("df_ctaes_raw"), pd.DataFrame) and not ss.get("df_ctaes_raw").empty
    hon_enriquecido = ss.get("honorarios_enriquecido")
    match_disponible = isinstance(hon_enriquecido, pd.DataFrame) and not getattr(hon_enriquecido, "empty", True)

    button_slot = st.empty()
    helper_slot = st.empty()

    match_hon_btn = False
    if not match_disponible:
        if honorarios_cargados:
            match_ready = cuentas_cargadas
            match_hon_btn = button_slot.button("Match con cuentas especiales", disabled=not match_ready)
            helper_slot.empty()
            if not cuentas_cargadas:
                helper_slot.caption("Necesitas cargar la maestra de cuentas para habilitar el match.")
        else:
            helper_slot.caption("Carga honorarios para habilitar el match con cuentas especiales.")
    else:
        button_slot.empty()
        helper_slot.empty()

    if match_hon_btn:
        df_hon = ss.get("honorarios")
        df_bancos = ss.get("df_ctaes_raw")
        if df_hon is None or (isinstance(df_hon, pd.DataFrame) and df_hon.empty):
            st.error("Carga honorarios antes de ejecutar el match.")
        elif df_bancos is None or (isinstance(df_bancos, pd.DataFrame) and df_bancos.empty):
            st.error("Debes cargar la maestra de cuentas especiales antes de cruzar honorarios.")
        else:
            try:
                df_hon_enr = merge_honorarios_con_bancos(df_hon, df_bancos)
                if df_hon_enr is None or df_hon_enr.empty:
                    ss["honorarios_enriquecido"] = None
                    st.warning("No se genero informacion enriquecida para honorarios.")
                else:
                    ss["honorarios_enriquecido"] = df_hon_enr
                    hon_enriquecido = df_hon_enr
                    hon_cols = set(df_hon.columns)
                    new_cols = [c for c in df_hon_enr.columns if c not in hon_cols]
                    match_col = next((c for c in ["codigo_contable", "banco", "cuenta_corriente", "sede_pago"] if c in new_cols), None)
                    if match_col:
                        matches = df_hon_enr[match_col].notna().sum()
                        ss["honorarios_summary"] = build_honorarios_summary(df_hon_enr)
                        message = f"Match completado: {matches:,} honorarios con datos bancarios de {len(df_hon_enr):,} filas."
                        _flash_and_rerun("success", message, tab=TAB_LABELS[3])
                    else:
                        st.warning("Match ejecutado pero no se detectaron columnas nuevas desde la maestra. Verifica la llave centro_costo_costeo / codigo_contable.")
            except Exception as e:
                st.error(f"No se pudo realizar el match con la maestra de bancos: {e}")

    hon_enriquecido = ss.get("honorarios_enriquecido")
    match_disponible = isinstance(hon_enriquecido, pd.DataFrame) and not getattr(hon_enriquecido, "empty", True)
    if match_disponible:
        ss["honorarios_summary"] = build_honorarios_summary(hon_enriquecido)
    elif not honorarios_cargados:
        ss["honorarios_summary"] = None

    hon_preview_open = False
    df_hon_norm = ss.get("honorarios")
    if isinstance(df_hon_norm, pd.DataFrame):
        hon_preview_open = not df_hon_norm.empty
    with st.expander("Honorarios normalizados (vista previa 100 filas)", expanded=hon_preview_open):
        _render_preview_table(ss.get("honorarios"))

safe_markdown('<div class="app-separator"></div>')

df = get_df_norm()
if df is None:
    df_cached = ss.get("df_cache")
    if isinstance(df_cached, pd.DataFrame) and not df_cached.empty:
        df = df_cached
        safe_markdown(
            """
            <div class="app-note">
                Mostrando el ultimo resumen calculado (sin cambios nuevos).
            </div>
            """,
        )
    else:
        safe_markdown(
            """
            <div class="app-note">
                Aun no hay base normalizada. Carga y mapea tus documentos para revisar el resumen.
            </div>
            """,
        )
        st.stop()
else:
    ss["df_cache"] = df
    ss["_match_timestamp_view"] = ss.get("_match_timestamp")

safe_markdown('<div class="app-separator"></div>')

safe_markdown(
    """
    <div style="text-align:center;font-size:0.85rem;color:var(--app-text-muted);padding:1.2rem 0 2rem;">
        Desarrollado por <strong>Danilo Parada Ulloa</strong>
    </div>
    """,
)

