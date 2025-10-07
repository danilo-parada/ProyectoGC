from __future__ import annotations
import pandas as pd
import streamlit as st

from lib_common import (
    # setup / UI
    init_session_keys, read_any, style_table, header_ui,
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
header_ui("Carga de Data", current_page="Inicio",
          subtitle="Primero carga la Maestra de Cuentas de Banco y luego tus documentos.")

# ---------------------------------------------------------------------
# Estado inicial
# ---------------------------------------------------------------------
init_session_keys()
ss = st.session_state

def _flash_and_rerun(level: str, message: str) -> None:
    """Persist a flash message and trigger UI refresh."""
    ss["_ui_flash"] = (level, message)
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def _count_rows(obj) -> int:
    if isinstance(obj, pd.DataFrame):
        return len(obj)
    return 0


def _card_html(
    title: str,
    rows: int,
    ready: bool,
    hint: str,
    extras: list[str] | None = None,
    actions: list[str] | None = None,
) -> str:
    status = "Listo" if ready else "Pendiente"
    color = "#42d6a4" if ready else "#f7b955"
    value = f"{rows:,}" if rows else "0"
    extras_html = "".join(extras) if extras else ""
    actions_html = "".join(actions) if actions else ""
    return (
        '<div class="app-card">'
        f'{actions_html}'
        f'<div class="app-card__title">{title}</div>'
        f'<div class="app-card__value">{value}</div>'
        f'<div class="app-card__trend" style="color:{color};"><span>&bull; {status}</span></div>'
        f"{extras_html}"
        f'<p style="margin-top:0.45rem;color:var(--app-text-muted);font-size:0.82rem;">{hint}</p>'
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

def build_honorarios_summary(df_hon_enr: pd.DataFrame) -> dict[str, float]:
    """Calcula totales y porcentaje de match bancario para honorarios."""
    if not isinstance(df_hon_enr, pd.DataFrame) or df_hon_enr.empty:
        return {}
    columnas_match = [c for c in ["codigo_contable", "banco", "cuenta_corriente"] if c in df_hon_enr.columns]
    matched = int(df_hon_enr[columnas_match].notna().all(axis=1).sum()) if columnas_match else 0
    total = len(df_hon_enr)
    pct = (matched / total * 100) if total else 0.0
    return {"total": total, "matched": matched, "pct": pct, "no_match": max(total - matched, 0)}

def build_facturas_summary(match_stats: dict) -> dict[str, object]:
    """Resume porcentajes y cuentas para facturas normalizadas."""
    if not isinstance(match_stats, dict) or not match_stats:
        return {}
    total = match_stats.get("total", 0)
    cuenta = match_stats.get("cuenta_especial", {}) or {}
    prov = match_stats.get("prov_prioritario", {}) or {}
    return {"total": total, "cuenta": cuenta, "prov": prov}

hon_summary = ss.get("honorarios_summary")
hon_enr = ss.get("honorarios_enriquecido")
if not hon_summary and isinstance(hon_enr, pd.DataFrame) and not hon_enr.empty:
    hon_summary = build_honorarios_summary(hon_enr)
    ss["honorarios_summary"] = hon_summary

fact_summary = ss.get("facturas_summary")
match_stats = None
if doc_rows > 0:
    try:
        match_stats = get_match_summary()
    except Exception:
        match_stats = None
    if match_stats:
        fact_summary = build_facturas_summary(match_stats)
        ss["facturas_summary"] = fact_summary
elif fact_summary is None and ss.get("_match_summary"):
    fact_summary = build_facturas_summary(ss["_match_summary"])
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
card_hon_html = _card_html("Honorarios", hon_rows, hon_rows > 0, hon_hint, extras=hon_extras)
card_fact_html = _card_html("Facturas normalizadas", doc_rows, doc_rows > 0, fact_hint, extras=fact_extras)

col_cta, col_prov, col_hon, col_fact = st.columns(4)

with col_cta:
    if st.button("Reset cuentas", key="card_reset_cta", use_container_width=True):
        reset_cuentas_especiales()
        _flash_and_rerun("warning", "Maestra de cuentas especiales eliminada.")
    st.markdown(card_cta_html, unsafe_allow_html=True)

with col_prov:
    if st.button("Reset proveedores", key="card_reset_prov", use_container_width=True):
        reset_proveedores()
        _flash_and_rerun("warning", "Proveedores prioritarios eliminados.")
    st.markdown(card_prov_html, unsafe_allow_html=True)

with col_hon:
    if st.button("Reset honorarios", key="card_reset_hon", use_container_width=True):
        reset_honorarios()
        _flash_and_rerun("warning", "Honorarios eliminados de la sesion.")
    st.markdown(card_hon_html, unsafe_allow_html=True)

with col_fact:
    if st.button("Reset facturas", key="card_reset_fact", use_container_width=True):
        reset_docs()
        _flash_and_rerun("warning", "Facturas eliminadas de la sesion actual.")
    st.markdown(card_fact_html, unsafe_allow_html=True)

flash_payload = ss.pop("_ui_flash", None)
if flash_payload:
    level, message = flash_payload
    notifier = getattr(st, level, st.info)
    notifier(message)

tab_cta, tab_prov, tab_hon, tab_fact = st.tabs([
    "1. Cuentas especiales",
    "2. Proveedores prioritarios",
    "3. Honorarios",
    "4. Facturas",
])

with tab_cta:
    st.markdown("#### 1. Maestra de cuentas (obligatorio)")
    st.markdown(
        """
        <div class="app-note">
            Archivo requerido con columnas: <code>ano_proyecto</code>, <code>codigo_contable</code>,
            <code>proyecto</code>, <code>banco</code>, <code>cuenta_corriente</code>, <code>cuenta_contable</code>,
            <code>cuenta_cont_descripcion</code>, <code>sede_pago</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    maestra_file = st.file_uploader(
        "Sube la maestra de cuentas (xlsx/csv)",
        type=["xlsx", "xls", "csv"],
        key="upload_ctaes",
    )
    if maestra_file is not None:
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
    with st.expander("Vista previa maestra (200 filas)", expanded=False):
        if isinstance(ss.get("df_ctaes_raw"), pd.DataFrame):
            style_table(ss["df_ctaes_raw"].head(200))

with tab_prov:
    st.markdown("#### 2. Proveedores prioritarios (opcional)")
    st.markdown(
        """
        <div class="app-note">
            Archivo con columna <code>codigo_proveedor</code> (puede incluir otros campos).
        </div>
        """,
        unsafe_allow_html=True,
    )
    prov_file = st.file_uploader(
        "Sube listado de proveedores prioritarios (xlsx/csv)",
        type=["xlsx", "xls", "csv"],
        key="upload_prio",
    )
    if prov_file is not None:
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
    with st.expander("Vista previa proveedores prioritarios", expanded=False):
        if isinstance(ss.get("df_prio_raw"), pd.DataFrame):
            style_table(ss["df_prio_raw"].head(200))

with tab_hon:
    st.markdown("#### 3. Honorarios (opcional)")
    st.markdown(
        """
        <div class="app-note">
            Se descartan filas sin <code>estado_cuota</code> y se normalizan los campos clave antes de guardar.
        </div>
        """,
        unsafe_allow_html=True,
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
                ss["honorarios_summary"] = None
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
                        _flash_and_rerun("success", message)
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

    with st.expander("Honorarios normalizados (vista previa)", expanded=False):
        df_hon_norm = ss.get("honorarios")
        if isinstance(df_hon_norm, pd.DataFrame) and not df_hon_norm.empty:
            style_table(df_hon_norm.head(200))

with tab_fact:
    st.markdown("#### 4. Facturas")
    st.markdown(
        """
        <div class="app-note">
            Carga uno o varios archivos y luego mapea columnas para habilitar todos los tableros.
        </div>
        """,
        unsafe_allow_html=True,
    )

    files = st.file_uploader(
        "Sube facturas (xlsx/csv)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        key="upload_docs",
    )

    if files:
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
                if not ok:
                    st.error("Falta asignar fac_fecha_factura en el mapeo. Corrige y vuelve a intentar.")
                else:
                    df_norm = apply_mapping(df_raw, col_map)
                    register_documents(df_norm, dedup=dedup_flag)
                    ss["_facturas_count"] = len(df_norm)
                    try:
                        ss["facturas_summary"] = build_facturas_summary(get_match_summary())
                    except Exception:
                        pass
                    _flash_and_rerun(
                        "success",
                        f"Base normalizada, deduplicada y con banderas de prioridad/cuenta especial ({len(df_norm):,} facturas).",
                    )
        except Exception as e:
            st.error(f"Error durante la carga o normalizacion: {e}")

    with st.expander("Base normalizada (vista previa 200 filas)", expanded=False):
        if isinstance(ss.get("df"), pd.DataFrame):
            style_table(ss["df"].head(200))

st.markdown('<div class="app-separator"></div>', unsafe_allow_html=True)

df = get_df_norm()
if df is None:
    df_cached = ss.get("df_cache")
    if isinstance(df_cached, pd.DataFrame) and not df_cached.empty:
        df = df_cached
        st.markdown(
            """
            <div class="app-note">
                Mostrando el ultimo resumen calculado (sin cambios nuevos).
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="app-note">
                Aun no hay base normalizada. Carga y mapea tus documentos para revisar el resumen.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()
else:
    ss["df_cache"] = df
    ss["_match_timestamp_view"] = ss.get("_match_timestamp")

st.markdown('<div class="app-separator"></div>', unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center;font-size:0.85rem;color:var(--app-text-muted);padding:1.2rem 0 2rem;">
        Desarrollado por <strong>Danilo Parada Ulloa</strong>
    </div>
    """,
    unsafe_allow_html=True,
)

