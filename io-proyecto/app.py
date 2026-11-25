"""Solver IO II: transporte lineal y rutas más cortas con Floyd-Warshall."""

from __future__ import annotations

import math
import sys
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pulp
import streamlit as st


st.set_page_config(page_title="Solver IO II", page_icon="📦", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2.6rem; padding-bottom: 3rem; max-width: 1550px}
  [data-testid="stMetric"] {background:#172b3a!important;border:1px solid #315064;padding:16px;border-radius:14px}
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {color:#bfd3df!important;font-weight:600}
  [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {color:#fff!important;font-weight:700}
  [data-testid="stMetricDelta"] {color:#9de2c4!important}
  [data-testid="stAlert"] {border-radius:12px}
  div[data-testid="stDataFrame"] {border:1px solid #334b5d;border-radius:12px;overflow:hidden}
  .hero {padding:1.25rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,#102a43,#176b87);color:white;margin-bottom:1rem}
  .hero p {color:#d9eef5;margin-bottom:0}
  @media (max-width:800px) {.block-container{padding-left:1rem;padding-right:1rem}.hero h1{font-size:2rem}}
</style>
<div class="hero"><h1>📦 Solver IO II</h1><p>Investigación Operativa · optimización, redes y análisis paso a paso</p></div>
""", unsafe_allow_html=True)


# ───────────────────────── Utilidades del modelo de transporte ─────────────────────────
def transport_defaults(m: int, n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    origins = pd.DataFrame({"Origen": [f"Planta {i+1}" for i in range(m)], "Oferta": [100.0] * m})
    destinations = pd.DataFrame({"Destino": [f"Cliente {j+1}" for j in range(n)], "Demanda": [100.0] * n})
    costs = pd.DataFrame([[float((i + j) % 7 + 1) for j in range(n)] for i in range(m)],
                         index=origins["Origen"], columns=destinations["Destino"])
    return origins, destinations, costs


def make_case(case_name: str) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {
        "Caso Cañete Norte (Base 2x3)": {
            "supply": [45, 55], "demand": [25, 40, 35], "cost": [[7, 4, 9], [6, 8, 5]],
            "origin_names": ["Planta 1", "Planta 2"], "destination_names": ["Cliente 1", "Cliente 2", "Cliente 3"],
            "restricted": False,
        },
        "Caso Cañete Norte Restringido": {
            "supply": [45, 55], "demand": [25, 40, 35], "cost": [[7, 4, 9], [6, 8, 5]],
            "origin_names": ["Planta 1", "Planta 2"], "destination_names": ["Cliente 1", "Cliente 2", "Cliente 3"],
            "restricted": True,
        },
        "Caso Desbalanceado": {
            "supply": [120, 80], "demand": [70, 90, 60], "cost": [[4, 8, 6], [7, 5, 9]],
            "origin_names": ["Planta 1", "Planta 2"], "destination_names": ["Cliente 1", "Cliente 2", "Cliente 3"],
            "restricted": False,
        },
        "Comparativa Heurísticas (3x3)": {
            "supply": [70, 90, 140], "demand": [80, 100, 120],
            "cost": [[10, 2, 20], [12, 7, 9], [4, 14, 16]],
            "origin_names": ["Planta 1", "Planta 2", "Planta 3"],
            "destination_names": ["Cliente 1", "Cliente 2", "Cliente 3"], "restricted": False,
        },
    }
    case = cases[case_name]
    m, n = len(case["supply"]), len(case["demand"])
    origins = pd.DataFrame({"Origen": case["origin_names"], "Oferta": case["supply"]})
    destinations = pd.DataFrame({"Destino": case["destination_names"], "Demanda": case["demand"]})
    costs = pd.DataFrame(case["cost"], index=origins["Origen"], columns=destinations["Destino"])
    caps = pd.DataFrame(np.nan, index=origins["Origen"], columns=destinations["Destino"])
    allowed = pd.DataFrame(True, index=origins["Origen"], columns=destinations["Destino"])
    if case["restricted"]:
        caps.iloc[1, 2] = 25.0  # x_23 ≤ 25
    return {"m": m, "n": n, "origins": origins, "destinations": destinations,
            "costs": costs, "caps": caps, "allowed": allowed, "restricted": case["restricted"]}


def resize_transport_state(m: int, n: int) -> None:
    """Conserva filas/columnas existentes y agrega datos de ejemplo al ampliar."""
    default_o, default_d, default_c = transport_defaults(m, n)
    old_o = st.session_state.get("transport_origins")
    old_d = st.session_state.get("transport_destinations")
    old_c = st.session_state.get("transport_costs")
    if old_o is None or len(old_o) != m:
        new_o = default_o.copy()
        if old_o is not None:
            for i in range(min(m, len(old_o))):
                new_o.iloc[i] = old_o.iloc[i].values
        st.session_state.transport_origins = new_o
    if old_d is None or len(old_d) != n:
        new_d = default_d.copy()
        if old_d is not None:
            for j in range(min(n, len(old_d))):
                new_d.iloc[j] = old_d.iloc[j].values
        st.session_state.transport_destinations = new_d
    if old_c is None or old_c.shape != (m, n):
        new_c = default_c.copy()
        if old_c is not None:
            for i in range(min(m, old_c.shape[0])):
                for j in range(min(n, old_c.shape[1])):
                    new_c.iloc[i, j] = old_c.iloc[i, j]
        st.session_state.transport_costs = new_c
    origins = st.session_state.transport_origins
    destinations = st.session_state.transport_destinations
    st.session_state.transport_costs.index = origins["Origen"].astype(str).tolist()
    st.session_state.transport_costs.columns = destinations["Destino"].astype(str).tolist()
    expected_index, expected_columns = origins["Origen"].astype(str), destinations["Destino"].astype(str)
    caps = st.session_state.get("transport_caps")
    allowed = st.session_state.get("transport_allowed")
    if caps is None or caps.shape != (m, n):
        caps = pd.DataFrame(np.nan, index=expected_index, columns=expected_columns)
    else:
        caps.index, caps.columns = expected_index, expected_columns
    if allowed is None or allowed.shape != (m, n):
        allowed = pd.DataFrame(True, index=expected_index, columns=expected_columns)
    else:
        allowed.index, allowed.columns = expected_index, expected_columns
    st.session_state.transport_caps = caps
    st.session_state.transport_allowed = allowed


def balance_transport(supply: np.ndarray, demand: np.ndarray, costs: np.ndarray,
                      caps: np.ndarray, allowed: np.ndarray, origin_names: list[str],
                      destination_names: list[str]):
    """Equilibra el modelo con nodo ficticio si las sumas difieren."""
    total_supply, total_demand = float(supply.sum()), float(demand.sum())
    if math.isclose(total_supply, total_demand, abs_tol=1e-8):
        return supply.copy(), demand.copy(), costs.copy(), caps.copy(), allowed.copy(), origin_names[:], destination_names[:], False
    if total_supply > total_demand:
        demand = np.append(demand, total_supply - total_demand)
        costs = np.column_stack((costs, np.zeros(len(supply))))
        caps = np.column_stack((caps, np.full(len(supply), np.inf)))
        allowed = np.column_stack((allowed, np.ones(len(supply), dtype=bool)))
        destination_names = destination_names + ["Destino ficticio (excedente)"]
    else:
        supply = np.append(supply, total_demand - total_supply)
        costs = np.vstack((costs, np.zeros(len(demand))))
        caps = np.vstack((caps, np.full(len(demand), np.inf)))
        allowed = np.vstack((allowed, np.ones(len(demand), dtype=bool)))
        origin_names = origin_names + ["Origen ficticio (faltante)"]
    return supply, demand, costs, caps, allowed, origin_names, destination_names, True


def solve_transport_lp(supply, demand, costs, caps, allowed):
    model = pulp.LpProblem("Transporte", pulp.LpMinimize)
    variables = {
        (i, j): pulp.LpVariable(
            f"x_{i+1}_{j+1}", lowBound=0,
            upBound=None if np.isinf(caps[i, j]) else float(caps[i, j]),
        )
        for i in range(len(supply)) for j in range(len(demand)) if allowed[i, j]
    }
    model += pulp.lpSum(costs[i, j] * variables[i, j] for i, j in variables), "Costo_total"
    supply_constraints, demand_constraints = {}, {}
    for i, amount in enumerate(supply):
        constraint = pulp.lpSum(variables[i, j] for j in range(len(demand)) if (i, j) in variables) == float(amount)
        name = f"Oferta_{i+1}"
        model += constraint, name
        supply_constraints[i] = name
    for j, amount in enumerate(demand):
        constraint = pulp.lpSum(variables[i, j] for i in range(len(supply)) if (i, j) in variables) == float(amount)
        name = f"Demanda_{j+1}"
        model += constraint, name
        demand_constraints[j] = name
    # PuLP 3.x todavía ofrece PULP_CBC_CMD, que localiza el CBC de su instalación
    # (incluido el que expone cbcbox). COIN_CMD solo busca CBC en PATH en PuLP 3.3.
    solver = pulp.PULP_CBC_CMD(msg=False)
    if not solver.available():
        raise SolverSetupError("No se encontró el ejecutable CBC que PuLP necesita para resolver. Instala el extra CBC con el comando indicado y reinicia Streamlit.")
    try:
        status_code = model.solve(solver)
    except Exception as exc:
        raise SolverSetupError(f"CBC no pudo completar la resolución. Detalle: {type(exc).__name__}: {exc}\n\nVuelve a instalar el solver en el entorno activo y reinicia Streamlit.") from exc
    status = pulp.LpStatus.get(status_code, str(status_code))
    if status != "Optimal":
        return status, None, None, None
    allocation = np.zeros((len(supply), len(demand)))
    for (i, j), variable in variables.items():
        allocation[i, j] = variable.value() or 0.0
    duals = {
        "supply": [model.constraints[supply_constraints[i]].pi for i in range(len(supply))],
        "demand": [model.constraints[demand_constraints[j]].pi for j in range(len(demand))],
    }
    return status, allocation, float(pulp.value(model.objective)), duals


def transport_heuristic(supply, demand, costs, caps, allowed, method: str):
    """Asigna cantidades de forma factible usando NW, costo mínimo o Vogel."""
    remaining_s = np.asarray(supply, dtype=float).copy()
    remaining_d = np.asarray(demand, dtype=float).copy()
    residual_caps = np.where(allowed, caps, 0.0).astype(float)
    residual_caps[~np.isfinite(residual_caps)] = np.inf
    allocation = np.zeros_like(costs, dtype=float)
    eps = 1e-8
    while remaining_s.sum() > eps:
        feasible = [(i, j) for i in range(len(remaining_s)) for j in range(len(remaining_d))
                    if remaining_s[i] > eps and remaining_d[j] > eps and residual_caps[i, j] > eps]
        if not feasible:
            return None
        if method == "Esquina Noroeste":
            i, j = min(feasible)
        elif method == "Costo Mínimo":
            i, j = min(feasible, key=lambda ij: (costs[ij], ij))
        else:
            scores = []
            for i0 in range(len(remaining_s)):
                js = [j0 for j0 in range(len(remaining_d)) if remaining_d[j0] > eps and residual_caps[i0, j0] > eps]
                if remaining_s[i0] > eps and js:
                    vals = sorted(costs[i0, j0] for j0 in js)
                    penalty = vals[1] - vals[0] if len(vals) > 1 else vals[0]
                    best_cost = min(costs[i0, j0] for j0 in js)
                    best_routes = [(i0, j0) for j0 in js if costs[i0, j0] == best_cost]
                    max_allocation = max(min(remaining_s[i0], remaining_d[j0], residual_caps[i0, j0]) for i0, j0 in best_routes)
                    scores.append((penalty, best_cost, max_allocation, [(i0, j0) for j0 in js]))
            for j0 in range(len(remaining_d)):
                ii = [i0 for i0 in range(len(remaining_s)) if remaining_s[i0] > eps and residual_caps[i0, j0] > eps]
                if remaining_d[j0] > eps and ii:
                    vals = sorted(costs[i0, j0] for i0 in ii)
                    penalty = vals[1] - vals[0] if len(vals) > 1 else vals[0]
                    best_cost = min(costs[i0, j0] for i0 in ii)
                    best_routes = [(i0, j0) for i0 in ii if costs[i0, j0] == best_cost]
                    max_allocation = max(min(remaining_s[i0], remaining_d[j0], residual_caps[i0, j0]) for i0, j0 in best_routes)
                    scores.append((penalty, best_cost, max_allocation, [(i0, j0) for i0 in ii]))
            if not scores:
                return None
            _, _, _, routes = max(scores, key=lambda row: (row[0], -row[1], row[2]))
            i, j = min(routes, key=lambda ij: (costs[ij], ij))
        quantity = min(remaining_s[i], remaining_d[j], residual_caps[i, j])
        if quantity <= eps:
            residual_caps[i, j] = 0.0
            continue
        allocation[i, j] += quantity
        remaining_s[i] -= quantity
        remaining_d[j] -= quantity
        residual_caps[i, j] -= quantity
    return allocation if np.allclose(remaining_d, 0, atol=1e-6) else None


def read_transport_inputs(origins, destinations, costs_df, caps_df, allowed_df):
    supply = pd.to_numeric(origins["Oferta"], errors="raise").to_numpy(dtype=float)
    demand = pd.to_numeric(destinations["Demanda"], errors="raise").to_numpy(dtype=float)
    costs = costs_df.to_numpy(dtype=float, copy=True)
    caps = caps_df.to_numpy(dtype=float, copy=True)
    allowed = allowed_df.to_numpy(dtype=bool, copy=True)
    if not (np.isfinite(supply).all() and np.isfinite(demand).all() and np.isfinite(costs).all()):
        raise ValueError("Oferta, demanda y costos deben ser valores finitos.")
    if np.any(supply < 0) or np.any(demand < 0) or np.any(costs < 0):
        raise ValueError("Oferta, demanda y costos no pueden ser negativos.")
    if np.any(caps[np.isfinite(caps)] < 0):
        raise ValueError("Las capacidades máximas no pueden ser negativas.")
    caps[np.isnan(caps)] = np.inf
    raw_o, raw_d = origins["Origen"], destinations["Destino"]
    if raw_o.isna().any() or raw_d.isna().any():
        raise ValueError("Todos los orígenes y destinos deben tener nombre.")
    names_o = raw_o.astype(str).str.strip().tolist()
    names_d = raw_d.astype(str).str.strip().tolist()
    if any(not name for name in names_o + names_d):
        raise ValueError("Todos los orígenes y destinos deben tener nombre.")
    if len(set(names_o)) != len(names_o) or len(set(names_d)) != len(names_d):
        raise ValueError("Los nombres de orígenes deben ser únicos entre sí, y lo mismo para los destinos.")
    allowed &= caps > 0
    return supply, demand, costs, caps, allowed, names_o, names_d


def render_transport() -> None:
    case_names = ["Caso Cañete Norte (Base 2x3)", "Caso Cañete Norte Restringido",
                  "Caso Desbalanceado", "Comparativa Heurísticas (3x3)"]
    # Cargar una configuración antes de instanciar sus widgets evita errores
    # de Streamlit al cambiar valores asociados a controles ya renderizados.
    queued_case = st.session_state.pop("queued_transport_case", None)
    if queued_case in case_names:
        data = make_case(queued_case)
        st.session_state.trans_m, st.session_state.trans_n = data["m"], data["n"]
        st.session_state.transport_origins = data["origins"]
        st.session_state.transport_destinations = data["destinations"]
        st.session_state.transport_costs = data["costs"]
        st.session_state.transport_caps = data["caps"]
        st.session_state.transport_allowed = data["allowed"]
        st.session_state.use_route_limits = data["restricted"]
        st.session_state.transport_editor_version = st.session_state.get("transport_editor_version", 0) + 1

    st.session_state.setdefault("trans_m", 3)
    st.session_state.setdefault("trans_n", 3)
    with st.sidebar:
        st.header("Configuración de transporte")
        m = st.number_input("Número de orígenes", min_value=2, max_value=8, step=1, key="trans_m")
        n = st.number_input("Número de destinos", min_value=2, max_value=8, step=1, key="trans_n")
        preset = st.selectbox("Caso precargado", ["— Selecciona un caso —"] + case_names, key="transport_case_select")
        if st.button("Cargar caso", use_container_width=True, disabled=preset.startswith("—")):
            st.session_state.queued_transport_case = preset
            st.rerun()
        st.caption("El caso seleccionado reemplazará las tablas actuales.")
    resize_transport_state(int(m), int(n))
    version = st.session_state.get("transport_editor_version", 0)

    data_tab, solution_tab, network_tab, guide_tab = st.tabs(["🧾 Datos", "📊 Solución", "🕸️ Red", "📘 Guía"])
    with data_tab:
        col_o, col_d = st.columns(2)
        with col_o:
            st.subheader("Orígenes y oferta")
            origins = st.data_editor(
                st.session_state.transport_origins, key=f"transport_origins_{m}_{version}",
                hide_index=True, use_container_width=True, num_rows="fixed",
                column_config={"Oferta": st.column_config.NumberColumn(min_value=0, step=1, format="%.2f")},
            )
            st.session_state.transport_origins = origins
            st.subheader("Costos unitarios")
            costs_df = st.data_editor(
                st.session_state.transport_costs, key=f"transport_costs_{m}_{n}_{version}",
                use_container_width=True,
                column_config={name: st.column_config.NumberColumn(min_value=0, step=1, format="%.2f")
                               for name in st.session_state.transport_costs.columns},
            )
            st.session_state.transport_costs = costs_df
        with col_d:
            st.subheader("Destinos y demanda")
            destinations = st.data_editor(
                st.session_state.transport_destinations, key=f"transport_destinations_{n}_{version}",
                hide_index=True, use_container_width=True, num_rows="fixed",
                column_config={"Demanda": st.column_config.NumberColumn(min_value=0, step=1, format="%.2f")},
            )
            st.session_state.transport_destinations = destinations
            st.subheader("Restricciones por ruta")
            use_limits = st.toggle("Definir capacidades o bloquear rutas", key="use_route_limits")
            if use_limits:
                st.caption("Límite vacío = sin cota superior. Desmarca una casilla para bloquear esa ruta.")
                route_caps = st.session_state.transport_caps.copy()
                route_caps.index = origins["Origen"].astype(str).tolist()
                route_caps.columns = destinations["Destino"].astype(str).tolist()
                route_caps = route_caps.replace([np.inf, -np.inf], np.nan)
                route_allowed = st.session_state.transport_allowed.copy()
                route_allowed.index = origins["Origen"].astype(str).tolist()
                route_allowed.columns = destinations["Destino"].astype(str).tolist()
                cc, cb = st.columns(2)
                with cc:
                    st.markdown("**Capacidad máxima**")
                    caps_edit = st.data_editor(route_caps, key=f"transport_caps_{m}_{n}_{version}", use_container_width=True,
                                               column_config={name: st.column_config.NumberColumn(min_value=0, format="%.2f") for name in route_caps.columns})
                with cb:
                    st.markdown("**Ruta habilitada**")
                    allowed_edit = st.data_editor(route_allowed, key=f"transport_allowed_{m}_{n}_{version}", use_container_width=True,
                                                  column_config={name: st.column_config.CheckboxColumn() for name in route_allowed.columns})
                st.session_state.transport_caps = caps_edit
                st.session_state.transport_allowed = allowed_edit
            else:
                caps_edit = pd.DataFrame(np.inf, index=range(m), columns=range(n))
                allowed_edit = pd.DataFrame(True, index=range(m), columns=range(n))
        run_transport = st.button("🚀 Resolver y comparar métodos", type="primary", use_container_width=True)

    if run_transport:
        st.session_state.pop("transport_result", None)
        try:
            supply, demand, cost, caps, allowed, names_o, names_d = read_transport_inputs(
                origins, destinations, costs_df, caps_edit, allowed_edit)
            s, d, c, u, a, names_o, names_d, has_dummy = balance_transport(
                supply, demand, cost, caps, allowed, names_o, names_d)
            status, optimal, objective, duals = solve_transport_lp(s, d, c, u, a)
            methods = ["Esquina Noroeste", "Costo Mínimo", "Aproximación de Vogel"]
            heuristic_results = {method: transport_heuristic(s, d, c, u, a, method) for method in methods}
            st.session_state.transport_result = {
                "status": status, "optimal": optimal, "objective": objective, "duals": duals,
                "heuristics": heuristic_results, "costs": c, "supply": s, "demand": d,
                "caps": u, "allowed": a, "origins": names_o, "destinations": names_d,
                "dummy": has_dummy,
            }
        except SolverSetupError as exc:
            st.error(str(exc))
            st.code(cbc_install_command(), language="powershell")
            st.caption("Ejecuta el comando en PowerShell. Después detén y vuelve a iniciar Streamlit desde el mismo entorno Python.")
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Revisa los datos: {exc}")
        except Exception as exc:
            st.error(f"No se pudo resolver el modelo ({type(exc).__name__}). Revisa los datos y dependencias del solver.")

    result = st.session_state.get("transport_result")
    with solution_tab:
        if not result:
            st.info("Completa los datos y pulsa **Resolver y comparar métodos**.")
        elif result["status"] != "Optimal":
            st.error(f"El modelo no es factible o el solver no encontró óptimo. Estado: **{result['status']}**. Revisa la oferta, demanda, rutas bloqueadas y capacidades.")
        else:
            optimal = result["optimal"]
            opt_cost = result["objective"]
            x_count = int(np.count_nonzero(optimal > 1e-7))
            required_basic = optimal.shape[0] + optimal.shape[1] - 1
            met1, met2, met3 = st.columns(3)
            met1.metric("Óptimo global (PuLP)", f"{opt_cost:,.2f}")
            met2.metric("Unidades transportadas", f"{optimal.sum():,.2f}")
            met3.metric("Rutas activas", f"{x_count} / {required_basic} básicas")
            if x_count < required_basic:
                st.warning(f"⚠️ Solución Óptima Degenerada: hay {x_count} rutas con envío positivo; una base de transporte requiere {required_basic}. En una solución degenerada, algunas variables básicas pueden valer cero.")
            else:
                st.success("La solución tiene al menos m + n − 1 rutas activas.")
            if result["dummy"]:
                st.info("Se agregó un nodo ficticio de costo cero para equilibrar oferta y demanda. Su asignación representa excedente o faltante.")

            costs_by_method: dict[str, float | None] = {
                name: (float(np.sum(matrix * result["costs"])) if matrix is not None else None)
                for name, matrix in result["heuristics"].items()
            }
            costs_by_method["Óptimo Global con PuLP"] = opt_cost
            nw_cost = costs_by_method["Esquina Noroeste"]
            comparison_rows = []
            for method in ["Esquina Noroeste", "Costo Mínimo", "Aproximación de Vogel", "Óptimo Global con PuLP"]:
                value = costs_by_method[method]
                saving = (nw_cost - value) if nw_cost is not None and value is not None else np.nan
                gap = ((value - opt_cost) / abs(opt_cost) * 100) if value is not None and abs(opt_cost) > 1e-12 else (0.0 if value == opt_cost else np.nan)
                comparison_rows.append({"Método": method, "Costo Total (Z)": value,
                                        "Ahorro vs. Noroeste": saving, "Brecha vs. óptimo (%)": gap})
            st.subheader("Comparativa simultánea")
            st.dataframe(pd.DataFrame(comparison_rows).style.format({
                "Costo Total (Z)": "{:,.2f}", "Ahorro vs. Noroeste": "{:,.2f}",
                "Brecha vs. óptimo (%)": "{:.2f}%",
            }, na_rep="No factible").highlight_min(subset=["Costo Total (Z)"], color="#235b47"),
                hide_index=True, use_container_width=True)
            inspect_method = st.selectbox("Inspeccionar asignaciones de", list(result["heuristics"]), key="inspect_transport_method")
            selected = result["heuristics"][inspect_method]
            if selected is None:
                st.warning("Esta heurística no pudo construir un plan factible con las restricciones definidas.")
            else:
                st.caption(f"Matriz de asignación de **{inspect_method}** · costo: {costs_by_method[inspect_method]:,.2f}")
                st.dataframe(pd.DataFrame(selected, index=result["origins"], columns=result["destinations"]).style.format("{:,.2f}").background_gradient(cmap="Blues", axis=None), use_container_width=True)
            st.subheader("Plan óptimo de transporte")
            optimal_df = pd.DataFrame(optimal, index=result["origins"], columns=result["destinations"])
            st.dataframe(optimal_df.style.format("{:,.2f}").background_gradient(cmap="GnBu", axis=None), use_container_width=True)
            details = [{"Origen": result["origins"][i], "Destino": result["destinations"][j],
                        "Unidades": optimal[i, j], "Costo unitario": result["costs"][i, j],
                        "Costo de ruta": optimal[i, j] * result["costs"][i, j]}
                       for i in range(optimal.shape[0]) for j in range(optimal.shape[1]) if optimal[i, j] > 1e-7]
            with st.expander("Ver detalle de rutas y descargar"):
                st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)
                st.download_button("Descargar plan óptimo CSV", optimal_df.to_csv().encode("utf-8-sig"), "plan_transporte.csv", "text/csv")
            st.subheader("Análisis de sensibilidad: precios sombra")
            dual_rows = ([{"Restricción": f"Oferta · {result['origins'][i]}", "Precio sombra (π)": pi}
                          for i, pi in enumerate(result["duals"]["supply"])] +
                         [{"Restricción": f"Demanda · {result['destinations'][j]}", "Precio sombra (π)": pi}
                          for j, pi in enumerate(result["duals"]["demand"])])
            st.dataframe(pd.DataFrame(dual_rows).style.format({"Precio sombra (π)": "{:,.4f}"}), hide_index=True, use_container_width=True)
            st.caption("El precio sombra expresa el cambio marginal del costo óptimo ante una unidad adicional en el lado derecho de la restricción, mientras se mantenga la base óptima actual.")

    with network_tab:
        if not result or result["status"] != "Optimal":
            st.info("Resuelve un modelo factible para dibujar la red de transporte.")
        else:
            st.subheader("Red óptima de transporte")
            st.caption("El grosor representa el flujo. Las etiquetas se distribuyen a lo largo de cada flecha según el índice del origen para reducir cruces visuales.")
            matrix, cost = result["optimal"], result["costs"]
            graph = nx.DiGraph()
            oids = [f"o{i}" for i in range(len(result["origins"]))]
            dids = [f"d{j}" for j in range(len(result["destinations"]))]
            labels = {**dict(zip(oids, result["origins"])), **dict(zip(dids, result["destinations"]))}
            for node in oids: graph.add_node(node, bipartite=0)
            for node in dids: graph.add_node(node, bipartite=1)
            pos = {}
            oy = np.linspace(1, 0, len(oids)) if len(oids) > 1 else [0.5]
            dy = np.linspace(1, 0, len(dids)) if len(dids) > 1 else [0.5]
            for i, node in enumerate(oids): pos[node] = (0, float(oy[i]))
            for j, node in enumerate(dids): pos[node] = (1, float(dy[j]))
            edges = []
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if matrix[i, j] > 1e-7:
                        edge = (oids[i], dids[j])
                        graph.add_edge(*edge, flow=matrix[i, j], cost=cost[i, j])
                        edges.append((i, j, edge))
            fig, ax = plt.subplots(figsize=(12, max(4.8, .85 * max(len(oids), len(dids)))), facecolor="#101820")
            ax.set_facecolor("#101820")
            nx.draw_networkx_nodes(graph, pos, nodelist=oids, node_color="#147d92", edgecolors="#a7e3ea", linewidths=1.2, node_size=2100, ax=ax)
            nx.draw_networkx_nodes(graph, pos, nodelist=dids, node_color="#d98a16", edgecolors="#ffdda0", linewidths=1.2, node_size=2100, ax=ax)
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=9, font_color="white", font_weight="bold", ax=ax)
            max_flow = max(float(matrix.max()), 1.0)
            for k, (i, j, (u, v)) in enumerate(edges):
                flow = graph[u][v]["flow"]
                rad = (0.10 + 0.025 * (k % 3)) * (1 if (i + j) % 2 == 0 else -1)
                nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], width=1.5 + 4.0 * flow / max_flow,
                                       edge_color="#59a9bd", alpha=.82, arrows=True, arrowsize=18,
                                       min_source_margin=18, min_target_margin=18,
                                       connectionstyle=f"arc3,rad={rad}", ax=ax)
                # t escalonado por índice de origen: 0.28 ... 0.72 del trayecto.
                t = .28 + .44 * (i / max(1, matrix.shape[0] - 1))
                p0, p2 = np.array(pos[u], dtype=float), np.array(pos[v], dtype=float)
                delta = p2 - p0
                p1 = (p0 + p2) / 2 + rad * np.array([-delta[1], delta[0]])
                point = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
                ax.text(point[0], point[1], f"{flow:g} u. · c/u {cost[i, j]:g}",
                        ha="center", va="center", fontsize=8, color="#f4f8fb",
                        bbox={"boxstyle": "round,pad=.3", "facecolor": "#172b3a", "edgecolor": "#467084", "alpha": .97}, zorder=5)
            ax.text(0, 1.13, "ORÍGENES", ha="center", color="#9edce5", fontsize=9, fontweight="bold")
            ax.text(1, 1.13, "DESTINOS", ha="center", color="#ffdda0", fontsize=9, fontweight="bold")
            ax.set_xlim(-.28, 1.28); ax.set_ylim(-.16, 1.27); ax.axis("off")
            fig.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with guide_tab:
        st.subheader("Formulación matemática del caso actual")
        try:
            s0, d0, c0, u0, a0, no0, nd0 = read_transport_inputs(origins, destinations, costs_df, caps_edit, allowed_edit)
            s1, d1, c1, u1, a1, no1, nd1, dummy = balance_transport(s0, d0, c0, u0, a0, no0, nd0)
            terms = [f"{c1[i,j]:g}x_{{{i+1},{j+1}}}" for i in range(len(s1)) for j in range(len(d1)) if a1[i,j]]
            st.markdown("**Función objetivo:**")
            st.latex(r"\min Z = " + (" + ".join(terms) if terms else "0"))
            st.markdown("**Restricciones de oferta:**")
            for i, amount in enumerate(s1):
                vars_i = [f"x_{{{i+1},{j+1}}}" for j in range(len(d1)) if a1[i,j]]
                st.latex((" + ".join(vars_i) if vars_i else "0") + f" = {amount:g}")
            st.markdown("**Restricciones de demanda:**")
            for j, amount in enumerate(d1):
                vars_j = [f"x_{{{i+1},{j+1}}}" for i in range(len(s1)) if a1[i,j]]
                st.latex((" + ".join(vars_j) if vars_j else "0") + f" = {amount:g}")
            route_constraints = []
            for i in range(len(s1)):
                for j in range(len(d1)):
                    if not a1[i,j]: route_constraints.append(f"x_{{{i+1},{j+1}}}=0")
                    elif np.isfinite(u1[i,j]): route_constraints.append(f"0 \\le x_{{{i+1},{j+1}}} \\le {u1[i,j]:g}")
            st.markdown("**Rutas, capacidades y no negatividad:**")
            st.latex(r",\quad ".join(route_constraints + [f"x_{{{i+1},{j+1}}}\\ge 0" for i in range(len(s1)) for j in range(len(d1)) if a1[i,j]]) or r"x_{ij}\ge 0")
            if dummy:
                st.info("La formulación incluye el nodo ficticio para equilibrar las cantidades totales.")
        except Exception as exc:
            st.warning(f"Completa datos válidos para generar la formulación: {exc}")
        with st.expander("Métodos de solución"):
            st.markdown("**Esquina Noroeste** asigna desde la esquina superior izquierda. **Costo Mínimo** prioriza las celdas más baratas. **Aproximación de Vogel** elige una fila o columna según la diferencia entre sus dos menores costos. PuLP resuelve el modelo lineal y certifica el óptimo global si el estado es óptimo.")


# ───────────────────────────────── Floyd-Warshall ─────────────────────────────────
class SolverSetupError(RuntimeError):
    """Error recuperable de instalación o disponibilidad del solver CBC."""


def cbc_install_command() -> str:
    """Comando para instalar CBC en el mismo Python que ejecuta Streamlit."""
    return f'& "{sys.executable}" -m pip install --upgrade "pulp[cbc]>=3.0,<4.0"'


NO_EDGE = 9999.0


def default_adjacency(n: int) -> pd.DataFrame:
    matrix = np.full((n, n), NO_EDGE, dtype=float)
    np.fill_diagonal(matrix, 0.0)
    return pd.DataFrame(matrix, index=[f"Nodo {i+1}" for i in range(n)],
                        columns=[f"Nodo {i+1}" for i in range(n)])


def cañete_graph() -> pd.DataFrame:
    matrix = default_adjacency(4)
    for i, j, w in [(0, 1, 3), (0, 2, 7), (0, 3, 2), (1, 2, 2), (1, 3, 5), (2, 3, 1)]:
        matrix.iloc[i, j] = matrix.iloc[j, i] = w
    return matrix


def parse_adjacency(frame: pd.DataFrame, undirected: bool) -> np.ndarray:
    n = len(frame)
    matrix = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        for j in range(n):
            raw = str(frame.iloc[i, j]).strip().lower()
            if raw in {"", "inf", "+inf", "∞", "infinito", "9999"}:
                matrix[i, j] = np.inf
            else:
                try:
                    value = float(raw.replace(",", "."))
                except ValueError as exc:
                    raise ValueError(f"El valor de Nodo {i+1} a Nodo {j+1} no es un número, 9999 ni inf.") from exc
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Usa pesos finitos no negativos; 9999 o inf representan ausencia de arista.")
                matrix[i, j] = np.inf if value >= NO_EDGE else value
    np.fill_diagonal(matrix, 0.0)
    if undirected:
        for i in range(n):
            for j in range(i + 1, n):
                candidates = [v for v in (matrix[i, j], matrix[j, i]) if np.isfinite(v)]
                weight = min(candidates) if candidates else np.inf
                matrix[i, j] = matrix[j, i] = weight
    return matrix


def floyd_warshall_steps(adjacency: np.ndarray):
    n = adjacency.shape[0]
    dist = adjacency.copy()
    next_hop = np.full((n, n), -1, dtype=int)
    for i in range(n):
        for j in range(n):
            if np.isfinite(dist[i, j]):
                next_hop[i, j] = j
    snapshots = [dist.copy()]
    changes = [np.zeros_like(dist, dtype=bool)]
    for k in range(n):
        previous = dist.copy()
        for i in range(n):
            for j in range(n):
                if np.isfinite(previous[i, k]) and np.isfinite(previous[k, j]):
                    candidate = previous[i, k] + previous[k, j]
                    if candidate < dist[i, j] - 1e-10:
                        dist[i, j] = candidate
                        next_hop[i, j] = next_hop[i, k]
        snapshots.append(dist.copy())
        changes.append(dist < previous - 1e-10)
    return snapshots, changes, next_hop


def reconstruct_path(next_hop: np.ndarray, source: int, destination: int) -> list[int] | None:
    if next_hop[source, destination] < 0:
        return None
    path = [source]
    current = source
    for _ in range(next_hop.shape[0] + 1):
        if current == destination:
            return path
        current = int(next_hop[current, destination])
        if current < 0:
            return None
        path.append(current)
    return None


def render_floyd() -> None:
    st.session_state.setdefault("floyd_n", 4)
    if st.session_state.pop("queued_floyd_preset", False):
        st.session_state.floyd_n = 4
        st.session_state.floyd_matrix = cañete_graph().astype(str)
        st.session_state.floyd_editor_version = st.session_state.get("floyd_editor_version", 0) + 1
    with st.sidebar:
        st.header("Configuración de rutas")
        n = st.number_input("Número de nodos", min_value=3, max_value=8, step=1, key="floyd_n")
        undirected = st.checkbox("Grafo no dirigido", value=True, key="floyd_undirected")
        if st.button("Cargar Grafo Cañete 4 Nodos", use_container_width=True):
            st.session_state.queued_floyd_preset = True
            st.rerun()
        st.caption("9999, inf o ∞ indica que no existe conexión directa.")

    n = int(n)
    if "floyd_matrix" not in st.session_state or st.session_state.floyd_matrix.shape != (n, n):
        old = st.session_state.get("floyd_matrix")
        new = default_adjacency(n).astype(str)
        if old is not None:
            for i in range(min(n, old.shape[0])):
                for j in range(min(n, old.shape[1])):
                    new.iloc[i, j] = str(old.iloc[i, j])
        st.session_state.floyd_matrix = new
    matrix_df = st.session_state.floyd_matrix
    version = st.session_state.get("floyd_editor_version", 0)
    st.subheader("Matriz de adyacencia")
    st.caption("Los valores de la diagonal se fijan en cero. En un grafo no dirigido se usa el menor peso finito de cada par y se refleja en ambas direcciones.")
    adjacency_edit = st.data_editor(
        matrix_df, key=f"floyd_adjacency_{n}_{version}", use_container_width=True,
        column_config={column: st.column_config.TextColumn(label=column) for column in matrix_df.columns},
    )
    st.session_state.floyd_matrix = adjacency_edit
    left, right = st.columns(2)
    with left:
        source = st.selectbox("Nodo origen", range(1, n + 1), format_func=lambda v: f"Nodo {v}", key=f"floyd_source_{n}")
    with right:
        destination = st.selectbox("Nodo destino", range(1, n + 1), index=min(n - 1, 3), format_func=lambda v: f"Nodo {v}", key=f"floyd_destination_{n}")
    calculate = st.button("🧭 Calcular rutas y matrices", type="primary", use_container_width=True)
    if calculate:
        st.session_state.pop("floyd_result", None)
        try:
            adjacency = parse_adjacency(adjacency_edit, undirected)
            snapshots, changes, next_hop = floyd_warshall_steps(adjacency)
            st.session_state.floyd_result = {"adjacency": adjacency, "snapshots": snapshots,
                                             "changes": changes, "next_hop": next_hop,
                                             "undirected": undirected}
        except ValueError as exc:
            st.error(f"Revisa la matriz: {exc}")
    result = st.session_state.get("floyd_result")
    if not result:
        st.info("Edita la matriz y pulsa **Calcular rutas y matrices** para ver el algoritmo paso a paso.")
        return

    snapshots, changes = result["snapshots"], result["changes"]
    st.subheader("Evolución de las distancias")
    st.caption("Las celdas verdes son distancias que mejoraron al permitir el nodo intermedio k.")
    step_tabs = st.tabs([f"D⁽{k}⁾" for k in range(len(snapshots))])
    for k, step_tab in enumerate(step_tabs):
        with step_tab:
            matrix = snapshots[k]
            display = pd.DataFrame(matrix, index=[f"Nodo {i+1}" for i in range(n)],
                                   columns=[f"Nodo {i+1}" for i in range(n)])
            def mark_improvements(_data, mask=changes[k]):
                return np.where(mask, "background-color: #176b4d; color: #ffffff; font-weight: bold", "")
            styler = display.style.apply(mark_improvements, axis=None)
            st.dataframe(styler.format(lambda val: "∞" if not np.isfinite(val) else f"{val:g}"), use_container_width=True)
            if k > 0:
                st.caption(f"En D⁽{k}⁾ se permite usar Nodo {k} como nodo intermedio.")

    src, dst = source - 1, destination - 1
    final_dist = snapshots[-1][src, dst]
    path = reconstruct_path(result["next_hop"], src, dst)
    st.subheader("Ruta óptima seleccionada")
    if not np.isfinite(final_dist) or path is None:
        st.warning(f"No existe una ruta desde Nodo {source} hasta Nodo {destination}.")
        path_edges: set[tuple[int, int]] = set()
    else:
        m1, m2 = st.columns(2)
        m1.metric("Costo mínimo", f"{final_dist:g}")
        m2.metric("Aristas del camino", str(max(0, len(path) - 1)))
        st.success(" → ".join(f"Nodo {v+1}" for v in path))
        path_edges = set(zip(path[:-1], path[1:]))
    st.subheader("Grafo y camino resaltado")
    adjacency = result["adjacency"]
    graph = nx.Graph() if result["undirected"] else nx.DiGraph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(adjacency[i, j]) and (not result["undirected"] or i < j):
                graph.add_edge(i, j, weight=adjacency[i, j])
    positions = nx.spring_layout(graph, seed=19, weight="weight")
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#101820")
    ax.set_facecolor("#101820")
    nx.draw_networkx_nodes(graph, positions, node_color="#147d92", edgecolors="#b7e9ec", linewidths=1.5, node_size=1550, ax=ax)
    nx.draw_networkx_labels(graph, positions, labels={i: f"Nodo {i+1}" for i in range(n)}, font_color="white", font_weight="bold", font_size=9, ax=ax)
    def is_path_edge(u, v):
        return (u, v) in path_edges or (result["undirected"] and (v, u) in path_edges)
    ordinary = [(u, v) for u, v in graph.edges if not is_path_edge(u, v)]
    highlighted = [(u, v) for u, v in graph.edges if is_path_edge(u, v)]
    nx.draw_networkx_edges(graph, positions, edgelist=ordinary, edge_color="#607786", width=1.8,
                           alpha=.75, arrows=not result["undirected"], arrowsize=17, ax=ax)
    nx.draw_networkx_edges(graph, positions, edgelist=highlighted, edge_color="#39e879", width=4.5,
                           alpha=1, arrows=not result["undirected"], arrowsize=22, ax=ax)
    edge_labels = {(u, v): f"{graph[u][v]['weight']:g}" for u, v in graph.edges}
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_color="#f1f5f9",
                                 font_size=9, label_pos=.5, rotate=False,
                                 bbox={"alpha": .85, "color": "#172b3a", "pad": 2}, ax=ax)
    ax.axis("off"); fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─────────────────────────────────── Aplicación ───────────────────────────────────
with st.sidebar:
    st.header("Solver IO II")
    app_mode = st.selectbox("Módulo", ["Modelo de Transporte", "Ruta más Corta (Floyd-Warshall)"], key="app_mode")

if app_mode == "Modelo de Transporte":
    render_transport()
else:
    render_floyd()
