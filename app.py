import numpy as np
import pandas as pd
import streamlit as st
import time


st.set_page_config(page_title="Banker's Algorithm Dashboard", layout="wide")
st.title("Banker's Algorithm for Deadlock Avoidance")
st.markdown(
    """
    <style>
    a[href^="#"] {
        display: none !important;
    }
    [data-testid="stElementToolbar"] {
        display: none !important;
    }
    [data-testid="stDataFrameToolbar"] {

    
        display: none !important;
    }
    [data-testid="stDataFrame"] [role="columnheader"] button {
        display: none !important;
    }
    [data-testid="stDataFrameColumnMenu"] {
        display: none !important;
    }
    /* Disable all header interactions (sort/menu) but keep body cells editable. */
    [data-testid="stDataFrame"] [role="columnheader"] {
        pointer-events: none !important;
    }
    [data-testid="stDataFrame"] [role="columnheader"] * {
        pointer-events: none !important;
    }
    /* Hide common overflow/menu affordances in data-editor headers. */
    [data-testid="stDataFrame"] [aria-label*="menu"],
    [data-testid="stDataFrame"] [aria-label*="Menu"],
    [data-testid="stDataFrame"] [aria-label*="sort"],
    [data-testid="stDataFrame"] [aria-label*="Sort"] {
        display: none !important;
    }
    [data-testid="stDataFrame"] .gdg-menu-icon,
    [data-testid="stDataFrame"] .gdg-header-drop-down,
    [data-testid="stDataFrame"] .gdg-icon-overflow {
        display: none !important;
    }
    /* Make text inputs look like plain table cells. */
    div[data-testid="stTextInput"] input {
        border-radius: 4px !important;
        border: 1px solid #d9d9d9 !important;
        text-align: center !important;
        padding: 0.35rem 0.45rem !important;
    }
    /* Hide Streamlit top header toolbar (Deploy + menu). */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    [data-testid="stDecoration"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Helpers ----------
def init_state(n: int, m: int) -> None:
    """Initialize matrices and runtime state for given dimensions."""
    st.session_state.allocation = np.zeros((n, m), dtype=int)
    st.session_state.max_need = np.zeros((n, m), dtype=int)
    st.session_state.available = np.zeros(m, dtype=int)
    st.session_state.last_sequence = []
    st.session_state.last_result = None
    st.session_state.last_message = ""


def to_int_matrix(df: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    arr = np.array(df.values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.rint(arr).astype(int)
    if arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}")
    return arr


def to_int_vector(df: pd.DataFrame, length: int) -> np.ndarray:
    arr = np.array(df.values, dtype=float).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.rint(arr).astype(int)
    if arr.shape[0] != length:
        raise ValueError(f"Expected length {length}, got {arr.shape[0]}")
    return arr


def safety_algorithm(
    allocation: np.ndarray, need: np.ndarray, available: np.ndarray
) -> tuple[bool, list[int]]:
    n = allocation.shape[0]
    work = available.copy()
    finish = [False] * n
    safe_sequence: list[int] = []

    made_progress = True
    while len(safe_sequence) < n and made_progress:
        made_progress = False
        for i in range(n):
            if not finish[i] and np.all(need[i] <= work):
                work = work + allocation[i]
                finish[i] = True
                safe_sequence.append(i)
                made_progress = True

    return all(finish), safe_sequence


def build_rag_dot(
    allocation: np.ndarray,
    need: np.ndarray,
    available: np.ndarray,
    process_colors: dict[int, str],
) -> str:
    n, m = allocation.shape
    lines: list[str] = [
        "digraph RAG {",
        "rankdir=LR;",
        "graph [fontsize=10, labelloc=t, label=\"Resource Allocation Graph\", ranksep=0.28, nodesep=0.22, pad=0.02, margin=0.02];",
        "node [fontname=Helvetica, fontsize=9];",
        "edge [fontsize=8];",
    ]

    for i in range(n):
        color = process_colors.get(i, "lightgray")
        lines.append(
            f'P{i} [shape=circle, style=filled, fillcolor="{color}", fixedsize=true, width=0.42, height=0.42, label="P{i}"];'
        )

    for j in range(m):
        lines.append(
            f'R{j} [shape=box, style=filled, fillcolor="white", fixedsize=true, width=0.55, height=0.34, label="R{j}|A:{int(available[j])}"];'
        )

    for i in range(n):
        for j in range(m):
            if allocation[i, j] > 0:
                lines.append(
                    f'R{j} -> P{i} [label="{int(allocation[i, j])}", color="forestgreen", penwidth=1.5];'
                )
            if need[i, j] > 0:
                lines.append(
                    f'P{i} -> R{j} [label="{int(need[i, j])}", color="darkorange", style="dashed", penwidth=1.1];'
                )

    lines.append("}")
    return "\n".join(lines)


def update_graph(current_process: int | None, state: str) -> None:
    n = st.session_state.sim_allocation.shape[0]
    colors = {i: "lightgray" for i in range(n)}

    for i, done in enumerate(st.session_state.sim_finish):
        if done:
            colors[i] = "palegreen"

    if current_process is not None:
        if state == "checking":
            colors[current_process] = "khaki"
        elif state == "finished":
            colors[current_process] = "palegreen"
        elif state == "waiting":
            colors[current_process] = "lightcoral"

    dot = build_rag_dot(
        st.session_state.sim_allocation,
        st.session_state.sim_need,
        st.session_state.sim_available,
        colors,
    )
    st.session_state.graph_placeholder.graphviz_chart(dot, use_container_width=True)

    st.session_state.available_placeholder.dataframe(
        pd.DataFrame([st.session_state.sim_available], index=["Work/Available"], columns=[f"R{j}" for j in range(st.session_state.sim_available.shape[0])]),
        use_container_width=True,
    )
    st.session_state.finish_placeholder.dataframe(
        pd.DataFrame(
            {
                "Process": [f"P{i}" for i in range(n)],
                "Finish": st.session_state.sim_finish,
            }
        ),
        use_container_width=True,
    )


def render_stage(
    graph_placeholder,
    metrics_placeholders: list,
    log_placeholder,
    allocation: np.ndarray,
    need: np.ndarray,
    work: np.ndarray,
    process_states: list[str],
    logs: list[str],
) -> None:
    color_map = {
        "idle": "white",
        "processing": "khaki",
        "finished": "palegreen",
        "blocked": "lightcoral",
    }
    colors = {i: color_map[process_states[i]] for i in range(len(process_states))}

    graph_placeholder.graphviz_chart(
        build_rag_dot(allocation, need, work, colors),
        use_container_width=True,
    )

    for j, metric_placeholder in enumerate(metrics_placeholders):
        metric_placeholder.metric(label=f"R{j}", value=int(work[j]))

    recent_logs = logs[-8:] if logs else ["Ready."]
    log_placeholder.markdown("#### Algorithm Steps\n" + "\n".join([f"- {line}" for line in recent_logs]))


def run_safety_with_animation(
    allocation: np.ndarray,
    need: np.ndarray,
    available: np.ndarray,
    graph_placeholder,
    metrics_placeholders: list,
    log_placeholder,
    delay_seconds: float = 1.0,
) -> tuple[bool, list[int], list[str], np.ndarray, list[bool]]:
    n = allocation.shape[0]
    work = available.copy()
    finish = [False] * n
    process_states = ["idle"] * n
    sequence: list[int] = []
    logs: list[str] = ["Starting safety simulation..."]

    render_stage(graph_placeholder, metrics_placeholders, log_placeholder, allocation, need, work, process_states, logs)
    time.sleep(delay_seconds)

    made_progress = True
    while len(sequence) < n and made_progress:
        made_progress = False
        for i in range(n):
            if finish[i]:
                continue

            process_states[i] = "processing"
            logs.append(f"Checking P{i}...")
            render_stage(graph_placeholder, metrics_placeholders, log_placeholder, allocation, need, work, process_states, logs)
            time.sleep(delay_seconds)

            if np.all(need[i] <= work):
                work = work + allocation[i]
                finish[i] = True
                sequence.append(i)
                process_states[i] = "finished"
                logs.append(f"P{i} is safe. Releasing allocated resources.")
                made_progress = True
            else:
                process_states[i] = "blocked"
                logs.append(f"P{i} is blocked (Need > Work).")

            render_stage(graph_placeholder, metrics_placeholders, log_placeholder, allocation, need, work, process_states, logs)
            time.sleep(delay_seconds)

    is_safe = all(finish)
    if is_safe:
        logs.append("System is SAFE.")
    else:
        logs.append("System is UNSAFE.")

    render_stage(graph_placeholder, metrics_placeholders, log_placeholder, allocation, need, work, process_states, logs)
    return is_safe, sequence, logs, work, finish


def df_from_matrix(matrix: np.ndarray, row_prefix: str = "P", col_prefix: str = "R") -> pd.DataFrame:
    rows = [f"{row_prefix}{i}" for i in range(matrix.shape[0])]
    cols = [f"{col_prefix}{j}" for j in range(matrix.shape[1])]
    return pd.DataFrame(matrix, index=rows, columns=cols)


def matrix_text_inputs(
    title: str, matrix: np.ndarray, key_prefix: str
) -> tuple[np.ndarray, list[str]]:
    """Render a plain text-input matrix and parse to non-negative integers."""
    n, m = matrix.shape
    st.markdown(f"### {title}")

    header_cols = st.columns(m + 1)
    header_cols[0].markdown(" ")
    for j in range(m):
        header_cols[j + 1].markdown(f"**R{j}**")

    edited = np.zeros((n, m), dtype=int)
    invalid_cells: list[str] = []
    for i in range(n):
        row_cols = st.columns(m + 1)
        row_cols[0].markdown(f"**P{i}**")
        for j in range(m):
            widget_key = f"{key_prefix}_{i}_{j}"
            if widget_key not in st.session_state:
                initial_value = int(matrix[i, j])
                st.session_state[widget_key] = "" if initial_value == 0 else str(initial_value)
            elif st.session_state[widget_key] == "0":
                st.session_state[widget_key] = ""

            raw = row_cols[j + 1].text_input(
                label=f"{title}_{i}_{j}",
                key=widget_key,
                label_visibility="collapsed",
                placeholder="0",
            ).strip()

            if raw == "":
                edited[i, j] = 0
                continue

            try:
                value = int(raw)
                if value < 0:
                    raise ValueError
                edited[i, j] = value
            except ValueError:
                invalid_cells.append(f"P{i}, R{j}")
                edited[i, j] = int(matrix[i, j])

    return edited, invalid_cells


def vector_text_inputs(title: str, vector: np.ndarray, key_prefix: str) -> tuple[np.ndarray, list[str]]:
    """Render one-row text-input vector using same feel as matrix inputs."""
    m = vector.shape[0]
    st.markdown(f"### {title}")

    header_cols = st.columns(m)
    for j in range(m):
        header_cols[j].markdown(f"**R{j}**")

    value_cols = st.columns(m)
    edited = np.zeros(m, dtype=int)
    invalid_cells: list[str] = []

    for j in range(m):
        widget_key = f"{key_prefix}_{j}"
        if widget_key not in st.session_state:
            initial_value = int(vector[j])
            st.session_state[widget_key] = "" if initial_value == 0 else str(initial_value)
        elif st.session_state[widget_key] == "0":
            st.session_state[widget_key] = ""

        raw = value_cols[j].text_input(
            label=f"{title}_{j}",
            key=widget_key,
            label_visibility="collapsed",
            placeholder="0",
        ).strip()

        if raw == "":
            edited[j] = 0
            continue

        try:
            value = int(raw)
            if value < 0:
                raise ValueError
            edited[j] = value
        except ValueError:
            invalid_cells.append(f"R{j}")
            edited[j] = int(vector[j])

    return edited, invalid_cells


if "n" not in st.session_state or "m" not in st.session_state:
    st.session_state.n = 5
    st.session_state.m = 3
    init_state(5, 3)

st.subheader("Input Setup")
cfg1, cfg2 = st.columns(2)
with cfg1:
    n = int(
        st.number_input(
            "Number of Processes",
            min_value=1,
            max_value=20,
            value=int(st.session_state.n),
            step=1,
        )
    )
with cfg2:
    m = int(
        st.number_input(
            "Number of Resources",
            min_value=1,
            max_value=10,
            value=int(st.session_state.m),
            step=1,
        )
    )

if st.session_state.n != n or st.session_state.m != m:
    st.session_state.n = n
    st.session_state.m = m
    init_state(n, m)
    st.session_state.show_stage = False
    st.rerun()

matrix_col1, matrix_col2 = st.columns(2)
with matrix_col1:
    max_need_input, max_need_invalid = matrix_text_inputs(
        "Max Need Matrix", st.session_state.max_need, "max_need"
    )
with matrix_col2:
    allocation_input, allocation_invalid = matrix_text_inputs(
        "Allocation Matrix", st.session_state.allocation, "allocation"
    )

# Sync pending available values before creating avail_* widgets.
if "pending_available" in st.session_state:
    pending_available = np.array(st.session_state.pending_available, dtype=int)
    if pending_available.shape[0] == m:
        st.session_state.available = pending_available
        for j in range(m):
            value = int(pending_available[j])
            st.session_state[f"avail_txt_{j}"] = "" if value == 0 else str(value)
    del st.session_state["pending_available"]

avail_values, avail_invalid = vector_text_inputs(
    "Initial Available Resources", st.session_state.available, "avail_txt"
)
input_error = None
try:
    allocation = allocation_input.copy()
    max_need = max_need_input.copy()
    available = np.array(avail_values, dtype=int)

    invalid_matrix_cells = max_need_invalid + allocation_invalid
    invalid_vector_cells = avail_invalid
    if invalid_matrix_cells or invalid_vector_cells:
        invalid_all = invalid_matrix_cells + invalid_vector_cells
        raise ValueError(
            "Please enter valid non-negative integers in cells: "
            + ", ".join(invalid_all[:8])
            + ("..." if len(invalid_all) > 8 else "")
        )

    if np.any(allocation < 0) or np.any(max_need < 0) or np.any(available < 0):
        raise ValueError("All values must be non-negative integers.")

    need = max_need - allocation
    if np.any(need < 0):
        raise ValueError("Invalid data: Allocation cannot exceed Max Need.")

    st.session_state.allocation = allocation
    st.session_state.max_need = max_need
    st.session_state.available = available
except ValueError as exc:
    input_error = str(exc)
    allocation = st.session_state.allocation.copy()
    max_need = st.session_state.max_need.copy()
    available = st.session_state.available.copy()
    need = np.maximum(max_need - allocation, 0)

st.markdown("### Remaining Need Matrix")
st.table(df_from_matrix(need))
if input_error:
    st.error(input_error)

req_header_col1, req_header_col2 = st.columns([1, 2])
with req_header_col1:
    st.markdown("### Request Resource")
with req_header_col2:
    req_process = st.selectbox(
        "Process ID",
        options=list(range(n)),
        format_func=lambda x: f"P{x}",
    )

req_cols = st.columns(max(1, m))
request_values = []
for j in range(m):
    with req_cols[j]:
        request_values.append(
            int(
                st.number_input(
                    f"Req R{j}",
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"req_{j}",
                )
            )
        )
request = np.array(request_values, dtype=int)

btn_col1, btn_col2, btn_col3 = st.columns(3)
with btn_col1:
    show_graph_clicked = st.button("Show Current State Graph", use_container_width=True)
with btn_col2:
    check_safety_clicked = st.button("Check Safety", type="primary", use_container_width=True)
with btn_col3:
    execute_request_clicked = st.button("Execute Request", use_container_width=True)

if "show_stage" not in st.session_state:
    st.session_state.show_stage = False

if show_graph_clicked and not input_error:
    st.session_state.show_stage = True
    st.session_state.last_result = None
    st.session_state.last_message = "Current state graph loaded."

if (check_safety_clicked or execute_request_clicked) and input_error:
    st.session_state.last_result = "error"
    st.session_state.last_message = "Please fix invalid inputs before running the algorithm."

if check_safety_clicked and not input_error:
    st.session_state.show_stage = True

if execute_request_clicked and not input_error:
    st.session_state.show_stage = True

if st.session_state.show_stage and not input_error:
    st.markdown("---")
    st.subheader("Simulation Stage")
    stage_col_graph, stage_col_log = st.columns([1.75, 0.75])

    with stage_col_graph:
        graph_placeholder = st.empty()

    with stage_col_log:
        log_placeholder = st.empty()
        st.markdown("#### Work Resources")
        work_cols = st.columns(max(1, m))
        metric_placeholders = [work_cols[j].empty() for j in range(m)]

    process_states_preview = ["idle"] * n

    if check_safety_clicked:
        is_safe, sequence, _, _, _ = run_safety_with_animation(
            allocation,
            need,
            available,
            graph_placeholder,
            metric_placeholders,
            log_placeholder,
            delay_seconds=1.0,
        )
        st.session_state.last_sequence = sequence
        if is_safe:
            st.session_state.last_result = "safe"
            st.session_state.last_message = "System is in a SAFE state."
        else:
            st.session_state.last_result = "unsafe"
            st.session_state.last_message = "System is in an UNSAFE state."

    elif execute_request_clicked:
        if np.any(request > need[req_process]):
            st.session_state.last_result = "error"
            st.session_state.last_message = "Request denied: Request > Need for selected process."
            render_stage(
                graph_placeholder,
                metric_placeholders,
                log_placeholder,
                allocation,
                need,
                available,
                process_states_preview,
                ["Request validation failed: Request > Need."],
            )
        elif np.any(request > available):
            st.session_state.last_result = "error"
            st.session_state.last_message = "Request denied: Request > Available resources."
            render_stage(
                graph_placeholder,
                metric_placeholders,
                log_placeholder,
                allocation,
                need,
                available,
                process_states_preview,
                ["Request validation failed: Request > Available."],
            )
        else:
            tmp_allocation = allocation.copy()
            tmp_need = need.copy()
            tmp_available = available.copy()

            tmp_available -= request
            tmp_allocation[req_process] += request
            tmp_need[req_process] -= request

            req_safe, req_sequence, _, _, _ = run_safety_with_animation(
                tmp_allocation,
                tmp_need,
                tmp_available,
                graph_placeholder,
                metric_placeholders,
                log_placeholder,
                delay_seconds=1.0,
            )

            if req_safe:
                st.session_state.allocation = tmp_allocation
                st.session_state.available = tmp_available
                st.session_state.pending_available = tmp_available.tolist()
                st.session_state.last_sequence = req_sequence
                st.session_state.last_result = "safe"
                st.session_state.last_message = "Request granted. State remains SAFE."
                st.rerun()
            else:
                st.session_state.last_sequence = req_sequence
                st.session_state.last_result = "unsafe"
                st.session_state.last_message = "Request cannot be granted: system would become unsafe."

    else:
        render_stage(
            graph_placeholder,
            metric_placeholders,
            log_placeholder,
            allocation,
            need,
            available,
            process_states_preview,
            ["Current graph generated. Press Check Safety or Execute Request."],
        )

st.markdown("---")
if st.session_state.last_result == "safe":
    seq_text = " -> ".join([f"P_{i}" for i in st.session_state.last_sequence])
    st.markdown(
        f"""
        <div style="
            border: 2px solid #2e7d32;
            background: #eaf7ee;
            border-radius: 10px;
            padding: 14px 16px;
            margin-top: 6px;
        ">
            <div style="font-size: 1.05rem; font-weight: 700; color: #1b5e20;">{st.session_state.last_message}</div>
            <div style="font-size: 1.28rem; font-weight: 800; color: #1b5e20; margin-top: 6px;">
                Safe Sequence: {seq_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif st.session_state.last_result == "unsafe":
    seq_text = " -> ".join([f"P_{i}" for i in st.session_state.last_sequence])
    if seq_text:
        st.markdown(
            f"""
            <div style="
                border: 2px solid #b26a00;
                background: #fff6e8;
                border-radius: 10px;
                padding: 14px 16px;
                margin-top: 6px;
            ">
                <div style="font-size: 1.05rem; font-weight: 700; color: #8d4b00;">{st.session_state.last_message}</div>
                <div style="font-size: 1.22rem; font-weight: 800; color: #8d4b00; margin-top: 6px;">
                    Partial Sequence: {seq_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="
                border: 2px solid #b26a00;
                background: #fff6e8;
                border-radius: 10px;
                padding: 14px 16px;
                margin-top: 6px;
                font-size: 1.05rem;
                font-weight: 700;
                color: #8d4b00;
            ">
                {st.session_state.last_message}
            </div>
            """,
            unsafe_allow_html=True,
        )
elif st.session_state.last_result == "error":
    st.markdown(
        f"""
        <div style="
            border: 2px solid #c62828;
            background: #fdecec;
            border-radius: 10px;
            padding: 14px 16px;
            margin-top: 6px;
            font-size: 1.05rem;
            font-weight: 700;
            color: #7f1d1d;
        ">
            {st.session_state.last_message}
        </div>
        """,
        unsafe_allow_html=True,
    )
