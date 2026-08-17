const API_BASE = "http://localhost:8000";
const MAX_SIZE = 6;

const OP_CONFIG = {
  matmul: {
    label: "Matrix multiplication",
    symbol: "x",
    aKind: "matrix",
    bKind: "matrix",
    defaults: { A: [[1, 2], [3, 4]], B: [[5, 6], [7, 8]] },
  },
  matvec: {
    label: "Matrix vector",
    symbol: "x",
    aKind: "matrix",
    bKind: "vector",
    defaults: { A: [[1, 2], [3, 4]], B: [5, 6] },
  },
  addvec: {
    label: "Add vectors",
    symbol: "+",
    aKind: "vector",
    bKind: "vector",
    defaults: { A: [1, 2, 3], B: [4, 5, 6] },
  },
  dot: {
    label: "Dot product",
    symbol: ".",
    aKind: "vector",
    bKind: "vector",
    defaults: { A: [1, 2, 3], B: [4, 5, 6] },
  },
};

const state = {
  op: "",
  A: [[1]],
  B: [[1]],
  events: [],
  step: 0,
  busy: false,
  autoTimer: null,
  visualise: true,
};

const els = {
  opSelect: document.querySelector("#opSelect"),
  opField: document.querySelector("#opField"),
  aField: document.querySelector("#aField"),
  bField: document.querySelector("#bField"),
  aEditor: document.querySelector("#aEditor"),
  bEditor: document.querySelector("#bEditor"),
  aShape: document.querySelector("#aShape"),
  bShape: document.querySelector("#bShape"),
  visualiseToggle: document.querySelector("#visualiseToggle"),
  calculateBtn: document.querySelector("#calculateBtn"),
  message: document.querySelector("#message"),
  workspace: document.querySelector("#workspace"),
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function isVector(kind) {
  return kind === "vector";
}

function displayRows(value, kind) {
  return isVector(kind) ? value.map((item) => [item]) : value;
}

function resizeMatrix(matrix, rows, cols) {
  const next = [];
  for (let r = 0; r < rows; r += 1) {
    const row = [];
    for (let c = 0; c < cols; c += 1) {
      row.push(Number(matrix[r]?.[c] ?? 0));
    }
    next.push(row);
  }
  return next;
}

function resizeVector(vector, len) {
  return Array.from({ length: len }, (_, index) => Number(vector[index] ?? 0));
}

function shapeOf(value, kind) {
  if (!state.op) return "";
  if (isVector(kind)) return `${value.length} elements`;
  return `${value.length} x ${value[0]?.length ?? 0}`;
}

function toNumber(raw) {
  if (raw.trim() === "") return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : null;
}

function validateOperand(value, kind) {
  if (isVector(kind)) {
    return Array.isArray(value) && value.length > 0 && value.length <= MAX_SIZE && value.every((item) => Number.isFinite(item));
  }

  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.length <= MAX_SIZE &&
    Array.isArray(value[0]) &&
    value[0].length > 0 &&
    value[0].length <= MAX_SIZE &&
    value.every((row) => Array.isArray(row) && row.length === value[0].length && row.every((item) => Number.isFinite(item)))
  );
}

function dimensionsValid() {
  const config = OP_CONFIG[state.op];
  if (!config) return false;
  if (!validateOperand(state.A, config.aKind) || !validateOperand(state.B, config.bKind)) return false;

  if (state.op === "matmul") return state.A[0].length === state.B.length;
  if (state.op === "matvec") return state.A[0].length === state.B.length;
  if (state.op === "addvec" || state.op === "dot") return state.A.length === state.B.length;
  return false;
}

function setBusy(isBusy) {
  state.busy = isBusy;
  const disabled = isBusy || !state.op;
  els.opSelect.disabled = isBusy;
  els.visualiseToggle.disabled = isBusy;
  els.calculateBtn.disabled = isBusy || !dimensionsValid();
  els.aEditor.querySelectorAll("button,input").forEach((control) => {
    control.disabled = disabled;
  });
  els.bEditor.querySelectorAll("button,input").forEach((control) => {
    control.disabled = disabled;
  });
}

function renderEditor(which) {
  const config = OP_CONFIG[state.op];
  const editor = which === "A" ? els.aEditor : els.bEditor;
  const shape = which === "A" ? els.aShape : els.bShape;
  const field = which === "A" ? els.aField : els.bField;
  const kind = config ? (which === "A" ? config.aKind : config.bKind) : "matrix";
  const value = state[which];

  field.classList.toggle("inactive", !state.op);
  shape.textContent = config ? shapeOf(value, kind) : "";

  if (!config) {
    editor.innerHTML = "";
    return;
  }

  const rows = displayRows(value, kind);
  const cols = rows[0]?.length ?? 1;
  const disabled = state.busy || !state.op;

  editor.innerHTML = `
    <div class="grid-wrap">
      <div class="input-grid" style="grid-template-columns: repeat(${cols}, 50px)">
        ${rows
          .map((row, rowIndex) =>
            row
              .map(
                (cell, colIndex) => {
                  const dataRow = isVector(kind) ? 0 : rowIndex;
                  const dataCol = isVector(kind) ? rowIndex : colIndex;
                  return `
                    <input
                      class="cell-input"
                      type="number"
                      step="any"
                      value="${cell}"
                      data-cell="${which}"
                      data-row="${dataRow}"
                      data-col="${dataCol}"
                      aria-label="${which} ${isVector(kind) ? `index ${rowIndex + 1}` : `row ${rowIndex + 1} column ${colIndex + 1}`}"
                      ${disabled ? "disabled" : ""}
                    >
                  `;
                },
              )
              .join(""),
          )
          .join("")}
      </div>
    </div>
    <div class="editor-actions">
      ${
        isVector(kind)
          ? `
            <button class="small-btn icon" type="button" data-action="remove-col" data-target="${which}" title="Remove element" ${disabled || value.length <= 1 ? "disabled" : ""}>-</button>
            <button class="small-btn icon" type="button" data-action="add-col" data-target="${which}" title="Add element" ${disabled || value.length >= MAX_SIZE ? "disabled" : ""}>+</button>
          `
          : `
            <button class="small-btn" type="button" data-action="remove-row" data-target="${which}" ${disabled || rows.length <= 1 ? "disabled" : ""}>- row</button>
            <button class="small-btn" type="button" data-action="add-row" data-target="${which}" ${disabled || rows.length >= MAX_SIZE ? "disabled" : ""}>+ row</button>
            <button class="small-btn" type="button" data-action="remove-col" data-target="${which}" ${disabled || cols <= 1 ? "disabled" : ""}>- col</button>
            <button class="small-btn" type="button" data-action="add-col" data-target="${which}" ${disabled || cols >= MAX_SIZE ? "disabled" : ""}>+ col</button>
          `
      }
    </div>
  `;
}

function renderInputs() {
  renderEditor("A");
  renderEditor("B");
  els.calculateBtn.disabled = state.busy || !dimensionsValid();
}

function applyCellEdit(input) {
  const which = input.dataset.cell;
  const config = OP_CONFIG[state.op];
  const kind = which === "A" ? config.aKind : config.bKind;
  const row = Number(input.dataset.row);
  const col = Number(input.dataset.col);
  const nextValue = toNumber(input.value);

  if (nextValue === null) {
    input.classList.add("invalid");
    els.calculateBtn.disabled = true;
    return;
  }

  input.classList.remove("invalid");
  if (isVector(kind)) {
    state[which][col] = nextValue;
  } else {
    state[which][row][col] = nextValue;
  }
  renderInputs();
}

function resizeOperand(action, which) {
  const config = OP_CONFIG[state.op];
  const kind = which === "A" ? config.aKind : config.bKind;
  const value = state[which];

  if (isVector(kind)) {
    const len = value.length + (action === "add-col" ? 1 : -1);
    state[which] = resizeVector(value, Math.min(MAX_SIZE, Math.max(1, len)));
  } else {
    const rows = value.length + (action === "add-row" ? 1 : action === "remove-row" ? -1 : 0);
    const cols = value[0].length + (action === "add-col" ? 1 : action === "remove-col" ? -1 : 0);
    state[which] = resizeMatrix(value, Math.min(MAX_SIZE, Math.max(1, rows)), Math.min(MAX_SIZE, Math.max(1, cols)));
  }

  renderInputs();
}

function setOperation(op) {
  stopAuto();
  state.op = op;
  state.events = [];
  state.step = 0;
  els.message.textContent = "";

  if (op) {
    state.A = clone(OP_CONFIG[op].defaults.A);
    state.B = clone(OP_CONFIG[op].defaults.B);
    clearWorkspace();
  } else {
    state.A = [[1]];
    state.B = [[1]];
    clearWorkspace();
  }

  renderInputs();
}

function clearWorkspace() {
  els.workspace.className = "workspace empty";
  els.workspace.innerHTML = `
    <div class="empty-state">
      <p>Select an operation and enter A and B to initialise a calculation.</p>
    </div>
  `;
}

function makeZeroOutput(op, init) {
  if (op === "matmul") return resizeMatrix([], init.C_rows, init.C_cols);
  if (op === "matvec") return resizeVector([], init.C_elements);
  if (op === "addvec") return resizeVector([], init.C_len);
  return [0];
}

function computeResult(A, B, op) {
  if (op === "matmul") {
    return A.map((row) => B[0].map((_, colIndex) => row.reduce((sum, value, index) => sum + value * B[index][colIndex], 0)));
  }
  if (op === "matvec") {
    return A.map((row) => row.reduce((sum, value, index) => sum + value * B[index], 0));
  }
  if (op === "addvec") return A.map((value, index) => value + B[index]);
  return [A.reduce((sum, value, index) => sum + value * B[index], 0)];
}

function visibleOutput(finalResult, events, step, op, init) {
  const output = makeZeroOutput(op, init);
  const writes = events.slice(0, step + 1).filter((event) => event.event === "write");

  writes.forEach((event) => {
    if (op === "matmul") output[event.i][event.j] = finalResult[event.i][event.j];
    if (op === "matvec") output[event.C_element] = finalResult[event.C_element];
    if (op === "addvec") output[event.C_index] = finalResult[event.C_index];
    if (op === "dot") output[0] = finalResult[0];
  });

  return output;
}

function cellKey(prefix, row, col = 0) {
  return `${prefix}:${row}:${col}`;
}

function highlightsFor(event, op) {
  const indexed = new Set();
  const writes = new Set();
  const targets = new Set();

  if (!event) return { indexed, writes, targets };

  if (op === "matmul") {
    if (event.event === "compute") {
      indexed.add(cellKey("A", event.i, event.k));
      indexed.add(cellKey("B", event.k, event.j));
      targets.add(cellKey("C", event.i, event.j));
    }
    if (event.event === "write") writes.add(cellKey("C", event.i, event.j));
  }

  if (op === "matvec") {
    if (event.event === "compute") {
      indexed.add(cellKey("A", event.A_row, event.A_col));
      indexed.add(cellKey("B", 0, event.B_element));
      targets.add(cellKey("C", 0, event.A_row));
    }
    if (event.event === "write") writes.add(cellKey("C", 0, event.C_element));
  }

  if (op === "addvec") {
    if (event.event === "compute") {
      indexed.add(cellKey("A", 0, event.A_index));
      indexed.add(cellKey("B", 0, event.B_index));
      targets.add(cellKey("C", 0, event.A_index));
    }
    if (event.event === "write") writes.add(cellKey("C", 0, event.C_index));
  }

  if (op === "dot") {
    if (event.event === "compute") {
      indexed.add(cellKey("A", 0, event.index));
      indexed.add(cellKey("B", 0, event.index));
      targets.add(cellKey("C", 0, 0));
    }
    if (event.event === "write") writes.add(cellKey("C", 0, 0));
  }

  return { indexed, writes, targets };
}

function formatValue(value) {
  if (value === null || value === undefined) return "";
  if (!Number.isFinite(Number(value))) return String(value);
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 5 });
}

function renderVizGrid(label, value, kind, prefix, marks, writtenKeys = new Set()) {
  const rows = displayRows(value, kind);
  const cols = rows[0]?.length ?? 1;
  const cells = rows
    .map((row, rowIndex) =>
      row
        .map((cell, colIndex) => {
          const keyRow = isVector(kind) ? 0 : rowIndex;
          const keyCol = isVector(kind) ? rowIndex : colIndex;
          const key = cellKey(prefix, keyRow, keyCol);
          const classes = ["viz-cell"];
          if (marks.indexed.has(key)) classes.push("indexed");
          if (marks.writes.has(key)) classes.push("write");
          if (marks.targets.has(key)) classes.push("target");
          if (prefix === "C" && !writtenKeys.has(key)) classes.push("pending");
          return `<div class="${classes.join(" ")}">${prefix === "C" && !writtenKeys.has(key) ? "..." : formatValue(cell)}</div>`;
        })
        .join(""),
    )
    .join("");

  return `
    <div class="viz-group">
      <div class="viz-label">${label}</div>
      <div class="viz-grid" style="grid-template-columns: repeat(${cols}, 48px)">
        ${cells}
      </div>
    </div>
  `;
}

function writtenOutputKeys(events, step, op) {
  const keys = new Set();
  events.slice(0, step + 1).forEach((event) => {
    if (event.event !== "write") return;
    if (op === "matmul") keys.add(cellKey("C", event.i, event.j));
    if (op === "matvec") keys.add(cellKey("C", 0, event.C_element));
    if (op === "addvec") keys.add(cellKey("C", 0, event.C_index));
    if (op === "dot") keys.add(cellKey("C", 0, 0));
  });
  return keys;
}

function eventDescription(event, op) {
  if (!event) return "";
  if (event.event === "init") return "Initialising input/output shapes from the engine.";
  if (event.event === "write") {
    if (op === "matmul") return `Writing the calculated value to C[${event.i}, ${event.j}].`;
    if (op === "matvec") return `Writing the calculated value to C[${event.C_element}].`;
    if (op === "addvec") return `Writing the calculated value to C[${event.C_index}].`;
    return "Writing the calculated value to C[0].";
  }
  if (op === "matmul") return `Indexing scalars A[${event.i}, ${event.k}] and B[${event.k}, ${event.j}] to calculate C[${event.i}, ${event.j}].`;
  if (op === "matvec") return `Indexing scalars A[${event.A_row}, ${event.A_col}] and B[${event.B_element}] to calculate C[${event.A_row}].`;
  if (op === "addvec") return `Indexing scalars A[${event.A_index}] and B[${event.B_index}] to calculate C[${event.A_index}].`;
  return `Indexing scalars A[${event.index}] and B[${event.index}] to calculate C[0].`;
}

function tallyCounters(events) {
  return events.reduce(
    (totals, event) => ({
      muls: Math.max(totals.muls, event.muls ?? 0),
      adds: Math.max(totals.adds, event.adds ?? 0),
      scalarsIndexed: Math.max(totals.scalarsIndexed, event.scalars_indexed ?? 0),
      writes: Math.max(totals.writes, event.writes ?? 0),
    }),
    { muls: 0, adds: 0, scalarsIndexed: 0, writes: 0 },
  );
}

function renderReport(events, complete) {
  const totals = tallyCounters(events);
  return `
    <section class="report-panel ${complete ? "complete" : ""}">
      <h2>${complete ? "Final report" : "Report"}</h2>
      <div class="report-list">
        <div class="report-row"><span>Multiplications</span><strong>${totals.muls}</strong></div>
        <div class="report-row"><span>Additions</span><strong>${totals.adds}</strong></div>
        <div class="report-row"><span>Scalars indexed</span><strong>${totals.scalarsIndexed}</strong></div>
        <div class="report-row"><span>Output writes</span><strong>${totals.writes}</strong></div>
      </div>
    </section>
  `;
}

function renderVisualiser() {
  const config = OP_CONFIG[state.op];
  const events = state.events;
  const init = events[0];
  const event = events[state.step];
  const marks = state.visualise ? highlightsFor(event, state.op) : highlightsFor(events[events.length - 1], state.op);
  const finalResult = computeResult(state.A, state.B, state.op);
  const output = visibleOutput(finalResult, events, state.step, state.op, init);
  const outputKind = state.op === "matmul" ? "matrix" : "vector";
  const writtenKeys = writtenOutputKeys(events, state.step, state.op);
  const visibleEvents = events.slice(0, state.step + 1);
  const complete = state.step === events.length - 1;

  els.workspace.className = "workspace";
  els.workspace.innerHTML = `
    <div class="visualiser">
      <section class="stage-card">
        <div class="stage-main">
          <div class="stage-title">
            <h2>${config.label}</h2>
            <span class="step-index">step ${state.step + 1} / ${events.length}</span>
          </div>
          <div class="matrix-zone">
            ${renderVizGrid("A", state.A, config.aKind, "A", marks)}
            <div class="operator">${config.symbol}</div>
            ${renderVizGrid("B", state.B, config.bKind, "B", marks)}
            <div class="operator">=</div>
            ${renderVizGrid("C", output, outputKind, "C", marks, writtenKeys)}
          </div>
        </div>

        <div class="stage-steps">
          <h2>Steps</h2>
          <div class="step-controls">
            <button class="step-btn" type="button" data-step-action="prev" ${state.step === 0 ? "disabled" : ""}>Previous</button>
            <button class="step-btn ${state.autoTimer ? "active" : ""}" type="button" data-step-action="auto">${state.autoTimer ? "Stop" : "Auto"}</button>
            <button class="step-btn" type="button" data-step-action="next" ${complete ? "disabled" : ""}>Next</button>
          </div>
        </div>
      </section>

      <div class="trace-dashboard">
        <div class="detail-row">
          <section class="event-panel">
            <h2>Event</h2>
            <div class="event-kind">${event.event}</div>
            <p class="event-copy">${eventDescription(event, state.op)}</p>
            <pre class="event-json">${JSON.stringify(event, null, 2)}</pre>
          </section>

          ${renderReport(visibleEvents, complete)}
        </div>
      </div>
    </div>
  `;
}

function stopAuto() {
  if (state.autoTimer) {
    window.clearInterval(state.autoTimer);
    state.autoTimer = null;
  }
}

function nextStep() {
  if (!state.events.length) return;
  state.step = state.step >= state.events.length - 1 ? 0 : state.step + 1;
  renderVisualiser();
}

function nextAutoStep() {
  if (!state.events.length) return;
  if (state.step >= state.events.length - 1) {
    stopAuto();
    renderVisualiser();
    return;
  }

  state.step += 1;
  renderVisualiser();
}

function previousStep() {
  if (!state.events.length) return;
  state.step = Math.max(0, state.step - 1);
  renderVisualiser();
}

function toggleAuto() {
  if (state.autoTimer) {
    stopAuto();
    renderVisualiser();
    return;
  }

  state.autoTimer = window.setInterval(nextAutoStep, 850);
  renderVisualiser();
}

async function calculate() {
  const config = OP_CONFIG[state.op];
  if (!config || !dimensionsValid()) return;

  stopAuto();
  els.message.textContent = "";
  state.visualise = els.visualiseToggle.checked;
  setBusy(true);

  try {
    const response = await fetch(`${API_BASE}/${state.op}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ A: state.A, B: state.B }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();

    if (result.status === "fail") {
      els.message.textContent = result.reason || "The engine rejected the request.";
      return;
    }

    state.events = Array.isArray(result.events) ? result.events : [];
    state.step = state.visualise ? 0 : Math.max(0, state.events.length - 1);
    if (!state.events.length) {
      els.message.textContent = "The engine returned no events.";
      return;
    }
    renderVisualiser();
  } catch (error) {
    els.message.textContent = `Request failed: ${error.message}`;
  } finally {
    setBusy(false);
  }
}

els.opSelect.addEventListener("change", (event) => setOperation(event.target.value));
els.visualiseToggle.addEventListener("change", (event) => {
  state.visualise = event.target.checked;
});
els.calculateBtn.addEventListener("click", calculate);

document.addEventListener("input", (event) => {
  if (event.target.matches(".cell-input")) applyCellEdit(event.target);
});

document.addEventListener("click", (event) => {
  const resizeButton = event.target.closest("[data-action]");
  if (resizeButton) {
    resizeOperand(resizeButton.dataset.action, resizeButton.dataset.target);
    return;
  }

  const stepButton = event.target.closest("[data-step-action]");
  if (!stepButton) return;
  const action = stepButton.dataset.stepAction;
  if (action === "next") nextStep();
  if (action === "prev") previousStep();
  if (action === "auto") toggleAuto();
});

renderInputs();
