"""
Local dashboard for stock_advanced_strategies.py.

Run:
    python3 stock_strategy_dashboard.py --port 8765

Then open:
    http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from stock_advanced_strategies import (
    get_default_config,
    get_factor_registry,
    invalidate_dir_fingerprints,
    run_strategies,
)
from stock_data_refresh import (
    REFRESH_REPORT_FILE,
    load_json,
    refresh_before_server,
    resolve_python,
)


ROOT = Path(__file__).resolve().parent

# 页面"搜索参数"按钮触发的后台任务状态；同一时间只允许一个搜索在跑。
OPTIMIZE_LOCK = threading.Lock()
OPTIMIZE_STATE: Dict[str, Any] = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "ok": None,
    "error": "",
    "elapsed_sec": None,
}


def _run_optimizer_job() -> None:
    started = time.time()
    cmd = [resolve_python(), "-B", "stock_strategy_optimizer.py", "--iterations", "100"]
    ok = False
    error = ""
    try:
        completed = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800
        )
        ok = completed.returncode == 0
        if not ok:
            tail = (completed.stderr or completed.stdout or "").strip().splitlines()
            error = " | ".join(tail[-3:]) if tail else f"exit={completed.returncode}"
    except subprocess.TimeoutExpired:
        error = "参数搜索超时(1800秒)"
    except OSError as exc:
        error = str(exc)
    if ok:
        # 优化器写入了新的 optimized config 与策略结果，强制下次访问重新扫描
        invalidate_dir_fingerprints()
    with OPTIMIZE_LOCK:
        OPTIMIZE_STATE.update(
            running=False,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            ok=ok,
            error=error,
            elapsed_sec=round(time.time() - started, 1),
        )


def start_optimizer_job() -> bool:
    with OPTIMIZE_LOCK:
        if OPTIMIZE_STATE["running"]:
            return False
        OPTIMIZE_STATE.update(
            running=True,
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            finished_at=None,
            ok=None,
            error="",
            elapsed_sec=None,
        )
    threading.Thread(target=_run_optimizer_job, daemon=True).start()
    return True


def optimizer_state_snapshot() -> Dict[str, Any]:
    with OPTIMIZE_LOCK:
        return dict(OPTIMIZE_STATE)


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A股高性能策略配置台</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --soft: #f8fafc;
      --soft-2: #eef2f7;
      --ink: #1f2933;
      --muted: #64748b;
      --line: #d8dee8;
      --blue: #2563eb;
      --green: #14804a;
      --red: #c2410c;
      --amber: #b7791f;
      --cyan: #0e7490;
      --shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background: var(--bg);
      font-size: 14px;
    }
    button, input, select { font: inherit; }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(560px, 680px) minmax(0, 1fr);
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      overflow: auto;
      max-height: 100vh;
    }
    .side-sticky {
      position: sticky;
      top: 0;
      z-index: 3;
      background: rgba(255, 255, 255, 0.97);
      border-bottom: 1px solid var(--line);
      padding: 16px 18px 14px;
    }
    .side-body {
      padding: 2px 18px 18px;
    }
    main {
      padding: 18px 22px 32px;
      overflow: auto;
      max-height: 100vh;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.25;
    }
    h2 {
      margin: 18px 0 10px;
      font-size: 15px;
    }
    .status {
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }
    .tabs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 14px 0;
    }
    .tab {
      border: 1px solid var(--line);
      background: #f8fafc;
      border-radius: 8px;
      padding: 9px 10px;
      cursor: pointer;
      font-weight: 650;
      color: var(--muted);
    }
    .tab.active {
      border-color: var(--blue);
      color: var(--blue);
      background: #eff6ff;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin: 10px 0 12px;
    }
    .btn {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 10px;
      cursor: pointer;
      min-height: 36px;
    }
    .btn.primary {
      background: var(--blue);
      border-color: var(--blue);
      color: white;
      font-weight: 650;
    }
    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }
    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-top: 18px;
    }
    .section-head h2 {
      margin: 0;
    }
    .count-pill {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 3px 7px;
      color: var(--muted);
      background: var(--soft);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .field {
      display: grid;
      gap: 5px;
      min-width: 0;
    }
    label {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.2;
    }
    input[type="number"], input[type="text"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 8px;
      color: var(--ink);
      background: #fff;
      min-height: 34px;
    }
    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      accent-color: var(--blue);
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--blue);
    }
    .checkrow {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
    }
    .field.disabled label {
      color: #b6c0cf;
    }
    input[type="number"]:disabled {
      background: #f1f5f9;
      color: #94a3b8;
      border-style: dashed;
      cursor: not-allowed;
    }
    .dep-hint {
      color: #94a3b8;
      font-size: 11px;
      line-height: 1.3;
    }
    .overlay {
      position: fixed;
      inset: 0;
      z-index: 100;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(246, 247, 249, 0.82);
      backdrop-filter: blur(2px);
    }
    .overlay[hidden] {
      display: none;
    }
    .overlay-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: var(--shadow);
      padding: 26px 34px;
      display: grid;
      gap: 10px;
      justify-items: center;
      text-align: center;
      max-width: 360px;
    }
    .overlay-title {
      font-size: 15px;
      font-weight: 700;
    }
    .overlay-sub {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .spinner {
      width: 34px;
      height: 34px;
      border: 4px solid #dbe3ee;
      border-top-color: var(--blue);
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .factor-tools {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(120px, 150px);
      gap: 8px;
      margin: 10px 0 8px;
    }
    .factor-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 8px;
    }
    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 8px;
      min-height: 34px;
      color: var(--ink);
      background: #fff;
    }
    .factor-list {
      display: grid;
      gap: 10px;
      padding-bottom: 8px;
    }
    .factor-group {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    .factor-group > summary {
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      background: var(--soft);
      color: var(--ink);
      list-style: none;
    }
    .factor-group > summary::-webkit-details-marker {
      display: none;
    }
    .factor-group > summary::before {
      content: "▸";
      color: var(--muted);
      font-size: 12px;
      transition: transform 0.15s ease;
    }
    .factor-group[open] > summary::before {
      transform: rotate(90deg);
    }
    .factor-group-title {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      flex: 1;
    }
    .factor-group-title strong {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .factor-group-title span {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .factor-group-body {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      padding: 8px 10px 10px;
    }
    .factor-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 72px;
      grid-template-areas:
        "name name"
        "range value";
      gap: 7px 8px;
      align-items: center;
      padding: 9px 10px;
      border: 1px solid #e5eaf3;
      border-radius: 8px;
      background: #fff;
    }
    .factor-row + .factor-row {
      margin-top: 0;
    }
    .factor-name {
      grid-area: name;
      min-width: 0;
    }
    .factor-name strong {
      display: block;
      font-size: 13px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .factor-name span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .factor-desc {
      display: block;
      margin-top: 2px;
      color: #738199;
      font-size: 11px;
      line-height: 1.35;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .factor-control {
      grid-area: range;
      display: grid;
      gap: 4px;
      min-width: 0;
    }
    .factor-control input[type="range"] {
      margin: 0;
    }
    .factor-row input[type="number"] {
      grid-area: value;
      min-height: 30px;
      padding: 5px 7px;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(140px, 1fr));
      gap: 10px;
      margin: 12px 0 14px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .metric .label {
      color: var(--muted);
      font-size: 12px;
    }
    .metric .value {
      margin-top: 5px;
      font-size: 22px;
      font-weight: 750;
    }
    .notes {
      display: grid;
      gap: 7px;
      margin: 10px 0;
    }
    .note {
      border-left: 3px solid var(--amber);
      background: #fffbeb;
      color: #7c4a03;
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 12px;
    }
    .pool-rules {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      box-shadow: var(--shadow);
      margin: 0 0 14px;
    }
    .pool-rules-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }
    .pool-rules-head strong {
      font-size: 14px;
    }
    .rule-list {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .rule-item {
      border: 1px solid #e5eaf3;
      border-radius: 8px;
      background: var(--soft);
      padding: 9px 10px;
      min-width: 0;
    }
    .rule-item strong {
      display: block;
      font-size: 12px;
      margin-bottom: 4px;
      color: #334155;
    }
    .rule-item span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .score-visual {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      box-shadow: var(--shadow);
      margin-bottom: 14px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 118px minmax(0, 1fr) 54px;
      gap: 10px;
      align-items: center;
      margin: 7px 0;
    }
    .bar-label {
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      font-size: 12px;
      color: var(--muted);
    }
    .bar-track {
      height: 11px;
      background: #e5e9f0;
      border-radius: 8px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 8px;
      background: linear-gradient(90deg, var(--cyan), var(--green));
    }
    .bar-value {
      font-variant-numeric: tabular-nums;
      color: var(--ink);
      text-align: right;
      font-size: 12px;
    }
    .table-wrap {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 980px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }
    th {
      position: sticky;
      top: 0;
      background: #f8fafc;
      z-index: 1;
      font-size: 12px;
      color: var(--muted);
    }
    td.rank {
      width: 52px;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    .code {
      font-weight: 750;
      white-space: nowrap;
    }
    .name {
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
      white-space: nowrap;
    }
    .score {
      font-weight: 750;
      color: var(--green);
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      border-radius: 6px;
      padding: 3px 6px;
      background: #eef2f7;
      color: #344256;
      font-size: 12px;
      max-width: 240px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .chip.good { background: #e7f7ee; color: #0f6b3d; }
    .chip.warn { background: #fff7ed; color: #9a3412; }
    details {
      min-width: 210px;
    }
    summary {
      cursor: pointer;
      color: var(--blue);
      font-size: 12px;
    }
    .mini-table {
      min-width: 360px;
      margin-top: 6px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .mini-table table {
      min-width: 360px;
      font-size: 12px;
    }
    .mini-table th, .mini-table td {
      padding: 6px 7px;
    }
    .empty {
      padding: 24px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      aside { max-height: none; border-right: 0; border-bottom: 1px solid var(--line); }
      .side-sticky { position: static; }
      main { max-height: none; }
      .cards { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
    }
    @media (max-width: 560px) {
      main { padding: 14px; }
      .side-sticky, .side-body { padding-left: 14px; padding-right: 14px; }
      .grid, .cards, .rule-list { grid-template-columns: 1fr; }
      .factor-tools, .factor-actions, .factor-group-body { grid-template-columns: 1fr; }
      .factor-row { grid-template-columns: minmax(0, 1fr) 72px; }
      .bar-row { grid-template-columns: 1fr 1fr; }
      .bar-track { grid-column: 1 / -1; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="side-sticky">
        <div class="topbar">
          <h1>A股策略配置台</h1>
          <span id="status" class="status">启动中</span>
        </div>
        <div class="tabs">
          <button id="tab-long" class="tab active" type="button">长线</button>
          <button id="tab-short" class="tab" type="button">短线</button>
        </div>
        <div class="toolbar">
          <button id="run" class="btn primary" type="button">运行</button>
          <button id="save" class="btn" type="button">保存配置</button>
          <button id="reset" class="btn" type="button">重置</button>
          <button id="optimize" class="btn" type="button">搜索参数</button>
          <label class="toggle"><input id="auto" type="checkbox" checked>实时</label>
        </div>
      </div>
      <div class="side-body">
        <h2>参数</h2>
        <div id="params" class="grid"></div>
        <div class="section-head">
          <h2>因子权重</h2>
          <span id="factor-count" class="count-pill"></span>
        </div>
        <div class="factor-tools">
          <input id="factor-search" type="text" placeholder="搜索因子">
          <select id="factor-group"></select>
        </div>
        <div class="factor-actions">
          <button id="expand-factors" class="btn" type="button">展开分组</button>
          <button id="collapse-factors" class="btn" type="button">折叠分组</button>
        </div>
        <div id="factors" class="factor-list"></div>
      </div>
    </aside>
    <main>
      <div class="topbar">
        <div>
          <h1 id="title">长线策略</h1>
          <div id="subtitle" class="status"></div>
        </div>
        <div id="generated" class="status"></div>
      </div>
      <div id="metrics" class="cards"></div>
      <div id="pool-rules" class="pool-rules"></div>
      <div id="notes" class="notes"></div>
      <div id="bars" class="score-visual"></div>
      <div id="table" class="table-wrap"></div>
    </main>
  </div>
  <div id="optimize-overlay" class="overlay" hidden>
    <div class="overlay-card">
      <div class="spinner"></div>
      <div class="overlay-title">正在搜索参数…</div>
      <div id="optimize-elapsed" class="overlay-sub">已耗时 0 秒</div>
      <div class="overlay-sub">长线/短线各 100 次随机搜索回测，约需 1-2 分钟，请勿关闭页面</div>
    </div>
  </div>
  <script>
    let config = null;
    let registry = null;
    let latest = null;
    let active = "long";
    let runTimer = null;
    let factorSearch = "";
    let factorGroup = "all";
    let factorGroupsExpanded = true;
    let optimizeTimer = null;
    let optimizeStartedAt = null;

    const $ = (id) => document.getElementById(id);
    const status = (text) => { $("status").textContent = text; };
    const fmt = (v, digits = 2) => {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return "--";
      return Number(v).toFixed(digits);
    };
    const esc = (text) => String(text ?? "").replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[ch]));

    async function fetchConfig() {
      const resp = await fetch("/api/config");
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload.error || "config failed");
      return payload;
    }

    async function init() {
      try {
        const saved = localStorage.getItem("stockStrategyConfig");
        const payload = await fetchConfig();
        config = saved ? mergeDeep(payload.config, JSON.parse(saved)) : payload.config;
        registry = payload.factors;
      } catch (err) {
        status("失败");
        $("table").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
        return;
      }
      bindEvents();
      renderControls();
      await resumeOptimizeIfRunning();
      await runNow();
    }

    function bindEvents() {
      $("tab-long").onclick = () => switchStrategy("long");
      $("tab-short").onclick = () => switchStrategy("short");
      $("run").onclick = runNow;
      $("save").onclick = () => {
        localStorage.setItem("stockStrategyConfig", JSON.stringify(config));
        status("已保存");
      };
      $("reset").onclick = async () => {
        const msg = "是否真的要重置为默认搜索参数？当前页面配置会被清空。";
        if (!window.confirm(msg)) return;
        localStorage.removeItem("stockStrategyConfig");
        try {
          const payload = await fetchConfig();
          config = payload.config;
          registry = payload.factors;
        } catch (err) {
          status("失败");
          $("table").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
          return;
        }
        renderControls();
        await runNow();
      };
      $("factor-search").oninput = () => {
        factorSearch = $("factor-search").value.trim().toLowerCase();
        renderFactors();
      };
      $("factor-group").onchange = () => {
        factorGroup = $("factor-group").value;
        renderFactors();
      };
      $("expand-factors").onclick = () => setFactorGroupsOpen(true);
      $("collapse-factors").onclick = () => setFactorGroupsOpen(false);
      $("optimize").onclick = startOptimize;
    }

    async function startOptimize() {
      const msg = "将对长线/短线各运行 100 次参数搜索回测"
        + "（相当于 python stock_strategy_optimizer.py --iterations 100），"
        + "约需 1-2 分钟，期间页面将锁定。确定开始吗？";
      if (!window.confirm(msg)) return;
      try {
        const resp = await fetch("/api/optimize", {method: "POST"});
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload.error || "启动失败");
      } catch (err) {
        status("失败");
        $("table").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
        return;
      }
      showOptimizeOverlay();
    }

    function showOptimizeOverlay() {
      optimizeStartedAt = Date.now();
      $("optimize-overlay").hidden = false;
      status("搜索参数中");
      clearInterval(optimizeTimer);
      optimizeTimer = setInterval(pollOptimizeStatus, 3000);
    }

    async function pollOptimizeStatus() {
      $("optimize-elapsed").textContent = `已耗时 ${Math.round((Date.now() - optimizeStartedAt) / 1000)} 秒`;
      let state;
      try {
        const resp = await fetch("/api/optimize/status");
        state = await resp.json();
      } catch (err) {
        return; // 网络抖动，下一轮再试
      }
      if (state.running) return;
      clearInterval(optimizeTimer);
      optimizeTimer = null;
      $("optimize-overlay").hidden = true;
      if (state.ok) {
        // 旧的本地保存配置会覆盖新搜出的默认参数，搜索成功后直接作废
        localStorage.removeItem("stockStrategyConfig");
        try {
          const payload = await fetchConfig();
          config = payload.config;
          registry = payload.factors;
        } catch (err) {
          status("失败");
          $("table").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
          return;
        }
        renderControls();
        await runNow();
        status(`参数已更新(${state.elapsed_sec ?? "?"}s)`);
      } else {
        status("搜索失败");
        $("table").innerHTML = `<div class="empty">参数搜索失败: ${esc(state.error || "未知错误")}</div>`;
      }
    }

    async function resumeOptimizeIfRunning() {
      // 页面中途刷新时，若后台搜索仍在跑，恢复遮罩与轮询
      try {
        const resp = await fetch("/api/optimize/status");
        const state = await resp.json();
        if (state.running) showOptimizeOverlay();
      } catch (err) {
        /* 状态接口异常不阻塞页面初始化 */
      }
    }

    function switchStrategy(name) {
      active = name;
      factorGroup = "all";
      factorSearch = "";
      $("factor-search").value = "";
      $("tab-long").classList.toggle("active", name === "long");
      $("tab-short").classList.toggle("active", name === "short");
      renderControls();
      renderResults();
    }

    function mergeDeep(base, override) {
      const out = structuredClone(base);
      for (const [key, value] of Object.entries(override || {})) {
        if (value && typeof value === "object" && !Array.isArray(value) && out[key]) {
          out[key] = mergeDeep(out[key], value);
        } else {
          out[key] = value;
        }
      }
      return out;
    }

    function renderControls() {
      renderParams();
      renderFactors();
      $("title").textContent = active === "long" ? "长线大盘股策略" : "短线龙虎榜策略";
    }

    function paramSchema() {
      if (active === "long") {
        return [
          ["top_n", "输出数量", "number", 1, 100, 1],
          ["min_score", "最低分", "number", 0, 100, 1],
          ["min_market_cap_yi", "市值下限(亿)", "number", 0, 5000, 50],
          ["min_listing_years", "上市年限", "number", 0, 20, 1],
          ["min_csi300_persistence", "成分稳定分", "number", 0, 100, 1],
          ["require_csi300", "必须当前沪深300", "checkbox"],
          ["require_high_drawdown", "高点回撤过滤", "checkbox"],
          ["min_high_drawdown_pct", "高点至今跌幅下限(%)", "number", 0, 95, 1, "require_high_drawdown"],
          ["exclude_st", "排除ST", "checkbox"]
        ];
      }
      return [
        ["top_n", "输出数量", "number", 1, 100, 1],
        ["min_score", "最低分", "number", 0, 100, 1],
        ["min_lhb_count", "最少上榜", "number", 0, 20, 1],
        ["min_hot_money_concurrent", "最少共振席位", "number", 0, 10, 1],
        ["max_consecutive_limit_up", "最多连板", "number", 0, 10, 1],
        ["exclude_st", "排除ST", "checkbox"]
      ];
    }

    function renderParams() {
      const box = $("params");
      box.innerHTML = "";
      const schema = paramSchema();
      const labelByKey = Object.fromEntries(schema.map((row) => [row[0], row[1]]));
      for (const [key, label, type, min, max, step, dependsOn] of schema) {
        const field = document.createElement("div");
        field.className = "field";
        if (type === "checkbox") {
          field.innerHTML = `<label>${esc(label)}</label><div class="checkrow"><input id="p-${key}" type="checkbox"><span>${esc(label)}</span></div>`;
          const input = field.querySelector("input");
          input.checked = Boolean(config[active][key]);
          input.onchange = () => {
            updateParam(key, input.checked);
            renderParams(); // 同步依赖此开关的输入框禁用状态
          };
        } else {
          const enabled = !dependsOn || Boolean(config[active][dependsOn]);
          const hint = dependsOn
            ? `<small class="dep-hint">勾选「${esc(labelByKey[dependsOn] || dependsOn)}」后生效</small>`
            : "";
          field.innerHTML = `<label for="p-${key}">${esc(label)}</label><input id="p-${key}" type="number" min="${min}" max="${max}" step="${step}">${hint}`;
          if (!enabled) field.classList.add("disabled");
          const input = field.querySelector("input");
          input.value = config[active][key];
          input.disabled = !enabled;
          input.oninput = () => {
            const v = Number(input.value);
            if (input.value === "" || Number.isNaN(v)) return;
            updateParam(key, v);
          };
        }
        box.appendChild(field);
      }
    }

    function renderFactors() {
      const box = $("factors");
      const allFactors = registry[active];
      const groupNames = [...new Set(allFactors.map((factor) => factor.group))].sort();
      renderFactorGroupSelect(groupNames);
      const visibleFactors = allFactors.filter((factor) => {
        const inGroup = factorGroup === "all" || factor.group === factorGroup;
        const haystack = `${factor.label} ${factor.key} ${factor.group} ${factor.description}`.toLowerCase();
        const matches = !factorSearch || haystack.includes(factorSearch);
        return inGroup && matches;
      });
      $("factor-count").textContent = `${active === "long" ? "长线" : "短线"} ${visibleFactors.length}/${allFactors.length}`;
      if (!visibleFactors.length) {
        box.innerHTML = `<div class="empty">没有匹配的因子</div>`;
        return;
      }

      const groups = {};
      for (const factor of visibleFactors) {
        if (!groups[factor.group]) groups[factor.group] = [];
        groups[factor.group].push(factor);
      }
      box.innerHTML = Object.entries(groups).map(([group, factors]) => `
        <details class="factor-group" ${factorGroupsExpanded ? "open" : ""}>
          <summary>
            <span class="factor-group-title">
              <strong>${esc(group)}</strong>
              <span>${factors.length} 个</span>
            </span>
            <span class="count-pill group-weight" data-group="${esc(group)}">权重 ${fmt(groupWeightSum(factors), 2)}</span>
          </summary>
          <div class="factor-group-body">${factors.map((factor) => factorRow(factor)).join("")}</div>
        </details>
      `).join("");
      const updateGroupPill = (group) => {
        const pill = box.querySelector(`.group-weight[data-group="${CSS.escape(group)}"]`);
        if (pill) pill.textContent = `权重 ${fmt(groupWeightSum(groups[group] || []), 2)}`;
      };
      for (const factor of visibleFactors) {
        const range = $(`w-${factor.key}`);
        const num = $(`n-${factor.key}`);
        const setWeight = (value) => {
          const v = Math.max(0, Math.min(3, Number(value)));
          config[active].weights[factor.key] = v;
          updateGroupPill(factor.group);
          scheduleRun();
          return v;
        };
        range.oninput = () => {
          num.value = setWeight(range.value).toFixed(2);
        };
        num.oninput = () => {
          if (num.value === "" || Number.isNaN(Number(num.value))) return;
          range.value = setWeight(num.value);
        };
        num.onblur = () => {
          const v = Number(config[active].weights[factor.key] ?? factor.default_weight);
          num.value = v.toFixed(2);
          range.value = v;
        };
      }
    }

    function renderFactorGroupSelect(groupNames) {
      if (factorGroup !== "all" && !groupNames.includes(factorGroup)) {
        factorGroup = "all";
      }
      $("factor-group").innerHTML = [
        `<option value="all">全部分组</option>`,
        ...groupNames.map((group) => `<option value="${esc(group)}">${esc(group)}</option>`)
      ].join("");
      $("factor-group").value = factorGroup;
    }

    function groupWeightSum(factors) {
      return factors.reduce((sum, factor) => {
        const weight = Number(config[active].weights[factor.key] ?? factor.default_weight);
        return sum + (Number.isFinite(weight) ? weight : 0);
      }, 0);
    }

    function setFactorGroupsOpen(open) {
      factorGroupsExpanded = open;
      for (const detail of document.querySelectorAll(".factor-group")) {
        detail.open = open;
      }
    }

    function factorRow(factor) {
      const weight = Number(config[active].weights[factor.key] ?? factor.default_weight);
      return `
        <div class="factor-row" title="${esc(factor.description)}">
          <div class="factor-name">
            <strong>${esc(factor.label)}</strong>
            <span>${esc(factor.key)}</span>
            <small class="factor-desc">${esc(factor.description)}</small>
          </div>
          <div class="factor-control">
            <input id="w-${esc(factor.key)}" type="range" min="0" max="3" step="0.05" value="${weight}">
          </div>
          <input id="n-${esc(factor.key)}" type="number" min="0" max="3" step="0.05" value="${weight.toFixed(2)}">
        </div>
      `;
    }

    function updateParam(key, value) {
      config[active][key] = value;
      scheduleRun();
    }

    function scheduleRun() {
      if (!$("auto").checked) return;
      clearTimeout(runTimer);
      runTimer = setTimeout(runNow, 350);
    }

    async function runNow() {
      try {
        status("运行中");
        const resp = await fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({config})
        });
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload.error || "run failed");
        latest = payload;
        status("已完成");
        renderResults();
      } catch (err) {
        status("失败");
        $("table").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
      }
    }

    function renderResults() {
      if (!latest) return;
      const section = latest[active];
      if (!section) return;
      $("subtitle").textContent = section.title;
      $("generated").textContent = section.generated_at;
      renderMetrics(section);
      renderPoolRules(section);
      renderNotes(section.notes || []);
      renderBars(section.picks || []);
      renderTable(section.picks || []);
    }

    function renderMetrics(section) {
      const diag = section.diagnostics || {};
      const range = diag.score_range || ["--", "--"];
      $("metrics").innerHTML = [
        ["候选", section.candidate_count],
        ["入选", section.selected_count],
        [active === "long" ? "长线因子" : "短线因子", section.factor_count],
        ["均分", diag.avg_score ?? "--"],
        ["分数区间", `${range[0]} / ${range[1]}`],
        ["数据覆盖", diag.avg_data_quality === undefined ? "--" : `${fmt(diag.avg_data_quality * 100, 1)}%`]
      ].map(([label, value]) => `<div class="metric"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`).join("");
    }

    function renderPoolRules(section) {
      const rules = active === "long" ? longPoolRules() : shortPoolRules();
      $("pool-rules").innerHTML = `
        <div class="pool-rules-head">
          <strong>${active === "long" ? "长线股票池筛选" : "短线股票池筛选"}</strong>
          <span class="count-pill">候选 ${esc(section.candidate_count ?? "--")} / 入选 ${esc(section.selected_count ?? "--")}</span>
        </div>
        <div class="rule-list">
          ${rules.map((rule) => `
            <div class="rule-item">
              <strong>${esc(rule[0])}</strong>
              <span>${esc(rule[1])}</span>
            </div>
          `).join("")}
        </div>
      `;
    }

    function longPoolRules() {
      const cfg = config.long;
      return [
        ["基础范围", "A股基础股票池，叠加沪深300成员、行情快照、财务与K线数据"],
        ["ST过滤", cfg.exclude_st ? "剔除名称包含 ST、*ST 或 S 前缀的股票" : "不过滤ST股票，风险只进入结果提示"],
        ["沪深300", cfg.require_csi300 ? "必须是当前沪深300成分股" : "不强制当前成分股，成分稳定性只参与打分"],
        ["成分稳定", `成分稳定分不低于 ${fmt(cfg.min_csi300_persistence, 0)}，历史快照不足时使用当前成员、核心池与上市年限代理`],
        ["历史回撤", cfg.require_high_drawdown ? `已启用：历史最高收盘价至今跌幅不低于 ${fmt(cfg.min_high_drawdown_pct, 0)}%` : `未启用：勾选「高点回撤过滤」后，才会按跌幅下限 ${fmt(cfg.min_high_drawdown_pct, 0)}% 硬过滤`],
        ["市值门槛", `总市值不低于 ${fmt(cfg.min_market_cap_yi, 0)} 亿元`],
        ["上市年限", `上市时间不低于 ${fmt(cfg.min_listing_years, 0)} 年`],
        ["评分出池", `综合分不低于 ${fmt(cfg.min_score, 0)}，按得分取前 ${fmt(cfg.top_n, 0)} 只`],
      ];
    }

    function shortPoolRules() {
      const cfg = config.short;
      return [
        ["基础范围", "龙虎榜与资金榜股票池，优先使用本地席位、净买入、游资跟随与连板数据"],
        ["ST过滤", cfg.exclude_st ? "剔除名称包含 ST、*ST 或 S 前缀的股票" : "不过滤ST股票，风险只进入结果提示"],
        ["上榜次数", `近期龙虎榜上榜次数不低于 ${fmt(cfg.min_lhb_count, 0)} 次`],
        ["游资共振", Number(cfg.min_hot_money_concurrent || 0) > 0 ? `活跃游资/买方席位共振不低于 ${fmt(cfg.min_hot_money_concurrent, 0)} 个` : "不设置最低共振席位，交给游资网络因子打分"],
        ["连板过滤", `连续涨停不超过 ${fmt(cfg.max_consecutive_limit_up, 0)} 板`],
        ["评分出池", `综合分不低于 ${fmt(cfg.min_score, 0)}，按得分取前 ${fmt(cfg.top_n, 0)} 只`],
        ["交易约束", "可交易性、过热、机构分歧和风险控制不做硬剔除，进入短线因子权重"],
      ];
    }

    function renderNotes(notes) {
      $("notes").innerHTML = notes.map((n) => `<div class="note">${esc(n)}</div>`).join("");
    }

    function renderBars(picks) {
      const top = picks.slice(0, 12);
      if (!top.length) {
        $("bars").innerHTML = `<div class="empty">当前参数没有选出股票</div>`;
        return;
      }
      $("bars").innerHTML = top.map((p) => `
        <div class="bar-row">
          <div class="bar-label">${esc(p.code)} ${esc(p.name)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, p.score))}%"></div></div>
          <div class="bar-value">${fmt(p.score)}</div>
        </div>
      `).join("");
    }

    function renderTable(picks) {
      if (!picks.length) {
        $("table").innerHTML = `<div class="empty">当前参数没有选出股票</div>`;
        return;
      }
      $("table").innerHTML = `
        <table>
          <thead>
            <tr>
              <th>排名</th><th>股票</th><th>得分</th><th>理由</th><th>风险</th><th>主导因子</th><th>明细</th>
            </tr>
          </thead>
          <tbody>${picks.map(renderRow).join("")}</tbody>
        </table>
      `;
    }

    function renderRow(pick) {
      const topFactors = Object.entries(pick.factor_scores || {})
        .map(([key, v]) => ({key, ...v, power: Number(v.score || 0) * Number(v.weight || 0)}))
        .filter((v) => v.weight > 0)
        .sort((a, b) => b.power - a.power)
        .slice(0, 5);
      return `
        <tr>
          <td class="rank">${pick.rank}</td>
          <td><div class="code">${esc(pick.code)}</div><div class="name">${esc(pick.name)} ${esc(pick.industry || "")}</div></td>
          <td><span class="score">${fmt(pick.score)}</span><div class="name">覆盖 ${fmt((pick.data_quality || 0) * 100, 1)}%</div></td>
          <td><div class="chips">${(pick.reasons || []).map((r) => `<span class="chip good">${esc(r)}</span>`).join("")}</div></td>
          <td><div class="chips">${(pick.warnings || []).map((r) => `<span class="chip warn">${esc(r)}</span>`).join("") || `<span class="chip">无显著提示</span>`}</div></td>
          <td><div class="chips">${topFactors.map((f) => `<span class="chip">${esc(f.label)} ${fmt(f.score, 0)}×${fmt(f.weight, 1)}</span>`).join("")}</div></td>
          <td>${detailBlock(pick)}</td>
        </tr>
      `;
    }

    function detailBlock(pick) {
      const rows = Object.entries(pick.factor_scores || {})
        .sort((a, b) => Number(b[1].weight || 0) - Number(a[1].weight || 0))
        .map(([key, v]) => `
          <tr>
            <td>${esc(v.label || key)}</td>
            <td>${esc(v.raw)}</td>
            <td>${fmt(v.score, 1)}</td>
            <td>${fmt(v.weight, 1)}</td>
          </tr>
        `).join("");
      const followers = (pick.followers || []).map((f) => `
        <tr><td>${esc(f.date || "")}</td><td>${esc(f.category || "")}</td><td>${esc(f.seat || "")}</td><td>${esc(f.buy_est || "")}</td></tr>
      `).join("");
      return `
        <details>
          <summary>展开</summary>
          <div class="mini-table"><table><thead><tr><th>因子</th><th>原始值</th><th>分</th><th>权</th></tr></thead><tbody>${rows}</tbody></table></div>
          ${followers ? `<div class="mini-table"><table><thead><tr><th>日期</th><th>类别</th><th>席位</th><th>买入</th></tr></thead><tbody>${followers}</tbody></table></div>` : ""}
        </details>
      `;
    }

    init();
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "StockStrategyDashboard/1.0"

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_text(HTML, "text/html; charset=utf-8")
            return
        if self.path == "/api/config":
            self.send_json({"config": get_default_config(), "factors": get_factor_registry()})
            return
        if self.path == "/api/health":
            self.send_json({"ok": True, "refresh": load_json(REFRESH_REPORT_FILE, {})})
            return
        if self.path == "/api/optimize/status":
            self.send_json(optimizer_state_snapshot())
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.path == "/api/optimize":
            if start_optimizer_job():
                self.send_json({"started": True})
            else:
                self.send_json({"error": "参数搜索已在运行中"}, status=409)
            return
        if self.path != "/api/run":
            self.send_error(404, "Not found")
            return
        try:
            payload = self.read_json()
            # Force a fresh directory scan so each run sees data refreshed
            # externally, without the fingerprint TTL window.
            invalidate_dir_fingerprints()
            result = run_strategies(payload.get("config", {}), persist=False)
            self.send_json(result)
        except Exception as exc:  # Keep dashboard errors visible to the browser.
            self.send_json({"error": str(exc)}, status=500)

    def read_json(self) -> Dict[str, Any]:
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def send_text(self, text: str, content_type: str, status: int = 200) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


def run_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def warmup_strategy_cache() -> None:
    started = time.perf_counter()
    print("Warming strategy cache...", flush=True)
    try:
        invalidate_dir_fingerprints()
        result = run_strategies(get_default_config(), persist=False)
    except Exception as exc:
        print(f"Strategy cache warmup failed: {exc}", flush=True)
        return
    elapsed = time.perf_counter() - started
    long_count = result.get("long", {}).get("candidate_count", 0)
    short_count = result.get("short", {}).get("candidate_count", 0)
    print(
        "Strategy cache warmed in "
        f"{elapsed:.1f}s; long candidates={long_count}, short candidates={short_count}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local stock strategy dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--skip-refresh", action="store_true", help="start without refreshing data")
    parser.add_argument("--skip-warmup", action="store_true", help="start without precomputing strategy caches")
    parser.add_argument("--refresh-mode", choices=["full", "quick", "capital-only"], default="full")
    parser.add_argument("--strict-refresh", action="store_true", help="abort startup if any refresh step fails")
    parser.add_argument("--refresh-timeout", type=int, default=1800, help="per-step timeout seconds, 0=disabled")
    parser.add_argument("--refresh-python", default=None, help="python executable used for crawler subprocesses")
    parser.add_argument("--index-workers", type=int, default=20)
    parser.add_argument("--index-limit", type=int, default=0)
    args = parser.parse_args()

    if not args.skip_refresh:
        report = refresh_before_server(
            mode=args.refresh_mode,
            strict=args.strict_refresh,
            timeout=args.refresh_timeout or None,
            python=args.refresh_python,
            index_workers=args.index_workers,
            index_limit=args.index_limit,
        )
        state = "ok" if report.get("ok") else "with warnings"
        print(f"Data refresh finished {state}; report: {REFRESH_REPORT_FILE}", flush=True)
    if not args.skip_warmup:
        warmup_strategy_cache()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
