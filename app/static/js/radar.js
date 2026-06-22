(() => {
  const PHASE_ORDER = [
    "疑似吸筹(待确认)🟢", "吸筹🟢", "试盘🟡", "洗盘🟡", "吸筹+洗盘🟡",
    "▲突破🟠", "拉升中🟠", "出货预警🔴", "观望⚪",
  ];

  function normalizePhase(label) {
    return String(label || "").replace("空仓观望", "观望");
  }

  function phaseClass(label) {
    label = String(label || "");
    if (label.includes("🟢")) return "g";
    if (label.includes("🟡")) return "y";
    if (label.includes("🟠")) return "o";
    if (label.includes("🔴")) return "r";
    return "w";
  }
  const fmt = (v, d = 2) => (v === null || v === undefined || v === "" || Number.isNaN(Number(v)) ? "-" : Number(v).toFixed(d));
  const scoreValue = (s, key, fallbackKey) => {
    const v = s && s[key];
    if (v !== null && v !== undefined && v !== "" && !Number.isNaN(Number(v))) return Number(v);
    return s ? s[fallbackKey] : null;
  };
  const fmtMarketCap = (v) => {
    if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) return "-";
    const n = Number(v);
    if (Math.abs(n) >= 10000) return `${(n / 10000).toFixed(2)}万亿`;
    if (Math.abs(n) >= 1000) return `${n.toFixed(0)}亿`;
    if (Math.abs(n) >= 100) return `${n.toFixed(1)}亿`;
    return `${n.toFixed(2)}亿`;
  };
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  // 命中形态标色(实测有效高亮)：P3 正向买入→绿；P11 追突破=接盘→橙(同突破阶段色)；P19/P20 出货→红；其余中性
  const patTagClass = (code) =>
    (code === "P3" ? "pg" : code === "P11" ? "po" : (code === "P19" || code === "P20") ? "pr" : "pd");
  // 有效形态的悬停说明（与 PATTERN_EFFECTIVE 一致）
  const PAT_TIP = {
    P3: "★实测有效·吸筹买入：剔大盘各周期正、40日+4%",
    P11: "★实测有效·追突破=接盘：全周期显著负、回避不宜追",
    P19: "★实测有效·出货风控：巨量大阴，显著负",
    P20: "★实测有效·出货风控：均线放量破位，显著负",
  };
  const SIG_ACCENT = { buy: "sg", hold: "sh", sell: "ss" };
  // 冲突：吸筹分仍高却命中出货预警 → 多为低位放量破位(P20)，结构似吸筹但已破位，仍偏空
  const isBreakdownConflict = (s) =>
    String((s && s.pattern_phase) || "").includes("出货") && (Number(s && s.ambush_score) || 0) >= 50;
  const CONFLICT_TIP = "吸筹分仍高却出货预警：多为低位放量破位(P20)，回测属下跌中继/失败反弹(40日约−1.4%,t−2.0)，非高位派发，仍偏空回避";
  const KLINE_PERIOD_LABEL = { day: "日K", week: "周K", month: "月K" };
  const KLINE_DEFAULT_VISIBLE = { day: 140, week: 96, month: 72 };
  const KLINE_MIN_VISIBLE = { day: 24, week: 18, month: 12 };
  const KLINE_HISTORY_LIMIT = 3600;

  function initRadar() {
    const root = document.querySelector("#radar-dashboard");
    if (!root || root.dataset.radarInit === "1") return;
    root.dataset.radarInit = "1";

    let disposed = false;
    let payload = null;
    let stocks = [];
    let selectedCode = null;
    let klineBars = [];
    let klineLayout = null;
    let hoverIdx = null;
    let klinePeriod = "day";
    let klineView = { start: 0, end: 0 };
    let klineDrag = null;
    let klineFetchSeq = 0;
    let pollTimer = null;
    let activeJobType = "";                                  // "run" | "data"，用于日志面板归属
    let sortState = { key: "opportunity_score", dir: "desc" }; // 默认按机会分：吸筹/出货百分位折扣
    let patternMap = {};                                    // code -> {name,category,signal,desc,effective}
    const $ = (id) => document.getElementById(id);

    function normalizePayload(raw) {
      const out = raw || {};
      out.stocks = Array.isArray(out.stocks) ? out.stocks.map((s) => {
        const accPct = scoreValue(s, "accumulation_percentile", "ambush_score");
        const distPct = scoreValue(s, "distribution_percentile", "distribution_score");
        return {
          ...s,
          accumulation_percentile: accPct,
          distribution_percentile: distPct,
          pattern_phase: normalizePhase(s && s.pattern_phase),
        };
      }) : [];
      const nextCounts = {};
      Object.entries(out.phase_counts || {}).forEach(([phase, count]) => {
        const key = normalizePhase(phase);
        nextCounts[key] = (nextCounts[key] || 0) + (Number(count) || 0);
      });
      out.phase_counts = nextCounts;
      return out;
    }

    async function fetchPatternCatalog() {
      try {
        const resp = await fetch("/api/radar/patterns");
        const data = await resp.json();
        (data.patterns || []).forEach((p) => { patternMap[p.code] = p; });
      } catch (err) { /* 解释模块降级为只显示编号 */ }
    }

    // ---------- 数据加载 ----------
    async function fetchData() {
      try {
        const resp = await fetch("/api/radar/data");
        const data = await resp.json();
        if (disposed) return null;
        payload = normalizePayload(data.payload || {});
        stocks = payload.stocks;
        renderMeta();
        renderPhaseFilter();
        renderPhaseChips();
        applyFilters();
        reflectJobs(data.run_job, data.data_job);
        return data;
      } catch (err) {
        $("radar-status").textContent = "数据读取失败";
        return null;
      }
    }

    function renderMeta() {
      const p = payload || {};
      if (!p.generated_at) {
        $("radar-status").textContent = "尚无结果";
        $("radar-meta").textContent = "还没有结果文件，点「运行 · 刷新结果」生成。";
        return;
      }
      $("radar-status").textContent = "就绪";
      const cap = p.max_market_cap_yi;
      // 「含大盘」勾选框同步到当前结果口径（null=全市值=勾上）；程序化赋值不触发 change，不会误触重跑
      $("radar-include-large").checked = (cap === null || cap === undefined);
      if (p.pool) $("radar-pool").value = p.pool;
      const pool = p.pool === "hotmoney"
        ? "游资小盘池(龙虎榜活跃·流通≤100亿·反转分排序)"
        : (cap ? `≤${cap}亿小中盘龙头` : "全市值龙头(含大盘)");
      const asOf = p.as_of ? ` · 数据截至 ${p.as_of}(历史复盘)` : "";
      const theme = (p.theme_source && p.theme_source.available)
        ? `题材热度 ${p.theme_source.generated_at || ""}${p.theme_source.stale ? " ⚠️偏旧" : ""}`
        : "题材热度: 缺/未挂载";
      const cc = p.capital_counts || {};
      const capLine = p.capital_available
        ? `<span>资金面: 户数降<strong>${cc.holder_down ?? 0}</strong> · 回购<strong>${cc.repurchase ?? 0}</strong> · 上榜避雷<strong>${cc.lhb_avoid ?? 0}</strong> · 户数/回购已并入吸筹总分</span>`
        : `<span>资金面: 未挂载（跑 stock_crawl_holders / stock_crawl_capital 后生效）</span>`;
      const mr = p.market_regime;
      const regimeLine = (mr && mr.available)
        ? `<span>大盘 <strong>${mr.above_ma20 ? "强势(>MA20)·反转分可做多" : "弱势(<MA20)·做多易接刀"}</strong></span>`
        : "";
      $("radar-meta").innerHTML =
        `<span>生成 <strong>${esc(p.generated_at)}</strong>${asOf}</span>` +
        `<span>候选 <strong>${p.candidate_count ?? "-"}</strong>（${esc(pool)}）</span>` +
        `<span>已打分 <strong>${p.scored_count ?? "-"}</strong></span>` +
        `<span>${esc(theme)}</span>` + capLine + regimeLine;
    }

    function orderedPhases() {
      const counts = (payload && payload.phase_counts) || {};
      const known = PHASE_ORDER.filter((ph) => counts[ph]);
      const extra = Object.keys(counts).filter((ph) => !PHASE_ORDER.includes(ph));
      return known.concat(extra);
    }

    function renderPhaseChips() {
      const counts = (payload && payload.phase_counts) || {};
      const chips = orderedPhases().map((ph) =>
        `<button type="button" class="phase-chip pc-${phaseClass(ph)}" data-phase="${esc(ph)}">${esc(ph)} ${counts[ph]}</button>`
      ).join("");
      $("radar-phases").innerHTML = chips || "<span class='status'>无阶段分布</span>";
      $("radar-phases").querySelectorAll(".phase-chip").forEach((btn) => {
        btn.onclick = () => {
          const sel = $("radar-phase-filter");
          sel.value = sel.value === btn.dataset.phase ? "" : btn.dataset.phase;
          applyFilters();
        };
      });
    }

    function renderPhaseFilter() {
      const sel = $("radar-phase-filter");
      const cur = sel.value;
      const opts = ['<option value="">全部阶段</option>']
        .concat(orderedPhases().map((ph) => `<option value="${esc(ph)}">${esc(ph)}</option>`));
      sel.innerHTML = opts.join("");
      if (cur && orderedPhases().includes(cur)) sel.value = cur;
    }

    // ---------- 筛选 + 表格 ----------
    function currentFilters() {
      return {
        q: ($("radar-search").value || "").trim().toLowerCase(),
        phase: $("radar-phase-filter").value || "",
        minScore: Number($("radar-min-score").value || 0),
        hideDist: $("radar-hide-dist").checked,
      };
    }

    function applyFilters() {
      const f = currentFilters();
      const list = stocks.filter((s) => {
        if (f.phase && s.pattern_phase !== f.phase) return false;
        if (f.hideDist && String(s.pattern_phase || "").includes("出货")) return false;
      if ((Number(s.ambush_score) || 0) < f.minScore) return false;
        if (f.q) {
          const hay = `${s.code} ${s.name || ""} ${s.tracking_theme || ""} ${s.parent_segment || ""} ${s.segment_name || ""}`.toLowerCase();
          if (!hay.includes(f.q)) return false;
        }
        return true;
      });
      if (sortState.key) {
        const k = sortState.key;
        const mul = sortState.dir === "asc" ? 1 : -1;
        list.sort((a, b) => ((Number(a[k]) || 0) - (Number(b[k]) || 0)) * mul);
      }
      $("radar-count").textContent = `${list.length} / ${stocks.length} 只`;
      renderTable(list);
    }

    function exportTopOpportunityStocks() {
      const limitInput = $("radar-export-limit");
      const limit = Math.max(1, Math.floor(Number(limitInput.value) || 10));
      limitInput.value = String(limit);
      const rows = stocks
        .filter((s) => !Number.isNaN(Number(s.opportunity_score)))
        .slice()
        .sort((a, b) =>
          (Number(b.opportunity_score) - Number(a.opportunity_score)) ||
          (Number(b.ambush_score) - Number(a.ambush_score)) ||
          String(a.code || "").localeCompare(String(b.code || ""))
        )
        .slice(0, limit);
      if (!rows.length) {
        $("radar-status").textContent = "暂无可导出股票";
        return;
      }
      const cleanField = (value) => String(value ?? "").replace(/[\r\n]+/g, " ").trim();
      const text = rows.map((s) => `${cleanField(s.code)},${cleanField(s.name)},`).join("\n") + "\n";
      const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      const today = new Date().toISOString().slice(0, 10).replace(/-/g, "");
      anchor.href = url;
      anchor.download = `hot_money_radar_opportunity_top${limit}_${today}.txt`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      $("radar-status").textContent = `已导出机会分Top${rows.length}`;
    }

    function themeCell(s) {
      const t = s.tracking_theme || "";
      if (!t) return "<span class='dim'>-</span>";
      const heat = (s.theme_heat_pctile === null || s.theme_heat_pctile === undefined)
        ? "" : ` <small>热${Math.round(s.theme_heat_pctile)}</small>`;
      return esc(t) + heat;
    }

    function sw2Cell(s) {
      const industry = s.parent_segment || "";
      if (!industry) return "<span class='dim'>-</span>";
      const heat = (s.sw2_heat_pctile === null || s.sw2_heat_pctile === undefined)
        ? "" : ` <small>热${Math.round(s.sw2_heat_pctile)}</small>`;
      return esc(industry) + heat;
    }

    function renderTable(list) {
      if (!list.length) {
        $("radar-table").innerHTML = "<div class='radar-empty'>无匹配结果</div>";
        return;
      }
      const sortTh = (key, label) => {
        const active = sortState.key === key;
        const arrow = active ? (sortState.dir === "asc" ? "▲" : "▼") : "↕";
        return `<th class="num sortable${active ? " active" : ""}" data-key="${key}">${label} <span class="sort-arrow">${arrow}</span></th>`;
      };
      const head = `<tr>
        <th>#</th><th>代码</th><th>名称</th>${sortTh("market_cap_yi", "市值")}${sortTh("opportunity_score", "机会分")}${sortTh("ambush_score", "吸筹分")}${sortTh("distribution_score", "出货分")}
        <th class="num">量比</th><th class="num">价分位</th><th class="num">换手分位</th><th class="num">CMF</th><th class="num">筹码</th>
        <th class="num">连板</th><th>命中形态</th><th>阶段·把握</th><th>走势相似行业</th><th>二级行业</th><th>依据</th>
      </tr>`;
      const rows = list.map((s, i) => {
        const sig = s.signals || {};
        const cls = phaseClass(s.pattern_phase);
        const conf = (s.phase_confidence === null || s.phase_confidence === undefined) ? "" : ` <small>把握${Math.round(s.phase_confidence)}</small>`;
        const sel = s.code === selectedCode ? " is-selected" : "";
        const patCell = (s.patterns && s.patterns.length)
          ? s.patterns.map((c) => `<span class="pat ${patTagClass(c)}"${PAT_TIP[c] ? ` title="${esc(PAT_TIP[c])}"` : ""}>${esc(c)}</span>`).join("")
          : "<span class='dim'>-</span>";
        const evCell = (s.evidence && s.evidence.length)
          ? s.evidence.map((e) => {
              const lab = (typeof e === "string") ? e : (e.label || "");
              const kind = (typeof e === "string") ? "" : (e.kind || "");
              const cls = kind === "bullish" ? "ev-bull" : kind === "bearish" ? "ev-warn" : "ev-neutral";
              return `<span class="ev ${cls}">${esc(lab)}</span>`;
            }).join("")
          : "<span class='dim'>-</span>";
        return `<tr class="rrow${sel}" data-code="${esc(s.code)}">
          <td class="dim">${i + 1}</td>
          <td class="mono">${esc(s.code)}</td>
          <td>${esc(s.name || "")}</td>
          <td class="num">${fmtMarketCap(s.market_cap_yi)}</td>
          <td class="num strong">${fmt(s.opportunity_score, 1)}</td>
          <td class="num" title="机会分计算用吸筹分位 ${fmt(s.accumulation_percentile, 1)}">${fmt(s.ambush_score, 1)}</td>
          <td class="num" title="机会分计算用出货分位 ${fmt(s.distribution_percentile, 1)}">${fmt(s.distribution_score, 1)}</td>
          <td class="num">${fmt(sig.vol_ratio, 2)}</td>
          <td class="num">${fmt(sig.close_pctile, 2)}</td>
          <td class="num${(sig.turnover_pctile != null && sig.turnover_pctile > 0.84) ? " hot-turnover" : ""}">${fmt(sig.turnover_pctile, 2)}</td>
          <td class="num">${fmt(sig.cmf, 3)}</td>
          <td class="num">${fmt(sig.chip_concentration, 2)}</td>
          <td class="num">${sig.limit_streak ?? 0}</td>
          <td class="pats mono">${patCell}</td>
          <td><span class="phase phase-${cls}">${esc(s.pattern_phase || "")}</span>${conf}${isBreakdownConflict(s) ? ` <span class="pi-warn" title="${esc(CONFLICT_TIP)}">⚠破位</span>` : ""}</td>
          <td>${themeCell(s)}</td>
          <td>${sw2Cell(s)}</td>
          <td class="ev-cell">${evCell}</td>
        </tr>`;
      }).join("");
      $("radar-table").innerHTML = `<table class="radar-grid"><thead>${head}</thead><tbody>${rows}</tbody></table>`;
      $("radar-table").querySelectorAll(".rrow").forEach((tr) => {
        tr.onclick = () => selectStock(tr.dataset.code);
      });
      $("radar-table").querySelectorAll("th.sortable").forEach((th) => {
        th.onclick = () => {
          const k = th.dataset.key;
          if (sortState.key === k) sortState.dir = sortState.dir === "asc" ? "desc" : "asc";
          else sortState = { key: k, dir: "desc" };
          applyFilters();
        };
      });
    }

    // ---------- K线 ----------
    async function selectStock(code) {
      selectedCode = code;
      $("radar-table").querySelectorAll(".rrow").forEach((tr) =>
        tr.classList.toggle("is-selected", tr.dataset.code === code));
      const s = stocks.find((x) => x.code === code) || {};
      $("radar-kline-title").textContent = `${code} ${s.name || ""} · 市值 ${fmtMarketCap(s.market_cap_yi)}`;
      $("radar-kline-sub").textContent = "加载K线…";
      renderPatternInfo(s);
      await loadKline(code);
    }

    function setKlinePeriod(period) {
      if (!KLINE_PERIOD_LABEL[period] || period === klinePeriod) return;
      klinePeriod = period;
      $("radar-kline-periods").querySelectorAll("button").forEach((btn) =>
        btn.classList.toggle("is-active", btn.dataset.period === period));
      if (selectedCode) loadKline(selectedCode);
      else drawKline();
    }

    async function loadKline(code) {
      const seq = ++klineFetchSeq;
      $("radar-kline-sub").textContent = `加载${KLINE_PERIOD_LABEL[klinePeriod]}…`;
      try {
        const url = `/api/radar/kline?code=${encodeURIComponent(code)}&period=${klinePeriod}&limit=${KLINE_HISTORY_LIMIT}`;
        const resp = await fetch(url);
        const data = await resp.json();
        if (disposed || selectedCode !== code || seq !== klineFetchSeq) return;
        klineBars = (data.bars || []).map((b) => ({
          date: b.date,
          startDate: b.start_date || b.date,
          open: +b.open,
          high: +b.high,
          low: +b.low,
          close: +b.close,
          volume: +b.volume || 0,
        }));
        hoverIdx = null;
        resetKlineView();
        updateKlineSub();
        drawKline();
        updateKlineDetail(klineBars.length ? klineBars.length - 1 : null);
      } catch (err) {
        $("radar-kline-sub").textContent = "K线加载失败";
      }
    }

    function renderPatternInfo(s) {
      const el = $("radar-pattern-info");
      if (!s || !s.code) { el.innerHTML = "点击左侧个股查看其命中形态的含义。"; return; }
      const codes = s.patterns || [];
      const meta = `<div class="pi-stock-meta"><span>市值 ${esc(fmtMarketCap(s.market_cap_yi))}</span></div>`;
      let html;
      if (!codes.length) {
        html = meta + `<div class="pi-empty">${esc(s.name || s.code)} 未命中任何形态，当前阶段：${esc(s.pattern_phase || "")}。</div>`;
      } else {
        html = meta + codes.map((c) => {
          const m = patternMap[c] || { name: c, category: "", signal: "", desc: "" };
          const acc = SIG_ACCENT[m.signal] || "sd";
          const eff = m.effective ? `<span class="pi-eff">★实测有效</span>` : "";
          const cat = [m.category, m.signal].filter(Boolean).join(" · ");
          return `<div class="pi-card ${acc}">
            <div class="pi-top">
              <span class="pat ${patTagClass(c)}">${esc(c)}</span>
              <strong>${esc(m.name)}</strong>
              <span class="pi-cat">${esc(cat)}</span>${eff}
            </div>
            <div class="pi-desc">${esc(m.desc) || "（无说明）"}</div>
          </div>`;
        }).join("");
      }
      if (isBreakdownConflict(s)) {
        html = `<div class="pi-conflict">⚠ 结构似吸筹（吸筹分 ${fmt(s.ambush_score, 0)}）却触发出货预警——` +
          `多为<strong>低位放量破位(P20)</strong>，回测属下跌中继/失败反弹（40日约 −1.4%，t−2.0），非高位派发，仍偏空回避。</div>` + html;
      }
      const inval = s.invalidations || [];
      if (inval.length) {
        html += `<div class="pi-inval"><span class="pi-inval-h">证伪 / 止损</span>` +
          inval.map((x) => `<span>${esc(x)}</span>`).join("") + `</div>`;
      }
      el.innerHTML = html;
    }

    function updateKlineDetail(idx) {
      const el = $("radar-kline-detail");
      if (idx === null || !klineBars[idx]) { el.textContent = ""; return; }
      const b = klineBars[idx];
      const prev = idx > 0 ? klineBars[idx - 1].close : b.open;
      const chg = prev ? (b.close / prev - 1) * 100 : 0;
      const up = chg >= 0;
      const dateText = b.startDate && b.startDate !== b.date ? `${b.startDate}~${b.date}` : b.date;
      el.innerHTML =
        `<span>${esc(dateText)}</span>` +
        `<span>开 ${fmt(b.open, 2)}</span><span>高 ${fmt(b.high, 2)}</span>` +
        `<span>低 ${fmt(b.low, 2)}</span><span>收 ${fmt(b.close, 2)}</span>` +
        `<span class="${up ? "up" : "down"}">${up ? "+" : ""}${fmt(chg, 2)}%</span>` +
        `<span>量 ${(b.volume / 1e4).toFixed(0)}万</span>`;
    }

    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

    function clampKlineView(start, end) {
      const total = klineBars.length;
      if (!total) return { start: 0, end: 0 };
      const minVisible = Math.min(total, KLINE_MIN_VISIBLE[klinePeriod] || 20);
      let span = Math.round(end - start);
      span = clamp(span, minVisible, total);
      start = Math.round(start);
      end = start + span;
      if (start < 0) { start = 0; end = span; }
      if (end > total) { end = total; start = total - span; }
      return { start, end };
    }

    function setKlineView(start, end) {
      klineView = clampKlineView(start, end);
      if (hoverIdx !== null) hoverIdx = clamp(hoverIdx, klineView.start, Math.max(klineView.start, klineView.end - 1));
      updateKlineSub();
      drawKline();
    }

    function resetKlineView() {
      const total = klineBars.length;
      if (!total) { klineView = { start: 0, end: 0 }; return; }
      const visible = Math.min(total, KLINE_DEFAULT_VISIBLE[klinePeriod] || 120);
      klineView = { start: total - visible, end: total };
    }

    function barRangeLabel(bar, compact = false) {
      if (!bar) return "";
      const a = String(bar.startDate || bar.date || "");
      const b = String(bar.date || "");
      const aa = compact ? a.slice(2) : a;
      const bb = compact ? b.slice(2) : b;
      return a && b && a !== b ? `${aa}~${bb}` : bb;
    }

    function updateKlineSub() {
      const total = klineBars.length;
      if (!total) {
        $("radar-kline-sub").textContent = "无数据";
        return;
      }
      const first = klineBars[klineView.start];
      const last = klineBars[Math.max(klineView.start, klineView.end - 1)];
      const visible = Math.max(0, klineView.end - klineView.start);
      $("radar-kline-sub").textContent =
        `${KLINE_PERIOD_LABEL[klinePeriod]} · ${barRangeLabel(first, true)} 至 ${barRangeLabel(last, true)} · ${visible}/${total}根`;
    }

    function drawKline() {
      const canvas = $("radar-kline-canvas");
      if (!canvas) return;
      const box = canvas.parentElement;
      const cs = getComputedStyle(box);
      const inner = box.clientWidth - parseFloat(cs.paddingLeft || 0) - parseFloat(cs.paddingRight || 0);
      const cssW = Math.max(280, Math.round(inner));
      const cssH = Math.max(340, Math.round(cssW * 0.76));
      const dpr = window.devicePixelRatio || 1;
      canvas.style.width = cssW + "px";
      canvas.style.height = cssH + "px";
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);
      ctx.font = "11px -apple-system,Segoe UI,sans-serif";
      if (!klineBars.length) {
        ctx.fillStyle = "#8a93a6";
        ctx.fillText("暂无K线数据", 12, 24);
        klineLayout = null;
        return;
      }
      klineView = clampKlineView(klineView.start, klineView.end);
      const visibleBars = klineBars.slice(klineView.start, klineView.end);
      const padL = 48, padR = 10, padT = 10, axisB = 18, gap = 10, miniGap = 24, miniH = 46;
      const n = visibleBars.length;
      const plotW = cssW - padL - padR;
      const chartH = cssH - padT - axisB - miniH - miniGap;
      const priceH = chartH * 0.70;
      const volH = chartH - priceH - gap;
      const step = plotW / n;
      const bw = Math.max(1, Math.min(11, step * 0.66));
      let lo = Infinity, hi = -Infinity, vmax = 0;
      for (const b of visibleBars) { lo = Math.min(lo, b.low); hi = Math.max(hi, b.high); vmax = Math.max(vmax, b.volume); }
      const pad = (hi - lo) * 0.05 || 1; lo -= pad; hi += pad;
      const y = (p) => padT + (hi - p) / (hi - lo) * priceH;
      const volTop = padT + priceH + gap;
      const vy = (v) => volTop + (1 - v / (vmax || 1)) * volH;
      const volBottom = volTop + volH;
      const miniTop = cssH - axisB - miniH;

      // 价格网格 + 标签
      ctx.strokeStyle = "rgba(140,150,170,.18)";
      ctx.fillStyle = "#8a93a6";
      ctx.lineWidth = 1;
      for (let k = 0; k <= 4; k++) {
        const pv = hi - (hi - lo) * k / 4;
        const yy = y(pv);
        ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(cssW - padR, yy); ctx.stroke();
        ctx.fillText(pv.toFixed(2), 4, yy + 3);
      }
      // 蜡烛 + 量
      for (let i = 0; i < n; i++) {
        const b = visibleBars[i];
        const cx = padL + step * i + step / 2;
        const up = b.close >= b.open;
        const col = up ? "#e23b3b" : "#1a9d5a";   // 红涨绿跌
        ctx.strokeStyle = col; ctx.fillStyle = col;
        ctx.beginPath(); ctx.moveTo(cx, y(b.high)); ctx.lineTo(cx, y(b.low)); ctx.stroke();
        const yo = y(b.open), yc = y(b.close);
        ctx.fillRect(cx - bw / 2, Math.min(yo, yc), bw, Math.max(1, Math.abs(yo - yc)));
        const vyy = vy(b.volume);
        ctx.fillRect(cx - bw / 2, vyy, bw, volBottom - vyy);
      }
      // 日期轴（首/中/末）
      ctx.fillStyle = "#8a93a6";
      const tickIndexes = Array.from(new Set([0, Math.floor(n / 3), Math.floor(n * 2 / 3), n - 1]));
      tickIndexes.forEach((i) => {
        const t = visibleBars[i] && barRangeLabel(visibleBars[i], true);
        if (!t) return;
        const cx = padL + step * i + step / 2;
        ctx.fillText(t, Math.min(cssW - padR - 58, Math.max(padL, cx - 24)), volBottom + 16);
      });

      drawMiniTimeline(ctx, padL, plotW, miniTop, miniH);

      klineLayout = {
        padL, padR, plotW, step,
        start: klineView.start, end: klineView.end, total: klineBars.length,
        main: { x0: padL, x1: cssW - padR, y0: padT, y1: volBottom },
        mini: { x0: padL, x1: cssW - padR, y0: miniTop, y1: miniTop + miniH },
      };
      if (hoverIdx !== null && hoverIdx >= klineView.start && hoverIdx < klineView.end) {
        const local = hoverIdx - klineView.start;
        const cx = padL + step * local + step / 2;
        ctx.strokeStyle = "rgba(120,140,200,.6)";
        ctx.beginPath(); ctx.moveTo(cx, padT); ctx.lineTo(cx, volBottom); ctx.stroke();
      }
    }

    function drawMiniTimeline(ctx, x0, w, y0, h) {
      const total = klineBars.length;
      if (!total) return;
      let lo = Infinity, hi = -Infinity;
      for (const b of klineBars) { lo = Math.min(lo, b.low); hi = Math.max(hi, b.high); }
      const range = hi - lo || 1;
      const x = (idx) => x0 + (total === 1 ? 0.5 : idx / (total - 1)) * w;
      const y = (p) => y0 + 4 + (hi - p) / range * (h - 8);
      ctx.fillStyle = "#f6f8fb";
      ctx.fillRect(x0, y0, w, h);
      ctx.strokeStyle = "rgba(120,130,150,.22)";
      ctx.strokeRect(x0, y0, w, h);
      ctx.beginPath();
      klineBars.forEach((b, i) => {
        const px = x(i), py = y(b.close);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.strokeStyle = "#5f7fd6";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      const sx = x(klineView.start);
      const ex = x(Math.max(klineView.start, klineView.end - 1));
      ctx.fillStyle = "rgba(28, 58, 120, .10)";
      ctx.fillRect(sx, y0, Math.max(3, ex - sx), h);
      ctx.fillStyle = "rgba(255,255,255,.72)";
      ctx.fillRect(x0, y0, Math.max(0, sx - x0), h);
      ctx.fillRect(ex, y0, Math.max(0, x0 + w - ex), h);
      ctx.fillStyle = "#315cbb";
      ctx.fillRect(sx - 2, y0, 4, h);
      ctx.fillRect(ex - 2, y0, 4, h);
    }

    function canvasPoint(ev) {
      const rect = ev.currentTarget.getBoundingClientRect();
      return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
    }

    function visibleIndexFromX(x) {
      if (!klineLayout || !klineBars.length) return;
      const local = Math.floor((x - klineLayout.padL) / klineLayout.step);
      return clamp(klineView.start + local, klineView.start, Math.max(klineView.start, klineView.end - 1));
    }

    function miniIndexFromX(x) {
      if (!klineLayout || !klineBars.length) return 0;
      const ratio = clamp((x - klineLayout.mini.x0) / (klineLayout.mini.x1 - klineLayout.mini.x0), 0, 1);
      return Math.round(ratio * (klineBars.length - 1));
    }

    function miniXForIndex(idx) {
      if (!klineLayout || klineBars.length <= 1) return klineLayout ? klineLayout.mini.x0 : 0;
      return klineLayout.mini.x0 + idx / (klineBars.length - 1) * (klineLayout.mini.x1 - klineLayout.mini.x0);
    }

    function onCanvasMove(ev) {
      if (!klineLayout || !klineBars.length) return;
      const { x, y } = canvasPoint(ev);
      if (klineDrag) {
        ev.preventDefault();
        if (klineDrag.type === "pan") {
          const dx = x - klineDrag.x;
          const shift = Math.round(-dx / klineLayout.step);
          setKlineView(klineDrag.start + shift, klineDrag.end + shift);
        } else if (klineDrag.type === "range") {
          const span = klineDrag.end - klineDrag.start;
          const center = miniIndexFromX(x) - klineDrag.offset;
          setKlineView(center, center + span);
        } else if (klineDrag.type === "start") {
          const idx = miniIndexFromX(x);
          setKlineView(idx, klineDrag.end);
        } else if (klineDrag.type === "end") {
          const idx = miniIndexFromX(x) + 1;
          setKlineView(klineDrag.start, idx);
        }
        return;
      }

      const inMain = y >= klineLayout.main.y0 && y <= klineLayout.main.y1 && x >= klineLayout.main.x0 && x <= klineLayout.main.x1;
      if (!inMain) return;
      const idx = visibleIndexFromX(x);
      if (idx === hoverIdx) return;
      hoverIdx = idx;
      updateKlineDetail(idx);
      drawKline();
    }

    function onCanvasDown(ev) {
      if (!klineLayout || !klineBars.length) return;
      const { x, y } = canvasPoint(ev);
      const canvas = ev.currentTarget;
      const inMini = y >= klineLayout.mini.y0 && y <= klineLayout.mini.y1 && x >= klineLayout.mini.x0 && x <= klineLayout.mini.x1;
      const inMain = y >= klineLayout.main.y0 && y <= klineLayout.main.y1 && x >= klineLayout.main.x0 && x <= klineLayout.main.x1;
      if (!inMini && !inMain) return;

      canvas.setPointerCapture(ev.pointerId);
      canvas.classList.add("is-dragging");
      if (inMini) {
        const sx = miniXForIndex(klineView.start);
        const ex = miniXForIndex(Math.max(klineView.start, klineView.end - 1));
        const idx = miniIndexFromX(x);
        if (Math.abs(x - sx) <= 8) klineDrag = { type: "start", start: klineView.start, end: klineView.end };
        else if (Math.abs(x - ex) <= 8) klineDrag = { type: "end", start: klineView.start, end: klineView.end };
        else {
          const span = klineView.end - klineView.start;
          const offset = x >= sx && x <= ex ? idx - klineView.start : Math.floor(span / 2);
          klineDrag = { type: "range", start: klineView.start, end: klineView.end, offset };
          setKlineView(idx - offset, idx - offset + span);
        }
      } else {
        klineDrag = { type: "pan", x, start: klineView.start, end: klineView.end };
      }
      ev.preventDefault();
    }

    function onCanvasUp(ev) {
      if (!klineDrag) return;
      klineDrag = null;
      ev.currentTarget.classList.remove("is-dragging");
      try { ev.currentTarget.releasePointerCapture(ev.pointerId); } catch (err) { /* pointer already released */ }
    }

    function onCanvasWheel(ev) {
      if (!klineLayout || !klineBars.length) return;
      ev.preventDefault();
      const total = klineBars.length;
      const count = klineView.end - klineView.start;
      const minVisible = Math.min(total, KLINE_MIN_VISIBLE[klinePeriod] || 20);
      const nextCount = clamp(Math.round(count * (ev.deltaY < 0 ? 0.82 : 1.22)), minVisible, total);
      const { x } = canvasPoint(ev);
      const anchor = visibleIndexFromX(x);
      const ratio = count > 1 ? (anchor - klineView.start) / count : 1;
      const nextStart = anchor - nextCount * ratio;
      setKlineView(nextStart, nextStart + nextCount);
    }

    // ---------- 任务（运行 / 刷新数据）----------
    function setBusy(busy) {
      $("radar-run").disabled = busy;
      $("radar-refresh-data").disabled = busy;
      $("radar-include-large").disabled = busy;
      $("radar-pool").disabled = busy;
    }

    function showJobLog(job, label) {
      const panel = $("radar-joblog");
      const lines = (job && job.log_lines) || [];
      if (!job || (!job.running && !lines.length)) { panel.hidden = true; return; }
      panel.hidden = false;
      $("radar-joblog-title").textContent = label + (job.running ? " · 运行中…" : (job.ok ? " · 完成" : " · 结束"));
      $("radar-joblog-count").textContent = `${lines.length} 行`;
      const body = $("radar-joblog-body");
      body.textContent = lines.slice(-200).join("\n") || "等待任务输出…";
      body.scrollTop = body.scrollHeight;
    }

    function reflectJobs(runJob, dataJob) {
      const running = (runJob && runJob.running) || (dataJob && dataJob.running);
      setBusy(running);
      if (dataJob && (dataJob.running || activeJobType === "data")) showJobLog(dataJob, "数据刷新");
      else if (runJob && (runJob.running || activeJobType === "run")) showJobLog(runJob, "运行");
      if (running && !pollTimer) pollTimer = setInterval(fetchData, 2000);
      if (!running && pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    }

    async function startJob(url, label, body) {
      setBusy(true);
      activeJobType = url.includes("refresh-data") ? "data" : "run";
      $("radar-status").textContent = label + "已启动…";
      try {
        const resp = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: body ? JSON.stringify(body) : null,
        });
        if (resp.status === 409) {
          const d = await resp.json();
          $("radar-status").textContent = d.error || "任务已在运行";
        }
      } catch (err) {
        $("radar-status").textContent = label + "启动失败";
      }
      await fetchData();
      if (!pollTimer) pollTimer = setInterval(fetchData, 2000);
    }

    // ---------- 绑定 ----------
    function boot() {
      const runBody = () => ({ include_large_cap: $("radar-include-large").checked, pool: $("radar-pool").value });
      $("radar-run").onclick = () => startJob("/api/radar/run", "运行", runBody());
      // 勾选/取消「含大盘」即自动重跑（市值口径变了，需要重新生成候选池）
      $("radar-include-large").addEventListener("change", () => {
        const scope = $("radar-include-large").checked ? "含大盘全市值" : "仅小中盘(剔除大盘)";
        startJob("/api/radar/run", `运行 · ${scope}`, runBody());
      });
      // 切换候选池(龙头/游资小盘)即自动重跑
      $("radar-pool").addEventListener("change", () => {
        const label = $("radar-pool").value === "hotmoney" ? "游资小盘池(反转分)" : "细分龙头池";
        startJob("/api/radar/run", `运行 · ${label}`, runBody());
      });
      $("radar-refresh-data").onclick = () => {
        if (!window.confirm("刷新数据将重爬全市场行情 + 板块 + 题材，大约需要 10–30 分钟，期间请勿关闭页面。\n\n确认现在开始吗？")) return;
        startJob("/api/radar/refresh-data", "数据刷新", null);
      };
      ["radar-search", "radar-phase-filter", "radar-min-score", "radar-hide-dist"].forEach((id) => {
        const el = $(id);
        el.addEventListener(el.tagName === "SELECT" || el.type === "checkbox" ? "change" : "input", applyFilters);
      });
      $("radar-export").onclick = exportTopOpportunityStocks;
      const canvas = $("radar-kline-canvas");
      $("radar-kline-periods").querySelectorAll("button").forEach((btn) => {
        btn.addEventListener("click", () => setKlinePeriod(btn.dataset.period));
      });
      canvas.addEventListener("pointerdown", onCanvasDown);
      canvas.addEventListener("pointermove", onCanvasMove);
      canvas.addEventListener("pointerup", onCanvasUp);
      canvas.addEventListener("pointercancel", onCanvasUp);
      canvas.addEventListener("wheel", onCanvasWheel, { passive: false });
      canvas.addEventListener("mouseleave", () => {
        if (klineDrag) return;
        hoverIdx = null;
        drawKline();
        updateKlineDetail(klineBars.length ? klineBars.length - 1 : null);
      });
      window.addEventListener("resize", onResize);
      fetchPatternCatalog();
      fetchData();
    }

    function onResize() { if (!disposed) drawKline(); }

    window.FinancialAnalysisPages = window.FinancialAnalysisPages || {};
    window.FinancialAnalysisPages.cleanup = () => {
      disposed = true;
      if (pollTimer) clearInterval(pollTimer);
      window.removeEventListener("resize", onResize);
    };
    boot();
  }

  window.FinancialAnalysisPages = window.FinancialAnalysisPages || {};
  window.FinancialAnalysisPages.radar = initRadar;
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initRadar, { once: true });
  } else {
    initRadar();
  }
})();
