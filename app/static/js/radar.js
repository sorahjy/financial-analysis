(() => {
  const PHASE_ORDER = [
    "疑似吸筹(待确认)🟢", "吸筹🟢", "试盘🟡", "洗盘🟡", "吸筹+洗盘🟡",
    "▲突破🟠", "拉升中🟠", "出货预警🔴", "观望⚪",
  ];

  function normalizePhase(label) {
    return String(label || "").replace("空仓观望", "观望");
  }

  const WARNING_COPY_OVERRIDES = new Map([
    ["近期上龙虎榜，长线按避雷处理", "近期上龙虎榜，波动加剧"],
    ["近期上龙虎榜(避雷)", "近期上龙虎榜，波动加剧"],
  ]);

  function normalizeEvidenceLabel(label) {
    return WARNING_COPY_OVERRIDES.get(String(label)) || String(label || "");
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
  const finiteNumber = (v) => (
    v === null || v === undefined || v === "" || !Number.isFinite(Number(v))
      ? Number.NaN
      : Number(v)
  );
  const fmtSignedPct = (v, d = 2) => {
    if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) return "-";
    const n = Number(v);
    return `${n > 0 ? "+" : ""}${n.toFixed(d)}%`;
  };
  const fmtSigned = (v, d = 2, forceNegativeZero = false) => {
    if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) return "-";
    const n = Number(v);
    if (n === 0 && forceNegativeZero) return `-${Math.abs(n).toFixed(d)}`;
    return `${n > 0 ? "+" : ""}${n.toFixed(d)}`;
  };
  const fmtRatioPct = (v) =>
    (v === null || v === undefined || v === "" || Number.isNaN(Number(v))
      ? "-"
      : `${(Number(v) * 100).toFixed(0)}%`);
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
  const SIG_ACCENT = { buy: "sg", hold: "sh", sell: "ss" };
  const PATTERN_BACKTEST_BUY_CODES = ["P1", "P2", "P3", "P5", "P21", "P23", "P24", "P25"];
  const DISTRIBUTION_WARNING_PATTERN_CODES = ["P14", "P15", "P16", "P17", "P19", "P20", "P22", "P26"];
  const PATTERN_SELL_PRESET_CODES = ["P11", ...DISTRIBUTION_WARNING_PATTERN_CODES];
  const PATTERN_BACKTEST_BUY_CODE_SET = new Set(PATTERN_BACKTEST_BUY_CODES);
  const PATTERN_SELL_PRESET_CODE_SET = new Set(PATTERN_SELL_PRESET_CODES);
  const NO_PATTERN_FILTER = "__none__";
  const patternRuleCodes = (rule, fallback) => {
    const source = Array.isArray(rule && rule.pattern_codes) ? rule.pattern_codes : fallback;
    return Array.from(new Set(source.map((code) => String(code || "").toUpperCase()).filter((code) => /^P\d+$/.test(code))))
      .sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
  };
  const patternAnyRuleText = (rule, fallback) => `${patternRuleCodes(rule, fallback).join("、")} 任一命中`;
  // 冲突：吸筹分仍高，但出货形态已达到统一预警门槛。
  const isDistributionConflict = (s) =>
    String((s && s.pattern_phase) || "").includes("出货") && (Number(s && s.ambush_score) || 0) >= 50;
  function distributionWarningRuleText(s) {
    const signals = (s && s.signals) || {};
    const rule = signals.distribution_warning_rule || {};
    return patternAnyRuleText(rule, DISTRIBUTION_WARNING_PATTERN_CODES);
  }
  const distributionConflictTip = (s) =>
    `吸筹分仍高，但已触发出货预警：${distributionWarningRuleText(s)}，仍应优先防守`;
  const KLINE_PERIOD_LABEL = { day: "日K", week: "周K", month: "月K" };
  const KLINE_DEFAULT_VISIBLE = { day: 140, week: 96, month: 72 };
  const KLINE_MIN_VISIBLE = { day: 24, week: 18, month: 12 };
  const KLINE_HISTORY_LIMIT = 0; // 0 = 该标的全部可用历史
  const PATTERN_MARKER_META = {
    bullish: { label: "买", color: "#34c759", placement: "below" },
    momentum: { label: "强", color: "#ff9f0a", placement: "below" },
    risk: { label: "卖", color: "#ff453a", placement: "above" },
  };
  const REALTIME_REFRESH_MS = 180000;
  const REALTIME_SOURCE_LABEL = { tencent_batch: "腾讯", sina_batch: "新浪", eastmoney_a_spot: "东财" };

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
    let loadedKlineCode = null;
    let loadedKlinePeriod = null;
    let patternBacktestHits = [];
    let patternBacktestMeta = null;
    let patternBacktestLoaded = false;
    let patternBacktestActive = false;
    let patternBacktestLoading = false;
    let patternBacktestError = "";
    let patternBacktestFetchSeq = 0;
    let klinePatternMarkers = new Map();
    let pollTimer = null;
    let realtimeTimer = null;
    let realtimeBusy = false;
    let radarRunBusy = false;
    let realtimeRefreshPending = false;
    let realtimeRefreshQueued = false;
    let jobsWereRunning = false;
    let dataJobWasRunning = false;
    let activeJobType = "";                                  // "run" | "data"，用于日志面板归属
    let sortState = { key: "opportunity_score", dir: "desc" }; // 默认按机会分：吸筹/出货百分位折扣
    let patternMap = {};                                    // code -> {name,category,signal,desc,effective}
    let patternCatalog = [];
    let scoringModel = null;
    let activeInsightModal = null;
    let insightReturnFocus = null;
    let industryHeatPayload = null;
    let industryHeatLoading = false;
    let industryHeatFetchSeq = 0;
    let sw2Options = [];
    let selectedSw2Industries = new Set();
    let patternOptions = [];
    let selectedPatterns = new Set();
    const $ = (id) => document.getElementById(id);

    function openInsightModal(id) {
      const modal = $(id);
      if (!modal) return;
      insightReturnFocus = document.activeElement;
      activeInsightModal = modal;
      modal.hidden = false;
      document.body.classList.add("radar-modal-open");
      const dialog = modal.querySelector("[role='dialog']");
      if (dialog) dialog.focus();
    }

    function closeInsightModal() {
      if (!activeInsightModal) return;
      activeInsightModal.hidden = true;
      activeInsightModal = null;
      document.body.classList.remove("radar-modal-open");
      if (insightReturnFocus && typeof insightReturnFocus.focus === "function") insightReturnFocus.focus();
      insightReturnFocus = null;
    }

    function industryHeatSparkline(row) {
      const daily = Array.isArray(row && row.daily) ? row.daily : [];
      const values = daily.map((point) => Number(point && point.heat_index));
      if (values.length < 2 || values.some((value) => !Number.isFinite(value))) return '<span class="industry-heat-no-chart">数据不足</span>';
      const width = 156;
      const height = 42;
      const pad = 3;
      const low = Math.min(...values);
      const high = Math.max(...values);
      const span = high - low || 1;
      const points = values.map((value, index) => {
        const x = pad + index / (values.length - 1) * (width - pad * 2);
        const y = height - pad - (value - low) / span * (height - pad * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(" ");
      const last = values[values.length - 1];
      const label = `${row.segment_name || row.segment_code}近20日热度指数，首5日均值100，最新${last.toFixed(1)}`;
      return `<svg class="industry-heat-sparkline" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(label)}"><title>${esc(label)}</title><line x1="3" y1="${(height / 2).toFixed(1)}" x2="153" y2="${(height / 2).toFixed(1)}"></line><polyline points="${points}"></polyline></svg>`;
    }

    function industryHeatCapText(row) {
      const cap = fmtMarketCap(row && row.market_cap_yi);
      if (cap === "-") return "成分市值暂无";
      const prefix = row.market_cap_is_estimate ? "成分市值约" : "成分总市值";
      return `${prefix}${cap} · 覆盖${fmt(row.market_cap_coverage_pct, 1)}%`;
    }

    const INDUSTRY_HEAT_SOURCE_LABELS = Object.freeze({
      sws_trend: "申万指数日频",
      sws_analysis_share_estimate: "申万成交份额反推",
      akshare_stock_zh_a_spot_component: "AkShare·新浪全A成分汇总",
      akshare_sina_spot_component_sum: "AkShare·新浪全A成分汇总",
      mixed: "多源混合",
    });

    function industryHeatSourceLabel(source) {
      const key = String(source || "").trim();
      return INDUSTRY_HEAT_SOURCE_LABELS[key] || key;
    }

    function industryHeatLatestSource(report) {
      const quality = report && report.data_quality || {};
      const explicitSources = Array.isArray(quality.latest_amount_sources)
        ? quality.latest_amount_sources.map(industryHeatSourceLabel).filter(Boolean)
        : [];
      if (explicitSources.length) return Array.from(new Set(explicitSources)).join("、");
      const explicit = quality.latest_amount_source || quality.latest_source;
      if (explicit) return industryHeatSourceLabel(explicit);
      const targetDate = String(report && report.as_of_date || "");
      const industries = Array.isArray(report && report.industries) ? report.industries : [];
      for (const row of industries) {
        const daily = Array.isArray(row && row.daily) ? row.daily : [];
        const point = daily.find((item) => String(item && item.date || "") === targetDate);
        if (point && point.amount_source) return industryHeatSourceLabel(point.amount_source);
      }
      return "未标注";
    }

    function industryHeatSourceSummary(report) {
      const industries = Array.isArray(report && report.industries) ? report.industries : [];
      const sources = new Set();
      industries.forEach((row) => {
        const rowSources = Array.isArray(row && row.amount_sources) ? row.amount_sources : [];
        rowSources.forEach((source) => {
          const label = industryHeatSourceLabel(source);
          if (label) sources.add(label);
        });
      });
      return sources.size ? Array.from(sources).join("、") : industryHeatLatestSource(report);
    }

    function industryHeatAmountNote(row) {
      const notes = [];
      if (row.amount_is_estimate) notes.push(`成交额含${row.amount_estimated_days || 0}日估算`);
      const daily = Array.isArray(row && row.daily) ? row.daily : [];
      const inferredDerivedDays = daily.filter((point) => {
        const source = String(point && point.amount_source || "");
        return Boolean(point && point.amount_is_derived) || source.startsWith("akshare_");
      }).length;
      const derivedDays = Number(row && row.amount_derived_days) || inferredDerivedDays;
      if (row.amount_is_derived || derivedDays > 0) notes.push(`成交额含${derivedDays || 1}日成分汇总`);
      const derivedCoverage = Number(row && row.latest_amount_member_coverage_pct);
      if (derivedDays > 0 && Number.isFinite(derivedCoverage) && derivedCoverage < 99.995) {
        notes.push(`最新汇总覆盖${fmt(derivedCoverage, 1)}%`);
      }
      return notes.length ? ` · ${notes.join(" · ")}` : "";
    }

    function industryHeatRow(row, kind, maxMetric) {
      const hot = kind === "hottest";
      const trendScore = finiteNumber(row.trend_score);
      const legacyRisingScore = finiteNumber(row.rising_score);
      const shareChange = finiteNumber(row.last5_vs_first5_share_pct);
      const falling = !hot && (
        row.direction === "falling"
        || row.is_falling_candidate === true
        || (Number.isFinite(trendScore) && trendScore < 0)
        || (!Number.isFinite(trendScore) && Number.isFinite(shareChange) && shareChange < 0)
      );
      const trendMetric = Number.isFinite(trendScore)
        ? trendScore
        : (Number.isFinite(legacyRisingScore) ? legacyRisingScore : shareChange);
      const metric = hot ? Number(row.amount_20d_yi) : Math.abs(trendMetric);
      const barWidth = Number.isFinite(metric) && maxMetric > 0
        ? Math.max(2, Math.min(100, metric / maxMetric * 100))
        : 0;
      const amountPrefix = row.amount_is_estimate ? "约" : "";
      const primary = hot
        ? `${amountPrefix}${fmtMarketCap(row.avg_daily_amount_yi)}/日`
        : fmtSignedPct(row.last5_vs_first5_share_pct, 1);
      const secondary = hot
        ? `20日 ${amountPrefix}${fmtMarketCap(row.amount_20d_yi)}`
        : (Number.isFinite(trendScore)
          ? `趋势分 ${fmtSigned(trendScore, 1, falling)}`
          : `20日相关 ${fmtSigned(row.trend_correlation, 2)}`);
      const detail = hot
        ? `最新 ${fmtMarketCap(row.latest_amount_yi)} · 份额 ${fmt(row.avg_daily_market_share_pct, 2)}%`
        : `末5日日均 ${fmtMarketCap(row.last5_avg_amount_yi)} · 相关 ${fmt(row.trend_correlation, 2)}`;
      const rowClass = hot ? "is-hot" : (falling ? "is-falling" : "is-rising");
      return `<article class="industry-heat-row ${rowClass}">
        <span class="industry-heat-rank">${esc(row.rank)}</span>
        <div class="industry-heat-name"><strong>${esc(row.segment_name || "未命名行业")}</strong><small>${esc(row.segment_code)} · ${esc(row.parent_segment || "-")}</small><div class="industry-heat-bar" aria-hidden="true"><i style="width:${barWidth.toFixed(1)}%"></i></div><em>${esc(industryHeatCapText(row))}${esc(industryHeatAmountNote(row))}</em></div>
        <div class="industry-heat-numbers"><b>${esc(primary)}</b><span>${esc(secondary)}</span><small>${esc(detail)}</small></div>
        <div class="industry-heat-chart">${industryHeatSparkline(row)}</div>
      </article>`;
    }

    function industryHeatRankedRows(report, rankingKey, rankField) {
      const industries = Array.isArray(report && report.industries) ? report.industries : [];
      const byCode = new Map(
        industries.map((row) => [String(row && row.segment_code || ""), row])
      );
      const refs = report && report.rankings && Array.isArray(report.rankings[rankingKey])
        ? report.rankings[rankingKey]
        : [];
      const joined = refs.map((ref) => {
        const row = byCode.get(String(ref && ref.segment_code || ""));
        return row ? { ...row, ...ref } : null;
      }).filter(Boolean);
      const source = joined.length
        ? joined
        : industries.filter((row) => {
            const rank = row && row[rankField];
            return rank !== null && rank !== undefined
              && Number.isFinite(Number(rank)) && Number(rank) > 0;
          });
      return source.map((row) => ({
        ...row,
        rank: Number(row.rank || row[rankField]),
      })).sort((left, right) => left.rank - right.rank);
    }

    function industryHeatTrendRows(report) {
      const ranked = industryHeatRankedRows(report, "trend", "trend_rank");
      if (ranked.length) return ranked;

      // v1/v2 报告没有合并趋势榜：保留原升温次序，再追加双负下降行业，
      // 让旧的原子快照也能立即展示负数；刷新后由后端 trend_rank 接管排序。
      const industries = Array.isArray(report && report.industries) ? report.industries : [];
      const rising = industryHeatRankedRows(report, "rising", "rising_rank").map((row) => ({
        ...row,
        trend_score: row.rising_score,
      }));
      const seen = new Set(rising.map((row) => String(row.segment_code || "")));
      const falling = industries.filter((row) => {
        if (seen.has(String(row && row.segment_code || ""))) return false;
        const growth = finiteNumber(row && row.last5_vs_first5_share_pct);
        const correlation = finiteNumber(row && row.trend_correlation);
        return Number.isFinite(growth) && growth < 0
          && Number.isFinite(correlation) && correlation < 0;
      }).sort((left, right) => {
        const growthOrder = Number(right.last5_vs_first5_share_pct) - Number(left.last5_vs_first5_share_pct);
        if (growthOrder) return growthOrder;
        const trendOrder = Number(right.trend_correlation) - Number(left.trend_correlation);
        if (trendOrder) return trendOrder;
        return String(left.segment_code || "").localeCompare(String(right.segment_code || ""));
      });
      return [...rising, ...falling].map((row, index) => ({ ...row, rank: index + 1 }));
    }

    function renderIndustryHeat(report) {
      const content = $("radar-industry-heat-content");
      if (!content) return;
      if (!report || !report.available) {
        const message = (report && report.error) || "尚未生成三级行业热度报告，请先刷新数据。";
        content.innerHTML = `<div class="industry-heat-state is-empty"><strong>暂无可展示报告</strong><p>${esc(message)}</p><button id="radar-industry-heat-retry" class="btn" type="button">重新读取</button></div>`;
        const retry = $("radar-industry-heat-retry");
        if (retry) retry.onclick = () => fetchIndustryHeat(true);
        return;
      }
      const hottest = industryHeatRankedRows(report, "hottest", "hottest_rank");
      const trend = industryHeatTrendRows(report);
      if (!hottest.length) {
        renderIndustryHeat({ available: false, error: "报告全量行业排名不完整，请重新刷新数据。" });
        return;
      }
      const quality = report.data_quality || {};
      const windowMeta = report.window || {};
      const methodology = report.methodology || {};
      const latestAmountSource = industryHeatLatestSource(report);
      const amountSourceSummary = industryHeatSourceSummary(report);
      const hottestMax = Math.max(...hottest.map((row) => Number(row.amount_20d_yi) || 0), 1);
      const trendMax = Math.max(...trend.map((row) => {
        const score = finiteNumber(row.trend_score);
        if (Number.isFinite(score)) return Math.abs(score);
        const legacyScore = finiteNumber(row.rising_score);
        if (Number.isFinite(legacyScore)) return Math.abs(legacyScore);
        return Math.abs(finiteNumber(row.last5_vs_first5_share_pct) || 0);
      }), 1);
      const trendRows = trend.length
        ? trend.map((row) => industryHeatRow(row, "trend", trendMax)).join("")
        : '<div class="industry-heat-panel-empty">本窗口没有份额变化与20日趋势同向的行业。</div>';
      content.innerHTML = `<div class="industry-heat-summary" role="group" aria-label="行业热度报告摘要">
        <div><span>数据截至</span><strong>${esc(report.as_of_date || "-")}</strong><small>最新日：${esc(latestAmountSource)} · 生成 ${esc(report.generated_at || "-")}</small></div>
        <div><span>统一窗口</span><strong>${esc(windowMeta.trading_days || 20)}个交易日</strong><small>${esc(windowMeta.start_date || "-")} 至 ${esc(windowMeta.end_date || "-")}</small></div>
        <div><span>行业覆盖</span><strong>${esc(quality.eligible_segment_count || 0)} / ${esc(quality.expected_segment_count || 0)}</strong><small>${fmt(quality.eligible_coverage_pct, 1)}% · ${esc(quality.estimated_industry_count || 0)}行业含成交额估算</small></div>
        <div><span>市值成员覆盖</span><strong>${fmt(quality.market_cap_coverage_pct, 1)}%</strong><small>不完整行业标注“约”</small></div>
      </div>
      <aside class="industry-heat-method"><strong>口径与来源</strong><p>${esc(methodology.hottest_rank || "热门榜按20日成交额排序")}；${esc(methodology.trend_score || methodology.rising_score || "趋势榜按份额增减幅与20日趋势相关性排序，下降显示负分")}。成交额来源：${esc(amountSourceSummary)}；“成分汇总”表示按已校验的 SW3 成分聚合，不等同于申万指数直接值。热度是研究用活跃度，不代表收益概率。</p></aside>
      <div class="industry-heat-grid">
        <section class="industry-heat-panel" aria-labelledby="industry-heat-hottest-title"><header><div><span>Amount ranking</span><h3 id="industry-heat-hottest-title">近20日成交额全量排名 · ${hottest.length}个</h3></div><p>条形长度=20日成交额；折线=每日成交份额热度指数</p></header><div class="industry-heat-list" aria-label="近20日成交额全量行业排名">${hottest.map((row) => industryHeatRow(row, "hottest", hottestMax)).join("")}</div></section>
        <section class="industry-heat-panel" aria-labelledby="industry-heat-rising-title"><header><div><span>Attention trend</span><h3 id="industry-heat-rising-title">热度升降趋势排名 · ${trend.length}个</h3></div><p>份额变化与20日趋势同向；下降行业显示负分</p></header><div class="industry-heat-list" aria-label="热度升降趋势行业排名">${trendRows}</div></section>
      </div>`;
    }

    function setIndustryHeatLoading(loading) {
      industryHeatLoading = loading;
      const button = $("radar-industry-heat");
      if (!button) return;
      button.disabled = loading;
      button.setAttribute("aria-busy", loading ? "true" : "false");
      button.textContent = loading ? "读取行业热度…" : "三级行业热度";
    }

    async function fetchIndustryHeat(force = false) {
      if (!force && industryHeatPayload) {
        renderIndustryHeat(industryHeatPayload);
        return;
      }
      if (industryHeatLoading && !force) return;
      const seq = ++industryHeatFetchSeq;
      setIndustryHeatLoading(true);
      const content = $("radar-industry-heat-content");
      if (content) content.innerHTML = '<div class="industry-heat-state is-loading">正在读取结构化行业热度报告…</div>';
      try {
        const resp = await fetch("/api/radar/industry-heat");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (disposed || seq !== industryHeatFetchSeq) return;
        industryHeatPayload = data && data.payload ? data.payload : { available: false, error: "接口未返回报告。" };
        renderIndustryHeat(industryHeatPayload);
      } catch (err) {
        if (disposed || seq !== industryHeatFetchSeq) return;
        renderIndustryHeat({ available: false, error: `读取失败：${err && err.message ? err.message : "网络异常"}` });
      } finally {
        if (!disposed && seq === industryHeatFetchSeq) setIndustryHeatLoading(false);
      }
    }

    function openIndustryHeat() {
      openInsightModal("radar-industry-heat-modal");
      fetchIndustryHeat();
    }

    function invalidateIndustryHeat() {
      const shouldReload = activeInsightModal && activeInsightModal.id === "radar-industry-heat-modal";
      industryHeatPayload = null;
      industryHeatFetchSeq += 1;
      setIndustryHeatLoading(false);
      if (shouldReload) fetchIndustryHeat(true);
    }

    function patternTagClass(code) {
      const meta = patternMap[code];
      if (!meta || isEtfMode()) return "pd";
      const style = meta.display_style || meta.effective_style || "neutral";
      if (style === "bullish") return "pg";
      if (style === "momentum") return "po";
      if (style === "risk") return "pr";
      return "pd";
    }

    function patternTitle(code) {
      const meta = patternMap[code];
      if (!meta) return "";
      const validation = meta.effective
        ? (isEtfMode() ? "股票池实测有效，ETF尚未专项回测" : "★全历史实测有效")
        : "";
      return [validation, meta.name, meta.desc].filter(Boolean).join(" · ");
    }

    function stockPatternCodes(stock) {
      const source = Array.isArray(stock && stock.patterns) ? stock.patterns : [];
      return Array.from(new Set(source
        .map((code) => String(code || "").toUpperCase())
        .filter((code) => /^P\d+$/.test(code))));
    }

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

    function currentPoolValue() {
      return (($("radar-pool") && $("radar-pool").value) || (payload && payload.pool) || "leader");
    }

    function acceptRadarPayload(raw) {
      const previousPool = currentPoolValue();
      const nextPayload = normalizePayload(raw || {});
      const nextPool = nextPayload.pool || previousPool;
      payload = nextPayload;
      stocks = payload.stocks;
      if (nextPool !== previousPool) clearSelectedStock();
    }

    function renderProductionPatterns(patterns) {
      const target = $("radar-pattern-factor-groups");
      if (!target) return;
      const rows = Array.isArray(patterns) ? patterns.filter((pattern) => pattern && pattern.production !== false) : [];
      const phaseOrder = ["吸筹", "试盘", "洗盘", "突破", "拉升", "出货"];
      const signalLabels = { buy: "偏多结构", hold: "阶段观察", sell: "风险结构", observe: "中性观察" };
      const statusClasses = { core: "core", experimental: "experimental", observation: "observation" };
      target.innerHTML = phaseOrder.map((phase) => {
        const phaseRows = rows.filter((pattern) => pattern.category === phase);
        if (!phaseRows.length) return "";
        const cards = phaseRows.map((pattern) => {
          const etfUnvalidated = isEtfMode() && pattern.effective;
          const statusClass = etfUnvalidated
            ? "observation"
            : (statusClasses[pattern.validation_status] || "observation");
          const validationLabel = etfUnvalidated
            ? "股票池有效·ETF未回测"
            : (pattern.validation_label || "形态观察");
          const usageClass = String(pattern.score_usage || "").includes("吸筹分") ? "score-buy"
            : (String(pattern.score_usage || "").includes("出货分") || pattern.score_usage === "出货预警") ? "score-risk" : "score-label";
          return `<article class="radar-pattern-card"><div class="radar-pattern-card-head"><b>${esc(pattern.code)}</b><h5>${esc(pattern.name)}</h5><span class="radar-pattern-status ${statusClass}">${esc(validationLabel)}</span></div><div class="radar-pattern-meta"><span class="${usageClass}">${esc(pattern.score_usage || "阶段标签")}</span><span>${esc(signalLabels[pattern.signal] || "结构观察")}</span></div><p>${esc(pattern.desc || "暂无说明")}</p></article>`;
        }).join("");
        return `<section class="radar-pattern-group"><div class="radar-pattern-group-title"><h4>${esc(phase)}</h4><span>${phaseRows.length}项</span></div><div class="radar-pattern-grid">${cards}</div></section>`;
      }).join("");
      if (!target.innerHTML) target.innerHTML = '<div class="radar-factor-loading">生产形态说明暂不可用。</div>';
      if ($("radar-pattern-factor-count")) $("radar-pattern-factor-count").textContent = String(rows.length);
      if ($("radar-effective-pattern-count")) $("radar-effective-pattern-count").textContent = String(rows.filter((pattern) => pattern.effective).length);
    }

    async function fetchPatternCatalog() {
      try {
        const resp = await fetch("/api/radar/patterns");
        const data = await resp.json();
        const patterns = data.patterns || [];
        patternCatalog = patterns;
        patterns.forEach((p) => { patternMap[p.code] = p; });
        renderProductionPatterns(patterns);
      } catch (err) {
        if ($("radar-pattern-factor-groups")) $("radar-pattern-factor-groups").innerHTML = '<div class="radar-factor-loading">生产形态说明暂不可用，请刷新页面后重试。</div>';
      }
    }

    function renderScoringModel(model) {
      if (!model) return;
      const pool = (($("radar-pool") && $("radar-pool").value) || (payload && payload.pool) || "leader");
      const override = ((model.pool_overrides || {})[pool]) || {};
      const withWeights = (section, weights) => {
        if (!weights || !Object.keys(weights).length) return section || {};
        return {
          ...(section || {}),
          factors: ((section || {}).factors || [])
            .filter((factor) => Object.prototype.hasOwnProperty.call(weights, factor.key))
            .map((factor) => ({
              ...factor,
              weight: weights[factor.key],
              weight_pct: Math.round(Number(weights[factor.key]) * 10000) / 100,
            })),
        };
      };
      const accumulation = withWeights(model.accumulation, override.accumulation_weights);
      const distribution = withWeights(model.distribution, override.distribution_weights);
      const auxiliary = pool === "etf"
        ? (model.auxiliary || []).filter((factor) => !["market_regime", "industry_heat"].includes(factor.key))
        : (model.auxiliary || []);
      const opportunity = model.opportunity || {};
      if ($("radar-opportunity-formula")) $("radar-opportunity-formula").textContent = opportunity.display_formula || opportunity.formula || "";
      if ($("radar-opportunity-description")) $("radar-opportunity-description").textContent = opportunity.description || "";
      const example = opportunity.example || {};
      const penalty = Number(opportunity.distribution_penalty || 0);
      const acc = Number(example.accumulation_percentile || 0);
      const dist = Number(example.distribution_percentile || 0);
      const keep = 1 - penalty * dist / 100;
      if ($("radar-score-example")) {
        $("radar-score-example").innerHTML = `<div><small>吸筹百分位</small><strong>${fmt(acc, 0)}</strong></div><span>×</span><div><small>风险保留系数</small><strong>${fmt(keep, 2)}</strong></div><span>=</span><div class="result"><small>机会分</small><strong>${fmt(example.opportunity_score, 0)}</strong></div>`;
      }
      const sorting = model.sorting || {};
      if ($("radar-opportunity-sorting")) {
        $("radar-opportunity-sorting").textContent = `机会分表达的是候选池内的相对位置，不是上涨概率。${sorting[pool] || sorting.leader || ""}。`;
      }

      const weightHtml = (section) => ((section || {}).factors || []).map((factor) =>
        `<p><b>${esc(factor.weight_pct)}%</b><span>${esc(factor.label)}</span></p>`
      ).join("");
      if ($("radar-accumulation-weights")) $("radar-accumulation-weights").innerHTML = weightHtml(accumulation);
      if ($("radar-distribution-weights")) $("radar-distribution-weights").innerHTML = weightHtml(distribution);

      const factorHtml = (section) => ((section || {}).factors || []).map((factor) =>
        `<article><b>${esc(factor.key)}</b><h4>${esc(factor.label)} · ${esc(factor.weight_pct)}%</h4><p>${esc(factor.description)}</p></article>`
      ).join("");
      if ($("radar-accumulation-factors")) $("radar-accumulation-factors").innerHTML = factorHtml(accumulation);
      if ($("radar-distribution-factors")) $("radar-distribution-factors").innerHTML = factorHtml(distribution);
      if ($("radar-reversal-factors")) {
        $("radar-reversal-factors").innerHTML = ((model.reversal || {}).factors || []).map((factor) =>
          `<div><b>${esc(factor.weight_pct)}%</b><span>${esc(factor.label)}</span><small>${esc(factor.description)}</small></div>`
        ).join("");
      }
      const scoringCount = [accumulation, distribution, model.reversal]
        .reduce((total, section) => total + (((section || {}).factors || []).length), 0);
      if ($("radar-scoring-factor-count")) $("radar-scoring-factor-count").textContent = String(scoringCount);
      if ($("radar-auxiliary-factors")) {
        $("radar-auxiliary-factors").innerHTML = auxiliary.map((factor) =>
          `<article><b>${esc(factor.key)}</b><h4>${esc(factor.label)}</h4><p>${esc(factor.description)}</p><span>不重复计分</span></article>`
        ).join("");
      }
    }

    async function fetchScoringModel() {
      try {
        const resp = await fetch("/api/radar/model");
        if (!resp.ok) throw new Error("模型说明读取失败");
        scoringModel = await resp.json();
        renderScoringModel(scoringModel);
      } catch (err) {
        if ($("radar-accumulation-factors")) $("radar-accumulation-factors").innerHTML = "<article><h4>模型说明暂不可用</h4><p>请刷新页面后重试。</p></article>";
        if ($("radar-distribution-factors")) $("radar-distribution-factors").innerHTML = "<article><h4>模型说明暂不可用</h4><p>请刷新页面后重试。</p></article>";
        if ($("radar-auxiliary-factors")) $("radar-auxiliary-factors").innerHTML = "<article><h4>辅助判断说明暂不可用</h4><p>请刷新页面后重试。</p></article>";
      }
    }

    function stockSw2Name(s) {
      return String((s && s.parent_segment) || "").trim();
    }

    function sw2HeatValue(s) {
      const v = s && s.sw2_heat_pctile;
      if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) return null;
      return Number(v);
    }

    function heatText(v) {
      return v === null || v === undefined ? "热-" : `热${Math.round(v)}`;
    }

    function buildSw2Options() {
      const byName = new Map();
      stocks.forEach((s) => {
        const name = stockSw2Name(s);
        if (!name) return;
        const heat = sw2HeatValue(s);
        const row = byName.get(name) || { name, heat: null, count: 0 };
        row.count += 1;
        if (heat !== null && (row.heat === null || heat > row.heat)) row.heat = heat;
        byName.set(name, row);
      });
      return Array.from(byName.values()).sort((a, b) => {
        const ah = a.heat === null ? -1 : a.heat;
        const bh = b.heat === null ? -1 : b.heat;
        if (bh !== ah) return bh - ah;
        if (b.count !== a.count) return b.count - a.count;
        return a.name.localeCompare(b.name, "zh-Hans-CN");
      });
    }

    function pickedSw2Options() {
      return sw2Options.filter((o) => selectedSw2Industries.has(o.name));
    }

    function syncSw2Filter() {
      sw2Options = buildSw2Options();
      const validNames = new Set(sw2Options.map((o) => o.name));
      selectedSw2Industries = new Set(Array.from(selectedSw2Industries).filter((name) => validNames.has(name)));
      renderSw2Filter();
    }

    function visibleSw2Options() {
      const q = (($("radar-sw2-search") && $("radar-sw2-search").value) || "").trim().toLowerCase();
      const minHeatValue = Number(($("radar-sw2-min-heat") && $("radar-sw2-min-heat").value) || 0);
      const minHeat = Number.isNaN(minHeatValue) ? 0 : minHeatValue;
      return sw2Options.filter((o) => {
        if (!isEtfMode() && (o.heat === null || o.heat <= minHeat)) return false;
        if (!q) return true;
        return o.name.toLowerCase().includes(q);
      });
    }

    function renderSw2Filter() {
      const trigger = $("radar-sw2-trigger");
      const summary = $("radar-sw2-summary");
      const selectedBox = $("radar-sw2-selected");
      const optionsBox = $("radar-sw2-options");
      if (!trigger || !summary || !selectedBox || !optionsBox) return;

      const picked = pickedSw2Options();
      if (!picked.length) summary.textContent = sw2Options.length ? "全部" : (isEtfMode() ? "无类别" : "无行业");
      else if (picked.length === 1) summary.textContent = isEtfMode()
        ? picked[0].name
        : `${picked[0].name} ${heatText(picked[0].heat)}`;
      else summary.textContent = `已选 ${picked.length} 个`;
      trigger.disabled = !sw2Options.length;
      trigger.classList.toggle("is-filtered", picked.length > 0);

      selectedBox.innerHTML = picked.length
        ? picked.map((o) => `<button type="button" class="industry-chip" data-name="${esc(o.name)}">${esc(o.name)}${isEtfMode() ? "" : ` <small>${heatText(o.heat)}</small>`}</button>`).join("")
        : `<span>${isEtfMode() ? "全部ETF类别" : "全部二级行业"}</span>`;
      selectedBox.querySelectorAll("button").forEach((btn) => {
        btn.onclick = () => {
          selectedSw2Industries.delete(btn.dataset.name);
          renderSw2Filter();
          applyFilters();
        };
      });

      const shown = visibleSw2Options();
      optionsBox.innerHTML = shown.length ? shown.map((o) => {
        const checked = selectedSw2Industries.has(o.name) ? " checked" : "";
        return `<label class="industry-option" title="${esc(o.name)} ${heatText(o.heat)} · ${o.count}只">
          <input type="checkbox" value="${esc(o.name)}"${checked}>
          <span class="industry-option-name">${esc(o.name)}</span>
          <span class="industry-option-heat">${isEtfMode() ? "" : heatText(o.heat)}</span>
          <span class="industry-option-count">${o.count}只</span>
        </label>`;
      }).join("") : `<div class="industry-empty">${isEtfMode() ? "无匹配类别" : "无匹配行业"}</div>`;
      optionsBox.querySelectorAll("input[type='checkbox']").forEach((input) => {
        input.onchange = () => {
          if (input.checked) selectedSw2Industries.add(input.value);
          else selectedSw2Industries.delete(input.value);
          renderSw2Filter();
          applyFilters();
        };
      });
      const clear = $("radar-sw2-clear");
      if (clear) clear.disabled = picked.length === 0;
      const selectAll = $("radar-sw2-select-all");
      if (selectAll) selectAll.disabled = shown.length === 0;
    }

    function setSw2PanelOpen(open) {
      const panel = $("radar-sw2-panel");
      const trigger = $("radar-sw2-trigger");
      const container = $("radar-sw2-filter");
      if (!panel || !trigger) return;
      if (open) setPatternPanelOpen(false);
      panel.hidden = !open;
      trigger.setAttribute("aria-expanded", open ? "true" : "false");
      if (container) container.classList.toggle("is-open", open);
      if (open) {
        renderSw2Filter();
        const search = $("radar-sw2-search");
        if (search) search.focus({ preventScroll: true });
      }
    }

    function onDocumentClick(ev) {
      const box = $("radar-sw2-filter");
      if (box && !box.contains(ev.target)) setSw2PanelOpen(false);
      const patternBox = $("radar-pattern-filter");
      if (patternBox && !patternBox.contains(ev.target)) setPatternPanelOpen(false);
    }

    function onDocumentKeyDown(ev) {
      if (ev.key === "Escape" && activeInsightModal) {
        closeInsightModal();
        return;
      }
      if (ev.key !== "Escape") return;
      const sw2Panel = $("radar-sw2-panel");
      const patternPanel = $("radar-pattern-panel");
      const sw2WasOpen = Boolean(sw2Panel && !sw2Panel.hidden);
      const patternWasOpen = Boolean(patternPanel && !patternPanel.hidden);
      setSw2PanelOpen(false);
      setPatternPanelOpen(false);
      if (patternWasOpen) $("radar-pattern-trigger").focus();
      else if (sw2WasOpen) $("radar-sw2-trigger").focus();
    }

    // ---------- 数据加载 ----------
    async function fetchData() {
      try {
        const resp = await fetch("/api/radar/data");
        const data = await resp.json();
        if (disposed) return null;
        acceptRadarPayload(data.payload);
        syncSw2Filter();
        renderMeta();
        renderProductionPatterns(patternCatalog);
        renderScoringModel(scoringModel);
        renderPhaseFilter();
        syncPatternFilter();
        renderPhaseChips();
        applyFilters();
        if (selectedCode) {
          const selected = stocks.find((s) => s.code === selectedCode);
          if (selected) {
            renderPatternInfo(selected);
            renderStockInfo(selected, klineBars);
          } else clearSelectedStock();
        }
        reflectJobs(data.run_job, data.data_job);
        return data;
      } catch (err) {
        $("radar-status").textContent = "数据读取失败";
        return null;
      }
    }

    async function fetchJobState() {
      try {
        const resp = await fetch("/api/radar/jobs");
        const data = await resp.json();
        if (disposed) return null;
        reflectJobs(data.run_job, data.data_job);
        return data;
      } catch (err) {
        $("radar-status").textContent = "任务状态读取失败";
        return null;
      }
    }

    function updateRealtimeStatus(text, tone) {
      const el = $("radar-live-status");
      if (!el) return;
      el.textContent = text;
      el.classList.toggle("is-live", tone === "live");
      el.classList.toggle("is-error", tone === "error");
    }

    async function fetchRealtimeData(queueIfBusy = false) {
      const toggle = $("radar-live-quote");
      if (!toggle || !toggle.checked || disposed) return;
      if (radarRunBusy) {
        realtimeRefreshPending = true;
        updateRealtimeStatus("等待新股票池…", "live");
        return;
      }
      if (realtimeBusy) {
        if (queueIfBusy) realtimeRefreshQueued = true;
        return;
      }
      realtimeBusy = true;
      updateRealtimeStatus("刷新中…", "live");
      try {
        const resp = await fetch("/api/radar/realtime");
        const data = await resp.json();
        if (disposed || !toggle.checked) return;
        if (radarRunBusy) {
          realtimeRefreshPending = true;
          updateRealtimeStatus("等待新股票池…", "live");
          return;
        }
        acceptRadarPayload(data.payload);
        syncSw2Filter();
        renderMeta();
        renderProductionPatterns(patternCatalog);
        renderScoringModel(scoringModel);
        renderPhaseFilter();
        syncPatternFilter();
        renderPhaseChips();
        applyFilters();
        if (selectedCode) {
          const selected = stocks.find((s) => s.code === selectedCode);
          if (selected) {
            renderPatternInfo(selected);
            renderStockInfo(selected, klineBars);
          } else clearSelectedStock();
        }
        const rt = payload.realtime_quote || {};
        const sourceLabel = REALTIME_SOURCE_LABEL[rt.source] || "实时";
        if (rt.available) updateRealtimeStatus(`${sourceLabel} ${rt.updated_at || ""} · 3分钟`, "live");
        else updateRealtimeStatus(rt.error ? "实时失败" : "无实时数据", rt.error ? "error" : "");
      } catch (err) {
        if (!disposed) updateRealtimeStatus("实时失败", "error");
      } finally {
        realtimeBusy = false;
        if (realtimeRefreshQueued && !disposed && toggle.checked) {
          realtimeRefreshQueued = false;
          fetchRealtimeData(true);
        }
      }
    }

    function setRealtimeEnabled(enabled) {
      const toggle = $("radar-live-quote");
      if (toggle) toggle.checked = !!enabled;
      if (realtimeTimer) {
        clearInterval(realtimeTimer);
        realtimeTimer = null;
      }
      if (!enabled) {
        realtimeRefreshPending = false;
        realtimeRefreshQueued = false;
        updateRealtimeStatus("未开启", "");
        fetchData();
        return;
      }
      updateRealtimeStatus("等待刷新…", "live");
      fetchRealtimeData();
      realtimeTimer = setInterval(fetchRealtimeData, REALTIME_REFRESH_MS);
    }

    function isEtfMode() {
      return (($("radar-pool") && $("radar-pool").value) || (payload && payload.pool)) === "etf";
    }

    function syncPoolPresentation(busy = false) {
      const etf = isEtfMode();
      const includeLarge = $("radar-include-large");
      if (etf) includeLarge.checked = true;
      includeLarge.disabled = busy || etf;
      if ($("radar-industry-filter-label")) $("radar-industry-filter-label").textContent = etf ? "ETF类别" : "二级行业";
      if ($("radar-sw2-search")) $("radar-sw2-search").placeholder = etf ? "搜索ETF类别" : "搜索二级行业";
      if ($("radar-stock-cap-label")) $("radar-stock-cap-label").textContent = etf ? "基金规模" : "总市值";
      if ($("radar-stock-sw2-label")) $("radar-stock-sw2-label").textContent = etf ? "ETF类别" : "二级行业";
      const minHeat = $("radar-sw2-min-heat");
      if (minHeat) {
        minHeat.disabled = etf;
        if (etf) minHeat.value = "0";
        if (minHeat.closest("label")) minHeat.closest("label").hidden = etf;
      }
      if ($("radar-sw2-hot")) $("radar-sw2-hot").hidden = etf;
      renderProductionPatterns(patternCatalog);
      renderScoringModel(scoringModel);
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
      syncPoolPresentation(jobsWereRunning);
      const pool = p.pool === "hotmoney"
        ? "游资小盘池(龙虎榜活跃·流通≤100亿·反转分排序)"
        : p.pool === "etf"
          ? "ETF配置池(纯技术分析·机会分排序)"
          : (cap ? `≤${cap}亿小中盘龙头` : "全市值龙头(含大盘)");
      const asOf = p.as_of ? ` · 数据截至 ${p.as_of}(历史复盘)` : "";
      const theme = p.pool === "etf"
        ? "ETF分类: 来自 stock_etf_pool.py"
        : (p.theme_source && p.theme_source.available)
        ? `题材热度 ${p.theme_source.generated_at || ""}${p.theme_source.stale ? " ⚠️偏旧" : ""}`
        : "题材热度: 缺/未挂载";
      const cc = p.capital_counts || {};
      const capLine = p.capital_applicable === false
        ? `<span>公司资金面: ETF不适用（已跳过股东户数、公司回购、龙虎榜）</span>`
        : p.capital_available
        ? `<span>资金面: 户数降<strong>${cc.holder_down ?? 0}</strong> · 回购<strong>${cc.repurchase ?? 0}</strong> · 上榜避雷<strong>${cc.lhb_avoid ?? 0}</strong> · 户数/回购已并入吸筹总分</span>`
        : `<span>资金面: 未挂载（跑 stock_crawl_holders / stock_crawl_capital 后生效）</span>`;
      const mr = p.market_regime;
      const regimeLine = (mr && mr.available)
        ? `<span>大盘 <strong>${mr.above_ma20 ? "强势(>MA20)·反转分可做多" : "弱势(<MA20)·做多易接刀"}</strong></span>`
        : "";
      const rt = p.realtime_quote || {};
      const sourceLabel = REALTIME_SOURCE_LABEL[rt.source] || "实时";
      const realtimeLine = rt.available
        ? `<span>实时: ${esc(sourceLabel)} <strong>${esc(rt.updated_at || "")}</strong> · 匹配<strong>${rt.matched_count ?? 0}</strong> · 重算<strong>${rt.used_count ?? 0}</strong></span>`
        : (rt.error ? `<span>实时: <strong>失败</strong> ${esc(rt.error)}</span>` : "");
      $("radar-meta").innerHTML =
        `<span>生成 <strong>${esc(p.generated_at)}</strong>${asOf}</span>` +
        `<span>候选 <strong>${p.candidate_count ?? "-"}</strong>（${esc(pool)}）</span>` +
        `<span>已打分 <strong>${p.scored_count ?? "-"}</strong></span>` +
        `<span>${esc(theme)}</span>` + capLine + regimeLine + realtimeLine;
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

    function buildPatternOptions() {
      const counts = new Map();
      let unmatched = 0;
      stocks.forEach((stock) => {
        const codes = stockPatternCodes(stock);
        if (!codes.length) unmatched += 1;
        codes.forEach((code) => counts.set(code, (counts.get(code) || 0) + 1));
      });
      const catalogCodes = patternCatalog
        .map((pattern) => String((pattern && pattern.code) || "").toUpperCase())
        .filter((code) => counts.has(code));
      const seen = new Set(catalogCodes);
      const extras = Array.from(counts.keys())
        .filter((code) => !seen.has(code))
        .sort((a, b) => (Number(a.slice(1)) || 0) - (Number(b.slice(1)) || 0));
      const options = catalogCodes.concat(extras).map((code) => {
        const meta = patternMap[code] || {};
        return {
          code,
          name: meta.name || code,
          category: meta.category || "其他",
          signal: meta.signal || "observe",
          count: counts.get(code) || 0,
        };
      });
      if (unmatched) {
        options.push({
          code: NO_PATTERN_FILTER,
          name: "未命中形态",
          category: "其他",
          signal: "observe",
          count: unmatched,
        });
      }
      return options;
    }

    function patternOptionTone(option) {
      if (!option || option.code === NO_PATTERN_FILTER) return "";
      if (option.category === "拉升") return "rally";
      if (PATTERN_BACKTEST_BUY_CODE_SET.has(option.code)) return "buy";
      if (PATTERN_SELL_PRESET_CODE_SET.has(option.code)) return "sell";
      return "";
    }

    function isPatternBuyPreset(option) {
      return !!option && PATTERN_BACKTEST_BUY_CODE_SET.has(option.code);
    }

    function isPatternSellPreset(option) {
      return !!option && PATTERN_SELL_PRESET_CODE_SET.has(option.code);
    }

    function isPatternRallyPreset(option) {
      return !!option && option.category === "拉升";
    }

    function pickedPatternOptions() {
      return patternOptions.filter((option) => selectedPatterns.has(option.code));
    }

    function syncPatternFilter() {
      patternOptions = buildPatternOptions();
      const validCodes = new Set(patternOptions.map((option) => option.code));
      selectedPatterns = new Set(Array.from(selectedPatterns).filter((code) => validCodes.has(code)));
      renderPatternFilter();
    }

    function visiblePatternOptions() {
      const q = (($("radar-pattern-search") && $("radar-pattern-search").value) || "").trim().toLowerCase();
      if (!q) return patternOptions;
      return patternOptions.filter((option) =>
        `${option.code} ${option.name} ${option.category}`.toLowerCase().includes(q)
      );
    }

    function renderPatternFilter() {
      const trigger = $("radar-pattern-trigger");
      const summary = $("radar-pattern-summary");
      const selectedBox = $("radar-pattern-selected");
      const optionsBox = $("radar-pattern-options");
      if (!trigger || !summary || !selectedBox || !optionsBox) return;

      const picked = pickedPatternOptions();
      if (!picked.length) summary.textContent = patternOptions.length ? "全部" : "无形态";
      else if (picked.length === 1) summary.textContent = picked[0].code === NO_PATTERN_FILTER ? "未命中" : picked[0].code;
      else summary.textContent = `已选 ${picked.length} 个`;
      trigger.disabled = !patternOptions.length;
      trigger.classList.toggle("is-filtered", picked.length > 0);

      selectedBox.innerHTML = picked.length
        ? picked.map((option) => `<button type="button" class="industry-chip pattern-chip ${patternOptionTone(option)}" data-code="${esc(option.code)}" title="移除 ${esc(option.name)}">${esc(option.code === NO_PATTERN_FILTER ? "未命中" : option.code)}</button>`).join("")
        : "<span>全部命中形态</span>";
      selectedBox.querySelectorAll("button").forEach((button) => {
        button.onclick = () => {
          selectedPatterns.delete(button.dataset.code);
          renderPatternFilter();
          applyFilters();
        };
      });

      const shown = visiblePatternOptions();
      optionsBox.innerHTML = shown.length ? shown.map((option) => {
        const checked = selectedPatterns.has(option.code) ? " checked" : "";
        const displayCode = option.code === NO_PATTERN_FILTER ? "—" : option.code;
        return `<label class="industry-option pattern-option" title="${esc(option.code === NO_PATTERN_FILTER ? option.name : `${option.code} ${option.name}`)} · ${option.count}只">
          <input type="checkbox" value="${esc(option.code)}"${checked}>
          <span class="pattern-option-code">${esc(displayCode)}</span>
          <span class="industry-option-name">${esc(option.name)}</span>
          <span class="pattern-option-category">${esc(option.category)}</span>
          <span class="industry-option-count">${option.count}只</span>
        </label>`;
      }).join("") : '<div class="industry-empty">无匹配形态</div>';
      optionsBox.querySelectorAll("input[type='checkbox']").forEach((input) => {
        input.onchange = () => {
          if (input.checked) selectedPatterns.add(input.value);
          else selectedPatterns.delete(input.value);
          renderPatternFilter();
          applyFilters();
        };
      });

      const clear = $("radar-pattern-clear");
      if (clear) clear.disabled = picked.length === 0;
      const selectAll = $("radar-pattern-select-all");
      if (selectAll) selectAll.disabled = shown.length === 0;
      const presets = {
        "radar-pattern-buy": isPatternBuyPreset,
        "radar-pattern-sell": isPatternSellPreset,
        "radar-pattern-rally": isPatternRallyPreset,
      };
      Object.entries(presets).forEach(([id, predicate]) => {
        const button = $(id);
        if (!button) return;
        const presetCodes = patternOptions.filter(predicate).map((option) => option.code);
        const active = presetCodes.length > 0
          && presetCodes.length === selectedPatterns.size
          && presetCodes.every((code) => selectedPatterns.has(code));
        button.disabled = presetCodes.length === 0;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", active ? "true" : "false");
      });
    }

    function setPatternPanelOpen(open) {
      const panel = $("radar-pattern-panel");
      const trigger = $("radar-pattern-trigger");
      const container = $("radar-pattern-filter");
      if (!panel || !trigger) return;
      if (open) setSw2PanelOpen(false);
      panel.hidden = !open;
      trigger.setAttribute("aria-expanded", open ? "true" : "false");
      if (container) container.classList.toggle("is-open", open);
      if (open) {
        renderPatternFilter();
        const search = $("radar-pattern-search");
        if (search) search.focus({ preventScroll: true });
      }
    }

    function selectPatternPreset(predicate) {
      selectedPatterns = new Set(patternOptions.filter(predicate).map((option) => option.code));
      renderPatternFilter();
      applyFilters();
    }

    // ---------- 筛选 + 表格 ----------
    function currentFilters() {
      return {
        q: ($("radar-search").value || "").trim().toLowerCase(),
        phase: $("radar-phase-filter").value || "",
        patterns: new Set(selectedPatterns),
        minScore: Number($("radar-min-score").value || 0),
        hideDist: $("radar-hide-dist").checked,
        sw2: new Set(selectedSw2Industries),
      };
    }

    function applyFilters() {
      const f = currentFilters();
      const list = stocks.filter((s) => {
        if (f.phase && s.pattern_phase !== f.phase) return false;
        const patterns = stockPatternCodes(s);
        if (f.patterns.size) {
          const matchedSelected = patterns.some((code) => f.patterns.has(code));
          const matchedUnhit = !patterns.length && f.patterns.has(NO_PATTERN_FILTER);
          if (!matchedSelected && !matchedUnhit) return false;
        }
        if (f.hideDist && String(s.pattern_phase || "").includes("出货")) return false;
        if ((Number(s.ambush_score) || 0) < f.minScore) return false;
        if (f.sw2.size && !f.sw2.has(stockSw2Name(s))) return false;
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
      const hasRealtime = !!(payload && payload.realtime_quote && payload.realtime_quote.available);
      const etf = isEtfMode();
      const sortTh = (key, label) => {
        const active = sortState.key === key;
        const arrow = active ? (sortState.dir === "asc" ? "▲" : "▼") : "↕";
        const ariaSort = active ? (sortState.dir === "asc" ? "ascending" : "descending") : "none";
        return `<th class="num sortable${active ? " active" : ""}" data-key="${key}" aria-sort="${ariaSort}"><button class="sort-control" type="button">${label} <span class="sort-arrow" aria-hidden="true">${arrow}</span></button></th>`;
      };
      const realtimeHead = hasRealtime
        ? `${sortTh("realtime_price", "现价")}${sortTh("realtime_change_pct", "涨幅")}`
        : "";
      const scaleHead = etf
        ? sortTh("fund_scale_yi", "规模")
        : sortTh("market_cap_yi", "市值");
      const head = `<tr>
        <th>#</th><th>代码</th><th>名称</th>${scaleHead}${realtimeHead}${sortTh("opportunity_score", "机会分")}
        <th class="num" title="按筹码成本估算，当前价以下的筹码占比">获利盘</th><th>命中形态</th><th>阶段·把握</th><th>${etf ? "ETF类别" : "二级行业"}</th><th>依据</th>
      </tr>`;
      const rows = list.map((s, i) => {
        const sig = s.signals || {};
        const cls = phaseClass(s.pattern_phase);
        const conf = (s.phase_confidence === null || s.phase_confidence === undefined) ? "" : ` <small>把握${Math.round(s.phase_confidence)}</small>`;
        const sel = s.code === selectedCode ? " is-selected" : "";
        const patCell = (s.patterns && s.patterns.length)
          ? s.patterns.map((c) => {
              const title = patternTitle(c);
              return `<span class="pat ${patternTagClass(c)}"${title ? ` title="${esc(title)}"` : ""}>${esc(c)}</span>`;
            }).join("")
          : "<span class='dim'>-</span>";
        const evCell = (s.evidence && s.evidence.length)
          ? s.evidence.map((e) => {
              const lab = normalizeEvidenceLabel((typeof e === "string") ? e : (e.label || ""));
              const kind = (typeof e === "string") ? "" : (e.kind || "");
              const cls = kind === "bullish" ? "ev-bull" : kind === "bearish" ? "ev-warn" : "ev-neutral";
              return `<span class="ev ${cls}">${esc(lab)}</span>`;
            }).join("")
          : "<span class='dim'>-</span>";
        const scaleCell = etf
          ? fmtMarketCap(s.fund_scale_yi)
          : fmtMarketCap(s.market_cap_yi);
        return `<tr class="rrow${sel}" data-code="${esc(s.code)}" tabindex="0" aria-selected="${s.code === selectedCode ? "true" : "false"}" aria-label="查看 ${esc(s.code)} ${esc(s.name || "")} K线与形态">
          <td class="dim">${i + 1}</td>
          <td class="mono">${esc(s.code)}</td>
          <td>${esc(s.name || "")}</td>
          <td class="num">${scaleCell}</td>
          ${hasRealtime ? `<td class="num">${fmt(s.realtime_price, 2)}</td><td class="num ${Number(s.realtime_change_pct) > 0 ? "up" : Number(s.realtime_change_pct) < 0 ? "down" : ""}">${fmtSignedPct(s.realtime_change_pct)}</td>` : ""}
          <td class="num strong">${fmt(s.opportunity_score, 1)}</td>
          <td class="num" title="当前价以下的估算筹码占比">${fmtRatioPct(sig.chip_winner)}</td>
          <td class="pats mono">${patCell}</td>
          <td><span class="phase phase-${cls}">${esc(s.pattern_phase || "")}</span>${conf}${isDistributionConflict(s) ? ` <span class="pi-warn" title="${esc(distributionConflictTip(s))}">⚠风险</span>` : ""}</td>
          <td>${sw2Cell(s)}</td>
          <td class="ev-cell">${evCell}</td>
        </tr>`;
      }).join("");
      $("radar-table").innerHTML = `<table class="radar-grid"><thead>${head}</thead><tbody>${rows}</tbody></table>`;
      $("radar-table").querySelectorAll(".rrow").forEach((tr) => {
        tr.onclick = () => selectStock(tr.dataset.code);
        tr.onkeydown = (event) => {
          if (event.key !== "Enter" && event.key !== " ") return;
          event.preventDefault();
          selectStock(tr.dataset.code);
        };
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
    function patternStyle(pattern) {
      const style = String((pattern && pattern.effective_style) || ((patternMap[pattern && pattern.code] || {}).effective_style) || "");
      return PATTERN_MARKER_META[style] ? style : "";
    }

    function normalizePatternBacktestHits(rawHits, meta = {}) {
      if (!Array.isArray(rawHits)) return [];
      const buyPatternCodes = new Set(patternRuleCodes(meta.buy_rule || {}, PATTERN_BACKTEST_BUY_CODES));
      const sellPatternCodes = new Set(patternRuleCodes(meta.sell_rule || {}, DISTRIBUTION_WARNING_PATTERN_CODES));
      return rawHits.map((hit) => {
        const numericOrNull = (value) => {
          if (value === null || value === undefined || value === "") return null;
          const parsed = Number(value);
          return Number.isFinite(parsed) ? parsed : null;
        };
        const isSellPoint = Boolean(hit && hit.is_sell_point === true);
        const isSuspectAccumulation = Boolean(hit && hit.is_suspect_accumulation === true);
        const distributionWarningPoints = numericOrNull(hit && hit.distribution_warning_points) ?? 0;
        const effectiveSellPatternCount = numericOrNull(hit && hit.effective_sell_pattern_count) ?? 0;
        const accumulationScore = numericOrNull(hit && hit.accumulation_score);
        const accumulationThreshold = numericOrNull(hit && hit.accumulation_threshold);
        const matchedProductionPatternCount = numericOrNull(hit && hit.matched_production_pattern_count);
        const patterns = (Array.isArray(hit && hit.patterns) ? hit.patterns : []).map((pattern) => {
          const style = patternStyle(pattern);
          if (!style || !pattern || !pattern.code) return null;
          const catalog = patternMap[pattern.code] || {};
          const signal = String(pattern.signal || catalog.signal || "");
          const code = String(pattern.code);
          const synthetic = pattern.synthetic === true;
          const isSuspectPattern = code === "SUSPECT_ACCUM" || synthetic;
          if (isSuspectPattern && !(
            isSuspectAccumulation &&
            code === "SUSPECT_ACCUM" &&
            synthetic &&
            style === "bullish" &&
            signal === "buy" &&
            matchedProductionPatternCount === 0
          )) return null;
          if (style === "bullish" && !synthetic && (signal !== "buy" || !buyPatternCodes.has(code))) return null;
          if (style === "risk" && (!isSellPoint || signal !== "sell" || !sellPatternCodes.has(code))) return null;
          return {
            code,
            name: String(pattern.name || catalog.name || code),
            phase: String(pattern.phase || catalog.category || ""),
            signal,
            effective_style: style,
            synthetic,
            accumulation_score: synthetic ? accumulationScore : null,
            accumulation_threshold: synthetic ? accumulationThreshold : null,
          };
        }).filter(Boolean);
        const hasSuspectPattern = patterns.some((pattern) => pattern.synthetic === true);
        const hasBuyPattern = patterns.some((pattern) => !pattern.synthetic && buyPatternCodes.has(pattern.code));
        const hasSellPattern = patterns.some((pattern) => sellPatternCodes.has(pattern.code));
        return {
          date: String((hit && hit.date) || ""),
          patterns,
          is_sell_point: isSellPoint && hasSellPattern,
          is_buy_point: hasBuyPattern,
          is_suspect_accumulation: isSuspectAccumulation && hasSuspectPattern,
          accumulation_score: accumulationScore,
          accumulation_threshold: accumulationThreshold,
          matched_production_pattern_count: matchedProductionPatternCount,
          distribution_warning_points: distributionWarningPoints,
          effective_sell_pattern_count: effectiveSellPatternCount,
        };
      }).filter((hit) => hit.date && hit.patterns.length)
        .sort((a, b) => a.date.localeCompare(b.date));
    }

    function patternBacktestCounts() {
      const counts = { dates: patternBacktestHits.length, patterns: 0, bullish: 0, suspect: 0, momentum: 0, risk: 0 };
      patternBacktestHits.forEach((hit) => {
        counts.patterns += hit.patterns.length;
        if (hit.is_suspect_accumulation) counts.suspect += 1;
        new Set(hit.patterns.filter((pattern) => !pattern.synthetic).map(patternStyle)).forEach((style) => {
          if (Object.prototype.hasOwnProperty.call(counts, style)) counts[style] += 1;
        });
      });
      return counts;
    }

    function renderPatternBacktestUI() {
      const button = $("radar-pattern-backtest");
      const label = $("radar-pattern-backtest-label");
      const badge = $("radar-pattern-backtest-count");
      const legend = $("radar-pattern-backtest-legend");
      if (!button || !label || !badge || !legend) return;

      const counts = patternBacktestCounts();
      const klineReady = Boolean(
        selectedCode &&
        loadedKlineCode === selectedCode &&
        loadedKlinePeriod === klinePeriod &&
        klineBars.length
      );
      button.disabled = !klineReady || patternBacktestLoading || jobsWereRunning;
      button.classList.toggle("is-loading", patternBacktestLoading);
      button.classList.toggle("is-active", patternBacktestActive);
      button.setAttribute("aria-pressed", patternBacktestActive ? "true" : "false");
      button.title = patternBacktestLoading
        ? "正在逐交易日计算当前 K 线完整区间"
        : (patternBacktestActive ? "隐藏 K 线上的形态标记" : "逐交易日回放该标的全部历史 K 线");
      label.textContent = patternBacktestLoading ? "回测中" : "形态回测";
      badge.hidden = !patternBacktestActive;
      badge.textContent = String(counts.dates);

      if (patternBacktestLoading) {
        legend.hidden = false;
        legend.innerHTML = '<span class="radar-backtest-legend-title">正在逐日回放…</span><span>全部历史日线 · 历史不足 40 根不计算</span>';
      } else if (patternBacktestError) {
        legend.hidden = false;
        legend.innerHTML = `<span class="radar-backtest-legend-title">回测失败</span><span>${esc(patternBacktestError)}</span>`;
      } else if (patternBacktestActive) {
        const scope = (patternBacktestMeta && patternBacktestMeta.validation_scope) === "stock_reference"
          ? "股票池口径 · ETF待专项验证"
          : "产品买卖点口径 · 日线PIT回放";
        const total = Number(patternBacktestMeta && patternBacktestMeta.total_bars) || 0;
        const evaluated = Number(patternBacktestMeta && patternBacktestMeta.evaluated_bars) || 0;
        const sellRule = (patternBacktestMeta && patternBacktestMeta.sell_rule) || {};
        const buyRule = (patternBacktestMeta && patternBacktestMeta.buy_rule) || {};
        const suspectRule = (patternBacktestMeta && patternBacktestMeta.suspect_accumulation_rule) || {};
        const suspectHit = patternBacktestHits.find((hit) => hit.is_suspect_accumulation);
        const suspectThreshold = Number(
          suspectRule.threshold ?? (suspectHit && suspectHit.accumulation_threshold)
        ) || 35;
        const summary = counts.dates
          ? `全历史日线 ${total} 根 · 已计算 ${evaluated} 根 · 标记 ${counts.dates} 日 / ${counts.patterns} 个信号`
          : `全历史日线 ${total} 根 · 已计算 ${evaluated} 根 · 未命中回测信号`;
        legend.hidden = false;
        legend.innerHTML = `<span class="radar-backtest-legend-title">${esc(summary)}</span>` +
          `<span class="radar-backtest-legend-item bullish"><i>买</i>${counts.bullish}</span>` +
          `<span class="radar-backtest-legend-item bullish"><i>疑</i>${counts.suspect}</span>` +
          `<span class="radar-backtest-legend-item momentum"><i>强</i>${counts.momentum}</span>` +
          `<span class="radar-backtest-legend-item risk"><i>卖</i>${counts.risk}</span>` +
          `<span class="radar-backtest-legend-note">买点=${esc(patternAnyRuleText(buyRule, PATTERN_BACKTEST_BUY_CODES))}</span>` +
          `<span class="radar-backtest-legend-note">卖点=${esc(patternAnyRuleText(sellRule, DISTRIBUTION_WARNING_PATTERN_CODES))}</span>` +
          `<span class="radar-backtest-legend-note">疑似吸筹买点=未命中形态且吸筹分≥${fmt(suspectThreshold, 0)}</span>` +
          `<span class="radar-backtest-legend-note">${esc(scope)}</span>`;
      } else {
        legend.hidden = true;
        legend.textContent = "";
      }

      const canvas = $("radar-kline-canvas");
      if (canvas) {
        canvas.setAttribute(
          "aria-label",
          patternBacktestActive
            ? `股票 K 线图，完整区间已标记 ${counts.dates} 个回测信号交易日`
            : "股票 K 线图",
        );
      }
    }

    function resetPatternBacktest() {
      patternBacktestFetchSeq += 1;
      patternBacktestHits = [];
      patternBacktestMeta = null;
      patternBacktestLoaded = false;
      patternBacktestActive = false;
      patternBacktestLoading = false;
      patternBacktestError = "";
      klinePatternMarkers = new Map();
      renderPatternBacktestUI();
    }

    function clearKlinePresentation() {
      loadedKlineCode = null;
      loadedKlinePeriod = null;
      klineBars = [];
      klineLayout = null;
      hoverIdx = null;
      klineView = { start: 0, end: 0 };
      klineDrag = null;
      klinePatternMarkers = new Map();
      updateKlineDetail(null);
      drawKline();
      renderPatternBacktestUI();
    }

    function clearSelectedStock() {
      selectedCode = null;
      klineFetchSeq += 1;
      resetPatternBacktest();
      clearKlinePresentation();
      const table = $("radar-table");
      if (table) {
        table.querySelectorAll(".rrow").forEach((tr) => {
          tr.classList.remove("is-selected");
          tr.setAttribute("aria-selected", "false");
        });
      }
      $("radar-kline-title").textContent = "点击左侧个股查看 K 线";
      $("radar-kline-sub").textContent = "";
      renderStockInfo({}, []);
      renderPatternInfo(null);
    }

    function rebuildKlinePatternMarkers() {
      const markers = new Map();
      if (!patternBacktestActive || !patternBacktestHits.length || !klineBars.length) {
        klinePatternMarkers = markers;
        return;
      }
      let barIndex = 0;
      patternBacktestHits.forEach((hit) => {
        while (barIndex < klineBars.length && String(klineBars[barIndex].date || "") < hit.date) barIndex += 1;
        if (barIndex >= klineBars.length) return;
        const bar = klineBars[barIndex];
        const start = String(bar.startDate || bar.date || "");
        const end = String(bar.date || "");
        if (hit.date < start || hit.date > end) return;
        const marker = markers.get(barIndex) || { dates: [], patterns: [], suspectAccumulations: [] };
        if (!marker.dates.includes(hit.date)) marker.dates.push(hit.date);
        if (hit.is_suspect_accumulation) {
          marker.suspectAccumulations.push({
            date: hit.date,
            score: hit.accumulation_score,
            threshold: hit.accumulation_threshold,
          });
        }
        hit.patterns.forEach((pattern) => {
          if (!marker.patterns.some((item) => item.code === pattern.code && item.effective_style === pattern.effective_style)) {
            marker.patterns.push(pattern);
          }
        });
        markers.set(barIndex, marker);
      });
      klinePatternMarkers = markers;
    }

    function patternMarkerGroups(index) {
      const marker = klinePatternMarkers.get(index);
      if (!marker) return [];
      return Object.keys(PATTERN_MARKER_META).map((style) => ({
        style,
        marker,
        patterns: marker.patterns.filter((pattern) => patternStyle(pattern) === style),
      })).filter((group) => group.patterns.length);
    }

    function refreshPatternBacktestChart() {
      rebuildKlinePatternMarkers();
      renderPatternBacktestUI();
      drawKline();
      updateKlineDetail(klineBars.length ? (hoverIdx === null ? klineBars.length - 1 : hoverIdx) : null);
    }

    async function togglePatternBacktest() {
      const klineReady = Boolean(
        selectedCode &&
        loadedKlineCode === selectedCode &&
        loadedKlinePeriod === klinePeriod &&
        klineBars.length
      );
      if (!klineReady || patternBacktestLoading) return;
      if (patternBacktestActive) {
        patternBacktestActive = false;
        refreshPatternBacktestChart();
        return;
      }
      if (patternBacktestLoaded) {
        patternBacktestActive = true;
        refreshPatternBacktestChart();
        return;
      }

      const code = selectedCode;
      const pool = (($("radar-pool") && $("radar-pool").value) || "leader");
      const seq = ++patternBacktestFetchSeq;
      patternBacktestLoading = true;
      patternBacktestError = "";
      renderPatternBacktestUI();
      try {
        const url = `/api/radar/pattern-backtest?code=${encodeURIComponent(code)}&pool=${encodeURIComponent(pool)}&limit=${KLINE_HISTORY_LIMIT}`;
        const resp = await fetch(url);
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "形态回测请求失败");
        const currentPool = (($("radar-pool") && $("radar-pool").value) || "leader");
        if (disposed || selectedCode !== code || seq !== patternBacktestFetchSeq) return;
        if (currentPool !== pool) { resetPatternBacktest(); return; }
        patternBacktestHits = normalizePatternBacktestHits(data.hits, data);
        patternBacktestMeta = data;
        patternBacktestLoaded = true;
        patternBacktestActive = true;
        patternBacktestLoading = false;
        refreshPatternBacktestChart();
      } catch (err) {
        const currentPool = (($("radar-pool") && $("radar-pool").value) || "leader");
        if (disposed || selectedCode !== code || seq !== patternBacktestFetchSeq) return;
        if (currentPool !== pool) { resetPatternBacktest(); return; }
        patternBacktestLoading = false;
        patternBacktestLoaded = false;
        patternBacktestActive = false;
        patternBacktestError = (err && err.message) || "请稍后重试";
        refreshPatternBacktestChart();
      }
    }

    async function selectStock(code) {
      const changed = selectedCode !== code;
      selectedCode = code;
      if (changed) {
        klineFetchSeq += 1;
        resetPatternBacktest();
      }
      $("radar-table").querySelectorAll(".rrow").forEach((tr) => {
        const active = tr.dataset.code === code;
        tr.classList.toggle("is-selected", active);
        tr.setAttribute("aria-selected", active ? "true" : "false");
      });
      const s = stocks.find((x) => x.code === code) || {};
      $("radar-kline-title").textContent = `${code} ${s.name || ""}`;
      $("radar-kline-sub").textContent = "加载K线…";
      renderStockInfo(s, []);
      renderPatternInfo(s);
      await loadKline(code);
    }

    function setKlinePeriod(period) {
      if (!KLINE_PERIOD_LABEL[period] || period === klinePeriod) return;
      klinePeriod = period;
      $("radar-kline-periods").querySelectorAll("button").forEach((btn) => {
        const active = btn.dataset.period === period;
        btn.classList.toggle("is-active", active);
        btn.setAttribute("aria-selected", active ? "true" : "false");
      });
      if (selectedCode) loadKline(selectedCode);
      else drawKline();
    }

    async function loadKline(code) {
      const period = klinePeriod;
      const seq = ++klineFetchSeq;
      clearKlinePresentation();
      $("radar-kline-sub").textContent = `加载${KLINE_PERIOD_LABEL[period]}…`;
      try {
        const url = `/api/radar/kline?code=${encodeURIComponent(code)}&period=${period}&limit=${KLINE_HISTORY_LIMIT}`;
        const resp = await fetch(url);
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "K线加载失败");
        if (disposed || selectedCode !== code || klinePeriod !== period || seq !== klineFetchSeq) return;
        klineBars = (data.bars || []).map((b) => ({
          date: b.date,
          startDate: b.start_date || b.date,
          open: +b.open,
          high: +b.high,
          low: +b.low,
          close: +b.close,
          volume: +b.volume || 0,
        }));
        loadedKlineCode = code;
        loadedKlinePeriod = period;
        renderPatternBacktestUI();
        rebuildKlinePatternMarkers();
        hoverIdx = null;
        resetKlineView();
        updateKlineSub();
        drawKline();
        updateKlineDetail(klineBars.length ? klineBars.length - 1 : null);
        const selected = stocks.find((s) => s.code === code);
        if (selected) renderStockInfo(selected, klineBars);
      } catch (err) {
        if (disposed || selectedCode !== code || klinePeriod !== period || seq !== klineFetchSeq) return;
        clearKlinePresentation();
        $("radar-kline-sub").textContent = "K线加载失败";
      }
    }

    function stockInfoNumber(value) {
      if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return null;
      return Number(value);
    }

    function stockInfoIndustry(name, heat) {
      const label = String(name || "").trim();
      if (!label) return "—";
      const value = stockInfoNumber(heat);
      return value === null ? label : `${label} · 热${Math.round(value)}`;
    }

    function setStockInfoTone(element, value) {
      if (!element) return;
      element.classList.remove("up", "down");
      if (value > 0) element.classList.add("up");
      else if (value < 0) element.classList.add("down");
    }

    function renderStockInfo(s, bars = []) {
      const root = $("radar-stock-info");
      if (!root) return;
      const hasStock = Boolean(s && s.code);
      root.classList.toggle("is-empty", !hasStock);

      const list = Array.isArray(bars) ? bars : [];
      const latest = list.length ? list[list.length - 1] : null;
      const previous = list.length > 1 ? list[list.length - 2] : null;
      const livePrice = hasStock ? stockInfoNumber(s.realtime_price) : null;
      const liveChange = hasStock ? stockInfoNumber(s.realtime_change_pct) : null;
      const historicalChange = latest && previous && previous.close
        ? (latest.close / previous.close - 1) * 100
        : null;
      const price = livePrice !== null ? livePrice : stockInfoNumber(latest && latest.close);
      const change = liveChange !== null ? liveChange : historicalChange;
      const realtime = livePrice !== null;

      $("radar-stock-price").textContent = price === null ? "—" : fmt(price, 2);
      $("radar-stock-change").textContent = change === null ? "—" : fmtSignedPct(change, 2);
      $("radar-stock-cap").textContent = hasStock ? fmtMarketCap(isEtfMode() ? s.fund_scale_yi : s.market_cap_yi) : "—";
      $("radar-stock-change-label").textContent = realtime
        ? "实时涨跌"
        : (latest ? `${KLINE_PERIOD_LABEL[klinePeriod]}涨跌` : "涨跌幅");

      const rt = (payload && payload.realtime_quote) || {};
      const source = REALTIME_SOURCE_LABEL[rt.source] || "实时";
      $("radar-stock-price-source").textContent = realtime
        ? `${source} · ${s.realtime_quote_time || rt.updated_at || "刚刚"}`
        : (latest ? `${KLINE_PERIOD_LABEL[klinePeriod]}收盘 · ${latest.date || s.last_date || ""}` : "等待K线");

      $("radar-stock-sw2").textContent = hasStock ? stockInfoIndustry(s.parent_segment, s.sw2_heat_pctile) : "—";
      $("radar-stock-sw3").textContent = hasStock ? (String(s.segment_name || "").trim() || "—") : "—";
      $("radar-stock-theme").textContent = hasStock ? stockInfoIndustry(s.tracking_theme, s.theme_heat_pctile) : "—";
      $("radar-stock-date").textContent = hasStock ? (s.last_date || (latest && latest.date) || "—") : "—";
      setStockInfoTone($("radar-stock-price"), change);
      setStockInfoTone($("radar-stock-change"), change);
    }

    function renderPatternInfo(s) {
      const el = $("radar-pattern-info");
      if (!s || !s.code) { el.innerHTML = "点击左侧个股查看其命中形态的含义。"; return; }
      const codes = s.patterns || [];
      const etf = isEtfMode();
      const meta = `<div class="pi-stock-meta"><span>${etf ? "规模" : "市值"} ${esc(fmtMarketCap(etf ? s.fund_scale_yi : s.market_cap_yi))}</span></div>`;
      let html;
      if (!codes.length) {
        html = meta + `<div class="pi-empty">${esc(s.name || s.code)} 未命中任何形态，当前阶段：${esc(s.pattern_phase || "")}。</div>`;
      } else {
        html = meta + codes.map((c) => {
          const m = patternMap[c] || { name: c, category: "", signal: "", desc: "" };
          const acc = SIG_ACCENT[m.signal] || "sd";
          const eff = m.effective
            ? `<span class="pi-eff">${etf ? "股票池实测·ETF待验证" : "★实测有效"}</span>`
            : "";
          const cat = [m.category, m.signal].filter(Boolean).join(" · ");
          return `<div class="pi-card ${acc}">
            <div class="pi-top">
              <span class="pat ${patternTagClass(c)}">${esc(c)}</span>
              <strong>${esc(m.name)}</strong>
              <span class="pi-cat">${esc(cat)}</span>${eff}
            </div>
            <div class="pi-desc">${esc(m.desc) || "（无说明）"}</div>
          </div>`;
        }).join("");
      }
      if (isDistributionConflict(s)) {
        html = `<div class="pi-conflict">⚠ 结构似吸筹（吸筹分 ${fmt(s.ambush_score, 0)}）却触发出货预警——` +
          `${esc(distributionWarningRuleText(s))}，仍应优先防守。</div>` + html;
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
      const patternHtml = patternMarkerGroups(idx).map((group) => {
        const marker = PATTERN_MARKER_META[group.style];
        const hasSuspectAccumulation = group.patterns.some((pattern) => pattern.synthetic === true);
        const codes = group.patterns.map((pattern) => pattern.synthetic ? "疑似吸筹" : pattern.code).join("·");
        const regularNames = group.patterns
          .filter((pattern) => !pattern.synthetic)
          .map((pattern) => `${pattern.code} ${pattern.name}`);
        const suspectNames = hasSuspectAccumulation
          ? (group.marker.suspectAccumulations || []).map((item) => {
              const score = item.score === null ? "-" : fmt(item.score, 1);
              const threshold = item.threshold === null ? "35" : fmt(item.threshold, 0);
              return `疑似吸筹（待确认） · 吸筹分 ${score}/阈值 ${threshold} · ${item.date}`;
            })
          : [];
        const names = regularNames.concat(suspectNames).join("、");
        const dates = group.marker.dates.join("、");
        const label = hasSuspectAccumulation ? "疑" : marker.label;
        return `<span class="kline-pattern-hit ${group.style}" title="${esc(names)} · ${esc(dates)}">${label} ${esc(codes)}</span>`;
      }).join("");
      el.innerHTML =
        `<span>${esc(dateText)}</span>` +
        `<span>开 ${fmt(b.open, 2)}</span><span>高 ${fmt(b.high, 2)}</span>` +
        `<span>低 ${fmt(b.low, 2)}</span><span>收 ${fmt(b.close, 2)}</span>` +
        `<span class="${up ? "up" : "down"}">${up ? "+" : ""}${fmt(chg, 2)}%</span>` +
        `<span>量 ${(b.volume / 1e4).toFixed(0)}万</span>` + patternHtml;
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

    function roundedRectPath(ctx, x, y, width, height, radius) {
      const r = Math.min(radius, width / 2, height / 2);
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + width - r, y);
      ctx.quadraticCurveTo(x + width, y, x + width, y + r);
      ctx.lineTo(x + width, y + height - r);
      ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
      ctx.lineTo(x + r, y + height);
      ctx.quadraticCurveTo(x, y + height, x, y + height - r);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath();
    }

    function drawPatternMarker(ctx, cx, anchorY, style, level, step, padT, volTop, label = "") {
      const meta = PATTERN_MARKER_META[style];
      if (!meta) return;
      const above = meta.placement === "above";
      if (step < 5.5) {
        const radius = step < 3.5 ? 1.7 : 2.4;
        const cy = above
          ? Math.max(padT + radius, anchorY - 6 - level * 7)
          : Math.min(volTop - radius, anchorY + 6 + level * 7);
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fillStyle = meta.color;
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, .9)";
        ctx.lineWidth = .8;
        ctx.stroke();
        ctx.restore();
        return;
      }
      const width = step < 7 ? 14 : 18;
      const height = 16;
      const offset = level * 19;
      const boxY = above
        ? Math.max(padT + 1, anchorY - height - 8 - offset)
        : Math.min(volTop - height - 2, anchorY + 8 + offset);
      const boxX = cx - width / 2;

      ctx.save();
      ctx.fillStyle = meta.color;
      ctx.shadowColor = "rgba(0, 0, 0, .16)";
      ctx.shadowBlur = 5;
      ctx.shadowOffsetY = 2;
      roundedRectPath(ctx, boxX, boxY, width, height, 5);
      ctx.fill();
      ctx.shadowColor = "transparent";
      ctx.beginPath();
      if (above) {
        const tipY = Math.min(anchorY - 1, boxY + height + 5);
        ctx.moveTo(cx - 3, boxY + height - 1);
        ctx.lineTo(cx + 3, boxY + height - 1);
        ctx.lineTo(cx, tipY);
      } else {
        const tipY = Math.max(anchorY + 1, boxY - 5);
        ctx.moveTo(cx - 3, boxY + 1);
        ctx.lineTo(cx + 3, boxY + 1);
        ctx.lineTo(cx, tipY);
      }
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = `${width < 16 ? 9 : 10}px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label || meta.label, cx, boxY + height / 2 + .5);
      ctx.restore();
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
      const markerPadding = patternBacktestActive && klinePatternMarkers.size ? 0.14 : 0.05;
      const pad = (hi - lo) * markerPadding || 1; lo -= pad; hi += pad;
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
      // 统一回测形态：绿色买点/橙色动量放在 K 线下方，红色预警卖点放在上方。
      for (let i = 0; i < n; i++) {
        const globalIndex = klineView.start + i;
        const groups = patternMarkerGroups(globalIndex);
        if (!groups.length) continue;
        const b = visibleBars[i];
        const cx = padL + step * i + step / 2;
        const levels = { above: 0, below: 0 };
        groups.forEach((group) => {
          const meta = PATTERN_MARKER_META[group.style];
          const placement = meta.placement;
          const anchor = placement === "above" ? y(b.high) : y(b.low);
          const markerLabel = group.patterns.some((pattern) => pattern.synthetic === true) ? "疑" : meta.label;
          drawPatternMarker(ctx, cx, anchor, group.style, levels[placement], step, padT, volTop, markerLabel);
          levels[placement] += 1;
        });
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

    // ---------- 任务（运行 / 刷新数据）----------
    function setBusy(busy) {
      $("radar-run").disabled = busy;
      $("radar-refresh-data").disabled = busy;
      syncPoolPresentation(busy);
      $("radar-pool").disabled = busy;
      renderPatternBacktestUI();
      if (busy) $("radar-pattern-backtest").disabled = true;
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
      const wasRadarRunBusy = radarRunBusy;
      const wasRunning = jobsWereRunning;
      const wasDataRunning = dataJobWasRunning;
      radarRunBusy = Boolean(runJob && runJob.running);
      const dataRunning = Boolean(dataJob && dataJob.running);
      const running = (runJob && runJob.running) || (dataJob && dataJob.running);
      jobsWereRunning = Boolean(running);
      dataJobWasRunning = dataRunning;
      setBusy(running);
      if (dataJob && (dataJob.running || activeJobType === "data")) showJobLog(dataJob, "数据刷新");
      else if (runJob && (runJob.running || activeJobType === "run")) showJobLog(runJob, "运行");
      if (running && !pollTimer) pollTimer = setInterval(fetchJobState, 2000);
      if (!running && pollTimer) { clearInterval(pollTimer); pollTimer = null; }
      if (wasRunning && !running) {
        const dataRefreshCompleted = wasDataRunning && !dataRunning;
        let reloadKlineSeq = null;
        resetPatternBacktest();
        if (dataRefreshCompleted) invalidateIndustryHeat();
        if (dataRefreshCompleted && selectedCode) {
          reloadKlineSeq = ++klineFetchSeq;
          clearKlinePresentation();
          $("radar-kline-sub").textContent = "数据已更新，重新加载K线…";
        }
        fetchData().then((data) => {
          if (
            data && dataRefreshCompleted && selectedCode &&
            reloadKlineSeq !== null && klineFetchSeq === reloadKlineSeq
          ) loadKline(selectedCode);
        });
      }
      if (wasRadarRunBusy && !radarRunBusy && realtimeRefreshPending) {
        realtimeRefreshPending = false;
        fetchRealtimeData(true);
      }
    }

    async function startJob(url, label, body) {
      if (url === "/api/radar/run") {
        radarRunBusy = true;
        if ($("radar-live-quote").checked) {
          realtimeRefreshPending = true;
          updateRealtimeStatus("等待新股票池…", "live");
        }
      }
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
        } else if (resp.ok) {
          jobsWereRunning = true;
          if (activeJobType === "data") dataJobWasRunning = true;
        }
      } catch (err) {
        $("radar-status").textContent = label + "启动失败";
      }
      await fetchJobState();
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
      // 切换候选池即自动重跑
      $("radar-pool").addEventListener("change", () => {
        clearSelectedStock();
        syncPoolPresentation(false);
        const labels = { hotmoney: "游资小盘池(反转分)", etf: "ETF池(技术机会分)", leader: "细分龙头池" };
        const label = labels[$("radar-pool").value] || labels.leader;
        startJob("/api/radar/run", `运行 · ${label}`, runBody());
      });
      $("radar-refresh-data").onclick = () => {
        const message = "刷新数据将统一更新全市场股票、ETF配置池、板块和题材；ETF只抓行情，不抓财报、股东户数或回购。任务耗时较长，期间请勿关闭页面。\n\n确认现在开始吗？";
        if (!window.confirm(message)) return;
        startJob("/api/radar/refresh-data", "全量数据刷新", null);
      };
      $("radar-live-quote").addEventListener("change", (ev) => setRealtimeEnabled(ev.target.checked));
      $("radar-pattern-trigger").onclick = () => setPatternPanelOpen($("radar-pattern-panel").hidden);
      $("radar-pattern-search").addEventListener("input", renderPatternFilter);
      $("radar-pattern-clear").onclick = () => selectPatternPreset(() => false);
      $("radar-pattern-select-all").onclick = () => {
        const visibleCodes = new Set(visiblePatternOptions().map((option) => option.code));
        selectPatternPreset((option) => visibleCodes.has(option.code));
      };
      $("radar-pattern-buy").onclick = () => selectPatternPreset(isPatternBuyPreset);
      $("radar-pattern-sell").onclick = () => selectPatternPreset(isPatternSellPreset);
      $("radar-pattern-rally").onclick = () => selectPatternPreset(isPatternRallyPreset);
      $("radar-sw2-trigger").onclick = () => setSw2PanelOpen($("radar-sw2-panel").hidden);
      $("radar-sw2-search").addEventListener("input", renderSw2Filter);
      $("radar-sw2-min-heat").addEventListener("input", renderSw2Filter);
      $("radar-sw2-clear").onclick = () => {
        selectedSw2Industries.clear();
        renderSw2Filter();
        applyFilters();
      };
      $("radar-sw2-hot").onclick = () => {
        selectedSw2Industries = new Set(sw2Options.filter((o) => o.heat !== null).slice(0, 10).map((o) => o.name));
        renderSw2Filter();
        applyFilters();
      };
      $("radar-sw2-select-all").onclick = () => {
        selectedSw2Industries = new Set(visibleSw2Options().map((o) => o.name));
        renderSw2Filter();
        applyFilters();
      };
      ["radar-search", "radar-phase-filter", "radar-min-score", "radar-hide-dist"].forEach((id) => {
        const el = $(id);
        el.addEventListener(el.tagName === "SELECT" || el.type === "checkbox" ? "change" : "input", applyFilters);
      });
      $("radar-export").onclick = exportTopOpportunityStocks;
      $("radar-industry-heat").onclick = openIndustryHeat;
      $("radar-score-help").onclick = () => openInsightModal("radar-score-modal");
      $("radar-factor-help").onclick = () => openInsightModal("radar-factor-modal");
      $("radar-pattern-backtest").onclick = togglePatternBacktest;
      root.querySelectorAll(".radar-insight-modal").forEach((modal) => {
        modal.addEventListener("click", (event) => { if (event.target === modal) closeInsightModal(); });
        modal.querySelectorAll("[data-radar-modal-close]").forEach((button) => { button.onclick = closeInsightModal; });
      });
      const canvas = $("radar-kline-canvas");
      $("radar-kline-periods").querySelectorAll("button").forEach((btn) => {
        btn.addEventListener("click", () => setKlinePeriod(btn.dataset.period));
      });
      canvas.addEventListener("pointerdown", onCanvasDown);
      canvas.addEventListener("pointermove", onCanvasMove);
      canvas.addEventListener("pointerup", onCanvasUp);
      canvas.addEventListener("pointercancel", onCanvasUp);
      canvas.addEventListener("mouseleave", () => {
        if (klineDrag) return;
        hoverIdx = null;
        drawKline();
        updateKlineDetail(klineBars.length ? klineBars.length - 1 : null);
      });
      window.addEventListener("resize", onResize);
      document.addEventListener("click", onDocumentClick);
      document.addEventListener("keydown", onDocumentKeyDown);
      renderPatternBacktestUI();
      Promise.all([fetchPatternCatalog(), fetchScoringModel()]).finally(fetchData);
    }

    function onResize() { if (!disposed) drawKline(); }

    window.FinancialAnalysisPages = window.FinancialAnalysisPages || {};
    window.FinancialAnalysisPages.cleanup = () => {
      disposed = true;
      industryHeatFetchSeq += 1;
      if (pollTimer) clearInterval(pollTimer);
      if (realtimeTimer) clearInterval(realtimeTimer);
      if (activeInsightModal) activeInsightModal.hidden = true;
      activeInsightModal = null;
      insightReturnFocus = null;
      document.body.classList.remove("radar-modal-open");
      window.removeEventListener("resize", onResize);
      document.removeEventListener("click", onDocumentClick);
      document.removeEventListener("keydown", onDocumentKeyDown);
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
