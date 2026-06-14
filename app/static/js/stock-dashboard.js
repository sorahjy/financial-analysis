(() => {
  function initStockDashboard() {
    const dashboardRoot = document.querySelector("#stock-native-dashboard");
    if (!dashboardRoot || dashboardRoot.dataset.stockDashboardInitialized === "1") return;
    dashboardRoot.dataset.stockDashboardInitialized = "1";

    let disposed = false;
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
    let refreshTimer = null;
    let refreshShouldRunAfterDone = false;
    let runInFlight = false;
    let runAgainAfterDone = false;

    const $ = (id) => document.getElementById(id);
    const status = (text) => {
      const node = $("status");
      if (node) node.textContent = text;
    };
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

    async function fetchHealth() {
      const resp = await fetch("/api/health");
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload.error || "health failed");
      return payload;
    }

    async function fetchLatestResult() {
      const resp = await fetch("/api/stock/latest");
      const payload = await resp.json();
      if (!resp.ok) throw new Error(payload.error || "latest failed");
      return payload;
    }

    async function init() {
      let hasSavedConfig = false;
      try {
        const saved = localStorage.getItem("stockStrategyConfig");
        hasSavedConfig = Boolean(saved);
        const payload = await fetchConfig();
        if (disposed) return;
        config = saved ? mergeDeep(payload.config, JSON.parse(saved)) : payload.config;
        registry = payload.factors;
      } catch (err) {
        status("失败");
        $("table").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
        return;
      }
      bindEvents();
      renderControls();
      await loadDataRefreshStatus();
      if (disposed) return;
      await resumeOptimizeIfRunning();
      if (disposed) return;
      await resumeDataRefreshIfRunning();
      if (disposed) return;
      const loaded = await loadCachedResult();
      if (disposed) return;
      if (loaded) {
        status(hasSavedConfig ? "已加载上次结果，点击运行更新" : "已加载");
      } else {
        renderPendingResult("正在计算策略结果...");
        await runNow();
      }
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
      $("refresh-data").onclick = startDataRefresh;
    }

    function setRefreshButtonBusy(busy) {
      $("refresh-data").disabled = busy;
    }

    function renderDataRefreshStatus(payload) {
      const report = payload.refresh || {};
      const job = payload.refresh_job || {};
      const health = report.health || {};
      const reportTime = report.finished_at || health.strategy_generated_at || health.capital_generated_at || "未刷新";
      const mode = report.mode ? ` · ${report.mode}` : "";
      const okText = report.ok === true ? " · 成功" : (report.ok === false ? " · 失败" : "");
      $("data-refresh-time").textContent = `${reportTime}${mode}${okText}`;

      const lines = Array.isArray(job.log_lines) ? job.log_lines : [];
      const hasJob = Boolean(job.running || job.ok === true || job.ok === false || lines.length);
      $("data-refresh-log-panel").hidden = !hasJob;
      $("data-refresh-log-count").textContent = `${lines.length} 行`;
      $("data-refresh-log").textContent = lines.length ? lines.join("\n") : "等待刷新任务输出...";
      $("data-refresh-log").scrollTop = $("data-refresh-log").scrollHeight;
      setRefreshButtonBusy(Boolean(job.running));
      return job;
    }

    async function loadDataRefreshStatus() {
      try {
        const payload = await fetchHealth();
        if (disposed) return null;
        return renderDataRefreshStatus(payload);
      } catch (err) {
        $("data-refresh-time").textContent = "读取失败";
        return null;
      }
    }

    async function startDataRefresh() {
      if ($("refresh-data").disabled) return;
      const msg = "本次会执行股票数据全量拉取（mode=full，no-proxy），数据刷新时间会很久，大约10-30分钟。确定开始吗？";
      if (!window.confirm(msg)) return;
      setRefreshButtonBusy(true);
      $("data-refresh-log-panel").hidden = false;
      $("data-refresh-log").textContent = "正在启动刷新任务...";
      status("刷新数据中");
      refreshShouldRunAfterDone = true;
      try {
        const resp = await fetch("/api/stock/refresh", {method: "POST"});
        const payload = await resp.json();
        if (disposed) return;
        if (!resp.ok) throw new Error(payload.error || "启动失败");
      } catch (err) {
        status("刷新失败");
        $("data-refresh-log").textContent = esc(err.message);
        setRefreshButtonBusy(false);
        return;
      }
      clearInterval(refreshTimer);
      refreshTimer = setInterval(pollDataRefreshStatus, 2000);
      await pollDataRefreshStatus();
    }

    async function pollDataRefreshStatus() {
      const job = await loadDataRefreshStatus();
      if (disposed) return;
      if (!job || job.running) return;
      clearInterval(refreshTimer);
      refreshTimer = null;
      setRefreshButtonBusy(false);
      if (!refreshShouldRunAfterDone) return;
      refreshShouldRunAfterDone = false;
      if (job.ok) {
        status(`数据已刷新(${job.elapsed_sec ?? "?"}s)`);
        try {
          const payload = await fetchConfig();
          if (disposed) return;
          config = mergeDeep(payload.config, JSON.parse(localStorage.getItem("stockStrategyConfig") || "{}"));
          registry = payload.factors;
          renderControls();
          await runNow();
        } catch (err) {
          status("刷新后运行失败");
          $("table").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
        }
      } else if (job.ok === false) {
        status("刷新失败");
      }
    }

    async function resumeDataRefreshIfRunning() {
      const job = await loadDataRefreshStatus();
      if (job && job.running) {
        refreshShouldRunAfterDone = true;
        clearInterval(refreshTimer);
        refreshTimer = setInterval(pollDataRefreshStatus, 2000);
      }
    }

    async function startOptimize() {
      const msg = "将对长线/短线各运行 200 次参数搜索回测"
        + "（相当于 python stock_strategy_optimizer.py --iterations 200），"
        + "约需 1 分钟左右，期间页面将锁定。确定开始吗？";
      if (!window.confirm(msg)) return;
      try {
        const resp = await fetch("/api/optimize", {method: "POST"});
        const payload = await resp.json();
        if (disposed) return;
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
      if (disposed) return;
      if (state.running) return;
      clearInterval(optimizeTimer);
      optimizeTimer = null;
      $("optimize-overlay").hidden = true;
      if (state.ok) {
        // 旧的本地保存配置会覆盖新搜出的默认参数，搜索成功后直接作废
        localStorage.removeItem("stockStrategyConfig");
        try {
          const payload = await fetchConfig();
          if (disposed) return;
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
        if (disposed) return;
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
        bindRangeDrag(range, (value) => {
          num.value = setWeight(value).toFixed(2);
        });
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

    function bindRangeDrag(range, onValue) {
      let dragging = false;

      const numberFromAttr = (name, fallback) => {
        const value = Number(range.getAttribute(name));
        return Number.isFinite(value) ? value : fallback;
      };
      const min = numberFromAttr("min", 0);
      const max = numberFromAttr("max", 100);
      const step = numberFromAttr("step", 1);
      const decimals = String(range.getAttribute("step") || "").split(".")[1]?.length || 0;

      const valueAt = (clientX) => {
        const rect = range.getBoundingClientRect();
        const ratio = rect.width <= 0 ? 0 : Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        const raw = min + ratio * (max - min);
        const stepped = min + Math.round((raw - min) / step) * step;
        return Math.max(min, Math.min(max, stepped)).toFixed(decimals);
      };

      const applyPointer = (event) => {
        const value = valueAt(event.clientX);
        if (range.value !== value) {
          range.value = value;
          onValue(value);
        }
      };

      range.addEventListener("pointerdown", (event) => {
        if (range.disabled || (event.button !== undefined && event.button !== 0)) return;
        dragging = true;
        event.preventDefault();
        range.focus();
        range.setPointerCapture && range.setPointerCapture(event.pointerId);
        applyPointer(event);
      });

      range.addEventListener("pointermove", (event) => {
        if (!dragging) return;
        event.preventDefault();
        applyPointer(event);
      });

      const stopDragging = (event) => {
        if (!dragging) return;
        dragging = false;
        if (range.hasPointerCapture && range.hasPointerCapture(event.pointerId)) {
          range.releasePointerCapture(event.pointerId);
        }
      };

      range.addEventListener("pointerup", stopDragging);
      range.addEventListener("pointercancel", stopDragging);
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

    async function loadCachedResult() {
      try {
        const payload = await fetchLatestResult();
        if (disposed) return false;
        if (!payload.cached || !payload.result) return false;
        latest = payload.result;
        renderResults();
        return true;
      } catch (err) {
        return false;
      }
    }

    function renderPendingResult(message) {
      $("subtitle").textContent = "";
      $("generated").textContent = "";
      $("metrics").innerHTML = "";
      $("pool-rules").innerHTML = "";
      $("notes").innerHTML = "";
      $("bars").innerHTML = `<div class="empty">${esc(message)}</div>`;
      $("table").innerHTML = `<div class="empty">${esc(message)}</div>`;
    }

    async function runNow() {
      if (runInFlight) {
        runAgainAfterDone = true;
        return;
      }
      runInFlight = true;
      runAgainAfterDone = false;
      $("run").disabled = true;
      try {
        status("运行中");
        const resp = await fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({config})
        });
        const payload = await resp.json();
        if (disposed) return;
        if (!resp.ok) throw new Error(payload.error || "run failed");
        latest = payload;
        status("已完成");
        renderResults();
      } catch (err) {
        status("失败");
        $("table").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
      } finally {
        runInFlight = false;
        $("run").disabled = false;
        if (runAgainAfterDone && !disposed) {
          runAgainAfterDone = false;
          scheduleRun();
        }
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

    window.FinancialAnalysisPages = window.FinancialAnalysisPages || {};
    window.FinancialAnalysisPages.cleanup = () => {
      disposed = true;
      clearTimeout(runTimer);
      clearInterval(optimizeTimer);
      clearInterval(refreshTimer);
    };
    init();
  }

  window.FinancialAnalysisPages = window.FinancialAnalysisPages || {};
  window.FinancialAnalysisPages.stock = initStockDashboard;
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initStockDashboard, { once: true });
  } else {
    initStockDashboard();
  }
})();
