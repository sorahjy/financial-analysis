(() => {
  initNativeReport(document.querySelector("#fund-native-report"));

  const statusRoot = document.querySelector("[data-fund-status]");
  if (!statusRoot) return;
  const actionButtons = Array.from(document.querySelectorAll("[data-post-url]"));
  const logRoot = document.querySelector("[data-fund-log]");
  const logText = logRoot ? logRoot.querySelector('[data-field="job-log"]') : null;
  const logCommand = logRoot ? logRoot.querySelector('[data-field="job-command"]') : null;
  const logCount = logRoot ? logRoot.querySelector('[data-field="log-count"]') : null;

  const setText = (name, value) => {
    const node = statusRoot.querySelector(`[data-field="${name}"]`);
    if (node) node.textContent = value;
  };

  const latestTimestamp = (...values) => {
    const present = values.filter(Boolean);
    return present.sort().pop() || "未生成";
  };

  const describeRefresh = (state) => {
    if (!state || state.ok === null) return "空闲";
    if (state.running) return "运行中";
    if (state.ok === true) return `完成 ${state.elapsed_sec || 0}s`;
    if (state.ok === false) return `失败 ${state.error || ""}`;
    return "空闲";
  };

  const setRefreshBusy = (busy) => {
    actionButtons.forEach((button) => {
      button.disabled = busy;
      button.classList.toggle("is-busy", busy);
      button.setAttribute("aria-busy", busy ? "true" : "false");
    });
  };

  const updateRefreshLog = (state) => {
    if (!logRoot) return;
    const lines = Array.isArray(state && state.log_lines) ? state.log_lines : [];
    const command = state && (state.command_text || (Array.isArray(state.command) ? state.command.join(" ") : ""));
    const hasTask = Boolean(state && (state.running || state.ok !== null || lines.length));

    logRoot.hidden = !hasTask;
    if (logCommand) logCommand.textContent = command || "未启动";
    if (logCount) logCount.textContent = `${lines.length} 行`;
    if (logText) {
      logText.textContent = lines.length ? lines.join("\n") : "等待刷新任务输出...";
      logText.scrollTop = logText.scrollHeight;
    }
  };

  async function refreshStatus() {
    const resp = await fetch("/api/fund/status");
    const data = await resp.json();
    setText("data-updated", latestTimestamp(data.report.updated_at, data.signals.updated_at));
    setText("job-state", describeRefresh(data.refresh));
    setRefreshBusy(Boolean(data.refresh && data.refresh.running));
    updateRefreshLog(data.refresh);
    return data;
  }

  async function postAction(button) {
    const url = button.dataset.postUrl;
    if (!url || button.disabled) return;
    setRefreshBusy(true);
    try {
      const resp = await fetch(url, { method: "POST" });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "任务启动失败");
      await pollUntilIdle();
    } catch (error) {
      setText("job-state", error.message);
    } finally {
      await refreshStatus();
    }
  }

  async function pollUntilIdle() {
    for (let i = 0; i < 900; i += 1) {
      const data = await refreshStatus();
      if (!data.refresh.running) {
        if (data.refresh.ok) {
          window.location.reload();
        }
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }

  actionButtons.forEach((button) => {
    button.addEventListener("click", () => postAction(button));
  });

  refreshStatus();

  function initNativeReport(root) {
    if (!root) return;

    const $ = (selector, scope = root) => scope.querySelector(selector);
    const $$ = (selector, scope = root) => Array.from(scope.querySelectorAll(selector));
    const searchInput = $("#fund-search");
    const chips = $$(".filter-chip");
    const densityToggle = $("#density-toggle");
    const topButton = $("#top-button");
    let activeFilter = "all";

    const normalize = (value) => String(value || "").trim().toLowerCase();
    const numericValue = (text) => {
      const cleaned = String(text || "").replace(/[,%\s]/g, "");
      const value = Number.parseFloat(cleaned);
      return Number.isFinite(value) ? value : null;
    };

    function rowMatches(row) {
      const q = normalize(searchInput ? searchInput.value : "");
      const haystack = normalize([
        row.dataset.code,
        row.dataset.name,
        row.dataset.signal,
        row.textContent,
      ].join(" "));
      if (q && !haystack.includes(q)) return false;

      if (activeFilter === "held") return row.dataset.held === "1";
      if (activeFilter === "buy") return row.dataset.signal === "买入";
      if (activeFilter === "sell") return row.dataset.signal === "卖出" || row.dataset.signal === "force-sell";
      return true;
    }

    function updateSectionCounts() {
      $$(".report-section").forEach((section) => {
        const rows = $$("tbody tr", section);
        const visible = rows.filter((row) => !row.hidden).length;
        const counter = $(".visible-count", section);
        if (counter) counter.textContent = visible;
        section.classList.toggle("no-match", rows.length > 0 && visible === 0);
      });
    }

    function applyFilters() {
      $$("tbody tr").forEach((row) => {
        row.hidden = !rowMatches(row);
      });
      updateSectionCounts();
    }

    function setPressed(button, pressed) {
      if (!button) return;
      button.classList.toggle("active", pressed);
      button.setAttribute("aria-pressed", pressed ? "true" : "false");
    }

    chips.forEach((chip) => {
      chip.addEventListener("click", () => {
        activeFilter = chip.dataset.filter || "all";
        chips.forEach((item) => item.classList.toggle("active", item === chip));
        applyFilters();
      });
    });

    if (searchInput) {
      searchInput.addEventListener("input", applyFilters);
    }

    if (localStorage.getItem("fund-report-density") === "compact") {
      root.classList.add("compact");
      if (densityToggle) densityToggle.textContent = "舒展";
      setPressed(densityToggle, true);
    }

    if (densityToggle) {
      densityToggle.addEventListener("click", () => {
        const compact = root.classList.toggle("compact");
        localStorage.setItem("fund-report-density", compact ? "compact" : "normal");
        densityToggle.textContent = compact ? "舒展" : "紧凑";
        setPressed(densityToggle, compact);
      });
    }

    if (topButton) {
      topButton.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
    }

    $$("table").forEach((table) => {
      const header = table.tHead ? table.tHead.rows[table.tHead.rows.length - 1] : null;
      if (!header || !table.tBodies.length) return;
      Array.from(header.cells).forEach((th, index) => {
        th.classList.add("sortable");
        th.addEventListener("click", () => {
          const direction = th.dataset.sort === "asc" ? "desc" : "asc";
          Array.from(header.cells).forEach((cell) => {
            cell.dataset.sort = "";
            cell.classList.remove("sort-asc", "sort-desc");
          });
          th.dataset.sort = direction;
          th.classList.add(direction === "asc" ? "sort-asc" : "sort-desc");

          const tbody = table.tBodies[0];
          const rows = Array.from(tbody.rows);
          rows.sort((a, b) => {
            const av = a.cells[index] ? a.cells[index].innerText : "";
            const bv = b.cells[index] ? b.cells[index].innerText : "";
            const an = numericValue(av);
            const bn = numericValue(bv);
            let result;
            if (an !== null && bn !== null) {
              result = an - bn;
            } else {
              result = av.localeCompare(bv, "zh-Hans-CN");
            }
            return direction === "asc" ? result : -result;
          });
          rows.forEach((row) => tbody.appendChild(row));
          applyFilters();
        });
      });
    });

    applyFilters();
  }
})();
