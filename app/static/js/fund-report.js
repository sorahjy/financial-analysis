(() => {
  function initFundPage() {
    const esc = (text) => String(text ?? "").replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[ch]));

    let disposed = false;
    const reportRoot = document.querySelector("#fund-native-report");
    if (reportRoot && reportRoot.dataset.fundReportInitialized !== "1") {
      reportRoot.dataset.fundReportInitialized = "1";
      initNativeReport(reportRoot);
    }

    const statusRoot = document.querySelector("[data-fund-status]");
    if (!statusRoot) return;
    if (statusRoot.dataset.fundPageInitialized === "1") return;
    statusRoot.dataset.fundPageInitialized = "1";
    window.FinancialAnalysisPages = window.FinancialAnalysisPages || {};
    window.FinancialAnalysisPages.cleanup = () => {
      disposed = true;
    };
    const actionButtons = Array.from(document.querySelectorAll("[data-post-url]"));
    const logRoot = document.querySelector("[data-fund-log]");
    const logText = logRoot ? logRoot.querySelector('[data-field="job-log"]') : null;
    const logCommand = logRoot ? logRoot.querySelector('[data-field="job-command"]') : null;
    const logCount = logRoot ? logRoot.querySelector('[data-field="log-count"]') : null;
    initFundEditor();

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
      if (disposed) return data;
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

    async function reloadCurrentPageContent() {
      const navigate = window.FinancialAnalysisPages && window.FinancialAnalysisPages.navigate;
      if (typeof navigate === "function") {
        const handled = await navigate(window.location.href, false);
        if (handled) return;
      }
      window.location.reload();
    }

    async function pollUntilIdle() {
      for (let i = 0; i < 900; i += 1) {
        const data = await refreshStatus();
        if (disposed) return;
        if (!data.refresh.running) {
          if (data.refresh.ok) {
            await reloadCurrentPageContent();
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

    function initFundEditor() {
      const openButton = document.querySelector("[data-fund-editor-open]");
      const modal = document.querySelector("[data-fund-editor-modal]");
      if (!openButton || !modal) return;

      const codeInput = modal.querySelector("[data-fund-editor-code]");
      const highlight = modal.querySelector("[data-fund-editor-highlight]");
      const pathLabel = modal.querySelector("[data-fund-editor-path]");
      const statusLabel = modal.querySelector("[data-fund-editor-status]");
      const saveButton = modal.querySelector("[data-fund-editor-save]");
      const cancelButtons = Array.from(modal.querySelectorAll("[data-fund-editor-cancel]"));
      let originalContent = "";

      const setEditorStatus = (text) => {
        if (statusLabel) statusLabel.textContent = text;
      };

      const setEditorBusy = (busy) => {
        if (saveButton) saveButton.disabled = busy;
        if (codeInput) codeInput.disabled = busy;
      };

      const renderHighlight = () => {
        if (!highlight || !codeInput) return;
        highlight.innerHTML = highlightPython(codeInput.value || "") + "\n";
        highlight.parentElement.scrollTop = codeInput.scrollTop;
        highlight.parentElement.scrollLeft = codeInput.scrollLeft;
      };

      const closeEditor = () => {
        modal.hidden = true;
        if (codeInput) codeInput.value = originalContent;
        renderHighlight();
      };

      const openEditor = async () => {
        modal.hidden = false;
        setEditorBusy(true);
        setEditorStatus("读取中");
        if (pathLabel) pathLabel.textContent = "读取中";
        try {
          const resp = await fetch(openButton.dataset.readUrl);
          const payload = await resp.json();
          if (!resp.ok) throw new Error(payload.error || "读取失败");
          originalContent = payload.content || "";
          codeInput.value = originalContent;
          if (pathLabel) pathLabel.textContent = payload.path || "funds.py";
          setEditorStatus("已加载");
          renderHighlight();
          codeInput.focus();
        } catch (error) {
          setEditorStatus(error.message);
        } finally {
          setEditorBusy(false);
        }
      };

      const saveEditor = async () => {
        if (!codeInput || !saveButton || saveButton.disabled) return;
        setEditorBusy(true);
        setEditorStatus("保存中");
        try {
          const resp = await fetch(openButton.dataset.saveUrl, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content: codeInput.value }),
          });
          const payload = await resp.json();
          if (!resp.ok) throw new Error(payload.error || "保存失败");
          originalContent = codeInput.value;
          setEditorStatus(`已保存 ${payload.size || 0} bytes`);
        } catch (error) {
          setEditorStatus(error.message);
        } finally {
          setEditorBusy(false);
        }
      };

      if (codeInput) {
        codeInput.addEventListener("input", renderHighlight);
        codeInput.addEventListener("scroll", renderHighlight);
        codeInput.addEventListener("keydown", (event) => {
          if (event.key !== "Tab") return;
          event.preventDefault();
          codeInput.setRangeText("    ", codeInput.selectionStart, codeInput.selectionEnd, "end");
          renderHighlight();
        });
      }

      openButton.addEventListener("click", openEditor);
      if (saveButton) saveButton.addEventListener("click", saveEditor);
      cancelButtons.forEach((button) => button.addEventListener("click", closeEditor));
      modal.addEventListener("click", (event) => {
        if (event.target === modal) closeEditor();
      });
    }

    function highlightPython(source) {
      const tokenRe = /("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|#.*|\b(?:False|None|True|and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield)\b|\b\d+(?:\.\d+)?\b)/g;
      let output = "";
      let lastIndex = 0;
      for (const match of source.matchAll(tokenRe)) {
        const token = match[0];
        output += esc(source.slice(lastIndex, match.index));
        let cls = "py-number";
        if (token.startsWith("#")) {
          cls = "py-comment";
        } else if (token.startsWith("\"") || token.startsWith("'")) {
          cls = "py-string";
        } else if (/^[A-Za-z_]+$/.test(token)) {
          cls = "py-keyword";
        }
        output += `<span class="${cls}">${esc(token)}</span>`;
        lastIndex = match.index + token.length;
      }
      return output + esc(source.slice(lastIndex));
    }

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
  }

  window.FinancialAnalysisPages = window.FinancialAnalysisPages || {};
  window.FinancialAnalysisPages.fund = initFundPage;
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initFundPage, { once: true });
  } else {
    initFundPage();
  }
})();
