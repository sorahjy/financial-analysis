(() => {
  window.FinancialAnalysisPages = window.FinancialAnalysisPages || {};
  window.FinancialAnalysisPages.bootId = window.FinancialAnalysisPages.bootId
    || `${Date.now()}-${Math.random().toString(36).slice(2)}`;

  const navigablePaths = new Set(["/", "/fund", "/fund/report", "/stock", "/radar"]);
  let navigating = false;

  function normalizePath(url) {
    const parsed = new URL(url, window.location.href);
    return parsed.pathname.replace(/\/+$/, "") || "/";
  }

  function isNavigable(url) {
    const parsed = new URL(url, window.location.href);
    return parsed.origin === window.location.origin && navigablePaths.has(normalizePath(parsed.href));
  }

  function initCurrentPage() {
    const pages = window.FinancialAnalysisPages || {};
    if (document.querySelector("#stock-native-dashboard")) {
      pages.stock && pages.stock();
    } else if (document.querySelector("#radar-dashboard")) {
      pages.radar && pages.radar();
    } else if (document.querySelector("[data-fund-status]") || document.querySelector("#fund-native-report")) {
      pages.fund && pages.fund();
    } else {
      pages.cleanup = null;
    }
  }

  async function navigate(url, pushState = true) {
    if (navigating || !isNavigable(url)) return false;
    navigating = true;
    try {
      const response = await fetch(url, {
        headers: {"X-Requested-With": "fetch"},
      });
      if (!response.ok) return false;

      const html = await response.text();
      const nextDocument = new DOMParser().parseFromString(html, "text/html");
      const nextMain = nextDocument.querySelector(".app-main");
      const currentMain = document.querySelector(".app-main");
      if (!nextMain || !currentMain) return false;

      const pages = window.FinancialAnalysisPages || {};
      if (typeof pages.cleanup === "function") {
        pages.cleanup();
        pages.cleanup = null;
      }

      document.title = nextDocument.title || document.title;
      currentMain.innerHTML = nextMain.innerHTML;
      if (pushState) window.history.pushState({}, "", url);
      window.scrollTo({top: 0, left: 0});
      initCurrentPage();
      return true;
    } catch (error) {
      console.warn("局部导航失败，回退到整页加载", error);
      return false;
    } finally {
      navigating = false;
    }
  }

  document.addEventListener("click", (event) => {
    if (event.defaultPrevented || event.button !== 0) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest("a[href]");
    if (!link || link.target || link.hasAttribute("download")) return;
    if (!isNavigable(link.href)) return;
    event.preventDefault();
    navigate(link.href).then((handled) => {
      if (!handled) window.location.href = link.href;
    });
  });

  window.addEventListener("popstate", () => {
    navigate(window.location.href, false).then((handled) => {
      if (!handled) window.location.reload();
    });
  });

  window.FinancialAnalysisPages.navigate = navigate;
  window.FinancialAnalysisPages.initCurrentPage = initCurrentPage;
})();
