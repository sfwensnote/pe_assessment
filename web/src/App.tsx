import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { ActionItem } from "./types";
import AdminPortal from "./components/AdminPortal";
import MonitorPopup from "./components/MonitorPopup";
import UploadTasks from "./components/UploadTasks";
import UserPortal from "./components/UserPortal";

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) || "http://127.0.0.1:8001";
const WS_BASE = API_BASE.replace(/^http/, "ws");

type ViewKey = "realtime" | "upload" | "admin";
type UploadFilter = "all" | "running" | "completed" | "failed";

const STORAGE_LAST_VIEW = "pe_last_view";
const STORAGE_UPLOAD_FILTER = "pe_upload_filter";
const STORAGE_SIDEBAR_COLLAPSED = "pe_sidebar_collapsed";

const SIDEBAR_EXPANDED_WIDTH = 248;
const SIDEBAR_COLLAPSED_WIDTH = 78;

type HealthState = {
  device: string;
  yolo: boolean;
  action: boolean;
  phaseModelCount: number;
  quality: boolean;
} | null;

function App() {
  const initialState = resolveInitialState();
  const [view, setView] = useState<ViewKey>(initialState.view);
  const [uploadFilter, setUploadFilter] = useState<UploadFilter>(initialState.filter);
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(initialState.sidebarCollapsed);
  const [actions, setActions] = useState<ActionItem[]>([]);
  const [health, setHealth] = useState<HealthState>(null);
  const [healthMessage, setHealthMessage] = useState<string>("");

  useEffect(() => {
    void fetchActions();
    void fetchHealth();

    const healthTimer = window.setInterval(() => {
      void fetchHealth();
    }, 30000);

    const current = resolveStateFromLocation(window.location.pathname, window.location.search);
    if (
      current.view !== view ||
      (current.view === "upload" && current.filter !== uploadFilter) ||
      window.location.pathname === "/"
    ) {
      window.history.replaceState(null, "", pathFromState(current.view, current.filter));
      setView(current.view);
      setUploadFilter(current.filter);
    }

    const onPopState = () => {
      const parsed = resolveStateFromLocation(window.location.pathname, window.location.search);
      setView(parsed.view);
      setUploadFilter(parsed.filter);
    };
    window.addEventListener("popstate", onPopState);

    return () => {
      window.removeEventListener("popstate", onPopState);
      window.clearInterval(healthTimer);
    };
  }, []);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_LAST_VIEW, view);
  }, [view]);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_UPLOAD_FILTER, uploadFilter);
  }, [uploadFilter]);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_SIDEBAR_COLLAPSED, sidebarCollapsed ? "1" : "0");
  }, [sidebarCollapsed]);

  const navItems = useMemo(
    () => [
      {
        key: "realtime" as const,
        label: "实时训练",
        short: "训",
        desc: "摄像头训练与即时反馈",
      },
      {
        key: "upload" as const,
        label: "视频评估",
        short: "评",
        desc: "后台任务评估与历史",
      },
      {
        key: "admin" as const,
        label: "管理员",
        short: "管",
        desc: "系统观测与任务监控",
      },
    ],
    []
  );

  const fetchActions = async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/actions`);
      const data = await resp.json();
      setActions(data.actions || []);
    } catch {
      setActions([]);
    }
  };

  const fetchHealth = async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/health`);
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const data = await resp.json();
      setHealth({
        device: data.device,
        yolo: Boolean(data.models?.yolo),
        action: Boolean(data.models?.action),
        phaseModelCount: Number(data.models?.phase_model_count || 0),
        quality: Boolean(data.models?.quality),
      });
      setHealthMessage("");
    } catch {
      setHealth(null);
      setHealthMessage("无法连接后端，请确认服务已启动");
    }
  };

  const navigateToView = (next: ViewKey) => {
    if (next === view) {
      return;
    }
    const nextPath = pathFromState(next, uploadFilter);
    window.history.pushState(null, "", nextPath);
    setView(next);
  };

  const navigateUploadFilter = (next: UploadFilter) => {
    setUploadFilter(next);
    if (view === "upload") {
      window.history.replaceState(null, "", pathFromState("upload", next));
    }
  };

  const appGridStyle = {
    "--sidebar-width": sidebarCollapsed
      ? `${SIDEBAR_COLLAPSED_WIDTH}px`
      : `${SIDEBAR_EXPANDED_WIDTH}px`,
  } as CSSProperties;

  return (
    <div className={`app-shell app-grid ${sidebarCollapsed ? "sidebar-collapsed" : ""}`} style={appGridStyle}>
      <aside className="sidebar">
        <div className="sidebar-brand-row">
          <div className="sidebar-brand">
            <h1>体育教学评估系统</h1>
            <p>课程训练 · 动作质控 · 教学管理</p>
          </div>
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarCollapsed((prev) => !prev)}
            title={sidebarCollapsed ? "展开侧边栏" : "折叠侧边栏"}
          >
            {sidebarCollapsed ? "»" : "«"}
          </button>
        </div>

        <nav className="sidebar-nav">
          {navItems.map((item) => (
            <button
              key={item.key}
              className={`nav-item ${item.key === view ? "active" : ""}`}
              onClick={() => navigateToView(item.key)}
            >
              <span className="nav-short">{item.short}</span>
              <span className="nav-title">{item.label}</span>
              <span className="nav-desc">{item.desc}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-health">
          <div className="health-title">运行状态</div>
          <div className="health-chip-row">
            <span className="chip">设备 {health?.device || "未知"}</span>
            <span className={`chip ${health?.yolo ? "ok" : "warn"}`}>
              姿态 {health?.yolo ? "就绪" : "未就绪"}
            </span>
            <span className={`chip ${health?.action ? "ok" : "warn"}`}>
              动作 {health?.action ? "已加载" : "未加载"}
            </span>
            <span className={`chip ${health?.phaseModelCount ? "ok" : "warn"}`}>
              阶段 {health?.phaseModelCount || 0}
            </span>
            <span className={`chip ${health?.quality ? "ok" : "warn"}`}>
              质量 {health?.quality ? "已加载" : "未加载"}
            </span>
          </div>
        </div>
      </aside>

      <section className="main-area">
        {healthMessage ? (
          <div className="status-box error" role="status">
            {healthMessage}
          </div>
        ) : null}

        {view === "realtime" ? <UserPortal actions={actions} apiBase={API_BASE} wsBase={WS_BASE} /> : null}
        {view === "upload" ? (
          <UploadTasks
            actions={actions}
            apiBase={API_BASE}
            filter={uploadFilter}
            onFilterChange={navigateUploadFilter}
          />
        ) : null}
        {view === "admin" ? <AdminPortal apiBase={API_BASE} /> : null}
        <MonitorPopup apiBase={API_BASE} />
      </section>
    </div>
  );
}

function resolveViewFromPath(pathname: string): ViewKey {
  if (pathname === "/upload") {
    return "upload";
  }
  if (pathname === "/admin") {
    return "admin";
  }
  return "realtime";
}

function pathFromState(view: ViewKey, uploadFilter: UploadFilter): string {
  if (view === "upload") {
    return uploadFilter === "all" ? "/upload" : `/upload?status=${uploadFilter}`;
  }
  if (view === "admin") {
    return "/admin";
  }
  return "/realtime";
}

function resolveStateFromLocation(pathname: string, search: string): {
  view: ViewKey;
  filter: UploadFilter;
} {
  const view = resolveViewFromPath(pathname);
  const params = new URLSearchParams(search);
  const raw = params.get("status") || "all";
  const filter = isUploadFilter(raw) ? raw : "all";
  return { view, filter };
}

function resolveInitialState(): {
  view: ViewKey;
  filter: UploadFilter;
  sidebarCollapsed: boolean;
} {
  const savedView = window.localStorage.getItem(STORAGE_LAST_VIEW) as ViewKey | null;
  const savedFilterRaw = window.localStorage.getItem(STORAGE_UPLOAD_FILTER) || "all";
  const savedFilter = isUploadFilter(savedFilterRaw) ? savedFilterRaw : "all";
  const savedCollapsed = window.localStorage.getItem(STORAGE_SIDEBAR_COLLAPSED) === "1";

  if (window.location.pathname === "/") {
    return {
      view: savedView === "upload" || savedView === "admin" ? savedView : "realtime",
      filter: savedFilter,
      sidebarCollapsed: savedCollapsed,
    };
  }

  const parsed = resolveStateFromLocation(window.location.pathname, window.location.search);
  return {
    view: parsed.view,
    filter: parsed.filter,
    sidebarCollapsed: savedCollapsed,
  };
}

function isUploadFilter(value: string): value is UploadFilter {
  return value === "all" || value === "running" || value === "completed" || value === "failed";
}

export default App;
