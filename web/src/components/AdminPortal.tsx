import { useEffect, useState } from "react";
import { AdminOverview, IngestOverview, SystemOverview } from "../types";
import Metric from "./Metric";
import { formatTime, readApiError, statusLabel } from "./helpers";

type AdminPortalProps = {
  apiBase: string;
};

const ADMIN_POLL_INTERVAL_MS = 3000;
const SYSTEM_OVERVIEW_POLL_INTERVAL_MS = 15000;

function AdminPortal({ apiBase }: AdminPortalProps) {
  const [adminOverview, setAdminOverview] = useState<AdminOverview | null>(null);
  const [systemOverview, setSystemOverview] = useState<SystemOverview | null>(null);
  const [adminError, setAdminError] = useState<string>("");
  const [systemError, setSystemError] = useState<string>("");
  const [ingestError, setIngestError] = useState<string>("");
  const [adminToken, setAdminToken] = useState<string>("");
  const [lastRefreshAt, setLastRefreshAt] = useState<number>(0);
  const [ingestOverview, setIngestOverview] = useState<IngestOverview | null>(null);

  useEffect(() => {
    const cachedToken = window.localStorage.getItem("pe_admin_token") || "";
    if (cachedToken) {
      setAdminToken(cachedToken);
    }
  }, []);

  useEffect(() => {
    void fetchAdminOverview();
    const timer = window.setInterval(() => {
      void fetchAdminOverview();
    }, ADMIN_POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [adminToken]);

  useEffect(() => {
    void fetchSystemOverview();
    const timer = window.setInterval(() => {
      void fetchSystemOverview();
    }, SYSTEM_OVERVIEW_POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    void fetchIngestOverview();
    const timer = window.setInterval(() => {
      void fetchIngestOverview();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [adminToken]);

  const fetchAdminOverview = async () => {
    try {
      const resp = await fetch(`${apiBase}/api/admin/overview`, {
        headers: adminHeaders(adminToken),
      });
      if (!resp.ok) {
        throw new Error(await readApiError(resp, `HTTP ${resp.status}`));
      }
      const data = (await resp.json()) as AdminOverview;
      setAdminOverview(data);
      setAdminError("");
      setLastRefreshAt(Date.now());
    } catch (error) {
      setAdminError(`加载后台观测失败: ${String(error)}`);
    }
  };

  const fetchSystemOverview = async () => {
    try {
      const resp = await fetch(`${apiBase}/api/system/overview`);
      if (!resp.ok) {
        throw new Error(await readApiError(resp, `HTTP ${resp.status}`));
      }
      const data = (await resp.json()) as SystemOverview;
      setSystemOverview(data);
      setSystemError("");
      setLastRefreshAt(Date.now());
    } catch (error) {
      setSystemError(`加载系统能力概览失败: ${String(error)}`);
    }
  };

  const fetchIngestOverview = async () => {
    try {
      const resp = await fetch(`${apiBase}/api/admin/ingest/overview`, {
        headers: adminHeaders(adminToken),
      });
      if (!resp.ok) {
        throw new Error(await readApiError(resp, `HTTP ${resp.status}`));
      }
      const data = (await resp.json()) as IngestOverview;
      setIngestOverview(data);
      setIngestError("");
      setLastRefreshAt(Date.now());
    } catch (error) {
      setIngestError(`加载采集监控失败: ${String(error)}`);
    }
  };

  return (
    <main className="admin-layout">
      <section className="panel admin-top-panel">
        <div className="panel-title">后台总览</div>
        <div className="button-row">
          <button
            className="topbar-refresh"
            onClick={() => {
              void fetchAdminOverview();
              void fetchSystemOverview();
              void fetchIngestOverview();
            }}
          >
            立即刷新
          </button>
          <div className="empty-text">
            {lastRefreshAt
              ? `最近更新：${new Date(lastRefreshAt).toLocaleTimeString("zh-CN")}`
              : "等待首次加载"}
          </div>
        </div>
        <div className="admin-token-row">
          <label>
            管理员令牌
            <input
              type="password"
              value={adminToken}
              placeholder="若后端配置 ADMIN_TOKEN，请输入"
              onChange={(e) => {
                const value = e.target.value;
                setAdminToken(value);
                if (value) {
                  window.localStorage.setItem("pe_admin_token", value);
                } else {
                  window.localStorage.removeItem("pe_admin_token");
                }
              }}
            />
          </label>
        </div>
        {adminError ? <div className="status-box error">{adminError}</div> : null}
        <div className="admin-cards">
          <Metric
            label="在线会话"
            value={String(adminOverview?.realtime.active_session_count || 0)}
            highlight
          />
          <Metric label="任务总量" value={String(adminOverview?.video_tasks.stats.total || 0)} />
          <Metric
            label="评估中任务"
            value={String(adminOverview?.video_tasks.stats.running || 0)}
          />
          <Metric
            label="平均时长"
            value={`${adminOverview?.video_tasks.stats.avg_duration_sec || 0}s`}
          />
        </div>
      </section>

      <section className="panel admin-system-panel">
        <div className="panel-title">系统能力与模型就绪</div>
        {systemError ? <div className="status-box error">{systemError}</div> : null}
        {systemOverview ? (
          <>
            <div
              className={`status-box ${
                systemOverview.readiness.level === "full" ? "" : "error"
              }`}
            >
              {systemOverview.readiness.level === "full" ? "完整就绪" : "部分就绪"}
              {`：${systemOverview.readiness.summary}`}
            </div>

            <div className="admin-cards">
              <Metric label="运行设备" value={systemOverview.device} />
              <Metric
                label="阶段模型"
                value={String(systemOverview.models.phase_model_count_loaded)}
              />
              <Metric
                label="缺失文件"
                value={String(systemOverview.checkpoints.missing_files.length)}
              />
              <Metric
                label="能力开通"
                value={`${Object.values(systemOverview.features).filter(Boolean).length}/${
                  Object.keys(systemOverview.features).length
                }`}
              />
            </div>

            <div className="admin-feature-row">
              {Object.entries(systemOverview.features).map(([key, enabled]) => (
                <span key={key} className={`chip ${enabled ? "ok" : "warn"}`}>
                  {featureLabel(key)} {enabled ? "开" : "关"}
                </span>
              ))}
            </div>

            {systemOverview.checkpoints.missing_files.length > 0 ? (
              <div className="status-box error">
                缺失模型文件：{systemOverview.checkpoints.missing_files.join("、")}
              </div>
            ) : (
              <div className="status-box">模型文件检查通过（未发现缺失项）。</div>
            )}

            <ol className="setup-steps">
              {systemOverview.recommended_setup_steps.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ol>
          </>
        ) : null}
      </section>

      <section className="panel admin-session-panel">
        <div className="panel-title">实时会话观测</div>
        {(adminOverview?.realtime.active_sessions || []).length === 0 ? (
          <div className="empty-text">当前无在线实时会话。</div>
        ) : (
          <div className="admin-table-wrap">
            <table className="admin-table">
              <thead>
                <tr>
                  <th>会话ID</th>
                  <th>动作</th>
                  <th>来源</th>
                  <th>评分</th>
                  <th>计次</th>
                  <th>时长</th>
                </tr>
              </thead>
              <tbody>
                {adminOverview?.realtime.active_sessions.map((session) => (
                  <tr key={session.session_id}>
                    <td>{session.session_id.slice(0, 8)}</td>
                    <td>{session.current_action || session.action_hint || "-"}</td>
                    <td>{session.action_source || "-"}</td>
                    <td>{session.current_score ?? "-"}</td>
                    <td>
                      {session.rep_count}/{session.target_reps}
                    </td>
                    <td>{Math.round(session.elapsed_seconds)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="panel admin-task-panel">
        <div className="panel-title">后台评估任务监控</div>
        <div className="task-list">
          {(adminOverview?.video_tasks.latest || []).map((task) => (
            <div key={task.task_id} className="task-item admin-item">
              <div className="task-item-top">
                <strong>{task.filename}</strong>
                <span className={`status-pill ${task.status}`}>{statusLabel(task.status)}</span>
              </div>
              <div className="task-item-sub">
                <span>{task.action_type || "自动识别"}</span>
                <span>{formatTime(task.created_at)}</span>
                <span>{Math.round(task.progress)}%</span>
              </div>
              <div className="mini-progress">
                <div style={{ width: `${task.progress}%` }} />
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel admin-report-panel">
        <div className="panel-title">最近会话报告</div>
        <div className="history-grid">
          {(adminOverview?.realtime.latest_reports || []).map((item) => (
            <div key={item._report_file || item.session_id} className="history-item">
              <div>
                <strong>{item.action_type || "未知动作"}</strong>
                <p>{item._report_file || item.session_id || "-"}</p>
              </div>
              <div className="history-side">
                <span>{item.avg_score?.toFixed(1) || "-"} 分</span>
                <span>{item.total_reps || 0} 次</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel admin-task-panel">
        <div className="panel-title">采集与数据转化监控</div>
        {ingestError ? <div className="status-box error">{ingestError}</div> : null}
        {ingestOverview ? (
          <>
            <div className="status-box">
              运行状态：{ingestOverview.run_state.status || "-"} · 阶段：
              {ingestOverview.run_state.stage || "-"}
              {ingestOverview.run_state.message ? ` · ${ingestOverview.run_state.message}` : ""}
            </div>

            <div className="admin-cards">
              <Metric label="清单总数" value={String(ingestOverview.files.manifest_count)} />
              <Metric label="已打标签" value={String(ingestOverview.files.tagged_count)} />
              <Metric label="已拒绝" value={String(ingestOverview.files.rejected_count)} />
              <Metric label="清理日志" value={String(ingestOverview.files.cleanup_log_count)} />
            </div>

            <div className="admin-table-wrap">
              <table className="admin-table">
                <thead>
                  <tr>
                    <th>动作</th>
                    <th>下载</th>
                    <th>骨骼</th>
                    <th>标注</th>
                    <th>骨骼转化率</th>
                    <th>标注转化率</th>
                  </tr>
                </thead>
                <tbody>
                  {ingestOverview.conversion.map((item) => (
                    <tr key={item.action_id}>
                      <td>{item.action_name}</td>
                      <td>{item.downloaded}</td>
                      <td>{item.skeleton}</td>
                      <td>{item.annotation}</td>
                      <td>{item.skeleton_rate.toFixed(1)}%</td>
                      <td>{item.annotation_rate.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {ingestOverview.rejected_reason_stats.length > 0 ? (
              <div className="admin-feature-row">
                {ingestOverview.rejected_reason_stats.map((item) => (
                  <span key={item.reason} className="chip warn">
                    {item.reason} {item.count}
                  </span>
                ))}
              </div>
            ) : (
              <div className="empty-text">暂无被拒绝视频记录。</div>
            )}
          </>
        ) : null}
      </section>
    </main>
  );
}

function featureLabel(key: string): string {
  const labels: Record<string, string> = {
    realtime_camera_assessment: "实时摄像头评估",
    video_upload_assessment: "视频上传评估",
    background_video_tasks: "后台任务",
    admin_observability: "管理员观测",
    estimated_repetition_count: "计次估算",
    auto_action_recognition_model: "动作自动识别模型",
    phase_segmentation_model: "阶段分割模型",
    quality_scoring_model: "质量评分模型",
  };
  return labels[key] || key;
}

function adminHeaders(token: string): HeadersInit {
  if (!token) {
    return {};
  }
  return {
    "X-Admin-Token": token,
  };
}

export default AdminPortal;
