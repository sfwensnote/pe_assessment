import { useEffect, useState } from "react";
import { LiveMonitorSnapshot } from "../types";
import { readApiError } from "./helpers";

const POLL_MS = 2000;

type MonitorPopupProps = {
  apiBase: string;
};

function MonitorPopup({ apiBase }: MonitorPopupProps) {
  const [open, setOpen] = useState<boolean>(false);
  const [snapshot, setSnapshot] = useState<LiveMonitorSnapshot | null>(null);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    if (!open) {
      return;
    }

    void fetchSnapshot();
    const timer = window.setInterval(() => {
      void fetchSnapshot();
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [open]);

  const fetchSnapshot = async () => {
    try {
      const resp = await fetch(`${apiBase}/api/monitor/live`);
      if (!resp.ok) {
        throw new Error(await readApiError(resp, `HTTP ${resp.status}`));
      }
      const data = (await resp.json()) as LiveMonitorSnapshot;
      setSnapshot(data);
      setError("");
    } catch (err) {
      setError(`读取监控失败: ${String(err)}`);
    }
  };

  return (
    <>
      <button className="monitor-fab" onClick={() => setOpen(true)}>
        后台监控
      </button>

      {open ? (
        <div className="monitor-overlay" onClick={() => setOpen(false)}>
          <section className="monitor-modal" onClick={(e) => e.stopPropagation()}>
            <header className="monitor-header">
              <div>
                <h3>后台实时监控</h3>
                <p>{snapshot ? `更新时间 ${formatTs(snapshot.time)}` : "加载中"}</p>
              </div>
              <button className="secondary" onClick={() => setOpen(false)}>
                关闭
              </button>
            </header>

            {error ? <div className="status-box error">{error}</div> : null}

            {snapshot ? (
              <div className="monitor-grid">
                <div className="monitor-card">
                  <h4>服务状态</h4>
                  <p>设备：{snapshot.health.device}</p>
                  <p>姿态：{snapshot.health.yolo ? "就绪" : "未就绪"}</p>
                  <p>动作模型：{snapshot.health.action ? "就绪" : "未就绪"}</p>
                  <p>阶段模型：{snapshot.health.phase_model_count}</p>
                  <p>质量模型：{snapshot.health.quality ? "就绪" : "未就绪"}</p>
                </div>

                <div className="monitor-card">
                  <h4>运行任务</h4>
                  <p>在线会话：{snapshot.realtime.active_sessions}</p>
                  <p>后台任务总数：{snapshot.video_tasks.total}</p>
                  <p>进行中：{snapshot.video_tasks.running}</p>
                  <p>排队中：{snapshot.video_tasks.queued}</p>
                  <p>失败：{snapshot.video_tasks.failed}</p>
                </div>

                <div className="monitor-card">
                  <h4>采集流水线</h4>
                  <p>运行ID：{snapshot.ingest.run_id || "-"}</p>
                  <p>
                    状态：{snapshot.ingest.status || "-"} / 阶段：{snapshot.ingest.stage || "-"}
                  </p>
                  <p>消息：{snapshot.ingest.message || "-"}</p>
                  <p>扫描：{num(snapshot.ingest.summary.scanned_total)}</p>
                  <p>下载：{num(snapshot.ingest.summary.added_total)}</p>
                  <p>骨骼：{num(snapshot.ingest.summary.skeleton_added_total)}</p>
                  <p>标注：{num(snapshot.ingest.summary.annotation_added_total)}</p>
                </div>
              </div>
            ) : null}

            {snapshot?.ingest.latest_events?.length ? (
              <div className="monitor-events">
                <div className="panel-title">最新事件</div>
                <div className="task-list">
                  {snapshot.ingest.latest_events.slice(-6).map((item, index) => (
                    <div className="history-item" key={`${item.time || index}-${index}`}>
                      <div>
                        <strong>{item.level || "info"}</strong>
                        <p>{item.message || "-"}</p>
                      </div>
                      <div className="history-side">
                        <span>{item.action_id || "全局"}</span>
                        <span>{item.time ? formatTs(item.time) : "-"}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </section>
        </div>
      ) : null}
    </>
  );
}

function formatTs(value: number): string {
  return new Date(value * 1000).toLocaleTimeString("zh-CN");
}

function num(value: number | undefined): number {
  return Number.isFinite(value) ? Number(value) : 0;
}

export default MonitorPopup;
