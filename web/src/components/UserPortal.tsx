import { useEffect, useMemo, useRef, useState } from "react";
import { ActionItem, RealtimeResult, SessionReport } from "../types";
import Metric from "./Metric";

const SEND_INTERVAL_MS = 100;

type UserPortalProps = {
  actions: ActionItem[];
  apiBase: string;
  wsBase: string;
};

function UserPortal({ actions, apiBase, wsBase }: UserPortalProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<number | null>(null);
  const sessionIdRef = useRef<string>("");
  const runningRef = useRef<boolean>(false);

  const [selectedAction, setSelectedAction] = useState<string>("");
  const [targetReps, setTargetReps] = useState<number>(20);
  const [mirrorCamera, setMirrorCamera] = useState<boolean>(true);
  const [sessionId, setSessionId] = useState<string>("");
  const [running, setRunning] = useState<boolean>(false);
  const [starting, setStarting] = useState<boolean>(false);
  const [statusText, setStatusText] = useState<string>("准备开始实时训练");
  const [connectionText, setConnectionText] = useState<string>("未连接");
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [frameSendCount, setFrameSendCount] = useState<number>(0);
  const [frameReceiveCount, setFrameReceiveCount] = useState<number>(0);
  const [lastResultAt, setLastResultAt] = useState<number>(0);
  const [result, setResult] = useState<RealtimeResult | null>(null);
  const [report, setReport] = useState<SessionReport | null>(null);

  const wsUrl = useMemo(() => wsBase, [wsBase]);
  const actionNameMap = useMemo(() => {
    const map = new Map<string, string>();
    actions.forEach((item) => {
      map.set(item.id, item.name);
    });
    return map;
  }, [actions]);

  useEffect(() => {
    return () => {
      void stopRealtime();
    };
  }, []);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    runningRef.current = running;
  }, [running]);

  useEffect(() => {
    if (!running) {
      return;
    }

    const timer = window.setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);

    return () => window.clearInterval(timer);
  }, [running]);

  useEffect(() => {
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      if (!runningRef.current) {
        return;
      }
      event.preventDefault();
      event.returnValue = "";
    };

    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, []);

  const startRealtime = async () => {
    if (running || starting) {
      return;
    }

    setStarting(true);
    setReport(null);
    setResult(null);
    setElapsedSeconds(0);
    setFrameSendCount(0);
    setFrameReceiveCount(0);
    setLastResultAt(0);
    setConnectionText("连接中...");

    try {
      const startResp = await fetch(`${apiBase}/api/realtime/session/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action_type: selectedAction || null,
          target_reps: targetReps,
        }),
      });

      if (!startResp.ok) {
        const text = await startResp.text();
        throw new Error(text || "创建会话失败");
      }

      const startData = await startResp.json();
      const createdSessionId: string = startData.session_id;
      setSessionId(createdSessionId);

      await startCamera();
      openSocket(createdSessionId);
      setRunning(true);
      setStatusText("会话已启动，正在实时分析...");
    } catch (error) {
      setStatusText(`启动失败: ${String(error)}`);
      setConnectionText("连接失败");
      await cleanupMedia();
    } finally {
      setStarting(false);
    }
  };

  const stopRealtime = async () => {
    const activeSessionId = sessionIdRef.current;

    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    await cleanupMedia();

    if (activeSessionId) {
      try {
        const resp = await fetch(`${apiBase}/api/realtime/session/${activeSessionId}/stop`, {
          method: "POST",
        });
        if (resp.ok) {
          const data = await resp.json();
          setReport(data.report as SessionReport);
        }
      } catch (error) {
        setStatusText(`停止会话失败: ${String(error)}`);
      }
    }

    setRunning(false);
    setStarting(false);
    setSessionId("");
    setElapsedSeconds(0);
    setFrameSendCount(0);
    setFrameReceiveCount(0);
    setLastResultAt(0);
    sessionIdRef.current = "";
    setConnectionText("未连接");
    setStatusText("会话已停止");
  };

  const startCamera = async () => {
    if (streamRef.current) {
      return;
    }

    const media = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: "user" },
      audio: false,
    });

    streamRef.current = media;
    if (videoRef.current) {
      videoRef.current.srcObject = media;
      await videoRef.current.play();
    }
  };

  const cleanupMedia = async () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }
  };

  const openSocket = (createdSessionId: string) => {
    const ws = new WebSocket(`${wsUrl}/ws/realtime/${createdSessionId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionText("已连接");
      timerRef.current = window.setInterval(sendFrame, SEND_INTERVAL_MS);
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as RealtimeResult;
        if (payload.status === "error") {
          setStatusText(payload.message || "推理异常，请稍后重试");
          return;
        }

        setResult(payload);
        setFrameReceiveCount((prev) => prev + 1);
        setLastResultAt(Date.now());

        if (payload.message) {
          setStatusText(payload.message);
        } else if (payload.status === "ok") {
          setStatusText("实时评估中");
        }
      } catch {
        setStatusText("实时消息解析失败，请重试。");
      }
    };

    ws.onerror = () => {
      setStatusText("WebSocket 出现异常，请重试。");
      setConnectionText("连接异常");
    };

    ws.onclose = () => {
      setConnectionText("已断开");
      if (runningRef.current) {
        setStatusText("连接已断开，会话已结束。");
      }
    };
  };

  const sendFrame = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) {
      return;
    }

    if (videoRef.current.readyState < 2 || videoRef.current.videoWidth === 0) {
      return;
    }

    const ws = wsRef.current;
    if (ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const canvas = canvasRef.current;
    const width = 960;
    const height = 540;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx || !videoRef.current) {
      return;
    }

    ctx.drawImage(videoRef.current, 0, 0, width, height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
    const imageBase64 = dataUrl.split(",", 2)[1];

    ws.send(
      JSON.stringify({
        type: "frame",
        timestamp: Date.now(),
        image_base64: imageBase64,
        width,
        height,
      })
    );
    setFrameSendCount((prev) => prev + 1);
  };

  return (
    <main className="realtime-layout">
      <section className="panel panel-video">
        <div className="panel-title">实时训练画面</div>
        <div className="realtime-status-line">
          <span className={`chip ${running ? "ok" : "warn"}`}>{running ? "训练进行中" : "未开始"}</span>
          <span className="chip">会话 {sessionId ? sessionId.slice(0, 8) : "-"}</span>
          <span className="chip">连接 {connectionText}</span>
          <span className="chip">时长 {elapsedSeconds}s</span>
        </div>
        <div className={`camera-frame ${mirrorCamera ? "mirror" : ""}`}>
          <video ref={videoRef} className="camera-view" playsInline muted />
        </div>
        <div className="camera-hint">建议全身入镜，保持镜头稳定和光线充足。</div>
        <canvas ref={canvasRef} className="hidden-canvas" />
      </section>

      <section className="panel panel-control">
        <div className="panel-title">训练控制</div>
        <div className="status-box">连接状态：{connectionText}</div>

        <div className="control-group">
          <label>
            训练动作
            <select
              value={selectedAction}
              disabled={running}
              onChange={(e) => setSelectedAction(e.target.value)}
            >
              <option value="">自动识别</option>
              {actions.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>

          <label>
            目标次数
            <input
              type="number"
              min={1}
              max={500}
              value={targetReps}
              disabled={running}
              onChange={(e) => {
                const next = Number(e.target.value || 20);
                const safeValue = Number.isFinite(next) ? Math.min(500, Math.max(1, next)) : 20;
                setTargetReps(safeValue);
              }}
            />
          </label>
        </div>

        <div className="preset-row quick-targets">
          {[10, 20, 30, 50].map((count) => (
            <button
              key={count}
              className={targetReps === count ? "active" : ""}
              disabled={running}
              onClick={() => setTargetReps(count)}
            >
              {count} 次
            </button>
          ))}
        </div>

        <label className="toggle-row">
          <input
            type="checkbox"
            checked={mirrorCamera}
            onChange={(e) => setMirrorCamera(e.target.checked)}
          />
          镜像显示摄像头
        </label>

        <div className="button-row">
          <button className="primary-btn" disabled={running || starting} onClick={startRealtime}>
            {starting ? "启动中..." : "开始训练"}
          </button>
          <button className="secondary danger-btn" disabled={!running} onClick={() => void stopRealtime()}>
            停止训练
          </button>
        </div>

        <div className="status-box">状态：{statusText}</div>
        <div className="realtime-guide">
          <strong>使用建议</strong>
          <p>先选择动作和目标次数，点击开始后连续完成标准动作，不要频繁离开镜头范围。</p>
        </div>
      </section>

      <section className="panel panel-live-score">
        <div className="panel-title">实时结果</div>
        <div className="status-box">
          {frameReceiveCount > 0
            ? `已接收 ${frameReceiveCount} 帧结果（发送 ${frameSendCount} 帧）`
            : running
            ? `等待首帧结果...（已发送 ${frameSendCount} 帧）`
            : "尚未开始训练"}
          {lastResultAt ? ` · 最近更新 ${new Date(lastResultAt).toLocaleTimeString("zh-CN")}` : ""}
        </div>
        <div className="metric-grid">
          <Metric label="评分" value={result?.overall_score?.toFixed(1) || "-"} highlight />
          <Metric
            label="达成率"
            value={typeof result?.completion_rate === "number" ? `${Math.round(result.completion_rate * 100)}%` : "-"}
            highlight
          />
          <Metric label="动作" value={toActionLabel(result?.action_type, actionNameMap)} />
          <Metric label="阶段" value={result?.phase_name || "-"} />
          <Metric label="识别来源" value={result?.action_source || "-"} />
          <Metric label="计次" value={String(result?.rep_count ?? 0)} />
        </div>

        <div className="live-feedback-grid">
          <div className="feedback-block">
            <h3>错误检测</h3>
            <ul>
              {(result?.errors && result.errors.length > 0
                ? result.errors
                : [running ? "等待模型输出中" : "暂无明显错误"]).map((err) => (
                <li key={err}>{err}</li>
              ))}
            </ul>
          </div>

          <div className="feedback-block">
            <h3>纠正建议</h3>
            <ul>
              {(result?.tips && result.tips.length > 0
                ? result.tips
                : [running ? "请保持在镜头中央并完成动作" : "开始动作后会显示建议"]).map((tip) => (
                <li key={tip}>{tip}</li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="panel panel-report">
        <div className="panel-title">训练会话报告</div>
        {report ? (
          <div className="report-grid">
            <Metric label="平均分" value={report.avg_score.toFixed(1)} highlight />
            <Metric label="最佳分" value={report.best_score.toFixed(1)} />
            <Metric label="动作" value={toActionLabel(report.action_type, actionNameMap)} />
            <Metric label="时长" value={`${report.duration_seconds}s`} />
            <Metric label="总次数" value={String(report.total_reps)} />
            <Metric
              label="完成率"
              value={`${Math.round(report.completion_rate * 100)}% (${report.total_reps}/${report.target_reps})`}
            />
          </div>
        ) : (
          <div className="empty-text">停止训练后将显示本次会话报告。</div>
        )}
      </section>
    </main>
  );
}

function toActionLabel(actionId: string | undefined, actionNameMap: Map<string, string>): string {
  if (!actionId) {
    return "-";
  }
  return actionNameMap.get(actionId) || "未知动作";
}

export default UserPortal;
