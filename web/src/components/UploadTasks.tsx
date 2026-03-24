import { useEffect, useMemo, useRef, useState } from "react";
import { ActionItem, VideoAssessment, VideoTask } from "../types";
import Metric from "./Metric";
import { formatTime, readApiError, statusLabel } from "./helpers";

const TASK_POLL_INTERVAL_MS = 2000;

type UploadTasksProps = {
  actions: ActionItem[];
  apiBase: string;
  filter: "all" | "running" | "completed" | "failed";
  onFilterChange: (next: "all" | "running" | "completed" | "failed") => void;
};

const AUTO_ACTION_KEY = "__auto__";

function UploadTasks({ actions, apiBase, filter, onFilterChange }: UploadTasksProps) {
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const activeTaskIdRef = useRef<string>("");

  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadAction, setUploadAction] = useState<string>("");
  const [taskCreating, setTaskCreating] = useState<boolean>(false);
  const [tasks, setTasks] = useState<VideoTask[]>([]);
  const [taskFilter, setTaskFilter] = useState<
    "all" | "running" | "completed" | "failed"
  >(filter);
  const [actionFilter, setActionFilter] = useState<string>("all");
  const [activeTaskId, setActiveTaskId] = useState<string>("");
  const [uploadError, setUploadError] = useState<string>("");
  const [deletingTaskId, setDeletingTaskId] = useState<string>("");
  const [clearingHistory, setClearingHistory] = useState<boolean>(false);
  const [searchText, setSearchText] = useState<string>("");
  const [sortMode, setSortMode] = useState<"latest" | "oldest" | "progress">("latest");

  const activeTask = useMemo(
    () => tasks.find((item) => item.task_id === activeTaskId) || null,
    [tasks, activeTaskId]
  );

  const filteredTasks = useMemo(() => {
    const statusMatched =
      taskFilter === "all"
        ? tasks
        : taskFilter === "running"
        ? tasks.filter((task) => task.status === "queued" || task.status === "running")
        : tasks.filter((task) => task.status === taskFilter);

    const actionMatched =
      actionFilter === "all"
        ? statusMatched
        : statusMatched.filter((task) => resolveTaskActionKey(task) === actionFilter);

    const keyword = searchText.trim().toLowerCase();
    const searched = keyword
      ? actionMatched.filter((task) => task.filename.toLowerCase().includes(keyword))
      : actionMatched;

    const sorted = [...searched];
    if (sortMode === "oldest") {
      sorted.sort((a, b) => a.created_at - b.created_at);
    } else if (sortMode === "progress") {
      sorted.sort((a, b) => b.progress - a.progress || b.updated_at - a.updated_at);
    } else {
      sorted.sort((a, b) => b.created_at - a.created_at);
    }

    return sorted;
  }, [tasks, taskFilter, actionFilter, searchText, sortMode]);

  const taskStats = useMemo(() => {
    return {
      total: tasks.length,
      running: tasks.filter((task) => task.status === "queued" || task.status === "running").length,
      completed: tasks.filter((task) => task.status === "completed").length,
      failed: tasks.filter((task) => task.status === "failed").length,
    };
  }, [tasks]);

  const historyTasks = useMemo(
    () =>
      filteredTasks
        .filter((item) => item.status === "completed" || item.status === "failed")
        .slice(0, 8),
    [filteredTasks]
  );

  const actionFilterOptions = useMemo(() => {
    const options = [{ key: "all", label: "全部动作" }, { key: AUTO_ACTION_KEY, label: "自动识别" }];
    actions.forEach((item) => {
      options.push({ key: item.id, label: item.name });
    });
    return options;
  }, [actions]);

  useEffect(() => {
    activeTaskIdRef.current = activeTaskId;
  }, [activeTaskId]);

  useEffect(() => {
    setTaskFilter(filter);
  }, [filter]);

  const applyFilter = (next: "all" | "running" | "completed" | "failed") => {
    setTaskFilter(next);
    onFilterChange(next);
  };

  useEffect(() => {
    void refreshTasks();
    const timer = window.setInterval(() => {
      void refreshTasks();
    }, TASK_POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, []);

  const refreshTasks = async () => {
    try {
      const resp = await fetch(`${apiBase}/api/inference/video/tasks`);
      if (!resp.ok) {
        return;
      }
      const data = await resp.json();
      const nextTasks = (data.tasks || []) as VideoTask[];
      setTasks(nextTasks);

      const currentActiveTaskId = activeTaskIdRef.current;
      if (!currentActiveTaskId && nextTasks.length > 0) {
        setActiveTaskId(nextTasks[0].task_id);
      } else if (
        currentActiveTaskId &&
        !nextTasks.some((item) => item.task_id === currentActiveTaskId)
      ) {
        setActiveTaskId(nextTasks.length > 0 ? nextTasks[0].task_id : "");
      }
    } catch {
      // silent while polling
    }
  };

  const createUploadTask = async () => {
    if (!uploadFile) {
      setUploadError("请先选择一个视频文件。");
      return;
    }

    setUploadError("");
    setTaskCreating(true);

    const formData = new FormData();
    formData.append("file", uploadFile);
    if (uploadAction) {
      formData.append("action_type", uploadAction);
    }

    try {
      const resp = await fetch(`${apiBase}/api/inference/video/tasks`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) {
        throw new Error(await readApiError(resp, "创建任务失败"));
      }

      const data = await resp.json();
      const createdTask = data.task as VideoTask;
      setTasks((prev) => [createdTask, ...prev]);
      setActiveTaskId(createdTask.task_id);
      setUploadFile(null);
      if (uploadInputRef.current) {
        uploadInputRef.current.value = "";
      }
      await refreshTasks();
    } catch (error) {
      setUploadError(`上传任务失败: ${String(error)}`);
    } finally {
      setTaskCreating(false);
    }
  };

  const deleteHistoryTask = async (taskId: string) => {
    const target = tasks.find((task) => task.task_id === taskId);
    if (!target) {
      return;
    }

    const confirmed = window.confirm(`确认删除历史任务「${target.filename}」吗？`);
    if (!confirmed) {
      return;
    }

    setDeletingTaskId(taskId);
    setUploadError("");
    try {
      const resp = await fetch(`${apiBase}/api/inference/video/tasks/${taskId}`, {
        method: "DELETE",
      });
      if (!resp.ok) {
        throw new Error(await readApiError(resp, "删除失败"));
      }

      setTasks((prev) => prev.filter((task) => task.task_id !== taskId));
      if (activeTaskIdRef.current === taskId) {
        setActiveTaskId("");
      }
      await refreshTasks();
    } catch (error) {
      setUploadError(`删除历史失败: ${String(error)}`);
    } finally {
      setDeletingTaskId("");
    }
  };

  const clearHistory = async () => {
    const hasHistory = tasks.some(
      (task) => task.status === "completed" || task.status === "failed"
    );
    if (!hasHistory) {
      return;
    }

    const confirmed = window.confirm("确认清空全部评估历史吗？此操作不可恢复。");
    if (!confirmed) {
      return;
    }

    setClearingHistory(true);
    setUploadError("");
    try {
      const resp = await fetch(`${apiBase}/api/inference/video/history`, {
        method: "DELETE",
      });
      if (!resp.ok) {
        throw new Error(await readApiError(resp, "清空失败"));
      }

      setTasks((prev) =>
        prev.filter((task) => task.status === "queued" || task.status === "running")
      );
      if (
        activeTask &&
        (activeTask.status === "completed" || activeTask.status === "failed")
      ) {
        setActiveTaskId("");
      }
      await refreshTasks();
    } catch (error) {
      setUploadError(`清空历史失败: ${String(error)}`);
    } finally {
      setClearingHistory(false);
    }
  };

  return (
    <main className="upload-layout">
      <section className="panel upload-control-panel">
        <div className="panel-title">新建评估任务</div>
        <label>
          选择动作（可选）
          <select
            value={uploadAction}
            disabled={taskCreating}
            onChange={(e) => setUploadAction(e.target.value)}
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
          选择视频文件
          <input
            ref={uploadInputRef}
            type="file"
            accept="video/*"
            disabled={taskCreating}
            onChange={(e) => {
              const file = e.target.files?.[0] || null;
              setUploadFile(file);
            }}
          />
        </label>

        {uploadFile ? (
          <div className="status-box">
            已选择：{uploadFile.name}（{Math.max(1, Math.round(uploadFile.size / 1024 / 1024))} MB）
          </div>
        ) : null}

        <div className="button-row">
          <button disabled={taskCreating || !uploadFile} onClick={() => void createUploadTask()}>
            {taskCreating ? "创建中..." : "提交后台评估"}
          </button>
        </div>

        {uploadError ? <div className="status-box error">{uploadError}</div> : null}
      </section>

      <section className="panel upload-progress-panel">
        <div className="panel-title">任务进度</div>
        {activeTask ? (
          <>
            <div className="task-headline">
              <div>
                <strong>{activeTask.filename}</strong>
                <p>{statusLabel(activeTask.status)}</p>
              </div>
              <span className={`status-pill ${activeTask.status}`}>{statusLabel(activeTask.status)}</span>
            </div>

            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${activeTask.progress}%` }} />
            </div>
            <div className="progress-meta">
              <span>{activeTask.message}</span>
              <span>{Math.round(activeTask.progress)}%</span>
            </div>
          </>
        ) : (
          <div className="empty-text">请选择一个任务查看进度。</div>
        )}
      </section>

      <section className="panel upload-result-panel">
        <div className="panel-title">当前任务结果</div>
        {activeTask?.result?.ok ? (
          <TaskResultView result={activeTask.result} />
        ) : (
          <div className="empty-text">任务完成后会展示评估结果。</div>
        )}
      </section>

      <section className="panel task-list-panel">
        <div className="panel-title">评估任务列表</div>
        <div className="task-stats-strip">
          <div className="task-stat-pill">
            <span>总任务</span>
            <strong>{taskStats.total}</strong>
          </div>
          <div className="task-stat-pill running">
            <span>进行中</span>
            <strong>{taskStats.running}</strong>
          </div>
          <div className="task-stat-pill completed">
            <span>已完成</span>
            <strong>{taskStats.completed}</strong>
          </div>
          <div className="task-stat-pill failed">
            <span>失败</span>
            <strong>{taskStats.failed}</strong>
          </div>
        </div>
        <div className="task-toolbar">
          <div className="status-filter-row">
            <button className={taskFilter === "all" ? "active" : ""} onClick={() => applyFilter("all")}>
              全部
            </button>
            <button
              className={taskFilter === "running" ? "active" : ""}
              onClick={() => applyFilter("running")}
            >
              进行中
            </button>
            <button
              className={taskFilter === "completed" ? "active" : ""}
              onClick={() => applyFilter("completed")}
            >
              已完成
            </button>
            <button
              className={taskFilter === "failed" ? "active" : ""}
              onClick={() => applyFilter("failed")}
            >
              失败
            </button>
          </div>
          <div className="task-filter-grid">
            <label className="action-filter-control">
              按动作
              <select value={actionFilter} onChange={(e) => setActionFilter(e.target.value)}>
                {actionFilterOptions.map((item) => (
                  <option key={item.key} value={item.key}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="action-filter-control">
              排序
              <select
                value={sortMode}
                onChange={(e) => setSortMode(e.target.value as "latest" | "oldest" | "progress")}
              >
                <option value="latest">最新优先</option>
                <option value="oldest">最早优先</option>
                <option value="progress">进度优先</option>
              </select>
            </label>
            <label className="action-filter-control">
              搜索
              <input
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                placeholder="文件名关键词"
              />
            </label>
          </div>
        </div>

        {filteredTasks.length === 0 ? (
          <div className="empty-text">暂无任务。</div>
        ) : (
          <div className="task-list">
            {filteredTasks.map((task) => (
              <button
                key={task.task_id}
                className={`task-item ${activeTaskId === task.task_id ? "active" : ""}`}
                onClick={() => setActiveTaskId(task.task_id)}
              >
                <div className="task-item-top">
                  <strong>{task.filename}</strong>
                  <span className={`status-pill ${task.status}`}>{statusLabel(task.status)}</span>
                </div>
                <div className="task-item-sub">
                  <span>{renderActionLabel(task, actions)}</span>
                  <span>{Math.round(task.progress)}%</span>
                  <span>{formatTime(task.created_at)}</span>
                </div>
                <div className="mini-progress">
                  <div style={{ width: `${task.progress}%` }} />
                </div>
              </button>
            ))}
          </div>
        )}
      </section>

      <section className="panel task-history-panel">
        <div className="history-header-row">
          <div className="panel-title">评估历史</div>
          <button
            className="ghost-danger history-clear-btn"
            disabled={clearingHistory}
            onClick={() => void clearHistory()}
          >
            {clearingHistory ? "清空中..." : "清空历史"}
          </button>
        </div>
        {historyTasks.length === 0 ? (
          <div className="empty-text">当前筛选下暂无历史任务。</div>
        ) : (
          <div className="history-grid">
            {historyTasks.map((task) => (
              <div key={task.task_id} className="history-item">
                <div>
                  <strong>{task.filename}</strong>
                  <p>
                    {renderActionLabel(task, actions)} · {formatTime(task.updated_at)}
                  </p>
                </div>
                <div className="history-side">
                  <span className={`status-pill ${task.status}`}>{statusLabel(task.status)}</span>
                  <span>{task.result?.overall_score ? `${task.result.overall_score.toFixed(1)} 分` : "-"}</span>
                  <button
                    className="ghost-danger history-delete-btn"
                    disabled={deletingTaskId === task.task_id || clearingHistory}
                    onClick={() => void deleteHistoryTask(task.task_id)}
                  >
                    {deletingTaskId === task.task_id ? "删除中" : "删除"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

function resolveTaskActionKey(task: VideoTask): string {
  const actionFromResult = task.result?.action_type?.trim();
  const actionFromTask = task.action_type?.trim();
  return actionFromResult || actionFromTask || AUTO_ACTION_KEY;
}

function renderActionLabel(task: VideoTask, actions: ActionItem[]): string {
  const key = resolveTaskActionKey(task);
  if (key === AUTO_ACTION_KEY) {
    return "自动识别";
  }
  const found = actions.find((item) => item.id === key);
  if (!found) {
    return "未知动作";
  }
  return found.name;
}

function TaskResultView({ result }: { result: VideoAssessment }) {
  return (
    <div className="task-result-layout">
      <div className="result-summary-grid">
        <Metric label="评分" value={result.overall_score?.toFixed(1) || "-"} highlight />
        <Metric label="标准" value={result.is_standard ? "达标" : "待改进"} highlight />
        <Metric label="动作" value={result.action_name || "-"} />
        <Metric label="动作计数" value={String(result.estimated_reps ?? 0)} />
        <Metric label="帧数" value={String(result.total_frames || 0)} />
        <Metric label="识别来源" value={result.action_source || "-"} />
      </div>

      <div className="result-feedback-grid">
        <div className="feedback-block">
          <h3>错误检测</h3>
          <ul>
            {(result.errors && result.errors.length > 0 ? result.errors : ["暂无明显错误"]).map((err) => (
              <li key={err}>{err}</li>
            ))}
          </ul>
        </div>

        <div className="feedback-block">
          <h3>纠正建议</h3>
          <ul>
            {(result.tips && result.tips.length > 0 ? result.tips : ["暂无建议"]).map((tip) => (
              <li key={tip}>{tip}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default UploadTasks;
