import { VideoTask } from "../types";

export function statusLabel(status: VideoTask["status"]): string {
  switch (status) {
    case "queued":
      return "排队中";
    case "running":
      return "评估中";
    case "completed":
      return "已完成";
    case "failed":
      return "失败";
    default:
      return status;
  }
}

export function formatTime(ts?: number | null): string {
  if (!ts) {
    return "-";
  }
  const date = new Date(ts * 1000);
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export async function readApiError(
  resp: Response,
  fallback: string
): Promise<string> {
  try {
    const payload = await resp.json();
    if (typeof payload?.detail === "string" && payload.detail.length > 0) {
      return payload.detail;
    }
    if (typeof payload?.message === "string" && payload.message.length > 0) {
      return payload.message;
    }
  } catch {
    // ignore json parse errors
  }

  try {
    const text = await resp.text();
    if (text) {
      return text;
    }
  } catch {
    // ignore text parse errors
  }

  return fallback;
}
