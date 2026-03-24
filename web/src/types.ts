export type ActionItem = {
  id: string;
  name: string;
  num_phases: number;
};

export type RealtimeResult = {
  status: string;
  message?: string;
  action_type?: string;
  confidence?: number;
  action_source?: string;
  phase?: number;
  phase_name?: string;
  overall_score?: number;
  is_standard?: boolean;
  errors?: string[];
  tips?: string[];
  rep_count?: number;
  target_reps?: number;
  completion_rate?: number;
  cadence?: number;
  warmup_progress?: number;
};

export type SessionReport = {
  session_id: string;
  duration_seconds: number;
  action_type: string;
  total_reps: number;
  target_reps: number;
  completion_rate: number;
  avg_score: number;
  best_score: number;
  error_histogram: Record<string, number>;
  score_series: number[];
};

export type VideoAssessment = {
  ok: boolean;
  error?: string;
  video_name?: string;
  total_frames?: number;
  action_type?: string;
  action_name?: string;
  confidence?: number;
  action_source?: string;
  phase?: number;
  phase_name?: string;
  overall_score?: number;
  is_standard?: boolean;
  estimated_reps?: number;
  errors?: string[];
  tips?: string[];
};

export type VideoTask = {
  task_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  message: string;
  filename: string;
  action_type?: string | null;
  created_at: number;
  updated_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  duration_sec?: number | null;
  error?: string | null;
  result?: VideoAssessment | null;
};

export type ActiveRealtimeSession = {
  session_id: string;
  action_hint?: string | null;
  current_action?: string | null;
  action_source?: string | null;
  current_score?: number | null;
  rep_count: number;
  target_reps: number;
  elapsed_seconds: number;
  status: string;
  last_message?: string | null;
};

export type AdminOverview = {
  system: {
    device: string;
    models: {
      yolo: boolean;
      action: boolean;
      phase_model_count: number;
      quality: boolean;
    };
  };
  realtime: {
    active_session_count: number;
    active_sessions: ActiveRealtimeSession[];
    latest_reports: Array<{
      session_id?: string;
      action_type?: string;
      avg_score?: number;
      total_reps?: number;
      duration_seconds?: number;
      _report_file?: string;
    }>;
  };
  video_tasks: {
    stats: {
      total: number;
      queued: number;
      running: number;
      completed: number;
      failed: number;
      avg_duration_sec: number;
    };
    latest: VideoTask[];
  };
};

export type SystemOverview = {
  project: string;
  device: string;
  readiness: {
    level: "full" | "partial";
    summary: string;
  };
  models: {
    pose_model_loaded: boolean;
    action_model_loaded: boolean;
    phase_model_count_loaded: number;
    quality_model_loaded: boolean;
  };
  checkpoints: {
    dir: string;
    action_model_file: string;
    quality_model_file: string;
    phase_model_files: string[];
    missing_files: string[];
    missing_phase_actions: string[];
  };
  features: Record<string, boolean>;
  recommended_setup_steps: string[];
};

export type IngestOverview = {
  run_state: {
    run_id: string;
    status: string;
    stage: string;
    message: string;
    updated_at?: number;
    summary: Record<string, number>;
    pipeline: Record<string, string>;
    latest_events: Array<{
      time?: number;
      level?: string;
      message?: string;
      action_id?: string;
    }>;
  };
  conversion: Array<{
    action_id: string;
    action_name: string;
    downloaded: number;
    skeleton: number;
    annotation: number;
    skeleton_rate: number;
    annotation_rate: number;
  }>;
  files: {
    manifest_count: number;
    rejected_count: number;
    tagged_count: number;
    cleanup_log_count: number;
  };
  rejected_reason_stats: Array<{
    reason: string;
    count: number;
  }>;
  latest_cleanup_logs: Array<{
    time?: number;
    filename?: string;
    removed?: boolean;
    dry_run?: boolean;
    reasons?: string[];
  }>;
};

export type LiveMonitorSnapshot = {
  time: number;
  health: {
    device: string;
    yolo: boolean;
    action: boolean;
    phase_model_count: number;
    quality: boolean;
  };
  realtime: {
    active_sessions: number;
  };
  video_tasks: {
    total: number;
    queued: number;
    running: number;
    completed: number;
    failed: number;
    avg_duration_sec: number;
  };
  ingest: {
    run_id: string;
    status: string;
    stage: string;
    message: string;
    updated_at?: number;
    summary: Record<string, number>;
    latest_events: Array<{
      time?: number;
      level?: string;
      message?: string;
      action_id?: string;
    }>;
  };
};
