// Mirrors src/api/schemas.py — keep in sync with the backend response models.

export interface UserOut {
  email: string;
  full_name: string;
  role: string;
}

export interface Token {
  access_token: string;
  token_type: string;
  user: UserOut;
}

export interface ClassAreaOut {
  name_en: string;
  name_ru: string;
  color: string;
  fraction: number;
}

export interface AnalysisResult {
  case_id: string;
  verdict_label: string;
  is_malignant: boolean;
  tumor_area_fraction: number;
  class_areas: ClassAreaOut[];
  analysis_mode: "patch" | "wsi";
}

export interface CaseSummary {
  id: string;
  created_at: string;
  tissue_type: string;
  verdict_label: string;
  is_malignant: boolean;
  tumor_area_fraction: number;
  status: "pending" | "confirmed";
}

export interface CaseDetail extends CaseSummary {
  source_filename: string;
  analysis_mode: "patch" | "wsi";
  class_areas: ClassAreaOut[];
  ki67: number | null;
  er_status: string | null;
  pr_status: string | null;
  her2_status: string | null;
  report_available: boolean;
}

export interface JobStatusOut {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  stage: string;
  progress: number;
  message: string;
  case_id: string | null;
  error: string | null;
}

export interface JobAccepted {
  job_id: string;
}

export interface IhcMarkersInput {
  ki67?: number;
  er_status?: string;
  pr_status?: string;
  her2_status?: string;
}
