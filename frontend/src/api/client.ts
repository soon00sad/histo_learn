import type {
  AnalysisResult,
  CaseDetail,
  CaseReviewInput,
  CaseSummary,
  IhcMarkersInput,
  JobAccepted,
  JobStatusOut,
  Token,
} from "./types";

const API_BASE = "/api/v1";
const TOKEN_STORAGE_KEY = "histovision.token";
const USER_STORAGE_KEY = "histovision.user";

/** Backend timestamps (Case.created_at etc.) serialize as naive UTC —
 * "2026-08-27T05:06:06" with no "Z"/offset, since they come from
 * dt.datetime.utcnow() with no tzinfo. `new Date(...)` on a string like
 * that is parsed as LOCAL time by every JS engine, not UTC — on a machine
 * whose clock isn't already set to UTC, every displayed timestamp would be
 * silently wrong by the local UTC offset. Appending "Z" (only if the
 * string doesn't already carry a zone marker) makes it parse as UTC, so
 * .toLocaleString()/.toLocaleDateString() then convert to the browser's
 * real local time, as they're meant to be used. */
export function parseUtc(isoString: string): Date {
  const hasZone = /[Zz]|[+-]\d\d:?\d\d$/.test(isoString);
  return new Date(hasZone ? isoString : `${isoString}Z`);
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function getStoredUser(): Token["user"] | null {
  const raw = localStorage.getItem(USER_STORAGE_KEY);
  return raw ? (JSON.parse(raw) as Token["user"]) : null;
}

export function storeSession(token: Token): void {
  localStorage.setItem(TOKEN_STORAGE_KEY, token.access_token);
  localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(token.user));
}

export function clearSession(): void {
  localStorage.removeItem(TOKEN_STORAGE_KEY);
  localStorage.removeItem(USER_STORAGE_KEY);
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const token = getStoredToken();
  const headers = new Headers(options.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // response had no JSON body; fall back to statusText
    }
    if (response.status === 401) clearSession();
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function ihcToFormData(form: FormData, ihc?: IhcMarkersInput): void {
  if (!ihc) return;
  if (ihc.ki67 !== undefined) form.append("ki67", String(ihc.ki67));
  if (ihc.er_status) form.append("er_status", ihc.er_status);
  if (ihc.pr_status) form.append("pr_status", ihc.pr_status);
  if (ihc.her2_status) form.append("her2_status", ihc.her2_status);
}

export const api = {
  login: (email: string, password: string) =>
    request<Token>("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    }),

  me: () => request<Token["user"]>("/auth/me"),

  analyzePatch: (file: File, tissueType: string, ihc?: IhcMarkersInput) => {
    const form = new FormData();
    form.append("file", file);
    form.append("tissue_type", tissueType);
    ihcToFormData(form, ihc);
    return request<AnalysisResult>("/analyze/patch", { method: "POST", body: form });
  },

  analyzeWsi: (file: File, tissueType: string, ihc?: IhcMarkersInput) => {
    const form = new FormData();
    form.append("file", file);
    form.append("tissue_type", tissueType);
    ihcToFormData(form, ihc);
    return request<JobAccepted>("/analyze/wsi", { method: "POST", body: form });
  },

  getJob: (jobId: string) => request<JobStatusOut>(`/jobs/${jobId}`),

  listCases: (params?: { status?: string; verdict?: string; search?: string; sort?: "priority" | "date" }) => {
    const query = new URLSearchParams();
    if (params?.status) query.set("status", params.status);
    if (params?.verdict) query.set("verdict", params.verdict);
    if (params?.search) query.set("search", params.search);
    if (params?.sort) query.set("sort", params.sort);
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return request<CaseSummary[]>(`/cases${suffix}`);
  },

  getCase: (caseId: string) => request<CaseDetail>(`/cases/${caseId}`),

  updateCaseStatus: (caseId: string, status: "pending" | "confirmed" | "rejected") =>
    request<CaseSummary>(`/cases/${caseId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    }),

  reviewCase: (caseId: string, review: CaseReviewInput) =>
    request<CaseSummary>(`/cases/${caseId}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(review),
    }),

  caseImageUrl: (caseId: string) => `${API_BASE}/cases/${caseId}/image`,
  caseMaskUrl: (caseId: string) => `${API_BASE}/cases/${caseId}/mask`,
  caseReportUrl: (caseId: string) => `${API_BASE}/cases/${caseId}/report.pdf`,

  generateReport: (caseId: string) =>
    request<{ report_url: string }>(`/cases/${caseId}/report`, { method: "POST" }),
};

/** Authenticated image/PDF fetches need the Authorization header, which a
 * plain <img src=...> or <a href=...> cannot attach — so callers request an
 * object URL through this helper and set it as the element's src/href. */
export async function fetchAuthenticatedBlobUrl(path: string): Promise<string> {
  const token = getStoredToken();
  const headers = new Headers();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(path, { headers });
  if (!response.ok) throw new ApiError(response.status, response.statusText);
  const blob = await response.blob();
  return URL.createObjectURL(blob);
}
