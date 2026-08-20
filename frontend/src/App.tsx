import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider } from "./auth/AuthContext";
import { ProtectedRoute } from "./auth/ProtectedRoute";
import { LoginPage } from "./pages/LoginPage";
import { AnalysisPage } from "./pages/AnalysisPage";
import { ProcessingPage } from "./pages/ProcessingPage";
import { HistoryPage } from "./pages/HistoryPage";
import { ReportPage } from "./pages/ReportPage";
import { CaseResultPage } from "./pages/CaseResultPage";
import { AnalysisV2Preview } from "./pages/AnalysisV2Preview";

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route
            path="/analysis"
            element={
              <ProtectedRoute>
                <AnalysisPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/processing/:jobId"
            element={
              <ProtectedRoute>
                <ProcessingPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/history"
            element={
              <ProtectedRoute>
                <HistoryPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/cases/:caseId"
            element={
              <ProtectedRoute>
                <CaseResultPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/cases/:caseId/report"
            element={
              <ProtectedRoute>
                <ReportPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/preview/analysis-v2"
            element={
              <ProtectedRoute>
                <AnalysisV2Preview />
              </ProtectedRoute>
            }
          />
          <Route path="/" element={<Navigate to="/analysis" replace />} />
          <Route path="*" element={<Navigate to="/analysis" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
