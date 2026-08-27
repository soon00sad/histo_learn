import { useState, type FormEvent } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { Logo } from "../components/Logo";
import { ApiError } from "../api/client";

export function LoginPage() {
  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (isAuthenticated) return <Navigate to="/analysis" replace />;

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);
    try {
      await login(email, password);
      navigate("/analysis");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось подключиться к серверу");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div
      style={{
        position: "relative",
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: `radial-gradient(circle at 22% 12%, var(--hv-hero-glow-brand) 0%, oklch(0 0 0 / 0%) 42%),
          radial-gradient(circle at 82% 88%, var(--hv-hero-glow-accent) 0%, oklch(0 0 0 / 0%) 40%),
          var(--hv-hero-bg)`,
        fontFamily: "var(--hv-font-body)",
        padding: 24,
        overflow: "hidden",
      }}
    >
      <HeroDecoration />
      <div style={{ position: "relative", width: "100%", maxWidth: 420 }}>
        <div style={{ display: "flex", justifyContent: "center", marginBottom: 36 }}>
          <Logo height={34} variant="white" />
        </div>

        <form
          onSubmit={handleSubmit}
          style={{ background: "#fff", borderRadius: 22, padding: "38px 36px", boxShadow: "0 40px 80px -30px oklch(0 0 0 / 55%)" }}
        >
          <div style={{ fontSize: 23, fontWeight: 700, fontFamily: "var(--hv-font-display)", color: "oklch(0.22 0.05 264)", letterSpacing: "-0.01em" }}>
            Вход в систему
          </div>
          <div style={{ fontSize: 13, color: "var(--hv-text-muted)", marginTop: 6, marginBottom: 28 }}>
            Платформа поддержки диагностики рака молочной железы
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div>
              <div style={{ fontSize: 12.5, fontWeight: 600, color: "oklch(0.35 0.03 264)", marginBottom: 7 }}>
                Логин или e-mail
              </div>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="e.sokolova@clinic.ru"
                style={inputStyle}
              />
            </div>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 7 }}>
                <span style={{ fontSize: 12.5, fontWeight: 600, color: "oklch(0.35 0.03 264)" }}>Пароль</span>
              </div>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••"
                style={inputStyle}
              />
            </div>

            {error && (
              <div style={{ fontSize: 12.5, color: "var(--hv-malignant)", fontWeight: 600 }}>{error}</div>
            )}

            <button
              type="submit"
              disabled={isSubmitting}
              style={{
                marginTop: 8,
                textAlign: "center",
                border: "none",
                cursor: isSubmitting ? "default" : "pointer",
                padding: "14px 22px",
                borderRadius: "var(--hv-radius-btn)",
                background: "var(--hv-brand-gradient)",
                color: "#fff",
                fontSize: 14.5,
                fontWeight: 700,
                fontFamily: "var(--hv-font-display)",
                boxShadow: "var(--hv-shadow-btn)",
                opacity: isSubmitting ? 0.7 : 1,
              }}
            >
              {isSubmitting ? "Вход…" : "Войти"}
            </button>
          </div>

          <div style={{ marginTop: 26, paddingTop: 22, borderTop: "1px solid var(--hv-border-light)", display: "flex", gap: 11, alignItems: "flex-start" }}>
            <div
              style={{
                width: 17,
                height: 19,
                border: "1.6px solid oklch(0.45 0.15 150)",
                borderRadius: "3px 3px 8px 8px",
                position: "relative",
                flex: "none",
                marginTop: 1,
              }}
            >
              <div
                style={{
                  position: "absolute",
                  left: "50%",
                  top: -8,
                  transform: "translateX(-50%)",
                  width: 9,
                  height: 9,
                  border: "1.6px solid oklch(0.45 0.15 150)",
                  borderBottom: "none",
                  borderRadius: "9px 9px 0 0",
                }}
              />
            </div>
            <div style={{ fontSize: 12, lineHeight: 1.55, color: "var(--hv-text-muted)" }}>
              Данные препаратов и пациентов обрабатываются локально в контуре клиники и не покидают её инфраструктуру.
            </div>
          </div>
        </form>

        <div style={{ textAlign: "center", marginTop: 22, fontSize: 11.5, color: "oklch(0.55 0.02 264 / 80%)" }}>
          HistoVision · система поддержки принятия решений · v2.3.1
        </div>
      </div>
    </div>
  );
}

/** Purely decorative background layer — thin curved outline paths, in the
 * spirit of thinkbio.ai's hero line-art, rendered on our own accent-green
 * token rather than any asset of theirs. Absolutely positioned, no pointer
 * events, so it never interferes with the form. */
function HeroDecoration() {
  return (
    <svg
      aria-hidden="true"
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
      preserveAspectRatio="xMidYMid slice"
    >
      <path
        d="M -40 120 C 180 40, 260 40, 320 140 S 420 260, 620 200"
        fill="none"
        stroke="var(--hv-hero-line)"
        strokeWidth="1.5"
      />
      <path
        d="M 60 620 C 220 560, 260 480, 200 400 S 300 260, 520 300"
        fill="none"
        stroke="var(--hv-hero-line)"
        strokeWidth="1.5"
      />
      <circle cx="320" cy="140" r="3" fill="var(--hv-accent-green)" opacity="0.6" />
      <circle cx="200" cy="400" r="3" fill="var(--hv-accent-teal)" opacity="0.5" />
    </svg>
  );
}

const inputStyle = {
  width: "100%",
  boxSizing: "border-box" as const,
  padding: "12px 14px",
  borderRadius: 10,
  border: "1.5px solid oklch(0.88 0.008 264)",
  fontSize: 14,
  fontFamily: "inherit",
  outline: "none",
  background: "oklch(0.98 0.004 264)",
};
