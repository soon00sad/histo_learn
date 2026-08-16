export function Disclaimer({ modelVersion = "v2.3.1" }: { modelVersion?: string }) {
  return (
    <div style={{ padding: "20px 48px 32px", borderTop: "1px solid var(--hv-border-light)", marginTop: 8 }}>
      <div className="hv-disclaimer">
        Система поддержки принятия решений. Окончательное заключение формулирует врач. · Модель {modelVersion}
      </div>
    </div>
  );
}
