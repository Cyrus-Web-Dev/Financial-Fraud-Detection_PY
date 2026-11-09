import { groupedPages, pageDef } from "../../lib/router.ts";
import { formatInt } from "../../lib/format.ts";
import { useStore } from "../../state/store.ts";
import { Badge } from "../ui/ui.tsx";

export function Sidebar() {
  const page = useStore((s) => s.page);
  const open = useStore((s) => s.sidebarOpen);
  const setPage = useStore((s) => s.setPage);
  const hasDataset = useStore((s) => s.dataset !== null);
  const health = useStore((s) => s.health);
  const initStatus = useStore((s) => s.status.init);
  const apiOk = health?.status === "ok";
  return (
    <aside className={`sidebar ${open ? "open" : ""}`} aria-label="Navigasi utama">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">
          <svg width="22" height="22" viewBox="0 0 32 32"><path d="M5 22l7-8 5 5 10-12" stroke="#ff5a6e" strokeWidth="3.2" fill="none" strokeLinecap="round" strokeLinejoin="round" /></svg>
        </div>
        <div>
          <div className="brand-title">Deteksi Fraud</div>
          <div className="brand-sub">Laporan keuangan · Statistika</div>
        </div>
      </div>
      <nav className="nav">
        {groupedPages().map((g) => (
          <div key={g.group}>
            <div className="nav-group">{g.group}</div>
            {g.pages.map((p) => (
              <button key={p.id} className="nav-item" aria-current={p.id === page ? "page" : undefined} onClick={() => setPage(p.id)}>
                <span className="nav-dot" />
                <span className="nav-label">{p.label}</span>
                {p.needsDataset && !hasDataset && <span className="nav-lock" title="Butuh dataset aktif">data</span>}
              </button>
            ))}
          </div>
        ))}
      </nav>
      <div className="side-foot">
        {initStatus.loading ? "Menghubungkan ke API…" : apiOk ? `API terhubung · ${health?.model_tersedia.length ?? 0} model` : "API tidak terhubung"}
      </div>
    </aside>
  );
}

export function TopBar() {
  const page = useStore((s) => s.page);
  const dataset = useStore((s) => s.dataset);
  const toggle = useStore((s) => s.toggleSidebar);
  const clear = useStore((s) => s.clearDataset);
  const health = useStore((s) => s.health);
  const def = pageDef(page);
  return (
    <header className="topbar">
      <button className="menu-btn" aria-label="Buka menu" onClick={() => toggle()}>☰</button>
      <div>
        <div className="topbar-title">{def.label}</div>
        <div className="small muted">{def.hint}</div>
      </div>
      <div className="topbar-spacer" />
      {dataset && (
        <div className="row" style={{ gap: 8 }}>
          <Badge tone="info">{dataset.nama} · {formatInt(dataset.nRows)} baris</Badge>
          <button className="btn btn-ghost btn-sm" onClick={clear} title="Lepas dataset aktif">Lepas</button>
        </div>
      )}
      <Badge tone={health?.status === "ok" ? "ok" : "error"}>{health?.status === "ok" ? "Online" : "Offline"}</Badge>
    </header>
  );
}
