import { Component } from "react";
import type { ReactNode } from "react";

type Props = { children?: ReactNode; fallback?: ReactNode; label?: string };
type State = { failed: boolean; message: string };

/** Menangkap galat render agar satu komponen yang gagal (mis. WebGL) tidak merobohkan seluruh aplikasi. */
export class ErrorBoundary extends Component<Props, State> {
  override state: State = { failed: false, message: "" };

  static getDerivedStateFromError(error: unknown): State {
    return { failed: true, message: error instanceof Error ? error.message : String(error) };
  }

  override componentDidCatch(error: unknown): void {
    console.error("[ErrorBoundary]", error);
  }

  override render(): ReactNode {
    if (this.state.failed) {
      return this.props.fallback ?? (
        <div className="alert error" role="alert">
          {this.props.label ?? "Bagian ini"} gagal ditampilkan. Muat ulang halaman atau pindah ke halaman lain.
        </div>
      );
    }
    return this.props.children;
  }
}
