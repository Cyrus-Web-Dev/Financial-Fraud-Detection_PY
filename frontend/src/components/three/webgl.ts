/** True bila browser mendukung WebGL (agar komponen 3D dapat jatuh kembali ke tampilan 2D). */
export function webglAvailable(): boolean {
  try {
    if (typeof document === "undefined") return false;
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

/** Pembangkit acak deterministik (mulberry32) agar adegan 3D selalu sama untuk seed yang sama. */
export function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
