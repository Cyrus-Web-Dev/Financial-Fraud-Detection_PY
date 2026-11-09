import { useEffect, useRef } from "react";
import * as THREE from "three";
import { clamp, formatProb } from "../../lib/format.ts";
import type { Tone } from "../../lib/format.ts";
import { prefersReducedMotion } from "../../hooks/motion.ts";
import { webglAvailable } from "./webgl.ts";

const COLORS: Record<Tone, number> = { high: 0xd7263d, mid: 0xe08a14, low: 0x1a9a68 };
const CSS_COLORS: Record<Tone, string> = { high: "#d7263d", mid: "#e08a14", low: "#1a9a68" };
const SWEEP = Math.PI * 1.5; // busur 270 derajat, celah di bawah
const START = (Math.PI * 5) / 4; // ujung kiri-bawah
const FULL_SCALE = 0.5; // probabilitas >= 50% memenuhi busur

/** Gauge risiko 3D (busur torus). Tanpa WebGL: batang progres 2D. */
export function RiskGauge3D(props: { probability: number; tone: Tone; caption?: string }) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const target = useRef(0);
  const toneRef = useRef<Tone>(props.tone);
  const frac = clamp(props.probability / FULL_SCALE, 0, 1);
  target.current = frac;
  toneRef.current = props.tone;
  const gl = webglAvailable();

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !gl) return;
    let raf = 0;
    let renderer: THREE.WebGLRenderer | null = null;
    let cleanup = () => {};
    try {
      const w = host.clientWidth || 220;
      const h = host.clientHeight || 200;
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(w, h, false);
      host.appendChild(renderer.domElement);
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 50);
      camera.position.set(0, 0, 5.2);
      const group = new THREE.Group();
      scene.add(group);

      const trackGeo = new THREE.TorusGeometry(1.25, 0.13, 20, 96, SWEEP);
      const trackMat = new THREE.MeshBasicMaterial({ color: 0xdde5f1 });
      const track = new THREE.Mesh(trackGeo, trackMat);
      track.rotation.z = -Math.PI / 4;
      group.add(track);

      const progMat = new THREE.MeshBasicMaterial({ color: COLORS[toneRef.current] });
      let progGeo: THREE.TorusGeometry | null = null;
      const prog = new THREE.Mesh(new THREE.BufferGeometry(), progMat);
      group.add(prog);

      let current = 0;
      let built = -1;
      const reduced = prefersReducedMotion();
      const rebuild = (f: number) => {
        if (progGeo) progGeo.dispose();
        const arc = Math.max(0.001, f * SWEEP);
        progGeo = new THREE.TorusGeometry(1.25, 0.19, 20, 96, arc);
        prog.geometry = progGeo;
        prog.rotation.z = START - arc;
        prog.visible = f > 0.004;
        built = f;
      };
      const frame = (t: number) => {
        current = reduced ? target.current : current + (target.current - current) * 0.08;
        if (Math.abs(current - built) > 0.002 || built < 0) rebuild(current);
        progMat.color.setHex(COLORS[toneRef.current]);
        if (!reduced) {
          group.rotation.y = Math.sin(t * 0.0007) * 0.35;
          group.rotation.x = Math.sin(t * 0.0005) * 0.12;
        }
        renderer?.render(scene, camera);
        if (!reduced || Math.abs(current - target.current) > 0.002) raf = requestAnimationFrame(frame);
      };
      raf = requestAnimationFrame(frame);

      cleanup = () => {
        cancelAnimationFrame(raf);
        trackGeo.dispose();
        trackMat.dispose();
        progMat.dispose();
        if (progGeo) progGeo.dispose();
        if (renderer) {
          renderer.dispose();
          if (renderer.domElement.parentNode === host) host.removeChild(renderer.domElement);
        }
      };
    } catch {
      cleanup = () => {
        if (renderer) renderer.dispose();
      };
    }
    return () => cleanup();
  }, [gl]);

  return (
    <div>
      <div className="gauge" role="img" aria-label={`Probabilitas fraud ${formatProb(props.probability)}`}>
        <div ref={hostRef} style={{ position: "absolute", inset: 0 }} />
        <div className="gauge-center">
          <div className="gauge-value num" style={{ color: CSS_COLORS[props.tone] }}>{formatProb(props.probability)}</div>
          {props.caption && <div className="small muted">{props.caption}</div>}
        </div>
      </div>
      {!gl && (
        <div className="gauge-fallback">
          <div style={{ width: `${frac * 100}%`, background: CSS_COLORS[props.tone] }} />
        </div>
      )}
    </div>
  );
}
