import { useEffect, useRef } from "react";
import * as THREE from "three";
import { prefersReducedMotion } from "../../hooks/motion.ts";
import { seeded, webglAvailable } from "./webgl.ts";

const NODES = 90;
const LINK_DIST = 2.7;

/** Latar hero: jaringan partikel 3D yang berputar pelan dengan parallax pointer. Gagal -> tidak merender apa pun. */
export function HeroScene() {
  const hostRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !webglAvailable()) return;
    let raf = 0;
    let renderer: THREE.WebGLRenderer | null = null;
    const disposables: { dispose(): void }[] = [];
    let cleanup = () => {};
    try {
      let w = host.clientWidth || 900;
      let h = host.clientHeight || 340;
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(w, h, false);
      host.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 100);
      camera.position.set(0, 0, 9);
      const group = new THREE.Group();
      scene.add(group);

      const rnd = seeded(20240611);
      const pos = new Float32Array(NODES * 3);
      for (let i = 0; i < NODES; i++) {
        pos[i * 3] = (rnd() - 0.5) * 12;
        pos[i * 3 + 1] = (rnd() - 0.5) * 6;
        pos[i * 3 + 2] = (rnd() - 0.5) * 6;
      }
      const pointGeo = new THREE.BufferGeometry();
      pointGeo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      const pointMat = new THREE.PointsMaterial({ color: 0xff7b8b, size: 0.11, transparent: true, opacity: 0.95 });
      group.add(new THREE.Points(pointGeo, pointMat));

      const seg: number[] = [];
      for (let i = 0; i < NODES; i++) {
        for (let j = i + 1; j < NODES; j++) {
          const dx = (pos[i * 3] ?? 0) - (pos[j * 3] ?? 0);
          const dy = (pos[i * 3 + 1] ?? 0) - (pos[j * 3 + 1] ?? 0);
          const dz = (pos[i * 3 + 2] ?? 0) - (pos[j * 3 + 2] ?? 0);
          if (dx * dx + dy * dy + dz * dz < LINK_DIST * LINK_DIST) {
            seg.push(pos[i * 3] ?? 0, pos[i * 3 + 1] ?? 0, pos[i * 3 + 2] ?? 0, pos[j * 3] ?? 0, pos[j * 3 + 1] ?? 0, pos[j * 3 + 2] ?? 0);
          }
        }
      }
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(seg, 3));
      const lineMat = new THREE.LineBasicMaterial({ color: 0x8fb2f0, transparent: true, opacity: 0.3 });
      group.add(new THREE.LineSegments(lineGeo, lineMat));
      disposables.push(pointGeo, pointMat, lineGeo, lineMat);

      let px = 0;
      let py = 0;
      const onMove = (e: PointerEvent) => {
        const r = host.getBoundingClientRect();
        px = ((e.clientX - r.left) / Math.max(1, r.width) - 0.5) * 2;
        py = ((e.clientY - r.top) / Math.max(1, r.height) - 0.5) * 2;
      };
      host.addEventListener("pointermove", onMove);

      const resize = () => {
        w = host.clientWidth || w;
        h = host.clientHeight || h;
        renderer?.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      };
      window.addEventListener("resize", resize);

      const reduced = prefersReducedMotion();
      const frame = (t: number) => {
        if (!reduced) {
          group.rotation.y = t * 0.00008;
          group.rotation.x = Math.sin(t * 0.0002) * 0.12;
          camera.position.x += (px * 0.8 - camera.position.x) * 0.04;
          camera.position.y += (-py * 0.5 - camera.position.y) * 0.04;
          camera.lookAt(0, 0, 0);
        }
        renderer?.render(scene, camera);
        if (!reduced) raf = requestAnimationFrame(frame);
      };
      raf = requestAnimationFrame(frame);

      cleanup = () => {
        cancelAnimationFrame(raf);
        host.removeEventListener("pointermove", onMove);
        window.removeEventListener("resize", resize);
        for (const d of disposables) d.dispose();
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
  }, []);

  return <div ref={hostRef} aria-hidden="true" style={{ position: "absolute", inset: 0 }} />;
}
