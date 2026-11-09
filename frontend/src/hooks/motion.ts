import { useEffect, useRef, useState } from "react";
import gsap from "gsap";

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Animasi angka (GSAP) dari nilai sebelumnya ke ``target``. Gerak dikurangi -> langsung ke nilai akhir. */
export function useCountUp(target: number | null, duration = 0.9): number | null {
  const [value, setValue] = useState<number | null>(() => (target === null || prefersReducedMotion() ? target : 0));
  const last = useRef<number>(0);
  useEffect(() => {
    if (target === null || !Number.isFinite(target) || prefersReducedMotion()) {
      setValue(target);
      if (target !== null) last.current = target;
      return;
    }
    const obj = { v: last.current };
    const tween = gsap.to(obj, {
      v: target,
      duration,
      ease: "power2.out",
      onUpdate: () => setValue(obj.v),
      onComplete: () => {
        last.current = target;
        setValue(target);
      },
    });
    return () => {
      tween.kill();
      last.current = obj.v;
    };
  }, [target, duration]);
  return value;
}
