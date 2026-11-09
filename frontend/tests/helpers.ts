import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
export const FIXTURES = join(here, "fixtures");
export const PROCESSED = join(here, "..", "..", "data", "processed");

export function fixture<T>(name: string): T {
  return JSON.parse(readFileSync(join(FIXTURES, `${name}.json`), "utf-8")) as T;
}
export function processed<T>(name: string): T | null {
  const p = join(PROCESSED, `${name}.json`);
  return existsSync(p) ? (JSON.parse(readFileSync(p, "utf-8")) as T) : null;
}
