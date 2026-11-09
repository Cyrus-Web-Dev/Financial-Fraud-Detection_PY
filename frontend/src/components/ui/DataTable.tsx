import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { paginate, sortRows } from "../../state/logic.ts";
import type { SortDir } from "../../state/logic.ts";

export type Column<T> = {
  key: string;
  label: string;
  render: (row: T) => ReactNode;
  sortValue?: (row: T) => string | number | null | undefined;
  align?: "left" | "right";
};

export function DataTable<T>(props: {
  rows: readonly T[]; columns: readonly Column<T>[]; pageSize?: number; empty?: string; maxHeight?: number;
  initialSort?: { key: string; dir: SortDir }; onRowClick?: (row: T) => void; isSelected?: (row: T) => boolean; rowKey?: (row: T, i: number) => string;
}) {
  const [sort, setSort] = useState<{ key: string; dir: SortDir } | null>(props.initialSort ?? null);
  const [page, setPage] = useState(1);
  const sorted = useMemo(() => {
    const col = props.columns.find((c) => c.key === sort?.key);
    return sort && col?.sortValue ? sortRows(props.rows, col.sortValue, sort.dir) : [...props.rows];
  }, [props.rows, props.columns, sort]);
  const pg = paginate(sorted, page, props.pageSize ?? Math.max(1, sorted.length));
  const toggle = (c: Column<T>) => {
    if (!c.sortValue) return;
    setPage(1);
    setSort((s) => (s?.key === c.key ? { key: c.key, dir: s.dir === "asc" ? "desc" : "asc" } : { key: c.key, dir: "desc" }));
  };
  if (props.rows.length === 0) return <div className="empty">{props.empty ?? "Tidak ada data."}</div>;
  return (
    <div>
      <div className="table-wrap" style={props.maxHeight ? { maxHeight: props.maxHeight, overflowY: "auto" } : undefined}>
        <table className="tbl">
          <thead>
            <tr>
              {props.columns.map((c) => (
                <th key={c.key} className={`${c.sortValue ? "sortable" : ""} ${c.align === "right" ? "r" : ""}`} onClick={() => toggle(c)} aria-sort={sort?.key === c.key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}>
                  {c.label}
                  {sort?.key === c.key ? (sort.dir === "asc" ? " ▲" : " ▼") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pg.items.map((row, i) => (
              <tr key={props.rowKey ? props.rowKey(row, i) : i} className={`${props.onRowClick ? "clickable" : ""} ${props.isSelected?.(row) ? "selected" : ""}`} onClick={props.onRowClick ? () => props.onRowClick?.(row) : undefined}>
                {props.columns.map((c) => (
                  <td key={c.key} className={c.align === "right" ? "r" : ""}>{c.render(row)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {pg.pages > 1 && (
        <div className="pager">
          <span>{pg.total} baris · halaman {pg.page} dari {pg.pages}</span>
          <button className="btn btn-ghost btn-sm" disabled={pg.page <= 1} onClick={() => setPage(pg.page - 1)}>‹ Sebelumnya</button>
          <button className="btn btn-ghost btn-sm" disabled={pg.page >= pg.pages} onClick={() => setPage(pg.page + 1)}>Berikutnya ›</button>
        </div>
      )}
    </div>
  );
}
