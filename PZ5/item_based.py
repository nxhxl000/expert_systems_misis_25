from pathlib import Path
import csv
import math
from typing import Optional, Sequence, List, Tuple
from tabulate import tabulate

# ---------- утилиты ----------

def _guess_delimiter(sample_lines: Sequence[str]) -> str:
    if not sample_lines:
        return ';'
    cands = [';', ',', '\t']
    counts = {delim: sum(ln.count(delim) for ln in sample_lines) for delim in cands}
    return max(counts, key=counts.get)

def _to_number(s: str) -> float:
    s = s.strip().strip('"').strip("'").replace(",", ".")
    if s == "":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0

def _is_numberish(s: str) -> bool:
    s2 = s.strip().replace(",", ".")
    if s2 == "":
        return True
    try:
        float(s2)
        return True
    except ValueError:
        return False

def read_pref_csv(filename: str, delimiter: Optional[str] = None):
    csv_path = Path(__file__).parent / filename
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        return None, None, []
    if delimiter is None:
        delimiter = _guess_delimiter(lines[:10])
    rows = list(csv.reader(lines, delimiter=delimiter))

    has_header = not all(_is_numberish(x) for x in rows[0])
    headers = rows[0] if has_header else None
    data_rows = rows[1:] if has_header else rows

    has_row_labels = bool(data_rows and any(not _is_numberish(r[0]) for r in data_rows))
    row_labels = [r[0] for r in data_rows] if has_row_labels else None
    if has_row_labels:
        data_rows = [r[1:] for r in data_rows]
        if headers is not None:
            headers = headers[1:]

    data = [[_to_number(c) for c in r] for r in data_rows]
    return headers, row_labels, data

def render_pref_table(filename: str) -> str:
    csv_path = Path(__file__).parent / filename
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        return "(пустая таблица)"
    delim = _guess_delimiter(lines[:10])
    rows = list(csv.reader(lines, delimiter=delim))
    return tabulate(rows, tablefmt="fancy_grid", stralign="center", numalign="center")

def mean_ignore_zeros(values: List[float]) -> Optional[float]:
    nz = [v for v in values if v != 0]
    if not nz:
        return None
    return sum(nz) / len(nz)

def _cos_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def render_square_matrix(labels: List[str], M: List[List[float]], ndigits: int = 3) -> str:
    rows_out = []
    for i, row in enumerate(M):
        rows_out.append([labels[i], *[round(x, ndigits) for x in row]])
    headers = [""] + labels
    return tabulate(rows_out, headers=headers, tablefmt='fancy_grid', stralign='center', numalign='center')

# ---------- предметная логика  ----------

def find_best_items(data: List[List[float]], row_labels: Optional[List[str]], eps: float = 1e-9
                   ) -> Tuple[List[int], Optional[float], str]:
    def plabel(i: int) -> str:
        return row_labels[i] if row_labels is not None else f"P{i+1}"
    item_means = [mean_ignore_zeros(row) for row in data]
    finite = [m for m in item_means if m is not None]
    if not finite:
        return [], None, ""
    best = max(finite)
    idxs = [i for i, m in enumerate(item_means) if (m is not None) and abs(m - best) <= eps]
    return idxs, best, ", ".join(plabel(i) for i in idxs)

def compute_item_similarity_and_users(data: List[List[float]]):
    n_items = len(data)
    n_users = len(data[0]) if data else 0
    non_new_users = [j for j in range(n_users) if not all(data[r][j] == 0 for r in range(n_items))]
    new_users = [j for j in range(n_users) if j not in non_new_users]

    # обрезаем нулевые столбцы под сходства
    data_wo_new = [[row[j] for j in non_new_users] for row in data]

    S = [[0.0]*n_items for _ in range(n_items)]
    for a in range(n_items):
        for b in range(n_items):
            if a == b:
                continue
            S[a][b] = _cos_sim(data_wo_new[a], data_wo_new[b])
    return S, non_new_users, new_users

def show_neighbors_slices_topk_and_predict(data: List[List[float]],
                                           headers: Optional[List[str]],
                                           row_labels: Optional[List[str]],
                                           S: List[List[float]],
                                           non_new_users: List[int],
                                           top_k: int,
                                           pred_threshold: float) -> List[str]:
    def ulabel(j: int) -> str:
        return headers[j] if headers is not None else f"U{j+1}"
    def plabel(i: int) -> str:
        return row_labels[i] if row_labels is not None else f"P{i+1}"

    rec_lines: List[str] = []
    non_rec_lines: List[str] = []
    n_items = len(data)
    for u in non_new_users:
        for i in range(n_items):
            if data[i][u] != 0:
                continue 

            print("\n" + "-"*80)
            print(f"Пара для анализа: Пользователь = {ulabel(u)}, Товар = {plabel(i)}")

            # кандидаты-соседи: оценены пользователем u и имеют положительное сходство
            candidates = [
                (j, S[i][j]) for j in range(n_items)
                if j != i and data[j][u] != 0 and S[i][j] > 0
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)
            neighbor_items = [j for j, _ in candidates[:top_k]]

            kept_item_indices = [i] + neighbor_items
            kept_user_indices = non_new_users 

            # печать среза
            def row_name(idx: int) -> str:
                return row_labels[idx] if row_labels is not None else f"P{idx+1}"
            def col_name(idx: int) -> str:
                return headers[idx] if headers is not None else f"U{idx+1}"

            rows = []
            for r_i in kept_item_indices:
                rows.append([row_name(r_i), *[data[r_i][c_j] for c_j in kept_user_indices]])
            headers_row = [""] + [col_name(j) for j in kept_user_indices]

            print("\nБлижайшие соседние товары (TOP-{} по косинусному сходству):\n".format(top_k))
            print(tabulate(rows, headers=headers_row, tablefmt="fancy_grid",
                           stralign="center", numalign="center"))

            # ---- прогноз: sum(r_uj * cos) / sum(|cos|) ----
            num, den = 0.0, 0.0
            for j in neighbor_items:
                sim = S[i][j]
                r_uj = data[j][u]
                num += r_uj * sim
                den += abs(sim)

            if den == 0:
                print("\nПрогноз: недостаточно данных (нет подходящих соседей).")
                continue

            pred = num / den
            print(f"\nПрогноз по формуле (взвешенное среднее): {pred:.3f}")

            if pred >= pred_threshold:
                rec_lines.append(f"Пользователю {ulabel(u)} рекомендуем товар {plabel(i)} (прогноз {pred:.3f})")
            else:
                non_rec_lines.append(f"Пользователю {ulabel(u)} не рекомендуем товар {plabel(i)} (прогноз {pred:.3f})")

    return rec_lines, non_rec_lines

# ---------- main ----------

if __name__ == "__main__":
    FILENAME = "preference_matrix.csv"
    EPS = 1e-9
    TOP_K = 3               # число ближайших товаров
    PRED_THRESHOLD = 4.5    # порог «рекомендовать/не рекомендовать»

    # 1) Исходная матрица предпочтений
    print("Исходная матрица предпочтений:\n")
    print(render_pref_table(FILENAME))

    # 2) Читаем данные
    headers, row_labels, data = read_pref_csv(FILENAME)

    # 3) Топ-товары по средней (для новых пользователей)
    best_indices, best_mean, best_names = find_best_items(data, row_labels, eps=EPS)
    if not best_indices:
        print("\nТовар с максимальной оценкой: не найден (все строки пустые или нулевые).")
    else:
        if len(best_indices) == 1:
            print(f"\nТовар с максимальной средней оценкой: {best_names} (ср. {best_mean:.2f})")
        else:
            print(f"\nТовары с максимальной средней оценкой ({best_mean:.2f}): {best_names}")

    # 4) Сходства между товарами и разбиение пользователей
    S, non_new_users, new_users = compute_item_similarity_and_users(data)

    # 5)печать матрицы сходств
    def plabel(i: int) -> str:
        return row_labels[i] if row_labels is not None else f"P{i+1}"
    item_labels = [plabel(i) for i in range(len(data))]
    print("\nМатрица косинусных сходств между товарами:\n")
    print(render_square_matrix(item_labels, S, ndigits=3))

    # 6) TOP-K соседи, прогнозы и сбор рекомендаций
    rec_lines, non_rec_lines = show_neighbors_slices_topk_and_predict(
        data=data,
        headers=headers,
        row_labels=row_labels,
        S=S,
        non_new_users=non_new_users,
        top_k=TOP_K,
        pred_threshold=PRED_THRESHOLD
    )

    # 7) Итоговый вывод рекомендаций
    if rec_lines or non_rec_lines or new_users:
        print("\n" + "="*80)
        print(f"ИТОГОВЫЕ РЕКОМЕНДАЦИИ (порог прогноза ≥ {PRED_THRESHOLD:.1f}):")

        if rec_lines:
            print("\nРекомендованные товары:")
            for ln in rec_lines:
                print(" -", ln)

        if non_rec_lines:
            print("\nТовары, которые не рекомендуем:")
            for ln in non_rec_lines:
                print(" -", ln)

        if new_users and best_indices:
            best_names_str = ", ".join(plabel(i) for i in best_indices)
            print("\nРекомендации для новых пользователей:")
            for u in new_users:
                uname = headers[u] if headers is not None else f"U{u+1}"
                if len(best_indices) == 1:
                    print(f" - Пользователю {uname}: рекомендуем товар {best_names_str} (средняя оценка: {best_mean:.2f})")
                else:
                    print(f" - Пользователю {uname}: рекомендуем товары {best_names_str} (средняя оценка: {best_mean:.2f})")
    else:
        print("\nНет рекомендаций по заданным данным и правилам.")
