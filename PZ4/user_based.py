from pathlib import Path
import csv
import math
from typing import Optional, Sequence, List
from tabulate import tabulate

# ---------- вспомогательные ----------

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

def _transpose(mat: List[List[float]]) -> List[List[float]]:
    return list(map(list, zip(*mat))) if mat else []

def render_square_matrix(labels: List[str], M: List[List[float]], ndigits: int = 3) -> str:
    rows_out = []
    for i, row in enumerate(M):
        rows_out.append([labels[i], *[round(x, ndigits) for x in row]])
    headers = [""] + labels
    return tabulate(rows_out, headers=headers, tablefmt='fancy_grid', stralign='center', numalign='center')

# ---------- main ----------

if __name__ == "__main__":
    FILENAME = "preference_matrix.csv"
    PRED_THRESHOLD = 4.0  # порог для решения «рекомендовать/не рекомендовать»

    # 1) Исходная матрица предпочтений
    print("Исходная матрица предпочтений:")
    print()
    print(render_pref_table(FILENAME))

    # 2) Читаем данные и находим товар с максимальной средней
    headers, row_labels, data = read_pref_csv(FILENAME)

    best_idx = None
    best_mean = None
    for i, row in enumerate(data):
        m = mean_ignore_zeros(row)
        if m is None:
            continue
        if (best_mean is None) or (m > best_mean):
            best_mean = m
            best_idx = i

    prod_name = (row_labels[best_idx] if (row_labels is not None and best_idx is not None)
                 else f"P{(best_idx + 1) if best_idx is not None else '?'}")

    print()
    if best_idx is None:
        print("Товар с максимальной оценкой: не найден (все строки пустые или нулевые).")
    else:
        print(f"Товар с максимальной оценкой: {prod_name} - средняя оценка ({best_mean:.2f})")

    # 3) Подготовка индексов пользователей
    n_rows = len(data)
    n_cols = len(data[0]) if data else 0
    user_vectors = _transpose(data)

    def ulabel(j: int) -> str:
        return headers[j] if headers is not None else f"U{j+1}"

    def plabel(i: int) -> str:
        return row_labels[i] if row_labels is not None else f"P{i+1}"

    non_new_users = [j for j in range(n_cols) if not all(data[r][j] == 0 for r in range(n_rows))]
    new_users = [j for j in range(n_cols) if j not in non_new_users]

    # 4) матрица косинусных сходств по всем не-новым пользователям
    sub_labels = [ulabel(j) for j in non_new_users]
    sub_vectors = [user_vectors[j] for j in non_new_users]
    k = len(sub_vectors)

    S = [[0.0]*k for _ in range(k)]
    for a in range(k):
        for b in range(k):
            if a == b:
                S[a][b] = 0.0
            else:
                S[a][b] = _cos_sim(sub_vectors[a], sub_vectors[b])

    print("\nМатрица косинусных «расстояний» (сходств) по всем не-новым пользователям:\n")
    print(render_square_matrix(sub_labels, S, ndigits=3))

    # Быстрый доступ: глобальный индекс пользователя -> индекс в S
    idx_in_S = {global_j: pos for pos, global_j in enumerate(non_new_users)}

    # 5) Порог сходства и порог прогноза
    while True:
        thr_in = input("\nВведите порог косинусного сходства (например, 0.8): ").strip().replace(",", ".")
        try:
            cos_threshold = float(thr_in)
            break
        except ValueError:
            print("Некорректное число. Попробуйте ещё раз (пример: 0.8).")

    print(f"\nПорог сходства принят: {cos_threshold}")
    print(f"Порог для рекомендации (прогноза): {PRED_THRESHOLD:.1f}\n")

    # 6) Автопроход по парам (u, i) c нулём у не-новых пользователей
    rec_summary_lines: List[str] = []
    new_user_lines: List[str] = []

    # Сначала подготовим сообщения для новых пользователей
    for u in new_users:
        new_user_lines.append(
            f"Пользователь {ulabel(u)} — новый. Рекомендуем товар с наибольшей средней: "
            f"{prod_name} (ср. {best_mean:.2f})"
        )

    for u in non_new_users:
        for pi in range(n_rows):
            if data[pi][u] != 0:
                continue  # интересуют только нулевые места — нужно предсказать

            user_selected = ulabel(u)
            product_selected = plabel(pi)

            print("\n" + "-"*80)
            print(f"Пара для прогноза: Пользователь = {user_selected}, Товар = {product_selected}")

            # Соседи: не-новые пользователи, у кого сходство >= порога И есть оценка по товару pi
            u_pos = idx_in_S[u]
            neighbor_globals = [
                v for v in non_new_users
                if v != u
                and S[u_pos][idx_in_S[v]] >= cos_threshold
                and data[pi][v] != 0
            ]

            kept_col_indices = [u] + neighbor_globals

            # «Следующая матрица»: исходная матрица предпочтений только с отобранными пользователями
            filtered_headers = [ulabel(j) for j in kept_col_indices]
            filtered_rows = []
            if row_labels is not None:
                for r_i, row in enumerate(data):
                    filtered_rows.append([row_labels[r_i], *[row[c_j] for c_j in kept_col_indices]])
                headers_for_print = [""] + filtered_headers
            else:
                for row in data:
                    filtered_rows.append([row[c_j] for c_j in kept_col_indices])
                headers_for_print = filtered_headers

            print("\n Ближайшие соседи:\n")
            print(tabulate(filtered_rows, headers=headers_for_print,
                           tablefmt="fancy_grid", stralign="center", numalign="center"))

            # Прогноз CF с центрированием относительно средних
            r_a = mean_ignore_zeros(user_vectors[u])
            num, den = 0.0, 0.0
            for v in neighbor_globals:
                r_ui = data[pi][v]
                if r_ui == 0:
                    continue
                r_u = mean_ignore_zeros(user_vectors[v])
                sim = S[u_pos][idx_in_S[v]]
                num += (r_ui - r_u) * sim
                den += abs(sim)

            pred = r_a if (den == 0 or r_a is None) else (r_a + num / den)
            print(f"\nПрогноз для {user_selected} по {product_selected}: {pred:.3f}")

            if pred is None:
                line = f"Пользователю {user_selected} не рекомендуем {product_selected} (недостаточно данных)"
            else:
                decision = "рекомендуем" if pred >= PRED_THRESHOLD else "не рекомендуем"
                line = f"Пользователю {user_selected} {decision} {product_selected} (прогноз {pred:.3f})"
            rec_summary_lines.append(line)

    # 7) Сводка рекомендаций и блок для новых пользователей
    if rec_summary_lines:
        print("\n" + "="*80)
        print(f"Итоговые решения по порогу прогноза ≥ {PRED_THRESHOLD:.1f}:")
        for ln in rec_summary_lines:
            print(" - " + ln)

    if new_user_lines:
        print("\nРекомендации для новых пользователей (нет ни одной оценки):")
        for ln in new_user_lines:
            print(" - " + ln)
