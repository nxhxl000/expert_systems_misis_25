from pathlib import Path
import csv
from typing import Sequence, Optional, List, Tuple
from tabulate import tabulate

# --------- настройки ---------
ONE_BASED = False  # True -> 1-индексация для Col/Column

# --------- утилиты вывода ---------

def print_section(title: str, body: str, use_ansi_bold: bool = True) -> None:
    """Печатает: пустая строка, заголовок, пустая строка, затем body."""
    def bold(s: str) -> str:
        return f"\033[1m{s}\033[0m" if use_ansi_bold else s
    print()
    print(bold(title))
    print()
    print(body)

# --------- чтение и рендер исходной таблицы ---------

def _guess_delimiter(sample_lines: Sequence[str]) -> str:
    cands = [';', ',', '\t']
    counts = {d: sum(line.count(d) for line in sample_lines) for d in cands}
    return max(counts, key=counts.get)

def _as_display_cell(raw: str) -> str:
    s = raw.strip().strip('"').strip("'")
    if s == "":
        return ""
    s_norm = s.replace(",", ".")
    try:
        return "" if float(s_norm) == 0.0 else s
    except ValueError:
        return s

def render_matrix_csv(filename: str,
                      delimiter: Optional[str] = None,
                      tablefmt: str = 'fancy_grid') -> str:
    csv_path = Path(__file__).parent / filename
    text = csv_path.read_text(encoding="utf-8-sig").splitlines()
    if not text:
        return "(пустая таблица)"
    if delimiter is None:
        delimiter = _guess_delimiter(text[:10])
    rows = [[_as_display_cell(cell) for cell in row]
            for row in csv.reader(text, delimiter=delimiter)]
    if not rows:
        return "(пустая таблица)"
    return tabulate(rows, tablefmt=tablefmt, stralign='center', numalign='center')

# --------- парсинг в числовую матрицу и представления ---------

def _to_number(cell: str) -> float:
    s = cell.strip().strip('"').strip("'")
    if s == "":
        return 0.0
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0

def read_matrix_numeric(filename: str, delimiter: Optional[str] = None) -> List[List[float]]:
    csv_path = Path(__file__).parent / filename
    text_lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    if not text_lines:
        return []
    if delimiter is None:
        delimiter = _guess_delimiter(text_lines[:10])
    rows_raw = list(csv.reader(text_lines, delimiter=delimiter))
    return [[_to_number(c) for c in row] for row in rows_raw]

def _pretty_num(x: float):
    xi = int(round(x))
    return xi if abs(x - xi) < 1e-12 else x

def build_triplets(mat: List[List[float]], one_based: bool = False) -> Tuple[list, list, list]:
    V, R, C = [], [], []
    for r, row in enumerate(mat):
        for c, val in enumerate(row):
            if val != 0:
                V.append(_pretty_num(val))
                R.append(r + (1 if one_based else 0))
                C.append(c + (1 if one_based else 0))
    return V, R, C

def build_csr(mat: List[List[float]], one_based_cols: bool = False) -> Tuple[list, list, list]:
    Value, Col, RowIndex = [], [], [0]
    nnz = 0
    for row in mat:
        for c, val in enumerate(row):
            if val != 0:
                Value.append(_pretty_num(val))
                Col.append(c + (1 if one_based_cols else 0))
                nnz += 1
        RowIndex.append(nnz)
    return Value, Col, RowIndex

# --------- ELPACK (ELLPACK/ITPACK) ---------

def build_ellpack(mat: List[List[float]], one_based_cols: bool = False) -> Tuple[List[List], List[List]]:
    n_rows = len(mat)
    nnz_per_row = [sum(1 for x in row if x != 0) for row in mat]
    K = max(nnz_per_row, default=0)

    values_ell: List[List] = []
    cols_ell: List[List] = []

    for r, row in enumerate(mat):
        v_row, c_row = [], []
        for c, val in enumerate(row):
            if val != 0:
                v_row.append(_pretty_num(val))
                c_row.append(c + (1 if one_based_cols else 0))
        # добиваем нулями до K
        v_row += [0] * (K - len(v_row))
        c_row += [0] * (K - len(c_row))
        values_ell.append(v_row)
        cols_ell.append(c_row)

    return values_ell, cols_ell  # размеры: n_rows × K

def render_grid(rows: List[List], title: str) -> str:
    return f"{title}\n" + tabulate(rows, tablefmt='fancy_grid', stralign='center', numalign='center')

# ================== ПРЕДПОЧТЕНИЯ ==================

def _is_numberish(s: str) -> bool:
    s2 = s.strip().replace(",", ".")
    if s2 == "":
        return True
    try:
        float(s2)
        return True
    except ValueError:
        return False

def parse_preference_csv(filename: str, delimiter: Optional[str] = None):
    """
    Возвращает (headers, row_labels, data).
    Подписи из CSV допускаются, но далее мы будем использовать собственные P*/U*.
    """
    csv_path = Path(__file__).parent / filename
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    if delimiter is None and lines:
        delimiter = _guess_delimiter(lines[:10])

    rows = list(csv.reader(lines, delimiter=delimiter)) if lines else []
    if not rows:
        return None, None, []

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

def row_mean_ignore_zeros(row: List[float]) -> float:
    nz = [x for x in row if x != 0]
    return sum(nz) / len(nz) if nz else float("nan")

# ---------- генерация имён преобразованной матрицы ----------

def gen_row_names(n_rows: int) -> List[str]:
    return [f"P{i+1}" for i in range(n_rows)]

def gen_col_names(n_cols: int) -> List[str]:
    return [f"U{j+1}" for j in range(n_cols)]

# ---------- форматирование чисел в логах ----------

def _fmt_num(x: float) -> str:
    if x != x:  # NaN
        return "NaN"
    xi = int(round(x))
    return str(xi) if abs(x - xi) < 1e-12 else f"{x:.3f}"

# ---------- удаление нулевых строк и столбцов с логами ----------

def drop_all_zero_rows_and_cols_pref_with_logs(
    data: List[List[float]]
) -> Tuple[List[List[float]], List[str], List[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
    if not data:
        return [], [], [], [], []

    n_rows0 = len(data)
    n_cols0 = len(data[0]) if data else 0
    row_names_all = gen_row_names(n_rows0)
    col_names_all = gen_col_names(n_cols0)

    # 1) строки = все нули
    zero_rows = [i for i, row in enumerate(data) if all(v == 0 for v in row)]
    removed_rows_log = [(row_names_all[i], "все оценки = 0 (удалено на шаге очистки нулевых строк)")
                        for i in zero_rows]
    keep_rows = [i for i in range(len(data)) if i not in zero_rows]
    data = [data[i] for i in keep_rows]
    row_names = [row_names_all[i] for i in keep_rows]

    # 2) столбцы = все нули (после удаления нулевых строк)
    if data:
        n_cols = len(data[0])
        zero_cols = []
        keep_cols = []
        for j in range(n_cols):
            col_nonzero = any(data[i][j] != 0 for i in range(len(data)))
            if col_nonzero:
                keep_cols.append(j)
            else:
                zero_cols.append(j)

        removed_cols_log = [(col_names_all[j], "все оценки = 0 (удалено на шаге очистки нулевых столбцов)")
                            for j in zero_cols]

        data = [[row[j] for j in keep_cols] for row in data]
        col_names = [col_names_all[j] for j in keep_cols]
    else:
        removed_cols_log = [(name, "все оценки = 0 (удалено на шаге очистки нулевых столбцов)") for name in col_names_all]
        col_names = []

    return data, row_names, col_names, removed_rows_log, removed_cols_log

# ---------- фильтр строк по среднему с логами ----------

def filter_rows_by_mean_with_logs(
    data: List[List[float]],
    row_names: List[str],
    R_thresh: float
) -> Tuple[List[List[float]], List[str], List[Tuple[str, str]]]:
    removed_rows_log: List[Tuple[str, str]] = []
    keep = []

    for i, row in enumerate(data):
        m = row_mean_ignore_zeros(row)
        if m == m and m >= R_thresh:
            keep.append(i)
        else:
            if m != m:
                reason = "среднее = NaN (нет ненулевых оценок)"
            else:
                reason = f"среднее = {_fmt_num(m)} < R={_fmt_num(R_thresh)}"
            removed_rows_log.append((row_names[i], f"{reason} (удалено на шаге фильтра по R)"))

    data = [data[i] for i in keep]
    row_names = [row_names[i] for i in keep]
    return data, row_names, removed_rows_log

# ---------- рендер преобразованной матрицы ----------

def to_display_with_PU_headers(row_names: List[str], col_names: List[str], data: List[List[float]]) -> str:
    # округлим «почти целые» для красоты
    body = [[(int(v) if abs(v-round(v)) < 1e-12 else v) for v in row] for row in data]
    # подставим имена строк
    body = [[row_names[i]] + body[i] for i in range(len(body))]
    headers = [""] + col_names
    return tabulate(body, headers=headers, tablefmt='fancy_grid', stralign='center', numalign='center')

def render_pref_table(filename: str) -> str:
    csv_path = Path(__file__).parent / filename
    text = csv_path.read_text(encoding="utf-8-sig").splitlines()
    if not text:
        return "(пустая таблица)"
    delimiter = _guess_delimiter(text[:10])
    rows = list(csv.reader(text, delimiter=delimiter))
    return tabulate(rows, tablefmt='fancy_grid', stralign='center', numalign='center')

# --------- main ---------

if __name__ == "__main__":
    FILENAME = "sparse_matrix.csv"

    # 1) Исходные данные
    print_section("Исходные данные: ", render_matrix_csv(FILENAME))

    # 2) Векторное представление
    mat = read_matrix_numeric(FILENAME)
    V, R, C = build_triplets(mat, one_based=ONE_BASED)
    print_section(
        f"Три вектора (индексация: {'1-based' if ONE_BASED else '0-based'})",
        "\n".join([f"Value: {V}", f"Row:   {R}", f"Col:   {C}"])
    )

    # 3) CSR (Value, Col, RowIndex)
    Value, Col, RowIndex = build_csr(mat, one_based_cols=ONE_BASED)
    body_csr = "\n".join([f"Value: {Value}", f"Col:   {Col}", f"RowIndex: {RowIndex}"])
    print_section("CSR-представление", body_csr)

    # 4) ELPACK (Elpac)
    val_ell, col_ell = build_ellpack(mat, one_based_cols=ONE_BASED)
    body_ell = render_grid(val_ell, "Value") + "\n\n" + render_grid(col_ell, "Column")
    print_section("Elpac (ELLPACK) представление", body_ell)

    # ===== Матрица предпочтений =====
    PREF_FILE = "preference_matrix.csv"
    R_THRESH = 4

    # Исходная матрица предпочтений
    print_section("Исходная матрица предпочтений", render_pref_table(PREF_FILE))

    # Порог R, с которым сравниваем средние по продукту
    print_section("Порог для среднего R", f"R = {R_THRESH}")

    # Парсинг
    _, _, data = parse_preference_csv(PREF_FILE)

    # Шаг 1: удаление полностью нулевые строки и столбцы с логами
    data, row_names, col_names, removed_rows_log_1, removed_cols_log = drop_all_zero_rows_and_cols_pref_with_logs(data)

    # Шаг 2: фильтрация строк по среднему >= R с логами
    data, row_names, removed_rows_log_2 = filter_rows_by_mean_with_logs(data, row_names, R_THRESH)

    # Объединённые логи по строкам
    removed_rows_log = removed_rows_log_1 + removed_rows_log_2

    # Вывод удаленных строк
    if removed_rows_log:
        rows_report = tabulate(
            [[name, reason] for name, reason in removed_rows_log],
            headers=["Строка (P)", "Причина удаления"],
            tablefmt='fancy_grid',
            stralign='left'
        )
    else:
        rows_report = "— (ничего не удалено)"

    print_section("Удалённые строки и причины", rows_report)

    # Вывод удаленных столбцов
    if removed_cols_log:
        cols_report = tabulate(
            [[name, reason] for name, reason in removed_cols_log],
            headers=["Столбец (U)", "Причина удаления"],
            tablefmt='fancy_grid',
            stralign='left'
        )
    else:
        cols_report = "— (ничего не удалено)"
    print_section("Удалённые столбцы и причины", cols_report)

    # Вывод преобразованной матрицы 
    print_section("Преобразованная матрица предпочтений",
                  to_display_with_PU_headers(row_names, col_names, data))
