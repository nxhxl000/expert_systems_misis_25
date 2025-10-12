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
    """
    Возвращает матрицы (n_rows × K): Value_ell, Column_ell,
    где K = max(nnz в строке). Недостающие позиции заполняются нулями.
    """
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
    Заголовок — если в первой строке есть нечисловые значения.
    Первый столбец — метки строк, если в нём есть нечисловые значения.
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

def drop_all_zero_rows_and_cols_pref(data: List[List[float]],
                                     headers: Optional[List[str]],
                                     row_labels: Optional[List[str]]):
    keep_rows = [i for i, row in enumerate(data) if any(v != 0 for v in row)]
    data = [data[i] for i in keep_rows]
    if row_labels is not None:
        row_labels = [row_labels[i] for i in keep_rows]

    if data:
        n_cols = len(data[0])
        keep_cols = [j for j in range(n_cols) if any(data[i][j] != 0 for i in range(len(data)))]
        data = [[row[j] for j in keep_cols] for row in data]
        if headers is not None:
            headers = [headers[j] for j in keep_cols]
    return data, headers, row_labels

def row_mean_ignore_zeros(row: List[float]) -> float:
    nz = [x for x in row if x != 0]
    return sum(nz) / len(nz) if nz else float("nan")

def filter_rows_by_mean(data: List[List[float]],
                        row_labels: Optional[List[str]],
                        R_thresh: float):
    """
    Оставляет ТОЛЬКО те строки, у которых средняя (без нулей) >= R_thresh.
    Никаких жёстко заданных имён строк.
    """
    keep = []
    for i, row in enumerate(data):
        m = row_mean_ignore_zeros(row)
        if m == m and m >= R_thresh:   # m==m => не NaN
            keep.append(i)
    data = [data[i] for i in keep]
    if row_labels is not None:
        row_labels = [row_labels[i] for i in keep]
    return data, row_labels

def render_pref_table(filename: str) -> str:
    """Исходная матрица предпочтений: нули НЕ скрываем."""
    csv_path = Path(__file__).parent / filename
    text = csv_path.read_text(encoding="utf-8-sig").splitlines()
    if not text:
        return "(пустая таблица)"
    delimiter = _guess_delimiter(text[:10])
    rows = list(csv.reader(text, delimiter=delimiter))
    return tabulate(rows, tablefmt='fancy_grid', stralign='center', numalign='center')

def to_display_with_headers(headers: Optional[List[str]],
                            row_labels: Optional[List[str]],
                            data: List[List[float]]) -> str:
    body = [[(int(v) if abs(v-round(v)) < 1e-12 else v) for v in row] for row in data]
    if row_labels is not None:
        body = [[row_labels[i]] + body[i] for i in range(len(body))]
    if headers is not None:
        hdr = ([""] if row_labels is not None else []) + headers
        return tabulate(body, headers=hdr, tablefmt='fancy_grid', stralign='center', numalign='center')
    return tabulate(body, tablefmt='fancy_grid', stralign='center', numalign='center')

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

    # Парсинг и подготовка
    headers, row_labels, data = parse_preference_csv(PREF_FILE)

    # убрать полностью нулевые строки и столбцы — до расчёта средних
    data, headers, row_labels = drop_all_zero_rows_and_cols_pref(data, headers, row_labels)

    # оставить только строки, где средняя без нулей >= R
    data, row_labels = filter_rows_by_mean(data, row_labels, R_THRESH)

    # Вывод преобразованной матрицы
    print_section("Преобразованная матрица предпочтений",
                  to_display_with_headers(headers, row_labels, data))
