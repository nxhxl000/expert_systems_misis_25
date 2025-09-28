from pathlib import Path
import csv
import math
from tabulate import tabulate
from typing import List, Tuple

# ---------- вывод исходной таблицы ----------

def render_table(filename: str, delimiter: str = ';', tablefmt: str = 'fancy_grid') -> str:
    """
    Читает CSV (рядом со скриптом) и возвращает строку с таблицей (рамки делает tabulate).
    Ничего НЕ печатает.
    """
    csv_path = Path(__file__).parent / filename
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f, delimiter=delimiter))

    if not rows:
        return "(пустая таблица)"

    headers, data = rows[0], rows[1:]
    return tabulate(data, headers=headers, tablefmt=tablefmt, stralign='center', numalign='center')

def print_section(title: str, body: str, use_ansi_bold: bool = True) -> None:
    """
    Печатает: пустая строка, заголовок, пустая строка, затем body.
    """
    def bold(s: str) -> str:
        return f"\033[1m{s}\033[0m" if use_ansi_bold else s

    print()
    print(bold(title))
    print()
    print(body)

# ---------- чтение данных ----------

def _read_users_products(filename: str, delimiter: str = ';'):
    """
    Возвращает:
      users          — список имён пользователей из заголовка (U1..Un)
      products       — список имён продуктов из первого столбца (P1..Pm)
      ratings_matrix — список списков чисел размера m x n (строки — продукты, столбцы — пользователи)
    """
    path = Path(__file__).parent / filename
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f, delimiter=delimiter))

    if not rows or len(rows[0]) < 2:
        raise ValueError("Некорректный формат CSV: ожидается первая строка ';U1;U2;...' и строки продуктов P1..")

    users = rows[0][1:]
    products = [r[0] for r in rows[1:]]
    ratings_matrix = [[float(x) for x in r[1:]] for r in rows[1:]]
    return users, products, ratings_matrix

# ---------- косинусные вычисления ----------

def _cos_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def compute_users_cosine_similarity(filename: str, delimiter: str = ';') -> Tuple[List[str], List[List[float]]]:
    """
    Возвращает (user_labels, S), где S[i][j] — косинусное сходство пользователей i и j.
    """
    users, _, ratings = _read_users_products(filename, delimiter)
    user_vectors = list(map(list, zip(*ratings))) if ratings else [[]]
    n = len(users)
    S = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            S[i][j] = _cos_sim(user_vectors[i], user_vectors[j])
    return users, S

def compute_products_cosine_similarity(filename: str, delimiter: str = ';') -> Tuple[List[str], List[List[float]]]:
    """
    Возвращает (product_labels, S), где S[p][q] — косинусное сходство продуктов p и q.
    """
    _, products, ratings = _read_users_products(filename, delimiter)
    m = len(products)
    S = [[0.0]*m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            S[i][j] = _cos_sim(ratings[i], ratings[j])
    return products, S

# ---------- представление, сохранение и поиск максимумов ----------

def render_upper_triangle(labels: List[str], S: List[List[float]], ndigits: int = 3, blank: str = '') -> str:
    """
    Рендерит верхнетреугольную часть матрицы S (включая диагональ), ниже диагонали — blank.
    """
    n = len(labels)
    mat = [[blank]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = S[i][j]
            val = 0.0 if abs(val) < 10**(-(ndigits+1)) else round(val, ndigits)
            mat[i][j] = val
    data = [[labels[i], *row] for i, row in enumerate(mat)]
    headers = [""] + labels
    return tabulate(data, headers=headers, tablefmt='fancy_grid', stralign='center', numalign='center')

def save_upper_triangle_csv(labels: List[str], S: List[List[float]], out_name: str,
                            ndigits: int = 3, blank: str = '', delimiter: str = ';') -> Path:
    """
    Сохраняет верхнетреугольную часть матрицы S в CSV рядом со скриптом.
    Возвращает путь к созданному файлу.
    """
    script_dir = Path(__file__).parent
    out_path = script_dir / out_name

    n = len(labels)
    # Заголовок: ;L1;L2;...;Ln
    header = [''] + labels

    # Строки: Label_i; [blank... val(i,i)..val(i,n)]
    rows_out: List[List[str]] = []
    for i in range(n):
        row = [labels[i]]
        for j in range(n):
            if j < i:
                row.append(blank)
            else:
                val = S[i][j]
                v = 0.0 if abs(val) < 10**(-(ndigits+1)) else round(val, ndigits)
                row.append(str(v))
        rows_out.append(row)

    with out_path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(header)
        w.writerows(rows_out)

    return out_path

def find_most_similar_pairs(labels: List[str], S: List[List[float]], ndigits: int = 3) -> List[Tuple[str, str, float]]:
    """
    Ищет максимум S[i][j] для i<j (вне диагонали). Возвращает все пары с максимальным значением.
    """
    n = len(labels)
    best = -1.0
    pairs: List[Tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i+1, n):
            val = S[i][j]
            if val > best + 1e-12:
                best = val
                pairs = [(labels[i], labels[j], val)]
            elif abs(val - best) <= 1e-12:
                pairs.append((labels[i], labels[j], val))
    return [(a, b, round(v, ndigits)) for (a, b, v) in pairs]

# ---------- пример использования ----------

if __name__ == "__main__":
    FILENAME = "data.csv"  # ваш исходный CSV рядом со скриптом

    # 1) Исходные данные
    print_section("Исходные данные", render_table(FILENAME))

    # 2) Пользователи: сходства, вывод, сохранение, лучшие пары
    user_labels, S_users = compute_users_cosine_similarity(FILENAME, delimiter=';')
    users_upper_str = render_upper_triangle(user_labels, S_users, ndigits=3, blank='')
    print_section("Косинусное сходство пользователей (верхнетреугольная матрица)", users_upper_str)

    users_csv_path = save_upper_triangle_csv(user_labels, S_users, "users_similarity_upper.csv",
                                             ndigits=3, blank='', delimiter=';')
    print(f"Файл сохранён: {users_csv_path}")

    best_user_pairs = find_most_similar_pairs(user_labels, S_users, ndigits=3)
    print()
    print("Самые близкие пользователи: " + ", ".join(f"{a}–{b} ({v})" for a, b, v in best_user_pairs))

    # 3) Продукты: сходства, вывод, сохранение, лучшие пары
    product_labels, S_prod = compute_products_cosine_similarity(FILENAME, delimiter=';')
    prod_upper_str = render_upper_triangle(product_labels, S_prod, ndigits=3, blank='')
    print_section("Косинусное сходство продуктов (верхнетреугольная матрица)", prod_upper_str)

    prod_csv_path = save_upper_triangle_csv(product_labels, S_prod, "products_similarity_upper.csv",
                                            ndigits=3, blank='', delimiter=';')
    print(f"Файл сохранён: {prod_csv_path}")

    best_prod_pairs = find_most_similar_pairs(product_labels, S_prod, ndigits=3)
    print()
    print("Самые близкие продукты: " + ", ".join(f"{a}–{b} ({v})" for a, b, v in best_prod_pairs))
