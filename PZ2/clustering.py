# clustering_users.py  (лежит в PZ2/)
from pathlib import Path
import csv
from tabulate import tabulate

# === где искать данные ===
DATA_DIR = (Path(__file__).parent.parent / "PZ1").resolve()

SOURCE_FILE   = "data.csv"                      # для вывода "Исходные данные"
USERS_FILE    = "users_similarity_upper.csv"    # уже сохранённая верхняя матрица
DELIM = ';'

NDIGITS   = 3
THRESHOLD = 0.94  # порог косинусного сходства для остановки

# ---------- утилиты вывода ----------
def render_table_from_dir(filename: str, delimiter: str = ';', tablefmt: str = 'fancy_grid') -> str:
    """Читает CSV из DATA_DIR и возвращает строку с таблицей (tabulate)."""
    path = (DATA_DIR / filename).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f, delimiter=delimiter))
    if not rows:
        return "(пустая таблица)"
    headers, data = rows[0], rows[1:]
    return tabulate(data, headers=headers, tablefmt=tablefmt, stralign='center', numalign='center')

def print_section(title: str, body: str, bold: bool = True) -> None:
    print()
    print(f"\033[1m{title}\033[0m" if bold else title)
    print()
    print(body)

def render_upper_triangle(labels, S, ndigits=3, blank=''):
    """Рендерит верхнетреугольную часть S (включая диагональ)."""
    n = len(labels)
    mat = [[blank]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            v = S[i][j]
            v = 0.0 if abs(v) < 10**(-(ndigits+1)) else round(v, ndigits)
            mat[i][j] = v
    data = [[labels[i], *row] for i, row in enumerate(mat)]
    headers = [''] + labels
    return tabulate(data, headers=headers, tablefmt='fancy_grid',
                    stralign='center', numalign='center')

# ---------- чтение сохранённой верхней матрицы и восстановление полной ----------
def read_users_upper_similarity():
    path = (DATA_DIR / USERS_FILE).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f, delimiter=DELIM))
    labels = rows[0][1:]
    n = len(labels)
    # восстановим полную S
    S = [[0.0]*n for _ in range(n)]
    for i in range(n):
        S[i][i] = 1.0
    for i in range(1, len(rows)):
        vals = rows[i][1:]
        for j, cell in enumerate(vals):
            if cell == '' or cell is None:
                continue
            S[i-1][j] = float(cell)
    # симметрия вниз
    for i in range(n):
        for j in range(i):
            S[i][j] = S[j][i]
    return labels, S

# ---------- поиск пары с максимальным сходством (вне диагонали) ----------
def argmax_offdiag(S):
    n = len(S)
    best = -1.0
    best_pair = None
    for i in range(n):
        for j in range(i+1, n):
            v = S[i][j]
            if v > best:
                best = v
                best_pair = (i, j)
    return best_pair, best

# ---------- слияние кластеров по правилу max (single-link по сходству) ----------
def merge_max(labels, S, i, j):
    new_label = f"({labels[i]},{labels[j]})"
    n = len(S)
    keep = [k for k in range(n) if k not in (i, j)]

    # сходства нового кластера с оставшимися: max(S[i][k], S[j][k])
    new_sim = [max(S[i][k], S[j][k]) for k in keep]

    # собрать новую матрицу (n-1 x n-1)
    S_new = []
    new_labels = []
    for a in keep:
        row = [S[a][b] for b in keep]
        S_new.append(row)
        new_labels.append(labels[a])

    # добавить новый кластер (последняя строка/колонка)
    for r, v in zip(S_new, new_sim):
        r.append(v)
    S_new.append(new_sim + [1.0])
    new_labels.append(new_label)
    return new_labels, S_new

# ---------- основной сценарий ----------
if __name__ == "__main__":
    # Шаг 0: Исходные данные (для удобства)
    print_section("Исходные данные", render_table_from_dir(SOURCE_FILE, delimiter=DELIM))

    # Шаг 1: читаем сохранённую матрицу пользователей и показываем её
    labels, S = read_users_upper_similarity()
    print_section("Косинусное сходство пользователей (исходная верхнетреугольная матрица)",
                  render_upper_triangle(labels, S, ndigits=NDIGITS, blank=''))

    # Пошаговая кластеризация
    step = 1
    while True:
        pair, best = argmax_offdiag(S)
        if pair is None or best < THRESHOLD or len(labels) <= 1:
            print()
            print(f"Порог достигнут или больше нет пар: максимум={round(best, NDIGITS) if pair else '—'}; "
                  f"порог={THRESHOLD}. Остановка.")
            print(f"Текущие кластеры: {', '.join(labels)}")
            break

        i, j = pair
        print(f"\nОбъединяем шаг {step}: {labels[i]}–{labels[j]} (cos={round(best, NDIGITS)})")
        labels, S = merge_max(labels, S, i, j)

        print_section(f"Матрица после шага {step}",
                      render_upper_triangle(labels, S, ndigits=NDIGITS, blank=''))
        step += 1
