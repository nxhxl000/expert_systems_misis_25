
from pathlib import Path
import csv
from tabulate import tabulate


DATA_DIR = (Path(__file__).parent.parent / "PZ1").resolve()

# файлы
SOURCE_FILE   = "data.csv"                        # исходные данные
USERS_FILE    = "users_similarity_upper.csv"      # верхнетреугольная матрица пользователей
PRODUCTS_FILE = "products_similarity_upper.csv"   # верхнетреугольная матрица продуктов
DELIM = ';'
NDIGITS = 3

# ----- мини-утилиты для верхнетреугольной матрицы -----
def get_upper(S, i, j):
    """Вернуть S[i][j] для i<j; для i>=j вернуть None (диагональ/низ не используются)."""
    if i >= j:
        return None
    return S[i][j]

def set_upper(S, i, j, value):
    """Положить значение в верхний треугольник (i<j)."""
    if i < j:
        S[i][j] = value

# ----- ввод порога R -----
def ask_R_named(label: str):
    """
    Запрашивает у пользователя CLUSTER_SIZE (R) для заданной группы.
    label: строка для подсказки, например 'пользователей' или 'продуктов'.
    """
    while True:
        raw = input(f"Введите CLUSTER_SIZE (R) для {label} в диапазоне [0..1]: ").strip().replace(',', '.')
        try:
            R = float(raw)
            if 0.0 <= R <= 1.0:
                return R
        except:
            pass
        print("Некорректное значение. Пример: 0.90")

# ----- чтение верхнетреугольной матрицы из CSV -----
def read_upper_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f, delimiter=DELIM))
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"Неверный формат CSV: {path}")
    labels = rows[0][1:]
    n = len(labels)
    S = [[None]*n for _ in range(n)]
    for i in range(1, len(rows)):
        vals = rows[i][1:]
        r = i - 1
        for j, cell in enumerate(vals):
            if not cell or cell.strip() == '':
                continue
            if r < j:
                set_upper(S, r, j, float(cell.replace(',', '.')))
    return labels, S

# ----- печать текущей верхнетреугольной матрицы -----
def print_upper(labels, S, title):
    n = len(labels)
    view = []
    for i in range(n):
        row = [''] * n
        for j in range(i+1, n):
            v = S[i][j]
            if v is not None:
                row[j] = 0.0 if abs(v) < 10**(-(NDIGITS+1)) else round(v, NDIGITS)
        view.append([labels[i], *row])
    headers = [''] + labels
    print("\n\033[1m" + title + "\033[0m\n")
    print(tabulate(view, headers=headers, tablefmt='fancy_grid', stralign='center', numalign='center'))

# ----- поиск пары с максимумом в верхнем треугольнике -----
def argmax_offdiag(S):
    n = len(S)
    best = float('-inf')
    pair = None
    for i in range(n):
        for j in range(i+1, n):
            v = S[i][j]
            if v is None:
                continue
            if v > best:
                best = v
                pair = (i, j)
    return pair, best

# ----- одно слияние: single-link по сходству, новый кластер добавляется в КОНЕЦ -----
def merge_once(labels, S, i, j):
    """
    Объединяет i и j -> новый кластер "(li,lj)".
    Новые сходства: max( S[i,k], S[j,k] ) для каждого оставшегося k.
    Возвращает labels', S' размера (n-1) с новым кластером в конце, и имя кластера.
    """
    li, lj = labels[i], labels[j]
    new_label = f"({li},{lj})"
    n = len(labels)
    keep = [k for k in range(n) if k not in (i, j)]

    m = n - 1
    S_new = [[None]*m for _ in range(m)]
    labels_new = [labels[k] for k in keep] + [new_label]
    new_idx = m - 1

    # 1) переносим старые связи между keep
    for a_pos, a in enumerate(keep):
        for b_pos in range(a_pos+1, len(keep)):
            b = keep[b_pos]
            S_new[a_pos][b_pos] = get_upper(S, min(a, b), max(a, b))

    # 2) связи нового кластера с каждым из keep
    for k_pos, k in enumerate(keep):
        v1 = get_upper(S, min(i, k), max(i, k))
        v2 = get_upper(S, min(j, k), max(j, k))
        if v1 is None and v2 is None:
            sim = None
        elif v1 is None:
            sim = v2
        elif v2 is None:
            sim = v1
        else:
            sim = max(v1, v2)
        S_new[min(k_pos, new_idx)][max(k_pos, new_idx)] = sim

    return labels_new, S_new, new_label

# ----- единый прогон кластеризации для «любой» матрицы -----
def run_clustering(labels, S, R, header_title):
    """
    header_title: заголовок таблицы («Косинусное сходство ... (верхнетреугольная, без диагонали)»).
    """
    print_upper(labels, S, header_title)
    print(f"\nИспользуем R = {round(R, NDIGITS)}")

    step = 1
    cluster_order = []  # лейблы кластеров в порядке образования (для красивого порядка колонок в печати)

    while True:
        # максимум и сравнение с R
        pair, best = argmax_offdiag(S)
        best_show = round(best, NDIGITS) if pair else "—"
        if (pair is None) or (best is None) or (best < R) or (len(labels) <= 1):
            print(f"\nПорог достигнут или больше нет допустимых пар: максимум={best_show} ; R={round(R, NDIGITS)}. Остановка.")
            print(f"Текущие кластеры: {', '.join(labels)}")
            break

        i, j = pair
        li, lj = labels[i], labels[j]
        print(f"\nПара максимума: {li} – {lj} (cos={round(best, NDIGITS)})")
        print(f"Шаг {step}. Перестройка матрицы расстояний: {li} и {lj} объединены по расстоянию {round(best, NDIGITS)} > R={round(R, NDIGITS)}")

        # слияние
        labels, S, new_cluster = merge_once(labels, S, i, j)
        cluster_order.append(new_cluster)

        # печать: кластеры — первыми (стабильно), потом одиночки (только для ВЫВОДА)
        idx_by_label = {lab: idx for idx, lab in enumerate(labels)}
        front = [idx_by_label[c] for c in cluster_order if c in idx_by_label]
        rest  = [idx for idx, lab in enumerate(labels) if lab not in set(cluster_order)]
        order = front + rest

        n = len(labels)
        view = []
        hdrs = [''] + [labels[k] for k in order]
        for a_pos, a in enumerate(order):
            row = [''] * n
            for b_rel in range(a_pos+1, n):
                b = order[b_rel]
                v = get_upper(S, min(a, b), max(a, b))
                if v is not None:
                    row[b_rel] = 0.0 if abs(v) < 10**(-(NDIGITS+1)) else round(v, NDIGITS)
            view.append([labels[a], *row])

        print("\n\033[1mМатрица после шага " + str(step) + "\033[0m\n")
        print(tabulate(view, headers=hdrs, tablefmt='fancy_grid', stralign='center', numalign='center'))

        step += 1


if __name__ == "__main__":
    print("\n\033[1mАгломеративный алгоритм кластеризации\033[0m")


    src = (DATA_DIR / SOURCE_FILE)
    if src.exists():
        with src.open('r', encoding='utf-8-sig', newline='') as f:
            rows = list(csv.reader(f, delimiter=DELIM))
        print("\n\033[1mИсходные данные\033[0m\n")
        print(tabulate(rows[1:], headers=rows[0], tablefmt='fancy_grid', stralign='center', numalign='center'))

    # --- пороги отдельно ---
    R_users    = ask_R_named('пользователей')
    R_products = ask_R_named('продуктов')

    # --- 1) Пользователи ---
    try:
        u_labels, u_S = read_upper_csv(DATA_DIR / USERS_FILE)
        run_clustering(
            labels=u_labels,
            S=u_S,
            R=R_users,
            header_title="Косинусное сходство пользователей (верхнетреугольная, без диагонали)",
        )
    except Exception as e:
        print(f"\n(предупреждение) Не удалось обработать пользователей: {e}")

    # --- 2) Продукты ---
    try:
        p_labels, p_S = read_upper_csv(DATA_DIR / PRODUCTS_FILE)
        run_clustering(
            labels=p_labels,
            S=p_S,
            R=R_products,
            header_title="Косинусное сходство продуктов (верхнетреугольная, без диагонали)",
        )
    except Exception as e:
        print(f"\n(предупреждение) Не удалось обработать продукты: {e}")
