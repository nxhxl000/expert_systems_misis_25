# Коэффициенты
MD_values = [0.8, 0.75, 0.4, 0.85]
MND_values = [0.3, 0.25, 0.5, 0.2]

# правила нечеткой логики
def AND(MD1, MD2):
    return min(MD1, MD2)

def OR(MD1, MD2):
    return max(MD1, MD2)

def NOT(MD1):
    return 1 - MD1

# формула для расчета доверия гипотезе с учетом свидетельств
def MD(H, E):
    if type(H) is list:
        return MD(H[0], H[1]) + MD(H[0], E) * (1 - MD(H[0], H[1]))
    return H + E * (1 - H)

# формула для расчета недоверия гипотезе с учетом свидетельств
def MND(H, E):
    if type(H) is list:
        return MND(H[0], H[1]) + MND(H[0], E) * (1 - MND(H[0], H[1]))
    return H + E * (1 - H)

# формула для расчета итогового коэффициента уверенности (формула Шортлифа)
def KU(H, E):
    return H - E

# Основной блок
def main():
    print("Рассчитаем коэффициенты доверия, недоверия и итоговый коэффициент уверенности для нескольких предположений.")
    print('Исходные данные: ')
    print(f'E1 = Если X проживает в Y (MD1 = {MD_values[0]}, MND1 = {MND_values[0]}) и является членом партии Z (MD2 = {MD_values[1]}, MND2 = {MND_values[1]}), то X будет голосовать за кандидата B')
    print(f'E2 = Если X имеет возраст T (MD3 = {MD_values[2]}, MND1 = {MND_values[2]}) или X является ИП (MD4 = {MD_values[3]}, MND4 = {MND_values[3]}), то X будет голосовать за кандидата B')
    print("Меры доверия и недоверия с учетом свидетельств:")
    md = MD(AND(MD_values[0], MD_values[1]), OR(MD_values[2], MD_values[3]))
    mnd = MND(AND(MND_values[0], MND_values[1]), OR(MND_values[2], MND_values[3]))
    print(f'Меры доверия MD[H:E]: {md}')
    print(f'Меры недоверия MND[H:E]: {mnd}')
    ku = KU(md, mnd)
    print(f'Итоговый коэффициент уверенности (KU): {ku}')
    print("Проверка:")
    if ku > -1 and ku < 1:
        print(f'Расчет произведен верно: KU = {ku} находится в интервале (-1, 1)')
    else:
        print("Расчет произведен с ошибкой")

if __name__ == "__main__":  
    main()
