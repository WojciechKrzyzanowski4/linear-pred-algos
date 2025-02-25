import pandas as pd


data = {
    'cena': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    'popyt': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98],
    'wynik': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147]
}
df = pd.DataFrame(data)

keys = []
for key in data.keys():
    keys.append(key)

# współczynniki korelacji Pearsona
correlation_1_3 = df[keys[0]].corr(df[keys[2]])
correlation_2_3 = df[keys[1]].corr(df[keys[2]])
correlation_1_2 = df[keys[0]].corr(df[keys[1]])

print(f"Korelacja między {keys[0]} a {keys[2]}: {correlation_1_3}")
print(f"Korelacja między {keys[1]} a {keys[2]}: {correlation_2_3}")
print(f"Korelacja między {keys[0]} a {keys[1]}: {correlation_1_2}")