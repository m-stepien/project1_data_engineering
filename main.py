from dataload import load_from_file

df = load_from_file("resource/data.csv", {"sep": ",", "usecols": [0]})
print(df)