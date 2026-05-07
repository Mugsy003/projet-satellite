import pandas as pd
df = pd.read_csv(r"C:\Users\a951444\Workspace\projet-satellite\Outputs\Validation_Saisonniere_LST_2526.csv")
print(df[df['Site'] == 'Spain'].to_string())
