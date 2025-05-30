# === IMPORTAÇÃO DE BIBLIOTECAS ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === CARREGAMENTO DA BASE DE DADOS ===
df = pd.read_csv("flood_risk_dataset_india.csv")

# === FREQUÊNCIA - Variável Quantitativa Discreta ===
freq_floods = df['Historical Floods'].value_counts().sort_index()
print("Frequência - Histórico de Enchentes:")
print(freq_floods)

# === FREQUÊNCIA - Variável Quantitativa Contínua ===
faixas_chuva = pd.cut(df['Rainfall (mm)'], bins=5)
freq_chuva = faixas_chuva.value_counts().sort_index()
print("\nFrequência - Faixas de Chuva:")
print(freq_chuva)

# === GRÁFICO 1 - Histograma de Chuva ===
plt.figure(figsize=(8,6))
sns.histplot(df['Rainfall (mm)'], bins=20, color='blue')
plt.title("Distribuição da Chuva (mm)")
plt.xlabel("Chuva (mm)")
plt.ylabel("Frequência")
plt.grid(True)
plt.show()

# === GRÁFICO 2 - Boxplot de Temperatura por Ocorrência de Enchente ===
plt.figure(figsize=(8,6))
sns.boxplot(x='Flood Occurred', y='Temperature (°C)', data=df, palette='coolwarm')
plt.title("Temperatura vs Ocorrência de Enchente")
plt.xlabel("Enchente Ocorrida")
plt.ylabel("Temperatura (°C)")
plt.grid(True)
plt.show()

# === Estatística Descritiva para 'Rainfall (mm)' ===
chuva = df['Rainfall (mm)']
print("\n===== Estatísticas para Chuva (mm) =====")
print(f"Média: {chuva.mean():.2f}")
print(f"Mediana: {chuva.median():.2f}")
print(f"Moda: {chuva.mode()[0]:.2f}")
print(f"Mínimo: {chuva.min():.2f}")
print(f"Máximo: {chuva.max():.2f}")
print(f"Amplitude: {(chuva.max() - chuva.min()):.2f}")
print(f"Variância: {chuva.var():.2f}")
print(f"Desvio Padrão: {chuva.std():.2f}")
print(f"Coeficiente de Variação: {(chuva.std() / chuva.mean()) * 100:.2f}%")
print(f"Quartis:\n{chuva.quantile([0.25, 0.5, 0.75])}")

# === Estatística Descritiva para 'Temperature (°C)' ===
temp = df['Temperature (°C)']
print("\n===== Estatísticas para Temperatura (°C) =====")
print(f"Média: {temp.mean():.2f}")
print(f"Mediana: {temp.median():.2f}")
print(f"Moda: {temp.mode()[0]:.2f}")
print(f"Mínimo: {temp.min():.2f}")
print(f"Máximo: {temp.max():.2f}")
print(f"Amplitude: {(temp.max() - temp.min()):.2f}")
print(f"Variância: {temp.var():.2f}")
print(f"Desvio Padrão: {temp.std():.2f}")
print(f"Coeficiente de Variação: {(temp.std() / temp.mean()) * 100:.2f}%")
print(f"Quartis:\n{temp.quantile([0.25, 0.5, 0.75])}")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# === Variáveis: Rainfall para prever Water Level ===
X = df[['Rainfall (mm)']]  # variável independente (input)
y = df['Water Level (m)']  # variável dependente (output)

# === Dividir em treino e teste ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Modelo de Regressão ===
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# === Previsão ===
y_pred = modelo.predict(X_test)

# === Métricas de Avaliação ===
print("\n===== Regressão Linear: Rainfall vs Water Level =====")
print(f"Coeficiente Angular (a): {modelo.coef_[0]:.4f}")
print(f"Coeficiente Linear (b): {modelo.intercept_:.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"Erro Quadrático Médio (MSE): {mean_squared_error(y_test, y_pred):.4f}")

# === Plot da regressão ===
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label='Dados reais')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regressão linear')
plt.title("Previsão do Nível da Água pela Chuva")
plt.xlabel("Chuva (mm)")
plt.ylabel("Nível da Água (m)")
plt.legend()
plt.grid(True)
plt.show()
