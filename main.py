# ////// BIBLIOTECAS //////
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ////// BASE DE DADOS //////
df = pd.read_csv("flood_risk_dataset_india.csv")

# ////// Variável Discreta: Infraestrutura //////
infra_freq = df['Infrastructure'].value_counts().sort_index()
print("Frequência - Presença de Infraestrutura:")
print(infra_freq)

# ////// Variável Contínua: Umidade //////
faixas_umidade = pd.cut(df['Humidity (%)'], bins=5)
freq_umidade = faixas_umidade.value_counts().sort_index()
print("\nFrequência - Faixas de Umidade:")
print(freq_umidade)

# ////// Gráfico Histograma de Umidade //////
plt.figure(figsize=(8,6))
sns.histplot(df['Humidity (%)'], bins=20, color='green')
plt.title("Distribuição da Umidade (%)")
plt.xlabel("Umidade (%)")
plt.ylabel("Frequência")
plt.grid(True)
plt.show()

# ////// Gráfico boxplot de Umidade por Ocorrência de Enchente //////
plt.figure(figsize=(8,6))
sns.boxplot(x='Flood Occurred', y='Humidity (%)', data=df, palette='YlGnBu')
plt.title("Umidade vs Ocorrência de Enchente")
plt.xlabel("Enchente Ocorrida")
plt.ylabel("Umidade (%)")
plt.grid(True)
plt.show()

# ////// Estatística Descritiva para Umidade //////
umidade = df['Humidity (%)']
print("\n===== Estatísticas para Umidade (%) =====")
print(f"Média: {umidade.mean():.2f}")
print(f"Mediana: {umidade.median():.2f}")
print(f"Moda: {umidade.mode()[0]:.2f}")
print(f"Mínimo: {umidade.min():.2f}")
print(f"Máximo: {umidade.max():.2f}")
print(f"Amplitude: {(umidade.max() - umidade.min()):.2f}")
print(f"Variância: {umidade.var():.2f}")
print(f"Desvio Padrão: {umidade.std():.2f}")
print(f"Coeficiente de Variação: {(umidade.std() / umidade.mean()) * 100:.2f}%")
print(f"Quartis:\n{umidade.quantile([0.25, 0.5, 0.75])}")

# ////// Estatística Descritiva para Infraestrutura //////
infra = df['Infrastructure']
print("\n===== Estatísticas para Infraestrutura =====")
print(f"Total com Infraestrutura: {(infra == 1).sum()}")
print(f"Total sem Infraestrutura: {(infra == 0).sum()}")
print(f"Proporção com Infraestrutura: {(infra.mean()) * 100:.2f}%")

# ////// Regressão: Umidade (%) para prever Nível da Água (m) //////
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = df[['Humidity (%)']]  # variável independente
y = df['Water Level (m)']  # variável dependente

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("\n===== Regressão Linear: Umidade vs Nível da Água =====")
print(f"Coeficiente Angular (a): {modelo.coef_[0]:.4f}")
print(f"Coeficiente Linear (b): {modelo.intercept_:.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"Erro Quadrático Médio (MSE): {mean_squared_error(y_test, y_pred):.4f}")

plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='green', label='Dados reais')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regressão linear')
plt.title("Previsão do Nível da Água pela Umidade")
plt.xlabel("Umidade (%)")
plt.ylabel("Nível da Água (m)")
plt.legend()
plt.grid(True)
plt.show()
