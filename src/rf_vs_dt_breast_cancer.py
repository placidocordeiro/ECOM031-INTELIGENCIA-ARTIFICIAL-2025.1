import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# --- 1. Carregando Dataset Real e Desafiador ---
print("--- 1. Carregando Dataset Breast Cancer Wisconsin ---")

# Carrega o dataset de câncer de mama (bom equilíbrio de dificuldade)
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print(f"Características: {X.shape[1]} features médicas")
print(f"Amostras: {X.shape[0]} pacientes")
print(f"Distribuição das classes: {np.unique(y, return_counts=True)}")
print(f"Nomes das classes: {cancer.target_names}")
print("Classes: 0 = Maligno, 1 = Benigno")

# --- 2. Preparação para Múltiplas Execuções ---
N_EXECUCOES = 20
print(f"\n--- 2. Preparação: Rodando {N_EXECUCOES} experimentos ---")

# Listas para guardar os resultados de cada execução
acc_dt_treino_lista = []
acc_dt_teste_lista = []
acc_rf_treino_lista = []
acc_rf_teste_lista = []

for i in range(N_EXECUCOES):
    # Split aleatório dos dados
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25)

    # --- 3. Modelo 1: Árvore de Decisão (Overfitting) ---
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_treino, y_treino)
    
    # Avaliando a Árvore
    acc_dt_treino = accuracy_score(y_treino, dt_model.predict(X_treino))
    acc_dt_teste = accuracy_score(y_teste, dt_model.predict(X_teste))
    
    # Guarda os resultados
    acc_dt_treino_lista.append(acc_dt_treino)
    acc_dt_teste_lista.append(acc_dt_teste)

    # --- 4. Modelo 2: Random Forest (Estável) ---
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_treino, y_treino)
    
    # Avaliando a Floresta
    acc_rf_treino = accuracy_score(y_treino, rf_model.predict(X_treino))
    acc_rf_teste = accuracy_score(y_teste, rf_model.predict(X_teste))
    
    # Guarda os resultados
    acc_rf_treino_lista.append(acc_rf_treino)
    acc_rf_teste_lista.append(acc_rf_teste)

print("... Experimentos concluídos.")

# --- 5. Comparação Final (Médias) ---
print("\n--- 5. Comparação Final (Médias de 20 Execuções) ---")

# Calculando as médias
media_dt_treino = np.mean(acc_dt_treino_lista)
media_dt_teste = np.mean(acc_dt_teste_lista)
media_rf_treino = np.mean(acc_rf_treino_lista)
media_rf_teste = np.mean(acc_rf_teste_lista)

print("\n--- Árvore de Decisão (Especialista Único) ---")
print(f"  Acurácia MÉDIA no TREINO: {media_dt_treino * 100:.2f}%")
print(f"  Acurácia MÉDIA no TESTE:  {media_dt_teste * 100:.2f}%")
print(f"  -> GAP DE OVERFITTING:   {(media_dt_treino - media_dt_teste) * 100:.2f}%")

print("\n--- Random Forest (Comitê) ---")
print(f"  Acurácia MÉDIA no TREINO: {media_rf_treino * 100:.2f}%")
print(f"  Acurácia MÉDIA no TESTE:  {media_rf_teste * 100:.2f}%")
print(f"  -> GAP DE OVERFITTING:   {(media_rf_treino - media_rf_teste) * 100:.2f}%")

# --- 6. Plotando o Gráfico de Linhas (Comparação de Estabilidade) ---
print("\n--- 6. Gerando Gráfico de Estabilidade ---")
print("Feche a janela do gráfico para finalizar o script.")

# Eixo X: O número da execução (de 1 a 20)
testes_x_axis = range(1, N_EXECUCOES + 1)

# Eixo Y: Converte as acurácias para porcentagem
dt_treino_perc = [acc * 100 for acc in acc_dt_treino_lista]
dt_teste_perc = [acc * 100 for acc in acc_dt_teste_lista]
rf_treino_perc = [acc * 100 for acc in acc_rf_treino_lista]
rf_teste_perc = [acc * 100 for acc in acc_rf_teste_lista]

fig, ax = plt.subplots(figsize=(12, 7))

# Plotando as linhas conforme solicitado
ax.plot(testes_x_axis, dt_treino_perc, 'r--', label='Árvore (Treino)', alpha=0.7)
ax.plot(testes_x_axis, dt_teste_perc, 'r-', label='Árvore (Teste)', linewidth=2)
ax.plot(testes_x_axis, rf_treino_perc, 'g--', label='Random Forest (Treino)', alpha=0.7)
ax.plot(testes_x_axis, rf_teste_perc, 'g-', label='Random Forest (Teste)', linewidth=2)

# Adiciona títulos e legendas
ax.set_ylabel('Acurácia (%)')
ax.set_xlabel('Nº da Execução')
ax.set_title('Estabilidade: Árvore vs Random Forest (Dataset Breast Cancer)')
ax.legend(loc='lower left')
ax.grid(True, linestyle=':', alpha=0.7)

# Ajusta os limites do eixo Y para focar na variação
ax.set_ylim(60, 105)

# Ajusta o eixo X para mostrar todos os 20 testes
ax.set_xticks(range(1, N_EXECUCOES + 1))

fig.tight_layout()
plt.show()
