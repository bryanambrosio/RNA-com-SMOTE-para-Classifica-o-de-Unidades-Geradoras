# ---------------------------------------
# RNA (Rede Neural Artificial) com SMOTE
# ---------------------------------------
# Objetivo: Classificação do número mínimo de Unidades Geradoras com dados balanceados
# Autor: Bryan Ambrósio
# ---------------------------------------

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from collections import OrderedDict

# ----- Configuração de diretório de saída -----
output_dir = "rna__deep_advanced"
os.makedirs(output_dir, exist_ok=True)

# ----- Carregamento e limpeza dos dados -----
df = pd.read_excel('dadosfiguras1500_minUG.xlsx')
# Filtra apenas amostras com diferença angular >= 1
df = df[df['Vang_XES_0+'] - df['Vang_XES_0-'] >= 1].copy()
# Converte 'min_UGs' para inteiro, tratando valores não numéricos
df['min_UGs'] = pd.to_numeric(df['min_UGs'], errors='coerce')
df = df[df['min_UGs'].notna()].copy()
df['min_UGs'] = df['min_UGs'].astype(int)

# Define variáveis preditoras e alvo
X = df.drop(columns=['Arquivo', 'min_UGs'], errors='ignore')
X = X.select_dtypes(include=[np.number])
y = df['min_UGs']

# ----- Padronização dos dados -----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----- Divisão em treino e teste (estratificado) -----
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ----- Balanceamento do conjunto de treino com SMOTE -----
# Ajusta k_neighbors para evitar erro em classes pequenas
min_class_size = min(np.bincount(y_train))
k_neighbors = max(1, min_class_size - 1)
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ----- Codificação one-hot dos labels -----
y_train_res_cat = to_categorical(y_train_res)
y_test_cat = to_categorical(y_test)

# ----- Definição e compilação da RNA (rede neural profunda) -----
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_res.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y_train_res_cat.shape[1], activation='softmax') # Saída multiclasse
])

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ----- Early stopping para evitar overfitting -----
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# ----- Treinamento da rede -----
history = model.fit(
    X_train_res, y_train_res_cat,
    epochs=300,
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stop],
    verbose=1
)

# ----- Avaliação do modelo -----
y_pred = np.argmax(model.predict(X_test), axis=1)
y_train_pred = np.argmax(model.predict(X_train_res), axis=1)

# Gera e salva o classification report
report = classification_report(y_test, y_pred)
with open(os.path.join(output_dir, "classification_report_advanced.txt"), "w") as f:
    f.write(report)

# ----- Gráfico de Accuracy e Loss durante o treino -----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "acc_loss_side_by_side_advanced.png"))
plt.close()

# ----- Gráfico PCA destacando os falsos negativos (FN) -----
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_res)
X_test_pca = pca.transform(X_test)

colors = plt.cm.tab10.colors
unique_classes = np.unique(y)

fig, axs = plt.subplots(1, 2, figsize=(18, 7))

# --- PCA Treino ---
correct_train = (y_train_res == y_train_pred)
axs[0].scatter(
    X_train_pca[correct_train, 0], X_train_pca[correct_train, 1],
    c='lightgray', s=60, edgecolor='k', label='Correctly Classified', zorder=1
)
for cls in unique_classes:
    fn_mask = (y_train_res == cls) & (y_train_pred < y_train_res)
    if np.any(fn_mask):
        axs[0].scatter(
            X_train_pca[fn_mask, 0], X_train_pca[fn_mask, 1],
            marker='X', s=120, color=colors[int(cls) % 10],
            label=f'FN Class {cls}', edgecolor='k', zorder=2
        )
axs[0].set_title("Training Set (RNA with SMOTE)")
axs[0].set_xlabel("PC1")
axs[0].set_ylabel("PC2")
axs[0].grid(True)

# --- PCA Teste ---
correct_test = (y_test == y_pred)
axs[1].scatter(
    X_test_pca[correct_test, 0], X_test_pca[correct_test, 1],
    c='lightgray', s=60, edgecolor='k', label='Correctly Classified', zorder=1
)
for cls in unique_classes:
    fn_mask = (y_test == cls) & (y_pred < y_test)
    if np.any(fn_mask):
        axs[1].scatter(
            X_test_pca[fn_mask, 0], X_test_pca[fn_mask, 1],
            marker='X', s=120, color=colors[int(cls) % 10],
            label=f'FN Class {cls}', edgecolor='k', zorder=2
        )
axs[1].set_title("Test Set (RNA with SMOTE)")
axs[1].set_xlabel("PC1")
axs[1].set_ylabel("PC2")
axs[1].grid(True)

# Monta legenda única para as duas figuras
handles, labels = [], []
for ax in axs:
    h, l = ax.get_legend_handles_labels()
    handles += h
    labels += l
by_label = OrderedDict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, fontsize=10)
fig.suptitle("RNA with SMOTE – FN highlighted as (X)", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(output_dir, "rna_scatter_train_test_advanced.png"), dpi=300)
plt.close(fig)
