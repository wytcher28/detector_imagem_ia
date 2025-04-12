# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 16:42:42 2025

@author: a840760
"""
import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Extrai características simples da imagem com Pillow
def extrair_caracteristicas(caminho_imagem):
    img = Image.open(caminho_imagem).convert("L")  # Converte para escala de cinza
    img = img.resize((128, 128))  # Redimensiona
    img_array = np.array(img).flatten()  # Achata o array para uma lista 1D

    # Normaliza os pixels (0 a 255 -> 0 a 1)
    img_array = img_array / 255.0
    return img_array

# Carrega imagens e rotula os dados
def carregar_dados(pasta_real, pasta_ia):
    dados = []
    rotulos = []

    for arquivo in os.listdir(pasta_real):
        caminho = os.path.join(pasta_real, arquivo)
        feat = extrair_caracteristicas(caminho)
        dados.append(feat)
        rotulos.append(0)  # Real

    for arquivo in os.listdir(pasta_ia):
        caminho = os.path.join(pasta_ia, arquivo)
        feat = extrair_caracteristicas(caminho)
        dados.append(feat)
        rotulos.append(1)  # IA

    return np.array(dados), np.array(rotulos)

# Treina e salva o modelo
def treinar_modelo(pasta_real, pasta_ia):
    X, y = carregar_dados(pasta_real, pasta_ia)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    print(f"Acurácia no teste: {modelo.score(X_test, y_test):.2f}")
    joblib.dump(modelo, "modelo_sem_opencv.pkl")

# Verifica uma imagem individual
def verificar_imagem(caminho_imagem, modelo_path="modelo_sem_opencv.pkl"):
    modelo = joblib.load(modelo_path)
    feat = extrair_caracteristicas(caminho_imagem).reshape(1, -1)
    resultado = modelo.predict(feat)

    if resultado[0] == 1:
        print("🔍 A imagem foi gerada por IA.")
    else:
        print("✅ A imagem é considerada real.")

# Execução principal
if __name__ == "__main__":
    # 1ª vez: treine o modelo com imagens reais e de IA
    treinar_modelo("C:/Users/a840760/Documents/Projetos/Teste Projetos/imagens_reais", "C:/Users/a840760/Documents/Projetos/Teste Projetos/imagens_ia")

    # Depois de treinado, use para verificar
    verificar_imagem("C:/Users/a840760/Documents/Projetos/Teste Projetos/imagem_teste.png")
