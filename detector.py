# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 16:42:42 2025

@author: a840760
"""
# detector.py
import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

def extrair_lbp(imagem, P=8, R=1):
    # Converte a imagem para escala de cinza inteira
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY).astype('uint8')
    lbp = local_binary_pattern(imagem, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normaliza o histograma
    return hist

def carregar_imagens_e_extrair_caracteristicas(diretorio, rotulo):
    dados = []
    labels = []
    for nome_arquivo in os.listdir(diretorio):
        caminho = os.path.join(diretorio, nome_arquivo)
        imagem = cv2.imread(caminho)
        if imagem is not None:
            hist = extrair_lbp(imagem)
            dados.append(hist)
            labels.append(rotulo)
    return dados, labels


def treinar_modelo(caminho_reais, caminho_ia):
    print("[INFO] Carregando imagens...")
    dados_reais, labels_reais = carregar_imagens_e_extrair_caracteristicas(caminho_reais, 0)
    dados_ia, labels_ia = carregar_imagens_e_extrair_caracteristicas(caminho_ia, 1)

    X = np.array(dados_reais + dados_ia)
    y = np.array(labels_reais + labels_ia)

    print("[INFO] Treinando o modelo...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        modelo = SVC(kernel='linear', probability=True)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        print(classification_report(y_test, y_pred, zero_division=0))

        joblib.dump(modelo, "modelo_sem_opencv.pkl")
        print("[INFO] Modelo salvo como modelo_sem_opencv.pkl")

if __name__ == "__main__":
    caminho_reais = "C:/Users/a840760/Documents/Projetos/Teste Projetos/detector_imagem_ia/imagens_reais"
    caminho_ia = "C:/Users/a840760/Documents/Projetos/Teste Projetos/detector_imagem_ia/imagens_ia"
    treinar_modelo(caminho_reais, caminho_ia)
