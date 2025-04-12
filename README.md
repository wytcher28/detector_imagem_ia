# Detector de Imagens Geradas por IA

Este projeto utiliza aprendizado de máquina para identificar se uma imagem foi gerada por inteligência artificial ou é uma fotografia real.

## Como usar

1. Coloque imagens reais na pasta `imagens_reais/`.
2. Coloque imagens geradas por IA na pasta `imagens_ia/`.
3. Execute o script com `python detector.py` para treinar.
4. Use a função `verificar_imagem()` para verificar uma nova imagem.

## Dependências

- numpy
- scikit-learn
- scikit-image
