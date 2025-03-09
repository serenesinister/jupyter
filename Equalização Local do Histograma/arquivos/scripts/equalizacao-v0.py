import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para equalização local do histograma
def equalizacao_local(imagem, tamanho_janela):
    # Obter as dimensões da imagem
    altura, largura = imagem.shape
    imagem_eq = np.zeros_like(imagem)

    # Para cada pixel da imagem
    for i in range(altura):
        for j in range(largura):
            # Definir limites da janela
            x1 = max(i - tamanho_janela // 2, 0)
            x2 = min(i + tamanho_janela // 2, altura - 1)
            y1 = max(j - tamanho_janela // 2, 0)
            y2 = min(j + tamanho_janela // 2, largura - 1)

            # Extrair a janela local
            janela = imagem[x1:x2+1, y1:y2+1]

            # Equalização do histograma da janela local
            janela_eq = cv2.equalizeHist(janela)

            # Atribuir o valor equalizado à posição central da janela
            imagem_eq[i, j] = janela_eq[tamanho_janela // 2, tamanho_janela // 2]

    return imagem_eq

# Carregar a imagem
imagem = cv2.imread('C:\\Users\\Wemerson\\Downloads\\prova.jpg', 0)  # Carregar como imagem em escala de cinza

# Definir diferentes tamanhos de janela
tamanho_janela_1 = 5  # Exemplo de janela 5x5
tamanho_janela_2 = 15  # Exemplo de janela 15x15

# Aplicar a equalização local
imagem_eq_1 = equalizacao_local(imagem, tamanho_janela_1)
imagem_eq_2 = equalizacao_local(imagem, tamanho_janela_2)

# Mostrar as imagens
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(imagem, cmap='gray')
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(imagem_eq_1, cmap='gray')
plt.title(f"Equalização (janela {tamanho_janela_1}x{tamanho_janela_1})")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(imagem_eq_2, cmap='gray')
plt.title(f"Equalização (janela {tamanho_janela_2}x{tamanho_janela_2})")
plt.axis("off")

plt.show()
