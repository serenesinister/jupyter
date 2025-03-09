import os
from skimage.transform import rescale
from skimage.io import imread, imsave
from skimage.filters import gaussian
import numpy as np
import matplotlib.pyplot as plt

#parâmetros
fator_escala = 1/3  #fator redução
fator_highboost = 4  #fator A do filtro HighBoost

#diretórios
pasta_entrada = "C:/Users/Wemerson/Downloads"  #pasta de entrada das imagens
pasta_saida = "C:/Users/Wemerson/Downloads/placas_highboost"  #pasta para salvar imagens processadas
os.makedirs(pasta_saida, exist_ok=True)

#filtro HighBoost
def aplicar_highboost(imagem, A, sigma=1):
    #cálculo do passa-baixa (PB)
    imagem_borrada = gaussian(imagem, sigma=sigma)  #PB = Imagem Borrada (filtro passa-baixa)
    
    #cálculo do passa-alta (PA)
    frequencias_altas = imagem - imagem_borrada  #PA = O - PB (componentes de alta frequência)
    
    highboost = imagem + A * frequencias_altas  #HB = O + A * PA (implementado)
    
    #limitação dos valores no intervalo [0, 1]
    return np.clip(highboost, 0, 1)

#nomes das imagens das placas
nomes_arquivos = ["placa01.png", "placa02.png", "placa03.png", "placa04.png", "placa05.png"]

#processar imagens
for nome_arquivo in nomes_arquivos:
    caminho_arquivo = os.path.join(pasta_entrada, nome_arquivo)
    
    #o arquivo existe?
    if os.path.exists(caminho_arquivo):
        print(f"Lendo a imagem {nome_arquivo}...")

        #carregar a imagem em nível de cinza
        imagem = imread(caminho_arquivo, as_gray=True)
        
        #redimensionar e restaurar
        imagem_redimensionada = rescale(imagem, fator_escala, anti_aliasing=True)
        imagem_restaurada = rescale(imagem_redimensionada, 1/fator_escala, anti_aliasing=True)
        
        #aplicar o filtro HighBoost
        imagem_highboost = aplicar_highboost(imagem_restaurada, fator_highboost, sigma=1.5)
        
        #converter a imagem para o formato correto (0-255, inteiros de 8 bits)
        imagem_highboost = np.uint8(imagem_highboost * 255)

        #salvar o resultado
        caminho_saida_arquivo = os.path.join(pasta_saida, f"highboost_{nome_arquivo}")
        imsave(caminho_saida_arquivo, imagem_highboost)
        print(f"Imagem {nome_arquivo} processada e salva como {caminho_saida_arquivo}")
        
        #exibir imagens para comparação
        _, ax = plt.subplots(1, 4, figsize=(15, 5))
        ax[0].imshow(imagem, cmap='gray')
        ax[0].set_title("Original")
        ax[1].imshow(imagem_redimensionada, cmap='gray')
        ax[1].set_title("Reduzida")
        ax[2].imshow(imagem_restaurada, cmap='gray')
        ax[2].set_title("Restaurada")
        ax[3].imshow(imagem_highboost, cmap='gray')
        ax[3].set_title("Highboost")
        plt.show()
    else:
        print(f"Arquivo {nome_arquivo} não encontrado!")
