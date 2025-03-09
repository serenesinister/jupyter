import os
from skimage.io import imread, imsave
from skimage.transform import rescale
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt

#parâmetros
fator_escala = 1 / 3  #fator de redução

#diretórios
pasta_entrada = "C:/Users/Wemerson/Downloads"  #pasta de entrada das imagens
pasta_saida = "C:/Users/Wemerson/Downloads/placas_highboost"  #pasta para salvar imagens processadas
os.makedirs(pasta_saida, exist_ok=True)

#Kernel Laplaciano (3x3)
kernel_laplaciano = np.array([[ 0, -0.5,  0],
                              [-0.5,  2, -0.5],
                              [ 0, -0.5,  0]])

#filtro HighBoost usando Kernel Laplaciano
def aplicar_highboost(imagem, A):
    #cálculo do passa-alta com convolução do kernel Laplaciano
    frequencias_altas = convolve2d(imagem, kernel_laplaciano, mode='same', boundary='wrap')  # Aplica a convolução com o kernel
    highboost = imagem + A * frequencias_altas  # HB = O + A * PA (implementação)
    
    #limitação dos valores no intervalo [0, 1]
    return np.clip(highboost, 0, 1)

#nomes das imagens das placas
nomes_arquivos = ["placa01.png", "placa02.png", "placa03.png", "placa04.png", "placa05.png"]

#parâmetros personalizados por imagem
parametros_imagens = {
    "placa01.png": {"fator_highboost": 3},
    "placa02.png": {"fator_highboost": 5},
    "placa03.png": {"fator_highboost": 4},
    "placa04.png": {"fator_highboost": 3},
    "placa05.png": {"fator_highboost": 2},
}

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
        imagem_restaurada = rescale(imagem_redimensionada, 1 / fator_escala, anti_aliasing=True)
        
        #parâmetros específicos para a imagem atual
        parametros = parametros_imagens.get(nome_arquivo, {"fator_highboost": 4})  # Valor padrão
        
        #aplicar o filtro HighBoost
        imagem_highboost = aplicar_highboost(imagem_restaurada, parametros["fator_highboost"])
        
        #converter a imagem para o formato correto (0-255, 8 bits)
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
        ax[3].set_title("Highboost (Laplaciano)")
        plt.show()
    else:
        print(f"Arquivo {nome_arquivo} não encontrado!")
