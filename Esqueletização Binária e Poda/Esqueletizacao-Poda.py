#Esqueletização Binária e poda

import matplotlib.pyplot as plt #para exibir a imagem

from matplotlib.image import imread

#diretório da imagem
imagem = imread(r"C:\Users\Wemerson\Downloads\digital.png")

#exibir imagem
plt.imshow(imagem, cmap="gray")
plt.title("Impressão Digital Binária",loc="left")
plt.axis("off")
plt.show()

#Preparação da Imagem 
import numpy as np  #biblioteca NumPy para manipulação de arrays/matrizes

#verificar os valores únicos na imagem
valores_unicos = np.unique(imagem)
print("Valores únicos na imagem:", valores_unicos)

#função de erosão (A ⊖ B): erosão de uma imagem A com o elemento estruturante B
def erosao(imagem, elemento_estruturante):
    m, n = imagem.shape  #dimensões da imagem
    h, w = elemento_estruturante.shape  #dimensões do elemento estruturante
    resultado = np.zeros_like(imagem)  #imagem de saída
    pad_h, pad_w = h // 2, w // 2  #padding para bordas

    #iteração sobre a imagem (ignorando as bordas)
    for i in range(pad_h, m - pad_h):
        for j in range(pad_w, n - pad_w):
            region = imagem[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]  #região da imagem ao redor do pixel (i, j)
            
            #a erosão só mantém o pixel se a região for totalmente compatível com o elemento estruturante
            if np.all(region[elemento_estruturante == 1] == 1):
                resultado[i, j] = 1
    return resultado  #retorna a imagem erodida
    
#função de dilatação (A ⊕ B): dilatação de uma imagem A com o elemento estruturante B
def dilatacao(imagem, elemento_estruturante):
    m, n = imagem.shape  #dimensões da imagem
    h, w = elemento_estruturante.shape  #dimensões do elemento estruturante
    resultado = np.zeros_like(imagem)  #imagem de saída
    pad_h, pad_w = h // 2, w // 2  #padding para bordas

    #iteração sobre a imagem (ignorando as bordas)
    for i in range(pad_h, m - pad_h):
        for j in range(pad_w, n - pad_w):
            region = imagem[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]  #região da imagem ao redor do pixel (i, j)
            
            #a dilatação acontece se pelo menos um pixel da região corresponder ao elemento estruturante
            if np.any(region[elemento_estruturante == 1] == 1):
                resultado[i, j] = 1
    return resultado  #retorna a imagem dilatada

#abertura (A ⊖ B) ⊕ B: primeiro a erosão (A ⊖ B) e depois a dilatação ((A ⊖ B) ⊕ B)
def abertura(imagem, elemento_estruturante):
    return dilatacao(erosao(imagem, elemento_estruturante), elemento_estruturante)
    
#fechamento (A ⊕ B) ⊖ B: primeiro a dilatação (A ⊕ B) e depois a erosão ((A ⊕ B) ⊖ B)
def fechamento(imagem, elemento_estruturante):
    return erosao(dilatacao(imagem, elemento_estruturante), elemento_estruturante)
    
#função de esqueletização (usando erosões sucessivas)
def esqueletizacao(imagem, elemento_estruturante, max_k=10):
    A = imagem.copy()
    
    #limpeza com abertura e fechamento (operação preparatória)
    A = abertura(A, elemento_estruturante)  #abertura para remoção de ruído
    A = fechamento(A, elemento_estruturante)  #fechamento para suavizar os limites

    #criar a imagem de esqueleto (inicialmente toda preta)
    esqueleto = np.zeros_like(imagem)

    for k in range(max_k):
        #erosão sucessiva de A (A ⊖ B) repetido k vezes
        A_erosao = erosao(A, elemento_estruturante)
        
        #dilatação da erosão (A ⊖ B) ⊕ B
        A_dilatacao = dilatacao(A_erosao, elemento_estruturante)
        
        #calcular a diferença entre a imagem original e a dilatação da erosão para obter o esqueleto (miolo)
        esqueleto_k = A - A_dilatacao  # Sk(A) = (A ⊖ B) - (A ⊖ B) ⊕ B
        
        #acumular o esqueleto (união dos esqueletos parciais)
        esqueleto = np.bitwise_or(esqueleto, esqueleto_k)
        
        #atualizar A para a próxima iteração (acumula a erosão sucessiva)
        A = A_erosao
        
        #se a imagem se tornar completamente vazia, parar
        if np.all(A == 0):
            break
            
    return esqueleto
    
#elemento estruturante (kernel 3x3 em forma de cruz)
elemento_estruturante = np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]], dtype=np.uint8)

#realizar a esqueletização
imagem_esqueleto = esqueletizacao(imagem.astype(np.uint8), elemento_estruturante) 

#exibir o resultado da esqueletização
plt.figure(figsize=(6, 6))
plt.imshow(imagem_esqueleto.astype(np.uint8), cmap="gray") #tipo uint8 para exibição
plt.title("Imagem Esqueleto", loc="left")
plt.axis("off")
plt.show()

#função de poda (pruning)
def poda(imagem_esqueletizada, elemento_estruturante, max_iter=3):
    esqueleto_podado = imagem_esqueletizada.copy()
    
    for _ in range(max_iter):
        esqueleto_temp = esqueleto_podado.copy()  #criar uma cópia da imagem atual para verificar alterações
        
        #vizinhança 3x3
        vizinhanca = [[-1, -1], [-1, 0], [-1, 1],
                      [ 0, -1], [ 0, 1],
                      [ 1, -1], [ 1, 0], [ 1, 1]]
                    
        #percorre cada pixel da imagem
        for y in range(1, esqueleto_podado.shape[0] - 1):  #ignorar bordas verticais
            for x in range(1, esqueleto_podado.shape[1] - 1):  #ignorar bordas horizontais
                if esqueleto_podado[y, x] == 1:  #se o pixel é parte do esqueleto
                    #contar o número de vizinhos conectados
                    vizinhos_conectados = sum(
                        esqueleto_podado[y + dy, x + dx] == 1 for dy, dx in vizinhanca
                    )
                    
                    #se o pixel tem apenas um vizinho conectado, pode ser uma extremidade
                    if vizinhos_conectados <= 1:
                        esqueleto_temp[y, x] = 0  #remover o pixel
                        
        #verificar se houve alguma alteração. Se não houver, interromper a poda
        if np.array_equal(esqueleto_temp, esqueleto_podado):
            break  #se a imagem não mudou, parar a iteração
        
        #atualizar a imagem esqueleto podada
        esqueleto_podado = esqueleto_temp
        
    return esqueleto_podado
    
#aplicar a poda ao esqueleto obtido
esqueleto_podado = poda(imagem_esqueleto, elemento_estruturante, max_iter=5)

#exibir o resultado final da poda
plt.figure(figsize=(6, 6))
plt.imshow(esqueleto_podado, cmap='gray')
plt.title("Esqueleto Final Após Poda", loc="left")
plt.axis('off')
plt.show()
