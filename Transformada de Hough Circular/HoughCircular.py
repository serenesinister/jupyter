#Transformada de Hough Circular

import matplotlib.pyplot as plt #para exibir as imagens e gráficos
from matplotlib.image import imread #para ler a imagem a partir de um arquivo (PNG por exemplo)
import numpy as np #para manipulação de arrays e operações matemáticas (como arrays)
from skimage import color #para conversões entre diferentes representações de cores (como escala de cinza)
from skimage.draw import circle_perimeter #para desenhar o perímetro (contorno) de círculo na imagem

#diretório da imagem
imagem = imread(r"C:\Users\Wemerson\Downloads\rosto.png")

#converter para escala de cinza
imagem_cinza = color.rgb2gray(imagem)

#exibir a imagem
plt.imshow(imagem_cinza, cmap="gray")
plt.title("Imagem Original em Escala de Cinza", loc="left")
plt.show()

#filtro Sobel
def filtro_sobel(imagem):
    #máscara Sobel para a direções horizontal(Gx)
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])
    #máscara Sobel para a direções vertical(Gy)
    sobel_y = np.array([[-1,-2,-1], 
                        [ 0, 0, 0], 
                        [ 1, 2, 1]])
    
    #obter dimensões da imagem
    m, n = imagem.shape

    #imagens de saída para Gx e Gy
    gx = np.zeros_like(imagem)
    gy = np.zeros_like(imagem)

    #convolução com as máscaras Sobel
    for i in range(1, m - 1): #ignorar bordas da imagem
        for j in range(1, n - 1):
            regiao = imagem[i - 1:i + 2, j - 1:j + 2] #região 3x3 ao redor do pixel (i, j)
            gx[i, j] = np.sum(regiao * sobel_x) #aplicar máscara Sobel na direção x
            gy[i, j] = np.sum(regiao * sobel_y) #aplicar máscara Sobel na direção y

    #calcular magnitude do gradiente (combinação de Gx e Gy)
    magnitude_gradiente = np.sqrt(gx**2 + gy**2)

    return magnitude_gradiente

#aplicando filtro Sobel
bordas = filtro_sobel(imagem_cinza)

#exibir a imagem de bordas
plt.imshow(bordas, cmap='gray')
plt.title("Detecção de Bordas (Sobel)", loc="left")
plt.show()

#histograma
def exibir_histograma(imagem):
    #calcular histograma da imagem (intensidades de pixel)
    plt.hist(imagem.ravel(), bins=256, range=(0, 1), color='grey', alpha=0.7)
    plt.xlabel('Intensidade de pixel')
    plt.ylabel('Número de pixels')
    plt.show()

plt.title("Histograma da Imagem", loc="left")
#histograma da imagem de bordas
exibir_histograma(bordas)

#binarização
def binarizacao(imagem, limiar=0.1):
    #limite de intensidade para binarizar a imagem
    imagem_binaria = np.zeros_like(imagem)
    
    #imagem binária será 1 (branca) se o valor da borda for maior que o limiar
    imagem_binaria[imagem > limiar] = 1
    
    return imagem_binaria

#binarização da imagem de bordas
binaria = binarizacao(bordas)

#exibir imagem binarizada
plt.imshow(binaria, cmap='gray')
plt.title("Imagem Binarizada", loc="left")
plt.show()

#histograma da imagem de Binarizada
plt.title("Histograma da Imagem Binarizada", loc="left")
exibir_histograma(binaria) #terá apenas 0s e 1s

#função de erosão (A ⊖ B): erosão de uma imagem A com o elemento estruturante B
def erosao(imagem, elemento_estruturante):
    m, n = imagem.shape #dimensões da imagem
    h, w = elemento_estruturante.shape #dimensões do elemento estruturante
    resultado = np.zeros_like(imagem) #imagem de saída
    pad_h, pad_w = h//2, w//2 #padding para bordas

    #iteração sobre a imagem (ignorando as bordas)
    for i in range(pad_h, m - pad_h):
        for j in range(pad_w, n - pad_w):
            regiao = imagem[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1] #região da imagem ao redor do pixel (i, j)
            
            #a erosão só mantém o pixel se a região for totalmente compatível com o elemento estruturante
            if np.all(regiao[elemento_estruturante == 1] == 1):
                resultado[i, j] = 1
    return resultado #retorna a imagem erodida
    
#função de dilatação (A ⊕ B): dilatação de uma imagem A com o elemento estruturante B
def dilatacao(imagem, elemento_estruturante):
    m, n = imagem.shape #dimensões da imagem
    h, w = elemento_estruturante.shape #dimensões do elemento estruturante
    resultado = np.zeros_like(imagem) #imagem de saída
    pad_h, pad_w = h//2, w//2 #padding para bordas

    #iteração sobre a imagem (ignorando as bordas)
    for i in range(pad_h, m - pad_h):
        for j in range(pad_w, n - pad_w):
            regiao = imagem[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1] #região da imagem ao redor do pixel (i, j)
            
            #a dilatação acontece se pelo menos um pixel da região corresponder ao elemento estruturante
            if np.any(regiao[elemento_estruturante == 1] == 1):
                resultado[i, j] = 1
    return resultado #retorna a imagem dilatada
    
#abertura (A ⊖ B) ⊕ B: primeiro a erosão (A ⊖ B) e depois a dilatação ((A ⊖ B) ⊕ B)
def abertura(imagem, elemento_estruturante):
    return dilatacao(erosao(imagem, elemento_estruturante), elemento_estruturante)
    
#fechamento (A ⊕ B) ⊖ B: primeiro a dilatação (A ⊕ B) e depois a erosão ((A ⊕ B) ⊖ B)
def fechamento(imagem, elemento_estruturante):
    return erosao(dilatacao(imagem, elemento_estruturante), elemento_estruturante)
    
#elemento estruturante (kernel 3x3)
elemento_estruturante = np.array([[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]], dtype=np.uint8)

#limpeza da imagem binária com abertura e fechamento
binaria_fechada = fechamento(binaria, elemento_estruturante) #remover pequenos ruídos
binaria_limpa = abertura(binaria_fechada, elemento_estruturante) #preencher pequenos buracos

#exibir imagem limpa
plt.imshow(binaria_limpa, cmap='gray')
plt.title("Imagem Limpa (Abertura + Fechamento)", loc="left")
plt.show()

#Transformada de Hough Circular (CHT)
def hough_circular(imagem, intervalo_raio):
    m, n = imagem.shape #obter dimensões da imagem (m: altura, n: largura)
    acumulador = np.zeros((m, n, len(intervalo_raio))) #inicializa matriz do acumulador

    #obter índices dos pixels de borda (onde o valor é maior que zero)
    pixels_borda = np.argwhere(imagem > 0)

    #para cada pixel de borda, calcular os possíveis centros dos círculos
    for (i, j) in pixels_borda:
        for r_index, r in enumerate(intervalo_raio): #para cada raio no intervalo de raios
            #calcular os centros do círculo para cada ângulo de 0 a 360 graus
            thetas = np.arange(0, 360, 1) #cria vetor de ângulos de 0 a 360 graus
            a = (i - r * np.cos(np.deg2rad(thetas))).astype(int) #coordenada x (horizontal) do centro
            b = (j - r * np.sin(np.deg2rad(thetas))).astype(int) #coordenada y (vertical) do centro

            #filtrar índices da imagem para garantir que o centro está dentro dos limites
            mascara_valida = (0 <= a) & (a < m) & (0 <= b) & (b < n)
            a_valido = a[mascara_valida]
            b_valido = b[mascara_valida]

            #acumular votos nos centros válidos (dentro da imagem)
            for ai, bi in zip(a_valido, b_valido):
                acumulador[ai, bi, r_index] += 1  #incrementa valor do acumulador para o centro (ai, bi)

    return acumulador #retorna a matriz acumuladora

#intervalo de raios (definido de 15 a 75, com intervalo 1)
intervalo_raio = np.arange(15, 76, 1)

#executando a Transformada de Hough Circular na imagem binária limpa
acumulador = hough_circular(binaria_limpa, intervalo_raio)

#visualizar a grade de acumuladores para o raio 75 (índice 60 pois começa em 0)
plt.imshow(acumulador[:, :, 60], cmap='gray')
plt.title("Grade de Acumuladores", loc="left")
plt.show()

#função para encontrar os centros dos círculos na matriz acumuladora com base em um limiar
def encontrar_centros_circulos(acumulador, intervalo_raio, limiar):   
    centros = []
    #para cada raio, procura-se os pontos máximos na matriz acumuladora
    for r_index in range(acumulador.shape[2]):
        max_acum = np.max(acumulador[:, :, r_index])
        
        #só considera os picos com votos acima de um certo limiar
        if max_acum >= limiar * max_acum:
            #obter índices de todos os pontos com valor máximo no acumulador
            pontos_maximos = np.argwhere(acumulador[:, :, r_index] == max_acum)
            for (x, y) in pontos_maximos:
                centros.append((x, y, intervalo_raio[r_index])) #adiciona o centro e o raio correspondente

    return centros
    
#função para desenhar círculos na imagem
def desenhar_circulos(imagem, centros_circulos):
    imagem_com_circulos = imagem.copy() #cópia da imagem original para evitar alterações diretas
    
    #verificar se a imagem é do tipo float e os valores estão no intervalo [0, 1]
    if imagem_com_circulos.dtype == np.float32 or imagem_com_circulos.dtype == np.float64:
        if imagem_com_circulos.max() <= 1.0: #se os valores estão no intervalo [0, 1]
            imagem_com_circulos = (imagem_com_circulos * 255).astype(np.uint8) #convertendo para o intervalo [0, 255]
    
    for (x, y, r) in centros_circulos:
        #calcular perímetro do círculo usando a função circle_perimeter
        rr, cc = circle_perimeter(x, y, r, shape=imagem.shape)
        
        #garantir que os índices estão dentro da imagem
        rr = np.clip(rr, 0, imagem.shape[0] - 1)
        cc = np.clip(cc, 0, imagem.shape[1] - 1)
        
        #desenhar o círculo (marcando pixels do perímetro)
        imagem_com_circulos[rr, cc] = [0, 255, 255] #círculo verde água (RGB)

    return imagem_com_circulos

#encontra os centros dos círculos com votos acima de 99% do máximo no acumulador
centros = encontrar_centros_circulos(acumulador, intervalo_raio, limiar=0.99)
#desenhar os círculos na imagem original
imagem_com_circulos = desenhar_circulos(imagem, centros)

#exibir imagem com círculos
plt.imshow(imagem_com_circulos)
plt.title("Círculos Detectados na Imagem", loc="left")
plt.show()