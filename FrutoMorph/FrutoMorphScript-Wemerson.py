#bibliotecas
import matplotlib.pyplot as plt #para exibir as imagens e gráficos
from matplotlib.image import imread #para ler a imagem a partir de um arquivo (PNG por exemplo)
import numpy as np #para manipulação de arrays e operações matemáticas (como arrays)
from skimage import color #para conversões entre diferentes representações de cores (como escala de cinza)
from skimage.draw import circle_perimeter #para desenhar o perímetro (contorno) de círculo na imagem
from skimage.draw import disk  #para gerar os índices dos pixels dentro de um círculo em uma imagem



#diretório da imagem
imagem = imread(r"C:\Users\Wemerson\Downloads\FrutoMorph\dataset\frutas0.png")

#converter para escala de cinza
imagem_cinza = color.rgb2gray(imagem)

#exibir a imagem original e a imagem em escala de cinza
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(imagem, cmap='gray')
plt.title("Imagem Original", loc="left")

plt.subplot(1, 2, 2)
plt.imshow(imagem_cinza, cmap='gray')
plt.title("Imagem em Escala de Cinza", loc="left")

plt.show()



#Filtro Mediano
def filtro_mediano(imagem, tamanho_kernel=3):
    #obter dimensões da imagem
    m, n = imagem.shape
    
    #imagem para armazenar o resultado
    imagem_mediana = np.copy(imagem)

    #raio do kernel
    raio = tamanho_kernel // 2

    #sobre cada pixel (exceto as bordas)
    for i in range(raio, m - raio):
        for j in range(raio, n - raio):
            #extração da região de vizinhança 3x3
            vizinhanca = imagem[i - raio:i + raio + 1, j - raio:j + raio + 1]
            
            #ordenação dos valores da vizinhança para pegar o valor mediano
            imagem_mediana[i, j] = np.median(vizinhanca)
    
    return imagem_mediana
    
    
    
#aplicação do filtro mediano
imagem_com_mediana = filtro_mediano(imagem_cinza, tamanho_kernel=3)

#imagem original e imagem com filtro mediano
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(imagem_cinza, cmap='gray')
plt.title("Imagem em Escala de Cinza", loc="left")

plt.subplot(1, 2, 2)
plt.imshow(imagem_com_mediana, cmap='gray')
plt.title("Imagem com Filtro Mediano", loc="left")

plt.show()



#Filtro Sobel
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

#aplicação do filtro Sobel
bordas = filtro_sobel(imagem_cinza)
bordas_suavizadas = filtro_sobel(imagem_com_mediana)

#imagens lado a lado para comparação
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(bordas, cmap='gray')
plt.title("Detecção de Bordas (Sobel) - Sem Suavização", loc="left")

plt.subplot(1, 2, 2)
plt.imshow(bordas_suavizadas, cmap='gray')
plt.title("Detecção de Bordas (Sobel) - Com Suavização (Mediana)", loc="left")

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
def binarizacao(imagem, limiar=None):
    #normalizar imagem, se necessário
    imagem_normalizada = imagem / np.max(imagem) #normaliza para o intervalo [0, 1] se não for já
    
    if limiar is None:
        limiar = 0.05 * np.max(imagem_normalizada) #definir limiar como 5% do valor máximo da imagem

    imagem_binaria = np.zeros_like(imagem_normalizada)
    
    #a binarização será 1 (branca) se o valor da borda for maior que o limiar
    imagem_binaria[imagem_normalizada > limiar] = 1
    
    return imagem_binaria

#aplicação da binarização
binaria = binarizacao(bordas_suavizadas)

#imagem binarizada
plt.imshow(binaria, cmap='gray')
plt.title("Imagem Binarizada", loc="left")
plt.show()



plt.title("Histograma da Imagem Binarizada", loc="left")
#histograma da imagem de Binarizada
exibir_histograma(binaria)



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

#intervalo de raios de 25 a 60 pixels
intervalo_raio = np.arange(25, 60, 1)

#executando a Transformada de Hough Circular na imagem binária
acumulador = hough_circular(binaria, intervalo_raio)

#definir o raio desejado para visualização (um valor central para a análise)
raio_desejado = 50  #raio dentro do intervalo 50 px

#encontrar o índice correto dentro do intervalo
indice_raio = np.where(intervalo_raio == raio_desejado)[0][0]  #achar índice do raio 50

#visualizar a grade de acumuladores para o raio escolhido
plt.imshow(acumulador[:, :, indice_raio], cmap='gray')
plt.title(f"Grade de Acumuladores", loc="left")
plt.show()



#encontrar os centros dos círculos na matriz acumuladora com base em um limiar
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
    
    
    
#desenhar círculos na imagem
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
        imagem_com_circulos[rr, cc] = [0, 255, 255] #círculo turquesa (RGB)

    return imagem_com_circulos

#encontra os centros dos círculos com votos acima de 99% do máximo no acumulador
centros = encontrar_centros_circulos(acumulador, intervalo_raio, limiar=0.99)
#desenhar os círculos na imagem original
imagem_com_circulos = desenhar_circulos(imagem, centros)

#exibir imagem com círculos
plt.imshow(imagem_com_circulos)
plt.title("Círculos Detectados na Imagem", loc="left")
plt.show()



#eliminar círculos próximos
def eliminar_circulos_proximos(centros, distancia_minima=10):
    centros_filtrados = []
    #para cada círculo detectado
    for i, (x1, y1, r1) in enumerate(centros):
        #verifica se o círculo está suficientemente distante de outros
        adicionar = True
        for x2, y2, r2 in centros_filtrados:
            #calcular a distância entre os centros dos círculos
            distancia = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distancia < distancia_minima:
                adicionar = False
                break
        if adicionar:
            centros_filtrados.append((x1, y1, r1))  #adiciona o círculo à lista
    return centros_filtrados

#aplicar eliminação de círculos próximos
centros_filtrados = eliminar_circulos_proximos(centros, distancia_minima=35) #aqui foi ajustada manualmente ;(
#desenhar os círculos filtrados na imagem original
imagem_com_circulos_filtrados = desenhar_circulos(imagem, centros_filtrados)

#exibição
plt.imshow(imagem_com_circulos_filtrados)
plt.title("Círculos Detectados (Filtrados)", loc="left")
plt.show()

#número de frutas detectadas
print(f"Frutas detectadas: {len(centros_filtrados)}")



#calcular a cor média dentro de um círculo
def calcular_cor_media(imagem, centro, raio):
    rr, cc = disk(centro, raio, shape=imagem.shape)  #usa disk para pegar a área interna do círculo
    rr = np.clip(rr, 0, imagem.shape[0] - 1)
    cc = np.clip(cc, 0, imagem.shape[1] - 1)
    return np.mean(imagem[rr, cc], axis=0)  #calcular a cor média diretamente
    
    
    
#identificar a fruta com base na cor média (usando HSV)
def identificar_fruta(cor_media):
    hsv = color.rgb2hsv(cor_media[np.newaxis, np.newaxis, :])[0, 0, :] #converter a cor média de RGB para HSV
    hue, saturation, value = hsv[0] * 360, hsv[1], hsv[2]  # h em 0-360, s em 0-1, v em 0-1

    #determinar se a cor é escura ou clara com base no valor (luminosidade)
    if value < 0.5: #tom mais escura
        luminosidade = "escura"
    else: #cor mais clara
        luminosidade = "clara"

    #intervalos de cor para identificação das frutas de acordo com o valor de hue (matiz)
    if 0 <= hue <= 20 and luminosidade == "clara":
        return "Maçã" #vermelho mais claro (Maçã)
    
    if ((0 <= hue <= 15) or (250 <= hue <= 350)) and luminosidade == "escura" and saturation > 0.5:
        return "Ameixa"  # Variações de vermelho escuro, roxo ou magenta

    if 10 <= hue <= 20 and saturation < 0.3:
        return "Marrom"  #tons de marrom saturação baixa
        
    if 20 < hue <= 43 and saturation > 0.3: 
        return "Laranja" #tons de laranja

    if 43 < hue <= 60 and saturation > 0.4: 
        return "Limão Siciliano" #tons de amarelo forte e saturado
    
    if 40 < hue <= 60 and saturation < 0.4: 
        return "Pera" #tons de amarelo suave e pastel (saturação baixa)
    
    if 60 < hue <= 150: 
        return "Limão" #tons de verde

    return "Fruta de Cor Mista" #se a cor não corresponder a nenhum intervalo definido
    
    
    
#desenhar círculos e identificar frutas
def desenhar_circulos_e_frutas(imagem, centros):
    imagem_com_circulos = imagem.copy()
    
    if imagem_com_circulos.dtype in [np.float32, np.float64] and imagem_com_circulos.max() <= 1.0:
        imagem_com_circulos = (imagem_com_circulos * 255).astype(np.uint8)  #convertendo para [0, 255] se necessário
    
    for (x, y, r) in centros:
        rr, cc = circle_perimeter(x, y, r, shape=imagem.shape) #cálculo da circunferência
        rr = np.clip(rr, 0, imagem.shape[0] - 1)
        cc = np.clip(cc, 0, imagem.shape[1] - 1)
        
        imagem_com_circulos[rr, cc] = [0, 255, 255]  #desenha circunferência
        cor_media = calcular_cor_media(imagem, (x, y), r) #calcula cor média da área interna
        fruta = identificar_fruta(cor_media)  #identifica fruta com base na cor média
        
        plt.text(y, x, fruta, color="white", fontsize=8, ha="center", va="center")  #escreve o nome da fruta
    
    return imagem_com_circulos
    
    
    
#desenha círculos nas posições dos centros detectados e identifica as frutas com base na cor média de cada círculo
imagem_com_circulos_e_frutas = desenhar_circulos_e_frutas(imagem, centros_filtrados)

#exibição da imagem com círculos e nomes de frutas
plt.imshow(imagem_com_circulos_e_frutas)
plt.title(f"FrutoMorph: {len(centros_filtrados)} frutas detectadas", loc="left") #exibir o número de frutas detectadas
plt.gca().axis('off')
plt.savefig(r"C:\Users\Wemerson\Downloads\FrutoMorph\detected\deteccao.png", bbox_inches='tight', pad_inches=0)
plt.show()