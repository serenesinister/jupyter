import cv2
import subprocess
import difflib
import matplotlib.pyplot as plt
import numpy as np

#ler conteúdo de um arquivo.txt
def ler_arquivo(arquivo):
    with open(arquivo, 'r', encoding='utf-8') as file:
        return file.read()

#similaridade entre dois textos
def calcular_similaridade(texto1, texto2):
    seq = difflib.SequenceMatcher(None, texto1, texto2)
    return seq.ratio() * 100  # Retorna o percentual de acerto

#calcular histograma
def calcular_histograma(regiao):
    hist, _ = np.histogram(regiao.flatten(), bins=256, range=[0,256])
    return hist

#equalizar o histograma
def equalizar_histograma(hist):
    cdf = np.cumsum(hist)  # Função de distribuição acumulada
    cdf_normalizada = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    return np.uint8(cdf_normalizada)

#aplicar equalização local
def aplicar_equalizacao_local(imagem, tamanho_bloco=(8, 8), clip_limit=2.0):
    altura, largura = imagem.shape
    imagem_equalizada = np.zeros_like(imagem)
    
    #limitação de contraste
    def aplicar_clip_limit(hist, clip_limit):
        excessos = np.maximum(hist - clip_limit, 0)
        excesso_total = np.sum(excessos)
        hist[hist > clip_limit] = clip_limit
        hist = hist + excesso_total / 256  # Redistribui os excedentes
        return hist

    #processar imagem em blocos
    for i in range(0, altura, tamanho_bloco[0]):
        for j in range(0, largura, tamanho_bloco[1]):
            #limites de cada bloco
            x_end = min(i + tamanho_bloco[0], altura)
            y_end = min(j + tamanho_bloco[1], largura)
            
            #região (bloco) da imagem
            regiao = imagem[i:x_end, j:y_end]
            
            #histograma da região
            hist = calcular_histograma(regiao)
            
            #limitação do contraste clipLimit
            hist_clip = aplicar_clip_limit(hist, clip_limit)
            
            #equalização do histograma com clipLimit
            hist_normalizado = equalizar_histograma(hist_clip)
            
            #aplicar a equalização na região
            regiao_equalizada = cv2.LUT(regiao, hist_normalizado)
            
            #região equalizada na imagem final
            imagem_equalizada[i:x_end, j:y_end] = regiao_equalizada

    #filtro de média para suavizar a imagem (interpolação entre blocos)
    imagem_equalizada = cv2.medianBlur(imagem_equalizada, 3)

    return imagem_equalizada

#caminho da imagem original (no meu computador)
imagem_path = 'C:\\Users\\Wemerson\\Downloads\\prova.jpg'

#carregar imagem original em escala de cinza
imagem = cv2.imread(imagem_path, 0)

if imagem is None:
    raise FileNotFoundError(f"Erro: Não foi possível carregar a imagem no caminho: {imagem_path}")

#janela
tamanho_janela_1 = 5  # Janela 5x5
tamanho_janela_2 = 7  # Janela 7x7

#aplicação equalização local
imagem_eq_1 = aplicar_equalizacao_local(imagem, tamanho_bloco=(tamanho_janela_1, tamanho_janela_1), clip_limit=2.0)
imagem_eq_2 = aplicar_equalizacao_local(imagem, tamanho_bloco=(tamanho_janela_2, tamanho_janela_2), clip_limit=2.0)

#salvar imagens processadas.png
cv2.imwrite('imagem_bruta.png', imagem)
cv2.imwrite('imagem_equalizada_5x5.png', imagem_eq_1)
cv2.imwrite('imagem_equalizada_7x7.png', imagem_eq_2)

print("Imagens processadas salvas com sucesso.")

#realizar OCR nas imagens (criando PDFs com OCR)
subprocess.run(["ocrmypdf", "--image-dpi", "300", "imagem_bruta.png", "resultado_bruto.pdf"])
subprocess.run(["ocrmypdf", "--image-dpi", "300", "imagem_equalizada_5x5.png", "resultado_5x5.pdf"])
subprocess.run(["ocrmypdf", "--image-dpi", "300", "imagem_equalizada_7x7.png", "resultado_7x7.pdf"])

#caminho para pdftotext
poppler_path = r"C:\poppler\Library\bin\pdftotext.exe"

#gerar arquivos de texto a partir dos PDFs
subprocess.run([poppler_path, "resultado_bruto.pdf", "texto_bruto.txt"])
subprocess.run([poppler_path, "resultado_5x5.pdf", "texto_5x5.txt"])
subprocess.run([poppler_path, "resultado_7x7.pdf", "texto_7x7.txt"])

#texto original (referência digitada)
texto_original = """Questao 2 (4 pontos)

Escreva uma função para decidir se num tabuleiro de jogo da velha há um vencedor e, caso exista,
qual é o símbolo associado a ele.

Sua função deve verificar as seguintes situações anômalas:

1. Ocorrência de mais de um vencedor
2. Ocorrência de mais de dois tipos de símbolos associados aos jogadores"""

#ler textos extraídos dos arquivos.txt
texto_bruto = ler_arquivo("texto_bruto.txt")
texto_5x5 = ler_arquivo("texto_5x5.txt")
texto_7x7 = ler_arquivo("texto_7x7.txt")

#similaridade entre textos
percentual_bruto = calcular_similaridade(texto_original, texto_bruto)
percentual_5x5 = calcular_similaridade(texto_original, texto_5x5)
percentual_7x7 = calcular_similaridade(texto_original, texto_7x7)

#arquivo markdown com tabela de comparação
tabela_markdown = f"""
| Arquivo           | Imagem Processada              | Arquivo PDF Gerado    | Arquivo de Texto Gerado | Percentual de Acerto (em relação ao Original)
|-------------------|--------------------------------|-----------------------|-------------------------|-----------------------------------------------
| Bruto             | imagem_bruta.png               | resultado_bruto.pdf   | texto_bruto.txt         | {percentual_bruto:.2f}%
| Eq.Janela 5x5     | imagem_equalizada_5x5.png      | resultado_5x5.pdf     | texto_5x5.txt           | {percentual_5x5:.2f}%
| Eq.Janela 7x7     | imagem_equalizada_7x7.png      | resultado_7x7.pdf     | texto_7x7.txt           | {percentual_7x7:.2f}%
"""

#salvar tabela markdown
with open("tabela_comparacao.md", "w") as f:
    f.write(tabela_markdown)

print("Tabela Markdown gerada com sucesso.")
