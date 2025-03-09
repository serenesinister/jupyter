import cv2
import subprocess
import difflib
import matplotlib.pyplot as plt

# Função para ler o conteúdo de um arquivo .txt
def ler_arquivo(arquivo):
    with open(arquivo, 'r', encoding='utf-8') as file:
        return file.read()

# Função para calcular a similaridade entre dois textos
def calcular_similaridade(texto1, texto2):
    seq = difflib.SequenceMatcher(None, texto1, texto2)
    return seq.ratio() * 100  # Retorna o percentual de acerto

# Caminho da imagem original
imagem_path = 'C:\\Users\\Wemerson\\Downloads\\prova.jpg'

# Carregar a imagem original e em escala de cinza
imagem = cv2.imread(imagem_path, 0)

if imagem is None:
    raise FileNotFoundError(f"Erro: Não foi possível carregar a imagem no caminho: {imagem_path}")

# Definir diferentes tamanhos de janela
tamanho_janela_1 = 5  # Janela 5x5
tamanho_janela_2 = 15  # Janela 15x15

# Criar objetos CLAHE (Equalização de Histograma Local)
clahe_1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tamanho_janela_1, tamanho_janela_1))
clahe_2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tamanho_janela_2, tamanho_janela_2))

# Aplicar a equalização local
imagem_eq_1 = clahe_1.apply(imagem)
imagem_eq_2 = clahe_2.apply(imagem)

# Salvar as imagens processadas
cv2.imwrite('imagem_original.png', imagem)
cv2.imwrite('imagem_equalizada_5x5.png', imagem_eq_1)
cv2.imwrite('imagem_equalizada_15x15.png', imagem_eq_2)

print("Imagens processadas salvas com sucesso.")

# Realizar OCR nas imagens (criando PDFs com OCR)
subprocess.run(["ocrmypdf", "--image-dpi", "300", "imagem_original.png", "resultado_original.pdf"])
subprocess.run(["ocrmypdf", "--image-dpi", "300", "imagem_equalizada_5x5.png", "resultado_5x5.pdf"])
subprocess.run(["ocrmypdf", "--image-dpi", "300", "imagem_equalizada_15x15.png", "resultado_15x15.pdf"])

# Caminho para o pdftotext
poppler_path = r"C:\poppler\Library\bin\pdftotext.exe"

# Gerar os arquivos de texto a partir dos PDFs
subprocess.run([poppler_path, "resultado_original.pdf", "texto_original.txt"])
subprocess.run([poppler_path, "resultado_5x5.pdf", "texto_5x5.txt"])
subprocess.run([poppler_path, "resultado_15x15.pdf", "texto_15x15.txt"])

# Texto original
texto_original = """Questao 2 (4 pontos)

Escreva uma função para decidir se num tabuleiro de jogo da velha há um vencedor e, caso exista,
qual é o símbolo associado a ele.

Sua função deve verificar as seguintes situações anômalas:

1. Ocorrência de mais de um vencedor
2. Ocorrência de mais de dois tipos de símbolos associados aos jogadores"""

# Ler os textos extraídos dos arquivos .txt
texto_5x5 = ler_arquivo("texto_5x5.txt")
texto_15x15 = ler_arquivo("texto_15x15.txt")

# Calcular a similaridade entre o texto original e os textos extraídos
percentual_5x5 = calcular_similaridade(texto_original, texto_5x5)
percentual_15x15 = calcular_similaridade(texto_original, texto_15x15)

# Criar o arquivo markdown com a tabela de comparação
tabela_markdown = f"""
| Tamanho da Janela | Imagem Processada              | Arquivo PDF Gerado    | Arquivo de Texto Gerado | Qualidade do OCR | Percentual de Acerto |
|-------------------|--------------------------------|-----------------------|-------------------------|------------------|----------------------|
| Original          | imagem_original.png           | resultado_original.pdf| texto_original.txt      | Baixa            | {percentual_5x5:.2f}%   |
| 5x5               | imagem_equalizada_5x5.png     | resultado_5x5.pdf     | texto_5x5.txt           | Alta             | {percentual_5x5:.2f}%   |
| 15x15             | imagem_equalizada_15x15.png   | resultado_15x15.pdf   | texto_15x15.txt         | Média            | {percentual_15x15:.2f}%  |
"""

# Salvar a tabela em um arquivo Markdown
with open("tabela_comparacao.md", "w") as f:
    f.write(tabela_markdown)

print("Tabela Markdown gerada com sucesso.")
