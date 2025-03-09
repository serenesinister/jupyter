import cv2
import subprocess
import difflib
import matplotlib.pyplot as plt

#ler conteúdo de arquivo.txt
def ler_arquivo(arquivo):
    with open(arquivo, 'r', encoding='utf-8') as file:
        return file.read()

#similaridade entre dois textos
def calcular_similaridade(texto1, texto2):
    seq = difflib.SequenceMatcher(None, texto1, texto2)
    return seq.ratio() * 100  # Retorna o percentual de acerto

#caminho da imagem original (no meu computador)
imagem_path = 'C:\\Users\\Wemerson\\Downloads\\prova.jpg'

#carregar imagem original e em escala de cinza
imagem = cv2.imread(imagem_path, 0)

if imagem is None:
    raise FileNotFoundError(f"Erro: Não foi possível carregar a imagem no caminho: {imagem_path}")

#janelas
tamanho_janela_1 = 5  # Janela 5x5
tamanho_janela_2 = 7  # Janela 7x7

#Equalização de Histograma Local com objetos CLAHE
clahe_1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tamanho_janela_1, tamanho_janela_1))
clahe_2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tamanho_janela_2, tamanho_janela_2))

#aplicar equalização local
imagem_eq_1 = clahe_1.apply(imagem)
imagem_eq_2 = clahe_2.apply(imagem)

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

#arquivos de texto a partir dos PDFs
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

#calcular similaridade entre textos
percentual_bruto = calcular_similaridade(texto_original, texto_bruto)
percentual_5x5 = calcular_similaridade(texto_original, texto_5x5)
percentual_7x7 = calcular_similaridade(texto_original, texto_7x7)

#arquivo markdown com tabela de comparação
tabela_markdown = f"""
| Arquivo           | Imagem Processada              | Arquivo PDF Gerado    | Arquivo de Texto Gerado | Percentual de Acerto (em relação ao Original) |
|-------------------|--------------------------------|-----------------------|-------------------------|-----------------------------------------------|
| Bruto             | imagem_bruta.png               | resultado_bruto.pdf   | texto_bruto.txt         | {percentual_bruto:.2f}%                                        |
| Eq.Janela 5x5     | imagem_equalizada_5x5.png      | resultado_5x5.pdf     | texto_5x5.txt           | {percentual_5x5:.2f}%                                        |
| Eq.Janela 7x7     | imagem_equalizada_7x7.png      | resultado_7x7.pdf     | texto_7x7.txt           | {percentual_7x7:.2f}%                                        |
"""

#salvar tabela markdown
with open("tabela_comparacao.md", "w") as f:
    f.write(tabela_markdown)

print("Tabela Markdown gerada com sucesso.")
