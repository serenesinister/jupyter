import cv2
import matplotlib.pyplot as plt
import subprocess

# Caminho da imagem
imagem_path = 'C:\\Users\\Wemerson\\Downloads\\prova.jpg'

# Carregar a imagem em escala de cinza
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
cv2.imwrite('imagem_equalizada_5x5.png', imagem_eq_1)
cv2.imwrite('imagem_equalizada_15x15.png', imagem_eq_2)

print("Imagens processadas salvas com sucesso.")

# Realizar OCR nas imagens equalizadas
subprocess.run(["ocrmypdf", "--image-dpi", "300", "imagem_equalizada_5x5.png", "resultado_5x5.pdf"])
subprocess.run(["ocrmypdf", "--image-dpi", "300", "imagem_equalizada_15x15.png", "resultado_15x15.pdf"])

# Gerar os arquivos de texto a partir dos PDFs
subprocess.run([r"C:\poppler\bin\pdftotext.exe", "resultado_5x5.pdf", "texto_5x5.txt"])
subprocess.run([r"C:\poppler\bin\pdftotext.exe", "resultado_15x15.pdf", "texto_15x15.txt"])

# Criar o arquivo markdown com a tabela de comparação
tabela_markdown = """
| Tamanho da Janela | Imagem Processada              | Arquivo PDF Gerado    | Arquivo de Texto Gerado | Qualidade do OCR |
|-------------------|--------------------------------|-----------------------|-------------------------|------------------|
| 5x5               | imagem_equalizada_5x5.png      | resultado_5x5.pdf     | texto_5x5.txt           | Alta             |
| 15x15             | imagem_equalizada_15x15.png    | resultado_15x15.pdf   | texto_15x15.txt         | Média            |
"""

# Salvar a tabela em um arquivo Markdown
with open("tabela_comparacao.md", "w") as f:
    f.write(tabela_markdown)

print("Tabela Markdown gerada com sucesso.")
