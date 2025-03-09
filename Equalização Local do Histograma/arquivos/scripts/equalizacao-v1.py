import cv2
import matplotlib.pyplot as plt

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

plt.tight_layout()
plt.show()
