import cv2
import matplotlib.pyplot as plt
from skimage import io

# Carregar uma imagem usando OpenCV
imagem = cv2.imread("C:\\Users\\Wemerson\\Downloads\\prova.png")
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# Carregar uma imagem usando scikit-image
imagem_sk = io.imread("prova.png")

# Mostrar imagens
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(imagem_rgb)
ax[0].set_title("Imagem com OpenCV")
ax[0].axis("off")

ax[1].imshow(imagem_sk)
ax[1].set_title("Imagem com scikit-image")
ax[1].axis("off")

plt.show()
