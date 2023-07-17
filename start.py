import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import cosine
import numpy as np

# Carregar o modelo ResNet pré-treinado
modelo_resnet = models.resnet50(pretrained=True)
modelo_resnet.eval()

# Carregar e pré-processar as imagens de treinamento
imagens_treinamento = ['6238e4f0a87a7.png', '6238e4f0a825b.png', '6238e6c85b7ba.png', '6238e6c85c482.png', '6238e7b2b0bf1.png', '6238e7cc65fdf.png', '6238e7de8692b.png', '6238e42dda161.png', '6238e706cc8a4.png', '6238e729b1aa1.png', '6238e729b2d42.png', '6238e729b356b.png']
vetores_caracteristicas = []

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for imagem_treinamento in imagens_treinamento:
    imagem = Image.open(imagem_treinamento)
    imagem = transform(imagem)
    imagem = imagem.unsqueeze(0)

    vetor_caracteristicas = modelo_resnet(imagem)
    vetor_caracteristicas = torch.flatten(vetor_caracteristicas).detach().numpy()
    vetores_caracteristicas.append(vetor_caracteristicas)

# Carregar e pré-processar a imagem de teste
imagem_teste = '6238e7b2b0bf1.png'
imagem = Image.open(imagem_teste)
imagem = transform(imagem)
imagem = imagem.unsqueeze(0)

vetor_caracteristicas_teste = modelo_resnet(imagem)
vetor_caracteristicas_teste = torch.flatten(vetor_caracteristicas_teste).detach().numpy()

# Calcular a similaridade entre a imagem de teste e as imagens de treinamento
similaridades = []
for vetor_caracteristicas_treinamento in vetores_caracteristicas:
    distancia = cosine(vetor_caracteristicas_treinamento, vetor_caracteristicas_teste)
    similaridades.append(distancia)

# Obter os índices das três imagens mais similares
indices_similares = np.argsort(similaridades)[:5]

# Obter os nomes das três imagens mais similares
imagens_similares = [imagens_treinamento[indice] for indice in indices_similares]

print("As três imagens mais similares são:")
for imagem_similar in imagens_similares:
    print(imagem_similar)
