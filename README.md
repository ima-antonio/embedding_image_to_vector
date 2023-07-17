#API de Extração de Vetores de Imagens<br>
Esta é uma API simples construída com Flask que permite extrair vetores de características de imagens usando o modelo ResNet50 pré-treinado.

Instalação
Clone este repositório para o seu ambiente local:
`bash
Copy code
git clone https://github.com/ima-antonio/embedding_image_to_vector`
Acesse o diretório do projeto:
`bash
Copy code
cd api-imagem-vetor`
Crie um ambiente virtual (opcional):
`bash
Copy code
python -m venv env`
Ative o ambiente virtual (opcional):
No Windows:
`bash
Copy code
env\Scripts\activate`
No macOS/Linux:
`bash
Copy code
source env/bin/activate`
Instale as dependências do projeto:
`Copy code
pip install -r requirements.txt`
Uso
Inicie o servidor da API:
`Copy code
python api/vector.py`
Envie uma solicitação POST para a rota /vectorize com a URL da imagem que deseja extrair o vetor de características.
Exemplo usando o cURL:

`json
Copy code
curl -X POST -H "Content-Type: application/json" -d '{"url": "https://example.com/image.jpg"}' http://localhost:5000/vectorize`
A resposta da API será um objeto JSON contendo o vetor de características da imagem.
Exemplo de resposta:

`json
Copy code
{
  "vector": [-2.68184233e+00, -1.41177654e+00, 9.86026764e-01, -1.89144754e+00]
}`
Exemplo de Array de Imagens e Similaridades
Para calcular a similaridade entre uma imagem de teste e um array de imagens, você pode usar o seguinte exemplo de código:

`python
Copy code
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
imagens_treinamento = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg']
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
imagem_teste = 'image_test.jpg'
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
indices_similares = np.argsort(similaridades)[:3]

# Obter os nomes das três imagens mais similares
imagens_similares = [imagens_treinamento[indice] for indice in indices_similares]

print("As três imagens mais similares são:")
for imagem_similar in imagens_similares:
    print(imagem_similar)`
Certifique-se de substituir os nomes das imagens e as URLs de acordo com o seu caso de uso.

Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir problemas (issues) e enviar pull requests para aprimorar esta API.
