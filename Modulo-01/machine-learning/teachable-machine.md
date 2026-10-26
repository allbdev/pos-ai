# Teachable Machine

> Projeto prático: [`archive/`](./archive) — dataset de imagens de **100 esportes** (Kaggle) usado para treinar um classificador de imagens.
> Teoria por trás: [conceito-redes-neurais.md](./conceito-redes-neurais.md) · Código equivalente "na mão": [`exemplo-00/index.js`](./exemplo-00/index.js)

Usar **Kaggle + Teachable Machine** para treinar um modelo que reconhece dados específicos.

Durante a aula, Wendel treinou para reconhecer objetos do seu escritório + raças de cachorros.

Meu treino: reconhecer **esportes** com o dataset em [`archive/`](./archive).

---

## 1. O que é

- **Teachable Machine** (https://teachablemachine.withgoogle.com) — ferramenta do Google para treinar modelos **sem escrever código**, direto no navegador.
- Tipos de projeto: **imagem**, **áudio** e **pose** (corpo).
- Por baixo, usa **TensorFlow.js** — a mesma biblioteca do `exemplo-00`.
- O treino roda **no seu navegador** (as imagens não precisam ir para um servidor).

- **Kaggle** (https://www.kaggle.com) — plataforma com milhares de **datasets** prontos (e competições de ML). Foi de lá que veio o `archive/`.

---

## 2. O dataset: `archive/` (100 Sports Image Classification)

```
archive/
├── train/       100 pastas (1 por esporte) → 13.493 imagens   (aprender)
├── valid/       100 pastas                 →    500 imagens   (5 por esporte)
├── test/        100 pastas                 →    500 imagens   (5 por esporte)
├── sports.csv   índice de todas as imagens (14.493 linhas)
└── EfficientNetB0-100-(224 X 224)- 98.40.h5   modelo já treinado (~76 MB)
```

- **Cada pasta = uma classe (label)**. Ex.: `train/air hockey/001.jpg` → label `air hockey`.
- Todas as imagens são **JPEG 224×224 pixels, RGB (3 canais)**.

`sports.csv` descreve o dataset em formato tabular:

```csv
class id,filepaths,labels,data set
0,train/air hockey/001.jpg,air hockey,train
0,train/air hockey/002.jpg,air hockey,train
```

| coluna      | significado                                              |
|-------------|----------------------------------------------------------|
| `class id`  | o número da classe (0 a 99) → é o índice no one-hot      |
| `filepaths` | caminho da imagem                                        |
| `labels`    | nome da classe                                           |
| `data set`  | `train`, `valid` ou `test`                               |

### Por que três divisões?

| Pasta   | Para que serve                                                                 | Analogia             |
|---------|--------------------------------------------------------------------------------|----------------------|
| `train` | a rede aprende ajustando os pesos com essas imagens                            | estudar a matéria    |
| `valid` | medir o desempenho **durante** o treino e ajustar parâmetros (épocas, etc.)    | simulado             |
| `test`  | prova final, com imagens que o modelo **nunca viu**                            | prova de verdade     |

Se avaliarmos o modelo com as mesmas imagens do treino, ele pode só ter **decorado** (overfitting). O `test` mostra se ele realmente **generaliza**.

### O arquivo `.h5`

`EfficientNetB0-100-(224 X 224)- 98.40.h5` é um modelo **Keras (Python)** que o autor do dataset já treinou:

- **EfficientNetB0** → arquitetura da rede (uma CNN — rede convolucional, especializada em imagens)
- **100** → 100 classes
- **224 X 224** → tamanho da imagem de entrada
- **98.40** → acurácia de 98,40% no conjunto de teste

Serve como **referência** de até onde dá para chegar. Para usar em JS: `tensorflowjs_converter --input_format keras modelo.h5 ./modelo_tfjs` (pacote Python `tensorflowjs`).

---

## 3. Passo a passo no Teachable Machine

1. Acessar o site → **Get Started** → **Image Project** → **Standard image model**.
2. Criar uma **classe** para cada esporte (renomear `Class 1`, `Class 2`...).
3. Em cada classe: **Upload** → arrastar as imagens de `archive/train/<esporte>/`.
4. Clicar em **Train Model** (não trocar de aba durante o treino — ele roda no navegador).
5. **Preview**: testar com imagens de `archive/test/<esporte>/` (que o modelo não viu no treino).
6. **Export Model** → TensorFlow.js / TensorFlow (Keras) / TensorFlow Lite.

💡 **Dica prática:** comece com **3 a 5 esportes** visualmente diferentes (ex.: `swimming`, `basketball`, `formula 1 racing`). Subir as 13 mil imagens das 100 classes no navegador é lento e pesado. Depois teste com esportes parecidos (ex.: `figure skating men` × `figure skating pairs`) para ver o modelo se confundir.

### Configurações avançadas (Advanced)

São os mesmos conceitos do `model.fit()` do [`exemplo-00`](./exemplo-00/index.js):

| Teachable Machine | Padrão | No código (tfjs)               | O que é                                             |
|-------------------|--------|--------------------------------|-----------------------------------------------------|
| Epochs            | 50     | `epochs: 100`                  | quantas vezes a rede vê todas as imagens            |
| Batch Size        | 16     | `batchSize` (padrão 32)        | quantas imagens antes de ajustar os pesos           |
| Learning Rate     | 0.001  | taxa do `optimizer: 'adam'`    | tamanho do passo de ajuste dos pesos                |

Em **Advanced → Under the hood** dá para ver os gráficos de **accuracy** e **loss** por época — os mesmos números que o callback `onEpochEnd` mostraria no código.

O Teachable Machine também separa sozinho **15% das imagens** que você subiu para validação (o papel da pasta `valid/`).

---

## 4. Correlação com redes neurais (o que ele faz por baixo)

O Teachable Machine esconde exatamente os passos estudados em [conceito-redes-neurais.md](./conceito-redes-neurais.md):

| Etapa                      | `exemplo-00` (pessoas)                          | Teachable Machine (esportes)                                  |
|----------------------------|-------------------------------------------------|---------------------------------------------------------------|
| Dado bruto                 | `{ idade: 30, cor: "azul", ... }`               | um arquivo `.jpg`                                             |
| Virar tensor               | `[30, 1, 0, 0, 1, 0, 0]` → shape `[7]`          | pixels RGB → shape `[224, 224, 3]` (150.528 números)           |
| Normalização               | Min-Max na idade → `0 a 1`                      | pixels `0–255` → `-1 a 1` (`pixel / 127.5 - 1`)                 |
| Labels (one-hot)           | `premium = [1, 0, 0]`                           | a pasta/classe → `swimming = [0, 0, 1, 0, ...]`               |
| Camadas                    | `dense(80, relu)` → `dense(3, softmax)`         | **MobileNet** (pré-treinada) → `dense(n_classes, softmax)`     |
| Loss                       | `categoricalCrossentropy`                       | `categoricalCrossentropy`                                     |
| Treino                     | `model.fit(..., { epochs: 100 })`               | botão **Train Model** (epochs: 50)                            |
| Saída                      | `basic: 82.12%`                                 | `swimming: 97%` (barras no Preview)                           |

Uma imagem é só um **tensor 3D**: altura × largura × 3 canais (vermelho, verde, azul), cada valor sendo o brilho do pixel de 0 a 255.

### Transfer Learning: por que treina tão rápido?

Treinar uma rede de imagens do zero exigiria milhões de fotos e muitas horas. O Teachable Machine usa **transfer learning**:

```
          ┌──────────────────────────────────┐      ┌───────────────────────┐
imagem →  │ MobileNet (pesos CONGELADOS)     │  →   │ camada dense NOVA     │ → softmax
224x224x3 │ já treinada no ImageNet (milhões │      │ (só ela é treinada    │   [% por esporte]
          │ de imagens): sabe ver bordas,    │      │ com as SUAS imagens)  │
          │ texturas, formas, objetos        │      │                       │
          └──────────────────────────────────┘      └───────────────────────┘
            "extrator de características"              "classificador"
```

- A MobileNet já "sabe enxergar" — reaproveitamos esse conhecimento.
- Só a última camada é treinada para mapear essas características para **as nossas classes**.
- Por isso bastam dezenas de imagens por classe e alguns segundos de treino.
- A última camada é praticamente o mesmo `dense + softmax` do `exemplo-00`.

O `.h5` do `archive/` usa a mesma ideia com a **EfficientNetB0** no lugar da MobileNet.

---

## 5. Usando o modelo exportado no backend (Node.js)

Exportando em **TensorFlow.js → Download**, vem um `.zip` com:

- `model.json` → a arquitetura (as camadas)
- `weights.bin` → os pesos aprendidos
- `metadata.json` → os nomes das classes (`labels`) na ordem da saída

Exemplo de inferência com o mesmo `@tensorflow/tfjs-node` do `exemplo-00`:

```javascript
import tf from '@tensorflow/tfjs-node'
import fs from 'node:fs/promises'

const model = await tf.loadLayersModel('file://./modelo-esportes/model.json')
const { labels } = JSON.parse(await fs.readFile('./modelo-esportes/metadata.json', 'utf8'))

async function classificar(caminhoImagem) {
    const buffer = await fs.readFile(caminhoImagem)

    // tf.tidy libera da memória os tensores intermediários
    const probs = tf.tidy(() => {
        const imagem = tf.node.decodeImage(buffer, 3)           // [h, w, 3] valores 0-255
            .resizeBilinear([224, 224])                          // mesmo tamanho do treino
            .toFloat()
            .div(127.5).sub(1)                                   // normalização: -1 a 1
            .expandDims(0)                                       // vira um lote: [1, 224, 224, 3]

        return model.predict(imagem).dataSync()
    })

    return labels
        .map((label, i) => ({ label, prob: (probs[i] * 100).toFixed(2) + '%' }))
        .sort((a, b) => parseFloat(b.prob) - parseFloat(a.prob))
}

console.log(await classificar('../archive/test/swimming/1.jpg'))
```

Repare: é o **mesmo fluxo** do `exemplo-00` — preparar o dado do mesmo jeito do treino → `tensor` → `model.predict()` → mapear as probabilidades para os nomes das labels. A diferença é só o tipo do dado (imagem em vez de pessoa).

Daria para expor isso numa rota `POST /classificar` (upload de imagem) e ter uma API de reconhecimento de esportes.

---

## 6. Limitações e boas práticas

- **Qualidade > quantidade, mas variedade importa**: fotos com ângulos, luzes e fundos diferentes. Se todas as fotos de `swimming` forem azuis, o modelo pode aprender "azul = natação" (viés).
- **Classes balanceadas**: quantidades parecidas de imagens por classe, senão o modelo puxa para a classe com mais exemplos.
- **Ele só conhece as classes que viu**: uma foto de gato será classificada como *algum* esporte, porque o softmax sempre distribui 100% entre as classes existentes. Uma saída "não sei" precisa de uma classe extra ou de um limite mínimo de confiança.
- **Teachable Machine é para prototipar**: ótimo para validar uma ideia rápido. Para 100 classes / 13 mil imagens, o caminho é treinar em código (Python/Keras ou tfjs-node) — como foi feito com o `.h5`.

---

## Para praticar

- [ ] Treinar com 3 esportes diferentes e testar com `archive/test/`
- [ ] Treinar com 2 esportes parecidos e observar a confusão
- [ ] Mudar epochs (10 × 50 × 100) e comparar os gráficos de accuracy/loss
- [ ] Exportar em TensorFlow.js e rodar o script da seção 5
