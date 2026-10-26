# Conceito de Redes Neurais

> Projeto prático: [`exemplo-00/index.js`](./exemplo-00/index.js) — rede neural com TensorFlow.js (Node) que classifica pessoas em `premium`, `medium` ou `basic`.
> Rodar: `cd exemplo-00 && npm start`

**Redes neurais** = um monte de operações matemáticas conectadas tentando encontrar padrões escondidos nos dados.

A máquina não entende "Erick", "azul" ou "São Paulo". Ela só entende **números**. Por isso, todo o trabalho começa transformando dados em tensores.

---

## 1. Os dados brutos

Array de pessoas:

```javascript
const pessoas = [
    {nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo"},
    {nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio"},
    {nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba"},
]
```

- `idade` → dado **numérico** (contínuo)
- `cor` e `localizacao` → dados **categóricos** (texto, uma opção entre várias)
- `nome` → não entra no modelo (é só um identificador, não ajuda a encontrar padrão)

---

## 2. Tensor

**Tensor** = vetor multidimensional, ou uma lista representada por números.

| Dimensão | Nome     | Exemplo                                 | No projeto                                  |
|----------|----------|-----------------------------------------|---------------------------------------------|
| 0D       | escalar  | `0.33`                                  | a idade normalizada de uma pessoa           |
| 1D       | vetor    | `[0.33, 1, 0, 0, 1, 0, 0]`              | uma pessoa                                  |
| 2D       | matriz   | `[[...], [...], [...]]`                 | o dataset inteiro (`tf.tensor2d`)           |
| 3D       | —        | `[altura][largura][canais RGB]`         | uma imagem (ver [teachable-machine.md](./teachable-machine.md)) |
| 4D       | —        | `[lote][altura][largura][canais]`       | um lote de imagens                          |

O formato (**shape**) do tensor de pessoas é `[3, 7]` → 3 exemplos, 7 características (*features*) cada.

```javascript
const tensorPessoas = [
    //idade, azul, vermelho, verde, SP, Rio, Curitiba
    [30, 1, 0, 0, 1, 0, 0],  // Erick
    [25, 0, 1, 0, 0, 1, 0],  // Ana
    [40, 0, 0, 1, 0, 0, 1]   // Carlos
]
```

---

## 3. One-hot encoding (dados categóricos)

`cor` e `localizacao` ficam fixados em **zero OU um** (não há meio termo). Isso se chama **one-hot encoding**: cria-se uma coluna para cada opção possível e só uma delas fica "acesa" (`1`).

| cor      | azul | vermelho | verde |
|----------|------|----------|-------|
| azul     | 1    | 0        | 0     |
| vermelho | 0    | 1        | 0     |
| verde    | 0    | 0        | 1     |

**Por que não usar `azul = 1, vermelho = 2, verde = 3`?**
Porque a rede entenderia que *verde > vermelho > azul*, ou que *verde = 3× azul*. Isso cria uma relação de ordem que não existe. O one-hot deixa todas as opções "à mesma distância" umas das outras.

> Custo: cada nova opção vira uma coluna a mais. 100 cidades = 100 colunas.

---

## 4. Normalização (dados numéricos)

A idade tem valores como 30 e 40, enquanto o resto do tensor é 0 ou 1. Isso é um problema: a rede daria **peso exagerado** à idade só porque o número é maior. Por isso normalizamos para a escala `0 → 1` (**Min-Max**):

```
dado_normalizado = (dado_original - dado_min) / (dado_max - dado_min)
```

No dataset: `idade_min = 25`, `idade_max = 40`

```
Erick  = (30 - 25) / (40 - 25) =  5 / 15 = 0.333
Ana    = (25 - 25) / (40 - 25) =  0 / 15 = 0.0
Carlos = (40 - 25) / (40 - 25) = 15 / 15 = 1.0
```

```javascript
const tensorPessoasNormalizado = [
    //idade, azul, vermelho, verde, SP, Rio, Curitiba
    [0.33, 1, 0, 0, 1, 0, 0],  // Erick
    [0,    0, 1, 0, 0, 1, 0],  // Ana
    [1,    0, 0, 1, 0, 0, 1]   // Carlos
]
```

⚠️ **Regra de ouro:** dados novos (de previsão) precisam ser normalizados com o **mesmo `min` e `max` do treino**. No projeto, o Zé tem 28 anos → `(28 - 25) / (40 - 25) = 0.2`. Se usássemos outro min/max, o número 0.2 significaria outra idade e o modelo erraria.

Em JS puro:

```javascript
const normalizar = (valor, min, max) => (valor - min) / (max - min)
normalizar(28, 25, 40) // 0.2
```

Essa transformação (dados → números normalizados) é o que permite que a máquina aprenda padrões, preveja situações e recomende produtos.

---

## 5. Labels (a resposta certa)

As categorias de saída também viram one-hot:

```javascript
const categorias = ["premium", "medium", "basico"]

const tensorCategorias = [
    [1, 0, 0], // premium - representa Erick
    [0, 1, 0], // medium  - representa Ana
    [0, 0, 1]  // basico  - representa Carlos
]
```

- **X (inputs / features)** = o que a rede recebe → `tensorPessoasNormalizado` (`inputXs`)
- **Y (outputs / labels)** = o que ela deveria responder → `tensorCategorias` (`outputYs`)

Isso é **aprendizado supervisionado**: mostramos o exemplo *e* a resposta.

---

## 6. Anatomia da rede (como está no `exemplo-00`)

```
 Entrada (7)            Camada oculta (80)             Saída (3)
 ───────────            ──────────────────             ─────────
 idade      ─┐
 azul        │          ○ ○ ○ ○ ○ ○ ○ ○                ○ premium
 vermelho    │   ──►    ○ ○ ○ ○ ○ ○ ○ ○     ──►        ○ medium
 verde       ├─         ...  (ReLU)                    ○ basic
 SP          │          ○ ○ ○ ○ ○ ○ ○ ○                (softmax)
 Rio         │
 Curitiba   ─┘
```

```javascript
const model = tf.sequential()  // camadas empilhadas, uma após a outra
model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))
model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))
```

### O neurônio

Cada neurônio faz uma conta simples:

```
saida = ativacao( (x1 * w1) + (x2 * w2) + ... + (xn * wn) + bias )
```

- **x** = valores que chegam (as features)
- **w (pesos / weights)** = o "quanto importa" cada entrada. **É isso que a rede aprende.**
- **bias** = um ajuste fino, um deslocamento
- **ativação** = função que decide o que segue adiante

**`dense`** = camada totalmente conectada: cada neurônio recebe **todas** as saídas da camada anterior.
Na 1ª camada: `7 entradas × 80 neurônios + 80 bias = 640 parâmetros`.
Na 2ª: `80 × 3 + 3 = 243`. Total: **883 números** que o treino ajusta. (Dá pra conferir com `model.summary()`.)

### Funções de ativação

| Função    | O que faz                                                       | Onde usar                                   |
|-----------|-----------------------------------------------------------------|---------------------------------------------|
| `relu`    | `max(0, x)` → positivo passa, zero/negativo vira 0 (filtro)     | camadas ocultas (padrão)                    |
| `softmax` | transforma os valores em **probabilidades que somam 100%**      | saída de classificação com várias classes   |
| `sigmoid` | espreme o valor entre 0 e 1                                     | saída de classificação sim/não (binária)    |

Sem ativação, empilhar camadas seria só uma grande multiplicação linear — a rede não conseguiria aprender padrões complexos (não-lineares).

### Quantos neurônios?

- Mais neurônios → mais capacidade de aprender padrões complexos, mas mais processamento e mais risco de **decorar** os dados (overfitting).
- No projeto, 80 é bem mais do que precisaria para 3 exemplos — mas funciona como laboratório.

---

## 7. Compilar: como a rede vai aprender

```javascript
model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
})
```

- **loss (função de perda)** = mede o **tamanho do erro**. Compara o que o modelo "acha" (ex.: `[0.6, 0.3, 0.1]`) com a resposta certa (`[1, 0, 0]`). Quanto mais longe, maior o loss.
  - `categoricalCrossentropy` → várias classes, labels em one-hot (nosso caso)
  - `binaryCrossentropy` → sim/não
  - `meanSquaredError` → regressão (prever um número, ex.: preço)
- **optimizer** = a estratégia para ajustar os pesos e diminuir o loss. O **Adam** (*Adaptive Moment Estimation*) é o "personal trainer moderno": ajusta o tamanho do passo sozinho, usando o histórico de erros.
- **metrics** = o que queremos acompanhar como humanos. `accuracy` = % de acertos.

---

## 8. Treinar: o ciclo de aprendizado

**Treinar a rede** = mostrar vários exemplos e deixar o algoritmo ajustar sozinho os pesos internos dos neurônios até acertar o máximo possível.

O que acontece em cada rodada:

```
1. Forward pass    → os dados passam pela rede e ela dá um palpite
2. Loss            → mede o quanto errou
3. Backpropagation → calcula a "culpa" de cada peso no erro (derivadas / gradiente)
4. Optimizer       → ajusta cada peso um pouquinho na direção que diminui o erro
5. Repete
```

```javascript
await model.fit(inputXs, outputYs, {
    epochs: 100,     // quantas vezes a rede vê o dataset inteiro
    shuffle: true,   // embaralha a ordem a cada época (evita viés de ordem)
    verbose: 0,
    callbacks: {
        onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch}; loss = ${logs.loss}`)
    }
})
```

Vocabulário:

- **Epoch (época)** = uma passada completa por todos os dados de treino.
- **Batch size** = quantos exemplos a rede vê antes de ajustar os pesos (padrão do tfjs: 32; nosso dataset inteiro cabe em um batch).
- **Learning rate** = o tamanho do passo de ajuste. Grande demais → não converge; pequeno demais → demora.

💡 Experimento: descomente o `onEpochEnd` no `index.js` e veja o `loss` caindo a cada época. Depois teste `epochs: 5` e compare a confiança da previsão.

---

## 9. Prever (inferência)

```javascript
const pessoaTensorNormalizado = [
    //idade, azul, vermelho, verde, SP, Rio, Curitiba
    [0.2, 0, 0, 1, 0, 0, 1]   // Zé: 28 anos, verde, Curitiba
]
const pred = model.predict(tf.tensor2d(pessoaTensorNormalizado))
```

Mesmo sendo só **uma** pessoa, o input é 2D (`[1, 7]`): o modelo sempre espera um **lote** de exemplos.

Resultado real ao rodar o projeto:

```
Predições:  [
  { label: 'premium', prob: '8.40%' },
  { label: 'medium',  prob: '9.48%' },
  { label: 'basic',   prob: '82.12%' }
]
```

O Zé é parecido com o Carlos (verde + Curitiba) → **basic**. As probabilidades somam 100% por causa do `softmax`. Os números mudam um pouco a cada execução porque os pesos começam aleatórios.

Treinando o modelo com esses dados, conseguimos categorizar terceiras pessoas com base em idade, cor e localização.

---

## 10. Cuidados e limitações

- **Poucos dados**: 3 exemplos é um brinquedo. A rede basicamente decora "verde/Curitiba = basic". Quanto mais dados (e mais variados), melhor ela generaliza.
- **Overfitting** = o modelo decora o treino e erra em dados novos. **Underfitting** = nem o treino ele aprendeu.
- **Separar os dados**: em projetos reais, divide-se em *train* (aprender), *validation* (ajustar durante o treino) e *test* (prova final com dados nunca vistos). O dataset de esportes em [`archive/`](./archive) já vem assim — ver [teachable-machine.md](./teachable-machine.md).
- **Categoria nova**: se chegar alguém de "Recife", não existe coluna para isso. O encoding precisa ser definido antes do treino.
- **Memória (tfjs)**: tensores ficam fora do garbage collector do JS. Em um backend que faz muitas previsões, use `tf.tidy(() => ...)` ou `tensor.dispose()` para não vazar memória.

---

## 11. Pensando como backend

O fluxo do `exemplo-00` é o mesmo de uma API de recomendação real:

```
Treino (offline, uma vez)          Servindo (online, a cada request)
─────────────────────────          ─────────────────────────────────
dados do banco                     POST /categorizar { idade, cor, cidade }
  → encoding + normalização          → mesmo encoding + mesma normalização
  → model.fit()                      → model.predict()
  → model.save('file://./modelo')    → { categoria: 'basic', prob: 0.82 }
```

- Salvar: `await model.save('file://./modelo')`
- Carregar no servidor: `await tf.loadLayersModel('file://./modelo/model.json')`
- Guardar junto do modelo o `min`/`max` da idade e a ordem das colunas — sem isso, não dá para preparar os dados da request igual ao treino.

---

## Observação sobre o `package.json`

O `exemplo-00/package.json` tem `"types": "module"`, mas o campo correto é **`"type": "module"`**. Funciona hoje porque o Node 22 detecta sozinho a sintaxe `import`/top-level `await`, mas o Node exibe aviso (escondido pelo `--no-warnings`) e versões mais antigas falhariam.
