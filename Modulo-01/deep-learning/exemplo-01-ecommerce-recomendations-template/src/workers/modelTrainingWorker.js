import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

// Este arquivo roda em um Web Worker: o treino é pesado e bloquearia a UI
// se rodasse na thread principal. A comunicação com a tela é feita via postMessage.
console.log('Model training worker initialized');

// Estado do worker: o contexto (min/max, índices, vetores dos produtos)
// e o modelo treinado. Ambos são reutilizados pelo recommend().
let _globalCtx = {};
let _model = null;

// Importância de cada feature na recomendação (soma = 1).
// Categoria pesa mais que cor, que pesa mais que preço, que pesa mais que idade.
// Os pesos multiplicam as features ANTES de irem para a rede, então:
//  - a rede começa o treino com uma "dica" do que importa mais;
//  - os vetores dos produtos ficam prontos para busca por similaridade
//    (ex.: banco vetorial), onde a escala de cada feature faz diferença.
const WEIGHTS = {
    category: .4,
    color: .3,
    price: .2,
    age: .1
}

// Normalização min-max: coloca qualquer valor na escala 0–1.
// Ex.: preço 50 com min 10 e max 210 => (50 - 10) / (210 - 10) = 0.2
// Sem isso, o preço (centenas) "gritaria" mais alto que a idade (dezenas)
// só por causa da escala, não por ser mais importante.
// O "|| 1" evita divisão por zero quando min === max.
const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

/**
 * Monta o "contexto": tudo que precisa ser calculado UMA vez a partir dos dados
 * para depois conseguir transformar qualquer usuário/produto em números
 * (limites para normalizar, índices para one-hot, tamanho dos vetores).
 *
 * @param {Array<{id: number, name: string, category: string, price: number, color: string}>} products
 * @param {Array<{id: number, name: string, age: number, purchases: Array<{id: number, name: string, category: string, price: number, color: string}>}>} users
 */
function makeContext(products, users) {
    const ages = users.map(u => u.age)
    const prices = products.map(p => p.price)

    const maxAge = Math.max(...ages)
    const minAge = Math.min(...ages)

    const maxPrice = Math.max(...prices)
    const minPrice = Math.min(...prices)

    // Lista de valores únicos de cor e categoria (o Set remove duplicados)
    const colors = [...new Set(products.map(p => p.color))]
    const categories = [...new Set(products.map(p => p.category))]

    // Mapa "valor -> posição", usado no one-hot encoding.
    // Ex.: { preto: 0, prata: 1, azul: 2, ... }
    // A rede não entende "preto", mas entende "posição 0 ligada".
    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        })
    )

    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        })
    )

    // Computar a média de idade dos compradores por produto ajuda a personalizar
    // Ex.: se só pessoas de ~20 anos compram um tênis, esse tênis "tem cara" de 20 anos,
    // e a rede consegue relacionar a idade do usuário com a idade típica do produto.
    // Produtos que ninguém comprou recebem a idade do meio (valor neutro).
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(purchase => {
            ageSums[purchase.name] = (ageSums[purchase.name] || 0) + user.age
            ageCounts[purchase.name] = (ageCounts[purchase.name] || 0) + 1
        })
    })

    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ? ageSums[product.name] / ageCounts[product.name] : midAge
            const norm = normalize(avg, minAge, maxAge)

            return [product.name, norm]
        })
    )

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        productAvgAgeNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // Tamanho do vetor de um produto (e de um usuário):
        // 1 (preço) + 1 (idade) + N categorias + N cores
        dimensions: 2 + categories.length + colors.length
    }
}

// One-hot encoding com peso: transforma uma categoria em um vetor com uma
// única posição "ligada". Ex.: index 2 de 4, peso 0.4 => [0, 0, 0.4, 0]
// Usamos one-hot (e não 0, 1, 2, 3) para a rede não achar que
// "categoria 3" é maior ou melhor que "categoria 1" — não existe ordem entre elas.
const oneHotWeighted = (index, length, weigth) => tf.oneHot(index, length).cast('float32').mul(weigth)

// Transforma um produto em um vetor numérico:
// [preço, idade média dos compradores, ...categoria one-hot, ...cor one-hot]
function encodeProduct(product, context) {
    // Normalizando dados para ficar de 0 a 1 e
    // aplicar o peso na recomendação
    const price = tf.tensor1d([
        normalize(product.price, context.minPrice, context.maxPrice) * WEIGHTS.price
    ])

    const age = tf.tensor1d([
        (context.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age
    ])

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    )

    return tf.concat1d(
        [price, age, category, color]
    )
}

// Transforma um usuário em um vetor do MESMO formato do produto.
// O usuário é representado pelo que ele compra: a média dos vetores
// das compras dele é o "perfil de gosto" (ex.: 70% acessórios, 50% preto...).
function encodeUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(
            user.purchases.map(product => {
                return encodeProduct(product, context)
            })
        )
            .mean(0)
            .reshape([
                1,
                context.dimensions
            ])
    }

    // Cold start: usuário sem compras. A única informação que temos é a idade,
    // então o resto do vetor fica zerado e a recomendação se apoia só nela.
    return tf.concat1d(
        [
            tf.zeros([1]), // preço é ignorado
            tf.tensor1d([
                normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age
            ]), //
            tf.zeros([context.numCategories]), // categoria é ignorada
            tf.zeros([context.numColors]) // cor é ignorada
        ]
    ).reshape([
        1,
        context.dimensions
    ])
}

// Monta o dataset de treino: cada par (usuário, produto) vira um exemplo.
// - input: [...vetor do usuário, ...vetor do produto]
// - label: 1 se o usuário comprou o produto, 0 se não comprou
// Ou seja, a rede aprende a responder: "esse usuário compraria esse produto?"
// Com 5 usuários e 10 produtos são só 50 exemplos (12 com label 1).
function createTrainingData(context) {
    const inputs = []
    const labels = []
    context.users
        // usuários sem compras não ensinam nada sobre gosto
        .filter(user => user.purchases.length)
        .forEach(user => {
            const userVector = encodeUser(user, context).dataSync()
            context.products.forEach(product => {
                const productVector = encodeProduct(product, context).dataSync()

                // O "? 1 : 0" fica fora do some(): o label precisa ser número, não boolean
                const label = user.purchases.some(purchase => purchase.name === product.name) ? 1 : 0

                // combinar user + product
                inputs.push([...userVector, ...productVector])
                labels.push(label)
            })
        })
    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimension: context.dimensions * 2
        // tamanho = userVector + productVector
    }
}

// ====================================================================
// 📌 Exemplo de como um usuário é ANTES da codificação
// ====================================================================
/*
const exampleUser = {
    id: 201,
    name: 'Rafael Souza',
    age: 27,
    purchases: [
        { id: 8, name: 'Boné Estiloso', category: 'acessórios', price: 39.99, color: 'preto' },
        { id: 9, name: 'Mochila Executiva', category: 'acessórios', price: 159.99, color: 'cinza' }
    ]
};
*/

// ====================================================================
// 📌 Após a codificação, o modelo NÃO vê nomes ou palavras.
// Ele vê um VETOR NUMÉRICO (todos normalizados entre 0–1).
// Exemplo: [preço_normalizado, idade_normalizada, cat_one_hot..., cor_one_hot...]
//
// Suponha categorias = ['acessórios', 'eletrônicos', 'vestuário']
// Suponha cores      = ['preto', 'cinza', 'azul']
//
// Para Rafael (idade 27, categoria: acessórios, cores: preto/cinza),
// a categoria/cor de cada compra passa por 3 etapas:
//
//                    Boné (preto)       Mochila (cinza)
// 1. one-hot puro    cor [1, 0, 0]      cor [0, 1, 0]      (só 0 ou 1)
// 2. * peso (0.3)    cor [0.3, 0, 0]    cor [0, 0.3, 0]    (oneHotWeighted)
// 3. média (.mean)   cor [0.15, 0.15, 0]                   (encodeUser)
//
// Por isso o vetor final do usuário NÃO tem só 0 e 1:
//
// [
//   0.09,              // preço normalizado (média) * 0.2
//   0.06,              // idade média dos compradores (normalizada) * 0.1
//   0.4, 0, 0,         // categoria: as duas compras são acessórios => 1 * 0.4
//   0.15, 0.15, 0      // cor: metade preto, metade cinza => 0.5 * 0.3 cada
// ]
//
// Repare que a média gera "porcentagens": 0.15 em preto e cinza significa
// que metade das compras é de cada cor. É assim que o vetor guarda o gosto.
//
// São esses números que vão para a rede neural.
// ====================================================================



// ====================================================================
// 🧠 Configuração e treinamento da rede neural
// ====================================================================
async function configureNeuralNetAndTrain(trainData) {
    const model = tf.sequential()

    // Camada de entrada
    // - inputShape: Número de features por exemplo de treino (trainData.inputDim)
    //   Exemplo: Se o vetor produto + usuário = 20 números, então inputDim = 20
    // - units: 128 neurônios (muitos "olhos" para detectar padrões)
    // - activation: 'relu' (mantém apenas sinais positivos, ajuda a aprender padrões não-lineares)
    model.add(tf.layers.dense({ inputShape: [trainData.inputDimension], units: 128, activation: 'relu' }))

    // Camada oculta 1
    // - 64 neurônios (menos que a primeira camada: começa a comprimir informação)
    // - activation: 'relu' (ainda extraindo combinações relevantes de features)
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }))

    // Camada oculta 2
    // - 32 neurônios (mais estreita de novo, destilando as informações mais importantes)
    //   Exemplo: De muitos sinais, mantém apenas os padrões mais fortes
    // - activation: 'relu'
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }))

    // Camada de saída
    // - 1 neurônio porque vamos retornar apenas uma pontuação de recomendação
    // - activation: 'sigmoid' comprime o resultado para o intervalo 0–1
    //   Exemplo: 0.9 = recomendação forte, 0.1 = recomendação fraca
    model.add(
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
    )

    model.compile({
        // Adam: ajusta os pesos da rede a cada passo, aprendendo com o histórico
        // de erros. 0.01 é a taxa de aprendizado (o tamanho de cada ajuste).
        optimizer: tf.train.adam(0.01),
        // binaryCrossentropy: a função de erro certa para "sim ou não" (comprou / não comprou).
        // Pune mais forte quando a rede tem certeza e erra.
        loss: 'binaryCrossentropy',
        // Cuidado: com poucos 1s no dataset, a acurácia engana.
        // Responder sempre "não compra" já acertaria ~76% aqui.
        metrics: ['accuracy']
    })

    await model.fit(trainData.xs, trainData.ys, {
        // epochs: quantas vezes a rede vê o dataset inteiro
        epochs: 100,
        // batchSize: quantos exemplos ela vê antes de ajustar os pesos
        // (com 50 exemplos, são 2 ajustes por época)
        batchSize: 32,
        // shuffle: embaralha a ordem a cada época para a rede não decorar a sequência
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    })

    return model
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    const products = await (await fetch('/data/products.json')).json()

    const context = makeContext(products, users)

    // Pré-calcula o vetor de cada produto uma vez só.
    // Na recomendação, só o vetor do usuário muda; os dos produtos são reaproveitados.
    context.productVectors = products.map(product => ({
        name: product.name,
        meta: {
            ...product
        },
        vector: encodeProduct(product, context).dataSync()
    }))

    _globalCtx = context

    const trainData = createTrainingData(context)

    _model = await configureNeuralNetAndTrain(trainData)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });


}

function recommend({ user }) {
    if (!_model) return

    const context = _globalCtx

    // 1️⃣ Converta o usuário fornecido no vetor de features codificadas
    //    (preço ignorado, idade normalizada, categorias ignoradas)
    //    Isso transforma as informações do usuário no mesmo formato numérico
    //    que foi usado para treinar o modelo.
    const userVector = encodeUser(user, context).dataSync()


    // Em aplicações reais:
    //  Armazene todos os vetores de produtos em um banco de dados vetorial (como Postgres, Neo4j ou Pinecone)
    //  Consulta: Encontre os 200 produtos mais próximos do vetor do usuário
    //  Execute _model.predict() apenas nesses produtos

    // 2️⃣ Crie pares de entrada: para cada produto, concatene o vetor do usuário
    //    com o vetor codificado do produto.
    //    Por quê? O modelo prevê o "score de compatibilidade" para cada par (usuário, produto).
    const input = context.productVectors.map(({ vector }) => {
        return [...userVector, ...vector]
    })

    // 3️⃣ Converta todos esses pares (usuário, produto) em um único Tensor.
    //    Formato: [numProdutos, inputDim]
    const inputTensor = tf.tensor2d(input)

    // 4️⃣ Rode a rede neural treinada em todos os pares (usuário, produto) de uma vez.
    //    O resultado é uma pontuação para cada produto entre 0 e 1.
    //    Quanto maior, maior a probabilidade do usuário querer aquele produto.
    const predictions = _model.predict(inputTensor)

    // 5️⃣ Extraia as pontuações para um array JS normal.
    const scores = predictions.dataSync()
    const recommendations = context.productVectors.map((item, index) => {
        return {
            ...item.meta,
            name: item.name,
            score: scores[index] // previsão do modelo para este produto
        }
    })

    // 6️⃣ Ordene do maior score para o menor: o topo da lista é a melhor recomendação.
    //    Obs.: produtos que o usuário já comprou também entram na lista
    //    (e costumam ter score alto, porque foram label 1 no treino).
    const sortedItems = recommendations
        .sort((a, b) => b.score - a.score)

    // 7️⃣ Envie a lista ordenada de produtos recomendados
    //    para a thread principal (a UI pode exibi-los agora).
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: recommend,
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
