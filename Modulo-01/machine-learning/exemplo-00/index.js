import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()

    // Primeira camada da rede
    // entrada de sete posicoes (idade normailzada + cores + localizacoes)

    // 80 neuronios = por que tem pouca base de treinno
    // quanto mais neuronio mais complexidade a rede pode aprender
    // e consequentemente mais processamento ela vai usar

    //A ReLU age como um filtro:
    //É como se ela deixasse somente os dados interessantes seguirem viagem na rede
    //Se a informação que chegou nesse neuronio é positiva, passa pra frente!
    // Se for zero ou negativa, podejogar fora, nao vai servir pra nada
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))

    // Saída: 3 neuronios
    // Um para cada categoria (premium, medium, basic)

    // softmax = softmax normaliza a saida em probabilidade
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    // Compilando o model
    // optimizer Adam (Adaptative Moment Estimation)
    // é um treinador pessoal moderno para redes neurais
    // ajusta os pesos de forma eficiente e inteligente
    // aprender com historico de erros e acertos

    // losss: categoricalCrossentropy
    // Ele compara o que o modelo "acha" (os scores de cada categoria)
    // com a resposta certa 
    // ex: a categoria premium sera sempre [1, 0, 0]

    // metrics: [accuracy]
    // quanto mais distante da previsao do modelo da resposta correta
    // maior o erro (loss)
    // Exemplos classicos: classificacao de imagens, recomendacao, categorizacao de usuarios
    // qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    // Treinamento do modelo
    // verbose: desabilita o log interno (e usa só o callback)
    // epochs: numero de vezes que o modelo vai ver os dados
    // shuffle: embaralha os dados, para evitar viés
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                // onEpochEnd: (epoch, logs) => {
                //     console.log(`Epoch ${epoch}; loss = ${logs.loss}`)
                // }
            }
        }
    )

    return model
}

async function predict(model, pessoaTensorNormalizado) {
    // Transformar o array js para o tensor (tfjs)
    const tfInput = tf.tensor2d(pessoaTensorNormalizado)

    // Faz a predição (output será um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()

    return predArray[0].map((value, index) => {
        return { label: labelsNomes[index], prob: (value * 100).toFixed(2) + '%' }
    })
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// Quanto mais dado melhor
// assim o algoritmo consegue entender melhor os padroes complexos dos dados
const model = await trainModel(inputXs, outputYs)

// const pessoa = { nome: 'zé', cor: 'verde', idade: 28, localizacao: 'Curitiba' }

// Normalizando a idade da pessoa usando o mesmo padrã0 do treino
// Exemplo: idade_min = 25, idade_max = 40, então (28 - 25) / (40 - 25) = 0.2

const pessoaTensorNormalizado = [
    //idade, azul, vermelho, verde, SP, Rio, Curitiba
    [0.2, 0, 0, 1, 0, 0, 1]   // Zé
]

const predictions = await predict(model, pessoaTensorNormalizado)

console.log("Predições: ", predictions)