// ====================================================================
// 🧠 Train: rede neural que tenta adivinhar o tipo primário pelos stats
// ====================================================================
// Entrada: 6 stats + altura + peso, normalizados (encodeStats) → 8 números
// Saída:   probabilidade de cada tipo primário (softmax) → 15 números
//
// Rodar: npm run train
// Gera:  model/model.json + model/weights.bin (a rede treinada)
//        model/metadata.json (contexto + lista de tipos, necessários para prever)

import tf from './tf.js'
import { readFile, writeFile } from 'node:fs/promises'
import { makeContext } from './context.js'
import { encodeStats } from './encode.js'

const DATA_FILE = new URL('../data/pokemon.json', import.meta.url)
const MODEL_DIR = new URL('../model/', import.meta.url)

// 20% dos Pokémon ficam de fora do treino para medir se a rede
// aprendeu de verdade ou só decorou (overfitting).
const VALIDATION_SPLIT = 0.2

// Hiperparâmetros: configurações que NÓS escolhemos (a rede não aprende).
// Ficam aqui em cima para facilitar os experimentos.
const MAX_EPOCHS = 100

// Fração de neurônios desligados a cada passo do treino (0 = sem dropout)
// Early stopping: para o treino se o val_loss não melhorar por N épocas seguidas
//
// Valores escolhidos medindo (média de 10 treinos, todas as gerações, ver README):
// sem dropout e sem early stopping teve a MELHOR acurácia de validação (22.6%).
// Com 1025 Pokémon a rede quase não decora (acc de treino ~33%), então
// regularizar só atrapalha. Para testar de novo: DROPOUT_RATE = 0.3, PATIENCE = 10.
const DROPOUT_RATE = 0

// PATIENCE = MAX_EPOCHS na prática desliga o early stopping
const PATIENCE = 100

// Camadas escondidas: um número por camada = quantidade de neurônios.
// [32, 16] = duas camadas, a primeira com 32 neurônios e a segunda com 16.
// Mais neurônios/camadas = mais "capacidade" para aprender padrões,
// mas também mais facilidade para decorar (overfitting).
//
// Testamos [64, 32], [128, 64] e [128, 64, 32] (ver README): a acurácia de treino
// subiu (até 49%), mas a de validação ficou nos mesmos ~21%. Ou seja, a rede maior
// só decorou mais. Ficamos com a menor: mesmo resultado, mais simples e rápida.
const HIDDEN_LAYERS = [32, 16]

/**
 * Embaralha uma cópia do array (algoritmo Fisher-Yates).
 *
 * Por que embaralhar ANTES de separar treino/validação?
 * O validationSplit do TensorFlow pega sempre os ÚLTIMOS 20% do array, sem embaralhar.
 * Nosso arquivo está em ordem de Pokédex, então a validação seria só a última
 * geração e meia (ids ~821–1025), e a rede seria testada em Pokémon de
 * gerações que nunca viu no treino.
 */
function shuffle(array) {
    const copy = [...array]
    for (let i = copy.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[copy[i], copy[j]] = [copy[j], copy[i]]
    }
    return copy
}

/**
 * Baseline: o que um "modelo burro" acertaria?
 * Ele chuta sempre o tipo mais comum do TREINO e a gente mede quanto
 * isso acerta na VALIDAÇÃO. A rede só é útil se for melhor que isso.
 */
function baselineAccuracy(trainSet, validationSet) {
    const counts = {}
    trainSet.forEach(p => counts[p.types[0]] = (counts[p.types[0]] || 0) + 1)

    const [mostCommonType] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]
    const hits = validationSet.filter(p => p.types[0] === mostCommonType).length

    return { mostCommonType, accuracy: hits / validationSet.length }
}

function buildModel(numInputs, numClasses) {
    const model = tf.sequential()

    // Uma camada densa (+ dropout) para cada número em HIDDEN_LAYERS.
    // Só a primeira camada precisa do inputShape (o tamanho da entrada);
    // as seguintes descobrem o tamanho pela camada anterior.
    HIDDEN_LAYERS.forEach((units, index) => {
        model.add(tf.layers.dense({
            ...(index === 0 && { inputShape: [numInputs] }),
            units,
            activation: 'relu'
        }))

        // Dropout: a cada passo do treino, "desliga" uma fração dos neurônios aleatoriamente.
        // A rede não pode depender de um neurônio específico para lembrar de um Pokémon,
        // então é forçada a aprender padrões mais gerais (combate o overfitting).
        // Só age no treino: no predict(), todos os neurônios ficam ligados.
        model.add(tf.layers.dropout({ rate: DROPOUT_RATE }))
    })

    // Saída: 1 neurônio por tipo primário.
    // softmax transforma a saída em probabilidades que somam 1.
    // Ex.: { water: 0.6, ice: 0.3, normal: 0.1 } => a rede "acha" que é water.
    // (Na aula era 1 neurônio + sigmoid, porque a resposta era só sim/não.)
    model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }))

    model.compile({
        optimizer: tf.train.adam(0.01),
        // categoricalCrossentropy: a versão do binaryCrossentropy
        // para quando a resposta é UMA entre várias classes.
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    return model
}

async function main() {
    const pokemons = JSON.parse(await readFile(DATA_FILE, 'utf8'))
    const context = makeContext(pokemons)

    // Classes = tipos que aparecem como PRIMÁRIOS (18 com todas as gerações; eram 15 só na Gen 1).
    // Ordem alfabética, como no context.js: a posição 0 da saída é sempre o mesmo tipo.
    const classes = [...new Set(pokemons.map(p => p.types[0]))].sort()

    const shuffled = shuffle(pokemons)

    // Mesmo corte que o validationSplit vai fazer, para calcular o baseline
    // exatamente nos mesmos Pokémon que a rede vai usar na validação.
    const splitIndex = Math.floor(shuffled.length * (1 - VALIDATION_SPLIT))
    const trainSet = shuffled.slice(0, splitIndex)
    const validationSet = shuffled.slice(splitIndex)

    // xs: um vetor de 8 números por Pokémon (sem os tipos: é isso que queremos prever!)
    const xs = tf.tensor2d(shuffled.map(p => encodeStats(p, context)))

    // ys: one-hot do tipo primário. Ex.: water => [0, 0, ..., 1]
    // Aqui é one-hot de verdade (só uma posição), porque a resposta é UM tipo.
    const labels = shuffled.map(p => classes.indexOf(p.types[0]))
    const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), classes.length)

    const model = buildModel(xs.shape[1], classes.length)

    console.log(`Treino: ${trainSet.length} Pokémon | Validação: ${validationSet.length} | Classes: ${classes.length}\n`)

    // Formata uma linha de log: compare acc (treino) com val_acc (validação).
    // Se acc sobe e val_acc não acompanha, a rede está decorando.
    const formatEpoch = (label, epoch, logs) =>
        `${label} ${String(epoch + 1).padStart(3)} | ` +
        `loss ${logs.loss.toFixed(3)} acc ${logs.acc.toFixed(2)} | ` +
        `val_loss ${logs.val_loss.toFixed(3)} val_acc ${logs.val_acc.toFixed(2)}`

    let lastEpoch = { epoch: 0, logs: null }

    await model.fit(xs, ys, {
        epochs: MAX_EPOCHS,
        batchSize: 16,
        validationSplit: VALIDATION_SPLIT,
        // shuffle aqui embaralha só os dados de treino a cada época
        shuffle: true,
        // verbose 0 = sem a barra de progresso padrão, usamos o nosso log
        verbose: 0,
        callbacks: [
            // Early stopping: quando o val_loss para de melhorar, a rede começou
            // a decorar o treino. Esperamos PATIENCE épocas (o val_loss oscila)
            // e, se não melhorar, paramos.
            // Obs.: no TensorFlow.js não dá para voltar aos melhores pesos
            // (restoreBestWeights não existe), então ficamos com os pesos
            // de PATIENCE épocas depois do melhor ponto.
            tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: PATIENCE }),

            // Dentro de um array, o objeto { onEpochEnd } precisa ser embrulhado
            // em tf.CustomCallback (sozinho, fora de array, o tf faz isso por nós).
            new tf.CustomCallback({
                onEpochEnd: (epoch, logs) => {
                    lastEpoch = { epoch, logs }

                    // Loga a cada 10 épocas para não poluir o terminal
                    if ((epoch + 1) % 10 === 0) console.log(formatEpoch('Época', epoch, logs))
                }
            })
        ]
    })

    // Com early stopping, o treino pode parar numa época que não é múltipla de 10
    console.log(formatEpoch('Parou na época', lastEpoch.epoch, lastEpoch.logs))

    // Avaliação final: rede vs. chute do tipo mais comum, na validação
    const validationXs = tf.tensor2d(validationSet.map(p => encodeStats(p, context)))
    const predictions = model.predict(validationXs).argMax(1).dataSync()
    const hits = validationSet.filter((p, i) => classes[predictions[i]] === p.types[0]).length

    const baseline = baselineAccuracy(trainSet, validationSet)

    console.log(`\n📊 Validação (${validationSet.length} Pokémon que a rede nunca viu no treino)`)
    console.log(`   Rede neural:                     ${(hits / validationSet.length * 100).toFixed(1)}%`)
    console.log(`   Baseline (sempre "${baseline.mostCommonType}"):${' '.repeat(Math.max(1, 13 - baseline.mostCommonType.length))}${(baseline.accuracy * 100).toFixed(1)}%`)

    // Salva a rede (estrutura + pesos)...
    await model.save(MODEL_DIR.href.replace(/\/$/, ''))

    // ...e o que é preciso para usá-la depois. Sem isso, o servidor não saberia
    // normalizar uma entrada nova do mesmo jeito (min/max) nem traduzir
    // "posição 3 da saída" para o nome do tipo.
    await writeFile(
        new URL('metadata.json', MODEL_DIR),
        JSON.stringify({ classes, context }, null, 2)
    )

    console.log(`\n💾 Modelo salvo em model/`)
}

main()
