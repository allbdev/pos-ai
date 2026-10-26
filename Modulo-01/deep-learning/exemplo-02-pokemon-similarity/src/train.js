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

/**
 * Embaralha uma cópia do array (algoritmo Fisher-Yates).
 *
 * Por que embaralhar ANTES de separar treino/validação?
 * O validationSplit do TensorFlow pega sempre os ÚLTIMOS 20% do array, sem embaralhar.
 * Nosso arquivo está em ordem de Pokédex, então a validação seria só os ids 122–151
 * (cheio de dragões e lendários) e a rede seria testada em tipos que quase não viu.
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

    // Rede bem menor que a da aula (128-64-32): temos só ~120 exemplos de treino.
    // Rede grande + poucos dados = decora em vez de aprender.
    model.add(tf.layers.dense({ inputShape: [numInputs], units: 32, activation: 'relu' }))
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }))

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

    // Classes = tipos que aparecem como PRIMÁRIOS (15 na Gen 1).
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

    await model.fit(xs, ys, {
        epochs: 100,
        batchSize: 16,
        validationSplit: VALIDATION_SPLIT,
        // shuffle aqui embaralha só os dados de treino a cada época
        shuffle: true,
        // verbose 0 = sem a barra de progresso padrão, usamos o nosso log
        verbose: 0,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                // Loga a cada 10 épocas para não poluir o terminal.
                // Compare acc (treino) com val_acc (validação):
                // se acc sobe e val_acc não acompanha, a rede está decorando.
                if ((epoch + 1) % 10 !== 0) return

                console.log(
                    `Época ${String(epoch + 1).padStart(3)} | ` +
                    `loss ${logs.loss.toFixed(3)} acc ${logs.acc.toFixed(2)} | ` +
                    `val_loss ${logs.val_loss.toFixed(3)} val_acc ${logs.val_acc.toFixed(2)}`
                )
            }
        }
    })

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
