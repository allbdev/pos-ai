// ====================================================================
// 🔮 Classifier: carrega os modelos treinados e prevê o tipo de um Pokémon
// ====================================================================
// Usa o que o train.js salvou em model/<nome>/ (um por conjunto de features):
//  - model.json + weights.bin: a rede neural
//  - metadata.json: o conjunto de features, o contexto (min/max, golpes)
//    e a lista de classes (tipos)

import { readFile } from 'node:fs/promises'
import tf from './tf.js'
import { STAT_NAMES } from './context.js'
import { FEATURE_SETS } from './encode.js'

const MODELS_DIR = new URL('../model/', import.meta.url)

/**
 * Valida o corpo da requisição. Mesmo formato do data/pokemon.json:
 *   { stats: { hp, attack, ... }, height, weight, moves? }
 *
 * @param {Object<string, string>} moveTypes mapa "golpe -> tipo" do modelo,
 *   usado para recusar golpes que não existem (ex.: erro de digitação)
 * @returns uma mensagem de erro, ou null se estiver tudo certo
 */
export function validateInput(body, moveTypes) {
    const isValidNumber = value => typeof value === 'number' && Number.isFinite(value) && value >= 0

    if (!body || typeof body.stats !== 'object' || body.stats === null) {
        return 'o corpo precisa ter um objeto "stats"'
    }

    const missingStats = STAT_NAMES.filter(name => !isValidNumber(body.stats[name]))
    if (missingStats.length) {
        return `stats inválidos ou faltando: ${missingStats.join(', ')}`
    }

    if (!isValidNumber(body.height) || !isValidNumber(body.weight)) {
        return '"height" e "weight" precisam ser números >= 0'
    }

    // moves é opcional: sem ele, a requisição já está válida
    if (body.moves === undefined) return null

    const isListOfNames = Array.isArray(body.moves) &&
        body.moves.length > 0 &&
        body.moves.every(move => typeof move === 'string')

    if (!isListOfNames) {
        return '"moves" precisa ser uma lista não vazia de nomes de golpes, ex.: ["thunderbolt", "quick-attack"]'
    }

    // Recusa golpes desconhecidos em vez de ignorar em silêncio:
    // um erro de digitação ("thunderbot") mudaria a previsão sem ninguém perceber.
    // Object.hasOwn pelo mesmo motivo do server.js ("toString" não é um golpe).
    const unknownMoves = body.moves.filter(move => !Object.hasOwn(moveTypes, move))
    if (unknownMoves.length) {
        return `golpes desconhecidos: ${unknownMoves.join(', ')} (use os nomes da PokéAPI, ex.: "quick-attack")`
    }

    return null
}

/**
 * Carrega UM modelo (model/<nome>/) e devolve sua função predict.
 */
async function loadModel(name) {
    const modelDir = new URL(`${name}/`, MODELS_DIR)

    const model = await tf.loadLayersModel(new URL('model.json', modelDir).href)
    const { featureSet, classes, context } = JSON.parse(
        await readFile(new URL('metadata.json', modelDir), 'utf8')
    )

    // O mesmo encode usado no treino deste modelo (stats ou stats-moves)
    const encode = FEATURE_SETS[featureSet]

    /**
     * @returns os `top` tipos mais prováveis, ex.:
     *   [{ type: 'water', probability: 0.41 }, { type: 'ice', probability: 0.22 }, ...]
     */
    function predict(pokemon, top = 3) {
        // Mesmo encode do treino, com o MESMO contexto salvo no treino.
        // Se usássemos outro min/max, os números teriam outra escala
        // e a rede receberia algo diferente do que aprendeu.
        const input = tf.tensor2d([encode(pokemon, context)])

        // Saída: 1 linha com 18 probabilidades (uma por tipo), somando 1
        const output = model.predict(input)
        const probabilities = output.dataSync()

        // O TensorFlow não libera a memória dos tensores sozinho.
        // Num servidor que roda por dias, sem isso a memória só cresce.
        input.dispose()
        output.dispose()

        // Junta cada probabilidade com o nome do tipo da mesma posição
        // e ordena da mais provável para a menos provável
        return Array.from(probabilities)
            .map((probability, index) => ({ type: classes[index], probability }))
            .sort((a, b) => b.probability - a.probability)
            .slice(0, top)
    }

    return { predict, moveTypes: context.moveTypes }
}

/**
 * Carrega TODOS os modelos UMA vez (no início do servidor).
 * Carregar a cada requisição seria lento: ler arquivos e montar a rede de novo.
 *
 * Se um modelo não existir (ex.: ainda não rodou npm run train), ele só
 * fica de fora: o servidor sobe com os que encontrou.
 *
 * @returns {Promise<Object<string, { predict: Function, moveTypes: Object }>>}
 *   ex.: { 'stats': {...}, 'stats-moves': {...} }
 */
export async function loadClassifiers() {
    const classifiers = {}

    for (const name of Object.keys(FEATURE_SETS)) {
        try {
            classifiers[name] = await loadModel(name)
        } catch (error) {
            console.warn(`⚠️  Modelo "${name}" não encontrado (rode "npm run train"): ${error.message}`)
        }
    }

    return classifiers
}
