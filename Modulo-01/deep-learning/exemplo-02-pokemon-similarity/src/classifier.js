// ====================================================================
// 🔮 Classifier: carrega o modelo treinado e prevê o tipo de um Pokémon
// ====================================================================
// Usa o que o train.js salvou em model/:
//  - model.json + weights.bin: a rede neural
//  - metadata.json: o contexto (min/max) e a lista de classes (tipos)

import { readFile } from 'node:fs/promises'
import tf from './tf.js'
import { STAT_NAMES } from './context.js'
import { encodeStats } from './encode.js'

const MODEL_DIR = new URL('../model/', import.meta.url)

/**
 * Valida o corpo da requisição. Mesmo formato do data/pokemon.json:
 *   { stats: { hp, attack, ... }, height, weight }
 * Retorna uma mensagem de erro, ou null se estiver tudo certo.
 */
export function validateInput(body) {
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

    return null
}

/**
 * Carrega o modelo UMA vez (no início do servidor) e devolve uma função predict.
 * Carregar a cada requisição seria lento: ler arquivos e montar a rede de novo.
 */
export async function loadClassifier() {
    const model = await tf.loadLayersModel(new URL('model.json', MODEL_DIR).href)
    const { classes, context } = JSON.parse(
        await readFile(new URL('metadata.json', MODEL_DIR), 'utf8')
    )

    /**
     * @returns os `top` tipos mais prováveis, ex.:
     *   [{ type: 'water', probability: 0.41 }, { type: 'ice', probability: 0.22 }, ...]
     */
    function predict(pokemon, top = 3) {
        // Mesmo encode do treino, com o MESMO contexto salvo no treino.
        // Se usássemos outro min/max, os números teriam outra escala
        // e a rede receberia algo diferente do que aprendeu.
        const input = tf.tensor2d([encodeStats(pokemon, context)])

        // Saída: 1 linha com 15 probabilidades (uma por tipo), somando 1
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

    return predict
}
