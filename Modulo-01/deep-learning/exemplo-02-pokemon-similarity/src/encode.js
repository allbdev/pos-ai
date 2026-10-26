// ====================================================================
// 🔢 Encode: transforma um Pokémon em um vetor de números
// ====================================================================
// Mesma ideia do encodeProduct da aula (exemplo-01).
// O vetor final tem esse formato (25 números com os dados da Gen 1):
//
// [ hp, attack, defense, sp-attack, sp-defense, speed, height, weight, ...17 tipos ]
//   \____________________ 0 a 1 _____________________/ \___ 8 números ___/
//
// Diferente da aula, aqui usamos arrays JS normais em vez de tensores (tf):
//  - o pgvector recebe arrays comuns, não tensores;
//  - não precisamos nos preocupar em liberar memória (dispose);
//  - na hora de treinar, é só converter com tf.tensor2d(arrays).

import { STAT_NAMES } from './context.js'

// O que define "Pokémon parecidos"? Os pesos decidem.
// Com tipos pesando mais, os vizinhos tendem a ter o mesmo tipo.
// Com stats pesando mais, os vizinhos tendem a ter o mesmo perfil de batalha.
// Experimento sugerido: mudar esses valores e comparar os resultados.
export const WEIGHTS = {
    types: .5,
    stats: .4,
    size: .1 // altura e peso
}

// Normalização min-max (igual à aula): coloca o valor na escala 0–1.
// O "|| 1" evita divisão por zero quando min === max.
const normalize = (value, { min, max }) => (value - min) / ((max - min) || 1)

/**
 * Stats + tamanho normalizados, SEM pesos.
 * Exportada separada porque a parte 2 (classificador) usa só ela:
 * lá a rede recebe os stats e tenta adivinhar o tipo, então
 * os tipos não podem fazer parte da entrada.
 *
 * Ex. (Pikachu): [0.10, 0.39, 0.20, 0.25, 0.29, 0.56, 0.02, 0.01]
 */
export function encodeStats(pokemon, context) {
    const stats = STAT_NAMES.map(name =>
        normalize(pokemon.stats[name], context.stats[name])
    )

    const height = normalize(pokemon.height, context.height)
    const weight = normalize(pokemon.weight, context.weight)

    return [...stats, height, weight]
}

/**
 * Multi-hot dos tipos: um array com uma posição por tipo,
 * marcando 1 em TODOS os tipos do Pokémon (pode ser 1 ou 2).
 *
 * Ex. com tipos [bug, dragon, electric, ..., fire, flying, ...]:
 *   pikachu   (electric)      => [0, 0, 1, ..., 0, 0, ...]
 *   charizard (fire + flying) => [0, 0, 0, ..., 1, 1, ...]
 *
 * One-hot (aula) só permite UMA posição ligada; multi-hot permite várias.
 */
export function encodeTypes(pokemon, context) {
    const vector = new Array(context.numTypes).fill(0)

    pokemon.types.forEach(type => {
        vector[context.typesIndex[type]] = 1
    })

    return vector
}

/**
 * Vetor completo, COM pesos. É ele que vai para o pgvector (parte 1).
 * Cada grupo de features é multiplicado pelo seu peso, como o
 * oneHotWeighted da aula fazia com categoria e cor.
 */
export function encodePokemon(pokemon, context) {
    const [hp, attack, defense, spAttack, spDefense, speed, height, weight] = encodeStats(pokemon, context)

    const stats = [hp, attack, defense, spAttack, spDefense, speed]
        .map(value => value * WEIGHTS.stats)

    const size = [height, weight]
        .map(value => value * WEIGHTS.size)

    const types = encodeTypes(pokemon, context)
        .map(value => value * WEIGHTS.types)

    return [...stats, ...size, ...types]
}
