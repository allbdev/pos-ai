// ====================================================================
// 🔢 Encode: transforma um Pokémon em um vetor de números
// ====================================================================
// Mesma ideia do encodeProduct da aula (exemplo-01).
// O vetor final tem esse formato (44 números com todas as gerações):
//
// [ hp, attack, defense, sp-attack, sp-defense, speed, height, weight, ...18 tipos, ...18 golpes ]
//   \______________ 8 números, de 0 a 1 ______________________________/  multi-hot  % por tipo
//
// Diferente da aula, aqui usamos arrays JS normais em vez de tensores (tf):
//  - o pgvector recebe arrays comuns, não tensores;
//  - não precisamos nos preocupar em liberar memória (dispose);
//  - na hora de treinar, é só converter com tf.tensor2d(arrays).

import { STAT_NAMES, logScale } from './context.js'

// O que define "Pokémon parecidos"? Os pesos decidem.
// Com tipos pesando mais, os vizinhos tendem a ter o mesmo tipo.
// Com stats pesando mais, os vizinhos tendem a ter o mesmo perfil de batalha.
// Com golpes pesando mais, os vizinhos tendem a "jogar" parecido.
// Experimento sugerido: mudar esses valores e comparar os resultados.
//
// Tipos e golpes se sobrepõem (Pokémon de água aprendem golpes de água),
// então somar golpes sem tirar peso dos tipos empurraria tudo ainda mais
// para "mesmo tipo". Por isso os tipos caíram de 0.5 para 0.3.
//
// Peso NÃO é influência: influência = peso × quanto a feature VARIA entre Pokémon.
// Os tipos variam muito (0 ou 1 por posição); as % de golpes variam pouco
// (todo mundo tem ~45% de golpes normal). Com golpes em 0.3, eles eram só 4%
// da distância entre dois Pokémon quaisquer, e os tipos eram 88%.
// Com golpes em 1, eles viram ~30%: os tipos ainda definem a "vizinhança",
// e os golpes trazem a família de evolução (Pikachu => Raichu, Pichu).
// Os pesos não precisam somar 1: só a proporção entre eles importa.
export const WEIGHTS = {
    types: .3,
    moves: 1, // % dos golpes de cada tipo
    stats: .3,
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
 * Ex. (Pikachu): [0.13, 0.28, 0.16, 0.25, 0.14, 0.44, 0.20, 0.40]
 */
export function encodeStats(pokemon, context) {
    const stats = STAT_NAMES.map(name =>
        normalize(pokemon.stats[name], context.stats[name])
    )

    // Altura e peso: log primeiro, depois min-max (o contexto guarda min/max em log).
    // Os stats não precisam de log: vão de ~5 a ~255, sem valores extremos.
    const height = normalize(logScale(pokemon.height), context.height)
    const weight = normalize(logScale(pokemon.weight), context.weight)

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
 * Porcentagem dos golpes do Pokémon que é de cada tipo.
 * Mesmo formato do encodeTypes (uma posição por tipo, mesma ordem),
 * mas com frações em vez de 0/1. As posições somam 1.
 *
 * Ex. (Pikachu, 109 golpes): normal 0.46, electric 0.19, fighting 0.10, ...
 *   => [0, 0, 0.19, ..., 0.10, ..., 0.46, ...]
 *
 * Por que porcentagem e não contagem? O tamanho do learnset varia muito
 * (Weedle 4 golpes, Mew 375). Com contagem, Pokémon com muitos golpes
 * pareceriam "mais de tudo". A porcentagem compara o PERFIL dos golpes.
 */
export function encodeMoveTypes(pokemon, context) {
    const vector = new Array(context.numTypes).fill(0)
    let counted = 0

    pokemon.moves.forEach(move => {
        const index = context.typesIndex[context.moveTypes[move]]

        // Golpe sem tipo conhecido, ou de um tipo que nenhum Pokémon tem
        // (ex.: shadow): não entra na conta
        if (index === undefined) return

        vector[index]++
        counted++
    })

    // Sem nenhum golpe contado, devolve tudo zero (evita dividir por zero)
    if (counted === 0) return vector

    return vector.map(count => count / counted)
}

/**
 * Conjuntos de features do classificador (parte 2): cada um vira um modelo
 * separado, salvo em model/<nome>/. É o MESMO mapa no treino (train.js) e
 * na previsão (classifier.js), então a entrada é codificada igual nos dois.
 *
 * Sem pesos aqui: na rede neural, ela mesma aprende quanto cada entrada importa.
 * (Os WEIGHTS servem para a distância do pgvector, que não aprende nada.)
 */
export const FEATURE_SETS = {
    // 8 números: stats + tamanho (o modelo original)
    'stats': (pokemon, context) => encodeStats(pokemon, context),

    // 8 + 18 = 26 números: stats + tamanho + % dos golpes por tipo
    'stats-moves': (pokemon, context) => [
        ...encodeStats(pokemon, context),
        ...encodeMoveTypes(pokemon, context)
    ]
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

    const moves = encodeMoveTypes(pokemon, context)
        .map(value => value * WEIGHTS.moves)

    return [...stats, ...size, ...types, ...moves]
}
