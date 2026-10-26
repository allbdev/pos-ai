// ====================================================================
// 🧭 Contexto: tudo que precisa ser calculado UMA vez a partir dos dados
// ====================================================================
// Mesma ideia do makeContext da aula (exemplo-01): antes de transformar
// um Pokémon em vetor, precisamos conhecer o dataset inteiro para saber:
//  - os limites (min/max) de cada número, para normalizar de 0 a 1;
//  - quais tipos existem e em que posição cada um fica no multi-hot;
//  - o tamanho final do vetor.

// Os 6 stats base, na ordem em que vão aparecer no vetor.
// A ordem precisa ser SEMPRE a mesma: a posição 0 do vetor é sempre hp,
// a posição 5 é sempre speed. Se mudar a ordem, os vetores antigos
// deixam de ser comparáveis com os novos.
export const STAT_NAMES = [
    'hp',
    'attack',
    'defense',
    'special-attack',
    'special-defense',
    'speed'
]

// Escala logarítmica para altura e peso.
//
// Problema: o peso vai de 0.1 kg a 999.9 kg, mas quase todos os Pokémon são leves.
// Com min-max direto, 92% deles ficavam espremidos entre 0 e 0.2
// (Pikachu 6 kg => 0.006, Charizard 90 kg => 0.090): para a rede, quase iguais.
//
// O log "comprime" os valores grandes: cada vez que o peso multiplica por 10,
// o log só soma ~2.3. Assim 6 kg, 90 kg e 460 kg viram 0.40, 0.72 e 0.91,
// espalhados pela faixa 0–1. O que passa a importar é a PROPORÇÃO
// (2× mais pesado), não a diferença absoluta (+100 kg).
//
// log1p(x) = log(1 + x): igual ao log para valores grandes, mas não quebra
// com 0 (log(0) = -Infinito, e o /predict-type aceita peso 0).
export const logScale = value => Math.log1p(value)

// Menor e maior valor de uma lista. Com ~1000 itens, o spread (...) é tranquilo.
const minMax = values => ({
    min: Math.min(...values),
    max: Math.max(...values)
})

/**
 * @param {Array<{name: string, stats: Object<string, number>, types: string[], height: number, weight: number}>} pokemons
 */
export function makeContext(pokemons) {
    // Min/max de CADA stat separadamente.
    // Ex.: speed 90 é comparado com a faixa de speed (15–150),
    // não com a faixa de hp. Cada stat tem sua própria escala.
    const stats = Object.fromEntries(
        STAT_NAMES.map(name => [
            name,
            minMax(pokemons.map(p => p.stats[name]))
        ])
    )

    // Min/max calculados JÁ na escala log: o encode.js aplica o mesmo
    // logScale antes de normalizar, então as duas escalas batem.
    const height = minMax(pokemons.map(p => logScale(p.height)))
    const weight = minMax(pokemons.map(p => logScale(p.weight)))

    // Todos os tipos que aparecem nos dados (primários e secundários).
    // flatMap junta os arrays: [['fire', 'flying'], ['water']] => ['fire', 'flying', 'water']
    // O Set remove duplicados.
    // O sort() deixa a ordem alfabética: assim o índice de cada tipo não depende
    // da ordem em que os Pokémon aparecem no arquivo.
    const types = [...new Set(pokemons.flatMap(p => p.types))].sort()

    // Mapa "tipo -> posição" para o multi-hot.
    // Ex.: { bug: 0, dragon: 1, electric: 2, ... }
    const typesIndex = Object.fromEntries(
        types.map((type, index) => [type, index])
    )

    return {
        stats,
        height,
        weight,
        types,
        typesIndex,
        numTypes: types.length,
        // Tamanho do vetor de um Pokémon:
        // 6 stats + altura + peso + 1 posição por tipo
        dimensions: STAT_NAMES.length + 2 + types.length
    }
}
