// ====================================================================
// 🗳️ kNN: classificar o tipo pelos vizinhos mais próximos (partes 1 + 2)
// ====================================================================
// Ideia: Pokémon com stats parecidos tendem a ter o mesmo tipo?
// Para adivinhar o tipo de um Pokémon:
//   1. busca no pgvector os k Pokémon com stats mais parecidos (parte 1);
//   2. cada vizinho "vota" no seu tipo primário;
//   3. o tipo mais votado é o palpite (mesma pergunta da parte 2).
//
// Não tem treino: os próprios dados são o "modelo".
// Usa a coluna stats_embedding (sem tipos), a mesma entrada da rede neural,
// para a comparação ser justa.
//
// Pré-requisito: npm run seed
// Rodar: npm run knn

import { pool } from './db.js'

// Valores de k testados (quantos vizinhos votam)
const K_VALUES = [1, 3, 5, 10, 20, 50]
const MAX_K = Math.max(...K_VALUES)

// Mesma whitelist do server.js: operadores entram no texto do SQL,
// então só podem vir de valores fixos
const METRICS = {
    euclidean: '<->',
    cosine: '<=>'
}

/**
 * Para CADA Pokémon, busca os tipos primários dos seus MAX_K vizinhos
 * mais próximos, em ordem de distância. Tudo em uma query só.
 *
 * CROSS JOIN LATERAL = "para cada linha p, rode esta subquery usando p".
 * É como um forEach dentro do SQL: para cada Pokémon, uma busca de vizinhos.
 *
 * q.id <> p.id: o próprio Pokémon não vota em si mesmo (leave-one-out).
 * Se votasse, com k = 1 ele acertaria 100% "olhando a resposta".
 *
 * types[1]: arrays do Postgres começam em 1, não em 0!
 * types[1] é o tipo primário.
 *
 * array_agg(... ORDER BY distance) junta os tipos dos vizinhos em um array,
 * do mais perto para o mais longe. O pg converte para array JS.
 */
async function fetchNeighborTypes(operator) {
    const { rows } = await pool.query(
        `SELECT p.name,
                p.types[1] AS actual_type,
                nearest.neighbor_types
         FROM pokemon p
         CROSS JOIN LATERAL (
             SELECT array_agg(primary_type ORDER BY distance) AS neighbor_types
             FROM (
                 SELECT q.types[1] AS primary_type,
                        q.stats_embedding ${operator} p.stats_embedding AS distance
                 FROM pokemon q
                 WHERE q.id <> p.id
                 ORDER BY distance
                 LIMIT $1
             ) AS neighbors
         ) AS nearest`,
        [MAX_K]
    )

    return rows
}

/**
 * Votação: o tipo que mais aparece entre os vizinhos.
 * Empate: ganha o tipo que apareceu primeiro, ou seja, o do vizinho mais próximo.
 *
 * Ex.: ['water', 'ice', 'ice', 'water', 'normal'] => empate 2 x 2 => 'water'
 */
function vote(neighborTypes) {
    const counts = {}
    neighborTypes.forEach(type => counts[type] = (counts[type] || 0) + 1)

    // Percorre em ordem de distância e só troca de palpite com contagem MAIOR
    // (não maior-ou-igual): assim, no empate, fica o que apareceu antes.
    return neighborTypes.reduce((best, type) => counts[type] > counts[best] ? type : best)
}

async function main() {
    // Baseline: chutar sempre o tipo primário mais comum
    const { rows: [mostCommon] } = await pool.query(
        `SELECT types[1] AS type, count(*)::float / (SELECT count(*) FROM pokemon) AS share
         FROM pokemon
         GROUP BY types[1]
         ORDER BY count(*) DESC
         LIMIT 1`
    )

    const results = {}

    for (const [metric, operator] of Object.entries(METRICS)) {
        const rows = await fetchNeighborTypes(operator)

        results[metric] = K_VALUES.map(k => {
            // Para cada k, só os k primeiros vizinhos votam
            const hits = rows.filter(row =>
                vote(row.neighbor_types.slice(0, k)) === row.actual_type
            ).length

            return hits / rows.length
        })
    }

    const percent = value => `${(value * 100).toFixed(1)}%`.padStart(9)

    console.log(`kNN leave-one-out com ${MAX_K} vizinhos no máximo (cada Pokémon é classificado pelos outros)\n`)
    console.log('    k | euclidean |    cosine')
    console.log('------+-----------+----------')
    K_VALUES.forEach((k, index) => {
        console.log(`${String(k).padStart(5)} | ${percent(results.euclidean[index])} | ${percent(results.cosine[index])}`)
    })
    console.log(`\nBaseline (sempre "${mostCommon.type}"): ${(mostCommon.share * 100).toFixed(1)}%`)
    console.log('Rede neural (README, média de 10 treinos): ~22%')

    await pool.end()
}

main()
