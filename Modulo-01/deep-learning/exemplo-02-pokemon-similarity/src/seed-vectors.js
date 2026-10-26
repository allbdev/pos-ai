// ====================================================================
// 🌱 Seed: transforma cada Pokémon em vetor e salva no pgvector
// ====================================================================
// Fluxo: data/pokemon.json → makeContext → encodePokemon → tabela "pokemon"
//
// Pode rodar quantas vezes quiser (npm run seed). Rode de novo sempre que
// mudar o encode.js ou os WEIGHTS: os vetores antigos são substituídos.

import { loadData } from './data.js'
import { makeContext } from './context.js'
import { encodePokemon, encodeStats } from './encode.js'
import { pool, recreateTable, toSql } from './db.js'

async function main() {
    const { pokemons, moveTypes } = await loadData()

    // O contexto é calculado com TODOS os Pokémon (min/max, tipos),
    // antes de codificar qualquer um deles.
    const context = makeContext(pokemons, moveTypes)

    // Tabela nova e vazia a cada seed, já com o tamanho de vetor certo
    await recreateTable(context.dimensions, context.statsDimensions)

    // Pega UMA conexão do pool para usar em uma transação.
    // Transação = tudo ou nada: se um insert falhar no meio,
    // o ROLLBACK desfaz os anteriores e a tabela não fica pela metade.
    const client = await pool.connect()

    try {
        await client.query('BEGIN')

        for (const pokemon of pokemons) {
            const embedding = encodePokemon(pokemon, context)

            // Só stats + tamanho, sem tipos e sem pesos: a mesma entrada da rede neural
            const statsEmbedding = encodeStats(pokemon, context)

            // $1, $2... são parâmetros: o pg escapa os valores para nós.
            // Nunca monte SQL concatenando strings com dados (SQL injection).
            //
            // INSERT simples: a tabela acabou de ser recriada vazia,
            // então não existe Pokémon repetido para atualizar.
            await client.query(
                `INSERT INTO pokemon (id, name, types, sprite, embedding, stats_embedding)
                 VALUES ($1, $2, $3, $4, $5, $6)`,
                [pokemon.id, pokemon.name, pokemon.types, pokemon.sprite, toSql(embedding), toSql(statsEmbedding)]
            )
        }

        await client.query('COMMIT')
        console.log(`✅ ${pokemons.length} vetores de ${context.dimensions} dimensões salvos no pgvector`)
    } catch (error) {
        await client.query('ROLLBACK')
        throw error
    } finally {
        // Devolve a conexão para o pool e fecha o pool,
        // senão o Node fica esperando e o script não termina.
        client.release()
        await pool.end()
    }
}

main()
