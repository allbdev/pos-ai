// ====================================================================
// 🐘 DB: conexão com o Postgres (pgvector) e criação da tabela
// ====================================================================
// O banco roda no Docker (docker compose up -d), na porta 5440.

import pg from 'pg'

// A connection string pode vir de uma variável de ambiente;
// se não vier, usa o banco local do docker-compose.yml.
const DATABASE_URL = process.env.DATABASE_URL ?? 'postgres://pokemon:pokemon@localhost:5440/pokemon'

// Pool = um conjunto de conexões reaproveitáveis.
// Em vez de abrir/fechar uma conexão a cada query, o pool mantém
// algumas abertas e empresta para quem precisar. É o padrão para APIs.
export const pool = new pg.Pool({ connectionString: DATABASE_URL })

/**
 * O pgvector espera vetores no formato texto '[0.1,0.2,0.3]'.
 * Um array JS em JSON já fica exatamente assim.
 * (Existe o pacote "pgvector" no npm que faz isso, mas aqui é uma linha só.)
 */
export const toSql = vector => JSON.stringify(vector)

/**
 * Apaga e recria a tabela (e ativa a extensão, se preciso).
 *
 * @param {number} dimensions tamanho do vetor (context.dimensions: 25 na Gen 1, 26 com todas).
 *   Vem do contexto em vez de ser fixo: se os dados mudarem, o tamanho muda junto.
 * @param {number} statsDimensions tamanho do vetor só de stats (context.statsDimensions: 8).
 *
 * Por que apagar em vez de "CREATE TABLE IF NOT EXISTS"?
 * O IF NOT EXISTS não altera uma tabela que já existe. Quando passamos da Gen 1
 * para todas as gerações, apareceu o tipo dark e o vetor foi de 25 para 26 números:
 * a tabela antiga (vector(25)) recusaria os vetores novos.
 * Como o seed recria todos os dados a partir do JSON, recriar a tabela é seguro.
 */
export async function recreateTable(dimensions, statsDimensions) {
    // Ativa o pgvector neste banco (a imagem Docker tem a extensão,
    // mas cada banco precisa ativar).
    await pool.query('CREATE EXTENSION IF NOT EXISTS vector')

    await pool.query('DROP TABLE IF EXISTS pokemon')

    // embedding vector(N): coluna que guarda um vetor de N números.
    // É nela que o pgvector faz a busca por distância (<-> e <=>).
    // types text[]: array de texto do Postgres, ex.: {fire,flying}
    //
    // Duas colunas de vetor, para dois usos diferentes:
    //  - embedding: vetor completo COM tipos e pesos (busca por similaridade, parte 1)
    //  - stats_embedding: só stats + tamanho, SEM tipos (classificador kNN).
    //    Se o kNN usasse o embedding completo, os vizinhos teriam o mesmo tipo
    //    POR CAUSA do tipo: seria "colar" a resposta. É a mesma entrada da rede neural.
    await pool.query(`
        CREATE TABLE pokemon (
            id              integer PRIMARY KEY,
            name            text NOT NULL UNIQUE,
            types           text[] NOT NULL,
            sprite          text,
            embedding       vector(${dimensions}) NOT NULL,
            stats_embedding vector(${statsDimensions}) NOT NULL
        )
    `)
}
