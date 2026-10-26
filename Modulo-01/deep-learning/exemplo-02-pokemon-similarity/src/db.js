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
 * Cria a extensão e a tabela, se ainda não existirem.
 *
 * @param {number} dimensions tamanho do vetor (context.dimensions, 25 na Gen 1).
 *   Vem do contexto em vez de ser fixo: se os dados mudarem (ex.: todas as gerações),
 *   o tamanho muda junto.
 */
export async function createTable(dimensions) {
    // Ativa o pgvector neste banco (a imagem Docker tem a extensão,
    // mas cada banco precisa ativar).
    await pool.query('CREATE EXTENSION IF NOT EXISTS vector')

    // embedding vector(N): coluna que guarda um vetor de N números.
    // É nela que o pgvector faz a busca por distância (<-> e <=>).
    // types text[]: array de texto do Postgres, ex.: {fire,flying}
    await pool.query(`
        CREATE TABLE IF NOT EXISTS pokemon (
            id        integer PRIMARY KEY,
            name      text NOT NULL UNIQUE,
            types     text[] NOT NULL,
            sprite    text,
            embedding vector(${dimensions}) NOT NULL
        )
    `)
    // Atenção: IF NOT EXISTS não altera uma tabela que já existe.
    // Se o tamanho do vetor mudar, é preciso apagar a tabela
    // (DROP TABLE pokemon) ou zerar o banco (docker compose down -v).
}
