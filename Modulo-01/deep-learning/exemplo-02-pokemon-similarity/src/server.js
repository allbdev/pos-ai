// ====================================================================
// 🌐 Server: API HTTP para buscar Pokémon parecidos
// ====================================================================
// Pré-requisitos: banco no ar (docker compose up -d) e vetores salvos (npm run seed).
// Rodar: npm start
// Testar: curl "http://localhost:3001/pokemon/pikachu/similar?k=5"

import express from 'express'
import { pool } from './db.js'

// 3001 para não conflitar com o exemplo-01, que usa a 3000
const PORT = process.env.PORT ?? 3001

// Limites do parâmetro k (quantos vizinhos retornar)
const DEFAULT_K = 5
const MAX_K = 50

const app = express()

/**
 * GET /pokemon/:name/similar?k=5
 * Retorna os k Pokémon com o vetor mais próximo do Pokémon pedido.
 */
app.get('/pokemon/:name/similar', async (req, res) => {
    // Nomes no banco estão em minúsculo (vieram assim da PokéAPI)
    const name = req.params.name.toLowerCase()

    // Query params sempre chegam como string: "5", "abc" ou undefined.
    // Se não vier, usa o padrão. Se vier inválido, responde 400 (erro do cliente).
    const k = req.query.k === undefined ? DEFAULT_K : Number(req.query.k)

    if (!Number.isInteger(k) || k < 1 || k > MAX_K) {
        return res.status(400).json({ error: `k deve ser um inteiro entre 1 e ${MAX_K}` })
    }

    // 1️⃣ Busca o Pokémon pedido. Se não existir, 404.
    //    Fazer isso separado deixa claro a diferença entre
    //    "Pokémon não existe" e "não encontrou vizinhos".
    const target = await pool.query(
        'SELECT id, name, types, sprite, embedding FROM pokemon WHERE name = $1',
        [name]
    )

    if (target.rows.length === 0) {
        return res.status(404).json({ error: `Pokémon "${name}" não encontrado` })
    }

    const pokemon = target.rows[0]

    // 2️⃣ Busca os vizinhos mais próximos.
    //    <-> = distância euclidiana (linha reta entre os dois vetores).
    //    O embedding volta do banco como texto '[0.1,...]', e serve
    //    direto como parâmetro na próxima query.
    //    id <> $2 tira o próprio Pokémon do resultado (distância 0).
    const neighbors = await pool.query(
        `SELECT id, name, types, sprite, embedding <-> $1 AS distance
         FROM pokemon
         WHERE id <> $2
         ORDER BY distance
         LIMIT $3`,
        [pokemon.embedding, pokemon.id, k]
    )

    // O embedding não vai na resposta: é um detalhe interno,
    // quem consome a API só quer saber quais Pokémon são parecidos.
    res.json({
        pokemon: {
            id: pokemon.id,
            name: pokemon.name,
            types: pokemon.types,
            sprite: pokemon.sprite
        },
        similar: neighbors.rows
    })
})

// Middleware de erro: qualquer erro inesperado (ex.: banco fora do ar)
// cai aqui. O Express 5 já encaminha erros de funções async para cá.
// Logamos o erro completo no servidor, mas não expomos detalhes para o cliente.
app.use((error, req, res, next) => {
    console.error(error)
    res.status(500).json({ error: 'Erro interno no servidor' })
})

app.listen(PORT, () => {
    console.log(`🚀 API rodando em http://localhost:${PORT}`)
})
