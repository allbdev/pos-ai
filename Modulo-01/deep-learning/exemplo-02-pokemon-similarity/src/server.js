// ====================================================================
// 🌐 Server: API HTTP para buscar Pokémon parecidos
// ====================================================================
// Pré-requisitos: banco no ar (docker compose up -d), vetores salvos (npm run seed)
// e, para o /predict-type, modelos treinados (npm run train).
// Rodar: npm start
// Testar: curl "http://localhost:3001/pokemon/pikachu/similar?k=5"

import express from 'express'
import { pool } from './db.js'
import { loadClassifiers, validateInput } from './classifier.js'

// 3001 para não conflitar com o exemplo-01, que usa a 3000
const PORT = process.env.PORT ?? 3001

// Limites do parâmetro k (quantos vizinhos retornar)
const DEFAULT_K = 5
const MAX_K = 50

// Métricas de distância aceitas em ?metric= e o operador do pgvector de cada uma.
//  - euclidean (<->): distância em linha reta. Compara o "tamanho" (nível de poder).
//  - cosine    (<=>): ângulo entre os vetores. Compara o "formato" (perfil), ignora o tamanho.
//
// Por que um mapa (whitelist)? Operadores não podem ser parâmetros ($1) no SQL,
// então eles entram no texto da query. Se colocássemos ali o que o usuário
// digitou, ele poderia injetar SQL. Com o mapa, só esses dois textos fixos entram.
const METRICS = {
    euclidean: '<->',
    cosine: '<=>'
}
const DEFAULT_METRIC = 'euclidean'

// Carrega os classificadores antes de subir o servidor (top-level await).
// Se algum modelo ainda não foi treinado, a API sobe mesmo assim:
// a busca por similaridade funciona e só o /predict-type que precisa dele fica indisponível.
const classifiers = await loadClassifiers()
console.log(`🔮 Modelos de tipos carregados: ${Object.keys(classifiers).join(', ') || 'nenhum'}`)

const app = express()

// Lê corpos JSON das requisições (POST) e coloca em req.body
app.use(express.json())

/**
 * GET /pokemon/:name/similar?k=5&metric=euclidean
 * Retorna os k Pokémon com o vetor mais próximo do Pokémon pedido.
 * metric: "euclidean" (padrão) ou "cosine"
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

    const metric = req.query.metric ?? DEFAULT_METRIC

    // Object.hasOwn em vez de METRICS[metric]: evita que valores como
    // "toString" ou "constructor" (herdados de todo objeto JS) passem como válidos.
    if (!Object.hasOwn(METRICS, metric)) {
        return res.status(400).json({ error: `metric deve ser: ${Object.keys(METRICS).join(' ou ')}` })
    }

    const operator = METRICS[metric]

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

    // 2️⃣ Busca os vizinhos mais próximos, usando o operador da métrica escolhida.
    //    ${operator} é seguro aqui: só pode ser um dos valores fixos do METRICS.
    //    O embedding volta do banco como texto '[0.1,...]', e serve
    //    direto como parâmetro na próxima query.
    //    id <> $2 tira o próprio Pokémon do resultado (distância 0).
    const neighbors = await pool.query(
        `SELECT id, name, types, sprite, embedding ${operator} $1 AS distance
         FROM pokemon
         WHERE id <> $2
         ORDER BY distance
         LIMIT $3`,
        [pokemon.embedding, pokemon.id, k]
    )

    // O embedding não vai na resposta: é um detalhe interno,
    // quem consome a API só quer saber quais Pokémon são parecidos.
    res.json({
        metric,
        pokemon: {
            id: pokemon.id,
            name: pokemon.name,
            types: pokemon.types,
            sprite: pokemon.sprite
        },
        similar: neighbors.rows
    })
})

/**
 * POST /predict-type
 * Corpo: { "stats": { "hp": 35, "attack": 55, ... }, "height": 4, "weight": 60, "moves": [...] }
 * Retorna os 3 tipos primários mais prováveis segundo a rede neural.
 *
 * "moves" é opcional e deve ser o LEARNSET completo (todos os golpes que o Pokémon
 * pode aprender), não os 4 golpes de batalha: o modelo foi treinado com learnsets.
 *  - sem moves: modelo "stats"       (~22% de acerto)
 *  - com moves: modelo "stats-moves" (~79% de acerto)
 */
app.post('/predict-type', (req, res) => {
    // ?. porque o corpo pode nem existir (o validateInput responde esse caso)
    const modelName = req.body?.moves === undefined ? 'stats' : 'stats-moves'
    const classifier = classifiers[modelName]

    // 503 = serviço indisponível: o problema é do servidor (sem modelo), não do cliente
    if (!classifier) {
        return res.status(503).json({ error: `Modelo "${modelName}" não treinado. Rode "npm run train".` })
    }

    const error = validateInput(req.body, classifier.moveTypes)
    if (error) {
        return res.status(400).json({ error })
    }

    // "model" na resposta: quem chama sabe qual modelo respondeu (e o quanto confiar)
    res.json({ model: modelName, predictions: classifier.predict(req.body) })
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
