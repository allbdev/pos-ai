// ====================================================================
// 📥 Ingest dos tipos dos golpes: monta o mapa "golpe -> tipo"
// ====================================================================
// O data/pokemon.json só tem o NOME dos golpes ('thunderbolt').
// Para saber a porcentagem de golpes de cada tipo, precisamos do tipo
// de cada golpe ('thunderbolt' -> 'electric').
//
// Em vez de buscar os ~937 golpes um por um, usamos o /type/{nome}:
// cada tipo já lista todos os seus golpes. São só ~20 requisições.
//
// Roda UMA vez (npm run ingest:moves). Gera data/move-types.json:
//   { "absorb": "grass", "acid": "poison", ..., "thunderbolt": "electric", ... }

import { mkdir, writeFile } from 'node:fs/promises'

const TYPES_URL = 'https://pokeapi.co/api/v2/type'

const OUTPUT_DIR = new URL('../data/', import.meta.url)
const OUTPUT_FILE = new URL('move-types.json', OUTPUT_DIR)

async function fetchJson(url) {
    const response = await fetch(url)

    // fetch não lança erro em 404/500, só em falha de rede
    if (!response.ok) {
        throw new Error(`Falha ao buscar ${url}: HTTP ${response.status}`)
    }

    return response.json()
}

async function main() {
    // Lista de todos os tipos. São 21: os 18 normais + stellar, unknown e shadow.
    // limit=100 garante que vem tudo em uma página só.
    const { results: types } = await fetchJson(`${TYPES_URL}?limit=100`)

    const moveTypes = {}

    // Uma requisição por vez, como no ingest.js: API pública e gratuita
    for (const { name, url } of types) {
        const type = await fetchJson(url)

        // stellar e unknown não têm golpes. shadow tem 18, mas são golpes
        // que só existem nos jogos Colosseum/XD (nenhum Pokémon os aprende
        // nos jogos principais). Guardamos assim mesmo: o encode só conta
        // os tipos que aparecem nos Pokémon e ignora o resto.
        type.moves.forEach(move => {
            moveTypes[move.name] = name
        })

        console.log(`${name.padEnd(9)} ${type.moves.length} golpes`)
    }

    // Ordena as chaves (nomes dos golpes) para o JSON ficar estável no git
    const sorted = Object.fromEntries(
        Object.entries(moveTypes).sort(([a], [b]) => a.localeCompare(b))
    )

    await mkdir(OUTPUT_DIR, { recursive: true })
    await writeFile(OUTPUT_FILE, JSON.stringify(sorted, null, 2))

    console.log(`\n✅ ${Object.keys(sorted).length} golpes salvos em data/move-types.json`)
}

main()
