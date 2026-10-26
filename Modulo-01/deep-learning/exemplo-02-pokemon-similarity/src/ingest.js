// ====================================================================
// 📥 Ingest: busca os Pokémon na PokéAPI e salva em data/pokemon.json
// ====================================================================
// Roda UMA vez (npm run ingest). Depois disso, todos os outros scripts
// leem o arquivo local em vez de chamar a API. Assim:
//  - respeitamos a política de uso justo da PokéAPI (sem chamadas repetidas);
//  - os próximos passos ficam rápidos e sempre usam os mesmos dados.

import { mkdir, writeFile } from 'node:fs/promises'

const API_URL = 'https://pokeapi.co/api/v2/pokemon'

// Geração 1: ids de 1 a 151. Dataset pequeno, como na aula.
const FIRST_ID = 1
const LAST_ID = 151

// Caminho relativo a ESTE arquivo (src/), não a pasta onde o comando foi rodado.
// Assim o script funciona tanto com "npm run ingest" quanto com "node src/ingest.js".
const OUTPUT_DIR = new URL('../data/', import.meta.url)
const OUTPUT_FILE = new URL('pokemon.json', OUTPUT_DIR)

/**
 * Busca um Pokémon pelo id e devolve a resposta completa da API.
 * A resposta tem MUITA coisa (golpes, jogos, sons...), por isso
 * depois passamos ela pelo simplifyPokemon().
 */
async function fetchPokemon(id) {
    const response = await fetch(`${API_URL}/${id}`)

    // fetch não lança erro em 404/500, só em falha de rede.
    // Precisamos checar o status na mão.
    if (!response.ok) {
        throw new Error(`Falha ao buscar o Pokémon ${id}: HTTP ${response.status}`)
    }

    return response.json()
}

/**
 * Mantém só os campos que vamos usar para montar os vetores.
 *
 * Antes (resumido):
 *   stats: [{ base_stat: 35, stat: { name: 'hp' } }, ...]
 *   types: [{ slot: 1, type: { name: 'electric' } }]
 *
 * Depois:
 *   stats: { hp: 35, attack: 55, ... }
 *   types: ['electric']
 */
function simplifyPokemon(raw) {
    // Transforma a lista de stats em um objeto { nomeDoStat: valor }
    const stats = Object.fromEntries(
        raw.stats.map(item => [item.stat.name, item.base_stat])
    )

    // Ordena pelo slot para garantir que o tipo PRIMÁRIO venha primeiro.
    // Isso importa na parte 2: a rede vai prever o types[0].
    const types = raw.types
        .toSorted((a, b) => a.slot - b.slot)
        .map(item => item.type.name)

    return {
        id: raw.id,
        name: raw.name,
        stats,
        types,
        height: raw.height, // em decímetros (4 = 0.4 m)
        weight: raw.weight, // em hectogramas (60 = 6 kg)
        sprite: raw.sprites.front_default // só para exibir, não entra no vetor
    }
}

async function main() {
    const pokemons = []

    // Uma requisição por vez (sequencial), de propósito:
    // são só 151 chamadas, e assim não sobrecarregamos uma API pública e gratuita.
    for (let id = FIRST_ID; id <= LAST_ID; id++) {
        const raw = await fetchPokemon(id)
        const pokemon = simplifyPokemon(raw)
        pokemons.push(pokemon)

        console.log(`[${id}/${LAST_ID}] ${pokemon.name} (${pokemon.types.join(' + ')})`)
    }

    // Cria a pasta data/ se ela ainda não existir
    await mkdir(OUTPUT_DIR, { recursive: true })

    // null, 2 = JSON formatado com 2 espaços, fácil de ler e de revisar no git
    await writeFile(OUTPUT_FILE, JSON.stringify(pokemons, null, 2))

    console.log(`\n✅ ${pokemons.length} Pokémon salvos em data/pokemon.json`)
}

main()
