// ====================================================================
// 📂 Data: lê os arquivos gerados pelos ingests
// ====================================================================
// Um lugar só para ler os dados, em vez de cada script (seed, train...)
// ter o seu próprio readFile com o caminho do arquivo.

import { readFile } from 'node:fs/promises'

const DATA_DIR = new URL('../data/', import.meta.url)

const readJson = async fileName =>
    JSON.parse(await readFile(new URL(fileName, DATA_DIR), 'utf8'))

/**
 * @returns {{ pokemons: Array, moveTypes: Object<string, string> }}
 *   pokemons:  gerado pelo npm run ingest       (data/pokemon.json)
 *   moveTypes: gerado pelo npm run ingest:moves (data/move-types.json),
 *              mapa "golpe -> tipo", ex.: { thunderbolt: 'electric' }
 */
export async function loadData() {
    const [pokemons, moveTypes] = await Promise.all([
        readJson('pokemon.json'),
        readJson('move-types.json')
    ])

    return { pokemons, moveTypes }
}
