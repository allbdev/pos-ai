// ====================================================================
// 🖥️ Página: busca de Pokémon parecidos (usa a API do server.js)
// ====================================================================
// Fluxo:
//  1. ao abrir, busca a lista de nomes (GET /pokemon) para o autocompletar;
//  2. no submit, chama GET /pokemon/:name/similar?k=..&metric=..;
//  3. desenha o Pokémon buscado + os vizinhos como cards.
//
// A busca também fica na URL (?name=pikachu&metric=cosine&k=10):
// dá para recarregar a página ou mandar o link e ver o mesmo resultado.

// Cores oficiais (aproximadas) de cada tipo, para os badges
const TYPE_COLORS = {
    normal: '#A8A77A', fire: '#EE8130', water: '#6390F0', electric: '#F7D02C',
    grass: '#7AC74C', ice: '#96D9D6', fighting: '#C22E28', poison: '#A33EA1',
    ground: '#E2BF65', flying: '#A98FF3', psychic: '#F95587', bug: '#A6B91A',
    rock: '#B6A136', ghost: '#735797', dragon: '#6F35FC', dark: '#705746',
    steel: '#B7B7CE', fairy: '#D685AD'
}

// Tipos com cor clara: texto escuro no badge para continuar legível
const LIGHT_TYPES = new Set(['normal', 'electric', 'grass', 'ice', 'ground', 'bug', 'rock', 'steel'])

const form = document.querySelector('#search-form')
const nameInput = document.querySelector('#name-input')
const metricSelect = document.querySelector('#metric-select')
const kSelect = document.querySelector('#k-select')
const namesList = document.querySelector('#pokemon-names')
const status = document.querySelector('#status')
const results = document.querySelector('#results')
const targetContainer = document.querySelector('#target')
const neighborsContainer = document.querySelector('#neighbors')

/**
 * Cria um elemento com classe e texto.
 * Usamos textContent (e nunca innerHTML com dados da API): se um nome
 * tivesse "<script>", ele apareceria como texto, não seria executado.
 */
function createElement(tag, className, text) {
    const element = document.createElement(tag)
    if (className) element.className = className
    if (text !== undefined) element.textContent = text
    return element
}

// "dudunsparce-two-segment" => "dudunsparce two segment" (o CSS capitaliza)
const displayName = name => name.replaceAll('-', ' ')

function typeBadges(types) {
    const container = createElement('div', 'types')

    types.forEach(type => {
        const badge = createElement('span', 'type', type)
        badge.style.background = TYPE_COLORS[type] ?? '#888'
        badge.style.color = LIGHT_TYPES.has(type) ? '#1d2130' : '#ffffff'
        container.append(badge)
    })

    return container
}

/**
 * Card de um Pokémon. Os vizinhos são <button> (clicáveis e acessíveis
 * pelo teclado); o Pokémon buscado é um <div>.
 */
function pokemonCard(pokemon, { clickable = false, className = '' } = {}) {
    const card = createElement(clickable ? 'button' : 'div', `card ${className}`)
    if (clickable) card.type = 'button'

    const image = createElement('img')
    image.src = pokemon.sprite
    image.alt = displayName(pokemon.name)
    image.loading = 'lazy' // só baixa a imagem quando ela aparece na tela

    const info = createElement('div')
    info.append(
        createElement('div', 'number', `#${String(pokemon.id).padStart(4, '0')}`),
        createElement('div', 'name', displayName(pokemon.name)),
        typeBadges(pokemon.types)
    )

    card.append(image, info)
    return card
}

/**
 * Barra de proximidade: o mais parecido da lista fica com a barra cheia.
 * A distância "crua" depende da métrica (euclidiana e cosseno têm escalas
 * diferentes), então a barra é RELATIVA aos resultados desta busca.
 */
function closenessBar(distance, maxDistance) {
    const bar = createElement('div', 'closeness')
    const fill = createElement('span')

    // O mais longe da lista fica com 20% da barra, não zero
    const width = 100 * (1 - distance / (maxDistance * 1.25))
    fill.style.width = `${Math.max(width, 0)}%`

    bar.append(fill)
    return bar
}

function render({ pokemon, similar }) {
    targetContainer.replaceChildren(pokemonCard(pokemon, { className: 'card-target' }))

    const maxDistance = Math.max(...similar.map(item => item.distance))

    const cards = similar.map(item => {
        const card = pokemonCard(item, { clickable: true })
        card.append(
            closenessBar(item.distance, maxDistance),
            createElement('div', 'distance', `distância ${item.distance.toFixed(3)}`)
        )

        // Clicar num vizinho busca os parecidos com ELE
        card.addEventListener('click', () => {
            nameInput.value = item.name
            search()
        })

        return card
    })

    neighborsContainer.replaceChildren(...cards)
    results.hidden = false
}

function showStatus(message, isError = false) {
    status.textContent = message
    status.classList.toggle('error', isError)
}

async function search() {
    const name = nameInput.value.trim().toLowerCase()
    if (!name) return

    const params = new URLSearchParams({ k: kSelect.value, metric: metricSelect.value })

    // Atualiza a URL sem recarregar a página (replaceState não cria
    // uma entrada nova no histórico a cada busca)
    history.replaceState(null, '', `?${new URLSearchParams({ name, ...Object.fromEntries(params) })}`)

    showStatus('Buscando...')

    try {
        // encodeURIComponent: nomes com caracteres especiais não quebram a URL
        const response = await fetch(`/pokemon/${encodeURIComponent(name)}/similar?${params}`)
        const data = await response.json()

        // A API responde { error } com status 400/404/500
        if (!response.ok) {
            showStatus(data.error ?? 'Erro na busca', true)
            return
        }

        render(data)
        showStatus('')
        window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch {
        // fetch só lança erro em falha de rede (ex.: servidor desligado)
        showStatus('Não foi possível falar com o servidor. Ele está rodando (npm start)?', true)
    }
}

async function loadNames() {
    try {
        const response = await fetch('/pokemon')
        const pokemons = await response.json()

        namesList.replaceChildren(...pokemons.map(({ name }) => {
            const option = createElement('option')
            option.value = name
            return option
        }))
    } catch {
        // Sem a lista, a busca continua funcionando: só não tem autocompletar
    }
}

form.addEventListener('submit', event => {
    event.preventDefault() // sem isso, o form recarregaria a página
    search()
})

// Trocar métrica ou quantidade refaz a busca na hora
metricSelect.addEventListener('change', search)
kSelect.addEventListener('change', search)

// Estado inicial: o que estiver na URL, ou Pikachu
const initial = new URLSearchParams(location.search)
nameInput.value = initial.get('name') ?? 'pikachu'
if (initial.has('metric')) metricSelect.value = initial.get('metric')
if (initial.has('k')) kSelect.value = initial.get('k')

loadNames()
search()
