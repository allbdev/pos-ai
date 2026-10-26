# Pokémon Similarity & Type Prediction

> Status: **Parts 1 and 2 done** (similarity search + type classifier). Next: experiments.

A follow-up to [`exemplo-01`](../exemplo-01-ecommerce-recomendations-template). It uses the same techniques (normalization, one-hot encoding, feature weights, a small neural network) on a new dataset, with the work moved from the browser to a **Node backend**. The vectors are stored in **Postgres + pgvector**.

## Goal

Use data from [PokéAPI](https://pokeapi.co/) to build two things:

1. **Similarity search (vectors):** "Which Pokémon are most similar to Pikachu?" Each Pokémon becomes a numeric vector, stored in pgvector and searched by distance. No training is involved; this part is all about encoding the data well.
2. **Type prediction (deep learning):** "Given these base stats, what type is this Pokémon?" A neural network is trained on the stats and predicts the primary type. This is the part that continues the class subject.

Both parts share the same encoding code, just as `makeContext` and `encodeProduct` were shared in exemplo-01.

## The data

From `GET https://pokeapi.co/api/v2/pokemon/{id}`, we keep:

| Field | Example (Pikachu) | Notes |
|---|---|---|
| `name` | `pikachu` | |
| `stats` | hp 35, attack 55, defense 40, special-attack 50, special-defense 50, speed 90 | The 6 base stats |
| `types` | `electric` | Can be 1 or 2 types (for example, Charizard is fire + flying) |
| `height` | `4` | In decimetres |
| `weight` | `60` | In hectograms |
| `sprites.front_default` | image URL | Only for display |

We start with **Generation 1 (ids 1–151)**, a small dataset like the one in the class. Expanding to all generations later is an experiment in itself (see [Experiments](#experiments)).

## Encoding

The same ideas as exemplo-01, with one new concept: **multi-hot encoding**.

```
[ ...6 stats, height, weight, ...N types ]   → 8 + N numbers
```

`N` comes from the data, not from a hard-coded list (`makeContext` in `src/context.js`). Gen 1 has **17 types**, since there are no Dark types and PokéAPI returns current types (so Clefairy is Fairy and Magnemite is Steel), which makes each vector **25 numbers**.

- **Stats, height, weight:** min-max normalization to 0–1. Weight goes from 0.1 kg to hundreds of kg, so without normalization it would dominate everything.
- **Types:** a Pokémon can have **two**. One-hot encoding allows only one position "on"; **multi-hot** allows several:
  ```
  types = [normal, fire, water, electric, grass, ..., flying, ...]
  pikachu   (electric)        → [0, 0, 0, 1, 0, ..., 0, ...]
  charizard (fire + flying)   → [0, 1, 0, 0, 0, ..., 1, ...]
  ```
- **Feature weights:** a `WEIGHTS` object (for example, types 0.5, stats 0.4, size 0.1) decides what "similar" means. Weighting types heavily makes similar Pokémon share types; weighting stats heavily makes them share a battle profile. Changing the weights and comparing the results is one of the learning goals.

## Part 1: similarity search

1. Encode every Pokémon into its 25-number vector.
2. Store the vectors in a `vector(25)` column in Postgres.
3. Find the nearest neighbors with SQL:
   ```sql
   SELECT name FROM pokemon
   ORDER BY embedding <-> (SELECT embedding FROM pokemon WHERE name = 'pikachu')
   LIMIT 6;   -- the first result is Pikachu itself
   ```

pgvector has several distance operators. `<->` is Euclidean (straight-line) distance, and `<=>` is cosine distance (angle). Comparing their results is part of the exercise.

## Part 2: type prediction

The network learns: **input = normalized stats + height + weight (8 numbers) → output = primary type (15 classes: only 15 of the 17 types appear as a primary type)**.

The types are left out of the input; otherwise the network would just copy them.

| Layer | Units | Activation |
|---|---|---|
| Dense | 32 | ReLU |
| Dense | 16 | ReLU |
| Output | 15 | Softmax |

What's new compared to exemplo-01:

| | exemplo-01 | exemplo-02 |
|---|---|---|
| Problem | Binary (buy / don't buy) | **Multi-class** (1 of 15 types) |
| Output | 1 neuron, sigmoid | **15 neurons, softmax** (probabilities that add up to 1) |
| Loss | `binaryCrossentropy` | **`categoricalCrossentropy`** |
| Validation | None | **`validationSplit: 0.2`**, to watch for overfitting |
| Where it runs | Browser (Web Worker) | **Node** (`@tensorflow/tfjs-node`) |
| Persistence | Lost on refresh | **Saved with `model.save('file://...')`** |

**Expectations:** 151 examples across 15 classes is very little data, and stats alone don't fully determine type. The model will be wrong often, and that's part of the lesson. Success means **beating a baseline**: always guessing the most common type, which is water at 18.5% (28 of 151). The model should also show patterns that make sense, such as high defense pointing to rock or steel.

## Architecture

```
PokéAPI ──(once)──▶ ingest.js ──▶ data/pokemon.json      (cached raw data)
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
               seed-vectors.js                          train.js
          encode → insert into pgvector        encode → train → save model/
                     │                                     │
                     └──────────────────┬──────────────────┘
                                        ▼
                                    server.js
                  GET  /pokemon/:name/similar?k=5&metric=cosine → pgvector search
                  POST /predict-type  { stats... }  → loaded model
```

The API is called **once** and the result is cached in `data/pokemon.json`. This respects PokéAPI's fair-use policy and makes every later step fast and repeatable.

## Planned structure

```
exemplo-02-pokemon-similarity/
├── README.md
├── package.json
├── docker-compose.yml      # Postgres with the pgvector extension
├── data/
│   └── pokemon.json        # Cached PokéAPI data (generated by ingest)
├── model/                  # Saved trained model (generated, gitignored)
└── src/
    ├── ingest.js           # Fetches Pokémon 1–151 from PokéAPI → data/pokemon.json
    ├── context.js          # makeContext: min/max, type index, dimensions
    ├── encode.js           # encodePokemon: normalization + multi-hot + weights
    ├── db.js               # Postgres connection + table creation
    ├── seed-vectors.js     # Encodes all Pokémon and inserts them into pgvector
    ├── train.js            # Trains the type classifier and saves it to model/
    └── server.js           # HTTP API: similarity + type prediction
```

## Stack

- **Node.js** (ES modules, like exemplo-00)
- **@tensorflow/tfjs-node**: training and prediction
- **Postgres + pgvector**: vector storage and search, run with Docker
- **pg**: Postgres client
- No frontend at first. The API is tested with curl or Postman.

## Roadmap

- [x] `ingest.js`: fetch and cache Generation 1
- [x] `context.js` + `encode.js`: encoding, reusing the ideas from exemplo-01
- [x] `docker-compose.yml` + `db.js`: pgvector running
- [x] `seed-vectors.js` + `/similar` endpoint: **Part 1 done**
- [x] `train.js`: classifier + baseline comparison
- [x] `/predict-type` endpoint: **Part 2 done**
- [ ] Simple HTML page with sprites (optional)

## Experiments

Ideas to try once both parts work:

- **Change the `WEIGHTS`** and compare the similarity results. What makes two Pokémon "similar"?
- **Euclidean vs. cosine distance** (`?metric=euclidean` or `?metric=cosine`): with the default weights the rankings barely change, because types dominate. With `types: 0`, cosine finds evolution lines (Charmander → Charmeleon, Charizard) while Euclidean finds Pokémon of the same power level.
- **More data:** train with all generations (~1000 Pokémon). Does accuracy improve?
- **Predict both types:** use a multi-label output (17 sigmoid neurons, one per type, and `binaryCrossentropy`) instead of softmax.
- **Team builder:** suggest Pokémon whose types cover a team's weaknesses.
