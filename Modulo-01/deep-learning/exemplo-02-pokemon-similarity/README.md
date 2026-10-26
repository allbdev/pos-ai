# Pokémon Similarity & Type Prediction

> Status: **Done.** Similarity search + type classifier, using all 1025 Pokémon. See [What I learned](#what-i-learned).

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
| `moves` | `agility`, `thunderbolt`, ... (109 in total) | The **learnset**: every move the Pokémon *can* learn, by any method (level-up, TM, tutor, egg). Not the 4 moves used in a battle |

Each move's type comes from `GET /type/{name}`, which lists all the moves of a type. `ingest-move-types.js` saves the result as a move → type map in `data/move-types.json` (937 moves, about 20 requests). All 797 moves that Pokémon actually learn have a type in the map.

The project started with **Generation 1 (ids 1–151)**, a small dataset like the one in the class, and now uses **all 1025 species**. `ingest.js` reads the species count from `/pokemon-species`, because `/pokemon` also lists alternate forms (Megas, regional forms) with ids from 10001.

## Encoding

The same ideas as exemplo-01, with one new concept: **multi-hot encoding**.

```
[ ...6 stats, height, weight, ...N types, ...N move-type shares ]   → 8 + 2N numbers
```

`N` comes from the data, not from a hard-coded list (`makeContext` in `src/context.js`). All generations have **18 types**, which makes each vector **44 numbers**. Before moves were added it was 26; Gen 1 alone had only 17 types (25 numbers), since there are no Dark types there, and PokéAPI returns current types (so Clefairy is Fairy and Magnemite is Steel).

- **Stats, height, weight:** min-max normalization to 0–1. Weight goes from 0.1 kg to 999.9 kg, so without normalization it would dominate everything. Height and weight are also very skewed, so they go through a log scale first (`logScale` in `src/context.js`).
- **Types:** a Pokémon can have **two**. One-hot encoding allows only one position "on"; **multi-hot** allows several:
  ```
  types = [normal, fire, water, electric, grass, ..., flying, ...]
  pikachu   (electric)        → [0, 0, 0, 1, 0, ..., 0, ...]
  charizard (fire + flying)   → [0, 1, 0, 0, 0, ..., 1, ...]
  ```
- **Moves:** the share of the learnset that belongs to each type (`encodeMoveTypes`), in the same 18 positions as the types. Pikachu: Normal 0.46, Electric 0.19, Fighting 0.10... Shares instead of counts, because learnset sizes vary from 1 to 375 moves.
- **Feature weights:** `WEIGHTS = { types: .3, moves: 1, stats: .3, size: .1 }` decides what "similar" means. Weighting types heavily makes similar Pokémon share types; weighting stats heavily makes them share a battle profile; weighting moves heavily makes them play alike. Changing the weights and comparing the results is one of the learning goals.

## Part 1: similarity search

1. Encode every Pokémon into its 44-number vector.
2. Store the vectors in a `vector(44)` column in Postgres. The seed drops and recreates the table every time, so the column size always matches the data.
3. Find the nearest neighbors with SQL:
   ```sql
   SELECT name FROM pokemon
   ORDER BY embedding <-> (SELECT embedding FROM pokemon WHERE name = 'pikachu')
   LIMIT 6;   -- the first result is Pikachu itself
   ```

pgvector has several distance operators. `<->` is Euclidean (straight-line) distance, and `<=>` is cosine distance (angle). Comparing their results is part of the exercise.

## Part 2: type prediction

Two networks learn: **input → output = primary type (18 classes)**, one per feature set (`FEATURE_SETS` in `src/encode.js`):

| Model | Input |
|---|---|
| `stats` | normalized stats + height + weight (8 numbers) |
| `stats-moves` | the same 8 + the share of moves per type (26 numbers) |

`POST /predict-type` loads both models and picks one per request: **without `moves`** it uses `stats`; **with `moves`** it uses `stats-moves`. The response says which model answered (`"model": "stats-moves"`). `moves` must be the **full learnset**, because that's what the model was trained on; a 4-move battle set would produce very different shares. Unknown move names return a 400 that lists them, so a typo can't silently change the prediction.

The types are left out of the input; otherwise the network would just copy them.

| Layer | Units | Activation |
|---|---|---|
| Dense | 32 | ReLU |
| Dense | 16 | ReLU |
| Output | 18 | Softmax |

What's new compared to exemplo-01:

| | exemplo-01 | exemplo-02 |
|---|---|---|
| Problem | Binary (buy / don't buy) | **Multi-class** (1 of 18 types) |
| Output | 1 neuron, sigmoid | **18 neurons, softmax** (probabilities that add up to 1) |
| Loss | `binaryCrossentropy` | **`categoricalCrossentropy`** |
| Validation | None | **`validationSplit: 0.2`**, to watch for overfitting |
| Where it runs | Browser (Web Worker) | **Node** (`@tensorflow/tfjs-node`) |
| Persistence | Lost on refresh | **Saved with `model.save('file://...')`** |

**Expectations:** stats alone don't fully determine type, so the model will be wrong often, and that's part of the lesson. Success means **beating a baseline**: always guessing the most common type, which is water at 13.1% (134 of 1025). The model should also show patterns that make sense, such as high defense pointing to rock or steel.

**Result:** about **22%** validation accuracy with stats only and **79%** with stats + moves, against a 13% baseline, averaged over 10 runs. See [Experiments](#experiments) for how we got there.

## Running

Requires Node 22+ and Docker.

```bash
npm install
docker compose up -d     # Postgres + pgvector on port 5440
npm run ingest           # Only once: fetches all Pokémon → data/pokemon.json (~40 s)
npm run ingest:moves     # Only once: fetches each move's type → data/move-types.json
npm run seed             # Encodes every Pokémon and stores the vectors in pgvector
npm run train            # Trains both type classifiers → model/stats/ and model/stats-moves/
npm run knn              # Optional: evaluates the kNN classifier (no training)
npm start                # API on http://localhost:3001
```

```bash
# Part 1: the 5 most similar Pokémon (metric: euclidean or cosine)
curl "http://localhost:3001/pokemon/charizard/similar?k=5&metric=cosine"

# Part 2: predict the primary type from stats
curl -X POST http://localhost:3001/predict-type -H 'Content-Type: application/json' \
  -d '{"stats":{"hp":70,"attack":110,"defense":180,"special-attack":60,"special-defense":60,"speed":50},"height":21,"weight":3600}'
```

Rerun `npm run seed` after changing `encode.js` or `WEIGHTS`, and `npm run train` after changing `train.js`. The server loads the model only at startup, so restart it after training.

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
                  POST /predict-type  { stats, moves? } → stats or stats-moves model
```

The API is called **once** and the result is cached in `data/pokemon.json`. This respects PokéAPI's fair-use policy and makes every later step fast and repeatable.

## Structure

```
exemplo-02-pokemon-similarity/
├── README.md
├── package.json
├── docker-compose.yml      # Postgres with the pgvector extension
├── data/
│   ├── pokemon.json        # Cached PokéAPI data (generated by ingest)
│   └── move-types.json     # Move → type map (generated by ingest:moves)
├── model/                  # Saved trained models, one folder per feature set (generated, gitignored)
└── src/
    ├── ingest.js           # Fetches all Pokémon species from PokéAPI → data/pokemon.json
    ├── ingest-move-types.js # Fetches each move's type → data/move-types.json
    ├── data.js             # loadData: reads pokemon.json + move-types.json
    ├── context.js          # makeContext: min/max, type index, move → type map, dimensions
    ├── encode.js           # encodePokemon: normalization + multi-hot + move-type shares + weights
    ├── db.js               # Postgres connection + table creation
    ├── tf.js               # Imports tfjs-node with a Node 23+ compatibility fix
    ├── seed-vectors.js     # Encodes all Pokémon and inserts them into pgvector
    ├── train.js            # Trains one type classifier per feature set → model/<name>/
    ├── classifier.js       # Loads the saved model + input validation for /predict-type
    ├── evaluate-knn.js     # kNN classifier evaluation: type by majority vote of pgvector neighbors
    └── server.js           # HTTP API: similarity + type prediction
```

## Stack

- **Node.js** (ES modules, like exemplo-00)
- **@tensorflow/tfjs-node**: training and prediction
- **Postgres + pgvector**: vector storage and search, run with Docker
- **pg**: Postgres client
- No frontend: the API is tested with curl or Postman.

## Roadmap

- [x] `ingest.js`: fetch and cache Generation 1, then all generations
- [x] `context.js` + `encode.js`: encoding, reusing the ideas from exemplo-01
- [x] `docker-compose.yml` + `db.js`: pgvector running
- [x] `seed-vectors.js` + `/similar` endpoint: **Part 1 done**
- [x] `train.js`: classifier + baseline comparison
- [x] `/predict-type` endpoint: **Part 2 done**
- [x] Experiments: weights, distance metrics, overfitting, more data, log scale, network size
- [x] kNN classifier with pgvector (parts 1 + 2 together)
- [x] Moves: learnset ingest, move-type shares in the similarity vector, `stats-moves` model, optional `moves` in `/predict-type`
- [ ] Simple HTML page with sprites (optional)

## Experiments

Network accuracy numbers are averages over 10 training runs.

- **Changing the `WEIGHTS`:** with `{ types: .2, stats: .7, size: .1 }`, Pikachu's neighbors went from all Electric types (Voltorb, Raichu, Electabuzz) to small, fast, frail Pokémon of any type (Voltorb, Poliwag, Meowth).
- **Adding moves to the similarity vector:** the first try, `moves: .3` (same as types), barely changed the neighbors. Measuring each group's share of the squared distance showed why: between random Pokémon, **types were 88%** and moves only 4%. **A weight is not influence**: influence is weight × how much the feature varies. Types are 0 or 1; move shares vary by a few points, since every learnset is about 45% Normal.

  | Moves weight | Moves' share of distance | Pikachu's neighbors | Lucario's neighbors |
  |---|---|---|---|
  | 0.3 | 4% | Elekid, Blitzle, Voltorb, Electrike, Pichu | ...Primeape, Sirfetch'd |
  | **1 (chosen)** | **30%** | Elekid, **Raichu, Pichu**, Flaaffy, Blitzle | ...Primeape, **Machamp** |
  | 2 | 63% | **Raichu**, Flaaffy, Pichu, Electabuzz, Elekid | ...Machamp, **Riolu** |

  With a real share, moves bring in **evolution families** (relatives share most of their learnset): Snorlax's nearest neighbor became Munchlax. At 2, moves dominate, and Gyarados (Water/Flying) loses its Flying partners in favor of Water attackers (Feraligatr, Milotic). At 1, types still define the neighborhood and moves refine it.
- **Euclidean vs. cosine distance** (`?metric=euclidean` or `?metric=cosine`): with the default weights the rankings barely change, because types dominate. With `types: 0`, cosine finds evolution lines (Charmander → Charmeleon, Charizard) while Euclidean finds Pokémon of the same power level.
- **Fighting overfitting** (`DROPOUT_RATE` and `PATIENCE` in `src/train.js`). Averages over 10 runs each.

  **Generation 1** (151 Pokémon, 31 in validation):

  | Setup | Train acc | Val loss | Val acc |
  |---|---|---|---|
  | 100 epochs, no regularization | 72% | 2.74 | **36.8%** |
  | Dropout 0.3 | 44% | **2.18** | 28.7% |
  | Early stopping (patience 10) | 43% | 2.31 | 22.6% |
  | Dropout + early stopping | 33% | 2.20 | 24.2% |

  Both techniques closed the gap between training and validation and improved `val_loss`, but validation **accuracy** went down. `val_loss` punishes confident mistakes, while accuracy only checks the top guess. The overfit network was overconfident when wrong, yet its top guess was right more often. The real limit is the data: 151 examples, 15 classes and 8 features. Regularization can't add information that isn't there.

  **All generations** (1025 Pokémon, 205 in validation):

  | Setup | Train acc | Val loss | Val acc |
  |---|---|---|---|
  | 100 epochs, no regularization | 33% | 2.75 | **22.6%** |
  | Dropout 0.3 | 21% | **2.51** | 20.5% |
  | Early stopping (patience 10) | 26% | 2.57 | 19.9% |
  | Dropout + early stopping | 19% | 2.54 | 19.8% |

  With 7× more data, training accuracy fell from 72% to 33%: the network can no longer memorize, so overfitting mostly disappeared on its own, and regularization only hurt. Validation accuracy also fell (36.8% → 22.6%), but the task got harder (18 classes, baseline 16.5% → 12.8%), and the Gen 1 number came from just 31 Pokémon. Runs now vary by about ±5 points instead of ±15, so the new numbers are the honest ones. The network now **underfits**: the next levers are better features (such as `log(weight)`) or a bigger network, not more regularization. The code keeps the best setup (no dropout, no early stopping).
- **Log scale for height and weight** (`logScale` in `src/context.js`). With plain min-max, 92% of Pokémon had a normalized weight below 0.2 (Pikachu 0.006, Charizard 0.090). With `log1p` before min-max, the values spread across 0–1 (Pikachu 0.40, Charizard 0.72, Snorlax 0.91).
  - **Classifier:** validation accuracy didn't change (22.6% → 22.5%, 10 runs), and `val_loss` improved slightly (2.75 → 2.63). A neural network can learn non-linear relationships from its inputs, so it partly compensated for the skew already. Size also tells us little about type.
  - **Similarity:** neighbors changed only slightly, because size has just a 0.1 weight. For example, Snorlax's nearest neighbor was Greedent (6 kg) and is now Dudunsparce, with Greedent dropping to 5th.
  - The change stays: it's the correct encoding for skewed data, and it will matter more if `WEIGHTS.size` goes up.
- **k-nearest neighbors (kNN) classifier: parts 1 and 2 together** (`npm run knn`, `src/evaluate-knn.js`). Predicts a Pokémon's primary type by majority vote among its k nearest neighbors in pgvector, with no training. It searches a separate `stats_embedding` column (stats + size only, the same input as the network); the full `embedding` contains the types, which would give away the answer. Each Pokémon is classified by all the others (leave-one-out), so the result is deterministic: one run, no averaging.

  | k | Euclidean | Cosine |
  |---|---|---|
  | 1 | 16.9% | **25.3%** |
  | 3 | 17.2% | **25.5%** |
  | 5 | 17.5% | 24.2% |
  | 10 | 19.7% | 25.0% |
  | 20 | 21.3% | 25.0% |
  | 50 | 21.2% | 23.5% |

  **Cosine kNN beat the neural network** (about 25% vs. 22%), with a single SQL query. The reason is evolution lines: cosine compares stat *shape*, evolutions keep the same shape with bigger numbers, and they usually share a type. Bulbasaur's nearest cosine neighbor is Venusaur; its nearest Euclidean neighbor is Clauncher (Water). That's why cosine works best with few neighbors (the family is just 1–2 Pokémon), while Euclidean needs many votes to rise above noise.

  **Caveat:** kNN is partly *recognizing families*, not learning "stats → type" in general. A brand-new Pokémon from a family with no other members in the data would do worse. A stricter test would keep whole evolution families out of the reference data, which needs PokéAPI's `/evolution-chain` data. The network's random split has the same weakness, but a smooth network can't exploit exact profile matches the way 1-NN does.
- **Bigger network** (`HIDDEN_LAYERS` in `src/train.js`). Averages over 10 runs each:

  | Hidden layers | Train acc | Val loss | Val acc |
  |---|---|---|---|
  | [32, 16] | 34% | **2.69** | 21.1% |
  | [64, 32] | 39% | 2.91 | 21.7% |
  | [128, 64] | 49% | 3.37 | 20.5% |
  | [128, 64, 32] | 40% | 3.24 | 21.4% |

  Training accuracy rose with size, but validation accuracy stayed flat at about 21%, and `val_loss` got worse (more confident wrong answers). The extra capacity went into memorizing, not into learning. Together with the log experiment, this points to a **ceiling in the data**: base stats, height and weight only explain the primary type up to about 21–23% with this approach, against a 13% baseline. The code keeps the smallest network, since it gets the same result.

- **Adding moves to the classifier** (`FEATURE_SETS` in `src/encode.js`). `train.js` trains one model per feature set, on the **same split** in each run, so differences come from the features alone. Averages over 10 runs:

  | Model | Inputs | Train acc | Val acc |
  |---|---|---|---|
  | Baseline (always water) | – | – | 12.6% |
  | `stats` | 8 | 31.6% | 22.3% |
  | **`stats-moves`** | 26 | 94.8% | **79.2%** |
  | `stats-moves` with dropout 0.3 | 26 | 66.7% | 75.7% |

  The move-type shares carry far more type information than stats: accuracy went from 22% to 79%. Remember what this measures, though: "can a learnset predict type?" is a much easier question than "can stats predict type?", because Pokémon learn many moves of their own type. Features this close to the answer are a mild form of **leakage**: legitimate to use, but high accuracy means "the learnset is a strong clue", not "the network got smarter". Abilities (Blaze, Torrent...) would be even closer to the answer.

  The stats-moves model has a 15-point gap between training and validation, but dropout lowered validation accuracy, as it did for stats. **A gap only matters if closing it improves validation accuracy.**

### Ideas not tried yet

- **Family-aware evaluation:** hold out whole evolution families (from `/evolution-chain`) for both kNN and the network, to measure how much they rely on recognizing relatives.
- **Predict both types:** use a multi-label output (18 sigmoid neurons, one per type, and `binaryCrossentropy`) instead of softmax.
- **Team builder:** suggest Pokémon whose types cover a team's weaknesses.
- **HTML page with sprites**, to explore neighbors visually.

## What I learned

### Encoding and vectors

- **Encoding is most of the work.** The network is about 20 lines; turning Pokémon into good numbers (`context.js`, `encode.js`) took the most thought, just like in exemplo-01.
- **The context belongs with the model.** Min/max values and the type index are computed from the whole dataset, and the server must encode new inputs with the *same* context. That's why `train.js` saves `metadata.json` next to the model.
- **One-hot vs. multi-hot.** A Pokémon's types use multi-hot (one or two positions on); the classifier's label uses one-hot (exactly one answer).
- **Feature weights define what "similar" means.** With types at 0.5, every neighbor shares a type. Lowering them reveals Pokémon with the same battle profile instead.
- **Euclidean vs. cosine.** Euclidean compares *size* (power level), cosine compares *shape* (stat profile). With types removed, cosine found evolution lines on its own (Charmander → Charmeleon, Charizard).
- **Skewed features need a log scale.** Min-max squeezed 92% of weights near 0. The log fixed the distribution, but it only helps if the feature carried missing information, which size didn't.

### Training and evaluation

- **Always compare against a baseline.** "22% accuracy" means nothing alone; "22% vs. 13% for always guessing water" shows the network learned something.
- **Small validation sets lie.** With 31 Pokémon in validation, one run swung between 29% and 42%. Averaging 10 runs is the minimum for comparing two setups.
- **Shuffle before `validationSplit`.** TensorFlow.js takes the *last* 20% without shuffling, which would have meant validating only on the newest generations.
- **Read train and validation accuracy together.** High train / low validation = overfitting (Gen 1: 72% vs. 37%). Both low = underfitting. Both high and close = the goal.
- **Better features beat everything else.** No network size, encoding tweak or regularization moved the stats model past ~22%. Adding the move-type shares took it to 79%. Information in the input matters more than the model.
- **Regularization isn't free.** Dropout and early stopping closed the gap and improved `val_loss`, but lowered accuracy. `val_loss` punishes confident mistakes; accuracy only checks the top guess.
- **More data beats tricks.** Going from 151 to 1025 Pokémon removed most of the overfitting by itself, which no regularization setting did.
- **A simple method can beat a neural network.** kNN with cosine distance, with no training, reached about 25% against the network's 22%. Always try a simple baseline method, not just a "dumb" baseline.
- **A weight is not influence.** A feature's effect on distance is its weight × how much it varies. Measure each group's share of the distance instead of trusting the weights.
- **Check *why* something works.** The kNN gain came from finding evolution relatives, which is a different skill than the one we set out to measure. Related samples (families) in both the reference and test data can inflate results.
- **Find the ceiling, and question it.** More data, a better encoding and a bigger network all left the network at about 21–23%, which looked like a limit in the data. kNN then reached 25% with the same inputs, by exploiting evolution families, a pattern the network didn't capture. A ceiling is only a ceiling *for a given method*.

### Practical gotchas

- **tfjs-node 4.22 breaks on Node 23+** (`util.isNullOrUndefined` was removed). `src/tf.js` restores it.
- **TensorFlow.js doesn't free tensors automatically.** Call `.dispose()` (or use `tf.tidy`) in long-running code like a server.
- **Load the model once,** at server startup, not on every request.
- **In a `callbacks` array, wrap plain objects in `new tf.CustomCallback(...)`**, or `fit` crashes. `earlyStopping` can't restore the best weights in TensorFlow.js.
- **SQL operators can't be query parameters.** Choose them from a fixed whitelist (`METRICS` in `server.js`), never from raw user input.
- **A pgvector column has a fixed size.** When the data gained a new type (25 → 26 numbers), and again with moves (26 → 44), the table had to be recreated.
