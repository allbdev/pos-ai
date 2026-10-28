# AI Studies

My study notes and hands-on exercises from the course **Fundamentos de IA e LLMs para Programadores** (AI and LLM fundamentals for programmers).

I use this repository to track my progress: theory notes in Markdown, plus small JavaScript projects that apply each concept with [TensorFlow.js](https://www.tensorflow.org/js).

## Structure

```
IA/
└── Modulo-01/                          # Module 1: AI fundamentals
    ├── *.pdf / *.docx                  # Course material and reading list
    ├── machine-learning/
    │   ├── conceito-redes-neurais.md   # Notes: how neural networks work (data → tensors → model)
    │   ├── teachable-machine.md        # Notes: training an image classifier with Kaggle + Teachable Machine
    │   ├── exemplo-00/                 # Node + tfjs-node: classify people as premium / medium / basic
    │   └── archive/                    # Kaggle "100 sports" image dataset (train / valid / test)
    ├── deep-learning/
    │   ├── exemplo-01-ecommerce-recomendations-template/
    │   │                               # Browser app: product recommendations trained in a Web Worker
    │   └── exemplo-02-pokemon-similarity/
    │                                   # Node + pgvector: Pokémon similarity search + type classifier
    └── web-machine-learning/
        └── exemplo03-duck-hunt-js/     # Browser: pretrained YOLO plays Duck Hunt by itself
```

## Modules

| Module | Topic | Status |
|---|---|---|
| [Modulo-01](./Modulo-01) | Machine learning and deep learning fundamentals: neural networks, normalization, one-hot encoding, training with TensorFlow.js, vector search with pgvector, object detection with a pretrained YOLO model | In progress |

## Running the examples

Each example is its own npm project:

```bash
# Neural network classifier (Node)
cd Modulo-01/machine-learning/exemplo-00
npm install
npm start

# E-commerce recommendations (browser)
cd Modulo-01/deep-learning/exemplo-01-ecommerce-recomendations-template
npm install
npm start

# Pokémon similarity + type classifier (Node + Docker)
cd Modulo-01/deep-learning/exemplo-02-pokemon-similarity
npm install
docker compose up -d
npm run ingest && npm run ingest:moves && npm run seed && npm run train
npm start                # page on http://localhost:3001

# Duck Hunt played by YOLO (browser)
cd Modulo-01/web-machine-learning/exemplo03-duck-hunt-js
npm install
npm start                # http://localhost:8080
```

See each project's own README or notes file for details.
