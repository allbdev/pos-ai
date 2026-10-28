# Duck Hunt AI: Object Detection with YOLO + TensorFlow.js

A browser version of Duck Hunt where **the computer plays by itself**. Every 200 ms the game takes a screenshot of the canvas, sends it to a pretrained **YOLOv5n** model running in a Web Worker, and shoots at whatever the model detects.

This was the web machine learning class of Module 1. The main point: **you don't always need to train a model. A pretrained one can be reused for a problem it was never trained on.**

The twist: YOLO was trained on the [COCO dataset](https://cocodataset.org/), which has 80 classes and no "duck" among them. When it looks at the pixel-art ducks, it labels them **`kite`**. That's good enough to aim at, so the code filters for `kite` and shoots at it.

## Running

```bash
npm install
npm start
```

Open `http://localhost:8080` (it opens automatically). The crosshair is hidden until the model loads. After that, it jumps to every duck the model finds and fires on its own. The HUD in the top-right corner shows the last prediction.

`npm run build` writes the bundle to `dist/`. Webpack also copies the model files from `machine-learning/yolov5n_web_model/` to `dist/yolov5n_web_model/`, so the worker can `fetch` them at runtime.

## How it works

All the ML code lives in [`machine-learning/`](./machine-learning). The game itself (`src/`) is the original Duck Hunt JS, almost untouched. The only change is in [`main.js`](./main.js), which calls `machine-learning/main.js` once the game has loaded.

```
 main thread (machine-learning/main.js)          Web Worker (machine-learning/worker.js)
 ──────────────────────────────────────          ───────────────────────────────────────
                                                  load labels.json + model.json
                                                  warm-up run with a dummy tensor
 every 200 ms:
   renderer.extract.canvas(stage)  ── screenshot
   createImageBitmap(canvas)
   postMessage(bitmap, [bitmap])   ─────────────▶ preprocessImage()   pixels → [1, 640, 640, 3], 0–1
                                                  runInference()      boxes, scores, classes
                                                  processPrediction() score ≥ 0.4 and label 'kite'
                                                                      → center (x, y) of the box
   onmessage ◀───────────────────────────────────  postMessage({ type: 'prediction', x, y, score })
   aim.setPosition(x, y)
   game.handleClick({ global: aim position })  ── 🔫 shoot
   updateHUD()
```

### 1. Loading the model

YOLOv5n ("nano") was exported to the **TensorFlow.js graph-model** format: a `model.json` with the graph plus two binary `.bin` files with the weights (~8 MB in total). `labels.json` holds the 80 COCO class names in the same order as the model's class indexes, so `classes[i] = 33` means `labels[33] = 'kite'`.

- `tf.loadGraphModel()` loads it. A graph model is **inference only**: it can't be trained further like the `tf.sequential()` models from the previous classes.
- **Warm-up:** right after loading, the worker runs the model once on a tensor of ones. The first run is slow because the WebGL backend compiles its shaders. Doing it upfront keeps the first real frame fast.

### 2. Pre-processing a frame

The model expects a fixed input shape: `[1, 640, 640, 3]`.

| Step | Tensor shape | Why |
|---|---|---|
| `tf.browser.fromPixels(bitmap)` | `[H, W, 3]` | Image → RGB numbers 0–255 |
| `tf.image.resizeBilinear(..., [640, 640])` | `[640, 640, 3]` | The size the model was trained with |
| `.div(255)` | same | **Normalization** to 0–1, the same idea as min-max in the earlier classes |
| `.expandDims(0)` | `[1, 640, 640, 3]` | Models work on **batches**; this is a batch of one |

All of this runs inside `tf.tidy()`, which disposes the intermediate tensors automatically.

### 3. Reading the output

This export of YOLOv5 already includes non-max suppression (NMS), so it returns a short, clean list of detections instead of thousands of raw boxes. The code uses the first three outputs:

- **boxes:** `[x1, y1, x2, y2]` per detection, normalized to 0–1;
- **scores:** confidence from 0 to 1;
- **classes:** an index into `labels.json`.

`processPrediction()` is a generator (`function*`) that:

1. skips scores below `CLASS_THRESHOLD` (0.4);
2. skips any label other than `kite`;
3. scales the box back to canvas pixels (× width, × height);
4. yields the **center** of the box, which is where we shoot.

### 4. Shooting

For each prediction, the main thread moves the crosshair to `(x, y)` and calls `game.handleClick()` with the crosshair's global position. It goes through the exact same code path as a human mouse click: it spends a bullet, checks for hits, and updates the score.

### 5. Why a Web Worker

Inference on a 640×640 image takes tens of milliseconds. On the main thread, that would freeze the game's animation loop. In the worker, the game keeps running at full speed while the model thinks.

The screenshot is sent as an `ImageBitmap` in the **transfer list** (`postMessage(msg, [bitmap])`). That moves the image to the worker without copying it, which matters when you do it five times per second.

## What I learned

- **Reusing a pretrained model is a superpower.** Nobody labeled a single duck. A general-purpose detector was close enough, and filtering by the "wrong" class (`kite`) was all it took.
- **A model only knows its training classes.** YOLO can't say "duck" because COCO has no ducks. It picks the nearest thing it knows. That's both the trick and the limitation here.
- **Detection ≠ classification.** The classifiers in the previous classes answered "what is this?". A detector answers "what is where?": a class *and* a box for every object in the image.
- **Images are just tensors.** The pipeline is the same as before: turn the data into numbers, normalize, add a batch dimension, predict. The only difference is the shape, `[1, 640, 640, 3]` instead of `[1, 14]`.
- **The input must match the training format.** Same size (640×640), same scale (0–1), same channel order. Feed it anything else and the predictions get worse with no error.
- **Graph model vs. layers model.** `loadGraphModel` is a frozen, optimized graph for inference. `loadLayersModel` (and the `tf.sequential` models from the previous classes) can still be trained.
- **Memory is your job.** `tf.tidy()` and `dispose()` are needed in a loop that runs five times per second, or GPU memory grows until the tab crashes.
- **Workers and transferables keep the UI smooth.** This is the same lesson as the e-commerce class, now under real-time pressure.

## Notes and recommendations

These came up while reviewing the code after the class.

**Frames can pile up.** `setInterval` sends a frame every 200 ms whether or not the worker finished the previous one. On a slow machine, inference takes longer than 200 ms, so frames queue up and the shots land where the duck *was*. Better: send the next frame only after the worker replies (a simple `busy` flag), or use `requestAnimationFrame` with that same check.

**The CDN version isn't pinned.** `importScripts('.../@tensorflow/tfjs@latest')` means a new tfjs release can break the project overnight. Pin an exact version (for example `@tensorflow/tfjs@4.22.0`) or install it with npm and bundle it.

**The HUD's "Score" is the model's confidence,** not the game score. `updateHUD(data)` prints `data.score`, which is the detection score in %. Renaming it to "Confidence" would avoid confusion.

**The image gets stretched.** The canvas isn't square, but it's resized straight to 640×640. YOLO was trained on **letterboxed** images (resized with the aspect ratio kept, then padded to a square). It works here because the ducks are easy targets, but letterboxing usually gives better boxes.

**Every detection becomes a shot.** If the model sees two `kite`s, it fires twice, even when they're the same duck seen twice. Shooting only the highest-scoring detection per frame would save bullets.

**`kite` is a hack.** It works because the ducks happen to look like kites to this model. For a real product, you would **fine-tune** YOLO on a few hundred labeled duck screenshots, so it gets a real `duck` class and higher confidence. That's transfer learning, the natural next step after this class.

**Inherited files.** `infrastructure.tf`, `terraform.tfstate*` and `s3policy.json` deploy the original author's site (duckhuntjs.com) to AWS. They aren't used here and could be deleted.

## Project structure

```
├── main.js                     # Entry point: starts the game, then the AI (machine-learning/main.js)
├── machine-learning/           # ⭐ Everything added in this class
│   ├── main.js                 # Main thread: screenshots, worker messages, aim + shoot
│   ├── worker.js               # Web Worker: loads YOLO, preprocesses, runs inference, filters 'kite'
│   ├── layout.js               # PixiJS HUD with the last prediction
│   └── yolov5n_web_model/      # Pretrained YOLOv5n (TF.js graph model) + 80 COCO labels
├── src/                        # Original Duck Hunt JS game (PixiJS)
│   ├── modules/                # Game, Stage, Duck, Dog, Aim, Hud, Sound...
│   ├── data/levels.json
│   └── assets/                 # Sprites and sounds
├── dist/                       # Built bundle + copied model (served by the dev server)
└── webpack.config.js           # Bundles main.js; copies the model into dist/
```

## Original game

The game is [DuckHunt-JS](https://github.com/MattSurabian/DuckHunt-JS) by Matt Surabian (MIT license, see [`LICENSE`](./LICENSE)). It renders with PixiJS (WebGL with a Canvas fallback), plays audio with Howler, and animates with PixiJS sprites plus GreenSock tweens.

Other tasks from the original repo:

- `npm run lint`: ESLint on `src/`
- `npm run audio`: rebuild the audio sprite (needs [ffmpeg](https://ffmpeg.org/download.html))
- `npm run images`: rebuild the image sprite (needs [TexturePacker](https://www.codeandweb.com/texturepacker/download))

## References

- [YOLOv5 (Ultralytics)](https://github.com/ultralytics/yolov5), the model and its TF.js export
- [COCO dataset](https://cocodataset.org/), the 80 classes the model knows
- [TensorFlow.js: loading a graph model](https://www.tensorflow.org/js/guide/save_load)
- [MDN: Transferable objects](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects)
