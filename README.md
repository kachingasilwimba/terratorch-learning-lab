# TerraTorch Learning Lab

[Open the interactive lab website](https://kachingasilwimba.com/terratorch-learning-lab/)

Follow two satellite observations from six reflectance bands to Prithvi-EO patch embeddings. The lab uses the IBM–NASA **Prithvi-EO 2.0 Tiny-TL** encoder through TerraTorch. Start with a synthetic scene, then repeat the workflow on a public Harmonized Landsat Sentinel-2 (HLS) image pair from Mexico.

## Set up

Use Python 3.12 and run the commands from the repository root. On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

This installs the dependencies for inspecting the synthetic scene. Install the CPU build of PyTorch, followed by TerraTorch and the geospatial dependencies, for the remaining lessons. CPU is the default device. The first pretrained run downloads an approximately 129 MB checkpoint.

```sh
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-model.txt
```

On Apple Silicon use the regular PyPI build via `python -m pip install -r requirements-model.txt`. With a compatible CUDA build of PyTorch 2.8.0 and torchvision 0.23.0, use `encode --device cuda`.

## Lesson 1 · Read the six bands

```sh
python -m terratorch_lab inspect
```

Open `lab-output/scene.png`. The top row combines red, green and blue as natural color. The bottom row shows NDVI, computed as `(near infrared − red) / (near infrared + red)`. Find the patch with the largest change in vegetation and compare how it looks in both rows. The two dates and reflectance patterns in this first exercise are synthetic.

The encoder uses blue, green, red, narrow near infrared, shortwave infrared 1, and shortwave infrared 2, in that order. In HLS S30 these correspond to `B02`, `B03`, `B04`, `B8A`, `B11`, and `B12`. This lab keeps HLS reflectance scaled by 10,000 for compatibility with the checkpoint's per-band means and standard deviations. It assigns missing entries the normalized value zero as a simple teaching convention. Operational analyses also need cloud, shadow, snow, and other quality masks.

## Lesson 2 · Extract patch embeddings

```sh
python -m terratorch_lab encode
```

The normalized input has shape `[1, 6, 2, 64, 64]` in batch, band, time, height and width order. The encoder divides each date into a `4 × 4` grid of `16 × 16` patches. The final layer returns a scene token and 32 patch tokens, each with 192 features. The CLI saves these as `cls_token` with shape `[192]` and `patch_tokens` with shape `[2, 4, 4, 192]` in `lab-output/embeddings.npz`.

Open `lab-output/embedding-change.png`. Each cell shows the cosine distance between embeddings at the same image location on the first and last dates. A value of 0 means the vectors point in the same direction, and 2 means they point in opposite directions. The color scale is fitted to each run, so read the legend when comparing maps. See if the field from Lesson 1 stands out. A patch distance measures a difference in the encoder's representation. Seasonality, acquisition quality, and the neighboring patches can all affect it. A validated change-detection task requires labels, a task-specific prediction head or decision rule, and evaluation.

Try `encode --omit-coordinates --output-dir lab-output/no-coordinates` and compare the maps. With coordinates enabled, the encoder receives the year and day of year for each frame and latitude and longitude for the scene when available. The default synthetic scene provides dates only. The upstream model was trained with four frames. This exercise uses two dates so you can trace each embedding to its input patch.

To study the architecture offline after installing dependencies, pass `encode --random-weights --output-dir lab-output/random`. To use an existing trusted copy of the official checkpoint, pass `encode --checkpoint /path/to/Prithvi_EO_V2_tiny_TL.pt`. On a network that blocks Hugging Face Xet transfers, set `HF_HUB_DISABLE_XET=1` before running `encode` to use an HTTP download.

## Lesson 3 · Repeat with HLS observations

```sh
python -m terratorch_lab sample
python -m terratorch_lab inspect --input lab-output/hls-sample.npz --output-dir lab-output/hls
python -m terratorch_lab encode --input lab-output/hls-sample.npz --output-dir lab-output/hls
```

The sample command verifies and downloads two pinned upstream HLS S30 GeoTIFFs for tile T13REM dated January 26 and July 20, 2018. It extracts the same central `224 × 224` window from both rasters. The resulting embedding grid has `14 × 14` patches per date. Compare the RGB, NDVI and patch-distance views. Which patterns agree, and which might come from clouds, phenology or the embedding's spatial context?

For your own co-registered observations, supply an `.npz` archive with these arrays. Use consistent pixel grids and reflectance units across dates.

| Array | Contents |
| --- | --- |
| `pixels` | Numeric `[time, 6, height, width]` array. Height and width must be positive multiples of 16. Multiply unit-scale reflectance by 10,000 and use `-9999` or `NaN` for missing entries. |
| `bands` | Unicode strings `BLUE`, `GREEN`, `RED`, `NIR_NARROW`, `SWIR_1`, `SWIR_2` in that exact order. |
| `dates` | One `YYYY-MM-DD` string per frame, in strictly increasing order. |
| `location` | Optional two-element array `[latitude, longitude]` in decimal degrees. |
| `source` | Optional scalar string naming the scene. |

## Website and checks

Use the [companion site](https://kachingasilwimba.com/terratorch-learning-lab/) to compare the two HLS dates. The site includes true-color and false-color composites, mean NDVI, and precomputed cosine distances from the lab's pretrained Tiny-TL run with date and location coordinates. To serve the site locally, run `python -m http.server 8000 --directory dist` and open `http://localhost:8000`.

Run `python -m unittest discover -s tests` to check array layout, normalization, NDVI, and patch comparisons with the lightweight dependencies. All CLI artifacts are written under `lab-output/` by default.

## Sources

- [TerraTorch](https://github.com/torchgeo/terratorch/tree/v1.2.14)
- [IBM–NASA Prithvi-EO 2.0 Tiny-TL checkpoint, model card, and example GeoTIFFs](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL/tree/335eadc2c45ad5abe7bd307223e1c48c5b60c41b)
- [NASA HLS Sentinel-2 product and band definitions](https://hls.gsfc.nasa.gov/products-description/s30/)
