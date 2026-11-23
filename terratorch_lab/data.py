"""HLS inputs and preprocessing for Prithvi-EO 2.0."""

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

import numpy as np

BANDS = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")
PATCH_SIZE = 16
SCALE = 10_000
NODATA = -9999
MEAN = np.array([1087, 1342, 1433, 2734, 1958, 1363], dtype=np.float32)
STD = np.array([2248, 2179, 2178, 1850, 1242, 1049], dtype=np.float32)

SAMPLE_REPO = "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL"
SAMPLE_REVISION = "335eadc2c45ad5abe7bd307223e1c48c5b60c41b"
SAMPLE_ACQUISITIONS = (
    ("2018-01-26", "2018026T173609", "e34c1e8f6b69092bbf16f87da1a0c2337e8e53f28d172d8076e2efab292b795d"),
    ("2018-07-20", "2018201T172901", "fce050cc821ebec2974e85cfe702c0f093d74caf12196adb7ee88c8a30773d4f"),
)


@dataclass(frozen=True)
class Scene:
    """Co-registered frames in [time, band, height, width] order."""

    pixels: np.ndarray
    dates: tuple[date, ...]
    source: str
    location: tuple[float, float] | None = None  # latitude, longitude

    def __post_init__(self):
        shape = self.pixels.shape
        if len(shape) != 4 or shape[0] == 0 or shape[1] != len(BANDS):
            raise ValueError("Expected a nonempty [time, 6, height, width] pixel array.")
        if any(size == 0 or size % PATCH_SIZE for size in shape[-2:]):
            raise ValueError(f"Height and width must be positive multiples of {PATCH_SIZE}.")
        if len(self.dates) != shape[0] or any(a >= b for a, b in zip(self.dates, self.dates[1:])):
            raise ValueError("Supply one ISO date per frame in strictly increasing order.")
        if self.location is not None:
            latitude, longitude = self.location
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                raise ValueError("Location must be latitude and longitude in decimal degrees.")
        if not valid_pixels(self.pixels).any():
            raise ValueError("The scene contains no valid reflectance pixels.")


def valid_pixels(pixels: np.ndarray) -> np.ndarray:
    return np.isfinite(pixels) & (pixels != NODATA)


def synthetic_scene() -> Scene:
    """Generate two repeatable, illustrative frames with one changed region."""
    rng = np.random.default_rng(12)
    y, x = np.indices((64, 64))
    field = (x > 36) & (y > 28)
    water = abs(x - 14 - 3 * np.sin(y / 10)) < 6
    pixels = np.empty((2, 6, 64, 64), dtype=np.float32)
    for index in range(len(pixels)):
        pixels[index] = np.array([1000, 1400, 1500, 2700, 2100, 1800])[:, None, None]
        pixels[index][:, water] = np.array([640, 720, 490, 260, 180, 120])[:, None]
        if index == 1:
            pixels[index][:, field] = np.array([730, 1150, 660, 4700, 1900, 1200])[:, None]
        pixels[index] += rng.normal(0, 75, size=pixels[index].shape)
    pixels[:, :, :3, :6] = NODATA
    return Scene(pixels, (date(2020, 1, 26), date(2020, 7, 20)), "Synthetic practice scene")


def load_scene(path: Path) -> Scene:
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["bands"], BANDS):
            raise ValueError(f"Expected bands in this order: {', '.join(BANDS)}.")
        if archive["dates"].ndim != 1:
            raise ValueError("Dates must be a one-dimensional array of ISO dates.")
        dates = tuple(date.fromisoformat(value) for value in archive["dates"].tolist())
        location = archive["location"] if "location" in archive else None
        if location is not None and location.shape != (2,):
            raise ValueError("Location must be a [latitude, longitude] vector.")
        source = str(archive["source"].item()) if "source" in archive else path.name
        return Scene(
            archive["pixels"].astype(np.float32),
            dates,
            source,
            tuple(location.astype(float)) if location is not None else None,
        )


def model_input(scene: Scene) -> np.ndarray:
    """Standardize each band, fill missing values at its mean, and make [B,C,T,H,W]."""
    mean, std = MEAN[None, :, None, None], STD[None, :, None, None]
    raw = scene.pixels.astype(np.float32, copy=False)
    pixels = np.where(valid_pixels(raw), raw, mean)
    normalized = (pixels - mean) / std
    return np.ascontiguousarray(normalized.transpose(1, 0, 2, 3)[None, ...])


def ndvi(pixels: np.ndarray) -> np.ndarray:
    """Compute (near infrared - red) / (near infrared + red) for each date."""
    red, nir = pixels[:, 2:4].astype(np.float32, copy=False).transpose(1, 0, 2, 3)
    usable = valid_pixels(nir) & valid_pixels(red)
    red, nir = np.where(usable, red, 0), np.where(usable, nir, 0)
    denominator = nir + red
    result = np.full_like(red, np.nan, dtype=np.float32)
    return np.divide(nir - red, denominator, out=result, where=denominator != 0)


def download_sample(path: Path) -> None:
    """Fetch two pinned IBM–NASA example GeoTIFFs and save their common central crop."""
    from rasterio.io import MemoryFile
    from rasterio.warp import transform
    from rasterio.windows import Window

    frames = []
    grid = None
    for _, timestamp, checksum in SAMPLE_ACQUISITIONS:
        filename = f"Mexico_HLS.S30.T13REM.{timestamp}.v2.0_cropped.tif"
        url = f"{SAMPLE_REPO}/resolve/{SAMPLE_REVISION}/examples/{filename}"
        with urlopen(url, timeout=60) as response:
            geotiff = response.read(4_000_000)
        if sha256(geotiff).hexdigest() != checksum:
            raise ValueError(f"Sample checksum did not match for {filename}.")

        with MemoryFile(geotiff) as memory, memory.open() as raster:
            crop = Window((raster.width - 224) // 2, (raster.height - 224) // 2, 224, 224)
            cropped_transform = raster.window_transform(crop)
            current_grid = (raster.crs, raster.transform, raster.width, raster.height)
            if raster.count != len(BANDS) or (grid is not None and current_grid != grid):
                raise ValueError("Expected six bands and identical spatial grids for both sample dates.")
            grid = current_grid
            frame = raster.read(window=crop, masked=True).astype(np.float32)
            frames.append(frame.filled(NODATA))

    center_x, center_y = cropped_transform * (112, 112)
    longitudes, latitudes = transform(grid[0], "EPSG:4326", [center_x], [center_y])
    np.savez_compressed(
        path,
        pixels=np.stack(frames),
        bands=BANDS,
        dates=[acquisition[0] for acquisition in SAMPLE_ACQUISITIONS],
        location=[latitudes[0], longitudes[0]],
        source="IBM–NASA Mexico HLS sample · tile T13REM",
    )
