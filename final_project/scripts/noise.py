from opensimplex import noise2array
import numpy as np
from PIL import Image


def create_noise_map(shape: tuple, depth: int, scale: float, factor: float, persistence: float):
    final_noise = np.zeros(shape)
    amplitude = 1.0
    total_amplitude = 0.0
    for _ in range(depth):
        x = np.linspace(0, shape[0] / scale, shape[0])
        y = np.linspace(0, shape[1] / scale, shape[1])
        final_noise += noise2array(x, y) * amplitude
        total_amplitude += amplitude
        amplitude *= persistence
        scale *= factor
    return final_noise / total_amplitude


TERRAIN_COLORS = [
    (0.00, (15, 40, 90)),     # deep water
    (0.35, (30, 80, 160)),    # shallow water
    (0.42, (210, 190, 130)),  # sand
    (0.65, (50, 130, 50)),    # plains
    (0.82, (110, 100, 90)),   # mountains
    (0.95, (240, 240, 245)),  # mountain tops
]


def noise_to_image(noise_map: np.array):
    normalized = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    rgb = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    for i in range(len(TERRAIN_COLORS) - 1):
        t0, c0 = TERRAIN_COLORS[i]
        t1, _ = TERRAIN_COLORS[i + 1]
        mask = (normalized >= t0) & (normalized < t1)
        for ch in range(3):
            rgb[mask, ch] = c0[ch]
    mask_top = normalized >= TERRAIN_COLORS[-1][0]
    for ch in range(3):
        rgb[mask_top, ch] = TERRAIN_COLORS[-1][1][ch]
    return Image.fromarray(rgb, mode='RGB')


def generate_image():
    shape = (512, 512)
    depth = 6
    scale = 200
    factor = 0.45
    persistence = 0.4
    noise_map = create_noise_map(shape, depth, scale, factor, persistence)
    image = noise_to_image(noise_map)
    return image


def main():
    image = generate_image()
    image.show()


if __name__ == "__main__":
    main()