import os
from neural_networks.final_project.scripts.noise import create_noise_map, noise_to_image

PRESETS_DIR = os.path.join(os.path.dirname(__file__), "presets")

PRESETS = {
    "preset_01": {"shape": (128, 128), "depth": 6, "scale": 200, "factor": 0.45, "persistence": 0.4},
    "preset_02": {"shape": (128, 128), "depth": 7, "scale": 180, "factor": 0.48, "persistence": 0.5},
    "preset_03": {"shape": (128, 128), "depth": 5, "scale": 250, "factor": 0.42, "persistence": 0.45},
    "preset_04": {"shape": (128, 128), "depth": 8, "scale": 160, "factor": 0.52, "persistence": 0.35},
    "preset_05": {"shape": (128, 128), "depth": 4, "scale": 300, "factor": 0.40, "persistence": 0.55},
    "preset_06": {"shape": (128, 128), "depth": 6, "scale": 220, "factor": 0.55, "persistence": 0.42},
    "preset_07": {"shape": (128, 128), "depth": 7, "scale": 140, "factor": 0.46, "persistence": 0.48},
    "preset_08": {"shape": (128, 128), "depth": 5, "scale": 270, "factor": 0.50, "persistence": 0.38},
    "preset_09": {"shape": (128, 128), "depth": 9, "scale": 190, "factor": 0.43, "persistence": 0.32},
    "preset_10": {"shape": (128, 128), "depth": 6, "scale": 170, "factor": 0.38, "persistence": 0.52},
}


def main():
    os.makedirs(PRESETS_DIR, exist_ok=True)
    for name, params in PRESETS.items():
        print(f"Generating {name}...")
        noise_map = create_noise_map(**params)
        image = noise_to_image(noise_map)
        image.save(os.path.join(PRESETS_DIR, f"{name}.png"))
        print(f"  Saved presets/{name}.png")


if __name__ == "__main__":
    main()
