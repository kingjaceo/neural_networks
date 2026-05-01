import argparse
import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
import pygame
import numpy as np
import heapq
import math
from scripts.noise import create_noise_map, noise_to_image

GRID_SIZE = 128
MAP_DISPLAY = 512
SCALE = MAP_DISPLAY // GRID_SIZE  # 4 pixels per grid cell
WIN_W = 800
WIN_H = 640
MAP_X = (WIN_W - MAP_DISPLAY) // 2   # 144
MAP_Y = (WIN_H - MAP_DISPLAY) // 2   # 64
BASE_SPEED = 150  # display-pixels per second at 1.0x terrain

TERRAIN_SPEEDS = [
    (0.00, 0.0),   # deep water
    (0.35, 0.0),   # shallow water
    (0.42, 0.6),   # sand
    (0.65, 1.0),   # plains
    (0.82, 0.4),   # mountains
    (0.95, 0.2),   # mountain tops
]

LEGEND = [
    ("Deep Water",    (15,  40,  90),  "impassable"),
    ("Shallow Water", (30,  80, 160),  "impassable"),
    ("Sand",          (210,190, 130),  "0.6x speed"),
    ("Plains",        (50, 130,  50),  "1.0x speed"),
    ("Mountains",     (110,100,  90),  "0.4x speed"),
    ("Peaks",         (240,240, 245),  "0.2x speed"),
]

UI_BG   = (25,  25,  35)
UI_FG   = (220, 220, 220)
UI_DIM  = (130, 130, 150)
GOLD    = (255, 210,  50)
CYAN    = (0,   200, 255)
GREEN   = (0,   220,  80)
RED     = (220,  50,  50)


def build_speed_map(noise_map):
    norm = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    speed = np.zeros(norm.shape, dtype=np.float32)
    for i in range(len(TERRAIN_SPEEDS) - 1):
        t0, s0 = TERRAIN_SPEEDS[i]
        t1, _ = TERRAIN_SPEEDS[i + 1]
        speed[(norm >= t0) & (norm < t1)] = s0
    speed[norm >= TERRAIN_SPEEDS[-1][0]] = TERRAIN_SPEEDS[-1][1]
    return speed


def generate_terrain_perlin():
    noise_map = create_noise_map((GRID_SIZE, GRID_SIZE), 6, 200, 0.45, 0.4)
    return noise_to_image(noise_map), build_speed_map(noise_map)


def generate_terrain_gan(preset, checkpoint_dir):
    from scripts.gan_terrain import generate_terrain
    return generate_terrain(preset, checkpoint_dir=checkpoint_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Terrain Pathfinder")
    p.add_argument("--source", choices=["perlin", "gan"], default="perlin",
                   help="terrain generator: 'perlin' (default) or 'gan' (load a trained DCGAN)")
    p.add_argument("--preset", default="preset_01",
                   help="preset name for --source gan (must match a checkpoints/<preset>/ dir)")
    p.add_argument("--checkpoint_dir", default=os.path.join(_PROJECT_ROOT, "checkpoints"),
                   help="root dir holding <preset>/generator.npz files")
    return p.parse_args()


def pick_land_points(speed_map, min_dist=50):
    """Pick start and end in grid coordinates, at least min_dist grid cells apart."""
    land = np.argwhere(speed_map > 0)
    for _ in range(2000):
        i1, i2 = np.random.randint(len(land), size=2)
        s, e = land[i1], land[i2]
        if math.hypot(s[0] - e[0], s[1] - e[1]) >= min_dist:
            return (int(s[0]), int(s[1])), (int(e[0]), int(e[1]))
    return (int(land[0][0]), int(land[0][1])), (int(land[-1][0]), int(land[-1][1]))


def astar(speed_map, start, end):
    rows, cols = speed_map.shape
    INF = float('inf')
    g = np.full((rows, cols), INF, dtype=np.float64)
    g[start[0], start[1]] = 0.0
    came_from = {}
    closed = np.zeros((rows, cols), dtype=bool)
    max_spd = max(s for _, s in TERRAIN_SPEEDS if s > 0)
    er, ec = end
    SQRT2 = 1.4142135623730951
    DIRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]
    heap = [(math.hypot(start[0] - er, start[1] - ec) / max_spd, start[0], start[1])]

    while heap:
        _, r, c = heapq.heappop(heap)
        if closed[r, c]:
            continue
        closed[r, c] = True
        if r == er and c == ec:
            path, cr, cc = [], er, ec
            while (cr, cc) in came_from:
                path.append((cr, cc))
                cr, cc = came_from[(cr, cc)]
            path.append(start)
            path.reverse()
            return path, g[er, ec]
        gr = g[r, c]
        for dr, dc, dist in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not closed[nr, nc]:
                spd = speed_map[nr, nc]
                if spd > 0:
                    new_g = gr + dist / spd
                    if new_g < g[nr, nc]:
                        g[nr, nc] = new_g
                        came_from[(nr, nc)] = (r, c)
                        h = math.hypot(nr - er, nc - ec) / max_spd
                        heapq.heappush(heap, (new_g + h, nr, nc))

    return None, INF


def to_screen(display_x, display_y):
    return (int(display_x) + MAP_X, int(display_y) + MAP_Y)


def grid_to_display(row, col):
    return col * SCALE + SCALE // 2, row * SCALE + SCALE // 2


def terrain_at(speed_map, display_x, display_y):
    r = int(np.clip(display_y, 0, MAP_DISPLAY - 1)) // SCALE
    c = int(np.clip(display_x, 0, MAP_DISPLAY - 1)) // SCALE
    return speed_map[r, c]


def draw_panel_bg(surface):
    surface.fill(UI_BG)


def draw_legend(surface, font, small_font):
    x, y = 10, MAP_Y + 10
    label = font.render("TERRAIN", True, UI_DIM)
    surface.blit(label, (x, y))
    y += label.get_height() + 8
    for name, color, spd in LEGEND:
        pygame.draw.rect(surface, color, (x, y, 14, 14))
        pygame.draw.rect(surface, UI_DIM, (x, y, 14, 14), 1)
        surface.blit(small_font.render(name, True, UI_FG), (x + 20, y - 1))
        surface.blit(small_font.render(spd, True, UI_DIM), (x + 20, y + 13))
        y += 36


def draw_top_bar(surface, title_font):
    pygame.draw.line(surface, UI_DIM, (MAP_X, MAP_Y - 2), (MAP_X + MAP_DISPLAY, MAP_Y - 2), 1)
    t = title_font.render("TERRAIN PATHFINDER", True, UI_FG)
    surface.blit(t, (WIN_W // 2 - t.get_width() // 2, (MAP_Y - t.get_height()) // 2))


def draw_bottom_bar(surface, font, small_font, playing, finished, pulse=1.0):
    pygame.draw.line(surface, UI_DIM, (MAP_X, MAP_Y + MAP_DISPLAY + 2), (MAP_X + MAP_DISPLAY, MAP_Y + MAP_DISPLAY + 2), 1)
    bar_top = MAP_Y + MAP_DISPLAY + 2
    bar_h = WIN_H - bar_top

    if finished:
        t = small_font.render("R  restart   |   ESC  quit", True, UI_DIM)
        surface.blit(t, (WIN_W // 2 - t.get_width() // 2, bar_top + (bar_h - t.get_height()) // 2))
    elif playing:
        t = small_font.render("WASD / Arrows  move   |   ESC  quit   |   R  restart", True, UI_DIM)
        surface.blit(t, (WIN_W // 2 - t.get_width() // 2, bar_top + (bar_h - t.get_height()) // 2))
    else:
        space_color = tuple(int(c * pulse) for c in (255, 220, 60))
        st = font.render("PRESS  SPACE  TO  START", True, space_color)
        ht = small_font.render("WASD / Arrows  move   |   ESC  quit   |   R  restart", True, UI_DIM)
        total_h = st.get_height() + 4 + ht.get_height()
        y = bar_top + (bar_h - total_h) // 2
        surface.blit(st, (WIN_W // 2 - st.get_width() // 2, y))
        surface.blit(ht, (WIN_W // 2 - ht.get_width() // 2, y + st.get_height() + 4))


def draw_right_panel(surface, font, playing, finished, elapsed, optimal_time):
    x = MAP_X + MAP_DISPLAY + 8
    y = MAP_Y + 10

    if not playing and not finished:
        lines = [
            (font,       "Optimal",    UI_DIM),
            (font,       f"{optimal_time:.1f}s",  GOLD),
        ]
        for f_, text, color in lines:
            t = f_.render(text, True, color)
            surface.blit(t, (x, y))
            y += t.get_height() + 4
        return

    if playing or finished:
        label = font.render("Time", True, UI_DIM)
        surface.blit(label, (x, y))
        y += label.get_height() + 4
        time_color = CYAN if not finished else (GREEN if elapsed <= optimal_time * 1.5 else RED)
        t = font.render(f"{elapsed:.1f}s", True, time_color)
        surface.blit(t, (x, y))
        y += t.get_height() + 16

        opt_label = font.render("Optimal", True, UI_DIM)
        surface.blit(opt_label, (x, y))
        y += opt_label.get_height() + 4
        surface.blit(font.render(f"{optimal_time:.1f}s", True, GOLD), (x, y))
        y += font.get_height() + 16

    if finished:
        ratio = elapsed / optimal_time
        if ratio < 1.2:
            grade, color = "PERFECT", GREEN
        elif ratio < 1.6:
            grade, color = "GREAT",   GREEN
        elif ratio < 2.2:
            grade, color = "GOOD",    GOLD
        elif ratio < 3.5:
            grade, color = "OK",      GOLD
        else:
            grade, color = "TRY AGAIN", RED
        surface.blit(font.render(f"{ratio:.2f}x", True, UI_FG), (x, y))
        y += font.get_height() + 8
        g_surf = font.render(grade, True, color)
        surface.blit(g_surf, (x, y))


def main(cfg=None):
    if cfg is None:
        cfg = parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    caption = "Terrain Pathfinder" + (f" — DCGAN ({cfg.preset})" if cfg.source == "gan" else "")
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont(None, 30)
    font = pygame.font.SysFont(None, 26)
    small_font = pygame.font.SysFont(None, 21)

    # Loading screen
    screen.fill(UI_BG)
    loading_text = "Generating terrain (DCGAN)..." if cfg.source == "gan" else "Generating terrain..."
    msg = font.render(loading_text, True, UI_FG)
    screen.blit(msg, (WIN_W // 2 - msg.get_width() // 2, WIN_H // 2))
    pygame.display.flip()

    if cfg.source == "gan":
        image, speed_map = generate_terrain_gan(cfg.preset, cfg.checkpoint_dir)
    else:
        image, speed_map = generate_terrain_perlin()

    base_surface = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
    terrain_surface = pygame.transform.scale(base_surface, (MAP_DISPLAY, MAP_DISPLAY))

    while True:
        start, end = pick_land_points(speed_map)

        screen.fill(UI_BG)
        screen.blit(terrain_surface, (MAP_X, MAP_Y))
        sdx, sdy = grid_to_display(start[0], start[1])
        edx, edy = grid_to_display(end[0], end[1])
        pygame.draw.circle(screen, GREEN, to_screen(sdx, sdy), 8, 2)
        pygame.draw.circle(screen, RED,   to_screen(edx, edy), 8, 2)
        msg = font.render("Computing optimal path...", True, UI_FG)
        screen.blit(msg, (WIN_W // 2 - msg.get_width() // 2, WIN_H // 2))
        pygame.display.flip()

        path, optimal_cost = astar(speed_map, start, end)
        if path is not None:
            break

    optimal_time = optimal_cost * SCALE / BASE_SPEED

    path_surface = terrain_surface.copy()
    path_disp = [grid_to_display(r, c) for r, c in path]
    if len(path_disp) > 1:
        pygame.draw.lines(path_surface, (255, 220, 0), False, path_disp, 3)

    start_disp = grid_to_display(start[0], start[1])
    end_disp   = grid_to_display(end[0],   end[1])
    finish_radius_disp = SCALE * 3

    player_pos = [float(start_disp[0]), float(start_disp[1])]
    playing = False
    finished = False
    elapsed = 0.0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not playing and not finished:
                    playing = True
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    pygame.quit()
                    return main(cfg)

        if playing and not finished:
            elapsed += dt
            keys = pygame.key.get_pressed()
            dx, dy = 0.0, 0.0
            if keys[pygame.K_LEFT]  or keys[pygame.K_a]: dx -= 1
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: dx += 1
            if keys[pygame.K_UP]    or keys[pygame.K_w]: dy -= 1
            if keys[pygame.K_DOWN]  or keys[pygame.K_s]: dy += 1

            if dx != 0 or dy != 0:
                length = math.hypot(dx, dy)
                dx /= length
                dy /= length
                spd = terrain_at(speed_map, player_pos[0], player_pos[1]) * BASE_SPEED
                new_x = float(np.clip(player_pos[0] + dx * spd * dt, 0, MAP_DISPLAY - 1))
                new_y = float(np.clip(player_pos[1] + dy * spd * dt, 0, MAP_DISPLAY - 1))
                if terrain_at(speed_map, new_x, new_y) > 0:
                    player_pos[0] = new_x
                    player_pos[1] = new_y

            if math.hypot(player_pos[0] - end_disp[0], player_pos[1] - end_disp[1]) < finish_radius_disp:
                finished = True

        # --- draw ---
        pulse = 0.65 + 0.35 * math.sin(pygame.time.get_ticks() / 350.0)

        screen.fill(UI_BG)

        screen.blit(path_surface, (MAP_X, MAP_Y))

        # On the preview, show the player ghost so the spawn point is obvious
        if not playing and not finished:
            pygame.draw.circle(screen, CYAN, to_screen(*start_disp), SCALE)

        pygame.draw.circle(screen, GREEN, to_screen(*start_disp), 8, 2)
        pygame.draw.circle(screen, RED,   to_screen(*end_disp),   finish_radius_disp, 2)

        if playing or finished:
            pygame.draw.circle(screen, CYAN, to_screen(player_pos[0], player_pos[1]), SCALE)

        draw_top_bar(screen, title_font)
        draw_legend(screen, font, small_font)
        draw_right_panel(screen, font, playing, finished, elapsed, optimal_time)
        draw_bottom_bar(screen, font, small_font, playing, finished, pulse)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
