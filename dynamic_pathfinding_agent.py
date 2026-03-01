"""
╔══════════════════════════════════════════════════════════════════╗
║   AI 2002 – Artificial Intelligence  |  Spring 2026             ║
║   Assignment 2 – Question 6                                     ║
║   Dynamic Pathfinding Agent                                     ║
║                                                                  ║
║   Algorithms : Greedy Best-First Search (GBFS) · A*             ║
║   Heuristics : Manhattan Distance · Euclidean Distance          ║
║   Features   : Animated search · Dynamic obstacles              ║
║                Interactive map editor · Real-time metrics       ║
║                                                                  ║
║   Install    : pip install pygame                               ║
║   Run        : python dynamic_pathfinder_Q6.py                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pygame
import pygame.gfxdraw
import heapq
import math
import time
import random
import sys
from collections import defaultdict

# ═══════════════════════════════════════════════════════
#  COLOUR PALETTE  (dark-tech theme)
# ═══════════════════════════════════════════════════════
C = {
    # Background / structural
    "bg":           (10,  12,  20),
    "panel_bg":     (15,  18,  30),
    "panel_border": (40,  48,  70),
    "grid_bg":      (20,  24,  38),
    "grid_line":    (30,  36,  55),

    # Grid cell states
    "empty":        (26,  30,  46),
    "wall":         (14,  16,  26),
    "wall_hi":      (35,  40,  60),   # wall hover
    "start":        (16, 200, 120),
    "goal":         (240, 60,  80),
    "frontier":     (230, 180,  30),
    "explored":     (60,  90, 190),
    "path":         (0,  210, 140),
    "agent":        (255, 140,  30),
    "agent_trail":  (255, 100,  30, 80),
    "new_obstacle": (200,  50,  50),

    # Text
    "text_bright":  (220, 225, 240),
    "text_dim":     (100, 112, 140),
    "text_accent":  (0,  210, 180),
    "text_warn":    (240, 160,  40),
    "text_err":     (240,  70,  70),
    "text_ok":      (40,  210, 120),

    # Buttons
    "btn_run":      (20, 160, 110),
    "btn_run_hov":  (25, 190, 130),
    "btn_clear":    (60,  80, 130),
    "btn_clear_hov":(75, 100, 155),
    "btn_danger":   (170,  50,  50),
    "btn_danger_hov":(200, 65, 65),
    "btn_neutral":  (45,  55,  85),
    "btn_hov":      (60,  72, 108),
    "btn_active":   (30, 130, 200),
    "btn_active_hov":(40,155,235),
    "btn_text":     (220, 225, 240),
    "slider_track": (40,  48,  70),
    "slider_thumb": (30, 130, 200),
}

# ═══════════════════════════════════════════════════════
#  LAYOUT CONSTANTS  (responsive to window size)
# ═══════════════════════════════════════════════════════
MIN_W, MIN_H    = 900, 600
DEFAULT_W       = 1280
DEFAULT_H       = 740
PANEL_W         = 285
HEADER_H        = 44

# ═══════════════════════════════════════════════════════
#  HEURISTICS
# ═══════════════════════════════════════════════════════
def h_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def h_euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ═══════════════════════════════════════════════════════
#  GRID MODEL
# ═══════════════════════════════════════════════════════
class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.walls: set = set()

    def resize(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.walls = {w for w in self.walls if 0 <= w[0] < rows and 0 <= w[1] < cols}

    def in_bounds(self, pos):
        return 0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols

    def passable(self, pos):
        return pos not in self.walls

    def neighbors(self, pos):
        r, c = pos
        result = []
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            np_ = (r+dr, c+dc)
            if self.in_bounds(np_) and self.passable(np_):
                result.append(np_)
        return result

    def generate_maze(self, density, protected):
        self.walls.clear()
        for r in range(self.rows):
            for c in range(self.cols):
                p = (r, c)
                if p not in protected and random.random() < density:
                    self.walls.add(p)

# ═══════════════════════════════════════════════════════
#  SEARCH ALGORITHMS  (step generators for animation)
# ═══════════════════════════════════════════════════════
def reconstruct(came_from, node):
    path = []
    cur = node
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


def astar_gen(grid, start, goal, hfn):
    """Yields (explored_set, frontier_set, path_or_None) each step."""
    counter  = 0
    open_heap = []
    heapq.heappush(open_heap, (hfn(start, goal), counter, start))
    came_from  = {start: None}
    g_score    = defaultdict(lambda: math.inf)
    g_score[start] = 0
    closed     = set()
    open_set   = {start}
    explored   = set()

    while open_heap:
        f_cur, _, cur = heapq.heappop(open_heap)
        open_set.discard(cur)

        if cur in closed:
            continue
        closed.add(cur)
        explored.add(cur)

        if cur == goal:
            yield explored.copy(), open_set.copy(), reconstruct(came_from, goal), g_score[goal]
            return

        yield explored.copy(), open_set.copy(), None, None

        for nb in grid.neighbors(cur):
            tg = g_score[cur] + 1
            if tg < g_score[nb]:
                came_from[nb] = cur
                g_score[nb]   = tg
                f_nb = tg + hfn(nb, goal)
                counter += 1
                heapq.heappush(open_heap, (f_nb, counter, nb))
                open_set.add(nb)

    yield explored.copy(), open_set.copy(), None, None


def gbfs_gen(grid, start, goal, hfn):
    """Greedy BFS with strict visited list."""
    counter   = 0
    open_heap = []
    heapq.heappush(open_heap, (hfn(start, goal), counter, start))
    came_from = {start: None}
    visited   = {start}
    open_set  = {start}
    explored  = set()

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        open_set.discard(cur)

        if cur == goal:
            path = reconstruct(came_from, goal)
            yield explored.copy(), open_set.copy(), path, len(path)-1
            return

        explored.add(cur)
        yield explored.copy(), open_set.copy(), None, None

        for nb in grid.neighbors(cur):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = cur
                counter += 1
                heapq.heappush(open_heap, (hfn(nb, goal), counter, nb))
                open_set.add(nb)

    yield explored.copy(), open_set.copy(), None, None


# Instant versions (for agent launch / replanning)
def astar_instant(grid, start, goal, hfn):
    gen = astar_gen(grid, start, goal, hfn)
    explored, frontier, path, cost = None, None, None, None
    for explored, frontier, path, cost in gen:
        if path is not None:
            return path, explored, frontier, cost
    return None, explored or set(), frontier or set(), 0


def gbfs_instant(grid, start, goal, hfn):
    gen = gbfs_gen(grid, start, goal, hfn)
    explored, frontier, path, cost = None, None, None, None
    for explored, frontier, path, cost in gen:
        if path is not None:
            return path, explored, frontier, cost
    return None, explored or set(), frontier or set(), 0


# ═══════════════════════════════════════════════════════
#  UI WIDGETS
# ═══════════════════════════════════════════════════════
class Button:
    def __init__(self, text, color, hover_color, w=None, h=32):
        self.text        = text
        self.color       = color
        self.hover_color = hover_color
        self.w           = w
        self.h           = h
        self.rect        = pygame.Rect(0, 0, w or 100, h)
        self.hovered     = False
        self.active      = False
        self.enabled     = True

    def place(self, x, y, w=None):
        self.rect = pygame.Rect(x, y, w or self.w or 120, self.h)

    def draw(self, surf, font):
        base = self.color if not self.active else C["btn_active"]
        col  = (self.hover_color if self.hovered else base) if self.enabled else C["grid_line"]
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        # subtle top-highlight
        hi = pygame.Rect(self.rect.x+1, self.rect.y+1, self.rect.w-2, 2)
        pygame.draw.rect(surf, (*[min(255,c+40) for c in col[:3]],), hi, border_radius=6)
        txt_col = C["btn_text"] if self.enabled else C["text_dim"]
        t = font.render(self.text, True, txt_col)
        surf.blit(t, t.get_rect(center=self.rect.center))

    def handle(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos) and self.enabled
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) and self.enabled:
                return True
        return False


class Toggle(Button):
    """Two-state toggle that shows as active when on."""
    def __init__(self, text_off, text_on, **kw):
        super().__init__(text_off, **kw)
        self.text_off = text_off
        self.text_on  = text_on
        self.state    = False

    def draw(self, surf, font):
        self.text   = self.text_on  if self.state else self.text_off
        self.active = self.state
        super().draw(surf, font)


class Slider:
    def __init__(self, label, mn, mx, val, fmt="{:.0f}"):
        self.label   = label
        self.mn, self.mx = mn, mx
        self.val     = val
        self.fmt     = fmt
        self.rect    = pygame.Rect(0,0,200,16)
        self.dragging= False

    def place(self, x, y, w=200):
        self.rect = pygame.Rect(x, y, w, 16)

    @property
    def norm(self):
        return (self.val - self.mn) / (self.mx - self.mn)

    def draw(self, surf, font_sm):
        # Track
        tr = pygame.Rect(self.rect.x, self.rect.y+5, self.rect.w, 6)
        pygame.draw.rect(surf, C["slider_track"], tr, border_radius=3)
        # Fill
        fw = int(tr.w * self.norm)
        if fw > 0:
            pygame.draw.rect(surf, C["btn_active"], (tr.x, tr.y, fw, 6), border_radius=3)
        # Thumb
        tx = self.rect.x + int(self.rect.w * self.norm)
        pygame.draw.circle(surf, C["slider_thumb"], (tx, self.rect.y+8), 8)
        pygame.draw.circle(surf, C["text_bright"],  (tx, self.rect.y+8), 8, 2)
        # Label + value
        lbl = font_sm.render(f"{self.label}: {self.fmt.format(self.val)}", True, C["text_dim"])
        surf.blit(lbl, (self.rect.x, self.rect.y - 16))

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.inflate(0,16).collidepoint(event.pos):
                self.dragging = True
                self._update(event.pos[0])
                return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            self._update(event.pos[0])
            return True
        return False

    def _update(self, mx):
        t = max(0.0, min(1.0, (mx - self.rect.x) / self.rect.w))
        raw = self.mn + t * (self.mx - self.mn)
        self.val = max(self.mn, min(self.mx, raw))


# ═══════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════
class App:
    # ── Init ──────────────────────────────────────────
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Dynamic Pathfinding Agent  ·  AI 2002")
        self.screen = pygame.display.set_mode((DEFAULT_W, DEFAULT_H), pygame.RESIZABLE)
        self.W, self.H = DEFAULT_W, DEFAULT_H

        # Fonts
        self.f14 = pygame.font.SysFont("Consolas,Courier New,monospace", 14)
        self.f13 = pygame.font.SysFont("Consolas,Courier New,monospace", 13)
        self.f12 = pygame.font.SysFont("Consolas,Courier New,monospace", 12)
        self.f16 = pygame.font.SysFont("Consolas,Courier New,monospace", 16, bold=True)
        self.f11 = pygame.font.SysFont("Consolas,Courier New,monospace", 11)

        self.clock  = pygame.time.Clock()
        self.grid   = Grid(25, 40)
        self.cell_s = self._calc_cell()

        # Agent / search state
        self.start        = (self.grid.rows//2, 2)
        self.goal         = (self.grid.rows//2, self.grid.cols-3)
        self.explored     = set()
        self.frontier_set = set()
        self.path         = []
        self.path_cost    = 0
        self.search_gen   = None
        self.searching    = False
        self.search_done  = False
        self.start_time   = 0.0
        self.exec_ms      = 0.0
        self.nodes_vis    = 0

        self.agent_pos    = None
        self.agent_path   = []
        self.agent_idx    = 0
        self.agent_live   = False
        self.agent_timer  = 0
        self.trail        = []
        self.replan_count = 0
        self.dyn_timer    = 0
        self.flash_cells  = {}   # pos -> frames_remaining (red flash for new obstacles)

        # Interaction
        self.placing_start  = False
        self.placing_goal   = False
        self.drawing_wall   = False
        self.erasing_wall   = False
        self.last_wall_cell = None

        # Settings
        self.algorithm  = "A*"       # "A*" | "GBFS"
        self.heuristic  = "Manhattan"# "Manhattan" | "Euclidean"
        self.dyn_mode   = False
        self.show_legend= True

        self._build_ui()
        self._apply_grid_size()
        self.grid.generate_maze(0.28, {self.start, self.goal})

    # ── Responsive layout ─────────────────────────────
    def _calc_cell(self):
        gw = self.W - PANEL_W
        gh = self.H - HEADER_H
        return max(12, min(gw // self.grid.cols, gh // self.grid.rows))

    def _apply_grid_size(self):
        rows = int(self.sl_rows.val)
        cols = int(self.sl_cols.val)
        old_rows, old_cols = self.grid.rows, self.grid.cols
        self.grid.resize(rows, cols)
        # Clamp start / goal
        self.start = (min(self.start[0], rows-1), min(self.start[1], cols-1))
        self.goal  = (min(self.goal[0],  rows-1), min(self.goal[1],  cols-1))
        self.cell_s = self._calc_cell()
        self._reset_search()

    def _on_resize(self, w, h):
        self.W = max(MIN_W, w)
        self.H = max(MIN_H, h)
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        self.cell_s = self._calc_cell()
        self._layout_ui()

    # ── UI Build ──────────────────────────────────────
    def _build_ui(self):
        # Buttons
        self.btn_run    = Button("▶  Run Search",    C["btn_run"],     C["btn_run_hov"])
        self.btn_stop   = Button("■  Stop",          C["btn_danger"],  C["btn_danger_hov"])
        self.btn_agent  = Button("⬡  Launch Agent",  C["btn_run"],     C["btn_run_hov"])
        self.btn_clear  = Button("✕  Clear Walls",   C["btn_clear"],   C["btn_hov"])
        self.btn_reset  = Button("↺  Reset All",     C["btn_neutral"], C["btn_hov"])
        self.btn_maze   = Button("⬡  Random Maze",   C["btn_neutral"], C["btn_hov"])

        self.btn_astar  = Button("A*",   C["btn_neutral"], C["btn_hov"])
        self.btn_gbfs   = Button("GBFS", C["btn_neutral"], C["btn_hov"])
        self.btn_man    = Button("Manhattan", C["btn_neutral"], C["btn_hov"])
        self.btn_euc    = Button("Euclidean", C["btn_neutral"], C["btn_hov"])

        self.btn_ps     = Button("Set Start", C["btn_neutral"], C["btn_hov"])
        self.btn_pg     = Button("Set Goal",  C["btn_neutral"], C["btn_hov"])
        self.tog_dyn    = Toggle("Dynamic: OFF", "Dynamic: ON",
                                 color=C["btn_neutral"], hover_color=C["btn_hov"])

        # Sliders
        self.sl_rows    = Slider("Rows", 10, 40, self.grid.rows)
        self.sl_cols    = Slider("Cols", 15, 70, self.grid.cols)
        self.sl_speed   = Slider("Speed", 1, 20, 8)
        self.sl_density = Slider("Density %", 10, 60, 28, fmt="{:.0f}")

        self._layout_ui()

    def _layout_ui(self):
        """Place all widgets based on current window size."""
        px = self.W - PANEL_W + 10
        pw = PANEL_W - 20
        bw2 = (pw - 6) // 2   # half-width button
        y = HEADER_H + 8

        # ── Section: search control ──
        self.btn_run.place(px, y, pw);    y += 38
        self.btn_stop.place(px, y, pw);   y += 38
        self.btn_agent.place(px, y, pw);  y += 44

        # ── Section: algorithm ──
        y += 4
        self.btn_astar.place(px,       y, bw2)
        self.btn_gbfs.place(px+bw2+6,  y, bw2)
        y += 36

        # ── Section: heuristic ──
        self.btn_man.place(px,       y, bw2)
        self.btn_euc.place(px+bw2+6, y, bw2)
        y += 44

        # ── Section: map tools ──
        self.btn_ps.place(px,       y, bw2)
        self.btn_pg.place(px+bw2+6, y, bw2)
        y += 38
        self.tog_dyn.place(px, y, pw); y += 38
        self.btn_maze.place(px, y, pw); y += 38
        self.btn_clear.place(px,       y, bw2)
        self.btn_reset.place(px+bw2+6, y, bw2)
        y += 46

        # ── Sliders ──
        self.sl_rows.place(px, y+16, pw);  y += 48
        self.sl_cols.place(px, y+16, pw);  y += 48
        self.sl_speed.place(px, y+16, pw); y += 48
        self.sl_density.place(px, y+16, pw); y += 48

        self._ui_y_after_sliders = y

    # ── Reset helpers ──────────────────────────────────
    def _reset_search(self):
        self.explored     = set()
        self.frontier_set = set()
        self.path         = []
        self.path_cost    = 0
        self.search_gen   = None
        self.searching    = False
        self.search_done  = False
        self.exec_ms      = 0.0
        self.nodes_vis    = 0
        self._stop_agent()

    def _stop_agent(self):
        self.agent_live  = False
        self.agent_pos   = None
        self.agent_path  = []
        self.agent_idx   = 0
        self.agent_timer = 0
        self.trail       = []
        self.replan_count= 0
        self.dyn_timer   = 0

    # ── Heuristic selector ────────────────────────────
    def _hfn(self):
        return h_manhattan if self.heuristic == "Manhattan" else h_euclidean

    # ── Search control ────────────────────────────────
    def _run_search(self):
        self._reset_search()
        hfn = self._hfn()
        self.start_time = time.time()
        if self.algorithm == "A*":
            self.search_gen = astar_gen(self.grid, self.start, self.goal, hfn)
        else:
            self.search_gen = gbfs_gen(self.grid, self.start, self.goal, hfn)
        self.searching = True

    def _step_search(self):
        if not self.searching or self.search_gen is None:
            return
        steps = max(1, int(self.sl_speed.val))
        for _ in range(steps):
            try:
                exp, fro, path, cost = next(self.search_gen)
                self.explored     = exp
                self.frontier_set = fro
                self.nodes_vis    = len(exp)
                if path is not None:
                    self.path       = path
                    self.path_cost  = cost if cost is not None else len(path)-1
                    self.exec_ms    = (time.time() - self.start_time) * 1000
                    self.searching  = False
                    self.search_done= True
                    return
            except StopIteration:
                self.exec_ms   = (time.time() - self.start_time) * 1000
                self.searching = False
                self.search_done = True
                return

    def _launch_agent(self):
        hfn = self._hfn()
        t0  = time.time()
        if self.algorithm == "A*":
            path, exp, fro, cost = astar_instant(self.grid, self.start, self.goal, hfn)
        else:
            path, exp, fro, cost = gbfs_instant(self.grid, self.start, self.goal, hfn)
        self.exec_ms      = (time.time() - t0) * 1000
        self.explored     = exp
        self.frontier_set = fro
        self.nodes_vis    = len(exp)
        self.search_done  = True
        self._stop_agent()
        if path:
            self.path       = path
            self.path_cost  = cost if cost else len(path)-1
            self.agent_path = path
            self.agent_idx  = 0
            self.agent_pos  = path[0]
            self.agent_live = True
            self.trail      = [path[0]]
        else:
            self.path = []

    def _replan_agent(self):
        if not self.agent_live:
            return
        cur = self.agent_path[self.agent_idx] if self.agent_idx < len(self.agent_path) else self.agent_pos
        hfn = self._hfn()
        if self.algorithm == "A*":
            path, exp, fro, cost = astar_instant(self.grid, cur, self.goal, hfn)
        else:
            path, exp, fro, cost = gbfs_instant(self.grid, cur, self.goal, hfn)
        self.explored     |= exp
        self.frontier_set  = fro
        self.nodes_vis     = len(self.explored)
        if path:
            self.agent_path = path
            self.agent_idx  = 0
            self.path       = path
            self.path_cost  = cost if cost else len(path)-1
            self.replan_count += 1

    def _step_agent(self, dt_ms):
        if not self.agent_live:
            return
        speed_cells_per_sec = max(1, int(self.sl_speed.val)) * 2
        self.agent_timer += dt_ms
        ms_per_step = 1000 / speed_cells_per_sec
        if self.agent_timer >= ms_per_step:
            self.agent_timer = 0
            nxt = self.agent_idx + 1
            if nxt >= len(self.agent_path):
                self.agent_live = False
                return
            nxt_pos = self.agent_path[nxt]
            if nxt_pos in self.grid.walls:
                self._replan_agent()
                return
            self.agent_idx = nxt
            self.agent_pos = nxt_pos
            self.trail.append(nxt_pos)
            if len(self.trail) > 120:
                self.trail.pop(0)
            if self.agent_pos == self.goal:
                self.agent_live = False

    def _spawn_obstacle(self):
        for _ in range(20):
            r = random.randint(0, self.grid.rows-1)
            c = random.randint(0, self.grid.cols-1)
            pos = (r, c)
            if pos != self.start and pos != self.goal and pos != self.agent_pos and pos not in self.grid.walls:
                self.grid.walls.add(pos)
                self.flash_cells[pos] = 20
                if self.agent_live:
                    remaining = self.agent_path[self.agent_idx:]
                    if pos in remaining:
                        self._replan_agent()
                return

    # ── Cell coordinate helpers ───────────────────────
    def _grid_origin(self):
        gw  = self.W - PANEL_W
        gh  = self.H - HEADER_H
        ox  = (gw - self.grid.cols * self.cell_s) // 2
        oy  = HEADER_H + (gh - self.grid.rows * self.cell_s) // 2
        return ox, oy

    def _px_to_cell(self, mx, my):
        ox, oy = self._grid_origin()
        c = (mx - ox) // self.cell_s
        r = (my - oy) // self.cell_s
        pos = (r, c)
        if self.grid.in_bounds(pos):
            return pos
        return None

    def _cell_rect(self, r, c, ox, oy, margin=1):
        x = ox + c * self.cell_s + margin
        y = oy + r * self.cell_s + margin
        s = self.cell_s - margin * 2
        return pygame.Rect(x, y, s, s)

    # ── Drawing ───────────────────────────────────────
    def _draw_header(self):
        hdr = pygame.Rect(0, 0, self.W, HEADER_H)
        pygame.draw.rect(self.screen, C["panel_bg"], hdr)
        pygame.draw.line(self.screen, C["panel_border"], (0, HEADER_H), (self.W, HEADER_H), 1)
        title = self.f16.render("⬡  Dynamic Pathfinding Agent", True, C["text_accent"])
        self.screen.blit(title, (14, (HEADER_H - title.get_height()) // 2))
        # algorithm badge
        alg_str  = f" {self.algorithm} · {self.heuristic} "
        alg_surf = self.f13.render(alg_str, True, C["text_bright"])
        alg_rect = alg_surf.get_rect(midright=(self.W - PANEL_W - 12, HEADER_H//2))
        badge = alg_rect.inflate(10, 6)
        pygame.draw.rect(self.screen, C["btn_active"], badge, border_radius=4)
        self.screen.blit(alg_surf, alg_rect)
        # keyboard hints
        hints = self.f11.render("R=run  C=clear  M=maze  G=launch  ESC=quit", True, C["text_dim"])
        self.screen.blit(hints, (title.get_width() + 28, (HEADER_H - hints.get_height())//2))

    def _draw_grid(self):
        gw  = self.W - PANEL_W
        gh  = self.H - HEADER_H
        grid_surf_rect = pygame.Rect(0, HEADER_H, gw, gh)
        pygame.draw.rect(self.screen, C["grid_bg"], grid_surf_rect)

        ox, oy = self._grid_origin()
        cs = self.cell_s
        pad = 2 if cs > 14 else 1

        # Grid lines (light)
        for r in range(self.grid.rows + 1):
            pygame.draw.line(self.screen, C["grid_line"],
                             (ox, oy + r*cs), (ox + self.grid.cols*cs, oy + r*cs))
        for c in range(self.grid.cols + 1):
            pygame.draw.line(self.screen, C["grid_line"],
                             (ox + c*cs, oy), (ox + c*cs, oy + self.grid.rows*cs))

        # Agent trail (fading)
        for i, tp in enumerate(self.trail):
            tr_, tc_ = tp
            alpha_factor = i / max(1, len(self.trail))
            col = tuple(int(v * alpha_factor * 0.5) for v in (255, 140, 30))
            rect = self._cell_rect(tr_, tc_, ox, oy, pad)
            pygame.draw.rect(self.screen, col, rect, border_radius=3)

        # Cells
        path_set = set(self.path)
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                pos = (r, c)
                rect = self._cell_rect(r, c, ox, oy, pad)

                if pos in self.flash_cells:
                    intensity = self.flash_cells[pos] / 20
                    col = tuple(int(a + (b-a)*intensity) for a,b in zip(C["wall"], C["new_obstacle"]))
                elif pos in self.grid.walls:
                    col = C["wall"]
                elif pos == self.start:
                    col = C["start"]
                elif pos == self.goal:
                    col = C["goal"]
                elif pos in path_set and pos not in (self.start, self.goal):
                    col = C["path"]
                elif pos in self.explored:
                    col = C["explored"]
                elif pos in self.frontier_set:
                    col = C["frontier"]
                else:
                    continue  # empty cell already shows grid_bg

                br = max(1, cs // 6)
                pygame.draw.rect(self.screen, col, rect, border_radius=br)

                # ─ Labels for start / goal ─
                if pos == self.start and cs >= 18:
                    lbl = self.f12.render("S", True, C["grid_bg"])
                    self.screen.blit(lbl, lbl.get_rect(center=rect.center))
                elif pos == self.goal and cs >= 18:
                    lbl = self.f12.render("G", True, C["grid_bg"])
                    self.screen.blit(lbl, lbl.get_rect(center=rect.center))

        # Agent circle
        if self.agent_live and self.agent_pos:
            r, c = self.agent_pos
            rect = self._cell_rect(r, c, ox, oy, pad)
            cx, cy = rect.centerx, rect.centery
            rad = max(4, cs//2 - 2)
            pygame.draw.circle(self.screen, C["agent"],     (cx, cy), rad)
            pygame.draw.circle(self.screen, C["text_bright"],(cx, cy), rad, 2)
            # direction arrow approximation: dot
            pygame.draw.circle(self.screen, C["grid_bg"],   (cx, cy), max(2, rad//3))

        # Hover highlight
        mx, my = pygame.mouse.get_pos()
        if mx < self.W - PANEL_W:
            hc = self._px_to_cell(mx, my)
            if hc and hc not in (self.start, self.goal):
                hr, hcc = hc
                hrt = self._cell_rect(hr, hcc, ox, oy, pad)
                ov = pygame.Surface((hrt.w, hrt.h), pygame.SRCALPHA)
                ov.fill((255, 255, 255, 25))
                self.screen.blit(ov, hrt.topleft)

    def _draw_panel(self):
        px = self.W - PANEL_W
        panel_rect = pygame.Rect(px, 0, PANEL_W, self.H)
        pygame.draw.rect(self.screen, C["panel_bg"], panel_rect)
        pygame.draw.line(self.screen, C["panel_border"], (px, 0), (px, self.H), 2)

        # Section dividers
        def section(text, y):
            pygame.draw.line(self.screen, C["panel_border"], (px+6, y+7), (px+PANEL_W-6, y+7))
            lbl = self.f11.render(f" {text} ", True, C["text_dim"])
            lb  = lbl.get_rect(center=(px + PANEL_W//2, y+7))
            pygame.draw.rect(self.screen, C["panel_bg"], lb.inflate(4,2))
            self.screen.blit(lbl, lb)

        y0 = HEADER_H + 8
        section("SEARCH CONTROL", y0 - 6)
        y0 += 6

        self.btn_run.draw(self.screen, self.f13)
        self.btn_stop.draw(self.screen, self.f13)
        self.btn_agent.draw(self.screen, self.f13)

        section("ALGORITHM", self.btn_astar.rect.y - 12)
        self.btn_astar.active = (self.algorithm == "A*")
        self.btn_gbfs.active  = (self.algorithm == "GBFS")
        self.btn_astar.draw(self.screen, self.f13)
        self.btn_gbfs.draw(self.screen, self.f13)

        section("HEURISTIC", self.btn_man.rect.y - 12)
        self.btn_man.active = (self.heuristic == "Manhattan")
        self.btn_euc.active = (self.heuristic == "Euclidean")
        self.btn_man.draw(self.screen, self.f13)
        self.btn_euc.draw(self.screen, self.f13)

        section("MAP TOOLS", self.btn_ps.rect.y - 14)
        self.btn_ps.active  = self.placing_start
        self.btn_pg.active  = self.placing_goal
        self.btn_ps.draw(self.screen, self.f13)
        self.btn_pg.draw(self.screen, self.f13)
        self.tog_dyn.draw(self.screen, self.f13)
        self.btn_maze.draw(self.screen, self.f13)
        self.btn_clear.draw(self.screen, self.f13)
        self.btn_reset.draw(self.screen, self.f13)

        section("GRID SIZE & SPEED", self.sl_rows.rect.y - 30)
        self.sl_rows.draw(self.screen, self.f11)
        self.sl_cols.draw(self.screen, self.f11)
        self.sl_speed.draw(self.screen, self.f11)
        self.sl_density.draw(self.screen, self.f11)

        # ── Metrics ──────────────────────────────────
        my = self._ui_y_after_sliders + 6
        section("METRICS", my - 4)
        my += 10

        def metric(label, val, color=C["text_accent"], y=None):
            nonlocal my
            use_y = y if y is not None else my
            lbl = self.f12.render(label, True, C["text_dim"])
            vt  = self.f12.render(str(val), True, color)
            self.screen.blit(lbl, (px+10, use_y))
            self.screen.blit(vt,  (px+PANEL_W-10-vt.get_width(), use_y))
            my = use_y + 18

        status_text, status_col = self._status()
        metric("Status", status_text, status_col)
        metric("Nodes Visited", f"{self.nodes_vis:,}")
        metric("Path Length",   f"{len(self.path)-1 if self.path else '—'}")
        metric("Path Cost",     f"{self.path_cost:.1f}" if self.path_cost else "—")
        metric("Exec Time",     f"{self.exec_ms:.1f} ms")
        metric("Re-plans",      f"{self.replan_count}", C["text_warn"])
        metric("Grid Size",     f"{self.grid.rows}×{self.grid.cols}", C["text_dim"])
        metric("Walls",         f"{len(self.grid.walls)}", C["text_dim"])

        # ── Legend ───────────────────────────────────
        if my + 140 < self.H:
            section("LEGEND", my + 2)
            my += 16
            legend = [
                (C["start"],      "Start Node (S)"),
                (C["goal"],       "Goal Node (G)"),
                (C["wall"],       "Wall / Obstacle"),
                (C["frontier"],   "Frontier (Open)"),
                (C["explored"],   "Explored (Closed)"),
                (C["path"],       "Optimal Path"),
                (C["agent"],      "Agent"),
                (C["new_obstacle"],"New Obstacle"),
            ]
            for col, name in legend:
                if my + 16 > self.H - 4:
                    break
                pygame.draw.rect(self.screen, col,
                                 (px+10, my+2, 13, 11), border_radius=2)
                t = self.f11.render(name, True, C["text_dim"])
                self.screen.blit(t, (px+28, my+1))
                my += 16

        # ── Mode indicator top-right of panel ──────
        if self.dyn_mode:
            dyn_txt = self.f12.render("● DYN", True, C["text_warn"])
            self.screen.blit(dyn_txt, (px + PANEL_W - dyn_txt.get_width() - 8, 14))

    def _status(self):
        if self.agent_live:
            return "Agent moving…", C["text_warn"]
        if self.searching:
            return "Searching…", C["text_warn"]
        if self.search_done:
            if self.path:
                return "Path found ✓", C["text_ok"]
            else:
                return "No path found ✗", C["text_err"]
        return "Ready", C["text_dim"]

    # ── Event handling ────────────────────────────────
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE:
                self._on_resize(event.w, event.h)
                continue

            # Keyboard shortcuts
            if event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_ESCAPE:
                    return False
                elif k == pygame.K_r:
                    self._run_search()
                elif k == pygame.K_c:
                    self.grid.walls.clear(); self._reset_search()
                elif k == pygame.K_m:
                    self._do_maze()
                elif k == pygame.K_g:
                    self._launch_agent()
                elif k == pygame.K_SPACE:
                    if self.searching:
                        self.searching = False
                    else:
                        self._run_search()
                elif k == pygame.K_d:
                    self.tog_dyn.state = not self.tog_dyn.state
                    self.dyn_mode = self.tog_dyn.state

            # Sliders (before button check so drag doesn't activate buttons)
            for sl in (self.sl_rows, self.sl_cols, self.sl_speed, self.sl_density):
                if sl.handle(event):
                    break

            # Apply grid size when slider released
            if event.type == pygame.MOUSEBUTTONUP:
                new_r = int(self.sl_rows.val)
                new_c = int(self.sl_cols.val)
                if new_r != self.grid.rows or new_c != self.grid.cols:
                    self._apply_grid_size()

            # Buttons
            if self.btn_run.handle(event):
                self._run_search()
            if self.btn_stop.handle(event):
                self.searching  = False
                self.agent_live = False
            if self.btn_agent.handle(event):
                self._launch_agent()
            if self.btn_clear.handle(event):
                self.grid.walls.clear(); self._reset_search()
            if self.btn_reset.handle(event):
                self.grid.walls.clear()
                self.start = (self.grid.rows//2, 2)
                self.goal  = (self.grid.rows//2, self.grid.cols-3)
                self._reset_search()
                self.flash_cells.clear()
            if self.btn_maze.handle(event):
                self._do_maze()
            if self.btn_astar.handle(event):
                self.algorithm = "A*";    self._reset_search()
            if self.btn_gbfs.handle(event):
                self.algorithm = "GBFS";  self._reset_search()
            if self.btn_man.handle(event):
                self.heuristic = "Manhattan"; self._reset_search()
            if self.btn_euc.handle(event):
                self.heuristic = "Euclidean"; self._reset_search()
            if self.btn_ps.handle(event):
                self.placing_start = not self.placing_start
                self.placing_goal  = False
            if self.btn_pg.handle(event):
                self.placing_goal  = not self.placing_goal
                self.placing_start = False
            if self.tog_dyn.handle(event):
                self.tog_dyn.state = not self.tog_dyn.state
                self.dyn_mode      = self.tog_dyn.state

            # Grid mouse interaction
            mx, my = pygame.mouse.get_pos()
            in_grid = mx < self.W - PANEL_W and my > HEADER_H

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and in_grid:
                cell = self._px_to_cell(mx, my)
                if cell:
                    if self.placing_start:
                        if cell != self.goal:
                            self.start = cell
                            self.placing_start = False
                            self._reset_search()
                    elif self.placing_goal:
                        if cell != self.start:
                            self.goal = cell
                            self.placing_goal = False
                            self._reset_search()
                    elif cell not in (self.start, self.goal):
                        if cell in self.grid.walls:
                            self.erasing_wall = True
                            self.drawing_wall = False
                            self.grid.walls.discard(cell)
                        else:
                            self.drawing_wall = True
                            self.erasing_wall = False
                            self.grid.walls.add(cell)
                        self.last_wall_cell = cell
                        self._reset_search()

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.drawing_wall = False
                self.erasing_wall = False
                self.last_wall_cell = None

            if event.type == pygame.MOUSEMOTION and in_grid:
                if self.drawing_wall or self.erasing_wall:
                    cell = self._px_to_cell(mx, my)
                    if cell and cell not in (self.start, self.goal) and cell != self.last_wall_cell:
                        self.last_wall_cell = cell
                        if self.drawing_wall:
                            self.grid.walls.add(cell)
                        else:
                            self.grid.walls.discard(cell)
                        # Invalidate search overlay but keep searching
                        if self.search_done and not self.agent_live:
                            self._reset_search()

        return True

    def _do_maze(self):
        self._reset_search()
        self.flash_cells.clear()
        density = self.sl_density.val / 100
        self.grid.generate_maze(density, {self.start, self.goal})

    # ── Main Loop ─────────────────────────────────────
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60)

            running = self._handle_events()

            # Animate search
            if self.searching:
                self._step_search()

            # Animate agent
            if self.agent_live:
                self._step_agent(dt)
                # Dynamic obstacles
                if self.dyn_mode and self.agent_live:
                    self.dyn_timer += dt
                    if self.dyn_timer >= 600:  # every 600 ms
                        self.dyn_timer = 0
                        if random.random() < 0.65:
                            self._spawn_obstacle()

            # Flash timer
            expired = [p for p, f in self.flash_cells.items() if f <= 0]
            for p in expired:
                del self.flash_cells[p]
            for p in list(self.flash_cells):
                self.flash_cells[p] -= 1

            # Draw
            self.screen.fill(C["bg"])
            self._draw_grid()
            self._draw_panel()
            self._draw_header()
            pygame.display.flip()

        pygame.quit()
        sys.exit()


# ═══════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    App().run()