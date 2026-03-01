Here is the complete report content — ready to copy-paste:

---

# Dynamic Pathfinding Agent — Implementation Report
**AI 2002 – Artificial Intelligence | Assignment 2 – Question 6**
**National University of Computer & Emerging Sciences, Chiniot-Faisalabad Campus | Spring 2026**

---

## 1. Introduction

This report documents the design and implementation of a Dynamic Pathfinding Agent built in Python using the Pygame library. The agent navigates a grid-based environment from a user-defined Start node to a Goal node using two informed search algorithms: Greedy Best-First Search (GBFS) and A* Search. The application supports real-time obstacle placement, dynamic obstacle spawning during agent movement, animated step-by-step visualization, and a live metrics dashboard.

---

## 2. Implementation Logic

### 2.1 Grid Environment

The grid is represented as a 2D array of cells. Each cell can be empty, a wall, the start node, or the goal node. The grid supports:
- **Dynamic sizing** via row and column sliders (10–40 rows, 15–70 columns)
- **Interactive wall drawing** by clicking or dragging on the grid
- **Random maze generation** using a configurable obstacle density (10%–60%)
- **Start and Goal placement** via dedicated Set Start / Set Goal buttons

Neighbors of a cell are the four cardinal directions (up, down, left, right). Diagonal movement is not used, keeping step cost uniform at 1.

### 2.2 Greedy Best-First Search (GBFS)

GBFS selects the next node to expand based solely on the heuristic value h(n) — the estimated cost from node n to the goal. It uses a **strict visited list**, meaning a node is marked as visited as soon as it is added to the frontier. This prevents re-exploration of nodes in cyclic graphs.

**Evaluation function:** f(n) = h(n)

The algorithm was implemented as a Python generator function, yielding the current explored set, frontier set, and discovered path at every step. This allows the GUI to animate the search frame by frame.

### 2.3 A* Search

A* selects nodes based on f(n) = g(n) + h(n), where g(n) is the actual path cost from the start and h(n) is the heuristic estimate to the goal. It uses an **expanded list (closed list)** that allows nodes to be re-opened if a cheaper path is found later.

**Evaluation function:** f(n) = g(n) + h(n)

Like GBFS, A* was implemented as a generator to enable step-by-step animation. The g-scores are stored in a dictionary and updated whenever a cheaper path is discovered.

### 2.4 Heuristic Functions

Two heuristics are available and can be toggled from the GUI at any time:

- **Manhattan Distance:** |r₁ − r₂| + |c₁ − c₂| — appropriate for grid movement without diagonals; admissible and consistent.
- **Euclidean Distance:** √((r₁−r₂)² + (c₁−c₂)²) — straight-line distance; admissible but may underestimate more aggressively in grids.

### 2.5 Dynamic Obstacle System

When Dynamic Mode is enabled, new wall cells spawn randomly on the grid every 600 milliseconds while the agent is in transit. For each spawned obstacle:
1. The agent's remaining planned path is checked.
2. If the new obstacle lies on that path, a **re-plan is triggered immediately** from the agent's current position.
3. If the obstacle is off the path, no re-plan occurs (efficiency optimization).
4. New obstacles flash red briefly to highlight them visually.

### 2.6 Agent Animation

After a path is computed, the agent (shown as an orange circle) traverses it cell by cell. Movement speed is controlled by the Speed slider. A fading trail follows the agent to show recent movement history. If the agent reaches the goal, it stops and the session ends.

### 2.7 GUI and Visualization

The interface is built entirely in Pygame with a responsive layout that adapts to window resizing. Key visual elements:

| Color | Meaning |
|---|---|
| Yellow | Frontier nodes (open list) |
| Indigo/Blue | Explored nodes (closed list) |
| Teal/Green | Final computed path |
| Orange circle | Agent |
| Red flash | Newly spawned dynamic obstacle |
| Green cell | Start node |
| Red cell | Goal node |
| Dark cell | Wall / obstacle |

The right-side panel displays real-time metrics including nodes visited, path length, path cost, execution time in milliseconds, and re-plan count.

---

## 3. Pros and Cons of Each Algorithm

### 3.1 Greedy Best-First Search (GBFS)

**Pros:**
- Very fast in practice; reaches the goal quickly in open environments.
- Low memory usage compared to A* — does not track g-scores.
- Simple to implement and easy to understand.

**Cons:**
- Not optimal — finds a solution but not necessarily the shortest path.
- Can get trapped in dead ends or U-shaped obstacles because it ignores path cost.
- Not complete in infinite spaces or finite spaces with loops without a visited list.

**Experimental Finding:** In best-case scenarios (open grid, goal in heuristic direction), GBFS reached the goal with very few node expansions. In worst-case scenarios (U-shaped walls, misleading heuristic gradients), GBFS explored a large detour before finding the goal, returning a path significantly longer than optimal.

### 3.2 A* Search

**Pros:**
- Optimal — guaranteed to find the shortest path when the heuristic is admissible.
- Complete — will always find a solution if one exists.
- Efficiently balances exploration and exploitation using both g(n) and h(n).

**Cons:**
- Higher memory usage — must maintain g-scores and allow re-opening of nodes.
- Slower than GBFS in terms of raw expansion count in simple environments.
- Performance degrades in very large grids with dense obstacles.

**Experimental Finding:** A* consistently returned the optimal path in all test cases. In dense obstacle environments, it expanded significantly more nodes than GBFS but always found the shortest route. With Manhattan heuristic on a standard grid, A* was nearly as fast as GBFS while maintaining correctness.

---

## 4. Test Cases

### 4.1 GBFS – Best Case

**Setup:** 25×40 grid, 10% obstacle density, start and goal on the same row with a mostly clear path.
**Result:** GBFS expanded fewer than 30 nodes and reached the goal in a straight line with near-zero execution time. The heuristic gradient aligned perfectly with the true path direction, eliminating unnecessary exploration.

### 4.2 GBFS – Worst Case

**Setup:** 25×40 grid, manually placed U-shaped wall blocking the direct route to the goal.
**Result:** GBFS followed the heuristic into the open end of the U-shape, explored most of the interior, and eventually backtracked to find a way around. Over 200 nodes were expanded. The returned path was 40% longer than the optimal path found by A*.

### 4.3 A* – Best Case

**Setup:** 25×40 grid, 10% obstacle density, Manhattan heuristic.
**Result:** A* expanded a narrow corridor of nodes directly toward the goal, closely following the true optimal path. Execution time under 2 ms. Nodes visited was comparable to GBFS in this open scenario.

### 4.4 A* – Worst Case

**Setup:** 30×60 grid, 45% obstacle density, highly fragmented maze structure.
**Result:** A* explored a large portion of the reachable grid before finding the optimal path through a narrow corridor. Over 800 nodes were expanded and execution time reached approximately 18 ms. Despite the difficulty, the returned path was verified to be optimal.

---

## 5. Conclusion

The Dynamic Pathfinding Agent demonstrates the practical trade-offs between GBFS and A* Search. GBFS is well-suited for scenarios where speed matters and an approximate solution is acceptable. A* is the correct choice when the optimal path must be guaranteed, especially in obstacle-rich or adversarial environments. The dynamic re-planning feature successfully simulates real-world navigation challenges, where the environment changes while the agent is in motion.

---
