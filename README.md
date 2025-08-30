#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laberinto con Aprendizaje por Refuerzo (Q-Learning) en consola
==============================================================
Este programa cumple con el enunciado del ejercicio de "Aprendizaje por Refuerzo"
implementando un juego sencillo de laberinto con un agente que aprende por Q-Learning.

Incluye:
- Breve explicación (en comentarios) de aprendizaje por refuerzo (RL).
- Laberinto dibujado en consola, con el agente moviéndose paso a paso.
- Acciones: arriba, abajo, izquierda, derecha.
- Sistema de recompensas: +0.1 por moverse a una celda válida, -0.5 al chocar con pared,
  +10.0 al llegar a la meta.
- Agente que aprende mediante Q-Learning con política epsilon-greedy.
- Menú para elegir entre 3 laberintos prediseñados o generar uno aleatorio.
- Demostración final (post-entrenamiento) con animación en consola.

Cómo funciona el Aprendizaje por Refuerzo (resumen):
----------------------------------------------------
En RL un agente interactúa con un entorno tomando acciones. Tras cada acción recibe
una recompensa y observa un nuevo estado. El objetivo es aprender una política
(qué acción tomar en cada estado) para maximizar la recompensa acumulada.
Q-Learning estima el valor Q(s,a) (qué tan buena es una acción a en un estado s)
mediante la actualización:
    Q(s,a) ← Q(s,a) + α [ r + γ * max_a' Q(s', a') - Q(s,a) ]
donde α es la tasa de aprendizaje, γ el factor de descuento, r la recompensa
inmediata y s' el nuevo estado.
"""
import os
import random
import time
import sys
from collections import defaultdict

# ===== Utilidades de consola (limpiar y color) =====
def limpiar():
    os.system('cls' if os.name == 'nt' else 'clear')

# Habilitar colores ANSI en Windows (si es posible)
def _enable_vt_mode_windows():
    if os.name != 'nt':
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        pass

_enable_vt_mode_windows()

# Colores ANSI (con fallback si no se ven)
C_RESET = "\033[0m"
C_WALL  = "\033[92m"   # verde
C_AGENT = "\033[93m"   # amarillo
C_GOAL  = "\033[95m"   # magenta
C_START = "\033[96m"   # cian

# ===== Entorno de laberinto =====
WALL = '#'
PATH = '.'
START = 'S'
GOAL = 'G'

ACTIONS = ['U', 'D', 'L', 'R']
DELTA = {
    'U': (-1, 0),
    'D': ( 1, 0),
    'L': ( 0,-1),
    'R': ( 0, 1),
}

class Maze:
    def __init__(self, grid):
        """
        grid: lista de strings (todas del mismo ancho), con caracteres de {#, ., S, G}
        """
        self.grid = [list(row) for row in grid]
        self.h = len(self.grid)
        self.w = len(self.grid[0])
        self.start = None
        self.goal = None
        for r in range(self.h):
            for c in range(self.w):
                if self.grid[r][c] == START:
                    self.start = (r, c)
                elif self.grid[r][c] == GOAL:
                    self.goal = (r, c)
        if self.start is None or self.goal is None:
            raise ValueError("El laberinto debe contener 'S' (inicio) y 'G' (meta).")

    def inside(self, r, c):
        return 0 <= r < self.h and 0 <= c < self.w

    def is_wall(self, r, c):
        return self.grid[r][c] == WALL

    def state_id(self, pos):
        r, c = pos
        return r * self.w + c

    def id_to_pos(self, sid):
        return (sid // self.w, sid % self.w)

    def render(self, agent_pos=None):
        out_lines = []
        for r in range(self.h):
            line = []
            for c in range(self.w):
                ch = self.grid[r][c]
                if agent_pos is not None and (r, c) == agent_pos:
                    line.append(C_AGENT + 'O' + C_RESET)
                else:
                    if ch == WALL:
                        line.append(C_WALL + '#' + C_RESET)
                    elif ch == PATH:
                        line.append('.')
                    elif ch == START:
                        line.append(C_START + 'S' + C_RESET)
                    elif ch == GOAL:
                        line.append(C_GOAL + 'G' + C_RESET)
            out_lines.append(''.join(line))
        return '\n'.join(out_lines)

    def step(self, pos, action):
        """
        Ejecuta una acción desde la posición pos.
        Recompensas:
            - pared: -0.5 (no se mueve)
            - movimiento válido: +0.1
            - llegar a meta: +10.0 (y done=True)
        Devuelve: (new_pos, reward, done)
        """
        dr, dc = DELTA[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if not self.inside(nr, nc) or self.is_wall(nr, nc):
            return pos, -0.5, False  # choque con pared
        # movimiento válido
        new_pos = (nr, nc)
        if new_pos == self.goal:
            return new_pos, 10.0, True
        else:
            return new_pos, +0.1, False

# ===== Generador de laberintos aleatorios (backtracking) =====
def generar_laberinto_aleatorio(ancho=21, alto=15, seed=None):
    """
    Genera un laberinto rectangular con dimensiones impares usando backtracking.
    'S' en (1,1) y 'G' en (alto-2, ancho-2).
    """
    if seed is not None:
        random.seed(seed)
    # Forzar impares
    if ancho % 2 == 0: ancho += 1
    if alto % 2 == 0: alto += 1

    # Inicializar todo a paredes
    grid = [[WALL for _ in range(ancho)] for _ in range(alto)]

    # Comenzar en celda (1,1)
    def vecinos_celda(r, c):
        # vecinos a 2 celdas de distancia (arriba, abajo, izq, der)
        lst = [(r-2,c), (r+2,c), (r,c-2), (r,c+2)]
        random.shuffle(lst)
        return lst

    # Carvar con DFS
    stack = [(1,1)]
    grid[1][1] = PATH

    while stack:
        r, c = stack[-1]
        avanzó = False
        for nr, nc in vecinos_celda(r, c):
            if 1 <= nr < alto-1 and 1 <= nc < ancho-1 and grid[nr][nc] == WALL:
                # Carvar paso intermedio
                midr, midc = (r+nr)//2, (c+nc)//2
                grid[midr][midc] = PATH
                grid[nr][nc] = PATH
                stack.append((nr, nc))
                avanzó = True
                break
        if not avanzó:
            stack.pop()

    # Colocar inicio y meta
    grid[1][1] = START
    grid[alto-2][ancho-2] = GOAL

    # Convertir filas a strings
    return [''.join(row) for row in grid]

# ===== Laberintos prediseñados =====
MAZE_1 = [
    "#####################",
    "#S....#.......#.....#",
    "###.#.#.#####.#.###.#",
    "#...#.#.....#.#...#.#",
    "#.###.###.#.#.###.#.#",
    "#.#.....#.#.#...#.#.#",
    "#.#.###.#.#.###.#.#.#",
    "#.#.#...#.#...#.#.#.#",
    "#.#.#.#####.#.#.#.#.#",
    "#.#.#.......#.#.#.#.#",
    "#.#.#########.#.#.#.#",
    "#.#...........#.#.#.#",
    "#.#############.#.#.#",
    "#.................#G#",
    "#####################",
]

MAZE_2 = [
    "#####################",
    "#S..#...............#",
    "#.#.#.#############.#",
    "#.#.#.............#.#",
    "#.#.###########.#.#.#",
    "#.#.#.........#.#.#.#",
    "#.#.#.#######.#.#.#.#",
    "#...#.#.....#.#...#.#",
    "###.#.#.###.#.###.#.#",
    "#...#...#...#.....#.#",
    "#.#######.#######.#.#",
    "#.........#.....#.#.#",
    "####.#####.#.###.#.#",
    "#....#.....#...#...#",
    "##################G#",
]

MAZE_3 = [
    "#####################",
    "#S..................#",
    "#####.#############.#",
    "#...#.#...........#.#",
    "#.#.#.#.#########.#.#",
    "#.#...#.#.......#.#.#",
    "#.#####.#.#####.#.#.#",
    "#.....#.#.#...#.#.#.#",
    "#####.#.#.#.#.#.#.#.#",
    "#...#.#.#.#.#.#.#.#.#",
    "#.#.#.#.#.#.#.#.#.#.#",
    "#.#.#...#.#.#...#.#.#",
    "#.#.#####.#.#####.#.#",
    "#.#.......#.......#G#",
    "#####################",
]

PREDEFINED = {
    "1": MAZE_1,
    "2": MAZE_2,
    "3": MAZE_3,
}

# ===== Q-Learning =====
class QLearningAgent:
    def __init__(self, n_states, actions, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.q = defaultdict(float)  # (state, action) -> value
        self.n_states = n_states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def _qkey(self, s, a):
        return (s, a)

    def policy(self, s):
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # elegir acción con mayor Q
        best_a = None
        best_q = -float('inf')
        for a in self.actions:
            q = self.q[self._qkey(s, a)]
            if q > best_q:
                best_q = q
                best_a = a
        # si todas iguales, elige aleatoria
        return best_a if best_a is not None else random.choice(self.actions)

    def greedy(self, s):
        best_a = None
        best_q = -float('inf')
        for a in self.actions:
            q = self.q[self._qkey(s, a)]
            if q > best_q:
                best_q = q
                best_a = a
        return best_a if best_a is not None else random.choice(self.actions)

    def update(self, s, a, r, s_next):
        # max Q(s', a')
        max_q_next = max(self.q[self._qkey(s_next, an)] for an in self.actions)
        curr = self.q[self._qkey(s, a)]
        self.q[self._qkey(s, a)] = curr + self.alpha * (r + self.gamma * max_q_next - curr)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def entrenar_y_demo(maze, episodios=600, max_pasos=1000, mostrar_progreso=True):
    env = Maze(maze)
    n_states = env.h * env.w
    agent = QLearningAgent(n_states, ACTIONS)

    # Entrenamiento (sin renderizado para ser rápido)
    for ep in range(1, episodios+1):
        pos = env.start
        s = env.state_id(pos)
        total_r = 0.0
        for _ in range(max_pasos):
            a = agent.policy(s)
            new_pos, r, done = env.step(pos, a)
            s_next = env.state_id(new_pos)
            agent.update(s, a, r, s_next)
            pos, s = new_pos, s_next
            total_r += r
            if done:
                break
        agent.decay_epsilon()
        if mostrar_progreso and ep % max(1, (episodios//10)) == 0:
            print(f"[Entrenamiento] Episodio {ep}/{episodios} | Recompensa: {total_r:.2f} | epsilon={agent.epsilon:.3f}")
    if mostrar_progreso:
        print("\nEntrenamiento finalizado. Mostrando demostración con la política aprendida...")
        time.sleep(1.4)

    # Demostración post-entrenamiento (greedy) con animación
    pos = env.start
    s = env.state_id(pos)
    pasos = 0
    while True:
        limpiar()
        print(env.render(agent_pos=pos))
        print(f"\nPasos: {pasos}  |  Use Ctrl+C para salir")
        if pos == env.goal:
            print("\n¡Meta alcanzada!")
            break
        a = agent.greedy(s)
        new_pos, r, done = env.step(pos, a)
        pos = new_pos
        s = env.state_id(pos)
        pasos += 1
        if pasos > max_pasos:
            print("\nSe alcanzó el máximo de pasos en demo (posible atasco).")
            break
        time.sleep(0.06)

# ===== Menú principal =====
def elegir_laberinto():
    print("Elige el laberinto:")
    print("  1) Prediseñado 1")
    print("  2) Prediseñado 2")
    print("  3) Prediseñado 3")
    print("  4) Generar aleatorio")
    op = input("Opción [1-4]: ").strip()
    if op in PREDEFINED:
        return PREDEFINED[op]
    elif op == '4':
        try:
            w = int(input("Ancho (impar recomendado, ej. 21): ").strip() or "21")
            h = int(input("Alto  (impar recomendado, ej. 15): ").strip() or "15")
        except ValueError:
            print("Valores inválidos; usando 21x15 por defecto.")
            w, h = 21, 15
        return generar_laberinto_aleatorio(w, h)
    else:
        print("Opción inválida; usando Prediseñado 1.")
        return PREDEFINED["1"]

def main():
    try:
        maze = elegir_laberinto()
        # parámetros de entrenamiento
        try:
            episodios = int(input("Episodios de entrenamiento [600]: ").strip() or "600")
        except ValueError:
            episodios = 600
        entrenar_y_demo(maze, episodios=episodios, max_pasos=2000, mostrar_progreso=True)
    except KeyboardInterrupt:
        print("\nInterrupción por usuario. Saliendo...")

if __name__ == "__main__":
    main()

