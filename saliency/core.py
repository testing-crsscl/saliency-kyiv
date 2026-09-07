"""
CATALINA / GWT engine. Port of src/sim/*.ts — same arithmetic, same mulberry32.
Nothing here is a second model.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

Rng = Callable[[], float]

DEFAULT_PARAMS = {
    "idleThreshold": 0.3,
    "acousticBase": 0.93,
    "habituationTau": 230,
    "calendarTauDays": 201,
}

APPRAISAL_WEIGHTS = {
    "scepticismInv": 0.35,
    "confArmy": 0.25,
    "security": 0.15,
    "confGov": 0.1,
    "neighInsecurity": 0.1,
    "fight": 0.05,
}

SHELTER_IN_PLACE_METRES = 18
EVENT_T0 = -10
EVENT_T1 = 30
EVENT_MINUTES = EVENT_T1 - EVENT_T0 + 1
MINUTES = list(range(EVENT_T0, EVENT_T1 + 1))
ACTIONS = ("continue", "shelter_in_place", "move", "evacuate")

ASLEEP_BY_HOUR = [
    0.88, 0.9, 0.9, 0.88, 0.82, 0.7, 0.42, 0.18, 0.08, 0.05, 0.04, 0.04,
    0.05, 0.05, 0.04, 0.04, 0.05, 0.06, 0.07, 0.1, 0.16, 0.28, 0.48, 0.72,
]

PERIODS = [
    {"id": "2022-03/04", "label": "Mar–Apr", "startN": 0, "endN": 174, "typicalN": 40, "hour": 14, "wd": 2},
    {"id": "2022-05/06", "label": "May–Jun", "startN": 175, "endN": 305, "typicalN": 220, "hour": 14, "wd": 2},
    {"id": "2022-07/08/09", "label": "Jul–Sep", "startN": 306, "endN": 406, "typicalN": 360, "hour": 14, "wd": 2},
]


def _i32(x: int) -> int:
    x = x & 0xFFFFFFFF
    return x - 0x100000000 if x >= 0x80000000 else x


def _u32(x: int) -> int:
    return x & 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return _i32(_i32(a) * _i32(b))


def mulberry32(seed: int) -> Rng:
    """JS mulberry32. Unsigned >>> shifts, signed |0 arithmetic, Math.imul."""
    a = [_u32(int(seed))]

    def usr(x: int, n: int) -> int:
        return _u32(x) >> n

    def rng() -> float:
        a[0] = _i32(a[0])
        a[0] = _i32(a[0] + 0x6D2B79F5)
        t = _imul(a[0] ^ usr(a[0], 15), 1 | a[0])
        t = _i32((t + _imul(t ^ usr(t, 7), 61 | t)) ^ t)
        return usr(t ^ usr(t, 14), 0) / 4294967296.0

    return rng


def js_round(x: float) -> int:
    """JS Math.round for non-negative x: half-up toward +inf."""
    return int(math.floor(x + 0.5))


def clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def gauss(rng: Rng) -> float:
    u = max(1e-12, rng())
    v = rng()
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


def lognormal(rng: Rng, mu: float, sigma: float) -> float:
    return math.exp(mu + sigma * gauss(rng))


def pick_weighted(pool: List[dict], rng: Rng) -> dict:
    tot = sum(p.get("w", 1) or 0 for p in pool)
    u = rng() * tot
    for p in pool:
        u -= p.get("w", 1) or 0
        if u <= 0:
            return p
    return pool[-1]


def num(v, default=0.5) -> float:
    if v is None:
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(x):
        return default
    return x


def focused_attention_threshold(saliency_threshold: float) -> float:
    return saliency_threshold + (1 - saliency_threshold) / 2


def acoustic_saliency(exposure: float, p: dict = DEFAULT_PARAMS) -> float:
    tau = max(1, p["habituationTau"])
    return min(0.999, max(0.0, p["acousticBase"] * math.exp(-exposure / tau)))


def appraise_threat(agent: dict) -> float:
    W = APPRAISAL_WEIGHTS
    threat = (
        W["scepticismInv"] * (1 - agent["scepticism"])
        + W["confArmy"] * agent["confArmy"]
        + W["security"] * agent["security"]
        + W["confGov"] * agent["confGov"]
        + W["neighInsecurity"] * (1 - agent["neighSecure"])
        + W["fight"] * agent["fight"]
    )
    return min(1.0, max(0.0, threat))


def stay_safe_saliency(agent: dict, p: dict = DEFAULT_PARAMS) -> float:
    acoustic = acoustic_saliency(agent["exposure"], p)
    return min(0.999, acoustic * (0.4 + 0.6 * appraise_threat(agent)))


def siren_saliency(agent: dict, p: dict) -> float:
    return acoustic_saliency(agent["exposure"], p)


def draw_routine(hour: int, weekday: int, employed: bool, rng: Rng) -> str:
    h = ((hour % 24) + 24) % 24
    asleep = ASLEEP_BY_HOUR[h] if h < len(ASLEEP_BY_HOUR) else 0.1
    if rng() < asleep:
        return "asleep"
    workblock = employed and weekday < 5 and 8 <= h <= 18
    p_commit = 0.62 if workblock else (0.28 if employed else 0.18)
    if h >= 22 or h <= 6:
        p_commit *= 0.35
    if rng() < p_commit:
        return "committed"
    return "idle"


def draw_walk_speed_mpm(age: float, sex: str, rng: Rng) -> float:
    female = sex == "Female"
    v = 1.39 if female else 1.43
    if age > 30:
        v -= 0.0025 * (age - 30)
    if age > 60:
        v -= 0.006 * (age - 60)
    v += gauss(rng) * 0.18
    return clamp(v, 0.5, 1.85) * 60


def draw_decision_lag_min(age: float, rng: Rng) -> int:
    mu = math.log(3.2) + max(0.0, age - 40) * 0.012
    x = lognormal(rng, mu, 0.5)
    return js_round(clamp(x, 1, 12))


def draw_sip_bias(belief: dict, rng: Rng) -> float:
    age = num(belief.get("age"), 45)
    au = num(belief.get("au"), 0.5)
    elderly = 0.28 if age >= 65 else (0.12 if age >= 55 else 0)
    autonomy = 0.22 * au
    noise = (rng() - 0.5) * 0.16
    return clamp(0.22 + elderly + autonomy + noise, 0.05, 0.85)


def cell_center(env: dict, i: int, j: int) -> Tuple[float, float]:
    m = env["cellM"]
    return ((i + 0.5) * m, (j + 0.5) * m)


def cell_index(env: dict, i: int, j: int) -> int:
    n = env["n"]
    if i < 0 or j < 0 or i >= n or j >= n:
        return -1
    return env["lookup"][i + j * n]


def pick_occupied_weighted(env: dict, rng: Rng) -> dict:
    u = rng() * env["massTotal"]
    mass = env["mass"]
    cells = env["cells"]
    for i, m in enumerate(mass):
        u -= m
        if u <= 0:
            return cells[i]
    return cells[-1]


def build_env(grid: dict) -> dict:
    n = grid["n"]
    lookup = [-1] * (n * n)
    for idx, c in enumerate(grid["cells"]):
        lookup[c["i"] + c["j"] * n] = idx
    nearest = [-1] * (n * n)
    q = [0] * (n * n)
    head = tail = 0
    for c_i, cell in enumerate(grid["cells"]):
        packed = cell["i"] + cell["j"] * n
        nearest[packed] = c_i
        q[tail] = packed
        tail += 1
    while head < tail:
        packed = q[head]
        head += 1
        src = nearest[packed]
        i = packed % n
        j = packed // n
        nbrs = []
        if i > 0:
            nbrs.append(packed - 1)
        if i + 1 < n:
            nbrs.append(packed + 1)
        if j > 0:
            nbrs.append(packed - n)
        if j + 1 < n:
            nbrs.append(packed + n)
        for nb in nbrs:
            if nearest[nb] >= 0:
                continue
            nearest[nb] = src
            q[tail] = nb
            tail += 1
    mass = [0.0] * len(grid["cells"])
    pop_mass = 0.0
    for i, cell in enumerate(grid["cells"]):
        pop = cell.get("pop")
        if isinstance(pop, (int, float)) and pop > 0:
            mass[i] = float(pop)
            pop_mass += float(pop)
    if pop_mass > 0:
        mass_total = pop_mass
    else:
        mass_total = 0.0
        for i, cell in enumerate(grid["cells"]):
            mass[i] = max(1.0, float(cell.get("m2", 1)))
            mass_total += mass[i]
    return {
        "grid": grid,
        "cells": grid["cells"],
        "lookup": lookup,
        "nearest": nearest,
        "n": n,
        "cellM": grid["cellM"],
        "mass": mass,
        "massTotal": mass_total,
    }


def synthetic_city(n: int = 12, cell_m: int = 500) -> dict:
    cells = []
    occupied = 0
    elev = [110] * (n * n)
    for j in range(1, n - 1):
        for i in range(1, n - 1):
            if (i + j) % 5 == 0:
                continue
            shelter = 1 if ((i % 4 == 0 and j % 4 == 0) or (i == 6 and j == 6)) else 0
            si = i if shelter else 6
            sj = j if shelter else 6
            sd = math.hypot((si - i) * cell_m, (sj - j) * cell_m)
            cells.append({
                "i": i, "j": j,
                "b": 8 + ((i * 3 + j) % 20),
                "m2": 400 + i * 80 + j * 40,
                "lv": 4, "lvn": 1, "h": 12, "e": 110,
                "metro": 1 if shelter and i == 6 else 0,
                "water": 0, "shelter": shelter,
                "si": si, "sj": sj, "sd": sd,
                "pop": 8 + ((i * 3 + j) % 20) * 4,
            })
            occupied += 1
    return {
        "source": "synthetic 12×12 architecture city",
        "note": "Not Kyiv. Used only by interface checks.",
        "south": 50.4, "west": 30.5, "north": 50.46, "east": 30.58,
        "n": n, "cellM": cell_m, "mPerLat": 111320, "mPerLon": 71000,
        "buildings": occupied * 10, "levelsKnown": occupied, "occupied": occupied,
        "shelterCells": sum(1 for c in cells if c["shelter"]),
        "metroCells": sum(1 for c in cells if c["metro"]),
        "elevMin": 110, "elevMax": 110, "cells": cells, "water": [], "elev": elev,
    }


def empty_gw() -> dict:
    return {"beliefs": [], "desires": [], "intentions": [], "winner": "none", "ignited": False}


def apply_routine_thresholds(agent: dict, p: dict) -> None:
    if agent["routine"] == "idle":
        agent["focused"] = False
        agent["saliencyThreshold"] = p["idleThreshold"]
        agent["attentionThreshold"] = p["idleThreshold"]
        return
    sal = 0.8 if agent["routine"] == "asleep" else 0.5
    agent["focused"] = True
    agent["saliencyThreshold"] = sal
    agent["attentionThreshold"] = focused_attention_threshold(sal)


def seed_agent(id_: int, env: dict, belief: dict, p: dict, rng: Rng, exposure: float, hour=12, weekday=2) -> dict:
    cell = pick_occupied_weighted(env, rng)
    cx, cy = cell_center(env, cell["i"], cell["j"])
    age = num(belief.get("age"), 42)
    sex = "Male" if belief.get("sex") == "Male" else "Female"
    employed = num(belief.get("em"), 0) > 0.5
    routine = draw_routine(hour, weekday, employed, rng)
    agent = {
        "id": id_,
        "i": cell["i"], "j": cell["j"],
        "x": cx + (rng() - 0.5) * env["cellM"] * 0.4,
        "y": cy + (rng() - 0.5) * env["cellM"] * 0.4,
        "age": age, "sex": sex, "employed": employed,
        "scepticism": belief["sc"],
        "defiance": num(belief.get("df"), 0.32),
        "autonomy": num(belief.get("au"), 0.49),
        "trustArmyInv": num(belief.get("ta"), 0.4),
        "confArmy": num(belief.get("ca"), 0.58),
        "confGov": num(belief.get("cg"), 0.25),
        "security": num(belief.get("se"), 0.6),
        "socialTrust": num(belief.get("st"), 0.3),
        "neighSecure": num(belief.get("ns"), 0.7),
        "fight": num(belief.get("fi"), 0.6),
        "walkSpeedMpm": draw_walk_speed_mpm(age, sex, rng),
        "decisionLagMin": draw_decision_lag_min(age, rng),
        "sipBias": draw_sip_bias(belief, rng),
        "routine": routine,
        "focused": False,
        "saliencyThreshold": p["idleThreshold"],
        "attentionThreshold": p["idleThreshold"],
        "standing": [{"id": "Stay_Safe", "kind": "practical", "standing": True, "active": False, "saliency": 0, "precondition": True}],
        "active": [],
        "intentions": [],
        "gw": empty_gw(),
        "inhibited": set(),
        "exposure": exposure,
        "action": "continue",
        "destI": cell["i"], "destJ": cell["j"],
        "arrived": False, "wait": 0, "metres": 0,
        "heard": False, "ignited": False, "defied": False,
    }
    apply_routine_thresholds(agent, p)
    return agent


def make_population(n: int, env: dict, pool: List[dict], p: dict, rng: Rng, exposure=0, hour=12, weekday=2) -> List[dict]:
    return [seed_agent(i, env, pick_weighted(pool, rng), p, rng, exposure, hour, weekday) for i in range(n)]


def clone_agents(agents: List[dict]) -> List[dict]:
    out = []
    for a in agents:
        b = copy.deepcopy(a)
        b["inhibited"] = set(a["inhibited"])
        out.append(b)
    return out

