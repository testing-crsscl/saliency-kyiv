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

APPRAISAL_PRIOR = (0.35, 0.25, 0.15, 0.1, 0.1, 0.05)
APPRAISAL_WEIGHTS = {
    "scepticismInv": APPRAISAL_PRIOR[0],
    "confArmy": APPRAISAL_PRIOR[1],
    "security": APPRAISAL_PRIOR[2],
    "confGov": APPRAISAL_PRIOR[3],
    "neighInsecurity": APPRAISAL_PRIOR[4],
    "fight": APPRAISAL_PRIOR[5],
}
APPRAISAL_CUE_LABELS = (
    "1 − scepticism",
    "confArmy",
    "security vs freedom",
    "confGov",
    "1 − neighbourhood secure",
    "willingness to fight",
)
APPRAISAL_KAPPA = 12
DEFIANCE_PRIOR = (0.55, 0.2, 0.15)
DEFIANCE_KAPPA = 10
STAY_MIX_MEAN = 0.4
STAY_MIX_BETA = (8, 12)
IDLE_BETA = (12, 28)
ACOUSTIC_BETA = (28, 2.1)
TAU_LOG_SIGMA = 0.22

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


def gamma(rng: Rng, shape: float) -> float:
    """Marsaglia–Tsang gamma(shape, 1). Same loop as src/sim/rng.ts."""
    if shape < 1e-12:
        return 0.0
    if shape < 1:
        u = max(1e-12, rng())
        return gamma(rng, shape + 1) * (u ** (1.0 / shape))
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = 0.0
        v = 0.0
        while True:
            x = gauss(rng)
            v = 1.0 + c * x
            if v > 0:
                break
        v = v * v * v
        u = rng()
        x2 = x * x
        if u < 1.0 - 0.0331 * x2 * x2:
            return d * v
        if math.log(u) < 0.5 * x2 + d * (1.0 - v + math.log(v)):
            return d * v


def beta(rng: Rng, a: float, b: float) -> float:
    x = gamma(rng, a)
    y = gamma(rng, b)
    s = x + y
    if s <= 0:
        return a / (a + b)
    return x / s


def dirichlet(rng: Rng, alpha) -> List[float]:
    g = [gamma(rng, max(float(a), 1e-6)) for a in alpha]
    s = sum(g)
    if s <= 0:
        return [1.0 / len(alpha)] * len(alpha)
    return [x / s for x in g]


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


def union_probability(cues, weights) -> float:
    log_survive = 0.0
    n = max(len(cues), len(weights))
    for i in range(n):
        p = clamp(cues[i] if i < len(cues) else 0.0, 0.0, 1.0 - 1e-12)
        w = max(0.0, weights[i] if i < len(weights) else 0.0)
        log_survive += w * math.log(1.0 - p)
    return clamp(1.0 - math.exp(log_survive), 0.0, 0.999)


def threat_cues(agent: dict) -> List[float]:
    return [
        1.0 - agent["scepticism"],
        agent["confArmy"],
        agent["security"],
        agent["confGov"],
        1.0 - agent["neighSecure"],
        agent["fight"],
    ]


def defiance_cues(agent: dict) -> List[float]:
    return [agent["defiance"], 1.0 - agent["security"], 1.0 - agent["confArmy"]]


def focused_attention_threshold(saliency_threshold: float) -> float:
    return saliency_threshold + (1 - saliency_threshold) / 2


def acoustic_saliency(exposure: float, p: dict = None, agent: dict = None) -> float:
    p = p or DEFAULT_PARAMS
    base = (agent or {}).get("acousticBase", p["acousticBase"])
    tau = max(1.0, (agent or {}).get("habituationTau", p["habituationTau"]))
    return min(0.999, max(0.0, base * math.exp(-exposure / tau)))


def appraise_threat(agent: dict) -> float:
    w = agent.get("appraisalW") or list(APPRAISAL_PRIOR)
    return union_probability(threat_cues(agent), w)


def defiance_strength(agent: dict) -> float:
    w = agent.get("defianceW") or list(DEFIANCE_PRIOR)
    return union_probability(defiance_cues(agent), w)


def stay_safe_saliency(agent: dict, p: dict = None) -> float:
    p = p or DEFAULT_PARAMS
    acoustic = acoustic_saliency(agent["exposure"], p, agent)
    threat = appraise_threat(agent)
    a = agent.get("stayMix", STAY_MIX_MEAN)
    nominated = 1.0 - (1.0 - a) * (1.0 - threat)
    return min(0.999, acoustic * nominated)


def siren_saliency(agent: dict, p: dict) -> float:
    return acoustic_saliency(agent["exposure"], p, agent)


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


def draw_processor(rng: Rng, p: dict) -> dict:
    idle_mean = IDLE_BETA[0] / (IDLE_BETA[0] + IDLE_BETA[1])
    ac_mean = ACOUSTIC_BETA[0] / (ACOUSTIC_BETA[0] + ACOUSTIC_BETA[1])
    return {
        "appraisalW": dirichlet(rng, [m * APPRAISAL_KAPPA for m in APPRAISAL_PRIOR]),
        "defianceW": dirichlet(rng, [m * DEFIANCE_KAPPA for m in DEFIANCE_PRIOR]),
        "stayMix": clamp(beta(rng, STAY_MIX_BETA[0], STAY_MIX_BETA[1]), 0.12, 0.75),
        "idleBar": clamp(p["idleThreshold"] + (beta(rng, IDLE_BETA[0], IDLE_BETA[1]) - idle_mean), 0.12, 0.55),
        "acousticBase": clamp(p["acousticBase"] + (beta(rng, ACOUSTIC_BETA[0], ACOUSTIC_BETA[1]) - ac_mean), 0.45, 0.995),
        "habituationTau": clamp(lognormal(rng, math.log(max(1.0, p["habituationTau"])), TAU_LOG_SIGMA), 80, 1e9),
    }


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
        bar = agent.get("idleBar", p["idleThreshold"])
        agent["focused"] = False
        agent["saliencyThreshold"] = bar
        agent["attentionThreshold"] = bar
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
    x = cx + (rng() - 0.5) * env["cellM"] * 0.4
    y = cy + (rng() - 0.5) * env["cellM"] * 0.4
    walk = draw_walk_speed_mpm(age, sex, rng)
    lag = draw_decision_lag_min(age, rng)
    sip = draw_sip_bias(belief, rng)
    proc = draw_processor(rng, p)
    agent = {
        "id": id_,
        "i": cell["i"], "j": cell["j"],
        "x": x,
        "y": y,
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
        "walkSpeedMpm": walk,
        "decisionLagMin": lag,
        "sipBias": sip,
        **proc,
        "routine": routine,
        "focused": False,
        "saliencyThreshold": proc["idleBar"],
        "attentionThreshold": proc["idleBar"],
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

