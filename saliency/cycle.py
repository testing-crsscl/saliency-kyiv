from .core import *


def information_selection(agent: dict, p: dict, raw: bool):
    if not raw:
        return None
    return {"name": "siren", "saliency": siren_saliency(agent, p), "region": "local"}


def stimulus_inhibition(agent: dict, stim):
    if not stim:
        return None
    if stim["saliency"] < agent["attentionThreshold"]:
        return None
    if stim["region"] in agent["inhibited"]:
        return None
    return stim


def gw_maintenance(agent: dict, stim) -> None:
    agent["gw"]["beliefs"] = [stim] if stim else []
    agent["gw"]["ignited"] = bool(stim)
    agent["gw"]["winner"] = "siren" if stim else ("routine" if agent["focused"] else "none")
    agent["ignited"] = bool(stim)


def desire_deletion(agent: dict) -> None:
    agent["active"] = [d for d in agent["active"] if d["saliency"] >= agent["saliencyThreshold"]]


def intention_deletion(agent: dict) -> None:
    live = {d["id"] for d in agent["active"]}
    agent["intentions"] = [it for it in agent["intentions"] if it["desireId"] in live]
    agent["gw"]["intentions"] = agent["intentions"]
    agent["gw"]["desires"] = agent["active"]


def focus_agent(agent: dict) -> None:
    top = max(agent["intentions"], key=lambda it: it["saliency"])
    agent["focused"] = True
    agent["saliencyThreshold"] = top["saliency"]
    agent["attentionThreshold"] = focused_attention_threshold(top["saliency"])
    agent["inhibited"] = {"far"}


def unfocus_agent(agent: dict, p: dict) -> None:
    agent["focused"] = False
    agent["saliencyThreshold"] = p["idleThreshold"]
    agent["attentionThreshold"] = p["idleThreshold"]
    agent["inhibited"] = set()


def switching_to_stimulus(agent: dict, p: dict) -> bool:
    stim = next((b for b in agent["gw"]["beliefs"] if b["name"] == "siren"), None)
    if not stim:
        return False
    stay = next((d for d in agent["standing"] if d["id"] == "Stay_Safe"), None)
    if not stay:
        return False
    stay_sal = stay_safe_saliency(agent, p)
    defiance_gate = 0.55 * agent["defiance"] + 0.2 * (1 - agent["security"]) + 0.15 * (1 - agent["confArmy"])
    if defiance_gate > stay_sal:
        agent["defied"] = True
        agent["gw"]["winner"] = "routine" if agent["focused"] else "none"
        return False
    stay["standing"] = False
    stay["active"] = True
    stay["saliency"] = stay_sal
    if not any(d["id"] == "Stay_Safe" for d in agent["active"]):
        agent["active"].append(stay)
    agent["gw"]["desires"] = agent["active"]
    agent["gw"]["winner"] = "siren"
    agent["defied"] = False
    return True


def desire_promotion(agent: dict) -> bool:
    changed = False
    for d in agent["standing"]:
        if d["active"]:
            continue
        if not d["precondition"]:
            continue
        bar = agent["attentionThreshold"] if d["id"] in agent["inhibited"] else agent["saliencyThreshold"]
        if d["saliency"] > bar:
            d["active"] = True
            d["standing"] = False
            agent["active"].append(d)
            changed = True
    if changed:
        agent["gw"]["desires"] = agent["active"]
    return changed


def age_boost(age: float) -> float:
    if age >= 70:
        return 0.22
    if age >= 60:
        return 0.12
    return 0.0


def means_end_reasoner(agent: dict, env: dict, _p: dict) -> List[dict]:
    out = []
    for d in agent["active"]:
        if d["id"] != "Stay_Safe":
            continue
        idx = cell_index(env, agent["i"], agent["j"])
        here = env["cells"][idx] if idx >= 0 else None
        dist = here["sd"] if here else 5000
        if here and (here.get("shelter") or dist < 40):
            option = "shelter_in_place"
        elif dist <= 1500:
            option = "move"
        else:
            option = "evacuate"
        sip_score = agent["sipBias"] + (age_boost(agent["age"]) + agent["autonomy"] * 0.15) - dist / 4000
        if option != "shelter_in_place" and sip_score > 0.52:
            option = "shelter_in_place"
        dest_i = agent["i"] if option == "shelter_in_place" else (here["si"] if here else agent["i"])
        dest_j = agent["j"] if option == "shelter_in_place" else (here["sj"] if here else agent["j"])
        out.append({"desireId": d["id"], "option": option, "destI": dest_i, "destJ": dest_j, "saliency": d["saliency"]})
    return out


def filtering_process(options: List[dict]) -> List[dict]:
    return [o for o in options if o["option"] != "continue"]


def deliberation_process(agent: dict, options: List[dict]) -> bool:
    if not options:
        return False
    best = max(options, key=lambda o: o["saliency"])
    prev = agent["intentions"][0] if agent["intentions"] else None
    changed = (not prev) or prev["option"] != best["option"] or prev["destI"] != best["destI"]
    agent["intentions"] = [best]
    agent["gw"]["intentions"] = agent["intentions"]
    agent["action"] = best["option"]
    agent["destI"] = best["destI"]
    agent["destJ"] = best["destJ"]
    return changed


def plan_advancement_evaluation(agent: dict, env: dict) -> bool:
    if not agent["intentions"]:
        return False
    it = agent["intentions"][0]
    dx, dy = cell_center(env, it["destI"], it["destJ"])
    dx -= agent["x"]
    dy -= agent["y"]
    if math.hypot(dx, dy) < 8:
        agent["arrived"] = True
        agent["active"] = [d for d in agent["active"] if d["id"] != it["desireId"]]
        agent["intentions"] = []
        agent["gw"]["intentions"] = []
        agent["action"] = "shelter_in_place"
        return False
    return True


def plan_execution(agent: dict, env: dict, _p: dict, can_walk: bool) -> None:
    agent["metres"] = 0
    if not agent["intentions"]:
        return
    it = agent["intentions"][0]
    if it["option"] == "shelter_in_place":
        if not agent["arrived"]:
            agent["metres"] = SHELTER_IN_PLACE_METRES
            agent["arrived"] = True
        return
    if not can_walk:
        return
    dx, dy = cell_center(env, it["destI"], it["destJ"])
    dx -= agent["x"]
    dy -= agent["y"]
    dist = math.hypot(dx, dy)
    step = min(agent["walkSpeedMpm"], dist)
    if dist > 0:
        agent["x"] += (dx / dist) * step
        agent["y"] += (dy / dist) * step
    agent["metres"] = step
    n = env["n"]
    m = env["cellM"]
    agent["i"] = min(n - 1, max(0, int(math.floor(agent["x"] / m))))
    agent["j"] = min(n - 1, max(0, int(math.floor(agent["y"] / m))))
    if step >= dist - 1e-6:
        agent["arrived"] = True
        agent["action"] = "shelter_in_place"


def working_cycle(agent: dict, env: dict, p: dict, siren_on: bool, minute: int) -> None:
    agent["metres"] = 0
    agent["heard"] = False
    agent["ignited"] = False
    agent["defied"] = False
    raw = bool(siren_on)
    stim = stimulus_inhibition(agent, information_selection(agent, p, raw))
    gw_maintenance(agent, stim)
    if stim:
        agent["heard"] = True
    desire_deletion(agent)
    intention_deletion(agent)
    exogenous = switching_to_stimulus(agent, p)
    if not exogenous:
        desire_promotion(agent)
    intention_changed = False
    if agent["active"]:
        options = filtering_process(means_end_reasoner(agent, env, p))
        intention_changed = deliberation_process(agent, options)
    if intention_changed:
        if not agent["intentions"]:
            unfocus_agent(agent, p)
        else:
            focus_agent(agent)
    can_act = minute >= 0 and minute >= agent["decisionLagMin"] and siren_on
    if not can_act:
        agent["metres"] = 0
        return
    if agent["intentions"]:
        if plan_advancement_evaluation(agent, env):
            plan_execution(agent, env, p, True)


def run_event_window(agents_in: List[dict], env: dict, p: dict, duration_min: int = 46) -> dict:
    agents = clone_agents(agents_in)
    n = len(agents)
    displacement = [0.0] * EVENT_MINUTES
    n_heard = [0] * EVENT_MINUTES
    actions = [{k: 0.0 for k in ACTIONS} for _ in range(EVENT_MINUTES)]
    gw = [{"ignited": 0.0, "defied": 0.0, "staySafe": 0.0, "asleep": 0.0, "committed": 0.0, "idle": 0.0} for _ in range(EVENT_MINUTES)]
    density = [[0] * len(env["cells"]) for _ in range(EVENT_MINUTES)]
    n_asleep = n_committed = n_idle = 0
    walk = lag = 0.0
    for a in agents:
        if a["routine"] == "asleep":
            n_asleep += 1
        elif a["routine"] == "committed":
            n_committed += 1
        else:
            n_idle += 1
        walk += a["walkSpeedMpm"]
        lag += a["decisionLagMin"]
    for k in range(EVENT_MINUTES):
        minute = EVENT_T0 + k
        siren_on = minute >= 0 and minute < duration_min
        metres = 0.0
        counts = {k: 0 for k in ACTIONS}
        g = gw[k]
        for a in agents:
            working_cycle(a, env, p, siren_on, minute)
            metres += a["metres"]
            if not a["heard"] and not a["intentions"]:
                counts["continue"] += 1
            else:
                counts[a["action"]] += 1
            if a["heard"]:
                n_heard[k] += 1
            if a["ignited"]:
                g["ignited"] += 1
            if a["defied"]:
                g["defied"] += 1
            if a["action"] != "continue" and a["intentions"]:
                g["staySafe"] += 1
            if a["routine"] == "asleep":
                g["asleep"] += 1
            elif a["routine"] == "committed":
                g["committed"] += 1
            else:
                g["idle"] += 1
            packed = a["i"] + a["j"] * env["n"]
            idx = env["nearest"][packed] if 0 <= a["i"] < env["n"] and 0 <= a["j"] < env["n"] else -1
            if idx >= 0:
                density[k][idx] += 1
        displacement[k] = metres / n if n else 0.0
        for key in ACTIONS:
            actions[k][key] = counts[key] / n if n else 0.0
        if n:
            for key in g:
                g[key] /= n
    peak_minute = 0
    peak_metres = -1.0
    for k in range(EVENT_MINUTES):
        if displacement[k] > peak_metres:
            peak_metres = displacement[k]
            peak_minute = MINUTES[k]
    pre_trend_max = max(displacement[:10]) if displacement else 0.0
    return {
        "minutes": list(MINUTES),
        "displacement": displacement,
        "actions": actions,
        "density": density,
        "nAgents": n,
        "nHeard": n_heard,
        "gw": gw,
        "meanMetres": sum(displacement) / EVENT_MINUTES,
        "peakMinute": peak_minute,
        "peakMetres": peak_metres,
        "preTrendMax": pre_trend_max,
        "meanWalk": walk / n if n else 0,
        "meanLag": lag / n if n else 0,
        "nAsleep": n_asleep,
        "nCommitted": n_committed,
        "nIdle": n_idle,
    }


def mse(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if not n:
        return 0.0
    return sum((a[i] - b[i]) ** 2 for i in range(n)) / n


def norm(xs: List[float]) -> List[float]:
    m = max(xs) if xs else 0
    m = m if m > 1e-9 else 1e-9
    return [v / m for v in xs]


def calendar_response_scale(days: float, tau: float = 201) -> float:
    return math.exp(-days / tau)


def peak_ratio(a: float, b: float) -> float:
    if a < 1e-6 and b < 1e-6:
        return 1.0
    if b < 1e-6:
        return float("inf")
    return a / b


def run_dual_regions(env: dict, pool: List[dict], n_agents: int, p: dict, days: int, exposure_a: int, exposure_b: int, seed: int = 7) -> dict:
    rng = mulberry32(seed)
    pop_a = make_population(n_agents, env, pool, p, rng, exposure_a, 12, 2)
    pop_b = make_population(n_agents, env, pool, p, rng, exposure_b, 12, 2)
    hab_a = run_event_window(pop_a, env, p)
    hab_b = run_event_window(pop_b, env, p)
    scale = calendar_response_scale(days)
    cal_p = {**p, "habituationTau": 1e9, "acousticBase": p["acousticBase"] * scale}
    pop_c1 = make_population(n_agents, env, pool, cal_p, rng, 0, 12, 2)
    pop_c2 = make_population(n_agents, env, pool, cal_p, rng, 0, 12, 2)
    cal_a = run_event_window(pop_c1, env, cal_p)
    cal_b = run_event_window(pop_c2, env, cal_p)
    return {
        "exposureA": exposure_a,
        "exposureB": exposure_b,
        "days": days,
        "habituation": {"a": hab_a, "b": hab_b, "ratio": peak_ratio(hab_a["peakMetres"], hab_b["peakMetres"])},
        "calendar": {"a": cal_a, "b": cal_b, "ratio": peak_ratio(cal_a["peakMetres"], cal_b["peakMetres"])},
    }


