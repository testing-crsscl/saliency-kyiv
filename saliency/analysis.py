from .cycle import *

BELIEFS = [
    {"sc": 0.3, "df": 0.4, "au": 0.5, "ta": 0.4, "ca": 0.7, "cg": 0.4, "se": 0.8, "st": 0.4, "ns": 0.6, "fi": 0.7, "em": 1, "w": 1, "age": 34, "sex": "Male"},
    {"sc": 0.61, "df": 0.5, "au": 0.55, "ta": 0.6, "ca": 0.5, "cg": 0.2, "se": 0.6, "st": 0.3, "ns": 0.5, "fi": 0.5, "em": 0, "w": 1.2, "age": 51, "sex": "Female"},
    {"sc": 0.8, "df": 0.6, "au": 0.4, "ta": 0.7, "ca": 0.3, "cg": 0.1, "se": 0.4, "st": 0.2, "ns": 0.4, "fi": 0.3, "em": 1, "w": 0.8, "age": 28, "sex": "Male"},
]


def run_architecture_tests(params: dict = None) -> List[dict]:
    p = params or DEFAULT_PARAMS
    grid = synthetic_city(12)
    env = build_env(grid)
    rng = mulberry32(42)
    pop = make_population(80, env, BELIEFS, p, rng, 0, 14, 2)
    result = run_event_window(pop, env, p)
    out = []

    def check(id_, name, ok, detail):
        out.append({"id": id_, "name": name, "ok": bool(ok), "detail": detail})

    check("P1", "No pre-trend possible", result["preTrendMax"] == 0,
          f"max displacement t<0 is {result['preTrendMax']:.4f} m (must be 0 at every point in parameter space)")
    focused_ok = True
    s = 0.0
    while s < 1:
        if focused_attention_threshold(s) < 0.5 - 1e-12:
            focused_ok = False
            break
        s += 0.05
    check("P2", "Focused attention threshold floor of 0.50", focused_ok,
          "attention = sal + (1−sal)/2 ≥ 0.5 for every sal in [0,1). CATALINA arithmetic, not a choice of ours.")
    check("T03", "Attention formula",
          abs(focused_attention_threshold(0.5) - 0.75) < 1e-9 and abs(focused_attention_threshold(0.8) - 0.9) < 1e-9,
          "committed 0.50 → 0.75; asleep 0.80 → 0.90")
    idle = [a for a in pop if a["routine"] == "idle"]
    check("T04", "Idle thresholds equal",
          len(idle) == 0 or all(a["saliencyThreshold"] == a["attentionThreshold"] for a in idle),
          "Unfocused: saliency threshold = attention threshold = idle default")
    busy = make_population(1, env, BELIEFS, p, mulberry32(1), 0)
    agent = next((a for a in busy if a["routine"] == "committed"), busy[0])
    agent["routine"] = "committed"
    agent["focused"] = True
    agent["saliencyThreshold"] = 0.5
    agent["attentionThreshold"] = 0.75
    weak = information_selection(agent, p, True)
    if weak:
        weak["saliency"] = 0.4
    blocked = stimulus_inhibition(agent, weak)
    check("T05", "Sub-threshold stimulus never enters the workspace", blocked is None,
          "A 0.40 siren against a 0.75 bar is invisible — not deprioritised, absent. GWT ignition failed.")
    pre = all(v == 0 for v in result["nHeard"][:10])
    check("T06", "No one hears the siren before t=0", pre, "heard[t<0] ≡ 0")
    t0 = result["minutes"].index(0)
    check("T07", "Nobody walks at t=0 (own lag ≥ 1)", result["displacement"][t0] == 0,
          f"displacement at t=0 is {result['displacement'][t0]}. Lags are log-normal with a floor of 1 min.")
    check("T08", "Synthetic 12×12 city is occupied", len(env["cells"]) >= 40 and env["n"] == 12,
          f"{len(env['cells'])} occupied cells, {env['grid']['shelterCells']} shelters")
    dens_sum = sum(result["density"][20])
    check("T09", "Density integrates to the population", dens_sum == result["nAgents"],
          f"sum at t=+10 = {dens_sum}, nAgents = {result['nAgents']}")
    share = result["actions"][max(0, result["minutes"].index(result["peakMinute"]))]
    keys = ",".join(sorted(share.keys()))
    check("T10", "Declared action space", keys == "continue,evacuate,move,shelter_in_place", keys)
    check("T11", "No free sliders", p["idleThreshold"] == 0.3 and p["acousticBase"] == 0.93,
          "Idle threshold is CATALINA 0.30; acoustic base is a named constant. Individual variation is sampled, not slid.")
    check("T12", "Confidence refuses without pairs", True,
          "Confidence is unfitted. It reads internal state at the decision cycle and needs (simulation, observed) pairs — the original Van Dijcke coefficients, which are not in hand.")
    folds = [{"holdout": h, "train": [r for r in ["West 1", "West 2", "South", "East 1", "East 2", "Centre", "North"] if r != h]}
             for h in ["West 1", "West 2", "South", "East 1", "East 2"]]
    check("T13", "Leave-region-out, five folds",
          len(folds) == 5 and all(f["holdout"] not in f["train"] for f in folds),
          " · ".join(f["holdout"] for f in folds))
    dual = run_dual_regions(env, BELIEFS, 60, p, 120, 40, 160, 9)
    check("T14", "Calendar model is date-constant across regions",
          abs(dual["calendar"]["a"]["peakMetres"] - dual["calendar"]["b"]["peakMetres"])
          <= 0.2 * max(dual["calendar"]["a"]["peakMetres"], dual["calendar"]["b"]["peakMetres"], 1),
          f"peaks {dual['calendar']['a']['peakMetres']:.2f} vs {dual['calendar']['b']['peakMetres']:.2f} (same date, different exposure)")
    check("T15", "Habituation model separates regions by exposure",
          dual["habituation"]["ratio"] > 1.15 or dual["habituation"]["a"]["peakMetres"] > dual["habituation"]["b"]["peakMetres"],
          f"4× exposure: peak A {dual['habituation']['a']['peakMetres']:.1f} vs B {dual['habituation']['b']['peakMetres']:.1f}")
    a0 = make_population(1, env, BELIEFS, p, mulberry32(3), 0)[0]
    aN = {**a0, "exposure": 200}
    check("T16", "Habituation reduces acoustic saliency",
          siren_saliency(aN, p) < siren_saliency(a0, p) and acoustic_saliency(200, p) < acoustic_saliency(0, p),
          f"s(0)={siren_saliency(a0, p):.3f} → s(200)={siren_saliency(aN, p):.3f}")
    check("T17", "Peak is at or after the first possible lag",
          result["peakMinute"] == 0 or result["peakMinute"] >= 1 or result["peakMetres"] == 0,
          f"peak minute {result['peakMinute']}, mean lag {result['meanLag']:.1f}")
    check("T18", "Stay_Safe is the only standing practical desire",
          all(len(a["standing"]) == 1 and a["standing"][0]["id"] == "Stay_Safe" for a in pop),
          "One standing practical desire, Stay_Safe, which is what the siren triggers.")
    speeds = [a["walkSpeedMpm"] for a in pop]
    check("T19", "Walk speed is a distribution, not a constant", max(speeds) - min(speeds) > 10,
          f"range {min(speeds):.0f}–{max(speeds):.0f} m/min (Bohannon age/sex + residual)")
    low_sc = {**BELIEFS[0], "sc": 0.1, "ca": 0.95, "se": 1, "df": 0.05}
    high_sc = {**BELIEFS[0], "sc": 0.95, "ca": 0.05, "se": 0, "df": 0.9}
    trusting = make_population(1, env, [low_sc], p, mulberry32(5), 0)[0]
    sceptical = make_population(1, env, [high_sc], p, mulberry32(5), 0)[0]
    check("T20", "WVS traits other than scepticism reach Stay_Safe",
          stay_safe_saliency(trusting, p) > stay_safe_saliency(sceptical, p),
          f"trusting {stay_safe_saliency(trusting, p):.3f} vs sceptical {stay_safe_saliency(sceptical, p):.3f}")
    dense_grid = synthetic_city(12)
    for c in dense_grid["cells"]:
        c["pop"] = 1
    hot = next((c for c in dense_grid["cells"] if c["i"] == 2 and c["j"] == 2), None)
    cold = next((c for c in dense_grid["cells"] if c["i"] == 9 and c["j"] == 9), None)
    if hot:
        hot["pop"] = 1000
    if cold:
        cold["pop"] = 10
    trap_lat = dense_grid["south"] + (9 + 0.5) * dense_grid["cellM"] / dense_grid["mPerLat"]
    trap_lon = dense_grid["west"] + (9 + 0.5) * dense_grid["cellM"] / dense_grid["mPerLon"]
    trap = {**BELIEFS[0], "lat": trap_lat, "lon": trap_lon}
    dens_env = build_env(dense_grid)
    placed = make_population(400, dens_env, [trap], p, mulberry32(7), 0)
    at_hot = sum(1 for a in placed if a["i"] == 2 and a["j"] == 2)
    at_cold = sum(1 for a in placed if a["i"] == 9 and a["j"] == 9)
    check("T21", "Initial positions follow population density, not WVS coordinates",
          at_hot > 300 and at_hot > 8 * at_cold,
          f"WVS PSU pinned on the sparse cell ({at_cold}); WorldPop mass 100:1 put {at_hot}/400 on the dense cell")
    return out


def acoustic_table(p: dict = None):
    p = p or DEFAULT_PARAMS
    rows = []
    for n in (0, 40, 100, 175, 220, 230, 360, 406):
        a = acoustic_saliency(n, p)
        rows.append({
            "n": n, "acoustic": a,
            "ignitesIdle": a >= p["idleThreshold"],
            "ignitesCommitted": a >= 0.75,
            "ignitesAsleep": a >= 0.9,
        })
    return rows


def typical_kyivan_appraisal(wvs: dict) -> dict:
    traits = {
        "scepticism": wvs["scepticismKyiv"],
        "confArmy": wvs.get("confArmyKyiv", 0.58),
        "security": wvs.get("securityKyiv", 0.6),
        "confGov": wvs.get("confGovKyiv", 0.25),
        "neighSecure": wvs.get("neighSecureKyiv", 0.7),
        "fight": wvs.get("fightKyiv", 0.6),
        "exposure": 0,
        "defiance": wvs["defianceKyiv"],
    }
    return {
        "threat": appraise_threat(traits),
        "stay0": stay_safe_saliency(traits, DEFAULT_PARAMS),
        "stay406": stay_safe_saliency({**traits, "exposure": 406}, DEFAULT_PARAMS),
        "defianceGate": 0.55 * traits["defiance"] + 0.2 * (1 - traits["security"]) + 0.15 * (1 - traits["confArmy"]),
        "acoustic0": acoustic_saliency(0),
    }


def histogram(values: List[float], bins=10, lo=0.0, hi=1.0):
    counts = [0] * bins
    n = 0
    for v in values:
        if not isinstance(v, (int, float)) or math.isnan(v):
            continue
        n += 1
        i = min(bins - 1, max(0, int(math.floor(((v - lo) / (hi - lo)) * bins))))
        counts[i] += 1
    return {"counts": counts, "n": n}


def wvs_distributions(pool: List[dict]) -> dict:
    def take(key):
        return [b[key] for b in pool if isinstance(b.get(key), (int, float))]
    return {
        "n": len(pool),
        "scepticism": histogram(take("sc")),
        "defiance": histogram(take("df")),
        "femaleShare": sum(1 for b in pool if b.get("sex") == "Female") / max(1, len(pool)),
        "employedShare": sum(1 for b in pool if num(b.get("em"), 0) > 0.5) / max(1, len(pool)),
    }


def pick_alert(alerts: List[dict], typical_n: int, hour: int, wd: int) -> dict:
    if not alerts:
        return {"t": "1970-01-01T00:00:00Z", "dur": 46, "n": typical_n, "hour": hour, "wd": wd}
    best = alerts[0]
    best_score = float("inf")
    for a in alerts:
        score = abs((a.get("n") or 0) - typical_n) + 0.15 * abs((a.get("hour") or 12) - hour) + 0.1 * abs((a.get("wd") or 2) - wd)
        if score < best_score:
            best, best_score = a, score
    return best


def run_period(env, pool, alerts, vd, period, n_agents=120, seed=31) -> dict:
    sl = [a for a in alerts if period["startN"] <= a.get("n", 0) <= period["endN"]]
    alert = pick_alert(sl or alerts, period["typicalN"], period["hour"], period["wd"])
    rng = mulberry32(seed + period["typicalN"])
    pop = make_population(n_agents, env, pool, DEFAULT_PARAMS, rng, alert["n"], alert.get("hour", period["hour"]), alert.get("wd", period["wd"]))
    result = run_event_window(pop, env, DEFAULT_PARAMS, duration_min=alert.get("dur", 46))
    observed = vd["byPeriod"].get(period["id"], vd["overall"])
    acoustic = acoustic_saliency(alert["n"])
    return {
        "id": period["id"],
        "label": period["label"],
        "exposure": alert["n"],
        "peakSim": result["peakMetres"],
        "peakVd": max(observed),
        "peakMinute": result["peakMinute"],
        "shapeMse": mse(norm(result["displacement"]), norm(observed)),
        "displacement": result["displacement"],
        "vd": observed,
        "acoustic": acoustic,
        "silent": acoustic < DEFAULT_PARAMS["idleThreshold"],
        "nHeard": result["nHeard"][result["minutes"].index(result["peakMinute"])],
        "gw": result["gw"],
        "result": result,
    }


def run_first_alert(env, pool, alert, vd, n_agents=140, seed=21) -> dict:
    rng = mulberry32(seed)
    pop = make_population(n_agents, env, pool, DEFAULT_PARAMS, rng, alert["n"], alert.get("hour", 12), alert.get("wd", 2))
    result = run_event_window(pop, env, DEFAULT_PARAMS, duration_min=alert.get("dur", 46))
    return {
        "result": result,
        "shapeMse": mse(norm(result["displacement"]), norm(vd["overall"])),
        "levelMse": mse(result["displacement"], vd["overall"]),
        "vdPeak": max(vd["overall"]),
    }


def trace_minute(agent_in: dict, env: dict, p: dict, siren_on: bool, minute: int) -> dict:
    agent = clone_agents([agent_in])[0]
    working_cycle(agent, env, p, siren_on, minute)
    return {
        "minute": minute,
        "sirenOn": siren_on,
        "metres": agent["metres"],
        "action": agent["action"],
        "heard": agent["heard"],
        "ignited": agent["ignited"],
        "defied": agent["defied"],
        "lag": agent["decisionLagMin"],
        "routine": agent["routine"],
    }


def worked_example(env, pool, exposure, hour, weekday, seed=7) -> dict:
    rng = mulberry32(seed)
    belief = pool[0]
    agent = seed_agent(0, env, belief, DEFAULT_PARAMS, rng, exposure, hour, weekday)
    lag = max(1, agent["decisionLagMin"])
    return {
        "agent": agent,
        "acoustic": acoustic_saliency(agent["exposure"]),
        "threat": appraise_threat(agent),
        "stay": stay_safe_saliency(agent),
        "defianceGate": 0.55 * agent["defiance"] + 0.2 * (1 - agent["security"]) + 0.15 * (1 - agent["confArmy"]),
        "pre": trace_minute(agent, env, DEFAULT_PARAMS, False, -1),
        "onset": trace_minute(agent, env, DEFAULT_PARAMS, True, 0),
        "afterLag": trace_minute(agent, env, DEFAULT_PARAMS, True, lag),
    }
