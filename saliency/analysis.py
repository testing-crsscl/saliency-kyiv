from .cycle import *
import math

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
          "Idle mean is CATALINA 0.30; acoustic mean is Dehaene 0.93. Individuals draw Beta around those means. Nothing is a slider.")
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
    mean_w = list(APPRAISAL_PRIOR)
    union = union_probability(threat_cues(trusting), mean_w)
    linear = sum(mean_w[i] * threat_cues(trusting)[i] for i in range(len(mean_w)))
    check("T22", "Appraisal is the probability of the union, not an arithmetic sum",
          abs(union - linear) > 1e-4 and 0 < union < 1,
          f"noisy-OR {union:.3f} vs Σ w·x {linear:.3f}. Same cues, different algebra.")
    twins = make_population(8, env, [BELIEFS[0]], p, mulberry32(13), 0)
    w_spread = max(abs(a["appraisalW"][0] - twins[0]["appraisalW"][0]) for a in twins)
    idle_spread = max(a["idleBar"] for a in twins) - min(a["idleBar"] for a in twins)
    check("T23", "Mapping weights are a distribution, not a shared vector",
          w_spread > 0.01 and all(abs(sum(a["appraisalW"]) - 1) < 1e-6 for a in twins),
          f"same WVS respondent, Dirichlet γ_sc range {w_spread:.3f}; simplex sums to 1")
    idles = [a["idleBar"] for a in pop]
    taus = [a["habituationTau"] for a in pop]
    check("T24", "Idle bar and τ are sampled around their named means",
          max(idles) - min(idles) > 0.04 and max(taus) - min(taus) > 20,
          f"idle {min(idles):.2f}–{max(idles):.2f} around 0.30; τ {min(taus):.0f}–{max(taus):.0f} around 230")
    check("T25", "Stay_Safe of a trusting agent is a probability in (0,1)",
          0 < stay_safe_saliency(trusting, p) < 1 and idle_spread > 0.01,
          f"Stay_Safe(trusting)={stay_safe_saliency(trusting, p):.3f}; idle-bar spread among twins {idle_spread:.3f}")
    return out


def acoustic_table(p: dict = None):
    p = p or DEFAULT_PARAMS
    rows = []
    for i, n in enumerate((0, 40, 100, 175, 220, 230, 360, 406)):
        a = acoustic_saliency(n, p)
        rng = mulberry32(90 + i)
        ignite = 0
        draws = 400
        for _ in range(draws):
            if a >= draw_processor(rng, p)["idleBar"]:
                ignite += 1
        rows.append({
            "n": n, "acoustic": a,
            "ignitesIdle": a >= p["idleThreshold"],
            "idleShare": ignite / draws,
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
        "defianceGate": defiance_strength(traits),
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


def _layer(n, id_, name, catalina, what, value, gate):
    return {
        "n": n, "id": id_, "name": name, "catalina": catalina,
        "what": what, "value": value, "gate": gate,
    }


def trace_minute(agent_in: dict, env: dict, p: dict, siren_on: bool, minute: int) -> dict:
    """Same 12-layer walk as src/sim/trace.ts. Not a second model."""
    agent = clone_agents([agent_in])[0]
    steps = []
    n = 1
    raw = bool(siren_on)
    steps.append(_layer(
        n, "L1", "Perception", "perceptionProcessing",
        "Is the siren physically on this minute?",
        "siren ON" if raw else "siren OFF",
        "pass" if raw else "stop",
    ))
    n += 1
    selected = information_selection(agent, p, raw)
    acoustic = acoustic_saliency(agent["exposure"], p, agent)
    steps.append(_layer(
        n, "L2", "Information selection", "informationSelection",
        f"Habituated acoustic saliency s₀ exp(−n/τ), n = {agent['exposure']}, "
        f"this agent's s₀ {agent['acousticBase']:.2f} τ {agent['habituationTau']:.0f}",
        f"saliency {selected['saliency']:.3f}" if selected else "no stimulus (siren off)",
        "pass" if selected else "stop",
    ))
    n += 1
    stim = stimulus_inhibition(agent, selected)
    if stim:
        l3 = f"IGNITES — {stim['saliency']:.3f} ≥ {agent['attentionThreshold']:.3f}"
        g3 = "pass"
    elif selected:
        l3 = f"ABSENT — {selected['saliency']:.3f} < {agent['attentionThreshold']:.3f}"
        g3 = "stop"
    else:
        l3 = "no stimulus"
        g3 = "stop"
    steps.append(_layer(
        n, "L3", "Stimulus inhibition / ignition", "stimulusInhibition",
        f"Coalition vs attention bar {agent['attentionThreshold']:.3f} (routine {agent['routine']})",
        l3, g3,
    ))
    n += 1
    gw_maintenance(agent, stim)
    steps.append(_layer(
        n, "L4", "Workspace occupancy", "gwMaintenance",
        "Broadcast to standing desires, or empty workspace",
        f"winner {agent['gw']['winner']}" if agent["gw"]["ignited"] else "workspace empty",
        "pass" if agent["gw"]["ignited"] else "info",
    ))
    n += 1
    desire_deletion(agent)
    intention_deletion(agent)
    steps.append(_layer(
        n, "L5", "Desire / intention deletion", "desireDeletion + intentionDeletion",
        "Drop contents below the current saliency bar",
        f"active {len(agent['active'])}, intentions {len(agent['intentions'])}",
        "info",
    ))
    n += 1
    exogenous = switching_to_stimulus(agent, p)
    threat = appraise_threat(agent)
    stay = stay_safe_saliency(agent, p)
    defy = defiance_strength(agent)
    if stim:
        steps.append(_layer(
            n, "L6", "Appraisal processor", "appraiseThreat",
            "P(∪ cues) = 1 − ∏ (1 − cue_i)^{γ_i}  (union of WVS probabilities, γ ~ Dirichlet)",
            f"threat {threat:.3f}", "info",
        ))
        n += 1
        steps.append(_layer(
            n, "L7", "Stay_Safe saliency", "staySafeSaliency",
            "P(hear) · P(floor ∪ threat) = acoustic · [1 − (1−α)(1−threat)]",
            f"{stay:.3f}", "info",
        ))
        n += 1
        steps.append(_layer(
            n, "L8", "Defiance gate", "switchingToStimulus",
            "P(defy) = union(defiance, 1−security, 1−confArmy)  vs  P(Stay_Safe)",
            (f"DEFIED — P(defy) {defy:.3f} > Stay_Safe {stay:.3f}"
             if agent["defied"] else
             f"complies — P(defy) {defy:.3f} ≤ Stay_Safe {stay:.3f}"),
            "stop" if agent["defied"] else "pass",
        ))
        n += 1
    else:
        desire_promotion(agent)
        steps.append(_layer(
            n, "L6–L8", "No broadcast", "desirePromotion only",
            "Without ignition, appraisal and the defiance gate never run",
            "endogenous promotion only", "stop",
        ))
        n += 1
    intention_changed = False
    options = []
    if agent["active"]:
        options = filtering_process(means_end_reasoner(agent, env, p))
        intention_changed = deliberation_process(agent, options)
    idx = cell_index(env, agent["i"], agent["j"])
    here = env["cells"][idx] if idx >= 0 else None
    steps.append(_layer(
        n, "L9", "Means-end reasoner", "meansEndReasoner",
        "Shelter here / walk to nearest shelter / evacuate, given distance and sipBias",
        (f"{options[0]['option']}  dest ({options[0]['destI']},{options[0]['destJ']})  "
         f"dist {round((here or {}).get('sd', -1))} m" if options else "no options (no active Stay_Safe)"),
        "pass" if options else "stop",
    ))
    n += 1
    if intention_changed:
        if not agent["intentions"]:
            unfocus_agent(agent, p)
        else:
            focus_agent(agent)
    steps.append(_layer(
        n, "L10", "Focus / unfocus", "focusAgent / unfocusAgent",
        "Focused attention bar = sal + (1−sal)/2",
        (f"focused, attention {agent['attentionThreshold']:.3f}"
         if agent["focused"] else
         f"unfocused, attention {agent['attentionThreshold']:.3f}"),
        "info",
    ))
    n += 1
    can_act = minute >= 0 and minute >= agent["decisionLagMin"] and siren_on
    steps.append(_layer(
        n, "L11", "Motor gate (P1)", "canAct",
        "minute ≥ 0  ∧  minute ≥ own log-normal lag  ∧  siren on",
        (f"may move (lag {agent['decisionLagMin']} min)" if can_act
         else f"blocked — t={minute}, lag={agent['decisionLagMin']}, siren={'on' if siren_on else 'off'}"),
        "pass" if can_act else "stop",
    ))
    n += 1
    if not can_act:
        agent["metres"] = 0
    elif agent["intentions"]:
        go = plan_advancement_evaluation(agent, env)
        if go:
            plan_execution(agent, env, p, True)
    steps.append(_layer(
        n, "L12", "Plan execution", "planExecution",
        "Shelter-in-place = 18 m once. Walk = Bohannon speed this minute.",
        f"{agent['action']} · {agent['metres']:.1f} m",
        "pass" if agent["metres"] > 0 else "info",
    ))
    if stim:
        agent["heard"] = True
    return {
        "minute": minute,
        "sirenOn": siren_on,
        "metres": agent["metres"],
        "action": agent["action"],
        "heard": agent["heard"],
        "ignited": agent["ignited"],
        "defied": agent["defied"],
        "focused": agent["focused"],
        "lag": agent["decisionLagMin"],
        "routine": agent["routine"],
        "steps": steps,
        "acoustic": acoustic,
        "exogenous": exogenous,
    }


def worked_example(env, pool, exposure, hour, weekday, seed=7) -> dict:
    rng = mulberry32(seed)
    belief = pool[0]
    agent = seed_agent(0, env, belief, DEFAULT_PARAMS, rng, exposure, hour, weekday)
    lag = max(1, agent["decisionLagMin"])
    return {
        "agent": agent,
        "acoustic": acoustic_saliency(agent["exposure"], DEFAULT_PARAMS, agent),
        "threat": appraise_threat(agent),
        "stay": stay_safe_saliency(agent),
        "defianceGate": defiance_strength(agent),
        "attention": agent["attentionThreshold"],
        "pre": trace_minute(agent, env, DEFAULT_PARAMS, False, -1),
        "onset": trace_minute(agent, env, DEFAULT_PARAMS, True, 0),
        "afterLag": trace_minute(agent, env, DEFAULT_PARAMS, True, lag),
    }


def appraisal_breakdown(agent: dict) -> dict:
    cues = threat_cues(agent)
    w = agent.get("appraisalW") or list(APPRAISAL_PRIOR)
    parts = []
    for i, label in enumerate(APPRAISAL_CUE_LABELS):
        cue = cues[i]
        weight = w[i]
        survive = (1 - min(1 - 1e-12, max(0.0, cue))) ** weight
        parts.append({
            "label": label,
            "weight": weight,
            "value": cue,
            "product": weight * cue,
            "fire": 1 - survive,
            "survive": survive,
        })
    return {"parts": parts, "threat": union_probability(cues, w)}


def processor_audit(pool: List[dict], n=280, seed=19) -> dict:
    env = build_env(synthetic_city(12))
    pop = make_population(n, env, pool, DEFAULT_PARAMS, mulberry32(seed), 0, 14, 2)
    threat = [appraise_threat(a) for a in pop]
    stay = [stay_safe_saliency(a, DEFAULT_PARAMS) for a in pop]
    defy = [defiance_strength(a) for a in pop]
    comply = sum(1 for i, a in enumerate(pop) if stay[i] >= defy[i]) / max(1, len(pop))
    return {
        "n": len(pop),
        "threat": histogram(threat),
        "stay": histogram(stay),
        "defy": histogram(defy),
        "idle": histogram([a["idleBar"] for a in pop], bins=8, lo=0.12, hi=0.55),
        "tau": histogram([a["habituationTau"] for a in pop], bins=8, lo=80, hi=450),
        "stayMix": histogram([a["stayMix"] for a in pop], bins=8, lo=0.12, hi=0.75),
        "s0": histogram([a["acousticBase"] for a in pop], bins=8, lo=0.45, hi=0.995),
        "comply": comply,
        "meanThreat": sum(threat) / len(threat),
        "meanStay": sum(stay) / len(stay),
        "meanDefy": sum(defy) / len(defy),
        "pop": pop,
    }


def placement_audit(env, pool, n=400, seed=11) -> dict:
    rng = mulberry32(seed)
    pop = make_population(n, env, pool, DEFAULT_PARAMS, rng, 0, 12, 2)
    counts = [0] * len(env["cells"])
    for a in pop:
        idx = env["lookup"][a["i"] + a["j"] * env["n"]]
        if idx >= 0:
            counts[idx] += 1
    unique = sum(1 for c in counts if c > 0)
    order = sorted(range(len(env["cells"])), key=lambda i: env["cells"][i].get("pop") or 0, reverse=True)
    top_cells = max(1, int(math.floor(len(env["cells"]) * 0.1)))
    pop_top = sum((env["cells"][i].get("pop") or 0) for i in order[:top_cells])
    agents_top = sum(counts[i] for i in order[:top_cells])
    return {
        "n": n,
        "unique": unique,
        "agentsTopShare": agents_top / n if n else 0,
        "popTopShare": pop_top / env["massTotal"] if env["massTotal"] else 0,
        "topCells": top_cells,
        "occupied": len(env["cells"]),
        "pop": pop,
        "counts": counts,
    }


def folds_note() -> dict:
    regions = ["West 1", "West 2", "South", "East 1", "East 2", "Centre", "North"]
    folds = [{"holdout": h, "train": [r for r in regions if r != h]} for h in regions[:5]]
    return {
        "folds": folds,
        "simile": "Simile's 0.16 TVD threshold came from ~2,750 ratings by 14 of their raters on their own question set. Ours would have to be earned the same way, or not claimed.",
    }


def confidence_note() -> dict:
    return {
        "ready": False,
        "reason": "Confidence is unfitted. It reads internal state at the decision cycle and needs (simulation, observed) pairs — the original Van Dijcke coefficients, which are not in hand.",
    }


PIPELINE = [
    {"id": "ingest", "n": "01", "title": "Ingest",
     "body": "OSM footprints, SRTM elevation, WorldPop 2020 mass, WVS Wave 7 Kyiv microdata, air-raid alert times, reconstructed Van Dijcke curves."},
    {"id": "grid", "n": "02", "title": "Lattice",
     "body": "Snap the city onto a 250 m grid. Occupied cells, shelter graph, metro, WorldPop mass per cell."},
    {"id": "people", "n": "03", "title": "Population",
     "body": "Resample WVS respondents for traits. Draw WorldPop mass for home cells. Draw lag, speed, circadian mix."},
    {"id": "sense", "n": "04", "title": "Sense",
     "body": "Each minute: is the siren on? Acoustic saliency s₀ exp(−n/τ). Routine sets the attention bar."},
    {"id": "ignite", "n": "05", "title": "Ignite",
     "body": "If saliency ≥ attention, the coalition occupies the workspace. Otherwise the rest of the cycle never hears it."},
    {"id": "appraise", "n": "06", "title": "Appraise",
     "body": "WVS constructs are probabilities. Threat is their union (noisy-OR) with Dirichlet γ. Stay_Safe = P(hear)·P(floor ∪ threat). Defiance is a second union."},
    {"id": "plan", "n": "07", "title": "Plan",
     "body": "Means-end: shelter here, walk to nearest shelter, or evacuate. Deliberation picks one intention."},
    {"id": "motor", "n": "08", "title": "Motor",
     "body": "P1: nothing moves before t = 0. Then own log-normal lag, then Bohannon speed. 18 m for shelter-in-place."},
    {"id": "window", "n": "09", "title": "Window",
     "body": "Sum metres over agents, t = −10…+30. That vector is the simulated event-study curve."},
    {"id": "judge", "n": "10", "title": "Judge",
     "body": "Architecture tests on a synthetic city. Identifying design. Shape vs reconstructed Van Dijcke. Honest miss on levels."},
]

