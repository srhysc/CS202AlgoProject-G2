import os
import re
from dataclasses import dataclass
from itertools import combinations


@dataclass
class Activity:
    id: int
    duration: int
    resource_demands: list[int]
    successors: list[int]


@dataclass
class RCPSPInstance:
    num_activities: int
    num_resources: int
    resource_capacities: list[int]
    activities: dict[int, Activity]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _break_cycles(activities: dict) -> None:
    """
    Remove back edges until the precedence graph is a DAG.
    Uses Kahn's algorithm: any node that can't be topologically sorted
    is part of a cycle. We strip all incoming arcs to those stuck nodes
    and repeat until no cycles remain.
    """
    from collections import deque
    while True:
        in_degree = {aid: 0 for aid in activities}
        for act in activities.values():
            for s in act.successors:
                if s in in_degree:
                    in_degree[s] += 1

        queue = deque(aid for aid, d in in_degree.items() if d == 0)
        visited = set()
        while queue:
            node = queue.popleft()
            visited.add(node)
            for s in activities[node].successors:
                if s in in_degree:
                    in_degree[s] -= 1
                    if in_degree[s] == 0:
                        queue.append(s)

        stuck = {aid for aid in activities if aid not in visited}
        if not stuck:
            break  # DAG achieved

        # Break cycles by removing arcs from stuck nodes to other stuck nodes
        for act in activities.values():
            if act.id in stuck:
                act.successors = [s for s in act.successors if s not in stuck]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_psp(filepath: str) -> RCPSPInstance:
    """Parse a ProGenMax .SCH file into an RCPSPInstance."""
    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip()]

    def tokenize(line):
        return re.split(r'\s+', line)

    header = tokenize(lines[0])
    num_activities = int(header[0])
    num_resources = int(header[1])
    total_nodes = num_activities + 2  # includes dummy start (0) and end (n+1)

    activities = {}
    for i in range(1, total_nodes + 1):
        tokens = tokenize(lines[i])
        act_id = int(tokens[0])
        num_succs = int(tokens[2])
        raw_succs = [int(tokens[3 + j]) for j in range(num_succs)]
        lags = [int(tokens[3 + num_succs + j].strip('[]')) for j in range(num_succs)]
        # Keep only non-negative lag arcs (drop maximal-lag/backward arcs).
        successors = [s for s, lag in zip(raw_succs, lags) if lag >= 0]
        activities[act_id] = Activity(
            id=act_id, duration=0, resource_demands=[], successors=successors
        )

    for i in range(total_nodes + 1, 2 * total_nodes + 1):
        tokens = tokenize(lines[i])
        act_id = int(tokens[0])
        duration = int(tokens[2])
        demands = [int(tokens[3 + r]) for r in range(num_resources)]
        activities[act_id].duration = duration
        activities[act_id].resource_demands = demands

    # Break any remaining cycles caused by mutual positive-lag arcs
    # (ProGenMax cycle structures). Remove back edges until graph is a DAG.
    _break_cycles(activities)

    # After cycle-breaking, some activities may have lost all paths to the end.
    # Find them via backward BFS from end_id and add a direct arc to end.
    end_id = num_activities + 1
    preds_map = {aid: [] for aid in activities}
    for aid, act in activities.items():
        for s in act.successors:
            if s in preds_map:
                preds_map[s].append(aid)
    can_reach_end = set()
    stack = [end_id]
    while stack:
        node = stack.pop()
        if node in can_reach_end:
            continue
        can_reach_end.add(node)
        for pred in preds_map[node]:
            stack.append(pred)
    for aid, act in activities.items():
        if aid not in can_reach_end:
            act.successors.append(end_id)

    cap_tokens = tokenize(lines[2 * total_nodes + 1])
    capacities = [int(c) for c in cap_tokens[:num_resources]]

    return RCPSPInstance(
        num_activities=num_activities,
        num_resources=num_resources,
        resource_capacities=capacities,
        activities=activities,
    )


def load_all_instances(folder: str) -> list[RCPSPInstance]:
    """Load all .SCH files from a folder."""
    instances = []
    for fname in sorted(os.listdir(folder)):
        if fname.upper().endswith('.SCH'):
            path = os.path.join(folder, fname)
            try:
                instances.append(parse_psp(path))
            except Exception as e:
                print(f"  Warning: could not parse {fname}: {e}")
    return instances


# ---------------------------------------------------------------------------
# Auxiliary computations
# ---------------------------------------------------------------------------

def compute_longest_paths(instance: RCPSPInstance) -> dict[int, int]:
    """
    Longest path (time) from each activity to the project end, including its
    own duration. Computed via reverse topological DP.
    """
    acts = instance.activities

    # Kahn's algorithm for forward topological order
    in_degree = {aid: 0 for aid in acts}
    for act in acts.values():
        for succ in act.successors:
            if succ in in_degree:
                in_degree[succ] += 1

    queue = [aid for aid, deg in in_degree.items() if deg == 0]
    topo = []
    while queue:
        node = queue.pop()
        topo.append(node)
        for succ in acts[node].successors:
            if succ in in_degree:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

    # DP in reverse topological order (end → start)
    lp = {aid: acts[aid].duration for aid in acts}
    for aid in reversed(topo):
        for succ in acts[aid].successors:
            if succ in lp:
                lp[aid] = max(lp[aid], acts[aid].duration + lp[succ])

    return lp


def build_predecessors(instance: RCPSPInstance) -> dict[int, list[int]]:
    preds = {aid: [] for aid in instance.activities}
    for aid, act in instance.activities.items():
        for succ in act.successors:
            if succ in preds:
                preds[succ].append(aid)
    return preds


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def feasible_combos(
    eligible: list[int],
    resource_available: list[int],
    activities: dict[int, Activity],
    num_resources: int,
) -> list[tuple]:
    """
    All non-empty subsets of eligible activities whose combined resource
    demands fit within resource_available (multi-dimensional knapsack feasibility).
    """
    result = []
    for r in range(1, len(eligible) + 1):
        for combo in combinations(eligible, r):
            if all(
                sum(activities[a].resource_demands[k] for a in combo) <= resource_available[k]
                for k in range(num_resources)
            ):
                result.append(combo)
    return result


def solve(instance: RCPSPInstance, beam_width: int = 3) -> tuple[dict[int, int], int]:
    """
    Beam search over SSGS states.

    At each scheduling decision point t:
      1. Enumerate all feasible resource combinations of eligible activities
         (knapsack feasibility check).
      2. Rank combinations by total longest-path priority (critical-path first).
      3. Expand the top `beam_width` combinations as new beam states.
      4. Prune back to `beam_width` states by estimated remaining makespan.

    State representation:
      start_times  — dict {act_id: start_time}
      finished     — frozenset of activities whose finish time <= current t
      in_progress  — tuple of (finish_time, act_id) for started-but-not-done
      t            — current decision time
    """
    acts = instance.activities
    lp = compute_longest_paths(instance)
    preds = build_predecessors(instance)
    end_id = instance.num_activities + 1
    all_ids = set(acts.keys())

    # Dummy start (id=0) has duration 0 — immediately finished at t=0
    init = ({0: 0}, frozenset({0}), (), 0)
    beams = [init]
    best_makespan = float('inf')
    best_schedule = None

    while beams:
        next_beams = []

        for start_times, finished, in_progress, t in beams:

            # Terminal: dummy end has been finished
            if end_id in finished:
                ms = start_times[end_id]
                if ms < best_makespan:
                    best_makespan = ms
                    best_schedule = start_times
                continue

            # Activities that have been started (finished OR currently running)
            started = finished | frozenset(aid for _, aid in in_progress)

            # Eligible = not yet started, all predecessors finished
            eligible = [
                aid for aid in all_ids - started
                if all(p in finished for p in preds[aid])
            ]

            # Remaining resource capacity at time t
            avail = list(instance.resource_capacities)
            for _, aid in in_progress:
                for k in range(instance.num_resources):
                    avail[k] -= acts[aid].resource_demands[k]

            combos = feasible_combos(eligible, avail, acts, instance.num_resources) if eligible else []

            if not combos:
                # Nothing can be scheduled now — wait for next activity to finish
                if in_progress:
                    next_t = min(ft for ft, _ in in_progress)
                    done = frozenset(aid for ft, aid in in_progress if ft <= next_t)
                    rem = tuple((ft, aid) for ft, aid in in_progress if ft > next_t)
                    next_beams.append((start_times, finished | done, rem, next_t))
                continue

            # Rank combos: highest total longest-path first (most critical work first)
            combos.sort(key=lambda c: sum(lp.get(a, 0) for a in c), reverse=True)

            for combo in combos[:beam_width]:
                new_st = dict(start_times)
                new_ip = list(in_progress)
                new_fin = set(finished)

                for aid in combo:
                    new_st[aid] = t
                    finish_t = t + acts[aid].duration
                    if finish_t <= t:           # duration 0 → done immediately
                        new_fin.add(aid)
                    else:
                        new_ip.append((finish_t, aid))

                # Advance to next event (earliest finish time)
                if new_ip:
                    next_t = min(ft for ft, _ in new_ip)
                    done = frozenset(aid for ft, aid in new_ip if ft <= next_t)
                    rem = tuple((ft, aid) for ft, aid in new_ip if ft > next_t)
                    next_beams.append((new_st, frozenset(new_fin) | done, rem, next_t))
                else:
                    next_beams.append((new_st, frozenset(new_fin), (), t))

        # Prune: keep beam_width states with smallest estimated final makespan
        def estimate(state):
            _, fin, ip, t = state
            started = fin | frozenset(aid for _, aid in ip)
            unstarted = all_ids - started
            candidates = unstarted | frozenset(aid for _, aid in ip)
            return t + max((lp.get(aid, 0) for aid in candidates), default=0)

        next_beams.sort(key=estimate)
        beams = next_beams[:beam_width]

    return best_schedule, best_makespan


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate(instance: RCPSPInstance, schedule: dict[int, int]) -> list[str]:
    """Return a list of constraint violations (empty = valid schedule)."""
    acts = instance.activities
    preds = build_predecessors(instance)
    errors = []

    for aid, act in acts.items():
        if aid not in schedule:
            errors.append(f"Act {aid} has no start time")
            continue
        s_i = schedule[aid]
        for pred in preds[aid]:
            if pred not in schedule:
                continue
            s_pred = schedule[pred]
            d_pred = acts[pred].duration
            if s_i < s_pred + d_pred:
                errors.append(
                    f"Precedence violated: act {pred} (finish={s_pred+d_pred}) "
                    f"-> act {aid} (start={s_i})"
                )

    event_times = sorted({s for s in schedule.values()} |
                         {schedule[a] + acts[a].duration for a in schedule})
    for t in event_times:
        usage = [0] * instance.num_resources
        for aid, act in acts.items():
            if aid not in schedule:
                continue
            s = schedule[aid]
            if s <= t < s + act.duration:
                for k in range(instance.num_resources):
                    usage[k] += act.resource_demands[k]
        for k in range(instance.num_resources):
            if usage[k] > instance.resource_capacities[k]:
                errors.append(
                    f"Resource {k} exceeded at t={t}: "
                    f"used {usage[k]} > capacity {instance.resource_capacities[k]}"
                )

    return errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ── Switch ──────────────────────────────────────────────────────────────────
TEST_ALL = True   # False = single instance with full detail
#                  # True  = all instances with summary stats
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    folder = os.path.join(os.path.dirname(__file__), "sm_j20")
    instances = load_all_instances(folder)
    print(f"Loaded {len(instances)} instances from '{folder}'")

    if not TEST_ALL:
        inst = instances[0]
        t0 = time.perf_counter()
        schedule, makespan = solve(inst, beam_width=3)
        elapsed = time.perf_counter() - t0
        errors = validate(inst, schedule)

        print(f"\nInstance 0  makespan = {makespan}  ({elapsed*1000:.1f} ms)")
        if errors:
            print("  INVALID SCHEDULE:")
            for e in errors:
                print(f"    {e}")
        else:
            lower_bound = compute_longest_paths(inst)[0]
            print(f"  Valid schedule")
            print(f"  Critical path lower bound = {lower_bound}")
            print(f"  Gap from lower bound      = {makespan - lower_bound}")

        print(f"\n{'Act':>4}  {'Start':>5}  {'Finish':>6}")
        for aid in sorted(schedule):
            act = inst.activities[aid]
            print(f"{aid:>4}  {schedule[aid]:>5}  {schedule[aid] + act.duration:>6}")

    else:
        makespans = []
        solve_times = []
        failed = 0
        invalid = 0
        total_t0 = time.perf_counter()
        for i, inst in enumerate(instances):
            print(f"Solving instance {i+1}...", end="\r")
            t0 = time.perf_counter()
            schedule, makespan = solve(inst, beam_width=3)
            elapsed = time.perf_counter() - t0
            solve_times.append(elapsed)
            if schedule is None:
                failed += 1
                print(f"Instance {i+1:3d}  FAILED   ({elapsed*1000:.1f} ms)")
                continue
            errors = validate(inst, schedule)
            if errors:
                invalid += 1
                print(f"Instance {i+1:3d}  INVALID  {errors[0]}")
            else:
                makespans.append(makespan)

        total_elapsed = time.perf_counter() - total_t0
        print(f"\n── Results over {len(instances)} instances ──")
        print(f"  Valid   : {len(makespans)}")
        print(f"  Invalid : {invalid}")
        print(f"  Failed  : {failed}")
        if makespans:
            print(f"  Makespan  avg={sum(makespans)/len(makespans):.1f}  "
                  f"min={min(makespans)}  max={max(makespans)}")
        print(f"\n── Solve times ──")
        print(f"  Total   : {total_elapsed:.2f}s")
        print(f"  Avg     : {sum(solve_times)/len(solve_times)*1000:.1f} ms")
        print(f"  Max     : {max(solve_times)*1000:.1f} ms  (instance {solve_times.index(max(solve_times))+1})")
        print(f"  Min     : {min(solve_times)*1000:.1f} ms")
