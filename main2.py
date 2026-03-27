"""
RCPSP Solver - Bidirectional Alternating Scheduler
===================================================
Approach:
  1. Parse PSPLIB format
  2. Build DAG, compute EST/LST/slack via forward+backward passes
  3. Two players alternate placing tasks (forward from source, backward from sink)
  4. Left-shift squash compaction to tighten the final schedule
  5. Compare against greedy forward SGS as baseline

Other approaches kept in mind (can swap in):
  - Beam Search (top-K candidates at each step)
  - ACO (pheromone-based probabilistic selection)
  - Slack-based repair heuristic
  - Reverse SGS as second opinion
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional
import sys
import os


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Task:
    id: int
    duration: int
    resources: list[int]        # resource demand per resource type
    successors: list[int] = field(default_factory=list)
    predecessors: list[int] = field(default_factory=list)

    # Computed by critical path analysis
    est: int = 0                # Earliest Start Time
    lst: int = 0                # Latest Start Time
    eft: int = 0                # Earliest Finish Time
    lft: int = 0                # Latest Finish Time
    slack: int = 0              # LST - EST
    cp_remaining: int = 0       # Critical path length from this task to sink


@dataclass
class Schedule:
    start_times: dict[int, int] = field(default_factory=dict)
    
    def makespan(self, tasks: dict[int, Task]) -> int:
        if not self.start_times:
            return 0
        return max(self.start_times[t] + tasks[t].duration for t in self.start_times)

    def finish_time(self, task_id: int, tasks: dict[int, Task]) -> int:
        return self.start_times[task_id] + tasks[task_id].duration


# ─────────────────────────────────────────────
# PSPLIB Parser
# ─────────────────────────────────────────────

def parse_psplib(filepath: str) -> tuple[dict[int, Task], list[int], int]:
    """
    Parse the ProGenMax / PSPLIB .SCH format.

    File structure:
      Line 1:   n  K  ...          (n = real activities, K = resource types)

      Precedence section (n+2 lines, one per activity 0..n+1):
        activity_id  mode  n_successors  succ1 succ2 ...  [lag1] [lag2] ...
        (time lags in brackets are ignored — not needed for basic RCPSP)

      Resource section (n+2 lines, one per activity 0..n+1):
        activity_id  mode  duration  r1  r2  ...  rK

      Last line:  cap_r1  cap_r2  ...  cap_rK
    """
    with open(filepath) as f:
        raw_lines = [l.strip() for l in f.readlines()]

    # Drop empty lines and trailing whitespace
    lines = [l for l in raw_lines if l]

    # ── Line 1: header ──
    header = lines[0].split()
    n_real = int(header[0])   # real activities (excluding 2 dummies)
    n_resources = int(header[1])
    n_activities = n_real + 2  # include dummy source (0) and dummy sink (n+1)

    # ── Precedence section: lines 1 .. n_activities ──
    succ_data: dict[int, list[int]] = {}
    for i in range(1, 1 + n_activities):
        # Strip bracket tokens like [0], [9], [-22] — they're time lags, ignore them
        tokens = [t for t in lines[i].split() if not t.startswith("[")]
        task_id    = int(tokens[0])
        # tokens[1] = mode (always 1, skip)
        n_succ     = int(tokens[2])
        successors = [int(tokens[3 + j]) for j in range(n_succ)]
        succ_data[task_id] = successors

    # ── Resource section: lines n_activities+1 .. 2*n_activities ──
    duration_data:  dict[int, int]       = {}
    resource_data:  dict[int, list[int]] = {}
    base = 1 + n_activities
    for i in range(base, base + n_activities):
        tokens     = lines[i].split()
        task_id    = int(tokens[0])
        # tokens[1] = mode (skip)
        duration   = int(tokens[2])
        resources  = [int(tokens[3 + k]) for k in range(n_resources)]
        duration_data[task_id] = duration
        resource_data[task_id] = resources

    # ── Last line: resource capacities ──
    cap_line = lines[base + n_activities]
    resource_capacities = [int(x) for x in cap_line.split()]

    # ── Build Task objects ──
    tasks: dict[int, Task] = {}
    for task_id in succ_data:
        tasks[task_id] = Task(
            id=task_id,
            duration=duration_data.get(task_id, 0),
            resources=resource_data.get(task_id, [0] * n_resources),
            successors=succ_data[task_id],
        )

    # Break any cycles (e.g. from maximal time lag instances)
    _break_cycles(tasks)

    # Wire predecessors (after cycles broken so predecessors are clean)
    for task_id, task in tasks.items():
        for s in task.successors:
            if s in tasks:
                tasks[s].predecessors.append(task_id)

    return tasks, resource_capacities, n_real


def _break_cycles(activities: dict) -> None:
    """
    Remove back edges until the precedence graph is a DAG.
    Uses Kahn's algorithm: any node that can't be topologically sorted
    is part of a cycle. We strip all incoming arcs to those stuck nodes
    and repeat until no cycles remain.
    """
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


# ─────────────────────────────────────────────
# DAG Analysis — Forward + Backward Passes
# ─────────────────────────────────────────────

def topological_sort(tasks: dict[int, Task]) -> list[int]:
    in_degree = {t: len(tasks[t].predecessors) for t in tasks}
    queue = deque([t for t, d in in_degree.items() if d == 0])
    order = []
    while queue:
        t = queue.popleft()
        order.append(t)
        for s in tasks[t].successors:
            if s in tasks:
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(s)
    return order


def compute_critical_path(tasks: dict[int, Task]) -> int:
    """
    Forward pass  → EST, EFT
    Backward pass → LST, LFT, slack, cp_remaining
    Returns project makespan lower bound.
    """
    topo = topological_sort(tasks)

    # Forward pass
    for t in topo:
        task = tasks[t]
        task.est = max(
            (tasks[p].eft for p in task.predecessors if p in tasks),
            default=0
        )
        task.eft = task.est + task.duration

    makespan = max(tasks[t].eft for t in tasks)

    # Backward pass
    for t in topo:
        tasks[t].lft = makespan
        tasks[t].lst = makespan

    for t in reversed(topo):
        task = tasks[t]
        task.lft = min(
            (tasks[s].lst for s in task.successors if s in tasks),
            default=makespan
        )
        task.lst = task.lft - task.duration
        task.slack = task.lst - task.est

    # cp_remaining = longest path from task to sink
    for t in reversed(topo):
        task = tasks[t]
        task.cp_remaining = task.duration + max(
            (tasks[s].cp_remaining for s in task.successors if s in tasks),
            default=0
        )

    return makespan


# ─────────────────────────────────────────────
# Resource Feasibility Check
# ─────────────────────────────────────────────

def resource_usage_at(t: int, placed: dict[int, int], tasks: dict[int, Task], n_res: int) -> list[int]:
    usage = [0] * n_res
    for task_id, start in placed.items():
        task = tasks[task_id]
        if start <= t < start + task.duration:
            for k in range(n_res):
                usage[k] += task.resources[k]
    return usage


def can_place(task: Task, start: int, placed: dict[int, int],
              tasks: dict[int, Task], capacities: list[int]) -> bool:
    n_res = len(capacities)
    for t in range(start, start + task.duration):
        usage = resource_usage_at(t, placed, tasks, n_res)
        for k in range(n_res):
            if usage[k] + task.resources[k] > capacities[k]:
                return False
    return True


def earliest_feasible_start(task: Task, min_start: int, placed: dict[int, int],
                             tasks: dict[int, Task], capacities: list[int],
                             horizon: int = 1000) -> int:
    for t in range(min_start, horizon):
        if can_place(task, t, placed, tasks, capacities):
            return t
    return horizon


def latest_feasible_start(task: Task, max_start: int, placed: dict[int, int],
                           tasks: dict[int, Task], capacities: list[int]) -> int:
    """Finds latest start <= max_start where task fits resource-wise."""
    for t in range(max_start, -1, -1):
        if can_place(task, t, placed, tasks, capacities):
            return t
    return -1


# ─────────────────────────────────────────────
# Priority Functions
# ─────────────────────────────────────────────

def forward_priority(task: Task) -> tuple:
    """
    Higher = more urgent to place from the front.
    Primary:   critical path remaining (longest downstream chain)
    Secondary: resource demand (heavy tasks first)
    Tertiary:  number of successors (unblocks the most)
    """
    return (
        -task.cp_remaining,
        -sum(task.resources),
        -len(task.successors),
    )


def backward_priority(task: Task) -> tuple:
    """
    Higher = more urgent to place from the back.
    Primary:   latest start time (most time-pressured from sink)
    Secondary: resource demand
    Tertiary:  number of predecessors (most constrained)
    """
    return (
        task.lst,
        -sum(task.resources),
        -len(task.predecessors),
    )


# ─────────────────────────────────────────────
# Eligibility Checks
# ─────────────────────────────────────────────

def get_forward_eligible(remaining: set[int], placed: dict[int, int],
                          tasks: dict[int, Task]) -> list[int]:
    """Tasks whose ALL predecessors are placed."""
    return [
        t for t in remaining
        if all(p in placed for p in tasks[t].predecessors)
    ]


def get_backward_eligible(remaining: set[int], placed: dict[int, int],
                           tasks: dict[int, Task]) -> list[int]:
    """Tasks whose ALL successors are placed."""
    return [
        t for t in remaining
        if all(s in placed for s in tasks[t].successors)
    ]


# ─────────────────────────────────────────────
# Squash — Left-Shift Compaction
# ─────────────────────────────────────────────

def left_shift_squash(placed: dict[int, int], tasks: dict[int, Task],
                      capacities: list[int], horizon: int) -> dict[int, int]:
    """
    After bidirectional placement, slide every task as far left
    as possible while respecting precedence and resources.
    """
    topo = topological_sort(tasks)
    squashed = {}

    for task_id in topo:
        if task_id not in placed:
            continue
        task = tasks[task_id]

        # Earliest allowed by predecessors
        min_start = max(
            (squashed[p] + tasks[p].duration for p in task.predecessors if p in squashed),
            default=0
        )

        # Slide left until resource fits
        start = earliest_feasible_start(task, min_start, squashed, tasks, capacities, horizon)
        squashed[task_id] = start

    return squashed


# ─────────────────────────────────────────────
# Bidirectional Alternating Scheduler
# ─────────────────────────────────────────────

def bidirectional_schedule(tasks: dict[int, Task],
                            capacities: list[int]) -> Schedule:
    """
    Two players alternate placing tasks onto a shared board.

    Forward player : picks from tasks eligible from the source end
                     (all predecessors placed), places at earliest feasible time

    Backward player: picks from tasks eligible from the sink end
                     (all successors placed), places at latest feasible time

    Turn order: whichever player has fewer eligible tasks moves first
                (more constrained player gets priority)

    After both players exhaust their moves, squash left.
    """
    placed = {}       # task_id -> start_time
    remaining = set(tasks.keys())
    horizon = sum(tasks[t].duration for t in tasks) + 1

    turn = "forward"  # tiebreak: forward goes first

    while remaining:
        fwd_eligible = get_forward_eligible(remaining, placed, tasks)
        bwd_eligible = get_backward_eligible(remaining, placed, tasks)

        if not fwd_eligible and not bwd_eligible:
            # Deadlock — shouldn't happen with valid PSPLIB input
            print("WARNING: deadlock detected, forcing forward placement")
            break

        # More constrained player moves first
        if not fwd_eligible:
            turn = "backward"
        elif not bwd_eligible:
            turn = "forward"
        else:
            turn = "forward" if len(fwd_eligible) <= len(bwd_eligible) else "backward"

        if turn == "forward" and fwd_eligible:
            # Pick highest priority eligible task
            task_id = min(fwd_eligible, key=lambda t: forward_priority(tasks[t]))
            task = tasks[task_id]

            # Earliest start respecting predecessors
            min_start = max(
                (placed[p] + tasks[p].duration for p in task.predecessors if p in placed),
                default=0
            )
            start = earliest_feasible_start(task, min_start, placed, tasks, capacities, horizon)
            placed[task_id] = start
            remaining.remove(task_id)

        elif turn == "backward" and bwd_eligible:
            # Pick highest priority eligible task from sink end
            task_id = min(bwd_eligible, key=lambda t: backward_priority(tasks[t]))
            task = tasks[task_id]

            # Latest start respecting successors
            max_start = min(
                (placed[s] - task.duration for s in task.successors if s in placed),
                default=horizon - task.duration
            )
            start = latest_feasible_start(task, max_start, placed, tasks, capacities)

            if start < 0:
                # Can't fit late — fall back to earliest feasible
                min_start = max(
                    (placed[p] + tasks[p].duration for p in task.predecessors if p in placed),
                    default=0
                )
                start = earliest_feasible_start(task, min_start, placed, tasks, capacities, horizon)

            placed[task_id] = start
            remaining.remove(task_id)

        else:
            # One side exhausted, drain the other
            turn = "forward" if fwd_eligible else "backward"

    # Squash — compact left
    squashed = left_shift_squash(placed, tasks, capacities, horizon)
    return Schedule(start_times=squashed)


# ─────────────────────────────────────────────
# Baseline: Greedy Forward SGS
# ─────────────────────────────────────────────

def forward_sgs(tasks: dict[int, Task], capacities: list[int]) -> Schedule:
    """
    Standard Serial Schedule Generation Scheme.
    At each step: pick most critical eligible task, place at earliest feasible time.
    Baseline to compare bidirectional against.
    """
    placed = {}
    remaining = set(tasks.keys())
    horizon = sum(tasks[t].duration for t in tasks) + 1

    while remaining:
        eligible = get_forward_eligible(remaining, placed, tasks)
        if not eligible:
            break

        task_id = min(eligible, key=lambda t: forward_priority(tasks[t]))
        task = tasks[task_id]

        min_start = max(
            (placed[p] + tasks[p].duration for p in task.predecessors if p in placed),
            default=0
        )
        start = earliest_feasible_start(task, min_start, placed, tasks, capacities, horizon)
        placed[task_id] = start
        remaining.remove(task_id)

    return Schedule(start_times=placed)


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def validate_schedule(schedule: Schedule, tasks: dict[int, Task],
                      capacities: list[int]) -> list[str]:
    errors = []
    n_res = len(capacities)
    makespan = schedule.makespan(tasks)

    for task_id, start in schedule.start_times.items():
        task = tasks[task_id]

        # Precedence check
        for p in task.predecessors:
            if p in schedule.start_times:
                pred_finish = schedule.start_times[p] + tasks[p].duration
                if start < pred_finish:
                    errors.append(
                        f"Task {task_id} starts at {start} before predecessor "
                        f"{p} finishes at {pred_finish}"
                    )

    # Resource check at each time step
    for t in range(makespan):
        usage = resource_usage_at(t, schedule.start_times, tasks, n_res)
        for k in range(n_res):
            if usage[k] > capacities[k]:
                errors.append(
                    f"Resource {k} exceeded at t={t}: {usage[k]} > {capacities[k]}"
                )

    return errors


# ─────────────────────────────────────────────
# Pretty Print
# ─────────────────────────────────────────────

def print_schedule(schedule: Schedule, tasks: dict[int, Task],
                   capacities: list[int], label: str = "Schedule"):
    makespan = schedule.makespan(tasks)
    print(f"\n{'='*50}")
    print(f" {label}")
    print(f"{'='*50}")
    print(f" Makespan: {makespan}")
    print(f" Tasks: {len(schedule.start_times)}")
    print(f"\n {'Task':>6} {'Start':>6} {'Finish':>7} {'Slack':>6} {'Resources'}")
    print(f" {'-'*50}")

    for task_id in sorted(schedule.start_times):
        task = tasks[task_id]
        start = schedule.start_times[task_id]
        finish = start + task.duration
        print(
            f" {task_id:>6} {start:>6} {finish:>7} "
            f"{task.slack:>6}  {task.resources}"
        )

    errors = validate_schedule(schedule, tasks, capacities)
    if errors:
        print(f"\n ⚠ VALIDATION ERRORS ({len(errors)}):")
        for e in errors[:5]:
            print(f"   {e}")
    else:
        print(f"\n ✓ Schedule is feasible")


def print_resource_chart(schedule: Schedule, tasks: dict[int, Task],
                          capacities: list[int]):
    """ASCII resource usage chart over time."""
    makespan = schedule.makespan(tasks)
    n_res = len(capacities)
    print(f"\n Resource Usage Over Time (capacity: {capacities})")
    print(f" {'t':>4} ", end="")
    for k in range(n_res):
        print(f" R{k+1}({capacities[k]})", end="")
    print()
    print(f" {'-'*40}")
    for t in range(makespan):
        usage = resource_usage_at(t, schedule.start_times, tasks, n_res)
        print(f" {t:>4} ", end="")
        for k in range(n_res):
            bar = "█" * usage[k]
            over = usage[k] > capacities[k]
            print(f"  {usage[k]:>2}{'!' if over else ' '} {bar:<{capacities[k]}}", end="")
        print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def solve(filepath: str, verbose: bool = False) -> tuple[int, int, int, list, list]:
    """
    Solve one instance.
    Returns (cp_lower_bound, sgs_makespan, bidi_makespan, sgs_errors, bidi_errors).
    Set verbose=True to print the full schedule breakdown.
    """
    tasks, capacities, n_real = parse_psplib(filepath)
    cp = compute_critical_path(tasks)

    sgs_schedule = forward_sgs(tasks, capacities)
    sgs_makespan = sgs_schedule.makespan(tasks)
    sgs_errors = validate_schedule(sgs_schedule, tasks, capacities)

    bidi_schedule = bidirectional_schedule(tasks, capacities)
    bidi_makespan = bidi_schedule.makespan(tasks)
    bidi_errors = validate_schedule(bidi_schedule, tasks, capacities)

    if verbose:
        print(f"\nParsing: {filepath}")
        print(f"Tasks: {n_real} real + 2 dummy | Resources: {len(capacities)} | Capacities: {capacities}")
        print(f"Critical path lower bound: {cp}")
        print_schedule(sgs_schedule, tasks, capacities, "Baseline: Forward SGS")
        print_schedule(bidi_schedule, tasks, capacities, "Bidirectional Alternating + Squash")
        print(f"\n{'='*50}")
        print(f" COMPARISON")
        print(f"{'='*50}")
        print(f" Lower bound (CP):    {cp}")
        print(f" Forward SGS:         {sgs_makespan}  (gap: {sgs_makespan - cp})")
        print(f" Bidirectional:       {bidi_makespan}  (gap: {bidi_makespan - cp})")
        print(f" Improvement:         {sgs_makespan - bidi_makespan} time units")
        print_resource_chart(bidi_schedule, tasks, capacities)

    return cp, sgs_makespan, bidi_makespan, sgs_errors, bidi_errors


if __name__ == "__main__":
    import time

    folder = os.path.join(os.path.dirname(__file__), "sm_j10")
    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.upper().endswith(".SCH")
    )

    if not files:
        print(f"No .SCH files found in '{folder}'")
        sys.exit(1)

    print(f"Found {len(files)} instances in '{folder}'")
    print(f"\n{'#':>4}  {'File':<20}  {'CP':>5}  {'SGS':>5}  {'Bidi':>5}  {'Gap':>5}  {'Impr':>5}  {'Valid?'}")
    print("-" * 70)

    cps, sgs_ms, bidi_ms = [], [], []
    failed = 0
    invalid_bidi = 0
    invalid_sgs = 0

    t_total = time.perf_counter()
    for i, fp in enumerate(files, 1):
        fname = os.path.basename(fp)
        try:
            cp, sgs, bidi, sgs_errs, bidi_errs = solve(fp)
            cps.append(cp)
            sgs_ms.append(sgs)
            bidi_ms.append(bidi)
            if bidi_errs:
                invalid_bidi += 1
            if sgs_errs:
                invalid_sgs += 1
            validity = "OK" if not bidi_errs else f"INVALID({len(bidi_errs)})"
            print(f"{i:>4}  {fname:<20}  {cp:>5}  {sgs:>5}  {bidi:>5}  {bidi-cp:>5}  {sgs-bidi:>5}  {validity}")
            if bidi_errs:
                print(f"       {bidi_errs[0]}")
        except Exception as e:
            failed += 1
            print(f"{i:>4}  {fname:<20}  ERROR: {e}")

    elapsed = time.perf_counter() - t_total
    n = len(cps)
    if n:
        print(f"\n{'='*70}")
        print(f" Instances solved  : {n}  |  Failed: {failed}")
        print(f" Bidi invalid      : {invalid_bidi}  |  SGS invalid: {invalid_sgs}")
        print(f" Avg CP lb         : {sum(cps)/n:.1f}")
        print(f" Avg SGS makespan  : {sum(sgs_ms)/n:.1f}")
        print(f" Avg Bidi makespan : {sum(bidi_ms)/n:.1f}")
        print(f" Avg improvement   : {(sum(sgs_ms)-sum(bidi_ms))/n:.1f}")
        print(f" Total time        : {elapsed:.2f}s")