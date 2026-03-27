import os
import re
from dataclasses import dataclass


@dataclass
class Activity:
    id: int
    duration: int
    resource_demands: list[int]  # demand per resource type
    successors: list[int]        # successor activity ids


@dataclass
class RCPSPInstance:
    num_activities: int          # excludes dummy start/end
    num_resources: int
    resource_capacities: list[int]
    activities: dict[int, Activity]  # keyed by activity id


def parse_psp(filepath: str) -> RCPSPInstance:
    """Parse a ProGenMax .SCH file into an RCPSPInstance."""
    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip()]

    def tokenize(line: str) -> list[str]:
        return re.split(r'\s+', line)

    # --- Header ---
    header = tokenize(lines[0])
    num_activities = int(header[0])
    num_resources = int(header[1])
    total_nodes = num_activities + 2  # includes dummy start (0) and end

    # --- Precedence section (lines 1 .. total_nodes) ---
    activities: dict[int, Activity] = {}
    for i in range(1, total_nodes + 1):
        tokens = tokenize(lines[i])
        act_id = int(tokens[0])
        # tokens[1] = mode (always 1 for single-mode RCPSP)
        num_succs = int(tokens[2])
        successors = [int(tokens[3 + j]) for j in range(num_succs)]
        activities[act_id] = Activity(
            id=act_id,
            duration=0,          # filled in resource section
            resource_demands=[],  # filled in resource section
            successors=successors,
        )

    # --- Resource demand section (lines total_nodes+1 .. 2*total_nodes) ---
    for i in range(total_nodes + 1, 2 * total_nodes + 1):
        tokens = tokenize(lines[i])
        act_id = int(tokens[0])
        duration = int(tokens[2])
        demands = [int(tokens[3 + r]) for r in range(num_resources)]
        activities[act_id].duration = duration
        activities[act_id].resource_demands = demands

    # --- Resource capacities (last line) ---
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


if __name__ == "__main__":
    folder = os.path.join(os.path.dirname(__file__), "sm_j10")
    instances = load_all_instances(folder)
    print(f"Loaded {len(instances)} instances from {folder}")

    # Quick sanity-check on first instance
    inst = instances[0]
    print(f"\nInstance 0:")
    print(f"  Activities : {inst.num_activities}")
    print(f"  Resources  : {inst.num_resources}")
    print(f"  Capacities : {inst.resource_capacities}")
    for act_id, act in sorted(inst.activities.items()):
        print(f"  Act {act_id:2d}  dur={act.duration}  "
              f"demands={act.resource_demands}  "
              f"succs={act.successors}")
