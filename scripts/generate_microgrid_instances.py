#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic microgrid scheduling MILP instances."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/microgrid_lp"))
    parser.add_argument("--instances", type=int, default=8, help="Number of days to generate.")
    parser.add_argument("--time-steps", type=int, default=24, help="Time periods per day.")
    parser.add_argument("--generators", type=int, default=3, help="Number of thermal generators.")
    parser.add_argument("--evs", type=int, default=3, help="Number of EV fleets.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(1, args.instances + 1):
        instance_id = f"day_{idx:03d}"
        demand, pv, price = _sample_profiles(args.time_steps)
        import_price = [max(15.0, p + random.uniform(-2.0, 3.0)) for p in price]
        export_price = [6.0 for _ in range(args.time_steps)]
        lp_path = args.output_dir / f"{instance_id}.lp"
        content = build_lp_model(
            instance_id=instance_id,
            demand=demand,
            pv=pv,
            base_gen_cost=price,
            import_price=import_price,
            export_price=export_price,
            battery_capacity=10.0,
            battery_initial=5.0,
            battery_terminal=4.0,
            num_generators=args.generators,
            num_evs=args.evs,
        )
        lp_path.write_text(content, encoding="utf-8")


def _sample_profiles(time_steps: int) -> Tuple[List[float], List[float], List[float]]:
    demand_profile = []
    pv_profile = []
    gen_cost = []
    base_demand = random.uniform(3.0, 4.5)
    for t in range(time_steps):
        demand = base_demand + random.uniform(-0.8, 0.8) + 0.5 * _peak(t, time_steps)
        demand_profile.append(round(max(1.5, demand), 3))
        pv = max(0.0, 4.0 * _solar_shape(t, time_steps) + random.uniform(-0.3, 0.3))
        pv_profile.append(round(pv, 3))
        cost = 28.0 + 6.0 * _peak(t, time_steps) + random.uniform(-1.5, 1.5)
        gen_cost.append(round(cost, 3))
    return demand_profile, pv_profile, gen_cost


def _peak(t: int, n: int) -> float:
    mid = n / 2.0
    return max(0.0, 1.0 - abs(t - mid) / mid)


def _solar_shape(t: int, n: int) -> float:
    if t < n * 0.2 or t > n * 0.8:
        return 0.0
    center = n * 0.5
    width = n * 0.2
    return max(0.0, 1.0 - ((t - center) ** 2) / (width**2))


def build_lp_model(
    instance_id: str,
    demand: List[float],
    pv: List[float],
    base_gen_cost: List[float],
    import_price: List[float],
    export_price: List[float],
    battery_capacity: float,
    battery_initial: float,
    battery_terminal: float,
    num_generators: int,
    num_evs: int,
) -> str:
    T = len(demand)
    eta_charge = 0.92
    eta_discharge = 0.9
    charge_limit = 4.0
    discharge_limit = 4.0
    import_limit = 10.0
    export_limit = 6.0
    shed_penalty = 150.0
    ev_charge_limit = 3.0

    def var(name: str, *ids: int) -> str:
        return f"{name}_{'_'.join(str(i) for i in ids)}"

    # Generator fleet characteristics
    generators = []
    for g in range(num_generators):
        capacity = random.uniform(5.0, 12.0)
        min_output = capacity * random.uniform(0.15, 0.35)
        ramp = capacity * random.uniform(0.25, 0.5)
        startup_cost = random.uniform(40.0, 90.0)
        shutdown_cost = random.uniform(25.0, 60.0)
        min_up = random.randint(4, 10)
        min_down = random.randint(3, 7)
        initial_on = 1 if random.random() < 0.4 else 0
        cost_profile = [
            base_gen_cost[t] * random.uniform(0.9, 1.2) + g * 3.5 for t in range(T)
        ]
        generators.append(
            {
                "cap": round(capacity, 3),
                "min": round(min_output, 3),
                "ramp": round(ramp, 3),
                "startup": round(startup_cost, 3),
                "shutdown": round(shutdown_cost, 3),
                "min_up": min_up,
                "min_down": min_down,
                "initial": initial_on,
                "cost": [round(c, 3) for c in cost_profile],
            }
        )

    # EV fleets (arrival window and energy demand)
    evs = []
    for k in range(num_evs):
        arrival = random.randint(0, max(0, T // 2 - 2))
        departure = random.randint(arrival + 3, T - 1)
        energy_need = random.uniform(6.0, 12.0)
        efficiency = random.uniform(0.88, 0.94)
        evs.append(
            {
                "arrival": arrival,
                "departure": departure,
                "energy": round(energy_need, 3),
                "eff": round(efficiency, 3),
            }
        )

    unit_requirements = [
        random.uniform(max(1, num_generators // 2 - 0.5), max(1, num_generators // 2 + 2.0))
        for _ in range(T)
    ]
    reserve_multiplier = random.uniform(1.1, 1.3)

    lines: List[str] = []
    lines.append(f"\\* Synthetic microgrid MILP: {instance_id} *\\")
    lines.append("Minimize")
    obj_terms: List[str] = []
    for t in range(T):
        for g, gen in enumerate(generators):
            obj_terms.append(f"{gen['cost'][t]:.3f} {var('g', g, t)}")
            obj_terms.append(f"{gen['startup']:.3f} {var('u', g, t)}")
            obj_terms.append(f"{gen['shutdown']:.3f} {var('d', g, t)}")
        obj_terms.append(f"{import_price[t]:.3f} imp_{t}")
        obj_terms.append(f"4.000 chg_{t}")
        obj_terms.append(f"4.500 dis_{t}")
        obj_terms.append(f"{export_price[t]:.3f} exp_{t}")
        obj_terms.append(f"{shed_penalty:.3f} shed_{t}")
        for k, ev in enumerate(evs):
            obj_terms.append(f"2.000 {var('ev', k, t)}")
    lines.append(" obj: " + " + ".join(obj_terms))

    lines.append("Subject To")
    # Power balance
    for t in range(T):
        net_demand = demand[t] - pv[t]
        supply_terms = " + ".join(var("g", g, t) for g in range(num_generators))
        lhs_parts = [
            supply_terms if supply_terms else "0",
            f"+ dis_{t}",
            f"+ imp_{t}",
            f"+ shed_{t}",
            f"- chg_{t}",
            f"- exp_{t}",
        ]
        for k in range(num_evs):
            lhs_parts.append(f"- {var('ev', k, t)}")
        lhs = " ".join(lhs_parts)
        lines.append(f" balance_{t}: {lhs} = {net_demand:.3f}")

    # Generator constraints
    for g, gen in enumerate(generators):
        for t in range(T):
            lines.append(
                f" gen_cap_{g}_{t}: {var('g', g, t)} - {gen['cap']:.3f} {var('y', g, t)} <= 0"
            )
            lines.append(
                f" gen_min_{g}_{t}: {gen['min']:.3f} {var('y', g, t)} - {var('g', g, t)} <= 0"
            )
            if t > 0:
                lines.append(
                    f" commit_{g}_{t}: {var('y', g, t)} - {var('y', g, t-1)} - {var('u', g, t)} + {var('d', g, t)} = 0"
                )
                lines.append(
                    f" ramp_up_{g}_{t}: {var('g', g, t)} - {var('g', g, t-1)} <= {gen['ramp']:.3f}"
                )
                lines.append(
                    f" ramp_dn_{g}_{t}: {var('g', g, t-1)} - {var('g', g, t)} <= {gen['ramp']:.3f}"
                )
            else:
                lines.append(
                    f" initial_commit_{g}: {var('y', g, 0)} - {var('u', g, 0)} + {var('d', g, 0)} = {gen['initial']}"
                )
            # Minimum up-time window
            up_start = max(0, t - gen["min_up"] + 1)
            up_terms = [var("u", g, tau) for tau in range(up_start, t + 1)]
            if up_terms:
                up_expr = " + ".join(up_terms)
                lines.append(
                    f" min_up_{g}_{t}: {up_expr} - {var('y', g, t)} <= 0"
                )
            # Minimum down-time window
            down_start = max(0, t - gen["min_down"] + 1)
            down_terms = [var("d", g, tau) for tau in range(down_start, t + 1)]
            if down_terms:
                down_expr = " + ".join(down_terms)
                lines.append(
                    f" min_down_{g}_{t}: {down_expr} + {var('y', g, t)} <= 1"
                )

    # Crew requirements and reserve capacity
    for t in range(T):
        crew_expr = " + ".join(var("y", g, t) for g in range(num_generators))
        lines.append(f" crew_req_{t}: {crew_expr} >= {unit_requirements[t]:.3f}")

        reserve_expr = " + ".join(
            f"{generators[g]['cap']:.3f} {var('y', g, t)}" for g in range(num_generators)
        )
        reserve_rhs = demand[t] * reserve_multiplier
        lines.append(f" reserve_{t}: {reserve_expr} >= {reserve_rhs:.3f}")

    # Battery dynamics
    lines.append(f" soc_start: soc_0 = {battery_initial:.3f}")
    for t in range(T - 1):
        lines.append(
            f" storage_{t}: soc_{t+1} - soc_{t} - {eta_charge:.3f} chg_{t} + {1.0 / eta_discharge:.3f} dis_{t} = 0"
        )
    # Terminal requirement
    lines.append(f" soc_terminal: soc_{T-1} >= {battery_terminal:.3f}")

    # Variable bounds via inequalities
    for t in range(T):
        lines.append(f" charge_cap_{t}: chg_{t} <= {charge_limit:.3f}")
        lines.append(f" discharge_cap_{t}: dis_{t} <= {discharge_limit:.3f}")
        lines.append(f" import_cap_{t}: imp_{t} <= {import_limit:.3f}")
        lines.append(f" export_cap_{t}: exp_{t} <= {export_limit:.3f}")
        lines.append(f" soc_upper_{t}: soc_{t} <= {battery_capacity:.3f}")
        lines.append(f" shed_cap_{t}: shed_{t} <= 5.000")
        for k, ev in enumerate(evs):
            indicator = 1.0 if ev["arrival"] <= t <= ev["departure"] else 0.0
            limit = ev_charge_limit if indicator else 0.0
            lines.append(f" ev_cap_{k}_{t}: {var('ev', k, t)} <= {limit:.3f}")

    # EV energy requirements
    for k, ev in enumerate(evs):
        lhs = " + ".join(f"{ev['eff']:.3f} {var('ev', k, t)}" for t in range(ev["arrival"], ev["departure"] + 1))
        lines.append(f" ev_energy_{k}: {lhs} >= {ev['energy']:.3f}")

    lines.append("Bounds")
    for t in range(T):
        for g, gen in enumerate(generators):
            lines.append(f" 0 <= {var('g', g, t)} <= {gen['cap']:.3f}")
        lines.append(f" 0 <= chg_{t} <= {charge_limit:.3f}")
        lines.append(f" 0 <= dis_{t} <= {discharge_limit:.3f}")
        lines.append(f" 0 <= imp_{t} <= {import_limit:.3f}")
        lines.append(f" 0 <= exp_{t} <= {export_limit:.3f}")
        lines.append(f" 0 <= shed_{t} <= 5.000")
        lines.append(f" 0 <= soc_{t} <= {battery_capacity:.3f}")
        for k in range(num_evs):
            lines.append(f" 0 <= {var('ev', k, t)} <= {ev_charge_limit:.3f}")

    lines.append("Binary")
    binaries: List[str] = []
    for g in range(num_generators):
        for t in range(T):
            binaries.append(var("y", g, t))
            binaries.append(var("u", g, t))
            binaries.append(var("d", g, t))
    lines.append(" " + " ".join(binaries))

    lines.append("End")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
