
import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

# ------------------------ Helpers ------------------------

def michaelis_menten(x: float, K: float) -> float:
    """Michaelis-Menten saturating function (x / (K + x)), natural range [0, 1)."""
    x = max(0.0, x)
    return x / (K + x + 1e-12)

def sigmoid(x: float) -> float:
    """Logistic function sigma(x) with clamp to prevent overflow."""
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))

@dataclass
class AgentState:
    gamma: float
    h: float
    C: int
    B: Optional[float]
    alive: bool = True
    age: int = 0
    death_cause: Optional[str] = None
    death_counted: bool = False
    e_int: float = 0.0
    e_prev: float = 0.0

class Agent:
    """Base agent with channels I = {gamma, h, C, B}."""
    def __init__(self, agent_type: str, params: Dict):
        assert agent_type in ("EV-02C", "EV-04")
        self.agent_type = agent_type
        A = params["appendix_A"]
        self.gamma_min = A["gamma_min"]
        self.gamma_safe = A["gamma_safe"]
        self.h_max = A["h_max"]
        self.B_min = A["B_min"]
        self.k_rep = None  # set per regime externally
        self.c_rep = A.get("c_rep", 1.0)
        self.MM_K_gamma = A["MM_K_gamma"]
        self.y_cat = A.get("y_cat", 0.8)
        self.y_ano = A.get("y_ano", 0.5)
        self.delta_B = A.get("delta_B", 0.02)
        self.eta = A.get("eta", 0.6)

        B0 = np.random.uniform(0.8, 1.0) if agent_type == "EV-04" else None
        self.s = AgentState(
            gamma=np.random.uniform(0.4, 0.6),
            h=np.random.uniform(0.0, 0.1),
            C=0,
            B=B0,
        )

    def compute_viability(self) -> float:
        """V = product of safety scores s_c in [0,1]."""
        s_gamma = max(0.0, min(1.0, (self.s.gamma - self.gamma_min) / (self.gamma_safe - self.gamma_min + 1e-12)))
        s_h     = max(0.0, min(1.0, 1.0 - (self.s.h / (self.h_max + 1e-12))))
        s_C     = 1.0 if self.s.C < self.C_max else 0.0
        if self.agent_type == "EV-04":
            s_B = max(0.0, min(1.0, (self.s.B - self.B_min) / (1.0 - self.B_min + 1e-12)))
        else:
            s_B = 1.0
        return s_gamma * s_h * s_C * s_B

    def check_death(self) -> Tuple[bool, Optional[str]]:
        """Check thresholds; return (dead, cause)."""
        if self.s.gamma <= self.gamma_min:
            return True, "gamma"
        if self.s.h >= self.h_max:
            return True, "h"
        if self.s.C >= self.C_max:
            return True, "C"
        if self.agent_type == "EV-04" and self.s.B is not None and self.s.B <= self.B_min:
            return True, "B"
        return False, None

    def set_regime_params(self, regime_cfg: Dict):
        """Set per-regime constants."""
        self.P_basal = regime_cfg["P_basal"]
        self.phi_min = regime_cfg["phi_min"]
        self.k_rep   = regime_cfg["k_rep"]
        self.C_max   = regime_cfg["C_max"]

        # ---- heterogeneidad individual ----------
        # un poco de variación en metabolismo y daño
        self.P_basal *= np.random.normal(1.0, 0.10)   # ±10%
        self.phi_min *= np.random.normal(1.0, 0.15)   # ±15%

        # y un poquito en la tolerancia al daño
        self.h_max   *= np.random.normal(1.0, 0.05)   # ±5%
        # -----------------------------------------


    def step(self, u_eff: float):
        """Advance internal dynamics by one tick given effective demand u_eff (after governor)."""
        if not self.s.alive:
            return
        
        assert self.k_rep is not None, "k_rep must be set via set_regime_params() before stepping"

        if self.agent_type == "EV-02C":
            psi = self.k_rep * michaelis_menten(self.s.gamma, self.MM_K_gamma)
            self.s.gamma = self.s.gamma + self.eta * u_eff - self.P_basal - self.c_rep * psi
            self.s.h     = self.s.h + self.phi_min - psi
            self.s.C    += 1
        else:  # EV-04
            # un poco más realista: si ya está dañado, le cuesta sostener catabolismo
            damage_factor = max(0.2, 1.0 - 0.4 * (self.s.h / (self.h_max + 1e-12)))
            cat = self.y_cat * michaelis_menten(u_eff, 0.3) * damage_factor

            ano = self.y_ano * michaelis_menten(self.s.gamma, self.MM_K_gamma)
            psi = self.k_rep  * michaelis_menten(self.s.gamma, self.MM_K_gamma)

            self.s.gamma = self.s.gamma + cat - self.P_basal - self.c_rep * psi
            self.s.B     = self.s.B + ano - self.delta_B * self.s.B
            self.s.h     = self.s.h + self.phi_min - psi
            self.s.C    += 1
	    
            
        self.s.gamma = max(0.0, min(1.5, self.s.gamma))
        self.s.h     = max(0.0, min(5.0, self.s.h))
        if self.agent_type == "EV-04" and self.s.B is not None:
            self.s.B = max(0.0, min(5.0, self.s.B))

        self.s.age  += 1
        dead, cause = self.check_death()
        if dead:
            self.s.alive = False
            self.s.death_cause = cause

class Environment:
    """Shared resource E_env with E_{t+1} = E_t + F_E - demand."""
    def __init__(self, regime_cfg: Dict, appx: Dict, step_at: Optional[int] = None, 
                 step_scale: float = 1.5, step_dur: int = 50):
        self.F_E   = regime_cfg["F_E"]
        self.E_cap = regime_cfg["E_cap"]
        
        # Support explicit E_env0; fallback to E0 if not present
        # If value in [0,1], treat as fraction of E_cap
        E0_raw = regime_cfg.get("E_env0", regime_cfg.get("E0", 0.5))
        self.E = E0_raw * self.E_cap if E0_raw <= 1.0 else float(E0_raw)
        
        self.theta_low  = appx["theta_low"]
        self.theta_high = appx["theta_high"]
        
        # Step perturbation parameters
        self.step_at = step_at
        self.step_scale = step_scale
        self.step_dur = step_dur
    
    def _F_E_current(self, t: int) -> float:
        """Get current F_E value, applying step perturbation if active."""
        if self.step_at is None:
            return self.F_E
        if self.step_at <= t < self.step_at + self.step_dur:
            return self.F_E * self.step_scale
        return self.F_E

    def normalize_E(self) -> float:
        return max(0.0, min(1.0, self.E / (self.E_cap + 1e-12)))

    def governor(self) -> float:
        En = self.normalize_E()
        if En < self.theta_low:
            return 0.2
        elif En > self.theta_high:
            return 1.0
        else:
            return 0.2 + 0.8 * (En - self.theta_low) / (self.theta_high - self.theta_low + 1e-12)

    def step(self, total_demand: float, t: int) -> Tuple[float, float]:
        """Apply mass balance and return (E_env, G_env)."""
        G = self.governor()
        eff_demand = total_demand * G
        self.E = self.E + self._F_E_current(t) - eff_demand
        self.E = max(0.0, min(self.E_cap, self.E))
        return self.E, G

class Policy:
    def decide(self, agent: Agent, E_env: float) -> float:
        raise NotImplementedError

class PolicyVOFF(Policy):
    def __init__(self, u0: float):
        self.u0 = u0
    def decide(self, agent: Agent, E_env: float) -> float:
        return self.u0

class PolicyAPPFixed(Policy):
    def __init__(self, beta_g: float, g_min: float):
        self.beta_g = beta_g
        self.g_min = g_min
    def decide(self, agent: Agent, E_env: float) -> float:
        e = agent.gamma_safe - agent.s.gamma
        return max(self.g_min, sigmoid(self.beta_g * e))

class Simulation:
    def __init__(self, cfg: Dict, regime: str, policy_name: str, architecture: str,
                 n_agents: int, T: int, seed: int, outputs_dir: str,
                 step_at: Optional[int] = None, step_scale: float = 1.5, step_dur: int = 50):
        self.cfg = cfg
        self.regime = regime
        self.policy_name = policy_name
        self.architecture = architecture
        self.n_agents = n_agents
        self.T = T
        self.seed = seed
        self.outputs_dir = outputs_dir
        
        # Step perturbation params
        self.step_at = step_at
        self.step_scale = step_scale
        self.step_dur = step_dur

        np.random.seed(seed)
        random.seed(seed)

        self.regime_cfg = cfg["regimes"][regime]
        self.appx = cfg["appendix_A"]
        self.tail_w = cfg["general"].get("tail_window", 60)

        self.agents: List[Agent] = [Agent(architecture, cfg) for _ in range(n_agents)]
        for a in self.agents:
            a.set_regime_params(self.regime_cfg)

        self.env = Environment(self.regime_cfg, self.appx, 
                               step_at=self.step_at, step_scale=self.step_scale, step_dur=self.step_dur)

        if policy_name == "VOFF":
            u0 = cfg["policy_params"]["VOFF"]["u0"][regime]
            self.policy = PolicyVOFF(u0)
        elif policy_name == "APP-Fixed":
            self.policy = PolicyAPPFixed(self.appx["beta_g"], self.appx["g_min"])
        else:
            raise ValueError(f"Unsupported policy: {policy_name}")

        self.tick_rows: List[Dict] = []
        self.death_counts = {"gamma": 0, "h": 0, "C": 0, "B": 0}
        self.lifespans: List[int] = []
        
        # Per-agent tracking for AIM-4
        self.export_agents = self.cfg["general"].get("export_agents", False)
        self.agent_rows: Dict[int, List[Dict]] = {i: [] for i in range(n_agents)} if self.export_agents else {}

    def step(self, t: int):
        # Warm-start ramp: gradual metabolic induction
        warmup_ticks = self.cfg["general"].get("warmup_ticks", 30)
        ramp = min(1.0, t / float(warmup_ticks)) if warmup_ticks > 0 else 1.0
        
        demands = np.zeros(self.n_agents, dtype=float)
        for i, a in enumerate(self.agents):
            if a.s.alive:
                demands[i] = self.policy.decide(a, self.env.E) * ramp
            else:
                demands[i] = 0.0
        total_demand = float(np.sum(demands))

        E_env, G_env = self.env.step(total_demand, t)

        for i, a in enumerate(self.agents):
            if a.s.alive:
                a.step(demands[i] * G_env)
        
        # Record per-agent data if enabled
        if self.export_agents:
            for i, a in enumerate(self.agents):
                if a.s.alive:
                    row = {
                        "tick": t,
                        "gamma": a.s.gamma,
                        "h": a.s.h,
                        "C": a.s.C,
                        "u": demands[i],
                        "alive": 1
                    }
                    if a.agent_type == "EV-04":
                        row["B"] = a.s.B
                    self.agent_rows[i].append(row)

        alive_mask = np.array([a.s.alive for a in self.agents])
        n_alive = int(np.sum(alive_mask))
        mean_gamma = float(np.mean([a.s.gamma for a in self.agents]))
        mean_h     = float(np.mean([a.s.h for a in self.agents]))
        mean_C     = float(np.mean([a.s.C for a in self.agents]))
        
        B_values = [a.s.B for a in self.agents if a.agent_type == "EV-04" and a.s.B is not None]
        mean_B = float(np.mean(B_values)) if B_values else float("nan")

        row = dict(seed=self.seed, regime=self.regime, policy=self.policy_name, architecture=self.architecture,
                   tick=t, E_env=E_env, G_env=G_env, n_alive=n_alive, mean_gamma=mean_gamma,
                   mean_h=mean_h, mean_C=mean_C, mean_B=mean_B, total_demand=total_demand)
        self.tick_rows.append(row)

        for a in self.agents:
            if a.s.death_cause is not None and not a.s.death_counted:
                cause = a.s.death_cause
                if cause in self.death_counts:
                    self.death_counts[cause] += 1
                a.s.death_counted = True

    def run(self) -> Tuple[pd.DataFrame, Dict, Dict[int, pd.DataFrame]]:
        for t in range(self.T):
            self.step(t)

        for a in self.agents:
            self.lifespans.append(a.s.age)

        df_ticks = pd.DataFrame(self.tick_rows)
        
        # Convert agent rows to dataframes
        df_agents = {}
        if self.export_agents:
            for i, rows in self.agent_rows.items():
                if len(rows) > 0:
                    df_agents[i] = pd.DataFrame(rows)

        # Find last tick with alive agents
        alive_ticks = df_ticks[df_ticks["n_alive"] > 0]
        if len(alive_ticks) > 0:
            last_alive_tick = int(alive_ticks["tick"].max())
            # Use tail window, but only up to last alive tick
            tail_start = max(0, min(last_alive_tick - self.tail_w, self.T - self.tail_w))
            tail_end = last_alive_tick + 1
            tail = df_ticks[(df_ticks["tick"] >= tail_start) & (df_ticks["tick"] < tail_end)]
        else:
            # Fallback to standard tail if no one survives
            tail_start = max(0, self.T - self.tail_w)
            tail = df_ticks[df_ticks["tick"] >= tail_start]
        
        E = tail["E_env"].values
        dE = np.diff(E) if len(E) > 1 else np.array([0.0])

        var_ps = float(np.var(E)) if len(E) > 1 else 0.0
        p2p_ps = float(np.max(E) - np.min(E)) if len(E) > 0 else 0.0
        rugosity = float(np.var(dE)) if len(dE) > 1 else 0.0
        
        # Coefficient of Variation
        mu_E = float(np.mean(E)) if len(E) > 0 else 0.0
        sd_E = float(np.std(E)) if len(E) > 1 else 0.0
        cv_ps = (sd_E / mu_E) if mu_E > 1e-12 else float('inf')
        
        # Shannon entropy of death causes
        total_deaths = sum(self.death_counts.values())
        H_causes = 0.0
        if total_deaths > 0:
            for cause in ("gamma", "h", "C", "B"):
                p = self.death_counts.get(cause, 0) / total_deaths
                if p > 0:
                    H_causes -= p * math.log(p + 1e-12)
        
        # Convergence tick
        convergence_tick = -1
        if len(E) > 1:
            E_range = np.max(E) - np.min(E)
            threshold = 0.05 * E_range if E_range > 1e-12 else 0.01
            E_ss = float(np.median(E))
            for i, e_val in enumerate(df_ticks["E_env"].values):
                if abs(e_val - E_ss) < threshold:
                    convergence_tick = int(i)
                    break

        n_alive_final = int(df_ticks.iloc[-1]["n_alive"]) if len(df_ticks) else self.n_agents
        survival_rate = n_alive_final / self.n_agents if self.n_agents > 0 else 0.0
        mean_lifespan = float(np.mean(self.lifespans)) if self.lifespans else 0.0

        mean_V_tail = float(np.mean([
            max(0.0, min(1.0, (row["mean_gamma"] - self.appx["gamma_min"]) / (self.appx["gamma_safe"] - self.appx["gamma_min"] + 1e-12)))
            * max(0.0, min(1.0, 1.0 - row["mean_h"] / (self.appx["h_max"] + 1e-12)))
            for _, row in tail.iterrows()
        ])) if len(tail) else 0.0

        summary = dict(
            seed=self.seed, regime=self.regime, policy=self.policy_name, architecture=self.architecture,
            var_ps=var_ps, p2p_ps=p2p_ps, rugosity=rugosity, cv_ps=cv_ps, H_causes=H_causes,
            deaths_gamma=self.death_counts["gamma"], deaths_h=self.death_counts["h"],
            deaths_C=self.death_counts["C"], deaths_B=self.death_counts["B"],
            survival_rate=survival_rate, mean_lifespan=mean_lifespan,
            mean_V_tail=mean_V_tail, convergence_tick=convergence_tick
        )
        return df_ticks, summary, df_agents

def main():
    parser = argparse.ArgumentParser(description="APP/MB Simulation Base")
    parser.add_argument("--config", type=str, default="config_example.yaml")
    parser.add_argument("--architecture", type=str, default=None, help="EV-02C or EV-04")
    parser.add_argument("--regime", type=str, default=None, help="RE/RH/RC")
    parser.add_argument("--policy", type=str, default=None, help="VOFF/APP-Fixed")
    parser.add_argument("--n_agents", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=1, help="number of seeds (0..seeds-1)")
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    
    # Step perturbation (AIM-3)
    parser.add_argument("--step_at", type=int, default=None, help="Tick to apply step perturbation")
    parser.add_argument("--step_scale", type=float, default=1.5, help="F_E multiplier during step")
    parser.add_argument("--step_dur", type=int, default=50, help="Duration of step perturbation")
    
    args = parser.parse_args()

    import os
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        if os.path.exists(cfg_path):
            pass
        elif os.path.exists(os.path.join(os.path.dirname(__file__), cfg_path)):
            cfg_path = os.path.join(os.path.dirname(__file__), cfg_path)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    architecture = args.architecture or cfg["general"]["architecture"]
    regime       = args.regime       or cfg["general"]["regime"]
    policy_name  = args.policy       or cfg["general"]["policy"]
    n_agents     = args.n_agents     or cfg["general"]["N"]
    T            = args.T            or cfg["general"]["T"]
    seeds        = args.seeds

    assert architecture in ("EV-02C", "EV-04")
    assert regime in cfg["regimes"]
    assert policy_name in ("VOFF", "APP-Fixed")
    assert n_agents > 0 and T > 0 and seeds > 0

    outputs_dir = args.outputs_dir
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "tick_data"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "summary"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "figures"), exist_ok=True)

    all_summaries = []
    for seed in range(seeds):
        sim = Simulation(cfg, regime, policy_name, architecture, n_agents, T, seed, outputs_dir,
                        step_at=args.step_at, step_scale=args.step_scale, step_dur=args.step_dur)
        df_ticks, summary, df_agents = sim.run()
        all_summaries.append(summary)

        tick_path = os.path.join(outputs_dir, "tick_data", f"{regime}_{policy_name}_{architecture}_seed{seed}.csv")
        df_ticks.to_csv(tick_path, index=False)
        
        # Save per-agent data if exported
        if df_agents:
            agents_dir = os.path.join(outputs_dir, "agents")
            os.makedirs(agents_dir, exist_ok=True)
            for agent_id, df_agent in df_agents.items():
                agent_path = os.path.join(agents_dir, f"{regime}_{policy_name}_{architecture}_seed{seed}_agent{agent_id}.csv")
                df_agent.to_csv(agent_path, index=False)

    df_summary = pd.DataFrame(all_summaries)
    df_summary_path = os.path.join(outputs_dir, "summary", "all_metrics.csv")
    df_summary.to_csv(df_summary_path, index=False)

    causes = ["gamma", "h", "C", "B"]
    labels = [c for c in causes if f"deaths_{c}" in df_summary.columns]
    totals = [int(df_summary[f"deaths_{c}"].sum()) for c in labels]

    plt.figure()
    plt.bar(labels, totals)
    plt.title(f"Mortality Causes - {regime} - {policy_name} - {architecture}")
    plt.xlabel("Cause")
    plt.ylabel("Count")
    fig_path = os.path.join(outputs_dir, "figures", f"mortality_causes_{regime}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Wrote tick data to: {os.path.abspath(os.path.join(outputs_dir, 'tick_data'))}")
    print(f"Wrote summary to : {os.path.abspath(df_summary_path)}")
    print(f"Wrote figure to  : {os.path.abspath(fig_path)}")

if __name__ == "__main__":
    main()
