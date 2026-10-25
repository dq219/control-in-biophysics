"""Pre-train the RL dosing agents shown on the web visualiser (docs/rl.html).

The browser cannot train an agent, so this script trains one actor-critic agent per
virtual-patient preset offline and writes their policy weights to docs/rl/agents.js
(a script setting window.RL_AGENTS, so the page also works when opened from disk).
Everything follows ReinforcementLearning/adaptive_therapy.ipynb (model, reward, network,
hyperparameters), with one change: the Lotka-Volterra model is integrated with a fixed-step
RK4 (RK4_STEPS per weekly decision) instead of solve_ivp, so that docs/assets/rl.js, which
uses the same integrator, reproduces every decision exactly.

The reference patient uses the notebook's trained checkpoint
(ReinforcementLearning/adaptive_therapy_learned_policy.pt). The other presets are trained
here with several seeds each; the seed whose greedy policy earns the highest reward is kept.

Usage:  python3 docs/rl/train_agents.py            # train and export
        python3 docs/rl/train_agents.py --check    # only print CT / AT50 times to progression
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
NOTEBOOK_CKPT = os.path.join(REPO, "ReinforcementLearning", "adaptive_therapy_learned_policy.pt")
OUT = os.path.join(HERE, "agents.js")

# ---------------------------------------------------------------- model and reward

BASE = {"rS": 0.027, "rR": 0.027, "dS": 0.0, "dR": 0.0, "dD": 1.5, "K": 1.0, "S0": 0.74, "R0": 0.01, "DMax": 1.0}

PRESETS = [
    {"id": "reference", "name": "Reference patient",
     "desc": "The patient of the review and notebook.", "patient": {**BASE}},
    {"id": "resistant", "name": "More resistance at diagnosis",
     "desc": "Five times as many resistant cells at the start, same total size.",
     "patient": {**BASE, "S0": 0.70, "R0": 0.05}},
    {"id": "fitter", "name": "Fast-growing resistant cells",
     "desc": "Resistant cells proliferate faster than sensitive ones.",
     "patient": {**BASE, "rR": 0.035}},
    {"id": "costly", "name": "Costly resistance",
     "desc": "Resistance carries a fitness cost: resistant cells proliferate more slowly.",
     "patient": {**BASE, "rR": 0.02}},
]

DT_DAYS = 7.0
RK4_STEPS = 14  # substeps per decision interval (half a day each)
REWARD = {"base": 0.1, "hol": 0.05, "punish": -0.1, "survival_bonus": 5.0,
          "progression_factor": 1.2, "horizon_days": 20 * 365}
HOLIDAY, TREAT = 0, 1

AGENT_KWARGS = dict(hidden=(128, 128, 64), lr=1e-3, gamma=0.99, value_coef=0.5, entropy_coef=0.01)
N_EPISODES = 4000
SEEDS = (0, 1, 2)


def rhs(p, S, R, D):
    comp = 1.0 - (S + R) / p["K"]
    return (p["rS"] * comp * (1.0 - p["dD"] * D) * S - p["dS"] * S,
            p["rR"] * comp * R - p["dR"] * R)


def rk4_week(p, S, R, D, record=None):
    h = DT_DAYS / RK4_STEPS
    for _ in range(RK4_STEPS):
        k1 = rhs(p, S, R, D)
        k2 = rhs(p, S + 0.5 * h * k1[0], R + 0.5 * h * k1[1], D)
        k3 = rhs(p, S + 0.5 * h * k2[0], R + 0.5 * h * k2[1], D)
        k4 = rhs(p, S + h * k3[0], R + h * k3[1], D)
        S += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        R += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if record is not None:
            record.append((S, R))
    return S, R


class Env:
    """The notebook's CancerTreatmentEnv with the RK4 integrator."""

    def __init__(self, patient):
        self.p = patient
        self.n0 = patient["S0"] + patient["R0"]
        self.progression = REWARD["progression_factor"] * self.n0
        self.reset()

    def obs(self):
        return np.array([(self.S + self.R) / self.n0, self.D / self.p["DMax"]], dtype=np.float32)

    def reset(self):
        self.S, self.R, self.t, self.D = self.p["S0"], self.p["R0"], 0.0, 0.0
        return self.obs()

    def step(self, action):
        self.D = self.p["DMax"] if action == TREAT else 0.0
        self.S, self.R = rk4_week(self.p, self.S, self.R, self.D)
        self.t += DT_DAYS
        tumour = self.S + self.R
        if tumour > self.progression:
            return self.obs(), REWARD["punish"], True
        reward = REWARD["base"]
        if self.t >= REWARD["horizon_days"]:
            return self.obs(), reward + REWARD["survival_bonus"], True
        if action == HOLIDAY:
            if tumour > self.n0:
                reward += REWARD["punish"]
            reward += REWARD["hol"]
        return self.obs(), reward, False


def rollout(env, policy):
    """Return (time to progression in days, total reward) for policy(env, obs) -> action."""
    obs, total, done = env.reset(), 0.0, False
    while not done:
        obs, r, done = env.step(policy(env, obs))
        total += r
    return env.t, total


def ct_policy(env, obs):
    return TREAT


def at50_policy(env, obs):
    tumour = env.S + env.R
    if env.D > 0:
        return HOLIDAY if tumour <= 0.5 * env.n0 else TREAT
    return TREAT if tumour >= env.n0 else HOLIDAY


# ---------------------------------------------------------------- agent (as in the notebook)

def make_agent(seed):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical

    class ActorCritic(nn.Module):
        def __init__(self, n_in, n_out, hidden):
            super().__init__()
            sizes = [n_in, *hidden]
            self.trunk = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
            self.policy_head = nn.Linear(sizes[-1], n_out)
            self.value_head = nn.Linear(sizes[-1], 1)

        def forward(self, x):
            for layer in self.trunk:
                x = F.relu(layer(x))
            return self.policy_head(x), self.value_head(x)

    class PGAgent:
        def __init__(self, hidden, lr, gamma, value_coef, entropy_coef):
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.gamma, self.value_coef, self.entropy_coef = gamma, value_coef, entropy_coef
            self.net = ActorCritic(2, 2, hidden)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        def act(self, obs, greedy=False):
            with torch.no_grad():
                logits, _ = self.net(torch.as_tensor(obs, dtype=torch.float32))
                if greedy:
                    return int(torch.argmax(logits).item())
                return int(Categorical(logits=logits).sample().item())

        def update(self, states, actions, rewards):
            g, ret = 0.0, np.empty(len(rewards), dtype=np.float32)
            for i in reversed(range(len(rewards))):
                g = rewards[i] + self.gamma * g
                ret[i] = g
            states = torch.as_tensor(np.array(states), dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.int64)
            ret = torch.as_tensor(ret)
            logits, values = self.net(states)
            values = values.squeeze(1)
            dist = Categorical(logits=logits)
            adv = ret - values.detach()
            loss = (-(dist.log_prob(actions) * adv).mean()
                    + self.value_coef * F.mse_loss(values, ret)
                    - self.entropy_coef * dist.entropy().mean())
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()

    return PGAgent(**AGENT_KWARGS)


def weights_of(state_dict):
    """Policy network as plain lists: trunk layers then the policy head (value head not needed)."""
    r = lambda a: [float(f"{x:.6g}") for x in np.asarray(a).ravel()]
    layers = []
    i = 0
    while f"trunk.{i}.weight" in state_dict:
        w = state_dict[f"trunk.{i}.weight"]
        layers.append({"shape": list(w.shape), "W": r(w), "b": r(state_dict[f"trunk.{i}.bias"])})
        i += 1
    w = state_dict["policy_head.weight"]
    layers.append({"shape": list(w.shape), "W": r(w), "b": r(state_dict["policy_head.bias"])})
    return layers


def train(job):
    import copy
    import torch
    torch.set_num_threads(1)
    preset, seed = job
    env = Env(preset["patient"])
    agent = make_agent(seed)
    best, best_sd, curve = -np.inf, None, []
    for ep in range(N_EPISODES):
        obs, done = env.reset(), False
        states, actions, rewards = [], [], []
        while not done:
            a = agent.act(obs)
            nxt, r, done = env.step(a)
            states.append(obs); actions.append(a); rewards.append(r)
            obs = nxt
        agent.update(states, actions, rewards)
        if (ep + 1) % 20 == 0:
            ttp, g = rollout(env, lambda e, o: agent.act(o, greedy=True))
            curve.append([ep + 1, round(g, 3)])
            if g > best:
                best, best_sd = g, copy.deepcopy(agent.net.state_dict())
    print(f"  {preset['id']:<10} seed {seed}: best greedy reward {best:.2f}", flush=True)
    return preset["id"], seed, best, {k: v.numpy() for k, v in best_sd.items()}, curve


def greedy_from_weights(layers):
    def policy(env, obs):
        x = np.asarray(obs, dtype=np.float64)
        for k, L in enumerate(layers):
            W = np.asarray(L["W"]).reshape(L["shape"])
            x = W @ x + np.asarray(L["b"])
            if k < len(layers) - 1:
                x = np.maximum(x, 0)
        return int(np.argmax(x))
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    for pr in PRESETS:
        env = Env(pr["patient"])
        ct, at = rollout(env, ct_policy)[0], rollout(env, at50_policy)[0]
        print(f"{pr['id']:<10} CT {ct / 30.4:5.1f} mo   AT50 {at / 30.4:5.1f} mo")
    if args.check:
        return

    import torch
    jobs = [(pr, s) for pr in PRESETS if pr["id"] != "reference" for s in SEEDS]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        results = pool.map(train, jobs)

    out = {"dt_days": DT_DAYS, "rk4_steps": RK4_STEPS, "reward": REWARD,
           "agent": {**AGENT_KWARGS, "hidden": list(AGENT_KWARGS["hidden"]), "episodes": N_EPISODES},
           "presets": []}
    for pr in PRESETS:
        if pr["id"] == "reference":
            sd = torch.load(NOTEBOOK_CKPT, map_location="cpu", weights_only=True)
            layers = weights_of({k: v.numpy() for k, v in sd.items()})
            source, seed, curve = "notebook checkpoint", None, None
        else:
            cands = [r for r in results if r[0] == pr["id"]]
            _, seed, _, sd, curve = max(cands, key=lambda r: r[2])
            layers = weights_of(sd)
            source = f"trained here, best of seeds {list(SEEDS)}"
        env = Env(pr["patient"])
        check = {name: rollout(env, pol)[0] for name, pol in
                 [("ct", ct_policy), ("at50", at50_policy), ("rl", greedy_from_weights(layers))]}
        print(f"{pr['id']:<10} RL {check['rl'] / 30.4:5.1f} mo  ({source})")
        out["presets"].append({**pr, "source": source, "seed": seed, "curve": curve,
                               "ttp_days": check, "policy": layers})
    with open(OUT, "w") as f:
        f.write("// Generated by train_agents.py -- do not edit.\nwindow.RL_AGENTS = ")
        json.dump(out, f, separators=(",", ":"))
        f.write(";\n")
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e3:.0f} kB)")


if __name__ == "__main__":
    sys.exit(main())
