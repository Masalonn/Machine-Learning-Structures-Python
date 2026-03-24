import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


@dataclass(frozen=True)
class GAConfig:
    L: int = 35
    N: int = 100
    G: int = 150
    tournament_k: int = 3
    elitism: int = 1
    p_cross: float = 0.9
    p_mut: Optional[float] = None
    seed: Optional[int] = 42


def fitness(ind: List[int]) -> int:
    return ind.count(0)


def tournament(pop: List[List[int]], fits: List[int], k: int, rng: random.Random) -> List[int]:
    best = rng.randrange(len(pop))
    for _ in range(k - 1):
        i = rng.randrange(len(pop))
        if fits[i] > fits[best]:
            best = i
    return pop[best][:]


def crossover(a: List[int], b: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    c1 = [a[i] if rng.getrandbits(1) else b[i] for i in range(len(a))]
    c2 = [b[i] if rng.getrandbits(1) else a[i] for i in range(len(a))]
    return c1, c2


def mutate(ind: List[int], p: float, rng: random.Random) -> None:
    for i in range(len(ind)):
        if rng.random() < p:
            ind[i] ^= 1


def run_ga(cfg: GAConfig):
    rng = random.Random(cfg.seed)
    p_mut = cfg.p_mut if cfg.p_mut is not None else 1.0 / cfg.L

    pop = [[rng.getrandbits(1) for _ in range(cfg.L)] for _ in range(cfg.N)]
    pop_init = [x[:] for x in pop]

    h_best, h_mean = [], []
    best_global, best_fit_global = None, -1

    for gen in range(cfg.G):
        fits = [fitness(x) for x in pop]
        best_fit = max(fits)
        mean_fit = sum(fits) / len(fits)

        h_best.append(best_fit)
        h_mean.append(mean_fit)

        if best_fit > best_fit_global:
            best_fit_global = best_fit
            best_global = pop[fits.index(best_fit)][:]

        if best_fit_global == cfg.L:
            break

        new_pop = []
        elites = sorted(range(cfg.N), key=lambda i: fits[i], reverse=True)[:cfg.elitism]
        for i in elites:
            new_pop.append(pop[i][:])

        while len(new_pop) < cfg.N:
            p1 = tournament(pop, fits, cfg.tournament_k, rng)
            p2 = tournament(pop, fits, cfg.tournament_k, rng)

            if rng.random() < cfg.p_cross:
                c1, c2 = crossover(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]

            mutate(c1, p_mut, rng)
            mutate(c2, p_mut, rng)

            new_pop.append(c1)
            if len(new_pop) < cfg.N:
                new_pop.append(c2)

        pop = new_pop

    pop_final = [x[:] for x in pop]
    return best_global, best_fit_global, gen, h_best, h_mean, pop_init, pop_final


def plot_convergence(h_best, h_mean, L):
    if not HAS_PLOT:
        return
    plt.figure()
    plt.plot(h_best, label="best")
    plt.plot(h_mean, label="mean")
    plt.axhline(L, linestyle="--", label="max")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_hist(pop_init, pop_final):
    if not HAS_PLOT:
        return
    init_f = [fitness(x) for x in pop_init]
    final_f = [fitness(x) for x in pop_final]
    m = max(init_f + final_f)

    plt.figure()
    plt.hist(init_f, bins=range(0, m + 2), alpha=0.6, label="start")
    plt.hist(final_f, bins=range(0, m + 2), alpha=0.6, label="end")
    plt.xlabel("fitness")
    plt.ylabel("count")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    cfg = GAConfig()
    best, best_fit, gen, h_best, h_mean, pop_init, pop_final = run_ga(cfg)

    print("Best fitness:", best_fit)
    print("Generations:", gen)
    print("Best individual:", best)

    plot_convergence(h_best, h_mean, cfg.L)
    plot_hist(pop_init, pop_final)
