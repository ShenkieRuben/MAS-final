from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools

from MCTS import MCTS


def set_ranges(depths, nr_its):
    return np.arange(1, depths + 1), np.arange(nr_its)


def plot_rollout_ratio(cvals, depth, nr_its, mcts_its, nr_rollouts):
    ratios_accum = {}

    for c in cvals:
        for d in range(1, depth + 1):
            ratios_accum[(c, d)] = []

    for it in range(nr_its):
        print(it)
        for c in cvals:
            # TODO: Also experiments for (10, 1)?
            mcts = MCTS(c, depth, mcts_its, nr_rollouts)
            mcts.perform_iters()

            # mcts.tree.pp_tree(0, 0)

            root = 0
            data = mcts.tree.data
            min_left = 0
            max_right = 0
            for d in range(1, depth+1):
                res = 0
                min_left = min_left * 2 + 1
                max_right = max_right * 2 + 2

                for i in range(min_left, max_right+1, 2):
                    left = i
                    right = i + 1
                    min_val = min(data[left].n, data[right].n)
                    max_val = max(data[left].n, data[right].n)
                    if min_val == -1 and max_val == -1:
                        break

                    if max_val != 0:
                        res += min_val/max_val

                ratios_accum[(c, d)].append(res)

            # for d in ratios_accum.keys():
            #     val = np.array(ratios_accum[d])
            #     avg = np.mean(val)
            #     std = np.std(val)
            #     ratios_accum[d] = (avg, std)
            #     # print(d, len(val))
    for key in ratios_accum.keys():
        val = np.array(ratios_accum[key])
        avg = np.mean(val)
        std = np.std(val)
        ratios_accum[key] = (avg, std)

    print(ratios_accum)
    fig, axs = plt.subplots(3, 2)
    palette = itertools.cycle(sns.color_palette())

    ds = range(1, depth)
    for line, ax in enumerate(fig.axes):
        x = []
        y = []
        err = []
        c = cvals[line]
        col = next(palette)
        for d in ds:
            val = ratios_accum[(c, d)]
            x.append(d)
            y.append(val[0])
            err.append(val[1])
        ax.errorbar(x, y, err, fmt="o-", capsize=5,
                    label=f"c={c}", color=col, lw=2, alpha=0.5)

        ax.set_xticks(ds)
        ax.legend()

        for d in ds:
            if d % 2 == 0:
                val = ratios_accum[(c, d)]
                ax.annotate(f"{round(ratios_accum[(c,d)][0],2)}", xy=(
                    d, ratios_accum[(c, d)][0]))

    fig.suptitle("Average exploration score for different values of c.")
    fig.text(0.5, 0.04, 'Depth (d)', va='center', ha='center')
    fig.text(0.04, 0.5, 'Average exploration score', va='center', ha='center',
             rotation='vertical')

    plt.show()


def plot_regret(cvals, depths, nr_its, mcts_its, nr_rollouts):
    ds, its = set_ranges(depths, nr_its)
    ds_regret = {}
    perf = {}

    for d in ds:
        ds_regret[d] = []
        perf[d] = []

    for it in its:
        print(it)
        for d in ds:
            regrets = []
            for c in cvals:
                mcts = MCTS(c, d, mcts_its, nr_rollouts)
                mcts.perform_iters()
                max_val = mcts.tree.get_max()
                found_leave = mcts.tree.data[mcts.informed_path[-1]]
                found_val = found_leave.t / found_leave.n
                diff = max_val - found_val
                regrets.append(diff)
            ds_regret[d].append(regrets)

    for d in ds:
        regret_matrix = np.array(ds_regret[d])
        avgs = np.average(regret_matrix, axis=0)
        stds = np.std(regret_matrix, axis=0)
        perf[d] = (avgs, stds)

    print(perf)
    total_avgs = []
    total_stds = []

    for i in range(len(cs)):
        c_lines_avgs = []
        c_lines_stds = []
        for key, val in perf.items():
            cval_avg = val[0][i]
            cval_std = val[1][i]
            c_lines_avgs.append(cval_avg)
            c_lines_stds.append(cval_std)
        total_avgs.append(c_lines_avgs)
        total_stds.append(c_lines_stds)

    palette = itertools.cycle(sns.color_palette())
    fig, axs = plt.subplots(3, 2)

    for line, ax in enumerate(fig.axes):
        col = next(palette)
        ax.errorbar(ds, total_avgs[line],
                    total_stds[line], fmt="o-", capsize=5,  label=f"c={cs[line]}", color=col, lw=2, alpha=0.5)
        for d in range(len(ds)):
            if d % 2 == 0:
                ax.annotate(f"{round(total_avgs[line][d],2)}",
                            xy=(ds[d], total_avgs[line][d]))

        ax.set_xticks(ds)
        ax.legend()

    fig.suptitle("Average total regret for varying depths and c values.")
    fig.text(0.5, 0.04, 'Depth (d)', va='center', ha='center')
    fig.text(0.04, 0.5, 'Average regret', va='center', ha='center',
             rotation='vertical')

    plt.show()

    print(total_avgs)
    print(total_stds)


def plot_perf(depths, nr_its, mcts_its, nr_rollouts):

    ds, its = set_ranges(depths, nr_its)

    ds_time = {}
    perf = {}

    for d in ds:
        ds_time[d] = []
        perf[d] = ()

    for it in its:
        print(it)
        for d in ds:
            start = timer()
            mcts = MCTS(2, d, mcts_its, nr_rollouts)
            mcts.perform_iters()
            end = timer()
            ds_time[d].append(end - start)

    for key, val in ds_time.items():
        time_results = val
        perf[key] = (np.mean(time_results), np.std(time_results))

    res = list(perf.values())
    res = list(zip(*res))
    means = res[0]
    stds = res[1]
    plt.errorbar(list(perf.keys()), means, stds, fmt="-", capsize=5)
    plt.title("Performance of MCTS")
    plt.xlabel("Depth (d)")
    plt.ylabel("Time (s)")
    plt.xticks(np.arange(0, len(perf)+1, 1))
    plt.show()


if __name__ == "__main__":
    cs = [0.1, 1, 5, 10, 50, 100]
    # cs = [0.01, 1000]

    plot_perf(21, 100, 10, 1)
    # plot_regret(cs, 16, 100, 10, 1)
    # ratios = plot_rollout_ratio(cs, 17, 100, 10, 1)
    # for c, ratio in ratios:
    #     print(c, ratio)
    #     pass
