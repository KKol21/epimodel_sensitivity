from matplotlib import pyplot as plt
import numpy as np
import torch
import seaborn
import pandas as pd


def get_target(target_var):
    if target_var == "i_max":
        return "fertőzöttek száma a csúcson"
    elif target_var == "ic_max":
        return "intenzív betegek száma a csúcson"
    elif target_var == "d_max":
        return "halottak száma"


def generate_prcc_plot(params, target_var, prcc: np.ndarray, filename: str, r0):
    """
    Generate a tornado plot to visualize the Partial Rank Correlation Coefficient (PRCC) values.

    Args:
        params (list): The list of parameter names.
        target_var (str): The target variable.
        prcc (np.ndarray): The PRCC values.
        filename (str): The filename for saving the plot.
        r0: The value of R0.

    Returns:
        None
    """
    prcc = np.round(prcc, 3)
    target_var = get_target(target_var)
    plt.title(f"PRCC értékei a vakcinák elosztásának\n"
              f"Célváltozó: {target_var}\n "
              r"$\mathcal{R}_0=$" + str(r0), fontsize=15, wrap=True)

    ys = range(len(params))[::-1]
    # Plot the bars one by one
    for y, value in zip(ys, prcc):
        plt.broken_barh(
            [(value if value < 0 else 0, abs(value))],
            (y - 0.4, 0.8),
            facecolors=['white', 'white'],
            edgecolors=['black', 'black'],
            linewidth=1,
        )

        if value != 0:
            x = (value / 2) if np.abs(value) >= 0.2 else (- np.sign(value) * 0.1)
        else:
            x = -0.1
        plt.text(x, y - 0.01, str(value), va='center', ha='center')

    plt.axvline(0, color='black')

    # Position the x-axis on the top, hide all the other spines (=axis lines)
    axes = plt.gca()  # (gca = get current axes)
    axes.spines['left'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.xaxis.set_ticks_position('top')

    # Display age groups next to y axis
    def get_age_group(param):
        age_start = int(param.split('_')[2]) * 5
        return f'{age_start}-{age_start + 4} ' if age_start != 75 else '75+ '
    labels = list(map(get_age_group, params))
    plt.yticks(ys, labels)

    # Set the portion of the x- and y-axes to show
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1, len(params))
    # plt.text()
    plt.savefig(f'../sens_data/prcc_plots/prcc_tornado_plot_{filename}.pdf',
                format="pdf", bbox_inches='tight')
    plt.show()


def generate_epidemic_plot(sim_obj, vaccination, filename, target_var, r0, plot_title=None, compartments=None):
    from src.model.r0 import R0Generator

    if compartments is None:
        compartments = sim_obj.target_var_choices

    if plot_title is None:
        target = get_target(target_var)
        plot_title = "Járványgörbe korcsoportokra aggregálva \n"\
                     f"Vakcinálás célváltozója: {target}\n"\
                     f"R0={r0}"

    colors = ['orange', 'red', 'black']

    model = sim_obj.model
    sim_obj.params["susc"] = torch.ones(sim_obj.n_age).to(sim_obj.device)
    r0generator = R0Generator(param=sim_obj.params, device=sim_obj.data.device, n_age=sim_obj.n_age)
    # Calculate base transmission rate
    beta = r0 / r0generator.get_eig_val(contact_mtx=sim_obj.contact_matrix,
                                        susceptibles=sim_obj.susceptibles.reshape(1, -1),
                                        population=sim_obj.population)
    sim_obj.params["beta"] = beta
    model.get_constant_matrices()

    t_eval = torch.linspace(1, 1200, 1200).to(sim_obj.device)
    sol = sim_obj.model.get_solution(t_eval=t_eval[None, :],
                                     y0=sim_obj.model.get_initial_values()[None, :],
                                     lhs_table=torch.tensor(vaccination[None, :]).float()).ys[0, :, :]
    mask = torch.cat((torch.full((100, ), True),
                      sol[100:, model.idx('ic_0')].sum(axis=1) > 1))
    sol = sol[mask, :]
    t = t_eval[mask]

    for idx, comp in enumerate(compartments):
        comp_sol = model.aggregate_by_age(sol, comp)
        plt.plot(t, comp_sol, label=comp.upper(), color=colors[idx], linewidth=2)

    plt.legend()
    plt.gca().set_xlabel('Napok')
    plt.gca().set_ylabel('Kompartmentek méretei')
    plt.title(plot_title, y=1.03, fontsize=12)
    plt.savefig(f'../sens_data/epidemic_plots/epidemic_plot_{filename}.pdf',
                format="pdf", bbox_inches='tight')
    plt.show()
    plt.close()


def generate_epidemic_plot_(sim_obj, vaccination, vaccination_opt, filename, target_var, r0, r0_bad, plot_title=None, compartments=None):
    from src.model.r0 import R0Generator

    if compartments is None:
        compartments = sim_obj.target_var_choices

    if plot_title is None:
        target = get_target(target_var)
        plot_title = "Járványgörbe korcsoportokra aggregálva \n"\
                     f"Vakcinálás célváltozója: {target}\n"\
                     r"Szimuláció $\mathcal{R}_0=$" + str(r0)

    comp = 'ic'
    colors = ['orange', 'red']

    model = sim_obj.model
    sim_obj.params["susc"] = torch.ones(sim_obj.n_age).to(sim_obj.device)

    r0generator = R0Generator(param=sim_obj.params, device=sim_obj.data.device, n_age=sim_obj.n_age)
    ngm_ev = r0generator.get_eig_val(contact_mtx=sim_obj.contact_matrix,
                                        susceptibles=sim_obj.susceptibles.reshape(1, -1),
                                        population=sim_obj.population)
    # Calculate base transmission rate
    beta = r0 / ngm_ev
    sim_obj.params["beta"] = beta
    model.get_constant_matrices()

    t = torch.linspace(1, 1000, 1000).to(sim_obj.device)
    sol_real = sim_obj.model.get_solution(t=t, cm=sim_obj.contact_matrix,
                                          daily_vac=torch.tensor(vaccination_opt).float())
    sol_bad = sim_obj.model.get_solution(t=t, cm=sim_obj.contact_matrix,
                                         daily_vac=torch.tensor(vaccination).float())

    comp_sol = model.aggregate_by_age(sol_real, comp)
    comp_sol_bad = model.aggregate_by_age(sol_bad, comp)
    plt.plot(t, comp_sol_bad, color=colors[1], linewidth=2)
    plt.plot(t, comp_sol, '--', color=colors[0], linewidth=2)

    plt.legend([r'Optimális vakcinálás $\mathcal{R}_0 = 1.8$ esetén',
                r'Optimális vakcinálás $\mathcal{R}_0 = 3$ esetén'], fontsize=8)
    plt.gca().set_xlabel('Napok')
    plt.gca().set_ylabel('Intenzív betegek száma')
    plt.title(plot_title, y=1.03, fontsize=12)
    plt.savefig(f'../sens_data/epidemic_plots_/epidemic_plot_{filename}_{r0}.pdf',
                format="pdf", bbox_inches='tight')
    plt.show()
    plt.close()


def get_age_groups():
    return [f'{5 * age_start}-{5 * age_start + 5}' if 5 * age_start != 75 else '75+' for age_start in range(16)]


def get_hmap(cm):
    labels = get_age_groups()
    cm = pd.DataFrame(cm).loc[:, ::-1].T
    cmap = seaborn.color_palette("flare", as_cmap=True)
    hmap = seaborn.heatmap(cm, xticklabels=labels, yticklabels=labels[::-1], cmap=cmap)
    hmap = hmap.get_figure()
    hmap.suptitle('Kontakt mátrix hőtérképe')
    hmap.savefig('../cont_mtx_hmap.pdf', dpi=400)
