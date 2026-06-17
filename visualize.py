"""可视化模块"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

AGENT_NAMES = ['T1', 'T2', 'WT', 'PV', 'V1', 'V2']
AGENT_COLORS = ['#e74c3c', '#c0392b', '#2ecc71', '#27ae60', '#3498db', '#2980b9']
SCENARIO_NAMES = ['电力市场', '电-绿证耦合市场', '电-碳-绿证耦合市场']
SCENARIO_NAMES_EN = ['Electricity Market', 'Elec-Green Cert Market',
                     'Elec-Carbon-Green Cert Market']


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def smooth(data, window=50):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def smooth_window(data, target_window=100):
    data = np.array(data, dtype=float)
    if len(data) < 3:
        return data
    window = min(target_window, max(3, len(data) // 8))
    return smooth(data, window)


def smooth_ema(data, alpha=0.01):
    result = np.zeros_like(data, dtype=float)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result


def plot_training_rewards(scenario_results, save_dir):
    ensure_dir(save_dir)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    history = scenario_results[3]['history']

    for i, (ax, name) in enumerate(zip(axes, AGENT_NAMES)):
        raw = np.array(history['episode_rewards'][i])
        smoothed = smooth_ema(raw, alpha=0.03)
        episodes = np.arange(len(smoothed))

        ax.plot(episodes, smoothed / 1e4, color=AGENT_COLORS[i], linewidth=1.2)
        ax.set_ylabel(f'{name}收益/万元', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    axes[-2].set_xlabel('训练次数', fontsize=11)
    axes[-1].set_xlabel('训练次数', fontsize=11)
    fig.suptitle('图2 训练过程奖励变化曲线\nFig.2 Training process reward change curve',
                 fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'fig2_training_rewards.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved fig2_training_rewards.png")


def plot_scenario_comparison(scenario_results, save_dir):
    ensure_dir(save_dir)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    periods = np.arange(1, 25)
    bar_width = 0.13

    for sc_idx, sc in enumerate([1, 2, 3]):
        ax_q = axes[sc_idx, 0]
        ax_p = axes[sc_idx, 1]

        history = scenario_results[sc]['history']
        if 'final_eval' not in history:
            raise ValueError('Missing final_eval in scenario_results; rerun training.')
        final_eval = history['final_eval']
        hourly_agents = np.array(final_eval['outputs'], dtype=float)
        price_profile = np.array(final_eval['clearing_prices'], dtype=float)

        for i, name in enumerate(AGENT_NAMES):
            positions = periods + (i - 2.5) * bar_width
            ax_q.bar(positions, hourly_agents[:, i], bar_width,
                     color=AGENT_COLORS[i],
                     label=name if sc_idx == 0 else '', alpha=0.85)

        ax_q.set_ylabel('中标电量/(MW·h)', fontsize=10)
        ax_q.set_title(SCENARIO_NAMES[sc_idx], fontsize=11)
        ax_q.grid(True, alpha=0.2, axis='y')

        ax_p.plot(periods, price_profile, 'k-o', markersize=3, linewidth=1.5)
        ax_p.set_ylabel('出清电价/(元/(MW·h))', fontsize=10)
        ax_p.set_title(SCENARIO_NAMES[sc_idx], fontsize=11)
        ax_p.grid(True, alpha=0.3)

    axes[0, 0].legend(ncol=6, fontsize=8, loc='upper center',
                       bbox_to_anchor=(1.1, 1.25))

    for ax in axes[-1]:
        ax.set_xlabel('时段/h', fontsize=10)

    fig.suptitle('图3 不同场景下各市场主体中标电量以及出清电价\n'
                 'Fig.3 Bid quantity and clearing price under different scenarios',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig3_scenario_comparison.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved fig3_scenario_comparison.png")


def plot_vpp_internal(scenario_results, save_dir):
    ensure_dir(save_dir)
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    periods = np.arange(1, 25)

    component_names = ['储能放电', '燃气', '光伏', '风电']
    component_colors = ['#9b59b6', '#e67e22', '#f1c40f', '#2ecc71']
    charge_color = '#8e44ad'

    for sc_idx, sc in enumerate([1, 2, 3]):
        ax = axes[sc_idx]
        history = scenario_results[sc]['history']
        if 'final_eval' not in history:
            raise ValueError('Missing final_eval in scenario_results; rerun training.')
        components = history['final_eval']['vpp_components']['V1']
        storage_discharge = np.array(
            [row['storage_discharge'] for row in components], dtype=float)
        gas_out = np.array([row['gas'] for row in components], dtype=float)
        solar_out = np.array([row['solar'] for row in components], dtype=float)
        wind_out = np.array([row['wind'] for row in components], dtype=float)
        storage_charge = np.array(
            [row['storage_charge'] for row in components], dtype=float)

        bottoms = np.zeros(24)
        for data, label, color in zip(
                [storage_discharge, gas_out, solar_out, wind_out],
                component_names, component_colors):
            ax.bar(periods, data, bottom=bottoms, color=color,
                   label=label if sc_idx == 0 else '', alpha=0.85, width=0.8)
            bottoms += data

        ax.bar(periods, -storage_charge, color=charge_color, alpha=0.6,
               label='储能充电' if sc_idx == 0 else '', width=0.8)

        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.set_ylabel('V1内部出力/(MW·h)', fontsize=10)
        ax.set_title(SCENARIO_NAMES[sc_idx], fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')

    axes[0].legend(ncol=5, fontsize=8, loc='upper right')
    axes[-1].set_xlabel('时段/h', fontsize=10)

    fig.suptitle('图4 不同场景下虚拟电厂V1中市场主体中标电量\n'
                 'Fig.4 Bid quantity of each entity in VPP V1 under different scenarios',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_vpp_internal.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved fig4_vpp_internal.png")


def plot_quota_analysis(quota_results, save_dir):
    ensure_dir(save_dir)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    cq = np.array(quota_results['carbon_quotas'])
    gq = np.array(quota_results['green_quotas'])
    CQ, GQ = np.meshgrid(cq, gq)
    Z = quota_results['elec_prices'].T

    surf = ax.plot_surface(CQ, GQ * 100, Z, cmap='RdYlGn_r',
                           edgecolor='grey', linewidth=0.3, alpha=0.85)
    ax.set_xlabel('碳配额系数(t/MW·h)', fontsize=10, labelpad=10)
    ax.set_ylabel('绿证配额系数(%)', fontsize=10, labelpad=10)
    ax.set_zlabel('电价/(元/(MW·h))', fontsize=10, labelpad=10)

    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    ax.set_title('图5 不同配额系数下市场的电价\n'
                 'Fig.5 Electricity prices under different quota coefficients',
                 fontsize=12, pad=20)
    plt.savefig(os.path.join(save_dir, 'fig5_quota_analysis.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved fig5_quota_analysis.png")


def plot_penalty_analysis(penalty_results, save_dir):
    ensure_dir(save_dir)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    lp = np.array(penalty_results['lambda_plus'])
    lm = np.array(penalty_results['lambda_minus'])
    LP, LM = np.meshgrid(lp, lm)
    Z = penalty_results['elec_prices'].T

    surf = ax.plot_surface(LP, LM, Z, cmap='coolwarm',
                           edgecolor='grey', linewidth=0.3, alpha=0.85)
    ax.set_xlabel('少投惩罚系数', fontsize=10, labelpad=10)
    ax.set_ylabel('多投惩罚系数', fontsize=10, labelpad=10)
    ax.set_zlabel('电价/(元/(MW·h))', fontsize=10, labelpad=10)

    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    ax.set_title('图6 不同偏差惩罚系数下市场的电价\n'
                 'Fig.6 Electricity price under different penalty coefficients',
                 fontsize=12, pad=20)
    plt.savefig(os.path.join(save_dir, 'fig6_penalty_analysis.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved fig6_penalty_analysis.png")


def plot_algorithm_comparison(algo_results, save_dir):
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(10, 5))

    algo_colors = {'MA-CETD3': '#3498db', 'MATD3': '#e67e22'}

    smoothed_data = {}
    for algo in ['MA-CETD3', 'MATD3']:
        raw = np.array(algo_results[algo]['history']['total_rewards'])
        smoothed_data[algo] = smooth_ema(raw, alpha=0.005)

    cetd3 = smoothed_data['MA-CETD3']
    matd3 = smoothed_data['MATD3']
    final_cetd3 = np.mean(cetd3[int(len(cetd3)*0.8):])
    final_matd3 = np.mean(matd3[int(len(matd3)*0.8):])
    ax.plot(np.arange(len(cetd3)), cetd3 / 1e4, color=algo_colors['MA-CETD3'],
            linewidth=1.5, label='MA-CETD3')
    ax.plot(np.arange(len(matd3)), matd3 / 1e4, color=algo_colors['MATD3'],
            linewidth=1.5, label='MATD3')

    ax.set_ylabel('总收益/万元', fontsize=11)
    ax.set_xlabel('训练次数', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.legend(ncol=2, fontsize=10, loc='lower right')

    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin

    cetd3_eq = np.mean(cetd3[int(len(cetd3)*0.8):]) / 1e4
    matd3_eq = np.mean(matd3[int(len(matd3)*0.8):]) / 1e4
    ax.axhline(y=cetd3_eq, color=algo_colors['MA-CETD3'], linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axhline(y=matd3_eq, color=algo_colors['MATD3'], linestyle='--', alpha=0.4, linewidth=0.8)

    conv_thr = final_cetd3 * 0.95
    cetd3_conv = int(np.argmax(cetd3 > conv_thr)) if np.any(cetd3 > conv_thr) else len(cetd3)
    matd3_conv = int(np.argmax(matd3 > conv_thr)) if np.any(matd3 > conv_thr) else len(matd3)
    if len(cetd3) >= 100:
        ax.axvline(x=cetd3_conv, color=algo_colors['MA-CETD3'], linestyle=':', alpha=0.5)
        ax.axvline(x=matd3_conv, color=algo_colors['MATD3'], linestyle=':', alpha=0.5)
        ax.text(cetd3_conv + 30, ymin + yr*0.15, f'{cetd3_conv}轮', fontsize=8, color=algo_colors['MA-CETD3'])
        ax.text(matd3_conv + 30, ymin + yr*0.08, f'{matd3_conv}轮', fontsize=8, color=algo_colors['MATD3'])

    summary_text = (
        f"MA-CETD3均衡: {cetd3_eq:.2f}万元\n"
        f"MATD3均衡: {matd3_eq:.2f}万元"
    )
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))

    ax.set_title('图7 市场主体总收益对比\n'
                 'Fig.7 Comparison of total revenue among market entities',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig7_algorithm_comparison.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved fig7_algorithm_comparison.png")


def plot_bidding_convergence(algo_results, save_dir):
    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(10, 5))

    history = algo_results['MA-CETD3']['history']

    total_eps = len(history['bidding_coeffs'][0])
    start = 0
    end = total_eps
    alpha = 0.08 if total_eps < 100 else 0.03

    for i, name in enumerate(AGENT_NAMES):
        data = np.array(history['bidding_coeffs'][i][start:end])
        smoothed = smooth_ema(data, alpha=alpha)
        eps_s = np.arange(start, start + len(smoothed))
        ax.plot(eps_s, smoothed, color=AGENT_COLORS[i], linewidth=1.5,
                label=name)

    ax.set_xlabel('训练回合/次', fontsize=11)
    ax.set_ylabel('报价系数', fontsize=11)
    ax.legend(ncol=3, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('图8 市场主体单时段报价系数曲线\n'
                 'Fig.8 Market entities single-period bidding coefficient curve',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig8_bidding_convergence.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved fig8_bidding_convergence.png")


def generate_table1(scenario_results, save_dir):
    ensure_dir(save_dir)

    names = AGENT_NAMES
    print("\n" + "="*75)
    print("表1 各市场主体运行收益 / Table 1 Operating gains of each market entity")
    print("="*75)
    header = f"{'场景':<20}" + "".join([f"{n:>10}" for n in names]) + f"{'总计':>12}"
    print(header)
    print("-"*75)

    rows = []
    for sc in [1, 2, 3]:
        rev = scenario_results[sc]['revenues']
        row = f"{'场景'+str(sc):<20}"
        for r in rev:
            row += f"{r:>10.0f}"
        row += f"{np.sum(rev):>12.0f}"
        print(row)
        rows.append({'scenario': sc, 'revenues': rev.tolist(),
                     'total': float(np.sum(rev))})

    with open(os.path.join(save_dir, 'table1_revenues.txt'), 'w',
              encoding='utf-8') as f:
        f.write("表1 各市场主体运行收益 (单位: 元)\n")
        f.write("Table 1 Operating gains of each market entity (Unit: yuan)\n\n")
        f.write(header + "\n")
        f.write("-"*75 + "\n")
        for sc in [1, 2, 3]:
            rev = scenario_results[sc]['revenues']
            row = f"{'场景'+str(sc):<20}"
            for r in rev:
                row += f"{r:>10.0f}"
            row += f"{np.sum(rev):>12.0f}"
            f.write(row + "\n")

    print("  Saved table1_revenues.txt")
    return rows


def generate_table2(algo_results, save_dir):
    ensure_dir(save_dir)

    names = AGENT_NAMES
    print("\n" + "="*80)
    print("表2 不同求解算法下发电主体收益 / Table 2 Performance comparison")
    print("="*80)

    header = f"{'方法':<15}" + "".join([f"{n:>10}" for n in names]) + f"{'总收益':>12}"
    print(header)
    print("-"*80)

    rows = {}
    for algo in ['MATD3', 'MA-CETD3']:
        rev = algo_results[algo]['revenues']
        row = f"{algo:<15}"
        for r in rev:
            row += f"{r/1e4:>10.2f}"
        row += f"{np.sum(rev)/1e4:>12.2f}"
        print(row)
        rows[algo] = rev

    matd3_rev = rows['MATD3']
    macetd3_rev = rows['MA-CETD3']
    print(f"\n{'变化率/%':<15}", end='')
    for i in range(len(names)):
        if abs(matd3_rev[i]) > 1:
            rate = (macetd3_rev[i] - matd3_rev[i]) / abs(matd3_rev[i]) * 100
        else:
            rate = 0.0
        print(f"{rate:>10.2f}", end='')
    total_rate = (np.sum(macetd3_rev) - np.sum(matd3_rev)) / abs(np.sum(matd3_rev)) * 100
    print(f"{total_rate:>12.2f}")

    with open(os.path.join(save_dir, 'table2_algorithm_comparison.txt'), 'w',
              encoding='utf-8') as f:
        f.write("表2 不同求解算法下发电主体收益 (单位: 万元)\n")
        f.write("Table 2 Performance comparison (Unit: 10000 yuan)\n\n")
        f.write(header + "\n")
        f.write("-"*80 + "\n")
        for algo in ['MATD3', 'MA-CETD3']:
            rev = algo_results[algo]['revenues']
            row = f"{algo:<15}"
            for r in rev:
                row += f"{r/1e4:>10.2f}"
            row += f"{np.sum(rev)/1e4:>12.2f}"
            f.write(row + "\n")
        f.write(f"\n总收益提升: {total_rate:.2f}%\n")

    print(f"\n总收益提升: {total_rate:.2f}%")
    print("  Saved table2_algorithm_comparison.txt")


def plot_carbon_green_prices(scenario_results, save_dir):
    ensure_dir(save_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    history = scenario_results[3]['history']

    carbon = smooth(np.array(history['carbon_prices']), 150)
    ax1.plot(np.arange(len(carbon)), carbon, 'r-', linewidth=1.2)
    ax1.set_xlabel('训练次数', fontsize=11)
    ax1.set_ylabel('碳价/(元/t)', fontsize=11)
    ax1.set_title('实时碳价变化曲线', fontsize=11)
    ax1.grid(True, alpha=0.3)

    green = smooth(np.array(history['green_prices']), 150)
    ax2.plot(np.arange(len(green)), green, 'g-', linewidth=1.2)
    ax2.set_xlabel('训练次数', fontsize=11)
    ax2.set_ylabel('绿价/(元/张)', fontsize=11)
    ax2.set_title('实时绿价变化曲线', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('附录图D5 电-碳-绿证耦合市场下实时碳价和绿价变化曲线',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'figD5_carbon_green_prices.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved figD5_carbon_green_prices.png")


def plot_quota_carbon_price(quota_results, save_dir):
    ensure_dir(save_dir)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    cq = np.array(quota_results['carbon_quotas'])
    gq = np.array(quota_results['green_quotas'])
    CQ, GQ = np.meshgrid(cq, gq)
    Z = quota_results['carbon_prices'].T

    surf = ax.plot_surface(CQ, GQ * 100, Z, cmap='YlOrRd',
                           edgecolor='grey', linewidth=0.3, alpha=0.85)
    ax.set_xlabel('碳配额系数(t/MW·h)', fontsize=10, labelpad=10)
    ax.set_ylabel('绿证配额系数(%)', fontsize=10, labelpad=10)
    ax.set_zlabel('碳价/(元/t)', fontsize=10, labelpad=10)

    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    ax.set_title('图D6 不同配额系数下碳价\n'
                 'Fig.D6 Carbon price under different quota coefficients',
                 fontsize=12, pad=20)
    plt.savefig(os.path.join(save_dir, 'figD6_quota_carbon_price.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved figD6_quota_carbon_price.png")


def plot_quota_green_price(quota_results, save_dir):
    ensure_dir(save_dir)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    cq = np.array(quota_results['carbon_quotas'])
    gq = np.array(quota_results['green_quotas'])
    CQ, GQ = np.meshgrid(cq, gq)
    Z = quota_results['green_prices'].T

    surf = ax.plot_surface(CQ, GQ * 100, Z, cmap='YlGn',
                           edgecolor='grey', linewidth=0.3, alpha=0.85)
    ax.set_xlabel('碳配额系数(t/MW·h)', fontsize=10, labelpad=10)
    ax.set_ylabel('绿证配额系数(%)', fontsize=10, labelpad=10)
    ax.set_zlabel('绿证价格/(元/张)', fontsize=10, labelpad=10)

    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    ax.set_title('图D7 不同配额系数下绿证价格\n'
                 'Fig.D7 Green cert price under different quota coefficients',
                 fontsize=12, pad=20)
    plt.savefig(os.path.join(save_dir, 'figD7_quota_green_price.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved figD7_quota_green_price.png")


def plot_penalty_carbon_price(penalty_results, save_dir):
    ensure_dir(save_dir)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    lp = np.array(penalty_results['lambda_plus'])
    lm = np.array(penalty_results['lambda_minus'])
    LP, LM = np.meshgrid(lp, lm)
    Z = penalty_results['carbon_prices'].T

    surf = ax.plot_surface(LP, LM, Z, cmap='OrRd',
                           edgecolor='grey', linewidth=0.3, alpha=0.85)
    ax.set_xlabel('少投惩罚系数', fontsize=10, labelpad=10)
    ax.set_ylabel('多投惩罚系数', fontsize=10, labelpad=10)
    ax.set_zlabel('碳价/(元/t)', fontsize=10, labelpad=10)

    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    ax.set_title('图D8 不同偏差惩罚系数下碳价\n'
                 'Fig.D8 Carbon price under different penalty coefficients',
                 fontsize=12, pad=20)
    plt.savefig(os.path.join(save_dir, 'figD8_penalty_carbon_price.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved figD8_penalty_carbon_price.png")


def plot_penalty_green_price(penalty_results, save_dir):
    ensure_dir(save_dir)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    lp = np.array(penalty_results['lambda_plus'])
    lm = np.array(penalty_results['lambda_minus'])
    LP, LM = np.meshgrid(lp, lm)
    Z = penalty_results['green_prices'].T

    surf = ax.plot_surface(LP, LM, Z, cmap='BuGn',
                           edgecolor='grey', linewidth=0.3, alpha=0.85)
    ax.set_xlabel('少投惩罚系数', fontsize=10, labelpad=10)
    ax.set_ylabel('多投惩罚系数', fontsize=10, labelpad=10)
    ax.set_zlabel('绿证价格/(元/张)', fontsize=10, labelpad=10)

    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    ax.set_title('图D9 不同偏差惩罚系数下绿证价格\n'
                 'Fig.D9 Green cert price under different penalty coefficients',
                 fontsize=12, pad=20)
    plt.savefig(os.path.join(save_dir, 'figD9_penalty_green_price.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved figD9_penalty_green_price.png")


def plot_agent_count_convergence(agent_count_results, save_dir):
    ensure_dir(save_dir)
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    group_titles = {
        'group_thermal': ('不同火电数量', 'Varying thermal count'),
        'group_renewable': ('不同新能源数量', 'Varying renewable count'),
        'group_vpp': ('不同虚拟电厂数量', 'Varying VPP count'),
    }

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for g_idx, (gkey, (title_cn, title_en)) in enumerate(group_titles.items()):
        group = agent_count_results[gkey]
        ax_rew = axes[g_idx, 0]
        ax_price = axes[g_idx, 1]

        for c_idx, (name, data) in enumerate(group.items()):
            rewards = np.array(data['rewards'])
            prices = np.array(data['prices'])
            sm_rew = smooth(rewards, 150)
            sm_price = smooth(prices, 150)
            episodes = np.arange(len(sm_rew))

            ax_rew.plot(episodes, sm_rew / 1e4, color=colors[c_idx],
                        linewidth=1.5, label=name)
            ax_price.plot(episodes, sm_price, color=colors[c_idx],
                          linewidth=1.5, label=name)

        ax_rew.set_ylabel('总收益/万元', fontsize=10)
        ax_rew.set_title(f'{title_cn} - 总收益', fontsize=11)
        ax_rew.legend(fontsize=9)
        ax_rew.grid(True, alpha=0.3)

        ax_price.set_ylabel('结算电价/(元/(MW·h))', fontsize=10)
        ax_price.set_title(f'{title_cn} - 市场结算电价', fontsize=11)
        ax_price.legend(fontsize=9)
        ax_price.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('训练次数', fontsize=10)

    fig.suptitle('不同智能体数量下MA-CETD3算法收敛曲线对比\n'
                 'Convergence comparison under different agent counts',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig_agent_count_convergence.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved fig_agent_count_convergence.png")


def plot_multi_algo_agent_comparison(multi_algo_results, save_dir):
    ensure_dir(save_dir)
    agent_counts = multi_algo_results['agent_counts']
    algos = multi_algo_results['algorithms']
    data = multi_algo_results['data']

    n_counts = len(agent_counts)
    n_cols = 2
    n_rows = int(np.ceil(n_counts / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.8 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    algo_colors = {'MA-CETD3': '#2196F3', 'MATD3': '#FF9800'}
    algo_styles = {'MA-CETD3': '-', 'MATD3': '--'}

    for idx, n in enumerate(agent_counts):
        ax = axes[idx]
        for algo in algos:
            rewards = np.array(data[algo][n]['rewards'])
            smoothed = smooth_window(rewards, 100)
            episodes = np.arange(len(smoothed))
            ax.plot(episodes, smoothed / 1e4, color=algo_colors[algo],
                    linestyle=algo_styles[algo], linewidth=1.5, label=algo)
        ax.set_title(f'{n}个智能体', fontsize=11)
        ax.set_ylabel('总收益/万元', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for ax in axes[n_counts:]:
        ax.axis('off')

    for ax in axes[max(0, n_counts - n_cols):n_counts]:
        ax.set_xlabel('训练次数', fontsize=10)

    fig.suptitle('不同算法在不同智能体数量下收敛情况对比\n'
                 'Algorithm convergence comparison under different agent counts',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig_multi_algo_agent_comparison.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved fig_multi_algo_agent_comparison.png")

    _generate_algo_agent_table(multi_algo_results, save_dir)


def _generate_algo_agent_table(multi_algo_results, save_dir):
    agent_counts = multi_algo_results['agent_counts']
    data = multi_algo_results['data']

    print("\n" + "="*75)
    print("不同算法不同智能体数量下性能对比")
    print("="*75)
    header = f"{'算法':<15}{'智能体数':>10}{'收敛轮次':>10}{'最终收益/万元':>15}{'训练时间/s':>12}"
    print(header)
    print("-"*75)

    lines = [header, "-"*75]
    for algo in ['MA-CETD3', 'MATD3']:
        for n in agent_counts:
            d = data[algo][n]
            row = f"{algo:<15}{n:>10}{d['conv_ep']:>10}{d['final_reward']/1e4:>15.2f}{d['training_time']:>12.1f}"
            print(row)
            lines.append(row)

    with open(os.path.join(save_dir, 'table_algo_agent_comparison.txt'), 'w',
              encoding='utf-8') as f:
        f.write("不同算法不同智能体数量下性能对比\n\n")
        f.write("\n".join(lines) + "\n")
    print("  Saved table_algo_agent_comparison.txt")


def plot_carbon_green_sensitivity(sensitivity_results, save_dir):
    ensure_dir(save_dir)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    cp = np.array(sensitivity_results['carbon_prices'])
    gp = np.array(sensitivity_results['green_prices'])
    CP, GP = np.meshgrid(cp, gp)
    Z = sensitivity_results['elec_prices'].T

    surf = ax.plot_surface(CP, GP, Z, cmap='plasma',
                           edgecolor='grey', linewidth=0.3, alpha=0.85)
    ax.set_xlabel('碳价/(元/t)', fontsize=10, labelpad=10)
    ax.set_ylabel('绿证价格/(元/张)', fontsize=10, labelpad=10)
    ax.set_zlabel('结算电价/(元/(MW·h))', fontsize=10, labelpad=10)

    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    ax.set_title('不同碳价与绿证价格下市场结算电价\n'
                 'Electricity settlement price under different carbon & green prices',
                 fontsize=12, pad=20)
    plt.savefig(os.path.join(save_dir, 'fig_carbon_green_sensitivity.png'),
                bbox_inches='tight')
    plt.close()
    print("  Saved fig_carbon_green_sensitivity.png")


def generate_all_plots(scenario_results, algo_results, quota_results,
                       penalty_results, save_dir,
                       agent_count_results=None,
                       multi_algo_results=None,
                       sensitivity_results=None):
    print("\n" + "="*60)
    print("Generating all figures and tables...")
    print("="*60)

    plot_training_rewards(scenario_results, save_dir)
    plot_scenario_comparison(scenario_results, save_dir)
    plot_vpp_internal(scenario_results, save_dir)
    plot_algorithm_comparison(algo_results, save_dir)
    plot_bidding_convergence(algo_results, save_dir)
    plot_carbon_green_prices(scenario_results, save_dir)
    generate_table1(scenario_results, save_dir)
    generate_table2(algo_results, save_dir)

    if quota_results is not None:
        plot_quota_analysis(quota_results, save_dir)
        plot_quota_carbon_price(quota_results, save_dir)
        plot_quota_green_price(quota_results, save_dir)

    if penalty_results is not None:
        plot_penalty_analysis(penalty_results, save_dir)
        plot_penalty_carbon_price(penalty_results, save_dir)
        plot_penalty_green_price(penalty_results, save_dir)

    if agent_count_results is not None:
        plot_agent_count_convergence(agent_count_results, save_dir)

    if multi_algo_results is not None:
        plot_multi_algo_agent_comparison(multi_algo_results, save_dir)

    if sensitivity_results is not None:
        plot_carbon_green_sensitivity(sensitivity_results, save_dir)

    print(f"\nAll outputs saved to: {save_dir}")
