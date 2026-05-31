# ================================================================
# Pure Future Forecast: ARIMA vs ARIMAX  (2025–2026)
# Trained on full dataset 1996–2024
# ================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 0. REPRODUCIBILITY
# ================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ================================================================
# 0b. PLOT STYLE
# ================================================================
rcParams['font.family']      = 'serif'
rcParams['font.serif']       = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size']        = 11
rcParams['axes.labelsize']   = 12
rcParams['axes.titlesize']   = 13
rcParams['xtick.labelsize']  = 10
rcParams['ytick.labelsize']  = 10
rcParams['legend.fontsize']  = 10
rcParams['figure.dpi']       = 100
rcParams['savefig.dpi']      = 600
rcParams['savefig.format']   = 'pdf'
rcParams['savefig.bbox']     = 'tight'
rcParams['axes.linewidth']   = 1.0
rcParams['grid.linewidth']   = 0.5
rcParams['lines.linewidth']  = 1.5
rcParams['lines.markersize'] = 6

COLOR_OBS    = '#2C3E50'
COLOR_ARIMA  = '#3498DB'
COLOR_ARIMAX = '#E74C3C'
COLOR_GRID   = '#BDC3C7'
SCENARIO_COLORS = {'SC': '#8E44AD', 'CM': '#E74C3C', 'ML': '#E67E22'}
SCENARIO_LABELS = {
    'SC': 'SILSO Standard Curve',
    'CM': 'SILSO Combined Method',
    'ML': 'SILSO McNish & Lincoln',
}

# ================================================================
# 1. CONFIGURATION
# ================================================================
MIN_SPEED            = 1000
MIN_WIDTH, MAX_WIDTH = 0, 360
YEAR_START           = 1996
YEAR_TRAIN_END       = 2024
FORECAST_YEARS       = [2025, 2026]
FORECAST_STEPS       = len(FORECAST_YEARS)

print("=" * 80)
print("PURE FUTURE FORECAST: ARIMA vs ARIMAX  (2025-2026)")
print(f"Training period : {YEAR_START}-{YEAR_TRAIN_END}")
print(f"Forecast horizon: {FORECAST_YEARS[0]}-{FORECAST_YEARS[-1]}")
print("=" * 80)

# ================================================================
# 2. LOAD & FILTER CME DATA
# ================================================================
print("\n[1/8] Loading and filtering CME data...")
df_cmes = pd.read_csv("datos_procesados_2025_11_30.csv", low_memory=False)
df_cmes['Fecha'] = pd.to_datetime(df_cmes['Fecha'], errors='coerce')
df_cmes[['Central','Ancho','Rapidez']] = (
    df_cmes[['Central','Ancho','Rapidez']].apply(pd.to_numeric, errors='coerce')
)
df_cmes['Year'] = df_cmes['Fecha'].dt.year

df_filt = df_cmes[
    (df_cmes['Rapidez'] >= MIN_SPEED) &
    (df_cmes['Ancho']   >= MIN_WIDTH) &
    (df_cmes['Ancho']   <= MAX_WIDTH)
].copy()

conteo = (
    df_filt.groupby('Year').size()
    .rename('CMEs')
    .reindex(range(YEAR_START, YEAR_TRAIN_END + 1), fill_value=0)
    .reset_index()
)
conteo.columns = ['Year', 'CMEs']
print(f"   CMEs after filtering : {len(df_filt)}")
print(f"   Years with zero CMEs : {(conteo['CMEs'] == 0).sum()}")

# ================================================================
# 3. LOAD HISTORICAL SUNSPOT DATA
# ================================================================
print("[2/8] Loading historical sunspot data...")
df_sn_hist = pd.read_csv(
    "SN_y_tot_V2.0.txt", sep=r'\s+', header=None,
    usecols=[0, 1], names=['Year', 'SSN']
)
df_sn_hist['Year'] = df_sn_hist['Year'].astype(int)
df_sn_hist = df_sn_hist[
    (df_sn_hist['Year'] >= YEAR_START) &
    (df_sn_hist['Year'] <= YEAR_TRAIN_END)
].copy()

# ================================================================
# 4. BUILD FUTURE SSN FROM SILSO PREDICTIONS
# ================================================================
print("[3/8] Building future SSN scenarios from SILSO predictions...")

silso_monthly = {
    'SC': [
        (2025, 9,113.5),(2025,10,116.1),(2025,11,113.1),(2025,12,111.9),
        (2026, 1,110.4),(2026, 2,108.9),(2026, 3,107.3),(2026, 4,105.6),
        (2026, 5,103.8),(2026, 6,101.8),(2026, 7, 99.8),(2026, 8, 97.6),
        (2026, 9, 95.5),(2026,10, 93.0),(2026,11, 90.2),(2026,12, 87.5),
    ],
    'CM': [
        (2025, 9,115.0),(2025,10,113.1),(2025,11,113.6),(2025,12,114.6),
        (2026, 1,114.9),(2026, 2,114.9),(2026, 3,114.5),(2026, 4,113.0),
        (2026, 5,111.5),(2026, 6,109.4),(2026, 7,107.0),(2026, 8,105.3),
        (2026, 9,104.3),(2026,10,101.7),(2026,11, 98.6),(2026,12, 95.4),
    ],
    'ML': [
        (2025, 9,114.6),(2025,10,111.8),(2025,11,109.5),(2025,12,108.0),
        (2026, 1,106.3),(2026, 2,103.8),(2026, 3,100.3),(2026, 4, 96.4),
        (2026, 5, 92.8),(2026, 6, 89.1),(2026, 7, 85.9),(2026, 8, 83.5),
        (2026, 9, 81.3),(2026,10, 78.7),(2026,11, 75.7),(2026,12, 72.6),
    ],
}

try:
    df_sn_monthly = pd.read_csv(
        "SN_m_tot_V2.0.txt", sep=r'\s+', header=None,
        usecols=[0, 1, 3], names=['Year','Month','SSN']
    )
    df_sn_monthly['Year']  = df_sn_monthly['Year'].astype(int)
    df_sn_monthly['Month'] = df_sn_monthly['Month'].astype(int)
    jan_aug_2025 = df_sn_monthly[
        (df_sn_monthly['Year'] == 2025) & (df_sn_monthly['Month'] <= 8)
    ]['SSN'].values
    use_monthly = len(jan_aug_2025) > 0
    if use_monthly:
        print(f"   Monthly SSN file found -- Jan-Aug 2025 mean: {jan_aug_2025.mean():.1f}")
except FileNotFoundError:
    print("   INFO: SN_m_tot_V2.0.txt not found. Using historical annual data.")
    use_monthly = False
except Exception as e:
    print(f"   ERROR unexpectedly loading monthly data: {e}")
    use_monthly = False

ssn_scenarios = {}
for scenario, monthly_data in silso_monthly.items():
    annual_ssn = {}
    for yr in FORECAST_YEARS:
        silso_vals = [v for (y, m, v) in monthly_data if y == yr]
        if yr == 2025:
            if use_monthly:
                annual_ssn[yr] = float(np.mean(np.concatenate([jan_aug_2025, silso_vals])))
            else:
                hist_2025 = df_sn_hist[df_sn_hist['Year'] == 2025]['SSN'].values
                if len(hist_2025) > 0:
                    annual_ssn[yr] = float(hist_2025[0])
                else:
                    annual_ssn[yr] = float(np.mean(silso_vals))
                    print(f"   WARNING {scenario}: 2025 uses only Sep-Dec SILSO months")
        else:
            annual_ssn[yr] = float(np.mean(silso_vals))
    ssn_scenarios[scenario] = annual_ssn

print("\n   Annual SSN projections by scenario:")
print(f"   {'Year':<8}", end="")
for s in ssn_scenarios:
    print(f"  {s:>8}", end="")
print()
for yr in FORECAST_YEARS:
    print(f"   {yr:<8}", end="")
    for s, vals in ssn_scenarios.items():
        print(f"  {vals[yr]:>8.1f}", end="")
    print()

# ================================================================
# 5. MERGE & BUILD TRAINING SERIES
# ================================================================
print("\n[4/8] Building training series...")
df_merged = pd.merge(df_sn_hist, conteo, on='Year', how='inner')

missing = set(range(YEAR_START, YEAR_TRAIN_END + 1)) - set(df_merged['Year'])
if missing:
    raise ValueError(f"Year gaps in training data: {sorted(missing)}")

index_all = pd.to_datetime([str(y) for y in df_merged['Year']])
endog_all  = pd.Series(df_merged['CMEs'].values, index=index_all)
exog_all   = pd.DataFrame(df_merged['SSN'].values,
                           index=index_all, columns=['SSN'])

print(f"   Training observations: {len(endog_all)}")
print(f"   CME  -- mean: {endog_all.mean():.1f}  std: {endog_all.std():.1f}")
print(f"   SSN  -- mean: {exog_all['SSN'].mean():.1f}  std: {exog_all['SSN'].std():.1f}")

# ================================================================
# 6. STATIONARITY TEST -> DIFFERENCING ORDER
# ================================================================
print("\n" + "=" * 80)
print("STATIONARITY ANALYSIS (ADF TEST)")
print("=" * 80)

def adf_test(series, label):
    result = adfuller(series, autolag='AIC')
    stat, pval = result[0], result[1]
    cv = result[4]
    print(f"\n   {label}")
    print(f"   ADF statistic : {stat:.4f}")
    print(f"   p-value       : {pval:.4f}")
    print(f"   Critical values: 1%={cv['1%']:.3f}  5%={cv['5%']:.3f}  10%={cv['10%']:.3f}")
    stationary = pval < 0.05
    print(f"   -> {'STATIONARY' if stationary else 'NON-STATIONARY'} at 5% significance")
    return stationary, pval

stationary_0, pval_0 = adf_test(endog_all, "Level series (d=0)")

if stationary_0:
    d_suggested = 0
    print(f"\n   No differencing needed -- using d=0")
else:
    diff1 = endog_all.diff().dropna()
    stationary_1, pval_1 = adf_test(diff1, "First difference (d=1)")
    if stationary_1:
        d_suggested = 1
        print(f"\n   First difference is stationary -- using d=1")
    else:
        diff2 = diff1.diff().dropna()
        stationary_2, pval_2 = adf_test(diff2, "Second difference (d=2)")
        d_suggested = 2
        print(f"\n   {'Using d=2 (stationary)' if stationary_2 else 'Using d=2 (still non-stationary -- consider transformation)'}")

print(f"\n   Differencing order selected for ARIMA: d = {d_suggested}")

max_lags_lb = min(10, len(endog_all) // 4)
lb = acorr_ljungbox(endog_all, lags=max_lags_lb, return_df=True)
lb_pval = lb['lb_pvalue'].iloc[-1]
print(f"\n   LJUNG-BOX TEST (lag {max_lags_lb}): p = {lb_pval:.4f}")
print(f"   -> {'Significant autocorrelation -- modeling appropriate' if lb_pval < 0.05 else 'White noise -- forecasting may be difficult'}")
print("=" * 80)

# ================================================================
# 7. FIT MODELS ON FULL TRAINING DATA
# ================================================================
print(f"\n[5/8] Fitting models...")

max_order = min(5, len(endog_all) // 3)

# --- ARIMA ---
print(f"   Selecting ARIMA order (forced d={d_suggested})...")
arima_auto = auto_arima(
    endog_all, seasonal=False, trace=False,
    error_action='ignore', suppress_warnings=True,
    stepwise=False, random_state=RANDOM_SEED,
    start_p=0, start_q=0, max_p=max_order, max_q=max_order,
    information_criterion='aic',
    d=d_suggested
)
orden_arima = arima_auto.order
print(f"   ARIMA order selected : {orden_arima}")

modelo_arima    = SARIMAX(endog_all, order=orden_arima,
                          enforce_stationarity=False, enforce_invertibility=False)
resultado_arima = modelo_arima.fit(disp=False)

# --- ARIMAX ---
print("   Selecting ARIMAX order (allowing independent d selection)...")
arimax_auto = auto_arima(
    endog_all, X=exog_all, seasonal=False, trace=False,
    error_action='ignore', suppress_warnings=True,
    stepwise=False, random_state=RANDOM_SEED,
    start_p=0, start_q=0, max_p=max_order, max_q=max_order,
    information_criterion='aic',
    d=None
)
orden_arimax = arimax_auto.order
print(f"   ARIMAX order selected: {orden_arimax}")
if orden_arimax == orden_arima:
    print("   NOTE: same (p,d,q) as ARIMA.")

modelo_arimax    = SARIMAX(endog_all, exog=exog_all, order=orden_arimax,
                           enforce_stationarity=True, enforce_invertibility=True)
resultado_arimax = modelo_arimax.fit(disp=False)

fitted_arima  = resultado_arima.fittedvalues
fitted_arimax = resultado_arimax.fittedvalues

y_all          = endog_all.values
rmse_arima_is  = np.sqrt(mean_squared_error(y_all, fitted_arima.values))
mae_arima_is   = mean_absolute_error(y_all, fitted_arima.values)
r2_arima_is    = r2_score(y_all, fitted_arima.values)

rmse_arimax_is = np.sqrt(mean_squared_error(y_all, fitted_arimax.values))
mae_arimax_is  = mean_absolute_error(y_all, fitted_arimax.values)
r2_arimax_is   = r2_score(y_all, fitted_arimax.values)

print(f"\n   {'Metric':<12} {'ARIMA':>10} {'ARIMAX':>10}")
print(f"   {'-'*32}")
print(f"   {'AIC':<12} {resultado_arima.aic:>10.2f} {resultado_arimax.aic:>10.2f}")
print(f"   {'RMSE (IS)':<12} {rmse_arima_is:>10.3f} {rmse_arimax_is:>10.3f}")
print(f"   {'MAE  (IS)':<12} {mae_arima_is:>10.3f}  {mae_arimax_is:>10.3f}")
print(f"   {'R2   (IS)':<12} {r2_arima_is:>10.4f} {r2_arimax_is:>10.4f}")

# ================================================================
# 8. GENERATE FORECASTS PER SILSO SCENARIO
# ================================================================
print("\n[6/8] Generating forecasts 2025-2026 per SILSO scenario...")

forecast_index = pd.to_datetime([str(y) for y in FORECAST_YEARS])
results = {}

for scenario, ssn_dict in ssn_scenarios.items():
    exog_future = pd.DataFrame(
        {'SSN': [ssn_dict[y] for y in FORECAST_YEARS]},
        index=forecast_index
    )

    fc_arima     = resultado_arima.get_forecast(steps=FORECAST_STEPS)
    fc_arima_m   = fc_arima.predicted_mean.values
    ci_arima_raw = fc_arima.conf_int(alpha=0.05)
    ci_arima_lo  = np.clip(ci_arima_raw.iloc[:, 0].values, 0, None)
    ci_arima_hi  = ci_arima_raw.iloc[:, 1].values

    fc_arimax     = resultado_arimax.get_forecast(steps=FORECAST_STEPS, exog=exog_future)
    fc_arimax_m   = fc_arimax.predicted_mean.values
    ci_arimax_raw = fc_arimax.conf_int(alpha=0.05)
    ci_arimax_lo  = np.clip(ci_arimax_raw.iloc[:, 0].values, 0, None)
    ci_arimax_hi  = ci_arimax_raw.iloc[:, 1].values

    results[scenario] = {
        'ssn'        : [ssn_dict[y] for y in FORECAST_YEARS],
        'arima_mean' : fc_arima_m,
        'arima_lo'   : ci_arima_lo,
        'arima_hi'   : ci_arima_hi,
        'arimax_mean': fc_arimax_m,
        'arimax_lo'  : ci_arimax_lo,
        'arimax_hi'  : ci_arimax_hi,
    }

    print(f"\n   Scenario {scenario}  (SSN 2025={ssn_dict[2025]:.1f}, 2026={ssn_dict[2026]:.1f})")
    for i, yr in enumerate(FORECAST_YEARS):
        print(f"     {yr}  ARIMA : {fc_arima_m[i]:5.1f} "
              f"[{ci_arima_lo[i]:.1f}, {ci_arima_hi[i]:.1f}]   "
              f"ARIMAX: {fc_arimax_m[i]:5.1f} "
              f"[{ci_arimax_lo[i]:.1f}, {ci_arimax_hi[i]:.1f}]")

arima_mean = results['SC']['arima_mean']
arima_lo   = results['SC']['arima_lo']
arima_hi   = results['SC']['arima_hi']

# ================================================================
# 9. PUBLICATION-QUALITY FIGURE
# ================================================================
print("\n[7/8] Generating figure...")

hist_years = df_merged['Year'].values
hist_cmes  = df_merged['CMEs'].values

fig, axes = plt.subplots(2, 1, figsize=(13, 11),
                          gridspec_kw={'height_ratios': [3, 1.2]})
ax  = axes[0]
axs = axes[1]

# ── Forecast background shading ────────────────────────────────────────────
ax.axvspan(2024.5, 2026.5, alpha=0.06, color='gold', zorder=0)

# ── In-sample fitted values ────────────────────────────────────────────────
ax.plot(hist_years, fitted_arima.values,
        linestyle='--', linewidth=1.4, color=COLOR_ARIMA, alpha=0.50, zorder=2,
        label=f'ARIMA{orden_arima} ajustado  '
              f'(R$^2$={r2_arima_is:.3f}, RMSE={rmse_arima_is:.1f})')

ax.plot(hist_years, fitted_arimax.values,
        linestyle='-.', linewidth=1.4, color=COLOR_ARIMAX, alpha=0.50, zorder=2,
        label=f'ARIMAX{orden_arimax} ajustado '
              f'(R$^2$={r2_arimax_is:.3f}, RMSE={rmse_arimax_is:.1f})')

# ── ARIMA forecast (figura padre — se mantiene) ───────────────────────────
ax.fill_between(FORECAST_YEARS, arima_lo, arima_hi,
                alpha=0.15, color=COLOR_ARIMA, zorder=1, label='ARIMA 95% CI')
ax.plot(FORECAST_YEARS, arima_mean,
        linestyle='--', linewidth=2.4, color=COLOR_ARIMA,
        marker='D', markersize=7,
        markeredgecolor='white', markeredgewidth=0.8,
        zorder=5, label=f'ARIMA{orden_arima} predicción')

# ── ARIMAX forecasts (one per scenario) ───────────────────────────────────
for scenario, res in results.items():
    color = SCENARIO_COLORS[scenario]
    ax.fill_between(FORECAST_YEARS, res['arimax_lo'], res['arimax_hi'],
                    alpha=0.12, color=color, zorder=1)
    ax.plot(FORECAST_YEARS, res['arimax_mean'],
            linestyle='-.', linewidth=2.2, color=color,
            marker='s', markersize=7,
            markeredgecolor='white', markeredgewidth=0.8,
            zorder=5,
            label=f"ARIMAX{orden_arimax} -- {SCENARIO_LABELS[scenario]}")

# ── Observed data ──────────────────────────────────────────────────────────
ax.plot(hist_years, hist_cmes,
        marker='o', linestyle='-', linewidth=2.0, markersize=6,
        color=COLOR_OBS, zorder=6,
        markeredgewidth=0.8, markeredgecolor='white',
        label=' CMEs observadas (1996-2024)')

# ── Training boundary ──────────────────────────────────────────────────────
ax.axvline(x=2024.5, color='black', linestyle=':',
           linewidth=1.8, alpha=0.5, zorder=7, label='Límite de entrenamiento')

ax.set_ylabel('Conteo de CMEs  (eventos por año)', fontsize=12)

ax.set_xlim(1995.5, 2027.2)
ax.set_ylim(bottom=0)
ax.set_xticks(range(1996, 2027, 2))
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=COLOR_GRID)
ax.set_axisbelow(True)

legend = ax.legend(
    loc='upper left', frameon=True, framealpha=0.9,
    edgecolor='none', facecolor='#F8F9F9',
    fontsize=9, labelspacing=0.5,
    title="Comparación de modelos", title_fontsize=10
)

# ================================================================
# ZOOM INSET on forecast window 
# ================================================================
ax_inset = inset_axes(
    ax,
    width='50%',
    height='52%',
    loc='upper center',
    bbox_to_anchor=(0.10, 0.0, 0.9, 1.0),
    bbox_transform=ax.transAxes,
    borderpad=0,
)

# ── Background and border ──────────────────────────────────────────────────
ax_inset.set_facecolor('#FAFAFA')
for spine in ax_inset.spines.values():
    spine.set_edgecolor('#888888')
    spine.set_linewidth(0.8)

N_TAIL = 2
tail_years = hist_years[-N_TAIL:]
tail_cmes  = hist_cmes[-N_TAIL:]
ax_inset.plot(tail_years, tail_cmes,
              marker='o', linestyle='-', linewidth=1.8, markersize=5,
              color=COLOR_OBS, zorder=6,
              markeredgewidth=0.6, markeredgecolor='white')

# ── ARIMAX scenarios + CI ─────────────────────────────────────────────────
for scenario, res in results.items():
    color = SCENARIO_COLORS[scenario]
    ax_inset.fill_between(FORECAST_YEARS, res['arimax_lo'], res['arimax_hi'],
                          alpha=0.15, color=color, zorder=1)
    ax_inset.plot(FORECAST_YEARS, res['arimax_mean'],
                  linestyle='-.', linewidth=2.0, color=color,
                  marker='s', markersize=7,
                  markeredgecolor='white', markeredgewidth=0.8, zorder=5)

# ── Training boundary in inset ────────────────────────────────────────────
ax_inset.axvline(x=2024.5, color='black', linestyle=':', linewidth=1.4, alpha=0.5, zorder=7)
ax_inset.axvspan(2024.5, 2026.7, alpha=0.06, color='gold', zorder=0)


all_fc_vals = (
    [v for res in results.values()
     for v in list(res['arimax_mean']) + list(res['arimax_lo']) + list(res['arimax_hi'])]
    + list(tail_cmes)
)
y_pad = (max(all_fc_vals) - min(all_fc_vals)) * 0.05
inset_ylo = max(0, min(all_fc_vals) - y_pad)
inset_yhi = max(all_fc_vals) + y_pad * 1.5

ax_inset.set_xlim(2023 - 0.3, 2026 + 0.8)
ax_inset.set_ylim(inset_ylo, inset_yhi)
ax_inset.set_xticks([2023, 2024] + FORECAST_YEARS)
ax_inset.tick_params(axis='both', labelsize=7.5)
ax_inset.grid(True, alpha=0.3, linestyle='-', linewidth=0.4, color=COLOR_GRID)
ax_inset.set_axisbelow(True)

# ================================================================
# SSN PANEL (lower panel)
# ================================================================
axs.fill_between(hist_years, 0, df_merged['SSN'].values,
                 color='#BDC3C7', alpha=0.2, zorder=1)
axs.plot(hist_years, df_merged['SSN'].values,
         color='#7F8C8D', linewidth=1.6, linestyle='-',
         marker='o', markersize=3.5,
         markeredgewidth=0.5, markeredgecolor='white',
         label=' SSN observado', zorder=3)

for scenario, ssn_dict in ssn_scenarios.items():
    color  = SCENARIO_COLORS[scenario]
    ssn_fc = [ssn_dict[y] for y in FORECAST_YEARS]
    axs.plot([hist_years[-1], FORECAST_YEARS[0]],
             [df_merged['SSN'].values[-1], ssn_fc[0]],
             linestyle=':', linewidth=1.0, color=color, alpha=0.45, zorder=2)
    axs.plot(FORECAST_YEARS, ssn_fc,
             linestyle='-.', linewidth=1.8, color=color,
             marker='s', markersize=5,
             markeredgewidth=0.6, markeredgecolor='white',
             label=SCENARIO_LABELS[scenario], zorder=4)

axs.axvspan(2024.5, 2026.5, alpha=0.06, color='gold', zorder=0)
axs.axvline(x=2024.5, color='black', linestyle=':',
            linewidth=1.8, alpha=0.5, zorder=5)
axs.set_ylabel('Número de manchas solares', fontsize=11)
axs.set_xlabel('Año', fontsize=12)
axs.set_xlim(1995.5, 2027.2)
axs.set_ylim(bottom=0)
axs.set_xticks(range(1996, 2027, 2))
axs.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=COLOR_GRID)
axs.set_axisbelow(True)
legend_ssn = axs.legend(
    loc='upper center', frameon=True, fancybox=False, shadow=False,
    framealpha=0.75, edgecolor='#888888', facecolor='white',
    borderpad=0.6, labelspacing=0.3, fontsize=8.5
)
legend_ssn.get_frame().set_linewidth(0.6)

plt.tight_layout(h_pad=1.8)
plt.savefig('forecast_pure_2025_2026.pdf', dpi=600, bbox_inches='tight')
print("   Figure saved: 'forecast_pure_2025_2026.pdf'")
plt.close()

# ================================================================
# 10. SAVE NUMERICAL RESULTS
# ================================================================
print("\n[8/8] Saving results...")
rows = []
for scenario, res in results.items():
    for i, yr in enumerate(FORECAST_YEARS):
        rows.append({
            'Year'              : yr,
            'SILSO_scenario'    : scenario,
            'SSN_projected'     : round(res['ssn'][i], 1),
            'ARIMA_forecast'    : round(res['arima_mean'][i], 2),
            'ARIMA_CI_lower_95' : round(res['arima_lo'][i],   2),
            'ARIMA_CI_upper_95' : round(res['arima_hi'][i],   2),
            'ARIMAX_forecast'   : round(res['arimax_mean'][i], 2),
            'ARIMAX_CI_lower_95': round(res['arimax_lo'][i],   2),
            'ARIMAX_CI_upper_95': round(res['arimax_hi'][i],   2),
        })

df_out = pd.DataFrame(rows)
df_out.to_csv('forecast_pure_2025_2026.csv', index=False)
print("   Numerical results: 'forecast_pure_2025_2026.csv'")
print()
print(df_out.to_string(index=False))

with open('forecast_pure_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("PURE FUTURE FORECAST -- ARIMA vs ARIMAX  (2025-2026)\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Training period : {YEAR_START}-{YEAR_TRAIN_END} ({len(endog_all)} obs)\n")
    f.write(f"Forecast horizon: 2025-2026\n")
    f.write(f"CME filter      : speed >= {MIN_SPEED} km/s, "
            f"{MIN_WIDTH} <= width <= {MAX_WIDTH} deg\n\n")
    f.write("STATIONARITY:\n")
    f.write(f"  ADF level series : p={pval_0:.4f}  "
            f"-> {'stationary' if stationary_0 else 'non-stationary'}\n")
    f.write(f"  Differencing used (ARIMA) : d={d_suggested}\n")
    f.write(f"  Differencing used (ARIMAX): d={orden_arimax[1]}\n")
    f.write(f"  Ljung-Box p      : {lb_pval:.4f}\n\n")
    f.write("MODELS:\n")
    f.write(f"  ARIMA{orden_arima}   AIC={resultado_arima.aic:.2f}  "
            f"RMSE={rmse_arima_is:.3f}  R2={r2_arima_is:.4f}\n")
    f.write(f"  ARIMAX{orden_arimax}  AIC={resultado_arimax.aic:.2f}  "
            f"RMSE={rmse_arimax_is:.3f}  R2={r2_arimax_is:.4f}\n\n")
    f.write("SSN SCENARIOS:\n")
    for sc, sd in ssn_scenarios.items():
        f.write(f"  {sc}: 2025={sd[2025]:.1f}  2026={sd[2026]:.1f}\n")
    f.write("\nFORECAST RESULTS (95% CI, lower bound clipped at 0):\n\n")
    f.write(df_out.to_string(index=False))
    f.write("\n\nNOTES:\n")
    f.write("  - ARIMA forecast is identical across SILSO scenarios.\n")
    f.write("  - For 2025: annual SSN = Jan-Aug historical + Sep-Dec SILSO.\n")
    f.write(f"  - numpy.random.seed = {RANDOM_SEED}\n")
    f.write("=" * 80 + "\n")

print("   Summary: 'forecast_pure_summary.txt'")
print("\n" + "=" * 80)
print("FORECAST COMPLETED")
print("=" * 80)