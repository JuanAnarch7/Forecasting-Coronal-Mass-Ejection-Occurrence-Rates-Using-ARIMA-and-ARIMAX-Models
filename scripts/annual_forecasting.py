# ================================================================
# ARIMA vs ARIMAX Comparison for Annual CME Forecasting
# ================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 0. GLOBAL REPRODUCIBILITY SEED
# ================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ================================================================
# 0b. PLOT STYLE CONFIGURATION
# ================================================================
rcParams['font.family']       = 'serif'
rcParams['font.serif']        = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size']         = 11
rcParams['axes.labelsize']    = 12
rcParams['axes.titlesize']    = 13
rcParams['xtick.labelsize']   = 10
rcParams['ytick.labelsize']   = 10
rcParams['legend.fontsize']   = 10
rcParams['figure.titlesize']  = 14
rcParams['figure.dpi']        = 100
rcParams['savefig.dpi']       = 600
rcParams['savefig.format']    = 'pdf'
rcParams['savefig.bbox']      = 'tight'
rcParams['axes.linewidth']    = 1.0
rcParams['grid.linewidth']    = 0.5
rcParams['lines.linewidth']   = 1.5
rcParams['lines.markersize']  = 6

COLOR_OBSERVED = '#2C3E50'
COLOR_ARIMA    = '#3498DB'
COLOR_ARIMAX   = '#E74C3C'
COLOR_GRID     = '#BDC3C7'
COLOR_TRAIN    = '#95A5A6'

# ================================================================
# 1. CONFIGURATION PARAMETERS
# ================================================================
MIN_SPEED              = 450
MIN_WIDTH, MAX_WIDTH   = 120, 359
YEAR_START, YEAR_END   = 1996, 2024
TRAIN_END_YEAR         = 2019   # 80 % of data (1996–2019 = 23 years)
TEST_START_YEAR        = 2020   # 20 % of data (2020–2024 = 5 years)

print("=" * 80)
print("ARIMA vs ARIMAX MODEL COMPARISON")
print("=" * 80)
print(f"\nTrain period : {YEAR_START}–{TRAIN_END_YEAR}")
print(f"Test period  : {TEST_START_YEAR}–{YEAR_END}")

# ================================================================
# 2. LOAD CME DATA
# ================================================================
print("\n[1/9] Loading CME dataset...")
df_cmes = pd.read_csv("datos_procesados_2025_11_30.csv", low_memory=False)
df_cmes['Fecha'] = pd.to_datetime(df_cmes['Fecha'], errors='coerce')
df_cmes[['Central', 'Ancho', 'Rapidez']] = (
    df_cmes[['Central', 'Ancho', 'Rapidez']].apply(pd.to_numeric, errors='coerce')
)
df_cmes['Year'] = df_cmes['Fecha'].dt.year

# ================================================================
# 3. FILTER CME EVENTS AND HANDLE MISSING YEARS
# ================================================================
print("[2/9] Filtering CME events and handling missing years...")
df_cmes_filtrado = df_cmes[
    (df_cmes['Rapidez'] >= MIN_SPEED) &
    (df_cmes['Ancho']   >= MIN_WIDTH)  &
    (df_cmes['Ancho']   <= MAX_WIDTH)
].copy()

conteo_anual = (
    df_cmes_filtrado
    .groupby('Year')
    .size()
    .rename('CMEs_filtradas')
    .reindex(range(YEAR_START, YEAR_END + 1), fill_value=0)
    .reset_index()
)
conteo_anual.columns = ['Year', 'CMEs_filtradas']

print(f"   Total CMEs: {len(df_cmes)} → Filtered: {len(df_cmes_filtrado)}")
print(f"   Years with zero CMEs: {(conteo_anual['CMEs_filtradas'] == 0).sum()}")

# ================================================================
# 4. LOAD SUNSPOT DATA
# ================================================================
print("[3/9] Loading sunspot numbers...")
df_sn = pd.read_csv(
    "SN_y_tot_V2.0.txt", sep=r'\s+', header=None, usecols=[0, 1],
    names=['Year', 'SunspotNumber']
)
df_sn['Year'] = df_sn['Year'].astype(int)
df_sn = df_sn[(df_sn['Year'] >= YEAR_START) & (df_sn['Year'] <= YEAR_END)]

# ================================================================
# 5. MERGE DATASETS
# ================================================================
print("\n[4/9] Merging datasets...")
df_merged = pd.merge(df_sn, conteo_anual, on='Year', how='inner')

expected_years = set(range(YEAR_START, YEAR_END + 1))
observed_years = set(df_merged['Year'])
missing_after_merge = expected_years - observed_years
if missing_after_merge:
    raise ValueError(
        f"Year gaps after merge — missing: {sorted(missing_after_merge)}. "
        "Check sunspot data coverage."
    )

print(f"   Temporal range   : {df_merged['Year'].min()}–{df_merged['Year'].max()}")
print(f"   Total observations: {len(df_merged)}")
print(f"   NaN in CMEs      : {df_merged['CMEs_filtradas'].isna().sum()}")

# ================================================================
# 6. TRAIN / TEST SPLIT
# ================================================================
print("\n[5/9] Splitting data into train and test sets...")
df_train = df_merged[df_merged['Year'] <= TRAIN_END_YEAR].copy()
df_test  = df_merged[df_merged['Year'] >= TEST_START_YEAR].copy()

print(f"   Train : {len(df_train)} obs  ({df_train['Year'].min()}–{df_train['Year'].max()})")
print(f"   Test  : {len(df_test)} obs  ({df_test['Year'].min()}–{df_test['Year'].max()})")

index_train = pd.to_datetime([str(y) for y in df_train['Year']])
index_test  = pd.to_datetime([str(y) for y in df_test['Year']])

endog_train = pd.Series(df_train['CMEs_filtradas'].values, index=index_train)
endog_test  = pd.Series(df_test['CMEs_filtradas'].values,  index=index_test)

exog_train = pd.DataFrame(
    df_train['SunspotNumber'].values, index=index_train, columns=['Sunspots']
)
exog_test = pd.DataFrame(
    df_test['SunspotNumber'].values,  index=index_test,  columns=['Sunspots']
)

# ================================================================
# 7. STATIONARITY AND WHITE-NOISE TESTS
# ================================================================
print("\n" + "="*80)
print("PRELIMINARY STATISTICAL TESTS")
print("="*80)
print("[6/9] Testing stationarity (ADF) and serial correlation (Ljung-Box)...")

adf_result = adfuller(endog_train, autolag='AIC')
print(f"\n   AUGMENTED DICKEY-FULLER TEST (Stationarity):")
print(f"   ADF Statistic : {adf_result[0]:.4f}")
print(f"   p-value       : {adf_result[1]:.4f}")
print(f"   Critical values:")
for key, val in adf_result[4].items():
    print(f"      {key}: {val:.4f}")

if adf_result[1] < 0.05:
    print("   ✓ Series is STATIONARY (p < 0.05) — no differencing needed")
    diferenciacion_sugerida = 0
else:
    print("   ⚠ Series is NON-STATIONARY (p ≥ 0.05) — differencing required")
    diff1 = endog_train.diff().dropna()
    adf_d1 = adfuller(diff1, autolag='AIC')
    print(f"\n   ADF on first difference: stat={adf_d1[0]:.4f}, p={adf_d1[1]:.4f}")
    if adf_d1[1] < 0.05:
        print("   ✓ First difference stationary — d=1 recommended")
        diferenciacion_sugerida = 1
    else:
        diff2 = diff1.diff().dropna()
        adf_d2 = adfuller(diff2, autolag='AIC')
        print(f"\n   ADF on second difference: stat={adf_d2[0]:.4f}, p={adf_d2[1]:.4f}")
        diferenciacion_sugerida = 2 if adf_d2[1] < 0.05 else 2
        print("   ⚠ d=2 recommended (or consider transformation)")

max_lags_lb = min(10, len(endog_train) // 4)
lb_result = acorr_ljungbox(endog_train, lags=max_lags_lb, return_df=True)
print(f"\n   LJUNG-BOX TEST (White Noise — lag {max_lags_lb}):")
print(f"   p-value: {lb_result['lb_pvalue'].iloc[-1]:.4f}")
if lb_result['lb_pvalue'].iloc[-1] < 0.05:
    print("    Series shows SIGNIFICANT AUTOCORRELATION → modeling appropriate")
else:
    print("    Series behaves like WHITE NOISE → forecasting will be difficult")

print("="*80)

# ================================================================
# 8. ACF / PACF PLOTS
# ================================================================
print("\n[7/9] Generating ACF/PACF plots...")
max_lags_acf = min(10, len(endog_train) // 2 - 1)

fig_acf, axes_acf = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(endog_train,  lags=max_lags_acf, ax=axes_acf[0],
         color=COLOR_ARIMA, alpha=0.5)
axes_acf[0].set_title('Autocorrelation Function (ACF) — Training Data',
                       fontsize=12, fontweight='bold')
axes_acf[0].set_xlabel('Lag (years)', fontsize=11)
axes_acf[0].set_ylabel('ACF', fontsize=11)

plot_pacf(endog_train, lags=max_lags_acf, ax=axes_acf[1],
          color=COLOR_ARIMAX, alpha=0.5, method='ywm')
axes_acf[1].set_title('Partial Autocorrelation Function (PACF) — Training Data',
                       fontsize=12, fontweight='bold')
axes_acf[1].set_xlabel('Lag (years)', fontsize=11)
axes_acf[1].set_ylabel('PACF', fontsize=11)

plt.tight_layout()
plt.savefig('acf_pacf_annual_analysis.pdf', dpi=600, bbox_inches='tight')
print("   ✓ ACF/PACF saved: 'acf_pacf_annual_analysis.pdf'")
plt.close()

# ================================================================
# 9. MODEL 1 — ARIMA
# ================================================================
print("\n" + "="*80)
print("MODEL 1: ARIMA (WITHOUT EXOGENOUS VARIABLES)")
print("="*80)
print("[8a/9] Selecting optimal ARIMA order on training data...")

max_order = min(5, len(endog_train) // 3)

modelo_arima_auto = auto_arima(
    endog_train,
    seasonal=False,
    trace=False,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=False,
    random_state=RANDOM_SEED,
    start_p=0, start_q=0,
    max_p=max_order, max_q=max_order,
    information_criterion='aic',
    d=1
)
orden_arima = modelo_arima_auto.order
print(f"   Order selected: ARIMA{orden_arima}")

modelo_arima    = SARIMAX(endog_train, order=orden_arima,
                          enforce_stationarity=False, enforce_invertibility=False)
resultado_arima = modelo_arima.fit(disp=False)

pred_arima_train      = resultado_arima.get_prediction(
    start=endog_train.index[0], end=endog_train.index[-1]
)
pred_arima_train_mean = pred_arima_train.predicted_mean

forecast_arima_obj    = resultado_arima.get_forecast(steps=len(df_test))
forecast_arima_series = forecast_arima_obj.predicted_mean
ci_arima_raw          = forecast_arima_obj.conf_int(alpha=0.05)

ci_arima = ci_arima_raw.copy()
ci_arima.iloc[:, 0] = ci_arima.iloc[:, 0].clip(lower=0)

# Metrics
y_train_true          = endog_train.values
y_test_true           = endog_test.values
y_train_pred_arima    = pred_arima_train_mean.values
y_test_pred_arima     = forecast_arima_series.values

rmse_arima_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred_arima))
mae_arima_train  = mean_absolute_error(y_train_true, y_train_pred_arima)
r2_arima_train   = r2_score(y_train_true, y_train_pred_arima)

rmse_arima_test  = np.sqrt(mean_squared_error(y_test_true, y_test_pred_arima))
mae_arima_test   = mean_absolute_error(y_test_true, y_test_pred_arima)
r2_arima_test    = r2_score(y_test_true, y_test_pred_arima)

residuos_arima  = resultado_arima.resid
max_lags_res    = min(10, len(residuos_arima) // 4)
lb_arima        = acorr_ljungbox(residuos_arima, lags=max_lags_res, return_df=True)

print(f"\n   ARIMA METRICS — IN-SAMPLE (training):")
print(f"   RMSE : {rmse_arima_train:.3f}  MAE : {mae_arima_train:.3f}  R² : {r2_arima_train:.4f}")
print(f"\n   ARIMA METRICS — OUT-OF-SAMPLE (test) *** KEY ***:")
print(f"   RMSE : {rmse_arima_test:.3f}  MAE : {mae_arima_test:.3f}  R² : {r2_arima_test:.4f}")
print(f"   AIC  : {resultado_arima.aic:.2f}")
print(f"   Ljung-Box residuals p-value (lag {max_lags_res}): "
      f"{lb_arima['lb_pvalue'].iloc[-1]:.4f} "
      f"({'✓ white noise' if lb_arima['lb_pvalue'].iloc[-1] > 0.05 else '⚠ autocorrelation'})")

# ================================================================
# 10. MODEL 2 — ARIMAX  (FIX: pass exog to auto_arima)
# ================================================================
print("\n" + "="*80)
print("MODEL 2: ARIMAX (WITH SUNSPOT NUMBERS)")
print("="*80)
print("[8b/9] Selecting optimal ARIMAX order on training data...")

modelo_arimax_auto = auto_arima(
    endog_train,
    X=exog_train,                   
    seasonal=False,
    trace=False,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=False,
    random_state=RANDOM_SEED,
    start_p=0, start_q=0,
    max_p=max_order, max_q=max_order,
    information_criterion='aic',
    d=None
)
orden_arimax = modelo_arimax_auto.order
print(f"   Order selected: ARIMAX{orden_arimax}")
if orden_arimax == orden_arima:
    print("   NOTE: same order as ARIMA — sunspot regressor did not change "
          "the optimal p,q structure at this sample size.")

modelo_arimax = SARIMAX(
    endog_train, exog=exog_train, order=orden_arimax,
    enforce_stationarity=False, enforce_invertibility=False
)
resultado_arimax = modelo_arimax.fit(disp=False)

pred_arimax_train      = resultado_arimax.get_prediction(
    start=endog_train.index[0], end=endog_train.index[-1], exog=exog_train
)
pred_arimax_train_mean = pred_arimax_train.predicted_mean

forecast_arimax_obj    = resultado_arimax.get_forecast(steps=len(df_test),
                                                        exog=exog_test)
forecast_arimax_series = forecast_arimax_obj.predicted_mean
ci_arimax_raw          = forecast_arimax_obj.conf_int(alpha=0.05)

ci_arimax = ci_arimax_raw.copy()
ci_arimax.iloc[:, 0] = ci_arimax.iloc[:, 0].clip(lower=0)

# Metrics
y_train_pred_arimax = pred_arimax_train_mean.values
y_test_pred_arimax  = forecast_arimax_series.values

rmse_arimax_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred_arimax))
mae_arimax_train  = mean_absolute_error(y_train_true, y_train_pred_arimax)
r2_arimax_train   = r2_score(y_train_true, y_train_pred_arimax)

rmse_arimax_test  = np.sqrt(mean_squared_error(y_test_true, y_test_pred_arimax))
mae_arimax_test   = mean_absolute_error(y_test_true, y_test_pred_arimax)
r2_arimax_test    = r2_score(y_test_true, y_test_pred_arimax)

residuos_arimax = resultado_arimax.resid
lb_arimax       = acorr_ljungbox(residuos_arimax, lags=max_lags_res, return_df=True)

print(f"\n   ARIMAX METRICS — IN-SAMPLE (training):")
print(f"   RMSE : {rmse_arimax_train:.3f}  MAE : {mae_arimax_train:.3f}  R² : {r2_arimax_train:.4f}")
print(f"\n   ARIMAX METRICS — OUT-OF-SAMPLE (test) *** KEY ***:")
print(f"   RMSE : {rmse_arimax_test:.3f}  MAE : {mae_arimax_test:.3f}  R² : {r2_arimax_test:.4f}")
print(f"   AIC  : {resultado_arimax.aic:.2f}")
print(f"   Ljung-Box residuals p-value (lag {max_lags_res}): "
      f"{lb_arimax['lb_pvalue'].iloc[-1]:.4f} "
      f"({'✓ white noise' if lb_arimax['lb_pvalue'].iloc[-1] > 0.05 else '⚠ autocorrelation'})")

# ================================================================
# 11. MODEL COMPARISON
# ================================================================
print("\n" + "="*80)
print("MODEL COMPARISON (OUT-OF-SAMPLE PERFORMANCE)")
print("="*80)

mejora_rmse = ((rmse_arima_test - rmse_arimax_test) / rmse_arima_test * 100
               if rmse_arima_test > 0 else 0)
mejora_mae  = ((mae_arima_test  - mae_arimax_test)  / mae_arima_test  * 100
               if mae_arima_test  > 0 else 0)
mejora_r2   = ((r2_arimax_test  - r2_arima_test)    / abs(r2_arima_test) * 100
               if r2_arima_test  != 0 else 0)
mejora_aic  = resultado_arima.aic - resultado_arimax.aic

print(f"\n{'Metric':<20} {'ARIMA':<15} {'ARIMAX':<15} {'Improvement':<15}")
print("-" * 65)
print(f"{'RMSE (test)':<20} {rmse_arima_test:<15.3f} {rmse_arimax_test:<15.3f} {mejora_rmse:>+.2f}%")
print(f"{'MAE  (test)':<20} {mae_arima_test:<15.3f}  {mae_arimax_test:<15.3f}  {mejora_mae:>+.2f}%")
print(f"{'R²   (test)':<20} {r2_arima_test:<15.4f} {r2_arimax_test:<15.4f} {mejora_r2:>+.2f}%")
print(f"{'AIC':<20} {resultado_arima.aic:<15.2f} {resultado_arimax.aic:<15.2f} {mejora_aic:>+.2f}")
print("-" * 65)

if rmse_arimax_test < rmse_arima_test and resultado_arimax.aic < resultado_arima.aic:
    print(" ARIMAX is SUPERIOR: lower test error AND lower AIC")
elif rmse_arimax_test < rmse_arima_test:
    print(" ARIMAX has lower test error but comparable AIC — sunspot numbers help but add complexity")
else:
    print(" ARIMA is comparable or superior — sunspot numbers do not substantially improve the model")

print("="*80)

# ================================================================
# 12. VISUALISATION
# ================================================================

train_years = df_train['Year'].values
test_years  = df_test['Year'].values

# ---------- FIGURE 1: FORECAST COMPARISON ----------
fig1, ax1 = plt.subplots(figsize=(13, 6.5))


ax1.axvspan(TEST_START_YEAR - 0.5, YEAR_END + 0.5,
            alpha=0.07, color='gray', zorder=0, label='_nolegend_')

# Confidence intervals 
ax1.fill_between(test_years,
                 ci_arima.iloc[:, 0], ci_arima.iloc[:, 1],
                 alpha=0.18, color=COLOR_ARIMA, zorder=1,
                 label='ARIMA 95% CI')
ax1.fill_between(test_years,
                 ci_arimax.iloc[:, 0], ci_arimax.iloc[:, 1],
                 alpha=0.18, color=COLOR_ARIMAX, zorder=1,
                 label='ARIMAX 95% CI')

# Fitted values on training period 
ax1.plot(train_years, y_train_pred_arima,
         linestyle='--', linewidth=1.6, color=COLOR_ARIMA,
         alpha=0.45, zorder=2, label='_nolegend_')
ax1.plot(train_years, y_train_pred_arimax,
         linestyle='-.', linewidth=1.6, color=COLOR_ARIMAX,
         alpha=0.45, zorder=2, label='_nolegend_')

# Forecast lines on test period
ax1.plot(test_years, y_test_pred_arima,
         linestyle='--', linewidth=2.4, color=COLOR_ARIMA, zorder=3,
         label=f'ARIMA{orden_arima} forecast  (R²={r2_arima_test:.3f})')
ax1.plot(test_years, y_test_pred_arimax,
         linestyle='-.', linewidth=2.4, color=COLOR_ARIMAX, zorder=3,
         label=f'ARIMAX{orden_arimax} forecast (R²={r2_arimax_test:.3f})')

# Observed data — training
ax1.plot(train_years, y_train_true,
         marker='o', linestyle='-', linewidth=1.8, markersize=6,
         color=COLOR_TRAIN, alpha=0.65, zorder=4,
         markeredgewidth=0.8, markeredgecolor='white',
         label='Observed (training)')

# Observed data — test 
ax1.plot(test_years, y_test_true,
         marker='o', linestyle='-', linewidth=2.0, markersize=7,
         color=COLOR_OBSERVED, zorder=5,
         markeredgewidth=0.9, markeredgecolor='white',
         label='Observed (test)')

# Train/test split vertical line
ax1.axvline(x=TRAIN_END_YEAR + 0.5, color='black', linestyle=':',
            linewidth=1.8, alpha=0.6, zorder=6, label='Train / test split')

ax1.set_xlabel('Year', fontsize=12, fontweight='medium')
ax1.set_ylabel('CME count (events per year)', fontsize=12, fontweight='medium')
ax1.set_title(
    'Out-of-sample forecast: ARIMA vs ARIMAX for annual CME occurrence',
    fontsize=13, fontweight='bold', pad=14
)

ax1.set_xlim(YEAR_START - 0.5, YEAR_END + 0.5)
ax1.set_xticks(range(YEAR_START, YEAR_END + 1, 2))
ax1.set_ylim(bottom=0)      

ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=COLOR_GRID)
ax1.set_axisbelow(True)


legend = ax1.legend(
    loc='upper left', frameon=True, fancybox=False, shadow=False,
    framealpha=0.75,       
    edgecolor='#888888',
    facecolor='white',
    borderpad=0.7,
    labelspacing=0.4
)
legend.get_frame().set_linewidth(0.6)

textstr = (
    f'Out-of-sample RMSE improvement: {mejora_rmse:+.1f}%\n'
    f'Out-of-sample R² improvement:   {mejora_r2:+.1f}%'
)
props = dict(boxstyle='round,pad=0.4', facecolor='white',
             alpha=0.65, edgecolor='#aaaaaa', linewidth=0.7)
ax1.text(0.985, 0.03, textstr, transform=ax1.transAxes,
         fontsize=8.5, verticalalignment='bottom',
         horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('forecast_comparison_annual_with_validation.pdf',
            dpi=600, bbox_inches='tight')
print("\n   ✓ Figure 1 saved: 'forecast_comparison_annual_with_validation.pdf'")
plt.close()

# ---------- FIGURE 2: RESIDUAL DIAGNOSTICS ----------
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, residuos, color, label in [
    (axes[0, 0], residuos_arima,  COLOR_ARIMA,  f'ARIMA{orden_arima}'),
    (axes[0, 1], residuos_arimax, COLOR_ARIMAX, f'ARIMAX{orden_arimax}'),
]:
    ax.plot(train_years, residuos.values,
            marker='o', linestyle='-', color=color, alpha=0.6, markersize=5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.fill_between(train_years, residuos.values, 0, alpha=0.18, color=color)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Residual', fontsize=11)
    ax.set_title(f'{label} residuals (training data)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

for ax, residuos, color, label in [
    (axes[1, 0], residuos_arima,  COLOR_ARIMA,  f'ARIMA{orden_arima}'),
    (axes[1, 1], residuos_arimax, COLOR_ARIMAX, f'ARIMAX{orden_arimax}'),
]:
    ax.hist(residuos.values, bins=15, color=color, alpha=0.6, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{label} residual distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('residual_diagnostics_annual.pdf', dpi=600, bbox_inches='tight')
print("   ✓ Figure 2 saved: 'residual_diagnostics_annual.pdf'")
plt.close()

# ================================================================
# 13. SAVE NUMERICAL RESULTS
# ================================================================
print("\nSaving numerical results...")

all_years     = np.concatenate([train_years, test_years])
all_observed  = np.concatenate([y_train_true, y_test_true])
all_arima     = np.concatenate([y_train_pred_arima, y_test_pred_arima])
all_arimax    = np.concatenate([y_train_pred_arimax, y_test_pred_arimax])
all_sunspots  = np.concatenate([
    exog_train['Sunspots'].values, exog_test['Sunspots'].values
])

arima_ci_lower  = np.concatenate([np.full(len(train_years), np.nan),
                                   ci_arima.iloc[:, 0].values])
arima_ci_upper  = np.concatenate([np.full(len(train_years), np.nan),
                                   ci_arima.iloc[:, 1].values])
arimax_ci_lower = np.concatenate([np.full(len(train_years), np.nan),
                                   ci_arimax.iloc[:, 0].values])
arimax_ci_upper = np.concatenate([np.full(len(train_years), np.nan),
                                   ci_arimax.iloc[:, 1].values])

df_results = pd.DataFrame({
    'Year'              : all_years,
    'CMEs_observed'     : all_observed,
    'ARIMA_fitted'      : all_arima,
    'ARIMAX_fitted'     : all_arimax,
    'ARIMA_residuals'   : all_observed - all_arima,
    'ARIMAX_residuals'  : all_observed - all_arimax,
    'ARIMA_CI_lower_95' : arima_ci_lower,
    'ARIMA_CI_upper_95' : arima_ci_upper,
    'ARIMAX_CI_lower_95': arimax_ci_lower,
    'ARIMAX_CI_upper_95': arimax_ci_upper,
    'Sunspot_number'    : all_sunspots,
    'Dataset'           : ['Train'] * len(df_train) + ['Test'] * len(df_test),
})

df_results.to_csv('forecast_results_annual_with_validation.csv', index=False)
print("   ✓ Numerical results saved: 'forecast_results_annual_with_validation.csv'")

# ================================================================
# 14. METRICS SUMMARY FILE
# ================================================================
with open('model_metrics_summary_annual.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ARIMA vs ARIMAX MODEL COMPARISON — ANNUAL CME FORECASTING\n")
    f.write("WITH OUT-OF-SAMPLE VALIDATION\n")
    f.write("=" * 80 + "\n\n")

    f.write("DATA CONFIGURATION:\n")
    f.write(f"  Period           : {YEAR_START}–{YEAR_END} ({len(df_merged)} observations)\n")
    f.write(f"  CME filters      : speed ≥ {MIN_SPEED} km/s, "
            f"{MIN_WIDTH}° ≤ width ≤ {MAX_WIDTH}°\n")
    f.write(f"  Years with 0 CMEs: {(conteo_anual['CMEs_filtradas'] == 0).sum()}\n")
    f.write(f"  Train period     : {YEAR_START}–{TRAIN_END_YEAR} ({len(df_train)} years)\n")
    f.write(f"  Test period      : {TEST_START_YEAR}–{YEAR_END} ({len(df_test)} years)\n\n")

    f.write("PRELIMINARY TESTS:\n")
    f.write(f"  ADF (stationarity) : stat={adf_result[0]:.4f}, p={adf_result[1]:.4f}")
    f.write("  → STATIONARY\n" if adf_result[1] < 0.05 else "  → NON-STATIONARY\n")
    f.write(f"  Ljung-Box (lag {max_lags_lb}) : p={lb_result['lb_pvalue'].iloc[-1]:.4f}")
    f.write("  → AUTOCORRELATION present\n\n"
            if lb_result['lb_pvalue'].iloc[-1] < 0.05 else "  → WHITE NOISE\n\n")

    f.write(f"ARIMA{orden_arima} MODEL:\n")
    f.write(f"  Train RMSE : {rmse_arima_train:.3f}\n")
    f.write(f"  Train MAE  : {mae_arima_train:.3f}\n")
    f.write(f"  Train R²   : {r2_arima_train:.4f}\n")
    f.write(f"  Test RMSE  : {rmse_arima_test:.3f}  ***\n")
    f.write(f"  Test MAE   : {mae_arima_test:.3f}  ***\n")
    f.write(f"  Test R²    : {r2_arima_test:.4f}  ***\n")
    f.write(f"  AIC        : {resultado_arima.aic:.2f}\n")
    f.write(f"  LB p-value : {lb_arima['lb_pvalue'].iloc[-1]:.4f}\n\n")

    f.write(f"ARIMAX{orden_arimax} MODEL (with SSN):\n")
    f.write(f"  Train RMSE : {rmse_arimax_train:.3f}\n")
    f.write(f"  Train MAE  : {mae_arimax_train:.3f}\n")
    f.write(f"  Train R²   : {r2_arimax_train:.4f}\n")
    f.write(f"  Test RMSE  : {rmse_arimax_test:.3f}  ***\n")
    f.write(f"  Test MAE   : {mae_arimax_test:.3f}  ***\n")
    f.write(f"  Test R²    : {r2_arimax_test:.4f}  ***\n")
    f.write(f"  AIC        : {resultado_arimax.aic:.2f}\n")
    f.write(f"  LB p-value : {lb_arimax['lb_pvalue'].iloc[-1]:.4f}\n\n")

    f.write("OUT-OF-SAMPLE IMPROVEMENTS (ARIMAX vs ARIMA):\n")
    f.write(f"  RMSE : {mejora_rmse:+.2f}%\n")
    f.write(f"  MAE  : {mejora_mae:+.2f}%\n")
    f.write(f"  R²   : {mejora_r2:+.2f}%\n")
    f.write(f"  ΔAIC : {mejora_aic:+.2f}\n\n")

    f.write("REPRODUCIBILITY:\n")
    f.write(f"  numpy.random.seed : {RANDOM_SEED}\n")
    f.write(f"  auto_arima random_state : {RANDOM_SEED}\n\n")

    f.write("=" * 80 + "\n")
    f.write("*** Test metrics = TRUE out-of-sample forecasting performance\n")
    f.write("    CI lower bounds clipped to 0 (CME counts are non-negative)\n")
    f.write("=" * 80 + "\n")

print("   ✓ Metrics summary saved: 'model_metrics_summary_annual.txt'")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 80)
print("\nGenerated files:")
print("  1. acf_pacf_annual_analysis.pdf")
print("  2. forecast_comparison_annual_with_validation.pdf")
print("  3. residual_diagnostics_annual.pdf")
print("  4. forecast_results_annual_with_validation.csv")
print("  5. model_metrics_summary_annual.txt")
print(f"\nKey results:")
print(f"  ADF p-value            : {adf_result[1]:.4f}")
print(f"  ARIMA order            : {orden_arima}")
print(f"  ARIMAX order           : {orden_arimax}")
print(f"  RMSE improvement (OOS) : {mejora_rmse:+.1f}%")
print(f"  R² improvement (OOS)   : {mejora_r2:+.1f}%")
print("=" * 80)
