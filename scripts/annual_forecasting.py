# ================================================================
# ARIMA vs ARIMAX Comparison for Annual CME Forecasting
# ================================================================
# This script compares ARIMA and ARIMAX models with proper validation
# ================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
# ================================================================
# 0. PLOT STYLE CONFIGURATION 
# ================================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 14
rcParams['figure.dpi'] = 100
rcParams['savefig.dpi'] = 600
rcParams['savefig.format'] = 'pdf'
rcParams['savefig.bbox'] = 'tight'
rcParams['axes.linewidth'] = 1.0
rcParams['grid.linewidth'] = 0.5
rcParams['lines.linewidth'] = 1.5
rcParams['lines.markersize'] = 6

COLOR_OBSERVED = '#2C3E50'  
COLOR_ARIMA = '#3498DB'     
COLOR_ARIMAX = '#E74C3C'    
COLOR_GRID = '#BDC3C7'
COLOR_TRAIN = '#95A5A6'

#=================================================================
# 1. CONFIGURATION PARAMETERS
# ================================================================

MIN_SPEED = 450
MIN_WIDTH, MAX_WIDTH = 20, 120
YEAR_START, YEAR_END = 1996, 2024

# Train/test split configuration
TRAIN_END_YEAR = 2019  # 80% of data (1996-2019 = 24 years)
TEST_START_YEAR = 2020  # 20% of data (2020-2024 = 5 years)

print("=" * 80)
print("ARIMA vs ARIMAX MODEL COMPARISON")
print("=" * 80)
print(f"\nTrain period: {YEAR_START}-{TRAIN_END_YEAR}")
print(f"Test period:  {TEST_START_YEAR}-{YEAR_END}")

# ================================================================
# 2. LOAD CME DATA
# ================================================================

print("\n[1/9] Loading CME dataset...")
df_cmes = pd.read_csv("datos_procesados_2025_11_30.csv", low_memory=False)
df_cmes['Fecha'] = pd.to_datetime(df_cmes['Fecha'], errors='coerce')
df_cmes[['Central', 'Ancho', 'Rapidez']] = df_cmes[['Central', 'Ancho', 'Rapidez']].apply(pd.to_numeric, errors='coerce')
df_cmes['Year'] = df_cmes['Fecha'].dt.year

# ================================================================
# 3. FILTER CME EVENTS AND HANDLE MISSING YEARS
# ================================================================

print("[2/9] Filtering CME events and handling missing years...")
df_cmes_filtrado = df_cmes[
    (df_cmes['Rapidez'] >= MIN_SPEED) &
    (df_cmes['Ancho'] >= MIN_WIDTH) &
    (df_cmes['Ancho'] <= MAX_WIDTH)
].copy()

# Count CMEs per year
conteo_anual = (
    df_cmes_filtrado
    .groupby('Year')
    .size()
    .rename('CMEs_filtradas')
)

full_years = pd.RangeIndex(YEAR_START, YEAR_END + 1)
conteo_anual = (
    conteo_anual
    .reindex(full_years, fill_value=0)  # Fill missing years with 0
    .reset_index()
)
conteo_anual.columns = ['Year', 'CMEs_filtradas']

print(f"   Total CMEs: {len(df_cmes)} → Filtered CMEs: {len(df_cmes_filtrado)}")
print(f"   Years with zero CMEs: {(conteo_anual['CMEs_filtradas'] == 0).sum()}")

# ================================================================
# 4. LOAD SUNSPOT DATA
# ================================================================
print("[3/9] Loading sunspot numbers...")

df_sn = pd.read_csv("SN_y_tot_V2.0.txt", sep=r'\s+', header=None, usecols=[0, 1],
                    names=['Year', 'SunspotNumber'])
df_sn['Year'] = df_sn['Year'].astype(int)
df_sn = df_sn[(df_sn['Year'] >= YEAR_START) & (df_sn['Year'] <= YEAR_END)]

# ================================================================
# 5. MERGE DATASETS (NOW SAFE - conteo_anual HAS ALL YEARS)
# ================================================================
print("\n[4/9] Merging datasets...")

# Use inner join - now safe because conteo_anual has all years with 0s filled
df_merged = pd.merge(df_sn, conteo_anual, on='Year', how='inner')

print(f"   Temporal range: {df_merged['Year'].min()} - {df_merged['Year'].max()}")
print(f"   Total observations: {len(df_merged)}")
print(f"   NaN values in CMEs: {df_merged['CMEs_filtradas'].isna().sum()}")  # Should be 0

# ================================================================
# 6. TRAIN/TEST SPLIT
# ================================================================
print("\n[5/9] Splitting data into train and test sets...")

df_train = df_merged[df_merged['Year'] <= TRAIN_END_YEAR].copy()
df_test = df_merged[df_merged['Year'] >= TEST_START_YEAR].copy()

print(f"   Train set: {len(df_train)} observations ({df_train['Year'].min()}-{df_train['Year'].max()})")
print(f"   Test set:  {len(df_test)} observations ({df_test['Year'].min()}-{df_test['Year'].max()})")
print(f"   Train zeros: {(df_train['CMEs_filtradas'] == 0).sum()}")
print(f"   Test zeros: {(df_test['CMEs_filtradas'] == 0).sum()}")

index_train = pd.date_range(start=f'{YEAR_START}', periods=len(df_train), freq='YS')
index_test = pd.date_range(start=f'{TEST_START_YEAR}', periods=len(df_test), freq='YS')

endog_train = pd.Series(df_train['CMEs_filtradas'].values, index=index_train)
endog_test = pd.Series(df_test['CMEs_filtradas'].values, index=index_test)

exog_train = pd.DataFrame(df_train['SunspotNumber'].values, index=index_train, columns=['Sunspots'])
exog_test = pd.DataFrame(df_test['SunspotNumber'].values, index=index_test, columns=['Sunspots'])

# ================================================================
# 7. STATIONARITY AND WHITE NOISE TESTS
# ================================================================
print("\n" + "="*80)
print("PRELIMINARY STATISTICAL TESTS")
print("="*80)
print("[6/9] Testing stationarity (ADF) and serial correlation (Ljung-Box)...")

# ADF Test for stationarity
adf_result = adfuller(endog_train, autolag='AIC')
print(f"\n   AUGMENTED DICKEY-FULLER TEST (Stationarity):")
print(f"   ADF Statistic: {adf_result[0]:.4f}")
print(f"   p-value: {adf_result[1]:.4f}")
print(f"   Critical values:")
for key, value in adf_result[4].items():
    print(f"      {key}: {value:.4f}")

if adf_result[1] < 0.05:
    print(f"   ✓ Series is STATIONARY (p < 0.05) - No differencing needed")
    diferenciacion_sugerida = 0
else:
    print(f"   ⚠ Series is NON-STATIONARY (p >= 0.05) - Differencing required")
    # Test first difference
    diff_series = endog_train.diff().dropna()
    adf_diff = adfuller(diff_series, autolag='AIC')
    print(f"\n   ADF test on first difference:")
    print(f"   ADF Statistic: {adf_diff[0]:.4f}")
    print(f"   p-value: {adf_diff[1]:.4f}")
    if adf_diff[1] < 0.05:
        print(f"   ✓ First difference is stationary - d=1 recommended")
        diferenciacion_sugerida = 1
    else:
        print(f"   ⚠ May need d=2")
        diferenciacion_sugerida = 2

# Ljung-Box Test for white noise (serial correlation)
max_lags_lb = min(10, len(endog_train)//4)
lb_result = acorr_ljungbox(endog_train, lags=max_lags_lb, return_df=True)
print(f"\n   LJUNG-BOX TEST (White Noise Check):")
print(f"   Testing H0: No autocorrelation (series is white noise)")
print(f"   p-value (lag {max_lags_lb}): {lb_result['lb_pvalue'].iloc[-1]:.4f}")

if lb_result['lb_pvalue'].iloc[-1] < 0.05:
    print(f"    Series shows SIGNIFICANT AUTOCORRELATION (p < 0.05)")
    print(f"    Time series modeling is appropriate")
else:
    print(f"    Series behaves like WHITE NOISE (p >= 0.05)")
    print(f"    Forecasting may be difficult (random walk)")

print("="*80)

# ================================================================
# 8. ACF/PACF ANALYSIS (for model justification)
# ================================================================
print("\n[7/9] Generating ACF/PACF plots for model selection justification...")

fig_acf, axes_acf = plt.subplots(2, 1, figsize=(12, 8))

max_lags_acf = min(10, len(endog_train)//2 - 1)
plot_acf(endog_train, lags=max_lags_acf, ax=axes_acf[0], color=COLOR_ARIMA, alpha=0.5)
axes_acf[0].set_title('Autocorrelation Function (ACF) - Training Data', fontsize=12, fontweight='bold')
axes_acf[0].set_xlabel('Lag (years)', fontsize=11)
axes_acf[0].set_ylabel('ACF', fontsize=11)

plot_pacf(endog_train, lags=max_lags_acf, ax=axes_acf[1], color=COLOR_ARIMAX, alpha=0.5, method='ywm')
axes_acf[1].set_title('Partial Autocorrelation Function (PACF) - Training Data', fontsize=12, fontweight='bold')
axes_acf[1].set_xlabel('Lag (years)', fontsize=11)
axes_acf[1].set_ylabel('PACF', fontsize=11)

plt.tight_layout()
plt.savefig('acf_pacf_annual_analysis.pdf', dpi=600, bbox_inches='tight')
print("    ACF/PACF plots saved: 'acf_pacf_annual_analysis.pdf'")
plt.close()

# ========================================================================
# 9. MODEL 1: ARIMA (TRAINING PHASE)
# ========================================================================
print("\n" + "="*80)
print("MODEL 1: ARIMA (WITHOUT EXOGENOUS VARIABLES)")
print("="*80)
print("[8a/9] Selecting optimal ARIMA order on training data...")

# Adjust max_p and max_q based on sample size
max_order = min(5, len(endog_train)//3)

modelo_arima_auto = auto_arima(
    endog_train, 
    seasonal=False, 
    trace=False,
    error_action='ignore', 
    suppress_warnings=True, 
    stepwise=True,
    #Si se quieren probar modelos con d=0, d=1 y d=2, se comenta la linea siguiente y se descomenta la linea de abajo

    #max_p=5, max_q=5,                          #se descomenta cuando las lineas siguientes se comentan

    max_p=max_order, max_q=max_order, max_d=2,  #se comentan para utilizar el d sugerido por el ADF
    start_p=0, start_q=0,                       #este también se comenta
    information_criterion='aic'                 #este también se comenta

    #d=2                                        #se descomenta cuando las lineas anteriores se comentan
)

orden_arima = modelo_arima_auto.order
print(f"   Order selected: ARIMA{orden_arima}")

# Fit ARIMA model
modelo_arima = SARIMAX(
    endog_train,
    order=orden_arima,
    enforce_stationarity=False,
    enforce_invertibility=False
)
resultado_arima = modelo_arima.fit(disp=False)

# In-sample predictions (training)
pred_arima_train = resultado_arima.get_prediction(start=endog_train.index[0], end=endog_train.index[-1])
pred_arima_train_mean = pred_arima_train.predicted_mean

# Out-of-sample forecast (testing)
forecast_arima = resultado_arima.forecast(steps=len(df_test))
forecast_arima_series = pd.Series(forecast_arima, index=index_test)

# Metrics - Training set
y_train_true = endog_train.values
y_train_pred_arima = pred_arima_train_mean.values
rmse_arima_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred_arima))
mae_arima_train = mean_absolute_error(y_train_true, y_train_pred_arima)
r2_arima_train = r2_score(y_train_true, y_train_pred_arima)

# Metrics - Test set (OUT-OF-SAMPLE)
y_test_true = endog_test.values
y_test_pred_arima = forecast_arima_series.values
rmse_arima_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred_arima))
mae_arima_test = mean_absolute_error(y_test_true, y_test_pred_arima)
r2_arima_test = r2_score(y_test_true, y_test_pred_arima)

# Residual diagnostics
residuos_arima = resultado_arima.resid
max_lags_res = min(10, len(residuos_arima)//4)
lb_arima = acorr_ljungbox(residuos_arima, lags=max_lags_res, return_df=True)

print(f"\n   ARIMA METRICS (IN-SAMPLE - TRAINING):")
print(f"   RMSE:  {rmse_arima_train:.3f} CMEs/year")
print(f"   MAE:   {mae_arima_train:.3f} CMEs/year")
print(f"   R²:    {r2_arima_train:.4f}")

print(f"\n   ARIMA METRICS (OUT-OF-SAMPLE - TEST) *** KEY METRIC ***:")
print(f"   RMSE:  {rmse_arima_test:.3f} CMEs/year")
print(f"   MAE:   {mae_arima_test:.3f} CMEs/year")
print(f"   R²:    {r2_arima_test:.4f}")
print(f"   AIC:   {resultado_arima.aic:.2f}")

print(f"\n   Ljung-Box test (residuals whiteness):")
print(f"   p-value (lag {max_lags_res}): {lb_arima['lb_pvalue'].iloc[-1]:.4f}")
if lb_arima['lb_pvalue'].iloc[-1] > 0.05:
    print("    Residuals are white noise (p > 0.05)")
else:
    print("   Residuals show autocorrelation (p < 0.05)")

# ========================================================================
# 10. MODEL 2: ARIMAX (TRAINING PHASE WITH SUNSPOTS)
# ========================================================================
print("\n" + "="*80)
print("MODEL 2: ARIMAX (WITH SUNSPOT NUMBERS)")
print("="*80)
print("[8b/9] Selecting optimal ARIMAX order on training data...")

modelo_arimax_auto = auto_arima(
    endog_train, 
    exogenous=exog_train,
    seasonal=False, 
    trace=False,
    error_action='ignore', 
    suppress_warnings=True, 
    stepwise=True,
    #Si se quieren probar modelos con d=0, d=1 y d=2, se comenta la linea siguiente y se descomenta la linea de abajo

    #max_p=5, max_q=5,                          #se descomenta cuando las lineas siguientes se comentan

    max_p=max_order, max_q=max_order, max_d=2,  #se comentan para utilizar el d sugerido por el ADF
    start_p=0, start_q=0,                       #este también se comenta
    information_criterion='aic'                 #este también se comenta

    #d=2                                        #se descomenta cuando las lineas anteriores se comentan
)
orden_arimax = modelo_arimax_auto.order
print(f"   Order selected: ARIMA{orden_arimax}")

# Fit ARIMAX model
modelo_arimax = SARIMAX(
    endog_train, 
    exog=exog_train,
    order=orden_arimax,
    enforce_stationarity=False,
    enforce_invertibility=False
)
resultado_arimax = modelo_arimax.fit(disp=False)

# In-sample predictions
pred_arimax_train = resultado_arimax.get_prediction(start=endog_train.index[0], 
                                                     end=endog_train.index[-1], 
                                                     exog=exog_train)
pred_arimax_train_mean = pred_arimax_train.predicted_mean

# Out-of-sample forecast
forecast_arimax = resultado_arimax.forecast(steps=len(df_test), exog=exog_test)
forecast_arimax_series = pd.Series(forecast_arimax, index=index_test)

# Metrics - Training set
y_train_pred_arimax = pred_arimax_train_mean.values
rmse_arimax_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred_arimax))
mae_arimax_train = mean_absolute_error(y_train_true, y_train_pred_arimax)
r2_arimax_train = r2_score(y_train_true, y_train_pred_arimax)

# Metrics - Test set (OUT-OF-SAMPLE)
y_test_pred_arimax = forecast_arimax_series.values
rmse_arimax_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred_arimax))
mae_arimax_test = mean_absolute_error(y_test_true, y_test_pred_arimax)
r2_arimax_test = r2_score(y_test_true, y_test_pred_arimax)

# Residual diagnostics
residuos_arimax = resultado_arimax.resid
lb_arimax = acorr_ljungbox(residuos_arimax, lags=max_lags_res, return_df=True)

print(f"\n   ARIMAX METRICS (IN-SAMPLE - TRAINING):")
print(f"   RMSE:  {rmse_arimax_train:.3f} CMEs/year")
print(f"   MAE:   {mae_arimax_train:.3f} CMEs/year")
print(f"   R²:    {r2_arimax_train:.4f}")

print(f"\n   ARIMAX METRICS (OUT-OF-SAMPLE - TEST) *** KEY METRIC ***:")
print(f"   RMSE:  {rmse_arimax_test:.3f} CMEs/year")
print(f"   MAE:   {mae_arimax_test:.3f} CMEs/year")
print(f"   R²:    {r2_arimax_test:.4f}")
print(f"   AIC:   {resultado_arimax.aic:.2f}")

print(f"\n   Ljung-Box test (residuals whiteness):")
print(f"   p-value (lag {max_lags_res}): {lb_arimax['lb_pvalue'].iloc[-1]:.4f}")
if lb_arimax['lb_pvalue'].iloc[-1] > 0.05:
    print("    Residuals are white noise (p > 0.05)")
else:
    print("    Residuals show autocorrelation (p < 0.05)")

# ========================================================================
# 11. MODEL COMPARISON
# ========================================================================
print("\n" + "="*80)
print("MODEL COMPARISON (OUT-OF-SAMPLE PERFORMANCE)")
print("="*80)

mejora_rmse = ((rmse_arima_test - rmse_arimax_test) / rmse_arima_test) * 100 if rmse_arima_test > 0 else 0
mejora_mae = ((mae_arima_test - mae_arimax_test) / mae_arima_test) * 100 if mae_arima_test > 0 else 0
mejora_r2 = ((r2_arimax_test - r2_arima_test) / abs(r2_arima_test)) * 100 if r2_arima_test != 0 else 0
mejora_aic = resultado_arima.aic - resultado_arimax.aic

print(f"\n{'Metric':<20} {'ARIMA':<15} {'ARIMAX':<15} {'Improvement':<15}")
print("-"*65)
print(f"{'RMSE (test)':<20} {rmse_arima_test:<15.3f} {rmse_arimax_test:<15.3f} {mejora_rmse:>+.2f}%")
print(f"{'MAE (test)':<20} {mae_arima_test:<15.3f} {mae_arimax_test:<15.3f} {mejora_mae:>+.2f}%")
print(f"{'R² (test)':<20} {r2_arima_test:<15.4f} {r2_arimax_test:<15.4f} {mejora_r2:>+.2f}%")
print(f"{'AIC':<20} {resultado_arima.aic:<15.2f} {resultado_arimax.aic:<15.2f} {mejora_aic:>+.2f}")
print("-"*65)

if rmse_arimax_test < rmse_arima_test and resultado_arimax.aic < resultado_arima.aic:
    print(" ARIMAX is SUPERIOR: Lower error AND lower AIC")
    print(f"  Sunspot numbers SIGNIFICANTLY IMPROVE the model")
elif rmse_arimax_test < rmse_arima_test:
    print(" ARIMAX has lower error, but similar AIC")
    print(f"   Sunspot numbers help, but add complexity")
else:
    print(" ARIMA is comparable or superior")
    print(f"   Sunspot numbers DO NOT substantially improve the model")

print("="*80)

# ========================================================================
# 12. PUBLICATION-QUALITY VISUALIZATION (NUMPY ARRAYS)
# ========================================================================

# ========== FIGURE 1: FORECAST COMPARISON ==========
fig1, ax1 = plt.subplots(figsize=(12, 6))

train_years = endog_train.index.year.values 
test_years = endog_test.index.year.values    

# Plot observed data
ax1.plot(train_years, y_train_true, 
         marker='o', linestyle='-', linewidth=2.0, markersize=7,
         color=COLOR_TRAIN, label='Observed (training)',
         zorder=3, markeredgewidth=1.2, markeredgecolor='white', alpha=0.7)

ax1.plot(test_years, y_test_true, 
         marker='o', linestyle='-', linewidth=2.0, markersize=7,
         color=COLOR_OBSERVED, label='Observed (test)',
         zorder=3, markeredgewidth=1.2, markeredgecolor='white')

# Plot fitted values (training)
ax1.plot(train_years, y_train_pred_arima, 
         linestyle='--', linewidth=2.0, color=COLOR_ARIMA, 
         alpha=0.5, zorder=2)

ax1.plot(train_years, y_train_pred_arimax, 
         linestyle='-.', linewidth=2.0, color=COLOR_ARIMAX, 
         alpha=0.5, zorder=2)

# Plot forecasts (test) 
ax1.plot(test_years, y_test_pred_arima, 
         linestyle='--', linewidth=2.5, color=COLOR_ARIMA, 
         label=f'ARIMA{orden_arima} forecast (R²={r2_arima_test:.3f})',
         zorder=2, alpha=0.9)

ax1.plot(test_years, y_test_pred_arimax, 
         linestyle='-.', linewidth=2.5, color=COLOR_ARIMAX, 
         label=f'ARIMAX{orden_arimax} forecast (R²={r2_arimax_test:.3f})',
         zorder=2, alpha=0.9)

# Vertical line separating train/test
ax1.axvline(x=TRAIN_END_YEAR + 0.5, color='black', linestyle=':', 
            linewidth=2, alpha=0.7, label='Train/Test split')

# Shaded test region
ax1.axvspan(TEST_START_YEAR - 0.5, YEAR_END + 0.5, 
            alpha=0.1, color='gray', zorder=0)

ax1.set_xlabel('Year', fontsize=12, fontweight='medium')
ax1.set_ylabel('CME Count (per year)', fontsize=12, fontweight='medium')
ax1.set_title('Out-of-Sample Forecast: ARIMA vs ARIMAX for Annual CME Occurrence', 
             fontsize=13, fontweight='bold', pad=15)

ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=COLOR_GRID)
ax1.set_axisbelow(True)

legend = ax1.legend(loc='upper left', frameon=True, fancybox=False, 
                   shadow=False, framealpha=0.95, edgecolor='black',
                   facecolor='white')
legend.get_frame().set_linewidth(0.8)

ax1.set_xlim(YEAR_START - 0.5, YEAR_END + 0.5)
ax1.set_xticks(range(YEAR_START, YEAR_END + 1, 2))

textstr = f'Test RMSE improvement: {mejora_rmse:+.1f}%\nTest R² improvement: {mejora_r2:+.1f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.2, edgecolor='gray', linewidth=0.8)
ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)
# ========== SAVING FIGURE =========
plt.tight_layout()
plt.savefig('forecast_comparison_annual_with_validation.pdf', dpi=600, bbox_inches='tight')
print("   ✓ Figure 1 saved: 'forecast_comparison_annual_with_validation.pdf'")
plt.close()

# ========== FIGURE 2: RESIDUAL DIAGNOSTICS ==========
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# ARIMA residuals - time plot
axes[0, 0].plot(train_years, residuos_arima.values,  # .values to get numpy array
                marker='o', linestyle='-', color=COLOR_ARIMA, alpha=0.6, markersize=5)
axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 0].fill_between(train_years, residuos_arima.values, 0, 
                        alpha=0.2, color=COLOR_ARIMA)
axes[0, 0].set_xlabel('Year', fontsize=11)
axes[0, 0].set_ylabel('Residual', fontsize=11)
axes[0, 0].set_title(f'ARIMA{orden_arima} Residuals (Training Data)', 
                     fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# ARIMAX residuals - time plot
axes[0, 1].plot(train_years, residuos_arimax.values, 
                marker='o', linestyle='-', color=COLOR_ARIMAX, alpha=0.6, markersize=5)
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 1].fill_between(train_years, residuos_arimax.values, 0, 
                        alpha=0.2, color=COLOR_ARIMAX)
axes[0, 1].set_xlabel('Year', fontsize=11)
axes[0, 1].set_ylabel('Residual', fontsize=11)
axes[0, 1].set_title(f'ARIMAX{orden_arimax} Residuals (Training Data)', 
                     fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# ARIMA residuals - histogram
axes[1, 0].hist(residuos_arima.values, bins=15, color=COLOR_ARIMA, 
                alpha=0.6, edgecolor='black')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Residual Value', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title(f'ARIMA{orden_arima} Residual Distribution', 
                     fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# ARIMAX residuals - histogram
axes[1, 1].hist(residuos_arimax.values, bins=15, color=COLOR_ARIMAX, 
                alpha=0.6, edgecolor='black')
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Residual Value', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title(f'ARIMAX{orden_arimax} Residual Distribution', 
                     fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('residual_diagnostics_annual.pdf', dpi=600, bbox_inches='tight')
print("   ✓ Figure 2 saved: 'residual_diagnostics_annual.pdf'")
plt.close()

# ========================================================================
# 13. SAVE RESULTS
# ========================================================================
print("\nSaving numerical results...")

# Combine arrays for saving
all_years = np.concatenate([train_years, test_years])
all_observed = np.concatenate([y_train_true, y_test_true])
all_arima = np.concatenate([y_train_pred_arima, y_test_pred_arima])
all_arimax = np.concatenate([y_train_pred_arimax, y_test_pred_arimax])
all_sunspots = np.concatenate([exog_train['Sunspots'].values, exog_test['Sunspots'].values])

df_results = pd.DataFrame({
    "Year": all_years,
    "CMEs_observed": all_observed,
    "ARIMA_fitted": all_arima,
    "ARIMAX_fitted": all_arimax,
    "ARIMA_residuals": all_observed - all_arima,
    "ARIMAX_residuals": all_observed - all_arimax,
    "Sunspot_number": all_sunspots,
    "Dataset": ['Train']*len(df_train) + ['Test']*len(df_test)
})

df_results.to_csv("forecast_results_annual_with_validation.csv", index=False)
print("    Numerical results saved: 'forecast_results_annual_with_validation.csv'")

# Save comprehensive metrics summary
with open('model_metrics_summary_annual.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("ARIMA vs ARIMAX MODEL COMPARISON - ANNUAL CME FORECASTING\n")
    f.write("WITH OUT-OF-SAMPLE VALIDATION\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"DATA CONFIGURATION:\n")
    f.write(f"  Period: {YEAR_START}-{YEAR_END} ({len(df_merged)} observations)\n")
    f.write(f"  CME filters: speed ≥ {MIN_SPEED} km/s, {MIN_WIDTH}° ≤ width ≤ {MAX_WIDTH}°\n")
    f.write(f"  Years with zero CMEs: {(conteo_anual['CMEs_filtradas'] == 0).sum()}\n")
    f.write(f"  Train/Test split: {TRAIN_END_YEAR}/{TEST_START_YEAR}\n")
    f.write(f"  Train size: {len(df_train)} years\n")
    f.write(f"  Test size: {len(df_test)} years\n\n")
    
    f.write(f"PRELIMINARY TESTS:\n")
    f.write(f"  ADF test (stationarity): stat={adf_result[0]:.4f}, p={adf_result[1]:.4f}\n")
    if adf_result[1] < 0.05:
        f.write(f"     Series is STATIONARY\n")
    else:
        f.write(f"     Series is NON-STATIONARY (differencing needed)\n")
    f.write(f"  Ljung-Box test (white noise): p={lb_result['lb_pvalue'].iloc[-1]:.4f}\n")
    if lb_result['lb_pvalue'].iloc[-1] < 0.05:
        f.write(f"     Series shows AUTOCORRELATION (modeling appropriate)\n\n")
    else:
        f.write(f"     Series is WHITE NOISE (forecasting difficult)\n\n")
    
 
    f.write(f"ARIMA{orden_arima} MODEL:\n")
    f.write(f"  Training RMSE: {rmse_arima_train:.3f} CMEs/year\n")
    f.write(f"  Training MAE:  {mae_arima_train:.3f} CMEs/year\n")
    f.write(f"  Training R²:   {r2_arima_train:.4f}\n")
    f.write(f"  ---\n")
    f.write(f"  Test RMSE:     {rmse_arima_test:.3f} CMEs/year ***\n")
    f.write(f"  Test MAE:      {mae_arima_test:.3f} CMEs/year ***\n")
    f.write(f"  Test R²:       {r2_arima_test:.4f} ***\n")
    f.write(f"  AIC:           {resultado_arima.aic:.2f}\n")
    f.write(f"  Ljung-Box p:   {lb_arima['lb_pvalue'].iloc[-1]:.4f}\n\n")
    
    f.write(f"ARIMAX{orden_arimax} MODEL (with SSN):\n")
    f.write(f"  Training RMSE: {rmse_arimax_train:.3f} CMEs/year\n")
    f.write(f"  Training MAE:  {mae_arimax_train:.3f} CMEs/year\n")
    f.write(f"  Training R²:   {r2_arimax_train:.4f}\n")
    f.write(f"  ---\n")
    f.write(f"  Test RMSE:     {rmse_arimax_test:.3f} CMEs/year ***\n")
    f.write(f"  Test MAE:      {mae_arimax_test:.3f} CMEs/year ***\n")
    f.write(f"  Test R²:       {r2_arimax_test:.4f} ***\n")
    f.write(f"  AIC:           {resultado_arimax.aic:.2f}\n")
    f.write(f"  Ljung-Box p:   {lb_arimax['lb_pvalue'].iloc[-1]:.4f}\n\n")
    
    f.write(f"OUT-OF-SAMPLE IMPROVEMENTS (ARIMAX vs ARIMA):\n")
    f.write(f"  RMSE: {mejora_rmse:+.2f}%\n")
    f.write(f"  MAE:  {mejora_mae:+.2f}%\n")
    f.write(f"  R²:   {mejora_r2:+.2f}%\n")
    f.write(f"  ΔAIC: {mejora_aic:+.2f}\n\n")
    
    f.write("="*80 + "\n")
    f.write("*** Test metrics represent TRUE out-of-sample forecasting performance\n")
    f.write("="*80 + "\n")

print("    Metrics summary saved: 'model_metrics_summary_annual.txt'")

print("\n" + "="*80)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("="*80)
print("\nGenerated files:")
print("  1. acf_pacf_annual_analysis.pdf - ACF/PACF plots")
print("  2. forecast_comparison_annual_with_validation.pdf - Main forecast figure")
print("  3. residual_diagnostics_annual.pdf - Residual analysis")
print("  4. forecast_results_annual_with_validation.csv - Numerical results")
print("  5. model_metrics_summary_annual.txt - Complete metrics summary")
print(f"  - Stationarity: ADF p-value = {adf_result[1]:.4f}")
print(f"  - Out-of-sample RMSE improvement: {mejora_rmse:+.1f}%")
print(f"  - Out-of-sample R² improvement: {mejora_r2:+.1f}%")
print("="*80)
