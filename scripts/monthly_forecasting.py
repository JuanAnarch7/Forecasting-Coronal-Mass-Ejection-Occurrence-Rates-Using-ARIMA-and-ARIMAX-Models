# ================================================================
# SARIMA vs SARIMAX Comparison for Monthly CME Forecasting

# ================================================================
# This script compares SARIMA and SARIMAX models 
# ================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.dates import DateFormatter, YearLocator
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
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 13
rcParams['figure.dpi'] = 100
rcParams['savefig.dpi'] = 600
rcParams['savefig.format'] = 'pdf'
rcParams['savefig.bbox'] = 'tight'

COLOR_OBSERVED = '#2C3E50'  
COLOR_ARIMA = '#3498DB'     
COLOR_ARIMAX = '#E74C3C'    
COLOR_GRID = '#BDC3C7'
COLOR_TRAIN = '#95A5A6'

#=================================================================
# 1. CONFIGURATION PARAMETERS
# ================================================================

MIN_SPEED = 0
MIN_WIDTH, MAX_WIDTH = 0, 360
YEAR_START, YEAR_END = 1996, 2025

# Train/test split - 80/20 split
TRAIN_END = '2019-02'
TEST_START = '2019-03'

print("=" * 80)
print("SARIMA vs SARIMAX MODEL COMPARISON WITH OUT-OF-SAMPLE VALIDATION")
print("MONTHLY RESOLUTION")
print("=" * 80)
print(f"\nTrain period: {YEAR_START}-01 to {TRAIN_END}")
print(f"Test period:  {TEST_START} to {YEAR_END}-12")

# ================================================================
# 2. LOAD CME DATA
# ================================================================

print("\n[1/9] Loading CME dataset...")
df_cmes = pd.read_csv("datos_procesados_2025_11_30.csv", low_memory=False)
df_cmes['Fecha'] = pd.to_datetime(df_cmes['Fecha'], errors='coerce')
df_cmes[['Central', 'Ancho', 'Rapidez']] = df_cmes[['Central', 'Ancho', 'Rapidez']].apply(pd.to_numeric, errors='coerce')
df_cmes['Year'] = df_cmes['Fecha'].dt.year
df_cmes['Month'] = df_cmes['Fecha'].dt.month
df_cmes['YearMonth'] = df_cmes['Fecha'].dt.to_period('M')

# ================================================================
# 3. FILTER CME EVENTS
# ================================================================

print("[2/9] Filtering CME events...")
df_cmes_filtrado = df_cmes[
    (df_cmes['Rapidez'] >= MIN_SPEED) &
    (df_cmes['Ancho'] >= MIN_WIDTH) &
    (df_cmes['Ancho'] <= MAX_WIDTH)
].copy()

conteo_filtrado = (
    df_cmes_filtrado
    .groupby('YearMonth')
    .size()
    .rename('CMEs_filtradas')
)

full_range = pd.period_range(
    start=f"{YEAR_START}-01",
    end=f"{YEAR_END}-12",
    freq='M'
)

conteo_filtrado = (
    conteo_filtrado
    .reindex(full_range, fill_value=0)
    .reset_index()
)

conteo_filtrado.columns = ['YearMonth', 'CMEs_filtradas']
conteo_filtrado['YearMonth'] = conteo_filtrado['YearMonth'].astype(str)

print(f"   Total CMEs: {len(df_cmes)} → Filtered CMEs: {len(df_cmes_filtrado)}")


# ================================================================
# 4. LOAD SUNSPOT DATA (MONTHLY)
# ================================================================
print("[3/9] Loading monthly sunspot numbers...")

df_sn = pd.read_csv("SN_m_tot_V2.0.txt", sep=r'\s+', header=None, 
                    usecols=[0, 1, 3],
                    names=['Year', 'Month', 'SunspotNumber'])
df_sn['Year'] = df_sn['Year'].astype(int)
df_sn['Month'] = df_sn['Month'].astype(int)
df_sn['YearMonth'] = pd.to_datetime(df_sn[['Year', 'Month']].assign(day=1)).dt.to_period('M').astype(str)

# Filter by temporal range
df_sn = df_sn[(df_sn['Year'] >= YEAR_START) & (df_sn['Year'] <= YEAR_END)]

# Merge datasets
df_merged = pd.merge(df_sn, conteo_filtrado, on='YearMonth', how='inner')
df_merged = df_merged.sort_values('YearMonth').reset_index(drop=True)
print(f"   Temporal range: {df_merged['YearMonth'].min()} - {df_merged['YearMonth'].max()}")
print(f"   Total observations: {len(df_merged)} months")

# Correlation analysis
correlacion, p_valor = pearsonr(df_merged['SunspotNumber'], df_merged['CMEs_filtradas'])
print(f"\n   Pearson correlation (SSN vs CME): r = {correlacion:.4f} (p = {p_valor:.4e})")

# ================================================================
# 5. TRAIN/TEST SPLIT
# ================================================================
print("\n[4/9] Splitting data into train and test sets...")

# Create datetime index
df_merged['Date'] = pd.to_datetime(df_merged['YearMonth'].str.replace('-', '') + '01', format='%Y%m%d')

# Split based on date
df_train = df_merged[df_merged['YearMonth'] <= TRAIN_END].copy()
df_test = df_merged[df_merged['YearMonth'] >= TEST_START].copy()

print(f"   Train set: {len(df_train)} months ({df_train['YearMonth'].min()} to {df_train['YearMonth'].max()})")
print(f"   Test set:  {len(df_test)} months ({df_test['YearMonth'].min()} to {df_test['YearMonth'].max()})")

# Create time series objects
endog_train = pd.Series(df_train['CMEs_filtradas'].values, index=df_train['Date'])
endog_test = pd.Series(df_test['CMEs_filtradas'].values, index=df_test['Date'])

exog_train = pd.DataFrame(df_train['SunspotNumber'].values, 
                          index=df_train['Date'], columns=['Sunspots'])
exog_test = pd.DataFrame(df_test['SunspotNumber'].values, 
                         index=df_test['Date'], columns=['Sunspots'])

# ================================================================
# 6. STATIONARITY AND WHITE NOISE TESTS
# ================================================================
print("\n" + "="*80)
print("PRELIMINARY STATISTICAL TESTS")
print("="*80)
print("[5/9] Testing stationarity (ADF) and serial correlation (Ljung-Box)...")

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

# Ljung-Box Test for white noise
lb_result = acorr_ljungbox(endog_train, lags=20, return_df=True)
print(f"\n   LJUNG-BOX TEST (White Noise Check):")
print(f"   p-value (lag 20): {lb_result['lb_pvalue'].iloc[-1]:.4f}")

if lb_result['lb_pvalue'].iloc[-1] < 0.05:
    print(f"    Series shows SIGNIFICANT AUTOCORRELATION (p < 0.05)")
    print(f"    Time series modeling is appropriate")
else:
    print(f"    Series behaves like WHITE NOISE (p >= 0.05)")

print("="*80)

# ================================================================
# 7. ACF/PACF ANALYSIS
# ================================================================
print("\n[6/9] Generating ACF/PACF plots...")

fig_acf, axes_acf = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(endog_train, lags=40, ax=axes_acf[0], color=COLOR_ARIMA, alpha=0.5)
axes_acf[0].set_title('Autocorrelation Function (ACF) - Monthly Training Data', 
                      fontsize=12, fontweight='bold')
axes_acf[0].set_xlabel('Lag (months)', fontsize=11)
axes_acf[0].set_ylabel('ACF', fontsize=11)

plot_pacf(endog_train, lags=40, ax=axes_acf[1], color=COLOR_ARIMAX, alpha=0.5, method='ywm')
axes_acf[1].set_title('Partial Autocorrelation Function (PACF) - Monthly Training Data', 
                      fontsize=12, fontweight='bold')
axes_acf[1].set_xlabel('Lag (months)', fontsize=11)
axes_acf[1].set_ylabel('PACF', fontsize=11)

plt.tight_layout()
plt.savefig('acf_pacf_monthly_analysis.pdf', dpi=600, bbox_inches='tight')
print("    ACF/PACF plots saved: 'acf_pacf_monthly_analysis.pdf'")
plt.close()

# ========================================================================
# 8. MODEL 1: SARIMA
# ========================================================================
print("\n" + "="*80)
print("MODEL 1: SARIMA (SEASONAL ARIMA - NO EXOGENOUS)")
print("="*80)
print("[7a/9] Selecting optimal SARIMA order on training data...")

modelo_sarima_auto = auto_arima(
    endog_train, 
    seasonal=True,
    m=12,
    trace=False,
    error_action='ignore', 
    suppress_warnings=True, 
    stepwise=True,
    max_P=2, max_Q=2, max_D=1,
    #Si se recomienda usar un d diferente de 0 se descomenta la linea siguiente y se comenta las demás líneas

    #max_p=5, max_q=5,   #Se descomenta si se recomienda un d diferente de 0
    max_p=5, max_q=5, max_d=2,  #se comenta si se utiloiza un d diferente de 0
    information_criterion='aic' #también esta se comenta
    #d=1        #Se descomenta si se recomienda un d diferente de 0
)

orden_sarima = modelo_sarima_auto.order
orden_estacional_sarima = modelo_sarima_auto.seasonal_order
print(f"   Order selected: ARIMA{orden_sarima} × {orden_estacional_sarima}[12]")

# Fit SARIMA model
modelo_sarima = SARIMAX(
    endog_train,
    order=orden_sarima,
    seasonal_order=orden_estacional_sarima,
    enforce_stationarity=False,
    enforce_invertibility=False
)
resultado_sarima = modelo_sarima.fit(disp=False)

# In-sample predictions
pred_sarima_train = resultado_sarima.get_prediction(start=endog_train.index[0], 
                                                     end=endog_train.index[-1])
pred_sarima_train_mean = pred_sarima_train.predicted_mean

# Out-of-sample forecast
forecast_sarima = resultado_sarima.forecast(steps=len(df_test))
forecast_sarima_series = pd.Series(forecast_sarima, index=endog_test.index)

# Metrics - Training
y_train_true = endog_train.values
y_train_pred_sarima = pred_sarima_train_mean.values
rmse_sarima_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred_sarima))
mae_sarima_train = mean_absolute_error(y_train_true, y_train_pred_sarima)
r2_sarima_train = r2_score(y_train_true, y_train_pred_sarima)

# Metrics - Test (OUT-OF-SAMPLE)
y_test_true = endog_test.values
y_test_pred_sarima = forecast_sarima_series.values
rmse_sarima_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred_sarima))
mae_sarima_test = mean_absolute_error(y_test_true, y_test_pred_sarima)
r2_sarima_test = r2_score(y_test_true, y_test_pred_sarima)

# Residual diagnostics
residuos_sarima = resultado_sarima.resid
lb_sarima = acorr_ljungbox(residuos_sarima, lags=20, return_df=True)

print(f"\n   SARIMA METRICS (IN-SAMPLE - TRAINING):")
print(f"   RMSE:  {rmse_sarima_train:.3f} CMEs/month")
print(f"   MAE:   {mae_sarima_train:.3f} CMEs/month")
print(f"   R²:    {r2_sarima_train:.4f}")

print(f"\n   SARIMA METRICS (OUT-OF-SAMPLE - TEST) *** KEY METRIC ***:")
print(f"   RMSE:  {rmse_sarima_test:.3f} CMEs/month")
print(f"   MAE:   {mae_sarima_test:.3f} CMEs/month")
print(f"   R²:    {r2_sarima_test:.4f}")
print(f"   AIC:   {resultado_sarima.aic:.2f}")

print(f"\n   Ljung-Box test (residuals whiteness):")
print(f"   p-value (lag 20): {lb_sarima['lb_pvalue'].iloc[-1]:.4f}")
if lb_sarima['lb_pvalue'].iloc[-1] > 0.05:
    print("    Residuals are white noise (p > 0.05)")
else:
    print("    Residuals show autocorrelation (p < 0.05)")

# ========================================================================
# 9. MODEL 2: SARIMAX
# ========================================================================
print("\n" + "="*80)
print("MODEL 2: SARIMAX (SEASONAL ARIMAX WITH SUNSPOTS)")
print("="*80)
print("[7b/9] Selecting optimal SARIMAX order on training data...")

modelo_sarimax_auto = auto_arima(
    endog_train, 
    seasonal=True,
    m=12,
    trace=False,
    error_action='ignore', 
    suppress_warnings=True, 
    stepwise=True,
    max_P=2, max_Q=2, max_D=1,
    #Si se recomienda usar un d diferente de 0 se descomenta la linea siguiente y se comenta las demás líneas

    #max_p=5, max_q=5,   #Se descomenta si se recomienda un d diferente de 0
    max_p=5, max_q=5, max_d=2,  #se comenta si se utiloiza un d diferente de 0
    information_criterion='aic' #también esta se comenta
    #d=1        #Se descomenta si se recomienda un d diferente de 0
)
orden_sarimax = modelo_sarimax_auto.order
orden_estacional_sarimax = modelo_sarimax_auto.seasonal_order
print(f"   Order selected: ARIMA{orden_sarimax} × {orden_estacional_sarimax}[12]")

# Fit SARIMAX model
modelo_sarimax = SARIMAX(
    endog_train, 
    exog=exog_train,
    order=orden_sarimax,
    seasonal_order=orden_estacional_sarimax,
    enforce_stationarity=False,
    enforce_invertibility=False
)
resultado_sarimax = modelo_sarimax.fit(disp=False)

# In-sample predictions
pred_sarimax_train = resultado_sarimax.get_prediction(start=endog_train.index[0], 
                                                       end=endog_train.index[-1], 
                                                       exog=exog_train)
pred_sarimax_train_mean = pred_sarimax_train.predicted_mean

# Out-of-sample forecast
forecast_sarimax = resultado_sarimax.forecast(steps=len(df_test), exog=exog_test)
forecast_sarimax_series = pd.Series(forecast_sarimax, index=endog_test.index)

# Metrics - Training
y_train_pred_sarimax = pred_sarimax_train_mean.values
rmse_sarimax_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred_sarimax))
mae_sarimax_train = mean_absolute_error(y_train_true, y_train_pred_sarimax)
r2_sarimax_train = r2_score(y_train_true, y_train_pred_sarimax)

# Metrics - Test (OUT-OF-SAMPLE)
y_test_pred_sarimax = forecast_sarimax_series.values
rmse_sarimax_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred_sarimax))
mae_sarimax_test = mean_absolute_error(y_test_true, y_test_pred_sarimax)
r2_sarimax_test = r2_score(y_test_true, y_test_pred_sarimax)

# Residual diagnostics
residuos_sarimax = resultado_sarimax.resid
lb_sarimax = acorr_ljungbox(residuos_sarimax, lags=20, return_df=True)

print(f"\n   SARIMAX METRICS (IN-SAMPLE - TRAINING):")
print(f"   RMSE:  {rmse_sarimax_train:.3f} CMEs/month")
print(f"   MAE:   {mae_sarimax_train:.3f} CMEs/month")
print(f"   R²:    {r2_sarimax_train:.4f}")

print(f"\n   SARIMAX METRICS (OUT-OF-SAMPLE - TEST) *** KEY METRIC ***:")
print(f"   RMSE:  {rmse_sarimax_test:.3f} CMEs/month")
print(f"   MAE:   {mae_sarimax_test:.3f} CMEs/month")
print(f"   R²:    {r2_sarimax_test:.4f}")
print(f"   AIC:   {resultado_sarimax.aic:.2f}")

print(f"\n   Ljung-Box test (residuals whiteness):")
print(f"   p-value (lag 20): {lb_sarimax['lb_pvalue'].iloc[-1]:.4f}")
if lb_sarimax['lb_pvalue'].iloc[-1] > 0.05:
    print("    Residuals are white noise (p > 0.05)")
else:
    print("    Residuals show autocorrelation (p < 0.05)")

# ========================================================================
# 10. MODEL COMPARISON
# ========================================================================
print("\n" + "="*80)
print("MODEL COMPARISON (OUT-OF-SAMPLE PERFORMANCE)")
print("="*80)

mejora_rmse = ((rmse_sarima_test - rmse_sarimax_test) / rmse_sarima_test) * 100
mejora_mae = ((mae_sarima_test - mae_sarimax_test) / mae_sarima_test) * 100
mejora_r2 = ((r2_sarimax_test - r2_sarima_test) / abs(r2_sarima_test)) * 100 if r2_sarima_test != 0 else 0
mejora_aic = resultado_sarima.aic - resultado_sarimax.aic

print(f"\n{'Metric':<20} {'SARIMA':<15} {'SARIMAX':<15} {'Improvement':<15}")
print("-"*65)
print(f"{'RMSE (test)':<20} {rmse_sarima_test:<15.3f} {rmse_sarimax_test:<15.3f} {mejora_rmse:>+.2f}%")
print(f"{'MAE (test)':<20} {mae_sarima_test:<15.3f} {mae_sarimax_test:<15.3f} {mejora_mae:>+.2f}%")
print(f"{'R² (test)':<20} {r2_sarima_test:<15.4f} {r2_sarimax_test:<15.4f} {mejora_r2:>+.2f}%")
print(f"{'AIC':<20} {resultado_sarima.aic:<15.2f} {resultado_sarimax.aic:<15.2f} {mejora_aic:>+.2f}")
print("-"*65)

if rmse_sarimax_test < rmse_sarima_test and resultado_sarimax.aic < resultado_sarima.aic:
    print(" SARIMAX is SUPERIOR: Lower error AND lower AIC")
elif rmse_sarimax_test < rmse_sarima_test:
    print(" SARIMAX has lower error, but similar AIC")
else:
    print(" SARIMA is comparable or superior")

print("="*80)

# ========================================================================
# 11. PUBLICATION-QUALITY VISUALIZATION (NUMPY ARRAYS)
# ========================================================================
print("\n[8/9] Generating publication-quality figures...")

# Convert to numpy arrays for plotting (CRITICAL FIX)
train_dates = endog_train.index.to_numpy()
test_dates = endog_test.index.to_numpy()

# ========== FIGURE 1: FORECAST COMPARISON ==========
fig1, ax1 = plt.subplots(figsize=(14, 6))

# Plot observed data
ax1.plot(train_dates, y_train_true, 
         linestyle='-', linewidth=1.2, color=COLOR_TRAIN, 
         label='Observed (training)', alpha=0.6, zorder=2)

ax1.plot(test_dates, y_test_true, 
         linestyle='-', linewidth=1.5, color=COLOR_OBSERVED, 
         label='Observed (test)', zorder=3)

# Plot fitted values (training) - lighter
ax1.plot(train_dates, y_train_pred_sarima, 
         linestyle='--', linewidth=1.0, color=COLOR_ARIMA, 
         alpha=0.4, zorder=1)

ax1.plot(train_dates, y_train_pred_sarimax, 
         linestyle='-.', linewidth=1.0, color=COLOR_ARIMAX, 
         alpha=0.4, zorder=1)

# Plot forecasts (test) - KEY VISUALIZATION
ax1.plot(test_dates, y_test_pred_sarima, 
         linestyle='--', linewidth=2.0, color=COLOR_ARIMA, 
         label=f'SARIMA{orden_sarima}×{orden_estacional_sarima}[12] (R²={r2_sarima_test:.3f})',
         zorder=2, alpha=0.9)

ax1.plot(test_dates, y_test_pred_sarimax, 
         linestyle='-.', linewidth=2.0, color=COLOR_ARIMAX, 
         label=f'SARIMAX{orden_sarimax}×{orden_estacional_sarimax}[12] (R²={r2_sarimax_test:.3f})',
         zorder=2, alpha=0.9)

# Vertical line at train/test split
split_date = pd.to_datetime(TRAIN_END) + pd.DateOffset(days=15)
ax1.axvline(x=split_date, color='black', linestyle=':', 
            linewidth=2, alpha=0.7, label='Train/Test split')

# Shaded test region
ax1.axvspan(test_dates[0], test_dates[-1], 
            alpha=0.08, color='gray', zorder=0)

ax1.set_xlabel('Date', fontsize=12, fontweight='medium')
ax1.set_ylabel('CME Count (per month)', fontsize=12, fontweight='medium')
ax1.set_title('Out-of-Sample Forecast: SARIMA vs SARIMAX for Monthly CME Occurrence', 
             fontsize=13, fontweight='bold', pad=15)

ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=COLOR_GRID)
ax1.set_axisbelow(True)

legend = ax1.legend(loc='upper left', frameon=True, fancybox=False, 
                   shadow=False, framealpha=0.95, edgecolor='black',
                   facecolor='white', fontsize=9)
legend.get_frame().set_linewidth(0.8)

textstr = f'Test RMSE improvement: {mejora_rmse:+.1f}%\nTest R² improvement: {mejora_r2:+.1f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.2, edgecolor='gray', linewidth=0.8)
ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('forecast_comparison_monthly_with_validation.pdf', dpi=600, bbox_inches='tight')
print("   ✓ Figure 1 saved: 'forecast_comparison_monthly_with_validation.pdf'")
plt.close()

# ========== FIGURE 2: RESIDUAL DIAGNOSTICS ==========
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# SARIMA residuals - time plot
axes[0, 0].plot(train_dates, residuos_sarima.values, 
                linestyle='-', color=COLOR_ARIMA, alpha=0.5, linewidth=0.8)
axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 0].fill_between(train_dates, residuos_sarima.values, 0, 
                        alpha=0.15, color=COLOR_ARIMA)
axes[0, 0].set_xlabel('Date', fontsize=10)
axes[0, 0].set_ylabel('Residual', fontsize=10)
axes[0, 0].set_title(f'SARIMA{orden_sarima}×{orden_estacional_sarima}[12] Residuals', 
                     fontsize=11, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# SARIMAX residuals - time plot
axes[0, 1].plot(train_dates, residuos_sarimax.values, 
                linestyle='-', color=COLOR_ARIMAX, alpha=0.5, linewidth=0.8)
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 1].fill_between(train_dates, residuos_sarimax.values, 0, 
                        alpha=0.15, color=COLOR_ARIMAX)
axes[0, 1].set_xlabel('Date', fontsize=10)
axes[0, 1].set_ylabel('Residual', fontsize=10)
axes[0, 1].set_title(f'SARIMAX{orden_sarimax}×{orden_estacional_sarimax}[12] Residuals', 
                     fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# SARIMA residuals - histogram
axes[1, 0].hist(residuos_sarima.values, bins=30, color=COLOR_ARIMA, 
                alpha=0.6, edgecolor='black')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Residual Value', fontsize=10)
axes[1, 0].set_ylabel('Frequency', fontsize=10)
axes[1, 0].set_title(f'SARIMA Residual Distribution (σ={residuos_sarima.std():.2f})', 
                     fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# SARIMAX residuals - histogram
axes[1, 1].hist(residuos_sarimax.values, bins=30, color=COLOR_ARIMAX, 
                alpha=0.6, edgecolor='black')
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Residual Value', fontsize=10)
axes[1, 1].set_ylabel('Frequency', fontsize=10)
axes[1, 1].set_title(f'SARIMAX Residual Distribution (σ={residuos_sarimax.std():.2f})', 
                     fontsize=11, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('residual_diagnostics_monthly.pdf', dpi=600, bbox_inches='tight')
print("    Figure 2 saved: 'residual_diagnostics_monthly.pdf'")
plt.close()

# ========================================================================
# 12. SAVE RESULTS
# ========================================================================
print("\n[9/9] Saving numerical results...")

# Combine all data
all_dates = np.concatenate([train_dates, test_dates])
all_observed = np.concatenate([y_train_true, y_test_true])
all_sarima = np.concatenate([y_train_pred_sarima, y_test_pred_sarima])
all_sarimax = np.concatenate([y_train_pred_sarimax, y_test_pred_sarimax])
all_sunspots = np.concatenate([exog_train['Sunspots'].values, exog_test['Sunspots'].values])

df_results = pd.DataFrame({
    "Date": all_dates,
    "YearMonth": pd.to_datetime(all_dates).strftime('%Y-%m'),
    "CMEs_observed": all_observed,
    "SARIMA_fitted": all_sarima,
    "SARIMAX_fitted": all_sarimax,
    "SARIMA_residuals": all_observed - all_sarima,
    "SARIMAX_residuals": all_observed - all_sarimax,
    "Sunspot_number": all_sunspots,
    "Dataset": ['Train']*len(df_train) + ['Test']*len(df_test)
})

df_results.to_csv("forecast_results_monthly_with_validation.csv", index=False)
print("    Numerical results saved: 'forecast_results_monthly_with_validation.csv'")

# Save metrics summary
with open('model_metrics_summary_monthly.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("SARIMA vs SARIMAX MODEL COMPARISON - MONTHLY CME FORECASTING\n")
    f.write("WITH OUT-OF-SAMPLE VALIDATION\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"DATA CONFIGURATION:\n")
    f.write(f"  Period: {YEAR_START}-01 to {YEAR_END}-12 ({len(df_merged)} months)\n")
    f.write(f"  CME filters: speed ≥ {MIN_SPEED} km/s, {MIN_WIDTH}° ≤ width ≤ {MAX_WIDTH}°\n")
    f.write(f"  Train/Test split: {TRAIN_END} / {TEST_START}\n")
    f.write(f"  Train size: {len(df_train)} months\n")
    f.write(f"  Test size: {len(df_test)} months\n\n")
    
    f.write(f"PRELIMINARY TESTS:\n")
    f.write(f"  ADF test (stationarity): stat={adf_result[0]:.4f}, p={adf_result[1]:.4f}\n")
    if adf_result[1] < 0.05:
        f.write(f"     Series is STATIONARY\n")
    else:
        f.write(f"     Series is NON-STATIONARY\n")
    f.write(f"  Ljung-Box test (white noise): p={lb_result['lb_pvalue'].iloc[-1]:.4f}\n")
    if lb_result['lb_pvalue'].iloc[-1] < 0.05:
        f.write(f"     Series shows AUTOCORRELATION\n\n")
    else:
        f.write(f"     Series is WHITE NOISE\n\n")
    
    f.write(f"CORRELATION ANALYSIS:\n")
    f.write(f"  Pearson r (SSN vs CME) = {correlacion:.4f} (p = {p_valor:.4e})\n\n")
    
    f.write(f"SARIMA{orden_sarima} × {orden_estacional_sarima}[12] MODEL:\n")
    f.write(f"  Training RMSE: {rmse_sarima_train:.3f} CMEs/month\n")
    f.write(f"  Training MAE:  {mae_sarima_train:.3f} CMEs/month\n")
    f.write(f"  Training R²:   {r2_sarima_train:.4f}\n")
    f.write(f"  ---\n")
    f.write(f"  Test RMSE:     {rmse_sarima_test:.3f} CMEs/month ***\n")
    f.write(f"  Test MAE:      {mae_sarima_test:.3f} CMEs/month ***\n")
    f.write(f"  Test R²:       {r2_sarima_test:.4f} ***\n")
    f.write(f"  AIC:           {resultado_sarima.aic:.2f}\n")
    f.write(f"  Ljung-Box p:   {lb_sarima['lb_pvalue'].iloc[-1]:.4f}\n\n")
    
    f.write(f"SARIMAX{orden_sarimax} × {orden_estacional_sarimax}[12] MODEL (with SSN):\n")
    f.write(f"  Training RMSE: {rmse_sarimax_train:.3f} CMEs/month\n")
    f.write(f"  Training MAE:  {mae_sarimax_train:.3f} CMEs/month\n")
    f.write(f"  Training R²:   {r2_sarimax_train:.4f}\n")
    f.write(f"  ---\n")
    f.write(f"  Test RMSE:     {rmse_sarimax_test:.3f} CMEs/month ***\n")
    f.write(f"  Test MAE:      {mae_sarimax_test:.3f} CMEs/month ***\n")
    f.write(f"  Test R²:       {r2_sarimax_test:.4f} ***\n")
    f.write(f"  AIC:           {resultado_sarimax.aic:.2f}\n")
    f.write(f"  Ljung-Box p:   {lb_sarimax['lb_pvalue'].iloc[-1]:.4f}\n\n")
    
    f.write(f"OUT-OF-SAMPLE IMPROVEMENTS (SARIMAX vs SARIMA):\n")
    f.write(f"  RMSE: {mejora_rmse:+.2f}%\n")
    f.write(f"  MAE:  {mejora_mae:+.2f}%\n")
    f.write(f"  R²:   {mejora_r2:+.2f}%\n")
    f.write(f"  ΔAIC: {mejora_aic:+.2f}\n\n")
    
    f.write("="*80 + "\n")
    f.write("*** Test metrics represent TRUE out-of-sample forecasting performance\n")
    f.write("="*80 + "\n")

print("    Metrics summary saved: 'model_metrics_summary_monthly.txt'")

print("\n" + "="*80)
print("MONTHLY ANALYSIS COMPLETED SUCCESSFULLY")
print("="*80)
print("\nGenerated files:")
print("  1. acf_pacf_monthly_analysis.pdf")
print("  2. forecast_comparison_monthly_with_validation.pdf")
print("  3. residual_diagnostics_monthly.pdf")
print("  4. forecast_results_monthly_with_validation.csv")
print("  5. model_metrics_summary_monthly.txt")
print("\nKey findings:")
print(f"  - Stationarity: ADF p-value = {adf_result[1]:.4f}")
print(f"  - Serial correlation: LB p-value = {lb_result['lb_pvalue'].iloc[-1]:.4f}")
print(f"  - Out-of-sample RMSE improvement: {mejora_rmse:+.1f}%")
print(f"  - Out-of-sample R² improvement: {mejora_r2:+.1f}%")
print(f"  - Correlation (SSN-CME): r = {correlacion:.3f}")
print("="*80)
