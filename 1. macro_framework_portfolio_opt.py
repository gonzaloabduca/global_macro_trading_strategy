import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import cvxpy as cp
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from curl_cffi import requests
import quantstats as qs


def zscore(data, window: int):

    mean = data.rolling(window).mean()
    std = data.rolling(window).std()

    return (data - mean) / std


def macro_roc(data, fast: int, slow: int, zs_window: int, use_diff: bool = False):

    if use_diff:
        fast_change = data.diff(fast)
        slow_change = data.diff(slow)
    else:
        fast_change = data.pct_change(fast)
        slow_change = data.pct_change(slow)

    accel = fast_change - slow_change

    return zscore(accel, zs_window)


def trend_ind(data, trading_periods = 252):
    
    returns = data.pct_change()

    true_range = returns.rolling(60).std()*np.sqrt(trading_periods)
    true_range = true_range.squeeze()

    basic_upper_band = data * (1 + true_range)
    basic_lower_band = data * (1 - true_range)

    # Convert bands to Series we can modify
    final_upper_band = basic_upper_band.copy()
    final_lower_band = basic_lower_band.copy()

    # Initialize uptrend Series
    uptrend = pd.Series(index=data.index, dtype=bool)
    uptrend.iloc[0] = True  # Starting assumption: trend is up

    # Supertrend calculation loop
    for current in range(1, len(data)):

        previous = current - 1

        if data.iloc[current] > final_upper_band.iloc[previous]:
            uptrend.iloc[current] = True
        elif data.iloc[current] < final_lower_band.iloc[previous]:
            uptrend.iloc[current] = False
        else:
            uptrend.iloc[current] = uptrend.iloc[previous]

            if uptrend.iloc[current] and final_lower_band.iloc[current] < final_lower_band.iloc[previous]:
                final_lower_band.iloc[current] = final_lower_band.iloc[previous]

            if not uptrend.iloc[current] and final_upper_band.iloc[current] > final_upper_band.iloc[previous]:
                final_upper_band.iloc[current] = final_upper_band.iloc[previous]

    supertrend = pd.Series(index=data.index)
    supertrend[uptrend] = final_lower_band[uptrend]
    supertrend[~uptrend] = final_upper_band[~uptrend]

    return supertrend

def macd_zs(data, fast: int, slow: int, signal: int, zs_window: int):

    fast_ema = data.ewm(span=fast, adjust=False).mean()
    slow_ema = data.ewm(span=slow, adjust=False).mean()

    macd = fast_ema - slow_ema
    hist = macd.ewm(span=signal, adjust=False).mean()

    return zscore(macd-hist, window=zs_window)


def rolling_sortino_ratio(returns_df, rf_series, window=252, periods_per_year=252):
    
    df = returns_df.join(rf_series.rename('rf')).dropna()
    
    excess = df[returns_df.columns].sub(df['rf'], axis=0)
    
    rolling_mean = excess.rolling(window).mean() * periods_per_year
    
    downside = excess.clip(upper=0)
    rolling_downside_std = downside.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    
    rolling_sortino = rolling_mean / rolling_downside_std
    
    return rolling_sortino

def rolling_tail_ratio(df, window=252, upper_q=0.90, lower_q=0.10):
    upper = df.rolling(window).quantile(upper_q)
    lower = df.rolling(window).quantile(lower_q).abs()
    tail_ratio = upper / lower
    zs_tail = (tail_ratio - tail_ratio.rolling(756).mean()) / tail_ratio.rolling(756).std()
    return zs_tail


session = requests.Session(impersonate="chrome")

fred = Fred(api_key='a2fb338b4ef6e2dcb7c667c21b2d1c4e')

#MARKET DATA
spx = yf.download('^GSPC', start='1900-01-01', auto_adjust=True)['Close'].squeeze().pct_change().dropna()
ndx = yf.download('^IXIC', start='1900-01-01', auto_adjust=True)['Close'].squeeze().pct_change().dropna()
tlt = yf.download('TLT', start='1900-01-01', auto_adjust=True)['Close'].squeeze().pct_change().dropna()
sdy = yf.download('SDY', start='1900-01-01', auto_adjust=True)['Close'].squeeze().pct_change().dropna()


markets = pd.concat([
    spx, ndx, tlt, sdy
], axis=1).dropna()

rf = fred.get_series('DTB3').interpolate() / 100
rf = rf / 252
rf = rf.reindex(spx.index).ffill().dropna()

files = [
    "industry_daily_returns",
    "maket_size_daily_data",
    "momentum_market_size_daily_data",
    "operating_profit_market_size_daily_data",
    "size_book_to_market_daily_returns",
    "ff_factors_daily",
    "industries_10_daily"
]

dfs = []

for f in files:

    data = pd.read_csv(f"C:/Users/User/Desktop/Data Projects/Portfolio Optimisation/data/{f}.csv")

    data['Date'] = pd.to_datetime(data['Date'], format="%Y%m%d")

    data = data.set_index('Date')

    data = data.replace([-99.99, -999.99], np.nan)

    data = data / 100

    dfs.append(data)


industries = dfs[0]
market_size = dfs[1][['Lo 30', 'Med 40', 'Hi 30']]
momentum_size = dfs[2]
oprofit_size = dfs[3][['SMALL LoOP', 'SMALL HiOP', 'BIG LoOP','BIG HiOP']]
book_mkt_size = dfs[4][['SMALL LoBM', 'SMALL HiBM', 'BIG LoBM', 'BIG HiBM']]
cicl_def_ind = dfs[6]

### Cyclicals vs. Defensives ####

cyc = zscore(((1+ (cicl_def_ind[['Durbl','Manuf','HiTec']].mean(axis=1))).cumprod()).pct_change(252), window=756).dropna()
defn = zscore(((1+ (cicl_def_ind[['NoDur','Hlth','Utils']].mean(axis=1))).cumprod()).pct_change(252), window=756).dropna()

cyc_def = cyc - defn

### Momentum Factor ####

winners = zscore(((1+ (dfs[2][['SMALL HiPRIOR', 'BIG HiPRIOR']].mean(axis=1))).cumprod()).pct_change(252), window=756).dropna()
losers  = zscore(((1+ (dfs[2][['SMALL LoPRIOR', 'BIG LoPRIOR']].mean(axis=1))).cumprod()).pct_change(252), window=756).dropna()

momentum_factor = winners - losers

### High Beta vs Low Beta ####

high_beta = zscore(((1+ (cicl_def_ind[['Durbl','HiTec']].mean(axis=1))).cumprod()).pct_change(252), window=756).dropna()
low_beta  = zscore(((1+ (cicl_def_ind[['Utils','NoDur']].mean(axis=1))).cumprod()).pct_change(252), window=756).dropna()

beta_spread = high_beta - low_beta

### Market Breath ###

industries_index = (1+industries).cumprod().dropna()

rolling_mean = industries_index.rolling(252).mean()

above_ma = (industries_index > rolling_mean).mean(axis=1)

above_ma = above_ma.rolling(10).mean().dropna()

factors_features = (pd.concat({
    "cyclical_vs_defensives" : cyc_def,
    "momentum_factor" : momentum_factor,
    "beta_factor" : beta_spread,
    "market_breadth" : above_ma
    },axis=1).dropna()).resample('ME').last()


## Volatility_features

realized_vol = spx.rolling(21).std()

vol_last = realized_vol.resample('ME').last()
vol_max  = realized_vol.resample('ME').max()

vol_change = np.log(vol_last).diff()

vol_percentile = realized_vol.rolling(252).rank(pct=True).resample('ME').last()

vol_short = spx.rolling(10).std()
vol_long  = spx.rolling(63).std()

vol_ratio = (vol_short / vol_long).resample('ME').last()


vol_features = pd.concat(
    {'vol_last': vol_last,
    "vol_max": vol_max,
    "vol_change": vol_change,
    "vol_percentile" : vol_percentile,
    "vol_ratio" : vol_ratio}, axis=1).dropna()

vol_features_zs = zscore(vol_features, window=60).dropna()

market_features = pd.concat([factors_features, vol_features_zs], axis=1).dropna()

### CALCULATE SORTINO AND VOLATILITIES FOR PERFORMANCE MEASURES ###

market_dfs = [
    industries,
    market_size,
    oprofit_size,
    book_mkt_size,
    markets,
    cicl_def_ind
]

market_sortinos = []

for df in market_dfs:
    
    df_sortino = rolling_sortino_ratio(df, rf).dropna()
    market_sortinos.append(df_sortino)


market_vols = []

for df in market_dfs :
    
    neg_rets = df.clip(upper=0)
    df_volatility = neg_rets.rolling(60).std()*np.sqrt(252)
    vol_zs = (df_volatility - df_volatility.rolling(756).mean()) / df_volatility.rolling(756).std()
    market_vols.append(vol_zs)

market_tails = []

for df in market_dfs:

    tail_ratio = rolling_tail_ratio(df)
    market_tails.append(tail_ratio)

#######################################################################
################### MACROECONOMIC DATA ################################
#######################################################################


ism = pd.read_csv("C:/Users/User/Desktop/Data Projects/Portfolio Optimisation/data/ism_data.csv", index_col=0)
ism.index = pd.to_datetime(ism.index, errors='coerce') + pd.offsets.MonthEnd(0)

ism_momentum = macro_roc(ism, fast=3, slow = 12, zs_window = 60, use_diff=True).dropna()
ism_5y_zs = zscore(ism, window=60).dropna()

ism_vol = zscore(ism.diff().rolling(6).std(), window=60)

nfib = pd.read_excel("C:/Users/User/Desktop/Data Projects/Apps/Macro Dashboard App/misc/NFIB.xlsx").set_index('Date').interpolate()
nfib.index = pd.to_datetime(nfib.index, errors='coerce') + pd.offsets.MonthEnd(0)

nfib_roc = macro_roc(nfib, fast=3, slow = 12, zs_window = 60, use_diff=True).dropna()

nfib_zs = zscore(nfib, window=60).dropna()

nfib_vol = zscore(nfib.diff().rolling(3).std(), window=60)


permits = fred.get_series('PERMIT').interpolate()
permits.index = pd.to_datetime(permits.index, errors='coerce') + pd.offsets.MonthEnd(0)

permits_momentum = macro_roc(permits, fast=3, slow = 12, zs_window = 60, use_diff=False).dropna()

permits_level_zs = zscore(permits, window=60).dropna()
permits_3y_zs = zscore(permits.pct_change(12), window=60).dropna()
permits_5y_zs = zscore(permits.pct_change(12), window=60).dropna()
permits_10y_zs = zscore(permits.pct_change(12), window=120).dropna()

permits_vol = zscore(permits.pct_change().rolling(12).std(), window=60)

umcsi = fred.get_series('UMCSENT').interpolate()
umcsi.index = pd.to_datetime(umcsi.index, errors='coerce') + pd.offsets.MonthEnd(0)

umcsi_level_5y_zs = zscore(umcsi, window=60).dropna()

umcsi_momentum = macro_roc(umcsi, fast=3, slow = 12, zs_window = 60, use_diff=True).dropna()

ind_prod_series = fred.get_series('INDPRO').interpolate()
ind_prod = ind_prod_series.pct_change(12).dropna()
ind_prod.index = pd.to_datetime(ind_prod.index, errors='coerce') + pd.offsets.MonthEnd(0)

ind_prod_momentum = macro_roc(ind_prod, fast=3, slow = 12, zs_window = 60, use_diff=True).dropna()

ind_prod_level_5y_zs = zscore(ind_prod, window=60).dropna()

ind_prod_vol = zscore(ind_prod_series.pct_change().rolling(12).std(), window=60).resample('ME').last() 

jobless_claims = fred.get_series('ICSA').interpolate()
jobless_claims.index = pd.to_datetime(jobless_claims.index, errors='coerce')

jobless_claims_momentum = macro_roc(jobless_claims, fast=12, slow=48, zs_window=240, use_diff = False).resample('ME').last()

jobless_claims_3y_zs = zscore(jobless_claims, window=144).resample('ME').last()

jobless_claims_vol = zscore(jobless_claims.pct_change().rolling(52).std(),window=240).resample('ME').last()

cpi_series = fred.get_series('CPIAUCSL').interpolate()

cpi_series.index = pd.to_datetime(cpi_series.index, errors='coerce') + pd.offsets.MonthEnd(0)

cpi = cpi_series.pct_change(12).dropna()

cpi_momentum = macro_roc(cpi, fast=3, slow=12, zs_window=60, use_diff=True)

cpi_5y_zs = zscore(cpi, window=60)

cpi_vol = zscore(cpi_series.pct_change().rolling(12).std(), window=60)


macro_features = pd.concat({
    **{f'{c}_momentum' : ism_momentum[c] for c in ism_momentum.columns},
    **{f'{c}_5y_zs' : ism_5y_zs[c] for c in ism_5y_zs.columns},
    **{f'{c}_vol' : ism_vol[c] for c in ism_vol.columns},
    'permits_momentum': permits_momentum,
    'permits_level_zs' : permits_level_zs,
    'permits_3y_zs':permits_3y_zs,
    'permits_5y_zs':permits_5y_zs,
    'permits_10y_zs':permits_10y_zs,
    'permits_vol':permits_vol,
    'umcsi_level_5y_zs':umcsi_level_5y_zs,
    'umcsi_momentum':umcsi_momentum,
    'jobless_claims_momentum':jobless_claims_momentum,
    'jobless_claims_3y_zs':jobless_claims_3y_zs,
    'jobless_claims_vol':jobless_claims_vol,
    'ind_prod_momentum':ind_prod_momentum,
    'ind_prod_level_5y_zs':ind_prod_level_5y_zs,
    'ind_prod_vol':ind_prod_vol,
    'cpi_momentum':cpi_momentum,
    'cpi_5y_zs':cpi_5y_zs,
    'cpi_vol':cpi_vol
    },axis=1).dropna()


#######################################################################
################### Money Markets Data ################################
#######################################################################

dgs10 = fred.get_series('DGS10').interpolate()
dtb3 = fred.get_series('DTB3').interpolate()

yield_curve_5y_zs = zscore(dgs10 - dtb3, window=1260).resample('ME').last()

yield_curve_momentum = macro_roc(dgs10 - dtb3, fast=60, slow=252, zs_window=756, use_diff = True).resample('ME').last()

yield_curve_vol = zscore((dgs10 - dtb3).diff().rolling(60).std(), window=756).resample('ME').last()


dg10_5y_zs = zscore(dgs10, window=1260).resample('ME').last()

dgs10_momentum = macro_roc(dgs10, fast=60, slow=252, zs_window=756, use_diff = True).resample('ME').last()

dgs10_vol = zscore(dgs10.diff().rolling(60).std(), window=756).resample('ME').last()

m2_series = fred.get_series('M2SL').interpolate()
m2_series.index = pd.to_datetime(m2_series.index, errors='coerce') + pd.offsets.MonthEnd(0)

m2_yoy_5y_zs = zscore(m2_series.pct_change(12), window=60)

m2_momentum = macro_roc(m2_series, fast=3, slow=12, zs_window=60, use_diff = False)

m2_vol = zscore(m2_series.pct_change().rolling(6).std(), window=60)

rir = ((cpi.resample('ME').last()) - (dtb3/100).resample('ME').last()).dropna()

rir_5y_zs = zscore(rir, window=60)

rir_momentum = macro_roc(rir, fast=3, slow=12, zs_window=60, use_diff=True)

rir_vol = zscore(rir.diff().rolling(6).std(), window=60)

money_features = pd.concat({
                'yield_curve_5y_zs':yield_curve_5y_zs,
                'yield_curve_momentum':yield_curve_momentum,
                'yield_curve_vol':yield_curve_vol,
                'dg10_5y_zs':dg10_5y_zs,
                'dgs10_momentum':dgs10_momentum,
                'dgs10_vol':dgs10_vol,
                'm2_yoy_5y_zs':m2_yoy_5y_zs,
                'm2_momentum':m2_momentum,
                'm2_vol':m2_vol,
                'rir_5y_zs':rir_5y_zs,
                'rir_momentum':rir_momentum,
                'rir_vol':rir_vol,
                }, axis=1).dropna()


#######################################################################
################### Macro Modeling ####################################
#######################################################################

df_all = pd.concat([
    macro_features,
    money_features,
    market_features
    ], axis=1).dropna()

macro_cols = [c for c in df_all if c in macro_features.columns]
money_cols = [c for c in df_all if c in money_features.columns]
market_cols = [c for c in df_all if c in market_features.columns]

X_macro = StandardScaler().fit_transform(df_all[macro_cols])
X_money = StandardScaler().fit_transform(df_all[money_cols])
X_market = StandardScaler().fit_transform(df_all[market_cols])

pca_macro = PCA(n_components=0.8, random_state=42)
pca_money = PCA(n_components=0.8, random_state=42)
pca_market = PCA(n_components=0.8, random_state=42)

macro_factors = pca_macro.fit_transform(X_macro)
money_factors = pca_money.fit_transform(X_money)
market_factors = pca_market.fit_transform(X_market)

check_loadings_macro = pd.DataFrame(
    pca_macro.components_,  # only works after fitting
    columns=macro_features.columns,
    index=[f'PC{i+1}' for i in range(pca_macro.n_components_)]
)

check_loadings_market = pd.DataFrame(
    pca_market.components_,  # only works after fitting
    columns=market_features.columns,
    index=[f'PC{i+1}' for i in range(pca_market.n_components_)]
)

X = np.column_stack([
    macro_factors[:,:3],
    money_factors[:,:3],
    (market_factors * 1.25)
]
)

global_cols = ['growth', 'inflation/stress',
               'transition', 'liquidity_impulse',
               'rate_volatility', 'policy_tightness',
               'vol_regime', 'cyc_vs_def', 'volat_acc',
               'momentum']

pca_global = PCA(n_components=4)

X_global = pca_global.fit_transform(X)

pca_global.explained_variance_ratio_

check_loadings_global = pd.DataFrame(
    pca_global.components_,  # only works after fitting
    columns=global_cols,
    index=[f'PC{i+1}' for i in range(pca_global.n_components_)]
)

print(f'explained variance ratio  for global: {pca_global.explained_variance_ratio_}')
print('-------------------------------------------------------------------------')
print(f'Loadings for global pcas:')
print(f'              ')
print(check_loadings_global)


#### Hidden Markov Model ####

pca_global_cols= [
        'global_growth',
        'liquidity-inflation',
        'regime_transition',
        'market_volatility'
         ]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_global)

model = GaussianHMM(
    n_components=4,
    covariance_type="full",
    n_iter=1000,
    random_state=42
)

model.fit(X_scaled)

states = model.predict(X_scaled)

state_probs = model.predict_proba(X_scaled) 

transmat = model.transmat_

state_names = [f"Regime {i}" for i in range(model.n_components)]

transmat_df = pd.DataFrame(
    transmat,
    index=state_names,
    columns=state_names
)

print('Model Study using 4 PCA FACTORS multiplying market factors * 1.25:')

print('*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*')

print("Transition Matrix:")
print((transmat_df*100).round(2))
print("-.-.-.-.-.-.-.-.-.-.--.-.-.-.-.-.-.-.-.-.-")
means = pd.DataFrame(
    model.means_,
    columns=pca_global_cols,
    index=state_names
)

print(means)

industries_score = ((0.7 * market_sortinos[0] + 0.3 * market_tails[0])).dropna()

states_series = pd.Series(states, index=df_all.index, name='regime')

states_daily = states_series.reindex(industries_score.index, method='ffill')

df_score = industries_score.copy()

df_score.loc[:,'market_regime'] = states_daily.copy()

industries_regimes = df_score.groupby('market_regime').mean().T


print('Descriptive stats of industries score (0.7 * sortino + 0.3 * tail ratio)')
print(industries_regimes.describe())

print('Regimes by % of occurences')
print(f'{(round(states_daily.value_counts()/states_daily.count() * 100, 2))}')

regime_probs = (pd.DataFrame(
    state_probs,
    index=df_all.index,
    columns=[f"Regime_{i}" for i in range(model.n_components)]
)*100).round(2)

print('Crisis Testing:')
print('------------------')
print('Tequila Effect 1994')
print(regime_probs.loc['1994-01-01':'1996-01-01'])
print('------------------')
print('DOT-COM Bubble')
print(regime_probs.loc['1999-01-01':'2002-01-01'])
print('------------------')
print('Great Financial Crisis 2008')
print(regime_probs.loc['2007-01-01':'2010-01-01'])
print('------------------')
print('COVID - 19')
print(regime_probs.loc['2019-01-01':'2022-01-01'])


#######################################################################
################### TRADING AND PORTFOLIO #############################
######################## MANAGEMENT ###################################
#######################################################################


regime_industry_map = {}

for regime in range(0, 4):
    
    long_list = list(
        industries_regimes[regime]
        .loc[(industries_regimes[regime] > 1)
             |(industries_regimes[regime] > np.percentile(industries_regimes[regime], 70))]
        .sort_values(ascending=False)
        .index
    )

    short_list = list(
        industries_regimes[regime]
        .loc[(industries_regimes[regime] < 0.5)
             |(industries_regimes[regime] < np.percentile(industries_regimes[regime], 30))]
        .sort_values(ascending=True)
        .index
    )
    
    regime_industry_map[regime] = {
        "long": [l for l in long_list if l not in short_list],
        "short": [s for s in short_list if s not in long_list]
    }


industries_returns = industries

industry_indexes = (1+industries).cumprod()


data = industry_indexes.dropna().stack().to_frame('price')
data.index.names = ['Date', 'Industry']

data['trend'] = data.groupby('Industry')['price'].transform(lambda x: trend_ind(x, trading_periods=126))

data['returns'] = data.groupby('Industry')['price'].transform(lambda x: x.pct_change())

data['signal'] = np.where(data['price'] > data['trend'], 1, -1)
data['signal'] = data.groupby(level='Industry')['signal'].shift(1)

data['regime'] = data.index.get_level_values('Date').map(states_daily)

data.dropna(inplace=True)

#### For each monthly regime date,
#  build the long and short industry lists that are both regime-eligible and technically confirmed,
#  then store them at the next trading date for implementation.

portfolio_dict = {}

available_dates = pd.Index(sorted(data.index.get_level_values('Date').unique()))

for date, regime in states_series.items():

    if pd.isna(regime):
        continue

    regime = int(regime)

    # first available trading date strictly after the regime date
    next_idx = available_dates.searchsorted(date, side='right')

    if next_idx >= len(available_dates):
        continue

    calibration_date = available_dates[next_idx]
    daily_data = data.xs(calibration_date, level='Date')

    long_candidates = regime_industry_map[regime]["long"]
    short_candidates = regime_industry_map[regime]["short"]

    long_list = [
        ind for ind in long_candidates
        if ind in daily_data.index and daily_data.loc[ind, 'signal'] == 1
    ]

    short_list = [
        ind for ind in short_candidates
        if ind in daily_data.index and daily_data.loc[ind, 'signal'] == -1
    ]

    if len(short_list) < 5:
        short_list = short_candidates[:5]

    if len(long_list) < 5:
        long_list = long_candidates[:5]

    key = calibration_date.strftime('%Y-%m-%d')

    portfolio_dict[key] = {
        "regime": regime,
        "long": long_list,
        "short": short_list
    }



def compute_downside_covariance(
    returns_df: pd.DataFrame,
    mar: float = 0.0,
    annualize: bool = False,
    periods_per_year: int = 252,
    shrink_diag: float = 1e-6,
) -> pd.DataFrame:
    """
    Build a downside covariance matrix from asset returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset return matrix, columns are assets, rows are dates.
    mar : float, default 0.0
        Minimum acceptable return per period.
    annualize : bool, default False
        Whether to annualize the covariance.
    periods_per_year : int, default 252
        Used only if annualize=True.
    shrink_diag : float, default 1e-6
        Small diagonal ridge to ensure PSD / numerical stability.

    Returns
    -------
    pd.DataFrame
        Downside covariance matrix.
    """

    rets = returns_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if rets.empty:
        raise ValueError("No valid return rows after cleaning.")

    # Downside deviations relative to MAR
    downside = np.minimum(rets - mar, 0.0)
    
    sigma_down = downside.cov()

    if annualize:
        sigma_down = sigma_down * periods_per_year

    # Ridge regularization for stability
    sigma_down = sigma_down.copy()
    diag_idx = np.diag_indices_from(sigma_down.values)
    sigma_down.values[diag_idx] += shrink_diag

    return sigma_down


def convex_downside_risk_budgeting_optimizer(
    returns_df: pd.DataFrame,
    long_list: list[str],
    short_list: list[str],
    gross_target: float = 2.0,
    net_target: float = 0.0,
    max_position: float = 0.20,
    min_position_frac: float = 0.0,
    mar: float = 0.0,
    annualize_cov: bool = False,
    periods_per_year: int = 252,
    l2_penalty: float = 0.0,
    turnover_penalty: float = 0.0,
    prev_weights: pd.Series | None = None,
    solver=cp.SCS,
) -> dict:
    """
    Convex long/short downside-risk budgeting optimizer.

    IMPORTANT:
    - This is convex.
    - It is NOT exact equal-risk-contribution.
    - It minimizes downside variance subject to long/short exposure budgets.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset return matrix.
    long_list : list[str]
        Assets eligible only for long positions.
    short_list : list[str]
        Assets eligible only for short positions.
    gross_target : float
        Total gross exposure: sum(long) + sum(short).
    net_target : float
        Net exposure: sum(long) - sum(short).
    max_position : float
        Max gross weight per asset within its side.
    min_position_frac : float
        Fraction of equal-weight side budget enforced as minimum per asset.
        Example:
            if long_target=1.2, n_long=6, equal weight = 0.2,
            and min_position_frac=0.25 -> min long weight = 0.05
    mar : float
        Minimum acceptable return for downside covariance.
    annualize_cov : bool
        Whether to annualize downside covariance.
    periods_per_year : int
        Annualization factor.
    l2_penalty : float
        Ridge penalty on weights for smoother allocation.
    turnover_penalty : float
        L2 penalty against previous weights.
    prev_weights : pd.Series | None
        Previous signed weights indexed by asset names.
    solver
        CVXPY solver.

    Returns
    -------
    dict
        Optimization result dictionary.
    """
    if returns_df is None or returns_df.empty:
        return {
            "status": "empty_returns",
            "weights": None,
            "gross": None,
            "net": None,
            "downside_variance": None,
            "long_target": None,
            "short_target": None,
        }

    # Filter universe
    long_assets = [a for a in long_list if a in returns_df.columns]
    short_assets = [a for a in short_list if a in returns_df.columns]

    overlap = set(long_assets).intersection(short_assets)
    if overlap:
        long_assets = [a for a in long_assets if a not in overlap]
        short_assets = [a for a in short_assets if a not in overlap]

    if len(long_assets) == 0 or len(short_assets) == 0:
        return {
            "status": "invalid_universe",
            "weights": None,
            "gross": None,
            "net": None,
            "downside_variance": None,
            "long_target": None,
            "short_target": None,
        }
    
    # Exposure targets
    long_target = 0.5 * (gross_target + net_target)
    short_target = 0.5 * (gross_target - net_target)

    if long_target < 0 or short_target < 0:
        return {
            "status": "invalid_targets",
            "weights": None,
            "gross": None,
            "net": None,
            "downside_variance": None,
            "long_target": long_target,
            "short_target": short_target,
        }

    max_position = 0.25
    # Capacity-aware scaling so the problem stays feasible
    long_capacity = len(long_assets) * max_position
    short_capacity = len(short_assets) * max_position

    scale_long = min(1.0, long_capacity / long_target) if long_target > 0 else 1.0
    scale_short = min(1.0, short_capacity / short_target) if short_target > 0 else 1.0
    scale = min(scale_long, scale_short)

    long_target *= scale
    short_target *= scale

    selected_assets = long_assets + short_assets
    aligned_returns = returns_df[selected_assets].replace([np.inf, -np.inf], np.nan).dropna(how="any")

    if aligned_returns.empty:
        return {
            "status": "no_data_after_alignment",
            "weights": None,
            "gross": None,
            "net": None,
            "downside_variance": None,
            "long_target": long_target,
            "short_target": short_target,
        }

    sigma_down = compute_downside_covariance(
        aligned_returns,
        mar=mar,
        annualize=annualize_cov,
        periods_per_year=periods_per_year,
        shrink_diag=1e-6,
    )

    n_long = len(long_assets)
    n_short = len(short_assets)
    n_assets = n_long + n_short

    # PSD-safe matrix
    Sigma = sigma_down.values
    Sigma = 0.5 * (Sigma + Sigma.T)

    # Decision variables
    wL = cp.Variable(n_long, nonneg=True)
    wS = cp.Variable(n_short, nonneg=True)

    # Signed weights
    w = cp.hstack([wL, -wS])

    min_position_frac=0
    # Minimum position floors if desired
    min_long_position = min_position_frac * (long_target / n_long) if n_long > 0 else 0.0
    min_short_position = min_position_frac * (short_target / n_short) if n_short > 0 else 0.0

    constraints = [
        wL <= max_position,
        wS <= max_position,
        cp.sum(wL) == long_target,
        cp.sum(wS) == short_target,
    ]

    if min_position_frac > 0:
        constraints += [
            wL >= min_long_position,
            wS >= min_short_position,
        ]

    objective_terms = [
        cp.quad_form(w, Sigma)
    ]

    if l2_penalty > 0:
        objective_terms.append(l2_penalty * cp.sum_squares(w))

    if turnover_penalty > 0 and prev_weights is not None:
        prev = prev_weights.reindex(selected_assets).fillna(0.0).values
        objective_terms.append(turnover_penalty * cp.sum_squares(w - prev))

    objective = cp.Minimize(cp.sum(objective_terms))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    if wL.value is None or wS.value is None:
        return {
            "status": problem.status,
            "weights": None,
            "gross": None,
            "net": None,
            "downside_variance": None,
            "long_target": long_target,
            "short_target": short_target,
        }

    weights = pd.Series(
        np.concatenate([wL.value, -wS.value]),
        index=selected_assets,
        name="weight",
    )

    downside_variance = float(weights.values @ Sigma @ weights.values)

    return {
        "status": problem.status,
        "weights": weights,
        "gross": float(weights.abs().sum()),
        "net": float(weights.sum()),
        "downside_variance": downside_variance,
        "long_target": float(long_target),
        "short_target": float(short_target),
        "long_assets": long_assets,
        "short_assets": short_assets,
        "selected_assets": selected_assets,
        "downside_covariance": sigma_down,
    }




###################################################################

def dynamic_gross_target(
    spx_returns,
    start_date,
    vol_window=63,
    target_vol=0.18,
    base_gross=2.0,
    min_gross=0.75,
    max_gross=2.5
):
    """
    Gross exposure scales inversely with SPX realized volatility.

    Higher realized vol  -> lower gross
    Lower realized vol   -> higher gross
    """

    realized_vol = (
        spx_returns
        .loc[:start_date]
        .dropna()
        .tail(vol_window)
        .std()
        * np.sqrt(252)
    )

    if pd.isna(realized_vol) or realized_vol == 0:
        return base_gross, np.nan

    gross = base_gross * (target_vol / realized_vol)
    gross = np.clip(gross, min_gross, max_gross)

    return gross, realized_vol

final_returns = []

rebalance_dates = list(portfolio_dict.keys())

for i, start_date in enumerate(rebalance_dates):
    try:
        start_date = pd.to_datetime(start_date)

        # Holding period: until next rebalance date - 1 day
        if i < len(rebalance_dates) - 1:
            next_start = pd.to_datetime(rebalance_dates[i + 1])
            end_date = next_start - pd.Timedelta(days=1)
        else:
            end_date = start_date + pd.offsets.MonthEnd(0)

        entry = portfolio_dict[start_date.strftime('%Y-%m-%d')]
        long_list = entry['long']
        short_list = entry['short']
        regime = entry['regime']

        selected_assets = [x for x in (long_list + short_list) if x in industries.columns]

        if len(long_list) == 0 or len(short_list) == 0 or len(selected_assets) == 0:
            print(start_date, "skip empty universe")
            continue

        # Estimation window
        optimization_start_date = start_date - pd.DateOffset(months=12)
        optimization_end_date = start_date - pd.Timedelta(days=1)

        estimation_returns = industries.loc[
            optimization_start_date:optimization_end_date,
            selected_assets
        ].dropna()

        if estimation_returns.empty:
            print(start_date, "skip empty estimation window")
            continue

        # Actual tradable assets that survive estimation window
        long_assets = [x for x in long_list if x in estimation_returns.columns]
        short_assets = [x for x in short_list if x in estimation_returns.columns]

        if len(long_assets) == 0 or len(short_assets) == 0:
            print(start_date, "skip no valid long/short assets after estimation filter")
            continue

        nL = len(long_assets)
        nS = len(short_assets)

        breadth_signal = (nL - nS) / (nL + nS)
        
        regime_bias = {0:  0.25,
                       1: -0.25,
                       2:  0.10,
                       3: -0.10
                       }

        net_target = np.clip(1.5 * breadth_signal + regime_bias[regime], -1.0, 1.0)
        
        rf_window = rf.loc[optimization_start_date:optimization_end_date]

        opt_portfolio = convex_downside_risk_budgeting_optimizer(
            returns_df=estimation_returns,
            long_list=long_assets,
            short_list=short_assets,
            gross_target=2.0,
            net_target=net_target,
            max_position= 0.25,
            min_position_frac= 0.0
        )

        if opt_portfolio['weights'] is None:
            print(
                start_date,
                f"optimizer failed | status={opt_portfolio['status']} | "
                f"{opt_portfolio.get('message', '')}"
            )
            continue

        weights = opt_portfolio['weights']

        print(
                start_date,
                "gross:", weights.abs().sum(),
                "net:", weights.sum(),
                "min_w:", weights.min(),
                "max_w:", weights.max()
            )

        if weights.abs().sum() == 0:
            print(start_date, "skip zero weights")
            continue

        # Forward holding-period returns
        forward_returns = industries.loc[start_date:end_date, weights.index].dropna()

        if forward_returns.empty:
            print(start_date, "skip empty forward returns")
            continue

        stacked = forward_returns.stack().to_frame('return')
        stacked.index.names = ['Date', 'Industry']

        stacked['weights'] = stacked.index.get_level_values('Industry').map(weights)
        stacked['weighted_return'] = stacked['return'] * stacked['weights']

        period_returns = stacked.groupby(level='Date')['weighted_return'].sum()
        final_returns.append(period_returns)

        print(
            start_date,
            f"success | gross={opt_portfolio['gross']:.2f} | "
            f"net={opt_portfolio['net']:.2f} | "
            f"long_gross={opt_portfolio['long_gross']:.2f} | "
            f"short_gross={opt_portfolio['short_gross']:.2f}"
        )

    except Exception as e:
        print(f"{start_date}: {e}")

final_portfolio = pd.concat(final_returns).sort_index()
final_portfolio = final_portfolio[~final_portfolio.index.duplicated(keep='first')].to_frame('strat_return')

macro_cum_returns = (1 + final_portfolio['strat_return']).cumprod()
macro_cum_returns.plot()

macro_cum_returns.loc['2015-01-01':'2025-01-01'].plot()


def perf_stats(series, benchmark=None, threshold=0.0):
    """
    series: equity curve / price series of strategy
    benchmark: equity curve / price series of benchmark (optional)
    threshold: minimum acceptable daily return for Omega / Sortino
    """

    # Clean price series
    series = series.replace([np.inf, -np.inf], np.nan).dropna()

    if len(series) < 2:
        raise ValueError("Series must contain at least 2 observations.")

    rets = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    if benchmark is not None:
        benchmark = benchmark.replace([np.inf, -np.inf], np.nan).dropna()
        bench = benchmark.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        rets, bench = rets.align(bench, join='inner')

        mask = (~rets.isna()) & (~bench.isna()) & np.isfinite(rets) & np.isfinite(bench)
        rets = rets[mask]
        bench = bench[mask]
    else:
        bench = None

    if len(rets) == 0:
        raise ValueError("No valid returns after cleaning and alignment.")

    # Performance metrics
    n_years = len(rets) / 252
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / n_years) - 1

    drawdown = series / series.cummax() - 1
    mdd = drawdown.min()

    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan

    downside = np.minimum(rets - threshold, 0)
    downside_std = downside.std()
    sortino = ((rets.mean() - threshold) / downside_std) * np.sqrt(252) if downside_std > 0 else np.nan

    gains = (rets - threshold).clip(lower=0).sum()
    losses = (threshold - rets).clip(lower=0).sum()
    omega = gains / losses if losses > 0 else np.nan

    cagr2dd = cagr / abs(mdd) if mdd != 0 else np.nan

    skewness = skew(rets)
    kurt = kurtosis(rets, fisher=True)

    if bench is not None and len(bench) > 0:
        X = bench.values.reshape(-1, 1)
        y = rets.values.reshape(-1, 1)

        reg = LinearRegression().fit(X, y)
        beta = float(reg.coef_.item())
        alpha = float(reg.intercept_.item()) * 252
        tracking_error = (rets - bench).std() * np.sqrt(252)
        r2 = reg.score(X, y)

        active_return = (rets.mean() - bench.mean()) * 252
        info_ratio = active_return / tracking_error if tracking_error > 0 else np.nan
    else:
        alpha = beta = tracking_error = info_ratio = r2 = np.nan

    return pd.DataFrame({
        "CAGR (%)": [round(cagr * 100, 2)],
        "Max Drawdown (%)": [round(mdd * 100, 2)],
        "Sharpe": [round(sharpe, 2)],
        "Sortino": [round(sortino, 2)],
        "Omega": [round(omega, 2)],
        "CAGR/Drawdown": [round(cagr2dd, 2)],
        "Skewness": [round(skewness, 2)],
        "Kurtosis": [round(kurt, 2)],
        "Alpha (ann%)": [round(alpha * 100, 2)],
        "Beta": [round(beta, 4) if pd.notna(beta) else np.nan],
        "Tracking Error (%)": [round(tracking_error * 100, 2)],
        "Information Ratio": [round(info_ratio, 2)],
        "R2": [round(r2, 4) if pd.notna(r2) else np.nan],
    })


strategy_curve = macro_cum_returns
benchmark_curve = (1 + spx.loc[strategy_curve.index[0].strftime('%Y-%m-%d'):]).cumprod()

print(perf_stats(strategy_curve, benchmark_curve))

macro_cum_returns.loc['2006-01-01':'2010-01-01'].plot(title='GFC 2008')
benchmark_curve.loc['2006-01-01':'2010-01-01'].plot()

strategy_curve.plot(title='Historical Performance')

macro_cum_returns.loc['2019-01-01':'2022-01-01'].plot(title='COVID')
benchmark_curve.loc['2019-01-01':'2022-01-01'].plot()

macro_cum_returns.loc['2019-01-01':'2026-01-01'].plot(title='Pre-Covid onwards')
benchmark_curve.loc['2019-01-01':'2026-01-01'].plot()

qs.reports.full(final_portfolio['strat_return'], benchmark="^GSPC", mode='full')

# (qs.reports.
#  html(final_portfolio['strat_return'],
#       benchmark="^GSPC", mode='full', 
#       title='Macroeconomic Regime Long/Short Asset Allocation', 
#       output="C:/Users/User/Desktop/Data Projects/Portfolio Optimisation/macro_asset_alloc_2.html"))