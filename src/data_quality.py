"""
data_quality.py
Data quality gates for the biogas pipeline.

Handles:
  1. High-correlation sensor removal (e.g. 208 pairs in DataONE)
  2. Missing data pattern detection: MCAR vs MAR vs MNAR
  3. Automatic feature selection via mutual information + stability
  4. Sensor-group definitions for structured dropout training
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ─── 1. Correlation-Based Redundancy Removal ─────────────────────────────────

class CorrelationFilter:
    """
    Removes one feature from each highly correlated pair.
    Keeps the feature with higher variance (more information).

    Args:
      threshold: |r| above which features are considered redundant (default 0.85)
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold   = threshold
        self.kept_cols_  = None
        self.removed_    = None
        self.corr_pairs_ = None

    def fit(self, df: pd.DataFrame) -> "CorrelationFilter":
        numeric = df.select_dtypes(include=np.number)
        corr    = numeric.corr().abs()

        # Upper triangle only
        upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs   = [(col, row, upper.loc[row, col])
                   for col in upper.columns
                   for row in upper.index
                   if upper.loc[row, col] > self.threshold]

        to_remove = set()
        for c1, c2, r in pairs:
            if c1 in to_remove or c2 in to_remove:
                continue
            # Keep higher-variance feature
            v1, v2 = numeric[c1].var(), numeric[c2].var()
            to_remove.add(c2 if v1 >= v2 else c1)

        self.corr_pairs_ = pairs
        self.removed_    = list(to_remove)
        self.kept_cols_  = [c for c in df.columns if c not in to_remove]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.kept_cols_ is None:
            raise RuntimeError("Call fit() first")
        return df[[c for c in self.kept_cols_ if c in df.columns]]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def report(self) -> str:
        lines = [
            f"CorrelationFilter (threshold={self.threshold})",
            f"  Pairs found:  {len(self.corr_pairs_)}",
            f"  Removed:      {len(self.removed_)} features",
            f"  Kept:         {len(self.kept_cols_)} features",
            "",
            "  Removed features:",
        ]
        for col in sorted(self.removed_):
            lines.append(f"    - {col}")
        return "\n".join(lines)


# ─── 2. Missing Pattern Detector ─────────────────────────────────────────────

class MissingPatternDetector:
    """
    Tests for MCAR (Missing Completely At Random) vs MAR/MNAR.

    Method:
      - Little's MCAR test (chi-squared on missingness patterns)
      - Pairwise correlation between missingness indicators
      - Recommends appropriate imputation strategy
    """

    STRATEGIES = {
        "MCAR": "Simple imputation (mean/median/forward-fill) is unbiased.",
        "MAR":  "Multiple imputation or MICE. Consider model-based imputation.",
        "MNAR": "Selection model or pattern-mixture model needed. "
                "Flag sensor for maintenance check.",
    }

    def __init__(self):
        self.results_ = {}

    def fit(self, df: pd.DataFrame) -> "MissingPatternDetector":
        numeric = df.select_dtypes(include=np.number)
        miss    = numeric.isnull()

        col_results = {}
        for col in miss.columns:
            if miss[col].sum() == 0:
                continue

            miss_indicator = miss[col].astype(int)
            pattern        = self._classify(numeric, col, miss_indicator)
            col_results[col] = pattern

        self.results_ = col_results
        return self

    def _classify(self, df, target_col, miss_indicator) -> dict:
        """
        Heuristic classification based on correlation between
        missingness of target_col and values of other columns.
        """
        other_cols = [c for c in df.columns if c != target_col]
        correlations = {}
        for c in other_cols:
            valid = df[c].notna() & miss_indicator.notna()
            if valid.sum() < 10:
                continue
            r, p = stats.pointbiserialr(
                miss_indicator[valid], df.loc[valid, c].fillna(df[c].median())
            )
            correlations[c] = (r, p)

        # If any correlated columns → MAR; else MCAR
        significant = [(c, r, p) for c, (r, p) in correlations.items() if p < 0.05]

        if len(significant) == 0:
            mechanism = "MCAR"
        elif any(abs(r) > 0.4 for _, r, _ in significant):
            mechanism = "MNAR"
        else:
            mechanism = "MAR"

        return {
            "mechanism":   mechanism,
            "strategy":    self.STRATEGIES[mechanism],
            "n_missing":   int(miss_indicator.sum()),
            "pct_missing": float(miss_indicator.mean() * 100),
            "correlated_with": [c for c, _, _ in significant[:5]],
        }

    def report(self) -> str:
        if not self.results_:
            return "No missing values detected."
        lines = ["Missing Pattern Analysis:", ""]
        for col, r in self.results_.items():
            lines += [
                f"  {col}:",
                f"    Mechanism : {r['mechanism']}",
                f"    Missing   : {r['pct_missing']:.1f}%",
                f"    Strategy  : {r['strategy']}",
            ]
            if r["correlated_with"]:
                lines.append(f"    Corr. with: {', '.join(r['correlated_with'])}")
            lines.append("")
        return "\n".join(lines)


# ─── 3. Mutual Information Feature Selector ───────────────────────────────────

class MIFeatureSelector:
    """
    Ranks features by mutual information with the target,
    then applies a stability selection approach over bootstrap samples.

    Args:
      n_select:     number of top features to keep
      n_bootstrap:  bootstrap iterations for stability
      threshold:    minimum selection frequency to keep feature (0–1)
    """

    def __init__(self,
                 n_select:    int   = 15,
                 n_bootstrap: int   = 20,
                 threshold:   float = 0.5):
        self.n_select    = n_select
        self.n_bootstrap = n_bootstrap
        self.threshold   = threshold
        self.selected_   = None
        self.scores_     = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MIFeatureSelector":
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        X_num = X[numeric_cols].fillna(X[numeric_cols].median())
        y_arr = y.fillna(y.median()).values

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X_num)

        # Bootstrap stability
        counts = np.zeros(len(numeric_cols))
        rng    = np.random.RandomState(42)

        for _ in range(self.n_bootstrap):
            idx   = rng.choice(len(X_s), size=int(0.8 * len(X_s)), replace=False)
            mi    = mutual_info_regression(X_s[idx], y_arr[idx], random_state=42)
            top_k = np.argsort(mi)[-self.n_select:]
            counts[top_k] += 1

        freq = counts / self.n_bootstrap
        self.scores_   = dict(zip(numeric_cols, freq))
        stable         = [c for c, f in self.scores_.items() if f >= self.threshold]

        # If not enough stable features, fill with top-scoring
        if len(stable) < self.n_select:
            sorted_cols = sorted(self.scores_, key=self.scores_.get, reverse=True)
            stable      = sorted_cols[:self.n_select]

        self.selected_ = stable[:self.n_select]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.selected_ if c in X.columns]
        return X[available]

    def report(self) -> str:
        if self.scores_ is None:
            return "Not fitted yet."
        lines = [f"MIFeatureSelector: selected {len(self.selected_)} features", ""]
        sorted_features = sorted(self.scores_.items(), key=lambda x: -x[1])
        for col, score in sorted_features[:20]:
            mark = "✓" if col in self.selected_ else " "
            lines.append(f"  {mark} {col:40s}  stability={score:.2f}")
        return "\n".join(lines)


# ─── 4. Sensor Group Definitions (for structured dropout) ────────────────────

SENSOR_GROUPS = {
    "pH_alkalinity": ["pH", "pH_lab", "Alkalinity", "alkalinity"],
    "temperature":   ["Temperature", "Temp", "temperature", "temp"],
    "flow":          ["FlowRate", "Flow", "flow", "HRT", "SRT"],
    "organic":       ["COD", "COD_in", "COD_out", "VFA", "BOD", "substrate"],
    "gas":           ["GasFlow", "BiogasFlow", "MethaneFlow", "biogas", "methane"],
    "solids":        ["TotalSolids", "VolatileSolids", "TSS", "VSS", "MLSS"],
    "nutrients":     ["Ammonia", "TKN", "Phosphorus", "Nitrogen"],
}


def assign_sensor_groups(feature_cols: list) -> dict:
    """
    Maps each feature to a sensor group.
    Returns: {group_name: [indices_in_feature_cols]}
    """
    col_lower = [c.lower() for c in feature_cols]
    group_indices = {g: [] for g in SENSOR_GROUPS}

    for i, col in enumerate(col_lower):
        assigned = False
        for group, keywords in SENSOR_GROUPS.items():
            if any(kw.lower() in col for kw in keywords):
                group_indices[group].append(i)
                assigned = True
                break
        if not assigned:
            group_indices.setdefault("other", []).append(i)

    # Remove empty groups
    return {k: v for k, v in group_indices.items() if v}


# ─── Pipeline Runner ──────────────────────────────────────────────────────────

def run_data_quality_pipeline(df: pd.DataFrame,
                               target_col: str,
                               corr_thresh: float = 0.85,
                               n_select:    int   = 20) -> tuple:
    """
    Full pipeline:
      1. Correlation filter
      2. Missing pattern analysis
      3. MI feature selection

    Returns:
      cleaned_df      – DataFrame with selected features + target
      selected_cols   – list of selected feature column names
      reports         – dict of text reports
    """
    print("[DataQuality] Running pipeline …")

    # Step 1: Correlation filter
    cf = CorrelationFilter(threshold=corr_thresh)
    df_filtered = cf.fit_transform(df)
    print(f"  [Corr] Removed {len(cf.removed_)} redundant features")

    # Step 2: Missing pattern
    mpd = MissingPatternDetector()
    mpd.fit(df_filtered)

    # Step 3: Feature selection
    feature_cols = [c for c in df_filtered.columns if c != target_col]
    X = df_filtered[feature_cols]
    y = df_filtered[target_col] if target_col in df_filtered.columns else None

    if y is not None and y.notna().sum() > 30:
        mi_sel = MIFeatureSelector(n_select=n_select)
        mi_sel.fit(X, y)
        final_features = mi_sel.selected_
        print(f"  [MI]   Selected {len(final_features)} features")
    else:
        final_features = [c for c in feature_cols if c in df_filtered.columns]
        mi_sel         = None
        print("  [MI]   Skipped (no target labels)")

    cols_to_keep = final_features + ([target_col] if target_col in df_filtered.columns else [])
    cleaned_df   = df_filtered[[c for c in cols_to_keep if c in df_filtered.columns]]

    reports = {
        "correlation": cf.report(),
        "missing":     mpd.report(),
        "feature_sel": mi_sel.report() if mi_sel else "Skipped",
    }

    return cleaned_df, final_features, reports
