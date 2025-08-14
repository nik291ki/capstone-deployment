"""
Universal Grocery Clusterer 
----------------------------------------
Clusters customers from a transactions dataframe,
optionally auto-selects K via silhouette, and trains a small decision tree to
explain the clusters. Returns pandas DataFrames and plain dicts.

Input expectations (configurable via __init__):
- user_col: customer identifier
- item_col: product/category/department column used to build user×item matrix
- qty_col: numeric quantity/count column (defaults to 1 if missing)
- order_col/date_col/price_col: optional, not required for clustering core

Public API:
- fit(df, *, min_orders=1, auto_k=True, k=None, tree_max_depth=4) -> self
- get_features() -> DataFrame (user features + cluster)
- get_user_cluster_map() -> Dict[str, int]
- get_tree_rules() -> str (exported text of decision tree)
- get_cluster_sizes() -> DataFrame (cluster, size, pct)

Helper:
- build_user_item_matrix(df) -> DataFrame

Dependencies: pandas, numpy, scikit-learn
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, export_text


@dataclass
class ClustererConfig:
    user_col: str
    item_col: str
    qty_col: Optional[str] = None
    order_col: Optional[str] = None
    date_col: Optional[str] = None
    price_col: Optional[str] = None


class UniversalGroceryClusterer:
    def __init__(self, *, user_col: str, item_col: str, qty_col: Optional[str] = None,
                 order_col: Optional[str] = None, date_col: Optional[str] = None,
                 price_col: Optional[str] = None) -> None:
        self.cfg = ClustererConfig(user_col, item_col, qty_col, order_col, date_col, price_col)
        self._features: Optional[pd.DataFrame] = None
        self._labels: Optional[np.ndarray] = None
        self._kmeans: Optional[KMeans] = None
        self._tree_text: Optional[str] = None

    # ---------------------------------------------------------------------
    # Core build: user×item matrix
    # ---------------------------------------------------------------------
    def build_user_item_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        u, i = self.cfg.user_col, self.cfg.item_col
        q = self.cfg.qty_col
        work = df[[u, i]].copy()
        if q and q in df.columns:
            work[q] = pd.to_numeric(df[q], errors="coerce").fillna(0)
            mat = work.groupby([u, i])[q].sum().unstack(fill_value=0)
        else:
            # treat presence as 1
            work["__qty__"] = 1
            mat = work.groupby([u, i])["__qty__"].sum().unstack(fill_value=0)
        return mat.astype(float)

    # ---------------------------------------------------------------------
    # Fit: choose K, run KMeans, compute simple features, train tree
    # ---------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, *, min_orders: int = 1, auto_k: bool = True,
            k: Optional[int] = None, tree_max_depth: int = 4, random_state: int = 42) -> "UniversalGroceryClusterer":
        u, i = self.cfg.user_col, self.cfg.item_col
        # Basic filtering by min_orders (if order_col present)
        if self.cfg.order_col and self.cfg.order_col in df.columns:
            order_counts = df.groupby(u)[self.cfg.order_col].nunique()
            keep_users = order_counts[order_counts >= int(min_orders)].index
            df = df[df[u].isin(keep_users)].copy()
        # Build matrix
        mat = self.build_user_item_matrix(df)
        if mat.shape[0] < 2:
            raise ValueError("Not enough users after filtering to cluster.")

        # Auto-select K by silhouette on a sample
        if auto_k:
            sample = mat
            if mat.shape[0] > 4000:
                sample = mat.sample(4000, random_state=random_state)
            best_k, best_score = 0, -1.0
            for k_try in range(3, 9):
                km = KMeans(n_clusters=k_try, n_init=10, random_state=random_state)
                labels_try = km.fit_predict(sample)
                # silhouette requires > n_clusters samples
                if k_try < sample.shape[0]:
                    score = silhouette_score(sample, labels_try)
                    if score > best_score:
                        best_k, best_score = k_try, score
            final_k = best_k if best_k else 4
        else:
            if not k:
                raise ValueError("k must be provided when auto_k=False")
            final_k = int(k)

        km_final = KMeans(n_clusters=final_k, n_init=10, random_state=random_state)
        labels = km_final.fit_predict(mat)
        self._kmeans = km_final
        self._labels = labels

        # Build human-friendly features (can be extended)
        feats = pd.DataFrame(index=mat.index)
        feats["total_items"] = mat.sum(axis=1)
        feats["distinct_items"] = (mat > 0).sum(axis=1)
        if self.cfg.order_col and self.cfg.order_col in df.columns:
            feats["total_orders"] = df.groupby(u)[self.cfg.order_col].nunique().reindex(mat.index).fillna(0).astype(int)
        else:
            feats["total_orders"] = np.nan
        feats["cluster"] = labels
        feats[u] = feats.index
        self._features = feats.reset_index(drop=True)

        # Train a shallow decision tree to explain clusters
        try:
            X = feats.drop(columns=["cluster", u]).fillna(0)
            y = labels
            tree = DecisionTreeClassifier(max_depth=tree_max_depth, random_state=random_state)
            tree.fit(X, y)
            self._tree_text = export_text(tree, feature_names=list(X.columns))
        except Exception:
            self._tree_text = None

        return self

    # ---------------------------------------------------------------------
    # Accessors
    # ---------------------------------------------------------------------
    def get_features(self) -> pd.DataFrame:
        if self._features is None:
            raise RuntimeError("Call fit() first")
        return self._features.copy()

    def get_user_cluster_map(self) -> Dict[str, int]:
        feats = self.get_features()
        u = self.cfg.user_col
        return dict(zip(feats[u].astype(str), feats["cluster"].astype(int)))

    def get_tree_rules(self) -> Optional[str]:
        return self._tree_text

    def get_cluster_sizes(self) -> pd.DataFrame:
        feats = self.get_features()
        out = feats.groupby("cluster").size().rename("size").reset_index()
        out["pct"] = out["size"]/out["size"].sum()*100
        return out.sort_values("cluster").reset_index(drop=True)


# Convenience wrapper for one-shot run

def run_universal_pipeline(df: pd.DataFrame, *, user_col: str, item_col: str,
                           qty_col: Optional[str] = None, order_col: Optional[str] = None,
                           date_col: Optional[str] = None, price_col: Optional[str] = None,
                           min_orders: int = 1, auto_optimize: bool = True,
                           n_clusters: Optional[int] = None, create_decision_tree: bool = True,
                           tree_max_depth: int = 4) -> Tuple[UniversalGroceryClusterer, Dict]:
    """Backwards-compatible helper mirroring your previous app API.
    Returns (clusterer, descriptions_dict).
    """
    clu = UniversalGroceryClusterer(
        user_col=user_col, item_col=item_col, qty_col=qty_col,
        order_col=order_col, date_col=date_col, price_col=price_col
    )
    clu.fit(df, min_orders=min_orders, auto_k=auto_optimize, k=n_clusters,
            tree_max_depth=tree_max_depth)

    # Minimal description dictionary per cluster
    sizes = clu.get_cluster_sizes()
    desc: Dict[int, Dict] = {}
    for _, row in sizes.iterrows():
        cid = int(row["cluster"])  # type: ignore
        desc[cid] = {
            "name": f"Segment {cid}",
            "size": int(row["size"]),
            "percentage": float(row["pct"]),
            "rule": (clu.get_tree_rules() or "(tree not available)").splitlines()[0] if create_decision_tree else "",
            "description": "Auto-generated segment.",
            "marketing": "Consider targeted promotions."
        }
    return clu, desc
