import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class StreamlitHybridSegmentation:
    """
    Streamlit-optimized hybrid segmentation pipeline.
    Designed for efficient caching and interactive use.
    """
    
    def __init__(self, user_col='user_id', category_col='category', 
                 item_count_col='quantity', order_col='order_id'):
        self.user_col = user_col
        self.category_col = category_col
        self.item_count_col = item_count_col
        self.order_col = order_col
        
        # Results storage
        self.features = None
        self.features_scaled = None
        self.features_pca = None
        self.pca_model = None
        self.scaler = None
        self.segments = None
        self.segment_method = None
        self.interpretation_models = {}
        self.clustering_decision = None
        
    @st.cache_data
    def create_features(_self, data, min_orders=5):
        """Create features with Streamlit caching."""
        
        with st.spinner("🔬 Creating advanced behavioral features..."):
            # Validate columns
            required_cols = [_self.user_col, _self.category_col, _self.item_count_col]
            missing = [c for c in required_cols if c not in data.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
                return None
                
            st.write(f"📊 Dataset: {data.shape[0]:,} rows, {data[_self.user_col].nunique():,} users")
            
            # Filter users with minimum orders
            if _self.order_col and _self.order_col in data.columns:
                user_order_counts = data.groupby(_self.user_col)[_self.order_col].nunique()
            else:
                user_order_counts = data.groupby(_self.user_col).size()
                
            valid_users = user_order_counts[user_order_counts >= min_orders].index
            data_filtered = data[data[_self.user_col].isin(valid_users)]
            
            st.write(f"✅ Filtered to {len(valid_users):,} users with {min_orders}+ orders")
            
            # Create features DataFrame
            features = pd.DataFrame(index=valid_users)
            
            # Progress bar for feature creation
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. Volume metrics (20%)
            status_text.text("Creating volume metrics...")
            features['total_items'] = data_filtered.groupby(_self.user_col)[_self.item_count_col].sum()
            features['total_orders'] = user_order_counts[valid_users]
            features['avg_basket_size'] = features['total_items'] / features['total_orders']
            progress_bar.progress(0.2)
            
            # 2. Category behavior (40%)
            status_text.text("Analyzing category behavior...")
            category_counts = data_filtered.groupby([_self.user_col, _self.category_col])[_self.item_count_col].sum().reset_index()
            features['category_diversity'] = category_counts.groupby(_self.user_col)[_self.category_col].count()
            features['category_diversity_norm'] = features['category_diversity'] / data[_self.category_col].nunique()
            progress_bar.progress(0.4)
            
            # 3. Category concentration (60%)
            status_text.text("Computing concentration metrics...")
            def herfindahl_index(group):
                shares = group[_self.item_count_col] / group[_self.item_count_col].sum()
                return (shares ** 2).sum()
                
            features['category_concentration'] = category_counts.groupby(_self.user_col).apply(herfindahl_index)
            
            max_cat = category_counts.groupby(_self.user_col)[_self.item_count_col].max()
            features['top_category_pct'] = max_cat / features['total_items']
            progress_bar.progress(0.6)
            
            # 4. Entropy and advanced metrics (80%)
            status_text.text("Computing diversity metrics...")
            def shannon_entropy(group):
                shares = group[_self.item_count_col] / group[_self.item_count_col].sum()
                return -np.sum(shares * np.log2(shares + 1e-8))
                
            features['category_entropy'] = category_counts.groupby(_self.user_col).apply(shannon_entropy)
            
            # Consistency metrics
            features['shopping_intensity'] = features['avg_basket_size'].rank(pct=True)
            features['specialization_score'] = 1 - features['category_diversity_norm']
            progress_bar.progress(0.8)
            
            # 5. Clean and finalize (100%)
            status_text.text("Finalizing features...")
            features = features.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Remove constant features
            feature_variance = features.var()
            non_constant_features = feature_variance[feature_variance > 1e-6].index
            features = features[non_constant_features]
            
            progress_bar.progress(1.0)
            status_text.text("✅ Feature engineering complete!")
            
            _self.features = features
            return features
    
    def apply_preprocessing(self, transformation='yeo-johnson', scaling='robust', variance_threshold=0.90):
        """Apply preprocessing with progress indicators."""
        
        with st.spinner("🔄 Applying transformations and PCA..."):
            if self.features is None:
                st.error("Create features first!")
                return None
                
            X = self.features.copy()
            
            # Transformations
            if transformation == 'yeo-johnson':
                transformer = PowerTransformer(method='yeo-johnson', standardize=False)
                X_transformed = pd.DataFrame(
                    transformer.fit_transform(X),
                    index=X.index,
                    columns=X.columns
                )
            else:
                X_transformed = X
                
            # Scaling
            scaler = RobustScaler() if scaling == 'robust' else StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_transformed),
                index=X_transformed.index,
                columns=X_transformed.columns
            )
            
            # PCA
            pca_full = PCA()
            pca_full.fit(X_scaled)
            
            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = max(4, np.argmax(cumsum_var >= variance_threshold) + 1)
            
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            self.features_scaled = X_scaled
            self.scaler = scaler
            self.features_pca = pd.DataFrame(
                X_pca,
                index=X_scaled.index,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            self.pca_model = pca
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Features", X.shape[1])
            with col2:
                st.metric("PCA Components", n_components)
            with col3:
                st.metric("Variance Explained", f"{pca.explained_variance_ratio_.sum():.1%}")
                
            return self.features_pca
    
    def evaluate_clustering_viability(self, k_range=(2, 8), min_viable_clusters=3):
        """Evaluate clustering with interactive progress."""
        
        with st.spinner("🔍 Evaluating clustering algorithms..."):
            X = self.features_pca.values
            
            results = []
            algorithms = ['kmeans', 'gmm']
            
            # Progress tracking
            total_combinations = len(algorithms) * (k_range[1] - k_range[0] + 1)
            progress_bar = st.progress(0)
            current_step = 0
            
            for algorithm in algorithms:
                for k in range(k_range[0], k_range[1] + 1):
                    try:
                        if algorithm == 'kmeans':
                            model = KMeans(n_clusters=k, random_state=42, n_init=20)
                            labels = model.fit_predict(X)
                        else:
                            model = GaussianMixture(n_components=k, random_state=42)
                            labels = model.fit_predict(X)
                        
                        # Calculate metrics
                        if len(set(labels)) > 1:
                            silhouette = silhouette_score(X, labels)
                            calinski = calinski_harabasz_score(X, labels)
                            
                            # Check cluster balance
                            cluster_sizes = pd.Series(labels).value_counts()
                            min_cluster_size = cluster_sizes.min()
                            size_balance = min_cluster_size / len(labels)
                            
                            results.append({
                                'k': k,
                                'algorithm': algorithm,
                                'silhouette': silhouette,
                                'calinski': calinski,
                                'size_balance': size_balance,
                                'viable': size_balance >= 0.05 and silhouette > 0.2 and k >= min_viable_clusters
                            })
                    except Exception as e:
                        st.warning(f"Error with {algorithm} k={k}: {e}")
                    
                    current_step += 1
                    progress_bar.progress(current_step / total_combinations)
            
            # Analyze results
            scores_df = pd.DataFrame(results)
            viable_solutions = scores_df[scores_df['viable'] == True]
            
            if len(viable_solutions) > 0:
                best = viable_solutions.loc[viable_solutions['silhouette'].idxmax()]
                
                self.clustering_decision = {
                    'viable': True,
                    'best_algorithm': best['algorithm'],
                    'best_k': int(best['k']),
                    'best_silhouette': best['silhouette'],
                    'all_scores': scores_df
                }
                
                st.success(f"✅ Viable clustering found: {best['algorithm']} with k={best['k']}")
                return True
            else:
                self.clustering_decision = {
                    'viable': False,
                    'reason': 'No balanced clusters with good separation',
                    'all_scores': scores_df
                }
                
                st.warning("⚠️ No viable clustering solutions - will use continuum approach")
                return False
    
    def apply_segmentation(self, continuum_strategy='adaptive'):
        """Apply either clustering or continuum segmentation."""
        
        if self.clustering_decision['viable']:
            return self._apply_clustering()
        else:
            return self._apply_continuum_segmentation(continuum_strategy)
    
    def _apply_clustering(self):
        """Apply traditional clustering."""
        
        with st.spinner("🎯 Applying traditional clustering..."):
            X = self.features_pca.values
            algorithm = self.clustering_decision['best_algorithm']
            k = self.clustering_decision['best_k']
            
            if algorithm == 'kmeans':
                model = KMeans(n_clusters=k, random_state=42, n_init=50)
            else:
                model = GaussianMixture(n_components=k, random_state=42)
                
            labels = model.fit_predict(X)
            segments = [f"Cluster_{i}" for i in labels]
            
            self.segments = segments
            self.segment_method = 'clustering'
            self.final_model = model
            self.final_labels = labels
            
            # Add to features
            self.features['segment'] = segments
            self.features_pca['segment'] = segments
            
            # Display results
            st.success(f"✅ Created {k} clusters using {algorithm}")
            
            cluster_counts = pd.Series(segments).value_counts().sort_index()
            
            # Show cluster distribution
            fig = px.pie(
                values=cluster_counts.values,
                names=cluster_counts.index,
                title="Cluster Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            return segments
    
    def _apply_continuum_segmentation(self, strategy='adaptive'):
        """Apply continuum-based segmentation."""
        
        with st.spinner("🌈 Creating continuum-based segments..."):
            X_pca = self.features_pca.values
            n_components = X_pca.shape[1]
            
            segments = []
            
            if strategy == 'adaptive':
                pc1_values = X_pca[:, 0]
                pc1_high = np.percentile(pc1_values, 75)
                pc1_low = np.percentile(pc1_values, 25)
                
                if n_components >= 2:
                    pc2_values = X_pca[:, 1]
                    pc2_median = np.median(pc2_values)
                    
                    for i in range(len(X_pca)):
                        pc1, pc2 = X_pca[i, 0], X_pca[i, 1]
                        
                        if pc1 > pc1_high:
                            segments.append("High-Engagement-Bulk" if pc2 > pc2_median else "High-Engagement-Frequent")
                        elif pc1 < pc1_low:
                            segments.append("Low-Engagement-Bulk" if pc2 > pc2_median else "Low-Engagement-Frequent")
                        else:
                            segments.append("Medium-Engagement-Bulk" if pc2 > pc2_median else "Medium-Engagement-Frequent")
                else:
                    for pc1 in pc1_values:
                        if pc1 > pc1_high:
                            segments.append("High-Engagement")
                        elif pc1 < pc1_low:
                            segments.append("Low-Engagement")
                        else:
                            segments.append("Medium-Engagement")
            
            elif strategy == 'behavioral':
                # Behavioral interpretation
                pc1_values = X_pca[:, 0]
                pc1_high = np.percentile(pc1_values, 70)
                pc1_low = np.percentile(pc1_values, 30)
                
                if n_components >= 2:
                    pc2_values = X_pca[:, 1]
                    pc2_high = np.percentile(pc2_values, 70)
                    pc2_low = np.percentile(pc2_values, 30)
                    
                    for i in range(len(X_pca)):
                        pc1, pc2 = X_pca[i, 0], X_pca[i, 1]
                        
                        if pc1 > pc1_high and pc2 > pc2_high:
                            segments.append("Power-Users")
                        elif pc1 > pc1_high and pc2 < pc2_low:
                            segments.append("Focused-Buyers")
                        elif pc1 < pc1_low and pc2 > pc2_high:
                            segments.append("Occasional-Bulk")
                        elif pc1 < pc1_low and pc2 < pc2_low:
                            segments.append("Light-Users")
                        else:
                            segments.append("Balanced-Users")
            
            self.segments = segments
            self.segment_method = 'continuum'
            
            # Add to features
            self.features['segment'] = segments
            self.features_pca['segment'] = segments
            
            # Display results
            st.success(f"✅ Created continuum segments using {strategy} strategy")
            
            segment_counts = pd.Series(segments).value_counts()
            
            # Show segment distribution
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Segment Distribution (Continuum-Based)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            return segments
    
    def train_interpretation_models(self):
        """Train interpretation models with progress tracking."""
        
        with st.spinner("🧠 Training interpretation models..."):
            X = self.features_scaled.drop(['segment'], axis=1, errors='ignore')
            y = pd.Series(self.segments)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Train Decision Tree
            status_text.text("Training Decision Tree...")
            dt = DecisionTreeClassifier(
                max_depth=6,
                min_samples_split=30,
                min_samples_leaf=15,
                random_state=42,
                class_weight='balanced'
            )
            dt.fit(X_train, y_train)
            
            dt_cv_scores = cross_val_score(dt, X, y, cv=5)
            progress_bar.progress(0.5)
            
            # Train Random Forest
            status_text.text("Training Random Forest...")
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=20,
                random_state=42,
                class_weight='balanced'
            )
            rf.fit(X_train, y_train)
            
            rf_cv_scores = cross_val_score(rf, X, y, cv=5)
            progress_bar.progress(1.0)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'dt_importance': dt.feature_importances_,
                'rf_importance': rf.feature_importances_
            }).sort_values('rf_importance', ascending=False)
            
            # Store results
            self.interpretation_models = {
                'decision_tree': {
                    'model': dt,
                    'cv_scores': dt_cv_scores
                },
                'random_forest': {
                    'model': rf,
                    'cv_scores': rf_cv_scores
                },
                'feature_importance': feature_importance,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
            status_text.text("✅ Model training complete!")
            
            # Display performance
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Decision Tree CV", f"{dt_cv_scores.mean():.3f}")
            with col2:
                st.metric("Random Forest CV", f"{rf_cv_scores.mean():.3f}")
            
            return feature_importance
    
    def create_interactive_visualizations(self):
        """Create interactive Plotly visualizations for Streamlit."""
        
        st.subheader("📊 Segment Analysis Visualizations")
        
        # 1. PCA Scatter Plot
        if self.features_pca.shape[1] >= 2:
            fig = px.scatter(
                self.features_pca,
                x='PC1',
                y='PC2',
                color='segment',
                title=f"Segments in PCA Space ({self.segment_method})",
                labels={
                    'PC1': f'PC1 ({self.pca_model.explained_variance_ratio_[0]:.1%} variance)',
                    'PC2': f'PC2 ({self.pca_model.explained_variance_ratio_[1]:.1%} variance)'
                }
            )
            fig.update_traces(marker=dict(size=5, opacity=0.7))
            st.plotly_chart(fig, use_container_width=True)
        
        # 2. Feature Importance
        if 'feature_importance' in self.interpretation_models:
            importance_df = self.interpretation_models['feature_importance'].head(10)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=importance_df['rf_importance'],
                y=importance_df['feature'],
                orientation='h',
                name='Random Forest',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Top 10 Feature Importance (Random Forest)",
                xaxis_title="Importance",
                yaxis_title="Features"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 3. Segment Characteristics Heatmap
        numeric_features = self.features.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != 'segment'][:8]  # Top 8
        
        segment_profiles = self.features.groupby('segment')[numeric_features].mean()
        
        fig = px.imshow(
            segment_profiles.T,
            aspect='auto',
            title="Segment Characteristics Heatmap",
            labels=dict(x="Segment", y="Feature", color="Value")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def get_segment_insights_streamlit(self):
        """Get segment insights formatted for Streamlit display."""
        
        st.subheader("🔍 Segment Insights & Recommendations")
        
        if not self.interpretation_models:
            st.error("Train interpretation models first!")
            return
            
        top_features = self.interpretation_models['feature_importance']['feature'].head(5).tolist()
        
        for segment in sorted(set(self.segments)):
            with st.expander(f"📦 {segment}", expanded=False):
                segment_data = self.features[self.features['segment'] == segment]
                segment_size = len(segment_data)
                segment_pct = (segment_size / len(self.features)) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Size", f"{segment_size:,}")
                with col2:
                    st.metric("Percentage", f"{segment_pct:.1f}%")
                
                st.write("**Key Characteristics:**")
                characteristics = []
                
                for feature in top_features:
                    seg_mean = segment_data[feature].mean()
                    overall_mean = self.features[feature].mean()
                    
                    if overall_mean != 0:
                        pct_diff = ((seg_mean - overall_mean) / abs(overall_mean)) * 100
                        
                        if abs(pct_diff) > 10:
                            direction = "higher" if pct_diff > 0 else "lower"
                            st.write(f"• **{feature}**: {pct_diff:+.0f}% {direction} than average")
                
                st.write("**Business Recommendations:**")
                
                # Auto-generate recommendations based on segment characteristics
                if 'high' in segment.lower() or 'power' in segment.lower():
                    st.write("• Target for premium products and loyalty programs")
                    st.write("• Ideal for cross-selling and upselling campaigns")
                elif 'low' in segment.lower() or 'light' in segment.lower():
                    st.write("• Focus on activation and engagement campaigns")
                    st.write("• Offer introductory promotions to increase usage")
                elif 'bulk' in segment.lower():
                    st.write("• Emphasize bulk discounts and large pack sizes")
                    st.write("• Target with storage and convenience messaging")
                elif 'frequent' in segment.lower():
                    st.write("• Focus on convenience and quick shopping")
                    st.write("• Promote subscription or delivery services")
                else:
                    st.write("• Great candidates for A/B testing")
                    st.write("• Monitor for opportunities to increase engagement")
    
    def export_results_streamlit(self):
        """Export results with Streamlit download buttons."""
        
        st.subheader("💾 Export Results")
        
        # 1. Segment assignments
        segment_assignments = pd.DataFrame({
            'user_id': self.features.index,
            'segment': self.segments,
            'segment_method': self.segment_method
        })
        
        csv1 = segment_assignments.to_csv(index=False)
        st.download_button(
            label="📥 Download Segment Assignments",
            data=csv1,
            file_name="segment_assignments.csv",
            mime="text/csv"
        )
        
        # 2. Features with segments
        csv2 = self.features.to_csv()
        st.download_button(
            label="📥 Download Features with Segments",
            data=csv2,
            file_name="features_with_segments.csv",
            mime="text/csv"
        )
        
        # 3. Feature importance
        if 'feature_importance' in self.interpretation_models:
            csv3 = self.interpretation_models['feature_importance'].to_csv(index=False)
            st.download_button(
                label="📥 Download Feature Importance",
                data=csv3,
                file_name="feature_importance.csv",
                mime="text/csv"
            )
        
        # 4. Decision rules as text
        if 'decision_tree' in self.interpretation_models:
            dt = self.interpretation_models['decision_tree']['model']
            feature_names = [col for col in self.features.columns if col != 'segment']
            rules = export_text(dt, feature_names=feature_names[:len(dt.feature_importances_)])
            
            st.download_button(
                label="📥 Download Decision Rules",
                data=rules,
                file_name="decision_rules.txt",
                mime="text/plain"
            )
        
        st.success("✅ All results ready for download!")
        
        return segment_assignments

