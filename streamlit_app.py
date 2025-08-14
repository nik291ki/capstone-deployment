import os
import time
from datetime import datetime
from typing import Optional, Dict, List
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import streamlit as st

import requests
import io


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, export_text


# --- Page config ---
st.set_page_config(page_title="Supermarket AI – Simplified", layout="wide", initial_sidebar_state="expanded")

# --- Try to import your pipeline modules ---
ARP = None
try:
    from association_rules_pipeline import AssociationRulesPipeline as _ARP
    ARP = _ARP
except Exception:
    pass

UGC = None
try:
    from universal_grocery_clusterer import (
        UniversalGroceryClusterer as _UGC,
        run_universal_pipeline as _UGC_RUN,
    )
    UGC = {
        'cls': _UGC,
        'run': _UGC_RUN,
    }
except Exception:
    UGC = None

try:
    from hybrid_segmentation import StreamlitHybridSegmentation
    HYBRID_SEGMENTATION_AVAILABLE = True
    HYBRID_SEGMENTATION_ERR = None
except Exception as e:  # catch anything, not just ImportError
    HYBRID_SEGMENTATION_AVAILABLE = False
    HYBRID_SEGMENTATION_ERR = e

# --- Import data_upload (pandas-only) and provide fallbacks ---
DU_OK = False
DU_ERR = None
try:
    from data_upload import (
        DataUploadManager,
        handle_file_upload,
        handle_url_loading,
        handle_preloaded_data,
        handle_data_validation_and_preview,
    )
    DU_OK = True
except Exception as e:
    DU_OK = False
    DU_ERR = e


def _simple_uploader_fallback():
    """Minimal uploader that works even if data_upload.py failed to import."""
    st.warning("Using simple uploader fallback (data_upload module not available).")
    if DU_ERR:
        with st.expander("Import error details"):
            st.code(str(DU_ERR))
    up = st.file_uploader("Upload CSV/Excel/Parquet/JSON", type=["csv", "xlsx", "xls", "parquet", "json"])
    if up is not None:
        with st.spinner("Loading..."):
            if up.name.endswith((".parquet",)):
                df = pd.read_parquet(up)
            elif up.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(up)
            elif up.name.endswith((".json",)):
                try:
                    df = pd.read_json(up, lines=True)
                except ValueError:
                    up.seek(0)
                    df = pd.read_json(up)
            else:
                df = pd.read_csv(up, low_memory=False, on_bad_lines="skip")
        st.session_state.data = df
        st.session_state.file_info = {
            "name": up.name,
            "size_mb": getattr(up, "size", 0) / (1024 * 1024),
            "source": "upload",
        }
        st.session_state.data_uploaded = True
        st.success(f"Loaded {len(df):,} rows from {up.name}")
        st.dataframe(df.head(50), use_container_width=True)


# --- Session state ---
for k, v in {
    'data': None,
    'file_info': {},
    'clusters_df': None,
    'cluster_map': None,
    'cluster_user_col': None,
    'detector': None,  # This will store the EnhancedFruitDetector instance
    'fresh_model_loaded': False,
    'fd_log': [],
    'data_uploaded': False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v
        
# --- Constants for CV model ---
DEFAULT_CLASS_NAMES = [
    'good_apple', 'good_banana', 'good_bittergroud', 'good_carrot', 'good_cucumber',
    'good_grape', 'good_guava', 'good_jujube', 'good_mango', 'good_okra',
    'good_orange', 'good_pomegranate', 'good_potato', 'good_strawberry', 'good_sweetpepper',
    'good_tomato', 'stale_apple', 'stale_banana', 'stale_bittergroud', 'stale_carrot',
    'stale_cucumber', 'stale_grape', 'stale_guava', 'stale_jujube', 'stale_mango',
    'stale_okra', 'stale_orange', 'stale_pomegranate', 'stale_potato', 'stale_strawberry',
    'stale_sweetpepper', 'stale_tomato'
]
MODEL_PATH = "models/CNN_model_3.keras"

# --- Helpers ---

def human_size(num: int) -> str:
    for u in ['B','KB','MB','GB','TB']:
        if num < 1024:
            return f"{num:.1f} {u}"
        num /= 1024
    return f"{num:.1f} PB"


def require_data() -> Optional[pd.DataFrame]:
    if st.session_state.data is None:
        st.info("Please load data on Home → Data Loader or the Data Upload page.")
        return None
    return st.session_state.data


def big_preview_stats(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Columns", f"{df.shape[1]:,}")
    with c3:
        st.metric("Has cluster?", 'Yes' if 'cluster' in df.columns else 'No')
    with c4:
        st.metric("Last updated", datetime.now().strftime('%Y-%m-%d %H:%M'))
    st.markdown("---")

    with st.expander("Basic statistics (numeric cols)", expanded=True):
        num = df.select_dtypes(include=[np.number])
        if num.shape[1]:
            st.dataframe(num.describe().T, use_container_width=True)
        else:
            st.caption("No numeric columns detected.")

    cols = df.columns.tolist()
    colL, colR = st.columns(2)
    with colL:
        if 'department' in cols:
            st.subheader("Top departments")
            st.bar_chart(df['department'].value_counts().head(10))
        elif 'category' in cols:
            st.subheader("Top categories")
            st.bar_chart(df['category'].value_counts().head(10))
    with colR:
        if 'product_name' in cols:
            st.subheader("Top products")
            st.bar_chart(df['product_name'].value_counts().head(10))

# --- Sidebar ---
with st.sidebar:
    page = st.selectbox(
        "Navigate",
        [
            "Home",
            "Data Upload",
            "Customer Segmentation",
            "Association Rules",
            "Freshness Detection"
        ],
    )
    st.markdown("---")
    if st.session_state.data is not None:
        info = st.session_state.file_info or {}
        name = info.get('name', 'dataset')
        size_mb = info.get('size_mb')
        s = name if size_mb is None else f"{name} ({size_mb:.1f} MB)"
        st.success(f"✅ Data loaded: {s}")
    else:
        st.warning("No data loaded")

# --- Pages ---

def page_home():
    st.markdown("# Supermarket Analytics Dashboard")

    # Use your universal_data_loader UX if present
    can_load = False
    try:
        from universal_data_loader import UniversalDataLoader
        udl = UniversalDataLoader(page_context="dashboard")
        can_load = udl.show_data_loader(container_type="main")
    except Exception:
        pass

    if st.session_state.data is None and not st.session_state.get('data_uploaded', False):
        # Your requested welcome text
        st.markdown(
            """
            ### Welcome to your Retail Analytics Platform

            This platform combines the power of AI to give you valuable insights that can help optimize your supermarket's operations. 
            Here's what you can expect:
            """
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                **Customer Intelligence**
                - Advanced segmentation
                - Purchase pattern analysis
                - Targeted marketing insights
                - Product association mining
                """
            )
        with c2:
            st.markdown(
                """
                **Quality Control**
                - AI-powered freshness detection
                - Real-time monitoring
                - Automated quality alerts
                """
            )
        st.info("🚧 Load your data to see personalised dashboard statistics")
        return

    # If the Universal loader set data into session, capture it
    if can_load and st.session_state.get('uploaded_data') is not None:
        st.session_state.data = st.session_state.uploaded_data
        st.session_state.file_info = st.session_state.get('file_info', {})

    df = require_data()
    if df is None:
        return

    st.subheader("Dataset overview")
    big_preview_stats(df)

    st.markdown("### Data preview")
    n = st.slider("Rows to show", 5, min(2000, len(df)), min(10, len(df)))
    st.dataframe(df.head(n), use_container_width=True)

def page_upload():
    st.header("📥 Data Upload & Preview")
    
    source = st.radio("Choose data source", [
        "Upload Small CSV/Excel File (< 200MB)",
        "🚀 Load Online Retail Dataset (500K+ transactions)"
    ])

    data, info = None, None

    if source == "Upload Small CSV/Excel File (< 200MB)":
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading file..."):
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)
                    
                    info = {
                        'name': uploaded_file.name,
                        'size_mb': uploaded_file.size / (1024 * 1024),
                        'rows': len(data),
                        'columns': data.shape[1]
                    }
                    
                st.success(f"✅ File loaded: {len(data):,} rows")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    elif source == "🚀 Load Online Retail Dataset (500K+ transactions)":
        st.subheader("🌐 Online Retail Dataset")
        
        st.info("""
        **Online Retail Dataset (500k transactions)**
        - Description: UK retailer transactions 2010-2011
        - Size: ~18MB, 540k rows
        - Perfect for demonstrating retail analytics capabilities
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Load Dataset", type="primary"):
                try:
                    with st.spinner(f"🔄 Loading Online Retail Dataset..."):
                        
                        # Progress bar for visual effect
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("📡 Connecting to data source...")
                        progress_bar.progress(20)
                        
                        # Load the Online Retail dataset
                        dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
                        
                        status_text.text("📊 Downloading dataset...")
                        progress_bar.progress(40)
                        
                        data = pd.read_excel(dataset_url)
                        
                        # Clean column names for consistency
                        data.columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
                        
                        # Rename for consistency with your app
                        data = data.rename(columns={
                            'CustomerID': 'user_id',
                            'Description': 'product_name',
                            'StockCode': 'category',
                            'Quantity': 'quantity',
                            'InvoiceNo': 'order_id'
                        })
                        
                        # Clean the data
                        data = data.dropna(subset=['user_id'])  # Remove rows without customer ID
                        data = data[data['quantity'] > 0]  # Remove returns/cancellations
                        
                        progress_bar.progress(80)
                        status_text.text("🧮 Computing statistics...")
                        
                        # Calculate real statistics
                        size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
                        
                        info = {
                            'name': 'Online Retail Dataset',
                            'size_mb': size_mb,
                            'source': 'public_dataset',
                            'rows': len(data),
                            'columns': data.shape[1]
                        }
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Dataset loaded successfully!")
                        
                        # Clear progress
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success(f"✅ ONLINE RETAIL DATA LOADED: {len(data):,} rows!")
                        
                        # Show metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", f"{len(data):,}")
                        with col2:
                            st.metric("Size", f"{size_mb:.1f} MB")
                        with col3:
                            st.metric("Customers", f"{data['user_id'].nunique():,}")
                        
                        st.info("🎯 Real retail dataset ready for analytics - perfect for customer segmentation and market basket analysis!")
                        
                except Exception as e:
                    st.error(f"❌ Failed to load dataset: {str(e)}")
                    st.info("💡 Please check your internet connection or try again later")
        
        with col2:
            st.write("**Dataset Features:**")
            st.markdown("""
            - **540,000+** transaction records
            - **4,000+** unique customers
            - **4,000+** unique products
            - **37** countries
            - **Date range**: 2010-2011
            - Perfect for segmentation analysis
            """)

    # Handle data validation and preview
    if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
        st.session_state.data = data
        st.session_state.file_info = info or {}
        
        # Simple data preview and validation
        st.subheader("📊 Data Preview & Validation")
        
        # Show basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(data):,}")
        with col2:
            st.metric("Columns", f"{data.shape[1]}")
        with col3:
            if info and 'size_mb' in info:
                st.metric("Size", f"{info['size_mb']:.1f} MB")
        
        # Show data preview
        st.dataframe(data.head(10), use_container_width=True)
        
        # Show column info
        with st.expander("📋 Column Information"):
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Type': data.dtypes,
                'Non-Null': data.count(),
                'Null Count': data.isnull().sum()
            })
            st.dataframe(col_info, use_container_width=True)
            
        st.success("✅ Data ready for analysis on other pages!")
    
    elif data is None and source not in ["🚀 Load Online Retail Dataset (500K+ transactions)", "Upload Large Dataset (Advanced)"]:
        st.info("👆 Please select and load a data source to continue")
        

def page_segmentation():

    # Add this at the top of page_segmentation(), before the "Run Segmentation" button
    if st.button("🔄 Reset Segmentation", help="Clear previous results and start fresh"):
    # Clear all segmentation-related session state
        for key in ['clusters_df', 'cluster_map', 'cluster_user_col']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ Segmentation reset! You can now run a fresh analysis.")
        st.experimental_rerun()
        
    st.header("🧩 Advanced Customer Segmentation")
    st.write("Hybrid approach: Traditional clustering with automatic fallback to continuum segmentation")

    # Check if hybrid segmentation is available
    if not HYBRID_SEGMENTATION_AVAILABLE:
        st.error("❌ Hybrid segmentation module not available.")
        if 'HYBRID_SEGMENTATION_ERR' in globals() and HYBRID_SEGMENTATION_ERR:
            st.code(str(HYBRID_SEGMENTATION_ERR))
        st.stop()


    df = require_data()
    if df is None:
        st.markdown(
            """
            ### About Hybrid Customer Segmentation

            This advanced module uses a **hybrid approach** to customer segmentation:
            
            **🎯 Traditional Clustering First:**
            - Tests K-Means and Gaussian Mixture Models
            - Evaluates cluster quality and balance
            - Uses silhouette score and cluster size balance
            
            **🌈 Continuum Segmentation Fallback:**
            - When traditional clustering produces poor results (e.g., only 2 viable clusters)
            - Creates behavioral segments using PCA space
            - Produces interpretable business segments
            
            **🧠 Smart Interpretation:**
            - Decision trees explain segment characteristics
            - Feature importance analysis
            - Business-ready recommendations

            **Data requirements:**
            - Customer/User identifier
            - Product/Category column  
            - Purchase quantity/count
            - Optional: Order ID for transaction counting
            """
        )
        return

    cols = df.columns.tolist()

    # Column mapping
    st.subheader("📊 Data Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_col = st.selectbox(
            "Customer/User ID column", 
            options=cols, 
            index=cols.index('user_id') if 'user_id' in cols else 0
        )
        
        category_col = st.selectbox(
            "Product/Category column", 
            options=cols,
            index=cols.index('product_name') if 'product_name' in cols else (
                cols.index('category') if 'category' in cols else (
                    cols.index('department') if 'department' in cols else 0
                )
            )
        )
    
    with col2:
        quantity_col = st.selectbox(
            "Quantity/Count column", 
            options=['<count occurrences>'] + cols,
            index=0,
            help="Choose a quantity column or count occurrences"
        )
        
        order_col = st.selectbox(
            "Order ID column (optional)", 
            options=['<none>'] + cols,
            index=(cols.index('order_id') + 1) if 'order_id' in cols else 0
        )

    # Segmentation settings
    st.subheader("⚙️ Segmentation Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cluster_mode = st.radio(
            "Clustering approach:",
            ["Auto-detect optimal clusters", "Specify cluster count"],
            help="Auto-detect will find the best number of clusters, or fallback to continuum"
        )
    
    with col2:
        if cluster_mode == "Specify cluster count":
            target_clusters = st.slider("Target clusters", 2, 8, 4)
        else:
            target_clusters = None
            
        min_orders = st.selectbox("Minimum orders per customer", [2, 3, 5, 10], index=1)
    
    with col3:
        continuum_strategy = st.selectbox(
            "Continuum strategy (fallback)",
            ["adaptive", "behavioral"],
            help="Strategy to use if clustering fails"
        )
        
        variance_threshold = st.slider("PCA variance threshold", 0.8, 0.95, 0.90, 0.05)

    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        preprocessing_col1, preprocessing_col2 = st.columns(2)
        
        with preprocessing_col1:
            transformation = st.selectbox(
                "Data transformation",
                ["yeo-johnson", "none"],
                help="Yeo-Johnson handles skewed data better"
            )
        
        with preprocessing_col2:
            scaling_method = st.selectbox(
                "Scaling method",
                ["robust", "standard"],
                help="Robust scaling is less sensitive to outliers"
            )

    # Run segmentation button
    run_segmentation = st.button("🚀 Run Hybrid Segmentation", type="primary", use_container_width=True)

    if not run_segmentation:
        return

    # Prepare parameters
    quantity_param = None if quantity_col == '<count occurrences>' else quantity_col
    order_param = None if order_col == '<none>' else order_col

    try:
        # Initialize pipeline
        with st.spinner("🔧 Initializing segmentation pipeline..."):
            pipeline = StreamlitHybridSegmentation(
                user_col=user_col,
                category_col=category_col,
                item_count_col=quantity_param or 'count',  # Will handle counting internally
                order_col=order_param
            )

        # Step 1: Feature Engineering
        st.subheader("🔬 Step 1: Feature Engineering")
        
        # Handle quantity counting if needed
        if quantity_param is None:
            df_processed = df.copy()
            df_processed['count'] = 1
            pipeline.item_count_col = 'count'
        else:
            df_processed = df.copy()

        features = pipeline.create_features(df_processed, min_orders=min_orders)
        
        if features is None:
            st.error("❌ Feature creation failed")
            return

        # Display feature summary
        with st.expander("📊 Feature Summary", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Customers", f"{len(features):,}")
            with col2:
                st.metric("Features", f"{features.shape[1]}")
            with col3:
                st.metric("Avg Basket Size", f"{features['avg_basket_size'].mean():.1f}")
            with col4:
                st.metric("Avg Categories", f"{features['category_diversity'].mean():.1f}")

        # Step 2: Preprocessing
        st.subheader("🔄 Step 2: Data Preprocessing & PCA")
        
        features_pca = pipeline.apply_preprocessing(
            transformation=transformation,
            scaling=scaling_method,
            variance_threshold=variance_threshold
        )

        # Step 3: Clustering Evaluation
        st.subheader("🔍 Step 3: Clustering Viability Assessment")
        
        if cluster_mode == "Specify cluster count":
            # Force clustering with specified number
            st.info(f"🎯 Forcing {target_clusters} clusters (user specified)")
            
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            X = features_pca.values
            model = KMeans(n_clusters=target_clusters, random_state=42, n_init=50)
            labels = model.fit_predict(X)
            
            # Simulate clustering decision
            silhouette = silhouette_score(X, labels)
            pipeline.clustering_decision = {
                'viable': True,
                'best_algorithm': 'kmeans',
                'best_k': target_clusters,
                'best_silhouette': silhouette,
                'forced': True
            }
            
            st.success(f"✅ Using {target_clusters} clusters (silhouette: {silhouette:.3f})")
            
        else:
            # Auto-detect optimal clustering
            is_clustering_viable = pipeline.evaluate_clustering_viability(
                k_range=(2, 8),
                min_viable_clusters=3
            )
            
            if is_clustering_viable:
                decision = pipeline.clustering_decision
                st.success(f"✅ Optimal clustering: {decision['best_algorithm']} with {decision['best_k']} clusters")
                st.metric("Best Silhouette Score", f"{decision['best_silhouette']:.3f}")
            else:
                st.warning("⚠️ Traditional clustering not viable - using continuum approach")
                
                # Show why clustering failed
                with st.expander("📈 Clustering Evaluation Details"):
                    scores_df = pipeline.clustering_decision['all_scores']
                    st.dataframe(scores_df)

        # Step 4: Apply Segmentation
        st.subheader("🎯 Step 4: Segmentation")
        
        segments = pipeline.apply_segmentation(continuum_strategy=continuum_strategy)
        
        # Step 5: Customer Insights & Segment Characteristics
        st.subheader("🧠 Step 5: Customer Insights & Segment Characteristics")
        
        feature_importance = pipeline.train_interpretation_models()
        
        # Create user-friendly segment insights
        st.markdown("### 🎯 What Makes Each Segment Unique")
        
        # Get segment insights (assuming this method exists in your pipeline)
        try:
            segment_insights = pipeline.get_business_insights()
            
            for segment_name, insights in segment_insights.items():
                with st.expander(f"📊 {segment_name} Customers", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🛍️ Shopping Behavior:**")
                        for behavior in insights.get('behaviors', []):
                            st.write(f"• {behavior}")
                    
                    with col2:
                        st.markdown("**💡 Business Recommendations:**")
                        for recommendation in insights.get('recommendations', []):
                            st.write(f"• {recommendation}")
                    
                    # Show key metrics in a friendly way
                    if 'metrics' in insights:
                        st.markdown("**📈 Key Characteristics:**")
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        
                        with metrics_col1:
                            st.metric("Avg Orders", f"{insights['metrics'].get('avg_orders', 0):.1f}")
                        with metrics_col2:
                            st.metric("Avg Basket Size", f"${insights['metrics'].get('avg_spend', 0):.0f}")
                        with metrics_col3:
                            st.metric("Product Variety", f"{insights['metrics'].get('variety', 0):.1f}")
                            
        except:
            # Fallback to simplified feature importance if business insights aren't available
            st.markdown("### 🔍 Key Factors That Define Customer Segments")
            
            # Transform technical feature names to business-friendly descriptions
            feature_descriptions = {
                'avg_basket_size': 'Average Purchase Amount',
                'category_diversity': 'Product Variety Preference', 
                'purchase_frequency': 'Shopping Frequency',
                'total_spent': 'Total Customer Value',
                'avg_items_per_order': 'Items per Shopping Trip',
                'days_since_last_order': 'Recent Activity Level',
                'weekend_shopper': 'Weekend Shopping Preference',
                'weekday_shopper': 'Weekday Shopping Preference'
            }
            
            top_features = feature_importance.head(6)
            
            for idx, row in top_features.iterrows():
                feature_name = row['feature']
                friendly_name = feature_descriptions.get(feature_name, feature_name.replace('_', ' ').title())
                importance = row['rf_importance']
                
                # Create a visual importance bar
                importance_bar = "🟢" * int(importance * 10) + "⚪" * (10 - int(importance * 10))
                
                st.write(f"**{friendly_name}**: {importance_bar} (Key differentiator)")
            
            st.info("💡 These factors help us understand what makes each customer segment unique and how to serve them better.")

        # Step 6: Results & Visualizations
        st.subheader("📊 Step 6: Results & Analysis")
        
        # Create visualizations
        pipeline.create_interactive_visualizations()
        
        # Segment insights
        pipeline.get_segment_insights_streamlit()
        
        # Step 7: Export Results
        st.subheader("💾 Step 7: Export Results")
        
        segment_assignments = pipeline.export_results_streamlit()
        
        # Store results in session state for other pages
        st.session_state.clusters_df = pipeline.features.copy()
        st.session_state.cluster_map = dict(zip(pipeline.features.index, segments))
        st.session_state.cluster_user_col = user_col
        
        # Summary
        st.success("🎉 Hybrid segmentation complete!")
        
        # Final summary
        with st.expander("📋 Segmentation Summary", expanded=True):
            segment_counts = pd.Series(segments).value_counts()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", f"{len(segments):,}")
            with col2:
                st.metric("Segments Created", f"{len(segment_counts)}")
            with col3:
                st.metric("Method Used", pipeline.segment_method.title())
            
            st.write("**Segment Distribution:**")
            for segment, count in segment_counts.items():
                percentage = (count / len(segments)) * 100
                st.write(f"• **{segment}**: {count:,} customers ({percentage:.1f}%)")
        
    except Exception as e:
        st.error(f"❌ Segmentation failed: {str(e)}")
        
        # Debug information
        with st.expander("🐛 Debug Information"):
            st.code(f"Error: {e}")
            st.code(f"Data shape: {df.shape}")
            st.code(f"Columns: {df.columns.tolist()}")
            st.code(f"User column: {user_col}")
            st.code(f"Category column: {category_col}")

def page_rules():
    st.header("🔗 Market Basket Analysis")
    st.write("Discover products that are frequently bought together to optimize store layout and promotions")

    if ARP is None:
        st.info("association_rules_pipeline module not found.")
        return

    df = require_data()
    if df is None:
        st.markdown(
            """
            ### About Market Basket Analysis
            
            **🛒 What is Market Basket Analysis?**
            Market basket analysis helps you understand customer purchasing patterns by finding products that are frequently bought together.
            
            **💰 Business Benefits:**
            - **Cross-selling opportunities**: Recommend complementary products
            - **Store layout optimization**: Place related items near each other  
            - **Promotional strategies**: Bundle frequently bought items
            - **Inventory management**: Stock related items together
            
            **📊 What You'll Discover:**
            - "Customers who buy X also buy Y"
            - Strong product relationships
            - Seasonal buying patterns
            - Customer segment preferences
            
            **Data Requirements:**
            - Order/Transaction ID
            - Product/Item names
            - Customer segments (optional)
            """
        )
        return

    cols = df.columns.tolist()
    
    st.subheader("📊 Analysis Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        order_col = st.selectbox("Order/Transaction ID column", options=cols, index=(cols.index('order_id') if 'order_id' in cols else 0))
        item_col = st.selectbox("Product/Item column", options=cols, index=(cols.index('product_name') if 'product_name' in cols else 0))

    with col2:
        scope = st.radio("Analysis Scope", ["All customers", "Specific customer segment"], horizontal=True)
        
        # Simplified threshold setting
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Quick insights (loose criteria)", "Standard analysis (balanced)", "Deep insights (strict criteria)"],
            index=1,
            help="Choose how detailed you want the analysis to be"
        )

    # Set thresholds based on user-friendly selection
    if analysis_depth == "Quick insights (loose criteria)":
        min_support, min_conf, min_lift = 0.5, 0.2, 1.0
        explanation = "Finds many product relationships, including weaker ones"
    elif analysis_depth == "Standard analysis (balanced)":
        min_support, min_conf, min_lift = 1.0, 0.3, 1.2
        explanation = "Balanced approach - finds meaningful relationships"
    else:  # Deep insights
        min_support, min_conf, min_lift = 2.0, 0.5, 1.5
        explanation = "Finds only the strongest, most reliable relationships"
    
    st.info(f"📈 {explanation}")

    working = df.copy()
    if scope == "Specific customer segment":
        segment_found = False
        
        # Check for existing cluster column in the data
        if 'cluster' in df.columns:
            segments = sorted(map(str, pd.Series(df['cluster']).dropna().unique()))
            if segments:
                seg = st.selectbox("Choose customer segment", segments)
                working = df[df['cluster'].astype(str) == str(seg)]
                segment_found = True
                st.info(f"📊 Analyzing segment: {seg} ({len(working):,} transactions)")
        
        # Check for segments from session state (from Customer Segmentation page)
        elif st.session_state.cluster_map is not None and st.session_state.cluster_user_col in df.columns:
            ucol = st.session_state.cluster_user_col            
            
            # Check if user column exists and has data
            if ucol in df.columns:
                unique_users_in_data = df[ucol].dropna().unique()
                
                # Try multiple matching strategies for different data types
                users_with_segments = []
                
                # Strategy 1: Direct string matching
                users_with_segments_str = [u for u in unique_users_in_data if str(u) in st.session_state.cluster_map]
                
                # Strategy 2: Float matching (convert user IDs to float)
                try:
                    users_with_segments_float = [u for u in unique_users_in_data if float(u) in st.session_state.cluster_map]
                except:
                    users_with_segments_float = []
                
                # Strategy 3: Int matching (convert cluster map keys to int)
                try:
                    cluster_map_int_keys = {int(k): v for k, v in st.session_state.cluster_map.items()}
                    users_with_segments_int = [u for u in unique_users_in_data if int(u) in cluster_map_int_keys]
                except:
                    users_with_segments_int = []
                    cluster_map_int_keys = {}
                
                # Use the strategy that found the most matches
                if len(users_with_segments_float) > len(users_with_segments_str) and len(users_with_segments_float) > len(users_with_segments_int):
                    users_with_segments = users_with_segments_float
                    matching_strategy = "float"
                    cluster_map_to_use = st.session_state.cluster_map
                elif len(users_with_segments_int) > len(users_with_segments_str):
                    users_with_segments = users_with_segments_int
                    matching_strategy = "int"
                    cluster_map_to_use = cluster_map_int_keys
                else:
                    users_with_segments = users_with_segments_str
                    matching_strategy = "string"
                    cluster_map_to_use = {str(k): v for k, v in st.session_state.cluster_map.items()}
                
                if len(users_with_segments) > 0:
                    # Apply cluster mapping using the best strategy
                    tmp = df.copy()
                    
                    if matching_strategy == "float":
                        tmp['cluster'] = tmp[ucol].astype(float).map(cluster_map_to_use)
                    elif matching_strategy == "int":
                        tmp['cluster'] = tmp[ucol].astype(int).map(cluster_map_to_use)
                    else:
                        tmp['cluster'] = tmp[ucol].astype(str).map(cluster_map_to_use)
                    
                    # Remove rows where cluster mapping failed
                    tmp_with_clusters = tmp.dropna(subset=['cluster'])
                    
                    if len(tmp_with_clusters) > 0:
                        segments = sorted(map(str, pd.Series(tmp_with_clusters['cluster']).dropna().unique()))
                        if segments:
                            seg = st.selectbox("Choose customer segment", segments)
                            working = tmp_with_clusters[tmp_with_clusters['cluster'].astype(str) == str(seg)]
                            segment_found = True
                            st.info(f"📊 Analyzing segment: {seg} ({len(working):,} transactions)")
                    else:
                        st.warning("🔍 No transactions could be mapped to segments")
                else:
                    st.warning("🔍 No users from segmentation found in current dataset")
            else:
                st.warning(f"🔍 User column '{ucol}' not found in current dataset")
        
        # If no segments found, show helpful message
        if not segment_found:
            st.warning("🚫 No customer segments found!")
            st.markdown("""
            **To use segment-specific analysis:**
            
            1. **For the Online Retail dataset**: 
               - Go to **Customer Segmentation** page
               - Run segmentation on the **same dataset**
               - Come back here and segments will be available
            
            2. **For custom datasets**: 
               - Ensure you run segmentation on the **same data** you're analyzing here
               - Or add a 'cluster' column manually
            
            3. **Quick solution**: Use "All customers" analysis for now
            """)
            
            return  # Exit early if no segments available

    if st.button("🔍 Discover Product Relationships", type="primary", use_container_width=True):
        work = working[[order_col, item_col]].dropna()
        if work.empty:
            st.warning("No data available for analysis with the selected columns.")
            return
            
        with st.spinner("🔍 Analyzing purchasing patterns..."):
            pipe = ARP()
            rules = pipe.generate_rules(
                work,
                order_col=order_col,
                item_col=item_col,
                min_support=min_support,
                min_confidence=min_conf,
                min_lift=min_lift,
                verbose=False,
            )
            
        if rules is None or len(rules) == 0:
            st.warning("🚫 No strong product relationships found. Try using 'Quick insights' for a broader analysis.")
            return
        
        st.success(f"🎉 Found {len(rules):,} product relationships!")
        
        # Transform technical rules into business insights
        st.subheader("💡 Key Business Insights")
        
        # Sort by lift (strength of relationship) and show top insights
        top_rules = rules.nlargest(10, 'lift') if 'lift' in rules.columns else rules.head(10)
        
        
        insight_count = 1
        for idx, rule in top_rules.iterrows():
            try:
                # Use the correct column names from your data
                antecedents = rule.get('item_A', 'Unknown Product')
                consequents = rule.get('item_B', 'Unknown Product')
                confidence = rule.get('confidenceAtoB', 0)
                lift = rule.get('lift', 1)
                support = rule.get('supportAB', 0)
                
                # Convert confidence to percentage (it's already in decimal form)
                confidence_pct = confidence * 100
                
                # Support appears to already be in percentage form
                support_pct = support
                
            except Exception as e:
                st.error(f"Error processing rule {insight_count}: {e}")
                continue
            
            # Parse the product lists (they're stored as strings that look like lists)
            def parse_product_list(product_str):
                if isinstance(product_str, str):
                    # Remove brackets and quotes, split by comma
                    cleaned = product_str.strip("[]'\"").replace("'", "").replace('"', '')
                    products = [p.strip() for p in cleaned.split(',')]
                    return ', '.join(products)
                else:
                    return str(product_str)
            
            antecedents_str = parse_product_list(antecedents)
            consequents_str = parse_product_list(consequents)
            
            # Skip if we couldn't extract meaningful product names
            if antecedents_str in ["Unknown Product", ""] or consequents_str in ["Unknown Product", ""]:
                continue
            
            # Create business-friendly insight
            if lift > 3:
                strength = "Very Strong"
            elif lift > 2:
                strength = "Strong"
            else:
                strength = "Moderate"
            
            with st.expander(f"💡 Insight #{insight_count}: {antecedents_str} → {consequents_str}", expanded=insight_count <= 3):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🎯 The Pattern:**")
                    st.write(f"When customers buy **{antecedents_str}**, they also buy **{consequents_str}** about **{confidence_pct:.0f}%** of the time.")
                    
                    st.markdown("**📊 Relationship Strength:**")
                    st.write(f"**{strength}** relationship (customers are **{lift:.1f}x** more likely to buy these together)")
                
                with col2:
                    st.markdown("**💰 Business Actions:**")
                    if lift > 2:
                        actions = [
                            f"🛒 Place {consequents_str} near {antecedents_str} in store",
                            f"📦 Create bundle deals with these items",
                            f"🎯 Recommend {consequents_str} when customers buy {antecedents_str}",
                            f"📧 Include both in targeted promotions"
                        ]
                    else:
                        actions = [
                            f"🛒 Consider placing items nearby",
                            f"📧 Include in same promotional emails",
                            f"🎯 Cross-recommend during checkout"
                        ]
                    
                    for action in actions:
                        st.write(f"• {action}")
                
                # Show frequency info in user-friendly way
                if support_pct > 0:
                    st.info(f"📈 This pattern appears in **{support_pct:.1f}%** of all transactions")
                
                # Debug info for this rule (collapsed by default)
                with st.expander("🔧 Debug Info", expanded=False):
                    st.write(f"Raw item_A: {antecedents}")
                    st.write(f"Raw item_B: {consequents}")
                    st.write(f"Confidence: {confidence:.3f}")
                    st.write(f"Lift: {lift:.3f}")
                    st.write(f"Support: {support:.3f}")
            
            insight_count += 1
            
            # Limit to 10 insights to avoid clutter
            if insight_count > 10:
                break
        
        # Show message if no valid insights were found
        if insight_count == 1:
            st.warning("No valid product relationships could be displayed. Check the debug info above for data structure details.")
        
        # Summary recommendations
        st.subheader("🎯 Strategic Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏪 Store Layout Optimization:**")
            st.write("• Use the strongest relationships to reorganize product placement")
            st.write("• Create 'suggestion zones' near high-relationship items")
            st.write("• Design store flow to encourage complementary purchases")
        
        with col2:
            st.markdown("**📈 Marketing & Promotions:**")
            st.write("• Create bundle offers for frequently bought together items")
            st.write("• Design targeted campaigns based on purchase patterns")
            st.write("• Implement 'customers also bought' recommendations")
        
        # Option to view detailed data
        with st.expander("📋 Detailed Analysis Data", expanded=False):
            st.write("For analysts who want to see the raw numbers:")
            
            # Clean up the rules display
            display_rules = rules.copy()
            
            # Convert frozensets to strings for better display
            for col in display_rules.columns:
                if display_rules[col].dtype == 'object':
                    display_rules[col] = display_rules[col].apply(
                        lambda x: ', '.join(list(x)) if hasattr(x, '__iter__') and not isinstance(x, str) else x
                    )
            
            # Round numeric columns
            numeric_cols = display_rules.select_dtypes(include=[np.number]).columns
            display_rules[numeric_cols] = display_rules[numeric_cols].round(3)
            
            st.dataframe(display_rules, use_container_width=True)


def load_detector():
    """Load the fruit detection model and create detector instance"""
    try:
        if st.session_state.get('detector') is None:
            import tensorflow as tf
            from enhanced_fruit_detector import EnhancedFruitDetector
            
            # Load the trained model
            model = tf.keras.models.load_model(MODEL_PATH)
            
            # Create detector instance
            detector = EnhancedFruitDetector(model, DEFAULT_CLASS_NAMES)
            st.session_state.detector = detector
            st.session_state.fresh_model_loaded = True
            
        return st.session_state.detector
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
        

def classify_single_fruit(image_array, detector):
    """Classify a single fruit image without bounding boxes"""
    from PIL import Image
    import cv2
    
    # Get the expected input shape from the model
    try:
        expected_shape = detector.model.input_shape[1:3]  # Get (height, width) from (batch, height, width, channels)
        target_size = expected_shape
        st.info(f"🔧 Using model input size: {target_size}")
    except:
        # Fallback to common sizes
        target_size = (299, 299)  # Your model uses 299x299
        st.warning(f"⚠️ Using default input size: {target_size}")
    
    # Resize image to model input size
    image_resized = cv2.resize(image_array, target_size)
    
    # Ensure image is in correct format (normalize to 0-1)
    if image_resized.max() > 1.0:
        image_resized = image_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_resized, axis=0)
    
    
    # Get prediction
    try:
        predictions = detector.model.predict(image_batch, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = detector.class_names[predicted_class_idx]
        
        # Parse fruit type and freshness
        if predicted_class.startswith('good_'):
            fruit_type = predicted_class.replace('good_', '').replace('_', ' ').title()
            freshness = 'Fresh'
        elif predicted_class.startswith('stale_'):
            fruit_type = predicted_class.replace('stale_', '').replace('_', ' ').title()
            freshness = 'Stale'
        else:
            fruit_type = predicted_class.replace('_', ' ').title()
            freshness = 'Unknown'
        
        return {
            'fruit_type': fruit_type,
            'freshness': freshness,
            'confidence': confidence,
            'raw_prediction': predicted_class
        }
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        
        return {
            'fruit_type': 'Error',
            'freshness': 'Unknown',
            'confidence': 0.0,
            'raw_prediction': 'prediction_failed'
        }


def create_detection_visualization(image_array, detections):
    """Create matplotlib visualization with bounding boxes for multiple fruits"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_array)
    
    # Colors for different fruits
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    fruit_colors = {}
    color_idx = 0
    
    for detection in detections:
        fruit = detection['fruit_type']
        freshness = detection['freshness']
        conf = detection['confidence']
        bbox = detection['bbox']
        
        # Assign color for each fruit type
        if fruit not in fruit_colors:
            fruit_colors[fruit] = colors[color_idx % len(colors)]
            color_idx += 1
        
        color = fruit_colors[fruit]
        
        # Draw bounding box
        x, y, x2, y2 = bbox
        width = x2 - x
        height = y2 - y
        
        # Solid line for fresh, dashed for stale
        linestyle = '-' if freshness == 'Fresh' else '--'
        linewidth = 3
        
        rect = patches.Rectangle((x, y), width, height,
                               linewidth=linewidth, edgecolor=color, 
                               facecolor='none', linestyle=linestyle)
        ax.add_patch(rect)
        
        # Label
        label = f"{fruit}\n{freshness}\n{conf:.2f}"
        ax.text(x, y-5, label, fontsize=10, color='white', weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
    
    ax.set_title(f"Fruit Detection Results: {len(detections)} fruits found", fontsize=14, weight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig


def page_fresh():
    """Fruit detection page with separate modes for single vs multiple fruits"""
    st.title("🍎 Fruit Freshness Detection")
    st.write("Upload an image to detect fruits and assess their freshness")
    
    # Load model
    detector = load_detector()
    if detector is None:
        st.error("Could not load detection model. Please ensure the model file exists at: " + MODEL_PATH)
        return
    
    # Detection mode selection
    detection_mode = st.radio(
        "Detection Mode:",
        ["Single Fruit Classification", "Multiple Fruits Detection"],
        horizontal=True,
        help="Single mode: Classify entire image as one fruit. Multiple mode: Detect and locate multiple fruits with bounding boxes."
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Load image
        from PIL import Image
        import cv2
        
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert RGBA to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        # Show original image
        st.subheader("Original Image")
        st.image(image_array, use_container_width=True)
        
        # Detection button
        if st.button("🔍 Analyze Image", type="primary"):
            if detection_mode == "Single Fruit Classification":
                # Single fruit classification mode
                with st.spinner("Classifying fruit..."):
                    result = classify_single_fruit(image_array, detector)
                
                st.success("✅ Classification Complete!")
                
                # Display results in a nice format
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🍎 Fruit Type", result['fruit_type'])
                
                with col2:
                    freshness_color = "🟢" if result['freshness'] == 'Fresh' else "🔴"
                    st.metric("🌟 Freshness", f"{freshness_color} {result['freshness']}")
                
                with col3:
                    st.metric("🎯 Confidence", f"{result['confidence']:.1%}")
                
                # Recommendations based on freshness
                if result['freshness'] == 'Fresh':
                    st.success("✅ **Recommendation**: This fruit appears fresh and suitable for sale/consumption.")
                elif result['freshness'] == 'Stale':
                    st.warning("⚠️ **Recommendation**: This fruit appears stale. Consider removing from display or marking for discount.")
                else:
                    st.info("ℹ️ **Note**: Freshness status could not be determined.")
                    
            else:
                # Multiple fruits detection mode
                with st.spinner("Detecting multiple fruits..."):
                    result = detector.detect_fruits_multiscale(image_array, confidence_threshold=0.6)
                    detections = result['all_detections']
                
                if detections:
                    st.success(f"✅ Found {len(detections)} fruits!")
                    
                    # Show detection visualization with bounding boxes
                    fig = create_detection_visualization(image_array, detections)
                    st.pyplot(fig)
                    
                    # Show results table
                    st.subheader("🔍 Detection Details")
                    detection_data = []
                    fresh_count = 0
                    stale_count = 0
                    
                    for i, det in enumerate(detections, 1):
                        detection_data.append({
                            'ID': f"Fruit #{i}",
                            'Type': det['fruit_type'],
                            'Freshness': det['freshness'].lower(),
                            'Confidence': f"{det['confidence']:.1%}"
                        })
                        
                        if det['freshness'].lower() in ['fresh']:
                            fresh_count += 1
                        elif det['freshness'].lower() in ['stale']:
                            stale_count += 1
                        else:
                            # Handle unknown cases
                            st.warning(f"Unknown freshness value: {det['freshness']}")
                    
                    st.dataframe(detection_data, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Fruits", len(detections))
                    with col2:
                        st.metric("🟢 Fresh", fresh_count)
                    with col3:
                        st.metric("🔴 Stale", stale_count)
                    with col4:
                        fresh_percentage = (fresh_count / len(detections)) * 100
                        st.metric("Fresh %", f"{fresh_percentage:.1f}%")
                    
                    # Recommendations
                    if stale_count > 0:
                        st.warning(f"⚠️ **Quality Alert**: {stale_count} out of {len(detections)} fruits appear stale and should be reviewed.")
                    else:
                        st.success("✅ **Quality Check**: All detected fruits appear fresh!")
                        
                else:
                    st.warning("❌ No fruits detected. Try:")
                    st.write("- Ensuring the image contains clear, visible fruits")
                    st.write("- Using better lighting or image quality") 
                    st.write("- Trying single fruit classification mode instead")
    
    else:
        st.info("👆 Upload an image to get started")
        
        # Show example of what the model can detect
        with st.expander("🔍 What can this model detect?"):
            st.write("**Supported Fruits:**")
            fruits = list(set([name.replace('good_', '').replace('stale_', '').replace('_', ' ').title() 
                             for name in DEFAULT_CLASS_NAMES]))
            fruits.sort()
            
            # Display fruits in columns
            cols = st.columns(4)
            for i, fruit in enumerate(fruits):
                with cols[i % 4]:
                    st.write(f"• {fruit}")


# --- Router ---
if page == "Home":
    page_home()
elif page == "Data Upload":
    page_upload()
elif page == "Customer Segmentation":
    page_segmentation()
elif page == "Association Rules":
    page_rules()
elif page == "Freshness Detection":
    page_fresh()