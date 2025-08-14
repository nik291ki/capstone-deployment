import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, Tuple
import time
import gdown

class UniversalDataLoader:
    """
    Universal data loading component that can be embedded on any page.
    Handles preloaded demo data and user uploads.
    """
    
    def __init__(self, page_context: str = "general"):
        self.page_context = page_context
        self.demo_datasets = self._setup_demo_datasets()
    
    def _setup_demo_datasets(self) -> Dict:
        """Setup available demo datasets for the capstone project."""
        return {
            "instacart_full": {
                "name": "Instacart Dataset (Full)",
                "description": "Complete 2.4GB dataset with 3M+ orders, 200K+ users, comprehensive e-commerce data",
                "size_info": "2,400 MB • 3.2M orders • 200K customers",
                "file_path": "data/instacart_full.csv",
                "sample_size": None  # Load all data
            },
            "instacart_sample": {
                "name": "Instacart Dataset (Sample)", 
                "description": "500K orders sample - perfect for clustering and analysis",
                "size_info": "400 MB • 500K orders • 50K customers", 
                "file_path": "data/instacart_full.csv",
                "sample_size": 500000
            },
        }
    
    def show_data_loader(self, container_type: str = "main") -> bool:
        """
        Show the universal data loading interface.
        
        Args:
            container_type: "main" or "sidebar"
            
        Returns:
            bool: True if data was loaded successfully
        """
        
        # Check if data is already loaded
        if st.session_state.data_uploaded:
            return self._show_current_data_info(container_type)
        
        # Show data loading options
        return self._show_loading_options(container_type)
    
    def _show_current_data_info(self, container_type: str) -> bool:
        """Show information about currently loaded data with option to change."""
        
        if 'file_info' in st.session_state:
            file_info = st.session_state.file_info
            
            if container_type == "main":
                st.success(f"✅ **Data Loaded:** {file_info.get('name', 'Dataset')}")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"📊 {len(st.session_state.uploaded_data):,} rows × {len(st.session_state.uploaded_data.columns)} columns")
                    st.write(f"💾 Size: {file_info.get('size_mb', 0):.1f} MB")
                
                with col2:
                    if st.button("👁️ Preview Data", use_container_width=True):
                        self._show_data_preview()
                
                with col3:
                    if st.button("🔄 Load Different Data", use_container_width=True):
                        self._clear_data()
                        st.rerun()
            
            elif container_type == "sidebar":
                st.success(f"✅ {file_info.get('name', 'Data')}")
                st.caption(f"{len(st.session_state.uploaded_data):,} rows • {file_info.get('size_mb', 0):.1f} MB")
            
            return True
        
        return False
    
    def _show_loading_options(self, container_type: str) -> bool:
        """Show data loading options based on container type."""
        
        if container_type == "main":
            return self._show_main_loading_interface()
        elif container_type == "sidebar":
            return self._show_sidebar_loading_interface()
        
        return False
    
    def _show_main_loading_interface(self) -> bool:
        """Show full data loading interface for main content area."""
        
        st.markdown(f"""
        ### Load Data for {self.page_context.title()}
        
        Choose your data source to get started with analysis:
        """)
        
        # Simplified loading method selection - removed local directory option
        loading_method = st.radio(
            "Select data source:",
            [
                "Use Instacart Dataset",
                "Upload CSV/Excel File",
                "Load from URL",
            ],
            key=f"loading_method_{self.page_context}"
        )
        
        success = False
        
        if "Use Instacart Dataset" in loading_method:
            success = self._handle_preloaded_data()
        elif "Upload CSV/Excel File" in loading_method:
            success = self._handle_file_upload()
        elif "Load from URL" in loading_method:
            success = self._handle_url_loading()
        
        return success
    
    def _show_sidebar_loading_interface(self) -> bool:
        """Compact loading interface for sidebar."""
        
        st.markdown("### Load Data")
        
        load_option = st.selectbox(
            "Data source:",
            ["Select...", " Use Instacart Dataset", " Upload CSV/Excel File", " Load from URL"],
            key=f"sidebar_load_{self.page_context}"
        )
        
        if load_option == "Use Instacart Dataset":
            return self._handle_preloaded_data(compact=True)
        elif load_option == "Upload CSV/Excel File":
            return self._handle_file_upload(compact=True)
        elif load_option == "Load from URL":
            return self._handle_url_loading()
        
        return False
    
    def _handle_preloaded_data(self, compact: bool = False) -> bool:
        """Handle loading preloaded Instacart datasets."""
        
        if not compact:
            st.subheader("Use Instacart Dataset")
            st.markdown("""
            **Instacart retail dataset** prepared for demonstration:

            The Instacart dataset contains transaction data from a grocery delivery service, including over 3 million orders placed by over 200,000 users. 
            It provides information on products, order details, and customer behavior, with features like product names, categories, and order timestamps. 
            The dataset is loaded demonstration of the system. 
            """)
        
        # Dataset size selection
        dataset_options = list(self.demo_datasets.keys())
        dataset_labels = [self.demo_datasets[key]["name"] for key in dataset_options]
        
        if compact:
            selected_key = st.selectbox(
                "Dataset size:",
                dataset_options,
                format_func=lambda x: self.demo_datasets[x]["name"],
                key=f"demo_select_{self.page_context}"
            )
        else:
            # Show detailed options with descriptions
            st.markdown("**Choose dataset size:**")
            
            for i, (key, info) in enumerate(self.demo_datasets.items()):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if st.button(
                        "Load",
                        key=f"load_demo_{key}_{self.page_context}",
                        use_container_width=True
                    ):
                        return self._load_demo_dataset(key)
                
                with col2:
                    st.markdown(f"""
                    **{info['name']}**  
                    {info['description']}  
                    `{info['size_info']}`
                    """)
            
            return False
        
        # Load selected dataset (compact mode)
        if st.button("Load Data", key=f"load_demo_{self.page_context}"):
            return self._load_demo_dataset(selected_key)
        
        return False
    
    def _handle_file_upload(self, compact: bool = False) -> bool:
        """Handle user file upload."""
        
        if not compact:
            st.subheader("Upload Your Own Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose your retail data file:",
            type=['csv', 'xlsx', 'parquet'],
            key=f"file_upload_{self.page_context}",
            help="Supported: CSV, Excel, Parquet files"
        )
        
        if uploaded_file is not None:
            return self._process_uploaded_file(uploaded_file)
        
        return False
    
def load_large_gdrive_file(file_id: str, output="temp.csv"):
    """Download large Google Drive files bypassing virus scan."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, output, quiet=False)
    return output

def handle_url_loading_gdrive_safe():
    st.subheader("Load from Google Drive (Safe for Large Files)")

    gdrive_url = st.text_input("Enter Google Drive link:")
    if gdrive_url and st.button("Download & Load"):
        # Extract file ID
        import re
        match = re.search(r'/d/([a-zA-Z0-9_-]+)', gdrive_url)
        if not match:
            st.error("❌ Invalid Google Drive link.")
            return None, None

        file_id = match.group(1)

        with st.spinner("Downloading large file from Google Drive..."):
            local_path = load_large_gdrive_file(file_id, "sales_temp.csv")

        with st.spinner("Loading data..."):
            df = pd.read_csv(local_path, on_bad_lines="skip", low_memory=False)

        st.success(f"✅ Loaded {len(df):,} rows from Google Drive.")
        return df, {"name": os.path.basename(local_path), "source": "Google Drive"}

    return None, None
    
    def _process_uploaded_file(self, uploaded_file) -> bool:
        """Process an uploaded file."""
        
        file_size_mb = uploaded_file.size / (3000 * 3000)
        
        with st.spinner(f"Loading {uploaded_file.name}..."):
            try:
                # Load based on file type
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    data = pd.read_parquet(uploaded_file)
                else:
                    st.error("❌ Unsupported file format")
                    return False
                
                # Store in session state
                file_info = {
                    'name': uploaded_file.name,
                    'size_mb': file_size_mb,
                    'source': 'upload',
                    'type': uploaded_file.type
                }
                
                st.session_state.uploaded_data = data
                st.session_state.file_info = file_info
                st.session_state.data_uploaded = True
                
                st.success(f"✅ Uploaded {len(data):,} rows from {uploaded_file.name}")
                return True
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                return False
    
    def _process_url_data(self, url: str) -> bool:
        """Process data from URL."""
        
        with st.spinner("Loading data from URL..."):
            try:
                # Determine file type and load
                if url.endswith('.parquet'):
                    data = pd.read_parquet(url)
                elif url.endswith('.xlsx'):
                    data = pd.read_excel(url)
                else:
                    data = pd.read_csv(url)
                
                file_info = {
                    'name': url.split('/')[-1] or 'url_data',
                    'size_mb': len(data) * 0.001,
                    'source': 'url',
                    'type': 'URL',
                    'url': url
                }
                
                st.session_state.uploaded_data = data
                st.session_state.file_info = file_info
                st.session_state.data_uploaded = True
                
                st.success(f"✅ Loaded {len(data):,} rows from URL")
                return True
                
            except Exception as e:
                st.error(f"❌ Error loading from URL: {str(e)}")
                return False
    
    
    def _show_data_preview(self):
        """Show a preview of the current data."""
        if 'uploaded_data' in st.session_state:
            data = st.session_state.uploaded_data
            
            with st.expander("Data Preview", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Rows", f"{len(data):,}")
                with col2:
                    st.metric("Total Columns", len(data.columns))
                with col3:
                    memory_mb = data.memory_usage(deep=True).sum() / (1024**2)
                    st.metric("Memory Usage", f"{memory_mb:.1f} MB")
                
                # Show sample data
                st.dataframe(data.head(10), use_container_width=True)
                
                # Column information
                with st.expander("Column Details"):
                    col_info = pd.DataFrame({
                        'Column': data.columns,
                        'Type': data.dtypes.astype(str),
                        'Non-Null': data.count(),
                        'Unique Values': [data[col].nunique() for col in data.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
    
    def _clear_data(self):
        """Clear current data and reset session state."""
        keys_to_clear = [
            'uploaded_data', 'file_info', 'data_uploaded', 
            'clustering_complete', 'clustering_results',
            'column_mapping'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

# Utility functions for easy integration
def load_data_component(page_context: str = "general", container_type: str = "main") -> bool:
    """
    Easy-to-use function to add data loading to any page.
    
    Args:
        page_context: Context for the page (e.g., "clustering", "association")
        container_type: "main", "sidebar", or "expander"
    
    Returns:
        bool: True if data is loaded and ready for analysis
    """
    loader = UniversalDataLoader(page_context)
    return loader.show_data_loader(container_type)

def get_current_data() -> tuple[pd.DataFrame, dict]:
    """
    Get the currently loaded data and its metadata.
    
    Returns:
        tuple: (dataframe, file_info_dict) or (None, None) if no data loaded
    """
    if st.session_state.data_uploaded and 'uploaded_data' in st.session_state:
        return st.session_state.uploaded_data, st.session_state.get('file_info', {})
    return None, None

def ensure_data_loaded(page_context: str = "analysis") -> bool:
    """
    Ensure data is loaded before proceeding with analysis.
    Shows loading interface if no data is present.
    
    Args:
        page_context: Context for error messages
    
    Returns:
        bool: True if data is available, False otherwise
    """
    if not st.session_state.data_uploaded:
        st.warning(f"⚠️ No data loaded for {page_context}")
        st.info("👆 Please load data using one of the options above")
        return False
    
    return True

def show_data_summary() -> None:
    """Show a compact summary of currently loaded data."""
    data, file_info = get_current_data()
    
    if data is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{len(data):,}")
        with col2:
            st.metric("Columns", len(data.columns))
        with col3:
            unique_customers = data[data.columns[0]].nunique() if len(data.columns) > 0 else 0
            st.metric("Unique Customers", f"{unique_customers:,}")
        with col4:
            st.metric("Size", f"{file_info.get('size_mb', 0):.1f} MB")
    else:
        st.info("No data loaded")

if __name__ == "__main__":
    demo_usage_patterns()