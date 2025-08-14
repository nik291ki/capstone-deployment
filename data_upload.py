import streamlit as st
import pandas as pd

def load_large_csv(file, chunksize=50000, preview_rows=500):
    """
    Load a large CSV file in chunks, return only the first N rows for preview.
    """
    preview_df = pd.DataFrame()
    total_rows = 0

    # Read the file in chunks
    for chunk in pd.read_csv(file, chunksize=chunksize):
        total_rows += len(chunk)
        if len(preview_df) < preview_rows:
            preview_df = pd.concat([preview_df, chunk], ignore_index=True)
        else:
            break

    return preview_df, total_rows

def data_upload_page():
    st.header("📥 Upload Data")

    uploaded_file = st.file_uploader("Upload CSV, Excel, or Parquet", type=["csv", "xlsx", "xls", "parquet"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()

        if file_type == "csv":
            with st.spinner("Loading CSV in chunks..."):
                preview_df, total_rows = load_large_csv(uploaded_file)
                st.success(f"✅ CSV loaded. Estimated total rows: {total_rows:,}")
                st.dataframe(preview_df)

                # Optionally store in session state
                uploaded_file.seek(0)  # Reset pointer for full read
                st.session_state.data = pd.read_csv(uploaded_file)  # Only do this if you need full data

        elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ Excel loaded. Rows: {len(df):,}")
            st.dataframe(df.head(500))
            st.session_state.data = df

        elif file_type == "parquet":
            df = pd.read_parquet(uploaded_file)
            st.success(f"✅ Parquet loaded. Rows: {len(df):,}")
            st.dataframe(df.head(500))
            st.session_state.data = df

        else:
            st.error("Unsupported file type.")

    else:
        st.info("Upload a file to get started.")
