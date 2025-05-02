import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="RAG-Bench Dashboard", layout="wide")

# --- Load Evaluation Report ---
@st.cache_data
def load_report(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)

    # Convert relevant columns to numeric to prevent formatting errors
    metrics = ["precision@5_value", "recall@5_value", "ndcg@5_value"]
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    return df

report_path = st.sidebar.text_input("📂 Evaluation Report Path", "reports/retrieval_performance.json")
df = load_report(report_path)

# --- Setup Tabs ---
st.title("RAG-Bench: Retrieval Evaluation Dashboard")
tab1, tab2, tab3 = st.tabs(["📊 Metrics Overview", "🔍 Query Explorer", "⚠️ Failure Analysis"])

# --- Tab 1: Metrics Overview ---
with tab1:
    st.header("Average Metrics by Retrieval Strategy")

    metrics = ["precision@5_value", "recall@5_value", "ndcg@5_value"]
    grouped = df.groupby("retriever")[metrics].mean().reset_index()

    # Ensure formatting is only applied to numeric columns
    st.dataframe(grouped.style.format("{:.2f}", subset=metrics))

    st.subheader("Bar Chart Comparison")
    st.bar_chart(grouped.set_index("retriever"))

# --- Tab 2: Query Explorer ---
# --- Load Queries ---
@st.cache_data
def load_queries(path: str):
    with open(path, "r") as f:
        queries = json.load(f)
    return {q["query_id"]: q["text"] for q in queries}

queries_path = st.sidebar.text_input("📂 Queries File Path", "queries/query_list.json")
query_map = load_queries(queries_path)

# --- Load document Corpus ---
@st.cache_data
def load_corpus(path: str):
    with open(path, "r") as f:
        raw_list = json.load(f)
    return {doc["id"]: doc["text"] for doc in raw_list}  # returns {doc_id: doc_text}


corpus_path = st.sidebar.text_input("📂 Corpus File Path", "data/corpus.json")
corpus = load_corpus(corpus_path)

with tab2:
    st.header("Per-Query Retrieval Explorer")

    # Dropdown: Select query by natural language
    if query_map:
        selected_text = st.selectbox("Select a query:", list(query_map.values()))
        selected_qid = next(qid for qid, text in query_map.items() if text == selected_text)

        # Filter rows for this query
        query_df = df[df["query_id"] == selected_qid]

        if not query_df.empty:
            for idx, row in query_df.iterrows():
                retriever_name = row["retriever"]
                doc_ids = row.get("retrieved_doc_ids", [])

                st.subheader(f"{idx + 1}. `{retriever_name}`")

                st.markdown(f"**Precision@5:** {row['precision@5_value']:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
                            f"**Recall@5:** {row['recall@5_value']:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
                            f"**NDCG@5:** {row['ndcg@5_value']:.2f}")

                for doc_id in doc_ids:
                    with st.expander(f"📄 {doc_id}"):
                        st.write(corpus.get(doc_id, "*Document not found.*"))
        else:
            st.warning("No results available for this query.")





# --- Tab 3: Failure Analysis ---
with tab3:
    st.header("⚠️ Failure Mode Analysis")

    st.markdown("Use filters below to identify underperforming queries.")

    # Filtering controls
    precision_thresh = st.slider("Min Precision@5 to flag as failure", 0.0, 1.0, 0.2, 0.05)
    recall_thresh = st.slider("Min Recall@5 to flag as failure", 0.0, 1.0, 0.5, 0.05)
    ndcg_thresh = st.slider("Min NDCG@5 to flag as failure", 0.0, 1.0, 0.5, 0.05)

    # Apply filtering
    fail_df = df[
        (df["precision@5_value"] < precision_thresh) |
        (df["recall@5_value"] < recall_thresh) |
        (df["ndcg@5_value"] < ndcg_thresh)
    ]

    if fail_df.empty:
        st.success("✅ No queries matched the failure criteria.")
    else:
        st.subheader(f"🚨 {len(fail_df)} Query Failures Detected")

        # Add query text
        fail_df["query_text"] = fail_df["query_id"].apply(lambda qid: query_map.get(qid, "N/A"))

        # Display results
        st.dataframe(fail_df[[
            "query_id", "query_text", "retriever",
            "precision@5_value", "recall@5_value", "ndcg@5_value"
        ]].sort_values(by=["precision@5_value"]))
