import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
import os
from pathlib import Path
import time

# Set page configuration
st.set_page_config(
    page_title="ગુજરાતી સમાચાર શોધક",
    page_icon="📰",
    layout="wide",
)

# Constants
DATA_DIR = Path("data")

# Initialize session state
if 'index' not in st.session_state:
    st.session_state.index = None

def initialize_index():
    """Initialize the vector store index"""
    try:
        # Set up OpenAI credentials
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

        # Configure LlamaIndex
        llm = OpenAI(temperature=0, model="gpt-4")
        embed_model = OpenAIEmbedding()
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            chunk_size=1024
        )

        # Load documents
        if not DATA_DIR.exists():
            os.makedirs(DATA_DIR)
            st.warning("No documents found in data directory.")
            return None

        documents = SimpleDirectoryReader(
            input_dir=str(DATA_DIR),
            filename_as_id=True
        ).load_data()

        if not documents:
            st.warning("No documents found in data directory.")
            return None

        # Create index
        index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context,
            show_progress=True
        )

        return index

    except Exception as e:
        st.error(f"Error initializing index: {str(e)}")
        return None

def query_index(index, query, num_results=5):
    """Query the vector store index"""
    try:
        query_engine = index.as_query_engine(
            similarity_top_k=num_results,
            response_mode="tree_summarize"
        )
        response = query_engine.query(
            f"""Find relevant news about {query}. 
            For each relevant piece, provide:
            1. The original Gujarati text
            2. English translation
            3. Brief summary
            Format as:
            [Original Gujarati Text]
            [English Translation]
            [Brief Summary]
            ---"""
        )
        return response

    except Exception as e:
        st.error(f"Error querying index: {str(e)}")
        return None

def main():
    st.title("ગુજરાતી સમાચાર શોધક (Gujarati News Finder)")
    st.write("Search through indexed Gujarati newspapers")

    # Initialize index at startup
    if st.session_state.index is None:
        with st.spinner("Initializing document index..."):
            st.session_state.index = initialize_index()

    if st.session_state.index is None:
        st.error("Failed to initialize document index.")
        return

    # Search interface
    search_tag = st.text_input(
        "Enter search tag",
        placeholder="Enter topic in English or Gujarati",
        help="Enter the topic you want to search for in the newspapers"
    )

    num_results = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=5
    )

    # Process button
    if st.button("Search Newspapers 📰", key="search_btn"):
        if not search_tag:
            st.error("Please enter a search tag!")
            return

        try:
            with st.spinner("Searching newspapers..."):
                results = query_index(st.session_state.index, search_tag, num_results)

            if results:
                st.success("Search complete!")
                st.markdown("### 🔍 Search Results")
                st.markdown(str(results))
            else:
                st.warning("No relevant news found.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Help section
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Enter Tag**: Type the topic you want to search for
        2. **Adjust Results**: Select how many results you want to see
        3. **Search**: Click 'Search Newspapers' button
        4. **View Results**: See original text, translation, and summary

        **Note**:
        - The search uses AI to find and translate relevant content
        - Results include both Gujarati text and English translations
        """)

if __name__ == "__main__":
    main()
