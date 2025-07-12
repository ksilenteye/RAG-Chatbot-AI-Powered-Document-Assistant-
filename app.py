import streamlit as st
from src.rag_pipeline import ImprovedRAGPipeline
import time

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🧠 RAG Chatbot (AI-Powered Document Assistant)")

@st.cache_resource
def load_rag():
    return ImprovedRAGPipeline()

rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown(f"**Model:** `DialoGPT-medium`")
    st.markdown(f"**Device:** `{rag.device.upper()}`")
    st.markdown(f"**# of Chunks:** `{len(rag.chunks)}`")
    
    stats = rag.get_performance_stats()
    st.markdown("### 📊 Performance Stats")
    st.markdown(f"**Cache Size:** `{stats['embedding_cache_size']}`")
    st.markdown(f"**Model:** `{stats['model_name']}`")
    
    st.markdown("### 🎛️ Parameters")
    k_chunks = st.slider("Number of chunks to retrieve", 3, 10, 5)
    similarity_threshold = st.slider("Similarity threshold", 0.1, 0.8, 0.3, 0.1)
    max_tokens = st.slider("Max tokens", 100, 1000, 512, 50)
    
    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        if msg["role"] == "assistant" and "metadata" in msg:
            metadata = msg["metadata"]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Confidence", f"{metadata['confidence']:.2f}")
            with col2:
                st.metric("Time", f"{metadata['processing_time']:.2f}s")
            with col3:
                st.metric("Chunks", len(metadata['chunks']))
            with col4:
                hallucination_status = "⚠️ Potential" if metadata['hallucination_detection']['is_potential_hallucination'] else "✅ Good"
                st.metric("Hallucination", hallucination_status)

query = st.chat_input("Ask something about AI and Machine Learning...")
if query:
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            result = rag.query(
                question=query,
                k=k_chunks,
                max_tokens=max_tokens,
                similarity_threshold=similarity_threshold
            )
            
            st.write(result["answer"])
            
            with st.expander("📚 Source Context & Analysis"):
                if result["chunks_with_scores"]:
                    st.markdown("### Retrieved Chunks (with similarity scores):")
                    for i, (chunk, score) in enumerate(result["chunks_with_scores"]):
                        st.markdown(f"**Chunk {i+1} (Score: {score:.3f}):**")
                        st.markdown(f"```{chunk[:300]}{'...' if len(chunk) > 300 else ''}```")
                
                if result["chunks"]:
                    st.markdown("### Filtered Relevant Chunks:")
                    for i, chunk in enumerate(result["chunks"]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.markdown(chunk)
                
                st.markdown("### 🕵️ Hallucination Analysis:")
                hallucination = result["hallucination_detection"]
                st.markdown(f"**Overlap Ratio:** {hallucination['overlap_ratio']:.3f}")
                st.markdown(f"**Potential Hallucination:** {'Yes' if hallucination['is_potential_hallucination'] else 'No'}")
                st.markdown(f"**Detection Confidence:** {hallucination['confidence']:.3f}")
                
                if hallucination['is_potential_hallucination']:
                    st.warning("⚠️ This response may contain hallucinations. Please verify the information.")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": result["answer"],
        "metadata": {
            "confidence": result["confidence"],
            "processing_time": result["processing_time"],
            "chunks": result["chunks"],
            "hallucination_detection": result["hallucination_detection"]
        }
    })

st.markdown("---")
st.markdown("""
### 💡 Tips for Better Results:
- **Be specific** in your questions
- **Ask follow-up questions** for more details
- **Check the source context** to verify information
- **Look at confidence scores** - higher is better
- **Watch for hallucination warnings** ⚠️
""")

st.markdown("### 🎯 Example Questions:")
example_questions = [
    "What is machine learning?",
    "Explain the difference between supervised and unsupervised learning",
    "How do neural networks work?",
    "What are the applications of AI in healthcare?",
    "What is deep learning and how does it relate to machine learning?"
]

for question in example_questions:
    if st.button(question, key=f"example_{question}"):
        st.session_state.messages.append({"role": "user", "content": question})
        st.rerun() 