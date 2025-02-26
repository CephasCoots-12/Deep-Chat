import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from newsapi import NewsApiClient
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import plotly.express as px
import gtts
import speech_recognition as sr
import torch
import os
import gdown
from datetime import datetime

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NEWS_API_KEY = "601b0246d5c34acbbd94c17a1c23945e"

# Google Drive Direct Download Link
GDRIVE_MODEL_LINK = "https://drive.google.com/drive/folders/1phFOK6nT2Wvfq1JAzH35Rf-GbTV0N26P?usp=sharing"
MODEL_PATH = "mistral-7b-quantized"

# --- Function to Download Model from Google Drive ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("🚀 Downloading model from Google Drive... (This may take time)")
        gdown.download(GDRIVE_MODEL_LINK, MODEL_PATH, quiet=False)

# --- Core Functionality ---
@st.cache_resource
def load_models():
    # Download model if not available
    download_model()

    # 4-bit Quantized Mistral 7B
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    # Supporting models
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    sentiment = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    
    return tokenizer, model, embed_model, sentiment

tokenizer, model, embed_model, sentiment_analyzer = load_models()

# --- Real-Time Features ---
class FinanceAssistant:
    def __init__(self):
        self.newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        self.index = faiss.IndexFlatL2(384)
        self.articles = []
        
    def refresh_news(self):
        """Fetch and index latest financial news"""
        news = self.newsapi.get_everything(q="finance OR stocks OR acquisitions", language="en", sort_by="publishedAt", page_size=20)
        self.articles = news['articles']
        
        # Update FAISS index
        embeddings = embed_model.encode([f"{a['title']} {a['description']}" for a in self.articles])
        self.index.add(np.array(embeddings).astype('float32'))
        
    def rag_search(self, query, k=3):
        """Retrieve relevant news context"""
        query_embed = embed_model.encode([query])
        _, indices = self.index.search(np.array(query_embed).astype('float32'), k)
        return [self.articles[i] for i in indices[0]]

assistant = FinanceAssistant()

# --- Advanced UI ---
st.set_page_config(
    page_title="FinGPT Assistant",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #1a1a1a 0%, #2d2d2d 100%);
}
            
.stTextInput input {
    background: #333333 !important;
    color: white !important;
}
            
[data-testid="stSidebar"] {
    background: #262626 !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ Controls")
    voice_input = st.checkbox("🎤 Voice Input")
    risk_level = st.select_slider("🔍 Analysis Depth", ["Brief", "Normal", "Detailed"])
    st.divider()
    st.write("🔄 Last Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Main Interface
st.title("💹 FinGPT - Financial Intelligence Assistant")
st.caption("Your AI-powered guide through financial markets")

# Input Section
col1, col2 = st.columns([4,1])
with col1:
    query = st.text_input("Ask any financial question:", placeholder="E.g.: What's the impact of Zomato's latest acquisition?")
with col2:
    if st.button("🚀 Analyze"):
        pass

# Voice Input Handling
if voice_input:
    with st.spinner("Listening..."):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            audio = recognizer.listen(source, timeout=5)
            try:
                query = recognizer.recognize_google(audio)
            except:
                st.error("Voice input failed")

# Processing Pipeline
if query:
    with st.spinner("🔄 Analyzing markets & generating insights..."):
        # Refresh data
        assistant.refresh_news()
        
        # RAG Context
        context_articles = assistant.rag_search(query)
        context = "\n".join([f"Article {i+1}: {a['title']} - {a['description']}" 
                           for i, a in enumerate(context_articles)])
        
        # Generate Response
        prompt = f"""You are a senior financial analyst. Use this context:
        
        {context}
        
        Question: {query}
        
        Provide {risk_level.lower()} analysis including:
        1. Key event summary
        2. Historical context
        3. Market implications
        4. Expert opinions
        5. Potential risks/opportunities
        
        Answer:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=400)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-processing
        response = response.split("Answer:")[-1].strip()

    # Display Results
    with st.container():
        st.header("📈 Analysis Results")
        
        # Main Insights
        with st.expander("💡 Core Analysis", expanded=True):
            st.write(response)
            
            # Text-to-Speech
            audio_file = f"response_{datetime.now().timestamp()}.mp3"
            tts = gtts.gTTS(response, lang='en')
            tts.save(audio_file)
            st.audio(audio_file)

        # Supporting Visualizations
        tab1, tab2, tab3 = st.tabs(["📰 Related News", "📊 Sentiment Analysis", "🧠 AI Breakdown"])

        with tab1:
            for article in context_articles:
                st.subheader(article['title'])
                st.caption(f"Source: {article['source']['name']} | {article['publishedAt'][:10]}")
                st.write(article['description'])
                st.markdown(f"[Read Full Article]({article['url']})")
                st.divider()

        with tab2:
            sentiments = [sentiment_analyzer(a['title'])[0]['label'] for a in context_articles]
            fig = px.pie(values=[sentiments.count('positive'), sentiments.count('negative'), sentiments.count('neutral')],
                        names=['Positive', 'Negative', 'Neutral'],
                        color_discrete_sequence=['green', 'red', 'gray'])
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.write("### AI Reasoning Process")
            st.json({
                "query": query,
                "context_articles": [a['title'] for a in context_articles],
                "risk_level": risk_level,
                "generation_parameters": {
                    "model": MODEL_NAME,
                    "max_tokens": 400,
                    "temperature": 0.3
                }
            })
