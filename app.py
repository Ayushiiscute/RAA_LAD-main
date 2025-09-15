import streamlit as st
import os
from dotenv import load_dotenv
from backend import RAA_LAD_Runtime, RuntimeConfig

st.set_page_config(
    page_title="RAA-LAD | Threat Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    st.markdown("""
        <style>
            .stApp { background-color: #0E1117; color: #FAFAFA; }
            [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
            .stButton>button { background-color: #E50914; color: white; border-radius: 5px; border: none; font-weight: bold; }
            .stButton>button:hover { background-color: #F6121D; }
            .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid #30363D; }
            .stTabs [data-baseweb="tab"] { height: 48px; background-color: transparent; color: #8B949E; border-bottom: 2px solid transparent; }
            .stTabs [aria-selected="true"] { color: #FAFAFA; border-bottom: 2px solid #E50914; }
            .result-container { border: 1px solid #30363D; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }
            .high-risk-tag { background-color: #B80A12; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; display: inline-block; margin-top: 5px; }
        </style>
    """, unsafe_allow_html=True)

load_css()

def init_session_state():
    load_dotenv()
    defaults = {
        'runtime': None, 'model_loaded': False, 'results': [],
        'model_dir': "E:/Log/output/run_1757458008",
        'enable_network': True
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

with st.sidebar:
    st.title("🛡️ Configuration")
    st.markdown("---")
    
    st.session_state.model_dir = st.text_input("Enter Model Directory Path", value=st.session_state.model_dir)
    st.session_state.enable_network = st.checkbox("Enable Network for Threat Intel", value=st.session_state.enable_network)
    
    
    if st.button("Load Model", use_container_width=True):
        if not os.path.exists(st.session_state.model_dir):
            st.error(f"Directory not found: {st.session_state.model_dir}")
        else:
            with st.spinner("Loading model..."):
                try:
                    config = RuntimeConfig(
                        model_dir=st.session_state.model_dir,
                        enable_network=st.session_state.enable_network
                    )
                    st.session_state.runtime = RAA_LAD_Runtime(config)
                    st.session_state.model_loaded = True
                except Exception as e:
                    st.error(f"Model loading failed: {str(e)}")
                    st.session_state.model_loaded = False
    
    if st.session_state.model_loaded:
        st.success("Model is loaded and ready!")
        with st.expander("Loaded Configuration"):
            st.json({
                "Model Directory": st.session_state.model_dir,
                "Network Lookups": "Enabled" if st.session_state.enable_network else "Disabled",
                "Anomaly Threshold": f"{st.session_state.runtime.threshold:.3f}"
            })

st.title("RAA-LAD: Real-time Anomaly & Threat Detector")
st.markdown("Advanced log analysis using a dual-encoder model and threat intelligence.")

tab1, tab2 = st.tabs(["**Single Log Analysis**", "**Batch File Analysis**"])

with tab1:
    st.header("Analyze a Single Log Entry")
    log_message = st.text_area("Paste your log message here:", value="[2025-09-10 19:18:01] CRITICAL Multiple failed login attempts for user 'root' from 45.137.21.135", height=100, key="single_log_input")
    if st.button("Analyze Log"):
        if st.session_state.model_loaded and log_message.strip():
            with st.spinner("Analyzing..."):
                result = st.session_state.runtime.process_message(log_message.strip())
                st.session_state.results.insert(0, result)
        elif not st.session_state.model_loaded:
            st.warning("Please load the model using the sidebar first.")
        else:
            st.warning("Please paste a log message.")

with tab2:
    st.header("Analyze a Batch of Logs")
    uploaded_file = st.file_uploader("Upload a .txt file with one log per line.", type=['txt'])
    if st.button("Analyze File"):
        if st.session_state.model_loaded and uploaded_file is not None:
            lines = [line.strip() for line in uploaded_file.getvalue().decode("utf-8").splitlines() if line.strip()]
            new_results = []
            progress_bar = st.progress(0, text="Starting batch analysis...")
            for i, line in enumerate(lines):
                result = st.session_state.runtime.process_message(line)
                new_results.insert(0, result)
                progress_bar.progress((i + 1) / len(lines), text=f"Analyzed log {i+1}/{len(lines)}")
            st.session_state.results = new_results + st.session_state.results
            progress_bar.empty()
        elif not st.session_state.model_loaded:
            st.warning("Please load the model using the sidebar first.")
        else:
            st.warning("Please upload a file.")

if st.session_state.results:
    st.header("Analysis Results")
    for i, result in enumerate(st.session_state.results):
        is_anomaly = result.get('is_anomaly', False)
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([1, 4])
        with res_col1:
            st.metric(label=f"Result #{i+1}: {'ANOMALY' if is_anomaly else 'NORMAL'}", value=f"{result.get('score', 0):.4f}")
            if result.get('score', 0) > 0.9: st.markdown('<div class="high-risk-tag">⬆ High Risk</div>', unsafe_allow_html=True)
        with res_col2:
            st.code(result.get('message', ''), language='log')
        with st.expander(f"Show Details for Result #{i+1}"):
            st.markdown(f"**Explanation**")
            st.info(f"{'🔴' if is_anomaly else '🟢'} {result.get('explanation', 'N/A')}")
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.markdown("**Indicators of Compromise (IOCs)**"); st.json(result.get('iocs', {}))
            with detail_col2:
                st.markdown("**Threat Intelligence**"); st.json(result.get('threat_intel', {}))
        st.markdown('</div>', unsafe_allow_html=True)




