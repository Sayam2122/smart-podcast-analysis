"""
Streamlit web interface for the podcast RAG system.
Provides interactive UI for querying and exploring podcast content.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import time
import json
from datetime import datetime
from typing import List, Dict, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import components
try:
    from utils.logger import get_logger
    from rag_system.vector_database import VectorDatabase
    from rag_system.query_processor import QueryProcessor
    from pipeline.pipeline_runner import PipelineRunner, list_sessions
    
    logger = get_logger(__name__)
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import components: {e}")
    COMPONENTS_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="🎙️ Podcast RAG System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .result-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
    }
    
    .metadata-tag {
        background-color: #e9ecef;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    
    .highlight-text {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'query_processor' not in st.session_state:
        st.session_state.query_processor = None
    if 'current_session' not in st.session_state:
        st.session_state.current_session = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None


def initialize_components():
    """Initialize RAG system components"""
    if not COMPONENTS_AVAILABLE:
        st.error("System components not available. Please check installation.")
        return False
    
    if st.session_state.vector_db is None:
        try:
            with st.spinner("Initializing RAG system..."):
                st.session_state.vector_db = VectorDatabase()
                st.session_state.query_processor = QueryProcessor(st.session_state.vector_db)
            
            st.success("✅ RAG system initialized successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {e}")
            st.info("Please ensure ChromaDB and SentenceTransformers are installed:")
            st.code("pip install chromadb sentence-transformers")
            return False
    
    return True


def main_interface():
    """Main application interface"""
    st.markdown('<h1 class="main-header">🎙️ Podcast RAG System</h1>', unsafe_allow_html=True)
    
    # Initialize components
    if not initialize_components():
        return
    
    # Sidebar
    create_sidebar()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📊 Analytics", "⚙️ Process Audio", "📁 Manage Sessions"])
    
    with tab1:
        search_interface()
    
    with tab2:
        analytics_interface()
    
    with tab3:
        processing_interface()
    
    with tab4:
        session_management_interface()


def create_sidebar():
    """Create sidebar with session selection and filters"""
    st.sidebar.markdown("## 🎛️ Controls")
    
    # Session selection
    st.sidebar.markdown("### 📁 Session Filter")
    
    try:
        sessions = st.session_state.vector_db.list_sessions() if st.session_state.vector_db else []
        
        session_options = ["All Sessions"] + [
            f"{session['session_id'][:8]}... ({session.get('processing_date', 'Unknown')[:10]})"
            for session in sessions
        ]
        
        selected_session = st.sidebar.selectbox(
            "Select Session",
            session_options,
            key="session_selector"
        )
        
        # Update current session
        if selected_session == "All Sessions":
            st.session_state.current_session = None
        else:
            session_index = session_options.index(selected_session) - 1
            st.session_state.current_session = sessions[session_index]['session_id']
        
        # Show session info
        if st.session_state.current_session:
            session_info = st.session_state.vector_db.get_session_summary(st.session_state.current_session)
            if session_info:
                st.sidebar.info(f"""
                **Session Info:**
                - Duration: {session_info.get('total_duration', 0)/60:.1f} min
                - Embeddings: {session_info.get('total_embeddings', 0):,}
                - Speakers: {len(session_info.get('speakers', []))}
                """)
    
    except Exception as e:
        st.sidebar.error(f"Error loading sessions: {e}")
    
    # Quick filters
    st.sidebar.markdown("### 🎯 Quick Filters")
    
    content_type = st.sidebar.selectbox(
        "Content Type",
        ["All", "Semantic Blocks", "Segments", "Summaries", "Key Insights"],
        key="content_type_filter"
    )
    
    # Map content types
    content_type_mapping = {
        "All": None,
        "Semantic Blocks": ["semantic_block"],
        "Segments": ["segment"],
        "Summaries": ["overall_summary"],
        "Key Insights": ["key_insights"]
    }
    
    st.session_state.content_type_filter = content_type_mapping[content_type]
    
    # Database stats
    st.sidebar.markdown("### 📈 Database Stats")
    
    if st.sidebar.button("🔄 Refresh Stats"):
        show_database_stats_sidebar()


def show_database_stats_sidebar():
    """Show database statistics in sidebar"""
    try:
        if st.session_state.vector_db:
            stats = st.session_state.vector_db.get_database_stats()
            
            if stats:
                st.sidebar.metric("Total Sessions", stats.get('total_sessions', 0))
                st.sidebar.metric("Total Embeddings", f"{stats.get('total_embeddings', 0):,}")
            else:
                st.sidebar.warning("Unable to load stats")
        else:
            st.sidebar.warning("Database not initialized")
    
    except Exception as e:
        st.sidebar.error(f"Error: {e}")


def search_interface():
    """Main search interface"""
    st.markdown('<h2 class="section-header">🔍 Search Podcast Content</h2>', unsafe_allow_html=True)
    
    # Search input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your question or search query:",
            placeholder="e.g., 'What did they discuss about AI?' or 'summarize the main points'",
            key="search_query"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary")
    
    # Quick query examples
    st.markdown("**💡 Example queries:**")
    example_cols = st.columns(4)
    
    examples = [
        ("📝 Summarize", "summarize the main points"),
        ("👥 Speaker", "what did the host say"),
        ("😊 Emotions", "happy moments"),
        ("📚 Topics", "about technology")
    ]
    
    for i, (label, example_query) in enumerate(examples):
        if example_cols[i].button(label, key=f"example_{i}"):
            st.session_state.search_query = example_query
            st.experimental_rerun()
    
    # Process search
    if search_button and query:
        process_search_query(query)
    
    # Show query history
    if st.session_state.query_history:
        with st.expander("📚 Query History"):
            for i, hist_query in enumerate(reversed(st.session_state.query_history[-5:])):
                if st.button(f"🔄 {hist_query}", key=f"history_{i}"):
                    st.session_state.search_query = hist_query
                    st.experimental_rerun()


def process_search_query(query: str):
    """Process search query and display results"""
    try:
        with st.spinner(f"Searching for: '{query}'..."):
            # Add to history
            if query not in st.session_state.query_history:
                st.session_state.query_history.append(query)
                if len(st.session_state.query_history) > 20:
                    st.session_state.query_history.pop(0)
            
            # Process query
            result = st.session_state.query_processor.process_query(
                query, 
                st.session_state.current_session
            )
            
            # Display results
            display_search_results(result)
    
    except Exception as e:
        st.error(f"❌ Search failed: {e}")


def display_search_results(result: Dict):
    """Display formatted search results"""
    response = result.get('response', {})
    results = result.get('results', [])
    metadata = result.get('metadata', {})
    
    # Results summary
    st.success(f"📊 {response.get('summary', 'Search completed')}")
    
    # Processing info
    col1, col2, col3 = st.columns(3)
    col1.metric("Results Found", len(results))
    col2.metric("Processing Time", f"{metadata.get('processing_time', 0):.2f}s")
    col3.metric("Query Intent", response.get('query_intent', 'Unknown').replace('_', ' ').title())
    
    if not results:
        st.warning("No results found. Try different keywords or check suggestions below.")
        
        # Show suggestions
        if response.get('suggestions'):
            st.markdown("**💡 Try these suggestions:**")
            for suggestion in response['suggestions']:
                st.markdown(f"- {suggestion}")
        return
    
    # Key highlights
    if response.get('highlights'):
        st.markdown('<h3 class="section-header">🎯 Key Highlights</h3>', unsafe_allow_html=True)
        
        for i, highlight in enumerate(response['highlights'], 1):
            st.markdown(f"""
            <div class="highlight-text">
                <strong>Highlight {i}:</strong> {highlight}
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed results
    st.markdown('<h3 class="section-header">📋 Detailed Results</h3>', unsafe_allow_html=True)
    
    # Results per page
    results_per_page = st.selectbox("Results per page:", [5, 10, 20], value=5)
    
    # Pagination
    total_pages = (len(results) - 1) // results_per_page + 1
    
    if total_pages > 1:
        page = st.selectbox("Page:", range(1, total_pages + 1)) - 1
    else:
        page = 0
    
    start_idx = page * results_per_page
    end_idx = min(start_idx + results_per_page, len(results))
    
    # Display results
    for i in range(start_idx, end_idx):
        result_item = results[i]
        display_result_card(result_item, i + 1)
    
    # Related suggestions
    if response.get('suggestions'):
        st.markdown('<h3 class="section-header">💡 Related Suggestions</h3>', unsafe_allow_html=True)
        
        suggestion_cols = st.columns(min(len(response['suggestions']), 3))
        for i, suggestion in enumerate(response['suggestions'][:3]):
            if suggestion_cols[i].button(suggestion, key=f"suggestion_{i}"):
                st.session_state.search_query = suggestion
                st.experimental_rerun()


def display_result_card(result: Dict, rank: int):
    """Display individual result card"""
    doc = result['document']
    metadata = result['metadata']
    similarity = result.get('similarity', 0)
    
    with st.container():
        st.markdown(f"""
        <div class="result-card">
            <h4>Result #{rank} (Score: {similarity:.2f})</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Document content
        with st.expander(f"📄 Content ({len(doc)} chars)", expanded=True):
            st.markdown(doc)
        
        # Metadata
        col1, col2 = st.columns(2)
        
        with col1:
            if 'speaker' in metadata:
                st.markdown(f"**👤 Speaker:** {metadata['speaker']}")
            
            if 'type' in metadata:
                st.markdown(f"**📁 Type:** {metadata['type'].replace('_', ' ').title()}")
            
            if 'emotion_label' in metadata:
                st.markdown(f"**😊 Emotion:** {metadata['emotion_label']}")
        
        with col2:
            if 'time_range' in result:
                st.markdown(f"**⏰ Time:** {result['time_range']}")
            
            if 'confidence' in metadata:
                st.markdown(f"**🎯 Confidence:** {metadata['confidence']:.2f}")
            
            if 'key_topics' in metadata and metadata['key_topics']:
                topics = metadata['key_topics'].split(',')[:3]
                st.markdown(f"**🏷️ Topics:** {', '.join(topics)}")
        
        # Action buttons
        button_col1, button_col2, button_col3 = st.columns(3)
        
        with button_col1:
            if st.button(f"🔍 Similar Content", key=f"similar_{rank}"):
                st.info("Similar content search not yet implemented")
        
        with button_col2:
            if st.button(f"📝 Context", key=f"context_{rank}"):
                show_result_context(result)
        
        with button_col3:
            if st.button(f"💾 Export", key=f"export_{rank}"):
                st.download_button(
                    "Download JSON",
                    json.dumps(result, indent=2),
                    f"result_{rank}.json",
                    "application/json",
                    key=f"download_{rank}"
                )


def show_result_context(result: Dict):
    """Show context around a result"""
    try:
        context_results = st.session_state.query_processor.get_conversation_context(result)
        
        if context_results:
            st.markdown("**📖 Conversation Context:**")
            
            for ctx in context_results[:5]:  # Show up to 5 context items
                ctx_meta = ctx['metadata']
                start_time = ctx_meta.get('start_time', 0)
                
                st.markdown(f"""
                **Time {start_time/60:.1f}m:** {ctx['document'][:150]}...
                """)
        else:
            st.info("No context available for this result")
    
    except Exception as e:
        st.error(f"Error retrieving context: {e}")


def analytics_interface():
    """Analytics and visualization interface"""
    st.markdown('<h2 class="section-header">📊 Content Analytics</h2>', unsafe_allow_html=True)
    
    if not st.session_state.vector_db:
        st.warning("Database not initialized")
        return
    
    # Analytics tabs
    analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs(["📈 Overview", "🏷️ Topics", "👥 Speakers"])
    
    with analytics_tab1:
        show_overview_analytics()
    
    with analytics_tab2:
        show_topic_analytics()
    
    with analytics_tab3:
        show_speaker_analytics()


def show_overview_analytics():
    """Show overview analytics"""
    try:
        # Database stats
        stats = st.session_state.vector_db.get_database_stats()
        
        if not stats:
            st.warning("No analytics data available")
            return
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Sessions", stats.get('total_sessions', 0))
        col2.metric("Total Embeddings", f"{stats.get('total_embeddings', 0):,}")
        col3.metric("Embedding Model", stats.get('embedding_model', 'Unknown'))
        col4.metric("Database Size", "N/A")  # Could calculate actual size
        
        # Content distribution
        content_dist = stats.get('content_type_distribution', {})
        
        if content_dist:
            st.markdown("### 📊 Content Type Distribution")
            
            # Create pie chart
            df = pd.DataFrame(
                list(content_dist.items()),
                columns=['Content Type', 'Count']
            )
            
            fig = px.pie(
                df,
                values='Count',
                names='Content Type',
                title="Content Distribution by Type"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Session timeline (if we had timestamps)
        st.markdown("### 📅 Recent Activity")
        sessions = st.session_state.vector_db.list_sessions()
        
        if sessions:
            # Create timeline chart
            df_sessions = pd.DataFrame(sessions)
            
            if 'processing_date' in df_sessions.columns:
                # Convert to datetime
                df_sessions['date'] = pd.to_datetime(df_sessions['processing_date'], errors='coerce')
                df_sessions = df_sessions.dropna(subset=['date'])
                
                if not df_sessions.empty:
                    fig = px.scatter(
                        df_sessions,
                        x='date',
                        y='audio_duration',
                        size='total_segments',
                        hover_data=['session_id'],
                        title="Session Processing Timeline"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading analytics: {e}")


def show_topic_analytics():
    """Show topic analytics"""
    try:
        st.markdown("### 🏷️ Popular Topics")
        
        # Get popular topics
        topics = st.session_state.query_processor.get_popular_topics(
            session_id=st.session_state.current_session,
            limit=20
        )
        
        if topics:
            # Create topic chart
            df_topics = pd.DataFrame(topics)
            
            # Bar chart
            fig = px.bar(
                df_topics.head(10),
                x='count',
                y='topic',
                orientation='h',
                title="Top 10 Most Discussed Topics",
                labels={'count': 'Mentions', 'topic': 'Topic'}
            )
            
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Topic table
            st.markdown("### 📋 All Topics")
            st.dataframe(
                df_topics,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No topic data available")
    
    except Exception as e:
        st.error(f"Error loading topic analytics: {e}")


def show_speaker_analytics():
    """Show speaker analytics"""
    try:
        st.markdown("### 👥 Speaker Analysis")
        
        # This would require aggregating speaker data from the vector database
        # For now, show a placeholder
        st.info("Speaker analytics coming soon!")
        
        # You could implement speaker statistics here by:
        # 1. Querying all segments with speaker metadata
        # 2. Aggregating by speaker
        # 3. Creating visualizations for speaking time, emotion distribution, etc.
    
    except Exception as e:
        st.error(f"Error loading speaker analytics: {e}")


def processing_interface():
    """Audio processing interface"""
    st.markdown('<h2 class="section-header">⚙️ Process Audio Files</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=['mp3', 'wav', 'm4a', 'flac'],
        help="Supported formats: MP3, WAV, M4A, FLAC"
    )
    
    # Processing options
    with st.expander("⚙️ Processing Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            transcription_model = st.selectbox(
                "Transcription Model",
                ["base", "small", "medium", "large"],
                index=0
            )
            
            diarization_speakers = st.slider(
                "Max Speakers",
                min_value=1,
                max_value=10,
                value=5
            )
        
        with col2:
            summarization_model = st.selectbox(
                "Summarization Model",
                ["mistral:7b", "llama3.2:3b", "llama2:7b"],
                index=0
            )
            
            embedding_model = st.selectbox(
                "Embedding Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                index=0
            )
    
    # Process button
    if st.button("🚀 Start Processing", type="primary", disabled=uploaded_file is None):
        if uploaded_file:
            process_uploaded_file(uploaded_file, {
                'transcription_model': transcription_model,
                'max_speakers': diarization_speakers,
                'summarization_model': summarization_model,
                'embedding_model': embedding_model
            })
    
    # Show processing status
    if st.session_state.processing_status:
        show_processing_status()


def process_uploaded_file(uploaded_file, config: Dict):
    """Process uploaded audio file"""
    try:
        # Save uploaded file
        temp_path = f"temp_{uploaded_file.name}"
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.processing_status = {
            'filename': uploaded_file.name,
            'status': 'processing',
            'progress': 0,
            'current_step': 'Starting...'
        }
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            st.info(f"🎵 Processing {uploaded_file.name}...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize pipeline with config
                pipeline_config = {
                    'transcription': {'model_name': config['transcription_model']},
                    'diarization': {'max_speakers': config['max_speakers']},
                    'summarization': {'model_name': config['summarization_model']},
                    'semantic_segmentation': {'embedding_model': config['embedding_model']}
                }
                
                # Run pipeline
                runner = PipelineRunner(config=pipeline_config)
                
                # Process in steps with progress updates
                status_text.text("Step 1/7: Audio ingestion...")
                progress_bar.progress(1/7)
                
                results = runner.process_audio_file(temp_path)
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                # Add to RAG system
                if st.session_state.vector_db:
                    session_id = st.session_state.vector_db.add_podcast_session(results)
                    
                    st.success(f"✅ Processing completed successfully!")
                    st.info(f"🆔 Session ID: {session_id}")
                    
                    # Update current session
                    st.session_state.current_session = session_id
                else:
                    st.warning("⚠️ RAG system not available - results saved but not indexed")
                
                # Show session info
                session_info = results.get('final_report', {})
                if session_info:
                    col1, col2, col3 = st.columns(3)
                    
                    audio_info = session_info.get('audio_info', {})
                    content_info = session_info.get('content_analysis', {})
                    
                    col1.metric("Duration", f"{audio_info.get('duration', 0)/60:.1f} min")
                    col2.metric("Segments", content_info.get('total_segments', 0))
                    col3.metric("Speakers", content_info.get('speakers_detected', 0))
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                st.session_state.processing_status = None
    
    except Exception as e:
        st.error(f"❌ Processing failed: {e}")
        st.session_state.processing_status = None


def show_processing_status():
    """Show current processing status"""
    status = st.session_state.processing_status
    
    if status['status'] == 'processing':
        st.info(f"🔄 Processing {status['filename']}...")
        st.progress(status['progress'])
        st.text(status['current_step'])


def session_management_interface():
    """Session management interface"""
    st.markdown('<h2 class="section-header">📁 Session Management</h2>', unsafe_allow_html=True)
    
    # Tabs for different views
    mgmt_tab1, mgmt_tab2 = st.tabs(["📊 RAG Sessions", "⚙️ Pipeline Sessions"])
    
    with mgmt_tab1:
        show_rag_sessions()
    
    with mgmt_tab2:
        show_pipeline_sessions()


def show_rag_sessions():
    """Show RAG database sessions"""
    try:
        st.markdown("### 📊 Sessions in RAG Database")
        
        sessions = st.session_state.vector_db.list_sessions() if st.session_state.vector_db else []
        
        if not sessions:
            st.info("No sessions found in RAG database")
            return
        
        # Create session dataframe
        df = pd.DataFrame(sessions)
        
        # Display sessions
        for i, session in enumerate(sessions):
            with st.expander(f"Session {i+1}: {session['session_id'][:16]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Duration", f"{session.get('audio_duration', 0)/60:.1f} min")
                    st.metric("Segments", session.get('total_segments', 0))
                
                with col2:
                    st.metric("Blocks", session.get('semantic_blocks', 0))
                    st.metric("Speakers", session.get('speakers_detected', 0))
                
                # Actions
                action_col1, action_col2, action_col3 = st.columns(3)
                
                with action_col1:
                    if st.button(f"🔍 Explore", key=f"explore_{i}"):
                        st.session_state.current_session = session['session_id']
                        st.experimental_rerun()
                
                with action_col2:
                    if st.button(f"📊 Overview", key=f"overview_{i}"):
                        show_session_details(session['session_id'])
                
                with action_col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{i}", type="secondary"):
                        if st.button(f"⚠️ Confirm Delete", key=f"confirm_delete_{i}"):
                            delete_rag_session(session['session_id'])
    
    except Exception as e:
        st.error(f"Error loading RAG sessions: {e}")


def show_pipeline_sessions():
    """Show pipeline output sessions"""
    try:
        st.markdown("### ⚙️ Pipeline Output Sessions")
        
        sessions = list_sessions()
        
        if not sessions:
            st.info("No pipeline sessions found")
            return
        
        # Display sessions
        for i, session in enumerate(sessions):
            with st.expander(f"Session {i+1}: {session['session_id'][:16]}..."):
                st.markdown(f"**Audio File:** {session.get('audio_file', 'Unknown')}")
                st.markdown(f"**Status:** {session.get('last_completed_step', 'Unknown')}")
                st.markdown(f"**Created:** {session.get('created_at', 'Unknown')}")
                
                # Actions
                if st.button(f"📥 Import to RAG", key=f"import_{i}"):
                    import_pipeline_session(session)
    
    except Exception as e:
        st.error(f"Error loading pipeline sessions: {e}")


def show_session_details(session_id: str):
    """Show detailed session information"""
    try:
        overview = st.session_state.vector_db.get_session_summary(session_id)
        
        if overview:
            st.json(overview)
        else:
            st.warning("Session details not available")
    
    except Exception as e:
        st.error(f"Error loading session details: {e}")


def delete_rag_session(session_id: str):
    """Delete session from RAG database"""
    try:
        success = st.session_state.vector_db.delete_session(session_id)
        
        if success:
            st.success(f"✅ Session deleted: {session_id[:16]}...")
            st.experimental_rerun()
        else:
            st.error("Failed to delete session")
    
    except Exception as e:
        st.error(f"Error deleting session: {e}")


def import_pipeline_session(session_info: Dict):
    """Import pipeline session to RAG database"""
    try:
        # Load session results
        session_dir = Path(session_info['session_dir'])
        results_path = session_dir / "complete_results.json"
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Add to RAG system
            session_id = st.session_state.vector_db.add_podcast_session(results)
            
            st.success(f"✅ Imported session to RAG database: {session_id}")
        else:
            st.error("Session results not found")
    
    except Exception as e:
        st.error(f"Error importing session: {e}")


# Main application
def main():
    """Main application entry point"""
    initialize_session_state()
    main_interface()


if __name__ == "__main__":
    main()
