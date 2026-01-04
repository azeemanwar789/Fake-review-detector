import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import sys

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try importing the new fixed utils
try:
    from utils import load_fraud_detection_model, predict_review
    HAS_utils = True
except ImportError:
    from utils import load_fraud_detection_model, predict_review
    HAS_utils = False
    st.warning("⚠️ Using old utils.py - consider updating to utils.py")

# ================= CONFIGURATION =================
# Set your specific path here:
DATA_PATH = r"C:\Azeem's Work\Final year project\121025\data\\"

# Validate path exists
if not os.path.exists(DATA_PATH):
    st.error(f"❌ Data path not found: {DATA_PATH}")
    st.stop()
# ================================================

# Page configuration
st.set_page_config(
    page_title="Review Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .safe {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
    }
    .fraud {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
    }
    .calibrated-badge {
        background-color: #10B981;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    .uncalibrated-badge {
        background-color: #F59E0B;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #E0F2FE;
        border-left: 5px solid #0EA5E9;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stButton button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #2563EB;
        transform: translateY(-2px);
    }
    .probability-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin: 2px;
    }
    .low-risk { background-color: #10B981; color: white; }
    .medium-risk { background-color: #F59E0B; color: white; }
    .high-risk { background-color: #EF4444; color: white; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_info' not in st.session_state:
    st.session_state.model_info = {}
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None

def load_model():
    """Load the fraud detection model"""
    try:
        if HAS_utils:
            # Use the new fixed system
            from utils import load_fraud_detection_model, predict_review
            analyzer = load_fraud_detection_model()
            if analyzer:
                st.session_state.model_info = {
                    'model_type': 'XGBoost with Calibration',
                    'features': '1511 features (Core + Advanced)',
                    'calibration_status': 'CALIBRATED',
                    'backend': 'HybridFraudSystemFixed (from notebook)',
                    'version': '2.0 - Fixed Pipeline'
                }
                return analyzer
        else:
            # Fallback to old system
            st.warning("⚠️ Using legacy system - update to utils.py for better results")
            from utils import FraudReviewAnalyzer
            analyzer = FraudReviewAnalyzer(data_path=DATA_PATH)
            if analyzer:
                st.session_state.model_info = analyzer.get_model_info()
                return analyzer
        return None
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

def analyze_review_legacy(review_text, summary="", rating=5.0, user_id="user_001", product_id="prod_001"):
    """Legacy analysis using old system"""
    try:
        from utils import FraudReviewAnalyzer
        analyzer = FraudReviewAnalyzer(data_path=DATA_PATH)
        additional_features = {
            'review_length': len(review_text),
            'rating': rating,
            'user_age': 30,  # defaults
            'days_since_join': 365
        }
        return analyzer.analyze_review(review_text, additional_features)
    except Exception as e:
        return {'success': False, 'error': str(e)}

def analyze_review_new(review_text, summary="", rating=5.0, user_id="user_001", product_id="prod_001"):
    """New analysis using fixed backend pipeline"""
    try:
        result = predict_review(
            text=review_text,
            summary=summary,
            rating=rating,
            user_id=user_id,
            product_id=product_id
        )
        
        # Format result to match expected format
        is_fraud = result.get('is_spam', False)
        probability = result.get('probability', 0.5)
        confidence = result.get('confidence', 0.5)
        
        return {
            'success': True,
            'fraud_probability': probability,
            'raw_probability': result.get('raw_probability', probability),
            'calibrated_probability': probability if result.get('calibrated', False) else None,
            'is_fraud': is_fraud,
            'review_length': len(review_text),
            'word_count': len(review_text.split()),
            'exclamation_count': review_text.count('!'),
            'rating': rating,
            'calibrated': result.get('calibrated', False),
            'confidence': confidence,
            'flags': result.get('flags', {}),
            'features_used': result.get('features_used', 0),
            'probabilities': result.get('probabilities', {'non-spam': 1-probability, 'spam': probability})
        }
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return {'success': False, 'error': str(e)}

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Review Fraud & Spam Detector</h1>', unsafe_allow_html=True)
    
    # Load model if not already loaded
    if not st.session_state.model_loaded:
        with st.spinner('🔄 Loading fraud detection model... This may take a moment'):
            analyzer = load_model()
            if analyzer:
                st.session_state.analyzer = analyzer
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model")
                # Show file listing to help debug
                with st.expander("📁 Check Data Folder Contents"):
                    try:
                        files = os.listdir(DATA_PATH)
                        if files:
                            st.write(f"Files found in {DATA_PATH}:")
                            for file in sorted(files):
                                st.write(f"- {file}")
                        else:
                            st.write(f"❌ No files found in {DATA_PATH}")
                    except Exception as e:
                        st.write(f"❌ Cannot access {DATA_PATH}: {e}")
                return
    
    # Get model info
    model_info = st.session_state.model_info
    is_calibrated = model_info.get('calibration_status', '').lower() == 'calibrated'
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3067/3067256.png", width=100)
        st.title("⚙️ Settings & Control Panel")
        
        # Calibration status badge
        if is_calibrated:
            st.markdown('<div class="calibrated-badge">✅ CALIBRATED MODEL</div>', unsafe_allow_html=True)
            st.success("✓ Backend-calibrated probabilities (accurate)")
        else:
            st.markdown('<div class="uncalibrated-badge">⚠️ UNCALIBRATED MODEL</div>', unsafe_allow_html=True)
            st.warning("⚠️ Raw probabilities (may need adjustment)")
        
        # Model info
        with st.expander("📊 Model Information", expanded=True):
            st.json(model_info)
        
        # Backend type info
        st.subheader("🔧 Backend Type")
        backend_type = "Fixed (Notebook Pipeline)" if HAS_utils else "Legacy"
        st.info(f"Using: **{backend_type}**")
        
        # Threshold adjustment
        st.subheader("🎯 Detection Threshold")
        
        # Set default threshold based on calibration
        default_threshold = 0.5 if is_calibrated else 0.3
        min_threshold = 0.0
        max_threshold = 1.0
        step_size = 0.05
        
        if not is_calibrated:
            st.markdown('<div class="warning-box">⚠️ <b>Uncalibrated Model</b><br>Raw probabilities may be extreme. Use 0.3 threshold for testing.</div>', unsafe_allow_html=True)
        
        threshold = st.slider(
            "Adjust spam detection sensitivity",
            min_value=min_threshold,
            max_value=max_threshold,
            value=default_threshold,
            step=step_size,
            help=f"Higher = fewer false positives{' (0.5+ recommended for calibrated)' if is_calibrated else ' (0.3 recommended for uncalibrated)'}"
        )
        
        # Show current threshold info
        if is_calibrated:
            if threshold >= 0.5:
                st.success(f"✓ Threshold {threshold:.2f}: Balanced detection")
            elif threshold >= 0.3:
                st.info(f"ℹ️ Threshold {threshold:.2f}: Sensitive detection")
            else:
                st.warning(f"⚠️ Threshold {threshold:.2f}: Very sensitive (may have false positives)")
        else:
            if threshold <= 0.3:
                st.success(f"✓ Threshold {threshold:.2f}: Recommended for raw outputs")
            else:
                st.warning(f"⚠️ Threshold {threshold:.2f}: May miss actual spam")
        
        # Review metadata
        st.subheader("📝 Review Metadata")
        rating = st.selectbox("Rating", [1, 2, 3, 4, 5], index=4,
                            help="Product rating (1-5 stars)")
        
        # Show calibration effect
        if is_calibrated:
            with st.expander("🎯 Calibration Effects"):
                st.markdown("""
                **Calibration adjusts raw model outputs:**
                - Positive reviews (4-5 stars): Probability × 0.4
                - Negative reviews (1-2 stars): Probability + 0.3
                - Short reviews (<30 chars): +0.2
                - CAPS-heavy reviews (>40%): +0.3
                - Sigmoid smoothing applied
                """)
        
        # Action buttons
        st.subheader("⚡ Quick Actions")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                st.session_state.history = []
                st.success("History cleared!")
                st.rerun()
        
        with col_btn2:
            if st.button("🔍 Test Examples", type="secondary", use_container_width=True):
                st.session_state.run_examples = True
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Review Text")
        
        # Example reviews from your backend testing
        example_reviews = {
            "Obvious Spam": """PERFECT PRODUCT! BEST EVER! MUST BUY NOW!!! LIMITED TIME OFFER!!!""",
            
            "Template Spam": """I bought this product and it works great. Very satisfied with purchase. Would recommend to friends.""",
            
            "Legitimate Positive": """The camera quality is excellent in daylight, though low-light performance could be better. Battery lasts a full day with moderate use. The interface is intuitive but has a slight learning curve.""",
            
            "Legitimate Complaint": """Product arrived damaged. Customer service was unhelpful and refused replacement. Would not buy from this seller again.""",
            
            "Emoji Spam": """🔥🔥🔥 BEST PRODUCT EVER!!! 💯💯💯 MUST BUY!!! 🚀🚀🚀"""
        }
        
        # Quick example buttons
        st.write("💡 **Try example reviews (from backend testing):**")
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        with ex_col1:
            if st.button("🚨 Obvious Spam", use_container_width=True):
                st.session_state.example_review = example_reviews["Obvious Spam"]
                st.session_state.example_rating = 5
                st.rerun()
        with ex_col2:
            if st.button("✅ Legitimate", use_container_width=True):
                st.session_state.example_review = example_reviews["Legitimate Positive"]
                st.session_state.example_rating = 4
                st.rerun()
        with ex_col3:
            if st.button("😡 Complaint", use_container_width=True):
                st.session_state.example_review = example_reviews["Legitimate Complaint"]
                st.session_state.example_rating = 1
                st.rerun()
        
        # More examples
        ex_col4, ex_col5 = st.columns(2)
        with ex_col4:
            if st.button("📝 Template", use_container_width=True):
                st.session_state.example_review = example_reviews["Template Spam"]
                st.session_state.example_rating = 5
                st.rerun()
        with ex_col5:
            if st.button("🎭 Emoji Spam", use_container_width=True):
                st.session_state.example_review = example_reviews["Emoji Spam"]
                st.session_state.example_rating = 5
                st.rerun()
        
        # Clear example button if active
        if 'example_review' in st.session_state:
            if st.button("Clear Example", type="secondary", use_container_width=True):
                del st.session_state.example_review
                del st.session_state.example_rating
                st.rerun()
        
        # Text input area
        review_text = st.text_area(
            "Paste or type the review text here:",
            height=200,
            placeholder="Example: 'This product is amazing! I've bought 10 already and will buy 10 more tomorrow!!! BEST PRODUCT EVER!!!'",
            help="Enter the review text you want to analyze for spam/fraud indicators",
            value=st.session_state.get('example_review', '')
        )
        
        # Summary input
        summary_text = st.text_input(
            "Review Summary (Optional):",
            placeholder="Brief summary of the review",
            help="Short summary if available"
        )
        
        # Rating from sidebar or example
        current_rating = st.session_state.get('example_rating', rating)
        
        # Text statistics
        if review_text:
            with st.expander("📊 Text Statistics", expanded=False):
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Characters", len(review_text))
                with col_stat2:
                    st.metric("Words", len(review_text.split()))
                with col_stat3:
                    caps_ratio = sum(1 for c in review_text if c.isupper()) / max(len(review_text), 1)
                    st.metric("CAPS Ratio", f"{caps_ratio:.1%}")
                
                col_stat4, col_stat5 = st.columns(2)
                with col_stat4:
                    st.metric("Exclamations", review_text.count('!'))
                with col_stat5:
                    st.metric("Question Marks", review_text.count('?'))
        
        # Analyze button
        analyze_clicked = st.button("🔍 Analyze Review", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📈 Quick Stats")
        
        # Display metrics
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Reviews Analyzed", len(st.session_state.history))
        with metric_col2:
            if st.session_state.history:
                fraud_count = sum(1 for h in st.session_state.history if h.get('is_fraud', False))
                st.metric("Spam Detected", fraud_count)
        
        # Last analysis preview
        if st.session_state.history:
            st.subheader("📊 Last Result")
            last_result = st.session_state.history[-1]
            
            if last_result.get('success', False):
                result_color = "#EF4444" if last_result['is_fraud'] else "#10B981"
                result_text = "🚨 SPAM" if last_result['is_fraud'] else "✅ SAFE"
                
                # Risk level badge
                prob = last_result['fraud_probability']
                if prob < 0.3:
                    risk_badge = '<span class="probability-badge low-risk">Low Risk</span>'
                elif prob < 0.7:
                    risk_badge = '<span class="probability-badge medium-risk">Medium Risk</span>'
                else:
                    risk_badge = '<span class="probability-badge high-risk">High Risk</span>'
                
                st.markdown(f"""
                <div style='background-color:{result_color}20; padding:10px; border-radius:5px; border-left:4px solid {result_color}'>
                    <h4 style='margin:0; color:{result_color}'>{result_text} {risk_badge}</h4>
                    <p style='margin:5px 0;'><strong>Spam Probability:</strong> {last_result['fraud_probability']:.1%}</p>
                    <p style='margin:5px 0; font-size:0.9em;'>"{last_result.get('review_preview', '')[:50]}..."</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Backend info
        st.subheader("⚙️ Backend Info")
        st.info(f"Using: **{'Fixed Backend Pipeline' if HAS_utils else 'Legacy System'}**")
        
        if HAS_utils:
            st.caption("✓ Matches notebook preprocessing")
            st.caption("✓ 1511 features (Core + Advanced)")
            st.caption("✓ Backend calibration applied")
        else:
            st.warning("⚠️ Using legacy system - results may not match backend")
            st.caption("Update to utils.py for accurate results")
    
    # Process analysis
    if analyze_clicked and review_text:
        if not review_text.strip():
            st.warning("⚠️ Please enter some review text to analyze.")
        else:
            with st.spinner("🤖 Analyzing review for spam indicators..."):
                try:
                    # Choose analysis method based on available system
                    if HAS_utils:
                        result = analyze_review_new(
                            review_text, 
                            summary_text, 
                            current_rating,
                            f"user_{hash(review_text) % 10000}",
                            f"product_{hash(review_text) % 10000}"
                        )
                    else:
                        result = analyze_review_legacy(review_text)
                    
                    # Check for errors
                    if not result.get('success', False):
                        st.error(f"Analysis error: {result.get('error', 'Unknown error')}")
                    else:
                        # Add metadata
                        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        result['review_preview'] = review_text[:80] + "..." if len(review_text) > 80 else review_text
                        result['full_review'] = review_text
                        result['rating'] = current_rating
                        result['summary'] = summary_text
                        
                        # Add to history (keep last 50)
                        st.session_state.history.append(result)
                        if len(st.session_state.history) > 50:
                            st.session_state.history = st.session_state.history[-50:]
                        
                        # Display results
                        st.subheader("📊 Analysis Results")
                        
                        # Show probability comparison if available
                        if 'raw_probability' in result and result.get('calibrated', False):
                            with st.expander("🔧 Probability Details", expanded=True):
                                col_raw, col_cal, col_diff = st.columns(3)
                                with col_raw:
                                    raw_prob = result['raw_probability']
                                    st.metric("Raw Score", f"{raw_prob:.2%}", 
                                            delta="Before Calibration")
                                with col_cal:
                                    cal_prob = result['fraud_probability']
                                    st.metric("Calibrated", f"{cal_prob:.2%}", 
                                            delta="Final Score")
                                with col_diff:
                                    diff = cal_prob - raw_prob
                                    st.metric("Adjustment", f"{diff:+.2%}",
                                            delta_color="inverse" if diff > 0 else "normal")
                        
                        # Result box with color coding
                        is_fraud = result['fraud_probability'] > threshold
                        result_class = "fraud" if is_fraud else "safe"
                        result_text = "🚨 **SPAM DETECTED**" if is_fraud else "✅ **LEGITIMATE REVIEW**"
                        
                        # Calibration indicator
                        calibration_status = " (Calibrated)" if result.get('calibrated') else " (Raw)"
                        
                        st.markdown(f"""
                        <div class="result-box {result_class}">
                            <h3>{result_text}{calibration_status}</h3>
                            <p><strong>Spam Probability:</strong> <span style='font-size:1.3em; font-weight:bold;'>{result['fraud_probability']:.2%}</span></p>
                            <p><strong>Decision Threshold:</strong> {threshold:.0%} | <strong>Rating:</strong> {current_rating} stars</p>
                            <p><strong>Review Length:</strong> {result['review_length']} chars, {result['word_count']} words</p>
                            <p><strong>Analysis Time:</strong> {result['timestamp']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk assessment
                        prob = result['fraud_probability']
                        if prob < 0.3:
                            risk_level = "Low Risk"
                            risk_color = "green"
                            advice = "Likely legitimate review"
                        elif prob < 0.7:
                            risk_level = "Medium Risk"
                            risk_color = "orange"
                            advice = "Review with some suspicious elements"
                        else:
                            risk_level = "High Risk"
                            risk_color = "red"
                            advice = "High probability of spam"
                        
                        col_risk, col_advice = st.columns([1, 2])
                        with col_risk:
                            st.metric("Risk Level", risk_level)
                        with col_advice:
                            st.info(f"💡 **Assessment:** {advice}")
                        
                        # Display flags if any
                        if result.get('flags'):
                            st.subheader("🚩 Detected Flags")
                            flags = result['flags']
                            flag_cols = st.columns(3)
                            col_idx = 0
                            
                            for flag, value in flags.items():
                                if value:  # Only show True flags
                                    if flag == 'high_risk':
                                        badge = "🔴 High Risk"
                                        color = "#EF4444"
                                    elif flag == 'very_high_risk':
                                        badge = "🚨 Very High Risk"
                                        color = "#DC2626"
                                    elif flag == 'template_detected':
                                        badge = "📝 Template Detected"
                                        color = "#F59E0B"
                                    elif flag == 'extreme_sentiment':
                                        badge = "😲 Extreme Sentiment"
                                        color = "#8B5CF6"
                                    elif flag == 'burst_activity':
                                        badge = "⏱️ Burst Activity"
                                        color = "#0EA5E9"
                                    else:
                                        badge = f"⚠️ {flag}"
                                        color = "#6B7280"
                                    
                                    with flag_cols[col_idx % 3]:
                                        st.markdown(f"""
                                        <div style='background-color:{color}20; padding:10px; border-radius:5px; 
                                                    border-left:4px solid {color}; margin-bottom:10px;'>
                                            <strong>{badge}</strong>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    col_idx += 1
                        
                        # Probability gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result['fraud_probability'] * 100,
                            title={'text': "Spam Confidence Score", 'font': {'size': 20}},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "darkblue"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 30], 'color': "#10B981"},
                                    {'range': [30, 70], 'color': "#F59E0B"},
                                    {'range': [70, 100], 'color': "#EF4444"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': threshold * 100
                                }
                            }
                        ))
                        fig.update_layout(height=300, margin=dict(t=50, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature info if available
                        if result.get('features_used'):
                            st.info(f"📊 **Features Used:** {result['features_used']} features (Core + Advanced)")
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Run example tests if requested
    if st.session_state.get('run_examples', False):
        st.subheader("🧪 Example Review Tests")
        
        example_tests = [
            ("PERFECT! BEST EVER!!!", 5.0, "Obvious Spam"),
            ("Good product, works as expected", 4.0, "Legitimate Positive"),
            ("The camera quality is excellent in daylight", 4.0, "Detailed Review"),
            ("Worst product ever", 1.0, "Negative Review"),
            ("LOVE IT!!! MUST BUY NOW!!! 🔥🔥🔥", 5.0, "Emoji Spam"),
        ]
        
        results = []
        progress_bar = st.progress(0)
        
        for i, (text, rating, description) in enumerate(example_tests):
            progress_bar.progress((i + 1) / len(example_tests))
            
            if HAS_utils:
                result = analyze_review_new(text, "Test", rating, f"test_user_{i}", f"test_prod_{i}")
            else:
                result = analyze_review_legacy(text)
            
            if result.get('success', False):
                results.append({
                    'Description': description,
                    'Review': text[:40] + "..." if len(text) > 40 else text,
                    'Rating': rating,
                    'Spam Probability': f"{result['fraud_probability']:.2%}",
                    'Verdict': '🚨 SPAM' if result['is_fraud'] else '✅ SAFE',
                    'Calibrated': '✓' if result.get('calibrated') else '✗'
                })
        
        progress_bar.empty()
        
        if results:
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Summary
            st.success(f"✅ Tested {len(results)} examples")
            spam_count = sum(1 for r in results if '🚨' in r['Verdict'])
            st.info(f"📊 {spam_count}/{len(results)} flagged as spam")
        
        st.session_state.run_examples = False
    
    # Analysis History
    if st.session_state.history:
        st.subheader("📜 Analysis History")
        
        # Convert history to dataframe
        history_data = []
        for i, entry in enumerate(st.session_state.history):
            if entry.get('success', False):
                history_data.append({
                    '#': i + 1,
                    'Time': entry.get('timestamp', 'N/A').split(' ')[1],  # Just time
                    'Review': entry.get('review_preview', 'N/A'),
                    'Rating': entry.get('rating', 'N/A'),
                    'Score': f"{entry.get('fraud_probability', 0):.1%}",
                    'Verdict': '🚨 SPAM' if entry.get('is_fraud', False) else '✅ SAFE',
                    'Calibrated': '✓' if entry.get('calibrated', False) else '✗'
                })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            
            # Display history table
            st.dataframe(
                history_df,
                column_config={
                    "#": "#",
                    "Time": "Time",
                    "Review": st.column_config.TextColumn("Review", width="medium"),
                    "Rating": st.column_config.NumberColumn("Rating", format="%d⭐"),
                    "Score": st.column_config.TextColumn("Score", width="small"),
                    "Verdict": st.column_config.TextColumn("Result", width="small"),
                    "Calibrated": st.column_config.TextColumn("Cal", width="small", help="Calibration status")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Export option
            if st.button("📥 Export History as CSV"):
                export_df = pd.DataFrame(st.session_state.history)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"fraud_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    col_foot1, col_foot2, col_foot3 = st.columns(3)
    with col_foot1:
        st.caption(f"📂 Path: {DATA_PATH}")
    with col_foot2:
        if is_calibrated:
            st.caption("🛡️ Calibrated Fraud Detection v2.0")
            st.caption("Backend pipeline with 1511 features")
        else:
            st.caption("🛡️ Fraud Detection System v1.0")
            st.caption("Legacy system - update recommended")
    with col_foot3:
        st.caption(f"📊 Total Analyses: {len(st.session_state.history)}")
        if st.session_state.history:
            fraud_pct = sum(1 for h in st.session_state.history if h.get('is_fraud', False)) / len(st.session_state.history) * 100
            st.caption(f"📈 Spam Rate: {fraud_pct:.1f}%")

if __name__ == "__main__":
    main()