import torch
import numpy as np
import pandas as pd
import json
import re
import hashlib
import joblib
from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

# ================= PURE PYTHON TEXTBLOB REPLACEMENT =================
class TextBlob:
    """Pure Python TextBlob replacement with sentiment analysis"""
    
    def __init__(self, text):
        self.text = str(text)
        
        # Sentiment lexicon (simplified)
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'best', 
            'perfect', 'wonderful', 'fantastic', 'superb', 'outstanding', 'brilliant',
            'happy', 'satisfied', 'pleased', 'recommend', 'positive', 'helpful',
            'excellent', 'quality', 'working', 'nice', 'fine', 'okay', 'decent',
            'worth', 'value', 'recommended', 'useful', 'effective', 'efficient'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'horrible', 'awful', 'poor', 'worst', 'hate',
            'disappointed', 'waste', 'useless', 'broken', 'failed', 'negative',
            'avoid', 'unhappy', 'dislike', 'problem', 'issue', 'complaint',
            'garbage', 'trash', 'junk', 'scam', 'fraud', 'fake', 'counterfeit',
            'slow', 'broken', 'damaged', 'defective', 'missing', 'wrong', 'error'
        }
        
        self.intensifiers = {'very', 'really', 'extremely', 'absolutely', 'totally', 'highly'}
        self.negation_words = {'not', "n't", 'no', 'never', 'none', 'nothing', 'nowhere'}
        
    @property
    def sentiment(self):
        """Calculate sentiment polarity and subjectivity"""
        try:
            words = self.words
            if not words:
                class Sentiment:
                    polarity = 0.0
                    subjectivity = 0.0
                return Sentiment()
            
            # Count positive and negative words
            pos_count = 0
            neg_count = 0
            
            for i, word in enumerate(words):
                # Check for negations
                negated = False
                for j in range(max(0, i-2), i):
                    if words[j] in self.negation_words:
                        negated = True
                        break
                
                # Check for intensifiers
                intensified = False
                for j in range(max(0, i-2), i):
                    if words[j] in self.intensifiers:
                        intensified = True
                        break
                
                # Score the word
                if word in self.positive_words:
                    if negated:
                        neg_count += 1
                    else:
                        pos_count += 2 if intensified else 1
                elif word in self.negative_words:
                    if negated:
                        pos_count += 1
                    else:
                        neg_count += 2 if intensified else 1
            
            # Calculate polarity (-1 to 1)
            total = pos_count + neg_count
            if total > 0:
                polarity = (pos_count - neg_count) / total
            else:
                polarity = 0.0
            
            # Adjust for exclamation marks and caps
            excl_count = self.text.count('!')
            caps_ratio = sum(1 for c in self.text if c.isupper()) / max(len(self.text), 1)
            
            if excl_count > 0:
                polarity = polarity * (1 + min(0.3, excl_count * 0.05))
            
            if caps_ratio > 0.3:
                polarity = polarity * (1 + caps_ratio * 0.2)
            
            # Calculate subjectivity (0 to 1)
            sentiment_words = pos_count + neg_count
            total_words = max(len(words), 1)
            subjectivity = min(1.0, sentiment_words / total_words * 1.5)
            
            # Clip polarity to [-1, 1]
            polarity = max(-1.0, min(1.0, polarity))
            
            class Sentiment:
                polarity = polarity
                subjectivity = subjectivity
            
            return Sentiment()
            
        except Exception:
            class Sentiment:
                polarity = 0.0
                subjectivity = 0.0
            return Sentiment()
    
    @property
    def words(self):
        """Extract words from text"""
        try:
            # Simple word tokenization
            text_lower = self.text.lower()
            # Remove punctuation and split
            text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
            words = [w for w in text_clean.split() if w.isalpha()]
            return words
        except Exception:
            return []
    
    @property
    def sentences(self):
        """Split text into sentences"""
        try:
            # Simple sentence splitting
            sentences = []
            current = []
            
            for char in self.text:
                current.append(char)
                if char in '.!?':
                    sentence = ''.join(current).strip()
                    if sentence:
                        sentences.append(sentence)
                    current = []
            
            # Add last sentence if any
            if current:
                sentence = ''.join(current).strip()
                if sentence:
                    sentences.append(sentence)
            
            return sentences
        except Exception:
            return [self.text]

# ================= CONFIGURATION =================
class Config:
    """Configuration matching your backend"""
    TFIDF_MAX_FEATURES = 5000
    VECTORIZER_CHOICE = 'tfidf'
    RANDOM_STATE = 42

# ================= FEATURE ENGINEER =================
class FeatureEngineerLeakageFree:
    """EXACT replica of your backend feature engineer"""
    
    def __init__(self, vectorizer_choice='tfidf'):
        self.vectorizer_choice = vectorizer_choice
        self.tfidf_vectorizer = None
        self.hashing_vectorizer = None
        self.is_fitted = False
        
    def transform(self, df):
        """Transform data using pre-fitted vectorizers - matches your backend"""
        print("Generating core features...")
        
        df = df.copy()
        df['reviewText'] = df['reviewText'].astype(str)
        df['summary'] = df['summary'].astype(str)
        
        features_df = pd.DataFrame(index=df.index)
        
        # Text features - EXACTLY as in your backend
        if self.tfidf_vectorizer is not None:
            review_text_vectors = self.tfidf_vectorizer.transform(df['reviewText'])
            summary_vectors = self.tfidf_vectorizer.transform(df['summary'])
        else:
            # Fallback if vectorizer not loaded
            review_text_vectors = np.zeros((len(df), 1000))
            summary_vectors = np.zeros((len(df), 500))
        
        # Convert sparse to dense with limited features - EXACT dimensions as backend
        review_text_df = pd.DataFrame(
            review_text_vectors.toarray()[:, :1000],  # First 1000 features
            index=df.index,
            columns=[f'text_feature_{i}' for i in range(min(1000, review_text_vectors.shape[1] if hasattr(review_text_vectors, 'shape') else 1000))]
        )
        
        summary_df = pd.DataFrame(
            summary_vectors.toarray()[:, :500],  # First 500 features
            index=df.index,
            columns=[f'summary_feature_{i}' for i in range(min(500, summary_vectors.shape[1] if hasattr(summary_vectors, 'shape') else 500))]
        )
        
        features_df = pd.concat([features_df, review_text_df, summary_df], axis=1)
        
        # Basic features - EXACTLY as in your backend
        features_df['overall_rating'] = pd.to_numeric(df['overall'], errors='coerce').fillna(0)
        features_df['review_length'] = df['reviewText'].apply(len)
        features_df['summary_length'] = df['summary'].apply(len)
        
        # Time features
        if 'unixReviewTime' in df.columns:
            df['reviewTime'] = pd.to_datetime(df['unixReviewTime'], unit='s', errors='coerce')
            features_df['day_of_week'] = df['reviewTime'].dt.dayofweek.fillna(0)
            features_df['hour_of_day'] = df['reviewTime'].dt.hour.fillna(0)
        else:
            features_df['day_of_week'] = 0
            features_df['hour_of_day'] = 0
        
        # Add metadata for advanced features
        features_df['reviewerID'] = df['reviewerID']
        features_df['asin'] = df['asin']
        if 'unixReviewTime' in df.columns:
            features_df['unixReviewTime'] = df['unixReviewTime']
        else:
            features_df['unixReviewTime'] = datetime.now().timestamp()
        
        features_df['reviewText'] = df['reviewText']
        features_df['summary'] = df['summary']
        
        # Add label if present
        if 'class' in df.columns:
            features_df['class'] = df['class']
        
        print(f"Core features generated: {features_df.shape[1]} features")
        return features_df
    
    def load(self, path):
        """Load fitted vectorizers"""
        try:
            saved = joblib.load(path)
            self.tfidf_vectorizer = saved['tfidf_vectorizer']
            self.hashing_vectorizer = saved['hashing_vectorizer']
            self.vectorizer_choice = saved['vectorizer_choice']
            self.is_fitted = True
            print(f"✓ Vectorizer loaded with vocabulary size: {len(self.tfidf_vectorizer.vocabulary_) if self.tfidf_vectorizer else 0}")
            return True
        except Exception as e:
            print(f"✗ Error loading vectorizer: {e}")
            # Create a dummy vectorizer as fallback
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
            self.tfidf_vectorizer.fit(["dummy text for initialization"])
            return False

# ================= ADVANCED FEATURE EXTRACTOR =================
class AdvancedFraudFeaturesLeakageFree:
    """EXACT replica of your advanced feature extraction"""
    
    def __init__(self, for_inference=True):
        self.for_inference = for_inference
        
    def extract_all_features(self, df, reference_features=None):
        """Extract all advanced features - matches your backend exactly"""
        print("Extracting advanced fraud features...")
        
        sampled_df = df.copy()
        
        # Initialize feature dictionaries
        coordinated_features = self._detect_coordinated_attacks(sampled_df)
        linguistic_features = self._analyze_linguistic_fingerprints(sampled_df)
        template_features = self._detect_template_reviews(sampled_df)
        emotional_features = self._analyze_emotional_manipulation(sampled_df)
        timing_features = self._detect_timing_anomalies(sampled_df)
        burst_features = self._sliding_window_analysis(sampled_df)
        
        # Combine all features
        all_features = {}
        reviewer_ids = set(sampled_df['reviewerID'].unique())
        
        for reviewer_id in reviewer_ids:
            features = {}
            
            # Coordinated attacks
            if reviewer_id in coordinated_features:
                features.update({
                    f'coord_{k}': v for k, v in coordinated_features[reviewer_id].items()
                })
            
            # Linguistic features
            if reviewer_id in linguistic_features:
                features.update({
                    f'ling_{k}': v for k, v in linguistic_features[reviewer_id].items()
                })
            
            # Template features
            if reviewer_id in template_features:
                features.update({
                    f'temp_{k}': v for k, v in template_features[reviewer_id].items()
                })
            
            # Emotional features
            if reviewer_id in emotional_features:
                features.update({
                    f'emo_{k}': v for k, v in emotional_features[reviewer_id].items()
                })
            
            # Timing features
            if reviewer_id in timing_features:
                features.update({
                    f'time_{k}': v for k, v in timing_features[reviewer_id].items()
                })
            
            # Burst features
            if reviewer_id in burst_features:
                features.update({
                    f'burst_{k}': v for k, v in burst_features[reviewer_id].items()
                })
            
            all_features[reviewer_id] = features
        
        print(f"Advanced features extracted for {len(all_features)} reviewers")
        return all_features
    
    def _detect_coordinated_attacks(self, df):
        """Detect coordinated review attacks"""
        features = defaultdict(lambda: {
            'num_coord_reviews': 0,
            'avg_time_diff': 0,
            'rating_var': 0
        })
        
        if 'asin' not in df.columns or 'unixReviewTime' not in df.columns:
            return features
        
        # For single review case (common in inference)
        if len(df) <= 1:
            for _, row in df.iterrows():
                rid = row['reviewerID']
                features[rid] = {
                    'num_coord_reviews': 0,
                    'avg_time_diff': 0,
                    'rating_var': 0
                }
            return dict(features)
        
        # Group by product
        for asin, group in df.groupby('asin'):
            if len(group) > 1:
                group = group.sort_values('unixReviewTime')
                time_diffs = group['unixReviewTime'].diff().dropna()
                avg_time_diff = time_diffs.mean() if not time_diffs.empty else 0
                rating_var = group['overall'].var() if len(group) > 1 else 0
                
                for _, row in group.iterrows():
                    rid = row['reviewerID']
                    features[rid]['num_coord_reviews'] += len(group)
                    features[rid]['avg_time_diff'] += avg_time_diff
                    features[rid]['rating_var'] += rating_var
        
        # Normalize
        for rid in list(features.keys()):
            product_count = df[df['reviewerID'] == rid]['asin'].nunique()
            if product_count > 0:
                features[rid]['num_coord_reviews'] /= product_count
                features[rid]['avg_time_diff'] /= product_count
                features[rid]['rating_var'] /= product_count
        
        return dict(features)
    
    def _analyze_linguistic_fingerprints(self, df):
        """Analyze linguistic patterns"""
        features = defaultdict(lambda: {
            'avg_word_len': 0,
            'sentiment': 0,
            'vocab_rich': 0,
            'sentence_var': 0
        })
        
        for idx, row in df.iterrows():
            text = str(row.get('reviewText', ''))
            if not text.strip():
                features[row['reviewerID']] = {
                    'avg_word_len': 0,
                    'sentiment': 0,
                    'vocab_rich': 0,
                    'sentence_var': 0
                }
                continue
            
            try:
                blob = TextBlob(text)
                words = [w for w in blob.words if w.isalpha()]
                
                if words:
                    avg_word_len = sum(len(w) for w in words) / len(words)
                    vocab_rich = len(set(words)) / len(words)
                else:
                    avg_word_len = 0
                    vocab_rich = 0
                
                sentences = [str(s).strip() for s in blob.sentences if str(s).strip()]
                if len(sentences) > 1:
                    sentence_lens = [len(s.split()) for s in sentences]
                    sentence_var = np.var(sentence_lens)
                else:
                    sentence_var = 0
                
                features[row['reviewerID']] = {
                    'avg_word_len': avg_word_len,
                    'sentiment': blob.sentiment.polarity,
                    'vocab_rich': vocab_rich,
                    'sentence_var': sentence_var
                }
            except:
                features[row['reviewerID']] = {
                    'avg_word_len': 0,
                    'sentiment': 0,
                    'vocab_rich': 0,
                    'sentence_var': 0
                }
        
        return dict(features)
    
    def _detect_template_reviews(self, df):
        """Detect template reviews using text similarity"""
        features = defaultdict(lambda: {
            'template_score': 0,
            'template_count': 0
        })
        
        # For single review case
        if len(df) <= 1:
            for _, row in df.iterrows():
                rid = row['reviewerID']
                features[rid] = {
                    'template_score': 0,
                    'template_count': 0
                }
            return dict(features)
        
        # Simple hash-based template detection
        text_hashes = {}
        for idx, row in df.iterrows():
            text = str(row.get('reviewText', '')).lower()
            text = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text)).strip()
            
            if len(text) > 20:
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash not in text_hashes:
                    text_hashes[text_hash] = []
                text_hashes[text_hash].append(row['reviewerID'])
        
        # Find template groups
        for text_hash, reviewers in text_hashes.items():
            if len(reviewers) > 1:
                for rid in set(reviewers):
                    features[rid]['template_score'] = 1.0
                    features[rid]['template_count'] = len(reviewers) - 1
        
        return dict(features)
    
    def _analyze_emotional_manipulation(self, df):
        """Detect emotional manipulation patterns"""
        features = defaultdict(lambda: {
            'extreme_sent': 0,
            'excl_ratio': 0,
            'caps_ratio': 0
        })
        
        for idx, row in df.iterrows():
            text = str(row.get('reviewText', ''))
            if not text.strip():
                features[row['reviewerID']] = {
                    'extreme_sent': 0,
                    'excl_ratio': 0,
                    'caps_ratio': 0
                }
                continue
            
            # Extreme sentiment
            try:
                sentiment = TextBlob(text).sentiment.polarity
                extreme_sent = 1 if abs(sentiment) > 0.8 else 0
            except:
                extreme_sent = 0
            
            # Exclamation ratio
            excl_ratio = text.count('!') / max(len(text), 1)
            
            # Caps ratio
            caps_chars = sum(1 for c in text if c.isupper())
            caps_ratio = caps_chars / max(len(text), 1)
            
            features[row['reviewerID']] = {
                'extreme_sent': extreme_sent,
                'excl_ratio': excl_ratio,
                'caps_ratio': caps_ratio
            }
        
        return dict(features)
    
    def _detect_timing_anomalies(self, df):
        """Detect timing anomalies"""
        features = defaultdict(lambda: {
            'avg_interval': 0,
            'burst_score': 0
        })
        
        if 'unixReviewTime' not in df.columns:
            for rid in df['reviewerID'].unique():
                features[rid] = {
                    'avg_interval': 0,
                    'burst_score': 0
                }
            return dict(features)
        
        for rid, group in df.groupby('reviewerID'):
            if len(group) > 1:
                group = group.sort_values('unixReviewTime')
                intervals = group['unixReviewTime'].diff().dropna()
                
                if not intervals.empty:
                    avg_interval = intervals.mean()
                    burst_score = intervals.std() / avg_interval if avg_interval > 0 else 0
                    
                    features[rid] = {
                        'avg_interval': avg_interval,
                        'burst_score': burst_score
                    }
            else:
                features[rid] = {
                    'avg_interval': 0,
                    'burst_score': 0
                }
        
        return dict(features)
    
    def _sliding_window_analysis(self, df):
        """Sliding window burst detection"""
        features = defaultdict(lambda: {
            'burst_count': 0,
            'max_burst': 0
        })
        
        if 'unixReviewTime' not in df.columns:
            for rid in df['reviewerID'].unique():
                features[rid] = {
                    'burst_count': 0,
                    'max_burst': 0
                }
            return dict(features)
        
        df = df.copy()
        df['reviewTime'] = pd.to_datetime(df['unixReviewTime'], unit='s', errors='coerce')
        df = df.dropna(subset=['reviewTime'])
        
        for rid, group in df.groupby('reviewerID'):
            if len(group) >= 3:  # MIN_REVIEWS_IN_WINDOW
                group = group.sort_values('reviewTime')
                times = group['reviewTime'].tolist()
                
                burst_count = 0
                max_burst = 0
                
                for i in range(len(times)):
                    from datetime import timedelta
                    window_start = times[i]
                    window_end = window_start + timedelta(days=7)  # WINDOW_SIZE_DAYS
                    
                    reviews_in_window = sum(1 for t in times if window_start <= t < window_end)
                    
                    if reviews_in_window >= 3:
                        burst_count += 1
                        max_burst = max(max_burst, reviews_in_window)
                
                features[rid] = {
                    'burst_count': burst_count,
                    'max_burst': max_burst
                }
            else:
                features[rid] = {
                    'burst_count': 0,
                    'max_burst': 0
                }
        
        return dict(features)

# ================= MAIN FRAUD SYSTEM =================
class HybridFraudSystemFixed:
    """Main system class matching your backend"""
    
    def __init__(self, data_path=""):
        # HARDCODED PATH HERE
        self.data_path = r"C:\Azeem's Work\Final year project\121025\data"
        self.feature_engineer = FeatureEngineerLeakageFree(Config.VECTORIZER_CHOICE)
        self.advanced_feature_extractor = AdvancedFraudFeaturesLeakageFree(for_inference=True)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.train_features_avg = None
        self.expected_adv_columns = []
    
    def load_models(self):
        """Load pre-trained models - matches your backend"""
        try:
            # HARDCODED PATHS
            model_path = os.path.join(self.data_path, "xgboost_fraud_detection_model.joblib")
            scaler_path = os.path.join(self.data_path, "scaler.joblib")
            vectorizer_path = os.path.join(self.data_path, "tfidf_vectorizer.joblib")
            features_path = os.path.join(self.data_path, "feature_names.json")
            adv_cols_path = os.path.join(self.data_path, "expected_adv_columns.json")
            train_avg_path = os.path.join(self.data_path, "train_features_avg.json")
            
            print(f"🔍 Loading models from: {self.data_path}")
            print(f"  1. Model: {model_path}")
            print(f"  2. Scaler: {scaler_path}")
            print(f"  3. Vectorizer: {vectorizer_path}")
            print(f"  4. Features: {features_path}")
            
            # Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load files
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_engineer.load(vectorizer_path)
            
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
            
            # Load expected advanced columns if exists
            if os.path.exists(adv_cols_path):
                with open(adv_cols_path, 'r') as f:
                    self.expected_adv_columns = json.load(f)
            
            # Load average training features if available
            if os.path.exists(train_avg_path):
                with open(train_avg_path, 'r') as f:
                    self.train_features_avg = json.load(f)
            
            print(f"✅ Models loaded successfully")
            print(f"   - Features: {len(self.feature_names)}")
            print(f"   - Advanced columns: {len(self.expected_adv_columns)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _combine_features(self, core_features, advanced_features, core_feature_cols):
        """Combine features - ROBUST VERSION matching your backend"""
        if 'reviewerID' not in core_features.columns:
            core_features = core_features.copy()
            core_features['reviewerID'] = 'unknown'
        
        combined = core_features[['reviewerID'] + core_feature_cols].copy()
        
        # Track all advanced feature columns
        all_adv_columns = set()
        
        # Add advanced features if they exist
        if advanced_features:
            for rid, features in advanced_features.items():
                mask = combined['reviewerID'] == rid
                if mask.any() and features:
                    for feat_name, feat_value in features.items():
                        col_name = f'adv_{feat_name}'
                        all_adv_columns.add(col_name)
                        
                        # Initialize column if it doesn't exist
                        if col_name not in combined.columns:
                            combined[col_name] = 0.0
                        
                        combined.loc[mask, col_name] = feat_value
        
        # Ensure ALL possible advanced feature columns exist
        for col_name in all_adv_columns:
            if col_name not in combined.columns:
                combined[col_name] = 0.0
        
        # Add expected advanced columns if we have them
        if hasattr(self, 'expected_adv_columns') and self.expected_adv_columns:
            for col in self.expected_adv_columns:
                if col not in combined.columns:
                    combined[col] = 0.0
        
        # Fill NaN values with 0
        combined = combined.fillna(0)
        
        return combined
    
    def classify_review(self, review_text, summary, overall, reviewer_id, asin):
        """Classify a single review - FIXED VERSION matching your backend"""
        if not self.model or not self.scaler or not self.feature_names:
            if not self.load_models():
                raise RuntimeError("Models not loaded.")
        
        # Create single review dataframe - EXACTLY as in backend
        review_data = {
            'reviewText': str(review_text),
            'summary': str(summary),
            'overall': float(overall),
            'reviewerID': reviewer_id,
            'asin': asin,
            'unixReviewTime': datetime.now().timestamp(),
            'class': 0  # Placeholder
        }
        
        df_single = pd.DataFrame([review_data])
        
        # Extract core features
        core_features = self.feature_engineer.transform(df_single)
        
        # Save core feature columns
        core_feature_cols = [col for col in core_features.columns 
                           if col not in ['reviewerID', 'asin', 'unixReviewTime', 
                                        'reviewText', 'summary', 'class']]
        
        # Extract advanced features (using training patterns)
        advanced_features = self.advanced_feature_extractor.extract_all_features(
            df_single,
            reference_features=self.train_features_avg
        )
        
        # Combine features
        combined_features = self._combine_features(core_features, advanced_features, core_feature_cols)
        
        # CRITICAL: Ensure the combined_features has ALL expected feature columns
        # Create a template dataframe with all expected features initialized to 0
        template_df = pd.DataFrame(0.0, index=[0], columns=self.feature_names)
        
        # Fill in the actual values we have
        for col in self.feature_names:
            if col in combined_features.columns:
                template_df[col] = combined_features[col].values[0]
        
        # Prepare for prediction
        X_scaled = self.scaler.transform(template_df)
        
        # Predict
        proba = self.model.predict_proba(X_scaled)[0]
        fraud_prob = proba[1]  # Probability of being spam/fraud
        prediction = 1 if fraud_prob > 0.5 else 0
        
        # Apply calibration - EXACTLY as in your backend
        calibrated_prob = self._calibrate_probability(
            fraud_prob, 
            float(overall), 
            len(str(review_text)),
            sum(1 for c in str(review_text) if c.isupper()) / max(len(str(review_text)), 1)
        )
        
        # Update prediction based on calibrated probability
        prediction = 1 if calibrated_prob > 0.5 else 0
        
        # Generate flags
        flags = {}
        if calibrated_prob > 0.7:
            flags['high_risk'] = True
        if calibrated_prob > 0.9:
            flags['very_high_risk'] = True
        
        # Check advanced features for specific patterns
        if advanced_features:
            reviewer_features = advanced_features.get(reviewer_id, {})
            if reviewer_features.get('template_score', 0) > 0.5:
                flags['template_detected'] = True
            if reviewer_features.get('extreme_sent', 0) == 1:
                flags['extreme_sentiment'] = True
            if reviewer_features.get('burst_count', 0) > 0:
                flags['burst_activity'] = True
        
        return {
            'classification': 'spam' if prediction == 1 else 'non-spam',
            'confidence': calibrated_prob if prediction == 1 else (1 - calibrated_prob),
            'fraud_probability': float(calibrated_prob),
            'raw_probability': float(fraud_prob),
            'flags': flags,
            'probabilities': {
                'non-spam': float(proba[0]),
                'spam': float(proba[1])
            },
            'features_used': len(self.feature_names),
            'calibrated': True
        }
    
    def _calibrate_probability(self, original_prob, rating, text_length, caps_ratio):
        """Calibration function matching your backend EXACTLY"""
        calibrated = float(original_prob)
        
        # Fix rating bias: Positive reviews less likely to be spam
        if rating >= 4.0:
            calibrated *= 0.4  # Reduce by 60%
        elif rating <= 2.0:
            calibrated = min(1.0, calibrated + 0.3)  # Increase
        
        # Fix overconfidence: Apply sigmoid smoothing
        calibrated = 1 / (1 + np.exp(-10 * (calibrated - 0.5)))
        
        # Consider text features
        if text_length < 30:  # Very short reviews suspicious
            calibrated = min(1.0, calibrated + 0.2)
        
        if caps_ratio > 0.4:  # Too many CAPS suspicious
            calibrated = min(1.0, calibrated + 0.3)
        
        # Ensure valid probability
        return max(0.05, min(0.95, calibrated))

# ================= STREAMLIT APP FUNCTIONS =================
# Global instance for the app
_fraud_system = None

def load_fraud_detection_model():
    """Load the fraud detection model (for Streamlit app)"""
    global _fraud_system
    
    if _fraud_system is None:
        print("Loading fraud detection model...")
        _fraud_system = HybridFraudSystemFixed()
        if not _fraud_system.load_models():
            raise RuntimeError("Failed to load fraud detection model")
    
    return _fraud_system

def predict_review(text, summary="", rating=5.0, user_id="user_001", product_id="prod_001"):
    """Predict if a review is spam (for Streamlit app)"""
    try:
        system = load_fraud_detection_model()
        
        if system is None:
            return {
                'is_spam': False,
                'confidence': 0.5,
                'probability': 0.5,
                'error': 'Model not loaded'
            }
        
        result = system.classify_review(text, summary, rating, user_id, product_id)
        
        return {
            'is_spam': result['classification'] == 'spam',
            'confidence': result['confidence'],
            'probability': result['fraud_probability'],
            'raw_probability': result.get('raw_probability', result['fraud_probability']),
            'flags': result['flags'],
            'features_used': result['features_used'],
            'calibrated': result['calibrated'],
            'probabilities': result['probabilities']
        }
        
    except Exception as e:
        print(f"Error in predict_review: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: simple rule-based prediction
        text_lower = text.lower()
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Simple heuristics
        spam_score = 0
        if rating == 5.0:
            spam_score += 0.3
        if caps_ratio > 0.3:
            spam_score += 0.3
        if '!' in text:
            spam_score += 0.1 * text.count('!')
        if len(text) < 20:
            spam_score += 0.2
        
        probability = min(0.95, max(0.05, spam_score))
        is_spam = probability > 0.5
        
        return {
            'is_spam': is_spam,
            'confidence': probability if is_spam else (1 - probability),
            'probability': probability,
            'raw_probability': probability,
            'flags': {'fallback_used': True, 'simple_heuristics': True},
            'features_used': 0,
            'calibrated': False,
            'probabilities': {
                'non-spam': 1 - probability,
                'spam': probability
            }
        }

# ================= TEST FUNCTION =================
def test_model_with_examples():
    """Test the model with example reviews"""
    examples = [
        ("PERFECT! BEST EVER!!!", 5.0),
        ("Good product, works as expected", 4.0),
        ("Terrible product, waste of money", 1.0),
        ("The camera quality is excellent", 4.0),
        ("Worst.", 1.0),
    ]
    
    system = HybridFraudSystemFixed()
    if not system.load_models():
        print("Failed to load models")
        return
    
    print("Testing model with examples:")
    print("-" * 50)
    
    for text, rating in examples:
        result = system.classify_review(
            text, "Test summary", rating, 
            f"test_user_{hash(text) % 1000}", 
            f"test_product_{hash(text) % 1000}"
        )
        print(f"Text: '{text[:30]}...'")
        print(f"Rating: {rating} → {result['classification']} (prob: {result['fraud_probability']:.4f})")
        print(f"Flags: {result['flags']}")
        print()

# Run test if executed directly
if __name__ == "__main__":
    test_model_with_examples()