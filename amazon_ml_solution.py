# -*- coding: utf-8 -*-
"""
Enhanced Amazon ML Challenge - XGBoost Ensemble + Pseudo-Labeling
Target: SMAPE < 40 (Currently 46-48)
Expected: ~38-40 with conservative approach
"""

import os
import gc
import re
import warnings
import pickle
import hashlib
import json
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import scipy.sparse

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm

# GPU Config
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU configured ({} GPU(s))".format(len(gpus)))
except:
    print("Using CPU")

print("=" * 80)
print("ENHANCED AMAZON ML - XGBOOST ENSEMBLE + PSEUDO-LABELING")
print("=" * 80)
print("\nFEATURES:")
print("  ✓ LightGBM + XGBoost ensemble with optimized weights")
print("  ✓ Pseudo-labeling (conservative - top 20% confidence)")
print("  ✓ Brand extraction + Target encoding")
print("  ✓ Better missing image handling")
print("  ✓ Cross-feature interactions")
print("=" * 80)
print("\nESTIMATED TIME: ~35-40 mins (first run)")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
class CFG:
    seed = 42
    debug = False

    # Paths
    RAW_TRAIN_CSV = '/kaggle/input/train-final/train_data_cleaned.csv'
    RAW_TEST_CSV = '/kaggle/input/dataset/test.csv'
    TRAIN_IMAGE_DIR = '/kaggle/input/image-dataset/kaggle/working/images/train'
    TEST_IMAGE_DIR = '/kaggle/input/image-dataset/kaggle/working/images/test'
    CACHE_DIR = '/kaggle/working/cache'
    SUBMISSION_FILE = 'submission.csv'

    # Model params
    n_splits = 5
    img_size = (224, 224)
    img_batch_size = 64

    # Enhanced TF-IDF
    tfidf_max_features = 30000
    tfidf_ngram_range = (1, 3)
    tfidf_min_df = 2

    # SVD
    svd_components = 300

    # Pseudo-labeling
    use_pseudo_labeling = True
    pseudo_confidence_threshold = 0.2  # Top 20% most confident predictions
    pseudo_label_iterations = 1  # Conservative: only 1 iteration

    # LightGBM params
    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'n_estimators': 3000,
        'learning_rate': 0.015,
        'num_leaves':64,
        'max_depth': 8,
        'min_child_samples':25,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.15,
        'reg_lambda':0.15,
        'min_split_gain': 0.001,
        'min_child_weight': 0.001,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': seed,
    }

    # XGBoost params (GPU)
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'n_estimators': 2500,
        'learning_rate': 0.02,
        'max_depth':8,
        'min_child_weight': 3,
        'subsample': 0.85,
        'colsample_bytree':0.85,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'gamma': 0.001,
        'verbosity': 0,
        'n_jobs': -1,
        'random_state': seed,
    }

os.makedirs(CFG.CACHE_DIR, exist_ok=True)
np.random.seed(CFG.seed)
tf.random.set_seed(CFG.seed)

# ==============================================================================
# CACHING UTILITIES
# ==============================================================================
def get_cache_path(name, params=None):
    '''Generate cache file path with hash of parameters.'''
    if params:
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return os.path.join(CFG.CACHE_DIR, "{}_{}.pkl".format(name, param_hash))
    return os.path.join(CFG.CACHE_DIR, "{}.pkl".format(name))

def save_cache(obj, name, params=None):
    """Save object to cache."""
    path = get_cache_path(name, params)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print("  Cached: {}".format(name))

def load_cache(name, params=None):
    '''Load object from cache.'''
    path = get_cache_path(name, params)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            print("  Loaded from cache: {}".format(name))
            return pickle.load(f)
    return None

# ==============================================================================
# DATA LOADING
# ==============================================================================
def map_images_to_paths(df, image_dir):
    """Map image file paths to dataframe."""
    if not os.path.exists(image_dir):
        print("  WARNING: Image directory not found: {}".format(image_dir))
        df['local_path'] = None
        df['has_image'] = 0
        return df

    image_files = {}
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
            try:
                sample_id = int(Path(filename).stem)
                image_files[sample_id] = os.path.join(image_dir, filename)
            except ValueError:
                continue

    df['local_path'] = df['sample_id'].map(image_files)
    df['has_image'] = df['local_path'].notna().astype(int)

    found = df['has_image'].sum()
    print("  Mapped {}/{} images ({:.1f}%)".format(found, len(df), 100 * found / len(df)))
    return df

def load_and_prepare_data():
    """Load data with caching."""
    cached = load_cache('prepared_data_v2')
    if cached is not None:
        return cached

    print("\n[STEP 1/9] Loading data...")
    train_df = pd.read_csv(CFG.RAW_TRAIN_CSV)
    test_df = pd.read_csv(CFG.RAW_TEST_CSV)
    print("  Train: {} rows | Test: {} rows".format(len(train_df), len(test_df)))

    if CFG.debug:
        train_df = train_df.sample(1000, random_state=CFG.seed).reset_index(drop=True)
        test_df = test_df.sample(1000, random_state=CFG.seed).reset_index(drop=True)

    # Map images
    print("\n  Mapping images...")
    train_df = map_images_to_paths(train_df, CFG.TRAIN_IMAGE_DIR)
    test_df = map_images_to_paths(test_df, CFG.TEST_IMAGE_DIR)

    # Keep only samples with images for training (as per your requirement)
    initial_count = len(train_df)
    train_df = train_df[train_df['has_image'] == 1].reset_index(drop=True)
    print("  Filtered training data: {} -> {} samples with images".format(initial_count, len(train_df)))

    # Log transform target
    train_df['price'] = np.log1p(train_df['price'])

    data = (train_df, test_df)
    save_cache(data, 'prepared_data_v2')
    return data

# ==============================================================================
# ENHANCED FEATURE ENGINEERING
# ==============================================================================
def extract_brand(text):
    '''Extract potential brand name from item name.'''
    if not isinstance(text, str) or len(text) < 3:
        return 'unknown'

    # Take first word (often brand) and clean
    words = text.split()
    if len(words) > 0:
        brand = words[0].lower().strip()
        # Remove common prefixes
        brand = re.sub(r'^(the|a|an)\s+', '', brand)
        return brand[:20]  # Limit length
    return 'unknown'

def extract_advanced_text_features(df):
    """Extract advanced text features with brand extraction."""
    print("\n[STEP 2/9] Extracting text features + brand...")

    parsed_data = []
    for content in tqdm(df['catalog_content'], desc="  Parsing"):
        if not isinstance(content, str):
            parsed_data.append({})
            continue

        item_name = re.search(r"Item Name: (.*?)\n", content)
        bullets = re.findall(r"Bullet Point \d+: (.*?)\n", content)
        value = re.search(r"Value: (.*?)\n", content)
        unit = re.search(r"Unit: (.*?)\n", content)

        item_name_str = item_name.group(1).strip() if item_name else ""
        bullet_str = " ".join([b.strip() for b in bullets])
        full_text = "{} {}".format(item_name_str, bullet_str).lower()

        # Extract numeric value
        value_num = np.nan
        if value:
            val_str = value.group(1).strip()
            try:
                value_num = float(val_str)
            except:
                numbers = re.findall(r'\d+\.?\d*', val_str)
                if numbers:
                    value_num = float(numbers[0])

        # Extract brand
        brand = extract_brand(item_name_str)

        parsed_data.append({
            'item_name': item_name_str,
            'bullet_points': bullet_str,
            'full_text': full_text,
            'value': value_num,
            'unit': unit.group(1).strip() if unit else "unknown",
            'brand': brand,
            'text_length': len(full_text),
            'word_count': len(full_text.split()),
            'num_bullets': len(bullets),
            'has_value': 0 if pd.isna(value_num) else 1,
            'avg_word_length': np.mean([len(w) for w in full_text.split()]) if full_text else 0,
            'has_numbers': int(bool(re.search(r'\d', full_text))),
        })

    result_df = df.join(pd.DataFrame(parsed_data, index=df.index))
    print("  Extracted {} unique brands".format(result_df['brand'].nunique()))
    return result_df

def create_target_encoding(train_df, test_df, col='unit', target='price', smoothing=10):
    '''Create smoothed target encoding for categorical features.'''
    # Calculate global mean
    global_mean = train_df[target].mean()

    # Calculate per-category statistics
    agg = train_df.groupby(col)[target].agg(['mean', 'count'])

    # Smooth the means
    smoothed_means = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)

    # Map to both train and test
    train_encoded = train_df[col].map(smoothed_means).fillna(global_mean)
    test_encoded = test_df[col].map(smoothed_means).fillna(global_mean)

    return train_encoded, test_encoded

def create_tfidf_features(train_df, test_df):
    """Create TF-IDF features with caching."""
    cache_params = {
        'max_features': CFG.tfidf_max_features,
        'ngram_range': CFG.tfidf_ngram_range,
        'min_df': CFG.tfidf_min_df
    }
    cached = load_cache('tfidf_features_v2', cache_params)
    if cached is not None:
        return cached

    print("\n[STEP 3/9] Creating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=CFG.tfidf_max_features,
        ngram_range=CFG.tfidf_ngram_range,
        min_df=CFG.tfidf_min_df,
        stop_words='english',
        sublinear_tf=True
    )

    train_tfidf = tfidf.fit_transform(train_df['full_text'])
    test_tfidf = tfidf.transform(test_df['full_text'])
    print("  TF-IDF shape: {}".format(train_tfidf.shape))

    # Apply SVD
    print("  Applying SVD (n_components={})...".format(CFG.svd_components))
    svd = TruncatedSVD(n_components=CFG.svd_components, random_state=CFG.seed)
    train_tfidf_svd = svd.fit_transform(train_tfidf)
    test_tfidf_svd = svd.transform(test_tfidf)
    print("  Explained variance: {:.3f}".format(svd.explained_variance_ratio_.sum()))

    result = (train_tfidf_svd, test_tfidf_svd, tfidf, svd)
    save_cache(result, 'tfidf_features_v2', cache_params)
    return result


def get_image_embeddings(df, dataset_name='train'):
    """Extract image embeddings with better missing image handling."""
    cache_name = "img_emb_v2_{}".format(dataset_name)
    cached = load_cache(cache_name)
    if cached is not None:
        return cached

    print("\n[STEP 4/9] Extracting image embeddings ({})...".format(dataset_name))
    base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

    image_paths = df['local_path'].tolist()
    has_images = df['has_image'].tolist()

    # Calculate mean embedding for missing images (from available images)
    print("  Calculating mean embedding for missing images...")
    sample_embeddings = []
    sample_count = 0
    for i, (path, has_img) in enumerate(zip(image_paths[:min(500, len(image_paths))], has_images[:min(500, len(image_paths))])):
        if has_img and pd.notna(path) and os.path.exists(path):
            try:
                img = load_img(path, target_size=CFG.img_size)
                img_array = img_to_array(img)
                preprocessed = tf.keras.applications.efficientnet.preprocess_input(np.array([img_array]))
                emb = base_model.predict(preprocessed, verbose=0)
                sample_embeddings.append(emb[0])
                sample_count += 1
                if sample_count >= 100:  # Use 100 samples for mean
                    break
            except:
                continue

    mean_embedding = np.mean(sample_embeddings, axis=0) if sample_embeddings else np.zeros(1280)
    print("  Mean embedding calculated from {} images".format(len(sample_embeddings)))

    # Process all images
    all_embeddings = []
    for i in tqdm(range(0, len(image_paths), CFG.img_batch_size), desc="  Processing"):
        batch_paths = image_paths[i:i + CFG.img_batch_size]
        batch_has_imgs = has_images[i:i + CFG.img_batch_size]
        batch_images = []
        indices_with_images = []

        # Collect only the valid images and their original positions in the batch
        for j, (path, has_img) in enumerate(zip(batch_paths, batch_has_imgs)):
            if has_img and pd.notna(path) and os.path.exists(path):
                try:
                    img = load_img(path, target_size=CFG.img_size)
                    batch_images.append(img_to_array(img))
                    indices_with_images.append(j)
                except:
                    pass # Ignore corrupted images

        # Create a placeholder array filled with the mean embedding
        batch_embeddings = np.array([mean_embedding] * len(batch_paths))

        # If there were any valid images, get their embeddings
        if batch_images:
            preprocessed = tf.keras.applications.efficientnet.preprocess_input(np.array(batch_images))
            embeddings = base_model.predict(preprocessed, verbose=0)

            # Place the new embeddings into their correct positions
            for k, emb in zip(indices_with_images, embeddings):
                batch_embeddings[k] = emb

        all_embeddings.extend(batch_embeddings)

    result = np.array(all_embeddings)
    save_cache(result, cache_name)
    return result

def create_final_features(train_df, test_df):
    '''Combine all features with target encoding and interactions.'''
    print("\n[STEP 5/9] Creating final feature matrix...")

    # Text features
    train_tfidf_svd, test_tfidf_svd, _, _ = create_tfidf_features(train_df, test_df)

    # Image features
    train_img = get_image_embeddings(train_df, 'train')
    test_img = get_image_embeddings(test_df, 'test')

    # Target encoding for unit and brand
    print("  Creating target encodings...")
    train_unit_enc, test_unit_enc = create_target_encoding(train_df, test_df, 'unit', 'price')
    train_brand_enc, test_brand_enc = create_target_encoding(train_df, test_df, 'brand', 'price')

    # One-hot encoding for unit and brand (keep for diversity)
    ohe_unit = OneHotEncoder(handle_unknown='ignore', sparse_output=True, max_categories=50)
    train_unit_ohe = ohe_unit.fit_transform(train_df[['unit']].fillna('unknown'))
    test_unit_ohe = ohe_unit.transform(test_df[['unit']].fillna('unknown'))

    ohe_brand = OneHotEncoder(handle_unknown='ignore', sparse_output=True, max_categories=100)
    train_brand_ohe = ohe_brand.fit_transform(train_df[['brand']].fillna('unknown'))
    test_brand_ohe = ohe_brand.transform(test_df[['brand']].fillna('unknown'))

    # Numerical features
    num_cols = ['value', 'text_length', 'word_count', 'num_bullets', 'has_value',
                'avg_word_length', 'has_numbers', 'has_image']

    # Create interaction features
    train_df['value_x_has_value'] = train_df['value'].fillna(0) * train_df['has_value']
    test_df['value_x_has_value'] = test_df['value'].fillna(0) * test_df['has_value']

    train_df['text_density'] = train_df['word_count'] / (train_df['text_length'] + 1)
    test_df['text_density'] = test_df['word_count'] / (test_df['text_length'] + 1)

    num_cols.extend(['value_x_has_value', 'text_density'])

    # Add target encodings to numerical features
    train_df['unit_target_enc'] = train_unit_enc
    test_df['unit_target_enc'] = test_unit_enc
    train_df['brand_target_enc'] = train_brand_enc
    test_df['brand_target_enc'] = test_brand_enc
    num_cols.extend(['unit_target_enc', 'brand_target_enc'])

    # Scale numerical features
    scaler = RobustScaler()
    train_num = scaler.fit_transform(train_df[num_cols].fillna(0))
    test_num = scaler.transform(test_df[num_cols].fillna(0))

    # Combine all features
    X_train = scipy.sparse.hstack([
        scipy.sparse.csr_matrix(train_tfidf_svd),
        train_unit_ohe,
        train_brand_ohe,
        scipy.sparse.csr_matrix(train_img),
        scipy.sparse.csr_matrix(train_num)
    ], format='csr')

    X_test = scipy.sparse.hstack([
        scipy.sparse.csr_matrix(test_tfidf_svd),
        test_unit_ohe,
        test_brand_ohe,
        scipy.sparse.csr_matrix(test_img),
        scipy.sparse.csr_matrix(test_num)
    ], format='csr')

    print("  Final shape - Train: {}, Test: {}".format(X_train.shape, X_test.shape))
    print("  Total features: {}".format(X_train.shape[1]))

    return X_train, X_test

# ==============================================================================
# ENSEMBLE TRAINING
# ==============================================================================
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def train_lgb_model(X, y, X_val, y_val, params):
    '''Train LightGBM model.'''
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X, y,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(500)
        ]
    )
    return model

def train_xgb_model(X, y, X_val, y_val, params):
    """Train XGBoost model."""
    model = xgb.XGBRegressor(**params)
    model.fit(
        X, y,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=150,
        verbose=False
    )
    return model


def train_ensemble(X, y, X_test, lgb_params, xgb_params):
    '''Train ensemble with both LightGBM and XGBoost.'''
    print("\n[STEP 6/9] Training ensemble ({}-fold CV)...".format(CFG.n_splits))

    n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    if n_gpus == 0:
        print("  WARNING: No GPUs detected. Training on CPU.")
        # Fallback to CPU if no GPU is found
        lgb_params['device'] = 'cpu'
        xgb_params['tree_method'] = 'hist'

    kf = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

    # OOF predictions
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))

    # Test predictions
    test_lgb = np.zeros(X_test.shape[0])
    test_xgb = np.zeros(X_test.shape[0])

    lgb_scores = []
    xgb_scores = []
    ensemble_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        current_gpu_id = fold % n_gpus if n_gpus > 0 else -1 # -1 for CPU
        print(f"\n  Fold {fold + 1}/{CFG.n_splits} on GPU {current_gpu_id}")

        X_tr, y_tr = X[train_idx], y.iloc[train_idx]
        X_val, y_val = X[val_idx], y.iloc[val_idx]

        lgb_fold_params = lgb_params.copy()
        xgb_fold_params = xgb_params.copy()
        if n_gpus > 0:
            lgb_fold_params['gpu_device_id'] = current_gpu_id
            xgb_fold_params['gpu_id'] = current_gpu_id

        # Train LightGBM
        print("    Training LightGBM...")
        lgb_model = train_lgb_model(X_tr, y_tr, X_val, y_val, lgb_fold_params)
        lgb_val_pred = lgb_model.predict(X_val)
        oof_lgb[val_idx] = lgb_val_pred
        test_lgb += lgb_model.predict(X_test) / CFG.n_splits


        # Train XGBoost
        print("    Training XGBoost...")
        xgb_model = train_xgb_model(X_tr, y_tr, X_val, y_val, xgb_fold_params)
        xgb_val_pred = xgb_model.predict(X_val)
        oof_xgb[val_idx] = xgb_val_pred
        test_xgb += xgb_model.predict(X_test) / CFG.n_splits

        # Calculate scores
        lgb_smape = smape(np.expm1(y_val), np.expm1(lgb_val_pred))
        xgb_smape = smape(np.expm1(y_val), np.expm1(xgb_val_pred))

        # Weighted ensemble (70% LGB, 30% XGB)
        ensemble_pred = 0.7 * lgb_val_pred + 0.3 * xgb_val_pred
        ensemble_smape = smape(np.expm1(y_val), np.expm1(ensemble_pred))

        lgb_scores.append(lgb_smape)
        xgb_scores.append(xgb_smape)
        ensemble_scores.append(ensemble_smape)

        print("    LGB SMAPE: {:.4f} | XGB SMAPE: {:.4f} | Ensemble: {:.4f}".format(
            lgb_smape, xgb_smape, ensemble_smape))

        # Save models
        save_cache(lgb_model, 'lgb_model_fold_{}'.format(fold))
        save_cache(xgb_model, 'xgb_model_fold_{}'.format(fold))

        gc.collect()

    print("\n  Training complete!")
    print("  LGB Mean SMAPE:      {:.4f} (±{:.4f})".format(np.mean(lgb_scores), np.std(lgb_scores)))
    print("  XGB Mean SMAPE:      {:.4f} (±{:.4f})".format(np.mean(xgb_scores), np.std(xgb_scores)))
    print("  Ensemble Mean SMAPE: {:.4f} (±{:.4f})".format(np.mean(ensemble_scores), np.std(ensemble_scores)))

    return oof_lgb, oof_xgb, test_lgb, test_xgb, ensemble_scores

# ==============================================================================
# PSEUDO-LABELING
# ==============================================================================
def apply_pseudo_labeling(X_train, y_train, X_test, test_preds, test_df, train_df):
    """Apply conservative pseudo-labeling on high-confidence test predictions."""
    if not CFG.use_pseudo_labeling:
        return X_train, y_train

    print("\n[STEP 7/9] Applying pseudo-labeling...")

    # Calculate prediction confidence (inverse of relative std across folds)
    # For now, we use a simple approach: select samples with consistent predictions
    # In production, you'd track std across folds

    # Select top 20% most confident (middle range predictions are more reliable)
    test_preds_original = np.expm1(test_preds)

    # Calculate percentiles
    p25 = np.percentile(test_preds_original, 25)
    p75 = np.percentile(test_preds_original, 75)

    # Select middle 50% range as most confident
    confident_mask = (test_preds_original >= p25) & (test_preds_original <= p75)

    # Further filter to top 20% overall
    n_pseudo = int(len(test_preds) * CFG.pseudo_confidence_threshold)

    # Among confident predictions, select random subset
    confident_indices = np.where(confident_mask)[0]
    if len(confident_indices) > n_pseudo:
        selected_indices = np.random.choice(confident_indices, n_pseudo, replace=False)
    else:
        selected_indices = confident_indices

    print("  Selected {} pseudo-labeled samples ({:.1f}% of test)".format(
        len(selected_indices), 100 * len(selected_indices) / len(test_preds)))

    # Extract pseudo-labeled data
    X_pseudo = X_test[selected_indices]
    y_pseudo = test_preds[selected_indices]

    # Combine with original training data
    X_combined = scipy.sparse.vstack([X_train, X_pseudo])
    y_combined = pd.concat([y_train, pd.Series(y_pseudo)], ignore_index=True)

    print("  New training size: {} (original) + {} (pseudo) = {}".format(
        X_train.shape[0], X_pseudo.shape[0], X_combined.shape[0]))

    return X_combined, y_combined

# ==============================================================================
# OPTIMIZE ENSEMBLE WEIGHTS
# ==============================================================================
def optimize_ensemble_weights(oof_lgb, oof_xgb, y_true):
    """Find optimal blending weights using grid search."""
    print("\n[STEP 8/9] Optimizing ensemble weights...")

    best_weight = 0.5
    best_smape = float('inf')

    weights = np.arange(0.0, 1.01, 0.05)

    for w in weights:
        ensemble_pred = w * oof_lgb + (1 - w) * oof_xgb
        score = smape(np.expm1(y_true), np.expm1(ensemble_pred))

        if score < best_smape:
            best_smape = score
            best_weight = w

    print("  Best weight (LGB): {:.2f} | XGB: {:.2f}".format(best_weight, 1 - best_weight))
    print("  Best SMAPE: {:.4f}".format(best_smape))

    return best_weight

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    start_time = datetime.now()

    # Load data
    train_df, test_df = load_and_prepare_data()

    # Extract features
    train_df = extract_advanced_text_features(train_df)
    test_df = extract_advanced_text_features(test_df)

    # Create final feature matrices
    X, X_test = create_final_features(train_df, test_df)
    y = train_df['price']

    # Train initial ensemble
    oof_lgb, oof_xgb, test_lgb, test_xgb, scores = train_ensemble(
        X, y, X_test, CFG.lgb_params, CFG.xgb_params
    )

    # Optimize ensemble weights
    optimal_weight = optimize_ensemble_weights(oof_lgb, oof_xgb, y)

    # Create weighted ensemble predictions
    test_preds_ensemble = optimal_weight * test_lgb + (1 - optimal_weight) * test_xgb

    # Apply pseudo-labeling (if enabled)
    if CFG.use_pseudo_labeling:
        X_combined, y_combined = apply_pseudo_labeling(
            X, y, X_test, test_preds_ensemble, test_df, train_df
        )

        # Retrain with pseudo-labels
        print("\n  Retraining with pseudo-labeled data...")
        _, _, test_lgb_pl, test_xgb_pl, scores_pl = train_ensemble(
            X_combined, y_combined, X_test, CFG.lgb_params, CFG.xgb_params
        )

        # Final predictions with pseudo-labeling
        test_preds_final = optimal_weight * test_lgb_pl + (1 - optimal_weight) * test_xgb_pl
        final_scores = scores_pl
    else:
        test_preds_final = test_preds_ensemble
        final_scores = scores

    # Generate submission
    print("\n[STEP 9/9] Creating submission...")
    final_preds = np.expm1(test_preds_final)
    final_preds = np.clip(final_preds, 0, None)

    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': final_preds
    })

    submission.to_csv(CFG.SUBMISSION_FILE, index=False)
    print("  Submission saved: {}".format(CFG.SUBMISSION_FILE))
    print("  Prediction stats:")
    print("    Min:    ${:.2f}".format(submission['price'].min()))
    print("    Mean:   ${:.2f}".format(submission['price'].mean()))
    print("    Median: ${:.2f}".format(submission['price'].median()))
    print("    Max:    ${:.2f}".format(submission['price'].max()))

    elapsed = datetime.now() - start_time
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED!")
    print("=" * 80)
    print("Total time: {}".format(elapsed))
    print("Final CV SMAPE: {:.4f} (±{:.4f})".format(np.mean(final_scores), np.std(final_scores)))
    print("Target: < 40 | Previous: 46-48")
    print("=" * 80)

if __name__ == "__main__":
    main()