import os
import json
import time
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import io
import base64
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

ALLOWED_EXTENSIONS = {'csv', 'tsv', 'txt'}

# In-memory state
state = {
    'df': None,
    'filename': None,
    'model': None,
    'model_type': None,
    'task_type': None,
    'target': None,
    'features': None,
    'metrics': None,
    'encoders': {},
    'scaler': None,
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0f172a')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


def detect_task_type(series):
    n_unique = series.nunique()
    if series.dtype == object or n_unique <= 10:
        return 'classification'
    return 'regression'


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV/TSV files allowed'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    t0 = time.time()
    sep = '\t' if filename.endswith('.tsv') else ','
    try:
        df = pd.read_csv(filepath, sep=sep, low_memory=False)
    except Exception as e:
        return jsonify({'error': f'Could not parse file: {str(e)}'}), 400

    load_time = round(time.time() - t0, 3)
    state['df'] = df
    state['filename'] = filename

    # Basic info
    info = {
        'filename': filename,
        'rows': len(df),
        'cols': len(df.columns),
        'size_mb': round(os.path.getsize(filepath) / 1024 / 1024, 2),
        'load_time_s': load_time,
        'columns': list(df.columns),
        'dtypes': {c: str(df[c].dtype) for c in df.columns},
        'missing': df.isnull().sum().to_dict(),
        'preview': df.head(10).fillna('').to_dict(orient='records'),
        'stats': df.describe(include='all').fillna('').to_dict(),
    }
    return jsonify(info)


@app.route('/api/visualize', methods=['POST'])
def visualize():
    df = state.get('df')
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400

    body = request.get_json() or {}
    chart_type = body.get('chart_type', 'histogram')
    col_x = body.get('col_x')
    col_y = body.get('col_y')

    plt.style.use('dark_background')
    ACCENT = '#6366f1'
    BG = '#0f172a'
    GRID = '#1e293b'

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(GRID)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    try:
        if chart_type == 'histogram' and col_x:
            series = df[col_x].dropna()
            if series.dtype == object:
                vc = series.value_counts().head(20)
                ax.bar(vc.index, vc.values, color=ACCENT, edgecolor='#0f172a')
                ax.set_xlabel(col_x, color='#94a3b8')
                ax.set_ylabel('Count', color='#94a3b8')
                ax.tick_params(axis='x', rotation=45, colors='#94a3b8')
            else:
                ax.hist(series, bins=40, color=ACCENT, edgecolor='#0f172a', alpha=0.85)
                ax.set_xlabel(col_x, color='#94a3b8')
                ax.set_ylabel('Frequency', color='#94a3b8')
            ax.set_title(f'Distribution: {col_x}', color='#e2e8f0', fontsize=13)

        elif chart_type == 'scatter' and col_x and col_y:
            x = pd.to_numeric(df[col_x], errors='coerce').dropna()
            y = pd.to_numeric(df[col_y], errors='coerce').dropna()
            min_len = min(len(x), len(y), 5000)
            ax.scatter(x[:min_len], y[:min_len], alpha=0.4, s=10,
                       color=ACCENT, edgecolors='none')
            ax.set_xlabel(col_x, color='#94a3b8')
            ax.set_ylabel(col_y, color='#94a3b8')
            ax.set_title(f'{col_x} vs {col_y}', color='#e2e8f0', fontsize=13)

        elif chart_type == 'correlation':
            num_df = df.select_dtypes(include='number').dropna(axis=1, how='all')
            if len(num_df.columns) < 2:
                return jsonify({'error': 'Need ≥2 numeric columns for correlation'}), 400
            corr = num_df.corr()
            mask = np.zeros_like(corr, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            sns.heatmap(corr, mask=mask, ax=ax, cmap='coolwarm', center=0,
                        annot=len(corr) <= 12, fmt='.2f',
                        linewidths=0.5, linecolor='#0f172a',
                        cbar_kws={'shrink': 0.8})
            ax.set_title('Correlation Matrix', color='#e2e8f0', fontsize=13)

        elif chart_type == 'boxplot' and col_x:
            series = pd.to_numeric(df[col_x], errors='coerce').dropna()
            bp = ax.boxplot(series, patch_artist=True,
                            medianprops=dict(color='#f59e0b', linewidth=2))
            for patch in bp['boxes']:
                patch.set_facecolor(ACCENT)
                patch.set_alpha(0.7)
            ax.set_title(f'Boxplot: {col_x}', color='#e2e8f0', fontsize=13)
            ax.set_ylabel(col_x, color='#94a3b8')

        elif chart_type == 'missing':
            miss = df.isnull().mean().sort_values(ascending=False)
            miss = miss[miss > 0].head(30)
            if miss.empty:
                ax.text(0.5, 0.5, '✓ No missing values!', ha='center', va='center',
                        color='#10b981', fontsize=16, transform=ax.transAxes)
            else:
                ax.barh(miss.index, miss.values * 100, color='#ef4444', alpha=0.8)
                ax.set_xlabel('Missing %', color='#94a3b8')
                ax.set_title('Missing Values by Column', color='#e2e8f0', fontsize=13)
            ax.tick_params(colors='#94a3b8')

        else:
            return jsonify({'error': 'Invalid chart configuration'}), 400

        ax.tick_params(colors='#94a3b8')
        fig.tight_layout()
        return jsonify({'image': fig_to_base64(fig)})

    except Exception as e:
        plt.close(fig)
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train():
    df = state.get('df')
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400

    body = request.get_json() or {}
    target = body.get('target')
    features = body.get('features', [])
    model_name = body.get('model', 'auto')
    test_size = float(body.get('test_size', 0.2))

    if not target or target not in df.columns:
        return jsonify({'error': 'Invalid target column'}), 400
    if not features:
        features = [c for c in df.columns if c != target]

    # Prepare data
    df_work = df[features + [target]].copy().dropna()
    if len(df_work) < 20:
        return jsonify({'error': 'Not enough rows after dropping NaN'}), 400

    # Encode categoricals in features
    encoders = {}
    for col in features:
        if df_work[col].dtype == object:
            le = LabelEncoder()
            df_work[col] = le.fit_transform(df_work[col].astype(str))
            encoders[col] = le

    # Encode target if classification
    task_type = detect_task_type(df_work[target])
    target_encoder = None
    if task_type == 'classification' and df_work[target].dtype == object:
        le = LabelEncoder()
        df_work[target] = le.fit_transform(df_work[target].astype(str))
        target_encoder = le

    X = df_work[features].values
    y = df_work[target].values

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Choose model
    if task_type == 'classification':
        if model_name == 'logistic' or model_name == 'auto':
            model = LogisticRegression(max_iter=500, random_state=42)
            model_label = 'Logistic Regression'
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model_label = 'Random Forest Classifier'
    else:
        if model_name == 'linear' or model_name == 'auto':
            model = LinearRegression()
            model_label = 'Linear Regression'
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model_label = 'Random Forest Regressor'

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 3)

    y_pred = model.predict(X_test)

    # Metrics
    if task_type == 'regression':
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            'task': 'regression',
            'model': model_label,
            'r2': round(r2_score(y_test, y_pred), 4),
            'rmse': round(np.sqrt(mse), 4),
            'mse': round(mse, 4),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_time_s': train_time,
        }
    else:
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics = {
            'task': 'classification',
            'model': model_label,
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'f1_macro': round(report.get('macro avg', {}).get('f1-score', 0), 4),
            'precision': round(report.get('macro avg', {}).get('precision', 0), 4),
            'recall': round(report.get('macro avg', {}).get('recall', 0), 4),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_time_s': train_time,
        }

    # Save state
    state['model'] = model
    state['model_type'] = model_label
    state['task_type'] = task_type
    state['target'] = target
    state['features'] = features
    state['metrics'] = metrics
    state['encoders'] = encoders
    state['scaler'] = scaler
    state['X_test'] = X_test
    state['y_test'] = y_test
    state['y_pred'] = y_pred

    # Generate result plot
    plt.style.use('dark_background')
    BG, GRID = '#0f172a', '#1e293b'

    if task_type == 'regression':
        fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
        ax.set_facecolor(GRID)
        ax.scatter(y_test[:500], y_pred[:500], alpha=0.5, s=15, color='#6366f1')
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
        ax.set_xlabel('Actual', color='#94a3b8')
        ax.set_ylabel('Predicted', color='#94a3b8')
        ax.set_title(f'Actual vs Predicted  (R²={metrics["r2"]})', color='#e2e8f0', fontsize=12)
        ax.legend(facecolor=GRID, labelcolor='#94a3b8')
        ax.tick_params(colors='#94a3b8')
        for sp in ax.spines.values():
            sp.set_edgecolor('#334155')
    else:
        cm_arr = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
        ax.set_facecolor(GRID)
        sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', ax=ax,
                    linewidths=0.5, linecolor='#0f172a',
                    cbar_kws={'shrink': 0.8})
        ax.set_xlabel('Predicted', color='#94a3b8')
        ax.set_ylabel('Actual', color='#94a3b8')
        ax.set_title(f'Confusion Matrix  (Acc={metrics["accuracy"]})', color='#e2e8f0', fontsize=12)
        ax.tick_params(colors='#94a3b8')

    fig.tight_layout()
    metrics['plot'] = fig_to_base64(fig)

    # Feature importance (if RF)
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(15)
        fig2, ax2 = plt.subplots(figsize=(8, 4), facecolor=BG)
        ax2.set_facecolor(GRID)
        colors = cm.plasma(np.linspace(0.3, 0.9, len(imp)))
        ax2.barh(imp.index[::-1], imp.values[::-1], color=colors[::-1])
        ax2.set_xlabel('Importance', color='#94a3b8')
        ax2.set_title('Feature Importances', color='#e2e8f0', fontsize=12)
        ax2.tick_params(colors='#94a3b8')
        for sp in ax2.spines.values():
            sp.set_edgecolor('#334155')
        fig2.tight_layout()
        metrics['importance_plot'] = fig_to_base64(fig2)

    return jsonify(metrics)


@app.route('/api/status')
def status():
    df = state.get('df')
    return jsonify({
        'data_loaded': df is not None,
        'rows': len(df) if df is not None else 0,
        'model_trained': state['model'] is not None,
        'model_type': state.get('model_type'),
        'metrics': state.get('metrics'),
    })


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
