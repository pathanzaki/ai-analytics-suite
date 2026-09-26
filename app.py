"""
AI Analytics Suite - Complete Backend Application
Full-featured analytics platform with user authentication,
file processing, analytics, AI insights, favorites and sharing.
"""

import os
import json
import math
from datetime import datetime

import pandas as pd
import numpy as np

from werkzeug.utils import secure_filename

from flask import (
    Flask,
    jsonify,
    request,
    send_file,
    redirect,
    url_for
)

from flask_cors import CORS

from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user
)

from dotenv import load_dotenv

from config import config, Config
from models import (
    db,
    User,
    Analysis,
    Share,
    Favorite,
    AuditLog
)


# ============================================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================================

load_dotenv()


# ============================================================================
# INITIALIZE FLASK APP
# ============================================================================

app = Flask(__name__)

config_name = os.getenv("FLASK_ENV", "development")

if config_name not in config:
    config_name = "development"

app.config.from_object(config[config_name])


# ============================================================================
# INITIALIZE EXTENSIONS
# ============================================================================

db.init_app(app)

CORS(
    app,
    supports_credentials=True
)


# ============================================================================
# INITIALIZE LOGIN MANAGER
# ============================================================================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login_page"


@login_manager.user_loader
def load_user(user_id):
    """
    Load user from database for Flask-Login session.
    """
    try:
        return User.query.get(int(user_id))
    except (ValueError, TypeError):
        return None


@login_manager.unauthorized_handler
def unauthorized():
    """
    Handle unauthenticated users.

    API requests receive JSON.
    Browser page requests are redirected to login.
    """

    if request.path.startswith("/api/"):
        return jsonify({
            "error": "Authentication required",
            "message": "Please login first."
        }), 401

    return redirect(
        url_for(
            "login_page",
            next=request.path
        )
    )


# ============================================================================
# CREATE UPLOAD FOLDER
# ============================================================================

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """
    Check whether uploaded file extension is allowed.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


def clean_value(value):
    """
    Convert NumPy/Pandas values into JSON-safe Python values.
    """

    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    if pd.isna(value):
        return None

    return value


def clean_dict(data):
    """
    Recursively clean dictionary values for JSON serialization.
    """

    if isinstance(data, dict):
        return {
            str(key): clean_dict(value)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            clean_dict(value)
            for value in data
        ]

    return clean_value(data)


# ============================================================================
# STATISTICS
# ============================================================================

def calculate_statistics(df):
    """
    Calculate descriptive statistics for numeric columns.
    """

    stats = {}

    if df.empty:
        return stats

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    for col in numeric_cols:

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(series) == 0:
            continue

        stats[col] = {
            "mean": clean_value(series.mean()),
            "median": clean_value(series.median()),
            "std": clean_value(series.std()),
            "min": clean_value(series.min()),
            "max": clean_value(series.max()),
            "q25": clean_value(series.quantile(0.25)),
            "q75": clean_value(series.quantile(0.75))
        }

    return stats


# ============================================================================
# ANOMALY DETECTION
# ============================================================================

def detect_anomalies(df):
    """
    Detect anomalies using the IQR method.
    """

    anomalies = {}

    if df.empty:
        return anomalies

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    total_rows = len(df)

    if total_rows == 0:
        return anomalies

    for col in numeric_cols:

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        anomaly_count = int(
            ((series < lower_bound) |
             (series > upper_bound)).sum()
        )

        if anomaly_count > 0:

            anomalies[col] = {
                "count": anomaly_count,
                "percentage": clean_value(
                    (anomaly_count / total_rows) * 100
                ),
                "lower_bound": clean_value(
                    lower_bound
                ),
                "upper_bound": clean_value(
                    upper_bound
                )
            }

    return anomalies


# ============================================================================
# CORRELATIONS
# ============================================================================

def calculate_correlations(df):
    """
    Calculate correlations between numeric columns.
    """

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    if numeric_df.shape[1] < 2:
        return {}

    correlations = numeric_df.corr().to_dict()

    return clean_dict(correlations)


# ============================================================================
# DATA QUALITY SCORE
# ============================================================================

def calculate_quality_score(df):
    """
    Calculate data quality score.

    Completeness = 60%
    Duplicate-free = 20%
    Consistency = 20%
    """

    if df.empty or len(df.columns) == 0:
        return 0.0

    total_cells = len(df) * len(df.columns)

    missing_cells = int(
        df.isnull().sum().sum()
    )

    completeness = 1 - (
        missing_cells / total_cells
    )

    if len(df) > 0:
        duplicates = 1 - (
            df.duplicated().sum() / len(df)
        )
    else:
        duplicates = 1

    # Default consistency score.
    consistency = 0.90

    quality = (
        (completeness * 0.60)
        + (duplicates * 0.20)
        + (consistency * 0.20)
    )

    return round(
        min(max(quality * 100, 0), 100),
        2
    )


# ============================================================================
# AI INSIGHTS
# ============================================================================

def generate_ai_insights(df, filename):
    """
    Generate rule-based AI-style insights.
    """

    insights = []

    quality_score = calculate_quality_score(df)

    insights.append(
        f"Dataset '{filename}' contains "
        f"{len(df)} rows and "
        f"{len(df.columns)} columns."
    )

    insights.append(
        f"Data quality score is "
        f"{quality_score:.1f}%."
    )

    missing_total = int(
        df.isnull().sum().sum()
    )

    if missing_total > 0:

        insights.append(
            f"Found {missing_total} missing values."
        )

    else:

        insights.append(
            "No missing values were detected."
        )

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    if len(numeric_cols) > 0:

        insights.append(
            f"Contains {len(numeric_cols)} "
            f"numeric column(s)."
        )

    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns

    if len(categorical_cols) > 0:

        insights.append(
            f"Found {len(categorical_cols)} "
            f"categorical column(s)."
        )

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count > 0:

        insights.append(
            f"Detected {duplicate_count} "
            f"duplicate row(s)."
        )

    else:

        insights.append(
            "No duplicate rows were detected."
        )

    anomalies = detect_anomalies(df)

    if anomalies:

        insights.append(
            f"Detected anomalies in "
            f"{len(anomalies)} column(s)."
        )

    else:

        insights.append(
            "No significant IQR-based anomalies "
            "were detected."
        )

    if len(numeric_cols) >= 2:

        insights.append(
            "Correlation analysis is available "
            "for the numeric columns."
        )

    return insights


# ============================================================================
# AUDIT LOG
# ============================================================================

def log_audit(
    action,
    resource_type=None,
    resource_id=None,
    details=None
):
    """
    Log user action.
    """

    if not current_user.is_authenticated:
        return

    try:

        log = AuditLog(
            user_id=current_user.id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=request.remote_addr
        )

        db.session.add(log)
        db.session.commit()

    except Exception as e:

        db.session.rollback()

        print(
            f"Audit log error: {e}"
        )


# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@app.route(
    "/api/auth/signup",
    methods=["POST"]
)
def signup():
    """
    User registration.
    """

    try:

        data = request.get_json(
            silent=True
        ) or {}

        username = str(
            data.get("username", "")
        ).strip()

        email = str(
            data.get("email", "")
        ).strip()

        password = str(
            data.get("password", "")
        )

        first_name = str(
            data.get("first_name", "")
        ).strip()

        last_name = str(
            data.get("last_name", "")
        ).strip()

        if not username or not email or not password:

            return jsonify({
                "error": "Missing required fields"
            }), 400

        if len(password) < 6:

            return jsonify({
                "error": "Password must be at least 6 characters"
            }), 400

        existing_username = User.query.filter_by(
            username=username
        ).first()

        if existing_username:

            return jsonify({
                "error": "Username already exists"
            }), 409

        existing_email = User.query.filter_by(
            email=email
        ).first()

        if existing_email:

            return jsonify({
                "error": "Email already exists"
            }), 409

        user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name
        )

        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        return jsonify({
            "message": "User created successfully",
            "user": user.to_dict()
        }), 201

    except Exception as e:

        db.session.rollback()

        return jsonify({
            "error": str(e)
        }), 500


@app.route(
    "/api/auth/login",
    methods=["POST"]
)
def login():
    """
    User login using Flask-Login session.
    """

    try:

        data = request.get_json(
            silent=True
        ) or {}

        username = str(
            data.get("username", "")
        ).strip()

        password = str(
            data.get("password", "")
        )

        if not username or not password:

            return jsonify({
                "error": "Missing username or password"
            }), 400

        user = User.query.filter_by(
            username=username
        ).first()

        if not user:

            return jsonify({
                "error": "Invalid credentials"
            }), 401

        if not user.check_password(password):

            return jsonify({
                "error": "Invalid credentials"
            }), 401

        if not user.is_active:

            return jsonify({
                "error": "User account is inactive"
            }), 403

        # Create Flask-Login session.
        login_user(
            user,
            remember=False
        )

        user.last_login = datetime.utcnow()

        db.session.commit()

        log_audit(
            "user_login"
        )

        return jsonify({
            "message": "Login successful",
            "user": user.to_dict()
        }), 200

    except Exception as e:

        db.session.rollback()

        return jsonify({
            "error": str(e)
        }), 500


@app.route(
    "/api/auth/logout",
    methods=["POST"]
)
@login_required
def logout():
    """
    Logout current user.
    """

    log_audit(
        "user_logout"
    )

    logout_user()

    return jsonify({
        "message": "Logged out successfully"
    }), 200


@app.route(
    "/api/auth/me",
    methods=["GET"]
)
@login_required
def get_current_user():
    """
    Return currently authenticated user.
    """

    return jsonify(
        current_user.to_dict()
    ), 200


# ============================================================================
# FILE UPLOAD & ANALYSIS
# ============================================================================

@app.route(
    "/api/upload",
    methods=["POST"]
)
@login_required
def upload_file():
    """
    Upload CSV/Excel and perform analytics.
    """

    try:

        if "file" not in request.files:

            return jsonify({
                "error": "No file provided"
            }), 400

        file = request.files["file"]

        if not file.filename:

            return jsonify({
                "error": "No file selected"
            }), 400

        if not allowed_file(file.filename):

            return jsonify({
                "error":
                    "File type not allowed. "
                    f"Allowed: {', '.join(app.config['ALLOWED_EXTENSIONS'])}"
            }), 400

        original_filename = secure_filename(
            file.filename
        )

        timestamp = datetime.utcnow().strftime(
            "%Y%m%d_%H%M%S_"
        )

        filename = timestamp + original_filename

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        file_ext = filename.rsplit(
            ".",
            1
        )[1].lower()

        # --------------------------------------------------------------
        # READ FILE
        # --------------------------------------------------------------

        if file_ext == "csv":

            df = pd.read_csv(
                filepath
            )

        elif file_ext in ["xlsx", "xls"]:

            df = pd.read_excel(
                filepath
            )

        else:

            return jsonify({
                "error": "Unsupported file format"
            }), 400

        # --------------------------------------------------------------
        # ANALYTICS
        # --------------------------------------------------------------

        statistics = calculate_statistics(
            df
        )

        anomalies = detect_anomalies(
            df
        )

        correlations = calculate_correlations(
            df
        )

        quality_score = calculate_quality_score(
            df
        )

        insights = generate_ai_insights(
            df,
            original_filename
        )

        # --------------------------------------------------------------
        # CREATE ANALYSIS
        # --------------------------------------------------------------

        analysis = Analysis(
            user_id=current_user.id,

            filename=original_filename,

            file_path=filepath,

            file_size=os.path.getsize(
                filepath
            ),

            file_type=file_ext,

            rows=len(df),

            columns=len(df.columns),

            column_names=[
                str(col)
                for col in df.columns.tolist()
            ],

            data_types={
                str(col): str(df[col].dtype)
                for col in df.columns
            },

            statistics=clean_dict(
                statistics
            ),

            anomalies=clean_dict(
                anomalies
            ),

            correlations=clean_dict(
                correlations
            ),

            quality_score=quality_score,

            insights=insights,

            status="completed"
        )

        db.session.add(
            analysis
        )

        db.session.commit()

        log_audit(
            "file_upload",
            "analysis",
            analysis.id
        )

        return jsonify({
            "message": "File uploaded successfully",
            "analysis": analysis.to_dict()
        }), 201

    except Exception as e:

        db.session.rollback()

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================================
# GET ALL ANALYSES
# ============================================================================

@app.route(
    "/api/analyses",
    methods=["GET"]
)
@login_required
def get_analyses():
    """
    Get current user's analyses.
    """

    try:

        page = request.args.get(
            "page",
            1,
            type=int
        )

        if page < 1:
            page = 1

        per_page = app.config.get(
            "ITEMS_PER_PAGE",
            10
        )

        pagination = (
            Analysis.query
            .filter_by(
                user_id=current_user.id
            )
            .order_by(
                Analysis.created_at.desc()
            )
            .paginate(
                page=page,
                per_page=per_page
            )
        )

        return jsonify({
            "total": pagination.total,
            "pages": pagination.pages,
            "current_page": page,
            "analyses": [
                analysis.to_dict()
                for analysis in pagination.items
            ]
        }), 200

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================================
# GET SINGLE ANALYSIS
# ============================================================================

@app.route(
    "/api/analyses/<int:analysis_id>",
    methods=["GET"]
)
@login_required
def get_analysis(analysis_id):
    """
    Get one analysis.
    """

    try:

        analysis = Analysis.query.get(
            analysis_id
        )

        if not analysis:

            return jsonify({
                "error": "Analysis not found"
            }), 404

        if analysis.user_id != current_user.id:

            return jsonify({
                "error": "Unauthorized"
            }), 403

        return jsonify(
            analysis.to_dict()
        ), 200

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================================
# DOWNLOAD ANALYSIS FILE
# ============================================================================

@app.route(
    "/api/analyses/<int:analysis_id>/download",
    methods=["GET"]
)
@login_required
def download_analysis(analysis_id):
    """
    Download original uploaded file.
    """

    try:

        analysis = Analysis.query.get(
            analysis_id
        )

        if not analysis:

            return jsonify({
                "error": "Analysis not found"
            }), 404

        if analysis.user_id != current_user.id:

            return jsonify({
                "error": "Unauthorized"
            }), 403

        if not os.path.exists(
            analysis.file_path
        ):

            return jsonify({
                "error": "File not found"
            }), 404

        log_audit(
            "file_download",
            "analysis",
            analysis_id
        )

        return send_file(
            analysis.file_path,
            as_attachment=True,
            download_name=analysis.filename
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================================
# FAVORITE ANALYSIS
# ============================================================================

@app.route(
    "/api/analyses/<int:analysis_id>/favorite",
    methods=["POST"]
)
@login_required
def toggle_favorite(analysis_id):
    """
    Add/remove analysis from favorites.
    """

    try:

        analysis = Analysis.query.get(
            analysis_id
        )

        if not analysis:

            return jsonify({
                "error": "Analysis not found"
            }), 404

        if analysis.user_id != current_user.id:

            return jsonify({
                "error": "Unauthorized"
            }), 403

        favorite = Favorite.query.filter_by(
            user_id=current_user.id,
            analysis_id=analysis_id
        ).first()

        if favorite:

            db.session.delete(
                favorite
            )

            db.session.commit()

            log_audit(
                "favorite_removed",
                "analysis",
                analysis_id
            )

            return jsonify({
                "message": "Removed from favorites",
                "favorite": False
            }), 200

        else:

            favorite = Favorite(
                user_id=current_user.id,
                analysis_id=analysis_id
            )

            db.session.add(
                favorite
            )

            db.session.commit()

            log_audit(
                "favorite_added",
                "analysis",
                analysis_id
            )

            return jsonify({
                "message": "Added to favorites",
                "favorite": True
            }), 201

    except Exception as e:

        db.session.rollback()

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================================
# SHARE ANALYSIS
# ============================================================================

@app.route(
    "/api/analyses/<int:analysis_id>/share",
    methods=["POST"]
)
@login_required
def share_analysis(analysis_id):
    """
    Share analysis with another email.
    """

    try:

        data = request.get_json(
            silent=True
        ) or {}

        analysis = Analysis.query.get(
            analysis_id
        )

        if not analysis:

            return jsonify({
                "error": "Analysis not found"
            }), 404

        if analysis.user_id != current_user.id:

            return jsonify({
                "error": "Unauthorized"
            }), 403

        email = str(
            data.get("email", "")
        ).strip()

        if not email:

            return jsonify({
                "error": "Email required"
            }), 400

        permission = data.get(
            "permission",
            "view"
        )

        if permission not in ["view", "edit"]:

            permission = "view"

        share = Share(
            user_id=current_user.id,
            analysis_id=analysis_id,
            shared_email=email,
            permission=permission
        )

        db.session.add(
            share
        )

        db.session.commit()

        log_audit(
            "analysis_shared",
            "share",
            share.id,
            {
                "email": email
            }
        )

        return jsonify({
            "message": "Analysis shared successfully",
            "share": {
                "id": share.id,
                "shared_email": share.shared_email,
                "permission": share.permission
            }
        }), 201

    except Exception as e:

        db.session.rollback()

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================================
# PAGE SERVING ROUTES
# ============================================================================

@app.route("/")
def index():
    """
    Main entry point.

    If user is logged in:
        /dashboard

    If user is not logged in:
        /login
    """

    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )

    return redirect(
        url_for("login_page")
    )


@app.route("/dashboard")
@login_required
def dashboard():
    """
    Protected dashboard page.

    User must be logged in.
    """

    try:

        with open(
            "dashboard.html",
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()

    except Exception as e:

        return (
            "<h1>Error loading dashboard</h1>"
            f"<p>{str(e)}</p>"
        ), 500


@app.route("/login")
def login_page():
    """
    Login page.
    """

    # If already logged in, don't show login again.
    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )

    try:

        with open(
            "login.html",
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()

    except Exception as e:

        return (
            "<h1>Error loading login page</h1>"
            f"<p>{str(e)}</p>"
        ), 500


@app.route("/signup")
def signup_page():
    """
    Signup page.
    """

    # Already logged in users don't need signup.
    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )

    try:

        with open(
            "signup.html",
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()

    except Exception as e:

        return (
            "<h1>Error loading signup page</h1>"
            f"<p>{str(e)}</p>"
        ), 500


# ============================================================================
# API UTILITY ROUTES
# ============================================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():
    """
    Health check endpoint.
    """

    return jsonify({
        "status": "healthy",
        "service": "AI Analytics Suite",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }), 200


@app.route(
    "/api/config",
    methods=["GET"]
)
def get_config():
    """
    Application configuration.
    """

    return jsonify({
        "app_name": "AI Analytics Suite",
        "version": "1.0.0",
        "max_file_size": "100MB",
        "supported_formats": [
            "csv",
            "xlsx",
            "xls"
        ]
    }), 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """
    404 handler.
    """

    if request.path.startswith("/api/"):

        return jsonify({
            "error": "Endpoint not found"
        }), 404

    return (
        "<h1>404 - Page Not Found</h1>",
        404
    )


@app.errorhandler(500)
def server_error(error):
    """
    500 handler.
    """

    return jsonify({
        "error": "Internal server error"
    }), 500


@app.errorhandler(403)
def forbidden(error):
    """
    403 handler.
    """

    return jsonify({
        "error": "Forbidden"
    }), 403


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

@app.before_request
def create_tables():
    """
    Ensure database tables exist.
    """

    try:

        db.create_all()

    except Exception as e:

        print(
            f"Error creating tables: {e}"
        )


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":

    with app.app_context():

        db.create_all()

    print("\n" + "=" * 70)

    print(
        "🚀 AI ANALYTICS SUITE - COMPLETE VERSION"
    )

    print("=" * 70)

    print(
        "✓ User Authentication (Flask-Login)"
    )

    print(
        "✓ Session Based Authentication"
    )

    print(
        "✓ Database Integration (SQLite)"
    )

    print(
        "✓ Real File Processing (CSV, Excel)"
    )

    print(
        "✓ Advanced Analytics & Statistics"
    )

    print(
        "✓ Anomaly Detection (IQR Method)"
    )

    print(
        "✓ Correlation Analysis"
    )

    print(
        "✓ AI Insights Generation"
    )

    print(
        "✓ Share & Favorite Features"
    )

    print(
        "✓ Protected Dashboard"
    )

    print("=" * 70)

    print("\n🌐 Available URLs:")

    print(
        "  Home:       http://127.0.0.1:5000/"
    )

    print(
        "  Login:      http://127.0.0.1:5000/login"
    )

    print(
        "  Signup:     http://127.0.0.1:5000/signup"
    )

    print(
        "  Dashboard:  http://127.0.0.1:5000/dashboard"
    )

    print(
        "  API Health: http://127.0.0.1:5000/api/health"
    )

    print("=" * 70 + "\n")

    app.run(
        debug=os.getenv(
            "FLASK_DEBUG",
            "True"
        ).lower() == "true",

        host=os.getenv(
            "HOST",
            "127.0.0.1"
        ),

        port=int(
            os.getenv(
                "PORT",
                5000
            )
        )
    )