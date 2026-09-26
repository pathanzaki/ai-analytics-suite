"""
AI Analytics Suite - Production-ready Flask backend
Authentication: Flask-Login session
Uploads: CSV / XLSX / XLS
Analytics: statistics, correlations, IQR anomalies, quality score, insights
"""

import os
import json
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename

from config import config
from models import db, User, Analysis, Share, Favorite, AuditLog

load_dotenv()

# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
app = Flask(__name__)

config_name = os.getenv("FLASK_ENV", "development")
if config_name not in config:
    config_name = "development"

app.config.from_object(config[config_name])

# Safe production defaults. Environment variables can override these.
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024
app.config.setdefault("UPLOAD_FOLDER", os.path.join(app.root_path, "uploads"))
app.config.setdefault("ALLOWED_EXTENSIONS", {"csv", "xlsx", "xls"})
app.config.setdefault("ITEMS_PER_PAGE", 20)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db.init_app(app)

# Same-origin requests do not need CORS, but keeping credentials enabled is
# useful if the frontend is ever served from a separate allowed origin.
CORS(app, supports_credentials=True)

# ---------------------------------------------------------------------------
# FLASK LOGIN
# ---------------------------------------------------------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login_page"


@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except (TypeError, ValueError):
        return None


@login_manager.unauthorized_handler
def unauthorized():
    if request.path.startswith("/api/"):
        return jsonify({
            "error": "Authentication required",
            "message": "Please login first."
        }), 401
    return redirect(url_for("login_page", next=request.path))


# ---------------------------------------------------------------------------
# RESPONSE / CACHE HELPERS
# ---------------------------------------------------------------------------
@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Never cache authenticated pages or auth API responses.
    if request.path.startswith(("/dashboard", "/login", "/signup", "/api/auth/")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"

    return response


# ---------------------------------------------------------------------------
# JSON CLEANING
# ---------------------------------------------------------------------------
def clean_value(value):
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not math.isfinite(value) else value

    if isinstance(value, float):
        return None if not math.isfinite(value) else value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    return value


def clean_dict(data):
    if isinstance(data, dict):
        return {str(k): clean_dict(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [clean_dict(v) for v in data]
    return clean_value(data)


# ---------------------------------------------------------------------------
# FILE HELPERS
# ---------------------------------------------------------------------------
def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def read_uploaded_file(file_storage, extension):
    """Read the uploaded file directly, with CSV encoding fallbacks."""
    file_storage.stream.seek(0)

    if extension == "csv":
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                file_storage.stream.seek(0)
                return pd.read_csv(file_storage.stream, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        raise ValueError(f"Could not decode CSV file: {last_error}")

    if extension in ("xlsx", "xls"):
        file_storage.stream.seek(0)
        try:
            return pd.read_excel(file_storage.stream)
        except ImportError:
            raise ValueError(
                "Excel dependency is missing. For .xls files add xlrd==2.0.1 "
                "to requirements.txt. .xlsx uses openpyxl."
            )

    raise ValueError("Unsupported file format.")


# ---------------------------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------------------------
def calculate_statistics(df):
    stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        stats[str(col)] = {
            "mean": clean_value(series.mean()),
            "median": clean_value(series.median()),
            "std": clean_value(series.std()),
            "min": clean_value(series.min()),
            "max": clean_value(series.max()),
            "q25": clean_value(series.quantile(0.25)),
            "q75": clean_value(series.quantile(0.75)),
        }

    return stats


def detect_anomalies(df):
    anomalies = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    total_rows = len(df)

    if total_rows == 0:
        return anomalies

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        # Constant columns have IQR 0 and should not create false anomalies.
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        count = int(mask.sum())

        if count:
            anomalies[str(col)] = {
                "count": count,
                "percentage": clean_value(count / total_rows * 100),
                "lower_bound": clean_value(lower),
                "upper_bound": clean_value(upper),
            }

    return anomalies


def calculate_correlations(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {}
    return clean_dict(numeric_df.corr().to_dict())


def calculate_quality_score(df):
    if df.empty or len(df.columns) == 0:
        return 0.0

    total_cells = len(df) * len(df.columns)
    missing_cells = int(df.isnull().sum().sum())
    completeness = max(0.0, 1.0 - missing_cells / total_cells)

    duplicate_free = 1.0 - (df.duplicated().sum() / max(len(df), 1))

    # Keep the existing product's simple consistency component.
    consistency = 0.90

    score = (
        completeness * 0.60
        + duplicate_free * 0.20
        + consistency * 0.20
    ) * 100

    return round(min(max(score, 0), 100), 2)


def generate_ai_insights(df, filename):
    quality = calculate_quality_score(df)
    missing = int(df.isnull().sum().sum())
    numeric = df.select_dtypes(include=[np.number]).columns
    categorical = df.select_dtypes(include=["object", "category"]).columns
    duplicates = int(df.duplicated().sum())
    anomalies = detect_anomalies(df)

    insights = [
        f"Dataset '{filename}' contains {len(df):,} rows and {len(df.columns):,} columns.",
        f"Data quality score is {quality:.1f}%.",
        (
            f"Found {missing:,} missing values."
            if missing
            else "No missing values were detected."
        ),
        f"Contains {len(numeric)} numeric column(s).",
        f"Found {len(categorical)} categorical column(s).",
        (
            f"Detected {duplicates:,} duplicate row(s)."
            if duplicates
            else "No duplicate rows were detected."
        ),
        (
            f"Detected IQR anomalies in {len(anomalies)} numeric column(s)."
            if anomalies
            else "No significant IQR-based anomalies were detected."
        ),
    ]

    if len(numeric) >= 2:
        insights.append("Correlation analysis is available for the numeric columns.")

    return insights


# ---------------------------------------------------------------------------
# AUDIT
# ---------------------------------------------------------------------------
def log_audit(action, resource_type=None, resource_id=None, details=None):
    if not current_user.is_authenticated:
        return

    try:
        log = AuditLog(
            user_id=current_user.id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=request.remote_addr,
        )
        db.session.add(log)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        app.logger.warning("Audit log error: %s", exc)


# ---------------------------------------------------------------------------
# AUTH
# ---------------------------------------------------------------------------
@app.route("/api/auth/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json(silent=True) or {}

        username = str(data.get("username", "")).strip()
        email = str(data.get("email", "")).strip()
        password = str(data.get("password", ""))
        first_name = str(data.get("first_name", "")).strip()
        last_name = str(data.get("last_name", "")).strip()

        if not username or not email or not password:
            return jsonify({"error": "Username, email and password are required."}), 400

        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters."}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({"error": "Username already exists."}), 409

        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already exists."}), 409

        user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name,
        )
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        return jsonify({
            "message": "User created successfully.",
            "user": user.to_dict()
        }), 201

    except Exception as exc:
        db.session.rollback()
        app.logger.exception("Signup error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/auth/login", methods=["POST"])
def login():
    try:
        data = request.get_json(silent=True) or {}

        username = str(data.get("username", "")).strip()
        password = str(data.get("password", ""))

        if not username or not password:
            return jsonify({"error": "Username and password are required."}), 400

        user = User.query.filter_by(username=username).first()

        if not user or not user.check_password(password):
            return jsonify({"error": "Invalid username or password."}), 401

        if not user.is_active:
            return jsonify({"error": "User account is inactive."}), 403

        login_user(user, remember=False)
        user.last_login = datetime.utcnow()
        db.session.commit()

        return jsonify({
            "message": "Login successful.",
            "user": user.to_dict()
        }), 200

    except Exception as exc:
        db.session.rollback()
        app.logger.exception("Login error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/auth/logout", methods=["POST"])
@login_required
def logout():
    try:
        log_audit("user_logout")
        logout_user()
        return jsonify({"message": "Logged out successfully."}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/auth/me", methods=["GET"])
@login_required
def get_current_user():
    return jsonify(current_user.to_dict()), 200


# ---------------------------------------------------------------------------
# UPLOAD / ANALYSIS
# ---------------------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
@login_required
def upload_file():
    filepath = None

    try:
        if "file" not in request.files:
            return jsonify({
                "error": "No file field received. Please choose a CSV or Excel file."
            }), 400

        file = request.files["file"]

        if not file or not file.filename:
            return jsonify({"error": "No file selected."}), 400

        if not allowed_file(file.filename):
            allowed = ", ".join(sorted(app.config["ALLOWED_EXTENSIONS"]))
            return jsonify({
                "error": f"File type not allowed. Supported formats: {allowed}"
            }), 400

        original_filename = secure_filename(file.filename)
        if not original_filename:
            return jsonify({"error": "Invalid filename."}), 400

        extension = original_filename.rsplit(".", 1)[1].lower()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        stored_filename = f"{timestamp}_{original_filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], stored_filename)

        # Save first because Analysis stores the original file path.
        file.save(filepath)

        # Read from the saved file so the same path is also used by download.
        if extension == "csv":
            last_error = None
            for encoding in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    last_error = None
                    break
                except UnicodeDecodeError as exc:
                    last_error = exc
            if last_error:
                raise ValueError(f"Could not decode CSV file: {last_error}")
        elif extension in ("xlsx", "xls"):
            try:
                df = pd.read_excel(filepath)
            except ImportError as exc:
                raise ValueError(
                    "Excel dependency is missing. Add xlrd==2.0.1 to "
                    "requirements.txt if you want to accept .xls files."
                ) from exc
        else:
            raise ValueError("Unsupported file format.")

        if df is None or df.empty:
            raise ValueError("The uploaded file is empty.")

        # Normalize column names for stable JSON/chart keys.
        df.columns = [
            str(col).strip() if str(col).strip() else f"Column_{i + 1}"
            for i, col in enumerate(df.columns)
        ]

        statistics = calculate_statistics(df)
        anomalies = detect_anomalies(df)
        correlations = calculate_correlations(df)
        quality_score = calculate_quality_score(df)
        insights = generate_ai_insights(df, original_filename)

        analysis = Analysis(
            user_id=current_user.id,
            filename=original_filename,
            file_path=filepath,
            file_size=os.path.getsize(filepath),
            file_type=extension,
            rows=len(df),
            columns=len(df.columns),
            column_names=[str(c) for c in df.columns.tolist()],
            data_types={str(c): str(df[c].dtype) for c in df.columns},
            statistics=clean_dict(statistics),
            anomalies=clean_dict(anomalies),
            correlations=clean_dict(correlations),
            quality_score=quality_score,
            insights=clean_dict(insights),
            status="completed",
        )

        db.session.add(analysis)
        db.session.commit()

        log_audit("file_upload", "analysis", analysis.id)

        return jsonify({
            "message": "File uploaded successfully.",
            "analysis": analysis.to_dict()
        }), 201

    except Exception as exc:
        db.session.rollback()

        # Do not leave orphaned files after a failed DB/analytics operation.
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass

        app.logger.exception("UPLOAD ERROR")
        return jsonify({
            "error": str(exc),
            "message": "The file could not be processed."
        }), 500



# ---------------------------------------------------------------------------
# HR / BI DASHBOARD DATA
# ---------------------------------------------------------------------------
def _norm_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_column(df, aliases):
    normalized = {_norm_name(c): c for c in df.columns}
    for alias in aliases:
        key = _norm_name(alias)
        if key in normalized:
            return normalized[key]
    # Fuzzy fallback: alias appears inside a column name.
    for c in df.columns:
        cn = _norm_name(c)
        for alias in aliases:
            ak = _norm_name(alias)
            if ak and (ak in cn or cn in ak):
                return c
    return None


def _age_group(age):
    try:
        age = float(age)
    except (TypeError, ValueError):
        return "Unknown"
    if age < 25:
        return "Under 25"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55+"


def _weighted_count(frame, employee_count_col=None):
    if frame.empty:
        return 0
    if employee_count_col and employee_count_col in frame.columns:
        values = pd.to_numeric(frame[employee_count_col], errors="coerce").fillna(0)
        total = float(values.sum())
        return int(round(total))
    return int(len(frame))


def build_dashboard_data(df):
    """
    Build adaptive dashboard data.

    HR datasets get the HR/attrition metrics from the reference-style dashboard.
    Other datasets get generic BI metrics based only on columns actually present.
    """
    employee_count_col = _find_column(
        df, ["EmployeeCount", "Employee Count", "Headcount", "Employees"]
    )
    attrition_col = _find_column(
        df, ["Attrition", "Employee Attrition", "Left", "Churn"]
    )
    age_col = _find_column(df, ["Age"])
    department_col = _find_column(df, ["Department", "Dept"])
    gender_col = _find_column(df, ["Gender", "Sex"])
    satisfaction_col = _find_column(
        df,
        ["JobSatisfaction", "Job Satisfaction", "Satisfaction", "Job Satisfaction Rating"]
    )
    education_col = _find_column(
        df, ["EducationField", "Education Field", "Education"]
    )

    numeric_df = df.select_dtypes(include=[np.number])
    numeric_columns = [str(c) for c in numeric_df.columns]
    categorical_columns = [
        str(c) for c in df.select_dtypes(include=["object", "category", "bool"]).columns
    ]

    total_rows = int(len(df))
    total_columns = int(len(df.columns))
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    numeric_summary = []
    for col in numeric_df.columns[:12]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        numeric_summary.append({
            "name": str(col),
            "mean": clean_value(series.mean()),
            "min": clean_value(series.min()),
            "max": clean_value(series.max()),
            "median": clean_value(series.median()),
        })

    categorical_summary = []
    for col in df.select_dtypes(include=["object", "category", "bool"]).columns[:8]:
        counts = df[col].fillna("Missing").astype(str).value_counts().head(6)
        categorical_summary.append({
            "column": str(col),
            "total_unique": int(df[col].nunique(dropna=False)),
            "categories": [
                {"name": str(k), "count": int(v)}
                for k, v in counts.items()
            ]
        })

    # ---------- HR ----------
    employee_total = _weighted_count(df, employee_count_col)

    attrition_yes = pd.Series(False, index=df.index)
    if attrition_col:
        values = df[attrition_col].astype(str).str.strip().str.lower()
        attrition_yes = values.isin(["yes", "y", "true", "1", "left", "churned"])

    attrition_frame = df.loc[attrition_yes]
    attrition_count = _weighted_count(attrition_frame, employee_count_col)
    active_count = max(employee_total - attrition_count, 0)
    attrition_rate = (
        round((attrition_count / employee_total * 100), 2)
        if employee_total else 0.0
    )

    avg_age = None
    if age_col:
        ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
        if not ages.empty:
            avg_age = round(float(ages.mean()), 1)

    department = []
    if department_col:
        for name, group in df.groupby(department_col, dropna=False):
            label = "Unknown" if pd.isna(name) else str(name)
            department.append({
                "name": label,
                "count": _weighted_count(
                    group.loc[attrition_yes.loc[group.index]],
                    employee_count_col
                )
            })
        department.sort(key=lambda x: x["count"], reverse=True)
        department = department[:8]

    age_groups = ["Under 25", "25-34", "35-44", "45-54", "55+"]
    age_distribution = {k: 0 for k in age_groups}
    if age_col:
        tmp = df.copy()
        tmp["_age_group"] = tmp[age_col].apply(_age_group)
        for group_name, group in tmp.groupby("_age_group", dropna=False):
            if group_name in age_distribution:
                age_distribution[group_name] = _weighted_count(
                    group, employee_count_col
                )

    satisfaction = []
    if satisfaction_col:
        values = pd.to_numeric(df[satisfaction_col], errors="coerce")
        if values.notna().any():
            for raw in sorted(values.dropna().unique()):
                group = df.loc[values == raw]
                label = str(int(raw)) if float(raw).is_integer() else str(raw)
                satisfaction.append({
                    "rating": label,
                    "label": {
                        "1": "Low",
                        "2": "Medium",
                        "3": "High",
                        "4": "Very High"
                    }.get(label, "Rating"),
                    "employees": _weighted_count(group, employee_count_col),
                    "attrition": _weighted_count(
                        group.loc[attrition_yes.loc[group.index]],
                        employee_count_col
                    )
                })
        else:
            text_values = df[satisfaction_col].fillna("Unknown").astype(str).str.strip()
            for label, group in df.groupby(text_values):
                satisfaction.append({
                    "rating": str(label),
                    "label": "",
                    "employees": _weighted_count(group, employee_count_col),
                    "attrition": _weighted_count(
                        group.loc[attrition_yes.loc[group.index]],
                        employee_count_col
                    )
                })

    education = []
    if education_col:
        for name, group in df.groupby(education_col, dropna=False):
            label = "Unknown" if pd.isna(name) else str(name)
            education.append({
                "name": label,
                "count": _weighted_count(
                    group.loc[attrition_yes.loc[group.index]],
                    employee_count_col
                )
            })
        education.sort(key=lambda x: x["count"], reverse=True)
        education = education[:8]

    gender_age = []
    if age_col and gender_col:
        temp = df.copy()
        temp["_age_group"] = temp[age_col].apply(_age_group)
        gender_values = temp[gender_col].astype(str).str.strip()
        gender_counts = gender_values.value_counts()
        gender_names = list(gender_counts.head(2).index)

        for age_name in age_groups:
            age_frame = temp[temp["_age_group"] == age_name]
            entries = {}

            for gender_name in gender_names:
                gframe = age_frame[
                    gender_values.loc[age_frame.index] == gender_name
                ]
                denom = _weighted_count(gframe, employee_count_col)
                numer = _weighted_count(
                    gframe.loc[attrition_yes.loc[gframe.index]],
                    employee_count_col
                )
                entries[str(gender_name)] = (
                    round((numer / denom * 100), 1) if denom else 0.0
                )

            age_attrition = age_frame.loc[attrition_yes.loc[age_frame.index]]
            age_total = _weighted_count(age_frame, employee_count_col)
            age_attr_count = _weighted_count(age_attrition, employee_count_col)
            total_rate = (
                round((age_attr_count / age_total * 100), 1)
                if age_total else 0.0
            )

            gender_age.append({
                "age_group": age_name,
                "gender_rates": entries,
                "overall_rate": total_rate
            })

    # Require multiple recognizable HR signals before calling a dataset HR.
    hr_signals = sum(bool(x) for x in [
        attrition_col,
        age_col,
        department_col,
        gender_col,
        satisfaction_col,
        education_col,
    ])
    is_hr = bool(attrition_col and hr_signals >= 2)

    return clean_dict({
        "is_hr": is_hr,
        "columns": [str(c) for c in df.columns],
        "total_rows": total_rows,
        "total_columns": total_columns,
        "numeric_count": len(numeric_columns),
        "categorical_count": len(categorical_columns),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,

        # HR fields
        "employee_count": employee_total if is_hr else None,
        "attrition_count": attrition_count if is_hr else None,
        "active_employees": active_count if is_hr else None,
        "attrition_rate": attrition_rate if is_hr else None,
        "avg_age": avg_age if is_hr else None,
        "department_attrition": department if is_hr else [],
        "age_distribution": (
            [{"name": k, "count": v} for k, v in age_distribution.items()]
            if is_hr else []
        ),
        "job_satisfaction": satisfaction if is_hr else [],
        "education_attrition": education if is_hr else [],
        "gender_age_attrition": gender_age if is_hr else [],
    })

@app.route("/api/analyses/<int:analysis_id>/dashboard", methods=["GET"])
@login_required
def get_dashboard_data(analysis_id):
    """Return raw-data-derived data for the compact BI/HR dashboard."""
    analysis = Analysis.query.get(analysis_id)

    if not analysis:
        return jsonify({"error": "Analysis not found."}), 404

    if analysis.user_id != current_user.id:
        return jsonify({"error": "Unauthorized."}), 403

    if not analysis.file_path or not os.path.exists(analysis.file_path):
        return jsonify({
            "error": "The original uploaded file is no longer available."
        }), 404

    try:
        extension = str(analysis.file_type or "").lower()
        if extension == "csv":
            df = pd.read_csv(analysis.file_path)
        elif extension in ("xlsx", "xls"):
            df = pd.read_excel(analysis.file_path)
        else:
            return jsonify({"error": "Unsupported stored file format."}), 400

        if df.empty:
            return jsonify({"error": "The stored dataset is empty."}), 400

        return jsonify({
            "analysis_id": analysis.id,
            "filename": analysis.filename,
            "dashboard": build_dashboard_data(df)
        }), 200

    except Exception as exc:
        app.logger.exception("Dashboard data error")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# ANALYSIS APIs
# ---------------------------------------------------------------------------
@app.route("/api/analyses", methods=["GET"])
@login_required
def get_analyses():
    try:
        page = max(request.args.get("page", 1, type=int), 1)
        per_page = app.config.get("ITEMS_PER_PAGE", 20)

        pagination = (
            Analysis.query
            .filter_by(user_id=current_user.id)
            .order_by(Analysis.created_at.desc())
            .paginate(page=page, per_page=per_page)
        )

        return jsonify({
            "total": pagination.total,
            "pages": pagination.pages,
            "current_page": page,
            "analyses": [a.to_dict() for a in pagination.items],
        }), 200

    except Exception as exc:
        app.logger.exception("Get analyses error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/analyses/<int:analysis_id>", methods=["GET"])
@login_required
def get_analysis(analysis_id):
    try:
        analysis = Analysis.query.get(analysis_id)

        if not analysis:
            return jsonify({"error": "Analysis not found."}), 404

        if analysis.user_id != current_user.id:
            return jsonify({"error": "Unauthorized."}), 403

        # Keep the response compatible with both old and new dashboard code.
        return jsonify({
            "analysis": analysis.to_dict()
        }), 200

    except Exception as exc:
        app.logger.exception("Get analysis error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/analyses/<int:analysis_id>/download", methods=["GET"])
@login_required
def download_analysis(analysis_id):
    try:
        analysis = Analysis.query.get(analysis_id)

        if not analysis:
            return jsonify({"error": "Analysis not found."}), 404

        if analysis.user_id != current_user.id:
            return jsonify({"error": "Unauthorized."}), 403

        if not analysis.file_path or not os.path.exists(analysis.file_path):
            return jsonify({
                "error": "The uploaded file is no longer available on this server."
            }), 404

        log_audit("file_download", "analysis", analysis_id)

        return send_file(
            analysis.file_path,
            as_attachment=True,
            download_name=analysis.filename,
        )

    except Exception as exc:
        app.logger.exception("Download error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/analyses/<int:analysis_id>/favorite", methods=["POST"])
@login_required
def toggle_favorite(analysis_id):
    try:
        analysis = Analysis.query.get(analysis_id)

        if not analysis:
            return jsonify({"error": "Analysis not found."}), 404

        if analysis.user_id != current_user.id:
            return jsonify({"error": "Unauthorized."}), 403

        favorite = Favorite.query.filter_by(
            user_id=current_user.id,
            analysis_id=analysis_id
        ).first()

        if favorite:
            db.session.delete(favorite)
            db.session.commit()
            return jsonify({
                "message": "Removed from favorites.",
                "favorite": False
            }), 200

        favorite = Favorite(
            user_id=current_user.id,
            analysis_id=analysis_id
        )
        db.session.add(favorite)
        db.session.commit()

        return jsonify({
            "message": "Added to favorites.",
            "favorite": True
        }), 201

    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/analyses/<int:analysis_id>/share", methods=["POST"])
@login_required
def share_analysis(analysis_id):
    try:
        data = request.get_json(silent=True) or {}
        email = str(data.get("email", "")).strip()
        permission = data.get("permission", "view")

        analysis = Analysis.query.get(analysis_id)

        if not analysis:
            return jsonify({"error": "Analysis not found."}), 404

        if analysis.user_id != current_user.id:
            return jsonify({"error": "Unauthorized."}), 403

        if not email:
            return jsonify({"error": "Email is required."}), 400

        if permission not in ("view", "edit"):
            permission = "view"

        share = Share(
            user_id=current_user.id,
            analysis_id=analysis_id,
            shared_email=email,
            permission=permission,
        )

        db.session.add(share)
        db.session.commit()

        return jsonify({
            "message": "Analysis shared successfully.",
            "share": {
                "id": share.id,
                "shared_email": share.shared_email,
                "permission": share.permission,
            }
        }), 201

    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login_page"))


@app.route("/dashboard")
@login_required
def dashboard():
    try:
        with open("dashboard.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"<h1>Error loading dashboard</h1><p>{exc}</p>", 500


@app.route("/login")
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    try:
        with open("login.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"<h1>Error loading login page</h1><p>{exc}</p>", 500


@app.route("/signup")
def signup_page():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    try:
        with open("signup.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"<h1>Error loading signup page</h1><p>{exc}</p>", 500


# ---------------------------------------------------------------------------
# UTILITY
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "AI Analytics Suite",
        "version": "1.1.0",
        "timestamp": datetime.utcnow().isoformat(),
    }), 200


@app.route("/api/config")
def get_config():
    return jsonify({
        "app_name": "AI Analytics Suite",
        "version": "1.1.0",
        "max_file_size": f"{app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)}MB",
        "supported_formats": sorted(app.config["ALLOWED_EXTENSIONS"]),
    }), 200


# ---------------------------------------------------------------------------
# ERRORS
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(error):
    return jsonify({
        "error": "File is too large.",
        "message": "Maximum upload size is 100MB."
    }), 413


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found."}), 404
    return "<h1>404 - Page Not Found</h1>", 404


@app.errorhandler(500)
def server_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error."}), 500
    return "<h1>500 - Internal Server Error</h1>", 500


@app.errorhandler(403)
def forbidden(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Forbidden."}), 403
    return "<h1>403 - Forbidden</h1>", 403


# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------
with app.app_context():
    db.create_all()


# ---------------------------------------------------------------------------
# LOCAL RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("AI ANALYTICS SUITE")
    print("Login:     http://127.0.0.1:5000/login")
    print("Signup:    http://127.0.0.1:5000/signup")
    print("Dashboard: http://127.0.0.1:5000/dashboard")
    print("Health:    http://127.0.0.1:5000/api/health")
    print("=" * 70)

    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "True").lower() == "true",
    )
