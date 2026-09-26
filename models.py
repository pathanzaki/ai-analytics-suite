from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import UniqueConstraint


# ============================================================================
# DATABASE
# ============================================================================

db = SQLAlchemy()


# ============================================================================
# USER MODEL
# ============================================================================

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(
        db.String(80),
        unique=True,
        nullable=False,
        index=True
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False,
        index=True
    )

    password_hash = db.Column(
        db.String(255),
        nullable=False
    )

    first_name = db.Column(
        db.String(100),
        nullable=True,
        default=''
    )

    last_name = db.Column(
        db.String(100),
        nullable=True,
        default=''
    )

    is_active = db.Column(
        db.Boolean,
        default=True,
        nullable=False
    )

    last_login = db.Column(
        db.DateTime,
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    # Relationships
    analyses = db.relationship(
        'Analysis',
        backref='user',
        lazy=True,
        cascade='all, delete-orphan'
    )

    favorites = db.relationship(
        'Favorite',
        backref='user',
        lazy=True,
        cascade='all, delete-orphan'
    )

    shares = db.relationship(
        'Share',
        backref='user',
        lazy=True,
        cascade='all, delete-orphan'
    )

    audit_logs = db.relationship(
        'AuditLog',
        backref='user',
        lazy=True,
        cascade='all, delete-orphan'
    )

    def set_password(self, password):
        """Hash and store password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check password against stored hash."""
        return check_password_hash(
            self.password_hash,
            password
        )

    def to_dict(self):
        """Return safe user information."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name or '',
            'last_name': self.last_name or '',
            'is_active': self.is_active,
            'last_login': (
                self.last_login.isoformat()
                if self.last_login
                else None
            ),
            'created_at': (
                self.created_at.isoformat()
                if self.created_at
                else None
            )
        }

    def __repr__(self):
        return f'<User {self.username}>'


# ============================================================================
# ANALYSIS MODEL
# ============================================================================

class Analysis(db.Model):
    __tablename__ = 'analyses'

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id'),
        nullable=False,
        index=True
    )

    filename = db.Column(
        db.String(255),
        nullable=False
    )

    file_path = db.Column(
        db.String(500),
        nullable=False
    )

    file_size = db.Column(
        db.Integer,
        nullable=True
    )

    file_type = db.Column(
        db.String(20),
        nullable=True
    )

    rows = db.Column(
        db.Integer,
        nullable=True
    )

    columns = db.Column(
        db.Integer,
        nullable=True
    )

    column_names = db.Column(
        db.JSON,
        nullable=True
    )

    data_types = db.Column(
        db.JSON,
        nullable=True
    )

    statistics = db.Column(
        db.JSON,
        nullable=True
    )

    anomalies = db.Column(
        db.JSON,
        nullable=True
    )

    correlations = db.Column(
        db.JSON,
        nullable=True
    )

    quality_score = db.Column(
        db.Float,
        nullable=True
    )

    insights = db.Column(
        db.JSON,
        nullable=True
    )

    status = db.Column(
        db.String(50),
        default='pending',
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    def to_dict(self):
        """Return analysis data as JSON-friendly dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'filename': self.filename,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'file_type': self.file_type,
            'rows': self.rows,
            'columns': self.columns,
            'column_names': self.column_names or [],
            'data_types': self.data_types or {},
            'statistics': self.statistics or {},
            'anomalies': self.anomalies or {},
            'correlations': self.correlations or {},
            'quality_score': self.quality_score,
            'insights': self.insights or [],
            'status': self.status,
            'created_at': (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
            'updated_at': (
                self.updated_at.isoformat()
                if self.updated_at
                else None
            )
        }

    def __repr__(self):
        return f'<Analysis {self.filename}>'


# ============================================================================
# FAVORITE MODEL
# ============================================================================

class Favorite(db.Model):
    __tablename__ = 'favorites'

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id'),
        nullable=False,
        index=True
    )

    analysis_id = db.Column(
        db.Integer,
        db.ForeignKey('analyses.id'),
        nullable=False,
        index=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            'user_id',
            'analysis_id',
            name='unique_user_analysis_favorite'
        ),
    )

    analysis = db.relationship(
        'Analysis',
        backref=db.backref(
            'favorites',
            lazy=True,
            cascade='all, delete-orphan'
        )
    )

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'analysis_id': self.analysis_id,
            'created_at': (
                self.created_at.isoformat()
                if self.created_at
                else None
            )
        }

    def __repr__(self):
        return f'<Favorite user={self.user_id} analysis={self.analysis_id}>'


# ============================================================================
# SHARE MODEL
# ============================================================================

class Share(db.Model):
    __tablename__ = 'shares'

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id'),
        nullable=False,
        index=True
    )

    analysis_id = db.Column(
        db.Integer,
        db.ForeignKey('analyses.id'),
        nullable=False,
        index=True
    )

    shared_email = db.Column(
        db.String(120),
        nullable=False
    )

    permission = db.Column(
        db.String(20),
        default='view',
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    analysis = db.relationship(
        'Analysis',
        backref=db.backref(
            'shares',
            lazy=True,
            cascade='all, delete-orphan'
        )
    )

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'analysis_id': self.analysis_id,
            'shared_email': self.shared_email,
            'permission': self.permission,
            'created_at': (
                self.created_at.isoformat()
                if self.created_at
                else None
            )
        }

    def __repr__(self):
        return f'<Share {self.shared_email}>'


# ============================================================================
# AUDIT LOG MODEL
# ============================================================================

class AuditLog(db.Model):
    __tablename__ = 'audit_logs'

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey('users.id'),
        nullable=True,
        index=True
    )

    action = db.Column(
        db.String(100),
        nullable=False
    )

    resource_type = db.Column(
        db.String(50),
        nullable=True
    )

    resource_id = db.Column(
        db.Integer,
        nullable=True
    )

    details = db.Column(
        db.JSON,
        nullable=True
    )

    ip_address = db.Column(
        db.String(45),
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'details': self.details or {},
            'ip_address': self.ip_address,
            'created_at': (
                self.created_at.isoformat()
                if self.created_at
                else None
            )
        }

    def __repr__(self):
        return f'<AuditLog {self.action}>'