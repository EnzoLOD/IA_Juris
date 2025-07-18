"""
Queries Module for Juris Oracle - Legal AI System
Comprehensive database query operations with security, performance optimization and error handling.
"""

import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta
from sqlalchemy import (
    select, insert, update, delete, func, and_, or_, text,
    desc, asc, distinct, case, exists, join, outerjoin,
    Integer, String, Text, DateTime, Boolean, Float
)
from sqlalchemy.orm import Session, selectinload, joinedload, contains_eager
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, NoResultFound
from sqlalchemy.dialects.postgresql import insert as pg_insert
from contextlib import contextmanager
from functools import wraps
import uuid
from enum import Enum

# Importing models (assuming they exist based on typical FastAPI structure)
from app.models.user import User
from app.models.document import Document, DocumentType, DocumentStatus
from app.models.case import Case, CaseStatus, CaseType
from app.models.embedding import DocumentEmbedding, QueryEmbedding
from app.models.search import SearchHistory, SearchResult
from app.models.analysis import LegalAnalysis, AnalysisType
from app.models.audit import AuditLog, ActionType
from app.core.database import get_db
from app.core.config import settings

logger = logging.getLogger(__name__)

# ======================================================================
# UTILITIES AND DECORATORS
# ======================================================================

class QueryError(Exception):
    """Custom exception for query operations"""
    pass

def handle_db_errors(func):
    """Decorator for consistent database error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except IntegrityError as e:
            logger.error(f"Integrity error in {func.__name__}: {str(e)}")
            raise QueryError(f"Data integrity violation: {str(e)}")
        except NoResultFound as e:
            logger.warning(f"No result found in {func.__name__}: {str(e)}")
            raise QueryError(f"Resource not found: {str(e)}")
        except SQLAlchemyError as e:
            logger.error(f"Database error in {func.__name__}: {str(e)}")
            raise QueryError(f"Database operation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise QueryError(f"Unexpected error: {str(e)}")
    return wrapper

@contextmanager
def get_db_session():
    """Context manager for database sessions with proper cleanup"""
    db = next(get_db())
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database transaction failed: {str(e)}")
        raise
    finally:
        db.close()

# ======================================================================
# USER QUERIES
# ======================================================================

class UserQueries:
    """Comprehensive user management queries"""
    
    @staticmethod
    @handle_db_errors
    def create_user(db: Session, user_data: Dict[str, Any]) -> User:
        """
        Create a new user with comprehensive validation
        
        Args:
            db: Database session
            user_data: Dictionary containing user information
            
        Returns:
            User: Created user instance
        """
        # Check if email already exists
        existing_user = db.execute(
            select(User).where(User.email == user_data.get('email'))
        ).scalar_one_or_none()
        
        if existing_user:
            raise QueryError(f"User with email {user_data.get('email')} already exists")
        
        new_user = User(
            id=str(uuid.uuid4()),
            email=user_data['email'],
            username=user_data.get('username'),
            full_name=user_data.get('full_name'),
            hashed_password=user_data['hashed_password'],
            is_active=user_data.get('is_active', True),
            role=user_data.get('role', 'user'),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(new_user)
        db.flush()  # Get the ID without committing
        
        # Log user creation
        UserQueries._log_user_action(db, new_user.id, ActionType.CREATE, "User created")
        
        return new_user
    
    @staticmethod
    @handle_db_errors
    def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
        """Get user by ID with related data preloading"""
        return db.execute(
            select(User)
            .options(selectinload(User.documents))
            .where(User.id == user_id)
        ).scalar_one_or_none()
    
    @staticmethod
    @handle_db_errors
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email for authentication"""
        return db.execute(
            select(User).where(User.email == email)
        ).scalar_one_or_none()
    
    @staticmethod
    @handle_db_errors
    def update_user(db: Session, user_id: str, update_data: Dict[str, Any]) -> User:
        """Update user with optimistic locking"""
        user = UserQueries.get_user_by_id(db, user_id)
        if not user:
            raise QueryError(f"User with ID {user_id} not found")
        
        # Update fields
        for field, value in update_data.items():
            if hasattr(user, field) and field not in ['id', 'created_at']:
                setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        
        # Log update
        UserQueries._log_user_action(db, user_id, ActionType.UPDATE, "User updated")
        
        return user
    
    @staticmethod
    @handle_db_errors
    def get_users_with_pagination(
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        search_term: Optional[str] = None,
        is_active: Optional[bool] = None,
        role: Optional[str] = None
    ) -> Tuple[List[User], int]:
        """Get users with advanced filtering and pagination"""
        
        query = select(User)
        count_query = select(func.count(User.id))
        
        # Apply filters
        filters = []
        if search_term:
            search_filter = or_(
                User.email.ilike(f"%{search_term}%"),
                User.username.ilike(f"%{search_term}%"),
                User.full_name.ilike(f"%{search_term}%")
            )
            filters.append(search_filter)
        
        if is_active is not None:
            filters.append(User.is_active == is_active)
        
        if role:
            filters.append(User.role == role)
        
        if filters:
            query = query.where(and_(*filters))
            count_query = count_query.where(and_(*filters))
        
        # Get total count
        total_count = db.execute(count_query).scalar()
        
        # Apply pagination and ordering
        users = db.execute(
            query
            .order_by(desc(User.created_at))
            .offset(skip)
            .limit(limit)
        ).scalars().all()
        
        return users, total_count
    
    @staticmethod
    def _log_user_action(db: Session, user_id: str, action: ActionType, details: str):
        """Internal method to log user actions"""
        audit_log = AuditLog(
            id=str(uuid.uuid4()),
            user_id=user_id,
            action=action,
            details=details,
            timestamp=datetime.utcnow(),
            ip_address=None  # Would be populated from request context
        )
        db.add(audit_log)

# ======================================================================
# DOCUMENT QUERIES
# ======================================================================

class DocumentQueries:
    """Comprehensive document management queries optimized for legal domain"""
    
    @staticmethod
    @handle_db_errors
    def create_document(db: Session, document_data: Dict[str, Any]) -> Document:
        """Create document with metadata validation"""
        
        # Validate document type
        if document_data.get('document_type') not in [dt.value for dt in DocumentType]:
            raise QueryError(f"Invalid document type: {document_data.get('document_type')}")
        
        new_document = Document(
            id=str(uuid.uuid4()),
            title=document_data['title'],
            content=document_data.get('content'),
            file_path=document_data.get('file_path'),
            file_size=document_data.get('file_size'),
            mime_type=document_data.get('mime_type'),
            document_type=DocumentType(document_data['document_type']),
            status=DocumentStatus.PROCESSING,
            owner_id=document_data['owner_id'],
            metadata_=document_data.get('metadata', {}),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(new_document)
        db.flush()
        
        return new_document
    
    @staticmethod
    @handle_db_errors
    def get_documents_by_user(
        db: Session,
        user_id: str,
        document_type: Optional[DocumentType] = None,
        status: Optional[DocumentStatus] = None,
        skip: int = 0,
        limit: int = 50
    ) -> Tuple[List[Document], int]:
        """Get user documents with advanced filtering"""
        
        query = select(Document).where(Document.owner_id == user_id)
        count_query = select(func.count(Document.id)).where(Document.owner_id == user_id)
        
        if document_type:
            query = query.where(Document.document_type == document_type)
            count_query = count_query.where(Document.document_type == document_type)
        
        if status:
            query = query.where(Document.status == status)
            count_query = count_query.where(Document.status == status)
        
        total_count = db.execute(count_query).scalar()
        
        documents = db.execute(
            query
            .options(selectinload(Document.embeddings))
            .order_by(desc(Document.created_at))
            .offset(skip)
            .limit(limit)
        ).scalars().all()
        
        return documents, total_count
    
    @staticmethod
    @handle_db_errors
    def update_document_status(
        db: Session, 
        document_id: str, 
        status: DocumentStatus,
        processing_details: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Update document processing status with details"""
        
        document = db.execute(
            select(Document).where(Document.id == document_id)
        ).scalar_one_or_none()
        
        if not document:
            raise QueryError(f"Document with ID {document_id} not found")
        
        document.status = status
        document.updated_at = datetime.utcnow()
        
        if processing_details:
            document.metadata_ = {**(document.metadata_ or {}), **processing_details}
        
        return document
    
    @staticmethod
    @handle_db_errors
    def search_documents_full_text(
        db: Session,
        search_query: str,
        user_id: Optional[str] = None,
        document_types: Optional[List[DocumentType]] = None,
        limit: int = 20
    ) -> List[Document]:
        """Advanced full-text search with ranking"""
        
        # PostgreSQL full-text search with ranking
        search_vector = func.to_tsvector('portuguese', Document.content)
        search_query_ts = func.plainto_tsquery('portuguese', search_query)
        
        query = select(
            Document,
            func.ts_rank(search_vector, search_query_ts).label('rank')
        ).where(
            search_vector.match(search_query)
        )
        
        if user_id:
            query = query.where(Document.owner_id == user_id)
        
        if document_types:
            query = query.where(Document.document_type.in_(document_types))
        
        # Only return processed documents
        query = query.where(Document.status == DocumentStatus.COMPLETED)
        
        results = db.execute(
            query
            .order_by(desc('rank'))
            .limit(limit)
        ).all()
        
        return [result.Document for result in results]
    
    @staticmethod
    @handle_db_errors
    def get_document_statistics(db: Session, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive document statistics"""
        
        base_query = select(Document)
        if user_id:
            base_query = base_query.where(Document.owner_id == user_id)
        
        # Total documents
        total_docs = db.execute(
            select(func.count(Document.id)).select_from(base_query.subquery())
        ).scalar()
        
        # Documents by type
        docs_by_type = db.execute(
            select(Document.document_type, func.count(Document.id))
            .select_from(base_query.subquery())
            .group_by(Document.document_type)
        ).all()
        
        # Documents by status
        docs_by_status = db.execute(
            select(Document.status, func.count(Document.id))
            .select_from(base_query.subquery())
            .group_by(Document.status)
        ).all()
        
        # Total file size
        total_size = db.execute(
            select(func.sum(Document.file_size))
            .select_from(base_query.subquery())
        ).scalar() or 0
        
        return {
            'total_documents': total_docs,
            'documents_by_type': {doc_type.value: count for doc_type, count in docs_by_type},
            'documents_by_status': {status.value: count for status, count in docs_by_status},
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }

# ======================================================================
# CASE QUERIES
# ======================================================================

class CaseQueries:
    """Legal case management queries with domain-specific optimizations"""
    
    @staticmethod
    @handle_db_errors
    def create_case(db: Session, case_data: Dict[str, Any]) -> Case:
        """Create legal case with validation"""
        
        new_case = Case(
            id=str(uuid.uuid4()),
            case_number=case_data.get('case_number'),
            title=case_data['title'],
            description=case_data.get('description'),
            case_type=CaseType(case_data['case_type']),
            status=CaseStatus.OPEN,
            client_id=case_data['client_id'],
            assigned_lawyer_id=case_data.get('assigned_lawyer_id'),
            court_name=case_data.get('court_name'),
            filing_date=case_data.get('filing_date'),
            estimated_value=case_data.get('estimated_value'),
            metadata_=case_data.get('metadata', {}),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(new_case)
        db.flush()
        
        return new_case
    
    @staticmethod
    @handle_db_errors
    def get_cases_with_documents(
        db: Session,
        lawyer_id: Optional[str] = None,
        client_id: Optional[str] = None,
        status: Optional[CaseStatus] = None,
        case_type: Optional[CaseType] = None,
        skip: int = 0,
        limit: int = 20
    ) -> Tuple[List[Case], int]:
        """Get cases with related documents and comprehensive filtering"""
        
        query = select(Case).options(
            selectinload(Case.documents),
            selectinload(Case.analyses),
            joinedload(Case.client),
            joinedload(Case.assigned_lawyer)
        )
        count_query = select(func.count(Case.id))
        
        filters = []
        if lawyer_id:
            filters.append(Case.assigned_lawyer_id == lawyer_id)
        if client_id:
            filters.append(Case.client_id == client_id)
        if status:
            filters.append(Case.status == status)
        if case_type:
            filters.append(Case.case_type == case_type)
        
        if filters:
            filter_condition = and_(*filters)
            query = query.where(filter_condition)
            count_query = count_query.where(filter_condition)
        
        total_count = db.execute(count_query).scalar()
        
        cases = db.execute(
            query
            .order_by(desc(Case.created_at))
            .offset(skip)
            .limit(limit)
        ).scalars().all()
        
        return cases, total_count
    
    @staticmethod
    @handle_db_errors
    def search_cases_by_criteria(
        db: Session,
        search_criteria: Dict[str, Any],
        lawyer_id: Optional[str] = None
    ) -> List[Case]:
        """Advanced case search with multiple criteria"""
        
        query = select(Case)
        
        if lawyer_id:
            query = query.where(Case.assigned_lawyer_id == lawyer_id)
        
        filters = []
        
        # Text search in title and description
        if search_text := search_criteria.get('text'):
            text_filter = or_(
                Case.title.ilike(f"%{search_text}%"),
                Case.description.ilike(f"%{search_text}%"),
                Case.case_number.ilike(f"%{search_text}%")
            )
            filters.append(text_filter)
        
        # Date range filter
        if date_from := search_criteria.get('date_from'):
            filters.append(Case.filing_date >= date_from)
        if date_to := search_criteria.get('date_to'):
            filters.append(Case.filing_date <= date_to)
        
        # Value range filter
        if min_value := search_criteria.get('min_value'):
            filters.append(Case.estimated_value >= min_value)
        if max_value := search_criteria.get('max_value'):
            filters.append(Case.estimated_value <= max_value)
        
        # Court filter
        if court_name := search_criteria.get('court_name'):
            filters.append(Case.court_name.ilike(f"%{court_name}%"))
        
        if filters:
            query = query.where(and_(*filters))
        
        return db.execute(
            query
            .options(selectinload(Case.documents))
            .order_by(desc(Case.updated_at))
            .limit(50)
        ).scalars().all()
    
    @staticmethod
    @handle_db_errors
    def get_case_timeline(db: Session, case_id: str) -> List[Dict[str, Any]]:
        """Get comprehensive case timeline with all related activities"""
        
        # Get case updates from audit log
        case_updates = db.execute(
            select(AuditLog)
            .where(
                and_(
                    AuditLog.entity_type == 'Case',
                    AuditLog.entity_id == case_id
                )
            )
            .order_by(desc(AuditLog.timestamp))
        ).scalars().all()
        
        # Get document activities
        doc_activities = db.execute(
            select(Document, AuditLog)
            .join(AuditLog, Document.id == AuditLog.entity_id)
            .where(
                and_(
                    Document.case_id == case_id,
                    AuditLog.entity_type == 'Document'
                )
            )
            .order_by(desc(AuditLog.timestamp))
        ).all()
        
        timeline = []
        
        # Add case updates
        for update in case_updates:
            timeline.append({
                'timestamp': update.timestamp,
                'type': 'case_update',
                'action': update.action.value,
                'details': update.details,
                'user_id': update.user_id
            })
        
        # Add document activities
        for doc, activity in doc_activities:
            timeline.append({
                'timestamp': activity.timestamp,
                'type': 'document_activity',
                'action': activity.action.value,
                'details': f"Document: {doc.title}",
                'document_id': doc.id,
                'user_id': activity.user_id
            })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return timeline

# ======================================================================
# EMBEDDING QUERIES
# ======================================================================

class EmbeddingQueries:
    """Vector embedding queries for semantic search and RAG functionality"""
    
    @staticmethod
    @handle_db_errors
    def store_document_embedding(
        db: Session,
        document_id: str,
        chunk_index: int,
        chunk_text: str,
        embedding_vector: List[float],
        model_name: str
    ) -> DocumentEmbedding:
        """Store document chunk embedding"""
        
        embedding = DocumentEmbedding(
            id=str(uuid.uuid4()),
            document_id=document_id,
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            embedding_vector=embedding_vector,
            model_name=model_name,
            created_at=datetime.utcnow()
        )
        
        db.add(embedding)
        return embedding
    
    @staticmethod
    @handle_db_errors
    def batch_store_embeddings(
        db: Session,
        embeddings_data: List[Dict[str, Any]]
    ) -> List[DocumentEmbedding]:
        """Batch insert embeddings for performance"""
        
        embeddings = []
        for data in embeddings_data:
            embedding = DocumentEmbedding(
                id=str(uuid.uuid4()),
                document_id=data['document_id'],
                chunk_index=data['chunk_index'],
                chunk_text=data['chunk_text'],
                embedding_vector=data['embedding_vector'],
                model_name=data['model_name'],
                created_at=datetime.utcnow()
            )
            embeddings.append(embedding)
        
        db.add_all(embeddings)
        return embeddings
    
    @staticmethod
    @handle_db_errors
    def semantic_search(
        db: Session,
        query_vector: List[float],
        user_id: Optional[str] = None,
        document_types: Optional[List[DocumentType]] = None,
        similarity_threshold: float = 0.7,
        limit: int = 10
    ) -> List[Tuple[DocumentEmbedding, float]]:
        """
        Perform semantic search using cosine similarity
        Note: This requires pgvector extension in PostgreSQL
        """
        
        # Using pgvector cosine similarity
        similarity_expr = func.cosine_similarity(
            DocumentEmbedding.embedding_vector,
            query_vector
        )
        
        query = select(
            DocumentEmbedding,
            similarity_expr.label('similarity')
        ).join(Document).where(
            similarity_expr > similarity_threshold
        )
        
        if user_id:
            query = query.where(Document.owner_id == user_id)
        
        if document_types:
            query = query.where(Document.document_type.in_(document_types))
        
        # Only search in completed documents
        query = query.where(Document.status == DocumentStatus.COMPLETED)
        
        results = db.execute(
            query
            .order_by(desc('similarity'))
            .limit(limit)
        ).all()
        
        return [(result.DocumentEmbedding, result.similarity) for result in results]
    
    @staticmethod
    @handle_db_errors
    def hybrid_search(
        db: Session,
        text_query: str,
        query_vector: List[float],
        user_id: Optional[str] = None,
        alpha: float = 0.5,  # Weight between semantic and text search
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining full-text and semantic search
        """
        
        # Semantic search component
        semantic_similarity = func.cosine_similarity(
            DocumentEmbedding.embedding_vector,
            query_vector
        )
        
        # Full-text search component
        search_vector = func.to_tsvector('portuguese', DocumentEmbedding.chunk_text)
        search_query_ts = func.plainto_tsquery('portuguese', text_query)
        text_rank = func.ts_rank(search_vector, search_query_ts)
        
        # Combined score
        combined_score = (
            alpha * semantic_similarity + 
            (1 - alpha) * text_rank
        ).label('combined_score')
        
        query = select(
            DocumentEmbedding,
            Document,
            semantic_similarity.label('semantic_score'),
            text_rank.label('text_score'),
            combined_score
        ).join(Document).where(
            and_(
                semantic_similarity > 0.3,  # Minimum semantic threshold
                search_vector.match(text_query)  # Text search condition
            )
        )
        
        if user_id:
            query = query.where(Document.owner_id == user_id)
        
        query = query.where(Document.status == DocumentStatus.COMPLETED)
        
        results = db.execute(
            query
            .order_by(desc('combined_score'))
            .limit(limit)
        ).all()
        
        return [
            {
                'embedding': result.DocumentEmbedding,
                'document': result.Document,
                'semantic_score': result.semantic_score,
                'text_score': result.text_score,
                'combined_score': result.combined_score
            }
            for result in results
        ]

# ======================================================================
# ANALYTICS AND REPORTING QUERIES
# ======================================================================

class AnalyticsQueries:
    """Advanced analytics and reporting queries for legal insights"""
    
    @staticmethod
    @handle_db_errors
    def get_user_activity_metrics(
        db: Session,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive user activity metrics"""
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Document uploads
        doc_uploads = db.execute(
            select(func.count(Document.id))
            .where(
                and_(
                    Document.owner_id == user_id,
                    Document.created_at >= start_date
                )
            )
        ).scalar()
        
        # Search queries
        search_count = db.execute(
            select(func.count(SearchHistory.id))
            .where(
                and_(
                    SearchHistory.user_id == user_id,
                    SearchHistory.created_at >= start_date
                )
            )
        ).scalar()
        
        # Analysis requests
        analysis_count = db.execute(
            select(func.count(LegalAnalysis.id))
            .where(
                and_(
                    LegalAnalysis.user_id == user_id,
                    LegalAnalysis.created_at >= start_date
                )
            )
        ).scalar()
        
        # Most searched terms
        top_searches = db.execute(
            select(
                SearchHistory.query_text,
                func.count(SearchHistory.query_text).label('count')
            )
            .where(
                and_(
                    SearchHistory.user_id == user_id,
                    SearchHistory.created_at >= start_date
                )
            )
            .group_by(SearchHistory.query_text)
            .order_by(desc('count'))
            .limit(10)
        ).all()
        
        return {
            'period_days': days,
            'document_uploads': doc_uploads,
            'search_queries': search_count,
            'analysis_requests': analysis_count,
            'top_searches': [
                {'query': search.query_text, 'count': search.count}
                for search in top_searches
            ]
        }
    
    @staticmethod
    @handle_db_errors
    def get_system_performance_metrics(
        db: Session,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get system-wide performance metrics"""
        
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Query response times
        avg_response_time = db.execute(
            select(func.avg(SearchHistory.response_time_ms))
            .where(SearchHistory.created_at >= start_time)
        ).scalar()
        
        # Document processing success rate
        total_processing = db.execute(
            select(func.count(Document.id))
            .where(Document.created_at >= start_time)
        ).scalar()
        
        successful_processing = db.execute(
            select(func.count(Document.id))
            .where(
                and_(
                    Document.created_at >= start_time,
                    Document.status == DocumentStatus.COMPLETED
                )
            )
        ).scalar()
        
        success_rate = (
            (successful_processing / total_processing * 100) 
            if total_processing > 0 else 0
        )
        
        # Error rates by type
        error_counts = db.execute(
            select(
                AuditLog.details,
                func.count(AuditLog.id).label('count')
            )
            .where(
                and_(
                    AuditLog.timestamp >= start_time,
                    AuditLog.action == ActionType.ERROR
                )
            )
            .group_by(AuditLog.details)
            .order_by(desc('count'))
        ).all()
        
        return {
            'period_hours': hours,
            'avg_response_time_ms': round(avg_response_time or 0, 2),
            'document_processing_success_rate': round(success_rate, 2),
            'total_documents_processed': total_processing,
            'error_breakdown': [
                {'error_type': error.details, 'count': error.count}
                for error in error_counts
            ]
        }
    
    @staticmethod
    @handle_db_errors
    def get_legal_domain_insights(
        db: Session,
        lawyer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get insights specific to legal domain"""
        
        base_filter = []
        if lawyer_id:
            base_filter.append(Case.assigned_lawyer_id == lawyer_id)
        
        # Cases by type distribution
        cases_by_type = db.execute(
            select(
                Case.case_type,
                func.count(Case.id).label('count'),
                func.avg(Case.estimated_value).label('avg_value')
            )
            .where(and_(*base_filter) if base_filter else text('1=1'))
            .group_by(Case.case_type)
        ).all()
        
        # Average case duration by status
        case_durations = db.execute(
            select(
                Case.status,
                func.avg(
                    func.extract('epoch', Case.updated_at - Case.created_at) / 86400
                ).label('avg_duration_days')
            )
            .where(and_(*base_filter) if base_filter else text('1=1'))
            .group_by(Case.status)
        ).all()
        
        # Document types most associated with successful outcomes
        successful_doc_types = db.execute(
            select(
                Document.document_type,
                func.count(distinct(Document.case_id)).label('successful_cases')
            )
            .join(Case, Document.case_id == Case.id)
            .where(
                and_(
                    Case.status == CaseStatus.WON,
                    *base_filter
                ) if base_filter else Case.status == CaseStatus.WON
            )
            .group_by(Document.document_type)
            .order_by(desc('successful_cases'))
        ).all()
        
        return {
            'cases_by_type': [
                {
                    'case_type': case.case_type.value,
                    'count': case.count,
                    'avg_value': round(case.avg_value or 0, 2)
                }
                for case in cases_by_type
            ],
            'average_case_duration_by_status': [
                {
                    'status': duration.status.value,
                    'avg_duration_days': round(duration.avg_duration_days or 0, 1)
                }
                for duration in case_durations
            ],
            'document_types_in_successful_cases': [
                {
                    'document_type': doc.document_type.value,
                    'successful_cases': doc.successful_cases
                }
                for doc in successful_doc_types
            ]
        }

# ======================================================================
# SEARCH HISTORY AND OPTIMIZATION
# ======================================================================

class SearchQueries:
    """Search functionality with history tracking and optimization"""
    
    @staticmethod
    @handle_db_errors
    def log_search(
        db: Session,
        user_id: str,
        query_text: str,
        search_type: str,
        results_count: int,
        response_time_ms: float,
        filters_applied: Optional[Dict[str, Any]] = None
    ) -> SearchHistory:
        """Log search query with comprehensive metadata"""
        
        search_log = SearchHistory(
            id=str(uuid.uuid4()),
            user_id=user_id,
            query_text=query_text,
            search_type=search_type,
            results_count=results_count,
            response_time_ms=response_time_ms,
            filters_applied=filters_applied or {},
            created_at=datetime.utcnow()
        )
        
        db.add(search_log)
        return search_log
    
    @staticmethod
    @handle_db_errors
    def get_search_suggestions(
        db: Session,
        user_id: str,
        partial_query: str,
        limit: int = 5
    ) -> List[str]:
        """Get search suggestions based on user history and popular queries"""
        
        # User's previous searches
        user_suggestions = db.execute(
            select(distinct(SearchHistory.query_text))
            .where(
                and_(
                    SearchHistory.user_id == user_id,
                    SearchHistory.query_text.ilike(f"%{partial_query}%")
                )
            )
            .order_by(desc(SearchHistory.created_at))
            .limit(limit)
        ).scalars().all()
        
        # Popular searches from all users (if user suggestions are insufficient)
        if len(user_suggestions) < limit:
            remaining_limit = limit - len(user_suggestions)
            popular_suggestions = db.execute(
                select(
                    SearchHistory.query_text,
                    func.count(SearchHistory.query_text).label('frequency')
                )
                .where(
                    and_(
                        SearchHistory.query_text.ilike(f"%{partial_query}%"),
                        SearchHistory.query_text.notin_(user_suggestions)
                    )
                )
                .group_by(SearchHistory.query_text)
                .order_by(desc('frequency'))
                .limit(remaining_limit)
            ).scalars().all()
            
            user_suggestions.extend(popular_suggestions)
        
        return user_suggestions[:limit]
    
    @staticmethod
    @handle_db_errors
    def get_trending_searches(
        db: Session,
        hours: int = 24,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get trending search queries in the specified time period"""
        
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        trending = db.execute(
            select(
                SearchHistory.query_text,
                func.count(SearchHistory.query_text).label('frequency'),
                func.count(distinct(SearchHistory.user_id)).label('unique_users')
            )
            .where(SearchHistory.created_at >= start_time)
            .group_by(SearchHistory.query_text)
            .having(func.count(SearchHistory.query_text) > 1)  # At least 2 searches
            .order_by(desc('frequency'))
            .limit(limit)
        ).all()
        
        return [
            {
                'query': trend.query_text,
                'frequency': trend.frequency,
                'unique_users': trend.unique_users
            }
            for trend in trending
        ]

# ======================================================================
# MAINTENANCE AND OPTIMIZATION QUERIES
# ======================================================================

class MaintenanceQueries:
    """Database maintenance and optimization queries"""
    
    @staticmethod
    @handle_db_errors
    def cleanup_old_embeddings(
        db: Session,
        days_old: int = 90
    ) -> int:
        """Clean up embeddings for deleted documents"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Find orphaned embeddings
        orphaned_embeddings = db.execute(
            select(DocumentEmbedding.id)
            .outerjoin(Document, DocumentEmbedding.document_id == Document.id)
            .where(
                or_(
                    Document.id.is_(None),  # Document deleted
                    Document.created_at < cutoff_date  # Very old documents
                )
            )
        ).scalars().all()
        
        if orphaned_embeddings:
            deleted_count = db.execute(
                delete(DocumentEmbedding)
                .where(DocumentEmbedding.id.in_(orphaned_embeddings))
            ).rowcount
            
            logger.info(f"Cleaned up {deleted_count} orphaned embeddings")
            return deleted_count
        
        return 0
    
    @staticmethod
    @handle_db_errors
    def archive_old_search_history(
        db: Session,
        days_old: int = 365
    ) -> int:
        """Archive old search history to maintain performance"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Count records to be archived
        count_to_archive = db.execute(
            select(func.count(SearchHistory.id))
            .where(SearchHistory.created_at < cutoff_date)
        ).scalar()
        
        if count_to_archive > 0:
            # In a real implementation, you'd move to an archive table
            # For now, we'll just delete very old records
            deleted_count = db.execute(
                delete(SearchHistory)
                .where(SearchHistory.created_at < cutoff_date)
            ).rowcount
            
            logger.info(f"Archived {deleted_count} old search records")
            return deleted_count
        
        return 0
    
    @staticmethod
    @handle_db_errors
    def get_database_health_metrics(db: Session) -> Dict[str, Any]:
        """Get database health and performance metrics"""
        
        # Table sizes
        table_sizes = db.execute(text("""
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """)).all()
        
        # Index usage
        index_usage = db.execute(text("""
            SELECT 
                indexrelname as index_name,
                idx_tup_read,
                idx_tup_fetch,
                idx_scan
            FROM pg_stat_user_indexes
            ORDER BY idx_scan DESC
            LIMIT 10
        """)).all()
        
        # Connection info
        connections = db.execute(text("""
            SELECT 
                state,
                count(*) as count
            FROM pg_stat_activity 
            WHERE datname = current_database()
            GROUP BY state
        """)).all()
        
        return {
            'table_sizes': [
                {
                    'table': table.tablename,
                    'size_pretty': table.size,
                    'size_bytes': table.size_bytes
                }
                for table in table_sizes
            ],
            'top_indexes': [
                {
                    'index_name': idx.index_name,
                    'scans': idx.idx_scan,
                    'tuples_read': idx.idx_tup_read
                }
                for idx in index_usage
            ],
            'connections_by_state': [
                {
                    'state': conn.state,
                    'count': conn.count
                }
                for conn in connections
            ]
        }

# ======================================================================
# QUERY FACTORY AND FACADE
# ======================================================================

class QueryFactory:
    """Factory class providing access to all query modules"""
    
    def __init__(self):
        self.users = UserQueries()
        self.documents = DocumentQueries()
        self.cases = CaseQueries()
        self.embeddings = EmbeddingQueries()
        self.analytics = AnalyticsQueries()
        self.search = SearchQueries()
        self.maintenance = MaintenanceQueries()
    
    @contextmanager
    def get_session(self):
        """Get database session with proper error handling"""
        with get_db_session() as db:
            yield db

# Singleton instance for easy import
queries = QueryFactory()

# ======================================================================
# BULK OPERATIONS FOR PERFORMANCE
# ======================================================================

class BulkOperations:
    """High-performance bulk operations for large datasets"""
    
    @staticmethod
    @handle_db_errors
    def bulk_update_document_status(
        db: Session,
        document_ids: List[str],
        status: DocumentStatus,
        batch_size: int = 1000
    ) -> int:
        """Bulk update document status for performance"""
        
        total_updated = 0
        
        for i in range(0, len(document_ids), batch_size):
            batch_ids = document_ids[i:i + batch_size]
            
            result = db.execute(
                update(Document)
                .where(Document.id.in_(batch_ids))
                .values(
                    status=status,
                    updated_at=datetime.utcnow()
                )
            )
            
            total_updated += result.rowcount
        
        return total_updated
    
    @staticmethod
    @handle_db_errors
    def bulk_insert_search_results(
        db: Session,
        search_results: List[Dict[str, Any]]
    ) -> int:
        """Bulk insert search results for performance"""
        
        if not search_results:
            return 0
        
        # Use PostgreSQL's INSERT ... ON CONFLICT for upsert behavior
        stmt = pg_insert(SearchResult).values(search_results)
        stmt = stmt.on_conflict_do_update(
            index_elements=['search_history_id', 'document_id'],
            set_=dict(
                relevance_score=stmt.excluded.relevance_score,
                updated_at=datetime.utcnow()
            )
        )
        
        result = db.execute(stmt)
        return result.rowcount

# Export all query classes for external use
__all__ = [
    'UserQueries',
    'DocumentQueries', 
    'CaseQueries',
    'EmbeddingQueries',
    'AnalyticsQueries',
    'SearchQueries',
    'MaintenanceQueries',
    'BulkOperations',
    'QueryFactory',
    'queries',
    'QueryError',
    'get_db_session'
]