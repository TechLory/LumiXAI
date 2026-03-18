"""Database management and asynchronous job tracking module.

This module handles the persistence layer of the LumiXAI framework using SQLAlchemy and SQLite.
It utilizes a hybrid storage approach: lightweight metadata (status, timestamps, configuration) 
is stored in the SQLite database, while heavy attribution payloads (e.g., matrices, images) 
are saved as independent JSON files to prevent database bloat.
"""

import json
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import create_engine, Column, String, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker

# --- SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "jobs.db"
RESULTS_DIR = DATA_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- SETUP DATABASE ---
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Job(Base):
    """SQLAlchemy ORM Model representing an asynchronous explanation job.

    Attributes:
        id (str): The unique UUID identifier for the job.
        status (str): The current state of the job ("running", "completed", "failed").
        prompt (str): The input text prompt provided by the user.
        source_name (str): The origin of the model (e.g., "huggingface").
        model_name (str): The identifier of the model being explained.
        attributor_name (str): The identifier of the explanation algorithm used.
        created_at (datetime.datetime): The UTC timestamp when the job was initialized.
        completed_at (datetime.datetime): The UTC timestamp when the job finished processing.
        execution_time_sec (float): The total execution time in seconds.
        error_message (str): The error traceback if the job failed.
        result_file (str): The filename of the JSON payload saved on disk.
    """
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="running") 
    prompt = Column(String, nullable=False)
    source_name = Column(String, nullable=False) 
    model_name = Column(String, nullable=False)
    attributor_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    execution_time_sec = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    result_file = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

# --- HELPER FUNCTIONS ---
def create_job(prompt: str, source_name: str, model_name: str, attributor_name: str) -> str:
    """Initializes a new job record in the database.

    Args:
        prompt (str): The input text provided by the user.
        source_name (str): The registry source of the model.
        model_name (str): The specific model ID.
        attributor_name (str): The name of the attribution algorithm.

    Returns:
        str: The generated unique UUID representing the job.
    """
    job_id = str(uuid.uuid4())
    db = SessionLocal()
    new_job = Job(
        id=job_id,
        prompt=prompt,
        source_name=source_name,
        model_name=model_name,
        attributor_name=attributor_name
    )
    db.add(new_job)
    db.commit()
    db.close()
    return job_id

def update_job_success(job_id: str, payload: dict, start_time: float, end_time: float):
    """Marks a job as completed and persists its output payload.

    To optimize database performance, the heavy payload is dumped into a local JSON file 
    while only the reference filename is stored in the SQLite row.

    Args:
        job_id (str): The target job's UUID.
        payload (dict): The serialized attribution results.
        start_time (float): The timestamp when the inference started.
        end_time (float): The timestamp when the inference concluded.
    """
    file_path = RESULTS_DIR / f"{job_id}.json"
    with open(file_path, "w") as f:
        json.dump(payload, f)

    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if job:
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        job.execution_time_sec = round(end_time - start_time, 2)
        job.result_file = str(file_path.name)
        db.commit()
    db.close()

def update_job_failed(job_id: str, error_msg: str):
    """Marks a job as failed and records the error traceback.

    Args:
        job_id (str): The target job's UUID.
        error_msg (str): The stringified exception or error traceback.
    """
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if job:
        job.status = "failed"
        job.completed_at = datetime.now(timezone.utc)
        job.error_message = error_msg
        db.commit()
    db.close()

def get_job(job_id: str) -> dict | None:
    """Retrieves a specific job and its associated payload.

    If the job is marked as completed, this function automatically reads 
    the linked JSON file from the disk and injects it into the returned dictionary.

    Args:
        job_id (str): The requested job's UUID.

    Returns:
        dict | None: A dictionary containing the job metadata and payload, or None if not found.
    """
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    db.close()
    
    if not job:
        return None
    
    job_data = {
        "id": job.id,
        "status": job.status,
        "prompt": job.prompt,
        "source_name": job.source_name,
        "model_name": job.model_name,
        "attributor_name": job.attributor_name,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "execution_time_sec": job.execution_time_sec,
        "error_message": job.error_message,
        "payload": None
    }

    if job.status == "completed" and job.result_file:
        file_path = RESULTS_DIR / job.result_file
        if file_path.exists():
            with open(file_path, "r") as f:
                job_data["payload"] = json.load(f)
                
    return job_data

def get_all_jobs() -> list:
    """Retrieves metadata for all registered jobs.
    
    This function excludes the heavy payload files to ensure fast API responses 
    when populating history sidebars or summary dashboards.

    Returns:
        list: A chronological list of dictionaries containing job metadata (newest first).
    """
    db = SessionLocal()
    jobs = db.query(Job).order_by(Job.created_at.desc()).all()
    db.close()
    
    return [
        {
            "id": j.id,
            "status": j.status,
            "prompt": j.prompt,
            "source_name": j.source_name,
            "model_name": j.model_name,
            "attributor_name": j.attributor_name,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "execution_time_sec": j.execution_time_sec
        }
        for j in jobs
    ]

def delete_all_jobs():
    """Wipes the database table and purges all local JSON result files.

    Raises:
        Exception: If a database transaction fails or file deletion encounters an OS error.
    """
    db = SessionLocal()
    try:
        if RESULTS_DIR.exists():
            for file_path in RESULTS_DIR.glob("*.json"):
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        
        db.query(Job).delete()
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()