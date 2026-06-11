# db_models.py

from sqlalchemy import Column, String, Integer, Float, DateTime
from datetime import datetime
from database import Base


class SessionResult(Base):
    __tablename__ = "session_op"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(20), nullable=False)

    anger = Column(Float, default=0)
    anxiety = Column(Float, default=0)
    depression = Column(Float, default=0)
    normal_emotion = Column(Float, default=0)
    personality_disorder = Column(Float, default=0)
    sadness = Column(Float, default=0)
    suicidal = Column(Float, default=0)


class VideoData(Base):
    __tablename__ = "video_data"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(20), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Facial emotion percentages (from bar chart)
    angry = Column(Float, default=0)
    disgust = Column(Float, default=0)
    fear = Column(Float, default=0)
    happy = Column(Float, default=0)
    sad = Column(Float, default=0)
    surprise = Column(Float, default=0)
    neutral = Column(Float, default=0)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    patient_id = Column(String(20), unique=True, nullable=False)

    username = Column(String(100), unique=True, nullable=False)

    email = Column(String(150), unique=True, nullable=False)

    password = Column(String(255), nullable=False)