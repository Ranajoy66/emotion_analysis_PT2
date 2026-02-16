# db_models.py

from sqlalchemy import Column, String, Integer, Float
from database import Base

class SessionResult(Base):
    __tablename__ = "session_op"

    id = Column(Integer,  primary_key=True, index=True)
    patient_id = Column(String(20), nullable=False)

    anger = Column(Float, default=0)
    anxiety = Column(Float, default=0)
    depression = Column(Float, default=0)
    normal_emotion = Column(Float, default=0)
    personality_disorder = Column(Float, default=0)
    sadness = Column(Float, default=0)
    suicidal = Column(Float, default=0)
