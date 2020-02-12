from app import db
from sqlalchemy.dialects.postgresql import JSON


class Tweets(db.Model):
    __tablename__ = 'tweets'

    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String())
    prediction = db.Column(db.Float)

    def __init__(self, text: str, prediction: None) -> None:
        self.text = text
        self.prediction = prediction

    def __repr__(self):
        return f'<id {self.id}: {self.text[0:32]}>'
