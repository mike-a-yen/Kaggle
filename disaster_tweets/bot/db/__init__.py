from bot.db.base import make_session
from bot.db.models import Tweets


if __name__ == '__main__':
    session = make_session()
    q = session.query(Tweets)
    session.close()
    print(q.all())