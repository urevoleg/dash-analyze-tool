import datetime as dt

from sqlalchemy import Column, Integer, DateTime, Float, String
from sqlalchemy.ext.declarative import declarative_base

from logging import StreamHandler

Base = declarative_base()


class Weather(Base):
    __tablename__ = 'weather'
    __table_args__ = {'schema': 'sensors'}
    __features__ = ('id', 'ts', 'get_ts', 'tout', 'hout', 'pout', 'lout', 'tin', 'vcc', 'ur', 'mq2', 'infrared', 'wind_cur', 'flag')

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, default=dt.datetime.now, nullable=True)
    get_ts = Column(Float(), nullable=True)
    tout = Column(Float(4), nullable=True)
    hout = Column(Float(4), nullable=True)
    pout = Column(Float(4), nullable=True)
    lout = Column(Integer, nullable=True)
    tin = Column(Float(4), nullable=True)
    vcc = Column(Integer, nullable=True)
    ur = Column(Float(4), nullable=True)
    mq2 = Column(Integer, nullable=True)
    infrared = Column(Integer, nullable=True)
    wind_cur = Column(Float(4), nullable=True)
    flag = Column(Integer, nullable=True)


class PressureSpeed(Base):
    __tablename__ = 'tmp_pressure_speed'
    __table_args__ = {'schema': 'public'}
    __features__ = ('interval', 'ts', 'pspeed')

    interval = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime(timezone=True), default=dt.datetime.utcnow, nullable=False)
    pspeed = Column(Float(), nullable=True)

    def __init__(self, interval, ts, pspeed):
        self.interval = interval
        self.ts = ts
        self.pspeed = pspeed


class WeatherDWH(Base):
    __tablename__ = 'weather'
    __table_args__ = {'schema': 'dwh'}
    __features__ = ('id', 'ts', 'tout', 'hout', 'pout', 'lout', 'ur', 'wind_cur')

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, default=dt.datetime.now, nullable=True)
    tout = Column(Float(4), nullable=True)
    hout = Column(Float(4), nullable=True)
    pout = Column(Float(4), nullable=True)
    lout = Column(Integer, nullable=True)
    ur = Column(Float(4), nullable=True)
    wind_cur = Column(Float(4), nullable=True)


class Log(Base):
    __tablename__ = 'logs'
    __table_args__ = {'schema': 'public'}
    __features__ = ('id', 'ts', 'name', 'levelname', 'msg')

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime(timezone=False), default=dt.datetime.utcnow, nullable=False)
    name = Column(String, nullable=False)
    levelname = Column(String, nullable=False)
    msg = Column(String, nullable=True)

    def __init__(self, ts, name, levelname, msg):
        self.ts = ts
        self.name = name
        self.levelname = levelname
        self.msg = msg


class DbHandler(StreamHandler):

    def __init__(self, db_log_model, session):
        self.session = session
        self.db_log_model = db_log_model
        StreamHandler.__init__(self)

    def emit(self, record):
        r_dict = record.__dict__
        row = {
            'ts': dt.datetime.fromtimestamp(r_dict['created']),
            'name': r_dict['name'],
            'levelname': r_dict['levelname'],
            'msg': r_dict['msg']
        }

        self.session.add(self.db_log_model(**row))
        self.session.commit()
        self.session.close()


class AvgDailyTemperature(Base):
    __tablename__ = 'tmp_avg_daily_temperature'
    __table_args__ = {'schema': 'public'}
    __features__ = ('n_day', 'day', 'tout')

    n_day = Column(Integer, primary_key=True, index=True)
    day = Column(DateTime(timezone=True), nullable=True)
    tout = Column(Float(), nullable=True)

    def __init__(self, n_day, day, tout):
        self.n_day = n_day
        self.day = day
        self.tout = tout