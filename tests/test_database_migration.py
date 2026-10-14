from sqlalchemy import create_engine, inspect, text

from backend.app.core.database import Base
from backend.app.core.migrations import migrate
from backend.app.models.tables import Event


def test_legacy_migration_preserves_rows_and_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE events")
        connection.exec_driver_sql("DROP TABLE zone_visits")
        connection.exec_driver_sql("""CREATE TABLE events (
            id INTEGER PRIMARY KEY, job_id VARCHAR(36) NOT NULL, track_id INTEGER NOT NULL,
            event_type VARCHAR(50) NOT NULL, timestamp FLOAT NOT NULL, label VARCHAR(120),
            payload_json TEXT NOT NULL, created_at DATETIME NOT NULL,
            CONSTRAINT uq_job_track_event_time UNIQUE (job_id, track_id, event_type, timestamp))""")
        connection.exec_driver_sql("""CREATE TABLE zone_visits (
            id INTEGER PRIMARY KEY, job_id VARCHAR(36) NOT NULL, track_id INTEGER NOT NULL,
            zone_name VARCHAR(120) NOT NULL, entered_at FLOAT NOT NULL,
            exited_at FLOAT, dwell_seconds FLOAT)""")
        connection.execute(text("INSERT INTO processing_jobs (id, source_path, status, progress, entries, exits, current_occupancy, created_at) VALUES ('old', 'old.mp4', 'completed', 1, 0, 0, 0, CURRENT_TIMESTAMP)"))
        connection.execute(text("INSERT INTO events VALUES (7, 'old', 2, 'zone_enter', 1.5, 'cashier', '{}', '2026-10-12 08:00:00')"))
        connection.execute(text("INSERT INTO zone_visits VALUES (9, 'old', 2, 'cashier', 1.5, 3.5, 2)"))
    migrate(engine)
    migrate(engine)
    with engine.begin() as connection:
        assert connection.execute(text("SELECT id, label FROM events")).one() == (7, "cashier")
        assert connection.execute(text("SELECT created_at FROM events WHERE id=7")).scalar() == "2026-10-12 08:00:00"
        assert connection.execute(text("SELECT id, dwell_seconds, censored FROM zone_visits")).one() == (9, 2, None)
        connection.execute(Event.__table__.insert().values(job_id="old", track_id=2, event_type="zone_enter", timestamp=1.5, label="shelves", payload_json="{}"))
        assert connection.execute(text("SELECT COUNT(*) FROM events")).scalar() == 2
        assert connection.execute(text("SELECT COUNT(*) FROM schema_versions WHERE version=1")).scalar() == 1
        assert {index["name"] for index in inspect(connection).get_indexes("events")} == {index.name for index in Event.__table__.indexes}
    engine.dispose()
