"""Small, transactional migrations for the existing MVP database."""
from sqlalchemy import MetaData, inspect, text
from sqlalchemy.schema import CreateTable


def migrate(engine) -> None:
    from backend.app.models.tables import Event

    with engine.begin() as connection:
        if engine.dialect.name == "sqlite":
            # Explicit BEGIN makes SQLite DDL transactional as well as row changes.
            connection.exec_driver_sql("BEGIN")
        columns = {column["name"] for column in inspect(connection).get_columns("zone_visits")}
        for name, kind in [("censored", "BOOLEAN"), ("left_censored", "BOOLEAN"), ("close_reason", "VARCHAR(60)")]:
            if name not in columns:
                connection.execute(text(f"ALTER TABLE zone_visits ADD COLUMN {name} {kind}"))
        constraints = inspect(connection).get_unique_constraints("events")
        legacy = next((item for item in constraints if item["column_names"] == ["job_id", "track_id", "event_type", "timestamp"]), None)
        if legacy:
            if engine.dialect.name == "sqlite":
                metadata = MetaData()
                Event.__table__.metadata.tables["processing_jobs"].to_metadata(metadata)
                replacement = Event.__table__.to_metadata(metadata, name="events_migrated")
                connection.execute(CreateTable(replacement))
                columns_sql = ", ".join(f'"{column.name}"' for column in Event.__table__.columns)
                connection.exec_driver_sql(f"INSERT INTO events_migrated ({columns_sql}) SELECT {columns_sql} FROM events")
                connection.exec_driver_sql("DROP TABLE events")
                connection.exec_driver_sql("ALTER TABLE events_migrated RENAME TO events")
                for index in Event.__table__.indexes:
                    index.create(connection)
            else:
                connection.exec_driver_sql("ALTER TABLE events DROP CONSTRAINT uq_job_track_event_time")
                connection.exec_driver_sql("ALTER TABLE events ADD CONSTRAINT uq_job_track_event_time_label UNIQUE (job_id, track_id, event_type, timestamp, label)")
        exists = connection.execute(text("SELECT version FROM schema_versions WHERE version=1")).first()
        if not exists:
            connection.execute(text("INSERT INTO schema_versions (version) VALUES (1)"))
