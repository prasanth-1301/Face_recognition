import sqlite3
import json
import os
from datetime import datetime


class AttendanceDatabase:

    def __init__(self, db_path="data/attendance.db"):

        self.db_path = db_path

        os.makedirs(
            os.path.dirname(self.db_path),
            exist_ok=True
        )

        self.create_tables()


    def get_connection(self):

        connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False
        )

        connection.row_factory = sqlite3.Row

        return connection


    def create_tables(self):

        connection = self.get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                name TEXT NOT NULL,

                checkin_embedding TEXT NOT NULL,

                checkin_time TEXT NOT NULL,

                checkout_time TEXT,

                status TEXT NOT NULL DEFAULT 'CHECKED_IN',

                similarity REAL,

                duration_seconds INTEGER
            )
            """
        )

        connection.commit()

        connection.close()


    def check_in_person(
        self,
        name,
        embedding
    ):

        connection = self.get_connection()

        cursor = connection.cursor()

        checkin_time = datetime.now().isoformat()

        embedding_json = json.dumps(
            embedding.tolist()
        )

        cursor.execute(
            """
            INSERT INTO attendance (
                name,
                checkin_embedding,
                checkin_time,
                status
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                name,
                embedding_json,
                checkin_time,
                "CHECKED_IN"
            )
        )

        record_id = cursor.lastrowid

        connection.commit()

        connection.close()

        return record_id


    def get_active_people(self):

        connection = self.get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM attendance
            WHERE status = 'CHECKED_IN'
            ORDER BY checkin_time ASC
            """
        )

        rows = cursor.fetchall()

        connection.close()

        return [dict(row) for row in rows]


    def get_active_people_with_embeddings(self):

        people = self.get_active_people()

        for person in people:

            person["embedding"] = json.loads(
                person["checkin_embedding"]
            )

        return people


    def check_out_person(
        self,
        record_id,
        similarity
    ):

        connection = self.get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT checkin_time
            FROM attendance
            WHERE id = ?
            """,
            (record_id,)
        )

        row = cursor.fetchone()

        if row is None:

            connection.close()

            return None


        checkin_time = datetime.fromisoformat(
            row["checkin_time"]
        )

        checkout_time = datetime.now()

        duration = (
            checkout_time - checkin_time
        )

        duration_seconds = int(
            duration.total_seconds()
        )

        cursor.execute(
            """
            UPDATE attendance
            SET
                checkout_time = ?,
                status = ?,
                similarity = ?,
                duration_seconds = ?
            WHERE id = ?
            """,
            (
                checkout_time.isoformat(),
                "CHECKED_OUT",
                similarity,
                duration_seconds,
                record_id
            )
        )

        connection.commit()

        connection.close()

        return {
            "checkin_time": checkin_time,
            "checkout_time": checkout_time,
            "duration_seconds": duration_seconds
        }


    def get_completed_records(self):

        connection = self.get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM attendance
            WHERE status = 'CHECKED_OUT'
            ORDER BY checkout_time DESC
            """
        )

        rows = cursor.fetchall()

        connection.close()

        return [dict(row) for row in rows]


    def is_name_checked_in(
        self,
        name
    ):

        connection = self.get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id
            FROM attendance
            WHERE name = ?
            AND status = 'CHECKED_IN'
            """,
            (name,)
        )

        row = cursor.fetchone()

        connection.close()

        return row is not None