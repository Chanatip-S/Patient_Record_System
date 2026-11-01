"""Patient Record System (PRS). Synthetic patient data only."""
import getpass
import hashlib
import hmac
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime

CATEGORIES = ('Respiratory', 'Digestive', 'Musculoskeletal', 'General/Other')
DB_PATH = os.environ.get('PRS_DB', 'prs.db')

@dataclass(frozen=True)
class Patient:
    patient_id: str
    name: str
    date_of_birth: str
    contact: str

@dataclass(frozen=True)
class VisitNote:
    note_id: str
    patient_id: str
    visit_date: str
    description: str
    category: str | None
    suggestion: str | None
    review_status: str

@dataclass(frozen=True)
class User:
    username: str
    role: str

class DatabaseManager:
    def __init__(self, path=DB_PATH):
        # Flask serves requests in worker threads. SQLite connections must not be
        # shared across those threads, so each thread gets its own connection.
        self.path = os.path.abspath(path)
        self._local = threading.local()
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, role TEXT NOT NULL,
                salt BLOB NOT NULL, password_hash BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS patients(patient_id TEXT PRIMARY KEY, name TEXT NOT NULL,
                date_of_birth TEXT NOT NULL, contact TEXT NOT NULL DEFAULT '');
            CREATE TABLE IF NOT EXISTS visit_notes(note_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                visit_date TEXT NOT NULL, description TEXT NOT NULL,
                category TEXT, suggestion TEXT, review_status TEXT NOT NULL,
                created_by TEXT NOT NULL REFERENCES users(username));
            CREATE TRIGGER IF NOT EXISTS prevent_patient_id_change BEFORE UPDATE OF patient_id ON patients
                BEGIN SELECT RAISE(ABORT, 'Patient IDs are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS prevent_note_delete BEFORE DELETE ON visit_notes
                BEGIN SELECT RAISE(ABORT, 'Visit notes cannot be deleted'); END;
            CREATE TRIGGER IF NOT EXISTS prevent_note_content_change BEFORE UPDATE OF patient_id,visit_date,description,created_by ON visit_notes
                BEGIN SELECT RAISE(ABORT, 'Visit note contents are immutable'); END;
        ''')
        self.connection.commit()

    @property
    def connection(self):
        connection = getattr(self._local, 'connection', None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=15)
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA foreign_keys = ON')
            connection.execute('PRAGMA busy_timeout = 15000')
            self._local.connection = connection
        return connection

    def execute(self, sql, params=()):
        connection = self.connection
        with connection:
            return connection.execute(sql, params)

class AuthService:
    PERMISSIONS = {
        'staff': {'create_patient', 'read_patient', 'update_patient', 'add_note', 'read_history', 'review_note'},
        'doctor': {'read_patient', 'read_history', 'add_note', 'review_note'},
        'manager': {'create_patient', 'read_patient', 'update_patient', 'add_note', 'read_history', 'review_note'},
    }

    def __init__(self, db):
        self.db = db

    def register(self, username, password, role):
        if role not in self.PERMISSIONS or not username.strip() or len(password) < 8:
            raise ValueError('Invalid role, username, or password (minimum 8 characters).')
        salt = os.urandom(16)
        digest = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200000)
        self.db.execute('INSERT INTO users VALUES(?,?,?,?)', (username, role, salt, digest))

    def login(self, username, password):
        row = self.db.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
        if not row:
            raise PermissionError('Invalid credentials')
        digest = hashlib.pbkdf2_hmac('sha256', password.encode(), row['salt'], 200000)
        if not hmac.compare_digest(digest, row['password_hash']):
            raise PermissionError('Invalid credentials')
        return User(row['username'], row['role'])

    def require(self, user, permission):
        if user is None or permission not in self.PERMISSIONS.get(user.role, set()):
            raise PermissionError('Not authorized for this operation')

class PatientService:
    def __init__(self, db, auth):
        self.db, self.auth = db, auth

    def create(self, user, name, dob, contact=''):
        self.auth.require(user, 'create_patient')
        if not name.strip():
            raise ValueError('Patient name is required')
        date.fromisoformat(dob)
        for _ in range(5):
            patient_id = 'P-' + uuid.uuid4().hex[:12].upper()
            try:
                self.db.execute('INSERT INTO patients VALUES(?,?,?,?)', (patient_id, name.strip(), dob, contact))
                return patient_id
            except sqlite3.IntegrityError:
                continue
        raise RuntimeError('Unable to generate unique Patient ID')

    def get(self, user, patient_id):
        self.auth.require(user, 'read_patient')
        row = self.db.execute('SELECT * FROM patients WHERE patient_id=?', (patient_id,)).fetchone()
        return Patient(**dict(row)) if row else None

    def update(self, user, patient_id, name=None, contact=None):
        self.auth.require(user, 'update_patient')
        if not self.get(user, patient_id):
            raise ValueError('Patient not found')
        if name is not None:
            if not name.strip():
                raise ValueError('Name cannot be blank')
            self.db.execute('UPDATE patients SET name=? WHERE patient_id=?', (name.strip(), patient_id))
        if contact is not None:
            self.db.execute('UPDATE patients SET contact=? WHERE patient_id=?', (contact, patient_id))

class AICategorizer:
    """Real, optional local zero-shot model (not a keyword-based classifier).

    Errors and low-confidence results are explicit. A missing model never prevents
    a note from being saved. Suitable only for synthetic classroom examples.
    """
    MODEL_ID = os.environ.get('PRS_AI_MODEL', 'cross-encoder/nli-MiniLM2-L6-H768')

    def __init__(self):
        self.pipeline = None
        self.error = None
        self.last_error = None
        self.last_scores = []
        self.last_confidence = None
        try:
            from transformers import pipeline
        except ImportError as exc:
            self.error = ("AI packages are missing or incompatible. Run: "
                          "py -m pip install -r requirements-ai.txt "
                          f"(details: {exc})")
            return
        try:
            self.pipeline = pipeline('zero-shot-classification', model=self.MODEL_ID, device=-1)
        except Exception as exc:
            self.error = f"Cannot load AI model {self.MODEL_ID}: {type(exc).__name__}: {exc}"

    def suggest(self, description):
        self.last_error = self.error
        self.last_scores = []
        self.last_confidence = None
        if self.pipeline is None:
            return None
        try:
            result = self.pipeline(
                description,
                candidate_labels=list(CATEGORIES),
                hypothesis_template='This clinical visit note concerns {}.',
                multi_label=False
            )
            self.last_scores = [(label, round(float(score), 4))
                                for label, score in zip(result['labels'], result['scores'])]
            self.last_confidence = float(result['scores'][0])
            # This is a relative score across the provided labels, not a
            # calibrated probability of medical correctness.
            threshold = float(os.environ.get('PRS_AI_THRESHOLD', '0.45'))
            if self.last_confidence < threshold:
                self.last_error = (f"Model ran, but its highest score was "
                                   f"{self.last_confidence:.1%}, below the "
                                   f"{threshold:.0%} suggestion threshold.")
                return None
            return result['labels'][0]
        except Exception as exc:
            self.last_error = f"AI prediction failed: {type(exc).__name__}: {exc}"
            return None

class VisitService:
    def __init__(self, db, auth, patients, ai):
        self.db, self.auth, self.patients, self.ai = db, auth, patients, ai

    def add(self, user, patient_id, visit_date, description):
        self.auth.require(user, 'add_note')
        if not self.patients.get(user, patient_id):
            raise ValueError('Patient ID is not registered')
        date.fromisoformat(visit_date)
        if not description.strip():
            raise ValueError('Visit description is required')
        suggestion = self.ai.suggest(description)
        note_id = 'N-' + uuid.uuid4().hex[:12].upper()
        self.db.execute('INSERT INTO visit_notes VALUES(?,?,?,?,?,?,?,?)',
            (note_id, patient_id, visit_date, description.strip(), None, suggestion,
             'pending' if suggestion else 'unclassified', user.username))
        return note_id, suggestion

    def history(self, user, patient_id):
        self.auth.require(user, 'read_history')
        if not self.patients.get(user, patient_id):
            raise ValueError('Patient not found')
        rows = self.db.execute('SELECT note_id,patient_id,visit_date,description,category,suggestion,review_status FROM visit_notes WHERE patient_id=? ORDER BY visit_date,note_id', (patient_id,)).fetchall()
        return [VisitNote(**dict(r)) for r in rows]

    def review(self, user, note_id, accept):
        self.auth.require(user, 'review_note')
        row = self.db.execute('SELECT suggestion,review_status FROM visit_notes WHERE note_id=?', (note_id,)).fetchone()
        if not row or row['review_status'] != 'pending' or not row['suggestion']:
            raise ValueError('No pending suggestion for that note')
        self.db.execute('UPDATE visit_notes SET category=?,review_status=? WHERE note_id=?',
            (row['suggestion'] if accept else None, 'accepted' if accept else 'rejected', note_id))

class PRSApp:
    def __init__(self):
        self.db = DatabaseManager()
        self.auth = AuthService(self.db)
        self.patients = PatientService(self.db, self.auth)
        self.ai = AICategorizer()
        self.visits = VisitService(self.db, self.auth, self.patients, self.ai)

    def run(self):
        print('Patient Record System — SYNTHETIC DATA ONLY')
        if self.db.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 0:
            print('First-time setup: create a manager account')
            self.auth.register(input('Manager username: ').strip(), getpass.getpass('Password (8+ chars): '), 'manager')
        user = self.auth.login(input('Username: ').strip(), getpass.getpass('Password: '))
        print(f'Logged in as {user.username} ({user.role})')
        print('AI model ready' if self.ai.pipeline else 'AI unavailable: notes will save without category')
        while True:
            print('\n1 Create patient | 2 Find patient | 3 Update patient | 4 Add visit note | 5 Visit history | 6 Review suggestion | 7 Add user (manager) | 0 Exit')
            choice = input('Choice: ').strip()
            try:
                if choice == '0':
                    break
                if choice == '1':
                    pid = self.patients.create(user, input('Name: '), input('DOB (YYYY-MM-DD): '), input('Contact: '))
                    print('Created Patient ID:', pid)
                elif choice == '2':
                    print(self.patients.get(user, input('Patient ID: ')) or 'Patient not found')
                elif choice == '3':
                    pid = input('Patient ID: ')
                    self.patients.update(user, pid, name=input('New name: '), contact=input('New contact: '))
                    print('Updated (Patient ID and DOB unchanged)')
                elif choice == '4':
                    note_id, suggestion = self.visits.add(user, input('Patient ID: '), input('Visit date (YYYY-MM-DD): '), input('Description: '))
                    print('Saved note:', note_id, '| AI suggestion:', suggestion or 'None')
                    if suggestion:
                        decision = input('Accept (a), reject (r), or leave pending (p)? ').lower().strip()
                        if decision in ('a', 'r'):
                            self.visits.review(user, note_id, decision == 'a')
                elif choice == '5':
                    for note in self.visits.history(user, input('Patient ID: ')):
                        print(note)
                elif choice == '6':
                    self.visits.review(user, input('Note ID: '), input('Accept suggestion? (y/n): ').lower() == 'y')
                    print('Review saved')
                elif choice == '7':
                    if user.role != 'manager':
                        raise PermissionError('Manager only')
                    self.auth.register(input('New username: '), getpass.getpass('Password (8+ chars): '), input('Role (staff/doctor/manager): '))
                    print('User created')
                else:
                    print('Invalid menu choice')
            except (ValueError, PermissionError, sqlite3.Error, RuntimeError) as exc:
                print('Error:', exc)

if __name__ == '__main__':
    PRSApp().run()
