"""Core smoke tests; run with: py -m unittest test_core.py"""
import os
import tempfile
import threading
import unittest
from prs import DatabaseManager, AuthService, PatientService, VisitService

class NoAI:
    def suggest(self, description):
        return None

class PRSTests(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            db = DatabaseManager(os.path.join(directory, "test.db"))
            auth = AuthService(db)
            auth.register("testmanager", "demopass123", "manager")
            user = auth.login("testmanager", "demopass123")
            patients = PatientService(db, auth)
            visits = VisitService(db, auth, patients, NoAI())
            patient_id = patients.create(user, "Test Patient", "2000-01-01")
            note_id, suggestion = visits.add(user, patient_id, "2026-11-01", "Synthetic note")
            self.assertIsNone(suggestion)
            self.assertEqual(visits.history(user, patient_id)[0].note_id, note_id)
            self.assertEqual(patients.get(user, patient_id).name, "Test Patient")

    def test_threaded_access(self):
        with tempfile.TemporaryDirectory() as directory:
            db = DatabaseManager(os.path.join(directory, "thread-test.db"))
            errors = []
            results = []
            def read_db():
                try:
                    results.append(db.execute('SELECT COUNT(*) FROM users').fetchone()[0])
                except Exception as exc:
                    errors.append(exc)
            thread = threading.Thread(target=read_db)
            thread.start()
            thread.join()
            self.assertFalse(errors, errors)
            self.assertEqual(results, [0])

if __name__ == '__main__':
    unittest.main()
