"""Educational PRS web interface. Synthetic records only; not for clinical use."""
import os
from datetime import date
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, session
from prs import DatabaseManager, AuthService, PatientService, VisitService, AICategorizer, User

app = Flask(__name__)
app.secret_key = os.environ.get('PRS_SECRET_KEY') or os.urandom(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')
db = DatabaseManager(os.environ.get('PRS_DB', 'prs.db'))
auth = AuthService(db)
patients = PatientService(db, auth)

class LazyAI:
    """Load the model only on an explicit test or a visit-note submission."""
    def __init__(self):
        self.model = None
        self.last_error = None
        self.last_scores = []
        self.last_confidence = None

    def suggest(self, description):
        if self.model is None:
            self.model = AICategorizer()
        result = self.model.suggest(description)
        self.last_error = self.model.last_error or self.model.error
        self.last_scores = self.model.last_scores
        self.last_confidence = self.model.last_confidence
        return result

ai = LazyAI()
visits = VisitService(db, auth, patients, ai)

def current_user():
    username = session.get('username')
    if not username:
        return None
    row = db.execute('SELECT username, role FROM users WHERE username=?', (username,)).fetchone()
    return User(row['username'], row['role']) if row else None

def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not current_user():
            return redirect(url_for('login'))
        return fn(*args, **kwargs)
    return wrapped

@app.context_processor
def inject():
    return {'user': current_user(), 'today': date.today().isoformat()}

@app.errorhandler(403)
def forbidden(e):
    return render_template('message.html', title='Not authorized', message='You do not have permission for this operation.'), 403

@app.errorhandler(PermissionError)
def permission_denied(e):
    return render_template('message.html', title='Not authorized', message='You do not have permission for this operation.'), 403

@app.route('/')
def index():
    return redirect(url_for('dashboard') if current_user() else url_for('login'))

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    if db.execute('SELECT COUNT(*) FROM users').fetchone()[0]:
        return redirect(url_for('login'))
    if request.method == 'POST':
        try:
            auth.register(request.form['username'].strip(), request.form['password'], 'manager')
            flash('Manager account created. Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(str(e), 'error')
    return render_template('setup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if not db.execute('SELECT COUNT(*) FROM users').fetchone()[0]:
        return redirect(url_for('setup'))
    if request.method == 'POST':
        try:
            user = auth.login(request.form['username'].strip(), request.form['password'])
            session.clear()
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        except PermissionError as e:
            flash(str(e), 'error')
    return render_template('login.html')

@app.post('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.get('/dashboard')
@login_required
def dashboard():
    auth.require(current_user(), 'read_patient')
    count = db.execute('SELECT COUNT(*) FROM patients').fetchone()[0]
    notes = db.execute('SELECT COUNT(*) FROM visit_notes').fetchone()[0]
    pending = db.execute("SELECT COUNT(*) FROM visit_notes WHERE review_status='pending'").fetchone()[0]
    return render_template('dashboard.html', count=count, notes=notes, pending=pending)

@app.route('/patients/new', methods=['GET', 'POST'])
@login_required
def new_patient():
    auth.require(current_user(), 'create_patient')
    if request.method == 'POST':
        try:
            pid = patients.create(current_user(), request.form['name'], request.form['dob'], request.form.get('contact', ''))
            flash('Patient created: ' + pid, 'success')
            return redirect(url_for('patient_detail', pid=pid))
        except Exception as e:
            flash(str(e), 'error')
    return render_template('patient_form.html', patient=None)

@app.get('/patients')
@login_required
def search_patient():
    auth.require(current_user(), 'read_patient')
    pid = request.args.get('id', '').strip()
    if pid:
        return redirect(url_for('patient_detail', pid=pid))
    return render_template('search.html')

@app.get('/patients/<pid>')
@login_required
def patient_detail(pid):
    patient = patients.get(current_user(), pid)
    if not patient:
        return render_template('message.html', title='Patient not found', message='No patient matches that ID.'), 404
    notes = visits.history(current_user(), pid)
    return render_template('patient.html', patient=patient, notes=notes)

@app.route('/patients/<pid>/edit', methods=['GET', 'POST'])
@login_required
def edit_patient(pid):
    auth.require(current_user(), 'update_patient')
    patient = patients.get(current_user(), pid)
    if not patient:
        return render_template('message.html', title='Patient not found', message='No patient matches that ID.'), 404
    if request.method == 'POST':
        try:
            patients.update(current_user(), pid, name=request.form['name'], contact=request.form.get('contact', ''))
            flash('Patient updated. Patient ID and date of birth remain unchanged.', 'success')
            return redirect(url_for('patient_detail', pid=pid))
        except Exception as e:
            flash(str(e), 'error')
    return render_template('patient_form.html', patient=patient)

@app.route('/patients/<pid>/notes/new', methods=['GET', 'POST'])
@login_required
def new_note(pid):
    auth.require(current_user(), 'add_note')
    patient = patients.get(current_user(), pid)
    if not patient:
        return render_template('message.html', title='Patient not found', message='No patient matches that ID.'), 404
    if request.method == 'POST':
        try:
            note_id, suggestion = visits.add(current_user(), pid, request.form['visit_date'], request.form['description'])
            if suggestion:
                flash('Visit note saved. Please review the AI suggestion.', 'success')
                return redirect(url_for('review_note', note_id=note_id))
            flash('Visit note saved without an AI category. ' + (ai.last_error or 'No confident suggestion; visit the AI status page to test the model.'), 'success')
            return redirect(url_for('patient_detail', pid=pid))
        except Exception as e:
            flash(str(e), 'error')
    return render_template('note_form.html', patient=patient)

@app.route('/notes/<note_id>/review', methods=['GET', 'POST'])
@login_required
def review_note(note_id):
    auth.require(current_user(), 'review_note')
    row = db.execute('SELECT * FROM visit_notes WHERE note_id=?', (note_id,)).fetchone()
    if not row:
        return render_template('message.html', title='Note not found', message='No visit note matches that ID.'), 404
    if request.method == 'POST':
        choice = request.form.get('decision')
        if choice in ('accept', 'reject'):
            try:
                visits.review(current_user(), note_id, choice == 'accept')
                flash('Suggestion ' + ('accepted' if choice == 'accept' else 'rejected') + '.', 'success')
            except Exception as e:
                flash(str(e), 'error')
        else:
            flash('Suggestion left pending.', 'success')
        return redirect(url_for('patient_detail', pid=row['patient_id']))
    return render_template('review.html', note=row)

@app.route('/ai/status', methods=['GET', 'POST'])
@login_required
def ai_status():
    suggestion = None
    if request.method == 'POST':
        # Predefined synthetic example; patient records are not sent to this test.
        suggestion = ai.suggest(
            'Patient reports persistent cough, wheezing, nasal congestion and trouble breathing.'
        )
    return render_template('ai_status.html', model_id=AICategorizer.MODEL_ID,
                           loaded=ai.model is not None and ai.model.pipeline is not None,
                           attempted=ai.model is not None,
                           suggestion=suggestion, error=ai.last_error,
                           scores=ai.last_scores,
                           confidence=ai.last_confidence)

@app.route('/users/new', methods=['GET', 'POST'])
@login_required
def new_user():
    if current_user().role != 'manager':
        return render_template('message.html', title='Not authorized', message='Manager only.'), 403
    if request.method == 'POST':
        try:
            auth.register(request.form['username'].strip(), request.form['password'], request.form['role'])
            flash('User created.', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash(str(e), 'error')
    return render_template('user_form.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
