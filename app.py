from flask import Flask, render_template, request, redirect, url_for, session, flash, make_response
import datetime
from database import (
    init_db, verify_user, user_exists, create_user, save_user_profile, 
    get_user_profile, has_complete_profile, get_all_users, get_all_appointments, 
    get_all_medical_records, update_appointment_status, delete_appointment, delete_user,
    get_all_patients, get_patient_by_id, get_patient_medical_history, 
    get_patient_appointments, add_medical_record, add_patient_appointment,
    book_appointment, get_user_appointments, get_available_doctors
)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_this_in_production'

# Cache control middleware
@app.after_request
def after_request(response):
    # Prevent caching for all pages that require authentication
    if 'user_id' in session:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
    else:
        # Public pages can be cached
        response.headers['Cache-Control'] = 'public, max-age=300'
    return response

# Public Breast Cancer Information Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about-breast-cancer')
def about_breast_cancer():
    return render_template('about_breast_cancer.html')

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

@app.route('/risk-factors')
def risk_factors():
    return render_template('risk_factors.html')

@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

@app.route('/screening')
def screening():
    return render_template('screening.html')

# Authentication Routes
@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('login-email')
        password = request.form.get('login-password')
        
        user = verify_user(email, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            
            if not has_complete_profile(user['id']):
                flash('Please complete your profile information')
                return redirect(url_for('upload_profile'))
                
            return redirect(url_for('user_dashboard'))
        else:
            flash('Invalid email or password')
            return redirect(url_for('home'))
    
    return redirect(url_for('home'))

@app.route('/create-account', methods=['POST'])
def create_account():
    if request.method == 'POST':
        username = request.form.get('signup-name')
        email = request.form.get('signup-email')
        password = request.form.get('signup-password')
        confirm_password = request.form.get('signup-confirm')
        
        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('home'))
        elif user_exists(username, email):
            flash('Username or email already exists')
            return redirect(url_for('home'))
        else:
            user_id = create_user(username, email, password)
            if user_id:
                session['user_id'] = user_id
                session['username'] = username
                session['role'] = 'user'
                flash('Account created successfully! Please complete your profile.')
                return redirect(url_for('upload_profile'))
            else:
                flash('Error creating account')
                return redirect(url_for('home'))
    
    return redirect(url_for('home'))

# User Portal Routes (Protected)
@app.route('/user-dashboard')
def user_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    
    user_profile = get_user_profile(session['user_id'])
    return render_template('user_dashboard.html', username=session['username'], profile=user_profile)

@app.route('/upload-profile', methods=['GET', 'POST'])
def upload_profile():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        profile_data = {
            'full_name': request.form['full_name'],
            'age': request.form['age'],
            'date_of_birth': request.form['date_of_birth'],
            'gender': request.form['gender'],
            'blood_group': request.form['blood_group'],
            'address': request.form['address'],
            'phone': request.form['phone'],
            'emergency_contact': request.form['emergency_contact']
        }
        
        if save_user_profile(session['user_id'], profile_data):
            flash('Profile updated successfully!')
            if session.get('role') == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('user_dashboard'))
        else:
            flash('Error saving profile. Please try again.')
    
    existing_profile = get_user_profile(session['user_id'])
    return render_template('upload_profile.html', profile=existing_profile)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    
    user_profile = get_user_profile(session['user_id'])
    return render_template('profile.html', username=session['username'], profile=user_profile)

@app.route('/detect-cancer')
def detect_cancer():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    return render_template('detect_cancer.html', username=session['username'])

@app.route('/medical-reports')
def medical_reports():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    return render_template('medical_reports.html', username=session['username'])

@app.route('/appointments', methods=['GET', 'POST'])
def appointments():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        doctor_name = request.form.get('doctor_name')
        appointment_date = request.form.get('appointment_date')
        appointment_time = request.form.get('appointment_time')
        reason = request.form.get('reason')
        
        if doctor_name and appointment_date and appointment_time:
            appointment_id = book_appointment(
                session['user_id'], 
                doctor_name, 
                appointment_date, 
                appointment_time, 
                reason
            )
            if appointment_id:
                flash('Appointment booked successfully!')
                return redirect(url_for('appointments'))
            else:
                flash('Error booking appointment. Please try again.')
        else:
            flash('Please fill in all required fields.')
    
    # Get user's appointments
    user_appointments = get_user_appointments(session['user_id'])
    doctors = get_available_doctors()
    
    return render_template('appointments.html', 
                         username=session['username'], 
                         appointments=user_appointments,
                         doctors=doctors)

@app.route('/cancel-appointment', methods=['POST'])
def cancel_appointment():
    if 'user_id' not in session:
        return {'success': False, 'message': 'Not logged in'}, 401
    
    appointment_id = request.form.get('appointment_id')
    
    if delete_appointment(appointment_id):
        return {'success': True, 'message': 'Appointment cancelled successfully'}
    else:
        return {'success': False, 'message': 'Error cancelling appointment'}

# Admin Routes
@app.route('/admin-dashboard')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Access denied. Admin privileges required.')
        return redirect(url_for('home'))
    
    users = get_all_users()
    appointments = get_all_appointments()
    records = get_all_medical_records()
    
    return render_template('admin_dashboard.html', 
                         users=users, 
                         appointments=appointments, 
                         records=records,
                         username=session['username'])

@app.route('/admin/users')
def admin_users():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Access denied. Admin privileges required.')
        return redirect(url_for('home'))
    
    users = get_all_users()
    return render_template('admin_users.html', users=users, username=session['username'])

@app.route('/admin/appointments')
def admin_appointments():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Access denied. Admin privileges required.')
        return redirect(url_for('home'))
    
    appointments = get_all_appointments()
    return render_template('admin_appointments.html', appointments=appointments, username=session['username'])

@app.route('/admin/records')
def admin_records():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Access denied. Admin privileges required.')
        return redirect(url_for('home'))
    
    records = get_all_medical_records()
    return render_template('admin_records.html', records=records, username=session['username'])

@app.route('/admin/patients')
def admin_patients():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Access denied. Admin privileges required.')
        return redirect(url_for('home'))
    
    patients = get_all_patients()
    return render_template('admin_patients.html', patients=patients, username=session['username'])

@app.route('/admin/patient/<int:user_id>')
def admin_patient_detail(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Access denied. Admin privileges required.')
        return redirect(url_for('home'))
    
    patient = get_patient_by_id(user_id)
    medical_history = get_patient_medical_history(user_id)
    appointments = get_patient_appointments(user_id)
    
    if not patient:
        flash('Patient not found.')
        return redirect(url_for('admin_patients'))
    
    return render_template('admin_patient_detail.html', 
                         patient=patient, 
                         medical_history=medical_history,
                         appointments=appointments,
                         username=session['username'])

@app.route('/admin/update-appointment-status', methods=['POST'])
def update_appointment_status_route():
    if 'user_id' not in session or session.get('role') != 'admin':
        return {'success': False, 'message': 'Access denied'}, 403
    
    appointment_id = request.form.get('appointment_id')
    status = request.form.get('status')
    
    if update_appointment_status(appointment_id, status):
        return {'success': True, 'message': 'Appointment status updated'}
    else:
        return {'success': False, 'message': 'Error updating appointment'}

@app.route('/admin/delete-appointment', methods=['POST'])
def delete_appointment_route():
    if 'user_id' not in session or session.get('role') != 'admin':
        return {'success': False, 'message': 'Access denied'}, 403
    
    appointment_id = request.form.get('appointment_id')
    
    if delete_appointment(appointment_id):
        return {'success': True, 'message': 'Appointment deleted'}
    else:
        return {'success': False, 'message': 'Error deleting appointment'}

@app.route('/admin/delete-user', methods=['POST'])
def delete_user_route():
    if 'user_id' not in session or session.get('role') != 'admin':
        return {'success': False, 'message': 'Access denied'}, 403
    
    user_id = request.form.get('user_id')
    
    if delete_user(user_id):
        return {'success': True, 'message': 'User deleted'}
    else:
        return {'success': False, 'message': 'Error deleting user'}

@app.route('/admin/add-medical-record', methods=['POST'])
def add_medical_record_route():
    if 'user_id' not in session or session.get('role') != 'admin':
        return {'success': False, 'message': 'Access denied'}, 403
    
    user_id = request.form.get('user_id')
    record_type = request.form.get('record_type')
    description = request.form.get('description')
    
    if add_medical_record(user_id, record_type, description):
        return {'success': True, 'message': 'Medical record added successfully'}
    else:
        return {'success': False, 'message': 'Error adding medical record'}

@app.route('/admin/add-appointment', methods=['POST'])
def add_appointment_route():
    if 'user_id' not in session or session.get('role') != 'admin':
        return {'success': False, 'message': 'Access denied'}, 403
    
    user_id = request.form.get('user_id')
    doctor_name = request.form.get('doctor_name')
    appointment_date = request.form.get('appointment_date')
    appointment_time = request.form.get('appointment_time')
    reason = request.form.get('reason')
    
    if add_patient_appointment(user_id, doctor_name, appointment_date, appointment_time, reason):
        return {'success': True, 'message': 'Appointment scheduled successfully'}
    else:
        return {'success': False, 'message': 'Error scheduling appointment'}

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.')
    return redirect(url_for('home'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)