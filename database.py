import sqlite3
import hashlib
from datetime import datetime

# Database initialization


def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # User profiles table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            full_name TEXT,
            age INTEGER,
            date_of_birth DATE,
            gender TEXT,
            blood_group TEXT,
            address TEXT,
            phone TEXT,
            emergency_contact TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')

    # Appointments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            doctor_name TEXT NOT NULL,
            appointment_date DATE NOT NULL,
            appointment_time TIME NOT NULL,
            reason TEXT,
            status TEXT DEFAULT 'scheduled',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')

    # Medical records table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medical_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            record_type TEXT NOT NULL,
            description TEXT,
            file_path TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')

    # Create default admin user
    try:
        admin_password = hash_password('admin123')
        cursor.execute(
            'INSERT OR IGNORE INTO users (username, email, password, role) VALUES (?, ?, ?, ?)',
            ('admin', 'admin@breastcare.ai', admin_password, 'admin')
        )
    except sqlite3.IntegrityError:
        pass

    conn.commit()
    conn.close()

def get_db_connection():
    """Create a database connection with row factory"""
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row  # This enables dictionary-like access
    return conn

# Hash password for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Check if user exists
def user_exists(username, email):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
    user = cursor.fetchone()
    conn.close()
    return user is not None

# Verify user credentials
def verify_user(email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, password, role FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()

    if user and user['password'] == hash_password(password):
        return {'id': user['id'], 'username': user['username'], 'role': user['role']}
    return None

# Create new user
def create_user(username, email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            (username, email, hash_password(password))
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None

# Save user profile
def save_user_profile(user_id, profile_data):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Check if profile already exists
        cursor.execute('SELECT id FROM user_profiles WHERE user_id = ?', (user_id,))
        existing_profile = cursor.fetchone()

        if existing_profile:
            # Update existing profile
            cursor.execute('''
                UPDATE user_profiles
                SET full_name = ?, age = ?, date_of_birth = ?, gender = ?,
                    blood_group = ?, address = ?, phone = ?, emergency_contact = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (
                profile_data['full_name'],
                profile_data['age'],
                profile_data['date_of_birth'],
                profile_data['gender'],
                profile_data['blood_group'],
                profile_data['address'],
                profile_data['phone'],
                profile_data['emergency_contact'],
                user_id
            ))
        else:
            # Insert new profile
            cursor.execute('''
                INSERT INTO user_profiles
                (user_id, full_name, age, date_of_birth, gender, blood_group, address, phone, emergency_contact)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                profile_data['full_name'],
                profile_data['age'],
                profile_data['date_of_birth'],
                profile_data['gender'],
                profile_data['blood_group'],
                profile_data['address'],
                profile_data['phone'],
                profile_data['emergency_contact']
            ))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving profile: {e}")
        conn.close()
        return False

# Get user profile
def get_user_profile(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT full_name, age, date_of_birth, gender, blood_group, address, phone, emergency_contact
        FROM user_profiles
        WHERE user_id = ?
    ''', (user_id,))

    profile = cursor.fetchone()
    conn.close()

    if profile:
        return {
            'full_name': profile['full_name'],
            'age': profile['age'],
            'date_of_birth': profile['date_of_birth'],
            'gender': profile['gender'],
            'blood_group': profile['blood_group'],
            'address': profile['address'],
            'phone': profile['phone'],
            'emergency_contact': profile['emergency_contact']
        }
    return None

# Check if user has completed profile
def has_complete_profile(user_id):
    profile = get_user_profile(user_id)
    return profile is not None and all(profile.values())

# ==================== ADMIN FUNCTIONS ====================

def get_all_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT u.id, u.username, u.email, u.role, u.created_at, 
               p.full_name, p.phone
        FROM users u
        LEFT JOIN user_profiles p ON u.id = p.user_id
        ORDER BY u.created_at DESC
    ''')
    users = cursor.fetchall()
    conn.close()
    return [dict(user) for user in users]

def get_all_appointments():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.*, u.username, p.full_name 
        FROM appointments a
        JOIN users u ON a.user_id = u.id
        LEFT JOIN user_profiles p ON u.id = p.user_id
        ORDER BY a.appointment_date DESC, a.appointment_time DESC
    ''')
    appointments = cursor.fetchall()
    conn.close()
    return [dict(appt) for appt in appointments]

def get_all_medical_records():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT m.*, u.username, p.full_name 
        FROM medical_records m
        JOIN users u ON m.user_id = u.id
        LEFT JOIN user_profiles p ON u.id = p.user_id
        ORDER BY m.uploaded_at DESC
    ''')
    records = cursor.fetchall()
    conn.close()
    return [dict(record) for record in records]

def update_appointment_status(appointment_id, status):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE appointments SET status = ? WHERE id = ?',
        (status, appointment_id)
    )
    conn.commit()
    conn.close()
    return True

def delete_appointment(appointment_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM appointments WHERE id = ?', (appointment_id,))
    conn.commit()
    conn.close()
    return True

def delete_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    return True

# ==================== PATIENT MANAGEMENT FUNCTIONS ====================

def get_all_patients():
    """Get all patients (users with role 'user')"""
    conn = get_db_connection()
    patients = conn.execute(
        'SELECT u.id, u.username, u.email, u.role, p.full_name, p.age, p.gender, p.phone '
        'FROM users u LEFT JOIN user_profiles p ON u.id = p.user_id '
        'WHERE u.role = ?', ('user',)
    ).fetchall()
    conn.close()
    return [dict(patient) for patient in patients]

def get_patient_by_id(user_id):
    """Get patient details by user ID"""
    conn = get_db_connection()
    patient = conn.execute(
        'SELECT u.id, u.username, u.email, u.role, p.* '
        'FROM users u LEFT JOIN user_profiles p ON u.id = p.user_id '
        'WHERE u.id = ?', (user_id,)
    ).fetchone()
    conn.close()
    return dict(patient) if patient else None

def get_patient_medical_history(user_id):
    """Get medical history for a specific patient"""
    conn = get_db_connection()
    history = conn.execute(
        'SELECT m.*, u.username, p.full_name FROM medical_records m '
        'JOIN users u ON m.user_id = u.id '
        'LEFT JOIN user_profiles p ON u.id = p.user_id '
        'WHERE m.user_id = ? ORDER BY m.uploaded_at DESC',
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(record) for record in history]

def get_patient_appointments(user_id):
    """Get appointments for a specific patient"""
    conn = get_db_connection()
    appointments = conn.execute(
        'SELECT a.*, u.username, p.full_name FROM appointments a '
        'JOIN users u ON a.user_id = u.id '
        'LEFT JOIN user_profiles p ON u.id = p.user_id '
        'WHERE a.user_id = ? ORDER BY a.appointment_date DESC',
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(appt) for appt in appointments]

def add_medical_record(user_id, record_type, description, file_path=None):
    """Add a medical record for a patient"""
    try:
        conn = get_db_connection()
        conn.execute(
            'INSERT INTO medical_records (user_id, record_type, description, file_path) VALUES (?, ?, ?, ?)',
            (user_id, record_type, description, file_path)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error adding medical record: {e}")
        conn.close()
        return False

def add_patient_appointment(user_id, doctor_name, appointment_date, appointment_time, reason):
    """Schedule an appointment for a patient"""
    try:
        conn = get_db_connection()
        conn.execute(
            'INSERT INTO appointments (user_id, doctor_name, appointment_date, appointment_time, reason, status) VALUES (?, ?, ?, ?, ?, ?)',
            (user_id, doctor_name, appointment_date, appointment_time, reason, 'scheduled')
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error scheduling appointment: {e}")
        conn.close()
        return False
def book_appointment(user_id, doctor_name, appointment_date, appointment_time, reason):
    """Book a new appointment"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO appointments (user_id, doctor_name, appointment_date, appointment_time, reason, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, doctor_name, appointment_date, appointment_time, reason, 'scheduled'))
        conn.commit()
        appointment_id = cursor.lastrowid
        conn.close()
        return appointment_id
    except Exception as e:
        print(f"Error booking appointment: {e}")
        conn.close()
        return None

def get_user_appointments(user_id):
    """Get all appointments for a specific user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM appointments 
        WHERE user_id = ? 
        ORDER BY appointment_date DESC, appointment_time DESC
    ''', (user_id,))
    appointments = cursor.fetchall()
    conn.close()
    return [dict(appt) for appt in appointments]
def get_all_appointments():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.*, u.username, u.email, p.full_name, p.phone 
        FROM appointments a
        JOIN users u ON a.user_id = u.id
        LEFT JOIN user_profiles p ON u.id = p.user_id
        ORDER BY a.appointment_date DESC, a.appointment_time DESC
    ''')
    appointments = cursor.fetchall()
    conn.close()
    return [dict(appt) for appt in appointments]

def get_available_doctors():
    """Get list of available doctors"""
    return [
        {'id': 1, 'name': 'Dr. Sarah Johnson', 'specialty': 'Cardiologist'},
        {'id': 2, 'name': 'Dr. Michael Chen', 'specialty': 'Dermatologist'},
        {'id': 3, 'name': 'Dr. Emily Davis', 'specialty': 'Neurologist'},
        {'id': 4, 'name': 'Dr. Robert Wilson', 'specialty': 'Oncologist'},
        {'id': 5, 'name': 'Dr. Maria Garcia', 'specialty': 'Gynecologist'}
    ]

# Initialize database when imported
init_db()