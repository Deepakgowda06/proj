from flask import Flask, render_template, request, redirect, url_for, session, flash, make_response, send_file, jsonify
import datetime
import os
import io
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from werkzeug.utils import secure_filename
from database import (
    init_db, verify_user, user_exists, create_user, save_user_profile, 
    get_user_profile, has_complete_profile, get_all_users, get_all_appointments, 
    get_all_medical_records, update_appointment_status, delete_appointment, delete_user,
    get_all_patients, get_patient_by_id, get_patient_medical_history, 
    get_patient_appointments, add_medical_record, add_patient_appointment,
    book_appointment, get_user_appointments, get_available_doctors,
    add_medicine_record, get_user_medicine_records, get_all_medicine_records,
    get_medicine_record_by_id, delete_medicine_record
)

# =========================================================
# ⚙️ Flask App Setup
# =========================================================
app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_this_in_production'

# Configure upload settings
UPLOAD_FOLDER = 'static/uploads'
MEDICINE_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'medicine')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Create upload directories if they don't exist
os.makedirs(MEDICINE_UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size

# =========================================================
# 🧠 AI Model Setup
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Load Your Trained EfficientNet-B3 (TorchVision)
model = models.efficientnet_b3(weights=None)
num_ftrs = model.classifier[1].in_features

# Custom classification head
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 3)
)

# Load trained weights (if available)
try:
    model.load_state_dict(torch.load("efficientnet_b3_best.pth", map_location=device))
    model_loaded = True
    print("✅ AI Model loaded successfully")
except:
    model_loaded = False
    print("⚠️  AI Model not found, cancer detection will not work")

model.to(device).eval()

CLASS_LABELS = ['benign', 'malignant', 'normal']

# =========================================================
# 🎨 Transformations
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# =========================================================
# 🧾 Helper Functions
# =========================================================
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cv2_to_base64(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

# =========================================================
# 🔥 Grad-CAM Generator (Manual Implementation)
# =========================================================
def generate_gradcam(model, input_tensor, target_class):
    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    last_conv = model.features[-1]
    forward_handle = last_conv.register_forward_hook(forward_hook)
    backward_handle = last_conv.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    # Weighted feature maps
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(features.shape[1]):
        features[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(features, dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.detach().cpu()

    forward_handle.remove()
    backward_handle.remove()

    return heatmap

# =========================================================
# 🔐 Cache control middleware
# =========================================================
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

# =========================================================
# 🌐 Public Breast Cancer Information Routes
# =========================================================
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

# =========================================================
# 🔐 Authentication Routes
# =========================================================
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

# =========================================================
# 👤 User Portal Routes (Protected)
# =========================================================
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

# =========================================================
# 🧠 AI Cancer Detection Route
# =========================================================
@app.route('/detect-cancer', methods=['GET', 'POST'])
def detect_cancer():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    
    prediction = None
    confidence = None
    uploaded_img = None
    overlay_base64 = None
    error_message = None
    show_appointment_modal = False
    
    if request.method == 'POST':
        if not model_loaded:
            error_message = "AI model is not available. Please contact administrator."
        else:
            file = request.files.get('medical_image')
            if file and file.filename != '' and allowed_file(file.filename):
                try:
                    # Save the uploaded file
                    filename = secure_filename(file.filename)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{session['user_id']}_{timestamp}_{filename}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)

                    # Preprocess
                    img = Image.open(file_path).convert("RGB")
                    input_tensor = transform(img).unsqueeze(0).to(device)

                    # Predict
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        pred_class = probs.argmax(dim=1).item()
                        confidence = probs[0, pred_class].item() * 100

                    # Grad-CAM
                    heatmap = generate_gradcam(model, input_tensor, pred_class)
                    heatmap = heatmap.numpy()

                    # Overlay heatmap
                    img_cv = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
                    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
                    heatmap = cv2.applyColorMap((heatmap * 255).astype("uint8"), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

                    uploaded_img = f"data:image/png;base64,{cv2_to_base64(img_cv)}"
                    overlay_base64 = f"data:image/png;base64,{cv2_to_base64(overlay)}"
                    prediction = CLASS_LABELS[pred_class]
                    
                    # Show appointment modal if malignant or high-risk benign
                    if prediction == 'malignant' or (prediction == 'benign' and confidence > 70):
                        show_appointment_modal = True
                    
                    flash('Analysis completed successfully!', 'success')
                    
                except Exception as e:
                    print(f"Error in cancer detection: {e}")
                    error_message = f"Error processing image: {str(e)}"
            else:
                error_message = "Please upload a valid image file (JPG, JPEG, PNG)"
    
    # Get available doctors for the appointment modal
    doctors = get_available_doctors()
    
    # Get current date for the appointment form
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    return render_template('detect_cancer.html', 
                         username=session['username'],
                         prediction=prediction,
                         confidence=confidence,
                         uploaded_img=uploaded_img,
                         overlay_img=overlay_base64,
                         error_message=error_message,
                         model_loaded=model_loaded,
                         show_appointment_modal=show_appointment_modal,
                         doctors=doctors,
                         current_date=current_date)

# =========================================================
# 📅 Quick Appointment Booking Route
# =========================================================
@app.route('/quick-book-appointment', methods=['POST'])
def quick_book_appointment():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please log in to book appointment'})
    
    try:
        doctor_name = request.form.get('doctor_name')
        appointment_date = request.form.get('appointment_date')
        appointment_time = request.form.get('appointment_time')
        reason = request.form.get('reason', 'Follow-up from AI cancer detection analysis')
        
        if doctor_name and appointment_date and appointment_time:
            appointment_id = book_appointment(
                session['user_id'], 
                doctor_name, 
                appointment_date, 
                appointment_time, 
                reason
            )
            if appointment_id:
                return jsonify({
                    'success': True, 
                    'message': 'Appointment booked successfully!',
                    'redirect_url': url_for('appointments')
                })
            else:
                return jsonify({'success': False, 'message': 'Error booking appointment. Please try again.'})
        else:
            return jsonify({'success': False, 'message': 'Please fill in all required fields.'})
    
    except Exception as e:
        print(f"Error booking appointment: {e}")
        return jsonify({'success': False, 'message': 'Error booking appointment. Please try again.'})

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

# =========================================================
# 💊 Medicine Records Routes
# =========================================================
@app.route('/medicine-upload')
def medicine_upload():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    
    medicine_records = get_user_medicine_records(session['user_id'])
    return render_template('medicine_upload.html', 
                         username=session['username'], 
                         medicine_records=medicine_records)

@app.route('/upload-medicine', methods=['POST'])
def upload_medicine():
    if 'user_id' not in session:
        flash('Please log in to upload medicine records')
        return redirect(url_for('home'))
    
    try:
        medicine_data = {
            'medicine_name': request.form.get('medicine_name'),
            'dosage': request.form.get('dosage'),
            'frequency': request.form.get('frequency'),
            'duration': request.form.get('duration'),
            'prescribed_by': request.form.get('prescribed_by'),
            'notes': request.form.get('notes')
        }
        
        file_path = None
        if 'medicine_file' in request.files:
            file = request.files['medicine_file']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Add user ID and timestamp to filename to avoid conflicts
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{session['user_id']}_{timestamp}_{filename}"
                file_path = os.path.join(MEDICINE_UPLOAD_FOLDER, filename)
                file.save(file_path)
        
        record_id = add_medicine_record(session['user_id'], medicine_data, file_path)
        if record_id:
            flash('Medicine record uploaded successfully!')
        else:
            flash('Error uploading medicine record. Please try again.')
        
    except Exception as e:
        print(f"Error uploading medicine: {e}")
        flash('Error uploading medicine record. Please try again.')
    
    return redirect(url_for('medicine_upload'))

@app.route('/download-medicine-file/<int:record_id>')
def download_medicine_file(record_id):
    if 'user_id' not in session:
        return redirect(url_for('home'))
    
    record = get_medicine_record_by_id(record_id)
    if not record:
        flash('Record not found')
        return redirect(url_for('medicine_upload'))
    
    # Check if user has permission to download this file
    if session['role'] != 'admin' and record['user_id'] != session['user_id']:
        flash('Access denied')
        return redirect(url_for('medicine_upload'))
    
    if not record['file_path'] or not os.path.exists(record['file_path']):
        flash('File not found')
        return redirect(url_for('medicine_upload'))
    
    return send_file(record['file_path'], as_attachment=True)

@app.route('/delete-medicine-record', methods=['POST'])
def delete_medicine_record_route():
    if 'user_id' not in session:
        return {'success': False, 'message': 'Not logged in'}, 401
    
    record_id = request.form.get('record_id')
    record = get_medicine_record_by_id(record_id)
    
    if not record:
        return {'success': False, 'message': 'Record not found'}
    
    # Check if user has permission to delete this record
    if session['role'] != 'admin' and record['user_id'] != session['user_id']:
        return {'success': False, 'message': 'Access denied'}
    
    if delete_medicine_record(record_id):
        # Also delete the associated file if it exists
        if record['file_path'] and os.path.exists(record['file_path']):
            try:
                os.remove(record['file_path'])
            except Exception as e:
                print(f"Error deleting file: {e}")
        
        return {'success': True, 'message': 'Medicine record deleted successfully'}
    else:
        return {'success': False, 'message': 'Error deleting medicine record'}

# =========================================================
# 👨‍💼 Admin Routes
# =========================================================
@app.route('/admin-dashboard')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Access denied. Admin privileges required.')
        return redirect(url_for('home'))
    
    users = get_all_users()
    appointments = get_all_appointments()
    records = get_all_medical_records()
    medicine_records = get_all_medicine_records()
    
    return render_template('admin_dashboard.html', 
                         users=users, 
                         appointments=appointments, 
                         records=records,
                         medicine_records=medicine_records,
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

@app.route('/admin/medicine')
def admin_medicine():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Access denied. Admin privileges required.')
        return redirect(url_for('home'))
    
    medicine_records = get_all_medicine_records()
    return render_template('admin_medicine.html', medicine_records=medicine_records, username=session['username'])

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
    medicine_records = get_user_medicine_records(user_id)
    
    if not patient:
        flash('Patient not found.')
        return redirect(url_for('admin_patients'))
    
    return render_template('admin_patient_detail.html', 
                         patient=patient, 
                         medical_history=medical_history,
                         appointments=appointments,
                         medicine_records=medicine_records,
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

@app.route('/admin/delete-medicine-record', methods=['POST'])
def admin_delete_medicine_record():
    if 'user_id' not in session or session.get('role') != 'admin':
        return {'success': False, 'message': 'Access denied'}, 403
    
    record_id = request.form.get('record_id')
    
    if delete_medicine_record(record_id):
        return {'success': True, 'message': 'Medicine record deleted successfully'}
    else:
        return {'success': False, 'message': 'Error deleting medicine record'}

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.')
    return redirect(url_for('home'))

# Add this if using gunicorn
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)