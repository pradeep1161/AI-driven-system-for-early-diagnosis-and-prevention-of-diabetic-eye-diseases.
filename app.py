# =============================================================================
# 1. SETUP & IMPORTS
# =============================================================================
import os
import io
import base64
import traceback
import warnings
import logging
from datetime import datetime
from dotenv import load_dotenv

# --- PDF GENERATION IMPORTS ---
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Suppress Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Numpy Fix
import numpy as np
try:
    if not hasattr(np, 'object'):
        np.object = object
except:
    pass

# --- FLASK IMPORTS ---
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import tensorflow as tf
from PIL import Image as PILImage
import cv2

# Silence TensorFlow Logger
tf.get_logger().setLevel(logging.ERROR)

# GradCAM
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# =============================================================================
# 🔧 GEMINI AI LIBRARY SETUP
# =============================================================================
load_dotenv()

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "secret")

# ✅ FIX: Database stored in 'database' folder, version v5
basedir = os.path.abspath(os.path.dirname(__file__))
db_folder = os.path.join(basedir, 'database')
os.makedirs(db_folder, exist_ok=True)

db_path = os.path.join(db_folder, 'patients_v5.db')
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

# Global Models
models = {}
IMG_SIZE = (224, 224)
FUNDUS_CLASSES = ['Cataract', 'Diabetic_Retinopathy', 'Glaucoma', 'Normal']
OCT_CLASSES = ['DME', 'Normal']

class CastLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# =============================================================================
# 3. DATABASE MODELS
# =============================================================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password = db.Column(db.String(200))

class DiagnosisRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(200))
    age = db.Column(db.Integer)
    symptoms = db.Column(db.Text)
    
    # ✅ Added: Diabetes Type
    diabetes_type = db.Column(db.String(50)) 
    
    primary_diagnosis = db.Column(db.String(150))
    overall_confidence = db.Column(db.Float)
    
    # ✅ Specific columns for results
    severity = db.Column(db.String(50))
    fundus_diagnosis = db.Column(db.String(100)) 
    oct_diagnosis = db.Column(db.String(100))    
    
    timestamp = db.Column(db.DateTime, default=datetime.now)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def init_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='pradeep_116').first():
            db.session.add(User(username='pradeep_116', password='Pradeep@116'))
            db.session.commit()

# =============================================================================
# 4. AI HELPERS
# =============================================================================
def load_models():
    print("🔄 Loading Models...")
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        f_path = os.path.join(base, 'models/fundus_model.h5')
        o_path = os.path.join(base, 'models/oct_model.h5')

        with tf.keras.utils.custom_object_scope({'Cast': CastLayer}):
            models['fundus'] = tf.keras.models.load_model(f_path, compile=False)
            models['oct'] = tf.keras.models.load_model(o_path, compile=False)
            
        print("✅ Models Loaded!")
    except Exception as e:
        print(f"❌ Model Load Error: {e}")
        traceback.print_exc()

def process_img(file_storage):
    img = PILImage.open(file_storage).convert('RGB').resize(IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(arr, axis=0), arr, img

def generate_gradcam(model, img_array_expanded, img_array_original, class_index):
    try:
        gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
        score = CategoricalScore([class_index])
        cam = gradcam(score, img_array_expanded, penultimate_layer=-1)
        heatmap = np.uint8(cam[0] * 255)
        
        original_img = np.uint8(img_array_original)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        if heatmap_colored.shape != original_img.shape:
            heatmap_colored = cv2.resize(heatmap_colored, (original_img.shape[1], original_img.shape[0]))
            
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        return superimposed_img
    except Exception as e:
        print(f"Heatmap Error: {e}")
        return np.uint8(img_array_original)

def array_to_base64(arr):
    try:
        img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    except:
        img_rgb = arr
    img = PILImage.fromarray(img_rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def get_advice(disease, patient_context):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return get_fallback_advice(disease)
    
    prompt = f"""
    Act as a Senior Ophthalmologist. Review this patient case and provide a concise clinical assessment.
    CASE DETAILS:
    - Condition Detected: {disease}
    - Severity Status: {patient_context.get('severity', 'Unknown')}
    - Patient Name: {patient_context.get('name', 'Patient')}
    - Age: {patient_context.get('age', 'Unknown')}
    - Diabetes History: {patient_context.get('diabetes_type', 'None')} ({patient_context.get('duration', '0')} years)
    - Reported Symptoms: {patient_context.get('symptoms', 'None')}

    INSTRUCTIONS:
    1. Speak directly to the patient in a professional, empathetic tone.
    2. Explain the condition simply (or why their eyes are healthy if Normal).
    3. Provide 3 specific, actionable steps based on their diabetes history and age.
    4. State clearly when they need their next check-up.
    5. KEEP IT SHORT: Maximum 100 words.
    6. FORMAT: Plain text only. No markdown.
    """
    
    models_to_try = ['gemini-2.5-flash', 'gemini-pro-latest', 'gemini-2.0-flash-exp']
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                if hasattr(response, 'text') and response.text:
                    return response.text.replace('**', '').replace('##', '').replace('* ', '- ')
            except Exception:
                continue
    except ImportError:
        pass

    try:
        import google.generativeai as genai_old
        genai_old.configure(api_key=api_key)
        for model_name in models_to_try:
            try:
                model = genai_old.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response.text:
                    return response.text.replace('**', '').replace('##', '').replace('* ', '- ')
            except Exception:
                continue
    except ImportError:
        pass

    return get_fallback_advice(disease)

def get_fallback_advice(disease):
    advice_db = {
        'Cataract': "Cataract is clouding of the lens. Schedule an appointment. Surgery is effective.",
        'Diabetic_Retinopathy': "URGENT: See a specialist within 1-2 weeks. Control blood sugar strictly.",
        'Glaucoma': "Immediate evaluation needed. High pressure damages optic nerve.",
        'DME': "Fluid in macula. Requires urgent retina specialist referral.",
        'Normal': "Your eyes appear healthy. Continue annual exams and maintain blood sugar control to keep them safe."
    }
    return advice_db.get(disease, f"Consult an ophthalmologist regarding {disease}.")

# =============================================================================
# 5. ROUTES
# =============================================================================
@app.route('/')
def home(): return render_template('index.html')

@app.route('/diagnosis')
def diagnosis_page(): return render_template('diagnosis.html')

@app.route('/assets/<path:filename>')
def serve_asset(filename):
    return send_from_directory('assets', filename)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and user.password == request.form.get('password'):
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    records = DiagnosisRecord.query.order_by(DiagnosisRecord.timestamp.desc()).all()
    
    # Calculate Statistics
    disease_counts = {}
    age_groups = {'0-20': 0, '21-40': 0, '41-60': 0, '60+': 0}
    
    # ✅ FIX: Initialize proper nested structure for Age vs Disease Chart
    age_disease_data = {
        '0-20': {}, 
        '21-40': {}, 
        '41-60': {}, 
        '60+': {}
    }

    for r in records:
        d = r.primary_diagnosis or "Unknown"
        
        # 1. Global Disease Count
        disease_counts[d] = disease_counts.get(d, 0) + 1
        
        # 2. Age Group Calculation
        age_group = None
        try:
            age = int(r.age)
            if age <= 20: 
                age_group = '0-20'
            elif age <= 40: 
                age_group = '21-40'
            elif age <= 60: 
                age_group = '41-60'
            else: 
                age_group = '60+'
            
            # Update global age counts
            age_groups[age_group] += 1
            
            # ✅ FIX: Update nested Age vs Disease data
            if d not in age_disease_data[age_group]:
                age_disease_data[age_group][d] = 0
            age_disease_data[age_group][d] += 1
            
        except:
            pass

    return render_template(
        'dashboard.html', 
        records=records, 
        disease_counts=disease_counts, 
        age_groups=age_groups, 
        age_disease_data=age_disease_data
    )

@app.route('/predict', methods=['POST'])
def predict():
    if not models: load_models()
    
    res = {'fundus_result': None, 'oct_result': None, 'combined_result': None}
    
    # Gather Patient Details
    patient_info = {
        'name': request.form.get('patient_name', 'Patient'),
        'age': request.form.get('patient_age', 'Unknown'),
        'diabetes_type': request.form.get('diabetes_type', 'None'),
        'duration': request.form.get('diabetes_duration', 'Unknown'),
        'symptoms': request.form.get('symptoms', 'None'),
        'severity': 'Unknown' 
    }

    fundus_diag = "N/A"
    oct_diag = "N/A"
    
    primary_diag = "Normal"
    max_conf = 0.0
    fundus_advice_text = None
    oct_advice_text = None

    # --- Process Fundus ---
    if 'fundus_image' in request.files and request.files['fundus_image'].filename:
        try:
            arr_exp, arr_orig, pil = process_img(request.files['fundus_image'])
            pred = models['fundus'].predict(arr_exp)
            idx = np.argmax(pred[0])
            conf = float(np.max(pred[0]) * 100)
            fundus_diag = FUNDUS_CLASSES[idx]
            
            severity = "Healthy / No Risk" if fundus_diag == "Normal" else ("High" if conf > 90 else "Moderate")
            patient_info['severity'] = severity
            
            fundus_advice_text = get_advice(fundus_diag, patient_info)
            heatmap = generate_gradcam(models['fundus'], arr_exp, arr_orig, idx)
            
            res['fundus_result'] = {
                "disease": fundus_diag,
                "prediction": fundus_diag,
                "confidence": f"{conf:.1f}",
                "severity": severity,
                "original_image": array_to_base64(arr_orig.astype('uint8')),
                "heatmap_image": array_to_base64(heatmap),
                "ai_advice": fundus_advice_text
            }
            
            if fundus_diag != "Normal":
                primary_diag = fundus_diag
                max_conf = conf
            elif primary_diag == "Normal":
                 max_conf = conf
        except Exception as e:
            traceback.print_exc()

    # --- Process OCT ---
    if 'oct_image' in request.files and request.files['oct_image'].filename:
        try:
            arr_exp, arr_orig, pil = process_img(request.files['oct_image'])
            pred = models['oct'].predict(arr_exp)
            idx = np.argmax(pred[0])
            c = float(np.max(pred[0]) * 100)
            oct_diag = OCT_CLASSES[idx]
            
            severity = "Healthy / No Risk" if oct_diag == "Normal" else ("High" if c > 90 else "Moderate")
            patient_info['severity'] = severity
            
            oct_advice_text = get_advice(oct_diag, patient_info)
            heatmap = generate_gradcam(models['oct'], arr_exp, arr_orig, idx)

            res['oct_result'] = {
                "disease": oct_diag,
                "prediction": oct_diag,
                "confidence": f"{c:.1f}",
                "severity": severity,
                "original_image": array_to_base64(arr_orig.astype('uint8')),
                "heatmap_image": array_to_base64(heatmap),
                "ai_advice": oct_advice_text
            }
            
            if oct_diag != "Normal":
                if primary_diag == "Normal":
                    primary_diag = oct_diag
                    max_conf = c
        except Exception as e:
            traceback.print_exc()

    # --- Final Combined Result ---
    final_advice = fundus_advice_text if (fundus_diag == primary_diag) else oct_advice_text
    if not final_advice:
        final_advice = get_advice(primary_diag, patient_info)

    res['combined_result'] = {
        "primary_diagnosis": primary_diag,
        "overall_confidence": f"{max_conf:.1f}",
        "ai_advice": final_advice
    }

    try:
        rec = DiagnosisRecord(
            patient_name=patient_info['name'],
            age=int(patient_info['age']) if patient_info['age'].isdigit() else 0,
            symptoms=patient_info['symptoms'],
            diabetes_type=patient_info['diabetes_type'], # ✅ Added here
            primary_diagnosis=primary_diag,
            overall_confidence=max_conf,
            severity=patient_info['severity'],  
            fundus_diagnosis=fundus_diag,
            oct_diagnosis=oct_diag,
            timestamp=datetime.now()
        )
        db.session.add(rec)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"⚠️ Database Save Failed: {e}")

    return jsonify(res)

# =============================================================================
# 🔧 PDF GENERATION ROUTE
# =============================================================================
@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.json
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=1, textColor=colors.darkblue)
        elements.append(Paragraph("AI-DRIVE Diagnosis Report", title_style))
        elements.append(Spacer(1, 12))
        
        # Patient Info
        p_name = data.get('patient_name', 'N/A')
        p_age = data.get('patient_age', 'N/A')
        p_sym = data.get('symptoms', 'None')
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        info_data = [
            ['Patient Name:', p_name, 'Date:', date_str],
            ['Age:', p_age, 'Symptoms:', p_sym[:50]]
        ]
        t = Table(info_data, colWidths=[1.5*inch, 2*inch, 1*inch, 2.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.aliceblue),
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))

        # Helper
        def add_image_section(title, result_data):
            if not result_data: return
            elements.append(Paragraph(f"{title}: {result_data.get('prediction', 'Unknown')}", styles['Heading2']))
            elements.append(Paragraph(f"Confidence: {result_data.get('confidence', '0')}%", styles['Normal']))
            elements.append(Spacer(1, 10))
            
            img_row = []
            for key in ['original_image', 'heatmap_image']:
                b64_str = result_data.get(key)
                if b64_str and ',' in b64_str:
                    try:
                        img_data = base64.b64decode(b64_str.split(',')[1])
                        img_io = io.BytesIO(img_data)
                        rl_img = RLImage(img_io, width=3*inch, height=2.25*inch)
                        img_row.append(rl_img)
                    except: pass
            
            if img_row: elements.append(Table([img_row], colWidths=[3.2*inch, 3.2*inch]))
            
            elements.append(Spacer(1, 10))
            advice = result_data.get('ai_advice', 'No advice available.')
            elements.append(Paragraph("<b>AI Analysis & Advice:</b>", styles['Normal']))
            elements.append(Paragraph(advice, styles['BodyText']))
            elements.append(Spacer(1, 20))

        add_image_section("Fundus Analysis", data.get('fundus_result'))
        add_image_section("OCT Analysis", data.get('oct_result'))
        
        doc.build(elements)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name=f"Report_{p_name}.pdf", mimetype='application/pdf')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        init_db()
        load_models()
    except Exception as e:
        print(f"⚠️ Init Warning: {e}")
    app.run(host='0.0.0.0', port=8501, debug=True)