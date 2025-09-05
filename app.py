# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained model pipeline
try:
    loaded_pipeline = joblib.load("health_markers_dataset_model.pkl")
except FileNotFoundError:
    print("Error: The model file 'health_markers_dataset_model.pkl' was not found.")
    print("Please make sure the file is in the same directory as app.py.")
    loaded_pipeline = None

def Feature_Extraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with the following columns:
    ['blood_glucose', 'hba1c', 'systolic_bp', 'diastolic_bp',
     'ldl', 'hdl', 'triglycerides', 'haemoglobin', 'mcv']

    Returns the same DataFrame with additional engineered features:
    - glucose_hba1c_ratio
    - pulse_pressure, MAP, hypertension_flag
    - total_cholesterol, ldl_hdl_ratio, tg_hdl_ratio, non_hdl
    - anaemia_flag, microcytosis_flag, hb_mcv_ratio
    - risk_score
    """
    df.columns = df.columns.str.lower()
    df['glucose_hba1c_ratio'] = df['blood_glucose'] / df['hba1c']
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['MAP'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
    df['hypertension_flag'] = ((df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90)).astype(int)
    df['total_cholesterol'] = df['ldl'] + df['hdl'] + (df['triglycerides'] / 5)
    df['ldl_hdl_ratio'] = df['ldl'] / df['hdl']
    df['tg_hdl_ratio'] = df['triglycerides'] / df['hdl']
    df['non_hdl'] = df['total_cholesterol'] - df['hdl']
    df['anaemia_flag'] = (df['haemoglobin'] < 12).astype(int)
    df['microcytosis_flag'] = (df['mcv'] < 80).astype(int)
    df['hb_mcv_ratio'] = df['haemoglobin'] / df['mcv']
    df['risk_score'] = (df['blood_glucose'] / 200) + (df['ldl'] / 160) + (df['triglycerides'] / 200) + (df['systolic_bp'] / 140)

    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not loaded_pipeline:
        return jsonify({'error': 'Model not loaded. Please check the server logs.'}), 500

    try:
        data = request.json
        input_data = {
            'blood_glucose': float(data.get('blood_glucose', 0)),
            'HbA1C': float(data.get('HbA1C', 0)),
            'Systolic_BP': float(data.get('Systolic_BP', 0)),
            'Diastolic_BP': float(data.get('Diastolic_BP', 0)),
            'LDL': float(data.get('LDL', 0)),
            'HDL': float(data.get('HDL', 0)),
            'Triglycerides': float(data.get('Triglycerides', 0)),
            'Haemoglobin': float(data.get('Haemoglobin', 0)),
            'MCV': float(data.get('MCV', 0))
        }

        # Create DataFrame
        df = pd.DataFrame([input_data])

        # Apply feature extraction
        processed_df = Feature_Extraction(df.copy())

        # Make prediction
        prediction = loaded_pipeline.predict(processed_df)[0]
        
        disease_map = {
            0: 'Anemia',
            1: 'Fit',
            2: 'Hypertension',
            3: 'Diabetes',
            4: 'High_Cholesterol'
        }
        
        predicted_disease = disease_map.get(prediction, 'Unknown')
        
        # Prepare recommendations
        recommendations = {
            'Anemia': {
                'title': 'فقر الدم (الأنيميا)',
                'prevention': 'تغذية سليمة (لحوم حمراء، سبانخ، عدس). تناول فيتامين C لزيادة امتصاص الحديد. تجنب الشاي/القهوة بعد الأكل.',
                'treatment': 'أقراص حديد تحت إشراف الطبيب، وأحيانًا فيتامين B12 أو حمض الفوليك.',
                'suggested_plan': 'حبوب حديد يوميًا + نظام غذائي غني بالحديد + متابعة الهيموجلوبين.'
            },
            'Hypertension': {
                'title': 'ارتفاع ضغط الدم',
                'prevention': 'تقليل الملح، ممارسة الرياضة بانتظام، التحكم في الوزن، والبعد عن التدخين.',
                'treatment': 'أدوية خافضة للضغط تحت إشراف الطبيب ومتابعة ضغط الدم.',
                'suggested_plan': 'قياس الضغط يوميًا + أدوية (ACE inhibitors / Beta blockers) + تقليل الملح.'
            },
            'Diabetes': {
                'title': 'مرض السكر',
                'prevention': 'أكل صحي قليل السكر، الحفاظ على وزن مثالي، ممارسة الرياضة.',
                'treatment': 'أدوية خافضة للسكر (مثل Metformin) أو الأنسولين.',
                'suggested_plan': 'نظام غذائي متوازن + رياضة يومية + علاج دوائي حسب الحالة.'
            },
            'High_Cholesterol': {
                'title': 'ارتفاع الكوليسترول',
                'prevention': 'تقليل الدهون المشبعة (مقليات، سمن)، زيادة الألياف (خضار، فواكه، شوفان)، رياضة منتظمة.',
                'treatment': 'أدوية خافضة للدهون (Statins) تحت إشراف الطبيب ونظام غذائي صحي.',
                'suggested_plan': 'تقليل أكل الدهون + أدوية Statins + متابعة الدهون بالتحاليل.'
            },
            'Fit': {
                'title': 'سليم / طبيعي',
                'prevention': 'الحفاظ على نظام غذائي صحي، ممارسة الرياضة، الكشف الدوري.',
                'treatment': 'لا يوجد علاج، فقط الاستمرار على نمط الحياة الصحي.',
                'suggested_plan': 'متابعة سنوية + نمط حياة صحي.'
            },
            'Unknown': {
                'title': 'غير معروف',
                'prevention': 'لا يمكن تحديد توصيات بسبب عدم كفاية البيانات.',
                'treatment': 'يرجى استشارة طبيب متخصص.',
                'suggested_plan': 'يرجى استشارة طبيب متخصص.'
            }
        }
        
        result = {
            'disease': predicted_disease,
            'recommendation': recommendations.get(predicted_disease, recommendations['Unknown'])
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
