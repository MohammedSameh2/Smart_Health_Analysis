# import pandas as pd
# def Feature_Extraction(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Takes a DataFrame with the following columns:
#     ['Blood_glucose', 'HbA1C', 'Systolic_BP', 'Diastolic_BP',
#      'LDL', 'HDL', 'Triglycerides', 'Haemoglobin', 'MCV']

#     Returns the same DataFrame with additional engineered features:
#     - glucose_hba1c_ratio
#     - pulse_pressure, MAP, hypertension_flag
#     - total_cholesterol, ldl_hdl_ratio, tg_hdl_ratio, non_hdl
#     - anaemia_flag, microcytosis_flag, hb_mcv_ratio
#     - risk_score
#     """

#     df = df.rename(columns=str.lower)


#     df['glucose_hba1c_ratio'] = df['blood_glucose'] / df['hba1c']

#     df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
#     df['MAP'] = (df['systolic_bp'] + 2*df['diastolic_bp']) / 3
#     df['hypertension_flag'] = ((df['systolic_bp'] >= 140) |
#                                (df['diastolic_bp'] >= 90)).astype(int)


#     df['total_cholesterol'] = df['ldl'] + df['hdl'] + (df['triglycerides'] / 5)
#     df['ldl_hdl_ratio'] = df['ldl'] / df['hdl']
#     df['tg_hdl_ratio'] = df['triglycerides'] / df['hdl']
#     df['non_hdl'] = df['total_cholesterol'] - df['hdl']
#     df['anaemia_flag'] = (df['haemoglobin'] < 12).astype(int)
#     df['microcytosis_flag'] = (df['mcv'] < 80).astype(int)
#     df['hb_mcv_ratio'] = df['haemoglobin'] / df['mcv']


#     df['risk_score'] = (
#         (df['blood_glucose']/200) +
#         (df['ldl']/160) +
#         (df['triglycerides']/200) +
#         (df['systolic_bp']/140)
#     )

#     return df





# columns = [
#     "Blood_glucose",
#     "HbA1C",
#     "Systolic_BP",
#     "Diastolic_BP",
#     "LDL",
#     "HDL",
#     "Triglycerides",
#     "Haemoglobin",
#     "MCV"
# ]


# data = [
#     [107.38, 4.93, 109.25, 74.10, 129.20, 52.11, 68.84, 10.17, 61.54] #user input
# ]


# df_input = pd.DataFrame(data, columns=columns)

# df_input = Feature_Extraction(df_input)
# result = pipeline.predict(df_input)
# print("النتيجة:", result)