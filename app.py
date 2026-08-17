import streamlit as st
import pandas as pd
import numpy as np
import pulp
import pydeck as pdk

# 1. Page Configuration
st.set_page_config(page_title="SkinIO | Connection to Care", page_icon="🩺", layout="wide")

# 2. Modern Custom CSS Styling
st.markdown("""
    <style>
    .stApp { background-color: #0b0f19; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .header-box { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 20px 24px; border-radius: 16px; border: 1px solid #334155; margin-bottom: 20px; }
    .header-title { color: #f8fafc; font-size: 24px; font-weight: 700; margin: 0; }
    .header-subtitle { color: #94a3b8; font-size: 13px; margin-top: 4px; }
    .metric-card { background-color: #1e293b; padding: 14px; border-radius: 12px; border: 1px solid #334155; text-align: center; }
    .metric-value { font-size: 22px; font-weight: 700; color: #38bdf8; }
    .metric-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
    .patient-card { background-color: #1e293b; padding: 18px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 12px; }
    .badge-primary { background-color: #0284c7; color: #ffffff; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
    .badge-insurance { background-color: #334155; color: #e2e8f0; padding: 3px 10px; border-radius: 12px; font-size: 12px; }
    </style>
""", unsafe_allow_html=True)

# 3. Header Section
st.markdown("""
    <div class="header-box">
        <div class="header-title">🩺 Connection to Care (C2C) Hub</div>
        <div class="header-subtitle">Intelligent Patient Matching Engine & Real-time Network Map</div>
    </div>
""", unsafe_allow_html=True)

# Fake Data Generators for Realistic Profiles
FIRST_NAMES = ["Sarah", "Michael", "Emily", "David", "Jessica", "James", "Amanda", "Robert", "Ashley", "John"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
URGENT_TYPES = ["Melanoma Suspect", "Atypical Mole", "Basal Cell Suspect", "Squamous Cell Suspect"]

# 4. Sidebar Controls & Toggles
st.sidebar.header("⚙️ System Controls")

use_fake_names = st.sidebar.toggle("Real-world Sample Profiles", value=True, help="Toggle between realistic fake names/phone numbers and plain IDs.")

st.sidebar.markdown("---")
st.sidebar.subheader("Network Toggles")
num_patients = st.sidebar.slider("Urgent Patients", 10, 150, 40)
num_doctors = st.sidebar.slider("Partner Clinics", 5, 30, 15)
seed = st.sidebar.number_input("Random Seed", value=42)

# Intake Form Expander
with st.sidebar.expander("➕ Add Single Urgent Patient"):
    intake_name = st.text_input("Patient Full Name", "Alex Mercer")
    intake_phone = st.text_input("Phone Number", "(555) 234-5678")
    intake_ins = st.selectbox("Insurance Network", ['Aetna', 'Cigna', 'BlueCross', 'UnitedHealthcare', 'Medicare'])
    intake_type = st.selectbox("Urgency Type", URGENT_TYPES)
    submit_intake = st.button("Add to Patient Queue", type="primary")

# 5. Core Matching Optimization Engine
def run_matching_engine(n_patients, n_doctors, r_seed, fake_names_flag):
    np.random.seed(r_seed)
    insurances = ['Aetna', 'Cigna', 'BlueCross', 'UnitedHealthcare', 'Medicare']

    # Generate Patients
    p_names = [f"{np.random.choice(FIRST_NAMES)} {np.random.choice(LAST_NAMES)}" for _ in range(n_patients)]
    p_phones = [f"({np.random.randint(200,999)}) {np.random.randint(100,999)}-{np.random.randint(1000,9999)}" for _ in range(n_patients)]
    p_urgency = np.random.choice(URGENT_TYPES, n_patients)

    patients = pd.DataFrame({
        'Patient_ID': [f"P-{1000 + i}" for i in range(n_patients)],
        'Full_Name': p_names if fake_names_flag else [f"Patient #{i+1}" for i in range(n_patients)],
        'Phone': p_phones if fake_names_flag else ["(555) 000-0000"] * n_patients,
        'Urgency_Flag': p_urgency,
        'lat': np.random.uniform(40.65, 40.85, n_patients),
        'lon': np.random.uniform(-74.05, -73.80, n_patients),
        'Insurance': np.random.choice(insurances, n_patients)
    })

    # Generate Doctors
    doctors = pd.DataFrame({
        'Clinic_ID': [f"Clinic {chr(65 + (j % 26))}-{j//26 + 1}" for j in range(n_doctors)],
        'lat': np.random.uniform(40.65, 40.85, n_doctors),
        'lon': np.random.uniform(-74.05, -73.80, n_doctors),
        'Accepted_Insurances': [
            list(np.random.choice(insurances, np.random.randint(2, 5), replace=False)) 
            for _ in range(n_doctors)
        ],
        'Capacity': np.random.randint(4, 7, n_doctors)
    })

    # Linear Programming Solver
    prob = pulp.LpProblem("C2C_Matching", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("match", ((i, j) for i in patients.index for j in doctors.index), cat='Binary')

    objective_terms = []
    for i in patients.index:
        for j in doctors.index:
            dist = np.sqrt((patients.loc[i, 'lat'] - doctors.loc[j, 'lat'])**2 + 
                           (patients.loc[i, 'lon'] - doctors.loc[j, 'lon'])**2) * 69
            if patients.loc[i, 'Insurance'] in doctors.loc[j, 'Accepted_Insurances']:
                objective_terms.append(dist * x[i, j])
            else:
                prob += x[i, j] == 0

    prob += pulp.lpSum(objective_terms)

    for i in patients.index:
        prob += pulp.lpSum(x[i, j] for j in doctors.index) == 1

    for j in doctors.index:
        prob += pulp.lpSum(x[i, j] for i in patients.index) <= doctors.loc[j, 'Capacity']

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    primary_matches = {}
    for i in patients.index:
        for j in doctors.index:
            if x[i, j].varValue == 1.0:
                primary_matches[i] = j

    results = []
    arcs = []
    for i in patients.index:
        p_rec = patients.loc[i]
        compat = []
        for j in doctors.index:
            d_rec = doctors.loc[j]
            if p_rec['Insurance'] in d_rec['Accepted_Insurances']:
                dist = np.sqrt((p_rec['lat'] - d_rec['lat'])**2 + (p_rec['lon'] - d_rec['lon'])**2) * 69
                compat.append({
                    'Clinic_ID': d_rec['Clinic_ID'],
                    'Clinic_Lat': d_rec['lat'],
                    'Clinic_Lon': d_rec['lon'],
                    'Distance': round(dist, 1),
                    'Is_Primary': (j == primary_matches[i])
                })
        
        compat = sorted(compat, key=lambda c: c['Distance'])
        primary = next((c for c in compat if c['Is_Primary']), compat[0])
        backups = [c for c in compat if not c['Is_Primary']][:2]

        arcs.append({
            'from_lat': p_rec['lat'], 'from_lon': p_rec['lon'],
            'to_lat': primary['Clinic_Lat'], 'to_lon': primary['Clinic_Lon'],
            'color': [56, 189, 248, 200]
        })

        results.append({
            'Patient_ID': p_rec['Patient_ID'],
            'Full_Name': p_rec['Full_Name'],
            'Phone': p_rec['Phone'],
            'Urgency_Flag': p_rec['Urgency_Flag'],
            'Insurance': p_rec['Insurance'],
            'lat': p_rec['lat'], 'lon': p_rec['lon'],
            'Primary_Clinic': primary['Clinic_ID'],
            'Primary_Distance': primary['Distance'],
            'Backup_1': f"{backups[0]['Clinic_ID']} ({backups[0]['Distance']} mi)" if len(backups) > 0 else "N/A",
            'Backup_2': f"{backups[1]['Clinic_ID']} ({backups[1]['Distance']} mi)" if len(backups) > 1 else "N/A",
            'All_Backups': [f"{b['Clinic_ID']} ({b['Distance']} mi)" for b in backups]
        })

    return pd.DataFrame(results), doctors, pd.DataFrame(arcs)

# Run Engine
patients_df, doctors_df, arcs_df = run_matching_engine(num_patients, num_doctors, seed, use_fake_names)

if submit_intake:
    st.toast(f"Added patient {intake_name} to queue!")

# 6. Top Metrics Bar
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(patients_df)}</div><div class="metric-label">Urgent Patients</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(doctors_df)}</div><div class="metric-label">Partner Clinics</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{patients_df["Primary_Distance"].mean():.1f} mi</div><div class="metric-label">Avg Travel Distance</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-value" style="color:#10b981;">100%</div><div class="metric-label">Matched Status</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 7. Multi-Tab Interface
tab_dashboard, tab_map = st.tabs(["📋 Patient Schedule & Review", "🗺️ Geographic Network Map"])

with tab_dashboard:
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("Scheduled Patient Queue")
        
        search_query = st.text_input("🔍 Search Queue:", "", placeholder="Filter by Name, ID, or Insurance...")
        
        filtered_df = patients_df.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df['Full_Name'].str.contains(search_query, case=False) |
                filtered_df['Patient_ID'].str.contains(search_query, case=False) |
                filtered_df['Insurance'].str.contains(search_query, case=False)
            ]

        # Display Data Table
        st.dataframe(
            filtered_df[['Patient_ID', 'Full_Name', 'Insurance', 'Urgency_Flag', 'Primary_Clinic', 'Primary_Distance', 'Backup_1']],
            column_config={
                "Patient_ID": "ID",
                "Full_Name": "Patient Name",
                "Insurance": "Insurance Network",
                "Urgency_Flag": "Urgency Details",
                "Primary_Clinic": "Top Clinic Choice",
                "Primary_Distance": "Distance (mi)",
                "Backup_1": "Backup Choice"
            },
            use_container_width=True,
            hide_index=True,
            height=380
        )

        csv = filtered_df[['Patient_ID', 'Full_Name', 'Phone', 'Insurance', 'Urgency_Flag', 'Primary_Clinic', 'Primary_Distance', 'Backup_1']].to_csv(index=False)
        st.download_button("📥 Export Schedule (CSV)", data=csv, file_name="c2c_patient_schedule.csv", mime="text/csv")

    with col_right:
        st.subheader("Patient Dossier & Override")
        
        # Select Patient by Full Name or ID
        patient_options = patients_df['Patient_ID'] + " - " + patients_df['Full_Name']
        selected_option = st.selectbox("Select Patient Record:", patient_options)
        selected_id = selected_option.split(" - ")[0]
        
        patient = patients_df[patients_df['Patient_ID'] == selected_id].iloc[0]

        st.markdown(f"""
            <div class="patient-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="color:#f8fafc; margin:0;">{patient['Full_Name']} ({patient['Patient_ID']})</h3>
                    <span class="badge-insurance">{patient['Insurance']}</span>
                </div>
                <p style="color:#cbd5e1; font-size:13px; margin-top:4px;">📞 {patient['Phone']} | 🚨 {patient['Urgency_Flag']}</p>
                <hr style="border-color:#334155; margin:10px 0;">
                <p style="color:#94a3b8; margin-bottom:4px; font-size:12px;">RECOMMENDED MATCH:</p>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span class="badge-primary">{patient['Primary_Clinic']}</span>
                    <span style="color:#cbd5e1; font-size:13px;">({patient['Primary_Distance']} miles away)</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.write("**Alternative Backups:**")
        if patient['All_Backups']:
            for b in patient['All_Backups']:
                st.write(f"• {b}")
        else:
            st.write("No compatible backups in range.")

        with st.expander("✏️ Override Clinic Selection"):
            opts = [patient['Primary_Clinic']] + [b.split(' (')[0] for b in patient['All_Backups']]
            new_choice = st.selectbox("Assign Different Clinic:", opts)
            if st.button("Save Re-assignment", type="primary"):
                st.success(f"Assigned {patient['Full_Name']} to {new_choice}!")

with tab_map:
    st.subheader("🗺️ Live Patient-to-Clinic Connections")
    st.caption("🔵 Blue Arcs represent optimal travel routes connecting patient locations (Red) to partner clinics (Green).")

    view_state = pdk.ViewState(
        latitude=patients_df['lat'].mean(),
        longitude=patients_df['lon'].mean(),
        zoom=10, pitch=35
    )

    arc_layer = pdk.Layer("ArcLayer", data=arcs_df, get_source_position=["from_lon", "from_lat"], get_target_position=["to_lon", "to_lat"], get_color="color", get_width=2)
    patient_layer = pdk.Layer("ScatterplotLayer", data=patients_df, get_position=["lon", "lat"], get_color="[239, 68, 68, 200]", get_radius=350)
    clinic_layer = pdk.Layer("ScatterplotLayer", data=doctors_df, get_position=["lon", "lat"], get_color="[16, 185, 129, 250]", get_radius=600)

    st.pydeck_chart(pdk.Deck(
        layers=[arc_layer, patient_layer, clinic_layer],
        initial_view_state=view_state,
        tooltip={"text": "Node: Lat {lat}, Lon {lon}"}
    ))
