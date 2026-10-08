# add sa ug libraries and tools para unya, with initialization 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# stremlit title ra ni
st.set_page_config(page_title="LazyBugs NIDS", page_icon="lazybugs.png", layout="wide")

# define targets, automatic niya i detect ang target if naa same na column na sa sulod sa list 
TARGETS = ["class", "label", "target", "attack", "intrusion"]

# seeder or alternative data if walay csv na i import
def synthetic_data(n=2500, seed=42):
    rng = np.random.default_rng(seed)
    protocol = rng.choice(["tcp", "udp", "icmp"], n, p=[.72, .18, .10])
    service = rng.choice(["http", "private", "ftp_data", "smtp", "domain_u"], n,
                         p=[.35, .25, .12, .13, .15])
    flag = rng.choice(["SF", "S0", "REJ", "RSTR"], n, p=[.60, .22, .12, .06])
    duration = rng.exponential(2.5, n)
    src = rng.lognormal(6, 1.6, n).astype(int)
    dst = rng.lognormal(6.5, 1.7, n).astype(int)
    count = rng.integers(1, 250, n)
    risk = ((src > np.percentile(src, 75)).astype(int) +
            (dst < np.percentile(dst, 25)).astype(int) +
            (count > 150).astype(int) + (protocol == "icmp").astype(int) +
            (flag == "S0").astype(int) + (flag == "REJ").astype(int) +
            (service == "private").astype(int))
    p = np.clip(.08 + risk * .13, .03, .90)
    anomaly = rng.random(n) < p
    return pd.DataFrame({
        "duration": duration.round(3), "protocol_type": protocol,
        "service": service, "flag": flag, "src_bytes": src,
        "dst_bytes": dst, "count": count,
        "class": np.where(anomaly, "anomaly", "normal")
    })

#
def generate_single_random_packet(X_ref):
    """Generates random attributes cast safely as clean Python elements to keep UI states stable."""
    row = {}
    if "duration" in X_ref:
        row["duration"] = float(np.random.exponential(X_ref["duration"].mean() or 2.5))
    if "protocol_type" in X_ref:
        row["protocol_type"] = str(np.random.choice(X_ref["protocol_type"].dropna().unique()))
    if "src_bytes" in X_ref:
        q = int(X_ref["src_bytes"].quantile(0.95))
        row["src_bytes"] = int(np.random.randint(10, max(1000, q)))
    if "service" in X_ref:
        row["service"] = str(np.random.choice(X_ref["service"].dropna().unique()))
    if "dst_bytes" in X_ref:
        q = int(X_ref["dst_bytes"].quantile(0.95))
        row["dst_bytes"] = int(np.random.randint(10, max(1000, q)))
    if "count" in X_ref:
        mx = int(X_ref["count"].max() or 250)
        row["count"] = int(np.random.randint(1, max(10, mx)))
    if "flag" in X_ref:
        row["flag"] = str(np.random.choice(X_ref["flag"].dropna().unique()))
    return row

# suggest ug ways to protect from this kind of attack if naay ma detect ps. just a common ways not so technical
def generate_playbook(row_df, anomaly_score):
    protocol = str(row_df["protocol_type"].iloc[0]).upper() if "protocol_type" in row_df.columns else "UNKNOWN"
    service = str(row_df["service"].iloc[0]).lower() if "service" in row_df.columns else "unknown"
    src_bytes = int(row_df["src_bytes"].iloc[0]) if "src_bytes" in row_df.columns else 0
    count = int(row_df["count"].iloc[0]) if "count" in row_df.columns else 0
    
    st.markdown("### Security Incident Response Playbook")
    st.info(f"**Calculated Threat Urgency Status:** {anomaly_score:.2%} Confidence Index")
    
    # if ICMP kind na attack ang ma detect
    if protocol == "ICMP":
        st.error("**Vector Assessment: ICMP Flood / Subnet Scan Sweeping**")
        st.markdown("""
        * **1. Perimeter Containment:** Drop incoming echo-requests from the source gateway at the border firewall instantly.
        * **2. Rate-Limiting Configuration:** Implement an active network threshold rule limiting ICMP traffic to a maximum of 5 packets/sec per node.
        * **3. Investigation Focus:** Audit host logs for mapping diagnostics or subnet sweeping signatures.
        """)
    # if greater than 50k source bytes ang ma recieved ug more than 150 amount kay ma detect na DDoS
    # ps. 50k recieved from 150 pockets is suspicious, and that amount of bytes in short time can crash device/services
    elif src_bytes > 50000 and count > 150:
        st.error("**Vector Assessment: Distributed Denial of Service (DDoS) / Infiltration**")
        st.markdown("""
        * **1. Bandwidth Mitigation:** Trigger high-volume DDoS mitigation traffic scrubbing profiles (e.g., Cloudflare/AWS Shield rules).
        * **2. Connection Throttling:** Drop concurrent TCP handshakes targeting service components if the protocol flag remains un-established.
        * **3. Load Balancing:** Route ingress connections to alternate application node clusters to preserve uptime.
        """)
    
    # prevent niya ang suspicious reconnaissance
    #ps. reconnaissance is the first step to exploitation, gina scan ang port para maka kita ug weakness
    elif service == "private":
        st.warning("**Vector Assessment: Unauthorized Port Scan / Private Subnet Enumeration**")
        st.markdown("""
        * **1. Access Control Control:** Quarantine the initiating host system from internal network directory vectors.
        * **2. Security Audit:** Review IAM credentials and service ports mapping files to evaluate vulnerability exposure.
        * **3. Firewall Update:** Close unused default development ports on outer-facing proxy appliances.
        """)
    
    else:
        st.error("**Vector Assessment: General Anomaly Matrix Disruption**")
        st.markdown("""
        * **1. Host Isolation:** Temporarily block communication strings between the target IP addresses.
        * **2. Log Capture:** Dump system telemetry lines into a dedicated archive for deeper Wireshark capture evaluation.
        """)

#identify target
def target_column(df):
    lookup = {str(c).lower(): c for c in df.columns}
    return next((lookup[x] for x in TARGETS if x in lookup), None)

#identify ang classes exp.(anomaly/normal)
def encode_target(y):
    if y.dtype == object or str(y.dtype).startswith("string"):
        s = y.astype(str).str.strip().str.lower()
        anomaly = {"anomaly", "attack", "attacked", "malicious", "intrusion", "intrusive", "1", "yes", "true"}
        normal = {"normal", "benign", "safe", "0", "no", "false"}
        mapped = s.map(lambda x: 0 if x in anomaly else 1 if x in normal else np.nan)
        if mapped.notna().all() and mapped.nunique() == 2:
            return mapped.astype(int)
        codes, uniques = pd.factorize(s)
        if len(uniques) != 2:
            raise ValueError("Target must contain exactly two classes.")
        return pd.Series(codes, index=y.index).astype(int)
    vals = sorted(pd.Series(y).dropna().unique())
    if len(vals) != 2:
        raise ValueError("Target must contain exactly two classes.")
    if set(vals) == {0, 1}:
        return y.astype(int)
    return y.map({vals[0]: 0, vals[1]: 1}).astype(int)

#fetch selected type of classifier (randomforest/logisticregression/decisiontree)
def classifier(name):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=5000, random_state=42)
    return DecisionTreeClassifier(random_state=42)

#start traning the machine and catches it, para if mag change ug classifier dili na mag strain balik
@st.cache_resource(show_spinner="Training model...")
#train ang data file gamit ang selected model
def train(df, model_name):
    df = df.copy()
    target = target_column(df)
    if target is None:
        raise ValueError("No target column found. Use class, label, target, attack, or intrusion.")
    y = encode_target(df[target])
    X = df.drop(columns=[target]).dropna(axis=1, how="all")
    cat = X.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    transformers = []
    if num:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat))
    
    preprocessor = ColumnTransformer(transformers)
    pipe = Pipeline([("preprocessor", preprocessor),
                     ("classifier", classifier(model_name))])
    
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    
    clf_step = pipe.named_steps["classifier"]
    if hasattr(clf_step, "predict_proba"):
        pred_proba = pipe.predict_proba(Xte)[:, 0]
    else:
        pred_proba = np.where(pred == 0, 1.0, 0.0)
    
    try:
        feat_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                feat_names.extend(cols)
            elif name == "cat" and hasattr(trans.named_steps["encoder"], "get_feature_names_out"):
                feat_names.extend(trans.named_steps["encoder"].get_feature_names_out(cols))
    except Exception:
        feat_names = [f"Feature {i}" for i in range(pipe.named_steps["preprocessor"].transform(Xtr).shape[1])]

    return {"pipe": pipe, "X": X, "Xtr": Xtr, "Xte": Xte, "yte": yte, "pred": pred, "pred_proba": pred_proba,
            "accuracy": accuracy_score(yte, pred), "cm": confusion_matrix(yte, pred, labels=[0, 1]),
            "report": classification_report(yte, pred, labels=[0, 1], target_names=["Anomaly", "Normal"], output_dict=True, zero_division=0),
            "target": target, "cat": cat, "num": num, "feat_names": feat_names}


def custom_row(values, X):
    row = {}
    for c in X.columns:
        if c in values:
            row[c] = values[c]
        elif pd.api.types.is_numeric_dtype(X[c]):
            v = pd.to_numeric(X[c], errors="coerce").median()
            row[c] = 0 if pd.isna(v) else v
        else:
            mode = X[c].dropna().astype(str).mode()
            row[c] = mode.iloc[0] if not mode.empty else ""
    return pd.DataFrame([row], columns=X.columns)


def custom_controls(X):
    vals = {}
    a, b, c = st.columns(3)
    
    with a:
        if "duration" in X:
            vals["duration"] = st.number_input("Duration", min_value=0.0, step=0.5, value=0.0, key="widget_duration")
        if "protocol_type" in X:
            opts = sorted(X["protocol_type"].dropna().astype(str).unique())
            vals["protocol_type"] = st.selectbox("Protocol Type", opts or ["tcp"], key="widget_protocol")
    with b:
        if "src_bytes" in X:
            q = pd.to_numeric(X["src_bytes"], errors="coerce").quantile(.99)
            vals["src_bytes"] = st.number_input("Source Bytes", min_value=0, max_value=max(1000, int(q or 100000)), step=10, value=100, key="widget_src_bytes")
        if "service" in X:
            opts = sorted(X["service"].dropna().astype(str).unique())
            vals["service"] = st.selectbox("Service", opts or ["http"], key="widget_service")
    with c:
        if "dst_bytes" in X:
            q = pd.to_numeric(X["dst_bytes"], errors="coerce").quantile(.99)
            vals["dst_bytes"] = st.number_input("Destination Bytes", min_value=0, max_value=max(1000, int(q or 100000)), step=10, value=100, key="widget_dst_bytes")
        if "count" in X:
            mx = max(100, int(pd.to_numeric(X["count"], errors="coerce").max() or 500))
            vals["count"] = st.number_input("Connection Count", min_value=0, max_value=mx, step=1, value=10, key="widget_count")
            
    return vals


st.title("Network Intrusion Detection System")

st.sidebar.header("Configuration")
upload = st.sidebar.file_uploader("Upload CSV dataset", type="csv")
model_name = st.sidebar.selectbox("ML Model", ["Random Forest", "Logistic Regression", "Decision Tree"])

st.sidebar.subheader("Security Sensitivity")
threshold = st.sidebar.slider(
    "Anomaly Score Cutoff (Threshold)", 
    min_value=0.05, max_value=0.95, value=0.50, step=0.05,
    help="Lower values trigger high security mode (fewer missed threats but more false alarms)."
)

if upload:
    try:
        df = pd.read_csv(upload)
        if target_column(df) is None:
            st.sidebar.warning("No recognized target column; using synthetic fallback.")
            df = synthetic_data()
            source = "Synthetic fallback"
        else:
            source = "Uploaded CSV"
    except Exception as e:
        st.sidebar.error(f"CSV error: {e}")
        df, source = synthetic_data(), "Synthetic fallback"
else:
    df, source = synthetic_data(), "Synthetic network traffic"

st.sidebar.success(f"Source: {source}")

try:
    r = train(df, model_name)
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()

baseline_anomalies = (r["pred_proba"] >= threshold).sum()
baseline_normals = len(r["pred_proba"]) - baseline_anomalies

a, b, c = st.columns(3)
a.metric("Total Packets Analyzed", f"{len(r['Xte']):,}")
b.metric("Threats Detected (Dynamic)", f"{baseline_anomalies:,}")
c.metric("Normal Traffic Count (Dynamic)", f"{baseline_normals:,}")

with st.expander("Dataset Information"):
    x, y, z = st.columns(3)
    x.write(f"Rows: **{len(df):,}**")
    x.write(f"Columns: **{df.shape}**")
    x.write(f"Target: **{r['target']}**")
    y.write(f"Categorical: **{len(r['cat'])}**")
    y.write(f"Numerical: **{len(r['num'])}**")
    z.write(f"Training: **{len(r['Xtr']):,}**")
    z.write(f"Validation: **{len(r['Xte']):,}**")

st.header(f"{model_name} Evaluation Suite")

t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "Accuracy Dashboard", 
    "Confusion Matrix", 
    "Classification Report", 
    "Feature Drivers", 
    "ROC Curve",
    "Protocol Distribution",
    "Traffic Outliers"
])

with t1:
    st.subheader("Model Diagnostic Score")
    st.metric("Validation Accuracy", f"{r['accuracy']:.3%}")
    st.progress(float(r["accuracy"]))
    
with t2:
    st.subheader("Confusion Matrix Breakdown")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", xticklabels=["Anomaly", "Normal"], yticklabels=["Anomaly", "Normal"], ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    st.pyplot(fig)
    plt.close(fig)
    
with t3:
    st.subheader("Detailed Classification Analysis")
    st.dataframe(pd.DataFrame(r["report"]).T.round(3), use_container_width=True)
    
with t4:
    st.subheader("Key Feature Importance Predictors")
    clf_step = r["pipe"].named_steps["classifier"]
    importances = None
    title_label = "Feature Importance"
    
    if hasattr(clf_step, "feature_importances_"):
        importances = clf_step.feature_importances_
    elif hasattr(clf_step, "coef_"):
        importances = np.abs(clf_step.coef_[0]) if clf_step.coef_.ndim > 1 else np.abs(clf_step.coef_)
        title_label = "Feature Importance (Absolute Coefficients)"
        
    if importances is not None:
        imp_df = pd.DataFrame({"Feature": r["feat_names"], "Importance": importances})
        imp_df = imp_df.sort_values(by="Importance", ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=imp_df, x="Importance", y="Feature", palette="viridis", ax=ax)
        ax.set_title(f"Top Drivers — {model_name} ({title_label})")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Feature impact tracking is unavailable for the selected architecture profile.")

with t5:
    st.subheader("Receiver Operating Characteristic (ROC)")
    y_true_roc = np.where(r["yte"] == 0, 1, 0)
    fpr, tpr, _ = roc_curve(y_true_roc, r["pred_proba"])
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("NIDS Anomaly Discovery Trajectory")
    ax.legend(loc="lower right")
    st.pyplot(fig)
    plt.close(fig)

with t6:
    st.subheader("Protocol vs. Threat Vector Landscape")
    if "protocol_type" in df.columns:
        target_col_name = target_column(df)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.countplot(data=df, x="protocol_type", hue=target_col_name, palette="Set2", ax=ax)
        ax.set_title("Protocol Traffic Volume Stratified by Classification")
        ax.set_xlabel("Protocol Architecture Profile")
        ax.set_ylabel("Log Entry Records Count")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Protocol mapping fields missing from current vector schema.")

with t7:
    st.subheader("Ingress/Egress Byte Throughput Distribution")
    if "src_bytes" in df.columns and "dst_bytes" in df.columns:
        target_col_name = target_column(df)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.scatterplot(data=df, x="src_bytes", y="dst_bytes", hue=target_col_name, alpha=0.5, palette="coolwarm", ax=ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Network Payload Distribution (Logarithmic Scale)")
        ax.set_xlabel("Source Output Bytes (src_bytes)")
        ax.set_ylabel("Destination Input Bytes (dst_bytes)")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Byte dimension parameters not found in parsing structures.")

st.divider()
st.header("Live Security Operations Center (SOC) Simulation")
col1, col2 = st.columns(2)
with col1:
    stream_active = st.toggle("Activate Live Traffic Intake", value=False)
    stream_speed = st.slider("Transmission Loop Speed (seconds)", min_value=0.2, max_value=2.0, value=0.5, step=0.1)
    clear_logs = st.button("Reset Intake Register Logs")

if "soc_stream_log" not in st.session_state or clear_logs:
    st.session_state["soc_stream_log"] = []

log_container = st.empty()

if stream_active:
    for _ in range(10):
        packet_raw = generate_single_random_packet(r["X"])
        packet_df = custom_row(packet_raw, r["X"])
        
        clf_step = r["pipe"].named_steps["classifier"]
        if hasattr(clf_step, "predict_proba"):
            probs = r["pipe"].predict_proba(packet_df)[0]
            classes = clf_step.classes_
            pm = {int(k): float(v) for k, v in zip(classes, probs)}
            anomaly_score = pm.get(0, 0.0)
        else:
            anomaly_score = 1.0 if int(r["pipe"].predict(packet_df)[0]) == 0 else 0.0
            
        classification_result = "MALICIOUS ANOMALY" if anomaly_score >= threshold else "NORMAL TRAFFIC"
        timestamp = time.strftime("%H:%M:%S")
        
        entry = {
            "Timestamp": timestamp,
            "Protocol": packet_raw.get("protocol_type", "tcp").upper(),
            "Service": packet_raw.get("service", "http"),
            "Src Bytes": packet_raw.get("src_bytes", 0),
            "Dst Bytes": packet_raw.get("dst_bytes", 0),
            "Conn Count": packet_raw.get("count", 1),
            "Anomaly Index": f"{anomaly_score:.1%}",
            "Disposition": classification_result
        }
        
        st.session_state["soc_stream_log"].insert(0, entry)
        if len(st.session_state["soc_stream_log"]) > 40:
            st.session_state["soc_stream_log"].pop()
            
        log_df = pd.DataFrame(st.session_state["soc_stream_log"])
        
        with log_container.container():
            st.dataframe(log_df, use_container_width=True, hide_index=True)
        time.sleep(stream_speed)
else:
    if st.session_state["soc_stream_log"]:
        log_df = pd.DataFrame(st.session_state["soc_stream_log"])
        log_container.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        log_container.info("Traffic ingest pipeline offline.")

st.divider()
st.header("Single Connection Inspection Matrix")

ins1, ins2 = st.tabs(["Instant Random Simulation", "Manual Parameter Tweaking"])

with ins1:
    st.subheader("Automated Packet Simulation Diagnostics")
    st.markdown("Click below to randomly assemble attributes from historical distributions, run them against the classifier, and inspect metrics immediately.")
    
    if st.button("Generate & Analyze Random Connection Pipeline", type="primary", use_container_width=True):
        # 1. Generate clean payload data
        packet_raw = generate_single_random_packet(r["X"])
        row = custom_row(packet_raw, r["X"])
        
        # 2. Present payload table instantly to the analyst
        st.markdown("##### Simulated Connection Payload Features")
        st.dataframe(pd.DataFrame([packet_raw]), hide_index=True, use_container_width=True)
        
        # 3. Calculate inference metrics safely
        clf_step = r["pipe"].named_steps["classifier"]
        if hasattr(clf_step, "predict_proba"):
            probs = r["pipe"].predict_proba(row)[0]
            classes = clf_step.classes_
            pm = {int(k): float(v) for k, v in zip(classes, probs)}
            anomaly_p, normal_p = pm.get(0, 0.0), pm.get(1, 0.0)
        else:
            raw_p = int(r["pipe"].predict(row)[0])
            anomaly_p, normal_p = (1.0, 0.0) if raw_p == 0 else (0.0, 1.0)
            
        is_anomaly = anomaly_p >= threshold
            
        # 4. Immediate state rendering UI blocks
        if is_anomaly:
            st.error("CRITICAL MALICIOUS INTRUSION REJECTED")
        else:
            st.success("VERIFIED SECURE CONNECTION TRAFFIC")
            
        x, y = st.columns(2)
        x.metric("Anomaly Probability Index", f"{anomaly_p:.2%}")
        x.progress(anomaly_p)
        y.metric("Normal Safety Index", f"{normal_p:.2%}")
        y.progress(normal_p)
        
        if is_anomaly:
            st.divider()
            generate_playbook(row, anomaly_p)

with ins2:
    st.subheader("Manual Inspection Matrix Parameters")
    vals = custom_controls(r["X"])
    if st.button("Analyze Custom Configuration Traffic", type="secondary", use_container_width=True):
        row = custom_row(vals, r["X"])
        
        clf_step = r["pipe"].named_steps["classifier"]
        if hasattr(clf_step, "predict_proba"):
            probs = r["pipe"].predict_proba(row)[0]
            classes = clf_step.classes_
            pm = {int(k): float(v) for k, v in zip(classes, probs)}
            anomaly_p, normal_p = pm.get(0, 0.0), pm.get(1, 0.0)
        else:
            raw_p = int(r["pipe"].predict(row)[0])
            anomaly_p, normal_p = (1.0, 0.0) if raw_p == 0 else (0.0, 1.0)
            
        is_anomaly = anomaly_p >= threshold
            
        if is_anomaly:
            st.error("CRITICAL MALICIOUS INTRUSION BLOCKED")
        else:
            st.success("VERIFIED SECURE CONNECTION TRAFFIC")
            
        x, y = st.columns(2)
        x.metric("Anomaly Probability", f"{anomaly_p:.2%}")
        x.progress(anomaly_p)
        y.metric("Normal Probability", f"{normal_p:.2%}")
        y.progress(normal_p)
        
        if is_anomaly:
            st.divider()
            generate_playbook(row, anomaly_p)

#unsupervise diri
st.divider()
st.header("Unsupervised Data Production Batch Inference")
st.markdown("Upload completely unlabeled data (unsupervised CSV) here to calculate risks using your trained model pipeline and sensitivity thresholds.")


def prepare_batch(raw, X_ref):
    """Align an unlabeled dataframe to the training feature schema.
    Returns (aligned_df, missing_cols, ignored_cols)."""
    raw = raw.copy()
    # drop target-like column if the file happens to contain one
    tcol = target_column(raw)
    if tcol is not None:
        raw = raw.drop(columns=[tcol])
    # case-insensitive / whitespace-tolerant column matching
    ref_lookup = {str(c).strip().lower(): c for c in X_ref.columns}
    raw.columns = [ref_lookup.get(str(c).strip().lower(), c) for c in raw.columns]
    missing = [c for c in X_ref.columns if c not in raw.columns]
    ignored = [c for c in raw.columns if c not in X_ref.columns]
    aligned = raw.reindex(columns=X_ref.columns)
    # coerce numeric columns; unseen/missing values are handled by the pipeline imputers
    for c in X_ref.columns:
        if pd.api.types.is_numeric_dtype(X_ref[c]):
            aligned[c] = pd.to_numeric(aligned[c], errors="coerce")
        else:
            aligned[c] = aligned[c].astype("object").where(aligned[c].notna(), np.nan)
    return aligned, missing, ignored


def predict_batch(pipe, batch):
    """Return anomaly probability (class 0) for every row."""
    clf_step = pipe.named_steps["classifier"]
    if hasattr(clf_step, "predict_proba"):
        probs = pipe.predict_proba(batch)
        idx = list(clf_step.classes_).index(0)
        return probs[:, idx]
    return (pipe.predict(batch) == 0).astype(float)


batch_upload = st.file_uploader("Upload unlabeled CSV for prediction", type="csv", key="batch_unlabeled_csv")

if batch_upload is not None:
    try:
        raw_batch = pd.read_csv(batch_upload)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        raw_batch = None

    if raw_batch is not None:
        if raw_batch.empty:
            st.warning("The uploaded file has no rows.")
        else:
            batch, missing, ignored = prepare_batch(raw_batch, r["X"])

            if len(missing) == len(r["X"].columns):
                st.error("None of the uploaded columns match the training features. "
                         f"Expected columns: {', '.join(map(str, r['X'].columns))}")
            else:
                if missing:
                    st.warning(f"Missing columns filled using training-set median/most-frequent values: {', '.join(map(str, missing))}")
                if ignored:
                    st.info(f"Ignored columns not used by the model: {', '.join(map(str, ignored))}")

                scores = predict_batch(r["pipe"], batch)
                result = raw_batch.copy()
                result["Anomaly Probability"] = np.round(scores, 4)
                result["Prediction"] = np.where(scores >= threshold, "ANOMALY", "NORMAL")

                n_anom = int((scores >= threshold).sum())
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows Scored", f"{len(result):,}")
                m2.metric("Threats Detected", f"{n_anom:,}")
                m3.metric("Normal Traffic", f"{len(result) - n_anom:,}")
                m4.metric("Threat Rate", f"{n_anom / len(result):.1%}")
                st.caption(f"Model: {model_name} | Threshold: {threshold:.2f}")

                view = st.radio("Show", ["All rows", "Anomalies only", "Normal only"], horizontal=True, key="batch_view")
                shown = result
                if view == "Anomalies only":
                    shown = result[result["Prediction"] == "ANOMALY"]
                elif view == "Normal only":
                    shown = result[result["Prediction"] == "NORMAL"]
                st.dataframe(shown.sort_values("Anomaly Probability", ascending=False),
                             use_container_width=True, hide_index=True)

                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.hist(scores, bins=30, color="steelblue", edgecolor="white")
                ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.2f}")
                ax.set_xlabel("Anomaly Probability")
                ax.set_ylabel("Rows")
                ax.set_title("Anomaly Score Distribution (Uploaded Batch)")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

                st.download_button(
                    "Download Predictions (CSV)",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="nids_batch_predictions.csv",
                    mime="text/csv",
                )
else:
    st.info("Waiting for an unlabeled CSV. Train a model first (sidebar), then upload here.")