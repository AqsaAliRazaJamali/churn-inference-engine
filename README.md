# Live Customer Churn Risk Inference Engine
 
An advanced, responsive enterprise-grade dashboard built with **Streamlit**, leveraging a Gradient Boosting pipeline to compute real-time operational churn probabilities (**<15ms target latency**). Features an integrated eXplainable AI (XAI) interface powered by **SHAP** to visualize local feature impact vectors on-the-fly.
 
<p align="left">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-8A2BE2" alt="SHAP">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs Welcome">
  <img src="https://img.shields.io/badge/status-active-success" alt="Status">
</p>
---
 
## Key Features
 
- **Advanced Cyberpunk/Gemini UI** — Tailored dark-themed styling, glassmorphism telemetry widgets, and glowing neon visual cues.
- **Predictive Pipeline Integration** — Connects seamlessly with a production serialization pipeline (`preprocessor.pkl` and `churn_model.pkl`).
- **Explainable AI Array** — Local feature weights visualized instantly through interactive SHAP waterfall charts.
- **Environment Diagnostics** — Automated runtime asset verification for production readiness.
---
 
## Tech Toolkit
 
| Category | Tools |
|---|---|
| **Core** | Python, Streamlit |
| **Data & Modeling** | Pandas, Scikit-learn, Joblib |
| **Explainability & Plotting** | SHAP, Matplotlib |
 
---
 
## Local Installation & Setup
 
### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-intelligence-dashboard.git
cd customer-churn-intelligence-dashboard
```
 
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
 
### 3. Ensure model binaries exist
Place your production artifacts inside the `models/` directory:
```
models/
├── preprocessor.pkl
└── churn_model.pkl
```
 
### 4. Launch the dashboard
```bash
streamlit run app.py
```
 
---
 
## 📂 Project Structure
 
```
customer-churn-intelligence-dashboard/
├── app.py
├── models/
│   ├── preprocessor.pkl
│   └── churn_model.pkl
├── requirements.txt
└── README.md
```
 
---
 
## How It Works
 
1. User inputs (or batch data) are passed through the serialized **preprocessing pipeline**.
2. The **Gradient Boosting model** computes a real-time churn probability score.
3. **SHAP** generates a local explanation, breaking down which features pushed the prediction up or down.
4. Results are rendered live on the dashboard with interactive waterfall visualizations.
---
 
## 🤝 Contributing
 
Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](../../issues) or open a pull request.
 
---
 
## 📜 License
 
This project is licensed under the **MIT License**.
 
---
 
## 👨‍💻 Author

### **Aqsa Jamali**

Artificial Intelligence Student | Machine Learning Enthusiast | Python Developer

GitHub: **https://github.com/AqsaAliRazaJamali**

---

## ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.
