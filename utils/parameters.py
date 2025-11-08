"""
parameters.py
Defines the parameters, limits, weights, and fail-fast rules
for the İzmir Water Health Factor (HF) Model.
"""

PARAMETERS = {
    "E.COL": {"limit": 0, "weight": 0.133, "failfast": True},
    "KOLIF": {"limit": 0, "weight": 0.133, "failfast": True},
    "C.PER": {"limit": 0, "weight": 0.133, "failfast": True},
    "As": {"limit": 10, "weight": 0.088, "failfast": True},
    "NO2": {"limit": 0.5, "weight": 0.088, "failfast": True},
    "Al": {"limit": 200, "weight": 0.055},
    "Fe": {"limit": 200, "weight": 0.055},
    "NH4+": {"limit": 0.5, "weight": 0.055},
    "pH": {"limit": (6.5, 9.5), "weight": 0.040},
    "Cl-": {"limit": 250, "weight": 0.020},
    "ILETK": {"limit": 2500, "weight": 0.020},
    "OKSIT": {"limit": 5.0, "weight": 0.020},
    "BULAN": {"limit": None, "weight": 0.010},
    "TAT": {"limit": None, "weight": 0.010},
    "KOKU": {"limit": None, "weight": 0.010},
    "RENK": {"limit": None, "weight": 0.010},
    "TOPLA": {"limit": None, "weight": 0.010},
    "TUZLU": {"limit": None, "weight": 0.010},
}

ALPHA = 1.2  # Nonlinearity factor for risk amplification
HF_SCALE = 100  # Final scaling factor for output