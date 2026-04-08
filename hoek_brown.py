from io import BytesIO
from math import exp, asin, degrees

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from numpy import power as pow

st.set_page_config(
    page_title="HOEK-BROWN FAILURE CRITERION",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", "Roboto", sans-serif;
    background-color: #f3f4f6;
}

/* Title */
.main-title {
    font-size: 2rem;
    font-weight: 700;
    color: #111827;
}

/* Panels */
.panel-box {
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 12px;
    background: #ffffff;
    margin-bottom: 12px;
}

/* Section titles */
.section-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #374151;
    margin-bottom: 6px;
    text-transform: uppercase;
}

/* Result cards */
.result-card {
    border: 1px solid #e5e7eb;
    border-left: 4px solid #2563eb;
    border-radius: 6px;
    padding: 10px;
    background: #ffffff;
    margin-bottom: 8px;
}

.result-name {
    font-size: 0.75rem;
    color: #6b7280;
}

.result-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: #111827;
}

/* Buttons */
.stDownloadButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 6px;
    border: none;
}
</style>
""", unsafe_allow_html=True)


class HoekBrown:
    def __init__(
        self,
        UCS,
        GSI,
        mi,
        D,
        E_Method,
        sigma3_method,
        UW,
        H,
        MR=None,
        Name="Rock",
        Ei=None,
    ):
        self.UCS = UCS
        self.GSI = GSI
        self.mi = mi
        self.D = D
        self.E_Method = E_Method
        self.sigma3_method = sigma3_method
        self.UW = UW
        self.H = H
        self.MR = MR
        self.Name = Name
        self.Ei = Ei

        self.HBParameters()

        self.results = {
            "mb": round(self.mb, 3),
            "s": round(self.s, 5),
            "alpha": round(self.alpha, 3),
            "σcm (MPa)": round(self.sigmacm, 2),
            "σ3 (MPa)": round(self.sigma3, 2),
            "σt (MPa)": round(self.sigmat, 3) if np.isfinite(self.sigmat) else "N/A",
            "Erm (MPa)": round(self.Erm) if self.Erm is not None else "N/A",
            "c (kPa)": round(self.c * 1000) if np.isfinite(self.c) else "N/A",
            "φ (deg)": round(self.fi, 1) if np.isfinite(self.fi) else "N/A",
        }

        self.inputs = {
            "UCS (MPa)": round(self.UCS, 1),
            "GSI": round(self.GSI, 0),
            "mi": round(self.mi, 2),
            "D": round(self.D, 2),
            "Modulus method": self.E_Method,
            "Sigma3 method": self.sigma3_method,
        }

        if self.UW is not None:
            self.inputs["Unit weight (kN/m³)"] = self.UW
        if self.H is not None:
            self.inputs["Depth (m)"] = self.H
        if self.MR is not None:
            self.inputs["MR"] = self.MR
        if self.Ei is not None:
            self.inputs["Ei (MPa)"] = self.Ei

        self.HB_Figure()

    def HBParameters(self):
        self.mb = self.mi * exp((self.GSI - 100) / (28 - 14 * self.D))
        self.s = exp((self.GSI - 100) / (9 - 3 * self.D))
        self.alpha = 0.5 + (exp(-self.GSI / 15) - exp(-20 / 3)) / 6
        self.sigmacm_calc()
        self.sigmat_calc()
        self.MC()
        self.E_calc()

    def sigmacm_calc(self):
        self.sigmacm = (
            self.UCS
            * (self.mb + 4 * self.s - self.alpha * (self.mb - 8 * self.s))
            * pow(self.mb / 4 + self.s, self.alpha - 1)
            / (2 * (1 + self.alpha) * (2 + self.alpha))
        )
        self.sigma3_calc()

    def sigma3_calc(self):
        if self.sigma3_method in ("Tunnel", "Slope"):
            if self.UW is None or self.H is None or self.UW <= 0 or self.H <= 0:
                self.sigma3 = self.UCS / 4
                return

            ratio = self.sigmacm / (self.UW * self.H * 0.001)
            if ratio <= 0:
                self.sigma3 = self.UCS / 4
                return

            if self.sigma3_method == "Tunnel":
                self.sigma3 = 0.47 * pow(ratio, -0.94) * self.sigmacm
            else:
                self.sigma3 = 0.72 * pow(ratio, -0.91) * self.sigmacm
        else:
            self.sigma3 = self.UCS / 4

    def sigmat_calc(self):
        if abs(self.mb) < 1e-12:
            self.sigmat = float("nan")
        else:
            self.sigmat = -self.s * self.UCS / self.mb

    def MC(self):
        sigma3n = self.sigma3 / self.UCS
        base = self.s + self.mb * sigma3n

        if base <= 0:
            self.fi = float("nan")
            self.c = float("nan")
            return

        term = 6 * self.alpha * self.mb * pow(base, self.alpha - 1)
        denom = (
            2 * (1 + self.alpha) * (2 + self.alpha)
            + 6 * self.alpha * self.mb * pow(base, self.alpha - 1)
        )

        arg = term / denom
        arg = max(-1.0, min(1.0, arg))
        self.fi = degrees(asin(arg))

        self.c = (
            self.UCS
            * ((1 + self.alpha * 2) * self.s + (1 - self.alpha) * self.mb * sigma3n)
            * pow(base, self.alpha - 1)
            / (
                (1 + self.alpha)
                * (2 + self.alpha)
                * pow(
                    1
                    + (
                        6
                        * self.alpha
                        * self.mb
                        * pow(base, self.alpha - 1)
                    )
                    / ((1 + self.alpha) * (2 + self.alpha)),
                    0.5,
                )
            )
        )

    def E_calc(self):
        if self.E_Method == "Generalized Hoek & Diederichs (2006)":
            if self.Ei is None:
                self.Ei = self.MR * self.UCS
            self.Erm = self.Ei * (
                0.02 + (1 - self.D / 2) / (1 + exp((60 + 15 * self.D - self.GSI) / 11))
            )

        elif self.E_Method == "Simplified Hoek & Diederichs (2006)":
            self.Erm = 100000 * (
                (1 - self.D / 2) / (1 + exp((75 + 25 * self.D - self.GSI) / 11))
            )

        elif self.E_Method == "Hoek, Carranza-Torres, Corkum (2002)":
            if self.UCS <= 100:
                self.Erm = (
                    1000
                    * (1 - self.D / 2)
                    * pow(self.UCS / 100, 0.5)
                    * pow(10, (self.GSI - 10) / 40)
                )
            else:
                self.Erm = 1000 * (1 - self.D / 2) * pow(10, (self.GSI - 10) / 40)
        else:
            self.Erm = None

    def HB_Figure(self):
        sigma3_start = self.sigmat if np.isfinite(self.sigmat) else 0
        sigma3_array = np.linspace(sigma3_start, self.sigma3, 1000)

        sigma1_array = sigma3_array + self.UCS * pow(
            self.mb * sigma3_array / self.UCS + self.s, self.alpha
        )
        sigma1_array[sigma1_array < 0] = 0

        self.fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

        # Background
        self.fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Main curve
        ax.plot(
            sigma3_array,
            sigma1_array,
            color="#1f77b4",
            linewidth=2.5,
        )

        # Fill
        ax.fill_between(
            sigma3_array,
            sigma1_array,
            color="#1f77b4",
            alpha=0.08,
        )

        # Grid (software style)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        # Axis styling
        ax.set_xlabel("σ3 (MPa)", fontsize=10)
        ax.set_ylabel("σ1 (MPa)", fontsize=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(axis="both", labelsize=9)

        # Title
        ax.set_title(
            "Hoek-Brown Failure Envelope",
            fontsize=11,
            fontweight="bold",
        )

    def to_dataframe(self):
        rows = []
        for k, v in self.inputs.items():
            rows.append({"Category": "Input", "Parameter": k, "Value": v})
        for k, v in self.results.items():
            rows.append({"Category": "Result", "Parameter": k, "Value": v})
        return pd.DataFrame(rows)

    def create_download_files(self):
        df = self.to_dataframe()

        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="HoekBrown")
        excel_buffer.seek(0)

        pdf_buffer = BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            pdf.savefig(self.fig, bbox_inches="tight")

            fig2, ax2 = plt.subplots(figsize=(8.27, 11.69))
            fig2.patch.set_facecolor("white")
            ax2.axis("off")
            ax2.set_title(
                f"HOEK-BROWN FAILURE CRITERION — {self.Name}",
                fontweight="bold",
                fontsize=14,
                pad=20,
            )

            table_data = [["Category", "Parameter", "Value"]] + df.values.tolist()
            table = ax2.table(
                cellText=table_data,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.4)

            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

        pdf_buffer.seek(0)
        return excel_buffer, pdf_buffer


def result_box(label, value):
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-name">{label}</div>
            <div class="result-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def streamlitHoekBrown():
    st.markdown(
        '<div class="main-title">HOEK-BROWN FAILURE CRITERION</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.85rem;color:#6b7280;">Geotechnical Analysis Module</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.9rem;color:#6b7280;margin-bottom:1rem;">Rock mass strength evaluation and Hoek-Brown failure envelope analysis.</div>',
        unsafe_allow_html=True,
    )

    rock_db = {
        "Agglomerate": {"MR": 500, "MRSTD": 100, "Type": "Igneous", "mi": 19, "miSTD": 3},
        "Amphibolites": {"MR": 450, "MRSTD": 50, "Type": "Metamorphic", "mi": 26, "miSTD": 6},
        "Andesite": {"MR": 400, "MRSTD": 100, "Type": "Igneous", "mi": 25, "miSTD": 5},
        "Anhydrite": {"MR": 350, "MRSTD": 0, "Type": "Sedimentary", "mi": 12, "miSTD": 2},
        "Basalt": {"MR": 350, "MRSTD": 100, "Type": "Igneous", "mi": 25, "miSTD": 5},
        "Breccia-I": {"MR": 500, "MRSTD": 0, "Type": "Igneous", "mi": 19, "miSTD": 5},
        "Breccia-S": {"MR": 290, "MRSTD": 60, "Type": "Sedimentary", "mi": 19, "miSTD": 5},
        "Chalk": {"MR": 1000, "MRSTD": 0, "Type": "Sedimentary", "mi": 7, "miSTD": 2},
        "Claystones": {"MR": 250, "MRSTD": 50, "Type": "Sedimentary", "mi": 4, "miSTD": 2},
        "Conglomerates": {"MR": 350, "MRSTD": 50, "Type": "Sedimentary", "mi": 21, "miSTD": 3},
        "Dacite": {"MR": 400, "MRSTD": 50, "Type": "Igneous", "mi": 25, "miSTD": 3},
        "Diabase": {"MR": 325, "MRSTD": 25, "Type": "Igneous", "mi": 15, "miSTD": 5},
        "Diorite": {"MR": 325, "MRSTD": 25, "Type": "Igneous", "mi": 25, "miSTD": 5},
        "Dolerite": {"MR": 350, "MRSTD": 50, "Type": "Igneous", "mi": 12, "miSTD": 3},
        "Dolomites": {"MR": 425, "MRSTD": 75, "Type": "Sedimentary", "mi": 9, "miSTD": 3},
        "Gabbro": {"MR": 450, "MRSTD": 50, "Type": "Sedimentary", "mi": 27, "miSTD": 3},
        "Gneiss": {"MR": 525, "MRSTD": 225, "Type": "Metamorphic", "mi": 28, "miSTD": 5},
        "Granite": {"MR": 425, "MRSTD": 125, "Type": "Igneous", "mi": 32, "miSTD": 3},
        "Granodiorite": {"MR": 425, "MRSTD": 125, "Type": "Igneous", "mi": 29, "miSTD": 3},
        "Greywackes": {"MR": 350, "MRSTD": 0, "Type": "Sedimentary", "mi": 18, "miSTD": 3},
        "Gypsum": {"MR": 350, "MRSTD": 0, "Type": "Sedimentary", "mi": 8, "miSTD": 2},
        "Hornfels": {"MR": 550, "MRSTD": 150, "Type": "Metamorphic", "mi": 19, "miSTD": 4},
        "Limestone (Crystalline)": {"MR": 500, "MRSTD": 100, "Type": "Sedimentary", "mi": 12, "miSTD": 3},
        "Limestone (Micritic)": {"MR": 900, "MRSTD": 100, "Type": "Sedimentary", "mi": 9, "miSTD": 2},
        "Limestone (Sparitic)": {"MR": 700, "MRSTD": 100, "Type": "Sedimentary", "mi": 10, "miSTD": 2},
        "Marble": {"MR": 850, "MRSTD": 150, "Type": "Metamorphic", "mi": 9, "miSTD": 3},
        "Marls": {"MR": 175, "MRSTD": 25, "Type": "Sedimentary", "mi": 7, "miSTD": 2},
        "Metasandstones": {"MR": 250, "MRSTD": 50, "Type": "Metamorphic", "mi": 19, "miSTD": 3},
        "Migmatite": {"MR": 375, "MRSTD": 25, "Type": "Metamorphic", "mi": 29, "miSTD": 3},
        "Norite": {"MR": 375, "MRSTD": 25, "Type": "Igneous", "mi": 20, "miSTD": 5},
        "Peridotite": {"MR": 275, "MRSTD": 25, "Type": "Igneous", "mi": 25, "miSTD": 5},
        "Phyllites": {"MR": 550, "MRSTD": 250, "Type": "Metamorphic", "mi": 7, "miSTD": 3},
        "Porphyries": {"MR": 400, "MRSTD": 0, "Type": "Igneous", "mi": 20, "miSTD": 5},
        "Quartzites": {"MR": 375, "MRSTD": 75, "Type": "Metamorphic", "mi": 20, "miSTD": 3},
        "Rhyolite": {"MR": 400, "MRSTD": 100, "Type": "Igneous", "mi": 25, "miSTD": 5},
        "Sandstones": {"MR": 275, "MRSTD": 75, "Type": "Sedimentary", "mi": 17, "miSTD": 4},
        "Schists": {"MR": 675, "MRSTD": 425, "Type": "Metamorphic", "mi": 12, "miSTD": 3},
        "Shales": {"MR": 200, "MRSTD": 50, "Type": "Sedimentary", "mi": 6, "miSTD": 2},
        "Siltstones": {"MR": 375, "MRSTD": 25, "Type": "Sedimentary", "mi": 7, "miSTD": 2},
        "Slates": {"MR": 500, "MRSTD": 100, "Type": "Metamorphic", "mi": 7, "miSTD": 4},
        "Tuff": {"MR": 300, "MRSTD": 100, "Type": "Igneous", "mi": 13, "miSTD": 5},
    }

    left, right = st.columns([1, 1.6], gap="medium")

    with left:
        st.markdown('<div class="section-title">Input Panel</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="panel-box">', unsafe_allow_html=True)
            Name = st.text_input("Project / Rock Mass Name", value="Rock")
            rock_type = st.selectbox("Rock Type Database", list(rock_db.keys()))
            selected_rock = rock_db[rock_type]

            st.caption(
                f"Type: {selected_rock['Type']} | "
                f"mi: {selected_rock['mi']} ± {selected_rock['miSTD']} | "
                f"MR: {selected_rock['MR']} ± {selected_rock['MRSTD']}"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="panel-box">', unsafe_allow_html=True)
            st.markdown("**Rock Mass Parameters**")

            col1, col2 = st.columns(2)

            with col1:
                UCS = st.slider(
                    "Uniaxial Compressive Strength, UCS (MPa)",
                    min_value=0.1,
                    max_value=300.0,
                    value=30.0,
                    step=0.1,
                )
                mi = st.slider(
                    "Material Constant, mi",
                    min_value=1.0,
                    max_value=40.0,
                    value=float(selected_rock["mi"]),
                    step=0.1,
                )

            with col2:
                GSI = st.slider(
                    "Geological Strength Index, GSI",
                    min_value=1,
                    max_value=100,
                    value=60,
                    step=1,
                )
                D = st.slider(
                    "Disturbance Factor, D",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.01,
                )

            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="panel-box">', unsafe_allow_html=True)
            st.markdown("**Modulus Estimation**")
            E_Method = st.selectbox(
                "Modulus Calculation Method",
                (
                    "Generalized Hoek & Diederichs (2006)",
                    "Simplified Hoek & Diederichs (2006)",
                    "Hoek, Carranza-Torres, Corkum (2002)",
                ),
            )

            if E_Method == "Generalized Hoek & Diederichs (2006)":
                mode = st.radio("Ei Input Mode", ["Use MR", "Manual Ei"], horizontal=True)
                if mode == "Use MR":
                    MR = st.number_input(
                        "Modulus Ratio, MR (Ei/UCS)",
                        value=float(selected_rock["MR"]),
                        min_value=1.0,
                        step=1.0,
                    )
                    Ei = None
                else:
                    Ei = st.number_input(
                        "Intact Modulus, Ei (MPa)",
                        value=float(selected_rock["MR"] * 30),
                        min_value=1.0,
                        step=100.0,
                    )
                    MR = None
            else:
                MR = float(selected_rock["MR"])
                Ei = None
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="panel-box">', unsafe_allow_html=True)
            st.markdown("**Confining Stress Selection**")
            sigma3_method = st.selectbox(
                "Lateral Pressure Method",
                ("Tunnel", "Slope", "General"),
            )

            if sigma3_method != "General":
                col1, col2 = st.columns(2)
                with col1:
                    UW = st.number_input("Unit Weight, γ (kN/m³)", value=23.5, min_value=1.0)
                with col2:
                    H = st.number_input("Depth, H (m)", value=30, min_value=1)
            else:
                UW = None
                H = None
            st.markdown("</div>", unsafe_allow_html=True)

    rock = HoekBrown(UCS, GSI, mi, D, E_Method, sigma3_method, UW, H, MR, Name, Ei)
    excel_file, pdf_file = rock.create_download_files()

    with right:
        st.markdown('<div class="section-title">Output Panel</div>', unsafe_allow_html=True)

        k1, k2 = st.columns(2)

        with k1:
            result_box("σcm (MPa)", rock.results["σcm (MPa)"])
            result_box("σ3 (MPa)", rock.results["σ3 (MPa)"])
            result_box("σt (MPa)", rock.results["σt (MPa)"])

        with k2:
            result_box("c (kPa)", rock.results["c (kPa)"])
            result_box("φ (deg)", rock.results["φ (deg)"])
            result_box("Erm (MPa)", rock.results["Erm (MPa)"])

        st.pyplot(rock.fig, clear_figure=False)

        with st.expander("Calculation Summary"):
            st.dataframe(rock.to_dataframe(), use_container_width=True, hide_index=True)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                label="Download Excel Report",
                data=excel_file,
                file_name=f"{Name}_HoekBrown.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                label="Download PDF Report",
                data=pdf_file,
                file_name=f"{Name}_HoekBrown.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


streamlitHoekBrown()