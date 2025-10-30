import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO
from scipy.signal import find_peaks
import time
import random
import os

# ФІЗИЧНІ КОНСТАНТИ ДЛЯ НІОБІЮ
e = 1.602e-19  # Кл
m = 9.109e-31  # кг
kB = 1.3806e-23  # Дж/К

# Параметри ніобію
Tc = 9.2  # К
n0 = 2.8e28  # м⁻³
tau_imp = 2.0e-12  # с

def determine_state(T):
    return "Надпровідник" if T < Tc else "Звичайний метал"

def tau_temperature_dependence(T):
    return tau_imp * (1 + (T / Tc)**3) if T < Tc else tau_imp * (T / Tc)

def calculate_superconducting_current(t, E_type, E0=1.0, a=1.0, omega=1.0, j0=0.0, T=4.2):
    ns = n0 * (1.0 - (T / Tc)**4.0) if T < Tc else 0.0
    K = (e**2 * ns) / m
    if E_type == "Статичне":
        return j0 + K * E0 * t
    elif E_type == "Лінійне":
        return j0 + K * (a * t**2) / 2
    elif E_type == "Синусоїдальне":
        return j0 + (K * E0 / omega) * np.sin(omega * t)

def calculate_normal_current_drude(t, E_type, T, E0=1.0, a=1.0, omega=1.0, j0=0.0):
    tau_T = tau_temperature_dependence(T)
    sigma = (n0 * e**2 * tau_T) / m
    if E_type == "Статичне":
        return j0 * np.exp(-t/tau_T) + sigma * E0 * (1.0 - np.exp(-t/tau_T))
    elif E_type == "Лінійне":
        return j0 * np.exp(-t/tau_T) + sigma * a * (t - tau_T * (1.0 - np.exp(-t/tau_T)))
    elif E_type == "Синусоїдальне":
        omega_tau_sq = (omega * tau_T)**2.0
        amp_factor = (sigma * E0) / np.sqrt(1.0 + omega_tau_sq)
        phase_shift = np.arctan(omega * tau_T)
        J_steady = amp_factor * np.sin(omega * t - phase_shift)
        C = j0 - amp_factor * np.sin(-phase_shift)
        J_transient = C * np.exp(-t / tau_T)
        return J_transient + J_steady

def calculate_normal_current_ohm(t, E_type, T, E0=1.0, a=1.0, omega=1.0, j0=0.0):
    tau_T = tau_temperature_dependence(T)
    sigma = (n0 * e**2 * tau_T) / m
    if E_type == "Статичне":
        return sigma * E0 * np.ones_like(t)
    elif E_type == "Лінійне":
        return sigma * a * t
    elif E_type == "Синусоїдальне":
        return sigma * E0 * np.sin(omega * t)

def analyze_physical_characteristics(t, j_data, state_name, field_type, T, omega=1.0):
    dt = t[1] - t[0]
    dj_dt = np.gradient(j_data, dt)
    analysis = {
        'Стан': state_name,
        'Температура': f"{T} K",
        'j(0)': f"{j_data[0]:.2e} А/м²",
        'j(t_max)': f"{j_data[-1]:.2e} А/м²",
        'j_max': f"{np.max(j_data):.2e} А/м²",
        'j_min': f"{np.min(j_data):.2e} А/м²",
        'Амплітуда': f"{np.max(j_data) - np.min(j_data):.2e} А/м²",
        'Макс. швидкість': f"{np.max(dj_dt):.2e} А/м²с"
    }
    if field_type == "Статичне":
        analysis['Поведінка'] = "Лінійне зростання" if state_name == "Надпровідник" else "Експоненційне насичення"
    elif field_type == "Лінійне":
        analysis['Поведінка'] = "Квадратичне зростання" if state_name == "Надпровідник" else "Експоненційне насичення"
    elif field_type == "Синусоїдальне":
        if state_name == "Надпровідник":
            analysis['Поведінка'] = "Коливання"
            analysis['Фазовий зсув'] = "π/2"
        else:
            tau_val = tau_temperature_dependence(T)
            analysis['Поведінка'] = "Коливання з фазовим зсувом"
            analysis['Фазовий зсув'] = f"{np.arctan(omega * tau_val):.3f} рад"
    return analysis

def analyze_mathematical_characteristics(t, j_data, state_name, field_type, omega=1.0):
    dt = t[1] - t[0]
    dj_dt = np.gradient(j_data, dt)
    peaks, _ = find_peaks(j_data, prominence=np.max(j_data)*0.01)
    valleys, _ = find_peaks(-j_data, prominence=-np.min(j_data)*0.01)
    analysis = {
        'Функція': state_name,
        'f(0)': f"{j_data[0]:.2e}",
        'f(t_max)': f"{j_data[-1]:.2e}",
        'max f(t)': f"{np.max(j_data):.2e}",
        'min f(t)': f"{np.min(j_data):.2e}",
        'Середнє': f"{np.mean(j_data):.2e}",
        'Стандартне відхилення': f"{np.std(j_data):.2e}",
        "f'(max)": f"{np.max(dj_dt):.2e}",
        "f'(min)": f"{np.min(dj_dt):.2e}",
        "f'(середнє)": f"{np.mean(np.abs(dj_dt)):.2e}",
        'Максимуми': len(peaks),
        'Мінімуми': len(valleys),
        'Екстремуми': len(peaks) + len(valleys)
    }
    if field_type == "Статичне":
        analysis['Тип функції'] = "Лінійна" if state_name == "Надпровідник" else "Експоненційна"
    elif field_type == "Лінійне":
        analysis['Тип функції'] = "Квадратична" if state_name == "Надпровідник" else "Експоненційна"
    elif field_type == "Синусоїдальне":
        analysis['Тип функції'] = "Коливальна"
        analysis['Період'] = f"{2*np.pi/omega:.2f} с" if omega > 0 else "∞"
    return analysis

def create_pdf_report(input_data, physical_analyses, math_analyses, saved_plots):
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import io
        
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=landscape(A4))
        
        font_name = 'Helvetica'
        try:
            if os.path.exists("DejaVuSans.ttf"):
                pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
                font_name = 'DejaVuSans'
        except:
            pass
        
        pdf.setFont(font_name, 16)
        pdf.drawString(100, 520, "ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ В НІОБІЇ")
        pdf.setFont(font_name, 12)
        y = 490
        
        # Параметри
        pdf.drawString(100, y, "Параметри моделювання:")
        y -= 20
        for k, v in input_data.items():
            pdf.drawString(120, y, f"- {k}: {v}")
            y -= 20
        y -= 10

        # Фізичний аналіз
        if physical_analyses:
            pdf.drawString(100, y, "Фізичний аналіз:")
            y -= 25
            headers = ["Стан", "Температура", "j(0)", "j_max", "Поведінка"]
            col_widths = [120, 80, 100, 100, 180]
            row_height = 20
            pdf.setFillColorRGB(0.8, 0.8, 1.0)
            pdf.rect(100, y - row_height, sum(col_widths), row_height, fill=1)
            pdf.setFillColorRGB(0, 0, 0)
            x = 100
            for h in headers:
                pdf.drawString(x + 5, y - 15, h)
                x += col_widths[headers.index(h)]
            y -= row_height
            for i, a in enumerate(physical_analyses):
                pdf.setFillColorRGB(0.95, 0.95, 0.95) if i % 2 == 0 else pdf.setFillColorRGB(1, 1, 1)
                pdf.rect(100, y - row_height, sum(col_widths), row_height, fill=1)
                pdf.setFillColorRGB(0, 0, 0)
                x = 100
                for h in headers:
                    pdf.drawString(x + 5, y - 15, str(a.get(h, '')))
                    x += col_widths[headers.index(h)]
                y -= row_height
                if y < 100:
                    pdf.showPage()
                    y = 490

        pdf.save()
        buffer.seek(0)
        return buffer
    except Exception as e:
        buffer = BytesIO()
        buffer.write(f"PDF не створено: {e}".encode('utf-8'))
        buffer.seek(0)
        return buffer

# === СТОРІНКИ ===
def animations_page():
    st.header("Анімації")
    if st.button("Запустити анімацію"):
        progress = st.progress(0)
        placeholder = st.empty()
        for i, T in enumerate(np.linspace(1, 18, 35)):
            progress.progress(int((i+1)/35*100))
            t = np.linspace(0, 5, 200)
            js = calculate_superconducting_current(t, "Статичне", 1.0, T=T)
            jn = calculate_normal_current_drude(t, "Статичне", T, 1.0)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=js, name="Надпровідник", line=dict(color="red")))
            fig.add_trace(go.Scatter(x=t, y=jn, name="Метал", line=dict(color="blue")))
            fig.update_layout(title=f"T = {T:.1f} K", height=500)
            placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.15)
        st.success("Готово!")

def racing_page():
    st.header("Гонки")
    st.write("Порівняння двох станів — хто швидше?")

def prediction_game_page():
    st.header("Передбачення")
    if st.button("Нова задача"):
        T = random.uniform(3, 15)
        st.session_state.problem_T = T
        st.session_state.problem_state = "Надпровідник" if T < Tc else "Метал"
        st.info(f"Температура: {T:.1f} K")
    if 'problem_state' in st.session_state:
        choice = st.radio("Струм буде:", ["Нескінченно зростати", "Насичуватися"])
        if st.button("Перевірити"):
            correct = (st.session_state.problem_state == "Надпровідник" and "зростати" in choice) or \
                      (st.session_state.problem_state == "Метал" and "Насич" in choice)
            st.write("Правильно!" if correct else "Ні!")

# === ОСНОВНА СТОРІНКА ===
def main_page():
    st.title("Моделювання струму в ніобії")
    if 'saved_plots' not in st.session_state:
        st.session_state.saved_plots = []

    with st.sidebar:
        st.header("Налаштування")
        mode = st.radio("Режим:", ["Один стан", "Порівняння", "Збережені"])
        field_type = st.selectbox("Поле:", ["Статичне", "Лінійне", "Синусоїдальне"])
        E0 = st.slider("E₀", 0.1, 100.0, 1.0, 0.1)
        t_max = st.slider("Час", 0.1, 20.0, 5.0, 0.1)
        a = st.slider("a", 0.1, 10.0, 1.0) if "Лінійне" in field_type else 1.0
        omega = st.slider("ω", 0.1, 50.0, 5.0) if "Синусоїдальне" in field_type else 1.0
        j0 = st.slider("j₀", 0.0, 100.0, 0.0, 0.1)

        metal_model = "Модель Друде (з перехідним процесом)"
        T_val = 4.2
        if mode == "Порівняння":
            T_val = st.slider("T (K)", 0.1, 18.4, 4.2, 0.1)
        elif mode == "Один стан":
            T_val = st.slider("T (K)", 0.1, 18.4, 4.2, 0.1)
            if determine_state(T_val) != "Надпровідник":
                metal_model = st.radio("Модель:", ["Модель Друде (з перехідним процесом)", "Закон Ома (стаціонарний)"])
        else:
            T_val = st.slider("T (K)", 0.1, 18.4, 4.2, 0.1)

        if mode != "Збережені" and st.button("Зберегти графік"):
            t = np.linspace(0, t_max, 1000)
            data = {'t': t, 'field_type': field_type, 'E0': E0, 'j0': j0, 'a': a, 'omega': omega, 'T': T_val, 'mode': mode}
            if mode == "Один стан":
                state = determine_state(T_val)
                if state == "Надпровідник":
                    data['j'] = calculate_superconducting_current(t, field_type, E0, a, omega, j0, T_val)
                    data['label'] = "Надпровідник"
                else:
                    func = calculate_normal_current_drude if "Друде" in metal_model else calculate_normal_current_ohm
                    data['j'] = func(t, field_type, T_val, E0, a, omega, j0)
                    data['label'] = "Метал"
            else:
                data['js'] = calculate_superconducting_current(t, field_type, E0, a, omega, j0, T_val)
                data['jn'] = calculate_normal_current_drude(t, field_type, T_val, E0, a, omega, j0)
                data['label'] = "Порівняння"
            st.session_state.saved_plots.append(data)
            st.success("Збережено!")

        if st.session_state.saved_plots and st.button("Очистити"):
            st.session_state.saved_plots = []
            st.success("Очищено!")

    col1, col2 = st.columns([2, 1])
    with col1:
        if mode == "Збережені":
            st.header("Збережені графіки")
            if not st.session_state.saved_plots:
                st.info("Порожньо")
            else:
                fig = go.Figure()
                for i, p in enumerate(st.session_state.saved_plots):
                    if p['label'] == "Надпровідник":
                        fig.add_trace(go.Scatter(x=p['t'], y=p['j'], name=f"Надпр {i+1}"))
                    elif p['label'] == "Метал":
                        fig.add_trace(go.Scatter(x=p['t'], y=p['j'], name=f"Метал {i+1}"))
                    else:
                        fig.add_trace(go.Scatter(x=p['t'], y=p['js'], name=f"Надпр {i+1}"))
                        fig.add_trace(go.Scatter(x=p['t'], y=p['jn'], name=f"Метал {i+1}"))
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.header("Графік")
            t = np.linspace(0, t_max, 1000)
            fig = go.Figure()
            if mode == "Порівняння":
                js = calculate_superconducting_current(t, field_type, E0, a, omega, j0, T_val)
                jn = calculate_normal_current_drude(t, field_type, T_val, E0, a, omega, j0)
                fig.add_trace(go.Scatter(x=t, y=js, name="Надпровідник", line=dict(color="red")))
                fig.add_trace(go.Scatter(x=t, y=jn, name="Метал", line=dict(color="blue")))
            else:
                state = determine_state(T_val)
                if state == "Надпровідник":
                    j = calculate_superconducting_current(t, field_type, E0, a, omega, j0, T_val)
                else:
                    func = calculate_normal_current_drude if "Друде" in metal_model else calculate_normal_current_ohm
                    j = func(t, field_type, T_val, E0, a, omega, j0)
                fig.add_trace(go.Scatter(x=t, y=j, name=state, line=dict(color="red" if state == "Надпровідник" else "blue")))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.header("Інфо")
        st.write(f"**T:** {T_val} K → **{determine_state(T_val)}**")
        if st.button("PDF звіт"):
            t_rep = np.linspace(0, t_max, 1000)
            input_data = {'field_type': field_type, 'E0': E0, 'j0': j0, 't_max': t_max, 'T': T_val}
            phys, math_ = [], []
            if mode == "Порівняння":
                js = calculate_superconducting_current(t_rep, field_type, E0, a, omega, j0, T_val)
                jn = calculate_normal_current_drude(t_rep, field_type, T_val, E0, a, omega, j0)
                phys = [analyze_physical_characteristics(t_rep, js, "Надпровідник", field_type, T_val, omega),
                        analyze_physical_characteristics(t_rep, jn, "Метал", field_type, T_val, omega)]
            else:
                state = determine_state(T_val)
                if state == "Надпровідник":
                    j = calculate_superconducting_current(t_rep, field_type, E0, a, omega, j0, T_val)
                else:
                    func = calculate_normal_current_drude if "Друде" in metal_model else calculate_normal_current_ohm
                    j = func(t_rep, field_type, T_val, E0, a, omega, j0)
                phys = [analyze_physical_characteristics(t_rep, j, state, field_type, T_val, omega)]
            pdf = create_pdf_report(input_data, phys, [], st.session_state.saved_plots)
            st.download_button("Завантажити PDF", pdf, "звіт.pdf", "application/pdf")

# === НАВІГАЦІЯ ===
def main():
    st.set_page_config(page_title="Ніобій", layout="wide")
    with st.sidebar:
        page = st.radio("Сторінка:", ["Головна", "Анімація", "Гонки", "Гра"])
    if page == "Головна":
        main_page()
    elif page == "Анімація":
        animations_page()
    elif page == "Гонки":
        racing_page()
    elif page == "Гра":
        prediction_game_page()

if __name__ == "__main__":
    main()
