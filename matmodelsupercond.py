import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from io import BytesIO
import base64
from scipy.signal import find_peaks

# ФІЗИЧНІ КОНСТАНТИ ДЛЯ НІОБІЮ
e = 1.6e-19  # Кл
m = 9.1e-31  # кг
Tc = 9.2  # К
n0 = 1.0e29  # м⁻³
tau_imp = 5.0e-14  # с
A_ph = 3.0e8  # коефіцієнт фононного розсіювання

def calculate_superconducting_current(t, E_type, E0=1.0, a=1.0, omega=1.0, j0=0.0):
    """Розрахунок струму в надпровідному стані - рівняння Лондонів"""
    K = (e**2 * n0) / m
    
    if E_type == "Статичне":
        return j0 + K * E0 * t
    elif E_type == "Лінійне":
        return j0 + K * (a * t**2) / 2
    elif E_type == "Синусоїдальне":
        return j0 + (K * E0 / omega) * (1 - np.cos(omega * t))

def calculate_normal_current(t, E_type, T, E0=1.0, a=1.0, omega=1.0, j0=0.0):
    """Розрахунок струму в звичайному стані - модель Друде"""
    # ВИПРАВЛЕННЯ: використовуємо постійну концентрацію для звичайного металу
    ns = n0  # Для звичайного металу концентрація не залежить від температури так сильно
    tau = 1 / (1/tau_imp + A_ph * T**5)
    sigma = (ns * e**2 * tau) / m
    
    if E_type == "Статичне":
        return j0 * np.exp(-t/tau) + sigma * E0 * tau * (1 - np.exp(-t/tau))
    elif E_type == "Лінійне":
        return j0 * np.exp(-t/tau) + sigma * a * E0 * tau**2 * (1 - np.exp(-t/tau))
    elif E_type == "Синусоїдальне":
        # ВИПРАВЛЕННЯ: правильна формула для синусоїдального поля
        phase_shift = np.arctan(omega * tau)
        amplitude = (sigma * E0 * tau) / np.sqrt(1 + (omega * tau)**2)
        transient = j0 * np.exp(-t/tau)
        steady_state = amplitude * np.sin(omega * t - phase_shift)
        return transient + steady_state

def analyze_physical_characteristics(t, j_data, state_name, field_type, T, omega=1.0):
    """ФІЗИЧНИЙ аналіз характеристик струму"""
    analysis = {}
    analysis['Параметр'] = state_name
    
    # Основні фізичні характеристики
    analysis['j(0)'] = f"{j_data[0]:.2e} А/м²"
    analysis['j(t_max)'] = f"{j_data[-1]:.2e} А/м²"
    analysis['j_max'] = f"{np.max(j_data):.2e} А/м²"
    analysis['j_min'] = f"{np.min(j_data):.2e} А/м²"
    
    # Фізична інтерпретація
    if field_type == "Статичне":
        if state_name == "Надпровідник":
            analysis['Поведінка'] = "Лінійне зростання"
            analysis['Швидкість'] = f"{(j_data[-1] - j_data[0]) / t[-1]:.2e} А/м²с"
        else:
            analysis['Поведінка'] = "Експоненційне насичення"
                
    elif field_type == "Лінійне":
        if state_name == "Надпровідник":
            analysis['Поведінка'] = "Квадратичне зростання"
        else:
            analysis['Поведінка'] = "Експоненційне насилення"
                
    elif field_type == "Синусоїдальне":
        if state_name == "Надпровідник":
            analysis['Поведінка'] = "Коливання"
            analysis['Фазовий зсув'] = "π/2"
        else:
            tau_val = 1 / (1/tau_imp + A_ph * T**5)
            analysis['Поведінка'] = "Коливання з фазовим зсувом"
            analysis['Фазовий зсув'] = f"{np.arctan(omega * tau_val):.3f} рад"
    
    analysis['Температура'] = f"{T} K"
    
    return analysis

def analyze_mathematical_characteristics(t, j_data, state_name, field_type):
    """МАТЕМАТИЧНИЙ аналіз графіка функції"""
    analysis = {}
    analysis['Функція'] = state_name
    
    # Математичні характеристики
    analysis['f(0)'] = f"{j_data[0]:.2e}"
    analysis['f(t_max)'] = f"{j_data[-1]:.2e}"
    analysis['max f(t)'] = f"{np.max(j_data):.2e}"
    analysis['min f(t)'] = f"{np.min(j_data):.2e}"
    
    # Похідна
    dt = t[1] - t[0]
    dj_dt = np.gradient(j_data, dt)
    analysis["f'(max)"] = f"{np.max(dj_dt):.2e}"
    analysis["f'(min)"] = f"{np.min(dj_dt):.2e}"
    
    # Екстремуми
    peaks, _ = find_peaks(j_data, prominence=np.max(j_data)*0.01)
    valleys, _ = find_peaks(-j_data, prominence=-np.min(j_data)*0.01)
    
    analysis['Екстремуми'] = len(peaks) + len(valleys)
    
    if field_type == "Статичне":
        if state_name == "Надпровідник":
            analysis['Тип'] = "Лінійна"
        else:
            analysis['Тип'] = "Експоненційна"
    elif field_type == "Лінійне":
        if state_name == "Надпровідник":
            analysis['Тип'] = "Квадратична"
        else:
            analysis['Тип'] = "Експоненційна"
    elif field_type == "Синусоїдальне":
        analysis['Тип'] = "Коливальна"
    
    return analysis

def create_comprehensive_pdf_report(input_data, physical_analyses, math_analyses):
    """Створення красивого PDF звіту з таблицями"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib import colors
        import io
        
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
            font_name = 'DejaVuSans'
        except:
            font_name = 'Helvetica'
        
        y_position = 800
        
        # Заголовок
        pdf.setFont(font_name, 16)
        pdf.setFillColor(colors.darkblue)
        pdf.drawString(100, y_position, "ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ")
        y_position -= 40
        
        # Вхідні параметри
        pdf.setFont(font_name, 14)
        pdf.setFillColor(colors.darkgreen)
        pdf.drawString(100, y_position, "Вхідні параметри:")
        y_position -= 25
        
        pdf.setFont(font_name, 10)
        pdf.setFillColor(colors.black)
        params = [
            f"Тип поля: {input_data['field_type']}",
            f"Напруженість поля: {input_data['E0']} В/м", 
            f"Початковий струм: {input_data['j0']} А/м²",
            f"Час моделювання: {input_data['t_max']} с",
            f"Температура: {input_data['T_common']} K",
        ]
        
        for param in params:
            pdf.drawString(120, y_position, param)
            y_position -= 15
        y_position -= 20
        
        # Фізичний аналіз
        pdf.setFont(font_name, 14)
        pdf.setFillColor(colors.darkgreen)
        pdf.drawString(100, y_position, "Фізичний аналіз:")
        y_position -= 25
        
        for analysis in physical_analyses:
            pdf.setFont(font_name, 12)
            pdf.setFillColor(colors.darkred)
            pdf.drawString(100, y_position, f"{analysis['Параметр']}:")
            y_position -= 15
            
            pdf.setFont(font_name, 10)
            pdf.setFillColor(colors.black)
            for key, value in analysis.items():
                if key != 'Параметр':
                    pdf.drawString(120, y_position, f"{key}: {value}")
                    y_position -= 12
                    if y_position < 100:
                        pdf.showPage()
                        y_position = 800
                        pdf.setFont(font_name, 10)
            y_position -= 10
        
        y_position -= 20
        
        # Математичний аналіз
        pdf.setFont(font_name, 14)
        pdf.setFillColor(colors.darkgreen)
        pdf.drawString(100, y_position, "Математичний аналіз:")
        y_position -= 25
        
        for analysis in math_analyses:
            pdf.setFont(font_name, 12)
            pdf.setFillColor(colors.purple)
            pdf.drawString(100, y_position, f"{analysis['Функція']}:")
            y_position -= 15
            
            pdf.setFont(font_name, 10)
            pdf.setFillColor(colors.black)
            for key, value in analysis.items():
                if key != 'Функція':
                    pdf.drawString(120, y_position, f"{key}: {value}")
                    y_position -= 12
                    if y_position < 100:
                        pdf.showPage()
                        y_position = 800
                        pdf.setFont(font_name, 10)
            y_position -= 10
        
        # Висновки
        y_position -= 20
        pdf.setFont(font_name, 14)
        pdf.setFillColor(colors.darkgreen)
        pdf.drawString(100, y_position, "Основні спостереження:")
        y_position -= 25
        
        pdf.setFont(font_name, 10)
        pdf.setFillColor(colors.black)
        conclusions = [
            "• Надпровідник демонструє відмінну поведінку від звичайного металу",
            "• Різні типи полів викликають різну динаміку струму",
            "• Фазові зсуви спостерігаються при синусоїдальному полі",
            "• Моделі показують очікувані фізичні ефекти"
        ]
        
        for conclusion in conclusions:
            pdf.drawString(100, y_position, conclusion)
            y_position -= 15
        
        pdf.save()
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        buffer = BytesIO()
        report_text = "ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ\n\n"
        report_text += "Вхідні параметри:\n"
        for key, value in input_data.items():
            report_text += f"{key}: {value}\n"
        buffer.write(report_text.encode('utf-8'))
        buffer.seek(0)
        return buffer

def main():
    st.set_page_config(page_title="Моделювання струму", layout="wide")
    st.title("🔬 Моделювання динаміки струму")
    
    if 'saved_plots' not in st.session_state:
        st.session_state.saved_plots = []
    
    with st.sidebar:
        st.header("⚙️ Параметри моделювання")
        
        comparison_mode = st.radio(
            "Режим:",
            ["Один стан", "Порівняння", "Кілька графіків"]
        )
        
        st.subheader("Загальні параметри")
        field_type = st.selectbox("Тип поля:", ["Статичне", "Лінійне", "Синусоїдальне"])
        E0 = st.slider("Напруженість E₀ (В/м)", 0.1, 10.0, 1.0, 0.1)
        j0 = st.slider("Початковий струм j₀ (А/м²)", 0.0, 10.0, 0.0, 0.1)
        t_max = st.slider("Час моделювання (с)", 0.1, 10.0, 5.0, 0.1)
        
        if field_type == "Лінійне":
            a = st.slider("Швидкість росту a", 0.1, 5.0, 1.0, 0.1)
        else:
            a = 1.0
            
        if field_type == "Синусоїдальне":
            omega = st.slider("Частота ω (рад/с)", 0.1, 10.0, 1.0, 0.1)
        else:
            omega = 1.0
        
        st.subheader("Параметри станів")
        if comparison_mode == "Порівняння":
            T_common = st.slider("Температура (K)", 0.1, 15.0, 4.2, 0.1)
            current_temp = T_common
        elif comparison_mode == "Один стан":
            selected_state = st.radio("Стан:", ["Надпровідник", "Звичайний метал"])
            if selected_state == "Надпровідник":
                T_super = st.slider("Температура надпровідника (K)", 0.1, Tc-0.1, 4.2, 0.1)
                current_temp = T_super
            else:
                T_normal = st.slider("Температура металу (K)", 0.1, 15.0, 4.2, 0.1)
                current_temp = T_normal
        else:
            T_multi = st.slider("Температура (K)", 0.1, 15.0, 4.2, 0.1)
            current_temp = T_multi

    # Основний контент
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Графіки струму")
        
        t = np.linspace(0, t_max, 1000)
        fig = go.Figure()
        
        # ВИПРАВЛЕННЯ: таблиці будуються для всіх режимів
        physical_analyses = []
        math_analyses = []
        
        if comparison_mode == "Порівняння":
            j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0)
            j_normal = calculate_normal_current(t, field_type, T_common, E0, a, omega, j0)
            
            fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник', 
                                   line=dict(color='red', width=3)))
            fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний метал',
                                   line=dict(color='blue', width=3)))
            
            physical_analyses = [
                analyze_physical_characteristics(t, j_super, "Надпровідник", field_type, T_common, omega),
                analyze_physical_characteristics(t, j_normal, "Звичайний метал", field_type, T_common, omega)
            ]
            math_analyses = [
                analyze_mathematical_characteristics(t, j_super, "Надпровідник", field_type),
                analyze_mathematical_characteristics(t, j_normal, "Звичайний метал", field_type)
            ]
            
        elif comparison_mode == "Один стан":
            if 'T_super' in locals():
                j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0)
                fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник',
                                       line=dict(color='red', width=3)))
                physical_analyses = [analyze_physical_characteristics(t, j_super, "Надпровідник", field_type, T_super, omega)]
                math_analyses = [analyze_mathematical_characteristics(t, j_super, "Надпровідник", field_type)]
            else:
                j_normal = calculate_normal_current(t, field_type, T_normal, E0, a, omega, j0)
                fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний метал',
                                       line=dict(color='blue', width=3)))
                physical_analyses = [analyze_physical_characteristics(t, j_normal, "Звичайний метал", field_type, T_normal, omega)]
                math_analyses = [analyze_mathematical_characteristics(t, j_normal, "Звичайний метал", field_type)]
        
        else:
            j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0)
            j_normal = calculate_normal_current(t, field_type, T_multi, E0, a, omega, j0)
            
            fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник',
                                   line=dict(color='red', width=3)))
            fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний метал',
                                   line=dict(color='blue', width=3)))
            
            physical_analyses = [
                analyze_physical_characteristics(t, j_super, "Надпровідник", field_type, T_multi, omega),
                analyze_physical_characteristics(t, j_normal, "Звичайний метал", field_type, T_multi, omega)
            ]
            math_analyses = [
                analyze_mathematical_characteristics(t, j_super, "Надпровідник", field_type),
                analyze_mathematical_characteristics(t, j_normal, "Звичайний метал", field_type)
            ]
        
        fig.update_layout(
            title="Динаміка густини струму",
            xaxis_title="Час (с)",
            yaxis_title="Густина струму (А/м²)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ВИПРАВЛЕННЯ: таблиці показуються завжди, коли є дані
        if physical_analyses:
            st.header("📊 Фізичний аналіз")
            physical_df = pd.DataFrame(physical_analyses)
            st.dataframe(physical_df, use_container_width=True, height=200)
            
            st.header("🧮 Математичний аналіз")
            if len(math_analyses) == 2:
                col_math1, col_math2 = st.columns(2)
                with col_math1:
                    st.write("**Надпровідник:**")
                    math_df_super = pd.DataFrame([math_analyses[0]])
                    st.dataframe(math_df_super.T.rename(columns={0: 'Значення'}), use_container_width=True, height=300)
                with col_math2:
                    st.write("**Звичайний метал:**")
                    math_df_normal = pd.DataFrame([math_analyses[1]])
                    st.dataframe(math_df_normal.T.rename(columns={0: 'Значення'}), use_container_width=True, height=300)
            else:
                math_df = pd.DataFrame([math_analyses[0]])
                st.dataframe(math_df.T.rename(columns={0: 'Значення'}), use_container_width=True, height=300)

    with col2:
        st.header("📋 Інформація")
        
        st.subheader("Параметри розрахунку")
        st.write(f"**Тип поля:** {field_type}")
        st.write(f"**E₀ =** {E0} В/м")
        st.write(f"**j₀ =** {j0} А/м²")
        st.write(f"**Температура:** {current_temp} K")
        
        if current_temp < Tc:
            st.success("✅ Температура нижче Tкрит")
        else:
            st.warning("⚠️ Температура вище Tкрит")

        st.header("📄 Експорт результатів")
        if st.button("📥 Згенерувати звіт", use_container_width=True) and physical_analyses:
            input_data = {
                'field_type': field_type,
                'E0': E0,
                'j0': j0,
                't_max': t_max,
                'T_common': current_temp,
            }
            
            pdf_buffer = create_comprehensive_pdf_report(input_data, physical_analyses, math_analyses)
            st.download_button(
                label="⬇️ Завантажити PDF",
                data=pdf_buffer,
                file_name="звіт_моделювання.pdf",
                mime="application/pdf",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
