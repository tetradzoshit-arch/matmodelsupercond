import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from io import BytesIO
import base64
from scipy.signal import find_peaks

# ФІЗИЧНІ КОНСТАНТИ ДЛЯ НІОБІЮ (ПЕРЕВІРЕНІ)
e = 1.602e-19  # Кл (елементарний заряд)
m = 9.109e-31  # кг (маса електрона)
ħ = 1.054e-34  # Дж·с (постійна Дірака)
kB = 1.3806e-23  # Дж/К (стала Больцмана)

# Параметри ніобію
Tc = 9.2  # К (критична температура)
Δ0 = 1.76 * kB * Tc  # енергетична щілина при T=0
n0 = 2.8e28  # м⁻³ (концентрація електронів)
tau_imp = 1.0e-12  # с (час релаксації імпуріті)

def tau_temperature_dependence(T):
    """Залежність часу релаксації від температури"""
    return tau_imp / (1 + (T / Tc)**3)

def calculate_superconducting_current(t, E_type, E0=1.0, a=1.0, omega=1.0, j0=0.0, T=4.2):
    """Розрахунок струму в надпровідному стані - рівняння Лондонів"""
    # Концентрація надпровідних електронів
    if T < Tc:
        ns = n0 * (1.0 - (T / Tc)**4.0)
    else:
        ns = 0.0
    
    K = (e**2 * ns) / m
    
    if E_type == "Статичне":
        return j0 + K * E0 * t
    elif E_type == "Лінійне":
        return j0 + K * (a * t**2) / 2
    elif E_type == "Синусоїдальне":
        return j0 + (K * E0 / omega) * (1 - np.cos(omega * t))

def calculate_normal_current_drude(t, E_type, T, E0=1.0, a=1.0, omega=1.0, j0=0.0):
    """Розрахунок струму в звичайному стані - модель Друде з перехідним процесом"""
    tau_T = tau_temperature_dependence(T)
    sigma = (n0 * e**2 * tau_T) / m
    
    if E_type == "Статичне":
        return j0 * np.exp(-t/tau_T) + sigma * E0 * (1.0 - np.exp(-t/tau_T))
    elif E_type == "Лінійне":
        return j0 * np.exp(-t/tau_T) + sigma * a * (t - tau_T * (1.0 - np.exp(-t/tau_T)))
    elif E_type == "Синусоїдальне":
        omega_tau_sq = (omega * tau_T)**2.0
        amp_factor = sigma / np.sqrt(1.0 + omega_tau_sq)
        phase_shift = np.arctan(omega * tau_T)
        J_steady = E_0 * amp_factor * np.sin(omega * t - phase_shift)
        C = j0 - E_0 * amp_factor * np.sin(-phase_shift)
        J_transient = C * np.exp(-t / tau_T)
        return J_transient + J_steady

def calculate_normal_current_ohm(t, E_type, T, E0=1.0, a=1.0, omega=1.0, j0=0.0):
    """Розрахунок струму в звичайному стані - закон Ома (стаціонарний)"""
    tau_T = tau_temperature_dependence(T)
    sigma = (n0 * e**2 * tau_T) / m
    
    if E_type == "Статичне":
        return sigma * E0 * np.ones_like(t)
    elif E_type == "Лінійне":
        return sigma * a * t
    elif E_type == "Синусоїдальне":
        return sigma * E0 * np.sin(omega * t)

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
            tau_val = tau_temperature_dependence(T)
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
    
    # Ініціалізація збережених графіків
    if 'saved_plots' not in st.session_state:
        st.session_state.saved_plots = []
    
    with st.sidebar:
        st.header("⚙️ Параметри моделювання")
        
        comparison_mode = st.radio(
            "Режим:",
            ["Один стан", "Порівняння", "Кілька графіків", "Збережені графіки"]
        )
        
        st.subheader("Загальні параметри")
        field_type = st.selectbox("Тип поля:", ["Статичне", "Лінійне", "Синусоїдальне"])
        E0 = st.slider("Напруженість E₀ (В/м)", 0.1, 100.0, 1.0, 0.1)
        j0 = st.slider("Початковий струм j₀ (А/м²)", 0.0, 100.0, 0.0, 0.1)
        t_max = st.slider("Час моделювання (с)", 0.1, 20.0, 5.0, 0.1)
        
        if field_type == "Лінійне":
            a = st.slider("Швидкість росту a", 0.1, 10.0, 1.0, 0.1)
        else:
            a = 1.0
            
        if field_type == "Синусоїдальне":
            omega = st.slider("Частота ω (рад/с)", 0.1, 50.0, 5.0, 0.1)
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
                # ВИПРАВЛЕННЯ: температура тільки до 15K для нормального металу
                T_normal = st.slider("Температура металу (K)", 0.1, 15.0, 4.2, 0.1)
                current_temp = T_normal
                # ВИПРАВЛЕННЯ: вибір моделі для нормального металу
                metal_model = st.radio("Модель для металу:", 
                                     ["Модель Друде (з перехідним процесом)", "Закон Ома (стаціонарний)"])
        else:
            T_multi = st.slider("Температура (K)", 0.1, 15.0, 4.2, 0.1)
            current_temp = T_multi

        # Кнопка збереження поточного графіка
        if comparison_mode in ["Один стан", "Порівняння", "Кілька графіків"]:
            if st.button("💾 Зберегти поточний графік", use_container_width=True):
                plot_data = {
                    't': np.linspace(0, t_max, 1000),
                    'field_type': field_type,
                    'E0': E0,
                    'j0': j0,
                    'a': a,
                    'omega': omega,
                    'temperature': current_temp,
                    'mode': comparison_mode,
                    'timestamp': pd.Timestamp.now()
                }
                
                if comparison_mode == "Один стан":
                    if selected_state == "Надпровідник":
                        plot_data['j_super'] = calculate_superconducting_current(
                            plot_data['t'], field_type, E0, a, omega, j0, current_temp
                        )
                        plot_data['state'] = 'Надпровідник'
                        plot_data['model'] = 'Лондони'
                    else:
                        if metal_model == "Модель Друде (з перехідним процесом)":
                            plot_data['j_normal'] = calculate_normal_current_drude(
                                plot_data['t'], field_type, current_temp, E0, a, omega, j0
                            )
                        else:
                            plot_data['j_normal'] = calculate_normal_current_ohm(
                                plot_data['t'], field_type, current_temp, E0, a, omega, j0
                            )
                        plot_data['state'] = 'Звичайний метал'
                        plot_data['model'] = metal_model
                elif comparison_mode == "Порівняння":
                    plot_data['j_super'] = calculate_superconducting_current(
                        plot_data['t'], field_type, E0, a, omega, j0, T_common
                    )
                    plot_data['j_normal'] = calculate_normal_current_drude(
                        plot_data['t'], field_type, T_common, E0, a, omega, j0
                    )
                    plot_data['state'] = 'Порівняння'
                    plot_data['model'] = 'Друде'
                else:  # Кілька графіків
                    plot_data['j_super'] = calculate_superconducting_current(
                        plot_data['t'], field_type, E0, a, omega, j0, T_multi
                    )
                    plot_data['j_normal'] = calculate_normal_current_drude(
                        plot_data['t'], field_type, T_multi, E0, a, omega, j0
                    )
                    plot_data['state'] = 'Кілька графіків'
                    plot_data['model'] = 'Друде'
                
                st.session_state.saved_plots.append(plot_data)
                st.success(f"Графік збережено! Всього збережено: {len(st.session_state.saved_plots)}")

        # Кнопка очищення всіх збережених графіків
        if st.session_state.saved_plots and st.button("🗑️ Очистити всі збережені графіки", use_container_width=True):
            st.session_state.saved_plots = []
            st.success("Всі збережені графіки видалено!")

    # Основний контент
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if comparison_mode == "Збережені графіки":
            st.header("📊 Збережені графіки")
            
            if not st.session_state.saved_plots:
                st.info("Немає збережених графіків. Збережіть графіки в інших режимах.")
            else:
                fig_saved = go.Figure()
                
                for i, plot_data in enumerate(st.session_state.saved_plots):
                    color_idx = i % 10  # Для різних кольорів
                    
                    if plot_data['state'] == 'Надпровідник':
                        fig_saved.add_trace(go.Scatter(
                            x=plot_data['t'], 
                            y=plot_data['j_super'], 
                            name=f"Надпровідник {i+1} (T={plot_data['temperature']}K)",
                            line=dict(width=2),
                            opacity=0.7
                        ))
                    elif plot_data['state'] == 'Звичайний метал':
                        fig_saved.add_trace(go.Scatter(
                            x=plot_data['t'], 
                            y=plot_data['j_normal'], 
                            name=f"Метал {i+1} (T={plot_data['temperature']}K, {plot_data['model']})",
                            line=dict(width=2),
                            opacity=0.7
                        ))
                    elif plot_data['state'] in ['Порівняння', 'Кілька графіків']:
                        fig_saved.add_trace(go.Scatter(
                            x=plot_data['t'], 
                            y=plot_data['j_super'], 
                            name=f"Надпровідник {i+1}",
                            line=dict(width=2),
                            opacity=0.7
                        ))
                        fig_saved.add_trace(go.Scatter(
                            x=plot_data['t'], 
                            y=plot_data['j_normal'], 
                            name=f"Метал {i+1}",
                            line=dict(width=2),
                            opacity=0.7
                        ))
                
                fig_saved.update_layout(
                    title="Усі збережені графіки",
                    xaxis_title="Час (с)",
                    yaxis_title="Густина струму (А/м²)",
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig_saved, use_container_width=True)
                
                # Інформація про збережені графіки
                st.subheader("Інформація про збережені графіки")
                for i, plot_data in enumerate(st.session_state.saved_plots):
                    with st.expander(f"Графік {i+1} - {plot_data['state']}"):
                        st.write(f"Тип поля: {plot_data['field_type']}")
                        st.write(f"E₀: {plot_data['E0']} В/м")
                        st.write(f"j₀: {plot_data['j0']} А/м²")
                        st.write(f"Температура: {plot_data['temperature']} K")
                        st.write(f"Модель: {plot_data.get('model', 'Лондони')}")
                        st.write(f"Збережено: {plot_data['timestamp']}")
                        
                        if st.button(f"Видалити графік {i+1}", key=f"delete_{i}"):
                            st.session_state.saved_plots.pop(i)
                            st.rerun()
        
        else:
            st.header("📈 Графіки струму")
            
            t = np.linspace(0, t_max, 1000)
            fig = go.Figure()
            
            physical_analyses = []
            math_analyses = []
            
            if comparison_mode == "Порівняння":
                j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0, T_common)
                j_normal = calculate_normal_current_drude(t, field_type, T_common, E0, a, omega, j0)
                
                fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник', 
                                       line=dict(color='red', width=3)))
                fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний метал (Друде)',
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
                if selected_state == "Надпровідник":
                    j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0, T_super)
                    fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник',
                                           line=dict(color='red', width=3)))
                    physical_analyses = [analyze_physical_characteristics(t, j_super, "Надпровідник", field_type, T_super, omega)]
                    math_analyses = [analyze_mathematical_characteristics(t, j_super, "Надпровідник", field_type)]
                else:
                    if metal_model == "Модель Друде (з перехідним процесом)":
                        j_normal = calculate_normal_current_drude(t, field_type, T_normal, E0, a, omega, j0)
                        model_name = "Звичайний метал (Друде)"
                    else:
                        j_normal = calculate_normal_current_ohm(t, field_type, T_normal, E0, a, omega, j0)
                        model_name = "Звичайний метал (Ом)"
                    
                    fig.add_trace(go.Scatter(x=t, y=j_normal, name=model_name,
                                           line=dict(color='blue', width=3)))
                    physical_analyses = [analyze_physical_characteristics(t, j_normal, "Звичайний метал", field_type, T_normal, omega)]
                    math_analyses = [analyze_mathematical_characteristics(t, j_normal, "Звичайний метал", field_type)]
            
            else:  # Кілька графіків
                j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0, T_multi)
                j_normal = calculate_normal_current_drude(t, field_type, T_multi, E0, a, omega, j0)
                
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
            
            # Аналітичні таблиці
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

        # Інформація про фізичні константи
        with st.expander("Фізичні константи"):
            st.write(f"**e =** {e:.3e} Кл")
            st.write(f"**m =** {m:.3e} кг")
            st.write(f"**n₀ =** {n0:.2e} м⁻³")
            st.write(f"**τ_imp =** {tau_imp:.2e} с")
            st.write(f"**T_c =** {Tc} K")

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
