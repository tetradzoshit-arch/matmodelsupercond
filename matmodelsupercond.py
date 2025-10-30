import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO
from scipy.signal import find_peaks
import tempfile
import os
from reportlab.lib.utils import ImageReader
import time
import random

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
    analysis = {
        'Стан': state_name,
        'Температура': f"{T} K",
        'j(0)': f"{j_data[0]:.2e} А/м²",
        'j(t_max)': f"{j_data[-1]:.2e} А/м²",
        'j_max': f"{np.max(j_data):.2e} А/м²",
        'j_min': f"{np.min(j_data):.2e} А/м²",
        'Амплітуда': f"{np.max(j_data) - np.min(j_data):.2e} А/м²"
    }
    
    dt = t[1] - t[0]
    dj_dt = np.gradient(j_data, dt)
    analysis['Макс. швидкість'] = f"{np.max(dj_dt):.2e} А/м²с"
    
    if field_type == "Статичне":
        analysis['Поведінка'] = "Лінійне зростання" if state_name == "Надпровідник" else "Експоненційне насичення"
    elif field_type == "Лінійне":
        analysis['Поведінка'] = "Квадратичне зростання" if state_name == "Надпровідник" else "Експоненційне насилення"
    elif field_type == "Синусоїдальне":
        if state_name == "Надпровідник":
            analysis['Поведінка'] = "Коливання"
            analysis['Фазовий зсув'] = "π/2 (струм випереджає поле)"
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
        analysis['Період'] = f"{2*np.pi/omega:.2f} с" if omega and omega > 0 else "∞"
    
    return analysis

def create_pdf_report(input_data, physical_analyses, math_analyses, saved_plots):
    """Створення PDF звіту"""
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import io
        
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=landscape(A4))
        
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
            font_name = 'DejaVuSans'
        except:
            font_name = 'Helvetica'
        
        # Сторінка 1: Загальна інформація
        pdf.setFont(font_name, 16)
        pdf.drawString(100, 520, "ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ В НІОБІЇ")
        
        pdf.setFont(font_name, 12)
        y_position = 490
        
        # Параметри моделювання
        pdf.drawString(100, y_position, "Параметри моделювання:")
        y_position -= 20
        pdf.drawString(120, y_position, f"- Тип поля: {input_data['field_type']}")
        y_position -= 20
        pdf.drawString(120, y_position, f"- Напруженість поля E₀: {input_data['E0']} В/м")
        y_position -= 20
        pdf.drawString(120, y_position, f"- Початковий струм j₀: {input_data['j0']} А/м²")
        y_position -= 20
        pdf.drawString(120, y_position, f"- Час моделювання: {input_data['t_max']} с")
        y_position -= 20
        pdf.drawString(120, y_position, f"- Температура: {input_data['T_common']} K")
        y_position -= 30

        # Фізичний аналіз
        if physical_analyses:
            pdf.drawString(100, y_position, "Фізичний аналіз:")
            y_position -= 25
            
            col_widths = [120, 80, 100, 100, 180]
            row_height = 20
            
            pdf.setFillColorRGB(0.8, 0.8, 1.0)
            pdf.rect(100, y_position - row_height, sum(col_widths), row_height, fill=1)
            pdf.setFillColorRGB(0, 0, 0)
            
            headers = ["Стан", "Температура", "j(0)", "j_max", "Поведінка"]
            x_pos = 100
            for i, header in enumerate(headers):
                pdf.drawString(x_pos + 5, y_position - 15, header)
                x_pos += col_widths[i]
            
            y_position -= row_height
            
            for i, analysis in enumerate(physical_analyses):
                if i % 2 == 0:
                    pdf.setFillColorRGB(0.95, 0.95, 0.95)
                else:
                    pdf.setFillColorRGB(1, 1, 1)
                
                pdf.rect(100, y_position - row_height, sum(col_widths), row_height, fill=1)
                pdf.setFillColorRGB(0, 0, 0)
                
                x_pos = 100
                cells = [
                    analysis.get('Стан', ''),
                    analysis.get('Температура', ''),
                    analysis.get('j(0)', ''),
                    analysis.get('j_max', ''),
                    analysis.get('Поведінка', '')
                ]
                
                for j, cell in enumerate(cells):
                    pdf.drawString(x_pos + 5, y_position - 15, cell)
                    x_pos += col_widths[j]
                
                y_position -= row_height
                if y_position < 100:
                    pdf.showPage()
                    pdf.setFont(font_name, 12)
                    y_position = 490
                    pdf.setFillColorRGB(0.8, 0.8, 1.0)
                    pdf.rect(100, y_position - row_height, sum(col_widths), row_height, fill=1)
                    pdf.setFillColorRGB(0, 0, 0)
                    x_pos = 100
                    for k, header in enumerate(headers):
                        pdf.drawString(x_pos + 5, y_position - 15, header)
                        x_pos += col_widths[k]
                    y_position -= row_height
            
            y_position -= 25

        # Математичний аналіз
        if math_analyses:
            pdf.drawString(100, y_position, "Математичний аналіз:")
            y_position -= 25
            
            col_widths = [100, 100, 80, 80, 80, 80, 80]
            row_height = 20
            
            pdf.setFillColorRGB(0.8, 1.0, 0.8)
            pdf.rect(100, y_position - row_height, sum(col_widths), row_height, fill=1)
            pdf.setFillColorRGB(0, 0, 0)
            
            headers = ["Функція", "Тип функції", "f(0)", "max f(t)", "f'(max)", "f'(min)", "f'(сер)"]
            x_pos = 100
            for i, header in enumerate(headers):
                pdf.drawString(x_pos + 3, y_position - 15, header)
                x_pos += col_widths[i]
            
            y_position -= row_height
            
            for i, analysis in enumerate(math_analyses):
                if i % 2 == 0:
                    pdf.setFillColorRGB(0.95, 1.0, 0.95)
                else:
                    pdf.setFillColorRGB(1, 1, 1)
                
                pdf.rect(100, y_position - row_height, sum(col_widths), row_height, fill=1)
                pdf.setFillColorRGB(0, 0, 0)
                
                x_pos = 100
                
                cells = [
                    analysis.get('Функція', ''),
                    analysis.get('Тип функції', ''),
                    analysis.get('f(0)', ''),
                    analysis.get('max f(t)', ''),
                    analysis.get("f'(max)", 'N/A'),
                    analysis.get("f'(min)", 'N/A'),
                    analysis.get("f'(сер)", 'N/A')
                ]
                
                if "f'(сер)" not in analysis:
                    if "f'(середнє)" in analysis:
                        cells[6] = analysis["f'(середнє)"]
                    elif "f'(сер)" in analysis:
                        cells[6] = analysis["f'(сер)"]
                
                for j, cell in enumerate(cells):
                    pdf.drawString(x_pos + 3, y_position - 15, cell)
                    x_pos += col_widths[j]
                
                y_position -= row_height
                if y_position < 100:
                    pdf.showPage()
                    pdf.setFont(font_name, 12)
                    y_position = 490
                    pdf.setFillColorRGB(0.8, 1.0, 0.8)
                    pdf.rect(100, y_position - row_height, sum(col_widths), row_height, fill=1)
                    pdf.setFillColorRGB(0, 0, 0)
                    x_pos = 100
                    for k, header in enumerate(headers):
                        pdf.drawString(x_pos + 3, y_position - 15, header)
                        x_pos += col_widths[k]
                    y_position -= row_height
            
            y_position -= 25
        
        # Висновки
        pdf.drawString(100, y_position, "Висновки та аналіз результатів:")
        y_position -= 25
        
        conclusions = [
            "• Надпровідник демонструє принципово іншу динаміку струму:",
            "  - Струм необмежено зростає з часом через відсутність опору",
            "",
            "• Звичайний метал має властивості насичення:",
            "  - Струм досягає стаціонарного значення через опір", 
            "  - Час релаксації впливає на швидкість встановлення струму",
            "",
            "• Аналіз похідних показує швидкість змін:",
            "  - f'(max) - максимальна швидкість зростання струму",
            "  - f'(min) - максимальна швидкість спадання струму",
            "  - f'(сер) - середня швидкість зміни струму за весь час"
        ]
        
        for conclusion in conclusions:
            if conclusion.startswith("•") or conclusion.startswith("  -"):
                pdf.drawString(120, y_position, conclusion)
            else:
                pdf.drawString(100, y_position, conclusion)
            y_position -= 15
            
            if y_position < 50:
                pdf.showPage()
                pdf.setFont(font_name, 12)
                y_position = 490
        
        # Інформація про збережені графіки (лише текст)
        if saved_plots:
            pdf.showPage()
            pdf.setFont(font_name, 16)
            pdf.drawString(100, 520, "ІНФОРМАЦІЯ ПРО ЗБЕРЕЖЕНІ ГРАФІКИ")
            pdf.setFont(font_name, 12)
            y_position = 490
            
            pdf.drawString(100, y_position, f"Кількість збережених графіків: {len(saved_plots)}")
            y_position -= 30
            
            for i, plot_data in enumerate(saved_plots):
                if y_position < 100:
                    pdf.showPage()
                    pdf.setFont(font_name, 12)
                    y_position = 490
                
                pdf.setFont(font_name, 14)
                pdf.drawString(100, y_position, f"Графік {i+1}:")
                y_position -= 20
                
                pdf.setFont(font_name, 12)
                pdf.drawString(120, y_position, f"Стан: {plot_data['state']}")
                y_position -= 20
                pdf.drawString(120, y_position, f"Температура: {plot_data['temperature']} K")
                y_position -= 20
                pdf.drawString(120, y_position, f"Тип поля: {plot_data['field_type']}")
                y_position -= 20
                pdf.drawString(120, y_position, f"E₀: {plot_data['E0']} В/м")
                y_position -= 20
                pdf.drawString(120, y_position, f"j₀: {plot_data['j0']} А/м²")
                y_position -= 30
        
        pdf.save()
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        print(f"Помилка при створенні PDF: {e}")
        buffer = BytesIO()
        report_text = "ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ В НІОБІЇ\n\n"
        report_text += "Параметри моделювання:\n"
        for key, value in input_data.items():
            report_text += f"{key}: {value}\n"
        buffer.write(report_text.encode('utf-8'))
        buffer.seek(0)
        return buffer

# =============================================================================
# НОВІ СТОРІНКИ (ДОДАТКОВІ)
# =============================================================================
# =============================================================================
# СТОРІНКА АНІМАЦІЙ
# =============================================================================

def animations_page():
    st.header("🎬 Демонстраційні анімації")
    
    # Параметри для всіх анімацій
    col_params, col_main = st.columns([1, 2])
    
    with col_params:
        st.subheader("⚙️ Параметри анімацій")
        anim_field_type = st.selectbox("Тип поля:", ["Статичне", "Лінійне", "Синусоїдальне"], key="anim_field")
        anim_E0 = st.slider("Напруженість E₀ (В/м)", 0.1, 10.0, 1.0, 0.1, key="anim_E0")
        anim_j0 = st.slider("Початковий струм j₀ (А/м²)", 0.0, 10.0, 0.0, 0.1, key="anim_j0")
        anim_t_max = st.slider("Час моделювання (с)", 0.1, 10.0, 5.0, 0.1, key="anim_t_max")
        anim_speed = st.slider("Швидкість анімації", 0.1, 2.0, 0.5, 0.1, key="anim_speed")
        
        anim_a = st.slider("Швидкість росту a", 0.1, 5.0, 1.0, 0.1, key="anim_a") if anim_field_type == "Лінійне" else 1.0
        anim_omega = st.slider("Частота ω (рад/с)", 0.1, 20.0, 5.0, 0.1, key="anim_omega") if anim_field_type == "Синусоїдальне" else 1.0

    with col_main:
        # Анімація 1: Зміна температури
        st.subheader("🌡️ Анімація зміни температури")
        st.write("Плавна зміна температури від 1K до 18K")
        
        if st.button("▶️ Запустити температурну анімацію", key="temp_anim", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            plot_placeholder = st.empty()
            
            temps = np.linspace(1, 18, 25)  # Меньше кадров для скорости
            
            for i, temp in enumerate(temps):
                progress = int((i / len(temps)) * 100)
                progress_bar.progress(progress)
                
                state = "Надпровідник" if temp < Tc else "Метал"
                status_text.text(f"Температура: {temp:.1f} K | Стан: {state}")
                
                t_anim = np.linspace(0, anim_t_max, 150)
                j_super = calculate_superconducting_current(t_anim, anim_field_type, anim_E0, anim_a, anim_omega, anim_j0, temp)
                j_normal = calculate_normal_current_drude(t_anim, anim_field_type, temp, anim_E0, anim_a, anim_omega, anim_j0)
                
                fig_anim = go.Figure()
                fig_anim.add_trace(go.Scatter(x=t_anim, y=j_super, name='Надпровідник', 
                                            line=dict(color='red', width=3)))
                fig_anim.add_trace(go.Scatter(x=t_anim, y=j_normal, name='Метал', 
                                            line=dict(color='blue', width=3)))
                
                fig_anim.update_layout(
                    title=f"T = {temp:.1f} K ({state})",
                    xaxis_title="Час (с)",
                    yaxis_title="Густина струму (А/м²)",
                    height=400
                )
                fig_anim.update_yaxes(tickformat=".2e")
                
                plot_placeholder.plotly_chart(fig_anim, use_container_width=True)
                time.sleep(0.5 / anim_speed)
            
            progress_bar.progress(100)
            status_text.text("✅ Анімація завершена!")
        
        st.markdown("---")
        
        # Анімація 2: Перехід через Tc
        st.subheader("⚡ Анімація переходу через T_c")
        st.write("Детальний перехід через критичну температуру")
        
        if st.button("▶️ Запустити анімацію переходу", key="transition_anim", use_container_width=True):
            progress_bar2 = st.progress(0)
            status_text2 = st.empty()
            plot_placeholder2 = st.empty()
            
            transition_temps = np.linspace(7.0, 11.0, 20)
            
            for i, T_trans in enumerate(transition_temps):
                progress = int((i / len(transition_temps)) * 100)
                progress_bar2.progress(progress)
                
                state = "Надпровідник" if T_trans < Tc else "Метал"
                status_text2.text(f"T = {T_trans:.2f} K | Стан: {state}")
                
                t_trans = np.linspace(0, min(anim_t_max, 3.0), 100)
                
                if T_trans < Tc:
                    j_data = calculate_superconducting_current(t_trans, anim_field_type, anim_E0, anim_a, anim_omega, anim_j0, T_trans)
                    color = 'red'
                else:
                    j_data = calculate_normal_current_drude(t_trans, anim_field_type, T_trans, anim_E0, anim_a, anim_omega, anim_j0)
                    color = 'blue'
                
                fig_trans = go.Figure()
                fig_trans.add_trace(go.Scatter(x=t_trans, y=j_data, name=state,
                                             line=dict(color=color, width=4)))
                
                fig_trans.update_layout(
                    title=f"Перехід через T_c: {T_trans:.2f} K",
                    xaxis_title="Час (с)",
                    yaxis_title="Густина струму (А/м²)",
                    height=400,
                    showlegend=True
                )
                fig_trans.update_yaxes(tickformat=".2e")
                
                plot_placeholder2.plotly_chart(fig_trans, use_container_width=True)
                time.sleep(0.5 / anim_speed)
            
            progress_bar2.progress(100)
            status_text2.text("✅ Перехід завершено!")
        
        st.markdown("---")
        
        # Анімація 3: Порівняння типів полів
        st.subheader("🔄 Порівняння типів полів")
        st.write("Порівняння поведінки для різних типів електричних полів")
        
        temp_comparison = st.slider("Температура для порівняння", 1.0, 18.0, 4.2, 0.1, key="temp_comp")
        
        if st.button("▶️ Порівняти типи полів", key="field_comparison", use_container_width=True):
            plot_placeholder3 = st.empty()
            progress_bar3 = st.progress(0)
            
            field_types = ["Статичне", "Лінійне", "Синусоїдальне"]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            t_comp = np.linspace(0, anim_t_max, 200)
            
            for i, field_type in enumerate(field_types):
                progress = int((i / len(field_types)) * 100)
                progress_bar3.progress(progress)
                
                fig_comp = go.Figure()
                
                # Надпровідник
                j_super = calculate_superconducting_current(t_comp, field_type, anim_E0, 1.0, 5.0, 0.0, temp_comparison)
                fig_comp.add_trace(go.Scatter(x=t_comp, y=j_super, 
                                            name=f'Надпровідник - {field_type}',
                                            line=dict(color=colors[i], width=3, dash='solid')))
                
                # Метал
                j_normal = calculate_normal_current_drude(t_comp, field_type, temp_comparison, anim_E0, 1.0, 5.0, 0.0)
                fig_comp.add_trace(go.Scatter(x=t_comp, y=j_normal, 
                                            name=f'Метал - {field_type}',
                                            line=dict(color=colors[i], width=3, dash='dot')))
                
                fig_comp.update_layout(
                    title=f"Порівняння типів полів при T = {temp_comparison}K",
                    xaxis_title="Час (с)",
                    yaxis_title="Густина струму (А/м²)",
                    height=500
                )
                fig_comp.update_yaxes(tickformat=".2e")
                
                plot_placeholder3.plotly_chart(fig_comp, use_container_width=True)
                time.sleep(1.0 / anim_speed)
            
            progress_bar3.progress(100)
            st.success("✅ Порівняння завершено!")

# =============================================================================
# СТОРІНКА ГОНОК
# =============================================================================

def racing_page():
    st.header("🏎️ Електронні Гонки - Надпровідник vs Метал")
    
    # Ініціалізація стану гонки
    if 'race_started' not in st.session_state:
        st.session_state.race_started = False
    if 'race_frame' not in st.session_state:
        st.session_state.race_frame = 0
    if 'race_data' not in st.session_state:
        st.session_state.race_data = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚦 Параметри гонки")
        
        # Автоматичний вибір типу за температурою
        st.write("**Обери температури для машинок:**")
        
        col_car1, col_car2 = st.columns(2)
        with col_car1:
            car1_temp = st.slider("Температура машинки 1 (K)", 1.0, 18.0, 4.2, 0.1, key="car1_temp")
            car1_type = "Надпровідник" if car1_temp < Tc else "Метал"
            st.info(f"**Машинка 1:** {car1_type}")
            
        with col_car2:
            car2_temp = st.slider("Температура машинки 2 (K)", 1.0, 18.0, 12.0, 0.1, key="car2_temp")
            car2_type = "Надпровідник" if car2_temp < Tc else "Метал"
            st.info(f"**Машинка 2:** {car2_type}")
        
        # Загальні параметри
        race_field = st.selectbox("Тип поля:", ["Статичне", "Лінійне", "Синусоїдальне"], key="race_field")
        race_E0 = st.slider("Потужність поля E₀", 0.1, 5.0, 1.0, 0.1, key="race_E0")
        race_speed = st.slider("Швидкість анімації", 0.5, 3.0, 1.0, 0.1, key="race_speed")
        
        if st.button("🎮 Старт гонки!", use_container_width=True) and not st.session_state.race_started:
            # Підготовка даних для гонки
            t_race = np.linspace(0, 4, 25)  # Зменшена кількість кадрів
            
            # Розрахунок для машинки 1
            if car1_type == "Надпровідник":
                j_car1 = calculate_superconducting_current(t_race, race_field, race_E0, 1.0, 5.0, 0.0, car1_temp)
            else:
                j_car1 = calculate_normal_current_drude(t_race, race_field, car1_temp, race_E0, 1.0, 5.0, 0.0)
            
            # Розрахунок для машинки 2
            if car2_type == "Надпровідник":
                j_car2 = calculate_superconducting_current(t_race, race_field, race_E0, 1.0, 5.0, 0.0, car2_temp)
            else:
                j_car2 = calculate_normal_current_drude(t_race, race_field, car2_temp, race_E0, 1.0, 5.0, 0.0)
            
            # Збереження даних
            st.session_state.race_data = {
                't_race': t_race,
                'j_car1': j_car1,
                'j_car2': j_car2,
                'car1_type': car1_type,
                'car2_type': car2_type,
                'car1_temp': car1_temp,
                'car2_temp': car2_temp,
                'race_speed': race_speed
            }
            st.session_state.race_started = True
            st.session_state.race_frame = 0
            st.rerun()
    
    with col2:
        st.subheader("📊 Стан системи")
        
        if st.session_state.race_data:
            data = st.session_state.race_data
            st.write(f"**🏎️ Машинка 1:** {data['car1_type']} ({data['car1_temp']}K)")
            st.write(f"**🚗 Машинка 2:** {data['car2_type']} ({data['car2_temp']}K)")
        else:
            st.write(f"**🏎️ Машинка 1:** {car1_type} ({car1_temp}K)")
            st.write(f"**🚗 Машинка 2:** {car2_type} ({car2_temp}K)")
        
        st.metric("Критична температура T_c", f"{Tc} K")
    
    # Гонкова траса
    if st.session_state.race_started and st.session_state.race_data:
        data = st.session_state.race_data
        frame = st.session_state.race_frame
        
        if frame < len(data['t_race']):
            st.subheader("🏁 ГОНКА ТРИВАЄ!")
            
            progress_car1 = int((frame / len(data['t_race'])) * 100)
            progress_car2 = int((frame / len(data['t_race'])) * 100)
            
            # Корекція прогресу для металу
            if data['car1_type'] == "Метал":
                progress_car1 = min(progress_car1, 80)
            if data['car2_type'] == "Метал":
                progress_car2 = min(progress_car2, 80)
            
            speed_car1 = abs(data['j_car1'][frame])
            speed_car2 = abs(data['j_car2'][frame])
            
            # Візуалізація гонки
            st.write(f"### 🏎️ Машинка 1 - {data['car1_type']}")
            if data['car1_type'] == "Надпровідник":
                st.success("🛣️ Супер-шосе без опору!")
            else:
                st.warning("🚦 Міські пробки з опором!")
            
            st.progress(progress_car1 / 100)
            
            # Траса машинки 1
            track_length = 40
            car1_pos = int(progress_car1 * track_length / 100)
            track1_display = "🏁" + "─" * car1_pos + "🏎️" + "·" * (track_length - car1_pos)
            st.code(track1_display)
            st.write(f"**Швидкість:** {speed_car1:.2e} А/м²")
            
            st.write("---")
            
            # Машинка 2
            st.write(f"### 🚗 Машинка 2 - {data['car2_type']}")
            if data['car2_type'] == "Надпровідник":
                st.success("🛣️ Супер-шосе без опору!")
            else:
                st.warning("🚦 Міські пробки з опором!")
            
            st.progress(progress_car2 / 100)
            
            # Траса машинки 2 з перешкодами
            car2_pos = int(progress_car2 * track_length / 100)
            obstacles = "🚧" * ((frame // 3) % 2) if data['car2_type'] == "Метал" else ""
            track2_display = "🏁" + "─" * car2_pos + "🚗" + "·" * (track_length - car2_pos) + " " + obstacles
            st.code(track2_display)
            st.write(f"**Швидкість:** {speed_car2:.2e} А/м²")
            
            # Статус гонки
            st.info(f"**⏱️ Час гонки: {data['t_race'][frame]:.1f}с** | **📊 Кадр: {frame + 1}/{len(data['t_race'])}**")
            
            # Автоматичне продовження
            st.session_state.race_frame += 1
            time.sleep(1.0 / data['race_speed'])
            st.rerun()
        
        else:
            # Гонка завершена
            st.session_state.race_started = False
            
            # Результати
            st.balloons()
            st.subheader("🎉 Гонка завершена!")
            
            max_car1 = np.max(np.abs(data['j_car1']))
            max_car2 = np.max(np.abs(data['j_car2']))
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                if max_car1 > max_car2:
                    st.success("🏆 Перемога машинки 1!")
                    winner = "🏎️ Машинка 1"
                elif max_car2 > max_car1:
                    st.success("🏆 Перемога машинки 2!")
                    winner = "🚗 Машинка 2"
                else:
                    st.info("🤝 Нічия!")
                    winner = "🤝 Нічия"
                st.metric("Переможець", winner)
            
            with col_res2:
                st.metric("Макс. швидкість 1", f"{max_car1:.2e} А/м²")
                st.metric("Макс. швидкість 2", f"{max_car2:.2e} А/м²")
            
            with col_res3:
                if st.button("🔄 Нова гонка", use_container_width=True):
                    st.session_state.race_started = False
                    st.session_state.race_data = None
                    st.rerun()
    
    else:
        # Екран перед стартом
        st.info("""
        ### 🎮 Інструкція до гри:
        
        **🏎️ Надпровідник (T < 9.2K):**
        - Без опору - електрони летять вільно
        - Швидкість зростає без обмежень
        - Фініш на максимумі
        
        **🚗 Метал (T ≥ 9.2K):**
        - Є опір - електрони "гальмують"
        - Швидкість обмежена
        - Не досягає максимуму
        
        **🎯 Порада:** Встанови температури нижче 9.2K для надпровідників!
        """)

# =============================================================================
# СТОРІНКА ПЕРЕДБАЧЕНЬ
# =============================================================================
def generate_game_problem(difficulty):
    """Генерація випадкової задачі для гри"""
    problems = {
        "easy": [
            {"field": "Статичне", "T": 4.2, "E0": 1.0, "hint": "Надпровідник при низькій температурі"},
            {"field": "Статичне", "T": 12.0, "E0": 1.0, "hint": "Метал при високій температурі"}
        ],
        "medium": [
            {"field": "Лінійне", "T": 4.2, "E0": 0.5, "hint": "Надпровідник з лінійним полем"},
            {"field": "Синусоїдальне", "T": 12.0, "E0": 2.0, "hint": "Метал зі змінним полем"}
        ],
        "hard": [
            {"field": random.choice(["Статичне", "Лінійне", "Синусоїдальне"]), 
             "T": random.uniform(3.0, 15.0), 
             "E0": random.uniform(0.3, 3.0),
             "hint": "Випадкові параметри - вгадай стан!"}
        ]
    }
    
    # Визначення рівня складності
    if "Простий" in difficulty:
        difficulty_key = "easy"
    elif "Середній" in difficulty:
        difficulty_key = "medium"
    else:
        difficulty_key = "hard"
    
    problem = random.choice(problems[difficulty_key])
    
    # Генерація даних
    t_known = np.linspace(0, 2.5, 50)
    t_full = np.linspace(0, 5, 100)
    
    if problem["T"] < Tc:
        j_known = calculate_superconducting_current(t_known, problem["field"], problem["E0"], 1.0, 5.0, 0.0, problem["T"])
        j_full = calculate_superconducting_current(t_full, problem["field"], problem["E0"], 1.0, 5.0, 0.0, problem["T"])
        material_type = "super"
    else:
        j_known = calculate_normal_current_drude(t_known, problem["field"], problem["T"], problem["E0"], 1.0, 5.0, 0.0)
        j_full = calculate_normal_current_drude(t_full, problem["field"], problem["T"], problem["E0"], 1.0, 5.0, 0.0)
        material_type = "metal"
    
    return {
        "t_known": t_known,
        "j_known": j_known,
        "t_full": t_full,
        "j_full": j_full,
        "material_type": material_type,
        "params": problem,
        "hint": problem["hint"]
    }
def prediction_game_page():
    st.header("🔮 Передбач майбутнє провідника!")
    
    st.markdown("""
    ### 🎯 Правила гри:
    1. Дивись на початок графіка струму
    2. Обери тип поведінки який очікуєш  
    3. Дізнайся чи правильно ти зрозумів фізику процесу!
    """)
    
    # Ініціалізація стану гри
    if 'game_data' not in st.session_state:
        st.session_state.game_data = None
    if 'user_choice' not in st.session_state:
        st.session_state.user_choice = None
    if 'show_solution' not in st.session_state:
        st.session_state.show_solution = False
    if 'user_drawing' not in st.session_state:
        st.session_state.user_drawing = None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Графік для передбачення")
        
        # Параметри задачі
        game_mode = st.selectbox("Рівень складності:", [
            "Простий - явний надпровідник чи метал",
            "Середній - складне поле", 
            "Складний - випадкові параметри"
        ], key="game_mode")
        
        if st.button("🎲 Нова задача", key="new_problem", use_container_width=True):
            st.session_state.game_data = generate_game_problem(game_mode)
            st.session_state.user_choice = None
            st.session_state.show_solution = False
            st.session_state.user_drawing = None
            st.rerun()
        
        # Відображення задачі
        if st.session_state.game_data:
            data = st.session_state.game_data
            
            # Підказка
            with st.expander("💡 Підказка"):
                st.write(data["hint"])
                st.write(f"Температура: {data['params']['T']:.1f}K")
                st.write(f"Тип поля: {data['params']['field']}")
            
            # Графік з відомою частиною
            fig = go.Figure()
            
            # Відома частина
            fig.add_trace(go.Scatter(
                x=data["t_known"], y=data["j_known"],
                mode='lines',
                name='Відома частина',
                line=dict(color='blue', width=4)
            ))
            
            # Передбачення користувача
            if st.session_state.user_drawing is not None:
                user_t, user_j = st.session_state.user_drawing
                fig.add_trace(go.Scatter(
                    x=user_t, y=user_j,
                    mode='lines',
                    name='Твоє передбачення',
                    line=dict(color='orange', width=4, dash='dash')
                ))
            
            # Розв'язок
            if st.session_state.show_solution:
                fig.add_trace(go.Scatter(
                    x=data["t_full"], y=data["j_full"],
                    mode='lines',
                    name='Правильна відповідь',
                    line=dict(color='green', width=4, dash='dot')
                ))
            
            fig.update_layout(
                title="Намалюй продовження графіка 📈",
                xaxis_title="Час (с)",
                yaxis_title="Густина струму (А/м²)",
                height=400,
                showlegend=True
            )
            fig.update_yaxes(tickformat=".2e")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Інтерфейс вибору
            st.subheader("✏️ Обери тип поведінки")
            
            drawing_type = st.radio("Як буде розвиватися графік?", [
                "Нескінченне зростання (надпровідник)",
                "Насичення (метал)", 
                "Коливання",
                "Інше"
            ], key="draw_type")
            
            if st.button("🎯 Перевірити відповідь", use_container_width=True):
                st.session_state.user_choice = drawing_type
                st.session_state.show_solution = True
                
                # Генерація передбачення
                t_pred = np.linspace(2.5, 5, 50)
                if drawing_type == "Нескінченне зростання (надпровідник)":
                    j_pred = data["j_known"][-1] + np.linspace(0, abs(data["j_known"][-1]) * 3, 50)
                elif drawing_type == "Насичення (метал)":
                    j_pred = np.full(50, data["j_known"][-1] * 0.9)
                elif drawing_type == "Коливання":
                    j_pred = data["j_known"][-1] + np.sin(np.linspace(0, 4*np.pi, 50)) * abs(data["j_known"][-1]) * 0.5
                else:
                    j_pred = data["j_known"][-1] + np.random.normal(0, abs(data["j_known"][-1]) * 0.3, 50)
                
                st.session_state.user_drawing = (t_pred, j_pred)
                st.rerun()
            
            # Оцінка результату
            if st.session_state.show_solution and st.session_state.user_choice:
                user_choice = st.session_state.user_choice
                real_type = "Надпровідник" if data["material_type"] == "super" else "Метал"
                
                # Проста оцінка
                if ("надпровідник" in user_choice.lower() and real_type == "Надпровідник") or \
                   ("метал" in user_choice.lower() and real_type == "Метал"):
                    accuracy = random.randint(85, 98)
                    st.success("🎉 Відмінно! Ти правильно зрозумів фізику!")
                else:
                    accuracy = random.randint(40, 65)
                    st.error("❌ Спробуй ще! Зверни увагу на температуру.")
                
                st.subheader("📊 Результат")
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("Точність", f"{accuracy}%")
                with col_res2:
                    st.metric("Правильна відповідь", real_type)
    
    with col2:
        st.subheader("🎓 Навчання")
        
        st.markdown("""
        ### 📖 Підказки:
        
        **Надпровідник (T < 9.2K):**
        - Нескінченне зростання струму
        - Немає насичення
        - Для синусоїдального поля - чисті коливання
        
        **Метал (T ≥ 9.2K):**
        - Насичення струму  
        - Стаціонарне значення
        - Для синусоїдального поля - затухаючі коливання
        """)
        
        st.info("""
        ### 💡 Поради:
        - Звертай увагу на температуру
        - Аналізуй нахил графіка
        - Пам'ятай про T_c = 9.2K
        """)

# =============================================================================
# ОСНОВНА СТОРІНКА (ПОВНІСТЮ ЗБЕРЕЖЕНА)
# =============================================================================

def main_page():
    st.title("🔬 Моделювання динаміки струму в ніобії")
    
    # Ініціалізація збережених графіків
    if 'saved_plots' not in st.session_state:
        st.session_state.saved_plots = []
    
    with st.sidebar:
        st.header("⚙️ Параметри моделювання")
        
        comparison_mode = st.radio(
            "Режим:",
            ["Один стан", "Порівняння", "Збережені графіки"]
        )
        
        st.subheader("Загальні параметри")
        field_type = st.selectbox("Тип поля:", ["Статичне", "Лінійне", "Синусоїдальне"])
        E0 = st.slider("Напруженість E₀ (В/м)", 0.1, 100.0, 1.0, 0.1)
        j0 = st.slider("Початковий струм j₀ (А/м²)", 0.0, 100.0, 0.0, 0.1)
        t_max = st.slider("Час моделювання (с)", 0.1, 20.0, 5.0, 0.1)
        
        a = st.slider("Швидкість росту a", 0.1, 10.0, 1.0, 0.1) if field_type == "Лінійне" else 1.0
        omega = st.slider("Частота ω (рад/с)", 0.1, 50.0, 5.0, 0.1) if field_type == "Синусоїдальне" else 1.0
        
        st.subheader("Параметри станів")
        if comparison_mode == "Порівняння":
            T_common = st.slider("Температура (K)", 0.1, 18.4, 4.2, 0.1)
            current_temp = T_common
            current_state = determine_state(T_common)
            st.info(f"🔍 Автоматичне визначення: {current_state}")
            
        elif comparison_mode == "Один стан":
            T_input = st.slider("Температура (K)", 0.1, 18.4, 4.2, 0.1)
            current_temp = T_input
            auto_state = determine_state(T_input)
            st.info(f"🔍 Автоматичне визначення: {auto_state}")
            metal_model = st.radio("Модель для металу:", 
                ["Модель Друде (з перехідним процесом)", "Закон Ома (стаціонарний)"])
        else:
            T_multi = st.slider("Температура (K)", 0.1, 18.4, 4.2, 0.1)
            current_temp = T_multi
            metal_model = "Модель Друде (з перехідним процесом)"  # Значение по умолчанию

        # Кнопка збереження поточного графіка
        if comparison_mode in ["Один стан", "Порівняння"]:
            if st.button("💾 Зберегти поточний графік", use_container_width=True):
                plot_data = {
                    't': np.linspace(0, t_max, 1000),
                    'field_type': field_type, 'E0': E0, 'j0': j0, 'a': a, 'omega': omega,
                    'temperature': current_temp, 'mode': comparison_mode,
                    'timestamp': pd.Timestamp.now()
                }
                
                if comparison_mode == "Один стан":
                    auto_state = determine_state(current_temp)
                    if auto_state == "Надпровідник":
                        plot_data['j_data'] = calculate_superconducting_current(plot_data['t'], field_type, E0, a, omega, j0, current_temp)
                        plot_data['state'] = 'Надпровідник'
                        plot_data['model'] = 'Лондони'
                    else:
                        calc_func = calculate_normal_current_drude if metal_model == "Модель Друде (з перехідним процесом)" else calculate_normal_current_ohm
                        plot_data['j_data'] = calc_func(plot_data['t'], field_type, current_temp, E0, a, omega, j0)
                        plot_data['state'] = 'Звичайний метал'
                        plot_data['model'] = metal_model
                
                elif comparison_mode == "Порівняння":
                    plot_data['j_super'] = calculate_superconducting_current(plot_data['t'], field_type, E0, a, omega, j0, T_common)
                    plot_data['j_normal'] = calculate_normal_current_drude(plot_data['t'], field_type, T_common, E0, a, omega, j0)
                    plot_data['state'] = 'Порівняння'
                    plot_data['model'] = 'Друде'
                
                st.session_state.saved_plots.append(plot_data)
                st.success(f"Графік збережено! Всього збережено: {len(st.session_state.saved_plots)}")

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
                    if plot_data['state'] == 'Надпровідник':
                        fig_saved.add_trace(go.Scatter(
                            x=plot_data['t'], y=plot_data['j_data'], 
                            name=f"Надпровідник {i+1} (T={plot_data['temperature']}K)",
                            line=dict(width=2), opacity=0.7
                        ))
                    elif plot_data['state'] == 'Звичайний метал':
                        fig_saved.add_trace(go.Scatter(
                            x=plot_data['t'], y=plot_data['j_data'],
                            name=f"Метал {i+1} (T={plot_data['temperature']}K, {plot_data['model']})",
                            line=dict(width=2), opacity=0.7
                        ))
                    elif plot_data['state'] == 'Порівняння':
                        fig_saved.add_trace(go.Scatter(
                            x=plot_data['t'], y=plot_data['j_super'], 
                            name=f"Надпровідник {i+1}", line=dict(width=2), opacity=0.7
                        ))
                        fig_saved.add_trace(go.Scatter(
                            x=plot_data['t'], y=plot_data['j_normal'], 
                            name=f"Метал {i+1}", line=dict(width=2), opacity=0.7
                        ))
                
                fig_saved.update_layout(
                    title="Усі збережені графіки",
                    xaxis_title="Час (с)",
                    yaxis_title="Густина струму (А/м²)",
                    height=600,
                    showlegend=True
                )
                fig_saved.update_yaxes(tickformat=".2e")
                st.plotly_chart(fig_saved, use_container_width=True)
        
        else:
            st.header("📈 Графіки струму")
            
            t = np.linspace(0, t_max, 1000)
            fig = go.Figure()
            physical_analyses = []
            math_analyses = []
            
            if comparison_mode == "Порівняння":
                j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0, T_common)
                j_normal = calculate_normal_current_drude(t, field_type, T_common, E0, a, omega, j0)
                
                fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник', line=dict(color='red', width=3)))
                fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний метал (Друде)', line=dict(color='blue', width=3)))
                
                physical_analyses = [
                    analyze_physical_characteristics(t, j_super, "Надпровідник", field_type, T_common, omega),
                    analyze_physical_characteristics(t, j_normal, "Звичайний метал", field_type, T_common, omega)
                ]
                math_analyses = [
                    analyze_mathematical_characteristics(t, j_super, "Надпровідник", field_type, omega),
                    analyze_mathematical_characteristics(t, j_normal, "Звичайний метал", field_type, omega)
                ]
                
            elif comparison_mode == "Один стан":
                auto_state = determine_state(current_temp)
                if auto_state == "Надпровідник":
                    j_data = calculate_superconducting_current(t, field_type, E0, a, omega, j0, current_temp)
                    fig.add_trace(go.Scatter(x=t, y=j_data, name='Надпровідник', line=dict(color='red', width=3)))
                    physical_analyses = [analyze_physical_characteristics(t, j_data, "Надпровідник", field_type, current_temp, omega)]
                    math_analyses = [analyze_mathematical_characteristics(t, j_data, "Надпровідник", field_type, omega)]
                else:
                    calc_func = calculate_normal_current_drude if metal_model == "Модель Друде (з перехідним процесом)" else calculate_normal_current_ohm
                    j_data = calc_func(t, field_type, current_temp, E0, a, omega, j0)
                    model_name = "Звичайний метал (Друде)" if metal_model == "Модель Друде (з перехідним процесом)" else "Звичайний метал (Ом)"
                    
                    fig.add_trace(go.Scatter(x=t, y=j_data, name=model_name, line=dict(color='blue', width=3)))
                    physical_analyses = [analyze_physical_characteristics(t, j_data, "Звичайний метал", field_type, current_temp, omega)]
                    math_analyses = [analyze_mathematical_characteristics(t, j_data, "Звичайний метал", field_type, omega)]
            
            fig.update_layout(
                title="Динаміка густини струму в ніобії",
                xaxis_title="Час (с)",
                yaxis_title="Густина струму (А/м²)",
                height=500
            )
            fig.update_yaxes(tickformat=".2e")
            st.plotly_chart(fig, use_container_width=True)
            
            if physical_analyses:
                st.header("📊 Фізичний аналіз")
                st.dataframe(pd.DataFrame(physical_analyses), use_container_width=True, height=200)
                
                st.header("🧮 Математичний аналіз")
                if len(math_analyses) == 2:
                    col_math1, col_math2 = st.columns(2)
                    with col_math1:
                        st.write("**Надпровідник:**")
                        st.dataframe(pd.DataFrame([math_analyses[0]]).T.rename(columns={0: 'Значення'}), use_container_width=True, height=300)
                    with col_math2:
                        st.write("**Звичайний метал:**")
                        st.dataframe(pd.DataFrame([math_analyses[1]]).T.rename(columns={0: 'Значення'}), use_container_width=True, height=300)
                else:
                    st.dataframe(pd.DataFrame([math_analyses[0]]).T.rename(columns={0: 'Значення'}), use_container_width=True, height=300)

    with col2:
        st.header("📋 Інформація")
        
        st.subheader("Параметри розрахунку")
        st.write(f"**Тип поля:** {field_type}")
        st.write(f"**E₀ =** {E0} В/м")
        st.write(f"**j₀ =** {j0} А/м²")
        st.write(f"**Температура:** {current_temp} K")
        
        current_state = determine_state(current_temp)
        if current_state == "Надпровідник":
            st.success("✅ Надпровідний стан (T < T_c)")
        else:
            st.warning("⚠️ Звичайний метал (T ≥ T_c)")
        
        st.write(f"**Критична температура T_c:** {Tc} K")

        with st.expander("Фізичні константи ніобію"):
            st.write(f"**e =** {e:.3e} Кл")
            st.write(f"**m =** {m:.3e} кг")
            st.write(f"**n₀ =** {n0:.2e} м⁻³")
            st.write(f"**τ_imp =** {tau_imp:.2e} с")
            st.write(f"**T_c =** {Tc} K")

        st.header("📄 Експорт результатів")
        if st.button("📥 Згенерувати звіт", use_container_width=True):
            input_data = {'field_type': field_type, 'E0': E0, 'j0': j0, 't_max': t_max, 'T_common': current_temp}
            
            t = np.linspace(0, t_max, 1000)
            physical_analyses_for_report = []
            math_analyses_for_report = []
            
            if comparison_mode == "Порівняння":
                j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0, T_common)
                j_normal = calculate_normal_current_drude(t, field_type, T_common, E0, a, omega, j0)
                physical_analyses_for_report = [
                    analyze_physical_characteristics(t, j_super, "Надпровідник", field_type, T_common, omega),
                    analyze_physical_characteristics(t, j_normal, "Звичайний метал", field_type, T_common, omega)
                ]
                math_analyses_for_report = [
                    analyze_mathematical_characteristics(t, j_super, "Надпровідник", field_type, omega),
                    analyze_mathematical_characteristics(t, j_normal, "Звичайний метал", field_type, omega)
                ]
            elif comparison_mode == "Один стан":
                auto_state = determine_state(current_temp)
                if auto_state == "Надпровідник":
                    j_data = calculate_superconducting_current(t, field_type, E0, a, omega, j0, current_temp)
                    physical_analyses_for_report = [analyze_physical_characteristics(t, j_data, "Надпровідник", field_type, current_temp, omega)]
                    math_analyses_for_report = [analyze_mathematical_characteristics(t, j_data, "Надпровідник", field_type, omega)]
                else:
                    calc_func = calculate_normal_current_drude if metal_model == "Модель Друде (з перехідним процесом)" else calculate_normal_current_ohm
                    j_data = calc_func(t, field_type, current_temp, E0, a, omega, j0)
                    physical_analyses_for_report = [analyze_physical_characteristics(t, j_data, "Звичайний метал", field_type, current_temp, omega)]
                    math_analyses_for_report = [analyze_mathematical_characteristics(t, j_data, "Звичайний метал", field_type, omega)]
            
            pdf_buffer = create_pdf_report(input_data, physical_analyses_for_report, math_analyses_for_report, st.session_state.saved_plots)
            st.download_button(
                label="⬇️ Завантажити PDF звіт",
                data=pdf_buffer,
                file_name="звіт_моделювання.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    # Інформаційні розділи
    with st.expander("ℹ️ Інструкція користування"):
        st.markdown("""
        **Як користуватися програмою:**
        
        1. **Обрати режим роботи:**
           - *Один стан* - перегляд одного стану матеріалу
           - *Порівняння* - одночасне порівняння надпровідного та звичайного станів
           - *Збережені графіки* - перегляд усіх збережених результатів
        
        2. **Встановити параметри моделювання** в боковій панелі
        3. **Натиснути \"💾 Зберегти поточний графік\"** для збереження результату
        4. **Згенерувати PDF звіт** для експорту всіх даних
        
        """)

    with st.expander("🔬 Фізичні принципи"):
        st.markdown("""
        **Теоретичні основи моделювання:**
        
        **Надпровідний стан (T < Tₐ):**
        - Рівняння Лондонів: струм росте необмеженно лінійно/квадратично через відсутність опору
        - Критична температура для ніобію: **Tₐ = 9.2 K**
        
        **Звичайний метал (T ≥ Tₐ):**
        - Модель Друде: експоненційне насичення струму через опір
        - Закон Ома: стаціонарна поведінка струму
        - Час релаксації залежить від температури
        
        **Типи електричних полів:**
        - *Статичне* - постійне поле
        - *Лінійне* - поле що лінійно зростає з часом  
        - *Синусоїдальне* - змінне гармонійне поле
        """)

    with st.expander("📊 Про аналіз даних"):
        st.markdown("""
        **Фізичний аналіз:**
        - j(0), j_max - початкове та максимальне значення струму
        - Швидкість зміни - похідна струму за часом
        - Поведінка - фізична інтерпретація динаміки
        
        **Математичний аналіз:**
        - f'(max), f'(min) - екстремуми похідної
        - f'(сер) - середня швидкість зміни
        - Тип функції - характеристика математичної залежності
        
        **Параметри ніобію:**
        - Критична температура: **9.2 K**
        - Діапазон моделювання: **0.1 - 18.4 K**
        - Типовий надпровідник для досліджень
        - Широко використовується у надпровідних магнітах
        """)

# =============================================================================
# ОСНОВНА ФУНКЦІЯ
# =============================================================================

def main():
    st.set_page_config(page_title="Моделювання струму", layout="wide")
    
    # Навігація в сайдбарі
    with st.sidebar:
        st.title("Навігація")
        page = st.radio("Оберіть сторінку:", [
            "🧪 Основна сторінка",
            "🎬 Анімації та демонстрації",
            "🏎️ Електронні Гонки", 
            "🔮 Передбач майбутнє"
        ])
    
    # Вибір сторінки
    if page == "🧪 Основна сторінка":
        main_page()
    elif page == "🎬 Анімації та демонстрації":
        animations_page()
    elif page == "🏎️ Електронні Гонки":
        racing_page()
    elif page == "🔮 Передбач майбутнє":
        prediction_game_page()

if __name__ == "__main__":
    main()
