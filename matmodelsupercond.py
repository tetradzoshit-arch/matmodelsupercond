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
e = 1.6e-19  # Кл
m = 9.1e-31  # кг
Tc = 9.2  # К
n0 = 1.0e29  # м⁻³
tau_imp = 5.0e-14  # с
A_ph = 3.0e8 

def determine_state(T):
    return "Надпровідник" if T < Tc else "Звичайний стан"

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
        from reportlab.lib import colors
        import io
        
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=landscape(A4))
        width, height = landscape(A4)
        
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
            font_name = 'DejaVuSans'
        except:
            font_name = 'Helvetica'
        
        # Перевірка чи є дані для аналізу
        if not physical_analyses and saved_plots:
            physical_analyses = []
            math_analyses = []
            
            for plot_data in saved_plots:
                if 'j_data' in plot_data:
                    t_temp = plot_data['t']
                    j_temp = plot_data['j_data']
                    state = plot_data.get('state', 'Невідомий стан')
                    field_type = plot_data.get('field_type', 'Статичне')
                    temp = plot_data.get('temperature', 4.2)
                    omega = plot_data.get('omega', 1.0)
                    
                    physical_analyses.append(
                        analyze_physical_characteristics(t_temp, j_temp, state, field_type, temp, omega)
                    )
                    math_analyses.append(
                        analyze_mathematical_characteristics(t_temp, j_temp, state, field_type, omega)
                    )
                elif 'j_super' in plot_data and 'j_normal' in plot_data:
                    t_temp = plot_data['t']
                    temp = plot_data.get('temperature', 4.2)
                    field_type = plot_data.get('field_type', 'Статичне')
                    omega = plot_data.get('omega', 1.0)
                    
                    physical_analyses.append(
                        analyze_physical_characteristics(t_temp, plot_data['j_super'], "Надпровідник", field_type, temp, omega)
                    )
                    physical_analyses.append(
                        analyze_physical_characteristics(t_temp, plot_data['j_normal'], "Звичайний стан", field_type, temp, omega)
                    )
                    math_analyses.append(
                        analyze_mathematical_characteristics(t_temp, plot_data['j_super'], "Надпровідник", field_type, omega)
                    )
                    math_analyses.append(
                        analyze_mathematical_characteristics(t_temp, plot_data['j_normal'], "Звичайний стан", field_type, omega)
                    )
        
        # Сторінка 1: Загальна інформація
        pdf.setFont(font_name, 18)
        pdf.drawString(100, height - 80, "ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ В НІОБІЇ")
        
        pdf.setFont(font_name, 12)
        y_position = height - 120
        
        # Загальні параметри моделювання
        pdf.drawString(100, y_position, "Загальні параметри моделювання:")
        y_position -= 25
        pdf.drawString(120, y_position, f"- Критична температура T_c: {Tc} K")
        y_position -= 20
        pdf.drawString(120, y_position, f"- Густина електронів n₀: {n0:.1e} м⁻³")
        y_position -= 20
        pdf.drawString(120, y_position, f"- Час релаксації τ: {tau_imp:.1e} с")
        y_position -= 30

        # Параметри всіх збережених графіків
        if saved_plots:
            pdf.drawString(100, y_position, "Параметри збережених графіків:")
            y_position -= 25
            
            for i, plot_data in enumerate(saved_plots):
                if y_position < 150:
                    pdf.showPage()
                    pdf.setFont(font_name, 12)
                    y_position = height - 80
                
                pdf.setFont(font_name, 11)
                pdf.drawString(120, y_position, f"Графік {i+1}: {plot_data.get('state', 'Невідомий')}")
                y_position -= 18
                pdf.drawString(140, y_position, f"Температура: {plot_data.get('temperature', 'N/A')} K")
                y_position -= 16
                pdf.drawString(140, y_position, f"Тип поля: {plot_data.get('field_type', 'N/A')}")
                y_position -= 16
                pdf.drawString(140, y_position, f"E₀: {plot_data.get('E0', 'N/A')} В/м")
                y_position -= 16
                pdf.drawString(140, y_position, f"j₀: {plot_data.get('j0', 'N/A')} А/м²")
                y_position -= 16
                if plot_data.get('a', 1.0) != 1.0:
                    pdf.drawString(140, y_position, f"a: {plot_data.get('a', 'N/A')}")
                    y_position -= 16
                if plot_data.get('omega', 1.0) != 1.0:
                    pdf.drawString(140, y_position, f"ω: {plot_data.get('omega', 'N/A')} рад/с")
                    y_position -= 16
                y_position -= 10

        y_position -= 20

        # Фізичний аналіз з кольоровими таблицями
        if physical_analyses:
            if y_position < 200:
                pdf.showPage()
                pdf.setFont(font_name, 14)
                y_position = height - 80
            
            pdf.setFont(font_name, 16)
            pdf.drawString(100, y_position, "ФІЗИЧНИЙ АНАЛІЗ")
            y_position -= 35
            
            # Заголовок таблиці
            col_widths = [130, 90, 120, 120, 180]
            col_positions = [80, 210, 300, 420, 540]
            row_height = 30
            
            # Кольорові заголовки
            pdf.setFillColor(colors.lightblue)
            pdf.rect(80, y_position - row_height, sum(col_widths), row_height, fill=1, stroke=0)
            pdf.setFillColor(colors.black)
            
            headers = ["Стан", "Температура", "j(0)", "j_max", "Поведінка"]
            pdf.setFont(font_name, 12)
            for i, header in enumerate(headers):
                pdf.drawString(col_positions[i] + 8, y_position - 18, header)
            
            y_position -= row_height + 8
            
            # Дані з кольоровим фоном
            pdf.setFont(font_name, 10)
            for i, analysis in enumerate(physical_analyses):
                if y_position < 120:
                    pdf.showPage()
                    pdf.setFont(font_name, 14)
                    y_position = height - 80
                    # Повторюємо заголовки на новій сторінці
                    pdf.setFillColor(colors.lightblue)
                    pdf.rect(80, y_position - row_height, sum(col_widths), row_height, fill=1, stroke=0)
                    pdf.setFillColor(colors.black)
                    pdf.setFont(font_name, 12)
                    for j, header in enumerate(headers):
                        pdf.drawString(col_positions[j] + 8, y_position - 18, header)
                    y_position -= row_height + 8
                    pdf.setFont(font_name, 10)
                
                # Кольоровий фон для рядків
                if i % 2 == 0:
                    pdf.setFillColor(colors.lightgrey)
                else:
                    pdf.setFillColor(colors.whitesmoke)
                
                pdf.rect(80, y_position - row_height, sum(col_widths), row_height, fill=1, stroke=0)
                pdf.setFillColor(colors.black)
                
                cells = [
                    analysis.get('Стан', '')[:18],
                    analysis.get('Температура', '')[:12],
                    analysis.get('j(0)', '')[:15],
                    analysis.get('j_max', '')[:15],
                    analysis.get('Поведінка', '')[:25]
                ]
                
                for j, cell in enumerate(cells):
                    pdf.drawString(col_positions[j] + 8, y_position - 18, str(cell))
                
                y_position -= row_height + 8
            
            y_position -= 25

        # Математичний аналіз з кольоровими таблицями
        if math_analyses:
            if y_position < 200:
                pdf.showPage()
                pdf.setFont(font_name, 14)
                y_position = height - 80
            
            pdf.setFont(font_name, 16)
            pdf.drawString(100, y_position, "МАТЕМАТИЧНИЙ АНАЛІЗ")
            y_position -= 35
            
            # Заголовок таблиці - компактна версія
            col_widths = [120, 120, 100, 100, 100, 100, 100]
            col_positions = [50, 170, 290, 390, 490, 590, 690]
            row_height = 30
            
            # Кольорові заголовки
            pdf.setFillColor(colors.lightgreen)
            pdf.rect(50, y_position - row_height, sum(col_widths), row_height, fill=1, stroke=0)
            pdf.setFillColor(colors.black)
            
            headers = ["Функція", "Тип функції", "f(0)", "max f(t)", "f'(max)", "f'(min)", "f'(сер)"]
            pdf.setFont(font_name, 10)
            for i, header in enumerate(headers):
                pdf.drawString(col_positions[i] + 5, y_position - 18, header)
            
            y_position -= row_height + 8
            
            # Дані з кольоровим фоном
            pdf.setFont(font_name, 9)
            for i, analysis in enumerate(math_analyses):
                if y_position < 120:
                    pdf.showPage()
                    pdf.setFont(font_name, 14)
                    y_position = height - 80
                    # Повторюємо заголовки
                    pdf.setFillColor(colors.lightgreen)
                    pdf.rect(50, y_position - row_height, sum(col_widths), row_height, fill=1, stroke=0)
                    pdf.setFillColor(colors.black)
                    pdf.setFont(font_name, 10)
                    for j, header in enumerate(headers):
                        pdf.drawString(col_positions[j] + 5, y_position - 18, header)
                    y_position -= row_height + 8
                    pdf.setFont(font_name, 9)
                
                # Кольоровий фон для рядків
                if i % 2 == 0:
                    pdf.setFillColor(colors.lightgrey)
                else:
                    pdf.setFillColor(colors.whitesmoke)
                
                pdf.rect(50, y_position - row_height, sum(col_widths), row_height, fill=1, stroke=0)
                pdf.setFillColor(colors.black)
                
                # Отримуємо всі значення
                func_type = analysis.get('Тип функції', analysis.get('Тип функції', 'N/A'))
                f_0 = analysis.get('f(0)', 'N/A')
                f_max = analysis.get('max f(t)', 'N/A')
                f_prime_max = analysis.get("f'(max)", analysis.get('Макс. швидкість', 'N/A'))
                f_prime_min = analysis.get("f'(min)", 'N/A')
                f_prime_avg = analysis.get("f'(середнє)", analysis.get("f'(сер)", 'N/A'))
                
                cells = [
                    analysis.get('Функція', 'N/A')[:15],
                    func_type[:18],
                    f_0[:12],
                    f_max[:12],
                    str(f_prime_max)[:12],
                    str(f_prime_min)[:12],
                    str(f_prime_avg)[:12]
                ]
                
                for j, cell in enumerate(cells):
                    pdf.drawString(col_positions[j] + 5, y_position - 18, str(cell))
                
                y_position -= row_height + 8
        
        # Висновки
        if y_position < 150:
            pdf.showPage()
            y_position = height - 80
        
        pdf.setFont(font_name, 16)
        pdf.drawString(100, y_position, "ВИСНОВКИ")
        y_position -= 30
        
        pdf.setFont(font_name, 12)
        conclusions = [
            "• Надпровідник демонструє необмежене зростання струму через відсутність опору",
            "• Звичайний стан має властивості насичення через наявність опору",
            "• Аналіз похідних показує швидкість змін струму в часі",
            f"• Усього проаналізовано графіків: {len(saved_plots) if saved_plots else len(physical_analyses)}"
             "",
            "🔬 Примітка: Модель нехтує квантовими та магнітними ефектами",,
            "для спрощення математичного аналізу та зосередження на фундаменті поведінки струму."
        ]
        
        for conclusion in conclusions:
            if y_position < 50:
                pdf.showPage()
                y_position = height - 80
                pdf.setFont(font_name, 12)
            
            pdf.drawString(120, y_position, conclusion)
            y_position -= 20
        
        pdf.save()
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Помилка при створенні PDF: {e}")
        import traceback
        st.error(traceback.format_exc())
        
        buffer = io.BytesIO()
        report_text = "ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ В НІОБІЇ\n\n"
        report_text += "Загальні параметри:\n"
        report_text += f"T_c = {Tc} K, n₀ = {n0:.1e} м⁻³, τ = {tau_imp:.1e} с\n\n"
        report_text += "Збережені графіки:\n"
        for i, plot in enumerate(saved_plots):
            report_text += f"Графік {i+1}: {plot.get('state', 'N/A')}, T={plot.get('temperature', 'N/A')}K\n"
        buffer.write(report_text.encode('utf-8'))
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Помилка при створенні PDF: {e}")
        import traceback
        st.error(traceback.format_exc())
        
        buffer = io.BytesIO()
        report_text = "ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ В НІОБІЇ\n\n"
        report_text += "Загальні параметри:\n"
        report_text += f"T_c = {Tc} K, n₀ = {n0:.1e} м⁻³, τ = {tau_imp:.1e} с\n\n"
        report_text += "Збережені графіки:\n"
        for i, plot in enumerate(saved_plots):
            report_text += f"Графік {i+1}: {plot.get('state', 'N/A')}, T={plot.get('temperature', 'N/A')}K\n"
        buffer.write(report_text.encode('utf-8'))
        buffer.seek(0)
        return buffer
# =============================================================================
# СТОРІНКА АНІМАЦІЙ
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
                
                state = "Надпровідник" if temp < Tc else "Звичайний стан"
                status_text.text(f"Температура: {temp:.1f} K | Стан: {state}")
                
                t_anim = np.linspace(0, anim_t_max, 150)
                j_super = calculate_superconducting_current(t_anim, anim_field_type, anim_E0, anim_a, anim_omega, anim_j0, temp)
                j_normal = calculate_normal_current_drude(t_anim, anim_field_type, temp, anim_E0, anim_a, anim_omega, anim_j0)
                
                fig_anim = go.Figure()
                fig_anim.add_trace(go.Scatter(x=t_anim, y=j_super, name='Надпровідник', 
                                            line=dict(color='red', width=3)))
                fig_anim.add_trace(go.Scatter(x=t_anim, y=j_normal, name='Звичайний стан', 
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
                
                state = "Надпровідник" if T_trans < Tc else "Звичайний стан"
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
        
        # Анімація 3: Автоматичне визначення стану за температурою
        st.subheader("🎯 Анімація одного стану")
        st.write("Динаміка струму для автоматично визначеного стану за температурою")
        
        anim_temp = st.slider("Температура для аналізу", 1.0, 18.0, 4.2, 0.1, key="anim_temp")
        
        if st.button("▶️ Запустити анімацію стану", key="state_anim", use_container_width=True):
            plot_placeholder3 = st.empty()
            progress_bar3 = st.progress(0)
            
            # Визначаємо стан за температурою
            state = "Надпровідник" if anim_temp < Tc else "Звичайний стан"
            st.info(f"🔍 Автоматично визначено: {state} (T = {anim_temp}K)")
            
            t_state = np.linspace(0, anim_t_max, 200)
            
            if state == "Надпровідник":
                j_data = calculate_superconducting_current(t_state, anim_field_type, anim_E0, anim_a, anim_omega, anim_j0, anim_temp)
                color = 'red'
                behavior = "Лінійне/квадратичне зростання без опору"
            else:
                j_data = calculate_normal_current_drude(t_state, anim_field_type, anim_temp, anim_E0, anim_a, anim_omega, anim_j0)
                color = 'blue'
                behavior = "Експоненційне насичення через опір"
            
            # Створюємо графік
            fig_state = go.Figure()
            fig_state.add_trace(go.Scatter(x=t_state, y=j_data, 
                                         name=f'{state} (T={anim_temp}K)',
                                         line=dict(color=color, width=4)))
            
            fig_state.update_layout(
                title=f"Динаміка струму: {state}",
                xaxis_title="Час (с)",
                yaxis_title="Густина струму (А/м²)",
                height=500,
                annotations=[
                    dict(
                        x=0.02, y=0.98, xref='paper', yref='paper',
                        text=f"Поведінка: {behavior}",
                        showarrow=False,
                        bgcolor='white',
                        bordercolor='black',
                        borderwidth=1
                    )
                ]
            )
            fig_state.update_yaxes(tickformat=".2e")
            
            plot_placeholder3.plotly_chart(fig_state, use_container_width=True)
            
            # Додаємо аналіз
            analysis = analyze_physical_characteristics(t_state, j_data, state, anim_field_type, anim_temp, anim_omega)
            
            st.subheader("📊 Фізичний аналіз")
            col_anal1, col_anal2 = st.columns(2)
            with col_anal1:
                st.metric("Початковий струм", analysis['j(0)'])
                st.metric("Максимальний струм", analysis['j_max'])
                st.metric("Температура", f"{anim_temp} K")
            with col_anal2:
                st.metric("Стан", state)
                st.metric("Макс. швидкість зміни", analysis['Макс. швидкість'])
                st.metric("Поведінка", analysis['Поведінка'])
            
            progress_bar3.progress(100)
            st.success("✅ Аналіз завершено!")

# =============================================================================
# СТОРІНКА ГОНОК
# =============================================================================
# =============================================================================
# СТОРІНКА ГОНОК
# =============================================================================

def racing_page():
    st.header("🏎️ Електронні Гонки - Надпровідник vs Звичайний стан")
    
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
            car1_type = "Надпровідник" if car1_temp < Tc else "Звичайний стан"
            st.info(f"**Машинка 1:** {car1_type}")
            
        with col_car2:
            car2_temp = st.slider("Температура машинки 2 (K)", 1.0, 18.0, 12.0, 0.1, key="car2_temp")
            car2_type = "Надпровідник" if car2_temp < Tc else "Звичайний стан"
            st.info(f"**Машинка 2:** {car2_type}")
        
        # Загальні параметри
        race_field = st.selectbox("Тип поля:", ["Статичне", "Лінійне", "Синусоїдальне"], key="race_field")
        race_E0 = st.slider("Потужність поля E₀", 0.1, 5.0, 1.0, 0.1, key="race_E0")
        race_speed = st.slider("Швидкість анімації", 0.5, 3.0, 1.0, 0.1, key="race_speed")
        
        # Примітка про спрощення
        with st.expander("📝 Важлива примітка про фізику"):
            st.markdown("""
            **⚖️ Спрощена модель для наочності:**
            
            У реальності опір у звичайному стані залежить від температури за **правилом Маттісена**:
            ```
            ρ(T) = ρ₀ + ρ_phonon(T)
            ```
            де:
            - `ρ₀` - залишковий опір (незалежний від T)
            - `ρ_phonon(T)` - опір від фононів (зростає з T)
            
            **Чому ми спрощуємо:**
            - 🔬 Надпровідність - **квантовий ефект**, що вимагає мікроскопічного опису
            - 📈 Температурна залежність опору металу - окрема складна тема
            - 🎯 Мета гри - показати **принципову різницю**:  
              **нульовий опір** vs **наявність опору**
            
            **У реальності:**
            - Надпровідник: опір = 0 (до критичного струму)
            - Звичайний стан: опір зростає з температурою
            """)
        
        if st.button("🎮 Старт гонки!", use_container_width=True) and not st.session_state.race_started:
            # Підготовка даних для гонки
            t_race = np.linspace(0, 8, 30)  # Більше часу для плавного прогресу
            
            # Розрахунок прогресу через модель Друде для обох типів
            if car1_type == "Надпровідник":
                # Для надпровідника - лінійний прогрес (без опору)
                j_car1 = race_E0 * t_race
                # Нормалізуємо до повільного старту і прискорення
                progress_car1 = 0.1 * t_race + 0.05 * t_race**2
            else:
                # Для звичайного стану - модель Друде з насиченням
                tau = 2.0  # Час релаксації
                j_max = race_E0 * tau  # Максимальний струм
                j_car1 = j_max * (1 - np.exp(-t_race / tau))
                progress_car1 = j_car1 / j_max * 0.8  # Обмежуємо максимум 80%
            
            if car2_type == "Надпровідник":
                # Для надпровідника - лінійний прогрес (без опору)
                j_car2 = race_E0 * t_race
                progress_car2 = 0.1 * t_race + 0.05 * t_race**2
            else:
                # Для звичайного стану - модель Друде з насиченням
                tau = 2.0  # Час релаксації
                j_max = race_E0 * tau  # Максимальний струм
                j_car2 = j_max * (1 - np.exp(-t_race / tau))
                progress_car2 = j_car2 / j_max * 0.8  # Обмежуємо максимум 80%
            
            # Застосування множників швидкості
            if car1_type == "Надпровідник":
                progress_car1 = progress_car1 * 1.5  # Надпровідник швидший
                speed_multiplier1 = 1.5
            else:
                speed_multiplier1 = 1.0
                
            if car2_type == "Надпровідник":
                progress_car2 = progress_car2 * 1.5  # Надпровідник швидший
                speed_multiplier2 = 1.5
            else:
                speed_multiplier2 = 1.0
            
            # Обмеження прогресу до 100%
            progress_car1 = np.minimum(progress_car1, 1.0)
            progress_car2 = np.minimum(progress_car2, 1.0)
            
            # Збереження даних
            st.session_state.race_data = {
                't_race': t_race,
                'progress_car1': progress_car1,
                'progress_car2': progress_car2,
                'j_car1': j_car1,
                'j_car2': j_car2,
                'car1_type': car1_type,
                'car2_type': car2_type,
                'car1_temp': car1_temp,
                'car2_temp': car2_temp,
                'race_speed': race_speed,
                'speed_multiplier1': speed_multiplier1,
                'speed_multiplier2': speed_multiplier2
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
            
            # Безпечний доступ до множників швидкості
            speed_multiplier1 = data.get('speed_multiplier1', 1.0)
            speed_multiplier2 = data.get('speed_multiplier2', 1.0)
            
            # Показуємо множники швидкості
            if data['car1_type'] == "Надпровідник":
                st.success(f"⚡ Машинка 1: Супер-швидкість (x{speed_multiplier1})")
            else:
                st.warning(f"🚦 Машинка 1: Звичайна швидкість (x{speed_multiplier1})")
                
            if data['car2_type'] == "Надпровідник":
                st.success(f"⚡ Машинка 2: Супер-швидкість (x{speed_multiplier2})")
            else:
                st.warning(f"🚦 Машинка 2: Звичайна швидкість (x{speed_multiplier2})")
        else:
            st.write(f"**🏎️ Машинка 1:** {car1_type} ({car1_temp}K)")
            st.write(f"**🚗 Машинка 2:** {car2_type} ({car2_temp}K)")
        
        st.metric("Критична температура T_c", f"{Tc} K")
        
        # Фізичне пояснення
        st.info("""
        **🎯 Фізична суть гри:**
        - **T < 9.2K**: нульовий опір → необмежене прискорення
        - **T ≥ 9.2K**: є опір → обмежена швидкість
        """)
    
    # Гонкова траса
    if st.session_state.race_started and st.session_state.race_data:
        data = st.session_state.race_data
        frame = st.session_state.race_frame
        
        if frame < len(data['t_race']):
            st.subheader("🏁 ГОНКА ТРИВАЄ!")
            
            progress_car1 = int(data['progress_car1'][frame] * 100)
            progress_car2 = int(data['progress_car2'][frame] * 100)
            
            speed_car1 = abs(data['j_car1'][frame])
            speed_car2 = abs(data['j_car2'][frame])
            
            # Візуалізація гонки
            st.write(f"### 🏎️ Машинка 1 - {data['car1_type']}")
            if data['car1_type'] == "Надпровідник":
                st.success("🛣️ Супер-шосе без опору! ⚡")
            else:
                st.warning("🚦 Міські пробки з опором! 🐌")
            
            st.progress(progress_car1 / 100)
            
            # Траса машинки 1
            track_length = 40
            car1_pos = int(progress_car1 * track_length / 100)
            track1_display = "🏁" + "─" * car1_pos + "🏎️" + "·" * (track_length - car1_pos)
            st.code(track1_display)
            st.write(f"**Швидкість:** {speed_car1:.2e} А/м²")
            st.write(f"**Прогрес:** {progress_car1}%")
            
            st.write("---")
            
            # Машинка 2
            st.write(f"### 🚗 Машинка 2 - {data['car2_type']}")
            if data['car2_type'] == "Надпровідник":
                st.success("🛣️ Супер-шосе без опору! ⚡")
            else:
                st.warning("🚦 Міські пробки з опором! 🐌")
            
            st.progress(progress_car2 / 100)
            
            # Траса машинки 2
            car2_pos = int(progress_car2 * track_length / 100)
            # Додаємо перешкоди тільки для звичайного стану
            obstacles = "🚧" * ((frame // 3) % 2) if data['car2_type'] == "Звичайний стан" else ""
            track2_display = "🏁" + "─" * car2_pos + "🚗" + "·" * (track_length - car2_pos) + " " + obstacles
            st.code(track2_display)
            st.write(f"**Швидкість:** {speed_car2:.2e} А/м²")
            st.write(f"**Прогрес:** {progress_car2}%")
            
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
            
            final_progress1 = int(data['progress_car1'][-1] * 100)
            final_progress2 = int(data['progress_car2'][-1] * 100)
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                if final_progress1 > final_progress2:
                    st.success("🏆 Перемога машинки 1!")
                    winner = "🏎️ Машинка 1"
                elif final_progress2 > final_progress1:
                    st.success("🏆 Перемога машинки 2!")
                    winner = "🚗 Машинка 2"
                else:
                    st.info("🤝 Нічия!")
                    winner = "🤝 Нічия"
                st.metric("Переможець", winner)
            
            with col_res2:
                st.metric("Фінальний прогрес 1", f"{final_progress1}%")
                st.metric("Фінальний прогрес 2", f"{final_progress2}%")
            
            with col_res3:
                st.metric("Макс. швидкість 1", f"{np.max(np.abs(data['j_car1'])):.2e} А/м²")
                st.metric("Макс. швидкість 2", f"{np.max(np.abs(data['j_car2'])):.2e} А/м²")
            
            # Аналіз результатів
            st.subheader("📈 Аналіз гонки")
            if data['car1_type'] == "Надпровідник" and data['car2_type'] == "Звичайний стан":
                st.success("**Фізика підтверджується!** Надпровідник показав кращий результат через відсутність опору!")
            elif data['car1_type'] == "Звичайний стан" and data['car2_type'] == "Надпровідник":
                st.success("**Фізика підтверджується!** Надпровідник обігнав звичайний стан через нульовий опір!")
            else:
                st.info("**Цікавий результат!** Обидві машинки одного типу - порівнюй температури!")
            
            # Пояснення спрощення
            st.info("""
            **🔬 Примітка:** У реальності температурна залежність є в обох станах:

• **Звичайний стан**: опір зростає з T через розсіювання на фононах
• **Надпровідник**: густина надпровідних електронів nₛ(T) зменшується з T

Тут ми спрощуємо модель, виділяючи лише принципову різницю: 
наявність опору чи повна його відсутність.
            """)
            
            if st.button("🔄 Нова гонка", use_container_width=True):
                st.session_state.race_started = False
                st.session_state.race_data = None
                st.rerun()
    
    else:
        # Екран перед стартом
        st.info("""
        ### 🎮 Інструкція до гри:
        
        **🏎️ Надпровідник (T < 9.2K):**
        - ⚡ **ШВИДКІСТЬ x1.5** - без опору!
        - Постійне прискорення
        - Може досягти 100% прогресу
        
        **🚗 Звичайний стан (T ≥ 9.2K):**
        - 🐌 **ЗВИЧАЙНА ШВИДКІСТЬ** - є опір!
        - Швидке насичення (до 80%)
        - Повільний старт через опір
        
        **🎯 Порада:** Встанови температури нижче 9.2K для надпровідників!
        
        **⚡ Фізика в дії:** Надпровідник завжди швидший через нульовий опір!
        """)
        
        # Додаткове пояснення
        with st.expander("🔍 Детальніше про фізичні спрощення"):
            st.markdown("""
            ### Чому ми не враховуємо температурну залежність опору?
            
            **1. Мета гри** - показати принципову різницю:
            - **Надпровідник**: нульовий опір (до критичного струму)
            - **Звичайний стан**: ненульовий опір
            
            **2. Складність реальної фізики**:
            - Опір металу: `ρ(T) = ρ₀ + ρ_phonon(T)`
            - Надпровідність: квантовий ефект з мікроскопічним описом
            - Температурна залежність опору - окрема складна тема
            
            **3. Демонстраційна ціль**:
            - Зрозуміти **фундаментальну різницю** між станами
            - Не заплутатися в деталях температурних залежностей
            
            **У реальному досліді:** опір звичайного стану дійсно зростає з температурою, 
            але це не змінює головного висновку - надпровідник має нульовий опір!
            """)
# =============================================================================
# СТОРІНКА ПЕРЕДБАЧЕНЬ
# =============================================================================
def generate_game_problem(difficulty):
    """Генерація випадкової задачі для гри"""
    problems = {
        "easy": [
            {"field": "Статичне", "T": 4.2, "E0": 1.0, "hint": "Надпровідник при низькій температурі"},
            {"field": "Статичне", "T": 12.0, "E0": 1.0, "hint": "Звичайний стан при високій температурі"}
        ],
        "medium": [
            {"field": "Лінійне", "T": 4.2, "E0": 0.5, "hint": "Надпровідник з лінійним полем"},
            {"field": "Синусоїдальне", "T": 12.0, "E0": 2.0, "hint": "Звичайний стан зі змінним полем"}
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
        material_type = "normal"
    
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
            "Простий - явний надпровідник чи звичайний стан",
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
                "Насичення (звичайний стан)", 
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
                elif drawing_type == "Насичення (звичайний стан)":
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
                real_type = "Надпровідник" if data["material_type"] == "super" else "Звичайний стан"
                
                # Проста оцінка
                if ("надпровідник" in user_choice.lower() and real_type == "Надпровідник") or \
                   ("звичайний" in user_choice.lower() and real_type == "Звичайний стан"):
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
        
        **Звичайний стан (T ≥ 9.2K):**
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
            ["Один стан", "Порівняння", "Збережені графіки"],
            key="comparison_mode_radio"
        )
        
        st.subheader("Загальні параметри")
        field_type = st.selectbox("Тип поля:", ["Статичне", "Лінійне", "Синусоїдальне"], key="field_type_select")
        E0 = st.slider("Напруженість E₀ (В/м)", 0.1, 100.0, 1.0, 0.1, key="E0_slider")
        j0 = st.slider("Початковий струм j₀ (А/м²)", 0.0, 100.0, 0.0, 0.1, key="j0_slider")
        t_max = st.slider("Час моделювання (с)", 0.1, 20.0, 5.0, 0.1, key="t_max_slider")
        
        a = st.slider("Швидкість росту a", 0.1, 10.0, 1.0, 0.1, key="a_slider") if field_type == "Лінійне" else 1.0
        omega = st.slider("Частота ω (рад/с)", 0.1, 50.0, 5.0, 0.1, key="omega_slider") if field_type == "Синусоїдальне" else 1.0
        
        st.subheader("Параметри станів")
        if comparison_mode == "Порівняння":
            T_common = st.slider("Температура (K)", 0.1, 18.4, 4.2, 0.1, key="T_common_slider")
            current_temp = T_common
            current_state = determine_state(T_common)
            st.info(f"🔍 Автоматичне визначення: {current_state}")
            
        elif comparison_mode == "Один стан":
            T_input = st.slider("Температура (K)", 0.1, 18.4, 4.2, 0.1, key="T_input_slider")
            current_temp = T_input
            auto_state = determine_state(T_input)
            st.info(f"🔍 Автоматичне визначення: {auto_state}")
            metal_model = st.radio("Модель для звичайного стану:", 
                ["Модель Друде (з перехідним процесом)", "Закон Ома (стаціонарний)"],
                key="metal_model_radio")
        else:
            T_multi = st.slider("Температура (K)", 0.1, 18.4, 4.2, 0.1, key="T_multi_slider")
            current_temp = T_multi
            metal_model = "Модель Друде (з перехідним процесом)"

        # Кнопка збереження поточного графіка
        if comparison_mode in ["Один стан", "Порівняння"]:
            if st.button("💾 Зберегти поточний графік", use_container_width=True, key="save_plot_btn"):
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
                        plot_data['state'] = 'Звичайний стан'
                        plot_data['model'] = metal_model
                
                elif comparison_mode == "Порівняння":
                    plot_data['j_super'] = calculate_superconducting_current(plot_data['t'], field_type, E0, a, omega, j0, T_common)
                    plot_data['j_normal'] = calculate_normal_current_drude(plot_data['t'], field_type, T_common, E0, a, omega, j0)
                    plot_data['state'] = 'Порівняння'
                    plot_data['model'] = 'Друде'
                
                st.session_state.saved_plots.append(plot_data)
                st.success(f"Графік збережено! Всього збережено: {len(st.session_state.saved_plots)}")

        if st.session_state.saved_plots and st.button("🗑️ Очистити всі збережені графіки", use_container_width=True, key="clear_plots_btn"):
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
                physical_analyses_saved = []
                math_analyses_saved = []
                
                for i, plot_data in enumerate(st.session_state.saved_plots):
                    if plot_data['state'] == 'Надпровідник':
                        fig_saved.add_trace(go.Scatter(
                            x=plot_data['t'], y=plot_data['j_data'], 
                            name=f"Надпровідник {i+1} (T={plot_data['temperature']}K)",
                            line=dict(width=2), opacity=0.7
                        ))
                        # Аналіз для надпровідника
                        physical_analyses_saved.append(
                            analyze_physical_characteristics(
                                plot_data['t'], plot_data['j_data'], 
                                'Надпровідник', 
                                plot_data['field_type'], 
                                plot_data['temperature'],
                                plot_data.get('omega', 1.0)
                            )
                        )
                        math_analyses_saved.append(
                            analyze_mathematical_characteristics(
                                plot_data['t'], plot_data['j_data'],
                                'Надпровідник',
                                plot_data['field_type'],
                                plot_data.get('omega', 1.0)
                            )
                        )
                        
                    elif plot_data['state'] == 'Звичайний стан':
                        fig_saved.add_trace(go.Scatter(
                            x=plot_data['t'], y=plot_data['j_data'],
                            name=f"Звичайний стан {i+1} (T={plot_data['temperature']}K, {plot_data['model']})",
                            line=dict(width=2), opacity=0.7
                        ))
                        # Аналіз для звичайного стану
                        physical_analyses_saved.append(
                            analyze_physical_characteristics(
                                plot_data['t'], plot_data['j_data'],
                                'Звичайний стан',
                                plot_data['field_type'],
                                plot_data['temperature'],
                                plot_data.get('omega', 1.0)
                            )
                        )
                        math_analyses_saved.append(
                            analyze_mathematical_characteristics(
                                plot_data['t'], plot_data['j_data'],
                                'Звичайний стан',
                                plot_data['field_type'],
                                plot_data.get('omega', 1.0)
                            )
                        )
                        
                    elif plot_data['state'] == 'Порівняння':
                        fig_saved.add_trace(go.Scatter(
                            x=plot_data['t'], y=plot_data['j_super'], 
                            name=f"Надпровідник {i+1}", line=dict(width=2), opacity=0.7
                        ))
                        fig_saved.add_trace(go.Scatter(
                            x=plot_data['t'], y=plot_data['j_normal'], 
                            name=f"Звичайний стан {i+1}", line=dict(width=2), opacity=0.7
                        ))
                        # Аналіз для порівняння
                        physical_analyses_saved.append(
                            analyze_physical_characteristics(
                                plot_data['t'], plot_data['j_super'],
                                'Надпровідник',
                                plot_data['field_type'],
                                plot_data['temperature'],
                                plot_data.get('omega', 1.0)
                            )
                        )
                        physical_analyses_saved.append(
                            analyze_physical_characteristics(
                                plot_data['t'], plot_data['j_normal'],
                                'Звичайний стан',
                                plot_data['field_type'],
                                plot_data['temperature'],
                                plot_data.get('omega', 1.0)
                            )
                        )
                        math_analyses_saved.append(
                            analyze_mathematical_characteristics(
                                plot_data['t'], plot_data['j_super'],
                                'Надпровідник',
                                plot_data['field_type'],
                                plot_data.get('omega', 1.0)
                            )
                        )
                        math_analyses_saved.append(
                            analyze_mathematical_characteristics(
                                plot_data['t'], plot_data['j_normal'],
                                'Звичайний стан',
                                plot_data['field_type'],
                                plot_data.get('omega', 1.0)
                            )
                        )
                
                fig_saved.update_layout(
                    title="Усі збережені графіки",
                    xaxis_title="Час (с)",
                    yaxis_title="Густина струму (А/м²)",
                    height=600,
                    showlegend=True
                )
                fig_saved.update_yaxes(tickformat=".2e")
                st.plotly_chart(fig_saved, use_container_width=True, key="saved_plots_chart")
                
                # Додаємо таблиці аналізу
                if physical_analyses_saved:
                    st.header("📊 Фізичний аналіз збережених графіків")
                    st.dataframe(pd.DataFrame(physical_analyses_saved), use_container_width=True, height=300, key="physical_analysis_df")
                    
                if math_analyses_saved:
                    st.header("🧮 Математичний аналіз збережених графіків")
                    st.dataframe(pd.DataFrame(math_analyses_saved), use_container_width=True, height=300, key="math_analysis_df")
        
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
                fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний стан (Друде)', line=dict(color='blue', width=3)))
                
                physical_analyses = [
                    analyze_physical_characteristics(t, j_super, "Надпровідник", field_type, T_common, omega),
                    analyze_physical_characteristics(t, j_normal, "Звичайний стан", field_type, T_common, omega)
                ]
                math_analyses = [
                    analyze_mathematical_characteristics(t, j_super, "Надпровідник", field_type, omega),
                    analyze_mathematical_characteristics(t, j_normal, "Звичайний стан", field_type, omega)
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
                    model_name = "Звичайний стан (Друде)" if metal_model == "Модель Друде (з перехідним процесом)" else "Звичайний стан (Ом)"
                    
                    fig.add_trace(go.Scatter(x=t, y=j_data, name=model_name, line=dict(color='blue', width=3)))
                    physical_analyses = [analyze_physical_characteristics(t, j_data, "Звичайний стан", field_type, current_temp, omega)]
                    math_analyses = [analyze_mathematical_characteristics(t, j_data, "Звичайний стан", field_type, omega)]
            
            fig.update_layout(
                title="Динаміка густини струму в ніобії",
                xaxis_title="Час (с)",
                yaxis_title="Густина струму (А/м²)",
                height=500
            )
            fig.update_yaxes(tickformat=".2e")
            st.plotly_chart(fig, use_container_width=True, key="main_plot_chart")
            
            if physical_analyses:
                st.header("📊 Фізичний аналіз")
                st.dataframe(pd.DataFrame(physical_analyses), use_container_width=True, height=200, key="main_physical_df")
                
                st.header("🧮 Математичний аналіз")
                if len(math_analyses) == 2:
                    col_math1, col_math2 = st.columns(2)
                    with col_math1:
                        st.write("**Надпровідник:**")
                        st.dataframe(pd.DataFrame([math_analyses[0]]).T.rename(columns={0: 'Значення'}), use_container_width=True, height=300, key="math_super_df")
                    with col_math2:
                        st.write("**Звичайний стан:**")
                        st.dataframe(pd.DataFrame([math_analyses[1]]).T.rename(columns={0: 'Значення'}), use_container_width=True, height=300, key="math_normal_df")
                else:
                    st.dataframe(pd.DataFrame([math_analyses[0]]).T.rename(columns={0: 'Значення'}), use_container_width=True, height=300, key="math_single_df")

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
            st.warning("⚠️ Звичайний стан (T ≥ T_c)")
        
        st.write(f"**Критична температура T_c:** {Tc} K")

        # Виносимо expander з колонки - робимо його окремим елементом
        st.header("📄 Експорт результатів")
        if st.button("📥 Згенерувати звіт", use_container_width=True, key="generate_report_btn"):
            input_data = {'field_type': field_type, 'E0': E0, 'j0': j0, 't_max': t_max, 'T_common': current_temp}
            
            t = np.linspace(0, t_max, 1000)
            physical_analyses_for_report = []
            math_analyses_for_report = []
            
            if comparison_mode == "Порівняння":
                j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0, T_common)
                j_normal = calculate_normal_current_drude(t, field_type, T_common, E0, a, omega, j0)
                physical_analyses_for_report = [
                    analyze_physical_characteristics(t, j_super, "Надпровідник", field_type, T_common, omega),
                    analyze_physical_characteristics(t, j_normal, "Звичайний стан", field_type, T_common, omega)
                ]
                math_analyses_for_report = [
                    analyze_mathematical_characteristics(t, j_super, "Надпровідник", field_type, omega),
                    analyze_mathematical_characteristics(t, j_normal, "Звичайний стан", field_type, omega)
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
                    physical_analyses_for_report = [analyze_physical_characteristics(t, j_data, "Звичайний стан", field_type, current_temp, omega)]
                    math_analyses_for_report = [analyze_mathematical_characteristics(t, j_data, "Звичайний стан", field_type, omega)]
            
            pdf_buffer = create_pdf_report(input_data, physical_analyses_for_report, math_analyses_for_report, st.session_state.saved_plots)
            st.download_button(
                label="⬇️ Завантажити PDF звіт",
                data=pdf_buffer,
                file_name="звіт_моделювання.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_report_btn"
            )

    # Інформаційні розділи ВИНОСИМО З КОЛОНОК - робимо окремими елементами
    st.markdown("---")
    
    with st.expander("ℹ️ Інструкція користування", key="instructions_expander"):
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

    with st.expander("🔬 Фізичні принципи", key="physics_expander"):
        st.markdown("""
        **Теоретичні основи моделювання:**
        
        **Надпровідний стан (T < Tₐ):**
        - Рівняння Лондонів: струм росте необмеженно лінійно/квадратично через відсутність опору
        - Критична температура для ніобію: **Tₐ = 9.2 K**
        
        **Звичайний стан (T ≥ Tₐ):**
        - Модель Друде: експоненційне насичення струму через опір
        - Закон Ома: стаціонарна поведінка струму
        - Час релаксації залежить від температури
        
        **Типи електричних полів:**
        - *Статичне* - постійне поле
        - *Лінійне* - поле що лінійно зростає з часом  
        - *Синусоїдальне* - змінне гармонійне поле
        """)

    with st.expander("📊 Про аналіз даних", key="analysis_expander"):
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

    # Додаємо окремий expander для фізичних констант
    with st.expander("🧮 Фізичні константи ніобію", key="constants_expander"):
        st.write(f"**e =** {e:.3e} Кл")
        st.write(f"**m =** {m:.3e} кг")
        st.write(f"**n₀ =** {n0:.2e} м⁻³")
        st.write(f"**τ_imp =** {tau_imp:.2e} с")
        st.write(f"**T_c =** {Tc} K")
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
