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
e = 1.6e-19  # Кл (елементарний заряд)
m = 9.1e-31  # кг (маса електрона)
Tc = 9.2  # К (критична температура ніобію)
n0 = 1.0e29  # м⁻³ (концентрація електронів)
tau_imp = 5.0e-14  # с (час релаксації на домішках)
A_ph = 3.0e8  # коефіцієнт фононного розсіювання

def calculate_superconducting_current(t, E_type, E0=1.0, a=1.0, omega=1.0, j0=0.0):
    """Розрахунок струму в надпровідному стані - рівняння Лондонів"""
    K = (e**2 * n0) / m  # Константа з рівняння Лондонів
    
    if E_type == "Статичне":
        # j(t) = j₀ + (e²nₛ/m)E₀t - лінійне зростання
        return j0 + K * E0 * t
    elif E_type == "Лінійне":
        # j(t) = j₀ + (e²nₛ/m)(a·t²)/2 - квадратичне зростання
        return j0 + K * (a * t**2) / 2
    elif E_type == "Синусоїдальне":
        # j(t) = j₀ + (e²nₛE₀/mω)(1 - cos(ωt)) - коливання з амплітудою, що залежить від частоти
        return j0 + (K * E0 / omega) * (1 - np.cos(omega * t))

def calculate_normal_current(t, E_type, T, E0=1.0, a=1.0, omega=1.0, j0=0.0):
    """Розрахунок струму в звичайному стані - модель Друде з температурною залежністю"""
    # Температурна залежність параметрів
    ns = n0 * (1 - (T/Tc)**4)  # Концентрація надпровідних електронів
    tau = 1 / (1/tau_imp + A_ph * T**5)  # Час релаксації (правило Маттіссена)
    sigma = (ns * e**2 * tau) / m  # Провідність
    
    if E_type == "Статичне":
        # j(t) = j₀e^(-t/τ) + σE₀τ(1 - e^(-t/τ)) - експоненційне насичення
        return j0 * np.exp(-t/tau) + sigma * E0 * tau * (1 - np.exp(-t/tau))
    elif E_type == "Лінійне":
        # j(t) = j₀e^(-t/τ) + σaE₀τ²(1 - e^(-t/τ)) - експоненційне насичення до лінійного росту
        return j0 * np.exp(-t/tau) + sigma * a * E0 * tau**2 * (1 - np.exp(-t/tau))
    elif E_type == "Синусоїдальне":
        # j(t) = j₀e^(-t/τ) + [σE₀τ/√(1+(ωτ)²)]sin(ωt - arctg(ωτ)) - коливання з фазовим зсувом
        phase_shift = np.arctan(omega * tau)
        amplitude = (sigma * E0 * tau) / np.sqrt(1 + (omega * tau)**2)
        transient = j0 * np.exp(-t/tau)
        steady_state = amplitude * np.sin(omega * t - phase_shift)
        return transient + steady_state

def analyze_physical_characteristics(t, j_super, j_normal, field_type, T, omega=1.0):
    """ФІЗИЧНИЙ аналіз характеристик струму"""
    analyses = []
    
    for j_data, state_name in [(j_super, "Надпровідник"), (j_normal, "Звичайний метал")]:
        analysis = {}
        analysis['Параметр'] = state_name
        
        # Основні фізичні характеристики
        analysis['j(0)'] = f"{j_data[0]:.2e} А/м²"
        analysis['j(t_max)'] = f"{j_data[-1]:.2e} А/м²"
        analysis['j_max'] = f"{np.max(j_data):.2e} А/м²"
        analysis['j_min'] = f"{np.min(j_data):.2e} А/м²"
        
        # Фізична інтерпретація за типом поля
        if field_type == "Статичне":
            if state_name == "Надпровідник":
                analysis['Фізична поведінка'] = "Лінійне зростання (відсутність опору)"
                analysis['Швидкість зростання'] = f"{(j_data[-1] - j_data[0]) / t[-1]:.2e} А/м²с"
                analysis['Стаціонарний стан'] = "Не досягається"
            else:
                analysis['Фізична поведінка'] = "Експоненційне насичення (наявність опору)"
                analysis['Стаціонарний стан'] = f"j = {j_data[-1]:.2e} А/м²"
                
        elif field_type == "Лінійне":
            if state_name == "Надпровідник":
                analysis['Фізична поведінка'] = "Квадратичне зростання (інерційність)"
                analysis['Прискорення'] = f"{2*(j_data[-1] - j_data[0]) / (t[-1]**2):.2e} А/м²с²"
            else:
                analysis['Фізична поведінка'] = "Експоненційне насилення"
                
        elif field_type == "Синусоїдальне":
            if state_name == "Надпровідник":
                analysis['Фізична поведінка'] = "Коливання з постійною амплітудою"
                analysis['Фазовий зсув'] = "π/2 (90°)"
                analysis['Фізичний зміст'] = "Чиста індуктивність"
            else:
                # Фізичний аналіз для звичайного металу
                ns = n0 * (1 - (T/Tc)**4)
                tau_val = 1 / (1/tau_imp + A_ph * T**5)
                analysis['Фізична поведінка'] = "Затухаючі коливання"
                analysis['Фазовий зсув'] = f"{np.arctan(omega * tau_val):.3f} рад"
                analysis['Фізичний зміст'] = "Комбінація R та L"
                analysis['Час релаксації τ'] = f"{tau_val:.2e} с"
        
        analysis['Температура'] = f"{T} K"
        analysis['Стан'] = "Надпровідний" if T < Tc and state_name == "Надпровідник" else "Звичайний"
        
        analyses.append(analysis)
    
    return analyses

def analyze_mathematical_characteristics(t, j_data, state_name, field_type, omega=1.0):
    """МАТЕМАТИЧНИЙ аналіз графіка функції"""
    analysis = {}
    analysis['Функція'] = state_name
    
    # Математичні характеристики
    analysis['f(0)'] = f"{j_data[0]:.2e}"
    analysis['f(t_max)'] = f"{j_data[-1]:.2e}"
    analysis['max f(t)'] = f"{np.max(j_data):.2e}"
    analysis['min f(t)'] = f"{np.min(j_data):.2e}"
    analysis['f_avg'] = f"{np.mean(j_data):.2e}"
    
    # Похідна (швидкість зміни)
    dt = t[1] - t[0]
    dj_dt = np.gradient(j_data, dt)
    analysis["f'(max)"] = f"{np.max(dj_dt):.2e}"
    analysis["f'(min)"] = f"{np.min(dj_dt):.2e}"
    analysis["f'(final)"] = f"{dj_dt[-1]:.2e}"
    
    # Екстремуми
    peaks, _ = find_peaks(j_data, prominence=np.max(j_data)*0.01)
    valleys, _ = find_peaks(-j_data, prominence=-np.min(j_data)*0.01)
    
    analysis['Кількість екстремумів'] = len(peaks) + len(valleys)
    analysis['Локальні максимуми'] = len(peaks)
    analysis['Локальні мінімуми'] = len(valleys)
    
    # Математична класифікація
    if field_type == "Статичне":
        if state_name == "Надпровідник":
            analysis['Тип функції'] = "Лінійна: f(t) = at + b"
            analysis['Коефіцієнт a'] = f"{(j_data[-1] - j_data[0]) / t[-1]:.2e}"
        else:
            analysis['Тип функції'] = "Експоненційна: f(t) = A(1 - e^(-t/τ))"
            
    elif field_type == "Лінійне":
        if state_name == "Надпровідник":
            analysis['Тип функції'] = "Квадратична: f(t) = at² + b"
            analysis['Коефіцієнт a'] = f"{(j_data[-1] - j_data[0]) / (t[-1]**2):.2e}"
        else:
            analysis['Тип функції'] = "Експоненційно-лінійна"
            
    elif field_type == "Синусоїдальне":
        analysis['Тип функції'] = "Коливальна"
        if len(j_data) > 10:
            amplitude = (np.max(j_data) - np.min(j_data)) / 2
            analysis['Амплітуда'] = f"{amplitude:.2e}"
            
            # Оцінка періоду
            if len(peaks) > 1:
                period = t[peaks[1]] - t[peaks[0]]
                analysis['Період'] = f"{period:.2f} с"
                analysis['Частота'] = f"{1/period:.2f} Гц"
    
    # Монотонність
    strictly_increasing = np.all(dj_dt > 0)
    strictly_decreasing = np.all(dj_dt < 0)
    
    if strictly_increasing:
        analysis['Монотонність'] = "Строго зростаюча"
    elif strictly_decreasing:
        analysis['Монотонність'] = "Строго спадна"
    else:
        analysis['Монотонність'] = "Немонотонна"
    
    # Опуклість/угнутість (друга похідна)
    d2j_dt2 = np.gradient(dj_dt, dt)
    avg_curvature = np.mean(d2j_dt2)
    analysis['Середня кривизна'] = f"{avg_curvature:.2e}"
    
    return analysis

def create_comprehensive_pdf_report(input_data, physical_analyses, math_analyses):
    """Створення повного PDF звіту з усіма результатами"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
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
        pdf.drawString(100, y_position, "ПОВНИЙ ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ")
        y_position -= 30
        
        # Вхідні параметри
        pdf.setFont(font_name, 14)
        pdf.drawString(100, y_position, "1. ВХІДНІ ПАРАМЕТРИ:")
        y_position -= 20
        pdf.setFont(font_name, 12)
        
        params = [
            f"Тип поля: {input_data['field_type']}",
            f"Напруженість поля E₀: {input_data['E0']} В/м",
            f"Початковий струм j₀: {input_data['j0']} А/м²",
            f"Час моделювання: {input_data['t_max']} с",
            f"Температура: {input_data['T_common']} K",
        ]
        
        for param in params:
            pdf.drawString(120, y_position, param)
            y_position -= 15
        
        y_position -= 10
        
        # Фізичний аналіз
        pdf.setFont(font_name, 14)
        pdf.drawString(100, y_position, "2. ФІЗИЧНИЙ АНАЛІЗ:")
        y_position -= 20
        pdf.setFont(font_name, 12)
        
        for i, analysis in enumerate(physical_analyses):
            pdf.drawString(100, y_position, f"{analysis['Параметр']}:")
            y_position -= 15
            
            for key, value in analysis.items():
                if key != 'Параметр':
                    pdf.drawString(120, y_position, f"{key}: {value}")
                    y_position -= 12
                    if y_position < 50:
                        pdf.showPage()
                        y_position = 800
                        pdf.setFont(font_name, 12)
            
            y_position -= 10
        
        y_position -= 10
        
        # Математичний аналіз
        pdf.setFont(font_name, 14)
        pdf.drawString(100, y_position, "3. МАТЕМАТИЧНИЙ АНАЛІЗ:")
        y_position -= 20
        pdf.setFont(font_name, 12)
        
        for i, analysis in enumerate(math_analyses):
            pdf.drawString(100, y_position, f"{analysis['Функція']}:")
            y_position -= 15
            
            for key, value in analysis.items():
                if key != 'Функція':
                    pdf.drawString(120, y_position, f"{key}: {value}")
                    y_position -= 12
                    if y_position < 50:
                        pdf.showPage()
                        y_position = 800
                        pdf.setFont(font_name, 12)
            
            y_position -= 10
        
        # Висновки
        y_position -= 20
        pdf.setFont(font_name, 14)
        pdf.drawString(100, y_position, "4. ВИСНОВКИ:")
        y_position -= 20
        pdf.setFont(font_name, 12)
        
        conclusions = [
            "• Надпровідник демонструє ідеальну провідність",
            "• Звичайний метал має опір та час релаксації",
            "• Синусоїдальне поле виявляє фазові зсуви",
            "• Моделі коректно описують фізичні явища"
        ]
        
        for conclusion in conclusions:
            pdf.drawString(100, y_position, conclusion)
            y_position -= 15
        
        pdf.save()
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        # Резервний варіант
        buffer = BytesIO()
        report_text = "ПОВНИЙ ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ\n\n"
        report_text += "Вхідні параметри:\n"
        for key, value in input_data.items():
            report_text += f"{key}: {value}\n"
        
        buffer.write(report_text.encode('utf-8'))
        buffer.seek(0)
        return buffer

def main():
    st.set_page_config(page_title="Моделювання струму", layout="wide")
    st.title("🔬 Моделювання динаміки струму: надпровідник vs звичайний метал")
    
    if 'saved_plots' not in st.session_state:
        st.session_state.saved_plots = []
    
    with st.sidebar:
        st.header("⚙️ Параметри моделювання")
        
        comparison_mode = st.radio(
            "Режим відображення:",
            ["Один стан", "Порівняння", "Кілька графіків"]
        )
        
        st.subheader("Загальні параметри")
        field_type = st.selectbox("Тип електричного поля:", ["Статичне", "Лінійне", "Синусоїдальне"])
        E0 = st.slider("Напруженість поля E₀ (В/м)", 0.1, 10.0, 1.0, 0.1)
        j0 = st.slider("Початковий струм j₀ (А/м²)", 0.0, 10.0, 0.0, 0.1)
        t_max = st.slider("Час моделювання (с)", 0.1, 10.0, 5.0, 0.1)
        
        if field_type == "Лінійне":
            a = st.slider("Швидкість росту поля a", 0.1, 5.0, 1.0, 0.1)
        else:
            a = 1.0
            
        if field_type == "Синусоїдальне":
            omega = st.slider("Частота ω (рад/с)", 0.1, 10.0, 1.0, 0.1)
        else:
            omega = 1.0
        
        st.subheader("Параметри станів")
        if comparison_mode == "Порівняння":
            T_common = st.slider("Температура для порівняння (K)", 0.1, 15.0, 4.2, 0.1)
            current_temp = T_common
        elif comparison_mode == "Один стан":
            selected_state = st.radio("Оберіть стан:", ["Надпровідник", "Звичайний метал"])
            if selected_state == "Надпровідник":
                T_super = st.slider("Температура надпровідника (K)", 0.1, Tc-0.1, 4.2, 0.1)
                current_temp = T_super
            else:
                T_normal = st.slider("Температура звичайного металу (K)", 0.1, 15.0, 4.2, 0.1)
                current_temp = T_normal
        else:
            T_multi = st.slider("Температура для аналізу (K)", 0.1, 15.0, 4.2, 0.1)
            current_temp = T_multi
        
        if st.button("💾 Зберегти поточний графік"):
            current_params = {
                'field_type': field_type,
                'E0': E0,
                'j0': j0,
                't_max': t_max,
                'label': f"{field_type}, E₀={E0}, T={current_temp}K"
            }
            st.session_state.saved_plots.append(current_params)
            st.success("Графік збережено!")
        
        if st.button("🗑️ Очистити всі графіки"):
            st.session_state.saved_plots = []
            st.success("Всі графіки очищено!")

    # Основний контент
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Графіки струму")
        
        t = np.linspace(0, t_max, 1000)
        fig = go.Figure()
        
        if comparison_mode == "Порівняння":
            j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0)
            j_normal = calculate_normal_current(t, field_type, T_common, E0, a, omega, j0)
            
            fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник', 
                                   line=dict(color='red', width=3)))
            fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний метал',
                                   line=dict(color='blue', width=3)))
            
            # АНАЛІЗИ
            physical_analyses = analyze_physical_characteristics(t, j_super, j_normal, field_type, T_common, omega)
            math_analyses = [
                analyze_mathematical_characteristics(t, j_super, "Надпровідник", field_type, omega),
                analyze_mathematical_characteristics(t, j_normal, "Звичайний метал", field_type, omega)
            ]
            
        elif comparison_mode == "Один стан":
            if 'T_super' in locals():
                j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0)
                fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник',
                                       line=dict(color='red', width=3)))
            else:
                j_normal = calculate_normal_current(t, field_type, T_normal, E0, a, omega, j0)
                fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний метал',
                                       line=dict(color='blue', width=3)))
        
        else:
            j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0)
            j_normal = calculate_normal_current(t, field_type, T_multi, E0, a, omega, j0)
            
            fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник (поточний)',
                                   line=dict(color='red', width=3)))
            fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний (поточний)',
                                   line=dict(color='blue', width=3)))
            
            for i, saved_plot in enumerate(st.session_state.saved_plots):
                j_super_saved = calculate_superconducting_current(t, saved_plot['field_type'], 
                                                                saved_plot['E0'], a, omega, saved_plot['j0'])
                fig.add_trace(go.Scatter(x=t, y=j_super_saved, name=f'Надпровідник {i+1}',
                                       line=dict(dash='dash')))
        
        fig.update_layout(
            title="Динаміка густини струму",
            xaxis_title="Час (с)",
            yaxis_title="Густина струму (А/м²)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ТАБЛИЦІ АНАЛІЗУ
        if comparison_mode == "Порівняння":
            st.header("📊 ФІЗИЧНИЙ АНАЛІЗ")
            physical_df = pd.DataFrame(physical_analyses)
            st.dataframe(physical_df, use_container_width=True, height=200)
            
            st.header("🧮 МАТЕМАТИЧНИЙ АНАЛІЗ")
            col_math1, col_math2 = st.columns(2)
            
            with col_math1:
                st.write("**Надпровідник:**")
                math_df_super = pd.DataFrame([math_analyses[0]])
                st.dataframe(math_df_super.T.rename(columns={0: 'Значення'}), use_container_width=True, height=400)
            
            with col_math2:
                st.write("**Звичайний метал:**")
                math_df_normal = pd.DataFrame([math_analyses[1]])
                st.dataframe(math_df_normal.T.rename(columns={0: 'Значення'}), use_container_width=True, height=400)

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
        if st.button("📥 Згенерувати повний звіт", use_container_width=True) and comparison_mode == "Порівняння":
            input_data = {
                'field_type': field_type,
                'E0': E0,
                'j0': j0,
                't_max': t_max,
                'T_common': T_common,
            }
            
            pdf_buffer = create_comprehensive_pdf_report(input_data, physical_analyses, math_analyses)
            st.download_button(
                label="⬇️ Завантажити PDF звіт",
                data=pdf_buffer,
                file_name="повний_звіт_моделювання.pdf",
                mime="application/pdf",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
