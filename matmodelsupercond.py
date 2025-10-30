import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO
from scipy.signal import find_peaks
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
        analysis['Поведінка'] = "Квадратичне зростання" if state_name == "Надпровідник" else "Експоненційне насичення"
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
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import io
        
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=landscape(A4))
        
        try:
            # Спроба знайти шрифт
            font_path = "DejaVuSans.ttf"
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
                font_name = 'DejaVuSans'
            else:
                font_name = 'Helvetica'
        except:
            font_name = 'Helvetica'
        
        pdf.setFont(font_name, 16)
        pdf.drawString(100, 520, "ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ В НІОБІЇ")
        
        pdf.setFont(font_name, 12)
        y_position = 490
        
        # Параметри
        pdf.drawString(100, y_position, "Параметри моделювання:")
        y_position -= 20
        for k, v in input_data.items():
            pdf.drawString(120, y_position, f"- {k}: {v}")
            y_position -= 20
        y_position -= 10

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
                    pdf.drawString(x_pos + 5, y_position - 15, str(cell))
                    x_pos += col_widths[j]
                
                y_position -= row_height
                if y_position < 100:
                    pdf.showPage()
                    pdf.setFont(font_name, 12)
                    y_position = 490
        
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
                    analysis.get("f'(середнє)", 'N/A')
                ]
                
                for j, cell in enumerate(cells):
                    pdf.drawString(x_pos + 3, y_position - 15, str(cell))
                    x_pos += col_widths[j]
                
                y_position -= row_height
                if y_position < 100:
                    pdf.showPage()
                    pdf.setFont(font_name, 12)
                    y_position = 490
        
        # Висновки
        pdf.drawString(100, y_position, "Висновки:")
        y_position -= 25
        conclusions = [
            "• Надпровідник: необмежене зростання струму",
            "• Звичайний метал: насичення через опір",
            "• Фазовий зсув у синусоїдальному полі"
        ]
        for c in conclusions:
            pdf.drawString(120, y_position, c)
            y_position -= 18
            if y_position < 50:
                pdf.showPage()
                y_position = 490
        
        pdf.save()
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        buffer = BytesIO()
        text = "ЗВІТ НЕ ВДАЛОСЯ ЗГЕНЕРУВАТИ (PDF)\n" + str(e)
        buffer.write(text.encode('utf-8'))
        buffer.seek(0)
        return buffer

# =============================================================================
# СТОРІНКИ
# =============================================================================

def animations_page():
    st.header("Демонстраційні анімації")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Анімація зміни температури")
        if st.button("Запустити анімацію", key="temp_anim"):
            progress_bar = st.progress(0)
            plot_placeholder = st.empty()
            temps = np.linspace(1, 18, 35)
            
            for i, temp in enumerate(temps):
                progress_bar.progress(int((i / len(temps)) * 100))
                t_anim = np.linspace(0, 5, 200)
                j_super = calculate_superconducting_current(t_anim, "Статичне", 1.0, 1.0, 5.0, 0.0, temp)
                j_normal = calculate_normal_current_drude(t_anim, "Статичне", temp, 1.0, 1.0, 5.0, 0.0)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t_anim, y=j_super, name='Надпровідник', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=t_anim, y=j_normal, name='Метал', line=dict(color='blue')))
                fig.update_layout(title=f"T = {temp:.1f} K", xaxis_title="Час (с)", yaxis_title="j (А/м²)", height=500)
                fig.update_yaxes(tickformat=".2e")
                plot_placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(0.15)
            
            progress_bar.progress(100)
            st.success("Анімація завершена!")

def racing_page():
    st.header("Електронні Гонки")
    col1, col2 = st.columns(2)
    with col1:
        car1_type = st.radio("Машинка 1:", ["Надпровідник", "Метал"], key="car1")
        car1_temp = st.slider("T1", 1.0, 18.0, 4.2, key="t1")
    with col2:
        car2_type = st.radio("Машинка 2:", ["Надпровідник", "Метал"], key="car2")
        car2_temp = st.slider("T2", 1.0, influx 18.0, 12.0, key="t2")
    
    if st.button("Старт гонки"):
        st.info("Гонка запущена!")

def prediction_game_page():
    st.header("Передбач майбутнє")
    if st.button("Нова задача"):
        T = random.uniform(3.0, 15.0)
        state = "Надпровідник" if T < Tc else "Метал"
        st.info(f"Температура: {T:.1f} K")
        choice = st.radio("Як поводитиметься струм?", ["Нескінченне зростання", "Насичення"])
        if st.button("Перевірити"):
            correct = (state == "Надпровідник" and choice == "Нескінченне зростання") or \
                      (state == "Метал" and choice == "Насичення")
            st.write("Правильно!" if correct else "Ні, спробуй ще!")

# =============================================================================
# ОСНОВНА СТОРІНКА
# =============================================================================

def main_page():
    st.title("Моделювання динаміки струму в ніобії")
    
    if 'saved_plots' not in st.session_state:
        st.session_state.saved_plots = []
    
    with st.sidebar:
        st.header("Параметри")
        comparison_mode = st.radio("Режим:", ["Один стан", "Порівняння", "Збережені графіки"])
        
        field_type = st.selectbox("Тип поля:", ["Статичне", "Лінійне", "Синусоїдальне"])
        E0 = st.slider("E₀ (В/м)", 0.1, 100.0, 1.0, 0.1)
        j0 = st.slider("j₀ (А/м²)", 0.0, 100.0, 0.0, 0.1)
        t_max = st.slider("Час (с)", 0.1, 20.0, 5.0, 0.1)
        a = st.slider("a", 0.1, 10.0, 1.0, 0.1) if field_type == "Лінійне" else 1.0
        omega = st.slider("ω (рад/с)", 0.1, 50.0, 5.0, 0.1) if field_type == "Синусоїдальне" else 1.0
        
        # ЗАВЖДИ визначаємо metal_model
        metal_model = "Модель Друде (з перехідним процесом)"
        current_temp = 4.2
        
        if comparison_mode == "Порівняння":
            T_common = st.slider("Температура (K)", 0.1, 18.4, 4.2, 0.1)
            current_temp = T_common
        elif comparison_mode == "Один стан":
            T_input = st.slider("Температура (K)", 0.1, 18.4, 4.2, 0.1)
            current_temp = T_input
            auto_state = determine_state(T_input)
            if auto_state != "Надпровідник":
                metal_model = st.radio("Модель для металу:", 
                    ["Модель Друде (з перехідним процесом)", "Закон Ома (стаціонарний)"])
        else:
            T_multi = st.slider("Температура (K)", 0.1, 18.4, 4.2, 0.1)
            current_temp = T_multi

        # Кнопка збереження
        if comparison_mode != "Збережені графіки":
            if st.button("Зберегти графік"):
                t = np.linspace(0, t_max, 1000)
                plot_data = {
                    't': t, 'field_type': field_type, 'E0': E0, 'j0': j0,
                    'a': a, 'omega': omega, 'temperature': current_temp,
                    'mode': comparison_mode, 'timestamp': pd.Timestamp.now()
                }
                
                if comparison_mode == "Один стан":
                    state = determine_state(current_temp)
                    if state == "Надпровідник":
                        plot_data['j_data'] = calculate_superconducting_current(t, field_type, E0, a, omega, j0, current_temp)
                        plot_data['state'] = 'Надпровідник'
                    else:
                        func = calculate_normal_current_drude if metal_model.startswith("Друде") else calculate_normal_current_ohm
                        plot_data['j_data'] = func(t, field_type, current_temp, E0, a, omega, j0)
                        plot_data['state'] = 'Звичайний метал'
                        plot_data['model'] = metal_model
                else:
                    plot_data['j_super'] = calculate_superconducting_current(t, field_type, E0, a, omega, j0, current_temp)
                    plot_data['j_normal'] = calculate_normal_current_drude(t, field_type, current_temp, E0, a, omega, j0)
                    plot_data['state'] = 'Порівняння'
                
                st.session_state.saved_plots.append(plot_data)
                st.success(f"Збережено! Всього: {len(st.session_state.saved_plots)}")

        if st.session_state.saved_plots:
            if st.button("Очистити всі"):
                st.session_state.saved_plots = []
                st.success("Очищено!")

    # Основний контент
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if comparison_mode == "Збережені графіки":
            st.header("Збережені графіки")
            if not st.session_state.saved_plots:
                st.info("Немає збережених графіків.")
            else:
                fig = go.Figure()
                for i, p in enumerate(st.session_state.saved_plots):
                    if p['state'] == 'Надпровідник':
                        fig.add_trace(go.Scatter(x=p['t'], y=p['j_data'], name=f"Надпр. {i+1}"))
                    elif p['state'] == 'Звичайний метал':
                        fig.add_trace(go.Scatter(x=p['t'], y=p['j_data'], name=f"Метал {i+1}"))
                    elif p['state'] == 'Порівняння':
                        fig.add_trace(go.Scatter(x=p['t'], y=p['j_super'], name=f"Надпр. {i+1}"))
                        fig.add_trace(go.Scatter(x=p['t'], y=p['j_normal'], name=f"Метал {i+1}"))
                fig.update_layout(title="Усі збережені", height=600)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.header("Графіки")
            t = np.linspace(0, t_max, 1000)
            fig = go.Figure()
            phys, math_ = [], []
            
            if comparison_mode == "Порівняння":
                j_s = calculate_superconducting_current(t, field_type, E0, a, omega, j0, current_temp)
                j_n = calculate_normal_current_drude(t, field_type, current_temp, E0, a, omega, j0)
                fig.add_trace(go.Scatter(x=t, y=j_s, name="Надпровідник", line=dict(color='red')))
                fig.add_trace(go.Scatter(x=t, y=j_n, name="Метал (Друде)", line=dict(color='blue')))
                phys = [analyze_physical_characteristics(t, j_s, "Надпровідник", field_type, current_temp, omega),
                        analyze_physical_characteristics(t, j_n, "Звичайний метал", field_type, current_temp, omega)]
                math_ = [analyze_mathematical_characteristics(t, j_s, "Надпровідник", field_type, omega),
                         analyze_mathematical_characteristics(t, j_n, "Звичайний метал", field_type, omega)]
            else:
                state = determine_state(current_temp)
                if state == "Надпровідник":
                    j = calculate_superconducting_current(t, field_type, E0, a, omega, j0, current_temp)
                    fig.add_trace(go.Scatter(x=t, y=j, name="Надпровідник", line=dict(color='red')))
                else:
                    func = calculate_normal_current_drude if "Друде" in metal_model else calculate_normal_current_ohm
                    j = func(t, field_type, current_temp, E0, a, omega, j0)
                    fig.add_trace(go.Scatter(x=t, y=j, name="Метал", line=dict(color='blue')))
                phys = [analyze_physical_characteristics(t, j, state, field_type, current_temp, omega)]
                math_ = [analyze_mathematical_characteristics(t, j, state, field_type, omega)]
            
            fig.update_layout(title="Динаміка струму", height=500)
            fig.update_yaxes(tickformat=".2e")
            st.plotly_chart(fig, use_container_width=True)
            
            if phys:
                st.header("Фізичний аналіз")
                st.dataframe(pd.DataFrame(phys), use_container_width=True)
                st.header("Математичний аналіз")
                st.dataframe(pd.DataFrame(math_).T, use_container_width=True)

    with col2:
        st.header("Інформація")
        st.write(f"**Поле:** {field_type}")
        st.write(f"**E₀:** {E0} В/м")
        st.write(f"**T:** {current_temp} K")
        st.write(f"**Стан:** {determine_state(current_temp)}")
        
        if st.button("Згенерувати PDF"):
            input_data = {'field_type': field_type, 'E0': E0, 'j0': j0, 't_max': t_max, 'T_common': current_temp}
            t_rep = np.linspace(0, t_max, 1000)
            phys_rep, math_rep = [], []
            
            if comparison_mode == "Порівняння":
                js = calculate_superconducting_current(t_rep, field_type, E0, a, omega, j0, current_temp)
                jn = calculate_normal_current_drude(t_rep, field_type, current_temp, E0, a, omega, j0)
                phys_rep = [analyze_physical_characteristics(t_rep, js, "Надпровідник", field_type, current_temp, omega),
                            analyze_physical_characteristics(t_rep, jn, "Звичайний метал", field_type, current_temp, omega)]
                math_rep = [analyze_mathematical_characteristics(t_rep, js, "Надпровідник", field_type, omega),
                            analyze_mathematical_characteristics(t_rep, jn, "Звичайний метал", field_type, omega)]
            else:
                state = determine_state(current_temp)
                if state == "Надпровідник":
                    j = calculate_superconducting_current(t_rep, field_type, E0, a, omega, j0, current_temp)
                else:
                    func = calculate_normal_current_drude if "Друде" in metal_model else calculate_normal_current_ohm
                    j = func(t_rep, field_type, current_temp, E0, a, omega, j0)
                phys_rep = [analyze_physical_characteristics(t_rep, j, state, field_type, current_temp, omega)]
                math_rep = [analyze_mathematical_characteristics(t_rep, j, state, field_type, omega)]
            
            pdf = create_pdf_report(input_data, phys_rep, math_rep, st.session_state.saved_plots)
            st.download_button("Завантажити PDF", pdf, "звіт.pdf", "application/pdf")

# =============================================================================
# ОСНОВНА ФУНКЦІЯ
# =============================================================================

def main():
    st.set_page_config(page_title="Моделювання струму в ніобії", layout="wide")
    
    with st.sidebar:
        st.title("Навігація")
        page = st.radio("Сторінка:", [
            "Основна сторінка",
            "Анімації",
            "Гонки",
            "Передбачення"
        ], key="nav")
    
    if page == "Основна сторінка":
        main_page()
    elif page == "Анімації":
        animations_page()
    elif page == "Гонки":
        racing_page()
    elif page == "Передбачення":
        prediction_game_page()

if __name__ == "__main__":
    main()
