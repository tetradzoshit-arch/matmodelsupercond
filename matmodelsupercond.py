import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from io import BytesIO
import base64

# Константи для Ніобію (ТОЧНО ТІ, ЩО ТИ ЗАДАВАЛА)
e = 1.6e-19  # Кл
m = 9.1e-31  # кг
Tc = 9.2  # К
n0 = 1.0e29  # м⁻³
tau_imp = 5.0e-14  # с
A_ph = 3.0e8  # коефіцієнт фононного розсіювання

def calculate_superconducting_current(t, E_type, E0=1.0, a=1.0, omega=1.0, j0=0.0):
    """Розрахунок струму в надпровідному стані - рівняння Лондонів"""
    K = (e**2 * n0) / m  # Константа з твоїх виведень
    
    if E_type == "Статичне":
        # j(t) = j₀ + K·E₀·t
        return j0 + K * E0 * t
    elif E_type == "Лінійне":
        # j(t) = j₀ + K·(a·t²)/2
        return j0 + K * (a * t**2) / 2
    elif E_type == "Синусоїдальне":
        # j(t) = j₀ + (K·E₀/ω)·(1 - cos(ωt))
        return j0 + (K * E0 / omega) * (1 - np.cos(omega * t))

def calculate_normal_current(t, E_type, T, E0=1.0, a=1.0, omega=1.0, j0=0.0):
    """Розрахунок струму в звичайному стані - модель Друде з температурною залежністю"""
    # Температурна залежність параметрів (з твоєї роботи)
    ns = n0 * (1 - (T/Tc)**4)  # Концентрація надпровідних електронів
    tau = 1 / (1/tau_imp + A_ph * T**5)  # Час релаксації за правилом Маттіссена
    sigma = (ns * e**2 * tau) / m  # Провідність
    
    if E_type == "Статичне":
        # j(t) = j₀·e^(-t/τ) + σ·E₀·τ·(1 - e^(-t/τ))
        return j0 * np.exp(-t/tau) + sigma * E0 * tau * (1 - np.exp(-t/tau))
    elif E_type == "Лінійне":
        # j(t) = j₀·e^(-t/τ) + σ·a·E₀·τ²·(1 - e^(-t/τ))
        return j0 * np.exp(-t/tau) + sigma * a * E0 * tau**2 * (1 - np.exp(-t/tau))
    elif E_type == "Синусоїдальне":
        # j(t) = j₀·e^(-t/τ) + (σ·E₀·τ/√(1+(ωτ)²))·sin(ωt - arctg(ωτ))
        phase_shift = np.arctan(omega * tau)
        amplitude = (sigma * E0 * tau) / np.sqrt(1 + (omega * tau)**2)
        transient = j0 * np.exp(-t/tau)
        return transient + amplitude * np.sin(omega * t - phase_shift)

def analyze_current_characteristics(t, j_super, j_normal, field_type, T):
    """Аналіз характеристик струму для обох станів"""
    analyses = []
    
    for j_data, state_name in [(j_super, "Надпровідник"), (j_normal, "Звичайний метал")]:
        analysis = {}
        analysis['Стан'] = state_name
        analysis['Тип поля'] = field_type
        
        # Основні характеристики
        analysis['Кінцеве значення j(t)'] = f"{j_data[-1]:.2e} А/м²"
        analysis['Максимальне значення'] = f"{np.max(j_data):.2e} А/м²"
        analysis['Мінімальне значення'] = f"{np.min(j_data):.2e} А/м²"
        
        # Аналіз поведінки за типом поля
        if field_type == "Статичне":
            if state_name == "Надпровідник":
                analysis['Поведінка'] = "Лінійне зростання"
                analysis['Швидкість зростання'] = f"{(j_data[-1] - j_data[0]) / t[-1]:.2e} А/м²с"
            else:
                analysis['Поведінка'] = "Експоненційне насичення"
                analysis['Стаціонарний стан'] = "Досягається"
                
        elif field_type == "Лінійне":
            if state_name == "Надпровідник":
                analysis['Поведінка'] = "Квадратичне зростання"
            else:
                analysis['Поведінка'] = "Експоненційне насичення"
                
        elif field_type == "Синусоїдальне":
            if state_name == "Надпровідник":
                analysis['Поведінка'] = "Коливання з постійною амплітудою"
                analysis['Фазовий зсув'] = "π/2"
            else:
                # Для звичайного металу розраховуємо параметри
                ns = n0 * (1 - (T/Tc)**4)
                tau_val = 1 / (1/tau_imp + A_ph * T**5)
                analysis['Поведінка'] = "Затухаючі коливання"
                analysis['Фазовий зсув'] = f"{np.arctan(omega * tau_val):.3f} рад"
                analysis['Амплітуда'] = f"{np.std(j_data[len(j_data)//2:]) * np.sqrt(2):.2e} А/м²"
        
        # Додаткові параметри
        analysis['Температура'] = f"{T} K"
        
        analyses.append(analysis)
    
    return analyses

def create_pdf_report(data):
    """Створення PDF звіту"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import io
        
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        
        # Встановлюємо шрифт
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
            font_name = 'DejaVuSans'
        except:
            font_name = 'Helvetica'
        
        # Заголовок
        pdf.setFont(font_name, 16)
        pdf.drawString(100, 800, "ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ")
        
        pdf.setFont(font_name, 12)
        y_position = 750
        
        # Параметри моделювання
        pdf.drawString(100, y_position, "Параметри моделювання:")
        y_position -= 20
        pdf.drawString(120, y_position, f"- Тип поля: {data['field_type']}")
        y_position -= 20
        pdf.drawString(120, y_position, f"- Напруженість поля E: {data['E0']} В/м")
        y_position -= 20
        pdf.drawString(120, y_position, f"- Початковий струм j: {data['j0']} А/м²")
        y_position -= 20
        pdf.drawString(120, y_position, f"- Час моделювання: {data['t_max']} с")
        y_position -= 20
        pdf.drawString(120, y_position, f"- Температура: {data.get('T_common', 'N/A')} K")
        y_position -= 30
        
        pdf.save()
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        buffer = BytesIO()
        report_text = f"""
        ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ
        
        Параметри моделювання:
        - Тип поля: {data['field_type']}
        - Напруженість поля E: {data['E0']} В/м
        - Початковий струм j: {data['j0']} А/м²
        - Час моделювання: {data['t_max']} с
        - Температура: {data.get('T_common', 'N/A')} K
        
        Висновки: Порівняльний аналіз показує фундаментальну різницю у динаміці струму.
        """
        buffer.write(report_text.encode('utf-8'))
        buffer.seek(0)
        return buffer

def main():
    st.set_page_config(page_title="Моделювання струму", layout="wide")
    st.title("🎛️ Моделювання динаміки струму: надпровідник vs звичайний метал")
    
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
        
        t_extended = np.linspace(0, t_max * 2, 2000)
        t_visible = np.linspace(0, t_max, 1000)
        
        fig = go.Figure()
        
        if comparison_mode == "Порівняння":
            j_super_ext = calculate_superconducting_current(t_extended, field_type, E0, a, omega, j0)
            j_normal_ext = calculate_normal_current(t_extended, field_type, T_common, E0, a, omega, j0)
            
            fig.add_trace(go.Scatter(x=t_extended, y=j_super_ext, name='Надпровідник', 
                                   line=dict(color='red', width=3)))
            fig.add_trace(go.Scatter(x=t_extended, y=j_normal_ext, name='Звичайний метал',
                                   line=dict(color='blue', width=3)))
            
            # АНАЛІЗ ХАРАКТЕРИСТИК
            j_super_vis = calculate_superconducting_current(t_visible, field_type, E0, a, omega, j0)
            j_normal_vis = calculate_normal_current(t_visible, field_type, T_common, E0, a, omega, j0)
            analyses = analyze_current_characteristics(t_visible, j_super_vis, j_normal_vis, field_type, T_common)
            
        elif comparison_mode == "Один стан":
            if 'T_super' in locals():
                j_super_ext = calculate_superconducting_current(t_extended, field_type, E0, a, omega, j0)
                fig.add_trace(go.Scatter(x=t_extended, y=j_super_ext, name='Надпровідник',
                                       line=dict(color='red', width=3)))
            else:
                j_normal_ext = calculate_normal_current(t_extended, field_type, T_normal, E0, a, omega, j0)
                fig.add_trace(go.Scatter(x=t_extended, y=j_normal_ext, name='Звичайний метал',
                                       line=dict(color='blue', width=3)))
        
        else:
            j_super_ext = calculate_superconducting_current(t_extended, field_type, E0, a, omega, j0)
            j_normal_ext = calculate_normal_current(t_extended, field_type, T_multi, E0, a, omega, j0)
            
            fig.add_trace(go.Scatter(x=t_extended, y=j_super_ext, name='Надпровідник (поточний)',
                                   line=dict(color='red', width=3)))
            fig.add_trace(go.Scatter(x=t_extended, y=j_normal_ext, name='Звичайний (поточний)',
                                   line=dict(color='blue', width=3)))
            
            for i, saved_plot in enumerate(st.session_state.saved_plots):
                j_super_saved = calculate_superconducting_current(t_extended, saved_plot['field_type'], 
                                                                saved_plot['E0'], a, omega, saved_plot['j0'])
                fig.add_trace(go.Scatter(x=t_extended, y=j_super_saved, name=f'Надпровідник {i+1}',
                                       line=dict(dash='dash')))
        
        fig.update_layout(
            title="Динаміка густини струму",
            xaxis_title="Час (с)",
            yaxis_title="Густина струму (А/м²)",
            height=500,
            xaxis=dict(range=[0, t_max]),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ТАБЛИЦЯ АНАЛІЗУ ГРАФІКІВ
        if comparison_mode == "Порівняння":
            st.subheader("📊 Аналіз характеристик струму")
            
            analysis_df = pd.DataFrame(analyses)
            st.dataframe(
                analysis_df,
                use_container_width=True,
                hide_index=True,
                height=200
            )
        
        # ГРАФІК АМПЛІТУДА-ЧАСТОТА
        if field_type == "Синусоїдальне" and comparison_mode == "Порівняння":
            with st.expander("📡 Аналіз частотної залежності", expanded=False):
                st.subheader("Залежність амплітуди струму від частоти")
                
                frequencies = np.logspace(-1, 2, 100)
                amplitudes_super = []
                amplitudes_normal = []
                
                for freq in frequencies:
                    # Надпровідник: амплітуда = (K·E₀)/ω
                    K = (e**2 * n0) / m
                    amp_super = (K * E0) / freq
                    amplitudes_super.append(amp_super)
                    
                    # Звичайний метал: амплітуда = (σ·E₀·τ)/√(1+(ωτ)²)
                    ns = n0 * (1 - (T_common/Tc)**4)
                    tau = 1 / (1/tau_imp + A_ph * T_common**5)
                    sigma = (ns * e**2 * tau) / m
                    amp_normal = (sigma * E0 * tau) / np.sqrt(1 + (freq * tau)**2)
                    amplitudes_normal.append(amp_normal)
                
                fig_freq = go.Figure()
                fig_freq.add_trace(go.Scatter(x=frequencies, y=amplitudes_super, 
                                            name='Надпровідник', line=dict(color='red')))
                fig_freq.add_trace(go.Scatter(x=frequencies, y=amplitudes_normal,
                                            name='Звичайний метал', line=dict(color='blue')))
                fig_freq.update_layout(
                    xaxis_title="Частота ω (рад/с)",
                    yaxis_title="Амплітуда струму (А/м²)",
                    xaxis_type="log",
                    yaxis_type="log",
                    height=300
                )
                st.plotly_chart(fig_freq, use_container_width=True)

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
        if st.button("📥 Згенерувати PDF звіт", use_container_width=True):
            report_data = {
                'field_type': field_type,
                'E0': E0,
                'j0': j0,
                't_max': t_max,
                'T_common': current_temp,
            }
            
            pdf_buffer = create_pdf_report(report_data)
            st.download_button(
                label="⬇️ Завантажити PDF звіт",
                data=pdf_buffer,
                file_name="звіт_моделювання_струму.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    # ПОРІВНЯЛЬНА ТАБЛИЦЯ ВЛАСТИВОСТЕЙ
    st.header("📋 Порівняльна таблиця властивостей")
    with st.expander("🎯 Розгорнути таблицю порівняння", expanded=True):
        comparison_data = {
            "Характеристика": [
                "Поведінка струму в статичному полі", 
                "Наявність опору",
                "Фазовий зсув у змінному полі", 
                "Стаціонарний стан",
                "Час релаксації",
            ],
            "Надпровідник": [
                "Необмежене лінійне зростання",
                "Відсутній", 
                "π/2 (90°)",
                "Не досягається",
                "Не визначає динаміку",
            ],
            "Звичайний метал": [
                "Експоненційне насичення",
                "Присутній",
                "arctg(ωτ) - залежить від частоти", 
                "Досягається (j = σE)",
                "Ключовий параметр",
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True, height=300)

    # ДОВІДКА
    st.header("📖 Довідка")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        with st.expander("🔬 Фізичні основи моделі", expanded=False):
            st.write("""
            **Надпровідний стан (рівняння Лондонів):**
            - Відсутність опору
            - Необмежене зростання струму в постійному полі
            - Фазовий зсув π/2 у змінному полі
            
            **Звичайний стан (модель Друде):**
            - Наявність опору через зіткнення
            - Експоненційне насичення струму
            - Частотно-залежний фазовий зсув
            
            **Примітка:** Модель використовує параметри, близькі до ніобію (Tc = 9.2 K)
            """)
    
    with col_info2:
        with st.expander("🧮 Математичні моделі", expanded=False):
            st.write("""
            **Надпровідник:** 
            ```python
            dj/dt = (e²nₛ/m)E(t)
            ```
            
            **Звичайний метал:** 
            ```python
            dj/dt + j/τ = (σ/τ)E(t)
            ```
            
            **Температурна залежність:**
            - nₛ = n₀[1 - (T/Tc)⁴]
            - 1/τ = 1/τ_imp + A·T⁵
            """)

if __name__ == "__main__":
    main()
