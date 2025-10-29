import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from io import BytesIO
import base64
from scipy.signal import find_peaks

# Константи для Ніобію
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
    """Розрахунок струму в звичайному стані - модель Друде з температурною залежністю"""
    ns = n0 * (1 - (T/Tc)**4)
    tau = 1 / (1/tau_imp + A_ph * T**5)
    sigma = (ns * e**2 * tau) / m
    
    if E_type == "Статичне":
        return j0 * np.exp(-t/tau) + sigma * E0 * tau * (1 - np.exp(-t/tau))
    elif E_type == "Лінійне":
        return j0 * np.exp(-t/tau) + sigma * a * E0 * tau**2 * (1 - np.exp(-t/tau))
    elif E_type == "Синусоїдальне":
        phase_shift = np.arctan(omega * tau)
        amplitude = (sigma * E0 * tau) / np.sqrt(1 + (omega * tau)**2)
        transient = j0 * np.exp(-t/tau)
        return transient + amplitude * np.sin(omega * t - phase_shift)

def analyze_graph_characteristics(t, j_data, state_name, field_type, omega=1.0):
    """Детальний аналіз графіка функції: екстремуми, похідні, точки перегину"""
    analysis = {}
    analysis['Стан'] = state_name
    analysis['Тип поля'] = field_type
    
    # Основні характеристики функції
    analysis['j(0)'] = f"{j_data[0]:.2e} А/м²"
    analysis['j(t_max)'] = f"{j_data[-1]:.2e} А/м²"
    analysis['Максимум'] = f"{np.max(j_data):.2e} А/м²"
    analysis['Мінімум'] = f"{np.min(j_data):.2e} А/м²"
    analysis['Середнє'] = f"{np.mean(j_data):.2e} А/м²"
    
    # Похідна (швидкість зміни струму)
    dj_dt = np.gradient(j_data, t)
    analysis['Макс. швидкість зростання'] = f"{np.max(dj_dt):.2e} А/м²с"
    analysis['Кінцева швидкість'] = f"{dj_dt[-1]:.2e} А/м²с"
    
    # Друга похідна (прискорення)
    d2j_dt2 = np.gradient(dj_dt, t)
    analysis['Макс. прискорення'] = f"{np.max(d2j_dt2):.2e} А/м²с²"
    
    # Аналіз екстремумів
    peaks, _ = find_peaks(j_data, height=np.max(j_data)*0.1)
    valleys, _ = find_peaks(-j_data, height=-np.min(j_data)*0.1)
    
    analysis['Кількість максимумів'] = len(peaks)
    analysis['Кількість мінімумів'] = len(valleys)
    
    if len(peaks) > 0:
        analysis['Перший максимум'] = f"t = {t[peaks[0]]:.2f} с, j = {j_data[peaks[0]]:.2e} А/м²"
    if len(valleys) > 0:
        analysis['Перший мінімум'] = f"t = {t[valleys[0]]:.2f} с, j = {j_data[valleys[0]]:.2e} А/м²"
    
    # Аналіз поведінки
    if field_type == "Статичне":
        if state_name == "Надпровідник":
            analysis['Тип функції'] = "Лінійна"
            analysis['Коефіцієнт нахилу'] = f"{(j_data[-1] - j_data[0]) / t[-1]:.2e} А/м²с"
        else:
            analysis['Тип функції'] = "Експоненційна"
            # Оцінка часу релаксації з графіка
            if j_data[-1] > j_data[0]:
                tau_est = -t[np.argmin(np.abs(j_data - 0.63*(j_data[-1] - j_data[0])))]
                analysis['Оцінка τ з графіка'] = f"{tau_est:.2f} с"
    
    elif field_type == "Лінійне":
        if state_name == "Надпровідник":
            analysis['Тип функції'] = "Квадратична"
            analysis['Коефіцієнт при t²'] = f"{(j_data[-1] - j_data[0]) / (t[-1]**2):.2e} А/м²с²"
    
    elif field_type == "Синусоїдальне":
        analysis['Тип функції'] = "Коливальна"
        if len(j_data) > 10:
            # Оцінка амплітуди коливань
            amplitude = (np.max(j_data) - np.min(j_data)) / 2
            analysis['Амплітуда коливань'] = f"{amplitude:.2e} А/м²"
            
            # Оцінка періоду коливань
            if len(peaks) > 1:
                period = t[peaks[1]] - t[peaks[0]]
                analysis['Період коливань'] = f"{period:.2f} с"
                analysis['Частота коливань'] = f"{1/period:.2f} Гц"
    
    # Аналіз монотонності
    increasing = np.all(dj_dt >= 0)
    decreasing = np.all(dj_dt <= 0)
    
    if increasing:
        analysis['Монотонність'] = "Зростаюча"
    elif decreasing:
        analysis['Монотонність'] = "Спадна"
    else:
        analysis['Монотонність'] = "Немонотонна"
    
    # Стабільність (чи виходить на стаціонарний стан)
    if len(j_data) > 100:
        last_tenth = j_data[len(j_data)*9//10:]
        variation = np.std(last_tenth) / np.mean(last_tenth) if np.mean(last_tenth) != 0 else 0
        analysis['Стабільність'] = "Стабільна" if variation < 0.05 else "Нестабільна"
    
    return analysis

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
        
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
            font_name = 'DejaVuSans'
        except:
            font_name = 'Helvetica'
        
        pdf.setFont(font_name, 16)
        pdf.drawString(100, 800, "ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ")
        
        pdf.setFont(font_name, 12)
        y_position = 750
        
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
            
            # АНАЛІЗ ГРАФІКІВ ФУНКЦІЙ
            j_super_vis = calculate_superconducting_current(t_visible, field_type, E0, a, omega, j0)
            j_normal_vis = calculate_normal_current(t_visible, field_type, T_common, E0, a, omega, j0)
            
            analysis_super = analyze_graph_characteristics(t_visible, j_super_vis, "Надпровідник", field_type, omega)
            analysis_normal = analyze_graph_characteristics(t_visible, j_normal_vis, "Звичайний метал", field_type, omega)
            
        elif comparison_mode == "Один стан":
            if 'T_super' in locals():
                j_super_ext = calculate_superconducting_current(t_extended, field_type, E0, a, omega, j0)
                fig.add_trace(go.Scatter(x=t_extended, y=j_super_ext, name='Надпровідник',
                                       line=dict(color='red', width=3)))
                # Аналіз одного графіка
                j_super_vis = calculate_superconducting_current(t_visible, field_type, E0, a, omega, j0)
                analysis_super = analyze_graph_characteristics(t_visible, j_super_vis, "Надпровідник", field_type, omega)
            else:
                j_normal_ext = calculate_normal_current(t_extended, field_type, T_normal, E0, a, omega, j0)
                fig.add_trace(go.Scatter(x=t_extended, y=j_normal_ext, name='Звичайний метал',
                                       line=dict(color='blue', width=3)))
                # Аналіз одного графіка
                j_normal_vis = calculate_normal_current(t_visible, field_type, T_normal, E0, a, omega, j0)
                analysis_normal = analyze_graph_characteristics(t_visible, j_normal_vis, "Звичайний метал", field_type, omega)
        
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
        
        # ТАБЛИЦЯ АНАЛІЗУ ГРАФІКІВ ФУНКЦІЙ
        if comparison_mode == "Порівняння":
            st.subheader("📊 Математичний аналіз графіків")
            
            col_analysis1, col_analysis2 = st.columns(2)
            
            with col_analysis1:
                st.write("**Надпровідник:**")
                analysis_df_super = pd.DataFrame([analysis_super])
                st.dataframe(
                    analysis_df_super.T.rename(columns={0: 'Значення'}),
                    use_container_width=True,
                    height=400
                )
            
            with col_analysis2:
                st.write("**Звичайний метал:**")
                analysis_df_normal = pd.DataFrame([analysis_normal])
                st.dataframe(
                    analysis_df_normal.T.rename(columns={0: 'Значення'}),
                    use_container_width=True,
                    height=400
                )
        
        elif comparison_mode == "Один стан":
            st.subheader("📊 Математичний аналіз графіка")
            if 'T_super' in locals():
                analysis_df = pd.DataFrame([analysis_super])
                st.dataframe(
                    analysis_df.T.rename(columns={0: 'Значення'}),
                    use_container_width=True,
                    height=400
                )
            else:
                analysis_df = pd.DataFrame([analysis_normal])
                st.dataframe(
                    analysis_df.T.rename(columns={0: 'Значення'}),
                    use_container_width=True,
                    height=400
                )

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

if __name__ == "__main__":
    main()
