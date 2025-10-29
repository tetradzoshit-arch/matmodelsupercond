import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from io import BytesIO
import base64

# Константи для Ніобію
e = 1.6e-19  # Кл
m = 9.1e-31  # кг
Tc = 9.2  # К
n0 = 1.0e29  # м⁻³
tau_imp = 5.0e-14  # с
A_ph = 3.0e8  # коефіцієнт фононного розсіювання

def calculate_superconducting_current(t, E_type, E0=1.0, a=1.0, omega=1.0, j0=0.0):
    """Розрахунок струму в надпровідному стані"""
    K = (e**2 * n0) / m
    
    if E_type == "Статичне":
        return j0 + K * E0 * t
    elif E_type == "Лінійне":
        return j0 + K * (a * t**2) / 2
    elif E_type == "Синусоїдальне":
        return j0 + (K * E0 / omega) * (1 - np.cos(omega * t))

def calculate_normal_current(t, E_type, T, E0=1.0, a=1.0, omega=1.0, j0=0.0):
    """Розрахунок струму в звичайному стані"""
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

def create_pdf_report(data):
    """Створення PDF звіту з красивою таблицею"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import io
        
        # Реєстрація українського шрифту (якщо є) або використання стандартного
        try:
            # Спробуємо знайти стандартний шрифт, що підтримує кирилицю
            pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
            font_name = 'Arial'
        except:
            # Якщо не вдалося - використовуємо стандартний Helvetica
            font_name = 'Helvetica'
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # Стиль для українського тексту
        ukrainian_style = ParagraphStyle(
            'Ukrainian',
            parent=styles['Normal'],
            fontName=font_name,
            encoding='UTF-8'
        )
        
        story = []
        
        # Заголовок
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=font_name,
            fontSize=16,
            spaceAfter=30,
            alignment=1
        )
        title = Paragraph("ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ", title_style)
        story.append(title)
        
        # Параметри моделювання
        story.append(Paragraph("Параметри моделювання:", styles['Heading2']))
        params_data = [
            ["<b>Тип поля:</b>", data['field_type']],
            ["<b>Напруженість поля E₀:</b>", f"{data['E0']} В/м"],
            ["<b>Початковий струм j₀:</b>", f"{data['j0']} А/м²"],
            ["<b>Час моделювання:</b>", f"{data['t_max']} с"],
            ["<b>Температура:</b>", f"{data.get('T_common', data.get('T_super', data.get('T_normal', 'N/A')))} K"]
        ]
        
        # Створюємо таблицю з параметрами
        params_table = Table(params_data, colWidths=[5*cm, 7*cm])
        params_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(params_table)
        story.append(Spacer(1, 20))
        
        # Порівняльна таблиця
        story.append(Paragraph("Порівняльна таблиця властивостей:", styles['Heading2']))
        comparison_data = [
            ["<b>Характеристика</b>", "<b>Надпровідник</b>", "<b>Звичайний стан</b>"],
            ["Поведінка струму в статичному полі", "Необмежене лінійне зростання", "Експоненційне насичення"],
            ["Наявність опору", "Відсутній", "Присутній"],
            ["Фазовий зсув у змінному полі", "π/2 (90°)", "arctg(ωτ) - залежить від частоти"],
            ["Стаціонарний стан", "Не досягається", "Досягається (j = σE)"],
            ["Час релаксації", "Не визначає динаміку струму", "Ключовий параметр"]
        ]
        
        comp_table = Table(comparison_data, colWidths=[5*cm, 5*cm, 5*cm])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 20))
        
        # Висновки
        story.append(Paragraph("Висновки:", styles['Heading2']))
        conclusion_text = data.get('conclusion', 'Порівняльний аналіз показує фундаментальну різницю у динаміці струму між надпровідним та звичайним станами.')
        conclusion = Paragraph(conclusion_text, ukrainian_style)
        story.append(conclusion)
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except ImportError:
        # Резервний варіант - простий текст
        buffer = BytesIO()
        report_text = f"""
        ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ
        
        Параметри моделювання:
        Тип поля: {data['field_type']}
        Напруженість поля E₀: {data['E0']} В/м
        Початковий струм j₀: {data['j0']} А/м²
        Час моделювання: {data['t_max']} с
        Температура: {data.get('T_common', data.get('T_super', data.get('T_normal', 'N/A')))} K
        
        Висновки: {data.get('conclusion', 'Порівняльний аналіз показує фундаментальну різницю у динаміці струму')}
        """
        buffer.write(report_text.encode('utf-8'))
        buffer.seek(0)
        return buffer

def main():
    st.set_page_config(page_title="Моделювання струму", layout="wide")
    st.title("🎛️ Моделювання динаміки струму: надпровідник vs звичайний метал")
    
    # Ініціалізація сесії для зберігання графіків
    if 'saved_plots' not in st.session_state:
        st.session_state.saved_plots = []
    
    # Сайдбар з параметрами
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
        
        # Кнопка для збереження поточного графіку
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

    # Графіки
    st.header("📈 Графіки струму")
    t = np.linspace(0, t_max, 1000)
    fig = go.Figure()
    
    if comparison_mode == "Порівняння":
        j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0)
        j_normal = calculate_normal_current(t, field_type, T_common, E0, a, omega, j0)
        fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник', line=dict(color='red', width=3)))
        fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний метал', line=dict(color='blue', width=3)))
        
    elif comparison_mode == "Один стан":
        if 'T_super' in locals():
            j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0)
            fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник', line=dict(color='red', width=3)))
        else:
            j_normal = calculate_normal_current(t, field_type, T_normal, E0, a, omega, j0)
            fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний метал', line=dict(color='blue', width=3)))
    
    else:
        j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0)
        j_normal = calculate_normal_current(t, field_type, T_multi, E0, a, omega, j0)
        fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник (поточний)', line=dict(color='red', width=3)))
        fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний (поточний)', line=dict(color='blue', width=3)))
        
        for i, saved_plot in enumerate(st.session_state.saved_plots):
            j_super_saved = calculate_superconducting_current(t, saved_plot['field_type'], saved_plot['E0'], a, omega, saved_plot['j0'])
            fig.add_trace(go.Scatter(x=t, y=j_super_saved, name=f'Надпровідник {i+1}', line=dict(dash='dash')))
    
    fig.update_layout(title="Динаміка густини струму", xaxis_title="Час (с)", yaxis_title="Густина струму (А/м²)", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Аналіз та експорт
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📊 Аналіз результатів")
        st.subheader("Параметри розрахунку")
        st.write(f"**Тип поля:** {field_type}")
        st.write(f"**E₀ =** {E0} В/м")
        st.write(f"**j₀ =** {j0} А/м²")
        st.write(f"**Температура:** {current_temp} K")
        if current_temp < Tc:
            st.success("✅ Температура нижче Tкрит")
        else:
            st.warning("⚠️ Температура вище Tкрит")

    with col2:
        st.header("📄 Експорт результатів")
        if st.button("📥 Згенерувати PDF звіт", use_container_width=True):
            report_data = {
                'field_type': field_type,
                'E0': E0,
                'j0': j0,
                't_max': t_max,
                'T_common': current_temp,
                'conclusion': "Порівняльний аналіз показує фундаментальну різницю у динаміці струму між надпровідним та звичайним станами."
            }
            
            pdf_buffer = create_pdf_report(report_data)
            st.download_button(
                label="⬇️ Завантажити PDF звіт",
                data=pdf_buffer,
                file_name="звіт_моделювання_струму.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    # Таблиця порівняння
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
                "Не визначає динаміку струму",
            ],
            "Звичайний стан": [
                "Експоненційне насичення",
                "Присутній",
                "arctg(ωτ) - залежить від частоти", 
                "Досягається (j = σE)",
                "Ключовий параметр",
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True, height=300)
        st.caption("Таблиця 1: Порівняння динаміки струму в надпровідному та звичайному станах")

    # Довідка
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
            """)

if __name__ == "__main__":
    main()
