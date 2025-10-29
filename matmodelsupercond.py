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
    """Створення PDF звіту"""
    buffer = BytesIO()
    
    report_text = f"""
    ЗВІТ З МОДЕЛЮВАННЯ СТРУМУ
    =========================
    
    Параметри моделювання:
    - Тип поля: {data['field_type']}
    - Напруженість поля E₀: {data['E0']} В/м
    - Початковий струм j₀: {data['j0']} А/м²
    - Час моделювання: {data['t_max']} с
    - Температура: {data.get('T_common', data.get('T_super', data.get('T_normal', 'N/A')))} K
    
    Результати:
    - Надпровідник: {data.get('super_desc', 'N/A')}
    - Звичайний метал: {data.get('normal_desc', 'N/A')}
    
    Висновки: {data.get('conclusion', 'Порівняльний аналіз динаміки струму')}
    """
    
    buffer.write(report_text.encode())
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
            ["Один стан", "Порівняння", "Кілька графіків"],
            help="Режим 'Кілька графіків' дозволяє будувати декілька кривих на одному графіку"
        )
        
        st.subheader("Загальні параметри")
        field_type = st.selectbox(
            "Тип електричного поля:",
            ["Статичне", "Лінійне", "Синусоїдальне"]
        )
        
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
        elif comparison_mode == "Один стан":
            selected_state = st.radio("Оберіть стан:", ["Надпровідник", "Звичайний метал"])
            if selected_state == "Надпровідник":
                T_super = st.slider("Температура надпровідника (K)", 0.1, Tc-0.1, 4.2, 0.1)
            else:
                T_normal = st.slider("Температура звичайного металу (K)", 0.1, 15.0, 4.2, 0.1)
        else:  # Кілька графіків
            T_multi = st.slider("Температура для аналізу (K)", 0.1, 15.0, 4.2, 0.1)
        
        # Кнопка для збереження поточного графіку
        if st.button("💾 Зберегти поточний графік"):
            current_params = {
                'field_type': field_type,
                'E0': E0,
                'j0': j0,
                't_max': t_max,
                'label': f"{field_type}, E₀={E0}, T={T_common if comparison_mode == 'Порівняння' else T_multi}K"
            }
            st.session_state.saved_plots.append(current_params)
            st.success("Графік збережено!")
        
        # Кнопка для очищення всіх збережених графіків
        if st.button("🗑️ Очистити всі графіки"):
            st.session_state.saved_plots = []
            st.success("Всі графіки очищено!")

    # Основний контент - тільки графіки
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
        
    elif comparison_mode == "Один стан":
        if 'T_super' in locals():
            j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0)
            fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник',
                                   line=dict(color='red', width=3)))
        else:
            j_normal = calculate_normal_current(t, field_type, T_normal, E0, a, omega, j0)
            fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний метал',
                                   line=dict(color='blue', width=3)))
    
    else:  # Кілька графіків
        j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0)
        j_normal = calculate_normal_current(t, field_type, T_multi, E0, a, omega, j0)
        
        fig.add_trace(go.Scatter(x=t, y=j_super, name=f'Надпровідник (поточний)',
                               line=dict(color='red', width=3)))
        fig.add_trace(go.Scatter(x=t, y=j_normal, name=f'Звичайний (поточний)',
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

    # Тепер таблиця і вся інформація під графіками
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📊 Аналіз результатів")
        
        # Інформація про параметри
        st.subheader("Параметри розрахунку")
        st.write(f"**Тип поля:** {field_type}")
        st.write(f"**E₀ =** {E0} В/м")
        st.write(f"**j₀ =** {j0} А/м²")
        
        if comparison_mode == "Порівняння":
            st.write(f"**Температура:** {T_common} K")
            status = "✅ Температура нижче Tкрит" if T_common < Tc else "⚠️ Температура вище Tкрит"
            st.success(status) if T_common < Tc else st.warning(status)
        elif comparison_mode == "Один стан":
            if 'T_super' in locals():
                st.write(f"**Температура надпровідника:** {T_super} K")
            else:
                st.write(f"**Температура металу:** {T_normal} K")
        else:
            st.write(f"**Температура:** {T_multi} K")

    with col2:
        st.header("📄 Експорт результатів")
        if st.button("📥 Згенерувати PDF звіт", use_container_width=True):
            report_data = {
                'field_type': field_type,
                'E0': E0,
                'j0': j0,
                't_max': t_max,
                'super_desc': "Необмежене зростання струму",
                'normal_desc': "Експоненційне насичення", 
                'conclusion': "Порівняльний аналіз показує фундаментальну різницю у динаміці струму"
            }
            
            pdf_buffer = create_pdf_report(report_data)
            st.download_button(
                label="⬇️ Завантажити PDF звіт",
                data=pdf_buffer,
                file_name="звіт_моделювання_струму.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    # КРАСИВА ТАБЛИЦЯ З МОЖЛИВІСТЮ РОЗГОРНУТИ
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
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=300
        )
        
        st.caption("Таблиця 1: Порівняння динаміки струму в надпровідному та звичайному станах")

    
    col_info1, col_info2 = st.columns(2)
    
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
            
            **Примітка:** Модель використовує параметри, близькі до ніобію (Tc = 9.2 K) - 
            типового низькотемпературного надпровідника. Температура 4.2 K відповідає 
            температурі кипіння рідкого гелію. Сучасні дослідження сфокусовані на 
            високотемпературних надпровідниках (до 138 K), але низькотемпературні 
            надпровідники досі використовуються в техніці через кращі експлуатаційні 
            властивості.
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
            
            де:
            - τ - час релаксації
            - σ - провідність
            - nₛ - концентрація електронів
            """)

if __name__ == "__main__":
    main()
