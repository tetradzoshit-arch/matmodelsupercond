
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Константи для Ніобію
e = 1.6e-19  # Кл
m = 9.1e-31  # кг
Tc = 9.2  # К
n0 = 1.0e29  # м⁻³
tau_imp = 5.0e-14  # с
A_ph = 3.0e8  # коефіцієнт фононного розсіювання

def calculate_superconducting_current(t, E_type, E0=1.0, a=1.0, omega=1.0, j0=0.0):
    """Розрахунок струму в надпровідному стані"""
    # Константа K для надпровідника
    K = (e**2 * n0) / m
    
    if E_type == "Статичне":
        return j0 + K * E0 * t
    elif E_type == "Лінійне":
        return j0 + K * (a * t**2) / 2
    elif E_type == "Синусоїдальне":
        return j0 + (K * E0 / omega) * (1 - np.cos(omega * t))

def calculate_normal_current(t, E_type, T, E0=1.0, a=1.0, omega=1.0, j0=0.0):
    """Розрахунок струму в звичайному стані"""
    # Температурна залежність параметрів
    ns = n0 * (1 - (T/Tc)**4)
    tau = 1 / (1/tau_imp + A_ph * T**5)
    sigma = (ns * e**2 * tau) / m
    
    if E_type == "Статичне":
        return j0 * np.exp(-t/tau) + sigma * E0 * tau * (1 - np.exp(-t/tau))
    elif E_type == "Лінійне":
        return j0 * np.exp(-t/tau) + sigma * a * E0 * tau**2 * (1 - np.exp(-t/tau))
    elif E_type == "Синусоїдальне":
        # Спрощена формула для усталеного режиму
        phase_shift = np.arctan(omega * tau)
        amplitude = (sigma * E0 * tau) / np.sqrt(1 + (omega * tau)**2)
        transient = j0 * np.exp(-t/tau)
        return transient + amplitude * np.sin(omega * t - phase_shift)

def main():
    st.set_page_config(page_title="Моделювання струму", layout="wide")
    st.title("🎛️ Моделювання динаміки струму: надпровідник vs звичайний метал")
    
    # Сайдбар з параметрами
    with st.sidebar:
        st.header("⚙️ Параметри моделювання")
        
        # Режим відображення
        comparison_mode = st.radio(
            "Режим відображення:",
            ["Один стан", "Порівняння"],
            help="Порівняння покаже обидва стани одночасно"
        )
        
        # Загальні параметри
        st.subheader("Загальні параметри")
        field_type = st.selectbox(
            "Тип електричного поля:",
            ["Статичне", "Лінійне", "Синусоїдальне"]
        )
        
        E0 = st.slider("Напруженість поля E₀ (В/м)", 0.1, 10.0, 1.0, 0.1)
        j0 = st.slider("Початковий струм j₀ (А/м²)", 0.0, 10.0, 0.0, 0.1)
        t_max = st.slider("Час моделювання (с)", 0.1, 10.0, 5.0, 0.1)
        
        # Параметри поля
        if field_type == "Лінійне":
            a = st.slider("Швидкість росту поля a", 0.1, 5.0, 1.0, 0.1)
        else:
            a = 1.0
            
        if field_type == "Синусоїдальне":
            omega = st.slider("Частота ω (рад/с)", 0.1, 10.0, 1.0, 0.1)
        else:
            omega = 1.0
        
        # Параметри станів
        st.subheader("Параметри станів")
        if comparison_mode == "Порівняння":
            T_common = st.slider("Температура для порівняння (K)", 0.1, 15.0, 4.2, 0.1)
            T_super = T_common
            T_normal = T_common
        else:
            selected_state = st.radio("Оберіть стан:", ["Надпровідник", "Звичайний метал"])
            if selected_state == "Надпровідник":
                T_super = st.slider("Температура надпровідника (K)", 0.1, Tc-0.1, 4.2, 0.1)
                T_normal = None
            else:
                T_normal = st.slider("Температура звичайного металу (K)", 0.1, 15.0, 4.2, 0.1)
                T_super = None
    
    # Основний контент
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Графіки струму")
        
        # Часова вісь
        t = np.linspace(0, t_max, 1000)
        
        if comparison_mode == "Порівняння":
            # Порівняльний режим
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=('Надпровідний стан', 'Звичайний стан'),
                              vertical_spacing=0.1)
            
            # Надпровідник
            j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0)
            fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник', 
                                   line=dict(color='red')), row=1, col=1)
            
            # Звичайний метал
            j_normal = calculate_normal_current(t, field_type, T_common, E0, a, omega, j0)
            fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний метал',
                                   line=dict(color='blue')), row=2, col=1)
            
            fig.update_xaxes(title_text="Час (с)", row=2, col=1)
            fig.update_yaxes(title_text="Густина струму (А/м²)", row=1, col=1)
            fig.update_yaxes(title_text="Густина струму (А/м²)", row=2, col=1)
            fig.update_layout(height=600, showlegend=True)
            
        else:
            # Один стан
            fig = go.Figure()
            
            if T_super is not None:
                j_super = calculate_superconducting_current(t, field_type, E0, a, omega, j0)
                fig.add_trace(go.Scatter(x=t, y=j_super, name='Надпровідник',
                                       line=dict(color='red', width=3)))
                title = "Надпровідний стан"
            else:
                j_normal = calculate_normal_current(t, field_type, T_normal, E0, a, omega, j0)
                fig.add_trace(go.Scatter(x=t, y=j_normal, name='Звичайний метал',
                                       line=dict(color='blue', width=3)))
                title = "Звичайний стан"
            
            fig.update_layout(title=title, xaxis_title="Час (с)", 
                            yaxis_title="Густина струму (А/м²)", height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("📊 Аналіз")
        
        # Інформація про параметри
        st.subheader("Параметри розрахунку")
        st.write(f"**Тип поля:** {field_type}")
        st.write(f"**E₀ =** {E0} В/м")
        st.write(f"**j₀ =** {j0} А/м²")
        
        if comparison_mode == "Порівняння":
            st.write(f"**Температура:** {T_common} K")
            if T_common < Tc:
                st.success("✅ Температура нижче Tкрит - можливий надпровідний стан")
            else:
                st.warning("⚠️ Температура вище Tкрит - тільки звичайний стан")
        else:
            if T_super is not None:
                st.write(f"**Температура надпровідника:** {T_super} K")
            else:
                st.write(f"**Температура металу:** {T_normal} K")
        
        # Додаткова візуалізація для синусоїдального поля
        if field_type == "Синусоїдальне" and comparison_mode == "Порівняння":
            st.subheader("📡 Аналіз частотної залежності")
            
            # Розрахунок амплітуд для різних частот
            frequencies = np.logspace(-1, 1, 50)  # 0.1 до 10 рад/с
            amplitudes_super = []
            amplitudes_normal = []
            
            for freq in frequencies:
                # Амплітуда для надпровідника
                K = (e**2 * n0) / m
                amp_super = (K * E0) / freq
                amplitudes_super.append(amp_super)
                
                # Амплітуда для звичайного металу
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
            fig_freq.update_layout(xaxis_title="Частота ω (рад/с)", 
                                 yaxis_title="Амплітуда струму (А/м²)",
                                 xaxis_type="log", yaxis_type="log",
                                 height=300)
            st.plotly_chart(fig_freq, use_container_width=True)

if __name__ == "__main__":
    main()
