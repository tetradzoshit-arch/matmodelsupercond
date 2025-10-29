import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- ФІЗИЧНІ КОНСТАНТИ ДЛЯ НІОБІЮ ---
E_CHARGE = 1.6e-19
M_ELECTRON = 9.1e-31
N_0 = 5.0e28
T_C = 9.2
TAU_IMP = 5.0e-13
A_PHONON = 1.0e7

# --- ДОПОМІЖНІ ФУНКЦІЇ ---
def tau_temperature_dependence(T):
    if T <= 0.1:
        return TAU_IMP
    scattering_rate = (1 / TAU_IMP) + (A_PHONON * T**5)
    tau_T = 1.0 / scattering_rate
    return tau_T

def find_peaks_simple(signal, prominence=0.1):
    peaks = []
    max_val = np.max(signal)
    threshold = prominence * max_val
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            if signal[i] > threshold:
                peaks.append(i)
    return np.array(peaks)

def analyze_current_direct(t_array, j_array, field_type, model_name, is_superconductor):
    analysis = {}
    
    # Основні статистики
    analysis['max_current'] = np.max(j_array)
    analysis['min_current'] = np.min(j_array)
    analysis['mean_current'] = np.mean(j_array)
    analysis['final_current'] = j_array[-1]
    analysis['time_to_max'] = t_array[np.argmax(j_array)]
    analysis['dynamic_range'] = analysis['max_current'] - analysis['min_current']
    analysis['overshoot'] = analysis['max_current'] - analysis['final_current'] if analysis['max_current'] > analysis['final_current'] else 0
    
    # Ініціалізація
    analysis['peaks_count'] = 0
    analysis['amplitude'] = 0
    analysis['frequency_mhz'] = 0
    analysis['settling_time'] = 0
    analysis['quadratic_coeff'] = 0
    analysis['final_growth_rate'] = 0
    
    if "Синусоїдальне" in field_type:
        peaks = find_peaks_simple(j_array, prominence=0.1)
        valleys = find_peaks_simple(-j_array, prominence=0.1)
        analysis['peaks_count'] = len(peaks)
        
        if len(peaks) >= 2:
            periods = np.diff(t_array[peaks])
            period_mean = np.mean(periods)
            analysis['frequency_mhz'] = 1.0 / (period_mean * 1e-9) / 1e6 if period_mean > 0 else 0
            
            if len(valleys) > 0:
                analysis['amplitude'] = (np.max(j_array[peaks]) - np.min(j_array[valleys])) / 2
            else:
                analysis['amplitude'] = np.max(j_array[peaks])
                
    elif "Постійне" in field_type:
        final_val = j_array[-1]
        if final_val != 0:
            settling_threshold = 0.02 * abs(final_val)
            for i in range(len(j_array)-1, 0, -1):
                if abs(j_array[i] - final_val) > settling_threshold:
                    analysis['settling_time'] = t_array[i+1] if i+1 < len(t_array) else t_array[-1]
                    break
    
    elif "Лінійне" in field_type:
        if is_superconductor and len(t_array) > 1 and np.max(t_array) > 0:
            t_mid = t_array[len(t_array)//2]
            j_mid = j_array[len(j_array)//2]
            if t_mid > 0:
                analysis['quadratic_coeff'] = (j_mid - j_array[0]) / (t_mid**2)
        elif len(j_array) > 10:
            last_slope = (j_array[-1] - j_array[-10]) / (t_array[-1] - t_array[-10])
            analysis['final_growth_rate'] = last_slope
    
    return analysis

def create_comparison_table(simulation_runs):
    table_data = []
    for i, run in enumerate(simulation_runs):
        is_superconductor = run['state'] == "Надпровідник"
        analysis = analyze_current_direct(run['T_array'], run['J_array'], run['field_type'], run['model'], is_superconductor)
        
        row = {
            '№': i + 1,
            'Температура': f"{run['T']} K",
            'Стан': run['state'],
            'Модель': run['model'],
            'Поле': run['field_type'].split(':')[0],
            'Макс. струм': f"{analysis['max_current']:.2e} А/м²",
            'Кінц. струм': f"{analysis['final_current']:.2e} А/м²",
            'Час до макс.': f"{analysis['time_to_max']:.1f} нс",
            'Динам. діапазон': f"{analysis['dynamic_range']:.2e} А/м²",
        }
        
        if "Синусоїдальне" in run['field_type']:
            row['Амплітуда'] = f"{analysis['amplitude']:.2e} А/м²"
            row['Частота'] = f"{analysis['frequency_mhz']:.1f} МГц"
            row['Періоди'] = f"{analysis['peaks_count']}"
        elif "Постійне" in run['field_type']:
            row['Час встановлення'] = f"{analysis['settling_time']:.1f} нс"
            row['Перерегулювання'] = f"{analysis['overshoot']:.2e} А/м²" if analysis['overshoot'] > 0 else "Немає"
        elif "Лінійне" in run['field_type']:
            if is_superconductor:
                row['Квадрат. коеф.'] = f"{analysis['quadratic_coeff']:.2e}"
            else:
                row['Швидкість зрост.'] = f"{analysis['final_growth_rate']:.2e}"
        
        table_data.append(row)
    
    return pd.DataFrame(table_data)

# --- ОСНОВНИЙ КОД ---
st.set_page_config(page_title="Моделювання струму", layout="wide")
st.title("🔬 МОДЕЛЮВАННЯ ДИНАМІКИ ГУСТИНИ СТРУМУ")
st.markdown("---")

if 'simulation_runs' not in st.session_state:
    st.session_state.simulation_runs = []

# --- САЙДБАР ---
with st.sidebar:
    st.header("🎛️ Параметри моделювання")
    
    T = st.slider("🌡️ Температура T (K)", 0.1, 20.0, 4.2, 0.1)
    is_superconductor = (T < T_C)
    
    if is_superconductor:
        st.success(f"⚡ Надпровідний стан: T={T}K < T_c={T_C}K")
        N_S = N_0 * (1.0 - (T / T_C) ** 4.0)
        K_COEFF = (N_S * E_CHARGE**2) / M_ELECTRON
        st.metric("Коефіцієнт $K$", f"{K_COEFF:.2e} $A/(V \\cdot m \\cdot s)$")
    else:
        st.info(f"🔌 Звичайний метал: T={T}K $\\ge$ T_c={T_C}K")
        metal_model = st.selectbox("Оберіть модель для металу:", 
                                   ["Модель Друде (з перехідним процесом)", "Закон Ома (стаціонарний)"])
        tau_T_current = tau_temperature_dependence(T)
        SIGMA_COEFF = (N_0 * E_CHARGE**2 * tau_T_current) / M_ELECTRON
        st.metric("Час релаксації $\\tau(T)$", f"{tau_T_current:.2e} с")
        st.metric("Провідність $\\sigma$", f"{SIGMA_COEFF:.2e} См/м")
    
    J_0 = st.number_input("➡️ Початкова густина струму $j_0$ (А/м²)", min_value=0.0, max_value=1e11, value=0.0, step=1e6)
    
    st.subheader("📊 Тип зовнішнього поля")
    field_type = st.selectbox("Оберіть тип поля:", 
                              ["Постійне поле: E(t) = E₀", "Лінійне поле: E(t) = a · t", "Синусоїдальне: E(t) = E₀ · sin(ωt)"])
    
    E_0, A, OMEGA = None, None, None
    if "Постійне" in field_type:
        E_0 = st.number_input("E₀ (В/м)", 0.0, 1e4, 5000.0, 100.0)
    elif "Лінійне" in field_type:
        A = st.number_input("Швидкість зростання 'a' (В/(м·с))", 1e8, 1e12, 1e10, 1e9)
    else:
        E_0 = st.number_input("Амплітуда E₀ (В/м)", 0.0, 1e4, 5000.0, 100.0)
        F = st.number_input("Частота f (кГц)", 1, 10000, 100, 10)
        OMEGA = 2.0 * np.pi * F * 1000

    st.subheader("⏰ Часовий інтервал")
    TIME_END = st.selectbox("Час моделювання", ["1 нс", "10 нс", "100 нс", "1 мкс", "10 мкс", "100 мкс", "1 мс"], index=4)
    time_dict = {"1 нс": 1e-9, "10 нс": 10e-9, "100 нс": 100e-9, "1 мкс": 1e-6, "10 мкс": 10e-6, "100 мкс": 100e-6, "1 мс": 1e-3}
    T_END_UNIFIED = time_dict[TIME_END]

    st.subheader("📈 Управління графіками")
    
    if st.button("➕ Додати поточний графік"):
        T_ARRAY = np.linspace(0.0, T_END_UNIFIED, 1000)
        J_ARRAY = np.zeros_like(T_ARRAY)
        formula_label = ""
        
        if is_superconductor:
            if "Постійне" in field_type:
                J_ARRAY = J_0 + K_COEFF * E_0 * T_ARRAY
                formula_label = r'$j(t) = j_0 + K E_0 t$'
            elif "Лінійне" in field_type:
                J_ARRAY = J_0 + (K_COEFF * A * T_ARRAY**2.0) / 2.0
                formula_label = r'$j(t) = j_0 + \frac{1}{2} K a t^2$'
            else:
                J_ARRAY = J_0 + (K_COEFF * E_0 / OMEGA) * (1.0 - np.cos(OMEGA * T_ARRAY))
                formula_label = r'$j(t) = j_0 + \frac{K E_0}{\omega} (1 - \cos(\omega t))$'
        else:
            tau_T = tau_temperature_dependence(T)
            sigma = (N_0 * E_CHARGE**2.0 * tau_T) / M_ELECTRON

            if metal_model == "Модель Друде (з перехідним процесом)":
                if "Постійне" in field_type:
                    J_ARRAY = J_0 * np.exp(-T_ARRAY / tau_T) + sigma * E_0 * (1.0 - np.exp(-T_ARRAY / tau_T))
                    formula_label = r'$j(t) = j_0 e^{-t/\tau(T)} + \sigma(T) E_0 (1 - e^{-t/\tau(T)})$'
                elif "Лінійне" in field_type:
                    J_ARRAY = J_0 * np.exp(-T_ARRAY / tau_T) + sigma * A * (T_ARRAY - tau_T * (1.0 - np.exp(-T_ARRAY / tau_T)))
                    formula_label = r'$j(t) = j_0 e^{-t/\tau(T)} + \sigma(T) a [t - \tau(T)(1 - e^{-t/\tau(T)})]$'
                else:
                    tau = tau_T
                    omega_tau_sq = (OMEGA * tau)**2.0
                    amp_factor = sigma * tau / np.sqrt(1.0 + omega_tau_sq)
                    phase_shift = np.arctan(OMEGA * tau)
                    J_ST_CLASSIC = E_0 * amp_factor * np.sin(OMEGA * T_ARRAY - phase_shift)
                    C = J_0 - E_0 * amp_factor * np.sin(-phase_shift)
                    J_TR_CLASSIC = C * np.exp(-T_ARRAY / tau)
                    J_ARRAY = J_TR_CLASSIC + J_ST_CLASSIC
                    formula_label = r'$j(t) = j_{\text{tr}}(t) + j_{\text{st}}(t)$'
            else:
                if "Постійне" in field_type:
                    J_ARRAY = sigma * E_0 * np.ones_like(T_ARRAY)
                    formula_label = r'$j(t) = \sigma(T) E_0$'
                elif "Лінійне" in field_type:
                    J_ARRAY = sigma * A * T_ARRAY
                    formula_label = r'$j(t) = \sigma(T) a t$'
                else:
                    J_ARRAY = sigma * E_0 * np.sin(OMEGA * T_ARRAY)
                    formula_label = r'$j(t) = \sigma(T) E_0 \sin(\omega t)$'
        
        new_run = {
            'T': T, 'T_array': T_ARRAY * 1e9, 'J_array': J_ARRAY, 'formula': formula_label,
            'state': "Надпровідник" if is_superconductor else "Метал",
            'model': "Лондони" if is_superconductor else metal_model,
            'field_type': field_type, 'time_scale': TIME_END
        }
        st.session_state.simulation_runs.append(new_run)
        st.success(f"✅ Графік #{len(st.session_state.simulation_runs)} додано!")
    
    if st.button("🗑️ Очистити всі графіки"):
        st.session_state.simulation_runs = []
        st.success("✅ Всі графіки очищено!")

# --- ВІЗУАЛІЗАЦІЯ ---
st.subheader("📈 Порівняння графіків")

if st.session_state.simulation_runs:
    time_scale = st.session_state.simulation_runs[0]['time_scale']
    time_unit = time_scale.split()[1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, run in enumerate(st.session_state.simulation_runs):
        color = colors[i % len(colors)]
        
        if time_unit == "мкс":
            time_array = run['T_array'] / 1000
        elif time_unit == "мс":
            time_array = run['T_array'] / 1000000
        else:
            time_array = run['T_array']
            
        label = f"#{i+1}: {run['model']}, {run['field_type'].split(':')[0]}, T={run['T']}K"
        ax.plot(time_array, run['J_array'], color=color, linewidth=2.5, label=label)
    
    ax.set_xlabel(f'Час $t$ ({time_unit})', fontsize=12)
    ax.set_ylabel('Густина струму $j$ (А/м²)', fontsize=12)
    ax.set_title(f'Порівняння динаміки струму ({time_scale})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', fontsize=9)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    st.pyplot(fig)

    # --- ТАБЛИЦЯ ---
    st.subheader("📊 Порівняльна таблиця графіків")
    comparison_df = create_comparison_table(st.session_state.simulation_runs)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    csv = comparison_df.to_csv(index=False, encoding='utf-8')
    st.download_button("📥 Експортувати таблицю в CSV", data=csv, file_name="порівняння_графіків.csv", mime="text/csv")

else:
    st.info("👆 Додайте перший графік, використовуючи кнопку в боковій панелі!")

with st.expander("ℹ️ Інструкція"):
    st.markdown("""
    **Рекомендації для синусоїдального поля:**
    - **Частота:** 1-100 кГц
    - **Амплітуда E₀:** 1000-10000 В/м  
    - **Час моделювання:** 10-100 мкс
    """)
    # Додай цей код на рядок 270
st.write("🔍 **ДЕБАГ:**")
tau_T_debug = tau_temperature_dependence(T)
sigma_debug = (N_0 * E_CHARGE**2.0 * tau_T_debug) / M_ELECTRON
st.write(f"tau_T = {tau_T_debug:.2e} с")
st.write(f"sigma = {sigma_debug:.2e} См/м")
st.write(f"E_0 = {E_0} В/м")
st.write(f"Очікуваний струм = {sigma_debug * E_0:.2e} А/м²")
st.write(f"Фактичний струм = {np.max(J_ARRAY):.2e} А/м²")

# Тільки для синусоїдального поля
if "Синусоїдальне" in field_type:
    st.write(f"OMEGA = {OMEGA:.2e} рад/с")
