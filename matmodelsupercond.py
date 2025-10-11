import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Моделювання струму", layout="wide")
st.title("🔬 МОДЕЛЮВАННЯ ДИНАМІКИ ГУСТИНИ СТРУМУ")
st.markdown("---")

# --- ФИЗИЧЕСКИЕ КОНСТАНТЫ ---
E_CHARGE = 1.6e-19
M_ELECTRON = 9.1e-31
N_0 = 1.0e29
T_C = 9.2
TAU = 2e-14

# --- ИНИЦИАЛИЗАЦИЯ СЕССИИ ---
if 'simulation_runs' not in st.session_state:
    st.session_state.simulation_runs = []

# --- САЙДБАР ДЛЯ ВВОДА ПАРАМЕТРОВ ---
with st.sidebar:
    st.header("🎛 Параметри моделювання")
    
    # Выбор температуры
    T = st.slider("🌡 Температура T (K)", 0.1, 20.0, 4.2, 0.1)
    
    # Определение состояния
    is_superconductor = (T < T_C)
    if is_superconductor:
        st.success(f"⚡️ Надпровідний стан: T={T}K < T_c={T_C}K")
        N_S = N_0 * (1 - (T / T_C)  4)
        K_COEFF = (N_S * E_CHARGE**2) / M_ELECTRON
        st.metric("Константа електронного відгуку K", f"{K_COEFF:.2e}")
    else:
        st.info(f"🔌 Звичайний метал: T={T}K ≥ T_c={T_C}K")
        SIGMA_COEFF = (N_0 * E_CHARGE**2 * TAU) / M_ELECTRON
        st.metric("Провідність Друде σ", f"{SIGMA_COEFF:.2e} См/м")
    
    # Начальный ток
    J_0 = st.number_input("➡️ Початкова густина струму j₀ (А/м²)", 
                         min_value=0.0, max_value=1e11, value=0.0, step=1e6)
    
    # Тип поля
    st.subheader("📊 Тип зовнішнього поля")
    field_type = st.selectbox("Оберіть тип поля:", 
                             ["Постійне поле: E(t) = E₀", 
                              "Лінійне поле: E(t) = a · t", 
                              "Синусоїдальне: E(t) = E₀ · sin(ωt)"])
    
    # Параметры поля
    if "Постійне" in field_type:
        E_0 = st.number_input("E₀ (В/м)", 0.0, 1e4, 1e3, 100.0)
    elif "Лінійне" in field_type:
        A = st.number_input("Швидкість зростання 'a' (В/(м·с))", 1e8, 1e12, 1e10, 1e9)
    else:  # Синусоидальное
        E_0 = st.number_input("Амплітуда E₀ (В/м)", 0.0, 1e4, 1e3, 100.0)
        F = st.number_input("Частота f (Гц)", 1e6, 1e9, 1e7, 1e6)
        OMEGA = 2 * np.pi * F

    # --- КНОПКИ ДЛЯ УПРАВЛЕНИЯ ГРАФИКАМИ ---
    st.subheader("📈 Управління графіками")
    
    if st.button("➕ Додати поточний графік"):
        # Расчеты для текущих параметров
        T_END = 1e-9
        T_ARRAY = np.linspace(0, T_END, 1000)
        J_ARRAY = np.zeros_like(T_ARRAY)
        
        # Расчет тока (твой код)
        if is_superconductor:
            if "Постійне" in field_type:
                J_ARRAY = J_0 + K_COEFF * E_0 * T_ARRAY
                formula_label = r'$j(t) = j_0 + K E_0 t$'
            elif "Лінійне" in field_type:
                J_ARRAY = J_0 + (K_COEFF * A * T_ARRAY**2) / 2
                formula_label = r'$j(t) = j_0 + \frac{1}{2} K a t^2$'
            else:  # Синусоидальное
                J_ARRAY = J_0 + (K_COEFF * E_0 / OMEGA) * (1 - np.cos(OMEGA * T_ARRAY))
                formula_label = r'$j(t) = j_0 + \frac{K E_0}{\omega} (1 - \cos(\omega t))$'
        else:
         tau_T = tau_temperature_dependence(T)
        sigma = (N_0 * E_CHARGE**2 * tau_T) / M_ELECTRON
        if "Постійне" in field_type:
        J_ARRAY = J_0 * np.exp(-T_ARRAY / tau_T) + sigma * E_0 * (1 - np.exp(-T_ARRAY / tau_T))
        formula_label = r'$j(t) = j_0 e^{-t/\tau(T)} + \sigma(T) E_0 (1 - e^{-t/\tau(T)})$'
    elif "Лінійне" in field_type:
        J_ARRAY = J_0 * np.exp(-T_ARRAY / tau_T) + sigma * A * (T_ARRAY - tau_T * (1 - np.exp(-T_ARRAY / tau_T)))
        formula_label = r'$j(t) = j_0 e^{-t/\tau(T)} + \sigma(T) a [t - \tau(T)(1 - e^{-t/\tau(T)})]$'
    else:  # Синусоїдальне
        phase_shift = np.arctan(OMEGA * tau_T)
        amplitude_factor = sigma / np.sqrt(1 + (OMEGA * tau_T)**2)
        J_ST = E_0 * amplitude_factor * np.sin(OMEGA * T_ARRAY - phase_shift)
        C = J_0 - E_0 * amplitude_factor * np.sin(-phase_shift)
        J_TR = C * np.exp(-T_ARRAY / tau_T)
        J_ARRAY = J_TR + J_ST
        formula_label = r'$j(t) = j_{\text{tr}}(t) + j_{\text{st}}(t)$'
        
        # Сохраняем график
        new_run = {
            'T': T,
            'T_array': T_ARRAY * 1e9,  # время в нс
            'J_array': J_ARRAY,
            'formula': formula_label,
            'state': "Надпровідник" if is_superconductor else "Метал",
            'field_type': field_type
        }
        st.session_state.simulation_runs.append(new_run)
        st.success(f"✅ Графік #{len(st.session_state.simulation_runs)} додано!")
    
    if st.button("🗑 Очистити всі графіки"):
        st.session_state.simulation_runs = []
        st.success("✅ Всі графіки очищено!")

# --- ВИЗУАЛИЗАЦИЯ ВСЕХ ГРАФИКОВ ---
st.subheader("📈 Порівняння графіків")

if st.session_state.simulation_runs:
    # Создание графика
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Цвета для разных графиков
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Рисуем все графики
    for i, run in enumerate(st.session_state.simulation_runs):
        color = colors[i % len(colors)]
        label = f"#{i+1}: T={run['T']}K ({run['state']})"
        ax.plot(run['T_array'], run['J_array'], 
                color=color, linewidth=2.5, label=label)
    
    ax.set_xlabel('Час $t$ (нс)', fontsize=12)
    ax.set_ylabel('Густина струму $j$ (${\\text{A}}/{\\text{м}^2}$)', fontsize=12)
    ax.set_title('Порівняння динаміки густини струму', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.legend(loc='upper right')
    
    # Показать график в Streamlit
    st.pyplot(fig)
    
    # --- ИНФОРМАЦИЯ О ГРАФИКАХ ---
    st.subheader("📊 Інформація про графіки")
    for i, run in enumerate(st.session_state.simulation_runs):
        with st.expander(f"Графік #{i+1}: T={run['T']}K ({run['state']})"):
            st.latex(run['formula'])
            st.metric("Максимальний струм", f"{np.max(run['J_array']):.2e} А/м²")
else:
    st.info("👆 Додайте перший графік, використовуючи кнопку в боковій панелі!")

# --- ИНФОРМАЦИЯ ---
with st.expander("ℹ️ Інструкція"):
    st.markdown("""
    **Як користуватися:
    1. Встановіть параметри в боковій панелі
    2. Натисніть \"➕ Додати поточний графік\"
    3. Змініть параметри і додайте ще графіки для порівняння
    4. Видаліть графіки кнопкою \"🗑 Очистити всі графіки\"
    
    Порада: Спробуйте порівняти T=4K (надпровідник) та T=10K (звичайний стан)!
    """)
# --- ИНФОРМАЦИЯ ---
with st.expander("ℹ️ Довідка"):
    st.markdown("""
    Фізичні принципи:
    - Надпровідник: Рівняння Лондонів - струм росте без опору.
    - Звичайний метал: Модель Друде - струм виходить на стаціонарний рівень.
    
    Параметри за замовчуванням подібні до Ніобію (Nb).  T=4.2К - температура кипіння рідкого Гелію.
    """)
