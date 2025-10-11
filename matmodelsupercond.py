import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Моделювання струму", layout="wide")
st.title("🔬 МОДЕЛЮВАННЯ ДИНАМІКИ ГУСТИНИ СТРУМУ")
st.markdown("---")

# --- ФИЗИЧЕСКИЕ КОНСТАНТЫ ---
E_CHARGE = 1.6e-19       # Заряд електрона, Кл
M_ELECTRON = 9.1e-31     # Маса електрона, кг
N_0 = 1.0e29             # Загальна концентрація електронів, м⁻³
T_C = 9.2                # Критична температура (для Nb), К
TAU = 2e-14              # Час релаксації (для нормального стану), с

# --- ДОПОМІЖНІ ФУНКЦІЇ ---

# Функція для часу релаксації. Наразі використовуємо константу TAU.
def tau_temperature_dependence(T):
    """Повертає час релаксації tau. В простій моделі вважаємо константою."""
    # Тут можна додати реальну температурну залежність (наприклад, T^-3 для розсіювання на фононах)
    return TAU

# --- ИНИЦИАЛИЗАЦИЯ СЕССИИ ---
if 'simulation_runs' not in st.session_state:
    st.session_state.simulation_runs = []

# --- САЙДБАР ДЛЯ ВВОДА ПАРАМЕТРОВ ---
with st.sidebar:
    st.header("🎛️ Параметри моделювання")
    
    # Вибір температури
    T = st.slider("🌡️ Температура T (K)", 0.1, 20.0, 4.2, 0.1)
    
    # Определение состояния
    is_superconductor = (T < T_C)
    if is_superconductor:
        st.success(f"⚡ Надпровідний стан: T={T}K < T_c={T_C}K")
        # Розрахунок коефіцієнта Лондонів K
        N_S = N_0 * (1 - (T / T_C) ** 4)
        K_COEFF = (N_S * E_CHARGE**2) / M_ELECTRON
        st.metric("Коефіцієнт $K$", f"{K_COEFF:.2e} $A/(V \cdot m \cdot s)$")
    else:
        st.info(f"🔌 Звичайний метал: T={T}K $\\ge$ T_c={T_C}K")
        # Розрахунок провідності Друде σ (використовуючи константу TAU)
        SIGMA_COEFF = (N_0 * E_CHARGE**2 * TAU) / M_ELECTRON
        st.metric("Провідність $\\sigma$", f"{SIGMA_COEFF:.2e} См/м")
    
    # Начальный ток
    J_0 = st.number_input("➡️ Початкова густина струму $j_0$ (А/м²)", 
                          min_value=0.0, max_value=1e11, value=0.0, step=1e6)
    
    # Тип поля
    st.subheader("📊 Тип зовнішнього поля")
    field_type = st.selectbox("Оберіть тип поля:", 
                              ["Постійне поле: E(t) = E₀", 
                               "Лінійне поле: E(t) = a · t", 
                               "Синусоїдальне: E(t) = E₀ · sin(ωt)"])
    
    # Параметри поля
    if "Постійне" in field_type:
        E_0 = st.number_input("E₀ (В/м)", 0.0, 1e4, 1e3, 100.0)
        A = None # Забезпечуємо, що інші параметри не визначені
        OMEGA = None
    elif "Лінійне" in field_type:
        A = st.number_input("Швидкість зростання 'a' (В/(м·с))", 1e8, 1e12, 1e10, 1e9)
        E_0 = None
        OMEGA = None
    else:  # Синусоїдальне
        E_0 = st.number_input("Амплітуда E₀ (В/м)", 0.0, 1e4, 1e3, 100.0)
        F = st.number_input("Частота f (Гц)", 1e6, 1e9, 1e7, 1e6)
        OMEGA = 2 * np.pi * F
        A = None

    # --- КНОПКИ ДЛЯ УПРАВЛЕНИЯ ГРАФИКАМИ ---
    st.subheader("📈 Управління графіками")
    
    if st.button("➕ Додати поточний графік"):
        # Розміри розрахунків
        T_END = 1e-9
        T_ARRAY = np.linspace(0, T_END, 1000)
        J_ARRAY = np.zeros_like(T_ARRAY)
        formula_label = ""
        
        # --- РОЗРАХУНОК СТРУМУ ---
        if is_superconductor:
            # Рівняння Лондонів: dj/dt = K * E(t)
            if "Постійне" in field_type:
                J_ARRAY = J_0 + K_COEFF * E_0 * T_ARRAY
                formula_label = r'$j(t) = j_0 + K E_0 t$'
            elif "Лінійне" in field_type:
                J_ARRAY = J_0 + (K_COEFF * A * T_ARRAY**2) / 2
                formula_label = r'$j(t) = j_0 + \frac{1}{2} K a t^2$'
            else:  # Синусоїдальне
                J_ARRAY = J_0 + (K_COEFF * E_0 / OMEGA) * (1 - np.cos(OMEGA * T_ARRAY))
                formula_label = r'$j(t) = j_0 + \frac{K E_0}{\omega} (1 - \cos(\omega t))$'
        else:
            # Модель Друде: dj/dt + j/τ = σ/τ * E(t)
            # !!! ПОМИЛКУ ВИПРАВЛЕНО: Додано функцію tau_temperature_dependence та виправлено відступи
            tau_T = tau_temperature_dependence(T) 
            sigma = (N_0 * E_CHARGE**2 * tau_T) / M_ELECTRON
            
            if "Постійне" in field_type:
                J_ARRAY = J_0 * np.exp(-T_ARRAY / tau_T) + sigma * E_0 * (1 - np.exp(-T_ARRAY / tau_T))
                formula_label = r'$j(t) = j_0 e^{-t/\tau} + \sigma E_0 (1 - e^{-t/\tau})$'
            elif "Лінійне" in field_type:
                J_ARRAY = J_0 * np.exp(-T_ARRAY / tau_T) + sigma * A * (T_ARRAY - tau_T * (1 - np.exp(-T_ARRAY / tau_T)))
                formula_label = r'$j(t) = j_0 e^{-t/\tau} + \sigma a [t - \tau(1 - e^{-t/\tau})]$'
            else:  # Синусоїдальне
                phase_shift = np.arctan(OMEGA * tau_T)
                amplitude_factor = sigma / np.sqrt(1 + (OMEGA * tau_T)**2)
                
                # Стаціонарний режим
                J_ST = E_0 * amplitude_factor * np.sin(OMEGA * T_ARRAY - phase_shift)
                
                # Перехідний процес (визначається j₀)
                C = J_0 - E_0 * amplitude_factor * np.sin(-phase_shift)
                J_TR = C * np.exp(-T_ARRAY / tau_T)
                
                J_ARRAY = J_TR + J_ST
                formula_label = r'$j(t) = j_{\text{tr}}(t) + j_{\text{st}}(t)$'
            
        # Зберігаємо графік
        new_run = {
            'T': T,
            'T_array': T_ARRAY * 1e9,  # час в нс
            'J_array': J_ARRAY,
            'formula': formula_label,
            'state': "Надпровідник" if is_superconductor else "Метал",
            'field_type': field_type
        }
        st.session_state.simulation_runs.append(new_run)
        st.success(f"✅ Графік #{len(st.session_state.simulation_runs)} додано!")
    
    if st.button("🗑️ Очистити всі графіки"):
        st.session_state.simulation_runs = []
        st.success("✅ Всі графіки очищено!")

# --- ВИЗУАЛИЗАЦИЯ ВСЕХ ГРАФИКОВ ---
st.subheader("📈 Порівняння графіків")

if st.session_state.simulation_runs:
    # Створення графіка
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Кольори для різних графіків
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Рисуємо всі графіки
    for i, run in enumerate(st.session_state.simulation_runs):
        color = colors[i % len(colors)]
        label = f"#{i+1}: T={run['T']}K ({run['state']}), Поле: {run['field_type'].split(':')[0]}"
        ax.plot(run['T_array'], run['J_array'], 
                color=color, linewidth=2.5, label=label)
    
    ax.set_xlabel('Час $t$ (нс)', fontsize=12)
    ax.set_ylabel('Густина струму $j$ (${\\text{A}}/{\\text{м}^2}$)', fontsize=12)
    ax.set_title('Порівняння динаміки густини струму', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.legend(loc='upper right', fontsize=10)
    
    # Показати графік в Streamlit
    st.pyplot(fig)
    
    # --- ИНФОРМАЦИЯ О ГРАФИКАХ ---
    st.subheader("📊 Інформація про графіки")
    for i, run in enumerate(st.session_state.simulation_runs):
        with st.expander(f"Графік #{i+1}: T={run['T']}K ({run['state']}) | Поле: {run['field_type'].split(':')[0]}"):
            st.latex(run['formula'])
            st.markdown(f"**Стан:** {run['state']}")
            st.markdown(f"**Тип поля:** {run['field_type']}")
            st.metric("Максимальний струм", f"{np.max(run['J_array']):.2e} А/м²")
else:
    st.info("👆 Додайте перший графік, використовуючи кнопку в боковій панелі!")

# --- ИНФОРМАЦИЯ ---
with st.expander("ℹ️ Інструкція"):
    st.markdown("""
    **Як користуватися:**
    1. **Встановіть параметри** (температура, початковий струм, тип поля) в боковій панелі.
    2. Натисніть **\"➕ Додати поточний графік\"** для додавання результату на основний графік.
    3. **Змініть параметри** і додайте ще графіки для порівняння динаміки в різних режимах (надпровідник vs. метал).
    4. Видаліть графіки кнопкою **\"🗑️ Очистити всі графіки\"**.
    
    **Порада:** Спробуйте порівняти $T=4.2K$ (надпровідник) та $T=10K$ (звичайний стан)!
    """)
# --- ИНФОРМАЦИЯ ---
with st.expander("ℹ️ Довідка"):
    st.markdown("""
    **Фізичні принципи:**
    - **Надпровідник** ($T < T_c$): Динаміка описується **Рівняннями Лондонів**. Струм може зростати лінійно або квадратично в часі, оскільки опір дорівнює нулю.
    - **Звичайний метал** ($T \ge T_c$): Динаміка описується **Моделлю Друде**. Струм виходить на стаціонарний рівень, обмежений часом релаксації $\\tau$.
    
    **Параметри за замовчуванням подібні до Ніобію ($\text{Nb}$), де $T_c \approx 9.2K$.**
    """)
