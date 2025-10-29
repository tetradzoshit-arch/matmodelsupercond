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

# Параметри залежності часу релаксації (для нормального металу T >= T_C)
TAU_IMP = 5.0e-14        # Час релаксації при T->0 (розсіювання на домішках), с
A_PHONON = 3.0e8         # Коефіцієнт для розсіювання на фононах (T^5 залежність)

# --- ДОПОМІЖНІ ФУНКЦІЇ ---

def tau_temperature_dependence(T):
    """
    Розраховує час релаксації tau(T) для звичайного металу
    на основі правила Маттіссена: 1/tau = 1/tau_imp + 1/tau_phonon
    Використовується T^5 залежність для розсіювання на фононах (низькі T).
    """
    if T <= 0.1: # Захист від ділення на нуль і для T->0
        return TAU_IMP
        
    # Швидкість розсіювання (1/tau)
    scattering_rate = (1 / TAU_IMP) + (A_PHONON * T**5)
    
    # Час релаксації
    tau_T = 1.0 / scattering_rate
    return tau_T

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
    
    # Визначення моделі для нормального металу
    metal_model = None
    if is_superconductor:
        st.success(f"⚡ Надпровідний стан: T={T}K < T_c={T_C}K")
        # Розрахунок коефіцієнта Лондонів K
        N_S = N_0 * (1.0 - (T / T_C) ** 4.0) # Явно вказуємо float для степенів
        K_COEFF = (N_S * E_CHARGE**2) / M_ELECTRON
        st.metric("Коефіцієнт $K$", f"{K_COEFF:.2e} $A/(V \\cdot m \\cdot s)$")
    else:
        st.info(f"🔌 Звичайний метал: T={T}K $\\ge$ T_c={T_C}K")
        
        # --- НОВИЙ ВИБІР МОДЕЛІ ---
        metal_model = st.selectbox("Оберіть модель для металу:", 
                                   ["Модель Друде (з перехідним процесом)", 
                                    "Закон Ома (стаціонарний)"])

        # ВИКОРИСТАННЯ ТЕМПЕРАТУРНО ЗАЛЕЖНОГО TAU
        tau_T_current = tau_temperature_dependence(T)
        SIGMA_COEFF = (N_0 * E_CHARGE**2 * tau_T_current) / M_ELECTRON
        
        st.metric("Час релаксації $\\tau(T)$", f"{tau_T_current:.2e} с")
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
    
    # Параметри поля (всі явні числа є float)
    E_0, A, OMEGA = None, None, None # Ініціалізація
    if "Постійне" in field_type:
        E_0 = st.number_input("E₀ (В/м)", 0.0, 1e4, 1e3, 100.0)
    elif "Лінійне" in field_type:
        A = st.number_input("Швидкість зростання 'a' (В/(м·с))", 1e8, 1e12, 1e10, 1e9)
    else:  # Синусоїдальне
        E_0 = st.number_input("Амплітуда E₀ (В/м)", 0.0, 1e4, 1e3, 100.0)
        F = st.number_input("Частота f (Гц)", 1e6, 1e9, 1e7, 1e6)
        OMEGA = 2.0 * np.pi * F # Явно вказуємо float

    # --- КНОПКИ ДЛЯ УПРАВЛЕНИЯ ГРАФИКАМИ ---
    st.subheader("📈 Управління графіками") 
    if st.button("➕ Додати поточний графік"):
        # Розміри розрахунків
        if "Синусоїдальне" in field_type and not is_superconductor:
            # Для синусоїдального поля в металі - показуємо кілька періодів
            T_END = 3 * (2.0 * np.pi / OMEGA)  # 3 періоди
        else:
            T_END = 1e-9  # 1 наносекунда для інших випадків
    
    T_ARRAY = np.linspace(0.0, T_END, 1000)
    J_ARRAY = np.zeros_like(T_ARRAY)
    formula_label
        
        # --- РОЗРАХУНОК СТРУМУ ---
        if is_superconductor:
            # Рівняння Лондонів: dj/dt = K * E(t)
            # ФОРМУЛИ: j(t)=j_0+KE_0 t, j(t)=j_0+K (at^2)/2, j(t)=j_0+(KΕ_0)/ω(1-cos⁡(ωt))
            if "Постійне" in field_type:
                # 1. j(t)=j_0+KE_0 t
                J_ARRAY = J_0 + K_COEFF * E_0 * T_ARRAY
                formula_label = r'$j(t) = j_0 + K E_0 t$'
            elif "Лінійне" in field_type:
                # 2. j(t)=j_0+K (at^2)/2
                J_ARRAY = J_0 + (K_COEFF * A * T_ARRAY**2.0) / 2.0
                formula_label = r'$j(t) = j_0 + \frac{1}{2} K a t^2$'
            else:  # Синусоїдальне
                # 3. j(t)=j_0+(KΕ_0)/ω(1-cos⁡(ωt))
                J_ARRAY = J_0 + (K_COEFF * E_0 / OMEGA) * (1.0 - np.cos(OMEGA * T_ARRAY))
                formula_label = r'$j(t) = j_0 + \frac{K E_0}{\omega} (1 - \cos(\omega t))$'
        else:
            # ЗВИЧАЙНИЙ МЕТАЛ
            tau_T = tau_temperature_dependence(T) 
            sigma = (N_0 * E_CHARGE**2.0 * tau_T) / M_ELECTRON

            if metal_model == "Модель Друде (з перехідним процесом)":
                # Модель Друде: dj/dt + j/τ(T) = σ(T)/τ(T) * E(t)
                
                if "Постійне" in field_type:
                    # j(t)=j_0 e^((-t)/τ)+σΕτ(1-e^((-t)/τ))
                    J_ARRAY = J_0 * np.exp(-T_ARRAY / tau_T) + sigma * E_0 * tau_T * (1.0 - np.exp(-T_ARRAY / tau_T)) / tau_T
                    # Спрощення до оригінальної формули (з коду): j(t)=j_0 e^((-t)/τ)+σΕ_0 (1-e^((-t)/τ))
                    J_ARRAY = J_0 * np.exp(-T_ARRAY / tau_T) + sigma * E_0 * (1.0 - np.exp(-T_ARRAY / tau_T))
                    formula_label = r'$j(t) = j_0 e^{-t/\tau(T)} + \sigma(T) E_0 (1 - e^{-t/\tau(T)})$'
                elif "Лінійне" in field_type:
                    #формула для E(t)=at: j(t) = j_0 e^{-t/\tau} + \sigma a [t - \tau(1 - e^{-t/\tau})]
                    J_ARRAY = J_0 * np.exp(-T_ARRAY / tau_T) + sigma * A * (T_ARRAY - tau_T * (1.0 - np.exp(-T_ARRAY / tau_T)))
                    formula_label = r'$j(t) = j_0 e^{-t/\tau(T)} + \sigma(T) a [t - \tau(T)(1 - e^{-t/\tau(T)})]$'
                else:  # Синусоїдальне
                    # КОРЕКТНА формула для моделі Д
                    tau = tau_T
                    omega_tau_sq = (OMEGA * tau)**2.0
    
                    # Амплітуда і фаза стаціонарного режиму
                    amp_factor = sigma * tau / np.sqrt(1.0 + omega_tau_sq)  # ДОДАНО tau!
                    phase_shift = np.arctan(OMEGA * tau)
                    J_ST_CLASSIC = E_0 * amp_factor * np.sin(OMEGA * T_ARRAY - phase_shift)
    
                    # Перехідна складова
                    C = J_0 - E_0 * amp_factor * np.sin(-phase_shift)
                    J_TR_CLASSIC = C * np.exp(-T_ARRAY / tau)
    
                    J_ARRAY = J_TR_CLASSIC + J_ST_CLASSIC
    
                    formula_label = r'$j(t) = j_{\text{tr}}(t) + j_{\text{st}}(t)$'
                    
            else: # Закон Ома (стаціонарний)
                # Закон Ома: j(t) = σ(T) * E(t). j₀ ігнорується.
                if "Постійне" in field_type:
                    J_ARRAY = sigma * E_0 * np.ones_like(T_ARRAY)
                    formula_label = r'$j(t) = \sigma(T) E_0$'
                elif "Лінійне" in field_type:
                    J_ARRAY = sigma * A * T_ARRAY
                    formula_label = r'$j(t) = \sigma(T) a t$'
                else:  # Синусоїдальне
                    J_ARRAY = sigma * E_0 * np.sin(OMEGA * T_ARRAY)
                    formula_label = r'$j(t) = \sigma(T) E_0 \sin(\omega t)$'
            
        # Зберігаємо графік
        new_run = {
            'T': T,
            'T_array': T_ARRAY * 1e9,  # час в нс
            'J_array': J_ARRAY,
            'formula': formula_label,
            'state': "Надпровідник" if is_superconductor else "Метал",
            'model': "Лондони" if is_superconductor else metal_model,
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
        label = f"#{i+1}: T={run['T']}K ({run['model']}), Поле: {run['field_type'].split(':')[0]}"
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
        with st.expander(f"Графік #{i+1}: T={run['T']}K ({run['model']}) | Поле: {run['field_type'].split(':')[0]}"):
            st.latex(run['formula'])
            st.markdown(f"**Стан:** {run['state']}")
            st.markdown(f"**Модель:** {run['model']}")
            st.markdown(f"**Тип поля:** {run['field_type']}")
            st.metric("Максимальний струм", f"{np.max(run['J_array']):.2e} А/м²")
else:
    st.info("👆 Додайте перший графік, використовуючи кнопку в боковій панелі!")

# --- ИНФОРМАЦИЯ ---
with st.expander("ℹ️ Інструкція"):
    st.markdown("""
    **Як користуватися:**
    1. **Встановіть параметри** (температура, початковий струм, тип поля) в боковій панелі.
    2. Якщо $T \ge T_c$, оберіть модель: **Друде** (з перехідним процесом) або **Ома** (стаціонарний).
    3. Натисніть **\"➕ Додати поточний графік\"** для додавання результату на основний графік.
    4. **Змініть параметри** і додайте ще графіки для порівняння динаміки в різних режимах (надпровідник vs. метал).
    5. Видаліть графіки кнопкою **\"🗑️ Очистити всі графіки\"**.
    
    **Порада:** Порівняйте Модель Друде (яка має експоненціальне зростання) та Закон Ома (який миттєво досягає стаціонарного значення) при $T=15K$ і постійному полі.
    """)
# --- ИНФОРМАЦИЯ ---
with st.expander("ℹ️ Довідка"):
    st.markdown("""
    **Фізичні принципи:**
    - **Надпровідник** ($T < T_c$): Динаміка описується **Рівняннями Лондонів**. Струм може зростати лінійно або квадратично в часі, оскільки опір дорівнює нулю.
    - **Звичайний метал** ($T \ge T_c$):
        - **Модель Друде:** Враховує інерцію електронів та час релаксації $\\tau(T)$, що призводить до плавного (експоненційного) встановлення струму до стаціонарного значення $\\sigma(T)E_0$.
        - **Закон Ома:** Припускає, що струм встановлюється миттєво, тобто $j(t) = \sigma(T)E(t)$. Це наближення є коректним, коли час релаксації $\\tau(T)$ набагато менший за характерний час зміни зовнішнього поля.
    
    **Параметри за замовчуванням подібні до Ніобію ($\text{Nb}$).**
    """)
