import ipywidgets as widgets
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from IPython.display import display, Audio
from pathlib import Path
from scipy.io import wavfile

def reproductor_interactivo(ruta_audio, ruta_csv='conteo_disparos.csv', etiquetas_reales_csv=None):
    """
    Crea una interfaz gráfica e interactiva en Jupyter para auditar disparos.

    Genera un lienzo dinámico usando Plotly (FigureWidget) sincronizado con
    controles de ipywidgets. Permite:
     - Visualizar la forma de onda reducida de manera adaptativa
     - Inspeccionar marcadores existentes haciendo clic en ellos
     - Escuchar el audio circundante a los marcadaores con márgenes configurables 
     - Eliminar puntos, alternar etiquetas ("posible_disparo" / "otro")
     - Crear nuevos marcadores encontrando automáticamente el pico de amplitud 
       máxima dentro de una ventana de tiempo seleccionada por arrastre del cursor. 
       * El pico de la ventana seleccionada se marca con una línea azul punteada.
     - Visualizar disparos reales si se proporciona un CSV de etiquetas reales.
     - Inicializar la detección de disparos de forma manual si no se cuenta con
       un CSV de detección previamente.

    Cualquier cambio muta y guarda el CSV de manera inmediata.

    Args:
        ruta_audio (str o Path): Ubicación del archivo de audio WAV a auditar.
        ruta_csv (str o Path): Ubicación del reporte CSV con las clasificaciones.

    Returns:
        None: Despliega la interfaz interactiva mediante controles en pantalla.
    """
    fs, audio = wavfile.read(ruta_audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
        
    duracion = len(audio) / fs
    tiempo = np.linspace(0, duracion, num=len(audio))
    
    # Reducción adaptativa para optimizar el rendimiento gráfico (máximo 100k puntos)
    max_puntos = 100000
    if len(audio) > max_puntos:
        factor = len(audio) // max_puntos
        audio_grafico = audio[::factor]
        tiempo_grafico = tiempo[::factor]
    else:
        audio_grafico = audio
        tiempo_grafico = tiempo

    # Control de existencia del CSV y creación si no existe o está vacío
    ruta_csv = Path(ruta_csv)
    if not ruta_csv.exists() or ruta_csv.stat().st_size == 0:
        with open(ruta_csv, "w", encoding="utf-8") as f:
            f.write("tiempo_segundos,puntaje,clasificacion\n")

    df_global = pd.read_csv(ruta_csv)
    fig = go.FigureWidget()

    # Capa 0: Onda de audio de fondo
    fig.add_trace(go.Scatter(
        x=tiempo_grafico, y=audio_grafico, mode='lines', name='Onda de Audio',
        line=dict(color='rgba(180, 180, 180, 0.4)', width=1), hoverinfo='skip'
    ))

    # Capa 1: Marcadores de eventos clasificados como "Otros"
    fig.add_trace(go.Scatter(
        x=[], y=[], mode='markers', name='Otros (Ruido)',
        marker=dict(color='orange', size=8, symbol='circle-open', line=dict(width=2)),
        hovertemplate='Tiempo: %{x:.3f} s<br>Puntaje: %{text}<extra></extra>'
    ))

    # Capa 2: Marcadores de eventos clasificados como "Posibles Disparos"
    fig.add_trace(go.Scatter(
        x=[], y=[], mode='markers', name='Posibles Disparos',
        marker=dict(color='crimson', size=11, symbol='x', line=dict(width=2)),
        hovertemplate='<b>¡Disparo!</b><br>Tiempo: %{x:.3f} s<br>Puntaje: %{text}<extra></extra>'
    ))

    def actualizar_puntos_grafico():
        """Refresca la posición de los marcadores en el lienzo de Plotly."""
        otros = df_global[df_global['clasificacion'] == 'otro']
        disparos = df_global[df_global['clasificacion'] == 'posible_disparo']

        with fig.batch_update():
            fig.data[1].x = otros['tiempo_segundos']
            fig.data[1].y = np.zeros(len(otros))
            
            fig.data[2].x = disparos['tiempo_segundos']
            fig.data[2].y = np.zeros(len(disparos))

    actualizar_puntos_grafico()

    # Marcadores de disparos reales si se proporciona un CSV de etiquetas reales
    traza_reales = None
    if etiquetas_reales_csv is not None:
        df_reales = pd.read_csv(etiquetas_reales_csv)
        tiempos_reales = df_reales.iloc[:, 0].values
        
        # Guardo la traza de los disparos reales en una variable para poder escucharlos
        # con el reproductor al hacer clic sobre ellos.
        traza_reales = go.Scatter(
            x=tiempos_reales, 
            y=np.zeros(len(tiempos_reales)), 
            mode='markers', 
            name='Disparos Efectuados (Reales)',
            marker=dict(
                color='gold',
                size=12,
                symbol='star',
                line=dict(color='black', width=1)
            ),
            hovertemplate='<b>Etiqueta Real</b><br>Tiempo: %{x:.3f} s<extra></extra>'
        )
        fig.add_trace(traza_reales)

    fig.update_layout(
        title=dict(
            text="Reproductor con Visualización de Disparos",
            x=0.5, y=0.97, xanchor='center', yanchor='top',
            font=dict(size=20, color="#0003A7", weight='bold')
        ),
        xaxis_title="Tiempo (segundos)", 
        yaxis_title="Amplitud",
        template="plotly_white",
        xaxis=dict(
            range=[-0.125, duracion * 0.965],
            rangeslider=dict(visible=True),
            type="linear"
        ),
        height=400, width=1200, 
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=True,
        legend=dict(
            x=0.02, y=0.98, xanchor='left', yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.7)', bordercolor='rgba(200, 200, 200, 0.5)', borderwidth=1
        ),
        dragmode='select'
    )

    txt_seleccion = widgets.Label(
        value="👉 Hacé clic en un marcador ❌ para reproducir su audio o arrastrá el cursor mientras hacés click para reproducir un fragmento de audio",
        style=dict(font_size='12pt', font_weight='bold')
    )
    
    sld_antes = widgets.IntSlider(value=200, min=0, max=2000, step=10, description='t antes:', readout=False, layout=widgets.Layout(width='280px'))
    sld_despues = widgets.IntSlider(value=500, min=0, max=4000, step=10, description='t después:', readout=False, layout=widgets.Layout(width='280px'))

    num_antes = widgets.Text(value=str(sld_antes.value), layout=widgets.Layout(width='47px'))
    num_despues = widgets.Text(value=str(sld_despues.value), layout=widgets.Layout(width='47px'))
    num_antes.style.text_align = 'center'
    num_despues.style.text_align = 'center'
    
    widgets.dlink((sld_antes, 'value'), (num_antes, 'value'), transform=str)
    widgets.dlink((sld_despues, 'value'), (num_despues, 'value'), transform=str)
    
    def texto_a_int(val):
        try: return int(val)
        except ValueError: return 0

    widgets.dlink((num_antes, 'value'), (sld_antes, 'value'), transform=texto_a_int)
    widgets.dlink((num_despues, 'value'), (sld_despues, 'value'), transform=texto_a_int)
    
    lbl_antes_ms = widgets.Label(value="ms", layout=widgets.Layout(margin='0 0 0 4px'))
    lbl_despues_ms = widgets.Label(value="ms", layout=widgets.Layout(margin='0 0 0 4px'))
    
    output_reproductor = widgets.Output()

    btn_cambiar_clase = widgets.Button(description="🔄 Cambiar Etiqueta", button_style='info', icon='exchange-alt', layout=widgets.Layout(width='180px'))
    btn_eliminar_punto = widgets.Button(description="🗑️ Eliminar Punto", button_style='danger', icon='trash-alt', layout=widgets.Layout(width='160px'))
    
    btn_guardar_disparo = widgets.Button(description="🎯 Guardar como Disparo", button_style='success', icon='check', layout=widgets.Layout(width='200px'))
    btn_guardar_otro = widgets.Button(description="⚠️ Guardar como Otro", button_style='warning', icon='exclamation', layout=widgets.Layout(width='180px'))

    box_edicion = widgets.HBox([btn_cambiar_clase, btn_eliminar_punto], layout=widgets.Layout(justify_content='center', margin='5px 0 5px 0'))
    box_creacion = widgets.HBox([btn_guardar_disparo, btn_guardar_otro], layout=widgets.Layout(justify_content='center', margin='5px 0 5px 0'))

    bloque_antes = widgets.HBox([sld_antes, num_antes, lbl_antes_ms], layout=widgets.Layout(align_items='center', margin='8px 15px 15px 15px'))
    bloque_despues = widgets.HBox([sld_despues, num_despues, lbl_despues_ms], layout=widgets.Layout(align_items='center', margin='8px 15px 15px 15px'))
    
    contenedor_sliders = widgets.HBox(
        [bloque_antes, bloque_despues],
        layout=widgets.Layout(justify_content='center', align_items='center')
    )
    
    contenedor_sliders.layout.display = 'none' 
    output_reproductor.layout.display = 'none'
    box_edicion.layout.display = 'none'
    box_creacion.layout.display = 'none'

    estado = {"tiempo_centro": None, "modo": None}

    def actualizar_reproductor():
        """Aísla y reproduce el segmento de audio centrado en la posición activa."""
        if estado["tiempo_centro"] is None:
            return
            
        with output_reproductor:
            output_reproductor.clear_output(wait=True) 
            
            if sld_antes.value == 0 and sld_despues.value == 0:
                txt_seleccion.value = "⚠️ Incrementa los milisegundos para poder generar un fragmento de audio audible."
                return  
            
            if estado["modo"] == "edit":
                txt_seleccion.value = f"🔊 Auditando marcador en la posición: {estado['tiempo_centro']:.3f} s"
            else:
                txt_seleccion.value = f"➕ Pico detectado en {estado['tiempo_centro']:.3f} s | 💡 Haz DOBLE CLIC en una zona vacía para salir o limpiar la caja"
            
            muestras_antes = int((sld_antes.value / 1000) * fs)
            muestras_despues = int((sld_despues.value / 1000) * fs)
            muestras_centro = int(estado["tiempo_centro"] * fs)
            
            idx_inicio = max(0, muestras_centro - muestras_antes)
            idx_fin = min(len(audio), muestras_centro + muestras_despues)
            
            fragmento = audio[idx_inicio:idx_fin]
            if len(fragmento) == 0:
                return
                
            display(Audio(fragmento, rate=fs, autoplay=True))

    def al_cambiar_slider(change):
        if change['name'] == 'value' and estado["tiempo_centro"] is not None:
            actualizar_reproductor()

    sld_antes.observe(al_cambiar_slider)
    sld_despues.observe(al_cambiar_slider)

    def al_hacer_clic(trace, points, state_click):
        """Manejador de eventos para la selección de marcadores existentes."""
        if not points.xs:
            return
            
        fig.layout.shapes = []
        
        with fig.batch_update():
            fig.data[0].selectedpoints = None
            fig.data[1].selectedpoints = None
            fig.data[2].selectedpoints = None
            if len(fig.data) > 3:
                fig.data[3].selectedpoints = None
            trace.selectedpoints = points.point_inds

        contenedor_sliders.layout.display = 'flex'
        output_reproductor.layout.display = 'flex'
        box_edicion.layout.display = 'flex'
        box_creacion.layout.display = 'none'
            
        estado["tiempo_centro"] = float(points.xs[0])
        estado["modo"] = "edit"
        
        actualizar_reproductor()

    fig.data[1].on_click(al_hacer_clic)
    fig.data[2].on_click(al_hacer_clic)
    if traza_reales is not None:
        fig.data[3].on_click(al_hacer_clic)  # Disparos Reales


    def al_seleccionar_zona(trace, points, state_select):
        """Manejador para la selección de nuevas regiones temporales."""
        if state_select is None or state_select.xrange is None:
            return
            
        with fig.batch_update():
            fig.data[1].selectedpoints = None
            fig.data[2].selectedpoints = None

        t_min, t_max = state_select.xrange
        idx_min = int(max(0, t_min * fs))
        idx_max = int(min(len(audio), t_max * fs))

        if idx_min >= idx_max:
            return

        # Algoritmo del pico máximo local absoluto dentro de la ventana delimitada
        fragmento_zona = np.abs(audio[idx_min:idx_max])
        if len(fragmento_zona) == 0:
            return
            
        idx_pico_relativo = np.argmax(fragmento_zona)
        idx_pico_absoluto = idx_min + idx_pico_relativo

        estado["tiempo_centro"] = float(idx_pico_absoluto / fs)
        estado["modo"] = "create"

        # Seteo adaptativo de los márgenes del audio según los límites de la caja
        ms_antes_calculado = int(max(0, (estado["tiempo_centro"] - t_min) * 1000))
        ms_despues_calculado = int(max(0, (t_max - estado["tiempo_centro"]) * 1000))

        sld_antes.value = ms_antes_calculado
        sld_despues.value = ms_despues_calculado

        # Dibujo de la línea guía vertical sobre el pico acústico detectado
        amplitud_maxima_local = float(np.max(np.abs(audio_grafico)))
        linea_temporal = go.layout.Shape(
            type="line",
            x0=estado["tiempo_centro"], x1=estado["tiempo_centro"],
            y0=-amplitud_maxima_local, y1=amplitud_maxima_local,
            line=dict(color="rgba(0, 3, 167, 0.85)", width=2, dash="dash")
        )
        with fig.batch_update():
            fig.layout.shapes = [linea_temporal]

        contenedor_sliders.layout.display = 'flex'
        output_reproductor.layout.display = 'flex'
        box_creacion.layout.display = 'flex'
        box_edicion.layout.display = 'none'

        actualizar_reproductor()

    def al_hacer_clic_libre(trace, points=None):
        """Limpia el estado de selección de la interfaz gráfica."""
        fig.layout.shapes = []
        with fig.batch_update():
            fig.data[0].selectedpoints = None
            fig.data[1].selectedpoints = None
            fig.data[2].selectedpoints = None
            if len(fig.data) > 3:
                fig.data[3].selectedpoints = None

        estado["tiempo_centro"] = None
        estado["modo"] = None

        contenedor_sliders.layout.display = 'none'
        output_reproductor.layout.display = 'none'
        box_creacion.layout.display = 'none'
        box_edicion.layout.display = 'none'
        txt_seleccion.value = "👉 Hacé clic en un marcador ❌ para reproducir su audio o arrastrá el cursor mientras hacés click para reproducir ese fragmento de audio"

    fig.data[0].on_selection(al_seleccionar_zona)
    fig.data[0].on_deselect(al_hacer_clic_libre)

    def reetiquetar_marcador(b):
        """Intercambia la clasificación del marcador activo en el archivo CSV."""
        idx = df_global[np.abs(df_global['tiempo_segundos'] - estado["tiempo_centro"]) < 0.001].index
        if not idx.empty:
            clase_nueva = "otro" if df_global.loc[idx, 'clasificacion'].values[0] == "posible_disparo" else "posible_disparo"
            df_global.loc[idx, 'clasificacion'] = clase_nueva
            df_global.to_csv(ruta_csv, index=False)
            txt_seleccion.value = f"✅ Modificado con éxito: {estado['tiempo_centro']:.3f} s cambiado a '{clase_nueva}'."
            
            with fig.batch_update():
                fig.data[1].selectedpoints = None
                fig.data[2].selectedpoints = None
                
            actualizar_puntos_grafico()

    def eliminar_marcador(b):
        """Remueve de manera permanente el marcador activo del archivo CSV."""
        nonlocal df_global
        idx = df_global[np.abs(df_global['tiempo_segundos'] - estado["tiempo_centro"]) < 0.001].index
        if not idx.empty:
            df_global = df_global.drop(idx).reset_index(drop=True)
            df_global.to_csv(ruta_csv, index=False)
            txt_seleccion.value = f"🗑️ El punto {estado['tiempo_centro']:.3f} s fue removido permanentemente del CSV."
            estado["tiempo_centro"] = None
            box_edicion.layout.display = 'none'
            
            with fig.batch_update():
                fig.data[1].selectedpoints = None
                fig.data[2].selectedpoints = None
                
            actualizar_puntos_grafico()

    def agregar_nuevo_punto(clase_nueva):
        """Consolida e inyecta un nuevo marcador verificado en el CSV global."""
        nonlocal df_global
        nueva_fila = pd.DataFrame([{
            'tiempo_segundos': round(estado["tiempo_centro"], 3),
            'puntaje': 1.0000,
            'clasificacion': clase_nueva
        }])
        df_global = pd.concat([df_global, nueva_fila]).sort_values(by='tiempo_segundos').reset_index(drop=True)
        df_global.to_csv(ruta_csv, index=False)
        txt_seleccion.value = f"✨ Registrado nuevo marcador en el segundo {estado['tiempo_centro']:.3f} s como '{clase_nueva}'."
        
        fig.layout.shapes = []
        with fig.batch_update():
            fig.data[0].selectedpoints = None
            
        box_creacion.layout.display = 'none'
        actualizar_puntos_grafico()

    btn_cambiar_clase.on_click(reetiquetar_marcador)
    btn_eliminar_punto.on_click(eliminar_marcador)
    btn_guardar_disparo.on_click(lambda b: agregar_nuevo_punto("posible_disparo"))
    btn_guardar_otro.on_click(lambda b: agregar_nuevo_punto("otro"))

    controles = widgets.VBox(
        [txt_seleccion, contenedor_sliders, output_reproductor, box_edicion, box_creacion],
        layout=widgets.Layout(width='1200px', align_items='center', justify_content='center')
    )
    
    display(fig)
    display(controles)


def analisis_librosa(times, onset_env, onset_peaks, onset_detect, audio, fs=16000, etiquetas_reales=None, graficar=False):
    """
    Muestra estadísticas del detector Librosa y opcionalmente grafica las curvas.

    Imprime en consola los resultados del conteo de transitorios acústicos calculados 
    por Librosa contrastados con las etiquetas reales de control de laboratorio. 
    Si `graficar=True`, genera una visualización estática en dos subplots: la curva de 
    novedad espectral con los máximos locales aislados, y la forma de onda del archivo 
    junto con líneas de referencia discontinuas para mapear los aciertos visuales.

    Args:
        times (np.ndarray): Vector de tiempos en segundos para los cuadros espectrales.
        onset_env (np.ndarray): Vector unidimensional con la curva de fuerza de inicio.
        onset_peaks (np.ndarray): Máscara booleana o índices de los picos máximos locales.
        onset_detect (np.ndarray): Índices de los cuadros que superaron el umbral de disparo.
        audio (np.ndarray): Vector de muestras del audio normalizado.
        fs (int, opcional): Frecuencia de muestreo de la señal. Por defecto es 16000.
        etiquetas_reales (pd.Series o np.ndarray, opcional): Tiempos cronológicos de los
            disparos reales efectuados para realizar la validación.
        graficar (bool, opcional): Indica si debe renderizarse el gráfico de Matplotlib.
            Por defecto es False.

    Returns:
        None: Imprime reportes numéricos en consola y genera plots si se solicita.
    """
    print("--- Conteo de disparos con librosa ---")
    print("Cantidad de disparos detectados:", len(onset_detect))
    if etiquetas_reales is not None:
        print("Cantidad de disparos efectuados:", len(etiquetas_reales))
    else:
        print("No se proporcionaron etiquetas reales.")

    if graficar:
        _, ax = plt.subplots(nrows=2, sharex=True, height_ratios=(3, 1))
        
        librosa.display.waveshow(y=audio, sr=fs, ax=ax[1], label="Waveform")
        ax[1].legend()
        ax[0].plot(times, onset_env, label="Onset envelope", color="C1")
        ax[0].scatter(
            times[onset_peaks],
            onset_env[onset_peaks],
            marker="^",
            color="k",
            label="Localmax Peaks",
        )
        ax[0].scatter(
            times[onset_detect],
            onset_env[onset_detect],
            marker="o",
            edgecolor="C2",
            facecolor="none",
            label="Disparos detectados",
        )
        if etiquetas_reales is not None:
            ax[0].vlines(
                etiquetas_reales,
                ymin=0,
                ymax=max(onset_env[onset_peaks]),
                color='red',
                linestyle='dashed',
                linewidth=1,
                label='Disparos efectuados'
            )
        ax[0].legend()
        ax[0].label_outer()
        plt.show()


def comparador_ondas_disparos(ruta_audio, ruta_csv="conteo_disparos.csv", t_antes_ms=50, t_despues_ms=200):
    """Muestra una cuadrícula vertical paginada de 10 disparos simultáneos con controles compactos.

    Permite evaluar de un solo vistazo las firmas acústicas de 10 eventos alineados 
    en el tiempo relativo de su pico (0 ms). Incluye paginación de 10 en 10 (estilo Gmail) 
    y botones compactos alineados rígidamente al lado de cada señal de audio.

    Args:
        ruta_audio (str o Path): Ubicación del archivo de audio WAV original.
        ruta_csv (str o Path, opcional): Ubicación del reporte CSV con las clasificaciones. 
            Por defecto es "conteo_disparos.csv".
        t_antes_ms (int, opcional): Milisegundos a extraer antes del pico. Por defecto 50.
        t_despues_ms (int, opcional): Milisegundos a extraer después del pico. Por defecto 200.
    """
    from plotly.subplots import make_subplots
    from pathlib import Path

    # 1. Carga inicial de datos
    fs, audio = wavfile.read(ruta_audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    ruta_csv = Path(ruta_csv)
    if not ruta_csv.exists():
        print(f"❌ Error: No se encontró el archivo {ruta_csv}")
        return

    df_global = pd.read_csv(ruta_csv)
    total_eventos = len(df_global)
    if total_eventos == 0:
        print("⚠️ El archivo CSV de detecciones está vacío.")
        return

    muestras_antes = int((t_antes_ms / 1000) * fs)
    muestras_despues = int((t_despues_ms / 1000) * fs)

    # Estado de la paginación y salida de audio oculta en segundo plano
    estado_paginacion = {"pagina_actual": 0}
    output_audio_oculto = widgets.Output()

    # Contenedores principales de la UI superior
    output_interfaz = widgets.Output()
    lbl_paginacion = widgets.Label(value="")
    btn_prev = widgets.Button(description="", icon="chevron-left", layout=widgets.Layout(width='40px', height='30px'))
    btn_next = widgets.Button(description="", icon="chevron-right", layout=widgets.Layout(width='40px', height='30px'))

    def renderizar_pagina():
        """Dibuja dinámicamente el bloque de 5 disparos correspondiente a la página activa."""
        idx_start = estado_paginacion["pagina_actual"] * 5
        idx_end = min(total_eventos, idx_start + 5)
        eventos_visibles = idx_end - idx_start

        # Actualizar texto de navegación estilo Gmail
        lbl_paginacion.value = f"Mostrando los disparos etiquetados: {idx_start + 1}-{idx_end} de {total_eventos}"
        
        # Habilitar/Deshabilitar flechas según los límites físicos
        btn_prev.disabled = (estado_paginacion["pagina_actual"] == 0)
        btn_next.disabled = (idx_end >= total_eventos)

        with output_interfaz:
            output_interfaz.clear_output(wait=True)

            # Inicializar subplots sin títulos de subtramas
            # Aumentamos el vertical_spacing para dar espacio físico real entre señales
            fig = go.FigureWidget(make_subplots(
                rows=eventos_visibles, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.035 
            ),)

            lista_controles_filas = []

            for i, idx_global in enumerate(range(idx_start, idx_end)):
                n_fila = i + 1
                row = df_global.iloc[idx_global]
                t_centro = row['tiempo_segundos']
                clase_inicial = row['clasificacion']

                idx_centro = int(t_centro * fs)
                start = max(0, idx_centro - muestras_antes)
                end = min(len(audio), idx_centro + muestras_despues)
                segmento = audio[start:end]
                tiempo_relativo_ms = ((np.arange(start, end) / fs) - t_centro) * 1000

                fig.add_trace(
                    go.Scatter(x=tiempo_relativo_ms, y=segmento, mode='lines', 
                               line=dict(color='#0003A7', width=1.2), showlegend=False),
                    row=n_fila, col=1
                )
                
                fig.add_shape(type="line", x0=0, x1=0, y0=float(np.min(segmento)), y1=float(np.max(segmento)),
                              line=dict(color="rgba(230,0,0,0.5)", width=1, dash="dash"), row=n_fila, col=1)

                fig.update_yaxes(visible=False, row=n_fila, col=1)
                
                if n_fila == eventos_visibles:
                    fig.update_xaxes(showticklabels=True, title_text="Tiempo relativo al pico (ms)", row=n_fila, col=1)
                else:
                    fig.update_xaxes(showticklabels=False, row=n_fila, col=1)

                # --- 🛠️ SOLUCIÓN: BOTÓN DE AUDIO ULTRA COMPACTO (Sustituye al reproductor pesado) ---
                btn_play_local = widgets.Button(description="Escuchar", icon="play", layout=widgets.Layout(width='120px', height='28px'))
                
                # Inyección dinámica de reproducción al hacer clic
                def crear_callback_audio(seg_datos):
                    def callback_audio(b):
                        with output_audio_oculto:
                            output_audio_oculto.clear_output(wait=True)
                            display(Audio(seg_datos, rate=fs, autoplay=True))
                    return callback_audio
                
                btn_play_local.on_click(crear_callback_audio(segmento))

                # Botón de mutación optimizado en tamaño
                btn_mutar = widgets.Button(description="", layout=widgets.Layout(width='95px', height='28px'))
                
                def configurar_boton(boton, clase):
                    if clase == "posible_disparo":
                        boton.description = "Es Ruido"
                        boton.button_style = 'warning'
                    else:
                        boton.description = "Es Disparo"
                        boton.button_style = 'success'

                configurar_boton(btn_mutar, clase_inicial)

                def crear_callback_csv(b_instancia, idx_p_csv):
                    def callback_clic(b):
                        clase_actual = df_global.loc[idx_p_csv, 'clasificacion']
                        clase_nueva = "otro" if clase_actual == "posible_disparo" else "posible_disparo"
                        df_global.loc[idx_p_csv, 'clasificacion'] = clase_nueva
                        df_global.to_csv(ruta_csv, index=False)
                        configurar_boton(b_instancia, clase_nueva)
                    return callback_clic

                btn_mutar.on_click(crear_callback_csv(btn_mutar, idx_global))

                # Fila de control limpia de desbordamientos e inmune a comportamientos del navegador
                caja_fila = widgets.HBox(
                    [btn_play_local, btn_mutar], 
                    layout=widgets.Layout(align_items='center', justify_content='flex-start', gap='4px', height='54px', margin='0px')
                )
                lista_controles_filas.append(caja_fila)

            # Ajustes globales del lienzo Plotly: ancho a 650px para dar espacio físico real a la botonera
            # El alto por fila se estira a 48px para emparejar perfectamente la separación vertical
            fig.update_layout(
                template="plotly_white",
                height=48 * eventos_visibles + 70, 
                width=450,  
                margin=dict(l=10, r=10, t=15, b=45)
            )

            # Ensamblar los subplots y el panel rígido sin barras de scroll ocultas
            panel_derecho = widgets.VBox(lista_controles_filas, layout=widgets.Layout(justify_content='flex-start', gap='0px', padding='0px 0 0 5px'))
            interfaz_grafica = widgets.HBox([fig, panel_derecho], layout=widgets.Layout(align_items='flex-start', margin='0px'))
            display(interfaz_grafica)

    # 4. Manejadores de clics de la barra de navegación Gmail
    def ir_atras(b):
        if estado_paginacion["pagina_actual"] > 0:
            estado_paginacion["pagina_actual"] -= 1
            renderizar_pagina()

    def ir_adelante(b):
        if (estado_paginacion["pagina_actual"] + 1) * 5 < total_eventos:
            estado_paginacion["pagina_actual"] += 1
            renderizar_pagina()

    btn_prev.on_click(ir_atras)
    btn_next.on_click(ir_adelante)

    # 5. Construcción de la barra de herramientas superior
    barra_navegacion = widgets.HBox(
        [btn_prev, btn_next, lbl_paginacion], 
        layout=widgets.Layout(align_items='center', gap='14px', margin='5px 0 10px 15px')
    )
    
    # 🛠️ SOLUCIÓN: Envolvemos el output de audio en un contenedor HBox centrado horizontalmente
    audio_centrado = widgets.HBox(
        [output_audio_oculto], 
        layout=widgets.Layout(width='100%', justify_content='flex-start', margin='15px 0 0 75px')
    )
    
    # Despliegue de la estructura completa en Jupyter con el emisor centrado
    display(widgets.VBox([barra_navegacion, output_interfaz, audio_centrado]))
    renderizar_pagina()
