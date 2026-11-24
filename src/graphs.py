import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, Audio
from scipy.io import wavfile

def reproductor_interactivo(ruta_audio, ruta_csv):
    """
    Crea un gráfico interactivo de Plotly integrado con widgets de Jupyter.
    Permite reproducir fragmentos de audio, alternar etiquetas, eliminar puntos
    existentes del CSV o arrastrar una caja de selección para detectar el pico 
    máximo de volumen, dibujar una guía visual y reproducir el fragmento de la caja.
    """
    # 1. Cargar el audio original completo
    fs, audio = wavfile.read(ruta_audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
        
    duracion = len(audio) / fs
    tiempo = np.linspace(0, duracion, num=len(audio))
    
    # Reducción adaptativa para visualización rápida
    max_puntos = 100000
    if len(audio) > max_puntos:
        factor = len(audio) // max_puntos
        audio_grafico = audio[::factor]
        tiempo_grafico = tiempo[::factor]
    else:
        audio_grafico = audio
        tiempo_grafico = tiempo

    # 2. Cargar datos del CSV original
    df_global = pd.read_csv(ruta_csv)

    # 3. Construir la figura de Plotly (FigureWidget para interactuar)
    fig = go.FigureWidget()

    # Capa 0: Añadir onda de audio de fondo
    fig.add_trace(go.Scatter(
        x=tiempo_grafico, y=audio_grafico, mode='lines', name='Onda de Audio',
        line=dict(color='rgba(180, 180, 180, 0.4)', width=1), hoverinfo='skip'
    ))

    # Capa 1: Añadir marcadores de "Otros"
    fig.add_trace(go.Scatter(
        x=[], y=[], mode='markers', name='Otros (Ruido)',
        marker=dict(color='orange', size=8, symbol='circle-open', line=dict(width=2)),
        hovertemplate='Tiempo: %{x:.3f} s<br>Puntaje: %{text}<extra></extra>'
    ))

    # Capa 2: Añadir marcadores de "Posibles Disparos"
    fig.add_trace(go.Scatter(
        x=[], y=[], mode='markers', name='Posibles Disparos',
        marker=dict(color='crimson', size=11, symbol='x', line=dict(width=2)),
        hovertemplate='<b>¡Disparo!</b><br>Tiempo: %{x:.3f} s<br>Puntaje: %{text}<extra></extra>'
    ))

    # Función interna para refrescar los marcadores en Y=0 sin alterar el audio
    def actualizar_puntos_grafico():
        otros = df_global[df_global['clasificacion'] == 'otro']
        disparos = df_global[df_global['clasificacion'] == 'posible_disparo']
        
        with fig.batch_update():
            fig.data[1].x = otros['tiempo_segundos']
            fig.data[1].y = np.zeros(len(otros))
            fig.data[1].text = [f"{p:.4f}" for p in otros['puntaje']]
            
            fig.data[2].x = disparos['tiempo_segundos']
            fig.data[2].y = np.zeros(len(disparos))
            fig.data[2].text = [f"{p:.4f}" for p in disparos['puntaje']]

    actualizar_puntos_grafico()

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

    # 5. Crear los Widgets de la interfaz de usuario (UI)
    txt_seleccion = widgets.Label(
        value="👉 Haz clic en cualquier marcador del gráfico para cargar su audio o arrastra una caja de selección",
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

    # --- BOTONES DE ACCIÓN PARA EDICIÓN Y CREACIÓN ---
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

    # Estado de la sesión
    estado = {"tiempo_centro": None, "modo": None}

    # 6. Función para actualizar y renderizar el reproductor de audio
    def actualizar_reproductor():
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
                txt_seleccion.value = f"➕ Nuevo punto en {estado['tiempo_centro']:.3f} s | 💡 Haz DOBLE CLIC en una zona vacía para salir o limpiar la caja"
            
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

    # 7. Lógica de Interacción al hacer clic en marcadores existentes ('edit')
    def al_hacer_clic(trace, points, state_click):
        if not points.xs:
            return
            
        # 1. Limpieza absoluta y forzada de la línea guía azul del pico anterior
        fig.layout.shapes = []
        
        # 2. Actualización de las capas de selección para remover la caja gris de fondo
        with fig.batch_update():
            fig.data[0].selectedpoints = None  # Limpia la caja sobre la onda de audio
            fig.data[1].selectedpoints = None  # Limpia focos en otros marcadores
            fig.data[2].selectedpoints = None  # Limpia focos en disparos
            
            # Forzar el resalte visual únicamente del marcador que se acaba de cliquear
            trace.selectedpoints = points.point_inds

        # 3. Desplegar los componentes gráficos correspondientes en la interfaz
        contenedor_sliders.layout.display = 'flex'
        output_reproductor.layout.display = 'flex'
        box_edicion.layout.display = 'flex'
        box_creacion.layout.display = 'none'
            
        estado["tiempo_centro"] = float(points.xs[0])
        estado["modo"] = "edit"
        
        actualizar_reproductor()

    fig.data[1].on_click(al_hacer_clic)
    fig.data[2].on_click(al_hacer_clic)

    # 8. Lógica de Selección de áreas vacías en el lienzo de audio ('create')
    def al_seleccionar_zona(trace, points, state_select):
        if state_select is None or state_select.xrange is None:
            return
            
        # Al arrastrar una caja nueva: removemos los resaltes de selección previos 
        # en las capas de marcadores existentes para limpiar la vista.
        with fig.batch_update():
            fig.data[1].selectedpoints = None
            fig.data[2].selectedpoints = None
            
        # Extraer el rango de tiempo seleccionado por la caja
        t_min, t_max = state_select.xrange
        
        # Mapear el rango de tiempo a muestras de audio reales
        idx_min = int(max(0, t_min * fs))
        idx_max = int(min(len(audio), t_max * fs))
        
        if idx_min >= idx_max:
            return

        # --- ALGORITMO DE PICO MÁXIMO ---
        fragmento_zona = np.abs(audio[idx_min:idx_max])
        if len(fragmento_zona) == 0:
            return
        idx_pico_relativo = np.argmax(fragmento_zona)
        idx_pico_absoluto = idx_min + idx_pico_relativo
        
        # Convertimos la posición exacta del pico detectado a segundos
        estado["tiempo_centro"] = float(idx_pico_absoluto / fs)
        estado["modo"] = "create"
        
        # --- REPRODUCCIÓN ADAPTATIVA DEL CONTENIDO DE LA CAJA ---
        ms_antes_calculado = int(max(0, (estado["tiempo_centro"] - t_min) * 1000))
        ms_despues_calculado = int(max(0, (t_max - estado["tiempo_centro"]) * 1000))
        
        sld_antes.value = ms_antes_calculado
        sld_despues.value = ms_despues_calculado
        
        # --- MARCADOR TEMPORAL (LÍNEA VERTICAL DISCONTINUA) ---
        amplitud_maxima_local = float(np.max(np.abs(audio_grafico)))
        linea_temporal = go.layout.Shape(
            type="line",
            x0=estado["tiempo_centro"], x1=estado["tiempo_centro"],
            y0=-amplitud_maxima_local, y1=amplitud_maxima_local,
            line=dict(color="rgba(0, 3, 167, 0.85)", width=2, dash="dash")
        )
        with fig.batch_update():
            fig.layout.shapes = [linea_temporal]
            
        # Desplegar los componentes gráficos correspondientes
        contenedor_sliders.layout.display = 'flex'
        output_reproductor.layout.display = 'flex'
        box_creacion.layout.display = 'flex'
        box_edicion.layout.display = 'none'
        
        actualizar_reproductor()

    # --- NUEVA FUNCIÓN: LIMPIEZA CON UN SOLO CLIC ---
    def al_hacer_clic_libre(trace, points=None):
        # Limpiamos las líneas guía del layout primero de forma atómica
        fig.layout.shapes = []
        
        # Luego limpiamos todas las selecciones y cajas grises de las capas
        with fig.batch_update():
            fig.data[0].selectedpoints = None
            fig.data[1].selectedpoints = None
            fig.data[2].selectedpoints = None
            
        # Restaurar variables de estado y ocultar ipywidgets
        estado["tiempo_centro"] = None
        estado["modo"] = None
        
        contenedor_sliders.layout.display = 'none'
        output_reproductor.layout.display = 'none'
        box_creacion.layout.display = 'none'
        box_edicion.layout.display = 'none'
        txt_seleccion.value = "👉 Haz clic en cualquier marcador del gráfico para cargar su audio o arrastra una caja de selección"

    # Vinculamos la escucha de selección y deselección a la Capa 0 (Onda de audio)
    fig.data[0].on_selection(al_seleccionar_zona)
    fig.data[0].on_deselect(al_hacer_clic_libre)

    # 9. Lógica de Modificación de Datos en el archivo CSV
    def reetiquetar_marcador(b):
        idx = df_global[np.abs(df_global['tiempo_segundos'] - estado["tiempo_centro"]) < 0.001].index
        if not idx.empty:
            clase_nueva = "otro" if df_global.loc[idx, 'clasificacion'].values[0] == "posible_disparo" else "posible_disparo"
            df_global.loc[idx, 'clasificacion'] = clase_nueva
            df_global.to_csv(ruta_csv, index=False)
            txt_seleccion.value = f"✅ Modificado con éxito: {estado['tiempo_centro']:.3f} s cambiado a '{clase_nueva}'."
            
            # Limpiar selección visual después de mutar el elemento
            with fig.batch_update():
                fig.data[1].selectedpoints = None
                fig.data[2].selectedpoints = None
                
            actualizar_puntos_grafico()

    def eliminar_marcador(b):
        nonlocal df_global
        idx = df_global[np.abs(df_global['tiempo_segundos'] - estado["tiempo_centro"]) < 0.001].index
        if not idx.empty:
            df_global = df_global.drop(idx).reset_index(drop=True)
            df_global.to_csv(ruta_csv, index=False)
            txt_seleccion.value = f"🗑️ El punto {estado['tiempo_centro']:.3f} s fue removido permanentemente del CSV."
            estado["tiempo_centro"] = None
            box_edicion.layout.display = 'none'
            
            # Limpiar selección visual tras remover
            with fig.batch_update():
                fig.data[1].selectedpoints = None
                fig.data[2].selectedpoints = None
                
            actualizar_puntos_grafico()

    def agregar_nuevo_punto(clase_nueva):
        nonlocal df_global
        nueva_fila = pd.DataFrame([{
            'tiempo_segundos': round(estado["tiempo_centro"], 3),
            'puntaje': 1.0000,
            'clasificacion': clase_nueva
        }])
        df_global = pd.concat([df_global, nueva_fila]).sort_values(by='tiempo_segundos').reset_index(drop=True)
        df_global.to_csv(ruta_csv, index=False)
        txt_seleccion.value = f"✨ Registrado nuevo marcador en el segundo {estado['tiempo_centro']:.3f} s como '{clase_nueva}'."
        
        # Limpiar la línea guía temporal y las cajas de selección una vez consolidado el guardado
        fig.layout.shapes = []
        with fig.batch_update():
            fig.data[0].selectedpoints = None
            
        box_creacion.layout.display = 'none'
        actualizar_puntos_grafico()

    btn_cambiar_clase.on_click(reetiquetar_marcador)
    btn_eliminar_punto.on_click(eliminar_marcador)
    btn_guardar_disparo.on_click(lambda b: agregar_nuevo_punto("posible_disparo"))
    btn_guardar_otro.on_click(lambda b: agregar_nuevo_punto("otro"))

    # 10. Disponer la interfaz organizada por bloques centrados
    controles = widgets.VBox(
        [txt_seleccion, contenedor_sliders, output_reproductor, box_edicion, box_creacion],
        layout=widgets.Layout(width='1200px', align_items='center', justify_content='center')
    )
    
    display(fig)
    display(controles)

