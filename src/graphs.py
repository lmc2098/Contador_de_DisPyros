import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, Audio
from scipy.io import wavfile

def reproductor_interactivo(ruta_audio, ruta_csv):
    """
    Crea un gráfico interactivo de Plotly integrado con widgets de Jupyter.
    Permite hacer clic en cualquier marcador de disparo y reproducir un fragmento
    personalizado de audio alrededor de ese impacto.
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

    # 2. Cargar datos del CSV
    df = pd.read_csv(ruta_csv)
    posibles_disparos = df[df['clasificacion'] == 'posible_disparo'].copy()
    otros_candidatos = df[df['clasificacion'] == 'otro'].copy()

    # 3. Construir la figura de Plotly (Debe ser un FigureWidget para interactuar)
    fig = go.FigureWidget()

    # Añadir onda de audio de fondo
    fig.add_trace(go.Scatter(
        x=tiempo_grafico, y=audio_grafico, mode='lines', name='Onda de Audio',
        line=dict(color='rgba(180, 180, 180, 0.4)', width=1), hoverinfo='skip'
    ))

    # Añadir marcadores de "Otros"
    fig.add_trace(go.Scatter(
        x=otros_candidatos['tiempo_segundos'], y=np.zeros(len(otros_candidatos)),
        mode='markers', name='Otros (Ruido)',
        marker=dict(color='orange', size=8, symbol='circle-open', line=dict(width=2)),
        text=[f"Puntaje: {p:.4f}" for p in otros_candidatos['puntaje']],
        hovertemplate='Tiempo: %{x:.3f} s<br>%{text}<extra></extra>'
    ))

    # Añadir marcadores de "Posibles Disparos"
    fig.add_trace(go.Scatter(
        x=posibles_disparos['tiempo_segundos'], y=np.zeros(len(posibles_disparos)),
        mode='markers', name='Posibles Disparos',
        marker=dict(color='crimson', size=11, symbol='x', line=dict(width=2)),
        # text=[f"Puntaje: {p:.4f}" for p in posibles_disparos['puntaje']],
        hovertemplate='<b>¡Disparo!</b><br>Tiempo: %{x:.3f} s<br>%{text}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="Reproductor con Visualización de Disparos",
            x=0.5,                             # Centrado horizontalmente (0 = izquierda, 0.5 = centro, 1 = derecha)
            y=0.97,                            # Posición vertical (cerca del borde superior)
            xanchor='center',                  # El punto de anclaje horizontal es el centro del texto
            yanchor='top',                     # El punto de anclaje vertical es la parte superior del texto
            font=dict(size=20, color="#0003A7", weight='bold') # Opcional: cambia el tamaño y color de la fuente
            ),
        xaxis_title="Tiempo (segundos)", 
        yaxis_title="Amplitud",
        template="plotly_white",
        xaxis=dict(
            range=[-0.125, duracion*0.965],
            rangeslider=dict(visible=True)
            ),
        height=400, # Alto total de la figura en píxeles
        width=1200, # Ancho total de la figura en píxeles
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=True,
        legend=dict(
            x=0.02,                             # Coordenada X (0 = izquierda, 1 = derecha)
            y=0.98,                             # Coordenada Y (0 = abajo, 1 = arriba)
            xanchor='left',                     # Anclaje horizontal
            yanchor='top',                      # Anclaje vertical
            bgcolor='rgba(255, 255, 255, 0.7)', # Fondo blanco traslúcido para no obstruir la señal oculta
            bordercolor='rgba(200, 200, 200, 0.5)',
            borderwidth=1
        )
    )

    # 5. Crear los Widgets de la interfaz de usuario (UI)
    txt_seleccion = widgets.Label(
        value="👉 Haz clic en cualquier marcador del gráfico para cargar su audio",
        style=dict(font_size='12pt', font_weight='bold')
    )
    
    sld_antes = widgets.IntSlider(value=200, min=0, max=1000, step=10, description='t antes:', readout=False, layout=widgets.Layout(width='280px'))
    sld_despues = widgets.IntSlider(value=500, min=0, max=2000, step=10, description='t después:', readout=False, layout=widgets.Layout(width='280px'))

    # 2. Creamos campos de texto plano (widgets.Text) para evitar que el navegador dibuje las flechas
    # Ajustamos el ancho a '45px' para que quede bien compacto y centramos el texto adentro
    num_antes = widgets.Text(value=str(sld_antes.value), layout=widgets.Layout(width='47px'))
    num_despues = widgets.Text(value=str(sld_despues.value), layout=widgets.Layout(width='47px'))
    
    # Alineamos el texto al centro dentro de los recuadros
    num_antes.style.text_align = 'center'
    num_despues.style.text_align = 'center'
    
    # 3. Vinculación bidireccional usando dlink para convertir enteros a texto y viceversa
    # Del Slider al campo de texto (convierte el número a string)
    widgets.dlink((sld_antes, 'value'), (num_antes, 'value'), transform=str)
    widgets.dlink((sld_despues, 'value'), (num_despues, 'value'), transform=str)
    
    # Del campo de texto al Slider (convierte el string de vuelta a entero de forma segura)
    def texto_a_int(val):
        try:
            return int(val)
        except ValueError:
            return 0

    widgets.dlink((num_antes, 'value'), (sld_antes, 'value'), transform=texto_a_int)
    widgets.dlink((num_despues, 'value'), (sld_despues, 'value'), transform=texto_a_int)
    
    # Etiquetas para el sufijo "ms"
    lbl_antes_ms = widgets.Label(value="ms", layout=widgets.Layout(margin='0 0 0 4px'))
    lbl_despues_ms = widgets.Label(value="ms", layout=widgets.Layout(margin='0 0 0 4px'))
    
    output_reproductor = widgets.Output()

    # --- CONTENEDORES CON ALINEACIÓN CENTRADA ---
    bloque_antes = widgets.HBox([sld_antes, num_antes, lbl_antes_ms], layout=widgets.Layout(align_items='center', margin='8px 15px 15px 15px'))
    bloque_despues = widgets.HBox([sld_despues, num_despues, lbl_despues_ms], layout=widgets.Layout(align_items='center', margin='8px 15px 15px 15px'))
    
    contenedor_sliders = widgets.HBox(
        [bloque_antes, bloque_despues],
        layout=widgets.Layout(justify_content='center', align_items='center')
    )
    contenedor_sliders.layout.display = 'none' 
    output_reproductor.layout.display = 'none'

    # Guardamos el tiempo seleccionado en una lista para poder modificarlo internamente
    estado = {"tiempo_centro": None}

    # 6. Función para actualizar y renderizar el reproductor de audio
    def actualizar_reproductor():
        if estado["tiempo_centro"] is None:
            return
            
        with output_reproductor:
            output_reproductor.clear_output(wait=True) 
            
            # --- VALIDACIÓN PROTECTORA CONTRA TAMAÑO CERO ---
            if sld_antes.value == 0 and sld_despues.value == 0:
                txt_seleccion.value = "⚠️ Incrementa los milisegundos para poder generar un fragmento de audio audible."
                return  # Frenamos la ejecución antes de crear el objeto Audio vacío
            
            # Si hay tiempo válido, restauramos el mensaje estándar de escucha
            txt_seleccion.value = f"🔊 Escuchando marcador en la posición: {estado['tiempo_centro']:.3f} s"
            
            muestras_antes = int((sld_antes.value / 1000) * fs)
            muestras_despues = int((sld_despues.value / 1000) * fs)
            muestras_centro = int(estado["tiempo_centro"] * fs)
            
            idx_inicio = max(0, muestras_centro - muestras_antes)
            idx_fin = min(len(audio), muestras_centro + muestras_despues)
            
            fragmento = audio[idx_inicio:idx_fin]
            
            # Verificación extra de seguridad por si los índices colapsan en los bordes del archivo
            if len(fragmento) == 0:
                return
                
            display(Audio(fragmento, rate=fs, autoplay=True))

    # 7. Lógica de Interacción al cambiar los Sliders
    def al_cambiar_slider(change):
        if change['name'] == 'value' and estado["tiempo_centro"] is not None:
            actualizar_reproductor()

    sld_antes.observe(al_cambiar_slider)
    sld_despues.observe(al_cambiar_slider)

    # 8. Lógica de Interacción al hacer clic en los puntos del gráfico
    def al_hacer_clic(trace, points, state_click):
        if not points.xs:
            return
            
        # Revelar los controles dinámicamente si es el primer clic
        if estado["tiempo_centro"] is None:
            contenedor_sliders.layout.display = 'flex'
            output_reproductor.layout.display = 'flex'
            
        # CORRECCIÓN EXTRA: Extraemos explícitamente el primer elemento numérico de la lista 'xs'
        estado["tiempo_centro"] = float(points.xs[0])
        
        # Ahora el texto cambiará correctamente sin colgarse
        txt_seleccion.value = f"🔊 Escuchando marcador en la posición: {estado['tiempo_centro']:.3f} s"
        
        actualizar_reproductor()

    # Vincular los eventos de clic a las capas de marcadores (índices 1 y 2)
    fig.data[1].on_click(al_hacer_clic)
    fig.data[2].on_click(al_hacer_clic)

    # 9. Disponer la interfaz organizada por bloques
    controles = widgets.VBox(
        [txt_seleccion, contenedor_sliders, output_reproductor],
        layout=widgets.Layout(
            width='1200px', 
            align_items='center', 
            justify_content='center'
        )
    )
    
    display(fig)
    display(controles)