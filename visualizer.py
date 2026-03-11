import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd

class Visualizer:
    def __init__(self):
        # Paleta de colores formal y combinada (Azules corporativos y naranja de contraste)
        self.primary_color = '#1f77b4'
        self.secondary_color = '#00d4ff'
        self.accent_color = '#ff7f0e'
        self.template = 'plotly_white'

    def plot_histogram(self, df, column):
        fig = px.histogram(df, x=column, 
                           color_discrete_sequence=[self.primary_color],
                           title=f'Histograma de {column}',
                           template=self.template)
        fig.update_layout(xaxis_title=column, yaxis_title="Frecuencia")
        return fig

    def plot_time_series(self, df, date_col):
        # Preparar datos para serie de tiempo (contar registros por fecha)
        df_time = df.copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col], format='mixed', errors='coerce')
        df_time = df_time.dropna(subset=[date_col])
        counts = df_time[date_col].dt.date.value_counts().sort_index().reset_index()
        counts.columns = ['Fecha', 'Registros']
        
        fig = px.line(counts, x='Fecha', y='Registros', markers=True,
                      color_discrete_sequence=[self.accent_color],
                      title=f'Grafica Temporal: Registros a lo largo del tiempo',
                      template=self.template)
        return fig

    def plot_distribution(self, df, feature, target):
        fig = px.histogram(df, x=feature, color=target, barmode='group',
                           color_discrete_sequence=[self.primary_color, self.accent_color, '#2ca02c', '#d62728'],
                           title=f'Distribucion de {feature} dado el Objetivo',
                           template=self.template)
        fig.update_layout(yaxis_title="Cantidad")
        return fig
        
    def plot_probability_comparison(self, p_a, p_a_b):
        fig = go.Figure(data=[
            go.Bar(name='Probabilidad Base P(A)', x=['Comparativa'], y=[p_a], marker_color='#a6c8e6'),
            go.Bar(name='Posterior P(A|B) [Con Evidencia]', x=['Comparativa'], y=[p_a_b], marker_color=self.primary_color)
        ])
        fig.update_layout(barmode='group', 
                          title="Comparacion: Probabilidad de Fallo vs Fallo c/ Evidencia",
                          template=self.template, 
                          yaxis=dict(tickformat=".1%", range=[0, max(p_a, p_a_b) + 0.1]))
        return fig

    def plot_confusion_matrix(self, cm):
        x = ['Prediccion: Negativo', 'Prediccion: Positivo']
        y = ['Real: Negativo', 'Real: Positivo']

        # Usar escala de azules que combine con el CSS
        fig = ff.create_annotated_heatmap(
            z=cm, x=x, y=y, colorscale='Blues', showscale=True
        )
        fig.update_layout(title_text='Matriz de Confusion', 
                          xaxis_title="Prediccion del Modelo", 
                          yaxis_title="Valor Real",
                          template=self.template)
        return fig