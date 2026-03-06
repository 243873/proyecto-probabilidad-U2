import plotly.express as px
import plotly.figure_factory as ff

class Visualizer:
    def plot_distribution(self, df, feature, target):
        fig = px.histogram(df, x=feature, color=target, barmode='overlay',
                           title=f'Distribución de {feature} respecto al Objetivo',
                           opacity=0.7)
        return fig
        
    def plot_confusion_matrix(self, cm):
        # Nombres de los ejes
        x = ['Predicción: No ocurre', 'Predicción: Sí ocurre']
        y = ['Real: No ocurre', 'Real: Sí ocurre']

        # Crear Heatmap (Matriz de confusión)
        fig = ff.create_annotated_heatmap(
            z=cm, 
            x=x,
            y=y,
            colorscale='Blues',
            showscale=True
        )
        fig.update_layout(title_text='Matriz de Confusión', xaxis_title="Predicción del Modelo", yaxis_title="Valor Real")
        return fig