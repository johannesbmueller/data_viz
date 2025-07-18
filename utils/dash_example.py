import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# Initialize Dash app
app = dash.Dash(__name__)


df = pd.read_csv(r"D:\Notebooks\data_viz\data\Scivias_PCAN_proteins.tsv", sep='\t', index_col=0)

# Calculate correlation matrix with same slicing as your notebook
corr_matrix = df.iloc[:,:100].corr().fillna(0)

# Perform clustering (same as seaborn clustermap)
row_linkage = linkage(pdist(corr_matrix, metric='euclidean'), method='average')
col_linkage = linkage(pdist(corr_matrix.T, metric='euclidean'), method='average')

# Get dendrogram order
row_dendro = dendrogram(row_linkage, no_plot=True)
col_dendro = dendrogram(col_linkage, no_plot=True)

# Reorder correlation matrix based on clustering
clustered_corr = corr_matrix.iloc[row_dendro['leaves'], col_dendro['leaves']]
clustered_proteins = clustered_corr.columns.tolist()

# Create the heatmap figure
def create_heatmap():
    fig = go.Figure(data=go.Heatmap(
        z=clustered_corr.values,
        x=clustered_corr.columns,
        y=clustered_corr.index,
        colorscale='RdBu',  # 'coolwarm' equivalent in plotly
        zmid=0,
        colorbar=dict(title="Correlation"),
        hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Proteome Correlation Heatmap',
        xaxis={'showticklabels': False, 'showgrid': False},
        yaxis={'showticklabels': False, 'showgrid': False},
        width=800,
        height=800,
        plot_bgcolor='white'
    )
    
    # Add thin grid lines
    fig.update_traces(
        xgap=0.5,
        ygap=0.5
    )
    
    return fig

# App layout
app.layout = html.Div([
    html.H1('Interactive Protein Correlation Analysis', 
            style={'textAlign': 'center'}),
    
    html.Div([
        # Left: Heatmap
        html.Div([
            dcc.Graph(
                id='correlation-heatmap',
                figure=create_heatmap(),
                style={'height': '800px'}
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        # Right: Scatter plot
        html.Div([
            dcc.Graph(
                id='scatter-plot',
                style={'height': '600px'}
            ),
            html.Div(id='correlation-info', 
                    style={'textAlign': 'center', 'fontSize': 18, 'marginTop': 20})
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ])
])

# Callback for scatter plot on heatmap click
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('correlation-info', 'children')],
    Input('correlation-heatmap', 'clickData')
)
def update_scatter(clickData):
    if clickData is None:
        # Default scatter plot
        fig = go.Figure()
        fig.add_annotation(
            text="Click on the heatmap to see protein correlations",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            plot_bgcolor='white'
        )
        return fig, ""
    
    # Get clicked proteins from clustered matrix
    protein_x = clickData['points'][0]['x']
    protein_y = clickData['points'][0]['y']
    
    # Get correlation value from clustered matrix
    corr_value = clustered_corr.loc[protein_y, protein_x]
    
    # Get data and remove NaN values
    x_data = df.iloc[:,:100][protein_x]
    y_data = df.iloc[:,:100][protein_y]
    
    # Create dataframe and drop NaN values
    scatter_df = pd.DataFrame({
        'x': x_data,
        'y': y_data
    }).dropna()
    
    if len(scatter_df) < 2:
        # Not enough data points
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data points for correlation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            plot_bgcolor='white'
        )
        return fig, f"Insufficient data (n={len(scatter_df)})"
    
    # Create scatter plot
    fig = px.scatter(
        scatter_df,
        x='x',
        y='y',
        labels={'x': protein_x, 'y': protein_y},
        title=f'Correlation: {protein_x} vs {protein_y}'
    )
    
    # Add regression line only if we have enough points
    if len(scatter_df) > 2:
        try:
            x_range = np.array([scatter_df['x'].min(), scatter_df['x'].max()])
            slope, intercept = np.polyfit(scatter_df['x'], scatter_df['y'], 1)
            y_range = slope * x_range + intercept
            
            fig.add_scatter(
                x=x_range, 
                y=y_range,
                mode='lines',
                name=f'r = {corr_value:.3f}',
                line=dict(color='red', dash='dash')
            )
        except np.linalg.LinAlgError:
            # If polyfit still fails, just show scatter without line
            pass
    
    fig.add_scatter(
        x=x_range, 
        y=y_range,
        mode='lines',
        name=f'r = {corr_value:.3f}',
        line=dict(color='red', dash='dash')
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )
    
    # Correlation info text
    info_text = f"Pearson r = {corr_value:.3f} | n = {len(df)}"
    
    return fig, info_text

if __name__ == '__main__':
    app.run(debug=True, port=8051)