import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
import base64
import io
import os

# --- Load and parse data ---
cols = ['element', 'layer', 'x', 'y', 'z', 'thermal']
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        s = io.StringIO(decoded.decode('utf-8'))
        df = pd.read_csv(s, delim_whitespace=True, names=cols, engine='python')
        df = df.dropna()
        df['element'] = df['element'].astype(int)
        df['layer'] = df['layer'].astype(int)
        return df
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None

# --- Create the 3D figure with gradient lines, threshold highlight, and sequence ---
def create_figure(df, layer_idx, cumulative, threshold, elem_idx):
    # Filter by layer
    if cumulative:
        plot_df = df[df['layer'] <= layer_idx]
    else:
        plot_df = df[df['layer'] == layer_idx]

    # Filter by element sequence
    if elem_idx is not None:
        plot_df = plot_df[plot_df['element'] <= elem_idx]

    # Define colorscale and normalize thermal
    colorscale = [[0, 'blue'], [0.5, 'green'], [1, 'red']]
    norm_vals = (plot_df['thermal'] + 1) / 2  # -1→0, 0→0.5, 1→1

    # Full gradient path
    trace_line = go.Scatter3d(
        x=plot_df['x'], y=plot_df['y'], z=plot_df['z'],
        mode='lines',
        line=dict(
            color=norm_vals,
            colorscale=colorscale,
            cmin=0,
            cmax=1,
            width=4,
            showscale=True,
            colorbar=dict(
                title='Thermal Index',
                thickness=20,
                x=-0.2,
                tickmode='array',
                tickvals=[0, 0.5, 1],
                ticktext=['-1', '0', '1']
            )
        ),
        customdata=np.stack((plot_df['element'], plot_df['layer'], plot_df['thermal']), axis=-1),
        hovertemplate=(
            'Element: %{customdata[0]}<br>'
            'Layer: %{customdata[1]}<br>'
            'Thermal Index: %{customdata[2]:.3f}<extra></extra>'
        )
    )

    # Highlight segments above threshold
    mask = plot_df['thermal'] > threshold
    idxs = np.where(mask)[0]
    segments = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1) if len(idxs) > 0 else []
    thresh_traces = []
    for seg in segments:
        if len(seg) > 1:
            seg_df = plot_df.iloc[seg]
            thresh_traces.append(
                go.Scatter3d(
                    x=seg_df['x'], y=seg_df['y'], z=seg_df['z'],
                    mode='lines',
                    line=dict(color='red', width=6),
                    customdata=np.stack((seg_df['element'], seg_df['layer'], seg_df['thermal']), axis=-1),
                    hovertemplate=(
                        'Element: %{customdata[0]}<br>'
                        'Layer: %{customdata[1]}<br>'
                        'Thermal Index: %{customdata[2]:.3f}<extra></extra>'
                    ),
                    showlegend=False
                )
            )

    fig = go.Figure(data=[trace_line] + thresh_traces)
    fig.update_layout(
        margin=dict(l=100, r=10, b=10, t=40),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        uirevision='constant'
    )
    return fig

# --- Dash App ---
def run_viewer():
    app = dash.Dash(__name__)
    app.title = "3D Thermal Index Toolpath Viewer"

    app.layout = html.Div([
        html.H2("3D Thermal Index Toolpath Viewer"),
        dcc.Upload(
            id='upload-data',
            children=html.Button('Select Data File'),
            multiple=False,
            style={'marginBottom': '20px'}
        ),
        dcc.Store(id='stored-data'),
        dcc.Graph(id='3d-plot', style={'height': '75vh'}),
        # Layer Slider
        html.Div([
            html.Label("Layer:"),
            dcc.Slider(
                id='layer-slider',
                min=0,
                max=0,
                step=1,
                value=0,
                tooltip={"always_visible": True}
            )
        ], style={'width': '80%', 'padding': '0 20px'}),
        # Element Sequence Slider
        html.Div([
            html.Label("Element Sequence:"),
            dcc.Slider(
                id='elem-slider',
                min=0,
                max=0,
                step=1,
                value=0,
                tooltip={"always_visible": True}
            )
        ], style={'width': '80%', 'padding': '10px 20px 20px'}),
        # Controls
        html.Div([
            html.Label("Mode:"),
            dcc.RadioItems(
                id='cumulative-toggle',
                options=[
                    {'label': ' Cumulative View', 'value': 'cumulative'},
                    {'label': ' Single Layer', 'value': 'single'}
                ],
                value='cumulative',
                inline=True
            ),
            html.Label("Highlight threshold (> red):", style={'marginLeft': '40px'}),
            dcc.Input(
                id='threshold-input',
                type='number',
                min=-1,
                max=1,
                step=0.01,
                value=0.5,
                style={'width': '80px', 'marginLeft': '10px'}
            )
        ], style={'padding': '20px'})
    ])

    @app.callback(
        Output('stored-data', 'data'),
        Input('upload-data', 'contents'),
        State('upload-data', 'filename')
    )
    def store_data(contents, filename):
        if not contents:
            return None
        df = parse_contents(contents, filename)
        return df.to_json(date_format='iso', orient='split') if df is not None else None

    # Update layer slider only when new data is loaded
    @app.callback(
        Output('layer-slider', 'max'),
        Output('layer-slider', 'value'),
        Output('layer-slider', 'marks'),
        Input('stored-data', 'data')
    )
    def update_layer_slider(json_df):
        if not json_df:
            return 0, 0, {}
        df = pd.read_json(json_df, orient='split')
        layers = sorted(df['layer'].unique())
        max_layer = layers[-1]
        marks = {int(l): str(int(l)) for l in layers}
        # Default to show all layers
        return max_layer, max_layer, marks

    # Update element slider when data loaded or layer changed
    @app.callback(
        Output('elem-slider', 'min'),
        Output('elem-slider', 'max'),
        Output('elem-slider', 'value'),
        Output('elem-slider', 'marks'),
        Input('stored-data', 'data'),
        Input('layer-slider', 'value')
    )
    def update_elem_slider(json_df, layer_val):
        if not json_df:
            return 0, 0, 0, {}
        df = pd.read_json(json_df, orient='split')
        sub = df[df['layer'] == layer_val]
        elems = sorted(sub['element'].unique())
        min_e, max_e = elems[0], elems[-1]
        marks = {int(e): str(int(e)) for e in elems}
        # Default to show all elements in this layer
        return min_e, max_e, max_e, marks

    @app.callback(
        Output('3d-plot', 'figure'),
        Input('stored-data', 'data'),
        Input('layer-slider', 'value'),
        Input('cumulative-toggle', 'value'),
        Input('threshold-input', 'value'),
        Input('elem-slider', 'value')
    )
    def update_figure(json_df, layer_val, mode, thresh, elem_val):
        if not json_df:
            return go.Figure()
        df = pd.read_json(json_df, orient='split')
        cumulative = (mode == 'cumulative')
        return create_figure(df, layer_val, cumulative, thresh, elem_val)

    app.run(debug=False)

if __name__ == '__main__':
    run_viewer()
