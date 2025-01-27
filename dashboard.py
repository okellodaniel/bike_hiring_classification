import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Database connection
engine = create_engine(
    'postgresql://user:password@localhost:5432/bike_sharing')
db = SQLAlchemy()

# Fetch data from the database


def fetch_data():
    query = "SELECT * FROM ride_data"
    df = pd.read_sql(query, engine)
    return df


# Layout of the dashboard
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Bike Sharing Dashboard",
                className="text-center"), className="mb-5 mt-5")
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='bar-chart'), width=12)
    ])
])

# Callback to update the graph


@app.callback(
    dash.dependencies.Output('bar-chart', 'figure'),
    [dash.dependencies.Input('bar-chart', 'id')]
)
def update_graph(_):
    df = fetch_data()
    fig = {
        'data': [
            {'x': df['day'], 'y': df['prediction'],
                'type': 'bar', 'name': 'Predictions'},
        ],
        'layout': {
            'title': 'Predicted Bike Rentals'
        }
    }
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
