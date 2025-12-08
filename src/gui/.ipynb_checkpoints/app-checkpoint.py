import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

import sys
sys.path.append("../../")

from src.utils import backtest, risk

app = dash.Dash(__name__)
server = app.server  # for deployment

app.layout = html.Div([
    html.H1("Multi-Asset Transformer Backtest", style={"margin-bottom":"30px"}),

    # Date pickers
    html.Div([
        html.Label("Start Date:"),
        dcc.DatePickerSingle(id='start-date', date='2021-01-01'),
        html.Label("End Date:"),
        dcc.DatePickerSingle(id='end-date', date='2025-01-12'),
    ], style={'margin-bottom':'20px', 'display':'flex', 'gap':'20px'}),


    html.Div([
        # Slider 1: Long Threshold
        html.Div([
            html.Label("Long Threshold"),
            dcc.Slider(
                id='long-threshold', min=0.5, max=0.8, step=0.05, value=0.55,
                marks={0.5:'0.50', 0.55:'0.55', 0.6:'0.60', 0.65:'0.65', 0.7:'0.70', 0.75:'0.75', 0.8:'0.80'},
                tooltip={"placement":"top", "always_visible":False}
            ),
        ], style={'flex':'1 1 23%', 'margin-bottom':'20px'}),
    
        # Slider 2: Short Threshold
        html.Div([
            html.Label("Short Threshold"),
            dcc.Slider(
                id='short-threshold', min=0.2, max=0.5, step=0.05, value=0.45,
                marks={0.2:'0.20', 0.25:'0.25', 0.3:'0.30', 0.35:'0.35', 0.4:'0.40', 0.45:'0.45', 0.5:'0.50'},
                tooltip={"placement":"top", "always_visible":False}
            ),
        ], style={'flex':'1 1 23%', 'margin-bottom':'20px'}),
    
        # Slider 3: Target Volatility
        html.Div([
            html.Label("Target Volatility"),
            dcc.Slider(
                id='target-vol', min=0.05, max=0.25, step=0.05, value=0.15,
                marks={0.05:'0.05', 0.1:'0.10', 0.15:'0.15', 0.2:'0.20', 0.25:'0.25'},
                tooltip={"placement":"top", "always_visible":False}
            ),
        ], style={'flex':'1 1 23%', 'margin-bottom':'20px'}),
    
        # Slider 4: Take Profit
        html.Div([
            html.Label("Take Profit"),
            dcc.Slider(
                id='take-profit', min=0.00, max=0.3, step=0.01, value=0.15,
                marks={0.00:'0.00', 0.05:'0.05', 0.1:'0.10', 0.15:'0.15', 0.2:'0.20', 0.25:'0.25', 0.3:'0.30'},
                tooltip={"placement":"top", "always_visible":False}
            ),
        ], style={'flex':'1 1 23%', 'margin-bottom':'20px'}),
    
        # Slider 5: Stop Loss
        html.Div([
            html.Label("Stop Loss"),
            dcc.Slider(
                id='stop-loss', min=-0.3, max=0.00, step=0.01, value=-0.02,
                marks={-0.3:'-0.30', -0.25:'-0.25', -0.2:'-0.20', -0.15:'-0.15', -0.1:'-0.10', -0.05:'-0.05', 0:'0.00'},
                tooltip={"placement":"top", "always_visible":False}
            ),
        ], style={'flex':'1 1 23%', 'margin-bottom':'20px'}),
    
        # Slider 6: Max Hold Days
        html.Div([
            html.Label("Max Hold Days"),
            dcc.Slider(
                id='max-hold-days', min=10, max=50, step=5, value=30,
                marks={10:'10', 20:'20', 30:'30', 40:'40', 50:'50'},
                tooltip={"placement":"top", "always_visible":False}
            ),
        ], style={'flex':'1 1 23%', 'margin-bottom':'20px'}),
    
        # Slider 7: Max Drawdown
        html.Div([
            html.Label("Max Drawdown"),
            dcc.Slider(
                id='max-drawdown', min=0.1, max=0.5, step=0.05, value=0.2,
                marks={0.1:'0.10', 0.2:'0.20', 0.3:'0.30', 0.4:'0.40', 0.5:'0.50'},
                tooltip={"placement":"top", "always_visible":False}
            ),
        ], style={'flex':'1 1 23%', 'margin-bottom':'20px'}),
    
        # Slider 8: Leverage
        html.Div([
            html.Label("Leverage"),
            dcc.Slider(
                id='leverage', min=1, max=10, step=1, value=1,
                marks={1:'1', 5:'5', 10:'10'},
                tooltip={"placement":"top", "always_visible":False}
            ),
        ], style={'flex':'1 1 23%', 'margin-bottom':'20px'}),
        
    ], style={
        'display':'flex',
        'flex-wrap':'wrap',
        'gap':'2%',
        'width':'100%'
    }),

    # Metrics output and graph
    html.Div(id='risk-report-table', style={'margin-bottom':'20px'}),
    dcc.Loading(id="loading",
                type="circle",
                children=html.Div(dcc.Graph(id='equity-graph'))
    )
])


def format_risk_report_table(risk_df: pd.DataFrame) -> html.Div:
    """Format the full risk report as a scrollable, horizontally centered table."""
    # Table header
    header = html.Tr(
        [html.Th("", style={'font-weight':'bold', 'background-color':'#f0f0f0', 'padding':'6px'})] +
        [html.Th(col, style={'font-weight':'bold', 'background-color':'#f0f0f0', 'padding':'6px'}) for col in risk_df.columns]
    )

    # Table body with alternating row colors
    body = []
    for i, (metric, row) in enumerate(risk_df.iterrows()):
        bg_color = '#f9f9f9' if i % 2 == 0 else '#ffffff'
        body.append(
            html.Tr(
                [html.Td(metric, style={'padding':'6px', 'text-align': 'center'})] +
                [html.Td(row[col], style={'padding':'6px', 'text-align': 'center'}) for col in risk_df.columns],
                style={'background-color': bg_color}
            )
        )

    table = html.Table([html.Thead(header), html.Tbody(body)],
                       style={
                           'border': '1px solid #ddd',
                           'border-collapse': 'collapse',
                           'width': '100%',
                           'font-family': 'Arial, sans-serif',
                       })

    # Wrap in a scrollable div and center the table
    return html.Div(
        html.Div(table, style={'overflow-x': 'auto', 'padding-bottom': '10px', 'margin': '0 auto', 'text-align': 'center'}),
        style={
            'max-width': '90%',      # table can expand but stays within page width
            'background-color': '#f7f7f7',
            'padding': '15px',
            'border-radius': '8px',
            'box-shadow': '0 2px 5px rgba(0,0,0,0.1)',
            'margin': '20px auto'    # center the whole container
        }
    )
# -----------------------------
# CALLBACK
# -----------------------------
@app.callback(
    [Output('equity-graph', 'figure'),
     Output('risk-report-table', 'children')],
    [Input('start-date', 'date'),
     Input('end-date', 'date'),
     Input('long-threshold', 'value'),
     Input('short-threshold', 'value'),
     Input('target-vol', 'value'),
     Input('take-profit', 'value'),
     Input('stop-loss', 'value'),
     Input('max-hold-days', 'value'),
     Input('max-drawdown', 'value'),
     Input('leverage', 'value')])
def update_backtest(start_date, 
                    end_date, 
                    long_thresh, 
                    short_thresh, 
                    target_vol,
                    take_profit,
                    stop_loss,
                    max_hold_days,
                    max_drawdown,
                    leverage):
    
    if not start_date or not end_date:
        return go.Figure(), ""

    # Set up param_grid for your backtest
    param_grid = {'long_threshold': long_thresh,
                  'short_threshold': short_thresh,
                  'target_vol': target_vol,
                  'slippage': 1,
                  'commission': 1,
                  'take_profit': take_profit,
                  'stop_loss': stop_loss,
                  'max_hold_days': max_hold_days,
                  'max_drawdown': max_drawdown,
                  'leverage': leverage}

    # Run the backtest
    strategy_eq, benchmark_eq, rf_series = backtest.backtest(param_grid=param_grid,
                                                             start_date=start_date,
                                                             end_date=end_date,
                                                             sharpe_only=False,
                                                             output=False,
                                                             input_loc='../../data/processed/predicted_df.csv',
                                                             benchmark_loc='../../data/raw/ftse_index.csv',
                                                             rf_loc='../../data/raw/rf/rf.csv')

    # Compute metrics
    strategy_returns = risk.compute_returns(strategy_eq)
    drawdown_series = risk.compute_drawdown(strategy_eq)
    rolling_volatility = risk.rolling_vol(strategy_returns)
    rolling_sharpe = risk.rolling_sharpe(strategy_returns)

    risk_df = risk.full_risk_report(strategy_eq, benchmark_eq, rf_series).T
    risk_table_div = format_risk_report_table(risk_df)

    # Create subplot with 2 rows, shared X-axis
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02)

    # Equity curves
    fig.add_trace(go.Scatter(x=strategy_eq.index, y=strategy_eq.values, mode='lines', name='Strategy', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=benchmark_eq.index, y=benchmark_eq.values, mode='lines', name='Benchmark', line=dict(color='black')), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series.values, mode='lines', name='Drawdown', line=dict(color='red')), row=2, col=1)

    # Returns
    fig.add_trace(go.Scatter(x=strategy_returns.index, y=strategy_returns.values, mode='lines', name='Returns', line=dict(color='green')), row=3, col=1)

    # Rolling Volatility
    fig.add_trace(go.Scatter(x=rolling_volatility.index, y=rolling_volatility.values, mode='lines', name='Rolling Volatility', line=dict(color='red')), row=4, col=1)

    # Rolling Sharpe
    fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, mode='lines', name='Rolling Sharpe', line=dict(color='Purple')), row=5, col=1)
    
    # Layout adjustments
    fig.update_layout(title={'text': "Strategy Backtest Performance", 
                             'x':0.5,
                             'xanchor':'center',
                             'font': {'size': 24}},
                      height=1500)

    # Label Y axes
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    fig.update_yaxes(title_text="Returns", row=3, col=1)
    fig.update_yaxes(title_text="Rolling Volatility", row=4, col=1)
    fig.update_yaxes(title_text="Rolling Sharpe", row=5, col=1)    

    return fig, risk_table_div

if __name__ == "__main__":
    app.run()