import streamlit as st
import altair as alt
import pandas as pd
from simulate import simulate
from spice_ev.util import set_options_from_config
from generate import generate
import argparse
import os
import json

# stramplit wide
st.set_page_config(layout="wide")

strategies_list: list[str] = [
    "greedy", 
    # "balanced", 
    # "balanced_market", 
    # "peak_shaving", 
    # "peak_load_window", 
    # "flex_window", 
    # "schedule"
    ]

strategy: str = st.selectbox("Select a strategy", strategies_list)

prices_file = './b_on/input/energy-charts_Electricity_production_and_spot_prices_in_Germany_in_week_20_2024 (1).csv'
prices_data = pd.read_csv(prices_file, parse_dates=True)
columns = prices_data.columns
time_col_name = columns[0]
price_col_name = columns[1]

def create_price_chart(prices_data: pd.DataFrame, time_col_name: str, price_col_name: str):
    """
    Create an Altair chart with tooltips for Price (EUR/MWh)
    """
    brush = alt.selection_interval(bind='scales', encodings=['x'])

    price_chart = alt.Chart(prices_data).mark_line(color='red').encode(
        x=alt.X('{}:T'.format(time_col_name), title='Time'),
        y=alt.Y('{}:Q'.format(price_col_name), title=price_col_name),
        tooltip=[alt.Tooltip('{}:T'.format(time_col_name), title=time_col_name), alt.Tooltip('{}:Q'.format(price_col_name), title=price_col_name)]
    ).add_selection(
        brush
    ).properties(
        width=800,
        height=400
    )
    return price_chart

price_chart = create_price_chart(prices_data, time_col_name, price_col_name)
# st.altair_chart(price_chart)

def generate_config(run_id: str, scenario_id: str, prices_data: pd.DataFrame | None) -> str:
    """
    Generate a configuration file for a scenario
    Returns the path to the scenarion.json

    @param prices_data: a daaframe with two columns: time and price
    """
    config_file = "./b_on/generate.cfg"
    parser = argparse.ArgumentParser(description='Application description.')
    parser.add_argument('--config', type=str, help='Path to the configuration file', default=config_file)
    args = parser.parse_args()
    
    # it just updates the args
    set_options_from_config(args, verbose=True)
    
    output_json_path = "./tmp/{}/{}/scenario.json".format(run_id, scenario_id)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    args.output = output_json_path

    if prices_data is not None:        
        args.start_time = prices_data["time"].iloc[0]
        prices_csv_path = os.path.join(os.path.dirname(output_json_path), "prices.csv")
        prices_data = prices_data.to_csv(path_or_buf=prices_csv_path, index=False)
        
        args.include_price_csv = "prices.csv" # it will be relative to scenario.json
        args.include_price_csv_option=[['column', "price"], ['step_duration_s', 900]]
    else:
        st.warning("Prices data is None")
        
    with st.expander("Generating scenario {}".format(scenario_id), expanded=False):
        st.write(args)
    
    generate(args)

    return output_json_path

class Metric:
    def __init__(self, name: str, value: any, unit: str | None, detail: str | None = None):
        self.name = name
        self.value = value
        self.unit = unit
        self.detail = detail

    def __repr__(self):
        return f"Metric(name={self.name}, value={self.value}, unit={self.unit}, detail={self.detail})"

class ScenarioResult:
    raw_results: dict
    timeseries: pd.DataFrame

    def __init__(self, results: dict, timeseries: pd.DataFrame, soc: pd.DataFrame):
        self.raw_results = results
        self.timeseries = timeseries
        self.soc = soc

    def get_metrics(self):
        metrics = []
        data = self.raw_results

        # Commodity costs total for the period
        commodity_costs = data['costs']['electricity costs']['for simulation period']['grid fee']['commodity costs']['total costs']
        metrics.append(Metric(name='Commodity Costs Total', value=commodity_costs, unit='EUR', detail='Total commodity costs for the simulation period'))

        # Stationary battery cycles
        stationary_battery_cycles = data['stationary battery cycles']['value']
        metrics.append(Metric(name='Stationary Battery Cycles', value=stationary_battery_cycles, unit=None, detail='Number of load cycles of stationary batteries (averaged)'))

        # Power peaks
        power_peaks = data['power peaks']['total']
        metrics.append(Metric(name='Power Peaks', value=power_peaks, unit='kW', detail='Maximum drawn power by all loads'))

        # Sum of energy
        sum_of_energy = data['sum of energy']['value']
        metrics.append(Metric(name='Sum of Energy', value=sum_of_energy, unit='kWh', detail='Total drawn energy from grid connection point during simulation'))

        # Average drawn power
        avg_drawn_power = data['avg drawn power']['value']
        metrics.append(Metric(name='Average Drawn Power', value=avg_drawn_power, unit='kW', detail='Drawn power, averaged over all time steps'))

        # All vehicle battery cycles
        vehicle_battery_cycles = data['all vehicle battery cycles']['value']
        metrics.append(Metric(name='All Vehicle Battery Cycles', value=vehicle_battery_cycles, unit=None, detail='Number of load cycles per vehicle (averaged)'))

        # Times below desired SoC
        times_below_desired_soc = data['times below desired soc']['without margin']
        metrics.append(Metric(name='Times Below Desired SoC', value=times_below_desired_soc, unit=None, detail='Number of times vehicle SoC was below desired SoC on departure (without margin)'))

        return metrics
    
    def get_metrics_df(self):
        metrics = self.get_metrics()
        df = pd.DataFrame([vars(m) for m in metrics])
        return df

def simulate_scenario(scenario_file, strategy) -> ScenarioResult:
    if not strategy:
        raise ValueError("Strategy cannot be None")
    params = argparse.Namespace(
        input=scenario_file,
        save_timeseries = os.path.join(os.path.dirname(scenario_file), "timeseries.csv"),
        save_results = os.path.join(os.path.dirname(scenario_file), "simulation.json"),
        save_soc = os.path.join(os.path.dirname(scenario_file), "soc.csv"),
        strategy=strategy,
        cost_parameters_file = "./b_on/input/price_sheet.json",
        cost_calc=True,
        margin=0.05, 
        strategy_option=[], 
        visual=True)
    simulate(params)
    results = json.loads(open(params.save_results).read())
    timeseries = pd.read_csv(params.save_timeseries)
    soc = pd.read_csv(params.save_soc)
    return ScenarioResult(results=results, timeseries=timeseries, soc=soc)

def run_scenario(prices_data: pd.DataFrame, run_id: int, strategy: str):
    scenario_path = generate_config(run_id=run_id, scenario_id=strategy, prices_data=prices_data)
    res = simulate_scenario(scenario_file=scenario_path, strategy=strategy)
    return res


def display_results(results: ScenarioResult):
    st.write(results.get_metrics_df())

def consolidate_results(results: dict[str, ScenarioResult]):
    keys = list(results.keys())
    first_strategy = keys[0]
    df = results[first_strategy].get_metrics_df()
    df.rename(columns={"value": first_strategy}, inplace=True)
    for strategy in keys[1:]:
        df[strategy] = results[strategy].get_metrics_df()["value"]

    # move unit and detail column at the end
    df["unit"] = df.pop("unit")
    df["detail"] = df.pop("detail")
    return df


prices_data = prices_data.rename(columns={time_col_name: "time", price_col_name: "price"})

results = {}
for s in strategies_list:
    results[s] = run_scenario(prices_data=prices_data, run_id=1, strategy=s)

consolidated_res = consolidate_results(results)
st.write(consolidated_res)

# Metrics to plot
metrics = ['grid supply [kW]', '# CS in use [-]', 'battery power [kW]', 'bat. stored energy [kWh]', 'sum CS power [kW]', 'price [EUR/kWh]']
metrics = [metric.replace("[", "").replace("]", "") for metric in metrics]
# st.write("Timeseries data for strategy: {}".format(strategy))

def display_soc_chart(res: ScenarioResult):
    soc = res.soc
    soc.pop("timestep")
    soc = soc.rename(columns={c.replace("[", "").replace("]", ""): c.replace("[", "").replace("]", "") for c in soc.columns})
    soc_melted = soc.melt(id_vars=['time'], var_name='vehicle', value_name='SoC')
    st.write(soc_melted)
    base = alt.Chart(soc_melted).encode(
        x='time:T'
    ).properties(
        width=1200,
        height=300
    )

    chart = base.mark_line().encode(
        y='SoC:Q',
        color=alt.Color('vehicle:N', legend=None)
    )

    st.altair_chart(chart)

def display_chart(res: ScenarioResult, metrics: list[str]):
    data = res.timeseries
    cols = [c.replace("[", "").replace("]", "") for c in data.columns]
    data = data.rename(columns=dict(zip(data.columns, cols)))
    
    melted_data = data.melt(id_vars=['time'], value_vars=metrics, var_name='metric', value_name='value')
    nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['time'], empty='none')
    
    base = alt.Chart(melted_data).encode(
        x='time:T'
    ).properties(
        width=1200,
        height=100
    )

    bars = base.mark_bar().encode(
        y='value:Q',
        color=alt.Color('metric:N', legend=None)
    )

    selectors = base.mark_point().encode(
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    points = bars.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = bars.mark_text(
        align='left', dx=5, dy=-5, fontSize=25
    ).encode(
        text=alt.condition(nearest, 'value:Q', alt.value(' ')),
        color=alt.value('black')  # Text color
    )

    # Draw a rule at the location of the selection
    rules = base.mark_rule(color='gray').encode(
        opacity=alt.condition(nearest, alt.value(0.8), alt.value(0)),
        size=alt.value(2)
    )

    # Combine the layers and facet by metric
    layered_chart = alt.layer(
        bars, selectors, points, rules, text
    ).facet(
        row=alt.Row('metric:N', title=None, header=alt.Header(labels=True, labelAngle=0, labelAlign='left', labelFontSize=12)),
        spacing=0
    ).resolve_scale(
        y='independent'
    ).configure_view(
        strokeWidth=0
    )

    st.altair_chart(layered_chart)

def display_chart2(res: ScenarioResult, metrics: list[str], dual_metrics: tuple[str, str] = None):
    data = res.timeseries
    cols = [c.replace("[", "").replace("]", "") for c in data.columns]
    data = data.rename(columns=dict(zip(data.columns, cols)))
    
    # Melt the DataFrame for easier plotting with Altair
    melted_data = data.melt(id_vars=['time'], value_vars=metrics, var_name='metric', value_name='value')
    # nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['time'], empty='none')

    base = alt.Chart(melted_data).encode(
        x='time:T'
    ).properties(
        width=1200,
        height=100
    )

    def create_chart(metric):
        chart = base.transform_filter(
            alt.datum.metric == metric
        ).mark_bar().encode(
            y=alt.Y('value:Q', title=metric),
            color=alt.Color('metric:N', legend=None),
            tooltip=[alt.Tooltip('time:T', title='Time', format='%H:%M'), alt.Tooltip('value:Q', title=metric)]
        )
        return chart

    def create_dual_chart(metric1, metric2):
        chart1 = base.transform_filter(
            alt.datum.metric == metric1
        ).mark_line().encode(
            y=alt.Y('value:Q', title=f'{metric1} & {metric2}'),
            color=alt.value('blue'),
            tooltip=[alt.Tooltip('time:T', title='Time', format='%H:%M'), alt.Tooltip('value:Q', title=metric1)]
        )
        
        chart2 = base.transform_filter(
            alt.datum.metric == metric2
        ).mark_line().encode(
            y=alt.Y('value:Q'),
            color=alt.value('red'),
            tooltip=[alt.Tooltip('time:T', title='Time', format='%H:%M'), alt.Tooltip('value:Q', title=metric2)]
        )
        
        return alt.layer(chart1, chart2)

    charts = []
    for metric in metrics:
        if dual_metrics and metric in dual_metrics:
            if dual_metrics[0] == metric:
                charts.append(create_dual_chart(dual_metrics[0], dual_metrics[1]).properties(title=f'{dual_metrics[0]} & {dual_metrics[1]}'))
        else:
            charts.append(create_chart(metric).properties(title=metric))

    # Combine the charts and facet by metric
    layered_chart = alt.vconcat(*charts).resolve_scale(
        y='independent'
    ).configure_view(
        strokeWidth=0
    )

    st.altair_chart(layered_chart)

display_soc_chart(results[strategy])
display_chart(results[strategy], metrics)
