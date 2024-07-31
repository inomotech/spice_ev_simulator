import spice_ev
import streamlit as st
import altair as alt
import pandas as pd
from simulate import simulate
from spice_ev.util import set_options_from_config
from generate import generate
from vehicles import Vehicle, convert_to_vehicle_objects
import argparse
import os
import json

# stramplit wide
st.set_page_config(layout="wide")

def create_price_chart(prices_data: pd.DataFrame, time_col_name: str = "time", price_col_name: str = "price"):
    """
    Create an Altair chart with tooltips for Price (EUR/MWh)
    """
    brush = alt.selection_interval(bind='scales', encodings=['x'])
    
    price_chart = alt.Chart(prices_data).mark_line(color='red').encode(
        x=alt.X('{}:T'.format(time_col_name), title='Time'),
        y=alt.Y('{}:Q'.format(price_col_name), title=price_col_name),
        tooltip=[alt.Tooltip('{}:T'.format(time_col_name), title=time_col_name), alt.Tooltip('{}:Q'.format(price_col_name), title=price_col_name, format='%Y-%m-%d %H:%M')]
    ).add_selection(
        brush
    ).properties(
        width=800,
        height=400
    )
    return price_chart

def generate_vehicle_types_file(path: str, vehicles_df: pd.DataFrame) -> str:
    vehicles_file_path = os.path.join(os.path.dirname(path), "vehicles.json")
    vehicles = convert_to_vehicle_objects(vehicles_df)
    output = {}
    for i, v in enumerate(vehicles):
        output[v.name] = v.__dict__

    with open(vehicles_file_path, 'w') as f:
        json.dump(output, f, indent=4)

    return vehicles_file_path

def generate_pv_and_load_csv(path: str, pv_power_max: int, load_power: int = 0, num_days: int = 1) -> str:
    data = pd.read_csv("./input/pv_and_load.csv")
    # data["load"]
    pv_file_path = os.path.join(os.path.dirname(path), "pv_and_load.csv")
    st.write("will write to", pv_file_path)

    max_power_in_file = data["pv"].max()
    factor = pv_power_max / max_power_in_file
    data["pv"] = data["pv"] * factor
    
    data[["timestamp", "pv"]].to_csv(pv_file_path, index=False)
    return "pv_and_load.csv"

def generate_config(run_id: str, scenario_id: str, vehicles_df: pd.DataFrame, prices_data: pd.DataFrame | None, grid_power_kW: int) -> str:
    """
    Generate a configuration file for a scenario
    Returns the path to the scenarion.json

    @param prices_data: a daaframe with two columns: time and price
    """
    # load defaults from file
    config_file = "./b_on/generate.cfg"
    parser = argparse.ArgumentParser(description='Application description.')
    parser.add_argument('--config', type=str, help='Path to the configuration file', default=config_file)
    args = parser.parse_args()
    set_options_from_config(args, verbose=True)
    
    # base path for all tmp files for this run
    output_json_path = "./tmp/{}_{}/scenario.json".format(run_id, scenario_id)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    args.output = output_json_path

    # write vehicles to file which is then passed to the config
    vehicles_file_path = generate_vehicle_types_file(output_json_path, vehicles_df)
    args.vehicle_types = vehicles_file_path
    args.vehicles = [[v["count"], v["name"]] for _, v in vehicles_df.iterrows()]

    # write pv and load data to file
    pv_file_path = generate_pv_and_load_csv(output_json_path, pv_power_max=20, load_power=0, num_days=num_days)
    st.write(pv_file_path)
    # args.include_local_generation_csv = pv_file_path
    # args.include_local_generation_csv_option = [['column', "pv"], ['step_duration_s', 3600]]

    # set the battery capacity
    args.battery = [[battery_capacity, 1]] if battery_capacity > 0 else []
    args.buffer = 0.1

    args.gc_power = grid_power_kW
    
    args.days = num_days

    # st.write(args)

    if prices_data is not None:        
        args.start_time = prices_data["time"].iloc[0]
        prices_csv_path = os.path.join(os.path.dirname(output_json_path), "prices.csv")
        prices_data = prices_data.to_csv(path_or_buf=prices_csv_path, index=False)
        
        args.include_price_csv = "prices.csv" # it will be relative to scenario.json
        args.include_price_csv_option=[['column', "price"], ['step_duration_s', 3600]]
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

        # minutes betwen each timestep
        if len(self.timeseries) < 2:
            raise ValueError("Timeseries must have at least 2 rows")
        
        self.timeseries["time"] = pd.to_datetime(self.timeseries["time"])
        time_between_timesteps = self.timeseries["time"].diff().iloc[1].seconds / 60

        self.timeseries["cost"] = self.timeseries.apply(lambda row: -1 * row["grid supply [kW]"] * row["price [EUR/kWh]"] * time_between_timesteps / 60, axis=1)
        self.timeseries["total_cost"] = self.timeseries["cost"].cumsum()

    def get_metrics(self):
        # st.write(self.timeseries)
        metrics = []
        data = self.raw_results

        # Total costs calculated for the simulation period
        metrics.append(Metric(name='Commodity cost calculated', value=self.timeseries["total_cost"].iloc[-1], unit='EUR', detail='Calculated as grid * price for each timestep and summed'))

        # Commodity costs total for the period
        commodity_costs = data['costs']['electricity costs']['for simulation period']['grid fee']['commodity costs']['total costs']
        metrics.append(Metric(name='Commodity Costs (using fixed value)', value=commodity_costs, unit='EUR', detail='Total commodity costs for the simulation period'))

        # Stationary battery cycles
        if 'stationary battery cycles' in data:
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

def simulate_scenario(scenario_file, strategy, show_visual) -> ScenarioResult:
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
        visual=show_visual)
    simulate(params)
    results = json.loads(open(params.save_results).read())
    timeseries = pd.read_csv(params.save_timeseries)
    soc = pd.read_csv(params.save_soc)
    return ScenarioResult(results=results, timeseries=timeseries, soc=soc)

def run_scenario(run_id: int, strategy: str, grid_power_kW: int, vehicles_df: pd.DataFrame, prices_data: pd.DataFrame, show_visual: bool = False):
    scenario_path = generate_config(run_id=run_id, scenario_id=strategy, vehicles_df=vehicles_df, prices_data=prices_data, grid_power_kW=grid_power_kW)
    res = simulate_scenario(scenario_file=scenario_path, strategy=strategy, show_visual=show_visual)
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

def load_prices_data(prices_file: str, transport_cost: float) -> pd.DataFrame:
    """
    Load prices data from a CSV file and preprocess it
    Returns the (dataframe, name of time column, name of price column)
    """
    prices_data = pd.read_csv(prices_file, parse_dates=True)
    columns = prices_data.columns
    time_col_name = columns[0]
    price_col_name = columns[1]
    prices_data[price_col_name] = prices_data[price_col_name].apply(lambda x: x/1000) # convert from EUR/MWh to EUR/kWh
    prices_data = prices_data.rename(columns={time_col_name: "time", price_col_name: "price"})
    prices_data["price"] = prices_data["price"] + transport_cost
    return prices_data

c = st.columns(5)
with c[0]:
    strategies_list: list[str] = [
        "greedy", 
        "balanced", 
        "balanced_market", 
        "peak_shaving", 
        # "peak_load_window", 
        # "flex_window", 
        # "schedule"
        ]

    strategies_selected: list[str] = st.multiselect("Select a strategy", strategies_list, ["greedy"])

with c[1]:
    battery_capacity = st.number_input("Battery Capacity (kWh)", value=100)

with c[2]:
    grid_power_kW = st.number_input("Grid Power (kW)", value=50)
    show_visual = False #st.checkbox("Show SimVisual", value=False)

with c[3]:
    num_days = st.number_input("Number of Days", value=7)

# vehicles
vehicles_df = pd.read_csv("./input/vehicles.csv")
vehicles_df = vehicles_df.head(1)
vehicles_df.no_drive_days = vehicles_df.no_drive_days.apply(lambda x: json.loads(x) if not pd.isna(x) else [])
st.write(vehicles_df)

# prices
prices_file = './b_on/input/june_2024_prices.csv'
prices_data = load_prices_data(prices_file, transport_cost=0.1)
# st.write(prices_data)
price_chart = create_price_chart(prices_data)
# st.altair_chart(price_chart)

if len(strategies_selected) == 0:
    st.info("Please select at least one strategy")
    st.stop()

results = {}
for s in strategies_selected:
    results[s] = run_scenario(grid_power_kW=grid_power_kW, prices_data=prices_data, vehicles_df=vehicles_df, run_id=1, strategy=s, show_visual=show_visual)

consolidated_res = consolidate_results(results)
st.write(consolidated_res)

strategy = st.selectbox("Select a strategy to display", strategies_selected)

# Metrics to plot
metrics = ['grid supply [kW]', '# CS in use [-]', 'battery power [kW]', 'bat. stored energy [kWh]', 'sum CS power [kW]', 'cost', 'total_cost', 'price [EUR/kWh]']
if battery_capacity == 0:
    metrics.remove('battery power [kW]')
    metrics.remove('bat. stored energy [kWh]')
metrics = [metric.replace("[", "").replace("]", "") for metric in metrics]
# st.write("Timeseries data for strategy: {}".format(strategy))

def display_chart(res: ScenarioResult, metrics: list[str]):
    soc = res.soc
    soc.pop('timestep')
    soc_melted = soc.melt(id_vars=['time'], var_name='vehicle', value_name='SoC')

    base_soc = alt.Chart(soc_melted).encode(
        x='time:T'
    ).properties(
        width=1200,
        height=300
    )

    soc_chart = base_soc.mark_line().encode(
        y='SoC:Q',
        color=alt.Color('vehicle:N', legend=alt.Legend(title="Vehicles")),
        tooltip=[alt.Tooltip('vehicle:N'), alt.Tooltip('SoC:Q'), alt.Tooltip('time:T', title='Time', format='%Y-%m-%d %H:%M')]
    ).interactive()

    data = res.timeseries
    cols = [c.replace("[", "").replace("]", "") for c in data.columns]
    data = data.rename(columns=dict(zip(data.columns, cols)))
    
    melted_data = data.melt(id_vars=['time'], value_vars=metrics, var_name='metric', value_name='value')
    nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['time'], empty='none')
    
    base_metrics = alt.Chart(melted_data).encode(
        x='time:T'
    ).properties(
        width=1200,
        height=100
    )

    bars = base_metrics.mark_bar().encode(
        y='value:Q',
        color=alt.Color('metric:N', legend=None)
    )

    selectors = base_metrics.mark_point().encode(
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
    rules = base_metrics.mark_rule(color='gray').encode(
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
    )

    combined_chart = alt.vconcat(soc_chart, layered_chart).resolve_axis(
        x='shared'
    ).configure_view(
        strokeWidth=0
    )

    st.altair_chart(combined_chart)

# display_soc_chart(results[strategy])
display_chart(results[strategy], metrics)
