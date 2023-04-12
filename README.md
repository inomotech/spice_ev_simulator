# SpiceEV - Simulation Program for Individual Charging Events of Electric Vehicles 

A tool to generate scenarios of electric-vehicle fleets and simulate different charging
strategies.

# Documentation

Full documentation can be found [here](https://spice-ev.readthedocs.io/en/latest/index.html)

## Installation

Just clone this repository. This tool just has an optional dependency on
Matplotlib. Everything else uses the Python (>= 3.6) standard library.

## Examples

Generate a scenario and store it in a JSON file:

```sh
./generate.py --output example.json
```

Generate a 7-day scenario with 10 vehicles of different types and 15 minute timesteps:

```sh
./generate.py --days 7 --vehicles 6 golf --vehicles 4 sprinter --interval 15 -o example.json
```

Run a simulation of this scenario using the `greedy` charging strategy and show
plots of the results:

```sh
./simulate.py example.json --strategy greedy --visual
```

Include an external load in scenario:
```sh
./generate.py --include-fixed-load-csv fixed_load.csv -o example.json
```
Please note that included file paths are relative to the scenario file location. Consider this directory structure:

```sh
├── scenarios
│   ├── price
│   │   ├── price.csv
│   ├── my_scenario
│   │   ├── fixed_load.csv
│   │   ├── example.json
```
The fixed_load.csv file is in the same directory as the example.json, hence no relative path is specified.

To include the price and fixed load timeseries:
```sh
./generate.py --include-price-csv ../price/price.csv --include-fixed-load-csv fixed_load.csv -o example.json
```

Calculate and include schedule:
```sh
./generate_schedule.py --scenario example.json --input data/timeseries/NSM_1.csv --output data/schedules/NSM_1.csv
```

Show all command line options:

```sh
./generate.py -h
./simulate.py -h
```

There are also example configuration files in the example folder. The required input/output must still be specified manually:

```sh
./generate.py --config examples/configs/generate.cfg
./simulate.py --config examples/configs/simulate.cfg
```

## SimBEV integration

This tools supports scenarios generated by the [SimBEV](https://github.com/rl-institut/simbev) tool. Convert SimBEV output files to a SpiceEV scenario: 
```sh
generate.py simbev --simbev /path/to/simbev/output/ --output example.json
```

# License

SpiceEV is licensed under the MIT License as described in the file [LICENSE](https://github.com/rl-institut/spice_ev/blob/dev/LICENSE)
