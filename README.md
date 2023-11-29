# EnerGAIN

Entwicklung einer intelligenten Gebotsstrategie für den Strommarkt mittels Reinforcement Learning: Eine Untersuchung am
Day-Ahead-Markt und Primärregelleistungsmarkt

## Performance

The following table shows the performance of the model on the different environments. The performance is measured by the
profit

| model      | avg price | avg amount | buy count | sell count | reserve count | avg reward | total reward | total profit |
|------------|-----------|------------|-----------|------------|---------------|------------|--------------|--------------|
| base       | 0.11      | 1.43       | 286       | 89         | 0             | -277.50    | -104063.66   | 335.33       |
| trend      | 0.09      | 0.18       | 1954      | 1038       | 0             | -55.1      | -165108.19   | 1951.80      |
| no savings | 0.09      | 0.20       | 1893      | 776        | 0             | -66.5      | -177502.52   | 1647.47      |
| base prl   | 0.06      | 34.26      | 62        | 257        | 192           | -469.5     | -239953.3    | 4266         |
| multi      | 0.05      | 70.9       | 401       | 667        | 1328          | -96.3      | -230771.97   | 4679.02      |

## Installation

```bash
pip install -r requirements.txt

pip install stable-baselines3==2.0.0a13

```

# Running the Program

## Training the Model

To train the model, use the `main.py` script. This script accepts two command-line arguments:

- `--training_steps`: The number of training steps to run. This must be an integer.
- `--env`: The environment to use. This must be one of 'base', 'trend', 'no_savings', 'base_prl', 'multi'
- `--save`: To decide if the model should be saved or not.

For example, to train the model for 500,000 steps on the 'trend' environment, run:

```bash
python main.py --training_steps 100_000 --env trend --save
```

The trained model is saved to the 'agents' directory with a filename based on the chosen environment.

## Evaluating the Model

To evaluate the model, use the `validation.py` script. This script accepts one command-line argument:

- `--env`: The environment to use. This must be one of 'base', 'trend', 'no_savings', 'base_prl'.
- `--plot`: To decide if the model should be plotted or not.
- `--month`: The month to evaluate the model on. This must be an integer between 1 and 12. If 0 the whole year is
  evaluated.
  For example, to evaluate the model on the 'trend' environment, run:

```bash
python validation.py --env base --plot --month 0
```

The script prints various statistics about the model's performance, and also saves a CSV file with the model's trades to
the current directory.
To plot those trades, use the `plotting.py` script.