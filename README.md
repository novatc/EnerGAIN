# EnerGAIN

Entwicklung einer intelligenten Gebotsstrategie für den Strommarkt mittels Reinforcement Learning: Eine Untersuchung am
Day-Ahead-Markt und Primärregelleistungsmarkt

relative strength index
I.I.D -> annahme für die verteilung der daten

# TODO
- compare on policy model with off policy model


## Installation

```bash
pip install -r requirements.txt

pip install stable-baselines3==2.0.0a13
```
# Running the Program

## Training the Model

To train the model, use the `main.py` script. This script accepts two command-line arguments:

- `--training_steps`: The number of training steps to run. This must be an integer.
- `--env`: The environment to use. This must be one of 'base', 'trend', 'no_savings', 'base_prl'.
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
For example, to evaluate the model on the 'trend' environment, run:

```bash
python validation.py --env base --plot
```

The script prints various statistics about the model's performance, and also saves a CSV file with the model's trades to
the current directory.
To plot those trades, use the `plotting.py` script.