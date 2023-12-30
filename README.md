# EnerGAIN

Entwicklung einer intelligenten Gebotsstrategie für den Strommarkt mittels Reinforcement Learning: Eine Untersuchung am
Day-Ahead-Markt und Primärregelleistungsmarkt

## Performance

The following table shows the performance of the model on the different environments. The performance is measured by the
profit

| **Agent**                           | **$\diameter$ (€) Preis** | **$\diameter$ (kWh) Menge** | **Anzahl Kauf** | **Anzahl Verkauf** | **Anzahl Reserve** | **Anzahl Halte** | **Anzahl invalide** |
|-------------------------------------|---------------------------|-----------------------------|-----------------|--------------------|--------------------|------------------|---------------------|
| **Base**                            | 0,08                      | 60,63                       | 1179            | 814                | -                  | 428              | 6362                |
| **No Savings**                      | 0,09                      | 62,04                       | 810             | 616                | -                  | 507              | 6850                |
| **Trend**                           | 0,08                      | 70,77                       | 1421            | 1536               | -                  | 500              | 5826                |
| **Multi-Markt**                     | 0,01                      | 636,88                      | 3               | 3                  | 7532               | 10               | 22                  |
| **parallel Multi Markt**            | 0,02                      | 348,98                      | 118             | 392                | 3592               | 359              | 533                 |
| **parallel Multi Markt No Savings** | 0,02                      | 264,33                      | 192             | 461                | 4404               | 207              | 519                 |
| **parallel Multi Markt Trend**      | 0,02                      | 294,12                      | 113             | 249                | 4388               | 133              | 521                 |

| **Agent**                           | **$\diameter$ (€) Kauf Preis** | **$\diameter$ (€) Verkauf Preis** | **$\diameter$ (kWh) Kauf Menge** | **$\diameter$ (kW) Reserve** | **$\diameter$ (kWh) Verkauf Menge** | **Preis (€) Differenz Kaufen** | **Preis (€) Differenz Verkaufen** | **Kapital (€)** |
|-------------------------------------|--------------------------------|-----------------------------------|----------------------------------|------------------------------|-------------------------------------|--------------------------------|-----------------------------------|-----------------|
| **Base**                            | 0,35                           | 0,04                              | 51,47                            | -                            | 73,89                               | 0,29                           | 0,05                              | 1.982,69        |
| **No Savings**                      | 0,54                           | 0,05                              | 54,94                            | -                            | 71,37                               | 0,47                           | 0,06                              | 1.719           |
| **Trend**                           | 0,29                           | 0,04                              | 73,79                            | -                            | 67,97                               | 0,23                           | 0,05                              | 3.868           |
| **Multi-Markt**                     | 0,61                           | 0,001                             | 146,79                           | 745,13                       | 204,75                              | 0,54                           | 0,16                              | 16.853,94       |
| **parallel Multi Markt**            | 0,42                           | 0,02                              | 495,89                           | 333,67                       | 148,06                              | 0,34                           | 0,09                              | 13.863,22       |
| **parallel Multi Markt No Savings** | 0,47                           | 0,01                              | 439,06                           | 357,21                       | 182,31                              | 0,42                           | 0,08                              | 12.510,99       |
| **parallel Multi Markt Trend**      | 0,34                           | 0,03                              | 430,21                           | 296,38                       | 194,04                              | 0,27                           | 0,06                              | 12.149,91       |

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