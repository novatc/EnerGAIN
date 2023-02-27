# EnerGAIN
EnerGAINs - Energy Gainful Automated Intelligent Network System

**Motivation**
Großbatteriespeicher bieten in Zukunft die Möglichkeit, in einem von volatilen Erzeugern geprägten Energiemarkt entscheidend zur Stabilisierung des europäischen Verbundnetzes beizutragen. Denkbar ist dabei die Teilnahme an verschiedenen Märkten: Im klassischen Energy-only-Market (Spotmarkt) können Batterien als Arbitrageure auftreten, die Strom zu billigen Zeiten einkaufen und zu teuren Zeiten verkaufen. Daneben können Batterien an den Primärregelleistungsmärkten zur kurzfristigen oder an positiven wie negativen Sekundärregel- und Arbeitsmärkten zur mittelfristigen Frequenzstabilisierung genutzt werden. Besonders spannend und u.U. bereits heute lohnenswert wird ein solcher Batteriespeicher, wenn dieser all diese Märkte bedienen kann und zur günstigen Stromversorgung mit einem großen Solarpark kombiniert wird. Die Herausforderung liegt allerdings in der konkreten Betriebsführung der Batterie: Zu welchem Zeitpunkt fließt welche Energiemenge in oder aus der Batterie und welcher Markt wird dazu zu welchem Preis bedient.

---

**KI-basiertes Entscheidungshilfesystem**

Aufgabe ist die Entwicklung eines KI-gestützten Entscheidungshilfesystems (Decision Support System, DSS), das den Betreibern hilft, in Echtzeit zu entscheiden, wann und wie die Batterie geladen oder entladen werden soll und an welchen Märkten sie teilnehmen sollen.
Dabei soll sich auf die Entwicklung eines KI-basierten Entscheidungssystems konzentriert werden. Dieses Programm soll in Echtzeit Empfehlungen zu Entscheidungen über den Betrieb eines größeren Energiespeichers liefern.

Inputs in dieses System ist eine Kombination aus Prognosedaten über Preis- und Nachfrageentwicklung für den Primärregelleistungsmärkten und den Sekundärregelmarkt.
Vorhersagen über die Energieerzeugung des Solarparks sind auch denkbar.

Ausgabe des Systems ist der Markt, an dem teilgenommen werden soll, die Uhrzeit der Teilnahme und die Menge an Energie, um die es geht.

Um Vorhersagen über Preis und Nachfrage zu treffen, werden Varianten von rekurrenten neuronalen Netzen verwendet. Für jeden Markt wird ein eigenes Netz mittels historischen Daten trainiert. Dabei geht es allerdings nicht um die Optimierung dieser RNNs, es werden Architekturen und Parameter gewählt, die sich bereits in der Literatur bewährt haben.

Nachgelagert befindet sich das Reinforcement Learning (RL) System. RL ist eine Art des maschinellen Lernens, bei dem ein Agent darauf trainiert wird, in einer Umgebung Entscheidungen zu treffen, die ein sogenanntes Belohnungssignal maximieren. Der Agent lernt durch die Interaktion mit seiner Umgebung, es fällt somit in die Kategorie des unbeaufsichtigten Lernens.
Der Agent entwickelt von alleine eine Strategie, die ihn am meisten belohnt. Belohnungen und Strafen müssen gut gewählt werden, nur dann kann der Agent zielführende Aktionen lernen.

Im Zusammenhang mit Batteriespeichersystemen kann RL verwendet werden, um den Entscheidungsprozess des Batteriespeichersystems zu modellieren und den optimalen Zeitplan für das Laden und Entladen der Batterie zu bestimmen. Die Umgebung kann als zukünftige Bedingungen modelliert werden, wie z.B. Energiebedarf, PV-Erzeugung, und Marktpreise, und die Aktionen des Agenten können als Lade- und Entladeentscheidungen und die Wahl des Marktes  modelliert werden. Die Strategie des Agenten kann durch die Maximierung des erwarteten kumulativen Gewinns gelernt werden, der durch die Teilnahme an verschiedenen Märkten erzielt werden kann.

Als Methode für RL bietet sich Q-Learning an. Q-Learning ist ein Ansatz, bei dem der Agent die optimale Strategie erlernt, in dem die erwartete kumulative Belohnung für jede Aktion in jedem Zustand schätzt.

RL ist eine leistungsstarke Methode, um mit Entscheidungsproblemen unter Unsicherheit umzugehen, die Zukunft bei Entscheidungen zu berücksichtigen und aus Erfahrungen zu lernen. Dabei ist diese Methodik aber rechenintensiv und bedarf vieler Daten. Glücklicherweise gibt es Zugriff auf historische Strompreisdaten und auch Daten eines Stromspeichers, gegen die das entwickelte DSS getestet werden kann. Der Zeitraum wird auf ein Jahr begrenzt.
