# Sustiprintojo mokymosi taikymas LiDAR pagrįstai TurtleBot3 navigacijai

Šioje saugykloje pateikiami bakalauro darbe naudoti įgyvendinimo failai ir „Gazebo“ eksperimentų rezultatų failai:

**Sustiprintojo mokymosi taikymas LiDAR pagrįstai TurtleBot3 navigacijai trajektorijoje, nužymėtoje vartais**  
**Application of Reinforcement Learning for LiDAR-based TurtleBot3 Navigation along a Gate-marked Trajectory**

Projekte tiriama „TurtleBot3 Burger“ roboto navigacija trajektorijoje, pažymėtoje vertikaliais vartais arba stulpais. Darbe lyginami šie metodai:

1. Taisyklėmis pagrįstas LiDAR vartų vidurio sekimo valdiklis.
2. PPO politika, naudojanti neapdorotus LiDAR ir odometrijos stebėjimus.
3. PPO politika su iš LiDAR duomenų gaunamu geometrijos pagalbiniu moduliu.

## Saugyklos struktūra

- `tb3_gate_rl/` – ROS 2 „Python“ mazgai, naudoti „Gazebo“ vertinime.
- `training/raw_lidar_ppo/` – dvimatė „Gymnasium“ aplinka ir skriptai neapdoroto LiDAR PPO metodui.
- `training/residual_lidar_ppo/` – dvimatė „Gymnasium“ aplinka ir skriptai PPO metodui su LiDAR geometrijos pagalbinėmis ypatybėmis.
- `gazebo_100_results/` – galutiniai 100 epizodų „Gazebo“ CSV rezultatų failai, naudoti darbo metodų palyginimui.
- `RESULTS.md` – trumpa galutinio „Gazebo“ palyginimo rezultatų santrauka.
- `REPRODUCING.md` – pavyzdinės komandos valdikliams ir eksperimento valdymo mazgui paleisti.

Išmokytų PPO modelių archyvai tiesiogiai šioje saugykloje nėra saugomi. Paleidžiant išmokytą valdiklį, kodas tikisi modelio katalogo, kuriame yra PPO modelis ir jam atitinkanti stebėjimų normalizavimo statistika.
