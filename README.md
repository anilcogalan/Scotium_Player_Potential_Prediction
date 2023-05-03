# CLASSIFICATION OF TALENT HUNTING WITH ARTIFICIAL LEARNING


<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/61653147/235911086-1c9cc73e-6b0d-42d8-ba36-57be55c57036.png">
</p>

## Business Problem 

Predicting which class (average, highlighted) players are according to the scores given to the characteristics of the football players followed by the Scouts.

## Data Set

The data set consists of information from Scoutium, which includes the features and scores of the football players evaluated by the scouts according to the characteristics of the footballers observed in the matches.

Attributes: It contains the points that the users who evaluate the players give to the characteristics of each player they watch and evaluate in a match. (Independent variables)

potential_labels: Contains potential tags from users who rate players, with their final opinions about the players in each match. (target variable)

9 Variables, 10730 Observations, 0.65 mb


### Table Description

This table contains information about a scout's evaluations of all players on a team's roster in a match. The table has the following columns:


Column Name      | Description                                                        
------------------|--------------------------------------------------------------------
task_response_id | The set of a scout's evaluations of all players on a team's roster in a match 
match_id         | The id of the corresponding match                                       
evaluator_id     | The id of the evaluator(scout)                                         
player_id        | The id of the respective player                                        
position_id      | The id of the position that the relevant player played in that match (1=Goalkeeper, 2=Stopper, 3=Right back, 4=Left back, 5=Defensive midfielder, 6=Central midfielder, 7=Right wing, 8=Left wing, 9=Offensive midfielder, 10=Striker) 
analysis_id      | The set of a scout's attribute evaluations of a player in a match   
attribute_id     | The id of each attribute that players were evaluated for              
attribute_value  | The value (points) given to a player's attribute by a scout            
potential_label  | Label that indicates the final decision of a scout regarding a player's potential in a match (target variable)


The position_id column indicates the position that the player played in that match. The following are the mappings for each position_id:

* Goalkeeper
* Stopper
* Right back
* Left back
* Defensive midfielder
* Central midfielder
* Right wing
* Left wing
* Offensive midfielder
* Striker

The potential_label column indicates the final decision of the scout regarding the potential of a player. This column serves as the target variable for predictive modeling tasks.



