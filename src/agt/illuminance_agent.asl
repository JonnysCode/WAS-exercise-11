/*
* The URL of the W3C Web of Things Thing Description of a lab environment
* Simulated lab URL: "https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl"
* Real lab URL: "https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl"
*/
learning_lab_environment("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl").
task_requirements([2,3]).

!start.

@start
+!start : learning_lab_environment(Url) & task_requirements([Z1Level, Z2Level])
<-
  .print("Lab environment URL: ", Url);
  .print("I want to achieve Z1Level=", Z1Level, " and Z2Level=",Z2Level);
  makeArtifact("Qlearner", "tools.QLearner", [Url], QlearnerId);
  calculateQ([Z1Level, Z2Level], 10, 0.1, 0.5, 0.2, 100).
