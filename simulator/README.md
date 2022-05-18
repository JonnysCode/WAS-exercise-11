# Simulator for Smart Factory Lighting

## Overview
The simulator is programmed in Node-Red environment. To get it running, you need to install Node-Red (basically a nodeJS based programming tool) and import the simulator_flow.json.

### Steps
1. Install Node-Red by following the instruction here https://nodered.org/docs/getting-started/local
2. Start Node-Red and import the flow (menu expander at top-right, then "Import").
3. Deploy the flow
4. The status endpoint is http://localhost:1880/was/rl/status Example response:

```json
{
  "Z1Level": 396.3840986447821,
  "Z2Level": 473.1920493223911,
  "Z1Light": false,
  "Z2Light": true,
  "Z1Blinds": true,
  "Z2Blinds": false,
  "Sunshine": 640.1482750972317,
  "TotalEnergyCost": 15,
  "EnergyCost": 0,
  "Hour": 1.5000000000000002
}
```

5. To send and action to the environment, POST the action as json payload to http://localhost:1880/was/rl/action
For example, to switch on the Z1 Lights:

```json
{

  "Z1Light": true

}
```

The response will confirm your action and provide a cost:

```json
{

  "Z1Light": true,
  "cost": 100
}
```

The simulator increments the time (Hour) by 0.1h every second and computes the new state of the environment. To keep things simple (intially), the Sunshine value hovers around 600..650. If you want to play with this, look at lines 11..20 in the "Update environment" node.

Postman collection: https://www.getpostman.com/collections/85b4707c30445040db33
