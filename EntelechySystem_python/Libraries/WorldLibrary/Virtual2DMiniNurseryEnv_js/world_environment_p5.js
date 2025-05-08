let agents = [];
let landmarks = [];
let actionSpaces = {
  speaking: { maxLength: 256 },
  movement: { low: -1, high: 1, dimensions: 2 },
  emotion: { states: ["静", "喜", "怒", "哀", "惧", "思"] },
  grabbing: { states: ["无抓取", "抓取"] },
  sleeping: { states: ["醒来", "睡觉"] },
  eating: { states: ["未进食", "进食"] },
};
let observationSpaces = {
  vision: { width: 640, height: 480, channels: 3 },
  hearing: { maxLength: 256 },
  emotion: { states: ["静", "喜", "怒", "哀", "惧", "思"] },
  touch: { states: ["无碰触", "轻触", "中触", "重触", "疼痛"] },
  smell: { states: ["无味觉", "有味觉"] },
  temperature: { range: [0, 1] },
  speaking: { states: ["不说话", "说话"] },
  grabbing: { states: ["无抓取", "抓取"] },
  sleepiness: { range: [0, 1] },
  hunger: { range: [0, 1] },
};

function setup() {
  createCanvas(320, 320);
  for (let i = 0; i < 3; i++) {
    agents.push({
      position: createVector(random(width), random(height)),
      velocity: createVector(random(-1, 1), random(-1, 1)),
      actionState: {
        speaking: 0,
        movement: createVector(0, 0),
        emotion: 0,
        grabbing: 0,
        sleeping: 0,
        eating: 0,
      },
      observationState: {
        vision: [],
        hearing: [],
        emotion: 0,
        touch: 0,
        smell: 0,
        temperature: 0.5,
        speaking: 0,
        grabbing: 0,
        sleepiness: 0,
        hunger: 0,
      },
    });
  }

  for (let i = 0; i < 3; i++) {
    landmarks.push({
      position: createVector(random(width), random(height)),
    });
  }
}

function draw() {
  background(240);

  // Update agents
  for (let agent of agents) {
    agent.position.add(agent.velocity);

    // Wrap around edges
    if (agent.position.x > width) agent.position.x = 0;
    if (agent.position.x < 0) agent.position.x = width;
    if (agent.position.y > height) agent.position.y = 0;
    if (agent.position.y < 0) agent.position.y = height;

    // Draw agent
    fill(52, 152, 219);
    ellipse(agent.position.x, agent.position.y, 20, 20);
  }

  // Draw landmarks
  for (let landmark of landmarks) {
    fill(231, 76, 60);
    ellipse(landmark.position.x, landmark.position.y, 15, 15);
  }
}