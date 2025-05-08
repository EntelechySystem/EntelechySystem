// import { defineComponent, Types, createWorld, addEntity, addComponent, System, pipe } from 'https://cdn.jsdelivr.net/npm/bitecs@0.3.40/+esm';

// Define components
const Position = defineComponent({ x: Types.f32, y: Types.f32 });
const Velocity = defineComponent({ x: Types.f32, y: Types.f32 });
const Color = defineComponent({ r: Types.ui8, g: Types.ui8, b: Types.ui8 });

// Create the ECS world
const world = createWorld();

// Initialize entities
const agents = [];
const landmarks = [];

function createAgent(x, y, vx, vy, r, g, b) {
  const entity = addEntity(world);
  addComponent(world, Position, entity);
  addComponent(world, Velocity, entity);
  addComponent(world, Color, entity);
  Position.x[entity] = x;
  Position.y[entity] = y;
  Velocity.x[entity] = vx;
  Velocity.y[entity] = vy;
  Color.r[entity] = r;
  Color.g[entity] = g;
  Color.b[entity] = b;
  agents.push(entity);
}

function createLandmark(x, y, r, g, b) {
  const entity = addEntity(world);
  addComponent(world, Position, entity);
  addComponent(world, Color, entity);
  Position.x[entity] = x;
  Position.y[entity] = y;
  Color.r[entity] = r;
  Color.g[entity] = g;
  Color.b[entity] = b;
  landmarks.push(entity);
}

// Define systems
const movementSystem = (world) => {
  const entities = world.query([Position, Velocity]);
  for (const entity of entities) {
    Position.x[entity] += Velocity.x[entity];
    Position.y[entity] += Velocity.y[entity];

    // Wrap around edges
    if (Position.x[entity] > width) Position.x[entity] = 0;
    if (Position.x[entity] < 0) Position.x[entity] = width;
    if (Position.y[entity] > height) Position.y[entity] = 0;
    if (Position.y[entity] < 0) Position.y[entity] = height;
  }
  return world;
};

const renderSystem = (world) => {
  background(240);

  // Render agents
  for (const entity of agents) {
    fill(Color.r[entity], Color.g[entity], Color.b[entity]);
    ellipse(Position.x[entity], Position.y[entity], 20, 20);
  }

  // Render landmarks
  for (const entity of landmarks) {
    fill(Color.r[entity], Color.g[entity], Color.b[entity]);
    ellipse(Position.x[entity], Position.y[entity], 15, 15);
  }

  return world;
};

const pipeline = pipe(movementSystem, renderSystem);

function setup() {
  createCanvas(320, 320);

  // Create agents
  for (let i = 0; i < 3; i++) {
    createAgent(random(width), random(height), random(-1, 1), random(-1, 1), 52, 152, 219);
  }

  // Create landmarks
  for (let i = 0; i < 3; i++) {
    createLandmark(random(width), random(height), 231, 76, 60);
  }
}

function draw() {
  pipeline(world);
}