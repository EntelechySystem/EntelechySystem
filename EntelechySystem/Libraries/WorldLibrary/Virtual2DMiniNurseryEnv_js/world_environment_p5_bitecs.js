// import { defineComponent, Types, createWorld, addEntity, addComponent, System, pipe } from 'https://cdn.jsdelivr.net/npm/bitecs@0.3.40/+esm';

/**
 * 定义组件：位置(Position)、速度(Velocity)、颜色(Color)
 */
const Position = defineComponent({ x: Types.f32, y: Types.f32 }); // 位置组件，包含 x 和 y 坐标
const Velocity = defineComponent({ x: Types.f32, y: Types.f32 }); // 速度组件，包含 x 和 y 方向的速度
const Color = defineComponent({ r: Types.ui8, g: Types.ui8, b: Types.ui8 }); // 颜色组件，包含 r、g、b 三种颜色值

/**
 * 创建 ECS 世界
 */
const world = createWorld(); // ECS 世界实例

/**
 * 初始化实体数组
 */
const agents = []; // 代理实体数组
const landmarks = []; // 地标实体数组

/**
 * 创建代理实体
 * @param {number} x - 初始 x 坐标
 * @param {number} y - 初始 y 坐标
 * @param {number} vx - 初始 x 方向速度
 * @param {number} vy - 初始 y 方向速度
 * @param {number} r - 颜色的红色分量
 * @param {number} g - 颜色的绿色分量
 * @param {number} b - 颜色的蓝色分量
 */
function createAgent(x, y, vx, vy, r, g, b) {
  const entity = addEntity(world); // 创建实体
  addComponent(world, Position, entity); // 添加位置组件
  addComponent(world, Velocity, entity); // 添加速度组件
  addComponent(world, Color, entity); // 添加颜色组件
  Position.x[entity] = x; // 设置 x 坐标
  Position.y[entity] = y; // 设置 y 坐标
  Velocity.x[entity] = vx; // 设置 x 方向速度
  Velocity.y[entity] = vy; // 设置 y 方向速度
  Color.r[entity] = r; // 设置红色分量
  Color.g[entity] = g; // 设置绿色分量
  Color.b[entity] = b; // 设置蓝色分量
  agents.push(entity); // 将实体添加到代理数组
}

/**
 * 创建地标实体
 * @param {number} x - 初始 x 坐标
 * @param {number} y - 初始 y 坐标
 * @param {number} r - 颜色的红色分量
 * @param {number} g - 颜色的绿色分量
 * @param {number} b - 颜色的蓝色分量
 */
function createLandmark(x, y, r, g, b) {
  const entity = addEntity(world); // 创建实体
  addComponent(world, Position, entity); // 添加位置组件
  addComponent(world, Color, entity); // 添加颜色组件
  Position.x[entity] = x; // 设置 x 坐标
  Position.y[entity] = y; // 设置 y 坐标
  Color.r[entity] = r; // 设置红色分量
  Color.g[entity] = g; // 设置绿色分量
  Color.b[entity] = b; // 设置蓝色分量
  landmarks.push(entity); // 将实体添加到地标数组
}

/**
 * 移动系统
 * @param {object} world - ECS 世界实例
 * @returns {object} 更新后的 ECS 世界实例
 */
const movementSystem = (world) => {
  const entities = world.query([Position, Velocity]); // 查询具有位置和速度组件的实体
  for (const entity of entities) {
    Position.x[entity] += Velocity.x[entity]; // 更新 x 坐标
    Position.y[entity] += Velocity.y[entity]; // 更新 y 坐标

    // 边界环绕逻辑
    if (Position.x[entity] > width) Position.x[entity] = 0;
    if (Position.x[entity] < 0) Position.x[entity] = width;
    if (Position.y[entity] > height) Position.y[entity] = 0;
    if (Position.y[entity] < 0) Position.y[entity] = height;
  }
  return world; // 返回更新后的世界
};

/**
 * 渲染系统
 * @param {object} world - ECS 世界实例
 * @returns {object} 更新后的 ECS 世界实例
 */
const renderSystem = (world) => {
  background(240);

  // 渲染代理实体
  for (const entity of agents) {
    fill(Color.r[entity], Color.g[entity], Color.b[entity]);
    ellipse(Position.x[entity], Position.y[entity], 20, 20);
  }

  // 渲染地标实体
  for (const entity of landmarks) {
    fill(Color.r[entity], Color.g[entity], Color.b[entity]);
    ellipse(Position.x[entity], Position.y[entity], 15, 15);
  }

  return world; // 返回更新后的世界
};

const pipeline = pipe(movementSystem, renderSystem);

/**
 * p5.js setup 函数
 */
function setup() {
  createCanvas(320, 320);

  // 创建代理实体
  for (let i = 0; i < 3; i++) {
    createAgent(random(width), random(height), random(-1, 1), random(-1, 1), 52, 152, 219);
  }

  // 创建地标实体
  for (let i = 0; i < 3; i++) {
    createLandmark(random(width), random(height), 231, 76, 60);
  }
}

/**
 * p5.js draw 函数
 */
function draw() {
  pipeline(world);
}