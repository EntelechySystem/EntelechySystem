import p5 from 'p5';

/**
 * 代理对象数组，每个代理包含位置、速度、动作状态和观察状态
 */
let agents: { position: p5.Vector; velocity: p5.Vector; actionState: any; observationState: any }[] = [];

/**
 * 地标对象数组，每个地标包含位置
 */
let landmarks: { position: p5.Vector }[] = [];

/**
 * 动作空间定义，包括说话、移动、情绪、抓取、睡眠和进食等动作
 */
const actionSpaces = {
  speaking: { maxLength: 256 }, // 说话动作的最大长度
  movement: { low: -1, high: 1, dimensions: 2 }, // 移动动作的范围和维度
  emotion: { states: ["静", "喜", "怒", "哀", "惧", "思"] }, // 情绪状态
  grabbing: { states: ["无抓取", "抓取"] }, // 抓取状态
  sleeping: { states: ["醒来", "睡觉"] }, // 睡眠状态
  eating: { states: ["未进食", "进食"] }, // 进食状态
};

/**
 * 观察空间定义，包括视觉、听觉、情绪、触觉、嗅觉、温度等观察状态
 */
const observationSpaces = {
  vision: { width: 640, height: 480, channels: 3 }, // 视觉的宽度、高度和通道数
  hearing: { maxLength: 256 }, // 听觉的最大长度
  emotion: { states: ["静", "喜", "怒", "哀", "惧", "思"] }, // 情绪状态
  touch: { states: ["无碰触", "轻触", "中触", "重触", "疼痛"] }, // 触觉状态
  smell: { states: ["无味觉", "有味觉"] }, // 嗅觉状态
  temperature: { range: [0, 1] }, // 温度范围
  speaking: { states: ["不说话", "说话"] }, // 说话状态
  grabbing: { states: ["无抓取", "抓取"] }, // 抓取状态
  sleepiness: { range: [0, 1] }, // 睡意范围
  hunger: { range: [0, 1] }, // 饥饿范围
};

/**
 * p5.js 的 setup 函数，用于初始化画布和对象
 */
function setup() {
  createCanvas(320, 320); // 创建画布，宽高为 320x320
  for (let i = 0; i < 3; i++) {
    agents.push({
      position: createVector(random(width), random(height)), // 随机生成代理的位置
      velocity: createVector(random(-1, 1), random(-1, 1)), // 随机生成代理的速度
      actionState: {
        speaking: 0, // 说话状态
        movement: createVector(0, 0), // 移动状态
        emotion: 0, // 情绪状态
        grabbing: 0, // 抓取状态
        sleeping: 0, // 睡眠状态
        eating: 0, // 进食状态
      },
      observationState: {
        vision: [], // 视觉状态
        hearing: [], // 听觉状态
        emotion: 0, // 情绪状态
        touch: 0, // 触觉状态
        smell: 0, // 嗅觉状态
        temperature: 0.5, // 温度状态
        speaking: 0, // 说话状态
        grabbing: 0, // 抓取状态
        sleepiness: 0, // 睡意状态
        hunger: 0, // 饥饿状态
      },
    });
  }

  for (let i = 0; i < 3; i++) {
    landmarks.push({
      position: createVector(random(width), random(height)), // 随机生成地标的位置
    });
  }
}

/**
 * p5.js 的 draw 函数，用于绘制画布内容
 */
function draw() {
  background(240); // 设置背景颜色为浅灰色

  // 更新代理的位置
  for (const agent of agents) {
    agent.position.add(agent.velocity); // 根据速度更新位置

    // 边界处理：超出画布边界时从另一侧出现
    if (agent.position.x > width) agent.position.x = 0;
    if (agent.position.x < 0) agent.position.x = width;
    if (agent.position.y > height) agent.position.y = 0;
    if (agent.position.y < 0) agent.position.y = height;

    // 绘制代理
    fill(52, 152, 219); // 设置颜色为蓝色
    ellipse(agent.position.x, agent.position.y, 20, 20); // 绘制圆形表示代理
  }

  // 绘制地标
  for (const landmark of landmarks) {
    fill(231, 76, 60); // 设置颜色为红色
    ellipse(landmark.position.x, landmark.position.y, 15, 15); // 绘制圆形表示地标
  }
}