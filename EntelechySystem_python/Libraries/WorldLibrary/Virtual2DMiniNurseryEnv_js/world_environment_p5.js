/**
 * @file world_environment_p5.js
 * @description 这是一个虚拟2D迷你环境的实现，使用p5.js库来模拟代理和地标的交互。
 */

let agents = []; // 代理数组，用于存储所有的代理对象
let landmarks = []; // 地标数组，用于存储所有的地标对象

/**
 * 动作空间定义，描述代理可以执行的动作类型及其参数范围
 */
let actionSpaces = {
    speaking: { maxLength: 256 }, // 说话动作，最大长度为256字符
    movement: { low: -1, high: 1, dimensions: 2 }, // 移动动作，二维向量范围为[-1, 1]
    emotion: { states: ["静", "喜", "怒", "哀", "惧", "思"] }, // 情绪状态
    grabbing: { states: ["无抓取", "抓取"] }, // 抓取状态
    sleeping: { states: ["醒来", "睡觉"] }, // 睡眠状态
    eating: { states: ["未进食", "进食"] }, // 进食状态
};

/**
 * 观察空间定义，描述代理可以感知的环境信息
 */
let observationSpaces = {
    vision: { width: 640, height: 480, channels: 3 }, // 视觉信息，分辨率为640x480，3通道
    hearing: { maxLength: 256 }, // 听觉信息，最大长度为256字符
    emotion: { states: ["静", "喜", "怒", "哀", "惧", "思"] }, // 情绪状态
    touch: { states: ["无碰触", "轻触", "中触", "重触", "疼痛"] }, // 触觉状态
    smell: { states: ["无味觉", "有味觉"] }, // 嗅觉状态
    temperature: { range: [0, 1] }, // 温度感知范围
    speaking: { states: ["不说话", "说话"] }, // 说话状态
    grabbing: { states: ["无抓取", "抓取"] }, // 抓取状态
    sleepiness: { range: [0, 1] }, // 困倦程度范围
    hunger: { range: [0, 1] }, // 饥饿程度范围
};

/**
 * p5.js的setup函数，用于初始化画布和代理、地标对象
 */
function setup() {
    createCanvas(320, 320); // 创建320x320的画布

    // 初始化代理
    for (let i = 0; i < 3; i++) {
        agents.push({
            position: createVector(random(width), random(height)), // 代理的位置向量
            velocity: createVector(random(-1, 1), random(-1, 1)), // 代理的速度向量
            actionState: {
                speaking: 0, // 说话状态
                movement: createVector(0, 0), // 移动状态
                emotion: 0, // 情绪状态
                grabbing: 0, // 抓取状态
                sleeping: 0, // 睡眠状态
                eating: 0, // 进食状态
            },
            observationState: {
                vision: [], // 视觉信息
                hearing: [], // 听觉信息
                emotion: 0, // 情绪状态
                touch: 0, // 触觉状态
                smell: 0, // 嗅觉状态
                temperature: 0.5, // 温度感知
                speaking: 0, // 说话状态
                grabbing: 0, // 抓取状态
                sleepiness: 0, // 困倦程度
                hunger: 0, // 饥饿程度
            },
        });
    }

    // 初始化地标
    for (let i = 0; i < 3; i++) {
        landmarks.push({
            position: createVector(random(width), random(height)), // 地标的位置向量
        });
    }
}

/**
 * p5.js的draw函数，用于绘制代理和地标，并更新代理的位置
 */
function draw() {
    background(240); // 设置背景颜色为浅灰色

    // 更新并绘制代理
    for (let agent of agents) {
        agent.position.add(agent.velocity); // 更新代理位置

        // 边界处理：代理超出画布时从另一侧出现
        if (agent.position.x > width) agent.position.x = 0;
        if (agent.position.x < 0) agent.position.x = width;
        if (agent.position.y > height) agent.position.y = 0;
        if (agent.position.y < 0) agent.position.y = height;

        // 绘制代理
        fill(52, 152, 219); // 设置代理颜色为蓝色
        ellipse(agent.position.x, agent.position.y, 20, 20); // 绘制代理为圆形
    }

    // 绘制地标
    for (let landmark of landmarks) {
        fill(231, 76, 60); // 设置地标颜色为红色
        ellipse(landmark.position.x, landmark.position.y, 15, 15); // 绘制地标为圆形
    }
}