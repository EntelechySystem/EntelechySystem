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
    speaking: {maxLength: 256}, // 说话动作，最大长度为256字符
    movement: {low: -1, high: 1, dimensions: 2}, // 移动动作，二维向量范围为[-1, 1]
    emotion: {states: ["静", "喜", "怒", "哀", "惧", "思"]}, // 情绪状态
    grabbing: {states: ["无抓取", "抓取"]}, // 抓取状态
    sleeping: {states: ["醒来", "睡觉"]}, // 睡眠状态
    eating: {states: ["未进食", "进食"]}, // 进食状态
};

/**
 * 观察空间定义，描述代理可以感知的环境信息
 */
let observationSpaces = {
    vision: {width: 640, height: 480, channels: 3}, // 视觉信息，分辨率为640x480，3通道
    hearing: {maxLength: 256}, // 听觉信息，最大长度为256字符
    emotion: {states: ["静", "喜", "怒", "哀", "惧", "思"]}, // 情绪状态
    touch: {states: ["无碰触", "轻触", "中触", "重触", "疼痛"]}, // 触觉状态
    smell: {states: ["无味觉", "有味觉"]}, // 嗅觉状态
    temperature: {range: [0, 1]}, // 温度感知范围
    speaking: {states: ["不说话", "说话"]}, // 说话状态
    grabbing: {states: ["无抓取", "抓取"]}, // 抓取状态
    sleepiness: {range: [0, 1]}, // 困倦程度范围
    hunger: {range: [0, 1]}, // 饥饿程度范围
};

let mapRadius = 150; // 地图半径
let mapCenter; // 地图中心点

/**
 * p5.js的setup函数，用于初始化画布和代理、地标对象
 */
function setup() {
    createCanvas(320, 320); // 创建320x320的画布
    mapCenter = createVector(width / 2, height / 2); // 设置地图中心点

    // 初始化地标
    for (let i = 0; i < 3; i++) {
        landmarks.push({
            position: generateRandomPositionInCircle(), // 在圆形地图内生成地标位置
        });
    }

    // 初始化代理
    for (let i = 0; i < 3; i++) {
        agents.push({
            position: generateRandomPositionInCircle(), // 在圆形地图内生成代理位置
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
}

/**
 * 在圆形地图内生成随机位置
 * @returns {p5.Vector} - 随机位置向量
 */
function generateRandomPositionInCircle() {
    let angle = random(TWO_PI); // 随机角度
    let radius = random(mapRadius); // 随机半径
    return createVector(
        mapCenter.x + cos(angle) * radius,
        mapCenter.y + sin(angle) * radius
    );
}

/**
 * 根据观测变量决策动作
 * @param {Object} observationState - 代理的观测状态
 * @returns {Object} - 决策的动作
 */
function decideAction(observationState) {
    // 简单的随机动作决策逻辑（可扩展为更复杂的算法）
    return {
        movement: createVector(random(-1, 1), random(-1, 1)), // 随机移动
        speaking: 0, // 不说话
        emotion: 0, // 保持情绪不变
        grabbing: 0, // 不抓取
        sleeping: 0, // 不睡觉
        eating: 0, // 不进食
    };
}

/**
 * 执行动作并更新环境
 * @param {Object} agent - 代理对象
 * @param {Object} action - 决策的动作
 */
function stepEnvironment(agent, action) {
    // 更新代理的动作状态
    agent.actionState = action;

    // 根据动作更新代理的位置
    agent.velocity = action.movement;
    agent.position.add(agent.velocity);

    // 限制代理在圆形地图内
    let distanceFromCenter = dist(agent.position.x, agent.position.y, mapCenter.x, mapCenter.y);
    if (distanceFromCenter > mapRadius) {
        // 停止代理的运动
        agent.velocity.set(0, 0);
        // 将代理位置调整到边界上
        let direction = p5.Vector.sub(agent.position, mapCenter).normalize();
        agent.position = p5.Vector.add(mapCenter, direction.mult(mapRadius));
    }
}

let decisionInterval = 30; // 决策间隔时间（帧数）
let frameCounter = 0; // 帧计数器

/**
 * p5.js的draw函数，用于绘制代理和地标，并更新代理的位置
 */
function draw() {
    background(240); // 设置背景颜色为浅灰色

    // 绘制地图边界（圆形）
    noFill();
    stroke(0);
    ellipse(mapCenter.x, mapCenter.y, mapRadius * 2, mapRadius * 2);

    frameCounter++; // 增加帧计数器

    // 绘制地标（最底层）
    for (let landmark of landmarks) {
        fill(231, 76, 60); // 设置地标颜色为红色
        ellipse(landmark.position.x, landmark.position.y, 15, 15); // 绘制地标为圆形
    }

    // 更新并绘制代理（覆盖在地标之上）
    for (let agent of agents) {
        // 每隔指定帧数才决策一次动作
        if (frameCounter % decisionInterval === 0) {
            const action = decideAction(agent.observationState); // 根据观测状态决策动作
            stepEnvironment(agent, action); // 执行动作并更新环境
        } else {
            stepEnvironment(agent, agent.actionState); // 保持当前动作状态
        }

        // 绘制代理
        fill(52, 152, 219); // 设置代理颜色为蓝色
        ellipse(agent.position.x, agent.position.y, 20, 20); // 绘制代理为圆形
    }
}

