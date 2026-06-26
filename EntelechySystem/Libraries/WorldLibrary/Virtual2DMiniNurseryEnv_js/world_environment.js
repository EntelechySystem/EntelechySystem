import { createWorld, defineComponent, defineQuery, addEntity, addComponent } from 'bitecs';
import Phaser from 'phaser';

// 定义组件
const Position = defineComponent({ x: 'float32', y: 'float32' });
const Velocity = defineComponent({ x: 'float32', y: 'float32' });
const ActionState = defineComponent({
    speaking: 'int8', // 0: 不说话, 1: 说话
    moving: 'float32', // -1 到 1 的连续动作
    emotion: 'int8', // 0: 静, 1: 喜, 2: 怒, 3: 哀, 4: 惧, 5: 思
    grabbing: 'int8', // 0: 无抓取, 1: 抓取
    sleeping: 'int8', // 0: 醒来, 1: 睡觉
    eating: 'int8', // 0: 未进食, 1: 进食
});
const ObservationState = defineComponent({
    vision: ['uint8', 640 * 480 * 3], // 视觉信息 (简化为图像像素)
    hearing: ['uint8', 256], // 听觉信息 (编码文本)
    emotion: 'int8', // 当前表情
    touch: 'int8', // 0: 无碰触, 1: 轻触, 2: 中触, 3: 重触, 4: 疼痛
    smell: 'int8', // 0: 无味觉, 1: 有味觉
    temperature: 'float32', // 0: 很冷, 1: 很热
    speaking: 'int8', // 0: 不说话, 1: 说话
    grabbing: 'int8', // 0: 无抓取, 1: 抓取
    sleepiness: 'float32', // 0: 不困倦, 1: 困倦
    hunger: 'float32', // 0: 不饥饿, 1: 饥饿
});

// 定义系统
const MovementSystem = (world) => {
    const query = defineQuery([Position, Velocity]);
    const entities = query(world);

    for (const eid of entities) {
        Position.x[eid] += Velocity.x[eid];
        Position.y[eid] += Velocity.y[eid];
    }

    return world;
};

const RenderSystem = (scene, sprites) => {
    const query = defineQuery([Position]);
    return (world) => {
        const entities = query(world);
        for (const eid of entities) {
            const circle = sprites[eid];
            circle.x = Position.x[eid];
            circle.y = Position.y[eid];
        }
        return world;
    };
};

// 定义 Phaser 配置
const config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    backgroundColor: '#ffffff',
    parent: 'phaser-example',
    scene: {
        preload: preload,
        create: create,
        update: update,
    },
};

// 创建 Phaser 游戏实例
const game = new Phaser.Game(config);

// 定义全局变量
let world;
let sprites = {};
let movementSystem;
let renderSystem;

function preload() {
    // 不加载图片，直接绘制矢量图
}

function create() {
    // 创建 ECS 世界
    world = createWorld();

    // 创建代理人
    for (let i = 0; i < 3; i++) {
        const eid = addEntity(world);
        addComponent(world, Position, eid);
        addComponent(world, Velocity, eid);
        addComponent(world, ActionState, eid);
        addComponent(world, ObservationState, eid);

        Position.x[eid] = Phaser.Math.Between(100, 700);
        Position.y[eid] = Phaser.Math.Between(100, 700);
        Velocity.x[eid] = Phaser.Math.FloatBetween(-1, 1);
        Velocity.y[eid] = Phaser.Math.FloatBetween(-1, 1);

        // 绘制圆形作为代理人
        const circle = this.add.circle(Position.x[eid], Position.y[eid], 20, 0x3498db); // 半径20，颜色蓝色
        sprites[eid] = circle;
    }

    // 创建地标
    for (let i = 0; i < 3; i++) {
        const eid = addEntity(world);
        addComponent(world, Position, eid);

        Position.x[eid] = Phaser.Math.Between(100, 700);
        Position.y[eid] = Phaser.Math.Between(100, 700);

        // 绘制圆形作为地标
        const circle = this.add.circle(Position.x[eid], Position.y[eid], 15, 0xe74c3c); // 半径15，颜色红色
        sprites[eid] = circle;
    }

    // 初始化系统
    movementSystem = MovementSystem;
    renderSystem = RenderSystem(this, sprites);
}

function update() {
    // 更新系统
    world = movementSystem(world);
    world = renderSystem(world);
}